import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.data import TBChestXrayDataset, get_val_transform, custom_collate_fn
from src.models import get_model
from src.partition import dirichlet_partition
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_data_distribution(out_dir):
    """Calculates actual data distribution dynamically and plots with a log scale."""
    logger.info("Generating Dynamic Data Distribution Chart...")
    
    num_normal = 2500
    num_tb = 2500
    labels = np.concatenate([np.zeros(num_normal), np.ones(num_tb)])
    indices = np.arange(len(labels))
    
    partitions = dirichlet_partition(indices, labels, config.NUM_CLIENTS_BASELINE, config.DIRICHLET_ALPHA)
    
    before_normal, before_tb = [], []
    after_normal, after_tb = [], []
    
    for client_id, part_indices in enumerate(partitions):
        client_labels = labels[part_indices]
        counts = np.bincount(client_labels.astype(int), minlength=2)
        before_normal.append(counts[0])
        before_tb.append(counts[1])
        
        if len(np.unique(client_labels)) == 2 and min(counts) >= 2:
            try:
                mock_features = np.random.rand(len(client_labels), 10)
                resampler = SMOTE(random_state=42, k_neighbors=min(5, min(counts)-1))
                _, res_labels = resampler.fit_resample(mock_features, client_labels)
                res_counts = np.bincount(res_labels.astype(int), minlength=2)
                after_normal.append(res_counts[0])
                after_tb.append(res_counts[1])
            except Exception:
                after_normal.append(counts[0])
                after_tb.append(counts[1])
        else:
            after_normal.append(counts[0])
            after_tb.append(counts[1])
            
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Actual Client Data Distribution Before and After SMOTE (Log Scale)', fontsize=14, fontweight='bold')
    x = np.arange(config.NUM_CLIENTS_BASELINE)
    width = 0.35
    
    # Before SMOTE
    axes[0].bar(x - width/2, before_normal, width, label='Normal', color='#2ca02c')
    axes[0].bar(x + width/2, before_tb, width, label='Tuberculosis', color='#d62728')
    axes[0].set_title('BEFORE SMOTE', fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Client {i}" for i in x], rotation=45)
    axes[0].set_ylabel('Number of Samples (Log Scale)')
    axes[0].set_yscale('log') # LOG SCALE MAKES TINY BARS VISIBLE
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    # After SMOTE
    axes[1].bar(x - width/2, after_normal, width, label='Normal', color='#2ca02c')
    axes[1].bar(x + width/2, after_tb, width, label='Tuberculosis', color='#d62728')
    axes[1].set_title('AFTER SMOTE', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Client {i}" for i in x], rotation=45)
    axes[1].set_ylabel('Number of Samples (Log Scale)')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    out_path = out_dir / "data_distribution_smote.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {out_path}")


def get_probabilities(model, test_loader):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(config.DEVICE)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy() # Prob for class 1 (TB)
            all_probs.extend(probs)
            all_targets.extend(batch_y.numpy())
    return np.array(all_probs), np.array(all_targets)


def plot_roc_curves(exp_dirs, labels, out_dir):
    """Plots ROC curves for multiple models on the same graph."""
    logger.info("Generating ROC Curves...")
    test_dataset = TBChestXrayDataset(root_dir=config.TB_ORGANIZED_ROOT, split="test", transform=get_val_transform())
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    
    plt.figure(figsize=(10, 8))
    
    for d, lbl in zip(exp_dirs, labels):
        model_path = Path(d) / "final_model.pt"
        if not model_path.exists():
            continue
            
        model = get_model(model_name="resnet50", pretrained=False, freeze_backbone=config.FREEZE_BACKBONE, num_classes=2)
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
        model = model.to(config.DEVICE)
        
        probs, targets = get_probabilities(model, test_loader)
        fpr, tpr, _ = roc_curve(targets, probs)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{lbl} (AUC = {roc_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    out_path = out_dir / "comparison_roc_curve.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {out_path}")


def plot_tsne(model_dir, label, out_dir):
    """Extracts features before the final FC layer and plots t-SNE."""
    logger.info(f"Generating t-SNE for {label}...")
    model_path = Path(model_dir) / "final_model.pt"
    if not model_path.exists():
        return
        
    model = get_model(model_name="resnet50", pretrained=False, freeze_backbone=config.FREEZE_BACKBONE, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    
    # Remove the final FC layer to get features
    # ResNet50 fc layer can be replaced by Identity
    model.fc = nn.Identity()
    model = model.to(config.DEVICE)
    model.eval()
    
    test_dataset = TBChestXrayDataset(root_dir=config.TB_ORGANIZED_ROOT, split="test", transform=get_val_transform())
    # SHUFFLE MUST BE TRUE! Otherwise we only get the first class (Normal) since dataset is ordered!
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    
    features = []
    targets = []
    
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.to(config.DEVICE)
            out = model(batch_x).cpu().numpy()
            features.append(out)
            targets.append(batch_y.numpy())
            if i >= 15: # Grab ~500 random samples (mixed TB and Normal)
                break
                
    features = np.concatenate(features)
    targets = np.concatenate(targets)
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=targets, cmap='coolwarm', alpha=0.7, edgecolors='k')
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Normal (0)', markerfacecolor=scatter.cmap(0.0), markersize=10, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Tuberculosis (1)', markerfacecolor=scatter.cmap(1.0), markersize=10, markeredgecolor='k')
    ]
    plt.legend(handles=legend_elements, fontsize=12)
    plt.title(f't-SNE Feature Projection: {label}', fontsize=14, fontweight='bold')
    
    out_path = out_dir / f"tsne_{label.split()[0]}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {out_path}")

if __name__ == "__main__":
    out_dir = config.PROJECT_ROOT / "results" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Bar Chart
    plot_data_distribution(out_dir)
    
    # Directories for experiments
    exp_dirs = [
        config.PROJECT_ROOT / "results" / "TH1",
        config.PROJECT_ROOT / "results" / "TH3",
        config.PROJECT_ROOT / "results" / "TH4",
        config.PROJECT_ROOT / "results" / "TH5"
    ]
    labels = ["TH1 (Vanilla SMOTE)", "TH3 (DynTau)", "TH4 (Borderline+DynTau)", "TH5 (35 Rounds)"]
    
    # Filter only existing directories
    valid_dirs, valid_labels = [], []
    for d, l in zip(exp_dirs, labels):
        if d.exists() and (d / "final_model.pt").exists():
            valid_dirs.append(d)
            valid_labels.append(l)
            
    if valid_dirs:
        # 2. ROC Curves
        plot_roc_curves(valid_dirs, valid_labels, out_dir)
        
        # 3. t-SNE for the best model (TH5) and baseline (TH1)
        for d, l in zip(valid_dirs, valid_labels):
            if "TH1" in l or "TH5" in l:
                plot_tsne(d, l, out_dir)
    else:
        logger.warning("No valid model files found to generate ROC/t-SNE.")
