"""
Phase E: Federated Learning Simulation
Main runner for Flower-based FL simulation with adaptive aggregation.

Pipeline:
  1. Data preparation (split, organize, resize)
  2. Data partitioning (Dirichlet non-IID)
  3. Client factory with DP-enabled training
  4. Server strategy with FedAvg/FedSGD switching
  5. Simulation execution (flwr.simulation)
  6. Metrics logging and export
"""

import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional
import numpy as np

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar

import torch
from torch.utils.data import DataLoader
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.prepare_data import prepare_tb_dataset, verify_prepared_dataset
from src.data import TBChestXrayDataset, create_dataloaders, get_train_transform, get_val_transform
from src.partition import dirichlet_partition
from src.models import get_model
from src.client import FlowerClient
from src.strategy import AdaptiveAggregationStrategy
from src.evaluation import MetricsLogger, evaluate_model, print_metrics_summary
from src.balancing import balance_client_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_simulation_data():
    """
    Phase 1: Prepare TB dataset (split, organize, resize).
    
    Returns:
        Path to organized dataset
    """
    logger.info("=" * 70)
    logger.info("PHASE 1: DATA PREPARATION")
    logger.info("=" * 70)
    
    # Check if data already organized
    organized_dir = config.TB_ORGANIZED_ROOT
    if organized_dir.exists() and len(list(organized_dir.glob("*/*/*"))) > 100:
        logger.info(f"Data already organized at {organized_dir}. Skipping preparation.")
    else:
        logger.info(f"Preparing TB dataset from {config.TB_DATA_ROOT}")
        prepare_tb_dataset(
            raw_dir=config.TB_DATA_ROOT,
            output_dir=config.TB_ORGANIZED_ROOT,
            target_size=(224, 224),
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        )
    
    # Verify prepared dataset
    verify_prepared_dataset(config.TB_ORGANIZED_ROOT)
    logger.info("✓ Data preparation complete\n")
    
    return organized_dir


def partition_data(num_clients: int = config.NUM_CLIENTS_BASELINE):
    """
    Phase 2: Partition training data using Dirichlet(α=0.5).
    
    Args:
        num_clients: Number of clients
    
    Returns:
        List of client indices partitions
    """
    logger.info("=" * 70)
    logger.info("PHASE 2: DATA PARTITIONING")
    logger.info("=" * 70)
    
    # Load training dataset
    train_dataset = TBChestXrayDataset(
        root_dir=config.TB_ORGANIZED_ROOT,
        split="train",
        transform=get_train_transform(),
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Partitioning into {num_clients} clients with Dirichlet(α={config.DIRICHLET_ALPHA})")
    
    # Extract labels from dataset
    labels = np.array([label for _, label in train_dataset.samples], dtype=np.int32)
    
    # Partition data
    partitions = dirichlet_partition(
        dataset_indices=np.arange(len(train_dataset), dtype=np.int32),
        labels=labels,
        num_clients=num_clients,
        alpha=config.DIRICHLET_ALPHA,
        seed=42,
    )
    
    logger.info(f"✓ Partitioning complete: {len(partitions)} clients\n")
    
    return partitions, train_dataset


def create_client_fn(
    partitions: List[np.ndarray],
    train_dataset: TBChestXrayDataset,
    model_name: str = "resnet50",
    dp_enabled: bool = config.DP_ENABLED,
) -> Callable[[str], FlowerClient]:
    """
    Create client factory function for Flower simulation.
    Apply SMOTE balancing to each client's training data.
    
    Args:
        partitions: List of client data partitions
        train_dataset: Full training dataset
        model_name: Model architecture (resnet50/vgg16)
        dp_enabled: Enable differential privacy
    
    Returns:
        Callable that creates a FlowerClient for a given client_id
    """
    
    # Store synthetic images cache per client
    synthetic_cache = {}
    
    def client_fn(cid: str) -> fl.client.Client:
        """Create a client for the given client ID."""
        nonlocal synthetic_cache
        
        cid_int = int(cid)
        
        # Get client data indices
        client_indices = partitions[cid_int]
        
        # Get image paths and labels for this client
        client_samples = [train_dataset.samples[i] for i in client_indices]
        client_paths = np.array([path for path, _ in client_samples])
        client_labels = np.array([label for _, label in client_samples])
        
        # Handle empty datasets
        if len(client_paths) == 0:
            logger.warning(f"Client {cid}: No training data assigned (empty partition)")
            balanced_paths = []
            balanced_labels = []
            balanced_indices = np.array([], dtype=int)
            subset = torch.utils.data.Subset(train_dataset, balanced_indices)
        else:
            # Apply SMOTE to balance client's data
            logger.info(f"Client {cid}: Applying SMOTE to balance training data...")
            balanced_paths, balanced_labels, synthetic_images = balance_client_dataset(
                client_paths.tolist(),
                client_labels.tolist(),
                random_state=42 + cid_int  # Different seed per client
            )
            
            # Cache synthetic images for this client
            if synthetic_images:
                synthetic_cache[cid_int] = synthetic_images
                logger.info(f"Client {cid}: Generated {len(synthetic_images)} synthetic images via SMOTE")
            
            logger.info(f"Client {cid}: Original samples: {len(client_paths)}, Balanced samples: {len(balanced_paths)}")
            
            # Create subset with balanced indices
            # For original samples, use partition indices
            # For synthetic samples, they'll be handled separately
            original_count = len(client_paths)
            balanced_indices = list(client_indices)  # Start with original indices
            
            # Add dummy indices for synthetic samples (they'll be intercepted in __getitem__)
            num_synthetic = len(synthetic_images)
            for i in range(num_synthetic):
                # Use negative indices as markers for synthetic samples
                balanced_indices.append(-(cid_int * 10000 + i + 1))  # Negative index to distinguish synthetic
            
            balanced_indices = np.array(balanced_indices, dtype=int)
            
            # Create a wrapper dataset that handles synthetic samples
            class BalancedClientDataset(torch.utils.data.Dataset):
                def __init__(self, base_dataset, balanced_idx, synthetic_dict, client_id):
                    self.base_dataset = base_dataset
                    self.balanced_idx = balanced_idx
                    self.synthetic_dict = synthetic_dict
                    self.client_id = client_id
                    self.original_count = original_count
                    
                def __len__(self):
                    return len(self.balanced_idx)
                
                def __getitem__(self, idx):
                    orig_idx = self.balanced_idx[idx]
                    
                    # Check if this is a synthetic sample (negative index)
                    if orig_idx < 0:
                        synthetic_idx = -orig_idx - 1
                        if self.client_id in self.synthetic_dict and synthetic_idx in self.synthetic_dict[self.client_id]:
                            # Get synthetic image array
                            img_array = self.synthetic_dict[self.client_id][synthetic_idx]
                            img = Image.fromarray(img_array)
                            label = 1  # Synthetic samples are always TB (minority class)
                        else:
                            raise IndexError(f"Synthetic image not found: client={self.client_id}, idx={synthetic_idx}")
                    else:
                        # Original sample from dataset
                        img, label = self.base_dataset[orig_idx]
                        if not isinstance(img, torch.Tensor):
                            img = get_train_transform()(img)
                    
                    return img, label
            
            subset = BalancedClientDataset(train_dataset, balanced_indices, synthetic_cache, cid_int)
        
        # Create dataloaders
        if len(subset) == 0:
            train_loader = DataLoader(
                subset,
                batch_size=config.BATCH_SIZE,
                sampler=torch.utils.data.SequentialSampler(subset),
                num_workers=0,
            )
        else:
            train_loader = DataLoader(
                subset,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=0,
            )
        
        val_loader = DataLoader(
            TBChestXrayDataset(
                root_dir=config.TB_ORGANIZED_ROOT,
                split="val",
                transform=get_val_transform(),
            ),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
        
        # Create model
        model = get_model(
            model_name=model_name,
            pretrained=True,
            freeze_backbone=True,
            num_classes=2,
        )
        
        # Create client
        client = FlowerClient(
            client_id=cid_int,
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            learning_rate=config.LEARNING_RATE,
            device=config.DEVICE,
            dp_enabled=dp_enabled,
        )
        
        logger.debug(f"Client {cid}: {len(subset)} balanced samples assigned")
        return client
    
    return client_fn


def run_simulation(
    num_clients: int = config.NUM_CLIENTS_BASELINE,
    num_rounds: int = 10,
    model_name: str = "resnet50",
    dp_enabled: bool = config.DP_ENABLED,
    min_available_clients: int = None,
    min_fit_clients: int = None,
    min_evaluate_clients: int = None,
):
    """
    Main simulation runner using Flower.
    
    Args:
        num_clients: Total number of clients to simulate
        num_rounds: Number of FL rounds
        model_name: Model architecture (resnet50/vgg16)
        dp_enabled: Enable differential privacy
        min_available_clients: Min clients needed (default: 90% of num_clients)
        min_fit_clients: Min clients for training per round (default: 100% of num_clients)
        min_evaluate_clients: Min clients for evaluation per round (default: 100% of num_clients)
    """
    logger.info("=" * 70)
    logger.info("FEDERATED LEARNING SIMULATION")
    logger.info("=" * 70)
    
    if min_available_clients is None:
        min_available_clients = int(0.9 * num_clients)
    if min_fit_clients is None:
        min_fit_clients = num_clients
    if min_evaluate_clients is None:
        min_evaluate_clients = num_clients
    
    logger.info(f"Config:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Clients: {num_clients}")
    logger.info(f"  Rounds: {num_rounds}")
    logger.info(f"  DP Enabled: {dp_enabled}")
    logger.info(f"  Batch Size: {config.BATCH_SIZE}")
    logger.info(f"  Learning Rate: {config.LEARNING_RATE}")
    logger.info(f"  Dirichlet α: {config.DIRICHLET_ALPHA}")
    logger.info(f"  Adaptive τ: {config.TAU_STATIC}\n")
    
    # Step 1: Prepare data
    prepare_simulation_data()
    
    # Step 2: Partition data
    partitions, train_dataset = partition_data(num_clients)
    
    # Step 3: Create client factory
    client_fn = create_client_fn(
        partitions=partitions,
        train_dataset=train_dataset,
        model_name=model_name,
        dp_enabled=dp_enabled,
    )
    
    # Step 4: Create strategy
    logger.info("=" * 70)
    logger.info("PHASE 3: STRATEGY INITIALIZATION")
    logger.info("=" * 70)
    
    strategy = AdaptiveAggregationStrategy(
        fraction_fit=1.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
    )
    
    logger.info("✓ Strategy initialized\n")
    
    # Step 5: Run simulation
    logger.info("=" * 70)
    logger.info("PHASE 4: SIMULATION EXECUTION")
    logger.info("=" * 70 + "\n")
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5: EVALUATION & LOGGING")
    logger.info("=" * 70)
    
    # Step 6: Evaluate on test set
    test_dataset = TBChestXrayDataset(
        root_dir=config.TB_ORGANIZED_ROOT,
        split="test",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    
    # Load global model weights and evaluate
    final_model = get_model(
        model_name=model_name,
        pretrained=True,
        freeze_backbone=False,  # Thaw for full fine-tune evaluation
        num_classes=2,
    )
    
    # Create a dummy client to deserialize final parameters
    test_client = client_fn("0")
    
    # Get final parameters from strategy
    if strategy.final_parameters is not None:
        test_client.set_parameters(
            fl.common.parameters_to_ndarrays(strategy.final_parameters)
        )
    else:
        logger.warning("No final parameters found from strategy")
    
    # Handle state_dict loaded from DP-wrapped model (has "_module" prefix)
    state_dict = test_client.model.state_dict()
    
    # Remove "_module" prefix if present (from Opacus PrivacyEngine wrapper)
    corrected_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_module."):
            corrected_state_dict[key[8:]] = value  # Remove "_module."
        else:
            corrected_state_dict[key] = value
    
    final_model.load_state_dict(corrected_state_dict, strict=False)
    final_model = final_model.to(config.DEVICE)
    
    test_loss, test_accuracy = evaluate_model(
        model=final_model,
        dataloader=test_loader,
        device=config.DEVICE,
    )
    
    logger.info(f"Test Loss: {test_loss:.6f}")
    logger.info(f"Test Accuracy: {test_accuracy:.6f}")
    
    # Step 7: Log metrics
    metrics_logger = MetricsLogger(output_dir=config.RESULTS_DIR)
    
    # Log training rounds
    divergence_history = strategy.get_metrics()["divergence_history"]
    algorithm_history = strategy.get_metrics()["algorithm_history"]
    loss_history = strategy.get_metrics()["loss_history"]
    
    for r in range(len(loss_history)):
        metrics_logger.log_round(
            round_num=r + 1,
            loss=loss_history[r] if r < len(loss_history) else 0,
            divergence=divergence_history[r] if r < len(divergence_history) else 0,
            tau=config.TAU_STATIC,
            algorithm=algorithm_history[r] if r < len(algorithm_history) else "N/A",
            num_clients=num_clients,
            split="train",
        )
    
    metrics_logger.log_round(
        round_num=num_rounds + 1,
        loss=test_loss,
        accuracy=test_accuracy,
        split="test",
    )
    
    metrics_logger.save_to_csv()
    
    test_results = {
        "Test Loss": test_loss,
        "Test Accuracy": test_accuracy,
        "Model": model_name,
        "Clients": num_clients,
        "Rounds": num_rounds,
        "DP Enabled": dp_enabled,
    }
    metrics_logger.log_test_results(test_results)
    
    # Print summary
    print_metrics_summary(
        metrics={
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy,
            "Training Rounds": num_rounds,
            "Clients": num_clients,
            "DP Enabled": dp_enabled,
        },
        title="SIMULATION COMPLETE - FINAL METRICS"
    )
    
    logger.info(f"✓ Results saved to {config.RESULTS_DIR}")
    logger.info("=" * 70)
    
    return history, test_loss, test_accuracy


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run FL simulation baseline")
    parser.add_argument(
        "--num-clients",
        type=int,
        default=config.NUM_CLIENTS_BASELINE,
        help=f"Number of clients (default: {config.NUM_CLIENTS_BASELINE})"
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=10,
        help="Number of FL rounds (default: 10)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "vgg16"],
        help="Model architecture (default: resnet50)"
    )
    parser.add_argument(
        "--dp",
        action="store_true",
        default=False,
        help="Enable differential privacy"
    )
    parser.add_argument(
        "--no-dp",
        action="store_true",
        help="Disable differential privacy"
    )
    
    args = parser.parse_args()
    # DP is enabled by config default, unless --no-dp is specified
    dp_enabled = config.DP_ENABLED and not args.no_dp
    
    # Run simulation
    history, test_loss, test_acc = run_simulation(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        model_name=args.model,
        dp_enabled=dp_enabled,
    )
    
    print(f"\n✓ Simulation complete!")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")
