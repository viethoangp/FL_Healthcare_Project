"""
Visualization: Generate charts from FL simulation results.
- Bar charts: Loss per round, Metrics summary, Divergence+Tau
- Confusion Matrix from saved final_model.pt
- Comparison chart: All experiments side by side

Usage:
    # Plot single experiment
    python src/visualize.py --label "TH1"

    # Plot single experiment from named subfolder
    python src/visualize.py --results-dir results/TH1 --label "TH1 - Baseline"

    # Compare all experiments side by side
    python src/visualize.py --compare results/TH1 results/TH2 results/TH3 results/TH4 results/TH5 \\
        --compare-labels "TH1 Baseline" "TH2 BorderSMOTE" "TH3 DynTau" "TH4 Both" "TH5 50rounds"
"""

import argparse
import csv
import json
import logging
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.data import TBChestXrayDataset, get_val_transform, custom_collate_fn
from src.models import get_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Color palette ──────────────────────────────────────────────────────────────
CLR_FEDAVG      = "#4C9BE8"
CLR_FEDSGD      = "#F28B3B"
CLR_LOSS        = "#E85C5C"
CLR_ACC         = "#50C878"
CLR_TAU         = "#A78BFA"
CLR_BG          = "#F7F9FC"
CLR_GRID        = "#E0E4EA"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD ROUNDS.CSV
# ══════════════════════════════════════════════════════════════════════════════
def load_rounds_csv(csv_path: Path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    train_rows = [r for r in rows if r["split"] == "train"]

    rounds      = [int(r["round"])             for r in train_rows]
    losses      = [float(r["loss"])            for r in train_rows]
    divergences = [float(r["divergence"]) if r["divergence"] else None for r in train_rows]
    taus        = [float(r["tau"])        if r["tau"]        else None for r in train_rows]
    algorithms  = [r["algorithm"]              for r in train_rows]
    epsilons    = [float(r["epsilon"]) if r["epsilon"] else None for r in train_rows]

    return rounds, losses, divergences, taus, algorithms, epsilons


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LOAD TEST_RESULTS.TXT
# ══════════════════════════════════════════════════════════════════════════════
def load_test_results(txt_path: Path):
    results = {}
    with open(txt_path) as f:
        for line in f:
            if ":" in line and not line.startswith("="):
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                try:
                    results[key] = float(val)
                except ValueError:
                    results[key] = val
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PLOT: LOSS PER ROUND (bar chart)
# ══════════════════════════════════════════════════════════════════════════════
def plot_loss_per_round(rounds, losses, algorithms, out_path: Path, label: str = ""):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=CLR_BG)
    ax.set_facecolor(CLR_BG)

    colors = [CLR_FEDSGD if a == "FedSGD" else CLR_FEDAVG for a in algorithms]
    bars = ax.bar(rounds, losses, color=colors, edgecolor="white", linewidth=0.8, zorder=3)

    # Value labels on bars
    for bar, val in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#444")

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    title = f"Training Loss per Round"
    if label:
        title += f"  [{label}]"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(rounds)
    ax.yaxis.grid(True, color=CLR_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(color=CLR_FEDAVG, label="FedAvg"),
        mpatches.Patch(color=CLR_FEDSGD, label="FedSGD"),
    ]
    ax.legend(handles=legend_patches, framealpha=0.85, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PLOT: METRICS SUMMARY BAR CHART (Accuracy / Precision / Recall / F1)
# ══════════════════════════════════════════════════════════════════════════════
def plot_metrics_summary(test_results: dict, out_path: Path, label: str = ""):
    metrics = {
        "Accuracy":     test_results.get("Test Accuracy", 0),
        "TB Precision": test_results.get("TB Precision",  0),
        "TB Recall":    test_results.get("TB Recall",     0),
        "TB F1":        test_results.get("TB F1",         0),
        "Normal Recall":test_results.get("Normal Recall", 0),
        "Normal F1":    test_results.get("Normal F1",     0),
    }

    names  = list(metrics.keys())
    values = [v * 100 for v in metrics.values()]   # convert to %
    colors = ["#4C9BE8", "#50C878", "#F28B3B", "#E85C5C", "#A78BFA", "#F9C74F"]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=CLR_BG)
    ax.set_facecolor(CLR_BG)

    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333")

    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)", fontsize=12)
    title = "Test Set Performance Metrics"
    if label:
        title += f"  [{label}]"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.yaxis.grid(True, color=CLR_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=15, ha="right", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PLOT: DIVERGENCE + TAU per round
# ══════════════════════════════════════════════════════════════════════════════
def plot_divergence(rounds, divergences, taus, algorithms, out_path: Path, label: str = ""):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=CLR_BG)
    ax.set_facecolor(CLR_BG)

    div_vals = [d if d is not None else 0 for d in divergences]
    colors = [CLR_FEDSGD if a == "FedSGD" else CLR_FEDAVG for a in algorithms]
    ax.bar(rounds, div_vals, color=colors, alpha=0.7, edgecolor="white", zorder=3, label="Divergence δ")

    tau_vals = [t for t in taus if t is not None]
    tau_rounds = [r for r, t in zip(rounds, taus) if t is not None]
    if tau_vals:
        ax.plot(tau_rounds, tau_vals, color=CLR_TAU, linewidth=2.5,
                marker="D", markersize=6, zorder=4, label="Threshold τ (adaptive)")

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Divergence / τ", fontsize=12)
    title = "Divergence δ and Adaptive Threshold τ per Round"
    if label:
        title += f"  [{label}]"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(rounds)
    ax.yaxis.grid(True, color=CLR_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(color=CLR_FEDAVG, label="FedAvg rounds"),
        mpatches.Patch(color=CLR_FEDSGD, label="FedSGD rounds"),
        plt.Line2D([0], [0], color=CLR_TAU, linewidth=2.5, marker="D", markersize=6, label="τ (adaptive)"),
    ]
    ax.legend(handles=legend_patches, framealpha=0.85, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PLOT: CONFUSION MATRIX (run inference on test set)
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(out_path: Path, label: str = ""):
    """Load the final global model and run inference on the test set."""
    logger.info("Loading model for confusion matrix inference...")

    # Look for saved model weights
    model_path = config.RESULTS_DIR / "final_model.pt"
    if not model_path.exists():
        logger.warning(
            "final_model.pt not found in results/. "
            "Re-run simulation with save_model=True or use --skip-cm flag."
        )
        return False

    model = get_model(
        model_name="resnet50",
        pretrained=False,
        freeze_backbone=config.FREEZE_BACKBONE,
        num_classes=2,
    )
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(config.DEVICE)

    test_dataset = TBChestXrayDataset(
        root_dir=config.TB_ORGANIZED_ROOT,
        split="test",
        transform=get_val_transform(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
    )

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(config.DEVICE)
            logits = model(batch_x)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())

    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=CLR_BG)
    ax.set_facecolor(CLR_BG)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    title = "Confusion Matrix — Test Set"
    if label:
        title += f"\n[{label}]"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PLOT: COMPARISON ACROSS ALL EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════
COMPARE_COLORS = ["#4C9BE8", "#50C878", "#F28B3B", "#E85C5C", "#A78BFA", "#F9C74F"]

def plot_comparison(exp_dirs: list, labels: list, out_dir: Path, skip_loss_curve: bool = False):
    """
    Side-by-side bar charts comparing key metrics across all experiments.
    Generates:
      - comparison_accuracy.png  : Accuracy + TB Recall + TB F1 grouped bars
      - comparison_loss.png      : Final test loss per experiment
      - comparison_loss_curve.png: Loss convergence curves overlaid
    """
    # Load test results for each experiment
    all_results = []
    valid_labels = []
    for d, lbl in zip(exp_dirs, labels):
        txt = Path(d) / "test_results.txt"
        if not txt.exists():
            logger.warning(f"Skipping {lbl}: test_results.txt not found in {d}")
            continue
        all_results.append(load_test_results(txt))
        valid_labels.append(lbl)

    if not all_results:
        logger.error("No valid experiment results found for comparison.")
        return

    n = len(valid_labels)
    x = np.arange(n)
    width = 0.22

    # ── Comparison: Accuracy / TB Recall / TB F1 ─────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, n * 2.5), 6), facecolor=CLR_BG)
    ax.set_facecolor(CLR_BG)

    metric_keys = ["Test Accuracy", "TB Recall", "TB F1", "Normal Recall"]
    metric_names = ["Accuracy", "TB Recall", "TB F1", "Normal Recall"]
    metric_colors = ["#4C9BE8", "#F28B3B", "#E85C5C", "#50C878"]

    for i, (key, name, clr) in enumerate(zip(metric_keys, metric_names, metric_colors)):
        vals = [r.get(key, 0) * 100 for r in all_results]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=clr,
                      edgecolor="white", linewidth=0.7, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=7, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels, fontsize=10, rotation=10, ha="right")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Experiment Comparison — Accuracy & Recall Metrics", fontsize=14,
                 fontweight="bold", pad=12)
    ax.yaxis.grid(True, color=CLR_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = out_dir / "comparison_accuracy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")

    # ── Comparison: Test Loss ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, n * 2), 5), facecolor=CLR_BG)
    ax.set_facecolor(CLR_BG)
    losses = [r.get("Test Loss", 0) for r in all_results]
    clrs = [COMPARE_COLORS[i % len(COMPARE_COLORS)] for i in range(n)]
    bars = ax.bar(valid_labels, losses, color=clrs, edgecolor="white", linewidth=0.8, zorder=3)
    for bar, val in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333")
    ax.set_ylabel("Test Loss", fontsize=12)
    ax.set_title("Experiment Comparison — Final Test Loss", fontsize=14,
                 fontweight="bold", pad=12)
    ax.yaxis.grid(True, color=CLR_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=10, ha="right", fontsize=10)
    plt.tight_layout()
    out = out_dir / "comparison_loss.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")

    # ── Comparison: Loss convergence curves ───────────────────────────────────
    if not skip_loss_curve:
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=CLR_BG)
        ax.set_facecolor(CLR_BG)
        for i, (d, lbl) in enumerate(zip(exp_dirs, valid_labels)):
            csv_p = Path(d) / "rounds.csv"
            if not csv_p.exists():
                continue
            rounds, losses_curve, _, _, _, _ = load_rounds_csv(csv_p)
            clr = COMPARE_COLORS[i % len(COMPARE_COLORS)]
            ax.plot(rounds, losses_curve, marker="o", markersize=5, linewidth=2,
                    color=clr, label=lbl, zorder=3)
                    
        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("Training Loss", fontsize=12)
        ax.set_title("Loss Convergence Comparison Across Experiments", fontsize=14,
                     fontweight="bold", pad=12)
        ax.yaxis.grid(True, color=CLR_GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(fontsize=10, framealpha=0.85)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        out = out_dir / "comparison_loss_curve.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {out}")

    # ── Execution time comparison ─────────────────────────────────────────────
    exec_times = [r.get("Execution time (s)", None) for r in all_results]
    if any(t is not None for t in exec_times):
        fig, ax = plt.subplots(figsize=(max(8, n * 2), 5), facecolor=CLR_BG)
        ax.set_facecolor(CLR_BG)
        valid_times = [(lbl, t) for lbl, t in zip(valid_labels, exec_times) if t is not None]
        lbls_t, vals_t = zip(*valid_times)
        clrs_t = [COMPARE_COLORS[i % len(COMPARE_COLORS)] for i in range(len(lbls_t))]
        bars = ax.bar(lbls_t, vals_t, color=clrs_t, edgecolor="white", linewidth=0.8, zorder=3)
        for bar, val in zip(bars, vals_t):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{val:.0f}s", ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color="#333")
        ax.set_ylabel("Execution Time (s)", fontsize=12)
        ax.set_title("Experiment Comparison — Execution Time", fontsize=14,
                     fontweight="bold", pad=12)
        ax.yaxis.grid(True, color=CLR_GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=10, ha="right", fontsize=10)
        plt.tight_layout()
        out = out_dir / "comparison_time.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {out}")

    print(f"\n✓ Comparison charts saved to: {out_dir}")
    print(f"  - comparison_accuracy.png")
    print(f"  - comparison_loss.png")
    if not skip_loss_curve:
        print(f"  - comparison_loss_curve.png")
    print(f"  - comparison_time.png")


def plot_comparison_confusion_matrices(exp_dirs: list, labels: list, out_dir: Path):
    """
    Load the final_model.pt for each experiment, run inference on the test set,
    and plot a side-by-side grid of confusion matrices.
    """
    logger.info("Generating side-by-side Confusion Matrices...")
    
    test_dataset = TBChestXrayDataset(
        root_dir=config.TB_ORGANIZED_ROOT,
        split="test",
        transform=get_val_transform(),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=custom_collate_fn
    )

    n = len(exp_dirs)
    if n == 4:
        cols = 2
        rows = 2
    else:
        cols = min(3, n)
        rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), facecolor=CLR_BG)
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (d, lbl) in enumerate(zip(exp_dirs, labels)):
        ax = axes[i]
        model_path = Path(d) / "final_model.pt"
        if not model_path.exists():
            ax.text(0.5, 0.5, f"Missing final_model.pt\nfor {lbl}", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(lbl, fontweight="bold")
            ax.axis('off')
            continue

        model = get_model(model_name="resnet50", pretrained=False, freeze_backbone=config.FREEZE_BACKBONE, num_classes=2)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model = model.to(config.DEVICE)

        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(config.DEVICE)
                preds = model(batch_x).argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(batch_y.numpy())

        cm = confusion_matrix(all_targets, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.CLASS_NAMES)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(lbl, fontsize=12, fontweight="bold")
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    out = out_dir / "comparison_confusion_matrices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")
    print(f"  - comparison_confusion_matrices.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Visualize FL simulation results")
    parser.add_argument("--results-dir", type=str, default=str(config.RESULTS_DIR),
                        help="Directory containing rounds.csv and test_results.txt")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory for plots (default: <results-dir>/plots)")
    parser.add_argument("--label", type=str, default="",
                        help="Label to add to chart titles (e.g. 'TH1 Baseline')")
    parser.add_argument("--skip-cm", action="store_true",
                        help="Skip generating the confusion matrix")
    parser.add_argument("--skip-loss-curve", action="store_true",
                        help="Skip generating the comparison loss curve")
    
    # Optional arguments for comparison mode
    parser.add_argument("--compare", nargs="+", metavar="DIR",
                        help="List of results directories to compare side by side")
    parser.add_argument("--compare-labels", nargs="+", metavar="LABEL",
                        help="Labels for each experiment in --compare (same order)")
    args = parser.parse_args()

    # ── COMPARE MODE ──────────────────────────────────────────────────────────
    if args.compare:
        labels = args.compare_labels or [Path(d).name for d in args.compare]
        out_dir = Path(args.out_dir) if args.out_dir else config.PROJECT_ROOT / "results" / "comparison"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison(args.compare, labels, out_dir, args.skip_loss_curve)
        if not args.skip_cm:
            plot_comparison_confusion_matrices(args.compare, labels, out_dir)
        return

    # ── SINGLE EXPERIMENT MODE ────────────────────────────────────────────────
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "rounds.csv"
    txt_path = results_dir / "test_results.txt"

    if not csv_path.exists():
        logger.error(f"rounds.csv not found at {csv_path}. Run simulation first.")
        sys.exit(1)
    if not txt_path.exists():
        logger.error(f"test_results.txt not found at {txt_path}. Run simulation first.")
        sys.exit(1)

    rounds, losses, divergences, taus, algorithms, epsilons = load_rounds_csv(csv_path)
    test_results = load_test_results(txt_path)
    lbl = args.label

    plot_loss_per_round(rounds, losses, algorithms,
                        out_dir / "loss_per_round.png", label=lbl)
    plot_metrics_summary(test_results,
                         out_dir / "metrics_summary.png", label=lbl)
    plot_divergence(rounds, divergences, taus, algorithms,
                    out_dir / "divergence_tau.png", label=lbl)
    if not args.skip_cm:
        # point confusion matrix at the correct results dir
        config.RESULTS_DIR = results_dir
        plot_confusion_matrix(out_dir / "confusion_matrix.png", label=lbl)

    print(f"\n✓ Charts saved to: {out_dir}")
    print(f"  - loss_per_round.png")
    print(f"  - metrics_summary.png")
    print(f"  - divergence_tau.png")
    if not args.skip_cm:
        print(f"  - confusion_matrix.png  (requires final_model.pt in results dir)")


if __name__ == "__main__":
    main()

