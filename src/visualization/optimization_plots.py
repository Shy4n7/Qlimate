"""
Optimization behavior plots: NN loss curves and VQC convergence.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def plot_nn_loss_curves(
    train_losses: list,
    val_losses: list,
    save_path: Path,
    early_stopping_epoch: Optional[int] = None,
) -> None:
    """Neural network train/val loss curves with early stopping annotation."""
    epochs = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, color="#3b82f6", linewidth=1.8, label="Train loss")
    ax.plot(epochs, val_losses, color="#f97316", linewidth=1.8, label="Val loss", linestyle="--")

    if early_stopping_epoch is not None:
        ax.axvline(early_stopping_epoch, color="red", linestyle=":", linewidth=1.2,
                   label=f"Early stop (epoch {early_stopping_epoch})")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Cross-entropy loss", fontsize=11)
    ax.set_title("Neural Network — Training Convergence", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(1, len(train_losses))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved NN loss curves: {save_path}")


def plot_vqc_no_convergence(
    vqc_meta: dict,
    save_path: Path,
) -> None:
    """
    Honest visualization of VQC training failure.
    Shows an empty loss plot with an annotated explanation.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, vqc_meta.get("maxiter_configured", 150))
    ax.set_ylim(0, 2)
    ax.set_facecolor("#f8f8f8")
    ax.set_xlabel("Optimizer iteration", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("VQC — Optimization Behavior", fontsize=13)
    ax.grid(alpha=0.2)

    explanation = (
        "0 optimizer iterations completed\n\n"
        f"Training time: {vqc_meta.get('training_time', 0):.1f}s\n"
        f"Configured max iterations: {vqc_meta.get('maxiter_configured', 150)}\n"
        f"Parameters: {vqc_meta.get('n_params', 24)}\n\n"
        "Cause: COBYLA callback interface incompatibility\n"
        "in qiskit-machine-learning 0.9.0.\n"
        "VQC.fit() completed but optimizer\n"
        "never updated circuit parameters.\n"
        "Predictions reflect unoptimized circuit."
    )
    ax.text(0.5, 0.5, explanation,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                      edgecolor="#e5e7eb", linewidth=1.5),
            color="#374151")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved VQC no-convergence plot: {save_path}")


def plot_combined_optimization(
    nn_meta: dict,
    vqc_meta: dict,
    save_path: Path,
) -> None:
    """Side-by-side: NN loss curves (left) and VQC status (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # Left: NN
    train_losses = nn_meta.get("train_losses", [])
    val_losses = nn_meta.get("val_losses", [])
    epochs = list(range(1, len(train_losses) + 1))
    ax1.plot(epochs, train_losses, color="#3b82f6", linewidth=1.8, label="Train")
    ax1.plot(epochs, val_losses, color="#f97316", linewidth=1.8, linestyle="--", label="Val")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Neural Network — Loss Convergence", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(1, len(train_losses))

    # Right: VQC status
    ax2.set_xlim(0, 150)
    ax2.set_ylim(0, 2)
    ax2.set_facecolor("#f8f8f8")
    ax2.set_xlabel("Optimizer iteration", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.set_title("VQC (COBYLA) — Optimization", fontsize=12)
    ax2.grid(alpha=0.2)
    text = (
        "0 of 150 iterations completed\n\n"
        "COBYLA callback interface\n"
        "incompatibility in qiskit-ml 0.9.0.\n\n"
        f"Wall time: {vqc_meta.get('training_time', 198):.0f}s\n"
        "Parameters: never updated."
    )
    ax2.text(0.5, 0.5, text, transform=ax2.transAxes,
             ha="center", va="center", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor="#d1d5db"),
             color="#374151")

    fig.suptitle("Optimization Behavior: Classical vs Quantum", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved combined optimization plot: {save_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    import joblib
    from pathlib import Path as P

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    nn_meta = joblib.load("results/models/neural_network_meta.pkl")
    vqc_meta = joblib.load("results/models/vqc_meta.pkl")
    vqc_meta["maxiter_configured"] = 150

    out_dir = Path("results/figures")
    plot_nn_loss_curves(nn_meta["train_losses"], nn_meta["val_losses"],
                        out_dir / "nn_loss_curves.png")
    plot_vqc_no_convergence(vqc_meta, out_dir / "vqc_no_convergence.png")
    plot_combined_optimization(nn_meta, vqc_meta, out_dir / "optimization_comparison.png")
