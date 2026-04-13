"""
Kernel matrix visualization for QSVC.
Generates heatmap of the quantum kernel matrix (simulator).
"""

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def plot_kernel_matrix_heatmap(
    kernel_matrix: np.ndarray,
    save_path: Path,
    title: str = "Quantum Kernel Matrix (Simulator)",
) -> None:
    """Seaborn-style heatmap of the QSVC kernel matrix."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(kernel_matrix, cmap="magma", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Kernel value (fidelity²)")
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Training sample index", fontsize=11)
    ax.set_ylabel("Training sample index", fontsize=11)
    n = kernel_matrix.shape[0]
    ticks = np.arange(0, n, max(1, n // 5))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    stats_text = (
        f"mean={kernel_matrix.mean():.3f}  "
        f"std={kernel_matrix.std():.3f}  "
        f"max={kernel_matrix.max():.3f}"
    )
    ax.text(0.5, -0.12, stats_text, transform=ax.transAxes,
            ha="center", fontsize=9, color="gray")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved kernel heatmap: {save_path}")


def kernel_matrix_from_model(
    X_train_q: np.ndarray,
    n_vis: int = 50,
    config: dict = None,
) -> np.ndarray:
    """Compute kernel matrix using StatevectorSampler (no hardware)."""
    from qiskit.primitives import StatevectorSampler
    from qiskit_machine_learning.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from src.models.quantum import QuantumModelTrainer

    if config is None:
        import yaml
        with open("config/config.yaml") as f:
            config = yaml.safe_load(f)

    trainer = QuantumModelTrainer(config)
    feature_map = trainer._build_feature_map()
    fidelity = ComputeUncompute(sampler=StatevectorSampler())
    kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

    X_vis = X_train_q[:n_vis]
    logger.info(f"Computing {n_vis}x{n_vis} kernel matrix...")
    K = kernel.evaluate(X_vis, X_vis)
    return np.clip(K, 0, 1)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    import yaml
    import joblib
    from src.features.engineering import load_splits

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(open("config/config.yaml"))
    _, quantum = load_splits(cfg)
    K = kernel_matrix_from_model(quantum["X_train_q"], n_vis=50, config=cfg)
    out = Path(cfg["paths"]["figures"]) / "kernel_matrix.png"
    plot_kernel_matrix_heatmap(K, out)
