"""
Feature compression diagram: raw features → PCA → qubit encoding.
Visualizes the data transformation pipeline for quantum ML.
"""

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

RAW_FEATURES = [
    "T2M (temperature)",
    "QV2M (humidity)",
    "U10M (wind E)",
    "V10M (wind N)",
    "PS (surface pressure)",
    "SLP (sea level pressure)",
    "PRECTOT (precipitation)",
    "EVAP (evaporation)",
    "CLDTOT (cloud fraction)",
]

PCA_LABELS = [
    "PC1  51.3% var",
    "PC2  21.1% var",
    "PC3  16.9% var",
    "PC4   4.9% var",
]

QUBIT_LABELS = [
    "Qubit 0  |ψ₀⟩",
    "Qubit 1  |ψ₁⟩",
    "Qubit 2  |ψ₂⟩",
    "Qubit 3  |ψ₃⟩",
]


def _box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=9):
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color="white" if facecolor not in ("white", "#f9fafb") else "#111827")


def plot_feature_compression_diagram(
    pca_variance: list,
    save_path: Path,
) -> None:
    """
    Three-column diagram:
      Col 1: 9 raw climate features
      Col 2: StandardScaler + PCA → 4 components
      Col 3: MinMaxScaler [0,π] + ZZFeatureMap → 4 qubits
    """
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    # Column X positions
    x_raw, x_pca, x_q = 2.2, 6.5, 10.8

    # Column headers
    for x, label, color in [
        (x_raw, "Raw Features\n(9 variables)", "#1d4ed8"),
        (x_pca, "PCA Compression\n(4 components, 94.1% variance)", "#6d28d9"),
        (x_q, "Qubit Encoding\n(ZZFeatureMap, 4 qubits)", "#0f766e"),
    ]:
        ax.text(x, 7.1, label, ha="center", va="center", fontsize=10,
                fontweight="bold", color=color)

    # Raw features (9 boxes)
    n_raw = len(RAW_FEATURES)
    raw_ys = np.linspace(6.0, 0.8, n_raw)
    for feat, y in zip(RAW_FEATURES, raw_ys):
        _box(ax, x_raw, y, 3.2, 0.42, feat, "#1d4ed8", "#1e40af", fontsize=8)

    # PCA boxes (4)
    pca_ys = np.linspace(5.2, 1.6, 4)
    for lbl, y in zip(PCA_LABELS, pca_ys):
        _box(ax, x_pca, y, 3.0, 0.45, lbl, "#6d28d9", "#5b21b6", fontsize=9)

    # Qubit boxes (4)
    q_ys = np.linspace(5.2, 1.6, 4)
    for lbl, y in zip(QUBIT_LABELS, q_ys):
        _box(ax, x_q, y, 2.8, 0.45, lbl, "#0f766e", "#0d9488", fontsize=9)

    # Arrows: all raw → PCA funnel
    for ry in raw_ys:
        ax.annotate("", xy=(x_pca - 1.5, pca_ys[0] + (pca_ys[-1] - pca_ys[0]) / 2),
                    xytext=(x_raw + 1.6, ry),
                    arrowprops=dict(arrowstyle="-", color="#9ca3af", lw=0.6))

    # PCA → Qubit arrows (1:1)
    for py, qy in zip(pca_ys, q_ys):
        ax.annotate("", xy=(x_q - 1.4, qy), xytext=(x_pca + 1.5, py),
                    arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=1.5))

    # Step labels between columns
    ax.text((x_raw + x_pca) / 2, 3.4, "StandardScaler\n+\nPCA",
            ha="center", va="center", fontsize=9, color="#6d28d9",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f3e8ff", edgecolor="#c4b5fd"))

    ax.text((x_pca + x_q) / 2, 3.4, "MinMaxScaler\n[0, π]\n+\nAngle encode",
            ha="center", va="center", fontsize=9, color="#0f766e",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ccfbf1", edgecolor="#5eead4"))

    # Summary bar at bottom
    summary = (
        "9 raw features  →  4 PCA components (94.1% variance retained)  →  4-qubit ZZFeatureMap"
    )
    ax.text(6.5, 0.2, summary, ha="center", va="center", fontsize=9,
            color="#374151",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f9fafb", edgecolor="#d1d5db"))

    ax.set_title("Feature Compression Pipeline: Classical Data → Quantum Circuit",
                 fontsize=13, pad=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved feature flow diagram: {save_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    import joblib

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pca = joblib.load("results/models/pca_model.pkl")
    out = Path("results/figures") / "feature_compression.png"
    plot_feature_compression_diagram(pca.explained_variance_ratio_.tolist(), out)
