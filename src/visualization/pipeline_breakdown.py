"""
Pipeline timing breakdown visualization.
Shows time distribution across pipeline stages per model.
"""

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

# Training times: classical not captured; quantum measured.
# Classical values are left as None and shown with a note.
PIPELINE_DATA = {
    "Random Forest": {
        "type": "classical",
        "color": "#2563eb",
        "training_s": None,
        "prediction_s": 0.08,
    },
    "SVM": {
        "type": "classical",
        "color": "#3b82f6",
        "training_s": None,
        "prediction_s": 0.04,
    },
    "XGBoost": {
        "type": "classical",
        "color": "#60a5fa",
        "training_s": None,
        "prediction_s": 0.09,
    },
    "Neural Network": {
        "type": "classical",
        "color": "#93c5fd",
        "training_s": None,
        "prediction_s": 0.03,
    },
    "QSVC": {
        "type": "quantum",
        "color": "#7c3aed",
        "training_s": 274.6,
        "prediction_s": 2078.6,
    },
    "VQC": {
        "type": "quantum",
        "color": "#a855f7",
        "training_s": 198.0,
        "prediction_s": 4.8,
    },
}


def plot_pipeline_timing(save_path: Path) -> None:
    """
    Stacked horizontal bar showing training + prediction time per model.
    Classical training times are not available; quantum times are measured.
    """
    models = list(PIPELINE_DATA.keys())
    train_times = [d["training_s"] or 0 for d in PIPELINE_DATA.values()]
    pred_times = [d["prediction_s"] for d in PIPELINE_DATA.values()]
    colors = [d["color"] for d in PIPELINE_DATA.values()]
    model_types = [d["type"] for d in PIPELINE_DATA.values()]

    fig, ax = plt.subplots(figsize=(10, 5))

    y = np.arange(len(models))
    bar_h = 0.5

    for i, (m, train, pred, col, mtype) in enumerate(
        zip(models, train_times, pred_times, colors, model_types)
    ):
        if mtype == "classical" and train == 0:
            # Training time not captured — show hatched placeholder
            ax.barh(i, 5, height=bar_h, color="#e5e7eb", edgecolor="#9ca3af",
                    hatch="///", linewidth=0.8)
            ax.text(5.5, i, "training time\nnot captured", va="center",
                    fontsize=7.5, color="#6b7280")
        else:
            ax.barh(i, train / 60, height=bar_h, color=col, alpha=0.85,
                    label="Training" if i == 0 else "")
            ax.barh(i, pred / 60, height=bar_h, left=train / 60,
                    color=col, alpha=0.45,
                    label="Prediction" if i == 0 else "")
            total = (train + pred) / 60
            ax.text(total + 0.5, i,
                    f"{train/60:.1f} + {pred/60:.1f} min",
                    va="center", fontsize=8, color="#374151")

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel("Time (minutes)", fontsize=11)
    ax.set_title(
        "Pipeline Timing: Training + Prediction\n"
        "(Quantum models trained on 400-sample subset; classical on 6,958 samples)",
        fontsize=12
    )

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2563eb", alpha=0.85, label="Classical — training"),
        Patch(facecolor="#2563eb", alpha=0.45, label="Classical — prediction"),
        Patch(facecolor="#7c3aed", alpha=0.85, label="Quantum — training"),
        Patch(facecolor="#7c3aed", alpha=0.45, label="Quantum — prediction"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    # Add divider between classical and quantum
    ax.axhline(3.5, color="#9ca3af", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(-2, 3.6, "Quantum", fontsize=8, color="#7c3aed", fontstyle="italic")
    ax.text(-2, 3.3, "Classical", fontsize=8, color="#2563eb", fontstyle="italic")

    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(-15, None)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved pipeline breakdown: {save_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    plot_pipeline_timing(Path("results/figures/pipeline_breakdown.png"))
