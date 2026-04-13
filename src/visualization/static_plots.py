"""
Publication-quality static visualizations using Matplotlib.

Generates all figures for the final report / README hero image.
All figures saved as 300 DPI PNG to results/figures/.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Publication style
STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

CLASS_NAMES = ["Normal", "Drought", "Wet_Flood", "Heat_Extreme", "Cold_Extreme"]

# Blue palette for classical, purple for quantum
MODEL_COLORS = {
    "Random Forest": "#1565C0",
    "SVM (RBF)": "#1976D2",
    "XGBoost": "#42A5F5",
    "Neural Network": "#90CAF9",
    "QSVC": "#6A1B9A",
    "VQC": "#AB47BC",
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close(fig)


def plot_model_comparison_bar(
    comparison_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Grouped bar chart: accuracy and F1 for all models (classical + quantum)."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        models = comparison_df["model"].tolist()
        x = np.arange(len(models))
        width = 0.35
        colors = [MODEL_COLORS.get(m, "#888888") for m in models]

        # Accuracy / F1
        bars1 = axes[0].bar(x - width / 2, comparison_df["accuracy"],
                             width, label="Accuracy",
                             color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        bars2 = axes[0].bar(x + width / 2, comparison_df["f1_macro"],
                             width, label="F1 Macro",
                             color=colors, alpha=0.55, edgecolor="black", linewidth=0.5,
                             hatch="//")

        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=25, ha="right")
        axes[0].set_ylabel("Score")
        axes[0].set_ylim(0, 1.12)
        axes[0].set_title("Classification Performance")
        axes[0].legend(loc="upper right")
        axes[0].axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

        # Add value labels
        for bar in bars1:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                         f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                         f"{h:.3f}", ha="center", va="bottom", fontsize=8)

        # Add quantum separator line
        if any("QSVC" in m or "VQC" in m for m in models):
            classical_count = sum(1 for m in models
                                  if m not in ("QSVC", "VQC"))
            axes[0].axvline(classical_count - 0.5, color="purple",
                             linestyle="--", linewidth=1, alpha=0.4)
            axes[0].text(classical_count - 0.5, 1.05, "Classical | Quantum",
                         ha="center", fontsize=9, color="purple", alpha=0.6)

        # Training time (log scale)
        train_times = comparison_df["training_time_s"].fillna(0)
        bars_t = axes[1].barh(models, train_times, color=colors,
                               edgecolor="black", linewidth=0.5, alpha=0.85)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Training Time (seconds, log scale)")
        axes[1].set_title("Training Time Comparison")

        for bar, val in zip(bars_t, train_times):
            if val > 0:
                axes[1].text(val * 1.05, bar.get_y() + bar.get_height() / 2,
                             f"{val:.1f}s", va="center", fontsize=9)

        # Color legend
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(color="#1565C0", label="Classical ML"),
            Patch(color="#6A1B9A", label="Quantum ML"),
        ]
        axes[1].legend(handles=legend_handles, loc="lower right")

        fig.suptitle("Classical vs Quantum ML — Climate Condition Classification",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        _save(fig, save_path)


def plot_confusion_matrices(
    results: dict,
    save_path: Path,
    class_names: list[str] = CLASS_NAMES,
) -> None:
    """Grid of row-normalized confusion matrices for all models."""
    n_models = len(results)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 4.5, nrows * 4))
        axes = np.array(axes).flatten()

        for ax, (name, r) in zip(axes, results.items()):
            cm = r["confusion_matrix"].astype(float)
            cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-10)
            n_cls = cm_norm.shape[0]
            labels = class_names[:n_cls]

            im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(n_cls))
            ax.set_yticks(range(n_cls))
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("True", fontsize=9)
            ax.set_title(
                f"{name}\nAcc={r['accuracy']:.3f}  F1={r['f1_macro']:.3f}",
                fontsize=10,
            )

            for i in range(n_cls):
                for j in range(n_cls):
                    val = cm_norm[i, j]
                    ax.text(j, i, f"{val:.2f}",
                            ha="center", va="center", fontsize=8,
                            color="white" if val > 0.5 else "black")
            plt.colorbar(im, ax=ax, shrink=0.75, label="Recall")

        # Hide unused axes
        for ax in axes[n_models:]:
            ax.set_visible(False)

        fig.suptitle("Confusion Matrices — Row Normalized (Classical & Quantum)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        _save(fig, save_path)


def plot_scalability(
    classical_df: pd.DataFrame,
    quantum_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Two-panel scalability figure: classical (size) + quantum (qubits)."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Classical scalability
        classical_colors = {"Random Forest": "#1565C0", "XGBoost": "#42A5F5"}
        for model in classical_df["model"].unique():
            sub = classical_df[classical_df["model"] == model]
            axes[0].plot(sub["train_size"], sub["f1_macro"],
                         "o-", label=model, linewidth=2, markersize=6,
                         color=classical_colors.get(model, "gray"))

        axes[0].set_xlabel("Training Dataset Size")
        axes[0].set_ylabel("F1 Macro")
        axes[0].set_title("Classical ML: F1 vs Training Size")
        axes[0].legend()
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3, linestyle="--")

        # Quantum scalability
        quantum_colors = {"QSVC": "#6A1B9A", "VQC": "#AB47BC"}
        for model in quantum_df["model"].unique():
            sub = quantum_df[quantum_df["model"] == model]
            axes[1].plot(sub["n_qubits"], sub["f1_macro"],
                         "s-", label=model, linewidth=2, markersize=7,
                         color=quantum_colors.get(model, "gray"))

        axes[1].set_xlabel("Number of Qubits")
        axes[1].set_ylabel("F1 Macro")
        axes[1].set_title("Quantum ML: F1 vs Qubit Count")
        axes[1].legend()
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3, linestyle="--")
        axes[1].set_xticks(quantum_df["n_qubits"].unique().tolist()
                           if not quantum_df.empty else range(2, 7))

        fig.suptitle("Scalability Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()
        _save(fig, save_path)


def plot_learning_curves(
    nn_train_losses: list[float],
    nn_val_losses: list[float],
    vqc_loss_history: list[float],
    save_path: Path,
) -> None:
    """Neural network and VQC training curves side by side."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        # NN
        ep = range(1, len(nn_train_losses) + 1)
        axes[0].plot(ep, nn_train_losses, label="Train", color="#1976D2", linewidth=1.5)
        axes[0].plot(ep, nn_val_losses, label="Validation", color="#F44336",
                     linewidth=1.5, linestyle="--")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cross-Entropy Loss")
        axes[0].set_title("Neural Network Learning Curves")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, linestyle="--")

        # VQC
        iters = range(1, len(vqc_loss_history) + 1)
        axes[1].plot(iters, vqc_loss_history, color="#7B1FA2", linewidth=1.5,
                     alpha=0.6, label="VQC (COBYLA)")
        if len(vqc_loss_history) > 10:
            smoothed = pd.Series(vqc_loss_history).rolling(7, center=True,
                                                             min_periods=1).mean()
            axes[1].plot(iters, smoothed, color="#E91E63", linewidth=2,
                         linestyle="--", label="Smoothed")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Objective")
        axes[1].set_title("VQC Convergence (COBYLA Optimizer)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, linestyle="--")

        fig.suptitle("Training Dynamics", fontsize=14, fontweight="bold")
        plt.tight_layout()
        _save(fig, save_path)


def plot_feature_importance(
    rf_importances: np.ndarray,
    xgb_importances: np.ndarray,
    feature_names: list[str],
    save_path: Path,
) -> None:
    """RF and XGBoost feature importance side-by-side horizontal bars."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        for ax, importances, title, color in [
            (axes[0], rf_importances, "Random Forest", "#1565C0"),
            (axes[1], xgb_importances, "XGBoost", "#42A5F5"),
        ]:
            if importances is None:
                ax.set_visible(False)
                continue
            n_show = min(12, len(importances))
            top_idx = np.argsort(importances)[-n_show:]
            names = [feature_names[i] for i in top_idx]
            vals = importances[top_idx]

            colors = [color if v > np.median(vals) else color + "80"
                      for v in vals]
            ax.barh(names, vals, color=colors, edgecolor="black", linewidth=0.4)
            ax.set_xlabel("Importance Score")
            ax.set_title(f"{title} Feature Importance (Top {n_show})")
            ax.grid(True, axis="x", alpha=0.3, linestyle="--")

        plt.tight_layout()
        _save(fig, save_path)


def create_summary_figure(
    comparison_df: pd.DataFrame,
    all_results: dict,
    vqc_loss: list[float],
    save_path: Path,
    class_names: list[str] = CLASS_NAMES,
) -> None:
    """Hero figure for README: 4-panel summary."""
    with plt.rc_context({**STYLE, "figure.dpi": 200}):
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        # (a) Model comparison bars
        ax_bar = fig.add_subplot(gs[0, :2])
        models = comparison_df["model"].tolist()
        x = np.arange(len(models))
        colors = [MODEL_COLORS.get(m, "#888888") for m in models]
        ax_bar.bar(x - 0.2, comparison_df["accuracy"], 0.4,
                   label="Accuracy", color=colors, alpha=0.85,
                   edgecolor="black", linewidth=0.5)
        ax_bar.bar(x + 0.2, comparison_df["f1_macro"], 0.4,
                   label="F1 Macro", color=colors, alpha=0.5,
                   edgecolor="black", linewidth=0.5, hatch="//")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(models, rotation=20, ha="right")
        ax_bar.set_ylabel("Score")
        ax_bar.set_ylim(0, 1.15)
        ax_bar.set_title("(a) Model Comparison — Accuracy & F1 Macro")
        ax_bar.legend(loc="upper right", fontsize=9)
        ax_bar.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

        # (b) Best classical confusion matrix
        ax_cm_c = fig.add_subplot(gs[0, 2])
        best_classical = [m for m in comparison_df["model"]
                          if m not in ("QSVC", "VQC")]
        if best_classical:
            bm = best_classical[0]
            cm = all_results[bm]["confusion_matrix"].astype(float)
            cm_n = cm / (cm.sum(axis=1, keepdims=True) + 1e-10)
            n_cls = cm_n.shape[0]
            labels = class_names[:n_cls]
            im = ax_cm_c.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
            ax_cm_c.set_xticks(range(n_cls))
            ax_cm_c.set_yticks(range(n_cls))
            ax_cm_c.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
            ax_cm_c.set_yticklabels(labels, fontsize=7)
            ax_cm_c.set_title(f"(b) Best Classical ({bm})\nAcc={all_results[bm]['accuracy']:.3f}",
                               fontsize=10)
            for i in range(n_cls):
                for j in range(n_cls):
                    v = cm_n[i, j]
                    ax_cm_c.text(j, i, f"{v:.2f}", ha="center", va="center",
                                 fontsize=7, color="white" if v > 0.5 else "black")
            plt.colorbar(im, ax=ax_cm_c, shrink=0.7)

        # (c) VQC convergence
        ax_vqc = fig.add_subplot(gs[1, 0])
        if vqc_loss:
            iters = range(1, len(vqc_loss) + 1)
            ax_vqc.plot(iters, vqc_loss, color="#AB47BC", linewidth=1.5,
                         alpha=0.5, label="VQC loss")
            if len(vqc_loss) > 10:
                smoothed = pd.Series(vqc_loss).rolling(7, min_periods=1,
                                                         center=True).mean()
                ax_vqc.plot(iters, smoothed, color="#6A1B9A",
                             linewidth=2, label="Smoothed")
            ax_vqc.set_xlabel("Iteration")
            ax_vqc.set_ylabel("Loss")
            ax_vqc.set_title("(c) VQC Convergence")
            ax_vqc.legend(fontsize=9)
            ax_vqc.grid(True, alpha=0.3)

        # (d) Training time comparison (log)
        ax_time = fig.add_subplot(gs[1, 1])
        train_t = comparison_df["training_time_s"].fillna(0.1)
        colors_t = [MODEL_COLORS.get(m, "#888888") for m in models]
        ax_time.barh(models, train_t, color=colors_t,
                      edgecolor="black", linewidth=0.4)
        ax_time.set_xscale("log")
        ax_time.set_xlabel("Training Time (s, log scale)")
        ax_time.set_title("(d) Training Time")
        ax_time.grid(True, axis="x", alpha=0.3)

        # (e) Radar / spider chart — best models per category
        ax_radar = fig.add_subplot(gs[1, 2], polar=True)
        metrics_keys = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
        metric_labels = ["Accuracy", "F1", "Precision", "Recall"]
        n_metrics = len(metrics_keys)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        radar_models = [m for m in models if m in all_results][:4]
        radar_colors = [MODEL_COLORS.get(m, "#888888") for m in radar_models]

        for m, col in zip(radar_models, radar_colors):
            r = all_results[m]
            vals = [r.get(k, 0) for k in metrics_keys]
            vals += vals[:1]
            ax_radar.plot(angles, vals, "o-", color=col, linewidth=1.5,
                           markersize=4, label=m, alpha=0.8)
            ax_radar.fill(angles, vals, color=col, alpha=0.08)

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metric_labels, fontsize=9)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax_radar.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
        ax_radar.set_title("(e) Metric Profile", pad=15, fontsize=10)
        ax_radar.legend(loc="upper right", bbox_to_anchor=(1.45, 1.1), fontsize=8)

        fig.suptitle(
            "Qlimate: Classical ML vs Quantum ML on NASA MERRA-2 Climate Data (India)",
            fontsize=15, fontweight="bold",
        )
        _save(fig, save_path)
