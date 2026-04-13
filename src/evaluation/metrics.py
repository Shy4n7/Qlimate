"""
Unified evaluation metrics for classical and quantum models.
"""

import logging
import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Normal", "Drought", "Wet_Flood", "Heat_Extreme", "Cold_Extreme"]


def evaluate_model(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    class_names: list[str] = CLASS_NAMES,
    precomputed_preds: np.ndarray | None = None,
) -> dict:
    """Compute classification metrics for a model.

    Args:
        predict_fn: callable that takes X and returns integer predictions
        X_test: test features
        y_test: true labels (integers)
        model_name: name for logging/reporting
        precomputed_preds: if provided, skip inference and use these predictions

    Returns dict with accuracy, f1_macro, precision_macro, recall_macro,
    confusion_matrix, classification_report, prediction_time.
    """
    t0 = time.perf_counter()
    preds = precomputed_preds if precomputed_preds is not None else predict_fn(X_test)
    prediction_time = time.perf_counter() - t0

    present_classes = sorted(np.unique(np.concatenate([y_test, preds])))
    names = [class_names[i] for i in present_classes
             if i < len(class_names)]

    result = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_macro": float(f1_score(y_test, preds, average="macro",
                                    zero_division=0)),
        "precision_macro": float(precision_score(y_test, preds, average="macro",
                                                   zero_division=0)),
        "recall_macro": float(recall_score(y_test, preds, average="macro",
                                            zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, preds),
        "classification_report": classification_report(
            y_test, preds, target_names=names, zero_division=0
        ),
        "prediction_time": prediction_time,
        "n_test": len(y_test),
    }

    logger.info(
        f"{model_name}: acc={result['accuracy']:.4f}, "
        f"f1={result['f1_macro']:.4f}, "
        f"pred_time={prediction_time:.3f}s"
    )
    return result


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Build a comparison DataFrame sorted by F1 macro.

    Args:
        results: dict mapping model_name -> metrics dict from evaluate_model()

    Returns DataFrame with columns:
        model, accuracy, f1_macro, precision_macro, recall_macro,
        training_time, prediction_time
    """
    rows = []
    for name, r in results.items():
        rows.append({
            "model": name,
            "accuracy": r.get("accuracy", np.nan),
            "f1_macro": r.get("f1_macro", np.nan),
            "precision_macro": r.get("precision_macro", np.nan),
            "recall_macro": r.get("recall_macro", np.nan),
            "training_time_s": r.get("training_time", np.nan),
            "prediction_time_s": r.get("prediction_time", np.nan),
        })
    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    return df.reset_index(drop=True)


def print_comparison_table(df: pd.DataFrame) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}" if not np.isnan(x) else "  N/A",
    ))
    print("=" * 80)


def scalability_summary(
    classical_df: pd.DataFrame,
    quantum_df: pd.DataFrame,
) -> None:
    """Print scalability analysis summary."""
    print("\n--- Classical Scalability (training size) ---")
    print(classical_df.to_string(index=False))
    print("\n--- Quantum Scalability (qubit count) ---")
    print(quantum_df.to_string(index=False))
