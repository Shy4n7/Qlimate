"""
Unified evaluation metrics for classical and quantum regression models.
"""

import logging
import math
import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)


def evaluate_regressor(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    precomputed_preds: np.ndarray | None = None,
) -> dict:
    """Compute regression metrics for a model.

    Args:
        predict_fn: callable that takes X and returns float predictions
        X_test: test features
        y_test: true target values (float, °C)
        model_name: name for logging/reporting
        precomputed_preds: if provided, skip inference and use these predictions

    Returns dict with model_name, mae, rmse, r2, prediction_time, n_test.
    """
    t0 = time.perf_counter()
    preds = precomputed_preds if precomputed_preds is not None else predict_fn(X_test)
    prediction_time = time.perf_counter() - t0

    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(math.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    result = {
        "model_name": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "prediction_time": prediction_time,
        "n_test": len(y_test),
    }

    logger.info(
        f"{model_name}: mae={mae:.4f}°C, rmse={rmse:.4f}°C, "
        f"r2={r2:.4f}, pred_time={prediction_time:.3f}s"
    )
    return result


_QUANTUM_KEYWORDS = ("qsvr", "vqr", "quantum")


def _infer_model_type(name: str) -> str:
    """Return 'quantum' if *name* contains a quantum keyword, else 'classical'."""
    lower = name.lower()
    return "quantum" if any(kw in lower for kw in _QUANTUM_KEYWORDS) else "classical"


def compare_regressors(results: dict[str, dict]) -> pd.DataFrame:
    """Build a comparison DataFrame sorted by RMSE ascending.

    Args:
        results: dict mapping model_name -> metrics dict from evaluate_regressor()

    Returns DataFrame with columns:
        model, model_type, mae, rmse, r2,
        training_time_s, prediction_time_s, training_samples
    """
    rows = []
    for name, r in results.items():
        rows.append({
            "model": name,
            "model_type": _infer_model_type(name),
            "mae": r.get("mae", np.nan),
            "rmse": r.get("rmse", np.nan),
            "r2": r.get("r2", np.nan),
            "training_time_s": r.get("training_time", np.nan),
            "prediction_time_s": r.get("prediction_time", np.nan),
            "training_samples": r.get("n_train", np.nan),
        })
    df = pd.DataFrame(rows).sort_values("rmse", ascending=True)
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
