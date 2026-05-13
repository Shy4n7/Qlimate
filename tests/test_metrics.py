"""
Unit tests for src/evaluation/metrics.py

Tests use synthetic numpy arrays only — no dependency on real data files.
"""

import time

import numpy as np
import pytest

from src.evaluation.metrics import compare_regressors, evaluate_regressor

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_SAMPLES = 100


@pytest.fixture
def y_test():
    """Ground-truth temperature values in °C, 100 samples."""
    rng = np.random.default_rng(42)
    return rng.uniform(15.0, 40.0, N_SAMPLES)


@pytest.fixture
def X_test(y_test):
    """Dummy feature matrix — shape (100, 17), values don't matter for metrics."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((len(y_test), 17))


@pytest.fixture
def minimal_result():
    """Factory for a minimal result dict accepted by compare_regressors."""
    def _make(mae=1.5, rmse=2.0, r2=0.85, training_time=1.5, n_train=400):
        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "training_time": training_time,
            "prediction_time": 0.1,
            "n_train": n_train,
        }
    return _make


# ---------------------------------------------------------------------------
# evaluate_regressor tests
# ---------------------------------------------------------------------------


def test_evaluate_regressor_perfect_predictions(X_test, y_test):
    """Perfect predictor should yield MAE=0, RMSE=0, R²=1.0."""
    result = evaluate_regressor(
        predict_fn=lambda X: y_test,
        X_test=X_test,
        y_test=y_test,
        model_name="perfect",
    )
    assert result["mae"] == pytest.approx(0.0, abs=1e-9)
    assert result["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert result["r2"] == pytest.approx(1.0, abs=1e-9)


def test_evaluate_regressor_constant_predictions(X_test, y_test):
    """Constant predictor (always mean) should have R² ≤ 0 and MAE > 0."""
    mean_pred = np.full(len(y_test), np.mean(y_test))
    result = evaluate_regressor(
        predict_fn=lambda X: mean_pred,
        X_test=X_test,
        y_test=y_test,
        model_name="constant",
    )
    assert result["mae"] > 0.0
    assert result["rmse"] >= result["mae"]
    assert result["r2"] <= 0.0


def test_evaluate_regressor_precomputed_preds_fast(X_test, y_test):
    """When precomputed_preds is supplied, inference is skipped and prediction_time < 0.01 s."""

    def slow_predict(X):
        time.sleep(1.0)
        return np.zeros(len(X))

    result = evaluate_regressor(
        predict_fn=slow_predict,
        X_test=X_test,
        y_test=y_test,
        model_name="precomputed",
        precomputed_preds=y_test,
    )
    assert result["prediction_time"] < 0.01


def test_evaluate_regressor_returns_required_keys(X_test, y_test):
    """Result dict must contain all required keys."""
    result = evaluate_regressor(
        predict_fn=lambda X: y_test,
        X_test=X_test,
        y_test=y_test,
        model_name="test_model",
    )
    for key in ("model_name", "mae", "rmse", "r2", "prediction_time", "n_test"):
        assert key in result, f"Missing key: {key}"


def test_evaluate_regressor_n_test_matches_input(X_test, y_test):
    """n_test in result must equal len(y_test)."""
    result = evaluate_regressor(
        predict_fn=lambda X: y_test,
        X_test=X_test,
        y_test=y_test,
        model_name="test_model",
    )
    assert result["n_test"] == len(y_test)


def test_evaluate_regressor_rmse_ge_mae(X_test, y_test):
    """RMSE must always be >= MAE for any prediction array."""
    rng = np.random.default_rng(99)
    noisy_preds = y_test + rng.normal(0, 2.0, len(y_test))
    result = evaluate_regressor(
        predict_fn=lambda X: noisy_preds,
        X_test=X_test,
        y_test=y_test,
        model_name="noisy",
    )
    assert result["rmse"] >= result["mae"] - 1e-9


# ---------------------------------------------------------------------------
# compare_regressors tests
# ---------------------------------------------------------------------------


def test_compare_regressors_row_count(minimal_result):
    """compare_regressors with 2 model results should return a DataFrame with 2 rows."""
    results = {
        "model_a": minimal_result(rmse=2.0),
        "model_b": minimal_result(rmse=3.0),
    }
    df = compare_regressors(results)
    assert len(df) == 2


def test_compare_regressors_sorted_by_rmse_ascending(minimal_result):
    """Returned DataFrame must be sorted by rmse in ascending order."""
    results = {
        "model_high_rmse": minimal_result(rmse=5.0),
        "model_low_rmse": minimal_result(rmse=1.0),
    }
    df = compare_regressors(results)
    assert df["rmse"].iloc[0] <= df["rmse"].iloc[1]


def test_compare_regressors_has_required_columns(minimal_result):
    """Returned DataFrame must contain all required columns."""
    results = {"model_a": minimal_result()}
    df = compare_regressors(results)
    required_cols = {"model", "model_type", "mae", "rmse", "r2",
                     "training_time_s", "prediction_time_s", "training_samples"}
    assert required_cols.issubset(set(df.columns))


def test_compare_regressors_model_type_quantum(minimal_result):
    """Models with 'qsvr' or 'vqr' in name must have model_type='quantum'."""
    results = {
        "QSVR": minimal_result(),
        "VQR": minimal_result(),
        "xgboost_regressor": minimal_result(),
    }
    df = compare_regressors(results)
    df_indexed = df.set_index("model")

    assert df_indexed.loc["QSVR", "model_type"] == "quantum"
    assert df_indexed.loc["VQR", "model_type"] == "quantum"
    assert df_indexed.loc["xgboost_regressor", "model_type"] == "classical"


def test_compare_regressors_no_classification_columns(minimal_result):
    """Returned DataFrame must NOT contain classification metric columns."""
    results = {"model_a": minimal_result()}
    df = compare_regressors(results)
    forbidden_cols = {"accuracy", "f1_macro", "precision_macro", "recall_macro"}
    assert forbidden_cols.isdisjoint(set(df.columns))


def test_compare_regressors_has_training_samples_column(minimal_result):
    """Returned DataFrame must contain a 'training_samples' column."""
    results = {
        "model_a": minimal_result(n_train=400),
        "model_b": minimal_result(n_train=200),
    }
    df = compare_regressors(results)
    assert "training_samples" in df.columns
