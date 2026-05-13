"""
Unit tests for ClassicalModelTrainer in src/models/classical.py.

All tests use synthetic data only — no real CSV files or config.yaml dependency.
"""

import pytest
import numpy as np

from src.models.classical import ClassicalModelTrainer

# ---------------------------------------------------------------------------
# Minimal config — avoids any dependency on config/config.yaml
# ---------------------------------------------------------------------------

MINIMAL_CONFIG = {
    "data_split": {"random_state": 42},
    "classical_ml": {
        "random_forest": {
            "n_estimators": [10],
            "max_depth": [3],
            "min_samples_split": [2],
            "cv_folds": 2,
        },
        "svm": {
            "C": [1.0],
            "gamma": ["scale"],
            "cv_folds": 2,
        },
        "xgboost": {
            "n_estimators": [10],
            "max_depth": [3],
            "learning_rate": [0.1],
            "cv_folds": 2,
        },
        "neural_net": {
            "hidden_dims": [8],
            "dropout": [0.1],
            "epochs": 3,
            "learning_rate": 0.01,
            "batch_size": 16,
            "early_stopping_patience": 2,
        },
        "logistic_regression": {
            "C": [1.0],
            "solver": "lbfgs",
            "max_iter": 100,
            "cv_folds": 2,
        },
        "knn": {
            "n_neighbors": 3,
            "cv_folds": 2,
        },
        "gradient_boosting": {
            "n_estimators": [10],
            "max_depth": [3],
            "learning_rate": [0.1],
            "cv_folds": 2,
        },
        "lightgbm": {
            "n_estimators": [10],
            "max_depth": [3],
            "learning_rate": [0.1],
            "num_leaves": [15],
            "cv_folds": 2,
        },
        "adaboost": {
            "n_estimators": 10,
            "learning_rate": 1.0,
        },
        "decision_tree": {
            "max_depth": [3, 5],
            "min_samples_split": [2],
            "cv_folds": 2,
        },
    },
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_TRAIN = 60
N_VAL = 20
N_FEATURES = 16
N_CLASSES = 5


@pytest.fixture(scope="module")
def synthetic_data():
    """
    60 training samples (12 per class) and 20 validation samples (4 per class),
    16 features, 5 classes. Deterministic via numpy default_rng(42).
    """
    rng = np.random.default_rng(42)

    # Build balanced labels
    y_train = np.repeat(np.arange(N_CLASSES), N_TRAIN // N_CLASSES)  # 60 samples
    y_val = np.repeat(np.arange(N_CLASSES), N_VAL // N_CLASSES)       # 20 samples

    X_train = rng.standard_normal((N_TRAIN, N_FEATURES)).astype(np.float32)
    X_val = rng.standard_normal((N_VAL, N_FEATURES)).astype(np.float32)

    return X_train, y_train, X_val, y_val


@pytest.fixture(scope="module")
def trainer():
    """ClassicalModelTrainer initialised with the minimal inline config."""
    return ClassicalModelTrainer(config=MINIMAL_CONFIG)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

REQUIRED_BASE_KEYS = {"model", "training_time", "val_preds"}


def _assert_base_keys(result: dict) -> None:
    assert REQUIRED_BASE_KEYS.issubset(result.keys()), (
        f"Missing keys: {REQUIRED_BASE_KEYS - result.keys()}"
    )


# ---------------------------------------------------------------------------
# Tests 1–9: required-key assertions for each model
# ---------------------------------------------------------------------------

def test_train_random_forest_returns_required_keys(trainer, synthetic_data):
    X_train, y_train, X_val, y_val = synthetic_data
    result = trainer.train_random_forest(X_train, y_train, X_val, y_val)
    _assert_base_keys(result)


def test_train_svm_returns_required_keys(trainer, synthetic_data):
    X_train, y_train, X_val, y_val = synthetic_data
    result = trainer.train_svm(X_train, y_train, X_val, y_val)
    _assert_base_keys(result)


def test_train_xgboost_returns_required_keys(trainer, synthetic_data):
    X_train, y_train, X_val, y_val = synthetic_data
    result = trainer.train_xgboost(X_train, y_train, X_val, y_val)
    _assert_base_keys(result)


def test_train_neural_network_returns_required_keys(trainer, synthetic_data):
    X_train, y_train, X_val, y_val = synthetic_data
    result = trainer.train_neural_network(X_train, y_train, X_val, y_val)
    required = {"model", "training_time", "train_losses", "val_preds"}
    assert required.issubset(result.keys()), (
        f"Missing keys: {required - result.keys()}"
    )


def test_train_logistic_regression_returns_required_keys(trainer, synthetic_data):
    X_train, y_train, X_val, y_val = synthetic_data
    result = trainer.train_logistic_regression(X_train, y_train, X_val, y_val)
    _assert_base_keys(result)


def test_train_knn_returns_required_keys(trainer, synthetic_data):
    X_train, y_train, X_val, y_val = synthetic_data
    result = trainer.train_knn(X_train, y_train, X_val, y_val)
    _assert_base_keys(result)


def test_train_gradient_boosting_returns_required_keys(trainer, synthetic_data):
    X_train, y_train, X_val, y_val = synthetic_data
    result = trainer.train_gradient_boosting(X_train, y_train, X_val, y_val)
    _assert_base_keys(result)


def test_train_adaboost_returns_required_keys(trainer, synthetic_data):
    X_train, y_train, X_val, y_val = synthetic_data
    result = trainer.train_adaboost(X_train, y_train, X_val, y_val)
    _assert_base_keys(result)


def test_train_decision_tree_returns_required_keys(trainer, synthetic_data):
    X_train, y_train, X_val, y_val = synthetic_data
    result = trainer.train_decision_tree(X_train, y_train, X_val, y_val)
    _assert_base_keys(result)


# ---------------------------------------------------------------------------
# Test 10: LightGBM — skip gracefully if not installed
# ---------------------------------------------------------------------------

def test_train_lightgbm_returns_required_keys_or_skips(trainer, synthetic_data):
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        pytest.skip("lightgbm not installed")

    X_train, y_train, X_val, y_val = synthetic_data
    result = trainer.train_lightgbm(X_train, y_train, X_val, y_val)
    assert result is not None, "train_lightgbm returned None despite lightgbm being installed"
    _assert_base_keys(result)


# ---------------------------------------------------------------------------
# Test 11: val_preds contain only valid class integers
# ---------------------------------------------------------------------------

def test_val_preds_valid_classes(trainer, synthetic_data):
    """Random Forest val_preds must be integers in [0, N_CLASSES-1]."""
    X_train, y_train, X_val, y_val = synthetic_data
    result = trainer.train_random_forest(X_train, y_train, X_val, y_val)
    val_preds = result["val_preds"]

    assert all(isinstance(int(p), int) for p in val_preds), (
        "val_preds should contain integer-like values"
    )
    assert all(0 <= int(p) <= N_CLASSES - 1 for p in val_preds), (
        f"val_preds values must be in [0, {N_CLASSES - 1}], got: {np.unique(val_preds)}"
    )


# ---------------------------------------------------------------------------
# Test 12: predict() output shape
# ---------------------------------------------------------------------------

def test_predict_output_shape(trainer, synthetic_data):
    """After training RF, trainer.predict('random_forest', X_val) shape == (N_VAL,)."""
    X_train, y_train, X_val, y_val = synthetic_data
    trainer.train_random_forest(X_train, y_train, X_val, y_val)
    preds = trainer.predict("random_forest", X_val)
    assert preds.shape == (N_VAL,), (
        f"Expected shape ({N_VAL},), got {preds.shape}"
    )
