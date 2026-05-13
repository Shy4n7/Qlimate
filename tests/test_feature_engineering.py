"""
Unit tests for src/features/engineering.py

All tests use synthetic data only — no real CSV files, no config/config.yaml dependency.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    engineer_features,
    prepare_splits,
    prepare_quantum_subset,
)

# ---------------------------------------------------------------------------
# Minimal config — no dependency on config/config.yaml
# ---------------------------------------------------------------------------
MINIMAL_CONFIG = {
    "data_split": {"train": 0.70, "val": 0.15, "test": 0.15, "random_state": 42}
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """200-row reproducible DataFrame with all required MERRA-2 columns."""
    rng = np.random.default_rng(42)
    n = 200

    data = {
        "T2M":     rng.uniform(270.0, 310.0, n),
        "PRECTOT": rng.uniform(0.0, 10.0, n),
        "QV2M":    rng.uniform(0.001, 0.02, n),
        "PS":      rng.uniform(95000.0, 102000.0, n),
        "SLP":     rng.uniform(99000.0, 103000.0, n),
        "SWGDN":   rng.uniform(50.0, 300.0, n),
        "LWGNT":   rng.uniform(20.0, 150.0, n),
        "CLDTOT":  rng.uniform(0.0, 1.0, n),
        "EVAP":    rng.uniform(0.0, 5.0, n),
        "U10M":    rng.uniform(-10.0, 10.0, n),
        "V10M":    rng.uniform(-10.0, 10.0, n),
        "month":   rng.integers(1, 13, n),          # int 1–12
        "label":   np.tile(np.arange(5), 40),       # 5 classes, 40 rows each
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Tests for engineer_features
# ---------------------------------------------------------------------------

def test_engineer_features_adds_derived_columns(synthetic_df):
    """engineer_features must add all 7 derived columns."""
    result = engineer_features(synthetic_df)
    derived = [
        "wind_speed",
        "wind_direction",
        "net_radiation",
        "precip_evap_ratio",
        "month_sin",
        "month_cos",
        "pressure_anomaly",
    ]
    for col in derived:
        assert col in result.columns, f"Missing derived column: {col}"


def test_wind_speed_non_negative(synthetic_df):
    """wind_speed = sqrt(U10M^2 + V10M^2) must be >= 0 for all rows."""
    result = engineer_features(synthetic_df)
    assert (result["wind_speed"] >= 0).all(), "wind_speed contains negative values"


def test_month_sin_cos_range(synthetic_df):
    """month_sin and month_cos must be in [-1.0, 1.0] for all rows."""
    result = engineer_features(synthetic_df)
    assert result["month_sin"].between(-1.0, 1.0).all(), \
        "month_sin out of [-1, 1] range"
    assert result["month_cos"].between(-1.0, 1.0).all(), \
        "month_cos out of [-1, 1] range"


# ---------------------------------------------------------------------------
# Tests for prepare_splits
# ---------------------------------------------------------------------------

def test_prepare_splits_row_count(synthetic_df):
    """Sum of train + val + test rows must equal total non-NaN input rows."""
    splits = prepare_splits(synthetic_df, MINIMAL_CONFIG)
    total = len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"])
    assert total == len(synthetic_df), (
        f"Row count mismatch: {total} != {len(synthetic_df)}"
    )


def test_prepare_splits_scaler_applied(synthetic_df):
    """X_train must be approximately zero-mean and unit-variance after StandardScaler."""
    splits = prepare_splits(synthetic_df, MINIMAL_CONFIG)
    X_train = splits["X_train"]
    mean = np.abs(X_train.mean())
    std = np.abs(X_train.std() - 1.0)
    assert mean < 0.1, f"X_train mean too far from 0: {mean:.4f}"
    assert std < 0.1, f"X_train std too far from 1: {std:.4f}"


# ---------------------------------------------------------------------------
# Tests for prepare_quantum_subset
# ---------------------------------------------------------------------------

def _get_splits(synthetic_df):
    """Helper: run prepare_splits and return the result dict."""
    return prepare_splits(synthetic_df, MINIMAL_CONFIG)


def test_quantum_subset_range(synthetic_df):
    """All values in X_train_q must be in [0.0, π]."""
    splits = _get_splits(synthetic_df)
    quantum = prepare_quantum_subset(
        splits["X_train"], splits["y_train"],
        splits["X_test"],  splits["y_test"],
        n_components=4,
        subset_size=80,
        random_state=42,
        scale_range=(0.0, np.pi),
    )
    X_train_q = quantum["X_train_q"]
    # Use a small tolerance to account for float32 → float64 rounding
    assert X_train_q.min() >= 0.0 - 1e-6, \
        f"X_train_q has values below 0: min={X_train_q.min()}"
    assert X_train_q.max() <= np.pi + 1e-6, \
        f"X_train_q has values above π: max={X_train_q.max()}"


def test_quantum_subset_size_limit(synthetic_df):
    """prepare_quantum_subset with subset_size=80 must return at most 80 training samples."""
    splits = _get_splits(synthetic_df)
    quantum = prepare_quantum_subset(
        splits["X_train"], splits["y_train"],
        splits["X_test"],  splits["y_test"],
        n_components=4,
        subset_size=80,
        random_state=42,
        scale_range=(0.0, np.pi),
    )
    assert len(quantum["X_train_q"]) <= 80, (
        f"X_train_q has {len(quantum['X_train_q'])} samples, expected <= 80"
    )
