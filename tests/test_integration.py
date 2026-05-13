"""
Integration tests for the full data pipeline.

Tests verify that feature engineering, splitting, and quantum subset
preparation work together correctly end-to-end using synthetic data only.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import prepare_splits, prepare_quantum_subset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MINIMAL_CONFIG = {
    "data_split": {"train": 0.70, "val": 0.15, "test": 0.15, "random_state": 42}
}

# Columns required by prepare_splits (raw MERRA-2 + derived-friendly columns)
_RAW_COLS = [
    "T2M", "PRECTOT", "QV2M", "PS", "SLP",
    "SWGDN", "LWGNT", "CLDTOT", "EVAP",
    "U10M", "V10M",
    "month",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df():
    """200-row DataFrame with 5 balanced classes (40 rows each)."""
    rng = np.random.default_rng(42)
    n_rows = 200
    n_classes = 5
    rows_per_class = n_rows // n_classes

    data = {col: rng.uniform(0.1, 10.0, size=n_rows) for col in _RAW_COLS}
    # month must be integer in [1, 12]
    data["month"] = rng.integers(1, 13, size=n_rows)
    # 5 balanced classes
    data["label"] = np.repeat(np.arange(n_classes), rows_per_class)

    return pd.DataFrame(data)


@pytest.fixture
def synthetic_df_with_nans():
    """200-row DataFrame where 10 rows have NaN in the T2M column."""
    rng = np.random.default_rng(42)
    n_rows = 200
    n_classes = 5
    rows_per_class = n_rows // n_classes

    data = {col: rng.uniform(0.1, 10.0, size=n_rows) for col in _RAW_COLS}
    data["month"] = rng.integers(1, 13, size=n_rows)
    data["label"] = np.repeat(np.arange(n_classes), rows_per_class)

    df = pd.DataFrame(data)
    # Inject NaN into 10 rows of T2M
    nan_indices = rng.choice(n_rows, size=10, replace=False)
    df.loc[nan_indices, "T2M"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pipeline_test_set_row_count_matches(synthetic_df):
    """X_test and X_test_q must have the same number of rows.

    Validates: Requirement 5, Acceptance Criterion 2
    """
    splits = prepare_splits(synthetic_df, MINIMAL_CONFIG)
    quantum = prepare_quantum_subset(
        splits["X_train"],
        splits["y_train"],
        splits["X_test"],
        splits["y_test"],
        n_components=4,
        subset_size=80,
        random_state=42,
        scale_range=(0.0, np.pi),
    )

    X_test = splits["X_test"]
    X_test_q = quantum["X_test_q"]

    assert len(X_test) == len(X_test_q), (
        f"X_test has {len(X_test)} rows but X_test_q has {len(X_test_q)} rows"
    )


def test_pipeline_stratification_preserved(synthetic_df):
    """Every class in y_train must also appear in y_train_q.

    Validates: Requirement 5, Acceptance Criterion 3
    """
    splits = prepare_splits(synthetic_df, MINIMAL_CONFIG)
    quantum = prepare_quantum_subset(
        splits["X_train"],
        splits["y_train"],
        splits["X_test"],
        splits["y_test"],
        n_components=4,
        subset_size=80,
        random_state=42,
        scale_range=(0.0, np.pi),
    )

    classes_train = set(np.unique(splits["y_train"]))
    classes_train_q = set(np.unique(quantum["y_train_q"]))

    missing = classes_train - classes_train_q
    assert not missing, (
        f"Classes present in y_train but absent from y_train_q: {missing}"
    )


def test_pipeline_nan_handling(synthetic_df_with_nans):
    """prepare_splits must drop NaN rows so returned arrays contain no NaNs.

    Validates: Requirement 5, Acceptance Criterion 4
    """
    splits = prepare_splits(synthetic_df_with_nans, MINIMAL_CONFIG)

    X_train = splits["X_train"]
    X_test = splits["X_test"]

    assert np.isnan(X_train).sum() == 0, (
        f"X_train contains {np.isnan(X_train).sum()} NaN values"
    )
    assert np.isnan(X_test).sum() == 0, (
        f"X_test contains {np.isnan(X_test).sum()} NaN values"
    )
