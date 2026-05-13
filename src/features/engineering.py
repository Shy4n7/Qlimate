"""
Feature engineering pipeline for Qlimate (regression).

Derives additional features from raw MERRA-2 variables,
creates chronological train/val/test splits for regression,
and prepares a reduced PCA subset for quantum ML.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import joblib

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# Features used for regression (after engineering) — 17 features, no T2M Kelvin
REGRESSION_FEATURES = [
    # Raw MERRA-2 variables (T2M excluded — it becomes the target as T2M_celsius)
    "PRECTOT",           # total precipitation (kg/m²/s)
    "QV2M",              # 2m specific humidity (kg/kg)
    "PS",                # surface pressure (Pa)
    "SLP",               # sea-level pressure (Pa)
    "SWGDN",             # surface incident shortwave radiation (W/m²)
    "LWGNT",             # net longwave radiation (W/m²)
    "CLDTOT",            # total cloud fraction (0–1)
    "EVAP",              # evaporation (kg/m²/s)
    # Derived features
    "wind_speed",        # sqrt(U10M² + V10M²) (m/s)
    "wind_direction",    # arctan2(V10M, U10M) (radians)
    "net_radiation",     # SWGDN - |LWGNT| (W/m²)
    "precip_evap_ratio", # PRECTOT / (|EVAP| + 1e-10)
    "month_sin",         # sin(2π * month / 12)
    "month_cos",         # cos(2π * month / 12)
    "pressure_anomaly",  # PS - SLP (Pa)
    # Temporal and spatial features
    "year",              # integer year (1995–2035), enables temporal extrapolation
    "state_encoded",     # LabelEncoder integer for state name
]

# Regression target
TARGET = "T2M_celsius"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive additional features from raw MERRA-2 variables.

    Adds to a copy of df:
      - T2M_celsius:        T2M - 273.15  (Kelvin → Celsius)
      - wind_speed:         sqrt(U10M^2 + V10M^2)
      - wind_direction:     arctan2(V10M, U10M) in radians
      - net_radiation:      SWGDN - abs(LWGNT)  [net surface radiation]
      - precip_evap_ratio:  PRECTOT / (abs(EVAP) + 1e-10)  [moisture balance]
      - month_sin/cos:      cyclical month encoding
      - pressure_anomaly:   PS - SLP
      - year:               integer year (already present in CSV, kept as-is)
      - state_encoded:      LabelEncoder integer for state column
    """
    df = df.copy()

    # Convert T2M from Kelvin to Celsius — this is the regression target
    df["T2M_celsius"] = df["T2M"] - 273.15

    # Derived meteorological features (unchanged from original)
    df["wind_speed"] = np.sqrt(df["U10M"] ** 2 + df["V10M"] ** 2)
    df["wind_direction"] = np.arctan2(df["V10M"], df["U10M"])
    df["net_radiation"] = df["SWGDN"] - df["LWGNT"].abs()
    df["precip_evap_ratio"] = df["PRECTOT"] / (df["EVAP"].abs() + 1e-10)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["pressure_anomaly"] = df["PS"] - df["SLP"]

    # year is already present in the CSV as an integer column — no transformation needed

    # Encode state as a non-negative integer
    le = LabelEncoder()
    df["state_encoded"] = le.fit_transform(df["state"].astype(str))

    return df


def prepare_regression_splits(
    df: pd.DataFrame,
    config: dict,
    feature_cols: Optional[list] = None,
) -> dict:
    """Create chronological train/val/test splits for regression.

    Split boundaries (by year):
      - train: 1995–2019
      - val:   2020–2021
      - test:  2022–2024

    Returns dict with:
      X_train, X_val, X_test   — StandardScaler-normalized float32 arrays
      y_train, y_val, y_test   — float32 arrays of T2M_celsius
      feature_names             — list of feature column names
      scaler                    — fitted StandardScaler
    """
    if feature_cols is None:
        feature_cols = REGRESSION_FEATURES

    df = engineer_features(df)

    # Drop rows with any NaN in features or target
    # year is already in feature_cols (it's one of REGRESSION_FEATURES), so no need to add it separately
    cols_needed = feature_cols + [TARGET]
    df_clean = df[cols_needed].dropna()
    logger.info(f"Rows after dropping NaN: {len(df_clean)} (from {len(df)})")

    # Chronological split — year is already in feature_cols so it's available in df_clean
    train_mask = df_clean["year"] <= 2019
    val_mask = (df_clean["year"] >= 2020) & (df_clean["year"] <= 2021)
    test_mask = (df_clean["year"] >= 2022) & (df_clean["year"] <= 2024)

    train_df = df_clean[train_mask]
    val_df = df_clean[val_mask]
    test_df = df_clean[test_mask]

    X_train = train_df[feature_cols].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)

    y_train = train_df[TARGET].values.astype(np.float32)
    y_val = val_df[TARGET].values.astype(np.float32)
    y_test = test_df[TARGET].values.astype(np.float32)

    logger.info(
        f"Chronological split — train: {len(X_train)} (1995–2019), "
        f"val: {len(X_val)} (2020–2021), test: {len(X_test)} (2022–2024)"
    )

    # Fit scaler on training data only; apply to val and test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": feature_cols,
        "scaler": scaler,
    }


def prepare_quantum_regression_subset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_components: int = 4,
    subset_size: int = 400,
    random_state: int = 42,
    scale_range: tuple = (0.0, np.pi),
) -> dict:
    """Prepare reduced dataset for quantum regression ML.

    Steps:
      1. Fit PCA on X_train, transform train and test
      2. Random subsample of training set to subset_size (no stratification for regression)
      3. Normalize PCA features to scale_range (default [0, pi])

    Returns dict with:
      X_train_q, y_train_q  — subset for QML training
      X_test_q, y_test_q    — full test set (PCA reduced, normalized)
      pca                   — fitted PCA model
      scaler_q              — fitted MinMaxScaler
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    explained = pca.explained_variance_ratio_.sum()
    logger.info(
        f"PCA({n_components}) explains {explained:.1%} of variance"
    )

    # Normalize to [0, pi] for quantum angle encoding
    lo, hi = scale_range
    q_scaler = MinMaxScaler(feature_range=(lo, hi))
    X_train_pca = q_scaler.fit_transform(X_train_pca)
    X_test_pca = q_scaler.transform(X_test_pca)

    # Random subsample of training set (no stratification needed for regression)
    rng = np.random.default_rng(random_state)
    n_available = len(y_train)
    actual_size = min(subset_size, n_available)
    indices = rng.choice(n_available, size=actual_size, replace=False)

    X_train_q = X_train_pca[indices].astype(np.float64)
    y_train_q = y_train[indices].astype(np.float64)
    X_test_q = X_test_pca.astype(np.float64)
    y_test_q = y_test.astype(np.float64)

    logger.info(
        f"Quantum training subset: {len(X_train_q)} samples, "
        f"{n_components} features (PCA), range [{lo:.2f}, {hi:.2f}]"
    )

    return {
        "X_train_q": X_train_q,
        "y_train_q": y_train_q,
        "X_test_q": X_test_q,
        "y_test_q": y_test_q,
        "pca": pca,
        "scaler_q": q_scaler,
        "pca_explained_variance": explained,
    }


def save_artifacts(splits: dict, quantum: dict, config: dict) -> None:
    """Save scaler, PCA, and split data to disk."""
    models_dir = Path(config["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(splits["scaler"], models_dir / "classical_scaler.pkl")
    joblib.dump(quantum["pca"], models_dir / "pca_model.pkl")
    joblib.dump(quantum["scaler_q"], models_dir / "quantum_scaler.pkl")

    processed_dir = Path(config["paths"]["processed_data"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        processed_dir / "regression_splits.npz",
        X_train=splits["X_train"],
        X_val=splits["X_val"],
        X_test=splits["X_test"],
        y_train=splits["y_train"],
        y_val=splits["y_val"],
        y_test=splits["y_test"],
    )
    np.savez(
        processed_dir / "quantum_regression_splits.npz",
        X_train_q=quantum["X_train_q"],
        y_train_q=quantum["y_train_q"],
        X_test_q=quantum["X_test_q"],
        y_test_q=quantum["y_test_q"],
    )
    logger.info(f"Artifacts saved to {models_dir} and {processed_dir}")


def load_splits(config: dict) -> tuple:
    """Load previously saved train/val/test regression splits and quantum subset."""
    processed_dir = Path(config["paths"]["processed_data"])
    models_dir = Path(config["paths"]["models"])

    classical = np.load(processed_dir / "regression_splits.npz")
    quantum_data = np.load(processed_dir / "quantum_regression_splits.npz")

    splits = {
        "X_train": classical["X_train"],
        "X_val": classical["X_val"],
        "X_test": classical["X_test"],
        "y_train": classical["y_train"],
        "y_val": classical["y_val"],
        "y_test": classical["y_test"],
        "scaler": joblib.load(models_dir / "classical_scaler.pkl"),
    }
    quantum = {
        "X_train_q": quantum_data["X_train_q"],
        "y_train_q": quantum_data["y_train_q"],
        "X_test_q": quantum_data["X_test_q"],
        "y_test_q": quantum_data["y_test_q"],
        "pca": joblib.load(models_dir / "pca_model.pkl"),
        "scaler_q": joblib.load(models_dir / "quantum_scaler.pkl"),
    }
    return splits, quantum


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config()
    df = pd.read_csv(cfg["paths"]["processed_data"] + "/merra2_india_states.csv")

    splits = prepare_regression_splits(df, cfg)
    quantum = prepare_quantum_regression_subset(
        splits["X_train"], splits["y_train"],
        splits["X_test"], splits["y_test"],
        n_components=cfg["quantum_ml"]["pca_components"],
        subset_size=cfg["quantum_ml"]["training_subset_size"],
        random_state=cfg["data_split"]["random_state"],
    )
    save_artifacts(splits, quantum, cfg)

    print(f"Train: {splits['X_train'].shape}, Val: {splits['X_val'].shape}, "
          f"Test: {splits['X_test'].shape}")
    print(f"Quantum train: {quantum['X_train_q'].shape}, "
          f"PCA explains: {quantum['pca_explained_variance']:.1%}")
