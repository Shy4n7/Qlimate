"""
Feature engineering pipeline for Qlimate.

Derives additional features from raw MERRA-2 variables,
creates stratified train/val/test splits, and prepares
a reduced PCA subset for quantum ML.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import joblib

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# Features used for classical ML (after engineering)
CLASSICAL_FEATURES = [
    "T2M", "PRECTOT", "QV2M", "PS", "SLP",
    "SWGDN", "LWGNT", "CLDTOT", "EVAP",
    "wind_speed", "wind_direction",
    "net_radiation", "precip_evap_ratio",
    "month_sin", "month_cos",
    "pressure_anomaly",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive additional features from raw MERRA-2 variables.

    Adds to a copy of df:
      - wind_speed:         sqrt(U10M^2 + V10M^2)
      - wind_direction:     arctan2(V10M, U10M) in radians
      - net_radiation:      SWGDN - abs(LWGNT)  [net surface radiation]
      - precip_evap_ratio:  PRECTOT / (EVAP + 1e-10)  [moisture balance]
      - month_sin/cos:      cyclical month encoding
      - pressure_anomaly:   PS - SLP
    """
    df = df.copy()

    df["wind_speed"] = np.sqrt(df["U10M"] ** 2 + df["V10M"] ** 2)
    df["wind_direction"] = np.arctan2(df["V10M"], df["U10M"])
    df["net_radiation"] = df["SWGDN"] - df["LWGNT"].abs()
    df["precip_evap_ratio"] = df["PRECTOT"] / (df["EVAP"].abs() + 1e-10)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["pressure_anomaly"] = df["PS"] - df["SLP"]

    return df


def prepare_splits(
    df: pd.DataFrame,
    config: dict,
    feature_cols: Optional[list[str]] = None,
) -> dict:
    """Create stratified train/val/test splits and fit scaler on train.

    Returns dict with:
      X_train, X_val, X_test   — scaled numpy arrays
      y_train, y_val, y_test   — integer label arrays
      feature_names             — list of feature column names
      scaler                    — fitted StandardScaler
      label_encoder             — fitted LabelEncoder
    """
    if feature_cols is None:
        feature_cols = [f for f in CLASSICAL_FEATURES if f in df.columns]

    df = engineer_features(df)

    # Drop rows with any NaN in features or label
    df_clean = df[feature_cols + ["label"]].dropna()
    logger.info(f"Rows after dropping NaN: {len(df_clean)} (from {len(df)})")

    X = df_clean[feature_cols].values.astype(np.float32)
    y = df_clean["label"].values.astype(int)

    split_cfg = config["data_split"]
    seed = split_cfg["random_state"]
    val_ratio = split_cfg["val"]
    test_ratio = split_cfg["test"]

    # First split off the test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=seed
    )

    # Then split val from trainval
    val_of_trainval = val_ratio / (1.0 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_of_trainval,
        stratify=y_trainval,
        random_state=seed,
    )

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info(
        f"Split sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
    )

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


def prepare_quantum_subset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_components: int = 4,
    subset_size: int = 400,
    random_state: int = 42,
    scale_range: tuple = (0.0, np.pi),
) -> dict:
    """Prepare reduced dataset for quantum ML.

    Steps:
      1. Fit PCA on X_train, transform train and test
      2. Stratified subsample of training set to subset_size
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

    # Stratified subsample of training set
    rng = np.random.default_rng(random_state)
    classes = np.unique(y_train)
    per_class = subset_size // len(classes)
    indices = []
    for c in classes:
        idx = np.where(y_train == c)[0]
        chosen = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        indices.extend(chosen.tolist())

    # Fill remaining slots randomly if needed
    remaining = subset_size - len(indices)
    if remaining > 0:
        leftover = np.setdiff1d(np.arange(len(y_train)), indices)
        extra = rng.choice(leftover, size=min(remaining, len(leftover)), replace=False)
        indices.extend(extra.tolist())

    indices = np.array(indices)
    rng.shuffle(indices)

    X_train_q = X_train_pca[indices].astype(np.float64)
    y_train_q = y_train[indices]
    X_test_q = X_test_pca.astype(np.float64)

    logger.info(
        f"Quantum training subset: {len(X_train_q)} samples, "
        f"{n_components} features (PCA), range [{lo:.2f}, {hi:.2f}]"
    )

    return {
        "X_train_q": X_train_q,
        "y_train_q": y_train_q,
        "X_test_q": X_test_q,
        "y_test_q": y_test,
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
    np.savez(
        processed_dir / "splits.npz",
        X_train=splits["X_train"],
        X_val=splits["X_val"],
        X_test=splits["X_test"],
        y_train=splits["y_train"],
        y_val=splits["y_val"],
        y_test=splits["y_test"],
    )
    np.savez(
        processed_dir / "quantum_splits.npz",
        X_train_q=quantum["X_train_q"],
        y_train_q=quantum["y_train_q"],
        X_test_q=quantum["X_test_q"],
        y_test_q=quantum["y_test_q"],
    )
    logger.info(f"Artifacts saved to {models_dir} and {processed_dir}")


def load_splits(config: dict) -> tuple[dict, dict]:
    """Load previously saved train/val/test splits and quantum subset."""
    processed_dir = Path(config["paths"]["processed_data"])
    models_dir = Path(config["paths"]["models"])

    classical = np.load(processed_dir / "splits.npz")
    quantum_data = np.load(processed_dir / "quantum_splits.npz")

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
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config()
    df = pd.read_csv(cfg["paths"]["processed_data"] + "/merra2_india_labeled.csv")

    splits = prepare_splits(df, cfg)
    quantum = prepare_quantum_subset(
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
