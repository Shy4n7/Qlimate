"""
scripts/precompute_quantum.py

Offline precomputation of QSVR and VQR predictions for all
(state × month × year) combinations (28 × 12 × 41 = 13,776 entries).

Uses multiprocessing to parallelize across all CPU cores.

Run once after training quantum models:
    python scripts/precompute_quantum.py

Output: results/quantum_predictions.json
Structure: {state: {str(month): {str(year): {"qsvr": float|null, "vqr": float|null}}}}
"""

import json
import logging
import multiprocessing as mp
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REGRESSION_FEATURES = [
    "PRECTOT", "QV2M", "PS", "SLP", "SWGDN", "LWGNT", "CLDTOT", "EVAP",
    "wind_speed", "wind_direction", "net_radiation", "precip_evap_ratio",
    "month_sin", "month_cos", "pressure_anomaly", "year", "state_encoded",
]

TEMP_MIN = -20.0
TEMP_MAX = 55.0
HISTORICAL_YEARS = list(range(1995, 2025))
ALL_YEARS        = list(range(1995, 2036))
ALL_MONTHS       = list(range(1, 13))

DATA_PATH   = Path("data/processed/merra2_india_states.csv")
MODELS_DIR  = Path("results/models")
OUTPUT_PATH = Path("results/quantum_predictions.json")

N_WORKERS = max(1, os.cpu_count() - 1)  # leave 1 core free for OS


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["wind_speed"]        = np.sqrt(df["U10M"] ** 2 + df["V10M"] ** 2)
    df["wind_direction"]    = np.arctan2(df["V10M"], df["U10M"])
    df["net_radiation"]     = df["SWGDN"] - df["LWGNT"].abs()
    df["precip_evap_ratio"] = df["PRECTOT"] / (df["EVAP"].abs() + 1e-10)
    df["month_sin"]         = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]         = np.cos(2 * np.pi * df["month"] / 12)
    df["pressure_anomaly"]  = df["PS"] - df["SLP"]
    return df


def _clamp(value: float) -> float:
    return float(np.clip(value, TEMP_MIN, TEMP_MAX))


# ---------------------------------------------------------------------------
# Worker function — runs in a separate process
# ---------------------------------------------------------------------------

def _worker(args):
    """Process a chunk of (state, month, year) combos. Returns list of result dicts."""
    chunk, models_dir_str, df_records, le_classes = args

    warnings.filterwarnings("ignore")

    # Reload models in each worker (can't pickle Qiskit objects across processes)
    models_dir = Path(models_dir_str)
    classical_scaler = joblib.load(models_dir / "classical_scaler.pkl")
    pca              = joblib.load(models_dir / "pca_model.pkl")
    quantum_scaler   = joblib.load(models_dir / "quantum_scaler.pkl")

    qsvr_path = models_dir / "qsvr.pkl"
    vqr_path  = models_dir / "vqr.pkl"
    qsvr_model = joblib.load(qsvr_path) if qsvr_path.exists() else None
    vqr_model  = joblib.load(vqr_path)  if vqr_path.exists()  else None

    # Reconstruct DataFrame and LabelEncoder
    df = pd.DataFrame.from_records(df_records)
    le = LabelEncoder()
    le.classes_ = np.array(le_classes)

    results = []

    for state, month, year in chunk:
        enc = int(le.transform([state])[0])

        mask = (df["state"] == state) & (df["month"] == month)
        rows = df[mask]

        if rows.empty:
            continue

        if year <= 2024:
            year_rows = rows[rows["year"] == year]
            if year_rows.empty:
                continue
            feat_row = year_rows[REGRESSION_FEATURES].mean()
            feat_row["state_encoded"] = enc
        else:
            hist_rows = rows[rows["year"].isin(HISTORICAL_YEARS)]
            if hist_rows.empty:
                continue
            feat_row = hist_rows[REGRESSION_FEATURES].mean()
            feat_row["state_encoded"] = enc
            year_idx = REGRESSION_FEATURES.index("year")
            feat_row.iloc[year_idx] = float(year)

        raw_vec = feat_row.values.astype(np.float64).reshape(1, -1)
        scaled  = classical_scaler.transform(raw_vec)
        pca_vec = pca.transform(scaled)
        q_vec   = quantum_scaler.transform(pca_vec)

        qsvr_pred = None
        vqr_pred  = None

        if qsvr_model is not None:
            try:
                qsvr_pred = _clamp(float(np.asarray(qsvr_model.predict(q_vec)).ravel()[0]))
            except Exception:
                pass

        if vqr_model is not None:
            try:
                vqr_pred = _clamp(float(np.asarray(vqr_model.predict(q_vec)).ravel()[0]))
            except Exception:
                pass

        results.append((state, month, year, qsvr_pred, vqr_pred))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=== Quantum Prediction Precomputer (parallel) ===")
    logger.info(f"Using {N_WORKERS} worker processes (out of {os.cpu_count()} cores)")

    if not DATA_PATH.exists():
        logger.error(f"Data file not found: {DATA_PATH}")
        sys.exit(1)

    df_raw = pd.read_csv(DATA_PATH)
    df = _engineer_features(df_raw)

    le = LabelEncoder()
    le.fit(df["state"].astype(str).unique())
    df["state_encoded"] = le.transform(df["state"].astype(str))

    states = sorted(df["state"].unique().tolist())
    logger.info(f"States: {len(states)}, Years: {len(ALL_YEARS)}, Months: {len(ALL_MONTHS)}")

    # Build all combos
    all_combos = [(s, m, y) for s in states for m in ALL_MONTHS for y in ALL_YEARS]
    total = len(all_combos)
    logger.info(f"Total combinations: {total}")

    # Split into chunks — one per worker
    chunk_size = max(1, total // N_WORKERS)
    chunks = [all_combos[i:i + chunk_size] for i in range(0, total, chunk_size)]
    logger.info(f"Splitting into {len(chunks)} chunks of ~{chunk_size} each")

    # Serialize DataFrame as records for pickling
    df_records = df.to_dict("records")
    le_classes = le.classes_.tolist()

    worker_args = [(chunk, str(MODELS_DIR), df_records, le_classes) for chunk in chunks]

    logger.info("Starting parallel workers...")
    with mp.Pool(processes=N_WORKERS) as pool:
        all_results_nested = pool.map(_worker, worker_args)

    # Flatten results
    flat = [item for sublist in all_results_nested for item in sublist]
    logger.info(f"Total predictions collected: {len(flat)}")

    # Build nested dict
    output = {}
    for state, month, year, qsvr_pred, vqr_pred in flat:
        if state not in output:
            output[state] = {}
        sm = str(month)
        if sm not in output[state]:
            output[state][sm] = {}
        output[state][sm][str(year)] = {"qsvr": qsvr_pred, "vqr": vqr_pred}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, separators=(",", ":"))

    logger.info(f"Saved to {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.1f} KB)")
    logger.info("Done.")


if __name__ == "__main__":
    mp.freeze_support()  # needed on Windows
    main()
