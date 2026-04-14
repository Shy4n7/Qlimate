"""
Qlimate Prediction Server

FastAPI backend that serves predictions from trained classical and quantum models.
Run from the Qlimate project root:

    uvicorn backend.predict_server:app --port 8000 --reload

Endpoints:
    GET  /states         → list of valid state names
    POST /predict        → { state, month } → classical + quantum predictions
    GET  /health         → server status
"""

import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.special import softmax

# Ensure project src is importable when run from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — must exactly match src/features/engineering.py
# ---------------------------------------------------------------------------

CLASS_NAMES = ["Normal", "Drought", "Wet_Flood", "Heat_Extreme", "Cold_Extreme"]

CLASS_DISPLAY = {
    "Normal":       "Normal",
    "Drought":      "Drought",
    "Wet_Flood":    "Flood / Wet",
    "Heat_Extreme": "Heat Extreme",
    "Cold_Extreme": "Cold Extreme",
}

CLASSICAL_FEATURES = [
    "T2M", "PRECTOT", "QV2M", "PS", "SLP",
    "SWGDN", "LWGNT", "CLDTOT", "EVAP",
    "wind_speed", "wind_direction",
    "net_radiation", "precip_evap_ratio",
    "month_sin", "month_cos",
    "pressure_anomaly",
]

RAW_COLS = ["T2M", "QV2M", "U10M", "V10M", "PS", "SLP",
            "PRECTOT", "EVAP", "SWGDN", "LWGNT", "CLDTOT"]

MODELS_DIR = PROJECT_ROOT / "results" / "models"
DATA_PATH  = PROJECT_ROOT / "data" / "processed" / "merra2_india_labeled.csv"

# ---------------------------------------------------------------------------
# Startup: load all artifacts once
# ---------------------------------------------------------------------------

class Artifacts:
    df: pd.DataFrame
    classical_scaler: Any
    pca: Any
    quantum_scaler: Any
    xgboost_model: Any
    qsvc_model: Any
    states: list[str]

artifacts = Artifacts()


def _load_artifacts() -> None:
    logger.info("Loading dataset...")
    artifacts.df = pd.read_csv(DATA_PATH)

    logger.info("Loading scalers and PCA...")
    artifacts.classical_scaler = joblib.load(MODELS_DIR / "classical_scaler.pkl")
    artifacts.pca              = joblib.load(MODELS_DIR / "pca_model.pkl")
    artifacts.quantum_scaler   = joblib.load(MODELS_DIR / "quantum_scaler.pkl")

    logger.info("Loading XGBoost...")
    import xgboost  # noqa: F401 — ensure joblib can deserialize XGBClassifier
    artifacts.xgboost_model = joblib.load(MODELS_DIR / "xgboost.pkl")

    logger.info("Loading QSVC...")
    qsvc_path = MODELS_DIR / "qsvc.pkl"
    if qsvc_path.stat().st_size < 10:
        logger.warning("qsvc.pkl appears empty — quantum predictions will use fallback")
        artifacts.qsvc_model = None
    else:
        try:
            artifacts.qsvc_model = joblib.load(qsvc_path)
        except Exception as e:
            logger.warning(f"Could not load QSVC: {e} — quantum predictions will use fallback")
            artifacts.qsvc_model = None

    artifacts.states = sorted(artifacts.df["state"].dropna().unique().tolist())
    logger.info(f"Ready — {len(artifacts.states)} states loaded")


# ---------------------------------------------------------------------------
# Feature engineering (mirrors src/features/engineering.py exactly)
# ---------------------------------------------------------------------------

def _engineer(row: pd.Series, month: int) -> np.ndarray:
    wind_speed        = np.sqrt(row["U10M"] ** 2 + row["V10M"] ** 2)
    wind_direction    = np.arctan2(row["V10M"], row["U10M"])
    net_radiation     = row["SWGDN"] - abs(row["LWGNT"])
    precip_evap_ratio = row["PRECTOT"] / (abs(row["EVAP"]) + 1e-10)
    month_sin         = np.sin(2 * np.pi * month / 12)
    month_cos         = np.cos(2 * np.pi * month / 12)
    pressure_anomaly  = row["PS"] - row["SLP"]

    return np.array([
        row["T2M"], row["PRECTOT"], row["QV2M"], row["PS"], row["SLP"],
        row["SWGDN"], row["LWGNT"], row["CLDTOT"], row["EVAP"],
        wind_speed, wind_direction, net_radiation, precip_evap_ratio,
        month_sin, month_cos, pressure_anomaly,
    ], dtype=np.float64)


FUTURE_YEAR_CUTOFF = 2024


def _get_feature_vector(state: str, month: int, year: int | None) -> tuple[np.ndarray, str, bool]:
    """Look up features for (state, month).

    Returns (features, historical_label, is_projection).
    - Exact year row used for 1995–2024.
    - Historical average used when year is None or year > 2024 (projection).
    """
    mask = (artifacts.df["state"] == state) & (artifacts.df["month"] == month)
    rows = artifacts.df[mask]
    if rows.empty:
        raise HTTPException(404, f"No data for state='{state}', month={month}")

    historical_label = rows["label_name"].mode().iloc[0]

    is_projection = year is None or year > FUTURE_YEAR_CUTOFF

    if not is_projection:
        year_rows = rows[rows["year"] == year]
        if year_rows.empty:
            raise HTTPException(404, f"No data for state='{state}', month={month}, year={year}")
        avg = year_rows[RAW_COLS].iloc[0]
    else:
        avg = rows[RAW_COLS].mean()

    features = _engineer(avg, month)
    return features, historical_label, is_projection


def _classical_predict(features: np.ndarray) -> dict:
    X = artifacts.classical_scaler.transform(features.reshape(1, -1))
    proba = artifacts.xgboost_model.predict_proba(X)[0]
    idx   = int(np.argmax(proba))
    return {
        "label":      CLASS_NAMES[idx],
        "label_display": CLASS_DISPLAY[CLASS_NAMES[idx]],
        "confidence": float(proba[idx]),
        "all_proba":  {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba)},
        "model":      "XGBoost",
        "n_train":    6958,
        "n_features": 16,
    }


def _quantum_predict(features: np.ndarray) -> dict:
    if artifacts.qsvc_model is None:
        return _quantum_fallback(features)

    X_scaled = artifacts.classical_scaler.transform(features.reshape(1, -1))
    X_pca    = artifacts.pca.transform(X_scaled)
    X_q      = artifacts.quantum_scaler.transform(X_pca)

    try:
        scores = artifacts.qsvc_model.decision_function(X_q)[0]
        proba  = softmax(scores).tolist()
        idx    = int(np.argmax(proba))
        return {
            "label":           CLASS_NAMES[idx],
            "label_display":   CLASS_DISPLAY[CLASS_NAMES[idx]],
            "confidence":      float(proba[idx]),
            "all_proba":       {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba)},
            "model":           "QSVC",
            "n_train":         400,
            "n_features":      4,
            "confidence_note": "Estimated from SVM decision scores via softmax — not a calibrated probability",
        }
    except Exception as e:
        logger.warning(f"QSVC inference failed: {e}")
        return _quantum_fallback(features)


def _quantum_fallback(_features: np.ndarray) -> dict:
    """Used when QSVC model is unavailable — returns uniform uncertainty."""
    uniform = 1.0 / len(CLASS_NAMES)
    return {
        "label":           "Normal",
        "label_display":   "Normal",
        "confidence":      uniform,
        "all_proba":       {c: uniform for c in CLASS_NAMES},
        "model":           "QSVC (unavailable)",
        "n_train":         400,
        "n_features":      4,
        "confidence_note": "Model unavailable — showing uniform uncertainty",
    }


def _difference_reason(classical: dict, quantum: dict) -> str:
    agree    = classical["label"] == quantum["label"]
    c_conf   = classical["confidence"]
    q_conf   = quantum["confidence"]
    HIGH     = 0.45

    if agree and c_conf >= HIGH and q_conf >= HIGH:
        return (
            "Both models agree with high confidence — the climate signal for this "
            "state-month is strong enough that even quantum's compressed, limited "
            "dataset picks it up clearly."
        )
    if agree and q_conf < HIGH:
        return (
            "Both models predict the same outcome, but quantum is less certain. "
            "Training on 17× fewer examples and only 4 compressed features means "
            "it has less evidence to be sure."
        )
    if not agree and c_conf >= HIGH:
        return (
            "The models disagree. Classical is more confident — with 16 full features "
            "and 7,000 training examples it can distinguish subtler climate patterns "
            "that quantum's compressed 4-feature representation misses."
        )
    return (
        "Both models are uncertain here. This state-month may have mixed historical "
        "patterns, or it sits near a boundary between climate classes where even "
        "the data-rich classical model hedges its prediction."
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Qlimate Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173", "http://localhost:5174",
                   "http://localhost:5175", "http://localhost:5176", "http://localhost:5177"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    _load_artifacts()


class PredictRequest(BaseModel):
    state: str = Field(..., description="Indian state name, e.g. 'Gujarat'")
    month: int = Field(..., ge=1, le=12, description="Month 1–12")
    year: int | None = Field(None, ge=1995, le=2030, description="Optional year 1995–2030 (2025+ uses historical average as projection)")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "states_loaded": len(artifacts.states)}


@app.get("/states")
def get_states() -> dict:
    return {"states": artifacts.states}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    state = req.state.strip()
    if state not in artifacts.states:
        raise HTTPException(404, f"Unknown state: '{state}'. Use GET /states for valid names.")

    features, historical_label, is_projection = _get_feature_vector(state, req.month, req.year)
    classical = _classical_predict(features)
    quantum   = _quantum_predict(features)

    agreement      = classical["label"] == quantum["label"]
    confidence_gap = abs(classical["confidence"] - quantum["confidence"])
    reason         = _difference_reason(classical, quantum)

    return {
        "state":            state,
        "month":            req.month,
        "year":             req.year,
        "is_projection":    is_projection,
        "historical_label": historical_label,
        "classical":        classical,
        "quantum":          quantum,
        "agreement":        agreement,
        "confidence_gap":   round(confidence_gap, 4),
        "difference_reason": reason,
    }
