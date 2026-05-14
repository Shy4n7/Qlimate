"""
Qlimate Prediction Server (Regression)

FastAPI backend that serves temperature predictions from trained classical and
quantum regression models.

Run from the Qlimate project root:

    uvicorn backend.predict_server:app --port 8000 --reload

Endpoints:
    GET  /health                        → server status
    GET  /states                        → sorted list of valid state names
    POST /predict                       → PredictResponse (classical + quantum temps)
    GET  /forecast?state=...&month=...  → 2025–2035 forecast series
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.preprocessing import LabelEncoder

# Ensure project src is importable when run from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — must exactly match src/features/engineering.py
# ---------------------------------------------------------------------------

REGRESSION_FEATURES = [
    "PRECTOT",
    "QV2M",
    "PS",
    "SLP",
    "SWGDN",
    "LWGNT",
    "CLDTOT",
    "EVAP",
    "wind_speed",
    "wind_direction",
    "net_radiation",
    "precip_evap_ratio",
    "month_sin",
    "month_cos",
    "pressure_anomaly",
    "year",
    "state_encoded",
]

# Raw columns needed from the CSV to compute derived features
RAW_COLS = ["QV2M", "U10M", "V10M", "PS", "SLP", "PRECTOT", "EVAP", "SWGDN", "LWGNT", "CLDTOT"]

MODELS_DIR = PROJECT_ROOT / "results" / "models"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "merra2_india_states.csv"
QUANTUM_PREDICTIONS_PATH = PROJECT_ROOT / "results" / "quantum_predictions.json"

# Physical plausibility bounds for Indian states (°C)
TEMP_MIN_C = -20.0
TEMP_MAX_C = 55.0

# Year boundary between historical and future
FUTURE_YEAR_CUTOFF = 2024


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    state: str = Field(..., description="Indian state name, e.g. 'Gujarat'")
    month: int = Field(..., ge=1, le=12, description="Month 1–12")
    year: int = Field(..., ge=1995, le=2035, description="Year 1995–2035")


class TemperaturePrediction(BaseModel):
    model: str
    predicted_temp_c: Optional[float]
    n_train: int
    n_features: int


class PredictResponse(BaseModel):
    state: str
    month: int
    year: int
    is_future: bool
    classical: TemperaturePrediction
    quantum: TemperaturePrediction
    temp_delta_c: Optional[float]
    divergence_note: str


# ---------------------------------------------------------------------------
# Startup: load all artifacts once
# ---------------------------------------------------------------------------

class Artifacts:
    df: pd.DataFrame
    classical_scaler: Any
    pca: Any
    quantum_scaler: Any
    xgboost_model: Any
    quantum_predictions: dict
    states: list
    label_encoder: LabelEncoder


artifacts = Artifacts()


def _load_artifacts() -> None:
    logger.info("Loading dataset from merra2_india_states.csv ...")
    artifacts.df = pd.read_csv(DATA_PATH)

    # Fit LabelEncoder on all states in the CSV (same as training)
    all_states = sorted(artifacts.df["state"].dropna().unique().tolist())
    artifacts.label_encoder = LabelEncoder()
    artifacts.label_encoder.fit(all_states)
    artifacts.states = all_states
    logger.info(f"States loaded: {len(artifacts.states)}")

    logger.info("Loading scalers and PCA ...")
    artifacts.classical_scaler = joblib.load(MODELS_DIR / "classical_scaler.pkl")
    artifacts.pca = joblib.load(MODELS_DIR / "pca_model.pkl")
    artifacts.quantum_scaler = joblib.load(MODELS_DIR / "quantum_scaler.pkl")

    logger.info("Loading XGBoost Regressor ...")
    xgb_path = MODELS_DIR / "xgboost_regressor.pkl"
    if not xgb_path.exists():
        # Fall back to xgboost.pkl if the regressor-specific file isn't present yet
        xgb_path = MODELS_DIR / "xgboost.pkl"
        logger.warning(
            "xgboost_regressor.pkl not found — falling back to xgboost.pkl. "
            "Re-run the training pipeline to generate the regressor."
        )
    artifacts.xgboost_model = joblib.load(xgb_path)

    logger.info("Loading quantum predictions lookup table ...")
    if QUANTUM_PREDICTIONS_PATH.exists():
        with open(QUANTUM_PREDICTIONS_PATH, "r") as f:
            artifacts.quantum_predictions = json.load(f)
        logger.info("quantum_predictions.json loaded successfully")
    else:
        logger.warning(
            "quantum_predictions.json not found — quantum predictions will be unavailable. "
            "Run scripts/precompute_quantum.py to generate it."
        )
        artifacts.quantum_predictions = {}

    logger.info("Server ready.")


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def _engineer_features(row: pd.Series, month: int, year: int, state: str) -> np.ndarray:
    """Compute the 17-feature regression vector from a raw MERRA-2 row.

    Mirrors the logic in src/features/engineering.py exactly.
    """
    wind_speed = np.sqrt(row["U10M"] ** 2 + row["V10M"] ** 2)
    wind_direction = np.arctan2(row["V10M"], row["U10M"])
    net_radiation = row["SWGDN"] - abs(row["LWGNT"])
    precip_evap_ratio = row["PRECTOT"] / (abs(row["EVAP"]) + 1e-10)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    pressure_anomaly = row["PS"] - row["SLP"]
    state_encoded = int(artifacts.label_encoder.transform([state])[0])

    return np.array([
        row["PRECTOT"],
        row["QV2M"],
        row["PS"],
        row["SLP"],
        row["SWGDN"],
        row["LWGNT"],
        row["CLDTOT"],
        row["EVAP"],
        wind_speed,
        wind_direction,
        net_radiation,
        precip_evap_ratio,
        month_sin,
        month_cos,
        pressure_anomaly,
        float(year),
        float(state_encoded),
    ], dtype=np.float64)


def _get_feature_vector(
    state: str,
    month: int,
    year: int,
    df: pd.DataFrame,
    scaler: Any,
) -> np.ndarray:
    """Build a scaled 17-feature vector for (state, month, year).

    - year <= 2024: filter df to rows where state==state AND month==month AND year==year,
      take mean of raw features.
    - year > 2024: filter df to rows where state==state AND month==month (all years),
      take mean of raw features, then set year=requested_year.

    Returns a (1, 17) scaled array ready for model inference.
    Raises HTTP 404 if no rows found for (state, month).
    """
    mask = (df["state"] == state) & (df["month"] == month)
    rows = df[mask]

    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No historical data for state='{state}', month={month}",
        )

    if year <= FUTURE_YEAR_CUTOFF:
        year_rows = rows[rows["year"] == year]
        if year_rows.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data for state='{state}', month={month}, year={year}",
            )
        avg = year_rows[RAW_COLS].mean()
        effective_year = year
    else:
        # Future year: use historical mean climate variables, override year
        avg = rows[RAW_COLS].mean()
        effective_year = year

    feature_vec = _engineer_features(avg, month, effective_year, state)
    scaled = scaler.transform(feature_vec.reshape(1, -1))
    return scaled


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _clamp_temp(value: float, state: str, month: int, year: int) -> tuple[float, bool]:
    """Clamp temperature to physical bounds [-20, 55] °C.

    Returns (clamped_value, was_clamped).
    """
    if value < TEMP_MIN_C:
        logger.warning(
            f"Predicted temp {value:.2f}°C below minimum for "
            f"state={state}, month={month}, year={year} — clamping to {TEMP_MIN_C}°C"
        )
        return TEMP_MIN_C, True
    if value > TEMP_MAX_C:
        logger.warning(
            f"Predicted temp {value:.2f}°C above maximum for "
            f"state={state}, month={month}, year={year} — clamping to {TEMP_MAX_C}°C"
        )
        return TEMP_MAX_C, True
    return value, False


def _classical_predict(
    state: str,
    month: int,
    year: int,
) -> TemperaturePrediction:
    """Run live XGBoost Regressor inference for (state, month, year)."""
    X = _get_feature_vector(state, month, year, artifacts.df, artifacts.classical_scaler)
    raw_pred = float(artifacts.xgboost_model.predict(X)[0])
    clamped, was_clamped = _clamp_temp(raw_pred, state, month, year)

    return TemperaturePrediction(
        model="XGBoost Regressor",
        predicted_temp_c=round(clamped, 4),
        n_train=6958,
        n_features=17,
    )


def _quantum_predict(
    state: str,
    month: int,
    year: int,
) -> TemperaturePrediction:
    """Look up precomputed QSVR prediction for (state, month, year).

    Returns a TemperaturePrediction with predicted_temp_c=None if unavailable.
    """
    try:
        raw_pred = artifacts.quantum_predictions[state][str(month)][str(year)]["qsvr"]
        clamped, was_clamped = _clamp_temp(float(raw_pred), state, month, year)
        return TemperaturePrediction(
            model="QSVR",
            predicted_temp_c=round(clamped, 4),
            n_train=400,
            n_features=4,
        )
    except (KeyError, TypeError):
        return TemperaturePrediction(
            model="QSVR (unavailable)",
            predicted_temp_c=None,
            n_train=400,
            n_features=4,
        )


def _divergence_note(
    classical: TemperaturePrediction,
    quantum: TemperaturePrediction,
) -> str:
    """Generate a human-readable note about classical vs quantum agreement."""
    if quantum.predicted_temp_c is None:
        return "Quantum predictions unavailable — showing classical forecast only"

    delta = abs(quantum.predicted_temp_c - classical.predicted_temp_c)

    if delta < 0.5:
        return "Classical and quantum models agree closely on this forecast"
    if delta < 2.0:
        return "Minor divergence between models — quantum trained on 17x fewer samples"
    return (
        "Significant divergence — quantum's compressed 4-feature representation "
        "diverges from classical's full 17-feature prediction"
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Qlimate Prediction API (Regression)", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:4173",
        "http://localhost:80",
        "http://localhost",
        # Render frontend — update this after deploying frontend
        "https://qlimate-frontend.onrender.com",
        # Allow any onrender.com subdomain for flexibility
        "https://*.onrender.com",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    _load_artifacts()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "states_loaded": len(artifacts.states)}


@app.get("/states")
def get_states() -> dict:
    return {"states": artifacts.states}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    state = req.state.strip()
    if state not in artifacts.states:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown state: '{state}'. Use GET /states for valid names.",
        )

    is_future = req.year > FUTURE_YEAR_CUTOFF

    classical = _classical_predict(state, req.month, req.year)
    quantum = _quantum_predict(state, req.month, req.year)

    if quantum.predicted_temp_c is not None and classical.predicted_temp_c is not None:
        temp_delta_c = round(quantum.predicted_temp_c - classical.predicted_temp_c, 4)
    else:
        temp_delta_c = None

    note = _divergence_note(classical, quantum)

    return PredictResponse(
        state=state,
        month=req.month,
        year=req.year,
        is_future=is_future,
        classical=classical,
        quantum=quantum,
        temp_delta_c=temp_delta_c,
        divergence_note=note,
    )


@app.get("/forecast")
def forecast(
    state: str = Query(..., description="Indian state name"),
    month: int = Query(..., ge=1, le=12, description="Month 1–12"),
) -> dict:
    """Return classical and quantum temperature forecasts for years 2025–2035."""
    state = state.strip()
    if state not in artifacts.states:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown state: '{state}'. Use GET /states for valid names.",
        )

    forecast_years = list(range(2025, 2036))
    classical_temps = []
    quantum_temps = []

    for yr in forecast_years:
        # Classical: live inference
        c_pred = _classical_predict(state, month, yr)
        classical_temps.append(c_pred.predicted_temp_c)

        # Quantum: precomputed lookup
        q_pred = _quantum_predict(state, month, yr)
        quantum_temps.append(q_pred.predicted_temp_c)

    return {
        "state": state,
        "month": month,
        "years": forecast_years,
        "classical_temps": classical_temps,
        "quantum_temps": quantum_temps,
    }
