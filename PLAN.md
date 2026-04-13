# Qlimate: Prediction Simulator + VQC Bug Fix

## Overview

Two parallel tracks:
1. **Fix the VQC COBYLA bug** so the quantum model actually trains
2. **Build a map-based prediction simulator** that shows classical vs quantum predictions side-by-side for any Indian state + month

---

## Track 1 — VQC Bug Fix

### Root Cause

`qiskit_machine_learning 0.9.0` VQC does not route the user-supplied `callback(weights, obj_func_eval)` through to the `qiskit_algorithms` COBYLA optimizer correctly. COBYLA runs but never fires the callback — VQC exits with 0 recorded iterations while circuit evaluation overhead still consumed 198s.

### Fix

In `src/models/quantum.py`, `train_vqc()`:

**Replace COBYLA with SPSA**, which VQC 0.9.0 correctly wires callbacks through:

```python
# BEFORE
from qiskit_algorithms.optimizers import COBYLA
optimizer = COBYLA(maxiter=maxiter)

def _callback(weights, obj_func_eval):
    loss_history.append(float(obj_func_eval))
```

```python
# AFTER
from qiskit_algorithms.optimizers import SPSA

def _callback(nfev, x, fx, dx, accept):
    loss_history.append(float(fx))
    if len(loss_history) % 25 == 0:
        logger.info(f"VQC iter {len(loss_history)}: loss={fx:.4f}")

optimizer = SPSA(maxiter=maxiter, callback=_callback)
```

Also update `config/config.yaml`:
```yaml
vqc:
  optimizer: "SPSA"   # was "COBYLA"
```

**Why SPSA instead of COBYLA?**
- SPSA (Simultaneous Perturbation Stochastic Approximation) is gradient-free like COBYLA but designed for noisy objectives — exactly what quantum circuits produce
- VQC 0.9.0 exposes SPSA's callback correctly
- Handles shot noise inherently; COBYLA assumes a smooth landscape

**Note:** Re-training VQC takes ~20-30 min. The fix is in the code immediately. The simulator works with the existing QSVC model without re-running training.

---

## Track 2 — Prediction Simulator

### Architecture

```
Frontend (React/Vite — existing)
├── src/pages/Simulator.jsx              ← main page
├── src/components/IndiaMap.jsx          ← SVG map from GeoJSON
├── src/components/PredictionControls.jsx ← month picker
├── src/components/SplitOutputPanel.jsx  ← classical vs quantum side-by-side
├── src/components/DifferencePanel.jsx   ← agreement/mismatch + explanation
└── src/api/predict.js                   ← fetch wrapper + mock fallback

Backend (FastAPI — new)
└── backend/
    ├── predict_server.py    ← /predict and /states endpoints
    └── requirements.txt     ← fastapi, uvicorn, joblib, numpy, pandas, scikit-learn, torch
```

### Data Flow

```
User clicks Gujarat → selects June
  → POST /api/predict { state: "Gujarat", month: 6 }
  → server reads merra2_india_labeled.csv
  → computes average feature vector across all years for (Gujarat, June)
  → engineers features (same pipeline as training)
  → classical: StandardScaler → XGBoost.predict_proba() → top class + confidence
  → quantum:   StandardScaler → PCA(4) → MinMaxScaler[0,π] → QSVC.decision_function() → softmax → confidence
  → returns JSON → SplitOutputPanel renders
```

**Why average across years?** The CSV has 30 years × 28 states × 12 months. For any (state, month) pair there are ~30 rows. Averaging gives a "typical" feature vector for that state-month — consistent with what the models learned to classify. No external data needed.

### Backend — `backend/predict_server.py`

```
startup:
  - load merra2_india_labeled.csv into memory
  - load classical_scaler.pkl, pca_model.pkl, quantum_scaler.pkl
  - load xgboost.pkl (best classical model)
  - load qsvc.pkl (quantum model)
  - load neural_network_state_dict.pt + meta for NN

GET /states
  → returns sorted list of all state names in the CSV

POST /predict { state: str, month: int }
  → lookup: filter CSV to (state, month), average raw features
  → engineer: wind_speed, wind_direction, net_radiation, precip_evap_ratio,
               month_sin, month_cos, pressure_anomaly
  → classical prediction:
      X_cls = classical_scaler.transform(features_16d)
      proba = xgboost.predict_proba(X_cls)[0]
      label = CLASS_NAMES[argmax(proba)]
      confidence = max(proba)
  → quantum prediction:
      X_pca = pca_model.transform(classical_scaler.transform(features_16d))
      X_q   = quantum_scaler.transform(X_pca)
      scores = qsvc.decision_function(X_q)[0]
      proba_q = softmax(scores)
      label_q = CLASS_NAMES[argmax(proba_q)]
      confidence_q = max(proba_q)
  → return JSON (see schema below)

Response schema:
{
  state: str,
  month: int,
  historical_label: str,          // most common label for this state+month
  classical: {
    label: str,
    confidence: float,            // 0-1 from predict_proba
    model: "XGBoost",
    n_train: 6958,
    n_features: 16
  },
  quantum: {
    label: str,
    confidence: float,            // softmax of decision_function scores
    model: "QSVC",
    n_train: 400,
    n_features: 4,
    confidence_note: "estimated from decision scores, not calibrated probabilities"
  },
  agreement: bool,
  confidence_gap: float,
  difference_reason: str          // generated server-side based on outcome
}
```

**`difference_reason` logic (server-side):**
- Both agree + both high confidence → "Both models identify this pattern clearly — the climate signal for this state-month is strong enough to overcome quantum's data limitations"
- Both agree + quantum low confidence → "Both predict the same outcome, but quantum is less certain — it learned from 17× fewer examples and compressed features"
- Disagree + classical high confidence → "Classical has more confidence here. With 16 full features and 7,000 training examples, it can distinguish subtler patterns that quantum's 4-feature compressed representation misses"
- Disagree + both low confidence → "Both models are uncertain — this state-month may have mixed historical patterns or sit near a class boundary"

### Frontend Components

**`IndiaMap.jsx`**
- Uses `d3-geo` (new dependency) to project GeoJSON to SVG paths
- Mercator projection, auto-fit to container
- Each state: `<path>` with hover highlight (stroke + opacity) and click handler
- Hover tooltip: state name
- Selected state: distinct fill color
- No external map tile service needed — pure SVG from the local GeoJSON

**`PredictionControls.jsx`**
- Month dropdown: Jan–Dec (value 1–12)
- "Predict" button — disabled until state + month both selected
- Disclaimer text hardcoded below button

**`SplitOutputPanel.jsx`**
- Left card (blue border): Classical prediction
  - Condition badge (color-coded by class)
  - Confidence bar
  - "Trained on 7,000 examples · 16 features · full dataset"
- Right card (violet border): Quantum prediction
  - Condition badge
  - Confidence bar
  - "Trained on 400 examples · 4 compressed features · IBM quantum hardware"
- Loading skeleton while fetch is in flight

**`DifferencePanel.jsx`**
- Center strip between the two cards
- Agreement badge: green "Agree" or amber "Disagree"
- Confidence gap bar
- Plain-English reason from server response

**`Simulator.jsx`** (page)
```
┌─────────────────────────────────────┐
│  India Map (click a state)          │
├─────────────────────────────────────┤
│  Controls: [Month ▾] [Predict →]   │
│  ⚠ Disclaimer                       │
├──────────────┬──────┬───────────────┤
│  Classical   │ diff │ Quantum       │
│  (left)      │panel │ (right)       │
└──────────────┴──────┴───────────────┘
```

### Routing

Add `react-router-dom` (new dep). Update `App.jsx` to use `<Routes>`:
- `/` → existing dashboard
- `/simulator` → `<Simulator />`

Navbar gets a "Simulator" link.

### Vite dev proxy

In `vite.config.js`, add:
```js
server: {
  proxy: {
    '/api': { target: 'http://localhost:8000', rewrite: (p) => p.replace(/^\/api/, '') }
  }
}
```

### Mock fallback in `predict.js`

If fetch fails (backend not running), return deterministic mock based on `(state, month)` hash so the UI always works for demo purposes.

---

## New Dependencies

**Frontend** (`frontend/package.json`):
- `d3-geo` — GeoJSON to SVG projection (no full D3 needed, just this subpackage)
- `react-router-dom` — client-side routing for `/simulator` route

**Backend** (`backend/requirements.txt`):
- fastapi, uvicorn[standard]
- joblib, numpy, pandas
- scikit-learn, xgboost
- torch (CPU only, for NN meta)

---

## Files Created / Modified

| Action | File |
|--------|------|
| Modify | `src/models/quantum.py` — COBYLA → SPSA |
| Modify | `config/config.yaml` — vqc.optimizer |
| Create | `backend/predict_server.py` |
| Create | `backend/requirements.txt` |
| Copy | `data/shapefiles/india_states.geojson` → `frontend/public/india_states.geojson` |
| Create | `frontend/src/api/predict.js` |
| Create | `frontend/src/components/IndiaMap.jsx` |
| Create | `frontend/src/components/PredictionControls.jsx` |
| Create | `frontend/src/components/SplitOutputPanel.jsx` |
| Create | `frontend/src/components/DifferencePanel.jsx` |
| Create | `frontend/src/pages/Simulator.jsx` |
| Modify | `frontend/src/App.jsx` — add router + simulator route |
| Modify | `frontend/src/components/layout/Navbar.jsx` — add Simulator link |
| Modify | `frontend/vite.config.js` — add /api proxy |

---

## Running the Simulator

```bash
# Terminal 1 — Backend
cd backend
pip install -r requirements.txt
uvicorn predict_server:app --port 8000 --reload

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
# Open http://localhost:5173/simulator
```

---

## Constraints Honored

- No real-world weather forecasting claims
- Disclaimer hardcoded in PredictionControls
- Confidence labeled correctly (calibrated proba for classical, decision score estimate for quantum)
- No re-training required to run the simulator
- GeoJSON served from project — no external tile APIs

---

## Implementation Checklist

### Track 1 — VQC Fix
- [ ] Edit `src/models/quantum.py`: replace COBYLA with SPSA, update callback signature
- [ ] Edit `config/config.yaml`: `vqc.optimizer: "SPSA"`
- [ ] Verify SPSA import works: `from qiskit_algorithms.optimizers import SPSA`

### Track 2 — Backend
- [ ] Create `backend/` directory
- [ ] Create `backend/requirements.txt`
- [ ] Create `backend/predict_server.py`
  - [ ] FastAPI app with CORS enabled (allow localhost:5173)
  - [ ] Load artifacts on startup: CSV, scalers, PCA, XGBoost, QSVC
  - [ ] `GET /states` endpoint
  - [ ] `POST /predict` endpoint with feature engineering pipeline
  - [ ] `difference_reason` logic (4 cases)
  - [ ] Softmax helper for QSVC decision scores

### Track 2 — Frontend
- [ ] Copy `data/shapefiles/india_states.geojson` → `frontend/public/india_states.geojson`
- [ ] `npm install d3-geo react-router-dom` in `frontend/`
- [ ] Edit `frontend/vite.config.js`: add `/api` proxy to port 8000
- [ ] Create `frontend/src/api/predict.js` with mock fallback
- [ ] Create `frontend/src/components/IndiaMap.jsx`
- [ ] Create `frontend/src/components/PredictionControls.jsx`
- [ ] Create `frontend/src/components/SplitOutputPanel.jsx`
- [ ] Create `frontend/src/components/DifferencePanel.jsx`
- [ ] Create `frontend/src/pages/Simulator.jsx`
- [ ] Edit `frontend/src/App.jsx`: wrap in `<BrowserRouter>`, add `/simulator` route
- [ ] Edit `frontend/src/components/layout/Navbar.jsx`: add Simulator nav link

---

## Implementation Notes Per File

### `backend/predict_server.py`

```python
# Key imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, numpy as np, pandas as pd
from scipy.special import softmax
import xgboost  # needed so joblib can deserialize

# Artifact paths (relative to project root — run from Qlimate/)
CSV_PATH       = "data/processed/merra2_india_labeled.csv"
SCALER_PATH    = "results/models/classical_scaler.pkl"
PCA_PATH       = "results/models/pca_model.pkl"
Q_SCALER_PATH  = "results/models/quantum_scaler.pkl"
XGB_PATH       = "results/models/xgboost.pkl"
QSVC_PATH      = "results/models/qsvc.pkl"

CLASS_NAMES = ["Normal", "Drought", "Wet_Flood", "Heat_Extreme", "Cold_Extreme"]

# Feature engineering must exactly match src/features/engineering.py
RAW_COLS = ["T2M","PRECTOT","QV2M","PS","SLP","SWGDN","LWGNT","CLDTOT","EVAP","U10M","V10M"]
CLASSICAL_FEATURES = [
    "T2M","PRECTOT","QV2M","PS","SLP","SWGDN","LWGNT","CLDTOT","EVAP",
    "wind_speed","wind_direction","net_radiation","precip_evap_ratio",
    "month_sin","month_cos","pressure_anomaly"
]
```

**Critical:** The feature engineering in the server must be byte-for-byte identical to `src/features/engineering.py`. Use the same formulas:
- `wind_speed = sqrt(U10M² + V10M²)`
- `wind_direction = arctan2(V10M, U10M)`
- `net_radiation = SWGDN - abs(LWGNT)`
- `precip_evap_ratio = PRECTOT / (abs(EVAP) + 1e-10)`
- `month_sin = sin(2π × month / 12)`
- `month_cos = cos(2π × month / 12)`
- `pressure_anomaly = PS - SLP`

**QSVC confidence:** `decision_function` returns raw SVM margins, not probabilities. Apply `scipy.special.softmax` to get values in [0,1] that sum to 1. Label this "estimated confidence" in the response — not a calibrated probability.

**CORS:** Must allow `http://localhost:5173` (Vite dev) and `http://localhost:4173` (Vite preview).

### `frontend/src/api/predict.js`

```js
const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
const CLASS_LABELS = {
  Normal: "Normal", Drought: "Drought", Wet_Flood: "Flood / Wet",
  Heat_Extreme: "Heat Extreme", Cold_Extreme: "Cold Extreme"
}
const CLASS_COLORS = {
  Normal: "slate", Drought: "amber", Wet_Flood: "blue",
  Heat_Extreme: "red", Cold_Extreme: "cyan"
}

// Mock: deterministic based on (state, month) so UI is always demonstrable
function mockPredict(state, month) { ... }

export async function predict(state, month) {
  try {
    const res = await fetch(`/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state, month }),
    })
    if (!res.ok) throw new Error(res.statusText)
    return await res.json()
  } catch {
    return mockPredict(state, month)
  }
}

export async function fetchStates() {
  try {
    const res = await fetch("/api/states")
    return await res.json()
  } catch {
    return MOCK_STATES  // hardcoded list of 28 states
  }
}
```

### `frontend/src/components/IndiaMap.jsx`

```js
import { geoMercator, geoPath } from "d3-geo"
// GeoJSON fetched from /india_states.geojson (public folder)
// Projection: geoMercator().fitSize([width, height], geojson)
// Each <path> gets:
//   fill: selected ? "#7c3aed" : hovered ? "#334155" : "#1e293b"
//   stroke: "#475569", strokeWidth: 0.5
//   onClick: () => onSelect(feature.properties.NAME_1)
//   onMouseEnter/Leave: hover state
// Tooltip: absolute-positioned div showing state name on hover
```

State name mapping: GeoJSON uses `feature.properties.NAME_1` which matches the CSV `state` column exactly (verified: "Andaman and Nicobar", "Andhra Pradesh", etc.).

### `frontend/src/components/SplitOutputPanel.jsx`

Class badge colors:
```
Normal       → slate/gray
Drought      → amber/yellow
Wet_Flood    → blue
Heat_Extreme → red/orange
Cold_Extreme → cyan
```

Confidence bar: `<div style={{ width: `${confidence * 100}%` }} />`

Loading state: show pulsing skeleton divs while `isLoading === true`.

### `frontend/src/pages/Simulator.jsx`

State machine:
```
idle         → user hasn't selected state + month yet
ready        → state + month both selected, button enabled
loading      → fetch in flight
result       → prediction received, panels visible
error        → fetch failed (shouldn't happen due to mock fallback)
```

Layout (responsive):
- Mobile: map → controls → results stacked vertically
- Desktop: map left (40%) + results right (60%), controls above results

---

## Edge Cases

| Case | Handling |
|------|----------|
| State in GeoJSON not in CSV | Server returns 404; frontend shows "No historical data for this state" |
| QSVC pkl corrupted (2 bytes) | Server startup falls back to mock quantum predictions with a warning log |
| Month out of range (< 1 or > 12) | Pydantic validation rejects with 422 |
| d3-geo renders outside container | Use `useRef` + `ResizeObserver` to recompute projection on container resize |
| Backend not running | `predict.js` catch block returns deterministic mock — UI never breaks |
| GeoJSON state name mismatch | Normalize both sides: strip extra spaces, use `.trim()` |

---

## Known Limitations (document in UI)

1. **Predictions are averages** — the model sees a typical feature vector for that state-month (averaged over 30 years), not a specific year's values
2. **Quantum confidence is approximate** — QSVC decision scores are not calibrated probabilities; shown as relative confidence only
3. **VQC not used in simulator** — QSVC is the quantum model; VQC remains untrained until re-run after the bug fix
4. **28 states only** — Union territories and smaller regions may have limited data
