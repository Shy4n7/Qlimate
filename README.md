# Qlimate

A comparative study of Classical ML and Quantum ML for **temperature forecasting** on 30 years of NASA MERRA-2 climate data across Indian states — with an interactive simulator that predicts future temperatures up to 2035.

> **This project compares computational regimes, not just performance.**
> Same climate problem. Fundamentally different computation.

---

## What This Project Does

Qlimate trains classical and quantum regression models to predict **surface temperature (°C)** for any Indian state, month, and year — including future projections up to 2035. The central question: *which computing paradigm predicts climate temperature more accurately, and how do their forecasts diverge?*

| Dimension | Classical ML | Quantum ML |
|---|---|---|
| Target | T2M temperature (°C) | T2M temperature (°C) |
| Training samples | 8,400 (1995–2019) | 400 (simulation limit) |
| Feature count | 17 | 4 (PCA compressed) |
| Best RMSE | **0.73°C** (Gradient Boosting) | 6.56°C (QSVR) |
| Best R² | **0.993** | 0.425 |
| Training time | seconds–minutes | 15 min (QSVR), 5 min (VQR) |
| Hardware | CPU/GPU workstation | Qiskit StatevectorSampler (CPU simulation) |

---

## Results

| Model | Type | MAE (°C) | RMSE (°C) | R² | Training samples |
|---|---|---|---|---|---|
| Gradient Boosting | Classical | 0.554 | **0.732** | **0.993** | 8,400 |
| XGBoost | Classical | 0.546 | 0.735 | 0.993 | 8,400 |
| Random Forest | Classical | 0.609 | 0.824 | 0.991 | 8,400 |
| Neural Network | Classical | 0.740 | 0.993 | 0.987 | 8,400 |
| Ridge Regression | Classical | 1.155 | 1.426 | 0.973 | 8,400 |
| **QSVR** | **Quantum** | 4.030 | 6.558 | 0.425 | 400 |
| **VQR** | **Quantum** | 21.883 | 23.239 | −6.22 | 400 |

Classical models achieve R² > 0.99 — predicting temperature within 0.7°C. Quantum models are constrained to 400 samples and 4 compressed features due to NISQ hardware limits, resulting in significantly higher error. This gap is the core finding.

---

## Interactive Simulator

The simulator lets you select any Indian state, month, and year (1995–2035) and see classical vs quantum temperature predictions side by side, along with a 2025–2035 forecast divergence chart.

**How future predictions work:** For years 2025–2035, both models use the 30-year historical average climate features for that state+month, with `year` set to the requested future year. The year feature captures the temperature trend learned from 1995–2024 data.

**Quantum inference note:** Quantum predictions are precomputed offline for all 13,776 (state × month × year) combinations and served from a lookup table. Real-time quantum inference is not feasible — training QSVR on 400 samples requires ~22 hours of IBM quantum computer time, exceeding available quotas. This is standard practice in quantum ML research.

---

## Data Pipeline

**Source:** NASA MERRA-2 Monthly Reanalysis (GES DISC / Earthdata)

| Property | Value |
|---|---|
| Collections | M2TMNXSLV · M2TMNXFLX · M2TMNXRAD |
| Variables | T2M, QV2M, U10M, V10M, PS, SLP, PRECTOT, EVAP, CLDTOT |
| Coverage | 1995–2024 (355 months) · 28 Indian states |
| Total rows | 9,940 |
| Split | Chronological — train 1995–2019, val 2020–2021, test 2022–2024 |
| Target | T2M_celsius = T2M − 273.15 |

**17 regression features:** PRECTOT, QV2M, PS, SLP, SWGDN, LWGNT, CLDTOT, EVAP, wind_speed, wind_direction, net_radiation, precip_evap_ratio, month_sin, month_cos, pressure_anomaly, **year**, **state_encoded**

The `year` feature is what enables temporal extrapolation to 2035 — models learn the temperature trend over 1995–2024 and project it forward.

---

## Setup

```bash
git clone <repo-url>
cd Qlimate
pip install -r requirements.txt
```

### Credentials

```bash
export EARTHDATA_PASSWORD="your_password"   # Required for data download
```

Update `config/config.yaml` with your Earthdata username.

---

## Running the Pipeline

```bash
# Full pipeline (after data download)
python run.py

# Individual stages
python run.py --only engineer     # Feature engineering + chronological splits
python run.py --only classical    # Train 5 classical regressors (~9 min)
python run.py --only quantum      # Train QSVR + VQR (~1 hour, CPU simulation)
python run.py --only precompute   # Generate quantum_predictions.json (~1.5 hours, parallel)
python run.py --only evaluate     # Compare all models, write model_comparison.csv

# Resume from a stage
python run.py --from evaluate
```

### Download MERRA-2 data

```bash
python src/data/download.py
```

~3.5 GB, 1,080 NetCDF4 files.

### Precompute quantum predictions (parallel)

```bash
python scripts/precompute_quantum.py
```

Generates `results/quantum_predictions.json` — 13,776 predictions (28 states × 12 months × 41 years) using all available CPU cores in parallel.

---

## Project Structure

```
Qlimate/
├── config/config.yaml              # All hyperparameters and paths
├── run.py                          # Pipeline orchestrator
├── scripts/
│   ├── precompute_quantum.py       # Parallel offline quantum prediction
│   └── export_metrics.py           # Export metrics to JSON
├── src/
│   ├── data/
│   │   ├── download.py             # MERRA-2 Earthdata download
│   │   └── preprocess.py           # Grid-to-state aggregation
│   ├── features/
│   │   └── engineering.py          # Feature engineering, chronological splits, PCA
│   ├── models/
│   │   ├── classical.py            # ClassicalRegressorTrainer (XGBoost, RF, Ridge, GB, NN)
│   │   └── quantum.py              # QuantumRegressorTrainer (QSVR, VQR)
│   ├── evaluation/
│   │   └── metrics.py              # MAE, RMSE, R² evaluation + comparison DataFrame
│   └── visualization/
│       └── ...                     # Static + interactive plots
├── backend/
│   └── predict_server.py           # FastAPI: classical live inference + quantum lookup
├── frontend/                       # React/Vite simulator
│   └── src/
│       ├── pages/Simulator.jsx     # Main simulator page (homepage)
│       ├── components/
│       │   ├── TemperatureOutputPanel.jsx
│       │   ├── ForecastDivergenceChart.jsx
│       │   ├── DifferencePanel.jsx
│       │   ├── PredictionControls.jsx
│       │   └── IndiaMap.jsx
│       └── api/predict.js          # API client + mock fallback
├── results/
│   ├── figures/                    # PNG figures
│   ├── models/                     # Saved model artifacts + quantum_predictions.json
│   ├── metrics/                    # JSON metric exports
│   └── model_comparison.csv        # Full comparison table (MAE, RMSE, R²)
└── data/
    ├── raw/                        # MERRA-2 NetCDF4 (gitignored)
    └── processed/                  # State CSVs + regression splits (gitignored)
```

---

## Tests

```bash
pytest tests/
```

Covers feature engineering, regression metrics, classical model interfaces, quantum model interfaces, and data pipeline integration. All tests use synthetic data — no dependency on real data files or config.

---

## Acknowledgements

- **Dataset:** [NASA MERRA-2](https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/) via GES DISC / Earthdata
- **Quantum framework:** [Qiskit](https://www.ibm.com/quantum/qiskit) + [qiskit-machine-learning](https://github.com/qiskit-community/qiskit-machine-learning) by IBM
- **India boundaries:** [Subhash9325/GeoJson-Data-of-Indian-States](https://github.com/Subhash9325/GeoJson-Data-of-Indian-States)
