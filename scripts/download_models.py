"""
Download model artifacts from cloud storage at container startup.

Set these environment variables:
  MODEL_STORAGE_URL  — base URL of your R2/S3 bucket (no trailing slash)
                       e.g. https://pub-abc123.r2.dev

Files downloaded:
  results/models/xgboost_regressor.pkl
  results/models/classical_scaler.pkl
  results/models/pca_model.pkl
  results/models/quantum_scaler.pkl
  results/quantum_predictions.json
  data/processed/merra2_india_states.csv

If MODEL_STORAGE_URL is not set, this script does nothing (assumes files
are already present, e.g. in local Docker build).
"""

import os
import sys
import urllib.request
from pathlib import Path

BASE_URL = os.environ.get("MODEL_STORAGE_URL", "").rstrip("/")

FILES = [
    ("results/models/xgboost_regressor.pkl",   "results/models/xgboost_regressor.pkl"),
    ("results/models/classical_scaler.pkl",    "results/models/classical_scaler.pkl"),
    ("results/models/pca_model.pkl",           "results/models/pca_model.pkl"),
    ("results/models/quantum_scaler.pkl",      "results/models/quantum_scaler.pkl"),
    ("results/quantum_predictions.json",       "results/quantum_predictions.json"),
    ("data/processed/merra2_india_states.csv", "data/processed/merra2_india_states.csv"),
]


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest} ...", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  -> {dest.stat().st_size / 1024:.0f} KB", flush=True)


def main() -> None:
    if not BASE_URL:
        print("MODEL_STORAGE_URL not set — skipping model download (using local files)")
        return

    print(f"Downloading models from {BASE_URL} ...")
    for remote_path, local_path in FILES:
        dest = Path(local_path)
        if dest.exists():
            print(f"  {dest} already exists, skipping")
            continue
        url = f"{BASE_URL}/{remote_path}"
        try:
            download(url, dest)
        except Exception as e:
            print(f"  ERROR downloading {url}: {e}", file=sys.stderr)
            sys.exit(1)

    print("All model files ready.")


if __name__ == "__main__":
    main()
