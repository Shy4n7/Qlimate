"""
Download model artifacts from GitHub Releases at container startup.

No environment variables needed — URLs are hardcoded to the GitHub release.
Set SKIP_MODEL_DOWNLOAD=1 to skip (when files are already present locally).
"""

import os
import sys
import urllib.request
from pathlib import Path

GITHUB_RELEASE = "https://github.com/Shy4n7/Qlimate/releases/download/v1.0-models"

FILES = [
    (f"{GITHUB_RELEASE}/xgboost_regressor.pkl",   "results/models/xgboost_regressor.pkl"),
    (f"{GITHUB_RELEASE}/classical_scaler.pkl",    "results/models/classical_scaler.pkl"),
    (f"{GITHUB_RELEASE}/pca_model.pkl",           "results/models/pca_model.pkl"),
    (f"{GITHUB_RELEASE}/quantum_scaler.pkl",      "results/models/quantum_scaler.pkl"),
    (f"{GITHUB_RELEASE}/quantum_predictions.json","results/quantum_predictions.json"),
    (f"{GITHUB_RELEASE}/merra2_india_states.csv", "data/processed/merra2_india_states.csv"),
]


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} ...", flush=True)
    urllib.request.urlretrieve(url, dest)
    size_kb = dest.stat().st_size / 1024
    print(f"  -> {size_kb:.0f} KB", flush=True)


def main() -> None:
    if os.environ.get("SKIP_MODEL_DOWNLOAD") == "1":
        print("SKIP_MODEL_DOWNLOAD=1 — using local files")
        return

    print(f"Downloading model artifacts from GitHub Releases...")
    all_present = all(Path(local).exists() for _, local in FILES)
    if all_present:
        print("All model files already present, skipping download.")
        return

    for url, local_path in FILES:
        dest = Path(local_path)
        if dest.exists():
            print(f"  {dest.name} already exists, skipping")
            continue
        try:
            download(url, dest)
        except Exception as e:
            print(f"  ERROR downloading {dest.name}: {e}", file=sys.stderr)
            sys.exit(1)

    print("All model files ready.")


if __name__ == "__main__":
    main()
