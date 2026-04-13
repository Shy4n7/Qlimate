"""
Climate condition labeling using per-state, per-calendar-month percentile thresholds.

Classes (by priority):
  3 = Heat_Extreme  (T2M > p90 for that state-month)
  4 = Cold_Extreme  (T2M < p10 for that state-month)
  1 = Drought       (PRECTOT < p15 for that state-month)
  2 = Wet_Flood     (PRECTOT > p85 for that state-month)
  0 = Normal        (everything else)

Priority ensures each sample gets exactly one label.
Thresholds are computed per (state, calendar_month) to account for seasonality.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

LABEL_MAP = {
    "Normal": 0,
    "Drought": 1,
    "Wet_Flood": 2,
    "Heat_Extreme": 3,
    "Cold_Extreme": 4,
}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_climatology(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute per-state, per-calendar-month percentile thresholds.

    Returns DataFrame with columns:
      state, month, t2m_p10, t2m_p90, prec_p15, prec_p85
    """
    pct = config["labeling"]["percentiles"]

    def percentiles(g):
        return pd.Series({
            "t2m_p10": np.percentile(g["T2M"].dropna(), pct["cold_low"]),
            "t2m_p90": np.percentile(g["T2M"].dropna(), pct["heat_high"]),
            "prec_p15": np.percentile(g["PRECTOT"].dropna(), pct["drought_low"]),
            "prec_p85": np.percentile(g["PRECTOT"].dropna(), pct["wet_high"]),
        })

    climatology = (
        df.groupby(["state", "month"])
        .apply(percentiles)
        .reset_index()
    )
    return climatology


def assign_labels(df: pd.DataFrame, climatology: pd.DataFrame) -> pd.DataFrame:
    """Assign climate condition labels based on percentile thresholds.

    Mutates a copy of df, adding columns: label (int), label_name (str).
    """
    df = df.copy()
    merged = df.merge(climatology, on=["state", "month"], how="left")

    # Start everyone as Normal
    label = np.zeros(len(merged), dtype=int)
    label_name = np.array(["Normal"] * len(merged))

    # Apply in reverse priority order (lower priority first, higher overwrites)
    # Wet_Flood
    mask = merged["PRECTOT"] > merged["prec_p85"]
    label[mask] = LABEL_MAP["Wet_Flood"]
    label_name[mask] = "Wet_Flood"

    # Drought (overwrites Wet_Flood if both conditions somehow met - shouldn't happen)
    mask = merged["PRECTOT"] < merged["prec_p15"]
    label[mask] = LABEL_MAP["Drought"]
    label_name[mask] = "Drought"

    # Cold_Extreme
    mask = merged["T2M"] < merged["t2m_p10"]
    label[mask] = LABEL_MAP["Cold_Extreme"]
    label_name[mask] = "Cold_Extreme"

    # Heat_Extreme (highest priority - overwrites all)
    mask = merged["T2M"] > merged["t2m_p90"]
    label[mask] = LABEL_MAP["Heat_Extreme"]
    label_name[mask] = "Heat_Extreme"

    df["label"] = label
    df["label_name"] = label_name
    return df


def verify_label_distribution(df: pd.DataFrame) -> dict:
    """Print and return class distribution statistics."""
    counts = df["label_name"].value_counts()
    total = len(df)
    print("\nClass distribution:")
    print("-" * 40)
    stats = {}
    for name, count in counts.items():
        pct = 100 * count / total
        print(f"  {name:20s}: {count:5d} ({pct:.1f}%)")
        stats[name] = {"count": count, "pct": pct}

    min_pct = min(s["pct"] for s in stats.values())
    if min_pct < 5.0:
        logger.warning(
            f"Class imbalance: smallest class is {min_pct:.1f}%. "
            "Consider adjusting percentile thresholds."
        )
    print(f"\nTotal samples: {total}")
    return stats


def label_dataset(config: Optional[dict] = None,
                   config_path: str = "config/config.yaml") -> pd.DataFrame:
    """Load the aggregated CSV, apply labels, and save labeled version.

    Returns the labeled DataFrame.
    """
    if config is None:
        config = load_config(config_path)

    processed_dir = Path(config["paths"]["processed_data"])
    in_path = processed_dir / "merra2_india_states.csv"
    out_path = processed_dir / "merra2_india_labeled.csv"

    if out_path.exists():
        logger.info(f"Labeled file exists: {out_path}. Loading...")
        return pd.read_csv(out_path)

    if not in_path.exists():
        raise FileNotFoundError(
            f"Aggregated data not found: {in_path}. Run preprocess.py first."
        )

    df = pd.read_csv(in_path)
    logger.info(f"Loaded {len(df)} rows from {in_path}")

    climatology = compute_climatology(df, config)
    labeled = assign_labels(df, climatology)

    # Verify no NaN labels
    null_count = labeled["label"].isna().sum()
    if null_count > 0:
        logger.warning(f"{null_count} rows have NaN labels")

    labeled.to_csv(out_path, index=False)
    logger.info(f"Saved labeled data to {out_path}")

    verify_label_distribution(labeled)
    return labeled


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    df = label_dataset()
    print(df.head())
