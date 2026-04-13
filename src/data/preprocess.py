"""
Preprocess MERRA-2 NetCDF files into a state-level aggregated DataFrame.

Steps:
  1. Download India state boundary GeoJSON
  2. Build grid-to-state mapping (cached as pickle)
  3. For each month: open the 3 NetCDF files, subset to India, aggregate per state
  4. Output: data/processed/merra2_india_states.csv
"""

import pickle
import logging
import requests
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import yaml
from shapely.geometry import Point
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Canonical state name normalization (handles spelling variants in shapefiles)
STATE_NAME_MAP = {
    "Orissa": "Odisha",
    "Uttaranchal": "Uttarakhand",
    "Pondicherry": "Puducherry",
    "Jammu & Kashmir": "Jammu and Kashmir",
    "Daman & Diu": "Daman and Diu",
    "Dadra & Nagar Haveli": "Dadra and Nagar Haveli",
    "Andaman & Nicobar Islands": "Andaman and Nicobar Islands",
    "The Dadra And Nagar Haveli And Daman And Diu": "Dadra and Nagar Haveli",
}


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_india_shapefile(output_dir: Path, config: dict) -> Path:
    """Download India state boundary GeoJSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "india_states.geojson"

    if out_path.exists():
        logger.info(f"Shapefile already exists: {out_path}")
        return out_path

    urls = [
        config["geography"]["shapefile_url"],
        config["geography"].get("shapefile_fallback_url", ""),
    ]

    for url in urls:
        if not url:
            continue
        try:
            logger.info(f"Downloading shapefile from {url}")
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            out_path.write_bytes(r.content)
            logger.info(f"Shapefile saved to {out_path}")
            return out_path
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")

    raise RuntimeError("Could not download India shapefile from any source")


def load_india_states(shapefile_path: Path) -> gpd.GeoDataFrame:
    """Load and normalize India state GeoDataFrame."""
    gdf = gpd.read_file(shapefile_path)

    # Ensure WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    # Find the state name column — prefer more specific matches
    priority_keys = ["st_nm", "statename", "state_name", "name_1", "name1"]
    fallback_keys = ["state", "name"]
    name_col = None
    for key in priority_keys:
        matches = [c for c in gdf.columns if c.lower() == key]
        if matches:
            name_col = matches[0]
            break
    if name_col is None:
        for key in fallback_keys:
            matches = [c for c in gdf.columns if key in c.lower()]
            if matches:
                name_col = matches[-1]  # take last match (NAME_1 over NAME_0)
                break
    if name_col is None:
        raise ValueError(f"Cannot find state name column. Columns: {list(gdf.columns)}")
    gdf = gdf.rename(columns={name_col: "state"})
    gdf["state"] = gdf["state"].str.strip().replace(STATE_NAME_MAP)

    logger.info(f"Loaded {len(gdf)} states/UTs from {shapefile_path}")
    return gdf[["state", "geometry"]]


def build_grid_to_state_mapping(
    lats: np.ndarray,
    lons: np.ndarray,
    states_gdf: gpd.GeoDataFrame,
    cache_path: Path,
) -> dict:
    """Map each MERRA-2 grid cell within India bounds to a state name.

    Uses a spatial index (STRtree) for efficient point-in-polygon queries.
    Result is cached to disk since the grid is fixed.

    Returns: dict mapping state_name -> list of (lat_idx, lon_idx) tuples
    """
    if cache_path.exists():
        logger.info(f"Loading cached grid mapping from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info(f"Building grid-to-state mapping for {len(lats)} x {len(lons)} grid...")

    # Build a spatial index on state geometries
    tree = states_gdf.sindex

    mapping: dict[str, list[tuple[int, int]]] = {
        state: [] for state in states_gdf["state"]
    }

    total = len(lats) * len(lons)
    with tqdm(total=total, desc="Mapping grid to states") as pbar:
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                pt = Point(lon, lat)
                # Candidate states via spatial index
                candidates = list(tree.intersection(pt.bounds))
                for idx in candidates:
                    row = states_gdf.iloc[idx]
                    if row.geometry.contains(pt):
                        mapping[row["state"]].append((i, j))
                        break
                pbar.update(1)

    # Remove states with no grid cells
    mapping = {k: v for k, v in mapping.items() if v}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(mapping, f)

    logger.info(f"Mapped {sum(len(v) for v in mapping.values())} grid cells "
                f"across {len(mapping)} states")
    return mapping


def open_monthly_datasets(raw_dir: Path, year: int, month: int,
                           config: dict) -> Optional[xr.Dataset]:
    """Open and merge the 3 MERRA-2 NetCDF files for a given year-month.

    Returns merged xr.Dataset or None if any file is missing.
    """
    from src.data.download import get_stream_id, build_download_url

    datasets = []
    for col_key, col_cfg in config["merra2"]["collections"].items():
        collection = col_cfg["short_name"]
        version = col_cfg["version"]
        _, filename = build_download_url(
            config["earthdata"]["base_url"], collection, version, year, month
        )
        path = raw_dir / col_key / str(year) / filename
        if not path.exists():
            logger.warning(f"Missing file: {path}")
            return None
        datasets.append(xr.open_dataset(path, engine="netcdf4"))

    # Merge all variables into one dataset
    merged = xr.merge(datasets)
    return merged


def aggregate_month_to_states(
    ds: xr.Dataset,
    india_bounds: dict,
    grid_mapping: dict,
    all_variables: list[str],
    year: int,
    month: int,
) -> list[dict]:
    """Aggregate a monthly MERRA-2 dataset to state-level means.

    Returns list of dicts, one per state.
    """
    lat_min = india_bounds["lat_min"]
    lat_max = india_bounds["lat_max"]
    lon_min = india_bounds["lon_min"]
    lon_max = india_bounds["lon_max"]

    # Subset to India bounding box
    subset = ds.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max),
    )

    # Squeeze time dimension (monthly files have time dim of size 1)
    subset = subset.squeeze("time", drop=True)

    lats = subset["lat"].values
    lons = subset["lon"].values

    rows = []
    for state, cell_indices in grid_mapping.items():
        row = {"year": year, "month": month, "state": state}

        for var in all_variables:
            if var not in subset.data_vars:
                row[var] = np.nan
                continue

            arr = subset[var].values  # shape: (lat, lon)
            values = [arr[i, j] for i, j in cell_indices
                      if i < arr.shape[0] and j < arr.shape[1]]

            valid = [v for v in values if not np.isnan(v)]
            row[var] = float(np.mean(valid)) if valid else np.nan

        rows.append(row)

    return rows


def process_all(config: Optional[dict] = None,
                 config_path: str = "config/config.yaml") -> pd.DataFrame:
    """Main pipeline: process all months and save state-aggregated CSV.

    Returns the resulting DataFrame.
    """
    if config is None:
        config = load_config(config_path)

    raw_dir = Path(config["paths"]["raw_data"])
    processed_dir = Path(config["paths"]["processed_data"])
    shapefile_dir = Path(config["paths"]["shapefiles"])
    cache_dir = Path(config["paths"].get("cache", "data/.cache"))

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "merra2_india_states.csv"

    if out_path.exists():
        logger.info(f"Processed file already exists: {out_path}. Loading...")
        return pd.read_csv(out_path)

    # Shapefile
    shapefile_path = download_india_shapefile(shapefile_dir, config)
    states_gdf = load_india_states(shapefile_path)

    # Collect all variables across collections
    all_variables = []
    for col_cfg in config["merra2"]["collections"].values():
        all_variables.extend(col_cfg["variables"])

    india_bounds = config["geography"]["india_bounds"]
    start_year = config["merra2"]["time_range"]["start_year"]
    end_year = config["merra2"]["time_range"]["end_year"]

    # Load one dataset to get grid coordinates for mapping
    logger.info("Loading a reference dataset to extract grid coordinates...")
    ref_ds = None
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            ref_ds = open_monthly_datasets(raw_dir, year, month, config)
            if ref_ds is not None:
                break
        if ref_ds is not None:
            break

    if ref_ds is None:
        raise RuntimeError("No downloaded files found. Run download.py first.")

    ref_subset = ref_ds.sel(
        lat=slice(india_bounds["lat_min"], india_bounds["lat_max"]),
        lon=slice(india_bounds["lon_min"], india_bounds["lon_max"]),
    )
    lats = ref_subset["lat"].values
    lons = ref_subset["lon"].values
    ref_ds.close()

    # Build/load grid mapping
    cache_path = cache_dir / "grid_to_state_mapping.pkl"
    grid_mapping = build_grid_to_state_mapping(lats, lons, states_gdf, cache_path)

    # Process all months
    all_rows = []
    months_processed = 0
    months_missing = 0

    total_months = (end_year - start_year + 1) * 12
    with tqdm(total=total_months, desc="Processing months") as pbar:
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                ds = open_monthly_datasets(raw_dir, year, month, config)
                if ds is None:
                    months_missing += 1
                    pbar.update(1)
                    continue

                rows = aggregate_month_to_states(
                    ds, india_bounds, grid_mapping, all_variables, year, month
                )
                all_rows.extend(rows)
                ds.close()
                months_processed += 1
                pbar.update(1)

    logger.info(f"Processed {months_processed} months, {months_missing} missing")

    df = pd.DataFrame(all_rows)
    # Sort for reproducibility
    df = df.sort_values(["year", "month", "state"]).reset_index(drop=True)

    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} rows to {out_path}")
    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    df = process_all()
    print(df.head())
    print(f"Shape: {df.shape}")
    print(df.dtypes)
    print(df.isnull().sum())
