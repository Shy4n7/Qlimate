"""
MERRA-2 monthly data download pipeline using NASA Earthdata authentication.

Strategy: Download each full NetCDF4 file (~50 MB), immediately subset to
India bounding box with xarray (~0.5 MB result), save the subset, delete
the full file. Total disk usage stays under 2 GB instead of ~100 GB.

Authentication: set EARTHDATA_PASSWORD environment variable.
"""

import os
import time
import logging
import tempfile
from pathlib import Path
from typing import Optional

import requests
import xarray as xr
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_stream_id(year: int) -> str:
    if year < 1992:
        return "100"
    elif year < 2001:
        return "200"
    elif year < 2011:
        return "300"
    else:
        return "400"


def build_download_url(base_url: str, collection: str, version: str,
                        year: int, month: int) -> tuple[str, str]:
    """Build the GES DISC direct download URL and expected filename.

    Returns: (url, filename)
    """
    stream = get_stream_id(year)
    yyyymm = f"{year}{month:02d}"

    collection_suffix_map = {
        "M2TMNXSLV": "tavgM_2d_slv_Nx",
        "M2TMNXFLX": "tavgM_2d_flx_Nx",
        "M2TMNXRAD": "tavgM_2d_rad_Nx",
    }
    suffix = collection_suffix_map[collection]
    filename = f"MERRA2_{stream}.{suffix}.{yyyymm}.nc4"
    url = f"{base_url}/{collection}.{version}/{year}/{filename}"
    return url, filename


def get_earthdata_session(username: str, password: str) -> requests.Session:
    """Create an authenticated NASA Earthdata session."""
    session = requests.Session()
    session.auth = (username, password)
    session.max_redirects = 10
    return session


def download_and_subset(
    session: requests.Session,
    url: str,
    output_path: Path,
    variables: list[str],
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    retries: int = 3,
) -> bool:
    """Download a full MERRA-2 file, subset to India, save subset, delete full file.

    Returns True if successful. Skips if output already exists.
    """
    if output_path.exists():
        logger.debug(f"Skip (exists): {output_path.name}")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(retries):
        tmp_full = Path(tempfile.mktemp(suffix=".nc4"))
        try:
            # Download full file to temp location
            response = session.get(url, stream=True, timeout=180)
            response.raise_for_status()

            with open(tmp_full, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 512):
                    f.write(chunk)

            # Subset to India and save
            ds = xr.open_dataset(tmp_full, engine="netcdf4")
            avail_vars = [v for v in variables if v in ds.data_vars]
            subset = ds[avail_vars].sel(
                lat=slice(lat_bounds[0], lat_bounds[1]),
                lon=slice(lon_bounds[0], lon_bounds[1]),
            )
            subset.load()  # pull into memory before closing
            ds.close()

            subset.to_netcdf(output_path)
            subset.close()
            return True

        except requests.HTTPError as e:
            if e.response.status_code == 401:
                logger.error("Authentication failed. Check EARTHDATA_PASSWORD.")
                raise
            wait = 5 * (3 ** attempt)
            logger.warning(
                f"HTTP {e.response.status_code} attempt {attempt + 1}, "
                f"retrying in {wait}s"
            )
            time.sleep(wait)

        except (requests.ConnectionError, requests.Timeout) as e:
            wait = 5 * (3 ** attempt)
            logger.warning(f"Connection error attempt {attempt + 1}, retrying in {wait}s: {e}")
            time.sleep(wait)

        except Exception as e:
            logger.warning(f"Unexpected error attempt {attempt + 1}: {e}")
            wait = 5 * (3 ** attempt)
            time.sleep(wait)

        finally:
            if tmp_full.exists():
                tmp_full.unlink(missing_ok=True)

    logger.error(f"Failed after {retries} attempts: {url}")
    return False


def download_all(
    config: Optional[dict] = None,
    config_path: str = "config/config.yaml",
    dry_run: bool = False,
) -> dict:
    """Download and subset all MERRA-2 monthly files for the configured time range.

    Saves India-subsetted NetCDF4 files to data/raw/<col_key>/<year>/<filename>.
    Full files are never kept on disk simultaneously — peak usage ~55 MB.

    Returns dict: {col_key: {(year, month): Path}}
    """
    if config is None:
        config = load_config(config_path)

    username = config["earthdata"]["username"]
    password = os.environ.get("EARTHDATA_PASSWORD")
    if not password:
        raise ValueError("Set EARTHDATA_PASSWORD environment variable")

    base_url = config["earthdata"]["base_url"]
    start_year = config["merra2"]["time_range"]["start_year"]
    end_year = config["merra2"]["time_range"]["end_year"]
    raw_dir = Path(config["paths"]["raw_data"])
    bounds = config["geography"]["india_bounds"]
    lat_bounds = (bounds["lat_min"], bounds["lat_max"])
    lon_bounds = (bounds["lon_min"], bounds["lon_max"])

    session = get_earthdata_session(username, password)

    # Build task list
    tasks = []
    for col_key, col_cfg in config["merra2"]["collections"].items():
        collection = col_cfg["short_name"]
        version = col_cfg["version"]
        variables = col_cfg["variables"]
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                url, filename = build_download_url(
                    base_url, collection, version, year, month
                )
                out_path = raw_dir / col_key / str(year) / filename
                tasks.append((col_key, collection, variables, year, month, url, out_path))

    logger.info(f"Total files: {len(tasks)} | India-subsetted, ~0.5 MB each")

    if dry_run:
        for t in tasks[:5]:
            print(f"[DRY RUN] {t[5]}")
        print(f"... and {len(tasks) - 5} more")
        return {}

    results: dict = {}
    failed: list = []

    with tqdm(total=len(tasks), desc="Downloading MERRA-2 (India subset)") as pbar:
        for col_key, collection, variables, year, month, url, out_path in tasks:
            success = download_and_subset(
                session, url, out_path, variables, lat_bounds, lon_bounds
            )
            if success:
                results.setdefault(col_key, {})[(year, month)] = out_path
            else:
                failed.append((col_key, year, month, url))
            pbar.update(1)
            time.sleep(0.3)  # polite pacing

    if failed:
        logger.warning(f"{len(failed)} downloads failed:")
        for item in failed:
            logger.warning(f"  {item[0]} {item[1]}-{item[2]:02d}")

    total_files = sum(len(v) for v in results.values())
    logger.info(f"Done. {total_files} subsetted files saved to {raw_dir}")
    return results


def verify_download(config: Optional[dict] = None,
                     config_path: str = "config/config.yaml") -> None:
    """Spot-check one downloaded subset file per collection."""
    if config is None:
        config = load_config(config_path)

    raw_dir = Path(config["paths"]["raw_data"])
    year = config["merra2"]["time_range"]["start_year"]

    for col_key, col_cfg in config["merra2"]["collections"].items():
        collection = col_cfg["short_name"]
        version = col_cfg["version"]
        _, filename = build_download_url(
            config["earthdata"]["base_url"], collection, version, year, 6
        )
        path = raw_dir / col_key / str(year) / filename

        if not path.exists():
            print(f"[MISSING] {path}")
            continue

        ds = xr.open_dataset(path)
        expected = col_cfg["variables"]
        present = [v for v in expected if v in ds.data_vars]
        missing = [v for v in expected if v not in ds.data_vars]
        size_kb = path.stat().st_size / 1024
        print(f"[OK] {col_key}/{year}/{filename} ({size_kb:.0f} KB)")
        print(f"     vars: {present}" + (f" | MISSING: {missing}" if missing else ""))
        print(f"     dims: {dict(ds.dims)}")
        ds.close()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    dry = "--dry-run" in sys.argv
    download_all(dry_run=dry)
    if not dry:
        verify_download()
