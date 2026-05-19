"""
build_tensor.py
================

Convert the 454 per-detector × 12 monthly CSV files into a single
(T, N, F) PyTorch tensor + (T, N) boolean mask + metadata, saved to
'berlin_traffic_tensor.pt'.

The information pipeline between raw CSVs and the ST-GNN.

Pipeline (built incrementally):
  Step 1: build_master_time_index() + find_valid_detectors()
  Step 2: load_one_detector()
  Step 3: quality mask + missing-value handling
  Step 4: feature engineering (cyclic time, etc.)
  Step 5: stack into (T, N, F) tensor
  Step 6: train-split-only z-score normalization
  Step 7: save .pt
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd


# --- Configuration -----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_ROOT = PROJECT_ROOT / "berlin_traffic_data" / "2023" / "CSV_data"
GEOJSON_PATH = PROJECT_ROOT / "Standorte_Verkehrsdetektion_Berlin.geojson"
OUTPUT_PATH = PROJECT_ROOT / "berlin_traffic_tensor.pt"

YEAR = 2023

# Detector IDs that match this pattern are the "clean" Berlin VMZ format.
# Files like 'teuscalaS00000DD...' use a legacy format some of which have no GeoJSON entry
# and are excluded.
DETECTOR_ID_PATTERN = re.compile(r"^TEU\d+_Det\d+$")


# --- Step 1: define the two axes of the tensor -------------------------------

def build_master_time_index(year: int = YEAR) -> pd.DatetimeIndex:
    """
    Return the canonical hourly UTC index for `year`.

    Why this exists
    ---------------
    Every detector's timeseries will be reindexed onto this axis. That turns
    'irregular, possibly-missing timestamps' into a regular grid where
    position == time. A model that consumes a sliding window of L hours can
    then trust that row i+1 is exactly one hour after row i — no silent gaps.

    For 2023 (non-leap) this is 365 * 24 = 8760 entries.
    """
    start = pd.Timestamp(year=year, month=1, day=1, hour=0, tz="UTC")
    end = pd.Timestamp(year=year + 1, month=1, day=1, hour=0, tz="UTC")
    # 'left'-closed, so we include 00:00 of Jan 1 and exclude 00:00 of next Jan 1.
    idx = pd.date_range(start=start, end=end, freq="1h", inclusive="left")
    assert len(idx) == 8760, f"Expected 8760 hourly stamps for {year}, got {len(idx)}"
    return idx


def find_valid_detectors(
    csv_root: Path = CSV_ROOT,
    geojson_path: Path = GEOJSON_PATH,
    sample_month: str = "01",
) -> list[str]:
    """
    Return the sorted list of detector IDs that have BOTH:
      - at least one monthly CSV file (we probe `sample_month`), and
      - a location entry in the GeoJSON.

    Why a sorted intersection
    -------------------------
    Sorting makes the node order deterministic, so 'row 42 of the tensor'
    always means the same detector across runs. Intersection prevents two
    kinds of silent bug: detectors with data but no location (we couldn't
    place them on the graph) and detectors with location but no data
    (they'd become an all-NaN column).

    Why probe only one month
    ------------------------
    All 12 monthly archives contain the same detector set (verified during
    EDA), so probing one is enough and avoids 12x the directory scanning.
    """
    # 1. Detectors that have CSV data (TEU-format only; some of the 19 legacy
    #    'teuscala*' files have no GeoJSON entry and would be dropped anyway).
    month_dir = csv_root / sample_month
    if not month_dir.is_dir():
        raise FileNotFoundError(f"Sample month directory not found: {month_dir}")

    csv_detectors = {
        p.stem  # filename without .csv
        for p in month_dir.glob("*.csv")
        if DETECTOR_ID_PATTERN.match(p.stem)
    }

    # 2. Detectors that have a location entry in the GeoJSON.
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    geo_detectors = {feat["properties"]["teuID"] for feat in gj["features"]}

    # 3. Intersection, sorted for reproducibility.
    valid = sorted(csv_detectors & geo_detectors)

    if not valid:
        raise RuntimeError(
            "No detectors are present in both the CSV folder and the GeoJSON. "
            "Did the extraction step run, and is the GeoJSON the right one?"
        )
    return valid


# --- Step 2: load one detector's full year, on the master axis ---------------

# Columns we keep from each CSV. Everything else (ZScore_*, hist_cor, redundant
# time columns) is dropped here so it can't accidentally feed the model later.
FEATURE_COLS = ["qkfz", "qpkw", "qlkw", "vkfz", "vpkw", "vlkw"]
QUALITY_COL = "Vollständigkeit"
TIME_COL = "utc"
KEEP_COLS = [TIME_COL, QUALITY_COL] + FEATURE_COLS

CSV_SEP = ";"  # Berlin VMZ CSVs use semicolon (European convention)


def load_one_detector(
    detector_id: str,
    master_index: pd.DatetimeIndex,
    csv_root: Path = CSV_ROOT,
) -> pd.DataFrame:
    """
    Load all 12 monthly CSVs for `detector_id` and return a DataFrame
    indexed on `master_index` (length 8760).

    Output
    ------
    DataFrame with:
      - index: master_index (tz-aware UTC, hourly, length 8760)
      - columns: [Vollständigkeit, qkfz, qpkw, qlkw, vkfz, vpkw, vlkw]
      - rows missing from the CSVs appear as NaN rows (do not delete them)

    Why this shape
    --------------
    Every detector becomes a column-stack of identical length on identical
    timestamps. That is exactly what the (T, N, F) tensor wants. Reindexing
    is the operation that gets us there from irregular per-month CSVs.

    Implementation notes
    --------------------
    1. We read only KEEP_COLS to save memory (we have ~5800 files total).
    2. We force the source index to tz-aware UTC because the master index is
       tz-aware UTC; reindex raises on a tz mismatch.
    3. We drop duplicate timestamps defensively — month archives can overlap
       by an hour at boundaries.
    4. Missing months are silently skipped (warned). For a detector in our
       valid intersection this shouldn't happen, but we don't crash on it.
    """
    # 1. Find this detector's 12 monthly files.
    parts: list[pd.DataFrame] = []
    for month_dir in sorted(csv_root.iterdir()):
        if not month_dir.is_dir():
            continue
        path = month_dir / f"{detector_id}.csv"
        if not path.is_file():
            continue  # quietly tolerate a missing month
        parts.append(
            pd.read_csv(path, sep=CSV_SEP, usecols=KEEP_COLS, low_memory=False)
        )

    if not parts:
        # All 12 months missing — caller-side bug, fail loud.
        raise FileNotFoundError(
            f"No monthly CSVs found for {detector_id} under {csv_root}"
        )

    # 2. Concatenate, parse utc, normalise tz, set as index.
    df = pd.concat(parts, ignore_index=True)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)  # forces tz-aware UTC

    # 3. Defensive: sort and dedupe before reindex (see docstring).
    df = (
        df.sort_values(TIME_COL)
          .drop_duplicates(subset=TIME_COL, keep="first")
          .set_index(TIME_COL)
    )

    # 4. The key step: align onto the canonical hourly grid.
    #    Missing hours become rows of NaN. Extra rows outside the year drop.
    df = df.reindex(master_index)

    # 5. Return only the feature/quality columns in a fixed order.
    return df[[QUALITY_COL] + FEATURE_COLS]


# --- Step 2: load one detector's full year onto the master axis -------------

# Columns we keep from each CSV. Everything else (ZScore_*, hist_cor,
# localTime, month, Datum, Stunde des Tages) is either redundant with `utc`
# or unused per the README's design decisions.
USE_COLS = [
    "utc",
    "Vollständigkeit",
    "qkfz", "qpkw", "qlkw",
    "vkfz", "vpkw", "vlkw",
]


def load_one_detector(
    detector_id: str,
    master_index: pd.DatetimeIndex,
    csv_root: Path = CSV_ROOT,
) -> pd.DataFrame:
    """
    Read all monthly CSVs for `detector_id`, concatenate, and reindex onto
    `master_index`.

    Returns
    -------
    pd.DataFrame
        Indexed by UTC, length == len(master_index). Hours not present in
        any monthly CSV become rows of NaN (visible as missing data).
        Columns: USE_COLS (minus 'utc', which is the index).

    Why reindex
    -----------
    The raw CSV rows are irregular: some hours are missing from the source
    (sensor offline, no record published), monthly files sometimes start or
    end mid-day, and there can be small overlaps at month boundaries.
    Reindexing onto the canonical hourly axis converts the data from
    'list of (time, value) tuples' into 'fixed-length arrays where
    position == time'. Every downstream operation (sliding windows, masking,
    normalization) assumes that property.
    """
    # Glob across all 12 monthly subdirectories. sorted() makes the
    # concat order deterministic (01, 02, ..., 12) — not strictly necessary
    # after sort_values('utc') below, but cheap insurance.
    monthly_files = sorted(csv_root.glob(f"*/{detector_id}.csv"))
    if not monthly_files:
        raise FileNotFoundError(
            f"No monthly CSVs found for {detector_id} under {csv_root}. "
            "Did extraction run for all 12 months?"
        )

    # Read each month, keeping only the columns we need.
    parts = [
        pd.read_csv(f, sep=";", usecols=USE_COLS, low_memory=False)
        for f in monthly_files
    ]
    df = pd.concat(parts, ignore_index=True)

    # Parse the UTC timestamp as a timezone-aware datetime. utc=True both
    # parses the '+00:00' suffix and ensures the dtype matches master_index.
    df["utc"] = pd.to_datetime(df["utc"], utc=True)

    # Defensive: drop duplicate timestamps if month boundaries overlapped.
    # 'keep=first' is arbitrary — these duplicates should be exact copies.
    df = df.drop_duplicates(subset="utc", keep="first")

    # Sort by time and set utc as the index, then reindex onto the master axis.
    df = df.sort_values("utc").set_index("utc")
    df = df.reindex(master_index)

    return df


# --- Sanity-check entry point (will be replaced with full pipeline in Step 7)

def _sanity_check_step1() -> None:
    """Print enough to confirm Step 1 produced reasonable axes."""
    print("--- Step 1 sanity check ---")
    time_idx = build_master_time_index()
    print(f"Time axis : {len(time_idx)} hourly stamps "
          f"({time_idx[0]}  →  {time_idx[-1]})")

    detectors = find_valid_detectors()
    print(f"Node list : {len(detectors)} detectors "
          f"(first 3: {detectors[:3]}, last 3: {detectors[-3:]})")

    assert all(DETECTOR_ID_PATTERN.match(d) for d in detectors), \
        "Some detector IDs don't match the expected TEU####_Det# pattern."


def _sanity_check_step2() -> None:
    """Probe one detector and confirm reindexing produces a length-T frame."""
    print("\n--- Step 2 sanity check ---")
    time_idx = build_master_time_index()

    det_id = "TEU00002_Det0"
    df = load_one_detector(det_id, master_index=time_idx)

    print(f"Detector       : {det_id}")
    print(f"Frame length   : {len(df)} (expected {len(time_idx)})")
    print(f"Columns        : {list(df.columns)}")

    nan_rows = df["vkfz"].isna().sum()
    print(f"Rows with NaN vkfz : {nan_rows} ({nan_rows/len(df)*100:.1f}% of year)")
    print("First valid row:")
    print(df.dropna(subset=["vkfz"]).head(1).to_string())

    # Hard checks — these would catch a regression.
    assert len(df) == len(time_idx), "Reindex did not produce the expected length."
    assert df.index.equals(time_idx), "Reindex index differs from master_index."


if __name__ == "__main__":
    _sanity_check_step1()
    _sanity_check_step2()
