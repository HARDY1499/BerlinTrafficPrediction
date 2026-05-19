"""
build_tensor.py
================

Convert the 454 per-detector × 12 monthly CSV files into a single
(Time, Nodes, Features) PyTorch tensor + (T, N) boolean mask + metadata, saved to
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

import numpy as np
import pandas as pd
import torch


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

    # 2. Concatenate, parse utc, normalise timezone, set as index.
    df = pd.concat(parts, ignore_index=True) # ignoring original row numbers, we reindex later anyway
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)  # forces timezone-aware UTC

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


# --- Step 3: quality mask + short-gap handling ------------------------------

VOLLSTAENDIGKEIT_THRESHOLD = 90.0  # %; rows below this are treated as missing
SHORT_GAP_LIMIT = 3                # hours; forward-fill up to this many in a row (hyperparameter to tune later)


def clean_one_detector(
    df: pd.DataFrame,
    quality_threshold: float = VOLLSTAENDIGKEIT_THRESHOLD,
    short_gap_limit: int = SHORT_GAP_LIMIT,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply quality filtering, short-gap forward-fill, and produce a validity mask.

    Parameters
    ----------
    df : DataFrame
        Output of load_one_detector — columns [Vollständigkeit, *FEATURE_COLS],
        indexed on the master hourly UTC axis.
    quality_threshold : float
        Vollständigkeit cutoff (%). Rows below this become NaN in the features.
    short_gap_limit : int
        Forward-fill at most this many consecutive NaN hours per column.

    Returns
    -------
    features : DataFrame of shape (T, F)
        Cleaned feature matrix on the same index. The Vollständigkeit column
        is dropped — it served its purpose as a filter and the model should
        not see it.
    mask : Series of bool, shape (T,)
        True where every feature column is non-NaN after cleaning.

    Why mask instead of drop
    -----------------------
    Deletion would break the time axis: sliding windows assume "row i+1 is
    one hour after row i". The mask tells downstream loss/eval code to ignore
    bad positions without disturbing the index.

    Why forward-fill ONLY short gaps
    -------------------------------
    A 1–3 hour gap is usually transient (brief outage, maintenance). Carrying
    the previous valid value forward is a defensible estimate. A 24-hour gap
    is qualitatively different — the sensor was offline, rush hours came and
    went, and filling yesterday's late-night speed into this morning's commute
    is actively wrong. Past short_gap_limit, NaN + mask is more honest.
    """
    cleaned = df.copy()

    # 1. Quality filter: low-completeness rows become NaN in the features.
    #    Note: cleaned[QUALITY_COL] itself is left intact — we just don't
    #    return it (the model should never see the quality flag).
    low_quality = cleaned[QUALITY_COL] < quality_threshold
    cleaned.loc[low_quality, FEATURE_COLS] = pd.NA

    # 2. Forward-fill ONLY short gaps. ffill with `limit=N` fills at most
    #    N consecutive NaN values per column, leaving longer gaps as NaN.
    features = cleaned[FEATURE_COLS].ffill(limit=short_gap_limit)

    # 3. Mask: True where every feature column has a value after cleaning.
    #    .all(axis=1) gives row-wise AND across all feature columns.
    mask = features.notna().all(axis=1)

    return features, mask


# --- Step 4: feature engineering --------------------------------------------

CYCLIC_FEATURE_COLS = ["sin_hod", "cos_hod", "sin_dow", "cos_dow"]
BERLIN_TZ = "Europe/Berlin"


def add_cyclic_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build a (T, 4) DataFrame of cyclic time encodings: sin/cos of
    hour-of-day and sin/cos of day-of-week, computed in Berlin local time.

    Why cyclic encoding
    -------------------
    Time is circular but raw integers are linear. With hour-as-int, hour 23
    and hour 0 are 23 units apart even though they're one hour apart
    physically. The model would have to learn this wraparound from data,
    which is wasted capacity. Sin/cos projects the hour onto a unit circle:
    adjacent hours are nearby in (sin, cos) coordinates and the wraparound
    is automatic. Both sin and cos are included for each cycle, providing
    a complete representation as only including one would mean losing information 
    about the phase (0 to 2π; whether it's 12 or 24). 
    Two features per cycle instead of 24 one-hots, with
    continuity built in. We do this for both hour-of-day and day-of-week 
    to capture daily and weekly patterns.

    Why Berlin local time
    --------------------
    Rush hour happens at 8 a.m. Berlin time, not 8 a.m. UTC. Converting to
    local time aligns the daily cycle with human activity and makes future
    attention plots interpretable. UTC would also work, just shifted.
    """
    local = index.tz_convert(BERLIN_TZ)
    hod = local.hour          # 0..23
    dow = local.dayofweek     # 0..6  (Mon=0, Sun=6)

    return pd.DataFrame(
        {
            "sin_hod": np.sin(2 * np.pi * hod / 24),
            "cos_hod": np.cos(2 * np.pi * hod / 24),
            "sin_dow": np.sin(2 * np.pi * dow / 7),
            "cos_dow": np.cos(2 * np.pi * dow / 7),
        },
        index=index,
    )


def impute_seasonal_mean(features: pd.DataFrame) -> pd.DataFrame:
    """
    Fill remaining NaNs in each column with the same-hour-of-week mean
    for that detector. Operates per column independently.

    Why hour-of-week, not hour-of-day
    --------------------------------
    Weekday vs weekend traffic patterns differ substantially. The mean
    for "Wed 15:00" is a much better estimate than "15:00 of any day".
    168 buckets per column (7 days * 24 hours) stay statistically stable
    with a year of data.

    Why we still keep the mask
    -------------------------
    Imputation makes the input dense (the GCN can compute predictions at
    every position) but the imputed value is not a real observation. The
    mask from Step 3 continues to flag "this was originally missing", so
    the training loss can ignore these positions. Result: the model
    operates on dense inputs but only learns from real targets — the
    standard ST-GNN convention.
    """
    out = features.copy()
    # Hour-of-week bucket: 0..167. We use Berlin local time for consistency
    # with the cyclic features above.
    local = out.index.tz_convert(BERLIN_TZ)
    how = local.dayofweek * 24 + local.hour  # monday 00:00 → 0, Sunday 23:00 → 167
    seasonal = out.groupby(how).transform("mean")

    # Fallback chain: seasonal mean -> column mean -> 0.
    # - seasonal handles "this hour-of-week has data on other weeks"
    # - column mean handles "this entire hour-of-week bucket is empty"
    # - 0 handles "this column is empty for the whole year" (mask kills it anyway)
    out = out.fillna(seasonal)
    out = out.fillna(out.mean())
    out = out.fillna(0.0)
    return out


def engineer_one_detector(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the whole of Step 4 in one call: impute residual NaNs, then concatenate
    cyclic time features. Returns a dense (T, F_in) DataFrame, where
    F_in = len(FEATURE_COLS) + len(CYCLIC_FEATURE_COLS) = 6 + 4 = 10.
    """
    imputed = impute_seasonal_mean(features)
    cyclic = add_cyclic_time_features(features.index)
    return pd.concat([imputed, cyclic], axis=1)


# --- Step 5: stack every detector into (T, N, F) tensors --------------------

ALL_FEATURE_COLS = FEATURE_COLS + CYCLIC_FEATURE_COLS  # the F dimension


def build_tensor(
    detector_ids: list[str],
    master_index: pd.DatetimeIndex,
    csv_root: Path = CSV_ROOT,
    show_progress: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run Steps 2-4 over every detector and stack the results.

    Returns
    -------
    X : torch.Tensor, shape (T, N, F), dtype float32
        Dense feature tensor. F = len(ALL_FEATURE_COLS) = 10.
    mask : torch.Tensor, shape (T, N), dtype bool
        True where the row is an originally-observed (post-Step-3) value
        for that detector. The training loss should multiply by this.

    Notes
    -----
    1. Pre-allocates the output arrays. Growing Python lists and calling
       np.stack at the end works but copies memory twice; for a 160 MB
       tensor that matters. Pre-allocating with a NaN sentinel also lets
       us assert at the end that every cell was filled — a cheap guard
       against silent indexing bugs.

    2. We use float32 (not float64). 4 bytes per cell instead of 8 halves
       the memory footprint and matches what PyTorch wants on the GPU.
       Speeds and flows fit fine in float32 precision.

    3. If a single detector fails (file missing, parse error, ...) we
       log it and continue — losing one column is better than aborting
       the whole 2-minute run. The mask for that detector stays all-False
       so downstream code ignores it.
    """
    T = len(master_index)
    N = len(detector_ids)
    F = len(ALL_FEATURE_COLS)

    # Pre-allocate. NaN sentinel for X so a fill bug becomes a loud failure.
    X = np.full((T, N, F), np.nan, dtype=np.float32)
    mask = np.zeros((T, N), dtype=bool)

    iterator = enumerate(detector_ids)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(
                iterator, total=N, desc="Building tensor", unit="det",
            )
        except ImportError:
            pass  # progress bar is nice-to-have, not essential

    failures: list[tuple[str, str]] = []
    for i, det_id in iterator:
        try:
            raw = load_one_detector(det_id, master_index=master_index,
                                    csv_root=csv_root)
            features, m = clean_one_detector(raw)
            engineered = engineer_one_detector(features)

            # Defensive: column order must match ALL_FEATURE_COLS exactly.
            X[:, i, :] = engineered[ALL_FEATURE_COLS].to_numpy(dtype=np.float32)
            mask[:, i] = m.to_numpy(dtype=bool)
        except Exception as e:
            failures.append((det_id, str(e)))
            # Fill this column with zeros so downstream code can still index it;
            # the mask stays all-False so the loss ignores it.
            X[:, i, :] = 0.0

    if failures:
        print(f"\n[warn] {len(failures)} detector(s) failed and were zeroed out:")
        for det_id, err in failures[:5]:
            print(f"  {det_id}: {err}")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more")

    # Hard check: pre-allocation sentinel should be fully overwritten.
    assert not np.isnan(X).any(), (
        f"X has {np.isnan(X).sum()} residual NaN cells. "
        "Some detector's engineering step didn't produce a dense frame."
    )

    return torch.from_numpy(X), torch.from_numpy(mask)


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


def _sanity_check_step3() -> None:
    """Cleaning + mask on a single detector. Shows quality recovery."""
    print("\n--- Step 3 sanity check ---")
    time_idx = build_master_time_index()
    det_id = "TEU00002_Det0"
    raw = load_one_detector(det_id, master_index=time_idx)
    features, mask = clean_one_detector(raw)

    T = len(features)
    # Rows that were valid BEFORE cleaning (raw V>=threshold and not NaN).
    raw_valid = (
        raw[QUALITY_COL].notna()
        & (raw[QUALITY_COL] >= VOLLSTAENDIGKEIT_THRESHOLD)
    )

    print(f"Detector            : {det_id}")
    print(f"Total rows          : {T}")
    print(f"Raw V >= {VOLLSTAENDIGKEIT_THRESHOLD:>2.0f}        : "
          f"{raw_valid.sum()} ({raw_valid.mean()*100:.1f}%)")
    print(f"Mask-valid rows     : {mask.sum()} ({mask.mean()*100:.1f}%)")
    print(f"Recovered by ffill  : {mask.sum() - raw_valid.sum()} hours")
    print(f"Feature columns     : {list(features.columns)}")
    print("NaN counts per feature (post-clean):")
    print(features.isna().sum().to_string())

    assert len(features) == T and len(mask) == T
    assert features.notna().all(axis=1).equals(mask)


def _sanity_check_step4() -> None:
    """Engineering: cyclic features + imputation. Check dense output + mask."""
    print("\n--- Step 4 sanity check ---")
    time_idx = build_master_time_index()
    det_id = "TEU00002_Det0"
    raw = load_one_detector(det_id, master_index=time_idx)
    features, mask = clean_one_detector(raw)

    pre_nan = features.isna().sum().sum()
    engineered = engineer_one_detector(features)
    post_nan = engineered.isna().sum().sum()

    print(f"Detector            : {det_id}")
    print(f"Shape after Step 3  : {features.shape}  (NaN cells: {pre_nan})")
    print(f"Shape after Step 4  : {engineered.shape}  (NaN cells: {post_nan})")
    print(f"Mask coverage       : {mask.sum()}/{len(mask)} "
          f"({mask.mean()*100:.1f}%) — UNCHANGED from Step 3 (correct)")
    print(f"Feature columns     : {list(engineered.columns)}")

    # Verify cyclic features at known timestamps. Use Berlin local time.
    sample = engineered.loc[[
        engineered.index[0],   # 2023-01-01 00:00 UTC = 01:00 Berlin (winter)
        engineered.index[24*7] # one week later, same Berlin clock time
    ]]
    print("\nCyclic features at two reference timestamps:")
    print(sample[CYCLIC_FEATURE_COLS].to_string())

    assert post_nan == 0, "Step 4 should leave a dense matrix; residual NaNs!"
    assert engineered.shape[1] == len(FEATURE_COLS) + len(CYCLIC_FEATURE_COLS)
    assert engineered.index.equals(features.index)


def _sanity_check_step5() -> None:
    """Build the full tensor on a small subset to verify shape, dtype, mask."""
    print("\n--- Step 5 sanity check ---")
    time_idx = build_master_time_index()
    all_dets = find_valid_detectors()
    subset = all_dets[:10]  # keep this cheap; full build is Step 6 onward

    X, mask = build_tensor(subset, time_idx, show_progress=False)
    T, N, F = X.shape

    print(f"Detectors used      : {N} (out of {len(all_dets)})")
    print(f"Tensor shape (X)    : {tuple(X.shape)}  dtype={X.dtype}")
    print(f"Mask shape          : {tuple(mask.shape)}  dtype={mask.dtype}")
    print(f"Memory footprint X  : {X.element_size() * X.numel() / 1e6:.1f} MB")
    print(f"Memory footprint M  : {mask.element_size() * mask.numel() / 1e6:.1f} MB")
    print(f"Mask coverage       : {mask.float().mean()*100:.1f}% "
          f"(real observations across all {N} detectors)")
    print(f"Per-detector valid% : "
          f"{[round(mask[:, i].float().mean().item()*100, 1) for i in range(N)]}")
    print(f"X[0, 0, :] (det 0 at t=0):")
    for name, v in zip(ALL_FEATURE_COLS, X[0, 0, :].tolist()):
        print(f"  {name:<10s}: {v:>10.4f}")

    assert X.shape == (len(time_idx), len(subset), len(ALL_FEATURE_COLS))
    assert mask.shape == (len(time_idx), len(subset))
    assert X.dtype == torch.float32
    assert mask.dtype == torch.bool
    assert not torch.isnan(X).any()


if __name__ == "__main__":
    _sanity_check_step1()
    _sanity_check_step2()
    _sanity_check_step3()
    _sanity_check_step4()
    _sanity_check_step5()
