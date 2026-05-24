"""
build_tensor.py
================

Convert the 454 per-detector × 12 monthly CSV files into a single
(Time, Nodes, Features) PyTorch tensor + (T, N) boolean mask + metadata, saved to
'berlin_traffic_tensor.pt'.

The information pipeline between raw CSVs and the ST-GNN.

Pipeline (built incrementally):
  Step 1: build_master_time_index() + find_valid_detectors()           [done]
  Step 2: load_one_detector()                                          [done]
  Step 3: quality mask + missing-value handling                        [done]
  Step 4: feature engineering (cyclic time, etc.)                      [done]
  Step 5: stack into (T, N, F) tensor                                  [done]
  Step 6: train-split-only z-score normalization                       [done]
  Step 7: orchestrate everything + save to a single .pt bundle         [done]

CLI
---
  python build_tensor.py            # run per-step sanity checks (10-det subset)
  python build_tensor.py --build    # full pipeline → berlin_traffic_tensor.pt

Saved bundle schema (the dict written by `save_artifacts`)
----------------------------------------------------------
  format_version      int       — schema version; bump on breaking changes.
  X                   Tensor    — (T, N, F) float32, traffic chans z-scored.
  mask                Tensor    — (T, N) bool; True = real (post-Step-3) obs.
  det_ids             list[str] — length N, sorted; row i ↔ det_ids[i].
  time_index          DatetimeIndex — length T, tz=UTC, hourly.
  feature_names       list[str] — length F=10; first 6 traffic, last 4 cyclic.
  n_traffic_features  int       — 6; convenience constant for slicing F.
  mu, sigma           Tensor    — (6,) float32 train-only stats on traffic.
  train_end, val_end  int       — chronological split right boundaries.
  config              dict      — thresholds/ratios used to build this bundle.
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


# --- Step 6: train-split-only z-score normalization -------------------------

# Chronological split. No shuffling: the time axis is the whole point and
# leakage from future hours into the training statistics would silently inflate
# eval scores. The first 70% of hours train; the next 15% validates; the last
# 15% tests.
SPLIT_RATIOS = (0.70, 0.15, 0.15)

# Numerical floor on sigma so a degenerate (constant) feature cannot trigger a
# divide-by-zero. Standard ML hygiene; the value is small enough not to perturb
# any real feature's scale.
SIGMA_EPS = 1e-6


def compute_split_boundaries(
    T: int,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
) -> tuple[int, int]:
    """
    Return `(train_end, val_end)` — half-open right boundaries on the time axis.

    A timestamp `t` belongs to:
        train if  0          <= t < train_end
        val   if  train_end  <= t < val_end
        test  if  val_end    <= t < T

    Why integers, not timestamps
    ----------------------------
    The tensor is indexed by position, not by datetime, so returning indices
    keeps the downstream code as plain slices (`X[:train_end]`). Round-trip
    to timestamps lives in the metadata dict saved at Step 7 if you need it
    for plots later.

    Why round instead of truncate
    -----------------------------
    `int(T * 0.7)` would systematically bias train smaller than asked when T
    isn't a multiple of 10. `round(...)` is fairer and the off-by-one is
    irrelevant statistically.
    """
    train_frac, val_frac, _ = ratios
    train_end = round(T * train_frac)
    val_end = round(T * (train_frac + val_frac))
    # Defensive: the three slices must be non-empty and partition [0, T).
    assert 0 < train_end < val_end < T, (
        f"Bad split boundaries for T={T}: train_end={train_end}, val_end={val_end}"
    )
    return train_end, val_end


# The 6 traffic feature channels live at indices [0, 6). The 4 cyclic channels
# live at [6, 10). We slice this explicitly so the intent is hard to misread.
TRAFFIC_FEATURE_SLICE = slice(0, len(FEATURE_COLS))                       # 0..5
CYCLIC_FEATURE_SLICE = slice(len(FEATURE_COLS), len(ALL_FEATURE_COLS))    # 6..9


def compute_normalization_stats(
    X: torch.Tensor,
    mask: torch.Tensor,
    train_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-feature mean and std for the 6 traffic channels using ONLY
    the training time slice AND ONLY positions the mask marks as real.

    Parameters
    ----------
    X : (T, N, F) float32
        Output of build_tensor.
    mask : (T, N) bool
        True where the value is an originally-observed (post-Step-3) reading.
    train_end : int
        Right boundary of the training slice (exclusive).

    Returns
    -------
    mu, sigma : 1-D tensors of length len(FEATURE_COLS) = 6.

    Why train-only stats
    --------------------
    Z-scoring uses the mean and std of a population to centre and scale data.
    If those statistics are computed over the *entire* year, the model has
    indirectly "seen" the validation and test distributions before training:
    its inputs at eval time have been pre-scaled with knowledge of what the
    future will look like. That's a textbook form of data leakage. By
    confining μ/σ to the training slice we keep the eval set honest.

    Why mask-out imputed positions
    -----------------------------
    Step 4 filled long gaps with the seasonal mean so the input is dense for
    the GCN, but those filled values are artefacts — they make the
    distribution look more concentrated than it is. Including them in μ/σ
    would underestimate σ and slightly compress the real signal after
    scaling. Computing stats over `mask=True` cells only gives an estimate
    of the real-world distribution.

    Why only the traffic channels
    -----------------------------
    The cyclic time features (sin/cos of hour-of-day, day-of-week) are
    already in [-1, 1] with zero mean by construction. Z-scoring them would
    destroy the unit-circle geometry — the wraparound midnight-to-1am
    property exists specifically because sin/cos are NOT centred and
    rescaled per observation. We leave them untouched.
    """
    # 1. Restrict to the training time slice across both X and mask.
    X_train = X[:train_end]                          # (T_train, N, F)
    mask_train = mask[:train_end]                    # (T_train, N)

    # 2. Pull out only the 6 traffic channels.
    X_train_traffic = X_train[:, :, TRAFFIC_FEATURE_SLICE]  # (T_train, N, 6)

    # 3. Boolean-indexing trick: X_train_traffic[mask_train] selects rows
    #    along the leading (T_train, N) dims wherever the mask is True,
    #    yielding a flat (num_valid, 6) tensor of real observations.
    valid_rows = X_train_traffic[mask_train]         # (num_valid, 6)

    if valid_rows.numel() == 0:
        raise RuntimeError(
            "No valid (mask=True) cells in the training slice. "
            "Cannot compute normalization stats."
        )

    # 4. Per-feature mean/std across the valid sample axis.
    #    torch.std uses Bessel's correction by default — at ~millions of
    #    samples per feature it's indistinguishable from the population std.
    mu = valid_rows.mean(dim=0)                      # (6,)
    sigma = valid_rows.std(dim=0)                    # (6,)

    # 5. Defensive floor: a degenerate constant feature would otherwise
    #    propagate inf/nan after division.
    sigma = sigma.clamp(min=SIGMA_EPS)

    return mu, sigma


def apply_normalization(
    X: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Apply (x - μ)/σ to the 6 traffic channels of X; leave the cyclic
    channels untouched. Returns a NEW tensor (the input is not mutated).

    Broadcasting recap
    ------------------
    `mu` and `sigma` are shape (6,). When we subtract them from a slice of
    shape (T, N, 6), PyTorch broadcasts the (6,) along the leading two
    dimensions, so every (t, n) position gets the same per-feature shift
    and scale. This is exactly what z-scoring wants — a feature-wise
    transformation that is identical across time and across detectors.

    Why return a new tensor
    ----------------------
    In-place ops on a tensor we still want to compare against (e.g. for
    "shape went from X to Z" debugging) are error-prone. The tensor is
    only ~160 MB; the copy is cheap compared to a training run.
    """
    out = X.clone()
    out[:, :, TRAFFIC_FEATURE_SLICE] = (
        X[:, :, TRAFFIC_FEATURE_SLICE] - mu
    ) / sigma
    # Cyclic channels at CYCLIC_FEATURE_SLICE pass through unchanged by virtue
    # of never being assigned to.
    return out


# --- Step 7: orchestrate end-to-end + save the bundle ----------------------

# Schema version stamped into the saved file. Bump this if you change the set
# of keys (or their meaning) in the bundle dict, so old .pt files can be
# detected and re-built rather than silently misread by future code.
BUNDLE_FORMAT_VERSION = 1


def build_artifacts(
    detector_ids: list[str] | None = None,
    master_index: pd.DatetimeIndex | None = None,
    csv_root: Path = CSV_ROOT,
    show_progress: bool = True,
) -> dict:
    """
    Run Steps 1-6 end-to-end and return everything the model will need.

    Parameters
    ----------
    detector_ids : list of str, optional
        Which detectors to include. Defaults to the full sorted intersection
        from `find_valid_detectors()`.
    master_index : pd.DatetimeIndex, optional
        Time axis. Defaults to the canonical 2023 hourly UTC index.

    Returns
    -------
    artifacts : dict
        A flat, self-describing bundle ready for `torch.save`. See the module
        docstring "Saved bundle schema" section for the exact keys.

    Why a single dict (vs. one file per array)
    -----------------------------------------
    The alternative is dropping X, mask, mu, sigma, ... as a folder of separate
    files. A single dict keeps everything atomically loadable in one
    `torch.load`, which prevents the canonical "I loaded X but forgot mu, so
    my predictions are off by a factor of σ" bug. It also gives the artifact
    a single hash for versioning.

    Why we save the NORMALIZED X
    ----------------------------
    The model wants normalized inputs at training time. If we saved the raw
    tensor and re-normalized on every training run we'd repeat the work AND
    risk silent drift between runs (e.g. someone changes SPLIT_RATIOS without
    realising it shifts μ/σ). We persist X_norm + (mu, sigma): de-normalising
    for human-readable predictions (MAE in vehicles/h, not σ-units) is a
    single broadcasted `x*σ+μ` in the eval loop.

    Why we still save mu/sigma even though X is pre-normalized
    ----------------------------------------------------------
    Three reasons. (1) To invert predictions back to physical units for
    metrics and plots. (2) To normalize *new* data (e.g. 2024 hours) using
    the same statistics — the whole point of train-only stats. (3) Audit:
    it makes the bundle self-describing without having to keep a separate
    config file in sync.
    """
    if master_index is None:
        master_index = build_master_time_index()
    if detector_ids is None:
        detector_ids = find_valid_detectors(csv_root=csv_root)

    # Steps 2-5 wrapped: per-detector load + clean + engineer, stacked.
    X_raw, mask = build_tensor(
        detector_ids, master_index, csv_root=csv_root, show_progress=show_progress
    )

    # Step 6a: chronological split boundaries on the time axis.
    train_end, val_end = compute_split_boundaries(X_raw.shape[0])

    # Step 6b: train-only μ/σ over real (mask=True) traffic-channel cells.
    mu, sigma = compute_normalization_stats(X_raw, mask, train_end)

    # Step 6c: z-score the traffic channels; cyclic channels pass through.
    X = apply_normalization(X_raw, mu, sigma)

    # Bundle. Keep keys simple and snake_case; downstream code will rely on
    # these names. Bump BUNDLE_FORMAT_VERSION if you ever change them.
    return {
        "format_version": BUNDLE_FORMAT_VERSION,
        # Core arrays.
        "X": X,                                    # (T, N, F) float32, normalized
        "mask": mask,                              # (T, N) bool
        # Identity / axes — so row indices can be mapped back to meaning.
        "det_ids": list(detector_ids),             # length N, sorted, stable order
        "time_index": master_index,                # length T, tz=UTC, hourly
        "feature_names": list(ALL_FEATURE_COLS),   # length F=10
        "n_traffic_features": len(FEATURE_COLS),   # 6; first n_traffic are z-scored
        # Normalization stats (apply (x*σ)+μ to invert; traffic channels only).
        "mu": mu,                                  # (6,) float32
        "sigma": sigma,                            # (6,) float32
        # Splits — half-open right boundaries, so the three slices partition
        # [0, T):   train=[0, train_end), val=[train_end, val_end), test=[val_end, T).
        "train_end": int(train_end),
        "val_end": int(val_end),
        # Reproducibility metadata. None of this is required to train; all of
        # it is useful when you come back in 3 weeks and ask "wait, which
        # quality threshold did I use for this tensor?".
        "config": {
            "year": YEAR,
            "quality_threshold": VOLLSTAENDIGKEIT_THRESHOLD,
            "short_gap_limit": SHORT_GAP_LIMIT,
            "split_ratios": SPLIT_RATIOS,
            "sigma_eps": SIGMA_EPS,
            "berlin_tz": BERLIN_TZ,
            "csv_root": str(csv_root),
            "geojson_path": str(GEOJSON_PATH),
        },
    }


def save_artifacts(artifacts: dict, path: Path = OUTPUT_PATH) -> None:
    """
    Persist the bundle to disk via `torch.save` and print a one-line summary.

    Why torch.save (and not numpy.savez / parquet / hdf5)
    -----------------------------------------------------
    - `torch.save` is the de-facto standard for PyTorch artifacts and
      preserves tensor dtype and shape exactly. The training loop will just
      `torch.load(path)` and have everything as tensors already — no
      `torch.from_numpy()` ceremony at the start of every run.
    - It uses pickle under the hood, so the non-tensor fields (DatetimeIndex,
      list[str], config dict) round-trip without manual (de)serialization.
    - One file = one atomic artifact. Easier to checksum, move, version.

    The trade-off is that pickle-based files are Python-specific and have a
    well-known security caveat (never `torch.load` untrusted .pt files). For
    a learning/portfolio project that's the right call; production pipelines
    that need cross-language interop would reach for parquet/HDF5 instead.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifacts, path)

    size_mb = path.stat().st_size / 1e6
    X = artifacts["X"]
    print(
        f"\n[save] Wrote {path.name} ({size_mb:.1f} MB on disk)\n"
        f"       X shape : {tuple(X.shape)} dtype={X.dtype}\n"
        f"       Splits  : train=[0, {artifacts['train_end']}), "
        f"val=[{artifacts['train_end']}, {artifacts['val_end']}), "
        f"test=[{artifacts['val_end']}, {X.shape[0]})\n"
        f"       Path    : {path}"
    )


def load_artifacts(path: Path = OUTPUT_PATH) -> dict:
    """
    Convenience loader. Mirrors `save_artifacts`; returns the same dict shape.

    Note on `weights_only=False`
    ---------------------------
    Newer PyTorch versions default `torch.load` to a "weights-only" safe
    loader that refuses any non-tensor Python object inside the pickle. Our
    bundle deliberately contains a DatetimeIndex, a config dict, and a list
    of detector IDs, so we disable that guard. The contract is: only ever
    call `load_artifacts` on files this script produced.
    """
    path = Path(path)
    return torch.load(path, weights_only=False)


# --- Per-step sanity checks -------------------------------------------------

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


def _sanity_check_step6() -> None:
    """Z-score: train slice μ≈0, σ≈1 on traffic channels; cyclic untouched."""
    print("\n--- Step 6 sanity check ---")
    time_idx = build_master_time_index()
    subset = find_valid_detectors()[:10]
    X, mask = build_tensor(subset, time_idx, show_progress=False)
    T, N, F = X.shape

    # 1. Boundaries on the time axis.
    train_end, val_end = compute_split_boundaries(T)
    print(f"T = {T:>4d} hours total")
    print(f"  train : [    0, {train_end:>4d})   "
          f"({train_end} h, {train_end/T*100:.1f}%)")
    print(f"  val   : [{train_end:>4d}, {val_end:>4d})   "
          f"({val_end-train_end} h, {(val_end-train_end)/T*100:.1f}%)")
    print(f"  test  : [{val_end:>4d}, {T:>4d})   "
          f"({T-val_end} h, {(T-val_end)/T*100:.1f}%)")

    # 2. Compute and inspect stats.
    mu, sigma = compute_normalization_stats(X, mask, train_end)
    print(f"\nμ per traffic feature (computed on train slice + mask only):")
    for name, m, s in zip(FEATURE_COLS, mu.tolist(), sigma.tolist()):
        print(f"  {name:<6s}  mu={m:>9.3f}   sigma={s:>9.3f}")

    # 3. Apply and verify.
    X_norm = apply_normalization(X, mu, sigma)

    # 3a. Cyclic channels must be byte-for-byte identical.
    cyclic_unchanged = torch.equal(
        X[:, :, CYCLIC_FEATURE_SLICE],
        X_norm[:, :, CYCLIC_FEATURE_SLICE],
    )
    print(f"\nCyclic channels untouched : {cyclic_unchanged}")

    # 3b. On the training slice + mask, normalized traffic should have
    #     ~zero mean and ~unit std per feature. (Validation/test slices
    #     intentionally won't — that's the whole point of train-only stats:
    #     drift between train and eval is preserved, not laundered out.)
    Xn_train = X_norm[:train_end][:, :, TRAFFIC_FEATURE_SLICE]
    valid_train = Xn_train[mask[:train_end]]   # (num_valid, 6)
    mu_check = valid_train.mean(dim=0)
    std_check = valid_train.std(dim=0)
    print(f"\nPost-normalization stats on train slice (should be ~0, ~1):")
    for name, m, s in zip(FEATURE_COLS, mu_check.tolist(), std_check.tolist()):
        print(f"  {name:<6s}  mu={m:>9.3e}   sigma={s:>9.4f}")

    # 3c. Show what happens on the val and test slices — drift is expected
    #     and informative; do NOT panic if mu/sigma aren't exactly 0/1 here.
    for label, lo, hi in [("val ", train_end, val_end), ("test", val_end, T)]:
        Xn_slice = X_norm[lo:hi][:, :, TRAFFIC_FEATURE_SLICE]
        m_slice = mask[lo:hi]
        valid = Xn_slice[m_slice]
        if valid.numel() == 0:
            print(f"\n[{label}] no valid samples in slice — skipping")
            continue
        print(f"\nPost-normalization stats on {label} slice "
              f"(expect mild drift from 0/1):")
        mu_d = valid.mean(dim=0).tolist()
        sd_d = valid.std(dim=0).tolist()
        for name, m, s in zip(FEATURE_COLS, mu_d, sd_d):
            print(f"  {name:<6s}  mu={m:>9.3f}   sigma={s:>9.4f}")

    # Hard checks.
    assert cyclic_unchanged, "Cyclic features were modified — they shouldn't be."
    assert torch.allclose(mu_check, torch.zeros_like(mu_check), atol=1e-4), \
        f"Train-slice mean is not ~0: {mu_check}"
    assert torch.allclose(std_check, torch.ones_like(std_check), atol=1e-3), \
        f"Train-slice std is not ~1: {std_check}"
    assert X_norm.shape == X.shape and X_norm.dtype == X.dtype


def _sanity_check_step7(tmp_path: Path | None = None) -> None:
    """
    End-to-end on a 10-detector subset: build → save → reload → verify equality.

    Why a round-trip (and not just "did the file get written?")
    ----------------------------------------------------------
    The interesting failure mode in a save step isn't "no file appeared";
    it's "the file appeared but a key is missing / a tensor's dtype was
    silently downcast / a non-tensor field didn't survive pickle". The only
    way to catch those is to load it back and `torch.equal` / `==` against
    the in-memory original. So that's exactly what this does.
    """
    print("\n--- Step 7 sanity check ---")
    if tmp_path is None:
        # Sub-set bundle lands next to the real one so it's easy to spot/delete.
        tmp_path = PROJECT_ROOT / "berlin_traffic_tensor_subset.pt"

    time_idx = build_master_time_index()
    subset = find_valid_detectors()[:10]

    artifacts = build_artifacts(
        detector_ids=subset, master_index=time_idx, show_progress=False
    )
    save_artifacts(artifacts, tmp_path)

    # Reload and verify.
    reloaded = load_artifacts(tmp_path)

    # 1. Schema: every documented key is present.
    expected_keys = {
        "format_version", "X", "mask", "det_ids", "time_index",
        "feature_names", "n_traffic_features", "mu", "sigma",
        "train_end", "val_end", "config",
    }
    missing = expected_keys - reloaded.keys()
    assert not missing, f"Reloaded bundle is missing keys: {missing}"

    # 2. Tensor round-trip: bit-exact (torch.save preserves dtype + values).
    assert torch.equal(artifacts["X"], reloaded["X"]), "X mismatch after round-trip"
    assert torch.equal(artifacts["mask"], reloaded["mask"]), "mask mismatch"
    assert torch.equal(artifacts["mu"], reloaded["mu"]), "mu mismatch"
    assert torch.equal(artifacts["sigma"], reloaded["sigma"]), "sigma mismatch"

    # 3. Non-tensor round-trip: pickle round-trips these by value.
    assert artifacts["det_ids"] == reloaded["det_ids"]
    assert artifacts["time_index"].equals(reloaded["time_index"])
    assert artifacts["feature_names"] == reloaded["feature_names"]
    assert artifacts["n_traffic_features"] == reloaded["n_traffic_features"]
    assert artifacts["train_end"] == reloaded["train_end"]
    assert artifacts["val_end"] == reloaded["val_end"]
    assert artifacts["config"] == reloaded["config"]
    assert artifacts["format_version"] == reloaded["format_version"]

    print(f"\nRound-trip OK. Bundle keys: {sorted(reloaded.keys())}")
    print(f"format_version    : {reloaded['format_version']}")
    print(f"X.shape           : {tuple(reloaded['X'].shape)}")
    print(f"det_ids[:3]       : {reloaded['det_ids'][:3]}")
    print(f"time_index range  : {reloaded['time_index'][0]} → {reloaded['time_index'][-1]}")
    print(f"config            : {reloaded['config']}")

    # Tidy up the subset artifact so it doesn't pollute the project root.
    tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Build the (T, N, F) PyTorch tensor + mask + metadata from the "
            "Berlin VMZ CSVs. Default mode runs per-step sanity checks on a "
            "small subset; pass --build for the real thing."
        )
    )
    parser.add_argument(
        "--build", action="store_true",
        help=(
            "Build the FULL tensor over all valid detectors and save it to "
            f"{OUTPUT_PATH.name} (~2 min on a laptop, ~160 MB on disk). "
            "Without this flag, only the per-step sanity checks run."
        ),
    )
    args = parser.parse_args()

    if args.build:
        print("[run] Building full tensor over all valid detectors...")
        artifacts = build_artifacts()
        save_artifacts(artifacts, OUTPUT_PATH)
        print("\n[run] Done. Load it elsewhere via:")
        print(f"        from build_tensor import load_artifacts")
        print(f"        bundle = load_artifacts()  # uses OUTPUT_PATH by default")
    else:
        _sanity_check_step1()
        _sanity_check_step2()
        _sanity_check_step3()
        _sanity_check_step4()
        _sanity_check_step5()
        _sanity_check_step6()
        _sanity_check_step7()
