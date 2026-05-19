# ST-GNN Berlin Traffic Prediction — Roadmap to Thesis-Quality Results

**Last updated:** 2026-05-16 (resumed after 6 months idle)
**Target model:** A3T-GCN
**Prediction horizons:** 3 h, 6 h, 12 h (multi-step)
**Data:** Berlin VMZ Fahrstreifendetektoren, 2023

---

## 0. Status snapshot

| Step | Status | Artifact |
|---|---|---|
| Download monthly `.tgz` archives | done | `berlin_traffic_data/2023/Fahrstreifendetektoren_tgz/` (12 files) |
| Extract CSVs | **done, but BROKEN — see §1.1** | `berlin_traffic_data/2023/CSV_data/` (485 files, **only Sep 2023**) |
| Explore CSV schema | done | `explore_traffic_csv.py` |
| Explore sensor locations | done | `explore_sensor_locations.py` |
| Build graph topology | done | `berlin_traffic_graph.pt` (208 nodes, 2 km binary adjacency, self-loops) |
| **Build [T,N,F] tensor** | not started | — |
| Train/val/test split | not started | — |
| A3T-GCN model | not started | — |
| Training loop | not started | — |
| Baselines | not started | — |
| Evaluation + ablations | not started | — |
| Visualization | not started | — |
| Write-up | not started | — |

---

## 1. Critical issues to fix before anything else

### 1.1. **Extraction overwrote 11 of 12 months**
`extract_traffic_archives.py` extracts all 12 monthly archives into the same flat directory. Each archive contains identically-named files (e.g. `TEU00002_Det0.csv`), so each subsequent `tar.extractall()` overwrites the previous month. Verified by sampling extracted CSVs: every file covers exactly `2023-08-31 22:00 UTC → 2023-09-30 21:00 UTC` (~720 rows = one month).

**Fix:** extract each archive into a per-month subdirectory, then merge into one long timeseries per sensor-detector during preprocessing. Add a few lines like:

```python
month_str = os.path.basename(archive_file).split("_")[-1].replace(".tgz","")  # "01" .. "12"
target = os.path.join(extract_path, month_str)
os.makedirs(target, exist_ok=True)
tar.extractall(path=target)
```

Without this fix the entire downstream pipeline is on ~720 rows × 466 nodes, which is too small for a credible ST-GNN result.

### 1.2. **Detector-level vs sensor-level nodes — pick one**
- CSV files are at **detector level**: `TEU####_Det#` (485 files, avg 2.24 detectors per sensor).
- `berlin_traffic_graph.pt` is at **sensor level**: 208 `teuID` nodes.
- These need to be reconciled. Two options:

| Option | Pros | Cons |
|---|---|---|
| **A. Aggregate detectors → sensor-level** (208 nodes) | Matches existing graph; smaller; standard for ST-GNN literature | Loses lane/direction information; aggregation must be principled (sum for `q*`, q-weighted mean for `v*`) |
| **B. Rebuild graph at detector level** (~466 nodes) | Captures lane/direction asymmetries; richer | Need a meaningful detector-to-detector adjacency (same lat/lon for siblings → distance heuristic alone collapses) |

**Recommendation:** Option A for the baseline experiment; treat Option B as an ablation in §9.

### 1.3. **Self-loops are double-counted**
`build_traffic_graph.py` sets `adj[d < 2000m] = 1` (which already includes the diagonal since `d_ii = 0`) and then sets the diagonal to 1 again — harmless, but worth noting. More important: the current adjacency is **unweighted binary**. Gaussian-kernel weighting is standard for traffic ST-GNNs and is a near-zero-cost upgrade (§4).

---

## 2. Data preprocessing pipeline

**Output artifact:** `berlin_traffic_tensor.pt` containing
- `X`: float32, shape `[T, N, F_in]` — driver features
- `Y`: float32, shape `[T, N, F_out]` — targets (typically just `vkfz`)
- `mask`: bool, shape `[T, N]` — `True` where Vollständigkeit ≥ 90 and value not missing
- `timestamps`: pandas `DatetimeIndex`, length `T`
- `sensor_ids`: list of 208 `teuID` (must match the order in `berlin_traffic_graph.pt`)
- `feature_names`: list of length `F_in`

### 2.1. Time index
After §1.1 fix, the full year gives `T = 8760` hourly steps. Use `utc` (not `localTime`) to avoid CEST/CET transitions. Build a master `pd.date_range("2023-01-01", "2024-01-01", freq="1H", tz="UTC")` and reindex every sensor's series onto it.

### 2.2. Quality filtering
- Drop any row with `Vollständigkeit < 90` (mark as missing, not deleted — the temporal index must stay regular).
- Drop any sensor whose surviving-row fraction over the year is below 70 % (these sensors are too sparse to learn from).

### 2.3. Detector → sensor aggregation (per timestep)
For each `teuID` at each timestamp:
- `qkfz_sensor = sum(qkfz_detectors)` (analogously for `qpkw`, `qlkw`)
- `vkfz_sensor = sum(qkfz_d * vkfz_d) / sum(qkfz_d)` — flow-weighted mean (fall back to simple mean when total flow is zero)
- Same flow-weighting for `vpkw`, `vlkw`

### 2.4. Missing value handling
- Forward-fill within sensor for gaps ≤ 3 hours.
- For longer gaps, leave as `NaN`, fill with the sensor's seasonal mean (same hour-of-week), and rely on `mask` during loss computation.

### 2.5. Feature engineering (`F_in ≈ 11`)
| Feature | Source | Notes |
|---|---|---|
| `qkfz`, `qpkw`, `qlkw` | aggregated | per-sensor z-score normalised |
| `vkfz`, `vpkw`, `vlkw` | aggregated | per-sensor z-score normalised |
| `sin(2π h/24)`, `cos(2π h/24)` | derived | hour-of-day cyclic encoding |
| `sin(2π d/7)`, `cos(2π d/7)` | derived | day-of-week cyclic encoding |
| `is_holiday` | external (Berlin school + public holidays) | binary; will need a small CSV |

### 2.6. Target
For the headline experiment, predict `vkfz` (all-vehicle speed). Reserve `qkfz` prediction as a follow-up.

### 2.7. Normalisation
Compute z-score statistics on the **training split only** (§3) and apply identically to val/test. Save `mean.npy` and `std.npy` alongside the tensor.

---

## 3. Splits and sequence framing

### 3.1. Chronological splits (no shuffling — temporal leakage is fatal)
- **Train:** Jan 1 – Aug 31 (8 months, ~70 %)
- **Validation:** Sep 1 – Oct 15 (~13 %)
- **Test:** Oct 16 – Dec 31 (~17 %)

### 3.2. Windowing
- **Input window** `L = 12` hours
- **Output window** `H = 12` hours, evaluated at `[+3, +6, +12]`
- Stride = 1, so each split yields ~`split_length - 24` samples.

### 3.3. Loader
A sliding-window `Dataset` returning `(X_in: [L,N,F_in], Y_out: [H,N,1], mask_out: [H,N])`.

---

## 4. Graph design

The current `berlin_traffic_graph.pt` is a fine starting point. For the headline run, **swap binary 2 km adjacency for a thresholded Gaussian kernel**, which is the standard in STGCN/DCRNN papers:

$$W_{ij} = \exp\!\left(-\frac{d_{ij}^2}{\sigma^2}\right) \cdot \mathbb{1}[d_{ij} \le \kappa]$$

with `σ = std(d)` (over all pairs) and `κ = 2000 m`. This is one extra function in `build_traffic_graph.py`. Keep the binary version as ablation §9.

For A3T-GCN, the convention is to row-normalise the symmetric adjacency: `D^{-1/2} A D^{-1/2}` after adding self-loops.

---

## 5. Model: A3T-GCN

### 5.1. Why A3T-GCN
- GCN spatial layer + GRU temporal layer + attention over GRU hidden states.
- Strong, well-cited baseline for hourly traffic forecasting.
- Implemented in **PyTorch Geometric Temporal** (`torch_geometric_temporal.nn.recurrent.A3TGCN`), avoiding a from-scratch implementation while still allowing modifications.

### 5.2. Architecture (proposed)
- **Input:** `X_in` of shape `[B, L, N, F_in]` reshaped to PyG-Temporal convention.
- **Encoder:** `A3TGCN(in_channels=F_in, out_channels=64, periods=L)`.
- **Head:** Two-layer MLP per node mapping `64 → H` (multi-step regression with linear output).
- **Output:** `[B, H, N, 1]` (predicting `vkfz`).

Why MLP head instead of decoder-style GRU: simpler, fewer params, and recent ST-GNN ablations show it matches encoder-decoder for short horizons.

### 5.3. Loss
Masked MAE (a.k.a. L1) — robust to outliers, standard in METR-LA / PEMS-BAY:

```python
loss = (mask * (pred - target).abs()).sum() / mask.sum().clamp(min=1)
```

### 5.4. Hyperparameter starting point
| Param | Value |
|---|---|
| Hidden dim | 64 |
| Dropout | 0.1 |
| Batch size | 32 |
| Optimizer | Adam |
| LR | 1e-3 |
| LR schedule | ReduceLROnPlateau on val MAE, patience 5 |
| Weight decay | 1e-5 |
| Gradient clip | 5.0 |
| Max epochs | 100 (early-stop, patience 15) |

### 5.5. Compute footprint sanity check
- `T_train ≈ 5832`, `B = 32` → ~180 batches/epoch.
- `N = 208`, `L = 12`, `F_in = 11` → input ~280k floats per batch. Trivially fits a CPU; trains fast on a single GPU. A laptop run is feasible.

---

## 6. Training protocol

- Seed `torch`, `numpy`, `random` from a single config seed.
- Save: best-val checkpoint, final checkpoint, train/val curves, full config (YAML), git commit hash.
- Log: per-epoch train loss, val MAE/RMSE/MAPE at each horizon.
- Use **Hydra** or a simple `argparse` + YAML for config — required so every result is reproducible by re-running with one command.

---

## 7. Baselines (essential for thesis credibility)

A respectable ST-GNN thesis presents **three tiers** of baselines:

| Tier | Model | Captures | Purpose |
|---|---|---|---|
| Naive | Last value (`y_{t+h} = y_t`) | nothing | floor |
| Naive | Historical Average (same hour-of-week) | seasonality | sanity check |
| Statistical | ARIMA per node | per-node temporal structure | non-graph temporal floor |
| Deep, no graph | Per-node LSTM | per-node nonlinearity | "do we even need the graph?" |
| Deep, graph | T-GCN (no attention) | spatial + temporal | "does attention help?" |
| **Headline** | **A3T-GCN** | full | — |

The "Last value" and "Historical Average" baselines often beat poorly-tuned deep models — having them in the table is what makes the headline result believable.

---

## 8. Evaluation

### 8.1. Metrics (per horizon, on test split)
- **MAE** (km/h) — primary
- **RMSE** (km/h) — penalises large misses
- **MAPE** (%) — interpretable, but mask out targets < 5 km/h to avoid divide-by-small

Report as a table:

| Model | h=3 MAE | h=6 MAE | h=12 MAE | h=3 RMSE | h=6 RMSE | h=12 RMSE |
|---|---|---|---|---|---|---|

### 8.2. Statistical significance
Run each deep model with 3 seeds, report mean ± std, and run a paired t-test of A3T-GCN vs the strongest baseline.

### 8.3. Per-sensor error maps
Compute per-sensor MAE on the test split, plot on a Berlin basemap (use the existing geojson + Folium or contextily). This is the single most compelling figure for a traffic-prediction thesis.

---

## 9. Ablations (pick at least three)

1. **Graph construction:** binary 2 km vs Gaussian kernel vs k-NN (k=8) vs identity (no graph).
2. **Attention:** A3T-GCN vs T-GCN (remove the attention module).
3. **Input length:** L ∈ {6, 12, 24, 48} hours.
4. **Feature subsets:** flow-only, speed-only, all features.
5. **Sensor-level vs detector-level nodes** (the §1.2 decision).

Each ablation is one row in the results table; the headline configuration is the one that wins on validation MAE.

---

## 10. Visualisations for the write-up

1. **Sensor location map**, coloured by mean test MAE.
2. **Predicted vs actual time series** on 3 representative sensors (one inner-ring, one autobahn, one quiet arterial) across one congested weekday and one calm Sunday.
3. **Attention weights over time** for a single sensor — show what the model is paying attention to.
4. **Error vs horizon** line plot, all baselines on one axis.
5. **Graph adjacency visualisation** (chord diagram or simple network plot) — shows the spatial structure the model exploits.

---

## 11. Proposed repository structure

```
ML Projects/
├── config/
│   ├── data.yaml
│   ├── model_a3tgcn.yaml
│   └── train.yaml
├── data/
│   ├── raw/               # current berlin_traffic_data/
│   ├── processed/
│   │   ├── berlin_traffic_tensor.pt
│   │   └── berlin_traffic_graph.pt
│   └── splits.json
├── src/
│   ├── data/
│   │   ├── download.py            # current download_berlin_traffic_data.py
│   │   ├── extract.py             # FIXED extract_traffic_archives.py
│   │   ├── build_graph.py         # current build_traffic_graph.py + Gaussian kernel
│   │   └── build_tensor.py        # NEW — implements §2
│   ├── models/
│   │   ├── a3tgcn.py
│   │   ├── baselines.py           # last-value, HA, LSTM, T-GCN
│   │   └── losses.py              # masked MAE/RMSE
│   ├── train.py
│   ├── eval.py
│   └── viz.py
├── notebooks/
│   └── exploratory/
├── results/
│   └── <run_id>/                  # checkpoints, configs, metrics.json
├── ROADMAP.md
└── README.md
```

Move the existing scripts into `src/data/` rather than rewriting them — they're already mostly right.

---

## 12. Reproducibility checklist

- [ ] Single command runs the full pipeline (`make all` or `python -m src.run --config config/main.yaml`).
- [ ] All random seeds set; `torch.use_deterministic_algorithms(True)` where possible.
- [ ] Config saved with every result, including git commit hash.
- [ ] `requirements.txt` or `pyproject.toml` pinned.
- [ ] One eval script that takes a checkpoint path and produces the full metrics table.

---

## 13. Suggested execution order (resume from here)

1. **Fix extraction** (§1.1) — re-run; confirm 12 months of data per sensor. *~1 hour, mostly waiting.*
2. **Build tensor** (§2) — write `build_tensor.py`. *Half a day.*
3. **Upgrade adjacency** (§4) — add Gaussian kernel option to `build_traffic_graph.py`. *Trivial.*
4. **Baselines** (§7, naive + HA + per-node LSTM) — get the floor numbers down. *One day.*
5. **A3T-GCN** (§5, §6) — first end-to-end training run, log to `results/run_000/`. *One day including debugging.*
6. **Per-sensor error map** (§10 figure 1) — first compelling visual. *Half a day.*
7. **Ablations** (§9, pick 3) — *2–3 days.*
8. **Thesis figures and tables** (§8, §10) — *2–3 days.*
9. **Write up methods + results sections.**

The critical path through 1–6 is roughly **one week of focused work** before you have a defensible baseline. Everything after that is depth.

---

## 14. Risks and open questions

- **Sensor coverage holes:** if the surviving-sensor count after the 70 % filter drops below ~120, the graph becomes sparse and A3T-GCN underperforms. Mitigation: lower the threshold to 60 %, or use multi-year data.
- **Class imbalance in congestion regimes:** free-flow dominates the year. Consider stratified evaluation: report metrics separately for "free flow" (`vkfz > 50`) and "congested" (`vkfz ≤ 50`).
- **Holiday calendar:** Berlin school holidays affect traffic significantly. Make sure `is_holiday` distinguishes school vs public holidays.
- **2024+ data:** if Berlin's open-data portal has 2024 published, adding it gives a second test year and is the cheapest path to a stronger result.

