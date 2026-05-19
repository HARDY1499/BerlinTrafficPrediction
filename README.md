# BerlinTrafficPrediction

> Forecasting urban traffic speeds across Berlin with a Spatio-Temporal Graph Neural Network (ST-GNN). M.Sc. portfolio project — Data Analytics & Decision Science, RWTH Aachen.

---

## At a glance

| | |
|---|---|
| **Task** | Forecast vehicle speeds 3, 6, and 12 hours ahead at every inductive-loop detector in Berlin |
| **Data** | Berlin VMZ open inductive-loop traffic detectors, 2023 (full year, hourly) |
| **Nodes** | 454 detectors (after filtering for data modernity and completeness + location availability in GeoJSON) |
| **Edges** | Spatial proximity (Gaussian-kernel adjacency, ≤ 2 km) |
| **Model** | A3T-GCN — Attention-based Temporal Graph Convolutional Network |
| **Baselines** | Last-value, historical average, per-node LSTM, T-GCN (no attention) |
| **Status** | Data engineering in progress (see [Current status](#current-status)) |

A detailed step-by-step plan from the current state to a thesis-quality result is in [ROADMAP.md](ROADMAP.md).

---

## Why this project

Urban traffic prediction is the canonical real-world test for Spatio-Temporal Graph Neural Networks. A traffic network is *literally* a graph: detectors are nodes, road segments connecting them are edges, and conditions propagate along those edges over time. Predicting tomorrow morning's congestion at one detector means combining its own recent history (temporal signal) with what's currently happening at upstream and adjacent detectors (spatial signal). Models that mix these two kinds of information — convolution over the graph + recurrence or attention over time — consistently beat models that use only one.

This project picks up Berlin's open inductive-loop data (publicly published by the Verkehrsmanagementzentrale, the city's traffic management center) and builds a complete ST-GNN pipeline end-to-end: ingestion, cleaning, graph construction, model training, evaluation against principled baselines, and ablation studies. The goal is both a defensible result on a real city's data and a thorough engineering walkthrough I can talk about in an interview.

---

## Background: what an ST-GNN actually does

A traffic prediction problem looks like this:

- **Inputs:** for each of `N` detectors, the last `L` hours of measurements (flow, speed, etc.). Shape `(L, N, F_in)`.
- **Outputs:** the speed at each detector for the next `H` hours. Shape `(H, N, 1)`.
- **Side information:** a graph `G = (V, E, W)` where vertices are detectors, edges connect spatially close detectors, and edge weights `W_ij` are higher when detectors are closer.

A vanilla LSTM trained on a single detector ignores the graph entirely — it doesn't know that a slowdown one kilometer upstream usually predicts a slowdown here in twenty minutes. A Graph Convolutional Network (GCN) without recurrence ignores time — it doesn't know that *this morning's pattern* is informative for *this morning's prediction*. An ST-GNN combines them: at each timestep it applies a graph convolution (each node aggregates a weighted average of its neighbours' features) and feeds the result into a temporal model (GRU, LSTM, attention, or temporal convolution).

**A3T-GCN** (Bai et al., 2019), the target architecture for this project, is one of the cleaner instances of that recipe:

1. **Spatial layer:** at each timestep, run a 2-layer GCN over the graph to produce a node embedding informed by neighbours.
2. **Temporal layer:** feed the per-timestep embeddings through a GRU to model how patterns evolve.
3. **Attention layer:** learn a weighted sum over the GRU's hidden states across time, so the model can decide which past timesteps matter most for predicting the future.
4. **Output head:** an MLP per node mapping the attention-weighted embedding to the multi-step speed prediction.

It's the right starting model here because it's well-cited, has a reference implementation in PyTorch Geometric Temporal, trains in minutes on a laptop, and is comfortably ablatable — turning off the attention gives plain T-GCN, replacing the GCN gives a per-node LSTM, etc. — which gives a thesis its required baselines for free.

---

## The data

### Source

[Verkehrsdetektion Berlin](https://daten.berlin.de/datensaetze/verkehrsdetektion-berlin) — official open dataset from the city's traffic management center. Two artifacts are downloaded:

1. **Time series:** twelve monthly `.tgz` archives at `https://mdhopendata.blob.core.windows.net/verkehrsdetektion/2023/neue_qualitaetssicherung/Fahrstreifendetektoren/detektoren_2023_<MM>.tgz`, each containing ~485 CSV files (one per detector). After fixing the extraction (see [Design decisions](#design-decisions)), concatenating all twelve gives a full hourly year per detector.

2. **Geography:** `Standorte_Verkehrsdetektion_Berlin.geojson` — one record per detector, with WGS-84 coordinates and metadata (direction, lane assignment, position description).

The 485 CSVs per month split into 466 in the modern `TEU####_Det#` naming and 19 in a legacy `teuscalaS00000DD#####D#` format. Of the modern files, 454 have a matching GeoJSON entry and 12 do not (decommissioned or unregistered). Of the 19 legacy files, 5 are mappable to modern IDs that do appear in the GeoJSON, but we **exclude all 19 on data-modernity grounds** — they come from a different recording pipeline with an unverified schema, and quietly mixing them in could corrupt the tensor (see [Design decisions §6](#design-decisions)). That leaves **454 detectors** usable.

### Physical setup

An *inductive-loop detector* is a wire coil embedded in the road surface that registers each vehicle passing over it. A single roadside *sensor site* (`TEU####`) typically hosts several detectors (`_Det0`, `_Det1`, …) — one per lane, sometimes split across both directions of travel. The dataset is therefore *detector-level*, not pole-level: 454 detectors across roughly 206 physical sites.

### CSV schema (one file = one detector, one month)

Each row is one hour. Columns:

| Column | Type | Meaning | Used in this project |
|---|---|---|---|
| `utc` | timestamp (UTC) | Hour start in UTC | **Yes** — time index |
| `Datum (Ortszeit)` | date | Local Berlin date | No — DST-prone |
| `Stunde des Tages (Ortszeit)` | int 0–23 | Local hour-of-day | Indirectly (cyclic encoding) |
| `localTime` | timestamp | Local Berlin timestamp | No |
| `month` | int 1–12 | UTC month | No (redundant with `utc`) |
| `Vollständigkeit` | float 0–100 | % of the hour successfully measured | **Yes** — quality filter |
| `qkfz` | int | Total vehicles in this hour (Kraftfahrzeug) | **Yes** — input feature |
| `qpkw` | int | Passenger cars (Personenkraftwagen) | **Yes** — input feature |
| `qlkw` | int | Lorries / trucks (Lastkraftwagen) | **Yes** — input feature |
| `vkfz` | float (km/h) | Avg. speed of all vehicles | **Yes** — input + prediction target |
| `vpkw` | float (km/h) | Avg. speed of passenger cars | **Yes** — input feature |
| `vlkw` | float (km/h) | Avg. speed of lorries | **Yes** — input feature |
| `ZScore_Det0/1/2` | float | Anomaly scores for sub-loops | No Use — let the GNN learn anomalies |
| `hist_cor` | float | Internal calibration metric | No |

### Reading a row

```
utc=2023-08-31 22:00 UTC  |  qkfz=186 vehicles  |  qpkw=154 cars + qlkw=32 trucks
                          |  vkfz=74 km/h       |  Vollständigkeit=100%
```

In plain English: during the hour starting at 00:00 local time on 1 September 2023, this lane saw 186 vehicles travelling at an average of 74 km/h, with full data completeness. That's one row out of ~8,400 useful rows per detector per year.

### Patterns visible in the raw data

Even without modelling, a one-detector exploration recovers what you'd expect:

- **Daily cycle:** an Autobahn detector sees ~555 vehicles/h at 3–5 a.m., rising to ~1,120 at midday and falling to ~801 in the evening peak. Models that fail to recover this cycle are broken before evaluation.
- **Weekly cycle:** Friday (~771 veh/h) is the busiest day, Sunday (~624) the quietest. The effect is small (≈25 %) but real and consistent.
- **Sibling-detector correlations are informative, not redundant.** Two detectors on the same physical pole but opposite directions show flow correlation around 0.85 but **speed correlation near zero** — both directions get busy, but only one gets congested. This is exactly the asymmetry we want the graph to capture, and it's the empirical justification for treating detectors (not pole-level sites) as graph nodes.

---

## Approach

### Pipeline

```
   raw .tgz archives                         GeoJSON
          │                                     │
          ▼                                     │
   monthly CSV files                            │
   (per detector, per month)                    │
          │                                     │
          ▼                                     ▼
   ┌─────────────────────┐         ┌─────────────────────┐
   │  build_tensor.py    │         │ build_traffic_graph │
   │  (T, N, F) tensor   │         │  adjacency + coords │
   │  + mask + metadata  │         │   (Gaussian kernel) │
   └──────────┬──────────┘         └──────────┬──────────┘
              │                                │
              └────────────┬───────────────────┘
                           ▼
                   ┌──────────────┐
                   │   A3T-GCN    │ ←── trained against baselines
                   └──────┬───────┘     (last-value, HA, LSTM, T-GCN)
                          ▼
            predictions at t+3h, t+6h, t+12h
                          ▼
              evaluation + ablations + figures
```

### Modelling choices

- **Detector-level graph (~454 nodes).** Sibling detectors on the same pole carry genuinely different signals (different directions, different lanes, different vehicle compositions). Aggregating to pole-level would average out the directional asymmetry that's the single most informative signal in this dataset.
- **Gaussian-kernel adjacency,** thresholded at 2 km:

  `W_ij = exp(-d²_ij / σ²) · 𝟙[d_ij ≤ 2000 m]`

  This is the standard from STGCN (Yu et al., 2018). At `d = 0` (sibling detectors on the same pole) the edge weight is 1; at `d = 2 km` it has decayed to a small value; beyond that it's zero. That cleanly distinguishes sibling pairs from distant neighbours, which a binary adjacency cannot.
- **Multi-step prediction** at horizons of 3, 6, and 12 hours, with `vkfz` (all-vehicle average speed) as the headline target. Multi-step lets the results table show how the model degrades with horizon — the more interesting story than a single-horizon number.
- **Masked MAE loss.** Missing values are masked (not deleted) so the temporal grid stays uniform; the loss ignores masked positions.
- **Train / val / test** is chronological — Jan 1 – Aug 31 / Sep 1 – Oct 15 / Oct 16 – Dec 31. Random shuffling of timeseries is a silent leakage bug and is never used.

---

## Repository structure

```
ML Projects/
├── README.md                              ← you are here
├── ROADMAP.md                             ← full plan from current state to results
├── download_berlin_traffic_data.py        ← download 12 monthly .tgz archives
├── extract_traffic_archives.py            ← extract into per-month subdirectories
├── explore_traffic_csv.py                 ← single-file CSV inspection (one-off)
├── explore_sensor_locations.py            ← GeoJSON inspection (one-off)
├── build_traffic_graph.py                 ← sensor-location → adjacency matrix
├── build_tensor.py                        ← raw CSVs → (T, N, F) tensor [in progress]
├── berlin_traffic_graph.pt                ← saved graph artifact
├── Standorte_Verkehrsdetektion_Berlin.geojson  ← detector locations
└── berlin_traffic_data/2023/
    ├── Fahrstreifendetektoren_tgz/        ← downloaded archives
    └── CSV_data/01../12/                  ← extracted per-detector × per-month CSVs
```

---

## How to reproduce

```bash
# 1. Clone the repository and create a virtual environment.
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # (requirements.txt is a TODO — see ROADMAP)

# 2. Download the raw data (~hundreds of MB).
python download_berlin_traffic_data.py

# 3. Extract into per-month subdirectories.
python extract_traffic_archives.py

# 4. Build the graph adjacency from the GeoJSON.
python build_traffic_graph.py

# 5. Build the (T, N, F) tensor from the CSVs.
python build_tensor.py                    # currently implements Step 1; see below

# 6. Train (not yet implemented). Tracked in ROADMAP.md.
```

---

## Current status

Project resumed May 2026 after a six-month pause. Phase-by-phase:

| Phase | Status | Notes |
|---|---|---|
| Data ingestion (download + extract) | done | Extraction bug fixed — see [Design decisions](#design-decisions) |
| Exploratory data analysis | done | Full-year coverage, Vollständigkeit distribution, sibling-correlation study |
| Graph topology (sensor-level prototype) | done | `berlin_traffic_graph.pt`; will be rebuilt at detector level |
| Detector-level graph (Gaussian kernel + node features) | pending | After tensor build is complete |
| `build_tensor.py` | in progress | Step 1 (time axis + node list) implemented |
| A3T-GCN model | pending | |
| Baselines (HA, LSTM, T-GCN) | pending | |
| Training & evaluation | pending | |
| Ablations & write-up | pending | |

See [ROADMAP.md](ROADMAP.md) for the full forward plan.

---

## Design decisions

These are the choices I'd talk about in an interview — the *what* is in the code, but the *why* lives here.

### 1. Detector-level vs pole-level nodes

The data is published at detector level (one file per lane per direction), but a naive "one node per physical pole" approach would collapse the ~454 detectors into ~206 nodes. I tested the aggregation hypothesis empirically before committing: at one Autobahn pole with detectors in both directions, the flow correlation across directions is around 0.85 but the **speed correlation is near zero** — a classic morning-rush asymmetry where both directions are busy but only one is congested. Aggregating speeds across directions would average that out. **Decision:** keep detectors as nodes; treat direction and lane as additional node features.

### 2. Gaussian-kernel adjacency, not binary

A binary `d < 2 km` rule gives edge weight 1 to a sibling detector 0 m away *and* a neighbour 1.9 km away — those are obviously not equally relevant. A Gaussian kernel `exp(-d²/σ²)` decays smoothly with distance, naturally weighting siblings higher than distant neighbours. The threshold (2 km) just prevents the matrix from becoming fully dense; the kernel does the weighting. This is the STGCN convention and one less bespoke choice to defend in the methods section.

### 3. Extraction bug discovery

The first version of `extract_traffic_archives.py` extracted all twelve monthly archives into the same flat directory. Each archive contains identically-named files (e.g. `TEU00002_Det0.csv`), so each subsequent month silently overwrote the previous one — only September data survived. Discovered during the re-audit; fix is to extract each archive into a per-month subdirectory and concatenate during preprocessing. This is the kind of bug that doesn't error and quietly wastes a model run; an interviewer asking "how do you catch silent data issues?" gets a real answer here.

### 4. Chronological splits, masked loss

Random train/test splits on a timeseries leak the future into the training set: adjacent hours are nearly i.i.d., so the model effectively memorises rather than generalises. Splits in this project are strict chronological. Missing values are masked rather than dropped, so sliding-window batches never contain a hidden gap. Both are unglamorous, both are essential, both are common interview red flags when a candidate has them backwards.

### 6. Legacy-format files excluded on modernity grounds

19 of the 485 monthly CSVs use a legacy `teuscalaS00000DD#####D#` naming convention. Mapping them to the modern `TEU####_Det#` scheme shows 5 of those 19 *do* have matching GeoJSON entries, so they could in principle be added as nodes. They are nonetheless excluded for now: the different naming almost certainly reflects a different recording or processing pipeline, and adding them without first validating the column schema, units, and quality-flag conventions could introduce a silent calibration drift in 5 of 459 columns of the tensor. Cheap to revisit later as an ablation; expensive to debug if it bites mid-training.

### 5. Reproducibility-first config

Every result will be written to `results/<run_id>/` with the full config (YAML), seed, and git commit hash. A thesis examiner asking "can you reproduce Figure 4?" should be answerable with `python train.py --config results/<run_id>/config.yaml`. (See ROADMAP §12.)

---

## What I'm learning by building this

- **Data engineering is the project.** The actual model is a few hundred lines of PyTorch; the preprocessing, masking, and reindexing logic is what determines whether the model gets a chance to learn anything. The interview soundbite: "I spent more time on `build_tensor.py` than on the model — that's the right ratio."
- **Defensive design pays for itself.** Locking the time axis and node list at the start of preprocessing (Step 1 of `build_tensor.py`) eliminates an entire category of alignment bug that would otherwise show up much later, in confusing ways, during training.
- **Graphs are a modelling choice, not a property of the data.** "Distance-based proximity" is one of many adjacencies you could build (k-NN, functional similarity, directed-by-road-direction). Each is a hypothesis about which signals matter, and ablating them is the most interesting part of the experiment.

---

## References

- Bai, J. et al. (2019). *A3T-GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting.* [arXiv:2006.11583](https://arxiv.org/abs/2006.11583)
- Yu, B., Yin, H., & Zhu, Z. (2018). *Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting.* IJCAI 2018. [arXiv:1709.04875](https://arxiv.org/abs/1709.04875)
- Li, Y. et al. (2018). *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting.* ICLR 2018. [arXiv:1707.01926](https://arxiv.org/abs/1707.01926)
- Verkehrsmanagementzentrale Berlin. *Verkehrsdetektion (Open Data).* [daten.berlin.de](https://daten.berlin.de/datensaetze/verkehrsdetektion-berlin)
- Rozemberczki, B. et al. (2021). *PyTorch Geometric Temporal: Spatiotemporal Signal Processing with Neural Machine Learning Models.* CIKM 2021.

---

## Author

Harjasdeep Singh Allahabadia — M.Sc. Data Analytics & Decision Science, RWTH Aachen.
Project ongoing; feedback and discussion welcome via repo issues.
