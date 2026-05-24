# ST-GNN Berlin Traffic Prediction — Roadmap to Thesis-Quality Results

**Last updated:** 2026-05-24 (added ML-engineer extension track, §15–§19)
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

---

# Part II — ML-Engineer extension track (post-baseline, 3 months)

The roadmap up to §14 produces a defensible *thesis*. The roadmap from §15 onward converts the same project into a *portfolio piece* that demonstrates the skills a working ML engineer is hired for: reproducibility, deployment, monitoring, retraining, and LLM-augmented interfaces.

**Pre-requisite:** A3T-GCN trained, evaluated, and producing forecasts via §5–§8. Everything below assumes that model exists as a checkpoint on disk.

**Why this matters (learning frame):** Most M.Sc. projects stop at "model trained, paper written." The gap between that and "could ship at a company" is the layers below. Each month adds one layer; each layer is a self-contained skill you can name on a CV and defend in an interview.

---

## 15. Timeline overview

| Month | Theme | Headline deliverable | Skills demonstrated |
|---|---|---|---|
| **Month 1 (Jun–Jul 2026)** | MLOps foundations | Containerised training + inference, reproducible runs tracked in MLflow, CI on every push | Experiment tracking, Docker, data versioning, CI/CD |
| **Month 2 (Jul–Aug 2026)** | Serving, monitoring, retraining | Deployed FastAPI service with drift detection and an automated weekly retraining pipeline | API design, observability, orchestration, champion/challenger model promotion |
| **Month 3 (Aug–Sep 2026)** | GenAI / LLM layer | Natural-language query interface and/or RAG-augmented forecast explanations | Tool-using agents, RAG, embeddings, LLM evaluation |

Build a *thin* version of each layer before deepening any of them. A working end-to-end story is more valuable than any single polished component.

---

## 16. Month 1 — MLOps foundations

The goal of this month is to make every result reproducible and every artifact versioned. This is the part most students skip, and it's the single biggest "ML engineer vs ML student" signal an interviewer probes for.

### 16.1. Experiment tracking (week 1)
- Add **MLflow** (or **Weights & Biases** — pick one and stay) to `src/train.py`. Log every run's hyperparameters, per-epoch metrics, final test metrics, model checkpoint, and the Hydra/YAML config.
- Log the **dataset hash** (sha256 of `berlin_traffic_tensor.pt`) so a run is traceable to the exact data it was trained on.
- *Learning hook:* this is what makes the question "how did model v3 differ from v2?" answerable. Without it, you'll find yourself unable to defend your own results in three months.

### 16.2. Data versioning (week 1–2)
- Initialise **DVC** in the repo. Track `data/raw/` and `data/processed/` with DVC; commit the `.dvc` pointer files to git.
- Push the data remote to a free tier (S3, GDrive, or Hugging Face Datasets — HF is easiest for public ML projects).
- *Learning hook:* this teaches the separation between code (git) and data (DVC), which is the foundational MLOps idea.

### 16.3. Containerisation (week 2–3)
- Write `Dockerfile.train` that builds an image capable of running `python -m src.train --config ...` end-to-end.
- Write `Dockerfile.serve` that exposes a FastAPI inference endpoint (stub for now — Month 2 fleshes this out).
- Wire both together with `docker-compose.yml` so `docker compose up` reproduces the system locally.
- *Learning hook:* once your project runs in a container, it runs anywhere. This is the single biggest unlock for "deployable" claims.

### 16.4. CI/CD (week 3–4)
- Add a **GitHub Actions** workflow that on every push: runs linting (`ruff`), runs unit tests on data-pipeline functions (use `pytest`), builds both Docker images, and pushes them to GitHub Container Registry.
- Optional stretch: auto-deploy `Dockerfile.serve` to **Fly.io** or **Railway** free tier so the model has a real public URL.
- *Learning hook:* CI catches breakage you don't notice; CD eliminates "works on my machine." Both are interview gold.

### 16.5. Month 1 exit criteria
- [ ] Every training run produces an MLflow entry traceable to a git commit and dataset hash.
- [ ] `git clone … && docker compose up` reproduces a working serving endpoint on a fresh machine.
- [ ] CI is green on `main` and blocks merges on red.

---

## 17. Month 2 — Serving, monitoring, retraining

The goal of this month is to treat the model like a production system, not a notebook output. By the end, the system should retrain itself, notice when it's drifting, and surface that to a dashboard.

### 17.1. Production inference API (week 1)
- Flesh out `Dockerfile.serve` with a real FastAPI app:
  - `POST /forecast` — accepts `{"sensor_ids": [...], "horizon_h": 12}`, returns predictions + confidence intervals.
  - `GET /healthz` and `GET /readyz` — Kubernetes-style health checks.
  - Pydantic models for request/response validation.
- Log every request (input + prediction + latency) to a structured logger.
- *Learning hook:* input validation, health checks, and structured logs are three things every production ML service has and most student projects don't.

### 17.2. Monitoring & drift detection (week 2)
- Integrate **Evidently AI** to compare incoming request distributions to the 2023 training distribution. Generate a drift report nightly.
- Add a **Grafana + Prometheus** stack (or Evidently Cloud free tier) to visualise: request rate, p95 latency, error rate, drift score per feature, prediction MAE on labelled-after-the-fact data.
- *Learning hook:* models silently rot. Drift detection is how you catch it before users do. This is currently one of the most-asked-about MLOps skills.

### 17.3. Automated retraining pipeline (week 3)
- Build a **Prefect** (or Airflow, or Dagster — Prefect is the easiest start) flow that runs weekly:
  1. Pull the past week of fresh VMZ data.
  2. Append to the existing tensor, re-compute normalisation stats on the rolling window.
  3. Retrain the model from the previous champion checkpoint.
  4. Evaluate on a frozen holdout.
  5. If new MAE beats champion by ≥ 2 %, promote it (champion/challenger pattern). Otherwise alert.
- *Learning hook:* "automated retraining with quality gates" is the sentence that separates hobby projects from production thinking. The promotion gate is the key design decision — discuss it explicitly in your README.

### 17.4. Operator dashboard (week 4)
- Single **Streamlit** (or Grafana) page exposing: current champion model version, last retrain timestamp, last drift report, live forecast for a user-selected sensor, predicted-vs-actual chart.
- *Learning hook:* visualisation isn't just for the thesis — operators need to *see* the system. Building this teaches you what telemetry actually matters.

### 17.5. Month 2 exit criteria
- [ ] Public (or local) URL serves forecasts with sub-500ms p95 latency.
- [ ] Drift report regenerates nightly and is visible in the dashboard.
- [ ] Retraining flow has run at least 4 weekly cycles and promoted (or rejected) a challenger.

---

## 18. Month 3 — GenAI / LLM layer

This is the month that makes the project stand out. Most M.Sc. traffic projects have zero LLM component; adding one demonstrates skills that didn't even exist on job descriptions two years ago. Pick **one** of the two angles below as the headline; treat the other as a stretch.

### 18.1. Angle A — Natural-language query interface
Build a small agent that lets a non-technical user ask questions like *"What's the predicted congestion on the A100 around 5pm Friday?"* or *"Compare this Friday's forecast to last Friday's actuals."*

- Use the **Anthropic SDK** (or OpenAI / LangGraph) with tool use.
- Expose three tools to the LLM:
  1. `get_forecast(sensor_ids, time_range)` → calls your §17.1 FastAPI.
  2. `get_historical(sensor_ids, time_range)` → queries the tensor.
  3. `get_sensor_by_location(address_or_lat_lon)` → resolves Berlin street names / coords to `teuID`.
- Build a thin Streamlit/Gradio chat UI on top.
- *Learning hook:* this teaches function calling, tool design, and the agent loop — the most-hired-for GenAI skill in 2026.

### 18.2. Angle B — RAG-augmented forecast explanations
The model predicts numbers; an LLM turns them into explanations grounded in real-world context.

- Curate a small knowledge base of Berlin events (concerts, demos, road closures, BVG strikes, weather alerts) — scrape `berlin.de` or use a manual CSV for the MVP.
- Embed entries (e.g. `text-embedding-3-small` or a local model) into **Chroma** or **Qdrant** (both free, local).
- On each forecast request, retrieve top-k events for the time/location, pass them + the numeric forecast to an LLM, return a *narrative* prediction: *"Expect heavier than usual congestion — Hertha BSC plays at 19:00 and there is roadwork on Kurfürstendamm."*
- *Learning hook:* this teaches the full RAG stack (chunking, embedding, retrieval, prompt assembly) plus the hybrid ML+LLM pattern, which is where industry is heading.

### 18.3. LLM evaluation (week 3–4, both angles)
- Build a small **golden eval set** (~30–50 prompts) with expected behaviours (tools called, facts cited, no hallucination).
- Score with a mix of: programmatic checks (was the right tool called?), LLM-as-judge (Claude grades Claude's output on faithfulness), and manual review.
- Track eval scores in MLflow alongside the ST-GNN metrics.
- *Learning hook:* "how do you know the LLM isn't hallucinating?" is the question every interviewer asks. Having an eval framework is the answer.

### 18.4. Month 3 exit criteria
- [ ] User can ask a natural-language traffic question and get a grounded answer (Angle A) or a narrative forecast (Angle B).
- [ ] Eval set passes at ≥ 80 % on the chosen angle.
- [ ] README has a demo GIF or recorded video — recruiters watch these.

---

## 19. Skills-on-CV mapping

What the completed Part II earns you, in interview-ready language:

| Skill bucket | The concrete claim | Evidence in the repo |
|---|---|---|
| **MLOps foundations** | "Versioned data with DVC and tracked experiments with MLflow across 50+ runs" | `.dvc/` files, MLflow tracking server, `results/` directory |
| **Containerisation** | "Containerised training and serving with Docker; orchestrated locally via compose" | `Dockerfile.train`, `Dockerfile.serve`, `docker-compose.yml` |
| **CI/CD** | "GitHub Actions pipeline runs tests, builds images, deploys to Fly.io on merge to main" | `.github/workflows/*.yml`, public deployed URL |
| **Production serving** | "FastAPI inference service with Pydantic validation, structured logging, p95 < 500ms" | `src/serve/`, logged latency metrics |
| **Monitoring** | "Drift detection with Evidently; Grafana dashboards on latency, error rate, drift score" | Evidently reports, Grafana JSON |
| **Orchestration** | "Weekly retraining pipeline in Prefect with champion/challenger promotion gate" | `flows/retrain.py`, promotion log |
| **GenAI / agents** | "Tool-using LLM agent over the forecast API; eval set scored 8x.x % faithfulness" | `src/agent/`, golden eval set, MLflow LLM metrics |
| **RAG** *(if angle B)* | "RAG pipeline over Berlin events with Chroma; hybrid numeric+narrative forecasts" | `src/rag/`, embedding index |

The full pitch — *"Built and deployed an ST-GNN traffic forecasting system on Berlin sensor data with MLflow experiment tracking, Dockerised FastAPI serving, automated weekly retraining via Prefect with drift detection, and an LLM-based natural-language query interface using tool calling and RAG"* — touches every bullet on a junior-to-mid ML engineer JD, all traceable to one coherent project.

---

## 20. Open questions for Part II

- **Cloud vs local:** is a Fly.io / Railway free tier sufficient, or is it worth burning AWS/GCP free-tier credits to demonstrate cloud-native deployment (SageMaker / Vertex AI) explicitly?
- **Which LLM provider:** Anthropic SDK is straightforward and the eval story is cleanest, but a local Llama / Mistral angle would additionally demonstrate self-hosted inference.
- **Scope discipline:** if Month 1 slips, drop Angle B from Month 3 rather than skipping the eval framework — the eval framework is the more interview-relevant of the two.

