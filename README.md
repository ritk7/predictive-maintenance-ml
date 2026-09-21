# Turbofan Predictive Maintenance

Predicts Remaining Useful Life (RUL) and maintenance risk for jet engines
using NASA's C-MAPSS FD001 turbofan degradation dataset.

**▶ Live demo: https://ritk7.github.io/predictive-maintenance-ml/**

An interactive console showing all 100 held-out test engines ranked by
predicted remaining life, each engine's real sensor trace, and a draggable
alert threshold that recomputes precision, recall and missed failures from
the actual sweep. Every figure on it is real model output — the page is
generated from the trained models by `make demo`, with the payload inlined
so it needs no backend.

All numbers below are real outputs from the committed code, including the
ones that are unflattering. Section 11 lists known limitations honestly.

## 1. Dataset

NASA C-MAPSS FD001: 100 training engines run to failure, 100 test engines
with trajectories truncated before failure (true RUL supplied separately).
Single operating condition, single fault mode (HPC degradation).

**Source**: NASA's original host is gone, so these are pulled from a
byte-identical GitHub mirror:
`https://github.com/hankroark/Turbofan-Engine-Degradation` (`CMAPSSData/`).
If it disappears, download `train_FD001.txt`, `test_FD001.txt`,
`RUL_FD001.txt` from any C-MAPSS mirror (e.g. the Kaggle "NASA C-MAPSS"
dataset) into `data/`.

Columns (no header): `unit, cycle, setting_1..3, sensor_1..21`.

## 2. Project structure

```
data/                     raw C-MAPSS txt files (gitignored)
src/pipeline/
  config.py               all tunable thresholds/paths/hyperparameters
  data_loader.py          raw file loading + RUL label computation
  features.py             rolling mean/std + slope features, warm-up drop
  ood.py                  Mahalanobis out-of-distribution input guard
  build_dataset.py        assembles splits, verifies split integrity
src/models/
  train_regression.py     Linear / RandomForest / XGBoost RUL regressors
  train_classification.py RandomForest / XGBoost maintenance classifiers
src/api/
  schemas.py              pydantic models + input validation
  inference.py            artifact loading, feature prep, prediction
  main.py                 FastAPI app
scripts/
  sample_request.py       one real prediction against a running API
  export_demo_data.py     dumps real model output for the hosted demo
  build_demo.py           inlines that payload into docs/index.html
frontend/
  index.html              live dashboard (needs the API running)
  demo_template.html      source for the hosted demo page
docs/                     GitHub Pages: the self-contained demo + payload
models_saved/             trained artifacts (gitignored)
Makefile                  setup / train / api / dashboard / demo targets
```

## 3. Feature engineering

- **Dropped sensors** (verified std ≈ 0 on training data): `sensor_1, 5, 6,
  10, 16, 18, 19`. Under FD001's single operating condition these are
  physically pinned constants and carry no degradation signal.
- **Rolling mean & std** per kept sensor (windows 5 and 10) — level and
  local volatility, both of which shift as an engine degrades.
- **Degradation slope** — 10-cycle trailing least-squares slope. The
  rolling mean says where a sensor *is*; the slope says which direction and
  how fast it is moving, which is what remaining life depends on.

### Causality — verified, not assumed

All rolling/slope features are computed within `groupby("unit")` using
trailing pandas windows. This was verified empirically rather than asserted:

- Perturbing a **future** cycle of an engine changes **0** feature cells in
  that engine's earlier rows.
- Perturbing **another engine entirely** changes **0** feature cells.
- Rolling mean at cycle *t* equals the mean of that unit's cycles in
  *(t-w, t]* exactly.

### Warm-up rows are dropped (train/serve consistency)

Each engine's first 9 cycles cannot fill a 10-cycle window, so their
rolling-std and slope features collapse to structural zeros. Training on
them teaches a mapping the API can never reproduce. They are dropped
(**900 rows, 4.36% of training data**) so the training feature
distribution matches what is served. This *raised* reported test RMSE
(those were easy high-RUL rows) — the pre-fix numbers were flattered.

## 4. Train/test split integrity

C-MAPSS is one row per (engine, cycle); splitting by row would put adjacent
cycles of the same engine on both sides. The split is **by unit number only**:

```
[internal train/val split check] train units: 80, val units: 20, overlap: 0
[internal train/val split check] PASSED — zero unit_id overlap confirmed.
```

`test_FD001.txt` is a separate file of 100 independent engines, never mixed
into train/val. C-MAPSS numbers units 1–100 independently *per file*, so
"unit 7" in train and "unit 7" in test are different physical engines — a
raw cross-file unit-number comparison would be a false leakage signal, and
`build_dataset.py` documents this rather than silently skipping the check.

**Scaler**: `StandardScaler` is fit on training data only, then applied to
val/test. Never fit on combined data. **RUL clipping** uses a fixed
constant from config, not a data-derived quantity, so it carries no label
information across splits.

## 5. RUL regression results

RUL is clipped at **125 cycles**. Engines run flat and healthy for a long
stretch before observable degradation, so regressing exact RUL during that
stretch (RUL 280 vs 300) asks for a number the sensors do not encode.
Clipping reframes the target as "healthy vs. degrading on a known
trajectory."

**Held-out test set** (all figures RMSE/MAE, lower is better):

| Model | RMSE (all rows) | MAE (all) | RMSE (last-cycle) | MAE (last) | RMSE (last, unclipped) |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 20.222 | 16.211 | 20.336 | 15.663 | 21.234 |
| Random Forest | 17.117 | 12.168 | 18.155 | 12.463 | 19.085 |
| **XGBoost** | **17.072** | 12.278 | **17.686** | 12.358 | 18.572 |

Two protocols are reported because they differ materially and quoting only
the flattering one would be misleading:

- **all rows** = every test engine-cycle (12,196 rows).
- **last-cycle** = one row per engine (100 rows). **This is the protocol
  published C-MAPSS FD001 benchmarks use.** Against it, XGBoost scores
  **17.69 RMSE**. That is a respectable classical-ML result but *behind*
  the best published deep-learning results (~12–14 RMSE); an earlier draft
  of this README claimed benchmark parity, which was wrong.

Model selection uses last-cycle RMSE, since scoring an engine "as of now"
is the decision the deployed system actually makes.

### Overfitting — reported, not hidden

| Model | Train RMSE | Val RMSE | Test RMSE | Val−Train gap |
|---|---:|---:|---:|---:|
| Linear Regression | 20.205 | 18.940 | 20.222 | −1.264 |
| Random Forest | 10.326 | 16.064 | 17.117 | +5.738 |
| XGBoost | 10.926 | 16.010 | 17.072 | +5.084 |

The tree models overfit by ~5 RMSE (train vs val, a like-for-like
comparison — both come from the same file, split by unit). The linear
baseline shows no gap, i.e. it is high-bias rather than high-variance.
Hyperparameters were set to sensible defaults, **not** systematically tuned
by cross-validated search — see §11.

## 6. Maintenance risk classification

Label: **"needs maintenance within 30 cycles"**.

**Why N=30**, measured rather than asserted — class balance by horizon:

| N | train %pos | val %pos | test %pos |
|---:|---:|---:|---:|
| 10 | 5.56% | 5.66% | 0.14% |
| 20 | 10.61% | 10.80% | 1.01% |
| 30 | 15.66% | 15.94% | 2.72% |
| 40 | 20.71% | 21.08% | 4.89% |
| 50 | 25.76% | 26.22% | 7.26% |

N=20 leaves only ~1% positives in the test set — too sparse to estimate
recall stably. N=30 gives a non-trivial 15.66% / 2.72% split and matches
realistic maintenance lead time. (Test rates are lower than train because
test trajectories are truncated at random points, not near failure.)

**Why accuracy is the wrong headline metric**: at this imbalance, always
predicting "no maintenance" scores ~97% accuracy while catching zero
failures. Accuracy is reported only alongside the note that it is
misleading.

**Why F2, not F1, and why the threshold is not 0.5**: a missed failure
(engine fails in service) costs far more than a false alarm (an
unnecessary inspection). F1 weights those two errors equally, which does
not reflect the business. Models are selected on **F-beta with beta=2**
(recall weighted 2× precision) and the decision threshold is tuned on the
sweep below instead of left at the arbitrary 0.5 default:

Random Forest threshold sweep (PR-AUC 0.811):

| thr | precision | recall | F1 | F2 | missed failures (FN) | false alarms (FP) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.4208 | 0.9277 | 0.5789 | 0.7476 | 24 | 424 |
| **0.30** | **0.5078** | **0.8825** | 0.6447 | **0.7690** | **39** | 284 |
| 0.40 | 0.5918 | 0.8253 | 0.6893 | 0.7649 | 58 | 189 |
| 0.50 | 0.6615 | 0.7771 | 0.7147 | 0.7509 | 74 | 132 |
| 0.70 | 0.8059 | 0.6627 | 0.7273 | 0.6871 | 112 | 53 |

Moving 0.50 → 0.30 cuts missed failures **74 → 39** for 152 extra false
alarms. In this domain that is the correct trade, and it is the
configured operating point (`DECISION_THRESHOLD = 0.30`).

**Final comparison @ threshold 0.30:**

| Model | Precision | Recall | F1 | F2 |
|---|---:|---:|---:|---:|
| **Random Forest** | 0.5078 | **0.8825** | 0.6447 | **0.7690** |
| XGBoost | 0.6361 | 0.7952 | 0.7068 | 0.7573 |

Confusion matrices `[[TN, FP], [FN, TP]]`:

```
Random Forest:  [[11580   284]      XGBoost:  [[11713   151]
                 [   39   293]]                [   68   264]]
```

Random Forest is selected: it catches 293 of 332 true failures vs
XGBoost's 264. Note XGBoost wins on F1 — selecting on F1 would have picked
the model that misses 29 more failures, which is exactly the trap F2 avoids.

**Imbalance handling**: `class_weight='balanced'` (RF) and
`scale_pos_weight` (XGB). No resampling: duplicating a minority row would
duplicate near-identical rolling windows from the same trajectory and
inflate apparent performance.

**Classifier overfitting** (F1 at threshold 0.30): RF train 0.873 → val
0.802 → test 0.645; XGB train 0.975 → val 0.848 → test 0.707. XGBoost in
particular is close to memorizing the training set.

## 7. API

```bash
uvicorn src.api.main:app --reload
```

- `POST /predict` — `{unit_id, readings: [{cycle, setting_1..3,
  sensor_1..21}, ...]}`, oldest→newest. Returns predicted RUL, risk band,
  maintenance flag + probability, the decision threshold used, an
  input-plausibility warning, and which models ran.
- `GET /engines` — risk status for all 100 test engines (dashboard source).
- `GET /engines/{unit_id}/history` — sensor history for trend charts.
- `GET /health` — liveness.

**Validation** (all rejected with **400** and an actionable message, never
a bare 500): missing/extra/wrong-typed fields, NaN/Inf, non-ascending or
duplicate cycles, reading count outside `[10, 500]`, per-sensor
out-of-plausible-range values, and **implausible sensor combinations**
(see §9). Verified: zero false rejections across all 33,727 real data rows
and all 100 test engines.

All thresholds live in `src/pipeline/config.py`.

## 8. Frontends

Two, for two different jobs:

- **`frontend/index.html` — live dashboard.** Talks to the running API. Card
  grid color-coded by risk, filter/search, click through for predicted RUL,
  maintenance probability, and sensor trend charts. Needs `make api`.
- **`docs/index.html` — hosted static demo** (the live link above, and the
  one to open first). A
  self-contained operations console with no backend: the fleet ranked as a
  thermal curve, per-engine sensor traces, and an interactive threshold
  explorer driven by the real precision/recall sweep. Severity is encoded as
  temperature because that is the actual physics here — LPT outlet
  temperature climbs measurably as the compressor degrades (engine 34:
  1390 → 1427 °R across its life).

Rebuild the demo after retraining with `make demo`, which runs
`scripts/export_demo_data.py` (dumps real predictions, traces and sweeps to
`docs/demo_data.json`) then `scripts/build_demo.py` (inlines that payload
into `docs/index.html`). Inlining rather than fetching keeps the demo a
single file that works identically over `file://`, any static host, and
GitHub Pages.

## 9. Issues found in audit and fixed

1. **Silent wrong answers on short histories (critical).** With fewer than
   10 cycles, rolling-std/slope collapsed to zero and the model read the
   engine as healthy. Engine 24 (true RUL 20) was scored **RUL 42.4,
   "medium" risk** from 1 cycle vs **RUL 15.9, "high"** from 10+ — an
   engine needing urgent maintenance reported as medium, with no warning.
   Fixed: histories under `MIN_HISTORY_CYCLES` are rejected with a 400
   explaining why, and matching warm-up rows are dropped from training.
2. **`/engines/{id}/history` rebuilt the entire dataset per request** —
   ~8.9s per call and a trivial DoS vector. Fixed with a process-lifetime
   cache: **8.9s → 0.01s**.
3. **Impossible sensor combinations accepted.** All 21 sensors pinned to
   their individual minima passed every per-field bound and scored
   "healthy, RUL 121". Per-field ranges cannot see correlations. Fixed with
   a Mahalanobis distance guard calibrated on training data (`src/pipeline/ood.py`):
   that input now returns 400 (distance 14522 vs threshold 22.8) while all
   100 real engines still pass.
4. **Arbitrary 0.5 decision threshold** hardcoded in two places, and model
   selection on F1 despite an asymmetric cost structure. Fixed: threshold
   moved to config and tuned to 0.30 on F2; selection now uses F2.
5. **Misleading benchmark claim.** The README claimed the all-rows RMSE was
   comparable to published FD001 results; those use the last-cycle
   protocol. Both are now reported and the claim corrected.
6. **Code quality**: removed dead `TEST_ENGINES_CACHE` config and an unused
   import; replaced private cross-module imports (`_REGRESSOR_KEY`) with
   public accessors; replaced deprecated `@app.on_event` with `lifespan`;
   moved a magic `500` to a named constant.

Verified clean and needing no fix: feature causality (§3), scaler fitting,
clipping consistency, concurrency (20 simultaneous requests: all 200,
byte-identical results — models are read-only after load).

## 10. Running it end-to-end

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
brew install libomp                          # macOS only: XGBoost needs OpenMP.
                                              # Linux/Windows wheels bundle OpenMP already —
                                              # nothing extra to install there.

python src/models/train_regression.py
python src/models/train_classification.py
uvicorn src.api.main:app --reload                 # API on :8000
python -m http.server 8080 --directory frontend   # dashboard on :8080
```

## 11. Known limitations

- **Fast-failing engines are mispredicted in the dangerous direction.** The
  model regresses toward fleet-typical degradation, so atypically
  short-lived engines are reported as healthier than they are. Worst case
  in the test set: unit 41 (true RUL 18) predicted **43.6**; unit 37 (true
  RUL 21) predicted 66.9. Slow-failing engines are under-predicted
  (unit 12: true 124, predicted 91.3), which errs safe. The RUL regressor
  should not be the sole trigger for grounding an engine; the classifier,
  tuned for recall, is the safer signal.
- **Hyperparameters are sensible defaults, not tuned.** No cross-validated
  search was run. The ~5 RMSE train/val gap suggests regularization or a
  CV sweep would help.
- **Single validation split.** One 80/20 unit split, not k-fold
  grouped CV, so metric variance across splits is unmeasured.
- **FD001 only.** Single operating condition and fault mode. The
  near-zero-variance sensor drops in §3 are specific to FD001 and would be
  wrong for FD002/FD004, which have six operating conditions.
- **The OOD guard is a Gaussian approximation** (Mahalanobis on raw
  sensors). It catches gross combination errors, not subtle ones.
- **`/engines` is a static snapshot** computed at startup from the test
  set; it is a dashboard demo surface, not a live fleet feed.

## 12. License

MIT — see [LICENSE](LICENSE). The C-MAPSS dataset itself is published by
NASA's Prognostics Center of Excellence and is not covered by this license.
