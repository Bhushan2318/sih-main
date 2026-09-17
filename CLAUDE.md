# Sanket — working agreement for Claude Code

## What this project is

Sanket predicts **P(forecast bust)** per district, per variable, Day 1–10, from
the NOAA GEFSv12 reforecast archive, with SHAP attribution for every prediction.

It does **not** make weather forecasts. It predicts when an existing forecast is
about to be badly wrong, and says why.

A **bust** is a forecast whose absolute error lands in the tail of that
variable's own historical error distribution — the 90th percentile, computed on
**training data only**.

Problem statement 26079, Smart India Hackathon 2026, set by NCMRWF (National
Centre for Medium Range Weather Forecasting, Ministry of Earth Sciences).

Live site: https://sanket-a0dd.onrender.com

## Architecture — training and serving never swap roles

```
GitHub Actions (16 GB, never serves)   Render free tier (512 MB, never trains)
  restore data artifact                  entrypoint fetches the model artifact
  pull newest GEFS cycle                 uvicorn, no training, ever
  ingest reforecast into a COPY          FastAPI + built SPA, one origin
  train -> versioned run dir             /api/*  /ws
  prove serving store untouched
  score the baseline ladder
  package -> GitHub Release  ─────────►  deploy hook, then CI warms caches
```

The serving box is **killed**, not throttled, at 512 MB. Training peaks ~2.3 GB.
That is why the split exists. Do not blur it.

Stack: FastAPI, Pydantic v2, XGBoost, scikit-learn, SHAP, pandas/NumPy/PyArrow,
Parquet + SQLite/SQLAlchemy, Shapely, React 18 + TypeScript + Vite, TanStack
Query, Zustand, Recharts, hand-written CSS. Hosting cost: $0.

---

## Non-negotiable rules

1. **Nothing is synthetic.** Every value must trace to a real GRIB2 message or a
   real observation file. Columns `source_grib` / `source_msgs` record which one.
   If a fetch fails, STOP and report. Never generate placeholder, mock, or
   example data in a production path. Test fixtures must be derived from real
   files and labelled as such. Random tensors are permitted only for shape and
   plumbing tests, clearly labelled, and must never reach anything that produces
   a metric.

2. **Metrics are served, never written down.** A number in a file is right until
   the next retrain and quietly wrong afterwards. Never hardcode a score in
   source, a README, a docstring, a comment, or the frontend. Read them from
   `/api/model/status`.

3. **Refuse rather than patch.** An incomplete cycle is rejected, not partially
   ingested. Missing never becomes zero — use an explicit mask channel. A model
   that got worse does not ship.

4. **`valid_date = init + (lead − 1)`.** Day *k* is built from forecast hours
   ((k−1)·24, k·24]. Labelling it `init + k` verified every forecast against the
   following day's observation; fixing it cut regressor error ~17%. This
   convention has already caused one bug. Do not change it.

5. **Measure, do not estimate.** Anything added at serve time must be measured
   with a real RSS measurement. The box is killed at 512 MB with ~388 MB already
   in use.

6. **Free tier only.** Render free, GitHub Actions, GitHub Releases (2 GB per
   asset). $0 hosting is part of the story. No paid infrastructure, ever.

7. **The archive is a public good.** 4.2 TB is pulled from a public NOAA bucket.
   Fetch once. Never casually re-run a fetch. Any fetch path must be idempotent,
   resumable, and write to the canonical layout.

8. **Never push to `main`.** Branch from `develop`. Open a PR. Do not modify
   anything in `.github/workflows/` without saying so explicitly.

9. **Explainability is a shipped feature.** The region panel's "what drove this
   prediction" is SHAP over XGBoost. Anything that breaks SHAP breaks the
   product.

10. **Limitations are written down.** If you discover a caveat, add it to the
    limitations doc. Do not quietly work around it.

---

## Before you write code

- **Read the module you are about to change.** Do not infer its API.
- **Grep for names.** If you need a module, function, class, or column name,
  search for it. Do not guess.
- **If a name you expect does not exist, say so and stop.** Do not create a
  parallel implementation to route around it.
- **Never invent a URL, S3 key, bucket path, or dataset name.** Fetch it, print
  what came back. If you cannot verify it, say so and stop.
- **Write the test first**, watch it fail, then implement.
- **If you are more than 30% unsure about an approach, ask** before writing 200
  lines of it.
- **If you disagree with the brief, say so before implementing it.**

## Before you claim you are done

- Run the full test suite. Paste the output.
- For any numeric claim, run the code and paste the printed number. Never state
  a figure you did not just produce.
- For any external path or dataset, print the object listing or first bytes.
- Confirm you did not touch `main`, the promotion gate, or the workflows unless
  the task said to.

---

## Data

**Forecast side.** NOAA GEFSv12 *reforecast*, public AWS S3
(`noaa-gefs-retrospective`). 2000–2019, one 00 UTC cycle per day, 5 members
(control + 4 perturbed), 0.25° global grid, 3-hourly, Day 1–10. Public domain.
Only needed GRIB2 messages are pulled, via HTTP Range requests keyed off each
file's `.idx` sidecar.

**Observation side.** Currently ERA5 via Open-Meteo Historical Weather API
(CC-BY 4.0, Copernicus C3S). Being replaced: IMD gauge-based gridded rainfall at
native 0.25° for precipitation, ERA5 read as Zarr on its native grid for
everything else.

**Pipeline.** format-agnostic parsers → confidence-scored `SchemaMapper` → geo
resolution → canonical store (hive-partitioned Parquet + SQLite lineage) →
feature engineering → models.

## Geography

666 districts from GADM 4.1 admin-2, with three deliberate corrections: Ladakh
reassigned out of J&K (GADM predates 2019); disputed-territory features kept and
dissolved into their parent district (GADM files all of J&K and Ladakh under a
`Z01` GID with no `IND.*` counterpart, so filtering to "India proper" would erase
those states); Delhi relabelled (GADM calls the whole NCT "West").

Each district's value is the **area-weighted mean of every 0.25° cell its
polygon overlaps** — not a nearest-point sample, because 10 districts (Kolkata,
Hyderabad, the Puducherry enclaves, Lakshadweep) contain no grid centre at all.

Validated against the 35 city points it replaces: correlation 0.9896, median
absolute difference 0.31 °C.

**Observations go through the same weight table**, so both sides of the bust
label are area means over the same polygon. There is exactly one weight table.
Do not write a second one.

**Note:** display and aggregation now use the same GADM 4.1 file (both were
separate before the districts migration; the display-only Datameet file is
vendored but unused). The open question is not which file to use but whether
GADM's boundary *lines* match Survey of India's own published boundary data at
the line level — India's 2021 Geospatial Data Guidelines name SoI data as the
standard for any political map of India. Full research, sources, and a
Datameet comparison: `docs/boundary-geometry-licensing.md`. Do not change the
display geometry without reading that first.

## Models

- **XGBoost is the served model.** SHAP over it is a shipped feature.
- **The CNN is a challenger.** Dilated convolutions (1,2,4,8 → 31 cells ≈ 860 km
  receptive field), no downsampling, so the feature map stays at grid resolution
  and the *same* district weight table pools it — both families provably see
  identical geography (torch vs numpy agree to 7e-06). One mask channel per input
  channel. 39,361 parameters, capped by a test.
- **The ladder decides.** Both are scored on identical held-out rows.
- **PyTorch must never enter the serving dependency set.** Serving is
  onnxruntime only.

**Promotion gate.** `make_current` refuses a >0.05 ROC-AUC regression against the
incumbent, and refuses anything below 0.55 absolutely. It used to be
unconditional, which meant a model that had quietly got worse shipped itself and
the failure looked like success. **Do not modify the gate or its thresholds.**

---

## Measured facts

Do not re-derive these, and do not contradict them without new measurement.

- GEFS reforecast download: **~2.3 GB per initialisation** (5 members × 9
  variable files)
- CI runners reach **~80 MB/s** to that bucket; local is ~90 KB/s. **The fetch is
  a CI job, always.**
- District store at serving scale (666 districts, 73 cycles, 19.4 M forecast
  rows): **116 MB on disk** with zstd
- Listing forecast cycles by scanning one column of every row cost **+253 MB**.
  Read from Parquet footers: **+2 MB**. On a box killed at 512 MB with ~388 MB in
  use, that was the difference between fitting and dying.
- Gridded fields for the CNN: **3.98 MB per cycle** (ensemble mean + spread,
  int16 at a 0.01 physical quantum, high/low bytes split so zlib can work). The
  obvious encoding — stretch each plane to full int16 — is *incompressible*.
- CNN serving without PyTorch: **+51 MB, 490 ms** for all 10 lead days via
  onnxruntime, scoring one lead at a time. Batching all ten is faster (78 ms) but
  costs +163 MB.
- District geo index: ~4 MB resident, ~24 MB transient while parsing.
- Serving currently uses **388 MB of 512**.

## Known limitations

Stated plainly, and several are actively being beaten. Do not paper over any of
them.

- **Coverage is sampled, not continuous** — currently ~17 initialisations per
  year. The density work fixes this.
- **5 of 31 GEFS ensemble members.** The reforecast archive only *has* 5 daily
  (11 on Wednesdays). 31 members exist solely in the operational feed. Being
  partly addressed via a time-lagged ensemble.
- **ERA5 precipitation is weak over India** relative to gauge-based products, and
  rainfall is the hardest variable — zero-inflated, heavily skewed, and the driver
  of most busts. Being addressed via IMD gauge data.
- **Open-Meteo does not serve ERA5 on its native 0.25° grid** — points snap to a
  finer internal grid. Being addressed via Zarr on the native grid.
- **The ground truth has its own error.** Measured over ~21,000 paired city-days,
  two leading reanalysis products disagree by roughly a quarter to a half of a
  bust threshold.
- **"Bust" is defined on surface-variable error**, not the synoptic criterion of
  Rodwell et al. (2013). Deliberate: surface error is what reaches agriculture and
  disaster response.
- **The CNN has never been trained on real data.** Every claim about it is
  architectural until that changes.

## Deadline

SIH grand finale **6 December 2026**, with a semi-final before it. Internal round
cleared.
