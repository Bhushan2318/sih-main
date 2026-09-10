# Sanket — project brief

**Snapshot: 2026-09-09.** Written to be pasted into a fresh Claude conversation that has
no other context. It is a point-in-time document: figures marked *live* are served by the
running system and may already have changed; figures marked *measured* were measured once
and are stable facts about cost, not about model quality.

If you are the Claude reading this: the person asking has already built what is described
below. They want strategic and technical advice, not a rebuild. Constraints in section 8
are real and should bound any recommendation. Where you disagree with a decision already
made, say so plainly and say why.

---

## 1. What the project is

Smart India Hackathon 2026, problem statement 26079, set by NCMRWF (National Centre for
Medium Range Weather Forecasting, Ministry of Earth Sciences, India).

Sanket does **not** make weather forecasts. It predicts **when an existing forecast is
about to be badly wrong**, and says why. The question it answers is the one a duty
forecaster actually has at 6am: *"how likely is the forecast I am holding to be badly
wrong today?"*

A **bust** is defined as a forecast whose absolute error lands in the tail of that
variable's own historical error distribution — the 90th percentile, computed on training
data only. The system predicts P(bust) per region, per variable, out to Day 10, and
attributes each prediction to the inputs that drove it (SHAP).

The framing that matters: *correction and confidence are different services*. NCMRWF
already corrects forecasts statistically, and busts still happen — correction removes the
errors a model makes **consistently**, while a bust comes from the particular weather
pattern of that day. Knowing when to distrust a forecast is what decides whether a warning
goes out.

Live site: https://sanket-a0dd.onrender.com

## 2. Data and how it flows

**Forecast side.** NOAA GEFSv12 *reforecast* archive on public AWS S3 (`noaa-gefs-
retrospective`). 2000–2019, one 00 UTC cycle per day, 5 members (control + 4 perturbed),
0.25° global grid, 3-hourly steps, Day 1–10. Public domain. Only the needed GRIB2 messages
are pulled, via HTTP Range requests keyed off each file's `.idx` sidecar.

**Observation side.** ERA5 reanalysis via the Open-Meteo Historical Weather API
(CC-BY 4.0; Copernicus C3S).

**Nothing is synthetic.** Every value traces to a specific GRIB2 message; the columns
`source_grib` / `source_msgs` record which one. The test suite runs against the real files.

Pipeline: format-agnostic parsers → confidence-scored `SchemaMapper` → geo resolution →
canonical store (hive-partitioned Parquet + SQLite lineage) → feature engineering → models.

**A convention that has already caused one bug:** `valid_date = init + (lead − 1)`. Day *k*
is built from forecast hours ((k−1)·24, k·24]. Labelling it `init + k` verified every
forecast against the following day's observation; fixing it cut regressor error ~17%.

## 3. Architecture, and the one load-bearing decision

Training and serving are deliberately split and never swap roles:

```
GitHub Actions (16 GB, never serves)      Render free tier (512 MB, never trains)
  restore data artifact                     entrypoint fetches the model artifact
  pull newest GEFS cycle                    uvicorn, no training, ever
  ingest reforecast archive into a COPY     FastAPI + built SPA, one origin
  train -> versioned run dir                /api/*  /ws
  prove the serving store is untouched
  score the baseline ladder
  package -> GitHub Release  ───────────►  deploy hook, then CI warms caches
```

**Why:** the serving box is *killed*, not throttled, at 512 MB. Training peaks ~2.3 GB. So
the model carries the full archive while the box carries only what it must serve.

Stack: FastAPI, Pydantic v2, XGBoost, scikit-learn, SHAP, pandas/NumPy/PyArrow, Parquet +
SQLite/SQLAlchemy, Shapely, React 18 + TypeScript + Vite, TanStack Query, Zustand,
Recharts, hand-written CSS. Hosting cost: $0.

## 4. The evidence — the project's strongest asset

Metrics are **served, never written down**, because a number in a file is right until the
next retrain and quietly wrong afterwards. Read them at `/api/model/status`.

*Live, 2026-09-08 retrain:* 346 cycles (242 train / 52 val / 52 held out), 3,761,987 paired
rows, first training date 2000-01-09. Held-out: 16,765 events over 52 cycles.

**The baseline ladder — every model scored on identical held-out rows:**

| model | Brier skill | ROC-AUC |
|---|---|---|
| climatology | 0.0000 | 0.500 |
| lead-day only | **−0.0024** | 0.482 |
| ensemble spread | +0.0143 | 0.558 |
| lead + spread + season | +0.0175 | 0.566 |
| **Sanket bust classifier** | **+0.3600** | **0.8427** |

This is the load-bearing claim. The best cheap baseline gets 0.018 skill; the model gets
0.360; and a lead-day-only model has **negative** skill — so the result is not a
rediscovery of "day 10 is worse than day 1". It holds because a bust is defined against
each variable's own error percentile, not an absolute error threshold.

**Note: this ladder currently lives on the About tab, three clicks deep.**

## 5. Where things stand right now

Two states, and they differ:

**Production** — running commit `d0916bb`, i.e. *before* any of the work in section 6.
It retrains on schedule (last: 2026-09-08), serves 35 states, 388 MB of 512.

**Local** — 13 commits ahead, **never pushed**. 311 tests passing (was 233).
Working tree clean.

## 6. What was built in the current work stream (13 unpushed commits)

Goal: move from 35 states read at one city point each, to **666 districts** aggregated
properly, and add a **convolutional model** that reads the forecast field rather than a
summary of it.

```
ae1c9d3 geo: district as the region unit, area-weighted from the grid
1f3ca11 ml: keep the gridded fields for a convolutional model
22cbbb8 fetch: one pass over the archive, two artifacts
2121890 obs: verify against districts, the same way forecasts are aggregated
967d804 ci: refetch the whole archive at district resolution, with grids
2fe27d6 ingest: resolve to districts, take a canonical id at its word
e42dc39 serve: list cycles from Parquet footers, not every forecast row
db1a20c map: 666-district TopoJSON, 374 KB
9589c3d map: draw the districts, and let people search them
b36280f ml: a convolutional bust model that reads the field
efa813e ml: refuse to publish a model that got worse
2751db1 ui: fix what running it locally showed
dcef508 map: states at a glance, districts when you open one
```

**Districts.** 666, from GADM 4.1 admin-2, with three deliberate corrections: Ladakh
reassigned out of J&K (GADM predates 2019); disputed-territory features kept and dissolved
into their parent district (GADM files *all* of J&K and Ladakh under a `Z01` GID with no
`IND.*` counterpart — filtering to "India proper" would erase those states from the map);
and Delhi relabelled (GADM calls the whole NCT "West").

Each district's value is the **area-weighted mean of every 0.25° cell its polygon
overlaps** — not a nearest-point sample, because 10 districts (Kolkata, Hyderabad, the
Puducherry enclaves, Lakshadweep) contain no grid centre at all. Verified against the city
points it replaces: correlation 0.9896, median absolute difference 0.31 °C, and every one
of the six largest disagreements is a mountain city that is colder as a district than as a
point (Shimla −3.7 °C, Leh −3.4). Observations go through the *same* weight table, so both
sides of the bust label are area means over the same polygon.

**CNN.** Dilated convolutions (1,2,4,8 → 31 cells ≈ 860 km receptive field), **no
downsampling**, so the feature map stays at grid resolution and the *same* district weight
table pools it — both model families provably see identical geography (torch vs numpy
agree to 7e-06). A mask channel per input channel, because missing is not zero (soil
moisture stops at day 3, wind at day 5, land variables are absent over sea). 39,361
parameters, capped by a test. **Not yet trained on real data, and not wired into serving.**

**Promotion gate.** `make_current` used to be unconditional — any run that finished without
raising became the served model and fired the deploy, so a model that had quietly got worse
shipped itself and the failure looked like success. Now: refuses a >0.05 ROC-AUC regression
against the incumbent, and refuses anything below 0.55 absolutely.

## 7. Measured facts (these bound any advice)

*Measured, stable:*

- GEFS reforecast download: **~2.3 GB per initialisation** (5 members × 9 variable files).
- Local network to that S3 bucket: ~90 KB/s under load. CI runners: ~80 MB/s. **The
  re-fetch is a CI job, always.**
- District store at serving scale (666 districts, 73 cycles, 19.4 M forecast rows):
  **116 MB on disk** with zstd — far less than the ~800 MB first estimated.
- Listing forecast cycles used to cost **+253 MB** of RAM (scanning one column of every
  row). Now read from Parquet footers: **+2 MB**. On a box killed at 512 MB with ~388 MB
  in use, that was the difference between fitting and dying.
- Gridded fields for the CNN: **3.98 MB per cycle** (ensemble mean + spread, int16 at a
  0.01 physical quantum, high/low bytes split so zlib can work). The obvious encoding —
  stretch each plane to full int16 — is *incompressible*.
- CNN serving without PyTorch: **+51 MB, 490 ms** for all 10 lead days via onnxruntime,
  scoring one lead at a time. Batching all ten is faster (78 ms) but costs +163 MB.
- District geo index: ~4 MB resident, ~24 MB transient while parsing.

*Density scaling, 666 districts, calibrated against the live store (311 paired rows per
cycle-region; classifier trains at 10 events per cycle-region; regressors at 50):*

| density | cycles | transfer | grid | classifier rows | one regressor |
|---|---|---|---|---|---|
| seasonal (17/yr, today) | 340 | 0.8 TB | 1.4 GB | 0.1 M | 0.6 M |
| **daily, 5 yr (chosen)** | **1,825** | **4.2 TB** | **7.3 GB** | **12.2 M (~1.9 GB)** | **60.8 M (~7.3 GB)** |
| daily, 20 yr | 7,320 | 16.8 TB | 29.1 GB | 48.8 M (~7.8 GB) | 243.8 M (~29.3 GB) |

At the chosen 5-year daily scope **everything still trains inside a 16 GB CI runner**, so
no subsampling and no paid infrastructure are required. Only the full 20-year daily case
breaks it.

## 8. Constraints that must be respected

1. **512 MB serving box, killed not throttled.** PyTorch cannot go on it. Anything added
   at serve time must be measured, not estimated.
2. **Free tier throughout** — Render free, GitHub Actions, GitHub Releases (2 GB per
   asset). Total hosting cost $0, and that is part of the story.
3. **Deadlines:** SIH grand finale **6 December 2026**, with a semi-final before it. The
   internal round is already cleared.
4. **Explainability is a shipped feature.** The region panel's "what drove this prediction"
   is SHAP over XGBoost. Dropping XGBoost breaks it.
5. **Honesty rules the codebase already follows, and should keep following:** refuse rather
   than patch (an incomplete cycle is rejected, not partially ingested); missing never
   becomes zero; metrics are served, never hardcoded; limitations are written down.
6. **16.8 TB (or 4.2 TB) is pulled from a public-good bucket.** Fetch once; do not casually
   re-run.

## 9. What is blocked, and on what

**Everything remaining is blocked on one action: pushing 13 local commits.**
GitHub Actions can only run what is on the remote. Recommended shape: push to a *branch*
(not `main`) so Render does not auto-deploy and the live site is untouched; then dispatch
**one year** as an end-to-end proof before spending 4.2 TB.

Then, in order: CI re-fetch → retrain at district scale → train the CNN → compare both on
identical held-out rows via the promotion gate → ship whichever wins.

## 10. Open questions currently being discussed

**Density.** Decided: **every day, 2015–2019 first** (1,825 cycles, 4.2 TB), extending to
all 20 years later if it proves out.

**Model choice.** Current position: XGBoost stays the served model; the CNN competes as a
challenger; the ladder decides. Considered and ranked:

- *CatBoost* — worth one experiment, because `region_id` cardinality jumps 35 → 666 and
  ordered target encoding handles that better than XGBoost's categorical splits.
- *ConvLSTM* — now possible because daily density makes consecutive cycles adjacent days.
  Right step **after** the CNN proves the spatial signal is real, not before.
- *GNN over district adjacency* — attractive (matches output geometry, is what GraphCast
  is) but new tooling and new failure modes before December.
- *Transformers / ViT* — ruled out: 18,250 field samples is orders of magnitude too few.
- *GraphCast / Pangu / Aurora* — ruled out: they *make* forecasts, they do not predict
  when one will bust. Different task.
- *Stacking XGBoost + CNN* — best raw score, worst answer on stage; muddies the ladder and
  the SHAP panel.

**The differentiation problem (the live question).** Many teams will have the same problem
statement. The expected crowd: a weather API, an LSTM or random forest, a red-and-green
map, and a test accuracy from a random split with no temporal separation. Their weakness is
**rigour**, which is exactly this project's strength — and rigour is invisible in a
ten-minute demo.

Proposed differentiators, ranked:

1. **Replay named disasters.** Not an anonymous cycle — Cyclone Ockhi (29 Nov–6 Dec 2017),
   the Chennai floods (1–2 Dec 2015), the Kerala floods (15–17 Aug 2018). **All three fall
   inside the chosen 2015–2019 daily scope**, and none is reachable at 17 dates a year, so
   the density decision unlocks the demo. Ockhi is the canonical Indian forecast failure —
   fishermen died, policy changed, everyone in the room knows it.
2. **Potential economic value / cost–loss analysis** (Richardson 2000). Meteorology
   evaluates warnings by decision value, not AUC. Almost no team will have it.
3. **EMOS / Non-homogeneous Gaussian Regression** in the baseline ladder (Gneiting 2005) —
   the recognised operational post-processing method. Without it, "did you compare against
   EMOS?" has no answer.
4. **Move the baseline ladder to the first screen.** Currently three clicks deep.
5. *Lower priority:* verify against **IMDAA**, NCMRWF's own 12 km regional reanalysis —
   highest credibility with this specific audience, but needs portal registration and a new
   ingestion path. And the **Z500 synoptic bust definition** (Rodwell 2013) alongside the
   surface one — vocabulary rather than substance.

## 11. Known limitations, stated plainly

- **Coverage is sampled, not continuous** — currently ~17 initialisations per year. This is
  what the density work fixes.
- **5 of 31 GEFS ensemble members.** The reforecast archive only *has* 5 daily (11 on
  Wednesdays). 31 members exist solely in the operational feed, so this can improve live
  serving but can never widen the historical spread features. Partly unfixable, and saying
  so is stronger than promising a fix.
- **ERA5 precipitation is weak over India** relative to gauge-based products, and rainfall
  is the hardest variable — zero-inflated, heavily skewed, and the driver of most busts.
- **Open-Meteo does not serve ERA5 on its native 0.25° grid** — points snap to a finer
  internal grid. So a "cell value" is ERA5 sampled *at* that cell centre, not ERA5's own
  grid-box mean. Consistent in area and location with the forecasts, but not literally the
  grid-box average.
- **The ground truth has its own error.** Measured over ~21,000 paired city-days, two
  leading reanalysis products disagree by roughly a quarter to a half of a bust threshold.
- **"Bust" is defined on surface-variable error**, not the synoptic criterion of Rodwell et
  al. (2013). A deliberate choice: surface error is what reaches agriculture and disaster
  response; Z500 is what reaches meteorologists.
- **The CNN has never been trained on real data.** Every claim about it is architectural.

## 12. What good advice looks like here

Useful: what to prioritise before December given the constraints; how to make rigour
legible in ten minutes; which of the section-10 differentiators is worth the time; whether
the model strategy is right; what a NCMRWF reviewer would attack first.

Less useful: suggestions requiring paid infrastructure (possible but deliberately avoided);
model families ruled out in section 10 without new argument; anything that trades away the
honesty rules in section 8.5 for a better-looking number.
