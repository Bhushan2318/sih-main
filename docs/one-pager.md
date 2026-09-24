# Sanket — संकेत, "signal"

**Predicting where tomorrow's weather forecast will be wrong, and saying why.**

Smart India Hackathon 2026 · Problem Statement 26079 · NCMRWF, Ministry of Earth Sciences
**Live: https://sanket-a0dd.onrender.com**

> **Open the site and click _Replay_ first.** It takes a real historical forecast cycle,
> scores it with the deployed model, and shows what the system would have told a
> forecaster on that day — before anyone knew the answer.

---

## The question it answers

Every operational centre already issues a forecast. Sanket does not try to make a better
one. It answers a different question, the one a duty forecaster actually has at 6am:

**"How likely is the forecast I am holding to be badly wrong today?"**

A *bust* is a forecast whose error lands in the tail of that variable's own historical
error distribution — the 90th percentile. Sanket predicts the probability of that,
per region, per variable, per lead day out to Day 10, and attributes each prediction to
the inputs that drove it, using SHAP.

Correction and confidence are different services. NCMRWF already corrects its forecasts
statistically, and busts still happen: correction removes the errors a model makes
*consistently*, while a bust comes from the particular weather pattern of that day, which
is not consistent and so is not corrected away. Knowing when to distrust a forecast is
what decides whether a warning goes out.

## How well it works

**The live figures are on the site's About tab, read from the deployed model at render
time.** They are deliberately not repeated here: a number copied into a document is
correct until the next retrain and quietly wrong afterwards, which is the exact failure
this project claims not to have. The About page also carries the baseline ladder —
climatology, lead-day, ensemble-spread — scored on the same held-out rows, because
a bare ROC-AUC claims nothing without "against what?".

What is stable enough to write down is the shape of the evidence:

- The model is scored only on **forecast cycles it never trained on**: it trains on daily
  reforecast cycles for 2000–2016 across all 666 districts and is tested on all of 2017 —
  one monsoon. 2018 and 2019 are in the archive and unused by the model; scoring them is
  the next evaluation.
- A lead-day-only baseline has little skill on this task. Measured on 2026-09-25, part of
  the reason is that for temperature, humidity and soil moisture most large errors are a
  district's steady bias rather than a failure on the day. The next retrain defines busts
  on bias-corrected error; on the serving data that makes busts grow with lead time as
  forecast failures should. The lead/bust correlation is published beside the table.
- Two findings from 2026-09-25 mean current scores overstate skill until that retrain: an
  input that used an observation from after issue time, and 2016–2017 rainfall truth taken
  from a one-day-late IMD file. Both are in `docs/known-issues.md`.
- The 17-year model is trained on a workstation GPU and published as a release; CI pulls
  a fresh forecast cycle every six hours; the site is served from a 512 MB instance that
  cannot train. The site reports training data and served data separately.

## Why the numbers can be trusted

- **No synthetic, mocked or placeholder data anywhere — not even as a fallback.** Where a
  number cannot be computed from real data, the interface shows an em dash **and the
  reason**. This is enforced in the product, not just claimed here.
- **Real data end to end.** NOAA GEFS (reforecast for training, operational feed live) and
  ERA5 reanalysis for verification, across all 666 districts.
- **Leakage is tested for, and what the tests missed is written down.** Bust thresholds
  are fitted on the training split only, out-of-fold folds are grouped by forecast cycle
  so no cycle spans a fold, and no observed day appears on both sides of the train/test
  split — each a test in the suite. An audit on 2026-09-25 still found one leaking input;
  it is documented and removed in the next retrain.
- **The ground truth's own uncertainty is measured.** Measured across ERA5 vs MERRA-2
  over **21,492** paired city-days: **24–43%** of the bust threshold, depending on the
  variable — rainfall 24%, humidity 36%, wind 38%, temperature 43%. Stated up front
  rather than waiting to be asked.

  Backed by `docs/analysis/verification_product_agreement.csv` (per-variable bias, MAE,
  RMSE, correlation and threshold) and the 21,492-row paired file beside it, regenerated
  by `python -m scripts.compare_verification_products`. Near-zero bias across all four
  variables means the two products agree on average; the percentages are the irreducible
  uncertainty in the label itself, which is why they are quoted relative to the bust
  threshold rather than in raw units.

## What it does not do

One held-out year so far. Some variables stop early in the reforecast archive (10 m wind
at Day 5, soil moisture at Day 3) and are not scored beyond that. 5 of 31 GEFS members, so
spread features are a noisy estimate of true ensemble spread. ERA5 precipitation is weak
over India relative to IMD gauge-based products, and rainfall carries that caveat. Bust is
defined on surface-variable error, not the synoptic Z500 criterion of Rodwell et al.
(2013) — a deliberate choice, because surface error is what reaches agriculture and
disaster response.

## How it runs

FastAPI + XGBoost · React + TypeScript · hive-partitioned Parquet. One origin, one
process. Trained on GitHub Actions, served on a 512 MB free-tier box that **cannot train**
— serving memory was cut from 947 MB to ~346 MB to fit, with scored output verified
byte-identical at every step. The live GEFS feed self-heals: when NOAA drops forecast
steps mid-pull, an S3 fallback refills them before the daily reduction, because a short
rainfall *sum* silently halves the quantity that drives most busts.

Total hosting cost: **$0**.
