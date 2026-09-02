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
the features that drove it via SHAP.

Correction and confidence are different services. NCMRWF already runs quantile mapping
and EMOS, and busts still happen, because correction removes *systematic* error while
busts are *flow-dependent*. Knowing when to distrust a forecast is what decides whether a
warning goes out.

## How well it works

Measured on held-out forecast cycles the model never trained on — ⟨PENDING: final figures
after the 10-year publish; live values today⟩:

| | |
|---|---|
| ROC-AUC | **0.846** |
| F1 | 0.748 |
| Brier | 0.162 |
| Held-out forecasts | 3,500 across 10 cycles |

**A score means nothing without "compared to what?"** Error grows with lead time, so a
model that knows only the lead day should already look good. It doesn't:

| baseline | Brier skill vs climatology |
|---|---|
| lead day only | ⟨PENDING⟩ |
| ensemble spread only | ⟨PENDING⟩ |
| lead + spread + season | ⟨PENDING⟩ |
| **Sanket** | **⟨PENDING⟩** |

Because a bust is defined against each variable's *own* error percentile rather than an
absolute error, the label does not simply grow with lead time — so the model's skill is
not a rediscovery of "day 10 is worse than day 1".

## Why the numbers can be trusted

- **No synthetic, mocked or placeholder data anywhere — not even as a fallback.** Where a
  number cannot be computed from real data, the interface shows an em dash **and the
  reason**. This is enforced in the product, not just claimed here.
- **Real data end to end.** NOAA GEFS (reforecast for training, operational feed live) and
  ERA5 reanalysis for verification, across 35 Indian regions.
- **Leakage is tested, not asserted.** Bust thresholds are fitted on the training split
  only, out-of-fold folds are grouped by forecast cycle so no cycle spans a fold, and no
  observed day appears on both sides of the train/test split. Each is a test in the suite,
  written so it cannot pass vacuously.
- **The ground truth's own uncertainty is measured.** Against MERRA-2 over 21,492 paired
  city-days, two leading reanalyses disagree by 24–43% of a bust threshold. Stated up
  front rather than waiting to be asked.

## What it does not do

One year of dense data plus a sampled multi-year reforecast archive — not a continuous
record. 35 city points, not full regional coverage. 5 of 31 GEFS members, so spread
features are a noisy estimate of true ensemble spread. ERA5 precipitation is weak over
India relative to IMD gauge-based products, and rainfall carries that caveat. Bust is
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
