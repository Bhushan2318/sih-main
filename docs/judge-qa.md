# Questions a panel will ask

Short answers, each pointing at where the full answer lives. No score is written here.
Scores are read from `/api/model/status` on the day, because a number in a file is right
only until the next retrain.

**What exactly is a "bust"?**
A forecast whose absolute error lands in the tail of that variable's own historical error
distribution: the 90th percentile, computed on training data only. Sanket predicts the
probability of that, per district, per variable, for Days 1-10.
See `docs/engineering-reference.md`.

**Is this a better forecast?**
No. It does not forecast the weather. It says how likely the forecast you already hold is
to be badly wrong, and which inputs drove that. Correction removes consistent errors; a
bust comes from that day's weather pattern. See the README, "The question it answers".

**Why NOAA GEFS and not NCMRWF's NEPS-G?**
Training needs a reforecast archive: the same model re-run over many past years. GEFSv12
has 20 years of it, public domain. We found no public NEPS-G equivalent and did not obtain
NEPS-G data. Everything below the fetcher is source-agnostic.
See `docs/forecast-source-portability.md`.

**Why ERA5 and not IMD gauge data as the truth?**
ERA5 covers every variable on one grid. It is weak for rainfall over India, which is the
hardest variable and the one behind most busts. An IMD fetcher with the 0830 IST
convention exists, but it is not yet the truth the served model is scored on.
See `docs/known-issues.md`, the IMD entries.

**Why is the bust rate so high, if the threshold is the 90th percentile?**
The threshold is per variable, and an event busts if any of its variables busts, so the
rate is far above 10% even before the variables' dependence is counted. Weighting for a
10% base rate would wreck calibration.
See `docs/engineering-reference.md`, "The bust base rate".

**Why does the map sometimes get calmer at longer lead days?**
Part of it is structural. Soil moisture is scored only to Day 3 and wind only to Day 5,
because the training archive stops there. With fewer variables scored there are fewer ways
to bust. The lead-day rail marks those days. Part of it is the weather of that cycle.
See `docs/known-issues.md`, "The map can get calmer".

**Why surface variables and not a synoptic bust definition (Rodwell et al. 2013)?**
Deliberate. Surface error is what reaches agriculture and disaster response.

**How do we know it is not just learning the lead day?**
The model is compared against simple baselines on identical held-out rows. One of them is
lead day alone. The ladder for the currently served model has not been scored yet, and
the site says so rather than showing another model's table. See `docs/known-issues.md`.

**Can it be trusted? What about leakage?**
Thresholds are fitted on training data only, and whole forecast runs stay on one side of
the split. One leaking input was found by audit, written down, and removed. The served
model is trained without it. See `docs/known-issues.md`, `forecast_error_lag`.

**Why does it predict what it predicts?**
Every district panel shows SHAP attribution over the served XGBoost model.

**Can an alerting system consume it?**
`/api/alerts.cap` serves the high-risk alerts as CAP 1.2. It is marked `status=Test`,
because this is decision support, not an issued warning.

**What does it cost to run?**
$0. Training runs on GitHub Actions or a workstation, and serving runs on Render's free
tier (512 MB). The two never swap roles. See `docs/engineering-reference.md`.

**What are its limits?**
They are written down, not hidden: `docs/known-issues.md` and the About page.
