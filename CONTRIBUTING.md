# Contributing to Sanket

Thanks for helping. Sanket predicts when a weather forecast is about to be badly wrong, and
people may act on what it says. So the bar here is less "does it run" than "is every number
it shows true". This page covers how to get set up, the rules that protect that, and how a
change gets merged.

## Getting set up

Follow [Quick start](README.md#quick-start) in the README: a Python 3.11 or 3.12 virtualenv
for `backend/`, Node 18+ for `frontend/`. Then check both halves:

```bash
cd backend && pip install -r requirements-dev.txt && python -m pytest -q
cd frontend && npm test
```

## The rules

These are the project's non-negotiables. A pull request that breaks one is not merged,
however good the rest of it is.

1. **Nothing is synthetic.** Every value must trace back to a real GRIB2 message or a real
   observation file; the `source_grib` / `source_msgs` columns record which. If a fetch
   fails, stop and report it. Never add placeholder, mock or example data to a production
   path. Test fixtures are derived from real files and labelled as such. Random arrays are
   allowed only in shape and plumbing tests, clearly labelled, and must never reach
   anything that produces a metric.
2. **Metrics are served, never written down.** Do not put a score in source, a README, a
   docstring, a comment or the frontend. It is right until the next retrain and quietly
   wrong afterwards. Read it from `/api/model/status`.
3. **Refuse rather than patch.** An incomplete forecast cycle is rejected, not partially
   ingested. Missing never becomes zero; use an explicit mask. A model that got worse
   does not ship.
4. **`valid_date = init + (lead - 1)`.** Day *k* is built from forecast hours
   ((k-1)·24, k·24]. Getting this wrong once verified every forecast against the next
   day's observation. Do not change it.
5. **Measure, do not estimate.** The serving box is killed at 512 MB. Anything added at
   serve time needs a real memory measurement in the pull request.
6. **Free tier only.** Render free, GitHub Actions, GitHub Releases. No paid services.
7. **The archive is a public good.** Data is pulled from NOAA's public bucket once. Any
   fetch must be idempotent, resumable and write to the canonical layout.
8. **Explainability is a feature.** The region panel's "what drove this prediction" is
   SHAP over XGBoost. A change that breaks SHAP breaks the product.
9. **Limitations are written down.** If you find a caveat, add it to
   [`docs/known-issues.md`](docs/known-issues.md) rather than working around it quietly.
10. **The promotion gate stays as it is.** `make_current` refuses a model that loses more
    than 0.05 ROC-AUC against the one being served, or scores below 0.55. Changing the
    gate or its thresholds needs a maintainer's explicit agreement.

## Licensing

Everything you contribute is released under the [MIT Licence](LICENSE).
[`REUSE.toml`](REUSE.toml) covers new files automatically. Third-party material is
different: data, geometry or code you did not write needs its own annotation in
`REUSE.toml`, and the full text of its licence in `LICENSES/`. Check with:

```bash
pip install reuse && reuse lint
```

If you add or upgrade a dependency, regenerate the notices with
`python backend/scripts/gen_third_party_notices.py` (its docstring lists what it needs).

## Why a green test suite is not enough here

This is a data pipeline. It fails on volume, shape and duration, and a handful of test
rows cannot show those failures. Several bugs in this project passed every test and were
wrong at real scale. For anything that touches data, include evidence from a run at real
scale in your pull request: row counts, memory use, timings, or the printed metric. Treat
the suite as a check on plumbing, not on correctness.

## Making a change

1. Branch from `main`. Never push to `main` directly.
2. Read the module you are changing, and grep for the names you use; do not guess an API.
3. Write the test first and watch it fail.
4. Open a pull request using the template. CI runs the backend suite on Windows and Linux
   (Python 3.11 and 3.12), trains on the sample data, boots the API and builds the
   frontend.
5. Say so explicitly if you touch anything in `.github/workflows/`.
6. Keep `frontend/src/api/types.ts` in step with `backend/app/api/schemas.py`. They are
   mirrored by hand.

Deploys go through the **Refresh model and data** workflow only, never Render's "Deploy
latest commit"; see [Deployment](README.md#deployment).

## Reporting problems

- **A bug or a number that looks wrong:** open an issue. The templates ask for what makes
  it reproducible: district, variable, lead day and forecast cycle.
- **A security issue:** see [SECURITY.md](SECURITY.md). Please do not open a public issue.

Everyone taking part agrees to the [Code of Conduct](CODE_OF_CONDUCT.md).
