# Forecast-source portability: why GEFS, and what NEPS-G would take

The question a reviewer from the forecasting side asks first: *why is this built on NOAA's
GEFS and not on NCMRWF's own ensemble?* This is the answer, and the map of what would change.

## Why GEFSv12

A bust predictor learns from many past forecasts paired with what then happened. That needs
a **reforecast**: the same model, frozen, re-run over many past years. NOAA's GEFSv12
reforecast covers 2000-2019, one 00 UTC cycle per day, and is public domain on AWS
(`noaa-gefs-retrospective`). The operational feed (`noaa-gefs-pds`, NOMADS) then uses the
same model for today's cycles, so training and serving see the same system.

We did not find a public reforecast archive for NCMRWF's NEPS-G, and we did not obtain NEPS-G
data. Access would go through NCMRWF / MoES. Nothing in this repo reads it.

## What is specific to GEFS

| Piece | Where | What it fixes |
|---|---|---|
| Live transport, GRIB message table, cycle timing | `backend/app/live/gefs.py` (`S3_BUCKET`, `NOMADS_URL`, `VAR_SPEC`, `latest_expected_cycle`) | Where GRIB2 comes from, which messages, when a cycle is due |
| Reforecast fetch | `backend/scripts/fetch_gefs_reforecast_sample.py` | The training archive's layout and members |
| Member list | `backend/app/config.py` (`live_gefs_members`, 5 members) | Spread features were trained on 5 members, so live must match |
| Grid | 0.25° GEFS grid | The district weight table is built for this grid |

## What is source-agnostic

| Piece | Where | Why it does not care |
|---|---|---|
| Column mapping | `FORECAST_MAPPINGS` in `backend/app/live/orchestrator.py` | A new source only has to emit the same columns (`t2m_c`, `apcp_mm`, ...) |
| Ingestion | `backend/app/ingestion/` (parsers, `SchemaMapper`, pipeline) | Format-agnostic by design |
| Contracts | `backend/app/contracts.py` | Canonical variables, ranges, member fields |
| Features and models | `backend/app/features/`, `backend/app/ml/` | Read canonical rows only |
| District pooling | `DistrictGridAggregator` (`backend/app/utils/india_districts.py`), weights built by `backend/scripts/build_district_geo.py` | Applies a polygon-overlap weight table to any field; only the table is tied to the 0.25° grid |
| Serving, SHAP, Replay | `backend/app/services/`, `backend/app/api/` | Read model outputs only |

## What a NEPS-G adapter would need

1. **A fetcher beside `gefs.py`** that writes the same canonical columns. It is the only new
   transport code.
2. **District weights for its grid.** Today's table is built for the 0.25° grid by
   `build_district_geo.py`. A new grid needs that script extended to emit a table for it,
   applied by the same `DistrictGridAggregator`. That gives one table per grid and one code
   path, never a second implementation.
3. **Members.** NEPS-G's member count differs from the 5 the spread features were trained
   on. Either subset the members to match or retrain on NEPS-G.
4. **A training archive.** Without NEPS-G hindcasts the model cannot be trained on NEPS-G
   errors. With them, the existing pipeline (`train_pooled`, the baseline ladder, the
   promotion gate) runs unchanged.
5. **A source tag on rows**, so GEFS and NEPS-G pairs are never pooled by accident.

**The one-line version:** the pipeline is model-agnostic below the fetcher. Give it NEPS-G
hindcasts and the same code trains on them, scores them against the same baselines and
explains them with the same SHAP.
