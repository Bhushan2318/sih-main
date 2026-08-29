# ForecastGuard AI

AI-based forecast bust detection for medium-range (Day 1–10) weather forecasts across
Indian regions — Smart India Hackathon 2026, Problem Statement 26079 (NCMRWF / Ministry
of Earth Sciences).

FastAPI + XGBoost backend, React dashboard centered on a clickable India map. See
[`docs/plan.md`](docs/plan.md) for the full architecture, canonical data schema, and
phased build order.

**Status**: Phases 1–4 and 6 complete (Phase 5, the design pass, is still deferred).

- **Phase 1** — stack, folder scaffolding, canonical schema.
- **Phase 2** — ingestion + schema-mapping. Format-agnostic parsers (CSV/TSV/XLSX/JSON,
  NASA POWER header blocks + wide-month melt), a confidence-scored `SchemaMapper`
  (synonym + range heuristics, ambiguity/collision handling, `SourceProfile` reuse),
  point-in-polygon region resolution, and a canonicalisation pipeline writing a
  batch-partitioned parquet store with full SQLite lineage.
- **Phase 3** — ML. One XGBoost error regressor per canonical variable (target
  `|forecast − observed|`), a bust classifier at forecast-event grain trained on the
  regressors' out-of-fold predictions (leakage-safe), data-driven per-variable
  thresholds + percentile risk bands, and SHAP `top_factors`. `python -m
  app.ml.train_pipeline` does a full time-split retrain (~40 s) and writes a versioned
  `data/models/{run_id}/`; `current.json` flips only on full success.
- **Phase 4a** — FastAPI layer. `POST /api/upload` (+ `confirm-mapping`),
  `GET /api/regions`, `/api/regions/{id}`, `/api/alerts`, `/api/model/status`, and a
  `/ws` WebSocket broadcasting typed retrain events. Uploads ingest synchronously and
  kick off a full retrain as a BackgroundTask. See
  [`backend/app/api/README.md`](backend/app/api/README.md) for the contract.
- **Phase 4b** — React dashboard: clickable India choropleth with a lead-day selector,
  region detail panel (bust-probability curve, per-variable forecast-vs-observed
  trajectories, SHAP factors), alerts panel, and an upload + column-confirmation flow —
  all live over the WebSocket. Styling is deliberately plain; design is Phase 5. See
  [`frontend/README.md`](frontend/README.md).
- **Phase 6** — live ingestion. A background scheduler pulls each published NOAA GEFS
  operational cycle (0.25°, Day 1–10, the same 5 members the models were trained on) and
  refreshes observations in two tiers: near-real-time *provisional* values that fill the
  charts within hours but are never trained on, and ERA5 *final* values that supersede
  them and trigger a retrain. `GET /api/ingest/status` reports real feed freshness, and
  the dashboard shows which cycle is actually loaded rather than implying "now". Off by
  default (`LIVE_INGEST_ENABLED`). See [`backend/app/live/README.md`](backend/app/live/README.md).

Real sample data (`backend/data/samples/`, built by `backend/scripts/`) is NOAA GEFSv12
reforecast (forecast) + ERA5 (observations), now joined by live operational cycles — no
synthetic data anywhere. `pytest` (136 tests) runs against the real files.

Held-out test metrics on the 2019 sample: temperature MAE 0.68 °C (R² 0.79), pressure
0.74 hPa (R² 0.56), humidity 4.45 %RH (R² 0.64), rainfall 2.95 mm (R² 0.41); bust
classifier ROC-AUC 0.76 / F1 0.65. Small-sample — one year, 17 forecast cycles, 30 points
— so directional.

> Regressor errors dropped ~17% on 2026-08-29 when a one-day forecast/observation
> misalignment was found and fixed: lead day *k* was built from forecast hours
> ((k−1)·24, k·24] but labelled `valid_date = init + k`, so every forecast was verified
> against the following day's observation. The convention is now
> `valid_date = init + (lead − 1)`; see
> [`backend/scripts/fix_forecast_valid_date_offset.py`](backend/scripts/fix_forecast_valid_date_offset.py).

Run both halves together:

```bash
cd backend && source .venv/bin/activate
uvicorn app.main:app --reload --port 8000     # API + docs at /docs

cd frontend && npm run dev                    # dashboard at :5173
```

Next: **Phase 5** — the design pass (palette, typography, layout polish).

## Layout

- `backend/` — FastAPI app, ingestion/schema-mapping, XGBoost training + inference, SQLite + parquet storage.
- `frontend/` — React + Vite + TypeScript dashboard.

## Backend setup (once dependencies are needed, from Phase 2 onward)

```
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Frontend setup

```
cd frontend
npm install
cp .env.example .env
npm run dev
```
