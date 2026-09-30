# Sanket: the dashboard

React 18 + TypeScript + Vite. It talks only to the FastAPI backend, holds no data of its
own and fabricates nothing: every figure on screen comes from an API response.

```bash
npm install
cp .env.example .env          # Windows PowerShell: Copy-Item .env.example .env
npm run dev                   # http://localhost:5173 (the backend must be on :8000)
```

| Script | What it does |
|---|---|
| `npm run dev` | Development server with hot reload |
| `npm run build` | Type-check (`tsc -b`), then a production build into `dist/` |
| `npm test` | The vitest suite (jsdom and Testing Library), run once |
| `npm run lint` | ESLint |
| `npm run preview` | Serve the production build locally |

| Variable | Meaning |
|---|---|
| `VITE_API_BASE_URL` | Where the API is, e.g. `http://localhost:8000`. In production the dashboard is served by the API itself, from the same origin. |
| `VITE_WS_URL` | The live-events WebSocket, e.g. `ws://localhost:8000/ws` |
| `VITE_ENABLE_UPLOAD` | `false` hides the upload panel. The production image sets it, because the serving box refuses writes. |

## Screens

`pages/DashboardPage.tsx` is the tab shell. Its state lives in the URL
(`lib/urlState.ts`): the tab, the lead day, the district and the risk-band filter. So a
view can be shared as a link, and the back button works.

- **Operations:** the national picture for the current cycle. It has the hero ("where the
  forecast comes apart"), the KPI strip, the district map with its lead-day rail, a
  district's detail panel with its SHAP factors, and the worst-district ticker.
- **Alerts:** every district and day above the watch level, worst first, with filters and
  CSV export.
- **Model:** the served run, its held-out metrics, its training data, the per-variable bust
  thresholds, the reliability and economic-value cards, and the pipeline log.
- **Replay a real bust:** a past cycle scored as of issue time and stepped through day by
  day, with the forecast against what was observed.
- **About:** the question, the evidence and the baseline ladder, the limitations, and the
  data attribution.

## Layout

```
src/
  api/          typed fetch clients: regions, alerts, modelStatus, ensemble, replay,
                ingest, upload; types.ts mirrors backend/app/api/schemas.py by hand
  hooks/        useDashboardData (TanStack Query), useLiveSocket (WS → cache
                invalidation), useMediaQuery, useElapsedSeconds
  store/        liveStore (Zustand): socket status and the last live event
  lib/          pure, unit-tested helpers: risk bands, formatting, display names, CSV,
                chart domains, URL state, replay prefetch, economic value
  components/
    dashboard/  HeroDivergence, KpiStrip, RiskTicker, WorstDistrictsPanel,
                BaselineLadderCard, FeedFreshness
    map/        IndiaChoroplethMap, LeadDayRail, LeadDaySelector, MapLegend
    detail/     RegionDetailPanel, BustProbabilityCurve, VariableTrajectoryChart,
                ShapFactorsList
    alerts/     AlertsPage
    model/      ModelPage, BaselineLadderTable, CorpReliabilityCard,
                EconomicValueCard, MissesCard, PipelineLog
    replay/     ReplayView, ReplayCyclePicker, ReplayEventHeader, ReplayFocusChart,
                ReplayProbabilityChart
    about/      AboutPage
    upload/     UploadPanel, ColumnMappingConfirmModal
    common/     States (empty, loading, error), CopyLinkButton, TopDistrictsList
  assets/geo/   india_districts.topojson, claimed_territory.geojson
  styles.css    the whole design system, hand-written
  theme.ts      chart colours, mirroring the CSS custom properties
```

## The map

`assets/geo/india_districts.topojson` holds the 666 GADM 4.1 districts, built by
`backend/scripts/build_district_geo.py` from the same geometry the backend aggregates with.
So the map and the numbers describe identical areas. `claimed_territory.geojson` (Natural
Earth, via `build_claimed_territory_geo.py`) draws the areas India claims but does not
administer, such as Gilgit-Baltistan, Aksai Chin and the Shaksgam Valley, for which GADM
has no district polygon.
`india_states.topojson` is from the earlier state-level map and is no longer imported. See
[`docs/boundary-geometry-licensing.md`](../docs/boundary-geometry-licensing.md) before
changing any of it.

## Live updates

`useLiveSocket` opens `/ws` and, on each event, invalidates the affected TanStack Query
keys, so the data is refetched over REST. Socket payloads are never merged into UI state,
which keeps the socket and REST shapes independent. The queries also refetch on a timer,
so the dashboard stays current on hosts that cannot carry a WebSocket, such as Render's
free tier.

## The honesty rules, in UI terms

- A district the API did not score is drawn in the explicit **"No data"** grey, never in a
  risk colour, so an absent prediction cannot be read as low risk.
- With no trained model, the views show an empty state carrying the backend's own
  explanation instead of colours or charts.
- A number that cannot be computed from real data shows as an em dash with the reason,
  never as a placeholder.
- Observed values are drawn only where the forecast has verified; an unverified cycle says
  so rather than leaving a line that looks like data.

## Two conventions

- **`src/api/types.ts` mirrors `backend/app/api/schemas.py` by hand.** There is no codegen.
  Change a response shape and both files move together.
- **`theme.ts` mirrors the custom properties in `styles.css`.** Recharts needs real colour
  strings and cannot read `var(--blue)`, so the duplication is deliberate.
