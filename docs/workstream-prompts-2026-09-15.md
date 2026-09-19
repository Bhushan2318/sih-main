# Ready-to-paste Claude Code prompts, one per workstream (2026-09-15)

Each block below is self-contained — paste the whole block as the first message in a
fresh Claude Code session for that person. Each one already states current repo state
(checked tonight) so nobody re-discovers what's already done. Full detail lives in
`docs/team-brief-2026-09-15-updated.md` and `CLAUDE.md` — these prompts point there
rather than repeating everything.

---

## WORKSTREAM A — Pipeline, fetch, CI (owner only)

```
Read CLAUDE.md, then docs/team-brief-2026-09-15-updated.md sections 0 and 6
("WORKSTREAM A"). I am doing Workstream A: pipeline, fetch, CI, contracts.

Current state, already verified — do not re-check or redo:
- develop and main both already carry the districts+CNN work (82af129, a24a663).
  No commits need pushing.
- .github/workflows/ already has 7 files splitting fetch/train/test concerns
  (setup.yml, fetch-daily-year.yml, backfill-reforecast.yml, refresh-data.yml,
  measure-serving-memory.yml, test-backfill-train.yml, warm-on-push.yml). Read
  all 7 before proposing a new workflow file - check it doesn't already exist
  in a different shape.
- No prod-* git tag exists anywhere (git tag -l has no prod tag). Tag current
  main before touching branch protection.

What's actually still open, in priority order:
1. Verify the Render dashboard's own auto-deploy toggle (a live setting, not
   in the repo) - warm-on-push.yml's header comment implies it may still be
   on. Two-minute check, don't assume.
2. Freeze three Pydantic contract models with tests, in one PR, BEFORE anyone
   else's parallel work depends on them: the paired-row parquet schema, the
   /api/model/status payload, the region-panel+SHAP payload. This is the
   highest-leverage single thing you can do today with 5 other people about
   to touch this code.
3. Decide branch protection + CODEOWNERS scope now that 6 people commit (it
   was deliberately descoped to "block force-push only" for solo work - put
   real protection back).
4. Confirm measure-serving-memory.yml is wired as a REQUIRED check, not just
   a workflow that exists.

Read the actual module(s) before writing anything. Report what you find - file
names, function signatures, current workflow trigger conditions - before
proposing a plan. Then wait for my go-ahead before implementing.

Constraints from CLAUDE.md: never push to main, never modify
.github/workflows/ without saying so explicitly first, don't touch the
promotion gate.
```

---

## WORKSTREAM B — Observations and truth

```
Read CLAUDE.md, then docs/team-brief-2026-09-15-updated.md sections 0, 2, and 6
("WORKSTREAM B"). I am doing Workstream B: observations and truth.

Current state, already verified — do not re-fetch anything:
- scripts/fetch_imd_district_rainfall.py exists and merges IMD gauge rainfall
  into the CDS/ERA5 observation files through the existing district weight
  table. Merged parquets already exist on disk for 2016-2019
  (imd_merged_district_observations_india_{2016..2019}.parquet).
- ERA5 is fetched via Copernicus CDS on its true native 0.25 degree grid for
  2013-2019 inclusive (era5_cds_district_observations_india_{2013..2019}.
  parquet, verified on disk). This already beats the "Open-Meteo snaps to a
  finer grid" limitation WITHOUT needing the Zarr path - do not re-fetch via
  Zarr for the variables already covered this way.
- Do not re-run any fetch for years already on disk. The archive is a public
  good (CLAUDE.md rule 7).

What's actually still open, in priority order:
1. THE REAL GAP: IMD daily rainfall accumulates 0830 IST to 0830 IST,
   attributed to the starting day. Nothing in the fetch/merge script or the
   docs currently states how this aligns with the 00 UTC-initialised
   valid_date convention (valid_date = init + (lead-1), CLAUDE.md rule 4, do
   not change this convention itself). Work out the correct alignment, WRITE
   IT DOWN in a docstring and in the limitations doc, and add a test pinning
   it. Get this wrong and every rainfall bust label is off by a day.
2. Z500 for the synoptic bust definition, via ERA5 Zarr, is the one part of
   B2 still genuinely open - lower priority than item 1.
3. B4 (IPED) stays last, as the original brief says.

Read backend/scripts/fetch_imd_district_rainfall.py and
backend/app/utils/district_observations.py before writing anything. Report
what you find before proposing a plan for the 0830 IST decision. Wait for
go-ahead before implementing.

Constraints from CLAUDE.md: no synthetic data ever, missing becomes explicit
NaN never zero, don't write a second weight table, don't rebuild the 666
districts.
```

---

## WORKSTREAM C — Features and the tabular model

```
Read CLAUDE.md, then docs/team-brief-2026-09-15-updated.md sections 0 and 6
("WORKSTREAM C"). I am doing Workstream C: features and the tabular model.

Current state, already verified: NONE of C1-C4 exist yet - a full-repo grep
tonight found zero hits for "jumpiness", "time-lagged"/"NEPS", "MJO"/"RMM", or
"district_descriptor"/"elevation_mean" anywhere in backend/. This workstream
is a genuinely blank page; the original task list is still exactly the plan.

Given ~15 days to 30 September (not the original 6 December scope), do C1
first - it is pure feature engineering off data that already exists (no new
fetch needed), is SHAP-legible, and no competing team will have it:

C1. FORECAST JUMPINESS. For each (valid_date, district, variable), how much
    the forecast for that date changed between consecutive initialisations:
    - absolute change between the two most recent cycles
    - standard deviation across the last k cycles
    - sign-flip count (did it oscillate?)
    - change relative to that district-variable's climatological jumpiness

C2 (time-lagged ensemble), C3 (MJO/MISO), C4 (district descriptors, replacing
region_id) are real and still wanted, but are December-scope unless you finish
C1 with time to spare - don't start them first.

Read backend/app/features/engineering.py and backend/app/ml/regressors.py
(feature_columns) before writing anything - this is where a new feature family
plugs in. Report what you find - column names, how existing features are
tested with hand-computed values - before proposing a plan for C1. Score the
ladder before and after adding it. Wait for go-ahead before implementing.

Constraints from CLAUDE.md: write the test first with a hand-computed expected
value, never hardcode a metric, don't rebuild the 666 districts.
```

---

## WORKSTREAM D — Verification and conformal

```
Read CLAUDE.md, then docs/team-brief-2026-09-15-updated.md sections 0 and 6
("WORKSTREAM D"). I am doing Workstream D: verification and conformal.

Current state, already verified: NONE of D1-D6 exist yet - a full-repo grep
tonight found zero hits for block-bootstrap, binormal/Z-AUC, CORP, isotonic,
relative-economic-value, SEDI, or conformal anywhere in backend/. This
workstream is a genuinely blank page; the original task list is still exactly
the plan, and it correctly has no prerequisites for D1-D5 (only D6 waits on
D1-D5, and F's screens wait on this).

Given ~15 days to 30 September, do these two first - both are pure functions
over (y_true, y_prob) arrays already available from /api/model/status, no
pipeline changes needed, and both are the first two things a NCMRWF reviewer
attacks per the brief's own ranking:

D1. BLOCK-BOOTSTRAP CONFIDENCE INTERVALS on every ladder rung. CRITICAL:
    resample BY CYCLE, not by row - rows within a cycle are heavily
    correlated, and a naive row bootstrap gives intervals far too narrow.
    1000 resamples, 95% intervals.
D2. BINORMAL Z-AUC alongside trapezoidal. Shanker, Sarkar & Mamgain (NCMRWF,
    QJRMS 2024, doi:10.1002/qj.4674): trapezoidal AUC underestimates for rare
    extreme events with small ensembles vs the binormal/Z-transform estimate.
    Cite it in the docstring.

D3 (CORP reliability), D4 (relative economic value + SEDI), D5 (logistic/EMOS/
IDR/analog ladder rungs) are real and still wanted but are December-scope
unless D1-D2 finish with time to spare. D6 (conformal) waits on all of D1-D5 -
do not start it.

This is a new module. Do NOT import the trainer - pure functions only. Read
backend/app/ml/thresholds.py and wherever the current ladder/baselines are
scored (check /api/model/status's "baselines" payload structure) before
writing anything, so your functions' inputs match what's actually available.
Report what you find before proposing a plan. Wait for go-ahead.

Constraints from CLAUDE.md: every function tested against a hand-computed or
literature value, never hardcode a metric, PR against develop.
```

---

## WORKSTREAM E — CNN

```
Read CLAUDE.md, then docs/team-brief-2026-09-15-updated.md sections 0 and 6
("WORKSTREAM E") before doing anything else.

STOP - this workstream is CLOSED, do not write CNN code. Verified tonight: the
CNN was trained on real data, scored on identical held-out rows through the
unmodified promotion gate, and lost to XGBoost by roughly 4x the seed spread
(documented in docs/roadmap-to-finale.md, "Standing constraints"). Reopening
it costs time before 30 September and changes no conclusion.

If you were assigned here, the two highest-value things you can actually do:
1. Write up the CNN-vs-XGBoost result clearly for the PPT/demo material - the
   honest "we tried a stronger model and it lost fairly, so the simpler one
   ships" result is itself good evidence the baseline ladder isn't rigged
   (this is literally attack #6 in the brief's "what a reviewer attacks
   first" list). Pull the real numbers from the training run, don't estimate.
2. Ask whoever is coordinating the team whether Workstream C or D needs a
   second person more than E needs anyone - both are 100% unstarted as of
   2026-09-15 and more urgent than anything left in E.

Do not propose reopening the CNN model itself unless you have a specific new
claim to test that the existing result doesn't already answer.
```

---

## WORKSTREAM F — Frontend, demo, boundary

```
Read CLAUDE.md, then docs/team-brief-2026-09-15-updated.md sections 0 and 6
("WORKSTREAM F"). I am doing Workstream F: frontend, demo, boundary.

Current state, already verified — check the actual frontend before assuming
either "done" or "not done":
- frontend/src/components/model/ModelPage.tsx is the "About tab" the brief
  means - the baseline ladder currently lives there, not on the dashboard's
  first screen (F2 not done).
- frontend/src/components/replay/ReplayView.tsx, ReplayFocusChart.tsx, and
  ReplayProbabilityChart.tsx already exist - a disaster-replay feature (F5) is
  at least partially built. Check the running frontend for whether it covers
  Ockhi/Chennai 2015/Kerala 2018 specifically and whether it shows model
  failures, not just successes, before rebuilding anything.
- CLAUDE.md already notes the aggregation-geometry vs display-geometry split
  is recognized (F1's core concern) - check docs/ for whether the actual
  legal/sourcing research (Survey of India requirement, GADM licence,
  datameet alternative) was written up, or whether that's still open.
- F3 (economic-value slider) and F4 (CORP diagram) are correctly blocked on
  Workstream D, which is 100% unstarted as of tonight - do not build these
  against real data yet; mock against the frozen contract if you want to get
  ahead.

Given ~15 days to 30 September, do F2 first - it's cheap, needs nothing from
anyone else, and per the brief's own words is "the strongest evidence in the
project," currently three clicks deep:

F2. Move the baseline ladder to the dashboard's first screen. Highlight the
    lead-day-only rung's NEGATIVE skill specifically - that number is what
    proves the model didn't just rediscover "day 10 is worse than day 1."

Read frontend/src/components/model/ModelPage.tsx and whatever the dashboard's
current first-screen component is before writing anything. Report what you
find before proposing a plan. Wait for go-ahead.

Constraints from CLAUDE.md: never hardcode a metric, read from
/api/model/status; test against fixtures for missing/null/API-down states;
watch the bundle size (district TopoJSON is already 374 KB).
```
