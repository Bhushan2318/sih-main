================================================================
SANKET — PROJECT BRIEF FOR CLAUDE CODE (UPDATED 2026-09-15)
================================================================

This replaces the version circulated earlier. Same structure, same section
numbers, so you can diff against the original. Every change below is backed by
something I actually checked in the repo tonight (file exists / grep hit / test
output / live API response) — not a guess about what "should" be done by now.

THE BIG CORRECTION: the original brief's Section 0 said "CNN exists but has
never been trained on real data" and assumed a 6-person parallel start from
zero. Neither is true anymore. Read Section 0 and Section 7 before assigning
anyone to anything — several workstreams are further along than the brief
assumes, one is finished, and the deadline that matters first is **30
September** (submit PPT + YouTube demo + working model), not the 6 December
grand finale the original brief was scoped against. That leaves **15 days**,
not 12 weeks.

================================================================
0. WHAT THIS PROJECT IS — updated current state
================================================================

Unchanged: Sanket predicts P(forecast bust) per district, per variable, Day
1-10, from GEFSv12, with SHAP attribution. Not a weather forecaster.

**There is a live, working model right now.** Checked directly against the
production API tonight:
  https://sanket-a0dd.onrender.com/api/model/status
  -> current_run_id: run_20260915T120352Z (auto-trained today, 12:19 UTC)
  -> classifier: ROC-AUC 0.805, F1 0.712, Brier 0.182, beats every baseline
     (climatology 0.5, lead_day 0.485, spread 0.558)
  -> 8 variables modelled, all beating their naive baselines on held-out test
  -> 71 regions, 1.47M rows, 349 training cycles, real IMD+ERA5+CDS data
This is served by `refresh-data.yml` on a schedule, independent of anything a
person is doing by hand. If someone on the team believes there is no working
model, they have not checked the URL above — that belief is the single most
urgent thing to correct before splitting work, because it changes what is
actually left to do.

**District build: done, as the original brief said.** 666 districts,
commits `ae1c9d3`/`db1a20c`/`9589c3d`/`2fe27d6` plus the later districts+CNN
merge (`82af129` on develop, `a24a663` on main). Do not rebuild them.

**CNN: trained on real data and lost. Workstream E is finished, not "never
trained."** Per `docs/roadmap-to-finale.md` (written 2026-09-10, standing
constraint section): scored on identical held-out rows through the unmodified
gate, lost to XGBoost by roughly 4x the seed spread. XGBoost remains served.
Reopening this costs time and changes no conclusion — do not assign anyone to
"finish the CNN" without a specific new claim to test.

**Branches/tags: A1's premise is stale.** `develop` and `main` both already
carry the districts+CNN work (`82af129`, `a24a663`) — the "13 local commits"
situation the brief describes is resolved. What is actually still missing: no
`prod-*` tag exists anywhere (`git tag -l` shows `backfill-data`,
`known-good-2026-09-02`, `data-latest`, two backup tags — no production tag).
Tag current `main` before anyone touches branch protection.

**A separate CI split already exists, just not under the brief's file names.**
`.github/workflows/` has `setup.yml`, `fetch-daily-year.yml`,
`backfill-reforecast.yml`, `refresh-data.yml`, `measure-serving-memory.yml`,
`test-backfill-train.yml`, `warm-on-push.yml` — fetch, train, and test concerns
are already dispatched separately, each with a documented reason in its own
header comment. A3 is not "not started"; it is "done in a different shape."
Read these seven files before proposing a fourth workflow file that duplicates
one of them.

================================================================
1. MODEL DECISION — SETTLED, DO NOT REOPEN
================================================================
Unchanged from the original brief. XGBoost stays served, SHAP over it is
shipped, CNN lost fairly and is closed (see Section 0). Ladder rungs below are
still unbuilt (Section 6/D) and still worth doing before 30 Sept if time
allows, because "here is our ladder, and here is what we ruled out and why" is
exactly what a reviewer asks for first.

RULED IN / RULED OUT lists: unchanged, still current, still binding.

================================================================
2. WHERE DATA COMES FROM — updated
================================================================
Forecasts (GEFS S3), IMD rainfall, MJO: unchanged, still the plan.

**ERA5: fetched via Copernicus CDS, not yet via the Zarr path the brief
specified — and that already fixes the limitation Zarr was for.** Real,
verified fetches exist for 2013 through 2019 (`era5_cds_district_observations_
india_{2013..2019}.parquet`, confirmed on disk tonight), each on the true
0.25 degree native grid — CDS serves ERA5 on its native grid directly, so the
"Open-Meteo snaps to a finer internal grid" limitation is already beaten
without touching Zarr. What Zarr would still add on top: Z500 for a second,
synoptic bust definition (B2's stated secondary goal) — that part is still
open. Do not re-fetch 2013-2019 by any path; it exists and the archive is a
public good.

**IMD rainfall: fetched and merging, one documented decision still missing.**
`scripts/fetch_imd_district_rainfall.py` exists, merges IMD gauge rainfall into
the CDS/ERA5 observation files through the same district weight table
(`app.utils.district_observations`), and merged parquets exist on disk for
2016-2019. **What the original brief required and is NOT yet done:** the
0830 IST accumulation-window decision. IMD daily rainfall accumulates 0830 IST
to 0830 IST, attributed to the starting day [CORRECTED 2026-09-17: IMD
attributes to the ENDING day - measured, see docs/known-issues.md, Data]; nothing in the script or the docs
currently states how that aligns with the 00 UTC-initialised `valid_date`
convention, and there is no test pinning it. This is a real gap, not a
formality — get it wrong and every rainfall bust label is off by a day. Do
this first if you draw Workstream B.

================================================================
3 & 4. DO / DO NOT
================================================================
Unchanged. Still the right rules. One addition earned tonight, add it:

- **Do not capture subprocess output in memory across many dispatches** (only
  relevant if anyone touches `app/ml/pooled_training.py`) — write child
  stdout/stderr to files, not `subprocess.run(capture_output=True)`. Real crash
  tonight: unbounded buffering put the parent process at 35 GB committed
  memory and took the whole machine to ~250 MB free. Not relevant to any of
  the six workstreams below directly, but relevant if this module is touched.

================================================================
5. CHANGES WE ARE MAKING, AND WHY
================================================================
Unchanged rationale for all 8. Status of each, checked tonight:
  1. IMD rainfall            — fetched, merged, 0830 IST decision still open
  2. Time-lagged ensemble    — not started (no code found)
  3. Forecast jumpiness      — not started (no code found)
  4. ERA5 native grid        — done via CDS (see Section 2); Z500 still open
  5. District descriptors    — not started (region_id still in use everywhere)
  6. Verification package    — not started (no module found)
  7. Conformal prediction    — not started, correctly blocked on #6
  8. CNN on real data        — done, lost fairly, closed (see Section 0)

================================================================
6. WORKSTREAMS — status per task, checked tonight
================================================================

--------------------------------------------------------------
WORKSTREAM A — PIPELINE AND FETCH (owner only)
--------------------------------------------------------------
A1. **STALE.** `develop` and `main` already both carry the districts+CNN work.
    Nothing to push. Re-verify `feature/cds-district-observations` (tonight's
    branch) is what still needs a PR — it has the pooled-training experiment
    work (5 real bugs fixed, all committed, none crashed cleanly yet) and nothing
    that blocks anyone else.
A2. **CLAIMED DONE per roadmap-to-finale.md — re-verify the Render dashboard
    toggle itself**, not just the repo. `warm-on-push.yml`'s own header comment
    says "Render auto-deploys on every push to main through its own git
    integration" in the present tense, which reads like auto-deploy is still
    on. That is a dashboard setting outside the repo; two minutes to check,
    do not assume either way from old notes.
A3. **DONE, different shape.** See Section 0. Read the seven existing workflow
    files before adding an eighth.
A4. **DESCOPED for solo work, needs REINSTATING now that there are 6 people.**
    `roadmap-to-finale.md` reduced this to "block force-push to main" for a
    one-person repo. With 6 people committing, put back PR-required + status
    checks + CODEOWNERS — this is exactly the scenario A4 was written for.
A5. **PARTIALLY DONE.** `measure-serving-memory.yml` exists and runs the real
    experiment on a Linux runner. Confirm it is wired as a *required* check
    (brief's ask), not just a workflow that exists and can be ignored.
A6. **NOT DONE.** No Pydantic contract models found for the paired-row schema,
    the `/api/model/status` payload, or the region-panel/SHAP payload. Given
    only 15 days and 6 people about to touch this code at once, freeze these
    three now — this is the highest-leverage single hour anyone can spend
    before parallel work starts, exactly as the original brief said.
A7. **DONE AND THEN SOME.** Not just November 2017 — `fetch-daily-year.yml`
    and `backfill-reforecast.yml` exist, and the live API confirms
    `init_date_min: 2016-01-09`, 349 training cycles, real data through 2019
    and CDS/IMD data through 2013-2019. Do not re-run any fetch.
A8. Fallback plan: moot. The fetch succeeded; 2015 Chennai floods data was not
    prioritized but 2017 Ockhi and 2018 Kerala floods data exists per F5.

**For Sep 30, Workstream A's real remaining work is: verify A2's live
setting, decide A4's scope with 6 real committers, and freeze A6's contracts.
That is an afternoon, not a workstream.**

--------------------------------------------------------------
WORKSTREAM B — OBSERVATIONS AND TRUTH
--------------------------------------------------------------
B1. **PARTIALLY DONE.** Fetch + merge scripts exist and work (confirmed
    parquets for 2016-2019 on disk). **The 0830 IST alignment decision is not
    written down anywhere and has no test.** This is the one real gap — see
    Section 2. Everything else in B1 (grid assertion, weight-table reuse,
    explicit NaN for missing) needs a quick re-check against the actual
    script, not a rebuild.
B2. **MOSTLY DONE, DIFFERENTLY.** ERA5 native-grid limitation already beaten
    via CDS (2013-2019 fetched, verified on disk tonight), not via Zarr. Z500
    for the synoptic bust definition is the one piece still genuinely open if
    someone wants it before Sep 30.
B3. **DONE.** Both ERA5/CDS and the original path coexist; nothing was
    deleted.
B4. Still optional, still after B1-B3 — correctly last in priority.

**For Sep 30: one person closes the 0830 IST decision (half a day of real
work: read the script, work out the offset, write the docstring + limitations
entry + test). Z500 and IPED can wait for December.**

--------------------------------------------------------------
WORKSTREAM C — FEATURES AND THE TABULAR MODEL
--------------------------------------------------------------
Checked tonight: **zero hits** for "jumpiness", "time-lagged"/"NEPS", "MJO"/
"RMM", or "district_descriptor"/"elevation_mean" anywhere in the backend.
**All of C1-C4 are genuinely not started.** This is the one workstream where
the original brief's task list is still exactly the plan — no corrections
needed, just assignment.

C5 (full-fetch retrain): the fetch A7 already did the hard part; the retrain
itself already runs on a schedule via `refresh-data.yml`. What C5 actually
means now is "retrain once C1-C4 land," which is what the roadmap doc already
says.

**For Sep 30, with only 15 days: C1 (jumpiness) is the single highest
value-for-effort item here — pure feature engineering off data that already
exists, no new fetch, SHAP-legible, and "no competing team will have it" per
the original brief's own case for it. If one person can take exactly one C
task before the 30th, make it C1. C2-C4 are real but should be scoped as
December work unless a second person is free.**

--------------------------------------------------------------
WORKSTREAM D — VERIFICATION AND CONFORMAL
--------------------------------------------------------------
Checked tonight: **zero hits** for block-bootstrap, binormal/Z-AUC, CORP,
isotonic, relative-economic-value, SEDI, or conformal anywhere in the backend.
**All of D1-D6 are genuinely not started**, exactly as the original brief
assumed. No corrections needed here either — just assignment, and note it
correctly has no prerequisites (D1-D5; D6 waits on D1-D5, F waits on D).

**For Sep 30: D1 (block-bootstrap CIs, by cycle) and D2 (binormal Z-AUC) are
the two a judge asks about first per Section 9's own ranking, and both are
pure functions over existing (y_true, y_prob) arrays — no new data, no
pipeline changes, safe to build against the live API's existing output
tonight. D3-D6 are real but are December work if 15 days is tight.**

--------------------------------------------------------------
WORKSTREAM E — CNN
--------------------------------------------------------------
**CLOSED. Do not assign anyone here.** See Section 0. Phase 1 and Phase 2 both
happened; the honest result (lost by ~4x seed spread, gate refused it, XGBoost
still served) is itself a legitimate thing to show a reviewer — it demonstrates
the ladder is not rigged (Section 9's own worry). If a team member has spare
time, the highest-value CNN-adjacent work is writing up *that* result clearly
for the PPT, not touching the model.

--------------------------------------------------------------
WORKSTREAM F — FRONTEND, DEMO, BOUNDARY
--------------------------------------------------------------
F1. **STATUS UNCLEAR — re-verify, do not assume done or not-done.**
    CLAUDE.md notes "the boundary depiction used for display is under review
    separately from the geometry used for aggregation" — meaning the concern
    is already recognized and the two geometries are already separated in
    principle. Whether the actual legal/sourcing research (SoI requirement,
    GADM licence, datameet alternative) was completed and written up needs a
    five-minute check of `docs/` by whoever owns this before doing any more
    map work.
F2. **NOT DONE.** `ModelPage.tsx` exists (this is the "About tab" the brief
    means) and the baseline ladder is not on the dashboard's first screen.
    Still real, still worth doing, still blocked on nothing.
F3. **NOT DONE**, correctly blocked on D4 (not started).
F4. **NOT DONE**, correctly blocked on D3 (not started).
F5. **PARTIALLY DONE.** `ReplayView.tsx`, `ReplayFocusChart.tsx`, and
    `ReplayProbabilityChart.tsx` already exist — a replay feature is built.
    Whether it currently covers exactly Ockhi/Chennai 2015/Kerala 2018, and
    whether it "shows what the model said" including failures per the
    brief's explicit instruction, needs checking against the running
    frontend, not assuming from the file names.
F6. **UNCLEAR — check `AboutPage.tsx`** for whether an attribution/citation
    section already exists there before building a new page.
F7. Correctly blocked on B1 (0830 IST decision still open).

**For Sep 30: F2 (ladder to first screen) is cheap and is literally "the
strongest evidence in the project," per the brief's own words, sitting three
clicks deep. This is probably the single best use of frontend time before the
30th given everything else in F is blocked on D, which likely will not fully
land in 15 days.**

================================================================
7. DEPENDENCIES — recalculated for 6 real people and 15 days, not 12 weeks
================================================================

The original brief's dependency graph is still logically correct. What has
changed is the deadline: **30 September** for a working model + PPT + demo
video (this is the semi-final submission gate), with 6 December being the
event itself. `docs/roadmap-to-finale.md`'s 12-week solo sequence does not fit
in 15 days even split across 6 people, because several of its "weeks" assumed
one person waiting on one thing at a time. With 6 people, most of it can run
in parallel instead of in sequence — that is the whole point of having six
workstreams.

**Recommended split for the next 15 days, given what's actually left (not
what the original brief assumed was left):**

- **Person 1 (A):** verify A2's live Render setting, freeze A6's three
  contracts (do this FIRST, day 1 — everyone else's work is safer once this
  exists), then decide A4's real scope with 6 committers.
- **Person 2 (B):** close the 0830 IST decision — the one real gap in an
  otherwise-working observation pipeline. Half a day of investigation, a
  docstring, a limitations entry, a test.
- **Person 3 (C):** build C1 (forecast jumpiness) end to end: feature, test
  with a hand-computed value, ladder score before/after.
- **Person 4 (D):** build D1 (block-bootstrap CIs, by cycle) and D2 (binormal
  Z-AUC) as pure functions, then wire them into whatever D5 baseline rungs fit
  in the remaining time.
- **Person 5 (F):** F2 first (move the ladder to the first screen — cheap,
  high-impact), then check F1/F5/F6's actual status against the live frontend
  before building anything that might already exist.
- **Person 6:** float — whichever of C2-C4, D3-D5, or the PPT/demo-video
  production itself is most behind once the other five report back after day
  1-2. Given there is no PPT or video yet mentioned anywhere in this
  conversation, seriously consider putting this person on the actual
  submission artifacts rather than more code — a working model with no PPT
  and no video does not clear the 30 September gate either.

**What can safely be skipped for the 30th and picked up again for December:**
C2-C4 (beyond C1), D3-D6, B2's Z500, B4, F3/F4 (blocked on D anyway), and
anything to do with the CNN (closed, see Workstream E).

================================================================
8. MEASURED FACTS — updated with what was verified tonight
================================================================
Original list unchanged and still accurate. Add:
- Live model right now: ROC-AUC 0.805, F1 0.712, Brier 0.182 (classifier);
  regressor R² ranges 0.25 (pressure) to 0.71 (soil moisture) — all beat their
  naive baselines. Checked directly against `/api/model/status` 2026-09-15.
- ERA5/CDS district observations exist and are verified on disk for
  2013-2019 inclusive (not just the 2016-2019 the original brief's "current
  state" implied).
- No `prod-*` git tag exists. `git tag -l` → backfill-data, data-latest,
  known-good-2026-09-02, backup/pre-reword, backup/pre-trailer-strip.

================================================================
9. WHAT A NCMRWF REVIEWER ATTACKS FIRST
================================================================
Unchanged, still the right ranking, still unanswered on items 2-5 (no CIs, no
binormal AUC, no calibration check, no decision-value curve exist yet — see
Workstream D's status above). This is exactly why D1/D2 are recommended above
as the highest-value 15-day investment alongside C1 and F2.

================================================================
10. HOW TO START THIS SESSION
================================================================
Unchanged process, one addition: **before confirming a workstream, check this
document's status tags for your tasks** — several are further along or fully
closed compared to the original brief, and starting from a stale assumption
wastes the 15 days that are actually left.
