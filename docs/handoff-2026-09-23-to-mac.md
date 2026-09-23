# Handoff — Windows laptop → Mac, 2026-09-23

Training is finished. The Windows/RTX 4060 laptop is being shut down and all work
continues on the Mac. This is what moved, what did not, and what is still open.

## Nothing is stranded except one thing

| Thing | Where it is now |
|---|---|
| All pooled-training commits | `origin/main` (`ab22d70` and its four parents) |
| Baseline-ladder work | `origin/feat/pooled-baseline-ladder`, 2 commits, no PR yet |
| Live model `run_20260922T043925Z` | `serving-model` release — serving never depended on the laptop |
| Ladder training rows | `eval-events-run_20260922T043925Z` release, 789,314,717 bytes |
| Held-out events | same release, 301,698,438 bytes |
| **64 GB pooled cache** | **the laptop's disk, and nowhere else — see below** |

Both worktrees were clean: no uncommitted tracked changes, no unpushed commits.

## Finishing the baseline ladder on the Mac

The two release assets are sufficient. The 301 MB deck asset was checked and carries
every column the ladder reads — all eight `actual_err_*`, ten `spread_*`, `y_bust`,
`model_proba`, `split`, and the event keys. The 671 MB full eval-events file is *not*
needed and was never uploaded.

```
# both assets into backend/data/analysis/eval_events/, with the deck one renamed
# to run_20260922T043925Z.parquet — run_baselines keys off <run_id>.parquet
python -m scripts.run_baselines --run-id run_20260922T043925Z --write-run-artifact
```

It picks up `<run_id>_baselinefit.parquet` by name because the events carry no train
split. That one command writes **both** `baselines.json` and `misses.json`. Do not
generate `misses.json` separately — one artifact, one writer.

### Warning: the analog baseline may not finish quickly

That command was started on the laptop at 20:22 IST and was still running 54 minutes of
CPU later with no output. The bottleneck is almost certainly `AnalogBaseline`:
`KNeighborsClassifier`, `n_neighbors=50`, fitted on 13,320,000 points and queried for
2,430,900. The previous ladder ran on a 2,004,660-row train split — this is 6.6× the
scale it has ever been run at, and KNN query cost does not scale linearly. The Mac will
not be faster.

**Do not quietly subsample to make it finish.** For a k-NN analog, *less* training data
makes the baseline *worse*, so a subsampled analog row flatters our own classifier —
precisely the silent, in-our-favour drift `run_baselines`' own docstring warns about. If
it must be subsampled, the artifact has to say so and Bhushan has to be told. The other
six baselines are cheap; the expensive one is isolated and can be deferred.

## What cannot be rescued

`backend/data/_pooled_cache` — 64 GB, 18 years, 1,375,496,949 paired rows. A GitHub free
release asset caps at 2 GB and rule 6 forbids paid infrastructure, so there is nowhere to
put it. While that disk survives it is fine. If it is wiped, **no pooled retrain is
possible anywhere** without re-fetching from the NOAA bucket, which rule 7 says not to do
casually and which cost weeks.

Treat `run_20260922T043925Z` as the last pooled model this project can produce. Before
scoping any retrain, check that the cache still exists, and stop if it does not rather
than proposing a re-fetch.

## Why three Model-page cards were blank

The pooled path never wrote `baselines.json`. Three cards read it — the economic-value
curve, the CORP reliability plot, and "where it was wrong" — and `EconomicValueCard`
returns `null` rather than an empty state, so it vanished with nothing on the page saying
why. `run_20260922T043925Z` has been live in that state.

The obvious fix — rebuild the 13.3M-row train split by running all eight regressors — is
not necessary. Measured, not assumed:

- Every baseline in `ALL_BASELINES` fits from 21 of the 110 event columns. Each was
  fitted twice on the same 1,048,576 real held-out rows, full frame against stripped
  frame: identical predictions, max difference `0.000e+00` for all seven.
- None of the 21 is a regressor output — no `pred_err_*`, no `conf_*`.
- So `pv.build_event_frame` driven with an all-NaN prediction vector reproduces them
  exactly. Checked row-for-row against the known-good events built the expensive way:
  79,920 real held-out events, all 20 non-key columns bit-identical.

Cost: 2,000 cycles rebuilt in a few minutes at ~6 GB peak. See
`pooled_training.BASELINE_FIT_BASE_COLUMNS`, which is a contract and is tested as one.

## Still open

- **`baselines.json` / `misses.json` are not in the pinned `serving-model` release.** Even
  once generated, the three cards stay blank until someone republishes it — and that
  reinstalls what the live site runs. Bhushan's call, deliberately not taken.
- **Conformal `q_hat` will be `nan`** for this run. It needs a `val` split carrying
  `model_proba`, which genuinely does require the regressors over 365 cycles. Not started.
- **Artifact-parity audit: done, and bounded.** The only run-directory artifact the pooled
  path drops that the non-pooled path produces is `baselines.json` (plus `misses.json`,
  which ships with it). `jump_climatology.json` is absent from both. Nothing else is
  missing. Recorded in `docs/known-issues.md`, together with the fact that nothing asserts
  the two retrain paths produce the same artifact set.

## Site state, verified 2026-09-23

666 districts across all 10 lead days, `data_available` true on every one. ROC-AUC
0.8435, 8 variables, 0 skipped. Region detail returns a 10-lead probability curve and
distinct SHAP drivers per district. The precomputed cycle and the SHAP-summary memory fix
are both serving correctly.

The site returns **429** if polled hard — back-to-back `curl`s tripped it twice during
verification and briefly looked like an outage. Space the checks.
