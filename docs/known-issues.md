# Known issues

Things that are true, unfixed, and deliberate to leave. Written down so they are found
here rather than discovered live.

## Behaviour a visitor could notice

- **First load after a deploy is slow.** Caches are per-process and start empty; CI warms
  the three expensive endpoints after each publish, but a visitor arriving during that
  window pays for a cold `/api/replay/cycles` (measured 50–120 s cold against ~1 s warm on
  the deployed instance). It is slow, never wrong.
- **The opening screen is one viewport on desktop only.** On phones the KPI strip, the
  cue row and the ticker sit below the fold. The page scrolls and what is visible is
  composed; only the single-screen effect is lost.
- **Replay offers the 10 most recent cycles**, not every cycle in the store
  (`replay_service._MAX_CYCLES`). Each candidate costs one scoring pass on first call.
- **Feature and variable names render raw, in snake_case.** The SHAP panel lists
  `spread_rainfall_mm`, `conf_pressure_hpa` and
  `historical_bust_frequency_region_season`; the variable tabs and the bust-threshold
  list show `atmospheric_moisture_kgm2` and its peers. Everything around them was put
  into plain language on 2026-09-02, so these are now the densest text on the page for a
  reader without a meteorology background. Fixing it needs a display-name map rather than
  a text edit — the names arrive from the model's own feature list, not from a string in
  the component — which is why it was left rather than rushed. Worth doing Friday morning
  if there is time before the freeze; it is cosmetic and nothing depends on it.

## Operational

- **Scheduled refreshes run 3–5 hours late.** GitHub's cron is best-effort and queues on
  shared capacity for public repos; all four daily slots fire, consistently late. It costs
  nothing because the refresh is catch-up driven — each run ingests whichever of the last
  four cycles the store lacks, oldest first — and the dashboard reports which cycle is
  actually loaded rather than implying "now". Moving the schedule off the hour (`:23`)
  already reduced queueing; the remaining delay is not controllable from here.
- **Serving memory runs close to the 512 MB ceiling.** Measured 442 MB after compaction,
  against 490 MB before. The instance is killed rather than throttled if it is exceeded,
  so anything that increases what is held at serve time needs measuring, not estimating.
  `/api/health` reports the live figure because the platform paywalls its own metrics.
- **The CI serving-memory harness is noisy.** The same store measured 566 MB and 510 MB
  peak on consecutive runs, so it cannot resolve differences below roughly ±55 MB. Use it
  for large effects only; for small ones read `/api/health` on the running instance, where
  readings are stable.
- **A district-grain ingest needs the machine to itself.**
  `scripts/ingest_districts_chunked.py` splits a year by cycle so peak memory is set by the
  chunk rather than the year. That holds — but the chunk is expensive. One cycle is 33,300
  wide rows across 666 districts, melting to 209,790 long, and it peaks at **5,761 MB**.
  Measured 2026-09-11 over 225 consecutive cycles of 2017: 4,947 MB on the first, rising to
  5,761 MB by cycle 24, then flat for the 201 since — the cost of one chunk, not an
  accumulation across them. So the script will not run on a box under 16 GB, and on a 16 GB
  box it cannot share with much. Run alongside a dev uvicorn holding 1.4 GB and a browser,
  it drove a laptop into continuous paging — 7.5 GB of 8 GB swap, load average above 13 —
  and per-cycle time went from 60–90 s to 200–330 s, roughly doubling the wall time of the
  run. Nothing is wrong when this happens and no cycle is corrupted; it is simply slow.
  A death from memory pressure costs the cycle in flight rather than the run, because a
  cycle already carrying its full district set is skipped on the next start.
- **A district-scale retrain needs far more memory than the documented ~2.3 GB.** That
  figure was measured at 36 districts. The 2017 retrain at 666 districts
  (run_20260911T041126Z, 75,208,857 paired rows) measured a maximum resident set of
  11.07 GB and a peak memory footprint of 42.9 GB including swap, and ran 10,425 s on a
  16 GB laptop before failing at the event-frame step - so the classifier and SHAP stages,
  which it never reached, may raise the true peak further. A 16 GB CI runner has no swap
  to absorb that; district-scale training in CI is not assumed to fit until it has been
  measured there. Measured 2026-09-11.
- **Reading the store while an ingest writes to it killed a retrain twice over.** A reader
  lists every batch file, then opens them; an ingest adds a file every ~18 s. Writing the
  file straight to its final name let a reader open it without a footer. The first fix
  wrote to `part-0.parquet.partial` and renamed — but pyarrow's discovery skips only names
  starting with `.` or `_`, so the temp file was listed, renamed away, and the 2017 retrain
  (run_20260911T082621Z) died with `FileNotFoundError` after 2,558 s. The temp name is now
  `.part-0.parquet.partial`, and `test_store_atomic_write` reproduces the listing race.
  Concurrency is safe for reads now, but the two still compete for memory: that run's
  footprint was 13.4 GB beside the ingest's 4.9 GB. On a 16 GB box, run them in sequence.
- **A two-year retrain (2017+2018, 152.3M paired rows) died with a MemoryError before a
  single model trained** (run_20260911T193709Z, 1948 s in, on a 23.7 GB Windows box).
  `_build_paired_in_chunks` builds each chunk, appends it to a list, then at the end
  concatenates the whole list and calls `.sort_values(...).reset_index(drop=True)`. Two
  wasteful copies stacked at exactly that point: the per-chunk `frames` list stayed alive
  after being concatenated into `out` (both copies resident at once), and the separate
  `.reset_index(drop=True)` call triggers pandas' block consolidation - a `np.vstack` of
  every same-dtype column into one contiguous array - which tried to allocate 9.08 GiB it
  did not need, since the sort had already produced the row order. Fixed by clearing
  `frames` right after the concat and using `sort_values(..., ignore_index=True)` instead
  of the separate reset (`test_sort_ignore_index_matches_sort_then_reset_index` pins the
  two produce an identical frame). Removing one redundant copy is unlikely to be enough
  headroom for three or more years pooled the same way - 152M rows x 16 float32 columns is
  9 GiB for one array alone - so a third year repeating this failure is the signal to
  change the structure (feed the model per year/chunk rather than materialising one pooled
  frame), not to hunt for the next copy to delete.

  **This is not only a "training on three years" limit - it bounds any two-train/one-test
  evaluation too, once the training window is itself two years.** `_build_paired_in_chunks`
  reads and materialises the whole frame spanning `[init_date_min, init_date_max]`
  *before* `_choose_split`/`_split_by_year` ever divides it into train/val/test. So "train
  on 2017+2018, test on 2019" and "train on 2017+2018+2019 pooled" cost the same memory
  during the build - both span three years, whichever rows the split later assigns to
  train versus test. "Train on 2018, test on 2019" is a different case and does fit: that
  frame spans only two years (~152.6M rows, ~15.2 GB), the same proven scale as the
  successful 2017+2018 run - which is exactly why it was chosen once the three-year
  version was caught and killed. Measured
  bytes/row on the downcast training frame: 99.31 (built from a real 2018-07-15 slice;
  the schema has no object-dtype columns left after downcasting, all category/numeric, so
  a reorder has no cheap pointer-only path either). Three years spanned = ~228.4M rows x
  99.31 bytes ~= 22.7 GB resident for one copy of the frame alone, on a 23.7 GB box -
  attempted 2026-09-12 (train 2017+2018, test 2019, `--init-date-min 2017-01-01
  --init-date-max 2019-12-31 --test-year 2019`) and killed within seconds of launch once
  this was worked out, before it could reach the point the first 2-year attempt died at.
  Evaluating "does another year of training help" therefore needs the same structural fix
  as pooling three years for training - an external-memory/QuantileDMatrix iterator fed
  per chunk, not a full materialised frame - not merely a workaround for a bigger pool.
  The fixed retry (run_20260911T201511Z) succeeded; its own peak was not measured - only
  a >=12.7 GB reading at ~20 minutes in,
  taken before other work intervened and the process finished unwatched. Measured
  2026-09-11.
- **A promotion string comparing two runs is only a like-for-like regression check when
  both used the same kind of split.** `_choose_split` added a second split shape,
  `_split_by_year`, alongside the original chronological `_split_by_cycle` - and
  `_promotion_decision` compares held-out ROC-AUC across whichever two runs are current
  and previous, with no awareness of which shape produced either number. run_20260911T163128Z
  was scored on the chronological tail of 2017 (55 cycles, Nov-Dec); run_20260911T201511Z,
  trained on 2017 and holding out all of 2018 (365 cycles), reported "held-out ROC-AUC
  0.8348, 0.0117 below the previous run (0.8466)" - true as arithmetic, but the two 0.05-
  tolerance numbers describe different test sets, not the same 55-cycle question asked
  twice. Reading a sequence of promotion strings as if each were graded on the prior run's
  own held-out set would be a mistake. The gate's behaviour is correct as designed and is
  not being changed for this - the fix is in how these numbers get read, not in the gate.
  A cross-year `--test-year` run's ROC-AUC should be compared to another cross-year run's,
  not to a within-year run's. Measured/found 2026-09-11.
- **The classifier's easiest two months are November and December, in both years
  measured, and that is why validation scores above training.** run_20260911T201511Z
  reported train ROC-AUC 0.8400, val 0.8796, test 0.8348 - val scoring above train looks
  backwards until the calendar is checked. `_split_by_cycle`'s val slice is always the
  chronological tail before test, which for a year trained start-to-finish is Nov-Dec; per
  calendar month on the same run's eval frame:

  | month | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | 2017 (train/val) | .865 | .863 | .838 | .843 | .842 | .843 | .814 | **.802** | .810 | .855 | **.876** | **.882** |
  | 2018 (test)       | .847 | .843 | .848 | .831 | .823 | .824 | .813 | **.741** | .805 | .829 | **.870** | **.866** |

  Nov-Dec are the top two or three months of the year in both 2017 and 2018 - training on
  Jan-Oct and validating on Nov-Dec means the validation set is drawn from the model's
  easiest season, not a representative one. The low point is consistently the monsoon
  window (Jul-Sep, worst August both years - 0.802 in 2017, 0.741 in 2018), matching the
  documented fact that rainfall is the hardest, most zero-inflated variable and the driver
  of most busts. This also explains why the within-year test split (Nov-Dec 2017, ROC-AUC
  0.8466) scored higher than the cross-year test (all of 2018, 0.8348, see the entry
  above): the within-year test set was, again, the easiest two months, and the cross-year
  test averages over an entire monsoon it had to face. Not a defect in the model or the
  gate - a property of Indian monsoon seasonality that the current within-year split
  happens to validate and test on its easiest months. Measured 2026-09-11 from
  run_20260911T201511Z's own eval_events frame (`model_proba` vs `y_bust` grouped by
  calendar month); no retrain required to reproduce.
- **2019 collides with a filename the test suite already owns, and it was nearly
  overwritten tonight.** `tests/conftest.py:37` hardcodes `gefs_reforecast_india_2019.
  parquet`/`.csv` as the small 36-city legacy sample (30,600 rows, 36 columns - city/
  state/region schema) that `_ingested_slice` and every test built on it depend on. 2019
  was the very first year this project ever sampled, back when the unit was a city point;
  nobody anticipated it would later also become a real district-scale archive year fetched
  the same way 2017 and 2018 were, under the exact same filename pattern. Real district-
  scale 2019 is 12,154,500 rows, 37 columns - a different row count, a different schema,
  and a different unit of geography, so overwriting the legacy file would not have failed
  loudly; it would have made every fixture that reads it silently wrong. `_finalise` wrote
  the real 660 MB district parquet over the tracked 1.6 MB legacy one before this was
  caught; `git status` showed it modified, `git restore` put it back, and a full green
  suite (386 passed) confirmed the fixture's row/column counts were intact. No data was
  lost, but it was close.

  **Why it isn't renamed:** the obvious fix - give the legacy sample a name that doesn't
  collide with a real archive year - touches `.github/workflows/setup.yml:47`, which
  ingests `data/samples/gefs_reforecast_india_2019.parquet --confirm-all` by that literal
  path. Editing a workflow is a decision this repo's standing rule reserves for the user
  to make explicitly, not something to do unilaterally overnight. Also referenced by
  `tests/conftest.py:37`, `scripts/README.md:10`, and `scripts/fetch_era5_observations.py`'s
  docstring (2026-09-12).

  **What was done instead, so 2017/2018/future years are unaffected:** the real 2019
  archive lives at `data/samples/gefs_reforecast_india_2019_district.parquet` - an
  inconsistent name next to 2017/2018's plain ones, forced by the collision, not a
  stylistic choice. `scripts/ingest_districts_chunked.py` gained `resolve_source()` /
  `--source` to point the ingest at a non-default filename (default behaviour for every
  other year is unchanged). `scripts/fetch_gefs_reforecast_sample.py` gained `--no-csv`:
  a district-scale year's CSV twin is tens of GB (measured ~12 GB for 2019 at daily
  density) and nothing reads it - the ingest, the training pipeline and every test read
  the parquet - so it is skipped for district years now rather than written and then
  fought over. Renaming the legacy fixture properly, so 2019 does not need a special-
  cased filename at all, is a real fix and is the user's call to make; not done tonight.
- **The cross-year generalisation gap is stable across two independent year-pairs, not a
  property of one pair.** Train-2017/test-2018 (run_20260911T201511Z): held-out ROC-AUC
  0.8348. Train-2018/test-2019 (run_20260912T005532Z): held-out ROC-AUC 0.8327 - each
  scored on a full calendar year it never trained on (n=2,400,930 events, 365 cycles).
  0.0021 apart. That answers "does more data help, and is what we're seeing about the
  model or about which year got picked": a model trained on one year loses only ~0.2-0.25
  percentage points of ROC-AUC on the very next year, consistently, not by chance of
  pairing. Measured 2026-09-12.
- **More training years measurably help, scored the same way.** Four models, each scored
  on the identical 2019 (n=2,400,930, via `scripts/score_run_on_year.py`, never trained on
  2019): train-2016 (run_20260912T030448Z) 0.8302, train-2017 (run_20260911T163128Z)
  0.8231, train-2018 (run_20260912T005532Z) 0.8327, train-2017+2018
  (run_20260912T042847Z) 0.8418. Pooling two years beats every single-year model by
  0.009-0.019 ROC-AUC. This required scoring, not a single `full_retrain` call: training
  on 2017+2018 and testing on 2019 in one call would materialise all three years'
  paired frame at once (the frame is built whole before the train/val/test split ever
  divides it), which does not fit in 23.7 GB. The fix is two bounded processes - train
  with `--init-date-min 2017-01-01 --init-date-max 2018-12-31 --dry-run` (2 years, fits),
  then score the saved run separately with `score_run_on_year.py --year 2019` (1 year,
  fits) - never more than 2 years resident in either process. Measured 2026-09-12.
- **NOAA's public bucket has at least one file where the `.idx` sidecar is live but the
  `.grib2` body 404s.** `soilw_bgrnd_2008112100_p01.grib2.idx` returns 200 and lists real
  messages; the body 404s after 5 retries, while `p02`-`p04` the same cycle and `p01` the
  day before/after are all fine - a genuine gap in the archive, not a transient fault. The
  fetch script had a path for a missing idx (treat the file as absent, let
  `cycle_is_complete` refuse just that cycle) but not for a missing body despite a live
  idx, which propagated out of `pull_one_file` uncaught and crashed the whole month's CI
  job rather than refusing the one cycle. Fixed 2026-09-12 to catch it the same way. Any
  other year may hit the same pattern; it will now cost one refused cycle, not a failed
  month.
- **Swapping IMD gauge rainfall in for ERA5 did not move the pooled classifier ROC-AUC.**
  `scripts/fetch_imd_district_rainfall.py` replaces precip_mm with IMD's gauge-based
  product for 2016-2019 (mean |IMD - ERA5| = 3.3-4.0 mm/day, real and verified against the
  live archive). A model trained on 2018 with IMD rainfall, scored on 2019 with IMD
  rainfall (run_20260912T193709Z): 0.8328. The original all-ERA5 pairing
  (run_20260912T005532Z): 0.8327. Indistinguishable. This is not the fetch or the merge
  failing quietly - the store was spot-checked row for row against the IMD source file and
  matches exactly - it is that the bust label is defined self-referentially: a variable
  busts when its error exceeds *that variable's own* p90, computed on training data.
  Swapping the ground truth moves the regressor's error, the p90 threshold, and the label
  together, so a genuinely more accurate rainfall product does not automatically make the
  classification problem more separable - it can relabel which days bust without changing
  how separable busts are from non-busts. "More accurate ground truth" and "an easier
  classification problem" are different claims; this measured that they can diverge.
  Whether IMD rainfall improves anything downstream of the pooled AUC - the rainfall
  regressor's own error, or SHAP attribution on rainfall-driven busts specifically - is
  unmeasured and would need its own before/after, not read off this number. Measured
  2026-09-13.
- **The CNN loses to XGBoost by a wide, consistent margin, on two independent years.**
  Same held-out cycles both models saw (run_20260912T193709Z, 2018, IMD rainfall
  version), 3-seed CNN ensemble vs the tabular classifier: train 0.7721 vs 0.8505, val
  0.7023 vs 0.8247, test 0.7454 vs 0.8711 - XGBoost ahead by 0.08-0.13 ROC-AUC on every
  split. A separate one-off run on 2017 (run_20260911T163128Z, 1 seed) showed the same
  pattern more sharply: XGBoost 0.8466 vs CNN 0.6840. (Both models are named explicitly
  there because that pair is written in the opposite order to the three above it, where
  the CNN is quoted first - read positionally rather than by name it says the CNN won.)
  Plausible cause: ~2,550 training samples against
  43,969 parameters is little sample efficiency margin for a CNN relative to a tree
  ensemble on effectively tabular-shaped inputs. The CNN remains a challenger per
  CLAUDE.md, not a regression to fix - but two years now agree it is not currently
  winning the ladder. Measured 2026-09-13.
- **Pooling three years beats the best two-year pool, not just single years.**
  `scripts/train_pooled.py` (app.ml.pooled_training) trains any number of years through
  an XGBoost external-memory `DataIter`, never materialising more than one year at once -
  built after three real crashes on this exact code path (see the entries above and
  below) finding every remaining spot a redundant allocation could still exceed 23.7 GB.
  Train-2016+2017+2018/test-2019 (run_20260913T162755Z): held-out ROC-AUC 0.8487, cross-
  checked with `score_run_on_year.py` independently of the training run's own number.
  Beats the best two-year pool (2017+2018, 0.8418) by 0.0069, and every single-year model
  by 0.016-0.026. More data keeps helping past two years, at least up to three - whether
  it keeps helping past three is unmeasured. Training ran in 7,511 s (2h5m) by splitting
  the 8 variables across a GPU thread and a CPU thread (XGBoost's `train()` releases the
  GIL, so this is genuine concurrent hardware use, not time-slicing) - measured 2.2x
  faster per fit on GPU before wiring it in, and this run finished faster than the
  CPU-only attempts even reached their crash point. Measured 2026-09-14.
- **Getting here cost three real OOM crashes on the identical code path, each a different
  allocation at the same ceiling.** In order: (1) a redundant `.copy()` after a boolean-
  mask filter, in the new event-building pass, forcing pandas to consolidate blocks into
  a second full-sized contiguous array; (2) an equally expensive `.drop(columns=...)`
  a few lines later, doing the same block-reindex internally; (3) `pv.build_event_frame`'s
  own internal `.copy()` (shared, unmodified code, safe for a single year alone) having
  nowhere to allocate because the held-out test year was staying resident for the whole
  function on top of whichever training year was being streamed. Each was found only by
  running at real multi-year scale - none reproduced in the unit tests, which use frames
  too small for a block-consolidation copy to be expensive. Fixed by removing the first
  two outright and deferring the test year's load until after the event-building pass
  that needed the headroom. Cost: roughly 9 hours of wall-clock across the failed
  attempts before the working run above. Measured 2026-09-13/14.
- **Block-bootstrap AUC intervals are coarse below roughly 20 held-out cycles.**
  `app/ml/verification.py`'s `block_bootstrap_ci` resamples whole forecast cycles
  (deliberately - see D1 in `docs/team-brief-2026-09-15-updated.md`), but with only N
  cycles there are only on the order of N distinct resample compositions the bootstrap
  distribution can concentrate near, so the reported 95% interval is itself quantized
  rather than smooth. Demonstrated directly with a hand-constructed 2-cycle case in
  `backend/tests/test_verification.py`: the interval collapses to exactly the full
  [min, max] range of the two cycles' own metric values, not a narrower band around the
  point estimate. This is correct - it is the honest answer when independence gives you
  only 2 samples - but it means the interval width should not be read as improving
  smoothly as the held-out set grows; it improves in steps, one per additional
  independent cycle.
- **`fit_streaming` (the CNN training loop) was not bit-reproducible on CUDA with the
  same seed - fixed 2026-09-17.** E1 of `docs/team-brief-2026-09-15-updated.md` Section 6
  required a test asserting two same-seed runs produce identical weights; none existed
  before 2026-09-17. Added (`tests/test_train_cnn.py::test_same_seed_gives_bit_identical_
  weights_on_{cpu,cuda}`): the CPU version passed outright from the start. The CUDA
  version initially did not - every learned tensor differed, starting at the first
  encoder layer, not late drift. Diagnosed, not just observed: forcing
  `torch.use_deterministic_algorithms(True)` alone raised `RuntimeError`, naming cuBLAS's
  GEMM algorithm selection specifically (CUDA >= 10.2 needs `CUBLAS_WORKSPACE_CONFIG` set
  before the process starts). Two other candidates were suspected first and ruled out by
  the same experiment: cuDNN convolution backward and `DistrictPooling`'s
  `torch.sparse.mm` (a scatter-add, a classic nondeterministic-on-GPU shape) - neither
  needed a fix once cuBLAS's was applied. **Fix applied, by the user's own instruction,
  the same day:** `app/ml/train_cnn.py` now sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` at
  module import time (`os.environ.setdefault`, before torch initialises CUDA) and
  `torch.backends.cudnn.deterministic = True` / `torch.use_deterministic_algorithms(True)`
  inside `fit_streaming` (and `_fit_one`, the older CPU-only path, for consistency) right
  after the seed is set. Both CUDA and CPU tests now pass for real, not `xfail`.
  Deterministic algorithms cost some speed; not separately measured, and acceptable for a
  43,969-parameter model that already trains in minutes. This does not reopen the
  CNN-vs-XGBoost result (Section 0) - every CNN report to date used the CUDA path's
  actual output, whatever it was seeded to produce; reproducibility of the weights was
  never the same claim as correctness of
  the comparison. Diagnosed 2026-09-17.
- **The ONNX-serving memory/timing claim in `app/ml/cnn.py`'s `export_encoder` docstring
  does not reproduce exactly on this machine.** Documented: "+51 MB, 490 ms for all 10
  lead days". Re-measured 2026-09-17 (Windows, RTX 4060 laptop) with
  `scripts/measure_cnn_onnx_serving_memory.py`, which - per E4 of
  `docs/team-brief-2026-09-15-updated.md` Section 6 - exports a real-shaped model (666
  districts, 24 data channels) and measures a genuinely separate, torch-free subprocess
  scoring 10 lead days one at a time: **+57.1 MB, 65 ms**. Memory is close (12% higher -
  plausibly Windows `peak_wset` accounting for RSS differently than whatever produced the
  original number, the same caveat already on record for `measure-serving-memory.yml`'s
  cross-platform RSS readings, not evidence of a leak). Timing is not close: 65 ms is
  roughly 7.5x faster than 490 ms, not slower, so this is not a regression - but it is a
  different number on different hardware, and per the brief this gets reported rather
  than quietly adopting the original. Neither `app/ml/cnn.py`'s docstring nor CLAUDE.md's
  measured facts have been edited to match; both are one specific run's number, on
  whatever machine and onnxruntime build actually produced it, and this repo's own rule
  is not to overwrite a measured fact with a different machine's reading without saying
  so - recorded here instead. Measured 2026-09-17.
- **`fetch_gefs_reforecast_sample.py`'s CSV twin can silently stop a multi-year finalise
  partway through, with no traceback.** Discovered 2026-09-17/18 finalising 2013-2015 from
  already-cached dense parts (`--years 2013-2015 --stride 1 --resume`, no `--no-csv`):
  2013 finalised correctly (parquet + a `.csv.partial` that never got renamed), but 2014
  and 2015 were never touched at all - `_finalise` iterates years in a plain loop with no
  per-year isolation, and the district-scale CSV twin is "tens of GB" (the docstring's own
  words) per year. On a machine with ~38 GB free, 2013's CSV alone (12 GB, still `.partial`
  when the process stopped) was enough to exhaust free disk before 2014 was ever reached -
  and nothing printed an error to the captured output; the process just stopped. `--no-csv`
  exists precisely because "nothing reads it downstream" (the ingest, training, and every
  test read the parquet only) - use it for any multi-year finalise, not just as an
  optimisation. Fixed by deleting the runaway `.partial` and re-running 2014/2015 with
  `--no-csv`; both finalised correctly once disk pressure was gone. Measured 2026-09-18.
- **Subprocess isolation for the per-year event-building worker was not, by itself,
  enough - the crash it was built to prevent recurred inside it.** `_build_pooled_year_
  events_worker.py`'s own docstring already documented the 2026-09-15 (v9) crash this
  subprocess exists to avoid: `ArrayMemoryError` in pandas' groupby internals, ~586 MiB,
  on a full year's row count. It recurred anyway, twice, each ~1.8 hours into a real
  3-year pooled run (2026-09-17/18) - process isolation stopped it accumulating ACROSS
  years, but within one worker's own lifetime, `fold_models` (real XGBoost Booster
  objects, up to 24 of them for 8 variables x 3 folds) and the `job` dict holding them
  stayed resident through the whole OOF-prediction loop and were never freed before
  `build_event_frame`'s own big allocation needed its contiguous block.
  `full_retrain_pooled` itself already does exactly this cleanup
  (`del fold_models; gc.collect()`) - but only after the subprocess RETURNS, which never
  helped the worker's own peak. Fixed by freeing `fold_models`/`job` explicitly inside
  the worker, right before the call that needs the headroom, plus a retry
  (`_run_year_events_subprocess` now tries twice) since a fresh OS process is a
  genuinely different memory state, not a hope - at ~1.8 hours to reach this point,
  losing the whole job to one allocation is a far worse trade than the retry's cost.
  Per the existing caveat on this exact class of fix elsewhere in this file: Python's GC
  cannot defragment a process's native heap, only the OS reclaiming the whole process
  can - `gc.collect()` reduces the chance, it does not guarantee it, which is why the
  retry exists too. Not yet re-verified against a full real run (each attempt costs
  ~1.8 hours to even reach this point) - fixed and reasoned from the diagnosis, not
  re-measured end to end. Found 2026-09-18.

## Geography

- **Display and aggregation boundary geometry are the same GADM 4.1 file now**, not two
  separate ones — CLAUDE.md's note about them being reviewed separately predates the
  districts migration. India's 2021 Geospatial Data Guidelines (DST, 15 Feb 2021, clause
  xiii) name Survey of India boundary data as the standard for any political map of
  India; this project's district inclusion already matches India's official territorial
  position by design (see `build_district_geo.py`), but nobody has compared GADM's
  boundary *lines* against SoI's own data, and no such comparison is recorded anywhere.
  Full writeup, sources, and a Datameet alternative comparison: `docs/boundary-geometry-licensing.md`.

## Data

- **`docs/results.md` is generated, not committed.** Baselines are written into the model
  run that produced them and served from there, so the numbers cannot describe a different
  model than the one answering requests. The repo previously held a results file naming a
  superseded run.
- **Provisional observations are excluded from training** and badged in the UI. Verifying
  against a different product than the model was trained on would shift both the error and
  the bust label derived from it.
- **The CDS needs two licences accepted, not one.** ERA5 requires
  `licence-to-use-copernicus-products` *and* `cc-by`. The second does not appear in the
  portal licence list (`get_licences(scope="dataset")`) - it is only reachable as a
  `rel: license` link on the dataset itself. With just the first accepted, every request
  returns `403 required licences not accepted`, which reads like a client fault and is
  not. Measured 2026-09-10.
- **The CDS ignores `download_format: unarchived`.** A multi-variable ERA5 request comes
  back as a zip of two NetCDF files split by `stepType` - accumulations
  (`total_precipitation`) in one, instantaneous fields in the other. Opening the
  downloaded `.nc` directly fails; it is a zip. The time coordinate is `valid_time`, not
  `time`. Both verified against a real download rather than documentation.
- **ERA5 stamps an accumulation with the end of its hour.** `total_precipitation` at
  00:00 on the 2nd is rain that fell 23:00-24:00 on the 1st, so every variable is grouped
  by `valid_time - 1h`. That makes an observation day the half-open window `(t-24h, t]`,
  which is deliberately the convention the forecast side already uses for day *k* -
  `((k-1)*24, k*24]` - so both sides of a bust label describe the same interval. The
  consequence: a complete day needs stamps `01:00..00:00` of the next day, so a per-month
  request loses its final day unless it also pulls the first hour of the month after. The
  fetch does, and drops the spillover.
- **IMD dates each rain day by the END of its 0830 IST window, so IMD's date D is model
  date D-1.** IMD's gauge day accumulates 0830 IST to 0830 IST (0300-0300 UTC); Sanket's day
  is midnight to midnight UTC on both sides - `((k-1)*24, k*24]` per rule 4, `(t-24h, t]` in
  `to_daily`. The rainfall IMD labels D covers 0830 IST on D-1 to 0830 IST on D, sharing 21
  hours with model day D-1 and 3 with model day D. `fetch_imd_district_rainfall.py`
  originally joined IMD's D to model day D, following the team brief's statement that IMD
  attributes to the *starting* day. That put every IMD rainfall value one model day late: a
  forecast for day D was verified against mostly the previous day's rain. IMD's own product
  page (imdpune.gov.in, `Rainfall_25_NetCDF.html`) states no date convention at all, so this
  was settled from data, measured 2026-09-17. Spearman correlation of IMD district rainfall
  on IMD date D against independent hourly reanalysis rainfall summed to UTC day D+k, at the
  district centroid:

  | reference, period | districts | peak at k=-1 | mean at k=-1 | k=0 | k=+1 |
  |---|---|---|---|---|---|
  | ERA5 (Open-Meteo), Jun-Sep 2018 | 8 | 8 of 8 | 0.698 | 0.556 | 0.411 |
  | MERRA-2 (NASA POWER), Jun-Sep 2018 | same 8 | 8 of 8 | 0.810 | 0.621 | 0.410 |
  | ERA5, Jun-Sep 2017 | 5 | 5 of 5 | | | |
  | ERA5, Oct-Dec 2017 (NE monsoon, Ockhi) | 5 | 5 of 5 | | | |

  2018: Idukki, Ernakulam, Dakshina Kannada, Kolkata, Kamrup Metropolitan, Nagpur, Bhopal,
  Mumbai City. 2017: Idukki, Dakshina Kannada, Kolkata, Nagpur, Mumbai City; then
  Kanniyakumari, Thiruvananthapuram, Chennai, Tirunelveli, Nellore (Kanniyakumari 0.756 at
  k=-1 against 0.572 at k=0). Re-dating IMD by one day moved every 2018 peak to k=-2 against
  both references, so the test tracks dates rather than an artefact of rain persistence. Two
  independent models agreeing rules out a timing bias in either. The merge now moves IMD's
  dates back one day, reads IMD for the following year too (model 31 Dec is IMD's 1 Jan),
  and `check_attribution` refuses to write any merge whose rainfall does not correlate best
  with ERA5 at lag 0.
- **Every `imd_merged_district_observations_india_*` file is one day out, and is now
  refused.** They were written by the old join. Being gitignored, they exist only on local
  disks - `docs/workstream-prompts-2026-09-15.md` records them for 2016-2019.
  `ingest_backfill.observation_file` raises on one unless an `imd_aligned_*` file for the
  same year exists, rather than letting it keep winning file selection. Regenerate with
  `python -m scripts.fetch_imd_district_rainfall --years <year>`, then delete the old file.
  Any model trained on data ingested from an `imd_merged_*` file learned from rainfall
  labels one day out, and its scores should not be compared with ones trained after this.
- **A three-hour residual remains, deliberately.** Even with the day corrected, rain falling
  05:30-08:30 IST is counted by IMD in the neighbouring model day. It is not corrected:
  re-cutting the model day to start at 0300 UTC would match IMD exactly and push Day 10 out
  to forecast hour 243, past the 240-hour end of the GEFSv12 reforecast, losing rainfall at
  the longest lead. Because a bust is the 90th percentile of a variable's *own* error
  distribution rather than a fixed millimetre count, a uniform inflation of error lifts the
  threshold with it; what does not cancel is districts and seasons where an unusual share of
  rain falls inside that window.
- **How much the three-hour residual costs has not been measured.** ERA5 is hourly, so the
  same bust labels can be built on the 00 UTC day and on the 03 UTC day and compared - a
  single percentage of labels that differ, on real data, with no assumption about IMD
  involved. Nobody has produced that number yet. Until someone does, the paragraph above is
  an argument, not evidence.

- **EMOS's ladder position is unstable - never carry it across runs.** This entry
  first claimed EMOS "flips negative at district scale", generalised from a single
  run. A second run at the same scale contradicted it, so the claim is now narrower:

  | scored on | EMOS BSS | EMOS ROC-AUC | Sanket |
  |---|---|---|---|
  | Nov-Dec 2017, within-year, 337,950 events (`run_20260911T163128Z`) | **-0.0264** | 0.5973 | BSS 0.3713, AUC 0.8466 |
  | all of 2019, cross-year, 2,400,930 events (`run_20260912T005532Z`) | **+0.0653** | 0.6431 | BSS 0.3288, AUC 0.8327 |

  Negative in one, positive in the other, at the same district scale. The two tests
  differ in *both* the split type and the year, so neither can be credited as the
  cause - claiming either would repeat the very mistake this entry is correcting.
  What is safe to say: EMOS moves between splits; the cheapest baseline (ensemble
  spread alone) beat it on Brier in the first (BSS 0.0130); and Sanket's classifier
  clears every baseline in both by a wide margin. Re-score the whole ladder per run.
  Measured 2026-09-11 and 2026-09-12.

- **A cycle too incomplete to publish is refused, not partially ingested.** A short
  rainfall *sum* is roughly half the real accumulation, and rainfall drives most busts, so
  publishing a thin cycle would be worse than publishing nothing.

- **The 2017 store places eight districts wrongly, and five of them are absent.** Until
  2026-09-11 the schema mapper did not recognise a `region_id` column, so district rows
  were placed by coordinate. Five districts' representative points lie inside a
  neighbour, and their rows were re-keyed onto it: Imphal East -> Senapati, Dadra and
  Nagar Haveli -> Valsad, Nagaon -> Karbi Anglong, Phek -> Ukhrul, Sundargarh -> Sambalpur.
  After dedupe, Karbi Anglong, Ukhrul and Sambalpur hold the neighbour's forecasts and
  observations (Karbi Anglong read 29.19 °C against its own 27.76 °C on 2017-07-15), and
  the five do not appear. The mapper is fixed; the 2017 rows stay wrong until those ten
  districts are re-ingested, and any model trained before that saw 661 districts with
  three mislabelled. Anything ingested after the fix - 2018 onward - is unaffected.
- **Wind stops at day 5 and soil moisture at day 3, in every year.** The reforecast
  archive does not carry those messages past 120 h and 72 h (`VAR_SPEC.max_lead_h` in the
  fetch script). Measured identically in 2017 and 2018. Models see no wind features for
  days 6-10 and no soil-moisture features for days 4-10; this is the source, not a gap in
  the fetch.
- **Soil moisture has intermittent holes inside its three days.** In 2017 some cycles are
  60 values short (four districts × 3 leads × 5 members), starting 2017-11-16 and
  recurring after; in 2018 the year is 6,240 values short per lead. The ingest propagates
  the gap rather than filling it. Which districts, and why, is not yet established.
- **The parser test's collected count is not portable across machines.**
  `tests/test_parsers.py` runs one test per real file `conftest.iter_sample_files()`
  finds, which scans two roots: `backend/data/samples/` (repo, real fetch output) and,
  if it exists, `~/Desktop/data` - a personal, non-repo folder specific to whichever
  account is running the suite. On this Mac that adds 51 files no other machine, and no
  CI runner, will ever have. Moving work to a second laptop (2026-09-11) also excluded
  `backend/data/samples/parts-2018/` from the copy - 365 real per-day fetch parts,
  deliberately left out of the transfer as redundant with the single assembled
  `gefs_reforecast_india_2018.parquet` they already produced, but still counted by this
  test's directory scan on the source machine. Net effect: the Mac collected 826 tests at
  commit f36795a; the second laptop collected 384 at the same commit, and both are
  correct for what each machine actually holds. Neither figure is "the" real count - use
  pass/fail/skip ratios and diagnose any gap in the total before assuming a real problem.
- **Observed soil moisture dips fractionally below zero in the two island districts.**
  Nicobar Islands and Lakshadweep, and only those, carry negative values in the CDS
  district observations: 400 of 243,090 district-days in 2017 and 364 in 2018. The most
  negative are −0.00045 and −0.00049 percentage points. Both districts are almost all sea
  at 0.25°, so their area-weighted soil moisture sits near zero and the negatives are
  float-noise sized. The observation files keep them as delivered, unclipped; whether the
  ingest preserves them has not been checked. A physical-range check that assumes ≥ 0 will
  flag them. Measured 2026-09-11.

### Forecast jumpiness (C1), added 2026-09-16

- **The jumpiness features are empty on every row the served model trains on.** They
  compare forecasts for the same valid date from different initialisations, and a cycle
  only reaches 10 days ahead, so two cycles must be under 10 days apart to overlap. The
  backfill archive (`backfill-2000-2019.tar.gz`) holds 17 cycles a year spaced 14-35 days
  apart: 0 of 304 consecutive pairs across 2000-2018 are under 10 days, measured
  2026-09-16. Until a daily or stride-2..9 year (e.g. `fetch-daily-year.yml`'s 2017) is in
  the training store, `jump_*` carries no signal, XGBoost cannot split on it, and SHAP
  attributes nothing to it. The code path is complete; the data is not there.
- **Jumpiness does not cross a change of region scheme.** The live feed moved from
  state-level IDs (`IN-AN`) to district IDs (`IN-AN-SOUTHANDAMAN`) on the 2026-09-09
  cycle. A district's jump history therefore starts there: 2026-09-10 has two usable
  cycles, and `jump_std` / `jump_sign_flips` (which need three) are NaN until 2026-09-11.
  Deliberate - a state mean and a district mean are not the same forecast.
- **The relative jump is per district and variable, not per lead.** Jumps grow with lead
  (on the 2026-09-10 serving cycle the median |change| was 0.316 at Day 1 against 0.758 at
  Day 2), so `jump_rel_climatology` mixes lead with unusualness. The model also sees
  `lead_time_days` and can separate them; a per-lead climatology is the refinement if the
  feature earns its place once daily data exists.
- **Pooled training does not fit the jump climatology.** `full_retrain_pooled` (a
  comparison tool that never promotes) leaves `jump_rel_climatology` NaN; the other three
  jump features are cached and used. `full_retrain` fits it on the training split and
  ships it as `jump_climatology.json`.
- **Scoring an uncached cycle is slower.** Earlier cycles are read one at a time and
  reduced to ensemble means as they are read, to keep resident memory at one cycle of
  member rows. Measured on the serving store (71 regions, Windows, 3 runs each): peak
  working set 327/347/348 MB before, 348/338/355 MB after - inside run-to-run noise - and
  1.2 s -> 3.4 s per uncached cycle. Not yet measured at 666 districts on the Linux box;
  `measure-serving-memory.yml` is the check that counts.
- **Training frames carry four more float32 columns per member row.** On the real slice
  `test_paired_frame_memory.py` builds, the downcast frame went from 2.84 MB to 3.2 MB
  (+14%). Not yet measured at year scale, where the paired frame was 7.4 GB for 365
  district cycles. That test's "at least a halving" assertion was already failing on
  `develop` (5.59 -> 2.84 MB) and still fails (6.2 -> 3.2 MB).

### Found while building C1, not caused by it

- **`forecast_error_lag` is present in training and absent at serving.** It is the
  previous lead's absolute error for the same cycle and member, which needs that lead's
  observation. Measured 2026-09-16 on real cycles: non-null on 100% of lead 2-10 rows of a
  training cycle (2016-04-10), and on 0% of rows at every lead when scoring the newest
  cycle (2026-09-16), because those observations do not exist yet. The regressors learn
  from a value serving never supplies. The same observation is also later than the
  forecast's issue time, so held-out metrics may include information a live forecast could
  not have had. Not fixed here: it changes served model behaviour and needs its own
  before/after ladder.

### Time-lagged ensemble (C2) and district descriptors (C4), added 2026-09-17

- **The time-lagged ensemble inherits C1's sparse-density limitation exactly.**
  `laf_pool_std` / `laf_spread_ratio` need an earlier cycle still valid for the same
  target date, same overlap condition as `jump_std`. In the 17-cycle-a-year backfill
  archive that is almost never true, so the pool usually collapses to this cycle's own
  members (`laf_spread_ratio` = 1) - real, not a bug, and it activates on the same daily
  or dense-stride data that unlocks C1.
- **`border_distance_km` (C4) is distance to the modelled landmass's edge, not to the
  coast.** It is built from the union of all 666 district polygons
  (`scripts/build_district_descriptors.py`), which has no separate reference to tell
  coastline from international land border (Pakistan, China, Nepal, Bhutan, Bangladesh,
  Myanmar) apart. A district near the Bangladesh border and a coastal one at the same
  distance get the same value. Fixing this needs a real coastline dataset this repo does
  not have.
- **The C2/C4 before/after ladder was measured on the small CI sample, not at district
  scale.** `data/samples/gefs_reforecast_india_2019.parquet` covers 36 districts (one per
  state) and 17 cycles - real data, but neither the volume nor the district density
  region_id's replacement is meant to help with. Measured 2026-09-17, same 2019 sample,
  identical train/val/test split: ROC-AUC 0.7440 -> 0.7426, Brier 0.2012 -> 0.2037, F1
  0.6339 -> 0.6400 - inside noise for 1,050 test events, nowhere near the promotion
  gate's 0.05 ROC-AUC regression bar. This confirms no regression, not a proven gain;
  re-score once a district-grain, denser-cadence store is available.

### C4 completed with elevation_mean, added 2026-09-18

- **`elevation_mean` depends on a free, third-party public API this repo does not
  control.** `scripts/fetch_grid_elevation.py` queries `api.open-elevation.com`, an
  open-source, no-auth service whose own setup docs (fetched 2026-09-18) name its
  dataset as the CGIAR-CSI SRTM 250m resampled product
  (https://srtm.csi.cgiar.org) - not a NOAA/Copernicus source like everything else this
  project fetches. If that service goes offline or changes its dataset, the cached
  `data/geo/grid_elevation_m.parquet` (4,902 cells, fetched once) keeps working; a fresh
  fetch would not until the service is back. Verified against three known points before
  trusting it for all 4,902: Everest 8771 m (real ~8849 m), Mumbai 6 m, Delhi 214 m (real
  ~216 m).
- **250 m resolution, aggregated through a 0.25 deg (~28 km) weight table.** Elevation
  is fetched at the same grid cells GEFS/ERA5 already use, then area-weighted per
  district by the same `DistrictGridAggregator` - so `elevation_mean` is precise to the
  0.25 deg cell, not to 250 m, for any district smaller than one cell. This matches every
  other district-level value in this project (temperature, rainfall, etc. are the same
  cell-level average), so it is consistent with the rest of the feature set, not a new
  weakness specific to elevation.
- **Re-scored on the same real 2019 CI sample as C2/C4's original ladder.** ROC-AUC
  unchanged at 0.7426, Brier 0.2037 -> 0.2038, F1 0.6400 -> 0.6450 - within noise, as
  expected for one more numeric feature on a 36-district sample. Same caveat as before:
  confirms no regression, not a proven gain at district scale.

### MJO (C3), added 2026-09-18

- **This is NOAA PSL's OMI index, not BOM's canonical RMM index.** BOM's real-time text
  file (http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt) returns HTTP 403
  - "The Bureau of Meteorology website does not support web scraping" - for any
  automated request, verified 2026-09-18. `scripts/fetch_mjo_index.py` uses NOAA PSL's OMI
  instead (real, no-auth, government-hosted, daily since 1991) and transforms it to the
  RMM1/RMM2 convention per PSL's own documented mapping (RMM1 = OMI PC2, RMM2 = -OMI
  PC1), correlation > 0.93 with BOM's RMM per PSL. Anyone comparing this project's
  mjo_rmm1/mjo_rmm2 against a paper or plot that cites BOM's RMM directly should expect
  small differences, not an exact match.
- **No discrete MJO phase (1-8).** Phase-boundary conventions differ across sources and
  could not be independently verified given BOM's access block, so only the continuous
  mjo_rmm1/mjo_rmm2/mjo_amplitude are built. A model can still learn phase-like structure
  from the continuous pair directly.
- **MISO (the Indian-region monsoon analogue C3 was also named for) is not built.** No
  verified public real-time source was found for it.
- **The MJO index is global, not per-district**, and is attached by a backward as-of join
  on init_date, tolerant of up to `MJO_ASOF_TOLERANCE_DAYS` (5) days of gap - a forecast
  issued longer after the last known reading gets an unknown MJO state, not a stale one.
  `scripts/fetch_mjo_index.py` is re-run in full each time (not idempotent like the geo
  builds): the source is a small, continuously-growing daily file (~13,000 rows, under
  1 MB as of 2026-09-18), not a multi-GB archive, so a full re-fetch is the simplest
  correct way to pick up new days.
- **`load_mjo_index` degrades to all-NaN, not a crash, when the cache file is absent** -
  the same "the code path is complete; the data is not there" discipline as C1's
  jump_* columns, so a fresh checkout that has not yet run the fetch script still trains,
  just without this feature contributing anything.
- **Measured on the same real 2019 CI sample.** Before C3 (with C1/C2/C4 complete):
  ROC-AUC 0.7426, Brier 0.2038, F1 0.6450. After C3: ROC-AUC 0.7516, Brier 0.2010, F1
  0.6681 - a small, real improvement in the same direction on every metric, though on
  1,050 test events this is not distinguishable from noise on its own (the reported 95%
  CI on ROC-AUC is roughly +-0.05 wide). Not proof C3 adds real skill; consistent with
  it not hurting.
