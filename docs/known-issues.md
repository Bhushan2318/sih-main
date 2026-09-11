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
- **Observed soil moisture dips fractionally below zero in the two island districts.**
  Nicobar Islands and Lakshadweep, and only those, carry negative values in the CDS
  district observations: 400 of 243,090 district-days in 2017 and 364 in 2018. The most
  negative are −0.00045 and −0.00049 percentage points. Both districts are almost all sea
  at 0.25°, so their area-weighted soil moisture sits near zero and the negatives are
  float-noise sized. The observation files keep them as delivered, unclipped; whether the
  ingest preserves them has not been checked. A physical-range check that assumes ≥ 0 will
  flag them. Measured 2026-09-11.
