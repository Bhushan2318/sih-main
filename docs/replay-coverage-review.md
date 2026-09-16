# Replay coverage review (F5)

Written 2026-09-16. Checks two things the brief asked about directly rather than assumed:
does Replay cover Ockhi (Nov 2017) / Chennai floods (Dec 2015) / Kerala floods (Aug 2018)
specifically, and does it show model failures or only successes. Both checked against the
actual code path and the live API, not the frontend file names alone.

## Does it cover Ockhi / Chennai 2015 / Kerala 2018?

**No, and it cannot, by construction.** Two things pin this down precisely:

- `backend/app/ml/inference.available_cycles()` returns every scoreable forecast cycle
  **"newest first"** (its own docstring), sorted descending from `pd.Timestamp`.
- `backend/app/services/replay_service.list_cycles()` takes
  `inference.available_cycles()[:_MAX_CYCLES]` with `_MAX_CYCLES = 10` — the ten *most
  recent* cycles in the store, always, with no path to reach further back.

Confirmed against the live API rather than just the code:
`GET https://sanket-a0dd.onrender.com/api/replay/cycles` (fetched 2026-09-16) returns
cycles dated `2026-09-09` through `2026-09-15` — the current week, nowhere near
2015/2017/2018. This isn't a data-store gap that will fix itself later; it is the
selection logic. Even if the reforecast archive already holds a scoreable cycle from one
of those three dates, `_MAX_CYCLES` will never surface it once ten more recent cycles
exist, which is true every single day.

**What replay actually is today: a rolling window onto the current week**, useful for
"here's what the model said about yesterday" but structurally incapable of the specific
disaster-replay narrative (Ockhi/Chennai/Kerala) the brief describes. That narrative would
need either a separate, pinned "featured cycles" list independent of `_MAX_CYCLES`, or a
UI path to browse the full archive rather than the newest ten. Both are backend/data
changes (which cycles are ingested and how `list_cycles` selects among them), not a
frontend fix — Workstream F doesn't own that data.

## Does it show model failures, not just successes?

**Partially, and only if a viewer happens to click into one.** Three separate mechanisms,
checked individually:

1. **Cycle selection** (`list_cycles`'s sort key) ranks by
   `(verified, medium_range_growth, peak_bust_probability, verified_lead_days)`, all
   descending. This surfaces *dramatic, high-confidence* cycles — nothing in it measures
   or rewards being *wrong*. A cycle where the model missed badly ranks no higher than one
   where it correctly called a high-confidence bust.
2. **Region/variable auto-focus** (`_build_focus` → `_focus_variable_for`) defaults to the
   *peak-probability region*, then within that region ranks variables by
   `abs_err / bust_threshold` and picks the worst-tracking one. This part does lean toward
   showing a real miss, if the peak region happens to have one — but it's a side effect of
   "show the noisiest variable," not a "show me a failure" selector.
3. **The chart itself** (`ReplayFocusChart.tsx`) plots forecast vs. observed with a
   shaded "not a bust" band, so a genuine miss *is* visible once a viewer clicks a region —
   but nothing in the UI calls one out. `ReplayView.tsx`'s cycle dropdown shows
   `init_date`, verified/outcome status, and peak risk — never whether that peak call was
   right or wrong (`peak_region_abs_error` is computed in `_cycle_summary` and sent to the
   frontend's `ReplayCycleSummary` type but never rendered anywhere in `ReplayView.tsx`).

So: the plumbing to show a failure exists and is real (real forecast, real observation,
real error), but nothing curates toward one, and the one number that would flag a bad call
at a glance (`peak_region_abs_error`) is computed and shipped over the API and then
dropped on the floor by the UI.

## Recommendation

- Don't claim "covers Ockhi/Chennai 2015/Kerala 2018" in the demo narration or PPT as
  written today — it doesn't, and `_MAX_CYCLES` is why.
- Cheapest real improvement, and it's genuinely frontend-only: surface
  `peak_region_abs_error` (already in the API response, already typed, just unused) next
  to each cycle in the dropdown, so a viewer can tell a well-tracked cycle from a badly-
  missed one before picking it — turns "click and hope" into an actual choice between a
  success story and a failure story.
- The disaster-specific ask needs a backend decision outside Workstream F: either a
  pinned "featured historical cycles" list that bypasses `_MAX_CYCLES`, or confirmation
  that those three specific dates are even in the store's reforecast archive today. Not
  attempted here — flagging it rather than guessing at ingestion state I haven't checked.
