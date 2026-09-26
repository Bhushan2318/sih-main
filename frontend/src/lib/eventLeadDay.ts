/**
 * Where Replay should open an event, instead of always on Day 1.
 *
 * An event's whole point is the day it peaked - Kerala's floods on 15 Aug 2018, not the
 * 13 Aug 2018 cycle that forecast them. Left at Day 1, a judge sees the day *before* the
 * story and near-nationwide risk that has not sharpened into the event's district yet.
 *
 * `valid_date = init + (lead - 1)` (CLAUDE.md rule 4, already enforced server-side in
 * app/services/replay_cases.py and scripts/build_replay_cases.py) - so lead is derived,
 * never a second hand-written number to drift out of step with the catalogue.
 */

/** A `YYYY-MM-DD` string as a UTC-midnight instant, or null if it does not parse. Dates
 * here are calendar days, not local-time instants - parsing any other way risks an
 * off-by-one depending on the viewer's timezone. */
function parseUtcDate(iso: string): number | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);
  if (!m) return null;
  const [, y, mo, d] = m;
  const t = Date.UTC(Number(y), Number(mo) - 1, Number(d));
  return Number.isNaN(t) ? null : t;
}

const MS_PER_DAY = 24 * 60 * 60 * 1000;

/** `lead = (validDate - initDate) + 1`. Null when either date fails to parse. */
export function leadDayFromDates(
  initDate: string | null | undefined,
  validDate: string | null | undefined,
): number | null {
  if (!initDate || !validDate) return null;
  const init = parseUtcDate(initDate);
  const valid = parseUtcDate(validDate);
  if (init == null || valid == null) return null;
  return Math.round((valid - init) / MS_PER_DAY) + 1;
}

/**
 * The step index (into a `steps`/`leadDays` array ordered the way Replay renders it) an
 * event should open on, or null when there is nothing to override - not an event
 * (`peakValidDate` absent), no init date yet, or no steps to land on.
 *
 * Clamps rather than throws: a catalogue entry whose peak falls outside the days actually
 * scored for this run (or, in principle, before its own init) still lands on the nearest
 * real step instead of producing an out-of-range index.
 */
export function initialReplayStepIndex(
  initDate: string | null | undefined,
  peakValidDate: string | null | undefined,
  leadDays: readonly number[],
): number | null {
  if (!peakValidDate) return null;
  if (leadDays.length === 0) return null;
  const target = leadDayFromDates(initDate, peakValidDate);
  if (target == null) return null;

  let bestIdx = 0;
  let bestDiff = Infinity;
  leadDays.forEach((d, i) => {
    const diff = Math.abs(d - target);
    if (diff < bestDiff) {
      bestDiff = diff;
      bestIdx = i;
    }
  });
  return bestIdx;
}
