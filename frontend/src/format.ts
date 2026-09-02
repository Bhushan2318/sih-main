/**
 * One readable IST stamp, used wherever the API hands back a raw ISO timestamp.
 *
 * The pipeline is UTC end to end and the store writes UTC, but a visitor is not a duty
 * forecaster: "trained 12:28 UTC" reads like a bug to anyone who does not think in Zulu
 * time. So an *instant* - when a run trained, when alerts were generated, when the
 * pipeline last ran - is converted to IST and labelled, which is lossless and unambiguous.
 *
 * What is NOT converted, anywhere, is a forecast *date*: `init_date`, `valid_date`, and
 * the NNZ cycle label. Those are not instants, they are identifiers. A GEFS 00Z cycle is
 * called 00Z in every timezone on earth, and a Day-3 forecast valid on 2026-09-02 covers
 * the UTC day - shifting that label by +5:30 makes a forecast point at the wrong day near
 * midnight. That is a real bug this project has already paid for once (the valid_date
 * off-by-one, fixed 2026-08-29). Those render as bare dates with no zone suffix, so there
 * is no "UTC" on screen to confuse anyone in the first place.
 */
export function stamp(iso: string | null | undefined): string {
  const d = parseUtc(iso);
  if (!d) return "—";
  return `${istParts(d)} IST`;
}

/** Same instant, date only - for places already tight on width. */
export function stampShort(iso: string | null | undefined): string {
  const d = parseUtc(iso);
  if (!d) return "—";
  return istParts(d);
}

/** The store writes some timestamps without a zone marker; they are all UTC. */
function parseUtc(iso: string | null | undefined): Date | null {
  if (!iso) return null;
  const d = new Date(iso.endsWith("Z") ? iso : `${iso}Z`);
  return Number.isNaN(d.getTime()) ? null : d;
}

/**
 * Format in Asia/Kolkata via the platform's own tz database rather than by adding 5.5
 * hours by hand: a manual offset silently produces the wrong date around midnight, which
 * is precisely the failure mode this whole comment exists to avoid.
 */
function istParts(d: Date): string {
  const p = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Kolkata",
    year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", hour12: false,
  }).formatToParts(d).reduce<Record<string, string>>((a, x) => ((a[x.type] = x.value), a), {});
  return `${p.year}-${p.month}-${p.day} ${p.hour}:${p.minute}`;
}
