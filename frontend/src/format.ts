const TIME_WITH_ZONE = /[T ]\d{2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?(Z|[+-]\d{2}(?::?\d{2})?(?::?\d{2})?)$/i;
const HAS_TIME = /[T ]\d{2}:\d{2}/;

/**
 * Parse an API timestamp without destroying an explicit UTC offset.
 *
 * The backend normally emits UTC (`Z`), but a proxy or a test fixture may emit
 * `+05:30`/`-0400`. Appending `Z` to those strings used to turn a valid instant into
 * `Invalid Date`. Values without a zone are treated as UTC, matching the API contract.
 */
export function parseIsoTimestamp(value: string | null | undefined): Date | null {
  if (typeof value !== "string") return null;
  let candidate = value.trim();
  if (!candidate) return null;

  const zoned = candidate.match(TIME_WITH_ZONE);
  if (zoned) {
    const zone = normaliseZone(zoned[1]);
    candidate = `${candidate.slice(0, -zoned[1].length)}${zone}`;
  } else if (!HAS_TIME.test(candidate)) {
    // Date-only values are calendar values from the API and are interpreted as UTC.
    candidate += "Z";
  } else {
    // A zone-less date-time is also UTC for this API.
    candidate += "Z";
  }

  // Date.parse accepts a decimal point, not all ISO producers' comma decimal mark.
  candidate = candidate.replace(
    /(\d{2}:\d{2}(?::\d{2})?),(?=\d)/,
    "$1.",
  );

  const parsed = new Date(candidate);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function normaliseZone(zone: string): string {
  const compact = zone.match(/^([+-])(\d{2})(\d{2})(?:\d{2})?$/);
  if (compact) return `${compact[1]}${compact[2]}:${compact[3]}`;
  if (/^[+-]\d{2}$/.test(zone)) return `${zone}:00`;
  return zone.toUpperCase();
}

export function stamp(iso: string | null | undefined): string {
  const d = parseIsoTimestamp(iso);
  if (!d) return "—";
  return `${istParts(d)} IST`;
}

export function stampShort(iso: string | null | undefined): string {
  const d = parseIsoTimestamp(iso);
  if (!d) return "—";
  return istParts(d);
}

function istParts(d: Date): string {
  // Never a hand-added +5:30: a manual offset gives the wrong DATE around midnight.
  const p = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Kolkata",
    year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", hour12: false,
  }).formatToParts(d).reduce<Record<string, string>>((a, x) => ((a[x.type] = x.value), a), {});
  return `${p.year}-${p.month}-${p.day} ${p.hour}:${p.minute}`;
}
