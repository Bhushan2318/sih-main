/**
 * One readable UTC stamp, used wherever the API hands back a raw ISO timestamp.
 *
 * UTC rather than the viewer's locale because every date in this product is already UTC -
 * cycle init hours, valid dates, training runs - and silently shifting one of them into
 * local time is how a 00Z cycle starts looking like it covers the wrong day.
 */
export function stamp(iso: string | null | undefined): string {
  if (!iso) return "—";
  // the store writes some timestamps without a zone marker; they are all UTC
  const d = new Date(iso.endsWith("Z") ? iso : `${iso}Z`);
  if (Number.isNaN(d.getTime())) return "—";
  const pad = (n: number) => String(n).padStart(2, "0");
  return (
    `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}` +
    ` ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())} UTC`
  );
}
