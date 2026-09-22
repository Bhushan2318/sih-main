/**
 * Today's alerts as a CSV a person can open.
 *
 * A duty forecaster or a district disaster-management officer does not want a screenshot
 * of a table, they want the rows. This is the smallest honest version of that: the same
 * alerts the page is showing, in the order the API ranked them, with the probability left
 * as a number so a spreadsheet can compute with it rather than a rendered "87%".
 *
 * Built from data already in the browser, so it costs the 512 MB box nothing.
 */
import type { Alert } from "../api/types";

const COLUMNS = [
  "region_id",
  "region_name",
  "lead_time_days",
  "valid_date",
  "bust_probability",
  "risk_band",
  "dominant_variable",
] as const;

/** Excel and Sheets execute a cell that opens with one of these. */
const FORMULA_LEAD = /^[=+\-@\t\r]/;

/** A plain date, which is all that is ever allowed into a filename. */
const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;

function cell(value: unknown): string {
  if (value == null) return "";
  let s = String(value);
  // CSV injection: this file is opened by a spreadsheet, and a leading = + - or @ is run
  // as a formula there. Prefixing an apostrophe is the standard defusing, and it is done
  // before quoting so the apostrophe ends up inside the quotes.
  if (FORMULA_LEAD.test(s)) s = `'${s}`;
  // RFC 4180: quote anything containing a comma, quote or newline, and double the quotes.
  if (/[",\r\n]/.test(s)) s = `"${s.replace(/"/g, '""')}"`;
  return s;
}

export function alertsToCsv(alerts: Alert[]): string {
  const lines = [COLUMNS.join(",")];
  for (const a of alerts) {
    lines.push(COLUMNS.map((c) => cell(a[c])).join(","));
  }
  // CRLF, which is what RFC 4180 specifies and what Excel is happiest with.
  return `${lines.join("\r\n")}\r\n`;
}

/**
 * A filename naming the cycle the rows came from.
 *
 * The date is used only if it really is a date - it reaches a download attribute, and a
 * value with a slash or a traversal in it has no business there.
 */
export function csvFilename(initDate: string | null | undefined): string {
  const d = (initDate ?? "").slice(0, 10);
  return ISO_DATE.test(d) ? `sanket-alerts-${d}.csv` : "sanket-alerts.csv";
}
