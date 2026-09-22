/**
 * The parts of the dashboard worth putting in a URL.
 *
 * Every piece of screen state used to live in `useState`, which meant nothing about this
 * site was addressable: you could not send someone a district, you could not bookmark a
 * lead day, and the browser back button did nothing at all. For a project whose demo is a
 * link in a submission, that is a real gap - a judge cannot be walked to "Chennai, day 5"
 * without being talked through four clicks.
 *
 * Pure functions over a query string. Nothing here touches `window`, so the parsing rules
 * are testable on their own and the component keeps the history calls.
 *
 * A URL is untrusted input and this one is meant to be pasted around, so every field is
 * validated rather than cast: an unknown view, a lead day of 900, a band the filter does
 * not offer and a region id that is not shaped like one all fall back to the default
 * instead of putting the app somewhere its own UI cannot leave.
 */
import type { RiskBand } from "../api/types";

export type View = "live" | "alerts" | "model" | "replay" | "about";

const VIEWS: readonly View[] = ["live", "alerts", "model", "replay", "about"];

/** The Alerts page filters on watch and bust only; "low" is a band but not a filter. */
const BANDS: readonly RiskBand[] = ["medium", "high"];

/** Day 1-10, which is the range the archive and the model both cover. */
const MIN_DAY = 1;
const MAX_DAY = 10;

/** `IN-TN-CHENNAI`, `IN-LD-LAKSHADWEEP` - the canonical ids from the geo registry. */
const REGION_ID = /^[A-Za-z0-9][A-Za-z0-9-]{0,63}$/;

export interface AppState {
  view: View;
  leadDay: number;
  region: string | null;
  band: RiskBand | undefined;
}

export const DEFAULT_STATE: AppState = {
  view: "live",
  leadDay: MIN_DAY,
  region: null,
  band: undefined,
};

export function parseAppState(search: string): AppState {
  const q = new URLSearchParams(search.startsWith("?") ? search.slice(1) : search);

  const view = q.get("view");
  const day = Number.parseInt(q.get("day") ?? "", 10);
  const region = (q.get("region") ?? "").trim();
  const band = q.get("band");

  return {
    view: VIEWS.includes(view as View) ? (view as View) : DEFAULT_STATE.view,
    leadDay: Number.isFinite(day)
      ? Math.min(MAX_DAY, Math.max(MIN_DAY, day))
      : DEFAULT_STATE.leadDay,
    region: REGION_ID.test(region) ? region : null,
    band: BANDS.includes(band as RiskBand) ? (band as RiskBand) : undefined,
  };
}

/**
 * The query string for a state, or "" when it matches the default.
 *
 * Only what differs is written, so an ordinary visit keeps a clean address bar and a
 * shared link carries nothing but the part that was meant to be shared.
 */
export function toSearch(state: AppState): string {
  const q = new URLSearchParams();
  if (state.view !== DEFAULT_STATE.view) q.set("view", state.view);
  if (state.leadDay !== DEFAULT_STATE.leadDay) q.set("day", String(state.leadDay));
  if (state.region) q.set("region", state.region);
  if (state.band) q.set("band", state.band);
  const s = q.toString();
  return s ? `?${s}` : "";
}
