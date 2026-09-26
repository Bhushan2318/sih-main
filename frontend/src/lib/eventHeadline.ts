import type { ReplayCycleSummary, ReplayFocusSeries, ReplayLeadStep } from "../api/types";
import { bandLabel } from "../theme";
import { variableUnit } from "./displayNames";
import { leadDayFromDates } from "./eventLeadDay";
import { dayLabel, formatByMagnitude, formatDateUtc } from "./format";

function valueWithUnit(v: number | null | undefined, unit: string | null): string {
  const s = formatByMagnitude(v ?? null);
  return s === "—" || !unit ? s : `${s} ${unit}`;
}

function pctOrDash(v: number | null | undefined): string {
  return typeof v === "number" && Number.isFinite(v) ? `${Math.round(v * 100)}%` : "—";
}

/**
 * The one data-derived sentence Replay shows under an event's title: the day it peaked,
 * what the forecast said there against what ERA5 recorded, and the model's own bust risk
 * that day. Every value is read off the served replay response - `steps` (per-lead,
 * per-district bust probability and band) and `focusOptions` (per-lead forecast vs
 * observed) - never a hand-written event fact (replay_cases.py only hand-writes titles).
 *
 * Null for a live forecast, or an event missing the catalogue fields a headline needs to
 * be about anything (its focus district, its variable, or the day it peaked). A value the
 * response does not carry prints as an em dash, never a zero.
 */
export function eventHeadline(
  cycle: Pick<ReplayCycleSummary, "kind" | "init_date" | "peak_valid_date" | "focus_region_id" | "focus_variable">,
  steps: readonly ReplayLeadStep[],
  focusOptions: readonly ReplayFocusSeries[],
): string | null {
  if (cycle.kind !== "event") return null;
  if (!cycle.peak_valid_date || !cycle.focus_region_id || !cycle.focus_variable) return null;
  const lead = leadDayFromDates(cycle.init_date, cycle.peak_valid_date);
  if (lead == null) return null;

  const series =
    focusOptions.find(
      (o) => o.region_id === cycle.focus_region_id && o.variable === cycle.focus_variable,
    ) ?? focusOptions.find((o) => o.region_id === cycle.focus_region_id) ?? null;
  const point = series?.points.find((p) => p.lead_time_days === lead) ?? null;
  const unit = series?.unit ?? variableUnit(cycle.focus_variable);
  const districtName = series?.region_name ?? cycle.focus_region_id;

  const step = steps.find((s) => s.lead_time_days === lead) ?? null;
  const row = step?.regions.find((r) => r.region_id === cycle.focus_region_id) ?? null;

  return (
    `Peak day ${formatDateUtc(cycle.peak_valid_date)} (${dayLabel(lead)}). ` +
    `The forecast gave ${districtName} ${valueWithUnit(point?.predicted_value, unit)}; ` +
    `ERA5 recorded ${valueWithUnit(point?.observed_value, unit)}. ` +
    `Sanket's bust risk that day: ${pctOrDash(row?.bust_probability)} ` +
    `(${bandLabel(row?.risk_band ?? null)}).`
  );
}
