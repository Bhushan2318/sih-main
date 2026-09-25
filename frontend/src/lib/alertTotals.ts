import type { AllRegionsResponse } from "../api/types";
import { riskBandForRegion } from "./riskBands";

/** Totals for the Alerts page's summary, from every district on every lead day.
 *
 * The alert list itself is capped at the 200 most severe, and the summary used to be
 * counted from that list: on a bad day all 200 are bust, so it read "0 in the watch band"
 * while the map showed 141 watch districts on Day 1 alone. Counting from the uncapped
 * /api/regions/all answer - already fetched by the dashboard - gives the real totals. */
export function alertTotals(all: AllRegionsResponse | undefined) {
  if (!all?.days.length) return null;

  let bust = 0;
  let watch = 0;
  const districts = new Set<string>();
  const drivers = new Map<string, number>();

  for (const day of all.days) {
    for (const r of day.regions ?? []) {
      const band = riskBandForRegion(r);
      if (band !== "high" && band !== "medium") continue;
      if (band === "high") bust += 1;
      else watch += 1;
      districts.add(r.region_id);
      if (r.dominant_variable) drivers.set(r.dominant_variable, (drivers.get(r.dominant_variable) ?? 0) + 1);
    }
  }

  const topDriver = [...drivers.entries()].sort((a, b) => b[1] - a[1])[0] ?? null;
  return { bust, watch, districts: districts.size, days: all.days.length, topDriver };
}
