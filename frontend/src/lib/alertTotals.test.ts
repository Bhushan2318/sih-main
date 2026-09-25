import { describe, expect, it } from "vitest";
import { alertTotals } from "./alertTotals";
import { REAL_ALL_REGIONS_20260925 } from "../test/fixtures/allRegions";

describe("alertTotals", () => {
  it("counts every bust and watch district-day across all lead days, not a capped list", () => {
    const all = REAL_ALL_REGIONS_20260925;
    let bust = 0;
    let watch = 0;
    const districts = new Set<string>();
    for (const day of all.days) {
      for (const r of day.regions) {
        if (r.risk_band === "high") bust += 1;
        if (r.risk_band === "medium") watch += 1;
        if (r.risk_band === "high" || r.risk_band === "medium") districts.add(r.region_id);
      }
    }
    const t = alertTotals(all);
    expect(t?.bust).toBe(bust);
    expect(t?.watch).toBe(watch);
    expect(t?.districts).toBe(districts.size);
    expect(t?.days).toBe(all.days.length);
    // the fixture has both bands, which the capped list reported as "0 watch"
    expect(t?.watch).toBeGreaterThan(0);
  });

  it("names the variable behind the most alert district-days", () => {
    const t = alertTotals(REAL_ALL_REGIONS_20260925);
    expect(typeof t?.topDriver?.[0]).toBe("string");
    expect(t?.topDriver?.[1]).toBeGreaterThan(0);
  });

  it("is null with nothing to count", () => {
    expect(alertTotals(undefined)).toBeNull();
    expect(alertTotals({ ...REAL_ALL_REGIONS_20260925, days: [] })).toBeNull();
  });
});
