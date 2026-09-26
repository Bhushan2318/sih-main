import { describe, expect, it } from "vitest";
import type { ReplayFocusSeries, ReplayLeadStep } from "../api/types";
import { REAL_REPLAY_CYCLES } from "../test/fixtures/replayCycles";
import { REAL_KERALA_DAY3_STEP, REAL_KERALA_IDUKKI_RAINFALL } from "../test/fixtures/replayKerala";
import { eventHeadline } from "./eventHeadline";

const KERALA = REAL_REPLAY_CYCLES.find((c) => c.title?.includes("Kerala"));
if (!KERALA) throw new Error("fixture no longer has the Kerala case");

// Real served values for the Kerala event's peak day (see replayKerala.ts).
const DAY3_STEP = REAL_KERALA_DAY3_STEP;
const IDUKKI_RAINFALL = REAL_KERALA_IDUKKI_RAINFALL;

describe("eventHeadline", () => {
  it("is null for a live forecast cycle, which has no event facts to state", () => {
    const forecast = REAL_REPLAY_CYCLES.find((c) => c.kind === "forecast");
    if (!forecast) throw new Error("fixture has no forecast cycle");
    expect(eventHeadline(forecast, [DAY3_STEP], [IDUKKI_RAINFALL])).toBeNull();
  });

  it("is null when the catalogue fields a headline needs are missing", () => {
    expect(eventHeadline({ ...KERALA, focus_region_id: null }, [DAY3_STEP], [IDUKKI_RAINFALL]))
      .toBeNull();
    expect(eventHeadline({ ...KERALA, peak_valid_date: null }, [DAY3_STEP], [IDUKKI_RAINFALL]))
      .toBeNull();
  });

  it("states the peak day, the forecast-vs-observed values, and the bust risk that day", () => {
    const headline = eventHeadline(KERALA, [DAY3_STEP], [IDUKKI_RAINFALL]);
    // Served: predicted 35.269 mm, observed 144.90 mm, bust probability 0.712 (medium).
    // formatByMagnitude drops to 0 decimals at |v| >= 100, as every chart in the app does.
    expect(headline).toBe(
      "Peak day 15 Aug 2018 (Day 3). The forecast gave Idukki, Kerala 35.3 mm; " +
      "ERA5 recorded 145 mm. Sanket's bust risk that day: 71% (watch).",
    );
  });

  it("shows an em dash, never zero, for a value the served response does not carry", () => {
    const noPoints: ReplayFocusSeries = { ...IDUKKI_RAINFALL, points: [] };
    const noRegionRow: ReplayLeadStep = { ...DAY3_STEP, regions: [] };
    const headline = eventHeadline(KERALA, [noRegionRow], [noPoints]);
    expect(headline).toBe(
      "Peak day 15 Aug 2018 (Day 3). The forecast gave Idukki, Kerala —; " +
      "ERA5 recorded —. Sanket's bust risk that day: — (no data).",
    );
    expect(headline).not.toContain("0 mm");
    expect(headline).not.toContain("0%");
  });

  it("falls back to any series for the district when the exact variable is unavailable", () => {
    const otherVariable: ReplayFocusSeries = { ...IDUKKI_RAINFALL, variable: "humidity_pct", unit: "%" };
    const headline = eventHeadline(KERALA, [DAY3_STEP], [otherVariable]);
    expect(headline).toContain("Idukki, Kerala 35.3 %");
  });
});
