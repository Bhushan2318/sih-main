import { describe, expect, it } from "vitest";
import type { ReplayFocusSeries, ReplayLeadStep } from "../api/types";
import { REAL_REPLAY_CYCLES } from "../test/fixtures/replayCycles";
import { eventHeadline } from "./eventHeadline";

// REAL_REPLAY_CYCLES (fetched from the live API, see that file's header) has no fixture
// yet carrying per-lead forecast-vs-observed series or per-district step rows for an
// event, so there is nothing real to slice for those two inputs. The event identity below
// (id, init_date, peak_valid_date, focus_region_id, focus_variable) is the real Kerala
// floods catalogue entry, copied from REAL_REPLAY_CYCLES itself; only the per-lead
// predicted/observed/probability numbers are constructed round numbers, chosen to
// exercise the sentence's formatting, not measurements of anything.
const KERALA = REAL_REPLAY_CYCLES.find((c) => c.title?.includes("Kerala"));
if (!KERALA) throw new Error("fixture no longer has the Kerala case");

const DAY3_STEP: ReplayLeadStep = {
  lead_time_days: 3,
  valid_date: "2018-08-15",
  regions: [
    { region_id: "IN-KL-IDUKKI", region_name: "Idukki, Kerala", bust_probability: 0.62,
      risk_band: "medium", confidence: 0.8, dominant_variable: "rainfall_mm" },
  ],
  n_high: 0,
  n_medium: 1,
  mean_bust_probability: 0.62,
  narration: "Day 3 test narration",
};

const IDUKKI_RAINFALL: ReplayFocusSeries = {
  region_id: "IN-KL-IDUKKI",
  region_name: "Idukki, Kerala",
  variable: "rainfall_mm",
  unit: "mm",
  bust_threshold: 40,
  points: [
    { lead_time_days: 1, valid_date: "2018-08-13", predicted_value: 12, observed_value: 18,
      observed_status: "final", ensemble_spread: 2 },
    { lead_time_days: 3, valid_date: "2018-08-15", predicted_value: 45.2, observed_value: 312.6,
      observed_status: "final", ensemble_spread: 30 },
  ],
};

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
    // 312.6 prints as "313" - formatByMagnitude (lib/format.ts) drops to 0 decimals at
    // |v| >= 100, the same precision rule every other chart and panel in the app uses.
    expect(headline).toBe(
      "Peak day 15 Aug 2018 (Day 3). The forecast gave Idukki, Kerala 45.2 mm; " +
      "ERA5 recorded 313 mm. Sanket's bust risk that day: 62% (watch).",
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
    expect(headline).toContain("Idukki, Kerala 45.2 %");
  });
});
