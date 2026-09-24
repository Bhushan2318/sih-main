import { describe, expect, it } from "vitest";
import type { RegionSummary } from "../api/types";
import {
  inferRiskCuts,
  isScoredRegion,
  resolveRiskCuts,
  riskBandForProbability,
  riskBandForRegion,
} from "./riskBands";

const region = (overrides: Partial<RegionSummary> = {}): RegionSummary => ({
  region_id: "IN-MH-MUMBAI",
  region_name: "Mumbai, Maharashtra",
  bust_probability: 0.2,
  risk_band: "low",
  confidence: 0.8,
  dominant_variable: "rainfall_mm",
  data_available: true,
  ...overrides,
});

describe("risk bands", () => {
  it("uses the backend cuts instead of a UI default", () => {
    const cuts = { medium: 0.2, high: 0.8 };
    expect(riskBandForProbability(0.19, cuts)).toBe("low");
    expect(riskBandForProbability(0.2, cuts)).toBe("medium");
    expect(riskBandForProbability(0.8, cuts)).toBe("high");
  });

  it("does not turn missing scores into low risk", () => {
    expect(riskBandForProbability(null, { medium: 0.3, high: 0.7 })).toBeNull();
    expect(riskBandForProbability(0, undefined)).toBeNull();
    expect(isScoredRegion(region({ bust_probability: null, data_available: false }))).toBe(false);
  });

  it("prefers a backend band and fills a missing band from cuts", () => {
    expect(riskBandForRegion(region({ risk_band: "high", bust_probability: 0.1 }))).toBe("high");
    expect(riskBandForRegion(region({ risk_band: null, bust_probability: 0.9 }), { medium: 0.3, high: 0.7 }))
      .toBe("high");
  });

  it("parses backend prose definitions and rejects contradictory prose", () => {
    expect(resolveRiskCuts({
      low: "bust risk below 33%",
      medium: "bust risk 33% to 66%",
      high: "bust risk 66% or above",
    })).toEqual({ medium: 0.33, high: 0.66 });
    expect(resolveRiskCuts({
      low: "bust risk below 33%",
      medium: "bust risk 33% to 66%",
      high: "bust risk 80% or above",
    })).toBeUndefined();
  });

  it("infers cuts only when labelled rows provide both edges", () => {
    expect(inferRiskCuts([
      region({ bust_probability: 0.1, risk_band: "low" }),
      region({ bust_probability: 0.4, risk_band: "medium" }),
      region({ bust_probability: 0.9, risk_band: "high" }),
    ])).toEqual({ medium: 0.25, high: 0.65 });
  });
});
