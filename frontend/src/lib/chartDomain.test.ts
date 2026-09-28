import { describe, expect, it } from "vitest";
import { REAL_KERALA_IDUKKI_RAINFALL } from "../test/fixtures/replayKerala";
import { closeEnoughBand, yDomainForVariable } from "./chartDomain";

describe("yDomainForVariable", () => {
  it("floors rainfall, wind speed, humidity and soil moisture at zero", () => {
    // The bug this exists to fix: the focus chart's rainfall axis reached -55mm on the
    // live site because domain={["auto","auto"]} let a wide ensemble spread pull it below
    // a value that cannot physically be negative.
    expect(yDomainForVariable("rainfall_mm")).toEqual([0, "auto"]);
    expect(yDomainForVariable("wind_speed_ms")).toEqual([0, "auto"]);
    expect(yDomainForVariable("humidity_pct")).toEqual([0, "auto"]);
    expect(yDomainForVariable("soil_moisture_pct")).toEqual([0, "auto"]);
  });

  it("leaves variables that can genuinely go negative fully automatic", () => {
    expect(yDomainForVariable("temperature_c")).toEqual(["auto", "auto"]);
    expect(yDomainForVariable("pressure_hpa")).toEqual(["auto", "auto"]);
    expect(yDomainForVariable("wind_direction_deg")).toEqual(["auto", "auto"]);
    expect(yDomainForVariable("atmospheric_moisture_kgm2")).toEqual(["auto", "auto"]);
  });

  it("is fully automatic for an unknown or missing variable", () => {
    expect(yDomainForVariable(null)).toEqual(["auto", "auto"]);
    expect(yDomainForVariable(undefined)).toEqual(["auto", "auto"]);
    expect(yDomainForVariable("some_new_variable")).toEqual(["auto", "auto"]);
  });
});

describe("closeEnoughBand", () => {
  it("never reaches below zero for a variable that cannot be negative", () => {
    // Replay's rainfall axis read -50 to -95 mm in every event: the band is observed
    // +/- the bust threshold, Recharts widens the axis to fit it, and the floor that
    // yDomainForVariable sets does not hold against data below it.
    const thr = REAL_KERALA_IDUKKI_RAINFALL.bust_threshold as number;
    const bands = REAL_KERALA_IDUKKI_RAINFALL.points
      .map((p) => closeEnoughBand("rainfall_mm", p.observed_value, thr))
      .filter((b): b is [number, number] => b != null);
    expect(bands.length).toBeGreaterThan(0);
    expect(bands.some((_, i) => (REAL_KERALA_IDUKKI_RAINFALL.points[i].observed_value ?? 0) < thr))
      .toBe(true);
    for (const [lo, hi] of bands) {
      expect(lo).toBeGreaterThanOrEqual(0);
      expect(hi).toBeGreaterThan(lo);
    }
  });

  it("is observed plus and minus the threshold otherwise", () => {
    expect(closeEnoughBand("rainfall_mm", 20, 9)).toEqual([11, 29]);
    expect(closeEnoughBand("rainfall_mm", 4, 9)).toEqual([0, 13]);
    expect(closeEnoughBand("temperature_c", -2, 3)).toEqual([-5, 1]);
  });

  it("is absent without an observation or a threshold", () => {
    expect(closeEnoughBand("rainfall_mm", null, 9)).toBeNull();
    expect(closeEnoughBand("rainfall_mm", 4, null)).toBeNull();
  });
});
