import { describe, expect, it } from "vitest";
import { yDomainForVariable } from "./chartDomain";

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
