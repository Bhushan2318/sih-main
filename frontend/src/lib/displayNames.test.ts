import { describe, expect, it } from "vitest";
import { featureLabel, variableLabel, variableUnit } from "./displayNames";

describe("variableLabel", () => {
  it("names the eight modelled variables in plain words", () => {
    expect(variableLabel("atmospheric_moisture_kgm2")).toBe("Atmospheric moisture");
    expect(variableLabel("wind_direction_deg")).toBe("Wind direction");
    expect(variableLabel("soil_moisture_pct")).toBe("Soil moisture");
  });

  it("falls back to a readable form rather than showing snake_case", () => {
    // The point of the fallback: a feature added after this table was written must still
    // read as English, not as a column name.
    expect(variableLabel("some_new_variable_xyz")).toBe("Some new variable xyz");
  });

  it("returns an empty string for nothing, not the word undefined", () => {
    expect(variableLabel(undefined)).toBe("");
    expect(variableLabel("")).toBe("");
  });
});

describe("variableUnit", () => {
  it("gives the unit separately so a caller can place it", () => {
    expect(variableUnit("temperature_c")).toBe("°C");
    expect(variableUnit("atmospheric_moisture_kgm2")).toBe("kg/m²");
    expect(variableUnit("rainfall_mm")).toBe("mm");
  });

  it("is null when there is no sensible unit", () => {
    expect(variableUnit("region_id")).toBeNull();
    expect(variableUnit("unknown_thing")).toBeNull();
  });
});

describe("featureLabel", () => {
  it("expands the four engineered prefixes", () => {
    expect(featureLabel("pred_err_humidity_pct")).toBe("Predicted humidity error");
    expect(featureLabel("conf_pressure_hpa")).toBe("Confidence in pressure");
    expect(featureLabel("spread_rainfall_mm")).toBe("Ensemble disagreement on rainfall");
    expect(featureLabel("fc_temperature_c")).toBe("Forecast temperature");
  });

  it("names the standalone features a reader actually sees", () => {
    expect(featureLabel("historical_bust_frequency_region_season"))
      .toBe("How often this district busts this season");
    expect(featureLabel("lead_time_days")).toBe("How far ahead the forecast is");
    expect(featureLabel("region_id")).toBe("Which district");
    expect(featureLabel("jump_std")).toBe("How much the forecast has been changing");
  });

  it("does not mistake a variable prefix inside a longer word", () => {
    // "forecast_value" starts with neither fc_ nor conf_, and must not be mangled.
    expect(featureLabel("forecast_value")).toBe("Forecast value");
  });

  it("falls back readably for an unmapped feature", () => {
    expect(featureLabel("brand_new_feature")).toBe("Brand new feature");
    expect(featureLabel("pred_err_brand_new")).toBe("Predicted brand new error");
  });

  it("returns an empty string for nothing", () => {
    expect(featureLabel(undefined)).toBe("");
  });
});
