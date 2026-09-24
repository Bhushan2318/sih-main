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
      .toBe("How often this district's forecasts go badly wrong in this season");
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

describe("featureLabel covers every feature the live model uses", () => {
  // Names copied from run_20260922T043925Z/feature_columns.json (2026-09-25), with the
  // per-variable suffix shown once. None may fall through to snake_case prettifying.
  const live = [
    "laf_pool_mean", "laf_pool_std", "laf_pool_size", "laf_spread_ratio",
    "laf_spread_ratio_rainfall_mm", "laf_pool_mean_rainfall_mm",
    "jump_abs_change_rainfall_mm", "jump_std_rainfall_mm", "jump_sign_flips_rainfall_mm",
    "mjo_amplitude", "mjo_rmm1", "mjo_rmm2",
    "area_km2", "border_distance_km", "centroid_lat", "centroid_lon", "elevation_mean",
  ];
  it.each(live)("%s reads as a sentence, not a column name", (name) => {
    const label = featureLabel(name);
    expect(label).not.toMatch(/_|kgm2|pct|km2|laf|rmm\d/i);
  });
  it("names the jumpiness of one variable in words", () => {
    expect(featureLabel("jump_std_rainfall_mm")).toBe("How much the rainfall forecast has been changing");
  });
});
