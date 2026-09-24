import { describe, expect, it } from "vitest";
import { confirmedRole, inferUnitConversion, unitConversionForChoice } from "./uploadMapping";

describe("upload mapping", () => {
  it("keeps a proposed conversion only for its original variable", () => {
    expect(unitConversionForChoice("wind_speed_kmh", "wind_speed_ms", "wind_speed_ms", "kmh_to_ms"))
      .toBe("kmh_to_ms");
    expect(unitConversionForChoice("wind_speed_kmh", "temperature_c", "wind_speed_ms", "kmh_to_ms"))
      .toBeNull();
  });

  it("preserves a structural role", () => {
    expect(confirmedRole({
      source_column: "value",
      normalized: "value",
      role: "value",
      sample_values: [],
      suggested_variable: "rainfall_mm",
      suggested_value_type: "forecast",
      confidence: 1,
      ambiguity_gap: 0,
      method: "structural",
      unit_conversion: null,
      decision: "needs_confirmation",
      alternatives: [],
    })).toBe("value");
  });

  it("infers a conversion from the source header after a variable edit", () => {
    expect(inferUnitConversion("wind_speed_kmh", "wind_speed_ms")).toBe("kmh_to_ms");
    expect(inferUnitConversion("wind_speed_km/h", "wind_speed_ms")).toBe("kmh_to_ms");
    expect(inferUnitConversion("temperature_K", "temperature_c")).toBe("K_to_C");
    expect(inferUnitConversion("pressure_Pa", "pressure_hpa")).toBe("Pa_to_hPa");
    expect(inferUnitConversion("soil_moisture_fraction", "soil_moisture_pct")).toBe("frac_to_pct");
  });
});
