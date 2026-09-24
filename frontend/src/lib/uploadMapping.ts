import type { MappingProposal } from "../api/types";

/**
 * Keep the backend's conversion attached to the column only while the user keeps the
 * variable it was detected for. If the variable changes, a stale km/h -> m/s conversion
 * would silently corrupt a different measurement.
 */
export function unitConversionForChoice(
  sourceColumn: string,
  variable: string,
  suggestedVariable: string | null | undefined,
  proposedConversion: string | null | undefined,
): string | null {
  if (!variable) return null;
  if (variable === suggestedVariable && proposedConversion) return proposedConversion;
  return inferUnitConversion(sourceColumn, variable);
}

/** Mirrors the small, deterministic unit hints emitted by the backend mapper. */
export function inferUnitConversion(sourceColumn: string, variable: string): string | null {
  const header = sourceColumn.toLowerCase().replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();

  if (variable === "wind_speed_ms" && /\bkm\s*(?:\/\s*h|h)\b|km per h|kph|km hr/.test(header)) {
    return "kmh_to_ms";
  }
  if (variable === "temperature_c" && /\bk\b|kelvin/.test(header)) return "K_to_C";
  if (variable === "pressure_hpa" && /\bpa\b/.test(header) && !/\bhpa\b/.test(header)) {
    return "Pa_to_hPa";
  }
  if (variable === "soil_moisture_pct" && /frac|m3 m3|proportion|vol frac/.test(header)) {
    return "frac_to_pct";
  }
  return null;
}

export function confirmedRole(proposal: MappingProposal): string {
  return proposal.role || "measurement";
}
