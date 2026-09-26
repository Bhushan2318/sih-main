/** Variables that cannot physically go negative (see VARIABLES in displayNames.ts for the
 * full set of modelled variable ids). Temperature, pressure, wind direction and
 * atmospheric moisture are left out deliberately - a genuinely negative value is real for
 * some of those, so their axis stays fully automatic. */
const NON_NEGATIVE_VARIABLES = new Set(["rainfall_mm", "wind_speed_ms", "humidity_pct", "soil_moisture_pct"]);

export type ChartYDomain = [number | "auto", "auto"];

/**
 * The Y-axis domain for a chart of one modelled variable.
 *
 * Recharts' `domain={["auto","auto"]}` fits the axis to whatever the series (forecast,
 * observed, and the ensemble-spread band around it) actually contains. For rainfall, wind
 * speed, humidity and soil moisture that let a wide spread pull the lower bound below a
 * value that cannot exist - measured on the live site, the rainfall focus chart's axis
 * reached -55mm. Flooring the lower bound at 0 for those variables fixes it without giving
 * up automatic scaling on the upper bound.
 */
export function yDomainForVariable(variable: string | null | undefined): ChartYDomain {
  return [variable && NON_NEGATIVE_VARIABLES.has(variable) ? 0 : "auto", "auto"];
}
