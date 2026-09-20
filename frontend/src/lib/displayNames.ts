/**
 * Column names, in English.
 *
 * Feature and variable names arrive from the model's own feature list, so every screen
 * that explains a prediction was printing them raw: `conf_atmospheric_moisture_kgm2`,
 * `historical_bust_frequency_region_season`, `atmospheric_moisture_kgm2 >= 8.16`. That
 * made the SHAP panel - the part that says *why* a district is at risk, and the thing
 * this project has that a bare score does not - unreadable to anyone who has not seen the
 * schema. docs/known-issues.md called it "the densest text on the page for a reader
 * without a meteorology background".
 *
 * A lookup with a readable fallback, rather than renaming anything upstream: the names
 * are a contract between the model artifact and the scorer, and must not change to suit
 * a UI. Anything added later still reads as English via `prettify`, just less precisely.
 */

const VARIABLES: Record<string, { label: string; unit: string | null }> = {
  atmospheric_moisture_kgm2: { label: "Atmospheric moisture", unit: "kg/m²" },
  humidity_pct: { label: "Humidity", unit: "%" },
  pressure_hpa: { label: "Pressure", unit: "hPa" },
  rainfall_mm: { label: "Rainfall", unit: "mm" },
  soil_moisture_pct: { label: "Soil moisture", unit: "%" },
  temperature_c: { label: "Temperature", unit: "°C" },
  wind_direction_deg: { label: "Wind direction", unit: "°" },
  wind_speed_ms: { label: "Wind speed", unit: "m/s" },
};

/** Engineered prefixes, longest first so `pred_err_` is tested before any `pred_`. */
const PREFIXES: { prefix: string; phrase: (v: string) => string }[] = [
  { prefix: "pred_err_", phrase: (v) => `Predicted ${v} error` },
  { prefix: "spread_", phrase: (v) => `Ensemble disagreement on ${v}` },
  { prefix: "conf_", phrase: (v) => `Confidence in ${v}` },
  { prefix: "fc_", phrase: (v) => `Forecast ${v}` },
];

/** Features that are not per-variable. Phrased as what they mean, not what they are. */
const FEATURES: Record<string, string> = {
  historical_bust_frequency_region_season: "How often this district busts this season",
  lead_time_days: "How far ahead the forecast is",
  region_id: "Which district",
  state_id: "Which state",
  season: "Season",
  month: "Month",
  spread_mean: "Average ensemble disagreement",
  spread_max: "Largest ensemble disagreement",
  ensemble_spread: "Ensemble disagreement",
  ensemble_member_count: "Ensemble members available",
  forecast_value: "Forecast value",
  forecast_error_lag: "Error on the previous forecast",
  moisture_rate_of_change: "How fast moisture is changing",
  pressure_rate_of_change: "How fast pressure is changing",
  // C1, forecast jumpiness: how much the forecast for one day moved between cycles.
  jump_abs_change: "How much the forecast changed since the last run",
  jump_std: "How much the forecast has been changing",
  jump_sign_flips: "How often the forecast has flipped direction",
  jump_rel_climatology: "Forecast churn against this district's normal",
};

/** `some_new_variable_xyz` -> `Some new variable xyz`. */
function prettify(name: string): string {
  const spaced = name.replace(/_/g, " ").trim();
  return spaced ? spaced[0].toUpperCase() + spaced.slice(1) : "";
}

/** A modelled variable's name, without its unit. */
export function variableLabel(name: string | null | undefined): string {
  if (!name) return "";
  return VARIABLES[name]?.label ?? prettify(name);
}

/** A modelled variable's unit, or null where one would be meaningless. */
export function variableUnit(name: string | null | undefined): string | null {
  if (!name) return null;
  return VARIABLES[name]?.unit ?? null;
}

/**
 * Any model feature's name, in English.
 *
 * Handles the per-variable prefixes by expanding both halves, so a variable added to
 * VARIABLES improves every derived feature at once.
 */
export function featureLabel(name: string | null | undefined): string {
  if (!name) return "";
  const known = FEATURES[name];
  if (known) return known;

  for (const { prefix, phrase } of PREFIXES) {
    if (name.startsWith(prefix)) {
      const rest = name.slice(prefix.length);
      // Lower-cased: it sits mid-sentence ("Predicted humidity error"), not at the front.
      const varName = (VARIABLES[rest]?.label ?? prettify(rest)).toLowerCase();
      return phrase(varName);
    }
  }
  return VARIABLES[name]?.label ?? prettify(name);
}
