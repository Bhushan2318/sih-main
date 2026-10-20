import { translateKey } from "../i18n";

/**
 * Column names, in the reader's language (English text lives in i18n/locales/en.json).
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

const VARIABLES: Record<string, { unit: string | null }> = {
  atmospheric_moisture_kgm2: { unit: "kg/m²" },
  humidity_pct: { unit: "%" },
  pressure_hpa: { unit: "hPa" },
  rainfall_mm: { unit: "mm" },
  soil_moisture_pct: { unit: "%" },
  temperature_c: { unit: "°C" },
  wind_direction_deg: { unit: "°" },
  wind_speed_ms: { unit: "m/s" },
};

/** Engineered prefixes, longest first so `pred_err_` is tested before any `pred_`. */
const PREFIXES: string[] = [
  // C1 jumpiness and C2 time-lagged ensemble, per variable.
  "jump_abs_change_",
  "jump_sign_flips_",
  "jump_std_",
  "laf_spread_ratio_",
  "laf_pool_mean_",
  "laf_pool_size_",
  "laf_pool_std_",
  "pred_err_",
  "spread_",
  "conf_",
  "fc_",
];


function varName(name: string): string | undefined {
  return name in VARIABLES ? translateKey(`var.${name}`) : undefined;
}

/** `some_new_variable_xyz` -> `Some new variable xyz`. */
function prettify(name: string): string {
  const spaced = name.replace(/_/g, " ").trim();
  return spaced ? spaced[0].toUpperCase() + spaced.slice(1) : "";
}

/** A modelled variable's name, without its unit. */
export function variableLabel(name: string | null | undefined): string {
  if (!name) return "";
  return varName(name) ?? prettify(name);
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
  const known = translateKey(`feat.${name}`);
  if (known !== undefined) return known;

  for (const prefix of PREFIXES) {
    if (name.startsWith(prefix)) {
      const rest = name.slice(prefix.length);
      // Lower-cased: it sits mid-sentence ("Predicted humidity error"), not at the front.
      // A no-op for scripts without case.
      const v = (varName(rest) ?? prettify(rest)).toLowerCase();
      return translateKey(`feat.${prefix}`, { v }) ?? prettify(name);
    }
  }
  return varName(name) ?? prettify(name);
}
