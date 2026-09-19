import type { ModelStatusResponse } from "../api/types";

type Baselines = NonNullable<ModelStatusResponse["baselines"]>;
type BaselineModel = NonNullable<Baselines["models"]>[number];
export type EconomicValuePoint = NonNullable<BaselineModel["economic_value"]>[number];

/** Points the chart can actually draw.
 *
 * `relative_economic_value` returns NaN for every alpha when the split is degenerate
 * (base rate 0 or 1), and JSON carries that through as `null`. Recharts renders a
 * missing y as a break in the line, but a NaN that survives into an axis domain poisons
 * the whole scale - so they are dropped here, once, rather than guarded at each use.
 */
export function finiteCurve(
  curve: EconomicValuePoint[] | null | undefined,
): EconomicValuePoint[] {
  if (!curve) return [];
  return curve.filter(
    (p) => Number.isFinite(p.value) && Number.isFinite(p.cost_loss_ratio),
  );
}

/**
 * The sampled point nearest `alpha`.
 *
 * The backend sweeps a fixed grid (99 points over [0.01, 0.99]) rather than exposing a
 * continuous function, so a slider has to snap. Nearest rather than interpolated on
 * purpose: each point is the *best achievable* value at that alpha, taken over every
 * threshold present in the data, so the curve is an upper envelope and interpolating
 * between two of its points would claim a value no real decision rule achieves.
 *
 * Clamps at both ends, so dragging to 0 or 1 shows the nearest measured value instead
 * of blanking the readout.
 */
export function valueAtRatio(
  curve: EconomicValuePoint[] | null | undefined,
  alpha: number,
): EconomicValuePoint | undefined {
  const pts = finiteCurve(curve);
  if (!pts.length) return undefined;
  return pts.reduce((best, p) =>
    Math.abs(p.cost_loss_ratio - alpha) < Math.abs(best.cost_loss_ratio - alpha) ? p : best,
  );
}

/**
 * The curve's maximum - the cost-loss ratio this model serves best.
 *
 * Worth showing next to the slider because the peak sits near the base rate, which is
 * the honest headline: "most useful to someone whose cost-loss ratio is about X".
 */
export function peakValue(
  curve: EconomicValuePoint[] | null | undefined,
): EconomicValuePoint | undefined {
  const pts = finiteCurve(curve);
  if (!pts.length) return undefined;
  return pts.reduce((best, p) => (p.value > best.value ? p : best));
}
