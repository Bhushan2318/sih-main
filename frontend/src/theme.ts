/**
 * Chart palette, kept in one place and mirroring the tokens in styles.css.
 *
 * Recharts needs real colour strings (it measures and interpolates them), so these cannot
 * be `var(--blue)`. Anything that changes here must change there too.
 *
 * The forecast/observed pair is deliberately a hue *and* a value contrast, not two blues:
 * two series drawn in near-identical colours is the single easiest way to make a
 * divergence chart lie about whether anything diverged.
 */
export const CHART = {
  /** the model's own trace */
  forecast: "#2b4eff",
  /** what actually happened - ink, dashed, so it reads as ground truth */
  observed: "#0b1220",
  /** predicted |error| */
  error: "#f5254a",
  /** ensemble members / secondary traces */
  member: "#a9b6d6",
  /** the "you are here" marker on a replayed cycle */
  marker: "#f08700",

  /** risk ramp */
  low: "#00a882",
  medium: "#f08700",
  high: "#f5254a",

  grid: "#e5e9f0",
  axis: "#7b8798",
} as const;

/**
 * One display word per risk band, used at every render site.
 *
 * The band *values* stay `low | medium | high` everywhere they matter as data or as a CSS
 * class suffix; only the text a person reads is mapped here. "watch" and "bust" are the
 * operational words - "bust" is the term in the problem statement, so the label reads as
 * deliberate rather than generic.
 */
export const BAND_LABEL: Record<string, string> = {
  low: "low",
  medium: "watch",
  high: "bust",
};

export const bandLabel = (band: string | null | undefined): string =>
  band ? BAND_LABEL[band] ?? band : "no data";
