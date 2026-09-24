import type { RegionSummary, RiskBand } from "../api/types";

/** The two probability edges published by the backend for the current run. */
export interface RiskCuts {
  medium: number;
  high: number;
}

const VALID_BANDS: readonly RiskBand[] = ["low", "medium", "high"];

/**
 * A probability is only a score when the API supplied one. In particular, an absent
 * probability is not a zero probability: treating null as zero was what made an unscored
 * model look like a completely low-risk run in a few places in the UI.
 */
export function isScoredProbability(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function isScoredRegion(
  region: RegionSummary,
): region is RegionSummary & { bust_probability: number } {
  return region.data_available !== false && isScoredProbability(region.bust_probability);
}

export function asRiskBand(value: unknown): RiskBand | null {
  return typeof value === "string" && (VALID_BANDS as readonly string[]).includes(value)
    ? (value as RiskBand)
    : null;
}

/**
 * Classify a probability only when the current run's cuts are available. There is
 * deliberately no guessed/default cut here: a stale or missing threshold must render as
 * unclassified, never as low risk.
 */
export function riskBandForProbability(
  probability: number | null | undefined,
  cuts?: RiskCuts | null,
): RiskBand | null {
  const valid = cuts ? normaliseCuts(cuts) : undefined;
  if (!isScoredProbability(probability) || !valid) return null;
  if (probability >= valid.high) return "high";
  if (probability >= valid.medium) return "medium";
  return "low";
}

/** Prefer the band returned by the API, and use cuts only to fill a missing band. */
export function riskBandForRegion(
  region: RegionSummary,
  cuts?: RiskCuts | null,
): RiskBand | null {
  if (!isScoredRegion(region)) return null;
  return asRiskBand(region.risk_band) ?? riskBandForProbability(region.bust_probability, cuts);
}

/**
 * Resolve cuts from the model status first, then from the definitions included with a
 * regions/replay response. The latter keeps a freshly rendered map correct while the
 * status query is still loading, without introducing another set of UI constants.
 */
export function resolveRiskCuts(
  ...sources: Array<Partial<RiskCuts> | Record<string, string> | null | undefined>
): RiskCuts | undefined {
  for (const source of sources) {
    if (!source) continue;

    const direct = normaliseCuts({
      medium: typeof source.medium === "number" ? source.medium : undefined,
      high: typeof source.high === "number" ? source.high : undefined,
    });
    if (direct) return direct;

    // Only treat a source as prose definitions after the direct numeric shape failed.
    // This avoids accidentally reading arbitrary Record<string, string> fields as cuts.
    if ("low" in source || "medium" in source || "high" in source) {
      const parsed = parseDefinitionCuts(source as Record<string, string>);
      if (parsed) return parsed;
    }
  }
  return undefined;
}

function normaliseCuts(source: Partial<RiskCuts>): RiskCuts | undefined {
  const medium = source.medium;
  const high = source.high;
  if (typeof medium !== "number" || typeof high !== "number") return undefined;
  if (!Number.isFinite(medium) || !Number.isFinite(high)) return undefined;
  if (medium < 0 || high > 1 || medium >= high) return undefined;
  return { medium, high };
}

function parseDefinitionCuts(definitions: Record<string, string>): RiskCuts | undefined {
  const low = definitions.low?.match(/below\s+(\d+(?:\.\d+)?)\s*%/i)?.[1];
  const medium = definitions.medium?.match(
    /(\d+(?:\.\d+)?)\s*%\s*(?:to|–|—|-)\s*(\d+(?:\.\d+)?)\s*%/i,
  );
  const high = definitions.high?.match(/(\d+(?:\.\d+)?)\s*%\s*or\s+above/i)?.[1];

  if (!low || !medium || !high) return undefined;
  const parsed = normaliseCuts({
    medium: Number(low) / 100,
    high: Number(medium[2]) / 100,
  });
  if (!parsed) return undefined;

  // The high definition is the authoritative upper edge. Reject malformed or
  // contradictory prose rather than silently displaying a guessed threshold.
  const describedHigh = Number(high) / 100;
  return Math.abs(describedHigh - parsed.high) < 0.0001 ? parsed : undefined;
}

/**
 * Infer edges from labelled rows only as a last resort for a standalone map consumer.
 * There are no numeric fallbacks: a set containing only one band cannot tell us where its
 * missing edge belongs, so it remains unclassified until the status/definitions arrive.
 */
export function inferRiskCuts(regions: readonly RegionSummary[]): RiskCuts | undefined {
  const byBand: Record<RiskBand, number[]> = { low: [], medium: [], high: [] };
  for (const region of regions) {
    if (!isScoredRegion(region)) continue;
    const band = asRiskBand(region.risk_band);
    if (band) byBand[band].push(region.bust_probability);
  }

  const edge = (below: readonly number[], above: readonly number[]): number | undefined => {
    if (!below.length || !above.length) return undefined;
    return (Math.max(...below) + Math.min(...above)) / 2;
  };
  const medium = edge(byBand.low, byBand.medium);
  const high = edge(byBand.medium, byBand.high);
  return normaliseCuts({ medium: medium ?? Number.NaN, high: high ?? Number.NaN });
}
