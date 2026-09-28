import type { RegionSummary, RiskBand } from "../api/types";
import { bandLabel } from "../theme";
import { variableLabel } from "./displayNames";

/** The tooltip's widest, as .map-tooltip's max-width in styles.css. */
export const TOOLTIP_MAX_W = 240;
const GAP = 12;

/**
 * Where the map tooltip goes, in the coordinates of .map-wrap - the box it is positioned
 * in. It was placed from the SVG's box, which sits below the search and controls inside
 * the wrap, so it landed ~80 px off the cursor. Right of and below the cursor; flipped to
 * its left where it would run past the wrap's right edge, where an absolutely positioned
 * box shrinks to fit (107 px wide on the live site, a word per line).
 */
export function tooltipPlacement(x: number, y: number, wrapWidth: number):
  { left: number; top: number; flip: boolean } {
  const flip = x + GAP + TOOLTIP_MAX_W > wrapWidth;
  return { left: flip ? x - GAP : x + GAP, top: y + GAP, flip };
}

/** A scored district's tooltip body. */
export function districtTooltipLines(region: RegionSummary, band: RiskBand | null): string[] {
  return [
    `Bust probability: ${((region.bust_probability ?? 0) * 100).toFixed(1)}% (${bandLabel(band)})`,
    region.dominant_variable ? `Driver: ${variableLabel(region.dominant_variable)}` : "",
    region.confidence != null ? `Mean confidence: ${(region.confidence * 100).toFixed(0)}%` : "",
  ].filter(Boolean);
}
