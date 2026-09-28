import { describe, expect, it } from "vitest";
import { REAL_ALL_REGIONS_20260925 } from "../test/fixtures/allRegions";
import { variableLabel } from "./displayNames";
import { TOOLTIP_MAX_W, districtTooltipLines, tooltipPlacement } from "./mapTooltip";

describe("tooltipPlacement", () => {
  it("sits right of and below the cursor when there is room", () => {
    expect(tooltipPlacement(100, 80, 600)).toEqual({ left: 112, top: 92, flip: false });
  });

  it("flips to the cursor's left near the right edge instead of squashing", () => {
    // Near the right edge the box shrank to 107 px wide and wrapped every word.
    const p = tooltipPlacement(560, 80, 600);
    expect(p.flip).toBe(true);
    expect(p.left).toBe(548);
    expect(600 - 560 - 12).toBeLessThan(TOOLTIP_MAX_W);
  });
});

describe("districtTooltipLines", () => {
  it("names the driver in words, never by its variable id", () => {
    // Read "Driver: soil_moisture_pct" on the live map.
    const day = REAL_ALL_REGIONS_20260925.days[0];
    const region = day.regions.find((r) => r.dominant_variable)!;
    const lines = districtTooltipLines(region, "high");
    const driver = lines.find((l) => l.startsWith("Driver:"))!;
    expect(driver).toBe(`Driver: ${variableLabel(region.dominant_variable)}`);
    expect(lines.join(" ")).not.toContain(region.dominant_variable!);
  });
});
