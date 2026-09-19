import { describe, expect, it } from "vitest";
import { finiteCurve, peakValue, valueAtRatio } from "./economicValue";

type Pt = { cost_loss_ratio: number; value: number };

// Hand-built curve, not fixture-derived: the arithmetic below is checked by eye so a
// regression in the lookup shows up as a wrong number rather than a wrong-but-plausible
// one. Deliberately not monotone - the real curve peaks near the base rate and falls
// away on both sides, and a "just take the last point" bug would pass on a rising curve.
const CURVE: Pt[] = [
  { cost_loss_ratio: 0.1, value: 0.20 },
  { cost_loss_ratio: 0.2, value: 0.50 },
  { cost_loss_ratio: 0.3, value: 0.40 },
];

describe("valueAtRatio", () => {
  it("returns the exact point when alpha lands on one", () => {
    expect(valueAtRatio(CURVE, 0.2)?.value).toBe(0.5);
  });

  it("snaps to the nearest point, not the next one along", () => {
    // 0.21 is 0.01 from 0.2 and 0.09 from 0.3.
    expect(valueAtRatio(CURVE, 0.21)?.cost_loss_ratio).toBe(0.2);
    // 0.26 is 0.06 from 0.2 and 0.04 from 0.3 - across the midpoint, so it moves on.
    expect(valueAtRatio(CURVE, 0.26)?.cost_loss_ratio).toBe(0.3);
  });

  it("clamps outside the sampled range instead of returning undefined", () => {
    expect(valueAtRatio(CURVE, 0.0)?.cost_loss_ratio).toBe(0.1);
    expect(valueAtRatio(CURVE, 1.0)?.cost_loss_ratio).toBe(0.3);
  });

  it("returns undefined for an empty or missing curve", () => {
    expect(valueAtRatio([], 0.5)).toBeUndefined();
    expect(valueAtRatio(undefined, 0.5)).toBeUndefined();
    expect(valueAtRatio(null, 0.5)).toBeUndefined();
  });
});

describe("peakValue", () => {
  it("finds the maximum, which is not the last point", () => {
    expect(peakValue(CURVE)).toEqual({ cost_loss_ratio: 0.2, value: 0.5 });
  });

  it("ignores NaN, which verification.py emits for a degenerate split", () => {
    const withNaN: Pt[] = [
      { cost_loss_ratio: 0.1, value: Number.NaN },
      { cost_loss_ratio: 0.2, value: 0.3 },
    ];
    expect(peakValue(withNaN)).toEqual({ cost_loss_ratio: 0.2, value: 0.3 });
  });

  it("returns undefined when every point is NaN", () => {
    expect(peakValue([{ cost_loss_ratio: 0.1, value: Number.NaN }])).toBeUndefined();
  });
});

describe("finiteCurve", () => {
  it("drops non-finite points so the chart does not draw a gap as zero", () => {
    const mixed: Pt[] = [
      { cost_loss_ratio: 0.1, value: Number.NaN },
      { cost_loss_ratio: 0.2, value: 0.3 },
    ];
    expect(finiteCurve(mixed)).toEqual([{ cost_loss_ratio: 0.2, value: 0.3 }]);
  });

  it("is empty, never undefined, for a missing curve", () => {
    expect(finiteCurve(undefined)).toEqual([]);
    expect(finiteCurve(null)).toEqual([]);
  });
});
