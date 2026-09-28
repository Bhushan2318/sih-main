import { describe, expect, it } from "vitest";
import { axisTick, cycleChipLabel, dayLabel, dayRange, formatByMagnitude, formatMetric } from "./format";

describe("dayLabel / dayRange", () => {
  it("names a lead day in words, never as D1", () => {
    expect(dayLabel(1)).toBe("Day 1");
    expect(dayLabel(10)).toBe("Day 10");
  });

  it("names a run of lead days, and a single day as itself", () => {
    expect(dayRange(1, 3)).toBe("Days 1–3");
    expect(dayRange(4, 4)).toBe("Day 4");
  });
});

describe("formatMetric", () => {
  it("formats a finite number at the requested precision", () => {
    expect(formatMetric(0.28210326, 4)).toBe("0.2821");
    expect(formatMetric(0.807, 3)).toBe("0.807");
  });

  it("shows an em dash for anything that isn't a finite number", () => {
    expect(formatMetric(null, 3)).toBe("—");
    expect(formatMetric(undefined, 3)).toBe("—");
    expect(formatMetric(NaN, 3)).toBe("—");
    expect(formatMetric("0.5", 3)).toBe("—");
  });
});

describe("formatByMagnitude", () => {
  it("does not round a genuinely small value down to zero", () => {
    // the bug this exists to fix: a real 0.03mm rainfall miss must not read as "0.0"
    // next to a headline risk percentage, which is indistinguishable from a perfect call
    expect(formatByMagnitude(0.03)).toBe("0.03");
  });

  it("scales precision down as magnitude grows", () => {
    expect(formatByMagnitude(6.186805)).toBe("6.19");
    expect(formatByMagnitude(42.3)).toBe("42.3");
    expect(formatByMagnitude(350)).toBe("350");
  });

  it("shows an em dash for null, undefined, or NaN", () => {
    expect(formatByMagnitude(null)).toBe("—");
    expect(formatByMagnitude(undefined)).toBe("—");
    expect(formatByMagnitude(NaN)).toBe("—");
  });
});

describe("cycleChipLabel", () => {
  it("names both dates, so init and valid never read as the same date twice", () => {
    expect(cycleChipLabel("2026-09-26", "2026-09-26", 1)).toBe("Issued 26 Sep · valid 26 Sep (Day 1)");
    expect(cycleChipLabel("2026-09-26", "2026-09-30", 5)).toBe("Issued 26 Sep · valid 30 Sep (Day 5)");
  });

  it("falls back to the issue date alone when there is no valid date", () => {
    expect(cycleChipLabel("2026-09-26", null, 1)).toBe("Issued 26 Sep");
  });
});

describe("axisTick", () => {
  it("labels one axis in one style, without padding zeros", () => {
    // formatByMagnitude per tick read "0.00 / 50.0 / 100" on one Replay axis.
    expect([0, 50, 100, 150].map(axisTick)).toEqual(["0", "50", "100", "150"]);
    expect([0, 2.5, 5, 7.5].map(axisTick)).toEqual(["0", "2.5", "5", "7.5"]);
    expect([0.25, 0.5].map(axisTick)).toEqual(["0.25", "0.5"]);
  });

  it("keeps whole numbers whole at any size, and signs", () => {
    expect(axisTick(1012)).toBe("1012");
    expect(axisTick(-5)).toBe("-5");
    expect(axisTick(-0.5)).toBe("-0.5");
  });
});
