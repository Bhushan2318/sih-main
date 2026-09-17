import { describe, expect, it } from "vitest";
import { formatByMagnitude, formatMetric } from "./format";

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
