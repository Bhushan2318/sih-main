import { describe, expect, it } from "vitest";
import { leadDayIsUninformative, leadDayRung, rungLabel } from "./baselineLadder";
import { REAL_BASELINES_RUN_20260916 } from "../test/fixtures/modelStatus";

describe("leadDayRung", () => {
  it("finds the lead_day rung in a real ladder", () => {
    const rung = leadDayRung(REAL_BASELINES_RUN_20260916!.models);
    expect(rung?.name).toBe("lead_day");
    expect(rung?.bss).toBeLessThan(0);
  });

  it("returns undefined when models is undefined", () => {
    expect(leadDayRung(undefined)).toBeUndefined();
  });

  it("returns undefined when the ladder has no lead_day rung", () => {
    const withoutLeadDay = REAL_BASELINES_RUN_20260916!.models!.filter(
      (m) => m.name !== "lead_day",
    );
    expect(leadDayRung(withoutLeadDay)).toBeUndefined();
  });
});

describe("leadDayIsUninformative", () => {
  it("is true for the real, negative-skill lead_day rung", () => {
    expect(leadDayIsUninformative(REAL_BASELINES_RUN_20260916!.models)).toBe(true);
  });

  it("is null when there is no lead_day rung to judge", () => {
    expect(leadDayIsUninformative([])).toBeNull();
    expect(leadDayIsUninformative(undefined)).toBeNull();
  });

  it("is null when bss is null rather than a number", () => {
    expect(
      leadDayIsUninformative([
        { name: "lead_day", brier: null, bss: null, roc_auc: null, f1: null, is_model: false },
      ]),
    ).toBeNull();
  });

  it("is false when a lead_day rung genuinely beats climatology", () => {
    expect(
      leadDayIsUninformative([
        { name: "lead_day", brier: 0.2, bss: 0.05, roc_auc: 0.6, f1: 0.4, is_model: false },
      ]),
    ).toBe(false);
  });
});

describe("leadDayIsUninformative, near zero", () => {
  it("treats a skill of +0.0001 as no better than climatology, not as signal", () => {
    // the live run's lead_day rung scores +0.0001 (read from /api/model/status 2026-09-26)
    expect(
      leadDayIsUninformative([
        { name: "lead_day", brier: 0.2531, bss: 0.0001, roc_auc: 0.5095, f1: null, is_model: false },
      ]),
    ).toBe(true);
  });
});

describe("rungLabel", () => {
  it("names every baseline rung in words", () => {
    expect(rungLabel("lead_day")).toBe("Lead day only");
    expect(rungLabel("lead+spread+season")).toBe("Lead + spread + season");
    expect(rungLabel("climatology")).toBe("Climatology");
    expect(rungLabel("spread")).toBe("Ensemble spread");
    expect(rungLabel("analog")).toBe("Analogues");
    expect(rungLabel("EMOS")).toBe("EMOS");
  });

  it("passes an unknown rung through unchanged", () => {
    expect(rungLabel("Sanket bust classifier")).toBe("Sanket bust classifier");
    expect(rungLabel("something_new")).toBe("something_new");
  });
});
