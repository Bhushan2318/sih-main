import { describe, expect, it } from "vitest";
import { leadDayIsUninformative, leadDayRung } from "./baselineLadder";
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
