import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { REAL_MODEL_STATUS_POOLED } from "../../test/fixtures/modelStatusPooled";
import { translate } from "../../i18n";
import { MissesCard } from "./MissesCard";

describe("MissesCard with no misses.json", () => {
  it("does not promise the cases arrive with a retrain", () => {
    // misses.json is written by scripts/run_baselines next to baselines.json; a retrain
    // alone does not produce it (the served run_20261006T200800Z shows "misses": {}).
    render(<MissesCard data={{ ...REAL_MODEL_STATUS_POOLED, misses: {} }} />);
    expect(screen.getByText(translate("miss.notYet")).textContent).not.toMatch(/retrain/i);
  });
});
