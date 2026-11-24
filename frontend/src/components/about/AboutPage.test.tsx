import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { REAL_MODEL_STATUS_POOLED } from "../../test/fixtures/modelStatusPooled";
import { translate } from "../../i18n";

// The served run_20261006T200800Z answers with "baselines": {} - its bundle carries no
// baselines.json, because the pinned-model path in CI does not score the ladder.
const STATUS = { ...REAL_MODEL_STATUS_POOLED, baselines: {} };

vi.mock("../../hooks/useDashboardData", () => ({
  useModelStatus: () => ({ data: STATUS, isLoading: false, error: null, failureCount: 0 }),
}));

import { AboutPage } from "./AboutPage";

describe("AboutPage with no baseline ladder", () => {
  it("does not blame the gap on the model's age or promise a retrain", () => {
    // It read "this model predates it and the table is written on the next retrain".
    // Neither is true: the run postdates the ladder, and the ladder is scored by
    // scripts/run_baselines, not by retraining.
    render(<AboutPage onReplay={() => {}} />);
    const note = screen.getByText(translate("ab.noLadder"));
    expect(note.textContent).not.toMatch(/predates|retrain/i);
    expect(document.body.textContent).not.toMatch(/next retrain/i);
  });
});
