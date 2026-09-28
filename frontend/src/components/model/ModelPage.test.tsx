import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { REAL_MODEL_STATUS_POOLED as STATUS } from "../../test/fixtures/modelStatusPooled";

vi.mock("../../hooks/useDashboardData", () => ({
  useModelStatus: () => ({ data: STATUS, isLoading: false, error: null, failureCount: 0 }),
}));
vi.mock("../../store/liveStore", () => ({
  useLiveStore: (pick: (s: object) => unknown) =>
    pick({ connectionStatus: "open", trainingInProgress: false }),
}));
// Cards with their own data and tests; this file is about the page's own copy.
vi.mock("./BaselineLadderTable", () => ({ BaselineLadderTable: () => null }));
vi.mock("./EconomicValueCard", () => ({ EconomicValueCard: () => null }));
vi.mock("./CorpReliabilityCard", () => ({ CorpReliabilityCard: () => null }));
vi.mock("./MissesCard", () => ({ MissesCard: () => null }));
vi.mock("./PipelineLog", () => ({ PipelineLog: () => null }));
vi.mock("../upload/UploadPanel", () => ({ UploadPanel: () => null }));

import { ModelPage } from "./ModelPage";

const td = STATUS.training_data!;
const n = (v: number) => v.toLocaleString();

describe("ModelPage says what the served run was trained on", () => {
  it("does not call every cycle in the archive a training cycle", () => {
    // It read "Trained on 6,558 forecast cycles from 2000 to 2016": 6,558 counts the
    // validation and held-out years too, and the models were fit on a sample of 2,000.
    render(<ModelPage />);
    expect(document.body.textContent).not.toMatch(new RegExp(`Trained on ${n(td.cycles!)}`));
    const line = screen.getByText(/fit on/i, { selector: "p" });
    expect(line).toHaveTextContent(n(td.fit_cycles!));
    expect(line).toHaveTextContent(n(td.train_cycles!));
    expect(line).toHaveTextContent(String(td.test_year));
  });

  it("counts initialisations per year over the years the cycles came from", () => {
    // train_cycles / 17 years gave "about 343" - the validation year's cycles sit inside
    // the same 2000-2016 span and were left out of the numerator.
    render(<ModelPage />);
    const years = td.last_train_year! - td.first_train_year! + 1;
    const perYear = Math.round((td.train_cycles! + td.val_cycles!) / years);
    expect(screen.getByText(/initialisations per year/)).toHaveTextContent(String(perYear));
  });

  it("describes the baseline the numbers were computed against", () => {
    // regressors._evaluate scores against the average error of the rows being scored
    // (baseline_mae_predict_mean), not persistence.
    render(<ModelPage />);
    expect(document.body.textContent).not.toMatch(/tomorrow is like today/);
    expect(screen.getByRole("heading", { name: /average error/i })).toBeInTheDocument();
  });
});

describe("ModelPage explains what counts as a bust", () => {
  it("tells the two threshold columns apart, with units", () => {
    render(<ModelPage />);
    expect(screen.getByRole("columnheader", { name: /unit/i })).toBeInTheDocument();
    expect(screen.getByText(/ensemble average/i, { selector: "p" })).toBeInTheDocument();
    expect(screen.getByText(/single member/i, { selector: "p" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "mm" })).toBeInTheDocument();
  });

  it("says why a per-variable top 10% makes about half of all forecasts busts", () => {
    const vars = Object.keys(STATUS.thresholds?.bust_threshold ?? {}).length;
    const q = Number(STATUS.thresholds?.threshold_percentile) / 100;
    const independent = Math.round((1 - q ** vars) * 100);
    const served = Math.round(Number(STATUS.validation_metrics?.classifier?.bust_rate) * 100);
    render(<ModelPage />);
    const why = screen.getByText(/any one of its/i, { selector: "p" });
    expect(why).toHaveTextContent(`${vars} variables`);
    expect(why).toHaveTextContent(`${independent}%`);
    expect(why).toHaveTextContent(`${served}%`);
  });
});
