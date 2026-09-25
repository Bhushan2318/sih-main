import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { BaselineLadderCard } from "./BaselineLadderCard";
import { REAL_MODEL_STATUS_TRAINED } from "../../test/fixtures/modelStatus";
import type { ModelStatusResponse } from "../../api/types";

describe("BaselineLadderCard", () => {
  it("renders the ladder and flags the lead_day rung's negative skill", () => {
    render(<BaselineLadderCard data={REAL_MODEL_STATUS_TRAINED} />);
    expect(screen.getAllByText("Lead day only").length).toBeGreaterThan(0);
    expect(screen.getAllByText("-0.0021").length).toBeGreaterThan(0);
    expect(screen.getByText(/at or below zero/)).toBeInTheDocument();
    // the served model's own row is still present
    expect(screen.getByText("Sanket bust classifier")).toBeInTheDocument();
  });

  it("renders nothing when data is undefined (loading or API-down)", () => {
    const { container } = render(<BaselineLadderCard data={undefined} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when the model has not been trained", () => {
    const untrained: ModelStatusResponse = {
      ...REAL_MODEL_STATUS_TRAINED,
      model_trained: false,
      message: "No model trained yet",
    };
    const { container } = render(<BaselineLadderCard data={untrained} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when baselines is missing (a run that predates the ladder)", () => {
    const noBaselines: ModelStatusResponse = { ...REAL_MODEL_STATUS_TRAINED, baselines: undefined };
    const { container } = render(<BaselineLadderCard data={noBaselines} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when models is an empty array", () => {
    const emptyModels: ModelStatusResponse = {
      ...REAL_MODEL_STATUS_TRAINED,
      baselines: { ...REAL_MODEL_STATUS_TRAINED.baselines, models: [] },
    };
    const { container } = render(<BaselineLadderCard data={emptyModels} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("still renders the table, without the callout, when lead_day is missing from the ladder", () => {
    const withoutLeadDay: ModelStatusResponse = {
      ...REAL_MODEL_STATUS_TRAINED,
      baselines: {
        ...REAL_MODEL_STATUS_TRAINED.baselines,
        models: REAL_MODEL_STATUS_TRAINED.baselines!.models!.filter((m) => m.name !== "lead_day"),
      },
    };
    render(<BaselineLadderCard data={withoutLeadDay} />);
    expect(screen.getByText("Climatology")).toBeInTheDocument();
    expect(screen.queryByText("Lead day only")).not.toBeInTheDocument();
    expect(screen.queryByText(/at or below zero/)).not.toBeInTheDocument();
  });

  it("calls a lead_day skill of +0.0001 effectively zero, not a signal", () => {
    const nearZero: ModelStatusResponse = {
      ...REAL_MODEL_STATUS_TRAINED,
      baselines: {
        ...REAL_MODEL_STATUS_TRAINED.baselines,
        models: REAL_MODEL_STATUS_TRAINED.baselines!.models!.map((m) =>
          m.name === "lead_day" ? { ...m, bss: 0.0001 } : m),
      },
    };
    render(<BaselineLadderCard data={nearZero} />);
    expect(screen.getByText(/effectively zero/)).toBeInTheDocument();
    expect(screen.queryByText(/carries some signal/)).not.toBeInTheDocument();
  });
});
