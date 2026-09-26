import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { IngestRunRow } from "../../api/types";

/**
 * Nine runs, shaped like a real /api/ingest/runs row - enough to prove the log caps at
 * five and the toggle reveals the rest. There is no captured fixture for this endpoint
 * yet, so these values are illustrative for shape, not a served measurement.
 */
function run(id: number, over: Partial<IngestRunRow> = {}): IngestRunRow {
  return {
    id,
    kind: "forecast",
    target: `2026-09-${10 + id}`,
    status: "complete",
    trigger: "schedule",
    rows_ingested: 33300,
    started_at: `2026-09-${10 + id}T00:00:00Z`,
    finished_at: `2026-09-${10 + id}T00:05:00Z`,
    seconds: 300,
    detail: null,
    error: null,
    ...over,
  };
}

let runs: IngestRunRow[] = [];
vi.mock("../../hooks/useDashboardData", () => ({
  useIngestRuns: () => ({ data: { runs }, error: null, isLoading: false }),
}));

import { PipelineLog } from "./PipelineLog";

describe("PipelineLog", () => {
  it("shows only the newest 5 runs by default", () => {
    runs = Array.from({ length: 9 }, (_, i) => run(9 - i));
    render(<PipelineLog />);
    // one header row plus 5 body rows
    expect(screen.getAllByRole("row")).toHaveLength(6);
    expect(screen.getByRole("button", { name: "Show all 9" })).toBeInTheDocument();
  });

  it("reveals every fetched run behind the 'Show all N' toggle", () => {
    runs = Array.from({ length: 9 }, (_, i) => run(9 - i));
    render(<PipelineLog />);
    fireEvent.click(screen.getByRole("button", { name: "Show all 9" }));
    expect(screen.getAllByRole("row")).toHaveLength(10);
  });

  it("shows no toggle when there are 5 or fewer runs", () => {
    runs = Array.from({ length: 3 }, (_, i) => run(3 - i));
    render(<PipelineLog />);
    expect(screen.getAllByRole("row")).toHaveLength(4);
    expect(screen.queryByRole("button", { name: /Show all/ })).not.toBeInTheDocument();
  });

  it("renders nothing when there are no runs", () => {
    runs = [];
    const { container } = render(<PipelineLog />);
    expect(container).toBeEmptyDOMElement();
  });
});
