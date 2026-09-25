import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { WorstDistrictsPanel } from "./WorstDistrictsPanel";
import { REAL_ALL_REGIONS_20260925 } from "../../test/fixtures/allRegions";
import { variableLabel } from "../../lib/displayNames";
import type { RegionsResponse } from "../../api/types";

const DAY1 = REAL_ALL_REGIONS_20260925.days[0];

describe("WorstDistrictsPanel", () => {
  it("heads the list with the lead day in words", () => {
    render(<WorstDistrictsPanel day={DAY1} onSelect={vi.fn()} />);
    expect(screen.getByRole("heading", { name: /Worst districts · Day 1/ })).toBeInTheDocument();
  });

  it("ranks districts worst first, with each one's main cause in words", () => {
    render(<WorstDistrictsPanel day={DAY1} onSelect={vi.fn()} />);
    const rows = screen.getAllByRole("button");
    const expected = [...DAY1.regions]
      .filter((r) => typeof r.bust_probability === "number")
      .sort((a, b) => (b.bust_probability as number) - (a.bust_probability as number));
    expect(rows).toHaveLength(expected.length);
    expect(within(rows[0]).getByText(expected[0].region_name as string)).toBeInTheDocument();
    const cause = expected[0].dominant_variable as string;
    expect(within(rows[0]).getByText(variableLabel(cause))).toBeInTheDocument();
    expect(within(rows[0]).queryByText(cause)).toBeNull();
  });

  it("opens a district when its row is clicked", () => {
    const onSelect = vi.fn();
    render(<WorstDistrictsPanel day={DAY1} onSelect={onSelect} />);
    fireEvent.click(screen.getAllByRole("button")[0]);
    expect(onSelect).toHaveBeenCalledWith(DAY1.regions[0].region_id);
  });

  it("falls back to the plain empty state when the day has nothing scored", () => {
    const empty: RegionsResponse = { ...DAY1, regions: [] };
    render(<WorstDistrictsPanel day={empty} onSelect={vi.fn()} />);
    expect(screen.getByText("No region selected")).toBeInTheDocument();
  });
});
