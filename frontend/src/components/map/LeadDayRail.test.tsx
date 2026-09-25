import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { LeadDayRail } from "./LeadDayRail";
import { REAL_ALL_REGIONS_20260925 } from "../../test/fixtures/allRegions";
import { variableLabel } from "../../lib/displayNames";

describe("LeadDayRail", () => {
  it("labels every lead day as 'Day N', not 'DN'", () => {
    render(<LeadDayRail all={REAL_ALL_REGIONS_20260925} value={1} onChange={vi.fn()} />);
    const rows = screen.getAllByRole("button");
    expect(rows).toHaveLength(10);
    rows.forEach((row, i) => {
      expect(within(row).getByText(`Day ${i + 1}`)).toBeInTheDocument();
    });
    expect(screen.queryByText(/^D\d+$/)).toBeNull();
  });

  it("shows each day's band counts and its main cause on the row itself", () => {
    render(<LeadDayRail all={REAL_ALL_REGIONS_20260925} value={1} onChange={vi.fn()} />);
    const day1 = screen.getAllByRole("button")[0];
    const regions = REAL_ALL_REGIONS_20260925.days[0].regions;
    const counts = new Map<string, number>();
    regions.forEach((r) => r.dominant_variable && counts.set(r.dominant_variable, (counts.get(r.dominant_variable) ?? 0) + 1));
    const top = [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
    expect(within(day1).getByText(new RegExp(variableLabel(top), "i"))).toBeInTheDocument();
    expect(within(day1).getByText(/\d+ bust/)).toBeInTheDocument();
  });

  it("writes the driver summary in days, not D-numbers", () => {
    const { container } = render(<LeadDayRail all={REAL_ALL_REGIONS_20260925} value={1} onChange={vi.fn()} />);
    const note = container.querySelector(".rail__driver");
    expect(note?.textContent ?? "").not.toMatch(/\bD\d/);
  });
});
