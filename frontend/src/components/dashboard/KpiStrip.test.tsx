import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { REAL_ALL_REGIONS_20260925 } from "../../test/fixtures/allRegions";
import { KpiStrip } from "./KpiStrip";

const all = REAL_ALL_REGIONS_20260925;
const day1 = all.days.find((d) => d.lead_time_days === 1)!;

describe("KpiStrip", () => {
  it("names the lead day in the card labels, so they can't be mistaken for the ten-day hero", () => {
    render(<KpiStrip all={all} day={day1} />);
    expect(screen.getByText("Mean bust risk · Day 1")).toBeInTheDocument();
  });

  it("opens the peak district when its card is clicked", () => {
    const onSelectRegion = vi.fn();
    render(<KpiStrip all={all} day={day1} onSelectRegion={onSelectRegion} />);
    const peak = [...day1.regions]
      .filter((r) => r.bust_probability != null)
      .sort((a, b) => (b.bust_probability as number) - (a.bust_probability as number))[0];

    fireEvent.click(screen.getByRole("button", { name: /Peak risk/ }));
    expect(onSelectRegion).toHaveBeenCalledWith(peak.region_id);
  });

  it("is not a button when nothing listens", () => {
    render(<KpiStrip all={all} day={day1} />);
    expect(screen.queryByRole("button", { name: /Peak risk/ })).toBeNull();
  });
});
