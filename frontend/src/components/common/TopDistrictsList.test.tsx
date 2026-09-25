import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { TopDistrictsList, type TopDistrict } from "./TopDistrictsList";
import { REAL_ALL_REGIONS_20260925 } from "../../test/fixtures/allRegions";

const ITEMS: TopDistrict[] = REAL_ALL_REGIONS_20260925.days[0].regions.slice(0, 3).map((r) => ({
  region_id: r.region_id,
  region_name: r.region_name,
  bust_probability: r.bust_probability as number,
  band: r.risk_band,
  dominant_variable: r.dominant_variable,
}));

describe("TopDistrictsList", () => {
  it("shows each district's risk as a whole percentage", () => {
    render(<TopDistrictsList items={ITEMS} onSelect={vi.fn()} />);
    expect(screen.getAllByText(`${(ITEMS[0].bust_probability * 100).toFixed(0)}%`).length).toBeGreaterThan(0);
  });

  it("disables rows that cannot be opened, and marks the active one", () => {
    const onSelect = vi.fn();
    render(
      <TopDistrictsList
        items={ITEMS}
        onSelect={onSelect}
        activeId={ITEMS[0].region_id}
        isSelectable={(id) => id !== ITEMS[1].region_id}
      />,
    );
    const rows = screen.getAllByRole("button");
    expect(rows[0]).toHaveClass("is-active");
    expect(rows[1]).toBeDisabled();
    fireEvent.click(rows[1]);
    expect(onSelect).not.toHaveBeenCalled();
    fireEvent.click(rows[2]);
    expect(onSelect).toHaveBeenCalledWith(ITEMS[2].region_id);
  });
});
