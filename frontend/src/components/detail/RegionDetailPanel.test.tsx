import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { RegionDetailResponse } from "../../api/types";
import { featureLabel, variableLabel } from "../../lib/displayNames";
import { REAL_REGION_LEHLADAKH, REAL_REGION_PURBAMEDINIPUR } from "../../test/fixtures/regionDetail";

let current: RegionDetailResponse | undefined;
vi.mock("../../hooks/useDashboardData", () => ({
  useRegionDetail: () => ({ data: current, isLoading: false, error: null }),
}));
// Recharts needs layout jsdom does not have; the panel's choice of series is what is tested.
vi.mock("./VariableTrajectoryChart", () => ({
  VariableTrajectoryChart: ({ series }: { series: { variable: string } }) => (
    <div data-testid="trajectory">{series.variable}</div>
  ),
}));
vi.mock("./BustProbabilityCurve", () => ({ BustProbabilityCurve: () => null }));

import { RegionDetailPanel } from "./RegionDetailPanel";

const peakDriver = (d: RegionDetailResponse) =>
  [...d.bust_probability_curve].sort((a, b) => b.bust_probability - a.bust_probability)[0]
    .dominant_variable;

describe("RegionDetailPanel opens on the leading driver", () => {
  beforeEach(() => { current = undefined; });

  it("puts SHAP right under the peak, above the driver's forecast-vs-actual chart", () => {
    // The panel scrolls inside itself; SHAP at the bottom sat below the fold of that
    // scroll, and it is the feature that explains the prediction.
    current = REAL_REGION_PURBAMEDINIPUR;
    render(<RegionDetailPanel regionId={current.region_id} onClose={() => {}} />);

    expect(screen.getByTestId("trajectory")).toHaveTextContent(peakDriver(current)!);
    expect(screen.getByRole("heading", { name: /Leading driver/ })).toBeInTheDocument();

    const headings = screen.getAllByRole("heading", { level: 3 }).map((h) => h.textContent ?? "");
    const at = (re: RegExp) => headings.findIndex((t) => re.test(t));
    expect(at(/Peak bust risk/)).toBeLessThan(at(/What the model relies on/));
    expect(at(/What the model relies on/)).toBeLessThan(at(/Leading driver/));
    expect(at(/Leading driver/)).toBeLessThan(at(/Bust probability by lead day/));
  });

  it("sums up the top SHAP factors in one line under the peak", () => {
    current = REAL_REGION_PURBAMEDINIPUR;
    render(<RegionDetailPanel regionId={current.region_id} onClose={() => {}} />);
    const top = current.top_factors.slice(0, 3).map((f) => featureLabel(f.feature));
    const line = screen.getByText(/What the model leans on here/);
    for (const name of top) expect(line).toHaveTextContent(name);
  });

  it("a different district opens on its own driver, not the previous tab", () => {
    current = REAL_REGION_PURBAMEDINIPUR;
    const { rerender } = render(
      <RegionDetailPanel regionId={current.region_id} onClose={() => {}} />,
    );
    // Leave the first district on some other tab.
    const tabs = screen.getByRole("tablist");
    fireEvent.click(within(tabs).getByRole("tab", { name: new RegExp(`^${variableLabel("temperature_c")}`) }));
    expect(screen.getByTestId("trajectory")).toHaveTextContent("temperature_c");

    current = REAL_REGION_LEHLADAKH;
    rerender(<RegionDetailPanel regionId={current.region_id} onClose={() => {}} />);
    expect(peakDriver(current)).not.toBe(peakDriver(REAL_REGION_PURBAMEDINIPUR));
    expect(screen.getByTestId("trajectory")).toHaveTextContent(peakDriver(current)!);
  });

  it("marks the driver tab", () => {
    current = REAL_REGION_LEHLADAKH;
    render(<RegionDetailPanel regionId={current.region_id} onClose={() => {}} />);
    const tab = within(screen.getByRole("tablist")).getByRole("tab", { selected: true });
    expect(tab).toHaveTextContent(variableLabel(peakDriver(current)!));
    expect(tab).toHaveTextContent("driver");
  });
  it("says when a variable stops short of Day 10 instead of silently ending", () => {
    // The real Purba Medinipur response, with the field the API now adds set as the API
    // sets it for soil moisture (contracts.ARCHIVE_MAX_LEAD_DAYS). Only that field differs.
    const base = REAL_REGION_PURBAMEDINIPUR;
    current = {
      ...base,
      variables: base.variables.map((v) =>
        v.variable === "soil_moisture_pct" ? { ...v, max_lead_day: 3 } : v),
    };
    render(<RegionDetailPanel regionId={current.region_id} onClose={() => {}} />);
    fireEvent.click(within(screen.getByRole("tablist"))
      .getByRole("tab", { name: new RegExp(`^${variableLabel("soil_moisture_pct")}`) }));
    expect(screen.getByText(/Modelled for Days 1–3 only/)).toBeInTheDocument();
  });
});
