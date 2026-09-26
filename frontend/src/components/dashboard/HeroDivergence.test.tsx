import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { REAL_ENSEMBLE_20260926 } from "../../test/fixtures/ensemble";

// Recharts needs layout jsdom does not have; the labels around the chart are what is tested.
vi.mock("recharts", async (orig) => {
  const actual = await orig<typeof import("recharts")>();
  return { ...actual, ResponsiveContainer: () => null };
});

import { HeroDivergence } from "./HeroDivergence";

/**
 * The ring is the mean over every district and all ten lead days, and its count is the
 * districts reaching the bust band on any day. The KPI cards beside it are one lead day.
 * Both were labelled "Mean bust risk", so 25% next to 40% read as a contradiction.
 */
describe("HeroDivergence says what its numbers cover", () => {
  it("labels the ring as all ten lead days", () => {
    render(<HeroDivergence data={REAL_ENSEMBLE_20260926} />);
    expect(screen.getByText(/all 10 days/i)).toBeInTheDocument();
  });

  it("counts districts that reach the bust band on at least one day", () => {
    render(<HeroDivergence data={REAL_ENSEMBLE_20260926} />);
    expect(
      screen.getByText(`${REAL_ENSEMBLE_20260926.n_high_regions} of ${REAL_ENSEMBLE_20260926.n_scored_regions} districts reach the bust band on at least one day`),
    ).toBeInTheDocument();
  });

  it("says what a bust is, without writing a threshold down", () => {
    render(<HeroDivergence data={REAL_ENSEMBLE_20260926} />);
    expect(screen.getByText(/worst 10%/i)).toBeInTheDocument();
  });
});
