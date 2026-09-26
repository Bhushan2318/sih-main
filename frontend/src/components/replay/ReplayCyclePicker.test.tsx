import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { ReplayCyclePicker } from "./ReplayCyclePicker";
import { ReplayEventHeader } from "./ReplayEventHeader";
import { REAL_REPLAY_CYCLES } from "../../test/fixtures/replayCycles";

const EVENTS = REAL_REPLAY_CYCLES.filter((c) => c.kind === "event");
const FORECASTS = REAL_REPLAY_CYCLES.filter((c) => c.kind !== "event");

describe("ReplayCyclePicker", () => {
  it("offers past events first, in their own group, by title", () => {
    const { container } = render(
      <ReplayCyclePicker cycles={REAL_REPLAY_CYCLES} value={EVENTS[0].init_date} onChange={vi.fn()} />,
    );
    const groups = container.querySelectorAll("optgroup");
    expect(groups).toHaveLength(2);
    expect(groups[0].label).toMatch(/past events/i);
    expect(groups[1].label).toMatch(/recent forecasts/i);
    const eventOptions = groups[0].querySelectorAll("option");
    expect(eventOptions).toHaveLength(EVENTS.length);
    expect(eventOptions[0].textContent).toBe(EVENTS[0].title);
    expect(groups[1].querySelectorAll("option")).toHaveLength(FORECASTS.length);
  });

  it("reports the picked cycle's init date", () => {
    const onChange = vi.fn();
    render(<ReplayCyclePicker cycles={REAL_REPLAY_CYCLES} value={EVENTS[0].init_date} onChange={onChange} />);
    fireEvent.change(screen.getByLabelText(/cycle/i), { target: { value: EVENTS[1].init_date } });
    expect(onChange).toHaveBeenCalledWith(EVENTS[1].init_date);
  });

  it("stays a flat list when the API offers no past events", () => {
    const { container } = render(
      <ReplayCyclePicker cycles={FORECASTS} value={FORECASTS[0].init_date} onChange={vi.fn()} />,
    );
    expect(container.querySelectorAll("optgroup")).toHaveLength(0);
    expect(container.querySelectorAll("option")).toHaveLength(FORECASTS.length);
  });
});

describe("ReplayEventHeader", () => {
  it("names the event, its sample, and what the model knew", () => {
    const ev = EVENTS[0];
    render(<ReplayEventHeader cycle={ev} />);
    expect(screen.getByText(ev.title as string)).toBeInTheDocument();
    expect(screen.getByText(ev.sample_note as string)).toBeInTheDocument();
    const [y, m, d] = ev.init_date.split("-").map(Number);
    const month = new Date(Date.UTC(y, m - 1, d)).toLocaleDateString("en-GB", { month: "short", timeZone: "UTC" });
    expect(screen.getByText(new RegExp(`known on ${d} ${month} ${y}, 00 UTC`))).toBeInTheDocument();
    expect(screen.getByText(/checked against ERA5/)).toBeInTheDocument();
  });
});
