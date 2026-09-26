import { describe, expect, it } from "vitest";
import { initialReplayStepIndex, leadDayFromDates } from "./eventLeadDay";

describe("leadDayFromDates", () => {
  it("computes lead = (valid - init) + 1, per the project's valid_date convention", () => {
    // The Kerala floods case: init 2018-08-13, peaked 2018-08-15 -> Day 3.
    expect(leadDayFromDates("2018-08-13", "2018-08-15")).toBe(3);
  });

  it("is Day 1 when the valid date is the init date itself", () => {
    expect(leadDayFromDates("2019-05-01", "2019-05-01")).toBe(1);
  });

  it("parses in UTC, not the viewer's local zone", () => {
    // A date string near a local-midnight boundary must not shift by a day depending on
    // where the browser thinks it is - only the calendar difference matters.
    expect(leadDayFromDates("2017-11-28", "2017-11-30")).toBe(3);
  });

  it("returns null for a malformed or missing date", () => {
    expect(leadDayFromDates("", "2018-08-15")).toBeNull();
    expect(leadDayFromDates("2018-08-13", "not-a-date")).toBeNull();
  });
});

describe("initialReplayStepIndex", () => {
  const TEN_DAYS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

  it("opens the Kerala case on Day 3 (index 2), not Day 1", () => {
    expect(initialReplayStepIndex("2018-08-13", "2018-08-15", TEN_DAYS)).toBe(2);
  });

  it("is null when there is no peak_valid_date - not an event, so no override", () => {
    expect(initialReplayStepIndex("2026-09-25", null, TEN_DAYS)).toBeNull();
    expect(initialReplayStepIndex("2026-09-25", undefined, TEN_DAYS)).toBeNull();
  });

  it("is null when the init date is missing", () => {
    expect(initialReplayStepIndex(null, "2018-08-15", TEN_DAYS)).toBeNull();
  });

  it("is null when there are no available steps to index into", () => {
    expect(initialReplayStepIndex("2018-08-13", "2018-08-15", [])).toBeNull();
  });

  it("clamps to the last available step when the peak falls beyond it", () => {
    const shortRun = [1, 2, 3, 4, 5, 6, 7, 8];
    // A peak 10 days out with only 8 lead days scored - clamp, do not go out of bounds.
    expect(initialReplayStepIndex("2018-08-13", "2018-08-23", shortRun)).toBe(7);
  });

  it("clamps to the first available step when the peak is at or before init", () => {
    expect(initialReplayStepIndex("2018-08-13", "2018-08-13", TEN_DAYS)).toBe(0);
    // A malformed catalogue entry (peak before init) must not produce a negative index.
    expect(initialReplayStepIndex("2018-08-13", "2018-08-10", TEN_DAYS)).toBe(0);
  });

  it("lands on the closest available lead day when the exact one is missing", () => {
    const gappy = [1, 2, 5, 8];
    // Day 3 is not scored; the closest available is Day 2 (index 1).
    expect(initialReplayStepIndex("2018-08-13", "2018-08-15", gappy)).toBe(1);
  });
});
