import { describe, expect, it } from "vitest";
import { cycleAgeHours, STALE_AFTER_HOURS } from "./FeedFreshness";

describe("cycleAgeHours", () => {
  // The label shape the live /api/ingest/status returned on 2026-09-24: "2026-09-24 12".
  const now = Date.UTC(2026, 8, 24, 22, 30);
  it("reads the UTC cycle label the API sends", () => {
    expect(cycleAgeHours("2026-09-24 12", now)).toBeCloseTo(10.5, 5);
  });
  it("flags a cycle more than a day and a bit old", () => {
    expect(cycleAgeHours("2026-09-23 00", now)! > STALE_AFTER_HOURS).toBe(true);
    expect(cycleAgeHours("2026-09-24 00", now)! > STALE_AFTER_HOURS).toBe(false);
  });
  it("says nothing when there is no cycle rather than inventing an age", () => {
    expect(cycleAgeHours(null, now)).toBeNull();
    expect(cycleAgeHours("not a cycle", now)).toBeNull();
  });
});
