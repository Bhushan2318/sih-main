import { describe, expect, it } from "vitest";
import { parseIsoTimestamp } from "./format";

describe("parseIsoTimestamp", () => {
  it("honours an explicit positive offset", () => {
    expect(parseIsoTimestamp("2026-09-24T10:00:00+05:30")?.toISOString())
      .toBe("2026-09-24T04:30:00.000Z");
  });

  it("normalises compact and negative offsets", () => {
    expect(parseIsoTimestamp("2026-09-24T10:00:00-0400")?.toISOString())
      .toBe("2026-09-24T14:00:00.000Z");
    expect(parseIsoTimestamp("2026-09-24T10:00:00+05")?.toISOString())
      .toBe("2026-09-24T05:00:00.000Z");
  });

  it("keeps UTC and treats zone-less API date-times as UTC", () => {
    expect(parseIsoTimestamp("2026-09-24T10:00:00Z")?.toISOString())
      .toBe("2026-09-24T10:00:00.000Z");
    expect(parseIsoTimestamp("2026-09-24T10:00:00")?.toISOString())
      .toBe("2026-09-24T10:00:00.000Z");
    expect(parseIsoTimestamp("2026-09-24")?.toISOString())
      .toBe("2026-09-24T00:00:00.000Z");
  });

  it("rejects malformed values instead of producing a misleading date", () => {
    expect(parseIsoTimestamp("not-a-timestamp")).toBeNull();
    expect(parseIsoTimestamp("2026-09-24T10:00:00+99:99")).toBeNull();
  });
});
