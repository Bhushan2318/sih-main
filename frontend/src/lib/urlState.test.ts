import { describe, expect, it } from "vitest";
import { DEFAULT_STATE, parseAppState, toSearch } from "./urlState";

/**
 * A URL is untrusted input, and this one is meant to be pasted into a submission and
 * handed to judges. Every field therefore has to survive being wrong: a view that does
 * not exist, a lead day outside 1-10, a band the filter does not offer. None of those may
 * throw or put the app in a state its own UI cannot get out of.
 */
describe("parseAppState", () => {
  it("gives the defaults for an empty query", () => {
    expect(parseAppState("")).toEqual(DEFAULT_STATE);
    expect(parseAppState("?")).toEqual(DEFAULT_STATE);
  });

  it("reads a full link", () => {
    expect(parseAppState("?view=alerts&day=5&region=IN-TN-CHENNAI&band=high")).toEqual({
      view: "alerts",
      leadDay: 5,
      region: "IN-TN-CHENNAI",
      band: "high",
    });
  });

  it("falls back to the default view rather than trusting the URL", () => {
    expect(parseAppState("?view=admin").view).toBe("live");
    expect(parseAppState("?view=").view).toBe("live");
  });

  it("clamps the lead day into the range the model actually serves", () => {
    expect(parseAppState("?day=0").leadDay).toBe(1);
    expect(parseAppState("?day=11").leadDay).toBe(10);
    expect(parseAppState("?day=-4").leadDay).toBe(1);
    expect(parseAppState("?day=abc").leadDay).toBe(1);
    expect(parseAppState("?day=3.7").leadDay).toBe(3);
  });

  it("only accepts the two bands the filter offers", () => {
    expect(parseAppState("?band=high").band).toBe("high");
    expect(parseAppState("?band=medium").band).toBe("medium");
    // "low" is a real risk band but not a filter - the Alerts page lists watch and bust.
    expect(parseAppState("?band=low").band).toBeUndefined();
    expect(parseAppState("?band=nonsense").band).toBeUndefined();
  });

  it("refuses a region id that is not shaped like one", () => {
    expect(parseAppState("?region=IN-LD-LAKSHADWEEP").region).toBe("IN-LD-LAKSHADWEEP");
    expect(parseAppState("?region=<script>").region).toBeNull();
    expect(parseAppState("?region=").region).toBeNull();
  });
});

describe("toSearch", () => {
  it("is empty when nothing differs from the default, so a plain visit stays clean", () => {
    expect(toSearch(DEFAULT_STATE)).toBe("");
  });

  it("writes only what differs", () => {
    expect(toSearch({ ...DEFAULT_STATE, view: "model" })).toBe("?view=model");
    expect(toSearch({ ...DEFAULT_STATE, leadDay: 7 })).toBe("?day=7");
  });

  it("round-trips a full state", () => {
    const state = { view: "alerts" as const, leadDay: 9, region: "IN-KL-ERNAKULAM", band: "medium" as const };
    expect(parseAppState(toSearch(state))).toEqual(state);
  });

  it("round-trips every view", () => {
    for (const view of ["live", "alerts", "model", "replay", "about"] as const) {
      expect(parseAppState(toSearch({ ...DEFAULT_STATE, view })).view).toBe(view);
    }
  });

  it("drops the region when there is none, rather than writing an empty parameter", () => {
    expect(toSearch({ ...DEFAULT_STATE, region: null })).toBe("");
  });
});
