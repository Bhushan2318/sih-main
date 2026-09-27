import { describe, expect, it } from "vitest";
import { REAL_REPLAY_CYCLES } from "../test/fixtures/replayCycles";
import { replayPrefetchOrder } from "./replayPrefetch";

const nothingCached = () => false;

describe("replayPrefetchOrder", () => {
  it("fetches the past events first, in the order Replay lists them", () => {
    expect(replayPrefetchOrder(REAL_REPLAY_CYCLES, "events", nothingCached)).toEqual([
      "2018-08-13", "2017-08-27", "2017-11-28", "2019-05-01",
    ]);
  });

  it("adds the recent forecasts only once Replay is open", () => {
    const all = replayPrefetchOrder(REAL_REPLAY_CYCLES, "all", nothingCached);
    expect(all.slice(0, 4)).toEqual(["2018-08-13", "2017-08-27", "2017-11-28", "2019-05-01"]);
    expect(all.slice(4)).toEqual(["2026-09-25", "2026-09-23", "2026-09-24"]);
  });

  it("never fetches a cycle that is already cached", () => {
    const cached = new Set(["2018-08-13", "2026-09-23"]);
    const order = replayPrefetchOrder(REAL_REPLAY_CYCLES, "all", (d) => cached.has(d));
    expect(order).not.toContain("2018-08-13");
    expect(order).not.toContain("2026-09-23");
    expect(order).toHaveLength(REAL_REPLAY_CYCLES.length - 2);
  });
});
