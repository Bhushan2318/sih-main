import type { ReplayCycleSummary } from "../api/types";

/**
 * Which Replay cycles to fetch ahead of a click, in order.
 *
 * "events" is the past busts: what Replay opens on and what a walkthrough visits, so they
 * are fetched while the opening screen sits idle. "all" adds the recent forecasts too, once
 * Replay is actually open. A cycle already in the cache is never fetched twice.
 */
export function replayPrefetchOrder(
  cycles: ReplayCycleSummary[],
  which: "events" | "all",
  isCached: (initDate: string) => boolean,
): string[] {
  const events = cycles.filter((c) => c.kind === "event");
  const rest = which === "all" ? cycles.filter((c) => c.kind !== "event") : [];
  return [...events, ...rest].map((c) => c.init_date).filter((d) => !isCached(d));
}
