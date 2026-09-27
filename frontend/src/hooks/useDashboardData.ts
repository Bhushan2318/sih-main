import { useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { fetchAlerts } from "../api/alerts";
import { fetchEnsembleDivergence } from "../api/ensemble";
import { fetchIngestRuns, fetchIngestStatus } from "../api/ingest";
import { fetchModelStatus } from "../api/modelStatus";
import { fetchAllRegions, fetchRegionDetail } from "../api/regions";
import { fetchReplay, fetchReplayCycles } from "../api/replay";
import type { RiskBand } from "../api/types";
import { replayPrefetchOrder } from "../lib/replayPrefetch";

const SAFETY_REFETCH_MS = 60_000;

export const useAllRegions = () =>
  useQuery({
    queryKey: ["regions", "all"],
    queryFn: fetchAllRegions,
    staleTime: 5 * 60_000,
    refetchInterval: 5 * 60_000,
  });

export const useEnsembleDivergence = (regionId?: string | null) =>
  useQuery({
    queryKey: ["ensemble", "divergence", regionId ?? "auto"],
    queryFn: () => fetchEnsembleDivergence(regionId),
    staleTime: 5 * 60_000,
    placeholderData: (prev) => prev,
  });

export const useRegionDetail = (regionId: string | null) =>
  useQuery({
    queryKey: ["regions", "detail", regionId],
    queryFn: () => fetchRegionDetail(regionId as string),
    enabled: Boolean(regionId),
  });

export const useAlerts = (limit = 25, riskBand?: RiskBand) =>
  useQuery({
    queryKey: ["alerts", limit, riskBand ?? null],
    queryFn: () => fetchAlerts(limit, riskBand),
    refetchInterval: SAFETY_REFETCH_MS,
  });

export const useModelStatus = () =>
  useQuery({
    queryKey: ["modelStatus"],
    queryFn: fetchModelStatus,
    refetchInterval: SAFETY_REFETCH_MS,
  });

export const useReplayCycles = (enabled: boolean) =>
  useQuery({
    queryKey: ["replay", "cycles"],
    queryFn: fetchReplayCycles,
    enabled,
    staleTime: Infinity,
  });

export const useReplay = (initDate: string | undefined, enabled: boolean) =>
  useQuery({
    queryKey: ["replay", initDate ?? "default"],
    queryFn: () => fetchReplay(initDate),
    enabled,
    staleTime: Infinity,
    // A scored cycle does not change within a session, so keep it: switching back to a
    // cycle seen five minutes ago should not download it again.
    gcTime: Infinity,
  });

/**
 * Fetch Replay's cycles before they are clicked, one at a time, so opening the tab and
 * switching between cycles is instant rather than a wait on the network.
 *
 * `which` is null until there is something to fetch for: the dashboard passes "events"
 * once its own opening screen has loaded, and Replay passes "all" once its first cycle is
 * on screen. A failure is ignored - this is only a head start, and each view still fetches
 * whatever it needs itself.
 */
export function usePrefetchReplay(which: "events" | "all" | null) {
  const qc = useQueryClient();
  useEffect(() => {
    if (!which) return;
    let cancelled = false;
    qc.setQueryDefaults(["replay"], { staleTime: Infinity, gcTime: Infinity });
    (async () => {
      const cycles = await qc.fetchQuery({ queryKey: ["replay", "cycles"], queryFn: fetchReplayCycles });
      if (cancelled) return;
      // Replay opens on its default cycle. Filed under its own date as well, so picking
      // that date from the dropdown is not a second download of the same answer.
      const opening = await qc.fetchQuery({ queryKey: ["replay", "default"], queryFn: () => fetchReplay() });
      if (opening.init_date && qc.getQueryData(["replay", opening.init_date]) === undefined) {
        qc.setQueryData(["replay", opening.init_date], opening);
      }
      const order = replayPrefetchOrder(
        cycles, which, (d) => qc.getQueryData(["replay", d]) !== undefined);
      for (const d of order) {
        if (cancelled) return;
        await qc.prefetchQuery({ queryKey: ["replay", d], queryFn: () => fetchReplay(d) });
      }
    })().catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [which, qc]);
}

export const useIngestStatus = () =>
  useQuery({
    queryKey: ["ingestStatus"],
    queryFn: fetchIngestStatus,
    refetchInterval: 30_000,

    retry: false,
  });

export const useIngestRuns = (limit = 25) =>
  useQuery({
    queryKey: ["ingestRuns", limit],
    queryFn: () => fetchIngestRuns(limit),
    refetchInterval: 60_000,
    retry: false,
  });
