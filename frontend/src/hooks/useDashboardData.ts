import { useQuery } from "@tanstack/react-query";
import { fetchAlerts } from "../api/alerts";
import { fetchIngestStatus } from "../api/ingest";
import { fetchModelStatus } from "../api/modelStatus";
import { fetchAllRegions, fetchRegionDetail, fetchRegions } from "../api/regions";
import { fetchReplay, fetchReplayCycles } from "../api/replay";
import type { RiskBand } from "../api/types";

// A retrain lands via the WebSocket, so polling is only a safety net for a missed frame.
const SAFETY_REFETCH_MS = 60_000;

export const useRegions = (leadTimeDays: number) =>
  useQuery({
    queryKey: ["regions", leadTimeDays],
    queryFn: () => fetchRegions(leadTimeDays),
    refetchInterval: SAFETY_REFETCH_MS,
  });

// All 10 lead days in one payload so the lead-day selector switches with zero requests.
// A scored cycle only changes on retrain / ingest, and the WebSocket invalidates
// ["regions"] then - so there is nothing to poll for and staleTime can be generous.
export const useAllRegions = () =>
  useQuery({
    queryKey: ["regions", "all"],
    queryFn: fetchAllRegions,
    staleTime: 5 * 60_000,
    refetchInterval: 5 * 60_000,
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

// Guided replay. A cycle's scored numbers never change unless the model is retrained, so
// there is nothing to poll - the WebSocket retrain event invalidates the cache instead.
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
  });

// Feed freshness. Polled a little faster than the rest: a cycle landing is the one change
// the WebSocket cannot always attribute to a query key.
export const useIngestStatus = () =>
  useQuery({
    queryKey: ["ingestStatus"],
    queryFn: fetchIngestStatus,
    refetchInterval: 30_000,
    // The feed is optional: a deployment with live ingestion switched off should render
    // the dashboard normally rather than surfacing an error banner.
    retry: false,
  });
