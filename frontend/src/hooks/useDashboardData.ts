import { useQuery } from "@tanstack/react-query";
import { fetchAlerts } from "../api/alerts";
import { fetchIngestStatus } from "../api/ingest";
import { fetchModelStatus } from "../api/modelStatus";
import { fetchRegionDetail, fetchRegions } from "../api/regions";
import type { RiskBand } from "../api/types";

// A retrain lands via the WebSocket, so polling is only a safety net for a missed frame.
const SAFETY_REFETCH_MS = 60_000;

export const useRegions = (leadTimeDays: number) =>
  useQuery({
    queryKey: ["regions", leadTimeDays],
    queryFn: () => fetchRegions(leadTimeDays),
    refetchInterval: SAFETY_REFETCH_MS,
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
