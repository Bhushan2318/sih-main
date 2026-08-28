import { useQuery } from "@tanstack/react-query";
import { fetchAlerts } from "../api/alerts";
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
