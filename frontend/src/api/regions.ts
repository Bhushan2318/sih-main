import { apiGet } from "./client";
import type { RegionDetailResponse, RegionsResponse } from "./types";

export const fetchRegions = (leadTimeDays: number) =>
  apiGet<RegionsResponse>(`/api/regions?lead_time_days=${leadTimeDays}`);

export const fetchRegionDetail = (regionId: string) =>
  apiGet<RegionDetailResponse>(`/api/regions/${encodeURIComponent(regionId)}`);
