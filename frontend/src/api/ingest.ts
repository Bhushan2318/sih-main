import { apiGet, apiPostJson } from "./client";
import type { IngestRunRow, IngestStatus } from "./types";

/** Live-feed health: which cycle is actually loaded, and when it arrived. */
export const fetchIngestStatus = () => apiGet<IngestStatus>("/api/ingest/status");

/** "Refresh now" - pulls the newest published GEFS cycle in the background. */
export const triggerCycleRun = () =>
  apiPostJson<{ status: string; target: string; note?: string }>("/api/ingest/run-cycle", {});

/** The pipeline's own history - every attempt, including refusals. */
export const fetchIngestRuns = (limit = 25) =>
  apiGet<{ runs: IngestRunRow[] }>(`/api/ingest/runs?limit=${limit}`);
