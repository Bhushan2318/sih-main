import { apiGet, apiPostJson } from "./client";
import type { IngestStatus } from "./types";

/** Live-feed health: which cycle is actually loaded, and when it arrived. */
export const fetchIngestStatus = () => apiGet<IngestStatus>("/api/ingest/status");

/** "Refresh now" - pulls the newest published GEFS cycle in the background. */
export const triggerCycleRun = () =>
  apiPostJson<{ status: string; target: string; note?: string }>("/api/ingest/run-cycle", {});
