"""Phase 6: automated live ingestion.

Pulls real operational GEFS forecast cycles and near-real-time / reanalysis observations
on a schedule, feeds them through the same ingestion pipeline as a manual upload, and
retrains when enough newly-verified observations have landed.

Nothing in here fabricates a value or a timestamp. When a cycle has not been published
yet, or a fetch fails, the run is recorded as skipped/failed and the dashboard keeps
showing the last cycle that genuinely exists.
"""
