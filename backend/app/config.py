"""Backend configuration. Values are read from environment variables / a .env file
(see .env.example) so paths can be adjusted without touching code."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # protected_namespaces=() because `model_dir` below would otherwise collide with
    # pydantic's own reserved "model_" prefix (BaseModel.model_config etc.) and emit a
    # spurious warning on every import.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    data_dir: Path = Path("data")
    db_path: Path = Path("metadata.db")
    model_dir: Path = Path("data/models")
    raw_upload_dir: Path = Path("data/raw_uploads")
    canonical_dir: Path = Path("data/canonical")
    geo_dir: Path = Path("data/geo")

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ---- Phase 6: live ingestion -------------------------------------------------
    # The scheduler is OFF by default. A fresh clone, a test run and CI must not start
    # reaching out to S3 on import; switch it on deliberately (env or .env).
    live_ingest_enabled: bool = False
    # Cycles to pull, as UTC hours. All four carry the full Day 1-10 at 0.25 deg.
    live_gefs_cycles: str = "00,06,12,18"
    # A cycle is only attempted once this many hours have passed since its issue time -
    # GEFS takes ~4-5 h to finish computing and land on the bucket.
    live_gefs_publish_lag_hours: float = 5.5
    # Exactly the five members the models were trained on (see live/gefs.py for why
    # this must not be widened without retraining).
    live_gefs_members: str = "gec00,gep01,gep02,gep03,gep04"
    live_download_workers: int = 20
    # How far back the provisional observation refresh looks for newly-verifiable days.
    live_obs_provisional_days: int = 14
    # ERA5 final-verification window: re-pull this many days back so provisional rows
    # get overwritten once the reanalysis lands (~5 day latency).
    live_obs_final_days: int = 21
    # Minimum number of newly-final observation rows before an automatic retrain.
    live_retrain_min_new_rows: int = 500

    # Training peaks around 2.3 GB. The serving container has 512 MB and is killed, not
    # throttled, when it exceeds that - so training is refused unless a process is
    # explicitly declared as one that may train. Defaults to FALSE: a new deployment is
    # safe by default, and the paths that legitimately train opt in.
    allow_local_retrain: bool = False

    # ---- serving ------------------------------------------------------------------
    # Warm the guided-replay ranking and the ensemble view in a background thread at
    # startup, so the first request for either is instant. Worth it on a workstation.
    #
    # Turn it OFF on a memory-constrained host. The warm scores every historical cycle,
    # and if a real request lands while it is still running the box pays for two scoring
    # passes at once - which is what OOM-killed the 512 MB container even after the read
    # path was cut to fit a single pass. Cold means the first replay call is slow, not
    # that anything is missing.
    warm_caches_on_startup: bool = True

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def gefs_cycle_list(self) -> list[str]:
        return [c.strip().zfill(2) for c in self.live_gefs_cycles.split(",") if c.strip()]

    @property
    def gefs_member_list(self) -> list[str]:
        return [m.strip() for m in self.live_gefs_members.split(",") if m.strip()]


settings = Settings()
