from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    data_dir: Path = Path("data")
    db_path: Path = Path("metadata.db")
    model_dir: Path = Path("data/models")
    raw_upload_dir: Path = Path("data/raw_uploads")
    canonical_dir: Path = Path("data/canonical")
    geo_dir: Path = Path("data/geo")

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    live_ingest_enabled: bool = False
    live_gefs_cycles: str = "00,06,12,18"
    live_gefs_publish_lag_hours: float = 5.5
    # These five only: the models were trained on them and spread features would skew.
    live_gefs_members: str = "gec00,gep01,gep02,gep03,gep04"
    live_download_workers: int = 20
    live_obs_provisional_days: int = 14
    live_obs_final_days: int = 21
    live_retrain_min_new_rows: int = 500

    allow_local_retrain: bool = False

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
