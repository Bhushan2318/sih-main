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

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
