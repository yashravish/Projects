"""Typed application configuration loaded from environment variables.

Uses pydantic-settings so that every value crossing the env boundary is
validated and typed. `get_settings()` is cached so the same instance is shared
across the process.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    Attributes are documented in `.env.example`. Required secrets that have no
    safe default (e.g. OPENAI_API_KEY) default to empty string and are checked
    by the components that need them — we don't want startup to fail in
    development just because a developer hasn't filled in every key.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Runtime ----------------------------------------------------------------
    env: Literal["development", "staging", "production", "test"] = "development"
    log_level: str = "INFO"
    seed_on_boot: bool = True
    allowed_origins: str = "http://localhost:5173"

    # Database ---------------------------------------------------------------
    database_url: str = (
        "postgresql+asyncpg://psdi:psdi_dev_password@postgres:5432/psdi"
    )
    database_url_sync: str = (
        "postgresql+psycopg://psdi:psdi_dev_password@postgres:5432/psdi"
    )

    # Redis ------------------------------------------------------------------
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/1"
    celery_result_backend: str = "redis://redis:6379/2"

    # MLflow -----------------------------------------------------------------
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "psdi"

    # OpenAI -----------------------------------------------------------------
    openai_api_key: str = ""
    openai_default_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # JWT --------------------------------------------------------------------
    jwt_secret: str = "change-me"
    jwt_private_key_path: str = ""
    jwt_public_key_path: str = ""
    jwt_access_ttl_minutes: int = 15
    jwt_refresh_ttl_days: int = 7
    jwt_issuer: str = "psdi"

    # Storage ----------------------------------------------------------------
    storage_backend: Literal["local", "s3"] = "local"
    local_upload_dir: str = "/data/uploads"
    s3_bucket: str = ""
    s3_region: str = "us-east-1"
    s3_endpoint_url: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    # Reranker ---------------------------------------------------------------
    reranker_backend: Literal["local", "sagemaker"] = "local"
    sagemaker_reranker_endpoint: str = ""
    sagemaker_role_arn: str = ""
    aws_region: str = "us-east-1"

    # ML model registry ------------------------------------------------------
    # Local directory the LocalModelRegistry writes to (mounted as a docker
    # volume in production). One subdirectory per logical model name, with
    # one nested subdirectory per version.
    models_dir: str = "/data/models"

    # Rate limits ------------------------------------------------------------
    rate_limit_query_per_min: int = 30
    rate_limit_upload_per_hour: int = 10

    # Seeded admin -----------------------------------------------------------
    seed_org_name: str = "Demo Agency"
    seed_org_slug: str = "demo-agency"
    seed_admin_email: str = "seed-admin@example.gov"
    seed_admin_password: str = "ChangeMe!2026"

    # ---- Validators --------------------------------------------------------

    @field_validator("allowed_origins")
    @classmethod
    def _no_wildcard_in_prod(cls, v: str, info: object) -> str:
        # Note: cross-field validation against `env` is done in __post_init__-style
        # at app startup (see app.main), where we have access to the full model.
        return v.strip()

    # ---- Derived helpers ---------------------------------------------------

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def jwt_uses_rs256(self) -> bool:
        return bool(self.jwt_private_key_path) and Path(self.jwt_private_key_path).is_file()

    @property
    def jwt_algorithm(self) -> str:
        return "RS256" if self.jwt_uses_rs256 else "HS256"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton."""
    return Settings()
