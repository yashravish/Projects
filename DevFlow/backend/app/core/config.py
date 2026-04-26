from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "DevFlow API"
    debug: bool = False
    database_url: str = Field(
        default="postgresql+asyncpg://devflow:devflow@localhost:5432/devflow",
        description="Async SQLAlchemy database URL",
    )
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    canary_rollback_error_rate: float = Field(
        default=0.15, description="Simulated error rate above this triggers auto-rollback"
    )
    pipeline_simulation_delay_ms: int = Field(
        default=8, description="Base delay per pipeline stage in milliseconds"
    )

    @field_validator("database_url", mode="before")
    @classmethod
    def _strip_database_url(cls, v: str) -> str:
        if isinstance(v, str) and v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql+asyncpg://", 1)
        return v

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def _empty_openai(cls, v: str | None) -> str | None:
        if v == "":
            return None
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
