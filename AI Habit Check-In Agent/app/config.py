from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    openai_api_key: str = "sk-test-key"
    openai_model: str = "gpt-4o-mini"
    database_url: str = "sqlite+aiosqlite:///./checkins.db"
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
