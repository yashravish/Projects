from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg2://vendorguard:vendorguard@localhost:5432/vendorguard"
    secret_key: str = "change-me-to-a-random-secret-key-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 480
    admin_username: str = "admin"
    admin_password: str = "admin123"
    admin_email: str = "admin@vendorguard.local"
    ai_enabled: bool = False
    openai_api_key: str = ""
    openai_model: str = "gpt-4"
    log_level: str = "INFO"
    environment: str = "development"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
