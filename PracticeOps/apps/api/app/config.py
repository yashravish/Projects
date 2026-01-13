"""Application configuration using Pydantic Settings."""

import secrets
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    database_url: str = "postgresql+asyncpg://practiceops:practiceops@localhost:5433/practiceops"

    @field_validator("database_url", mode="before")
    @classmethod
    def convert_database_url(cls, v: Any) -> str:
        """Convert Railway's postgresql:// to postgresql+asyncpg:// format."""
        if isinstance(v, str) and v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    # Environment
    environment: str = "development"

    # CORS - accepts comma-separated string from env var
    # Use str type to prevent Pydantic from trying to parse as JSON
    cors_origins: str | list[str] = "http://localhost:5173,http://127.0.0.1:5173"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        """Parse CORS origins from comma-separated string, JSON array, or list.

        Handles:
        - Comma-separated string: "url1,url2,url3"
        - Single URL string: "url1"
        - List: ["url1", "url2"]
        """
        if isinstance(v, str):
            # Split by comma and filter empty strings
            origins = [origin.strip() for origin in v.split(",") if origin.strip()]
            return origins if origins else ["http://localhost:5173"]
        if isinstance(v, list):
            return v
        # Fallback to default
        return ["http://localhost:5173", "http://127.0.0.1:5173"]

    # Frontend URL (for invite links)
    frontend_url: str = "http://localhost:5173"

    # JWT Configuration
    jwt_secret_key: str = secrets.token_urlsafe(32)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 30

    # SMTP Configuration (for production email)
    # Supports standard SMTP or Resend API
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_pass: str | None = None
    smtp_from: str = "noreply@practiceops.app"
    
    # Resend API (alternative to SMTP - https://resend.com)
    resend_api_key: str | None = None

    @property
    def smtp_enabled(self) -> bool:
        """Check if SMTP is configured for production email."""
        return bool(self.smtp_host and self.smtp_user and self.smtp_pass)
    
    @property
    def resend_enabled(self) -> bool:
        """Check if Resend API is configured."""
        return bool(self.resend_api_key)
    
    @property
    def email_enabled(self) -> bool:
        """Check if any email provider is configured."""
        return self.smtp_enabled or self.resend_enabled

    # OpenAI (optional - for AI summaries)
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"


settings = Settings()
