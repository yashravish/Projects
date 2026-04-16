from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────────
    APP_NAME: str = "Enterprise Integration Gateway"
    APP_VERSION: str = "2.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"

    # ── Database ───────────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql://eig_user:eig_password@localhost:5432/eig_db"
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10

    # ── Mock Provider URLs ─────────────────────────────────────────────────────
    CRM_BASE_URL: str = "http://localhost:8001"
    VENDOR_BASE_URL: str = "http://localhost:8001"

    # ── HTTP Client ────────────────────────────────────────────────────────────
    HTTP_TIMEOUT_SECONDS: int = 30
    HTTP_MAX_RETRIES: int = 3
    HTTP_RETRY_BACKOFF_FACTOR: float = 0.5

    # ── Scheduler ─────────────────────────────────────────────────────────────
    SCHEDULER_ENABLED: bool = True
    CRM_SYNC_INTERVAL_MINUTES: int = 15
    VENDOR_SYNC_INTERVAL_MINUTES: int = 15

    # ── Failure Handling ───────────────────────────────────────────────────────
    MAX_RETRY_COUNT: int = 3

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # ── CORS ───────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: str = "*"

    # ── Redis ──────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_ENABLED: bool = True
    CACHE_TTL_SECONDS: int = 60
    RATE_LIMIT_RPM: int = 30

    # ── Kafka ──────────────────────────────────────────────────────────────────
    KAFKA_ENABLED: bool = True
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_EVENTS_TOPIC: str = "eig.integration.events"
    KAFKA_INBOUND_TOPIC: str = "eig.inbound.sync.requests"
    KAFKA_CONSUMER_GROUP: str = "eig-gateway-group"

    def get_allowed_origins(self) -> List[str]:
        if self.ALLOWED_ORIGINS == "*":
            return ["*"]
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]


settings = Settings()
