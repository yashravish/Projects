import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_name: str = "Clinical Imaging QA Lab"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://ciqalab:ciqalab_pass@localhost:5432/ciqalab"
    )
    device_simulator_url: str = os.getenv(
        "DEVICE_SIMULATOR_URL",
        "http://localhost:8001"
    )
    frontend_origin: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:8080")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
