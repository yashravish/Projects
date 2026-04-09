from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:postgres@localhost:5432/processpilot"
    jwt_secret: str = "dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 480
    openai_api_key: str = ""
    ai_provider: str = "auto"
    app_env: str = "development"
    log_level: str = "INFO"
    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
