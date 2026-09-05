"""Centralized application configuration."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables and an optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    postgres_db: str = "pneumonia"
    postgres_user: str = "pneumonia"
    postgres_password: str = Field(default="pneumonia-local", repr=False)
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    minio_endpoint: str = "http://localhost:9000"
    minio_bucket: str = "pneumonia"
    minio_root_user: str = "minioadmin"
    minio_root_password: str = Field(default="minioadmin-local", repr=False)

    mlflow_tracking_uri: str = "http://localhost:5000"

    @property
    def database_url(self) -> str:
        """Return a PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
