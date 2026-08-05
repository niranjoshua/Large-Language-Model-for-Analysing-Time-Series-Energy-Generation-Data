"""Typed, environment driven settings.

All configuration flows through here so behaviour is reproducible and no module
reaches for os.environ directly. Values load from the process environment and
from a local .env file (see .env.example).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings.

    Secrets and deployment specific values belong here, never hard coded.
    """

    model_config = SettingsConfigDict(
        env_prefix="VATICORE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Runtime
    environment: str = Field(default="local", description="local, staging or production.")
    log_level: str = Field(default="INFO")

    # Storage. DuckDB is fine for local development and early backtests.
    # Point this at Postgres/TimescaleDB in staging and production.
    database_url: str = Field(default="duckdb:///vaticore.duckdb")

    # Weather provider. Open-Meteo needs no key to start.
    weather_provider: str = Field(default="open-meteo")
    weather_api_key: str | None = Field(default=None)

    # Experiment tracking.
    mlflow_tracking_uri: str | None = Field(default=None)

    # Default quantiles produced across the system.
    default_quantiles: tuple[float, ...] = Field(default=(0.1, 0.5, 0.9))


def get_settings() -> Settings:
    """Return a fresh Settings instance.

    Kept as a function (not a module level singleton) so tests can override the
    environment and construct isolated settings.
    """
    return Settings()


PROJECT_ROOT = Path(__file__).resolve().parent.parent
