"""Multi tenant, multi site persistence."""

from __future__ import annotations

from vaticore.config import Settings, get_settings
from vaticore.storage.duckdb_repository import DuckDBRepository
from vaticore.storage.repository import TimeSeriesRepository

__all__ = ["DuckDBRepository", "TimeSeriesRepository", "get_repository"]


def get_repository(settings: Settings | None = None) -> TimeSeriesRepository:
    """Build the configured repository from settings.

    Selects the backend from the database URL scheme: DuckDB for local and small
    deployments, Postgres/TimescaleDB for production. Both satisfy the same
    contract, so callers never change.
    """
    settings = settings or get_settings()
    url = settings.database_url
    if url.startswith("duckdb"):
        return DuckDBRepository(url)
    if url.startswith("postgres"):
        from vaticore.storage.postgres_repository import PostgresRepository

        return PostgresRepository(url)
    raise ValueError(
        f"unsupported database_url scheme: {url!r}. Use a duckdb:// or postgresql:// URL."
    )
