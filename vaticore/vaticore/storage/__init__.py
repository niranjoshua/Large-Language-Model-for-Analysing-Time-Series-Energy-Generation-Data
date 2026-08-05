"""Multi tenant, multi site persistence."""

from __future__ import annotations

from vaticore.config import Settings, get_settings
from vaticore.storage.duckdb_repository import DuckDBRepository
from vaticore.storage.repository import TimeSeriesRepository

__all__ = ["DuckDBRepository", "TimeSeriesRepository", "get_repository"]


def get_repository(settings: Settings | None = None) -> DuckDBRepository:
    """Build the configured repository from settings.

    DuckDB is supported today. Postgres/TimescaleDB is the production swap and
    lands behind the same interface; a clear error is raised until then so
    misconfiguration fails loudly rather than silently using the wrong store.
    """
    settings = settings or get_settings()
    url = settings.database_url
    if url.startswith("duckdb"):
        return DuckDBRepository(url)
    raise NotImplementedError(
        f"database_url scheme not supported yet: {url!r}. "
        "DuckDB is implemented; Postgres/TimescaleDB is the next backend."
    )
