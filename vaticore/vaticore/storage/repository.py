"""Persistence interface, scoped by operator and site from day one.

Retrofitting tenant scoping later is painful, so every read and write takes an
operator_id and site_id here at the interface, even before a concrete backend
(DuckDB locally, Postgres/TimescaleDB in production) is wired in.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class TimeSeriesRepository(Protocol):
    """Storage contract for validated site time series."""

    def upsert(self, frame: pd.DataFrame) -> int:
        """Insert or update rows for one or more sites. Returns rows written."""
        ...

    def read_history(
        self,
        operator_id: str,
        site_id: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return one site's validated history within an optional time range."""
        ...
