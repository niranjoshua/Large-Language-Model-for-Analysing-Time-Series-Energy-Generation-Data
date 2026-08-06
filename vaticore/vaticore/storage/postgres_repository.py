"""Postgres / TimescaleDB backed time-series repository.

The production store, behind the same interface as the DuckDB repository. SQL is
standard Postgres, so it runs on plain Postgres or TimescaleDB; when the
TimescaleDB extension is present the observations table is promoted to a
hypertable for time-series scale, and when it is absent that step is skipped
silently so a plain Postgres deployment still works.

Install the driver with the postgres extra:  uv sync --extra postgres
"""

from __future__ import annotations

import pandas as pd

from vaticore import schemas
from vaticore.schemas import (
    GENERATION_KW,
    LOAD_KW,
    OPERATOR_ID,
    SITE_ID,
    TIMESTAMP,
)

_TABLE = "observations"
_COLUMNS = [OPERATOR_ID, SITE_ID, TIMESTAMP, LOAD_KW, GENERATION_KW]

_DDL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    {OPERATOR_ID}   TEXT             NOT NULL,
    {SITE_ID}       TEXT             NOT NULL,
    {TIMESTAMP}     TIMESTAMPTZ      NOT NULL,
    {LOAD_KW}       DOUBLE PRECISION,
    {GENERATION_KW} DOUBLE PRECISION,
    PRIMARY KEY ({OPERATOR_ID}, {SITE_ID}, {TIMESTAMP})
);
"""

_UPSERT = f"""
INSERT INTO {_TABLE} ({", ".join(_COLUMNS)})
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT ({OPERATOR_ID}, {SITE_ID}, {TIMESTAMP}) DO UPDATE SET
    {LOAD_KW} = EXCLUDED.{LOAD_KW},
    {GENERATION_KW} = EXCLUDED.{GENERATION_KW}
"""


class PostgresRepository:
    """Store and retrieve validated site time series in Postgres/TimescaleDB."""

    def __init__(self, url: str, *, use_timescale: bool = True) -> None:
        import psycopg

        # Normalise a SQLAlchemy style URL to what psycopg expects.
        dsn = url.replace("postgresql+psycopg://", "postgresql://")
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute("SET TIME ZONE 'UTC'")
        self._conn.execute(_DDL)
        if use_timescale:
            self._maybe_hypertable()

    def _maybe_hypertable(self) -> None:
        """Promote to a TimescaleDB hypertable when the extension is available."""
        try:
            self._conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
            self._conn.execute(
                f"SELECT create_hypertable('{_TABLE}', '{TIMESTAMP}', "
                "if_not_exists => TRUE, migrate_data => TRUE)"
            )
        except Exception:
            # Plain Postgres without TimescaleDB: the standard table is fine.
            pass

    def upsert(self, frame: pd.DataFrame) -> int:
        validated = schemas.validate(frame)[_COLUMNS]
        rows = _to_rows(validated)
        with self._conn.cursor() as cur:
            cur.executemany(_UPSERT, rows)
        return len(rows)

    def read_history(
        self,
        operator_id: str,
        site_id: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        query = (
            f"SELECT {', '.join(_COLUMNS)} FROM {_TABLE} "
            f"WHERE {OPERATOR_ID} = %s AND {SITE_ID} = %s"
        )
        params: list[object] = [operator_id, site_id]
        if start is not None:
            query += f" AND {TIMESTAMP} >= %s"
            params.append(pd.Timestamp(start).to_pydatetime())
        if end is not None:
            query += f" AND {TIMESTAMP} <= %s"
            params.append(pd.Timestamp(end).to_pydatetime())
        query += f" ORDER BY {TIMESTAMP}"
        return self._query(query, params)

    def read_fleet(self) -> pd.DataFrame:
        query = (
            f"SELECT {', '.join(_COLUMNS)} FROM {_TABLE} "
            f"ORDER BY {OPERATOR_ID}, {SITE_ID}, {TIMESTAMP}"
        )
        return self._query(query, [])

    def list_sites(self) -> pd.DataFrame:
        query = f"""
            SELECT {OPERATOR_ID}, {SITE_ID},
                   COUNT(*) AS n_observations,
                   MIN({TIMESTAMP}) AS first_timestamp,
                   MAX({TIMESTAMP}) AS last_timestamp
            FROM {_TABLE}
            GROUP BY {OPERATOR_ID}, {SITE_ID}
            ORDER BY {OPERATOR_ID}, {SITE_ID}
        """
        with self._conn.cursor() as cur:
            cur.execute(query)
            names = [d.name for d in (cur.description or [])]
            return pd.DataFrame(cur.fetchall(), columns=names)

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {_TABLE}")
            row = cur.fetchone()
        return int(row[0]) if row is not None else 0

    def close(self) -> None:
        self._conn.close()

    # -- internals ---------------------------------------------------------

    def _query(self, query: str, params: list[object]) -> pd.DataFrame:
        with self._conn.cursor() as cur:
            cur.execute(query, params)
            names = [d.name for d in (cur.description or [])]
            rows = cur.fetchall()
        frame = pd.DataFrame(rows, columns=names)
        if frame.empty:
            frame = pd.DataFrame({c: [] for c in _COLUMNS})
        return schemas.validate(frame)


def _to_rows(frame: pd.DataFrame) -> list[tuple]:
    """Convert a validated frame to psycopg parameter tuples, NaN to None."""
    prepared = frame.copy()
    prepared[TIMESTAMP] = pd.DatetimeIndex(prepared[TIMESTAMP]).to_pydatetime()
    prepared = prepared.astype(object).where(pd.notna(prepared), None)
    return [tuple(row) for row in prepared[_COLUMNS].to_numpy()]
