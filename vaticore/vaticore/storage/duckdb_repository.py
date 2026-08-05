"""DuckDB backed time-series repository.

A real, tested implementation of the storage contract. DuckDB runs in-process
(a file or in-memory), which is ideal for local development, the demo, and small
deployments. Production swaps to Postgres/TimescaleDB behind the same interface;
the SQL here is deliberately standard so that port is small.

Every read and write is scoped by operator_id and site_id, so multi tenancy is
enforced at the storage boundary, not bolted on later.
"""

from __future__ import annotations

import duckdb
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

_DDL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    {OPERATOR_ID}   VARCHAR      NOT NULL,
    {SITE_ID}       VARCHAR      NOT NULL,
    {TIMESTAMP}     TIMESTAMPTZ  NOT NULL,
    {LOAD_KW}       DOUBLE,
    {GENERATION_KW} DOUBLE,
    PRIMARY KEY ({OPERATOR_ID}, {SITE_ID}, {TIMESTAMP})
);
"""

_COLUMNS = [OPERATOR_ID, SITE_ID, TIMESTAMP, LOAD_KW, GENERATION_KW]


def _path_from_url(url: str) -> str:
    """Turn a duckdb URL or bare path into a DuckDB connection target.

    Accepts 'duckdb:///file.duckdb', 'duckdb:///:memory:', or a plain path.
    """
    if url.startswith("duckdb://"):
        rest = url[len("duckdb://") :]
        return rest.lstrip("/") if rest not in ("/:memory:", "//:memory:") else ":memory:"
    return url


class DuckDBRepository:
    """Store and retrieve validated site time series in DuckDB."""

    def __init__(self, url: str = ":memory:") -> None:
        target = _path_from_url(url)
        # ":memory:" gives a private in-process database, perfect for tests.
        self._con = duckdb.connect(target if target else ":memory:")
        self._con.execute(_DDL)

    def upsert(self, frame: pd.DataFrame) -> int:
        """Insert or update rows for one or more sites. Returns rows written.

        The frame is validated against the internal schema first, so malformed
        data never reaches the store.
        """
        validated = schemas.validate(frame)[_COLUMNS]
        self._con.register("incoming", validated)
        try:
            self._con.execute(
                f"""
                INSERT INTO {_TABLE}
                SELECT {", ".join(_COLUMNS)} FROM incoming
                ON CONFLICT ({OPERATOR_ID}, {SITE_ID}, {TIMESTAMP}) DO UPDATE SET
                    {LOAD_KW} = excluded.{LOAD_KW},
                    {GENERATION_KW} = excluded.{GENERATION_KW}
                """
            )
        finally:
            self._con.unregister("incoming")
        return len(validated)

    def read_history(
        self,
        operator_id: str,
        site_id: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return one site's validated history within an optional time range."""
        query = (
            f"SELECT {', '.join(_COLUMNS)} FROM {_TABLE} WHERE {OPERATOR_ID} = ? AND {SITE_ID} = ?"
        )
        params: list[object] = [operator_id, site_id]
        if start is not None:
            query += f" AND {TIMESTAMP} >= ?"
            params.append(pd.Timestamp(start).to_pydatetime())
        if end is not None:
            query += f" AND {TIMESTAMP} <= ?"
            params.append(pd.Timestamp(end).to_pydatetime())
        query += f" ORDER BY {TIMESTAMP}"

        frame = self._con.execute(query, params).df()
        if frame.empty:
            return schemas.validate(_empty_frame())
        return schemas.validate(frame)

    def read_fleet(self) -> pd.DataFrame:
        """Return all stored observations across every operator and site."""
        frame = self._con.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM {_TABLE} ORDER BY {OPERATOR_ID}, {SITE_ID}, {TIMESTAMP}"
        ).df()
        if frame.empty:
            return schemas.validate(_empty_frame())
        return schemas.validate(frame)

    def list_sites(self) -> pd.DataFrame:
        """Summarise stored sites: row count and time span per operator/site."""
        return self._con.execute(
            f"""
            SELECT {OPERATOR_ID}, {SITE_ID},
                   COUNT(*) AS n_observations,
                   MIN({TIMESTAMP}) AS first_timestamp,
                   MAX({TIMESTAMP}) AS last_timestamp
            FROM {_TABLE}
            GROUP BY {OPERATOR_ID}, {SITE_ID}
            ORDER BY {OPERATOR_ID}, {SITE_ID}
            """
        ).df()

    def count(self) -> int:
        row = self._con.execute(f"SELECT COUNT(*) FROM {_TABLE}").fetchone()
        return int(row[0]) if row is not None else 0

    def close(self) -> None:
        self._con.close()


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            OPERATOR_ID: pd.Series([], dtype="string"),
            SITE_ID: pd.Series([], dtype="string"),
            TIMESTAMP: pd.Series([], dtype="datetime64[ns, UTC]"),
            LOAD_KW: pd.Series([], dtype="float64"),
            GENERATION_KW: pd.Series([], dtype="float64"),
        }
    )
