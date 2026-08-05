"""CSV ingestion into the internal schema.

This is the boundary where real operator data will hurt: malformed rows,
missing intervals, wrong timezones, duplicate timestamps. Everything here is
deliberately explicit. Timestamps are normalised to timezone aware UTC, gaps
are detected and reported rather than silently filled, and the result is
validated against the internal schema before it is allowed downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from vaticore import schemas
from vaticore.schemas import (
    GENERATION_KW,
    LOAD_KW,
    OPERATOR_ID,
    SITE_ID,
    TIMESTAMP,
)


@dataclass(frozen=True)
class GapReport:
    """Summary of missing intervals for one site at its native resolution."""

    operator_id: str
    site_id: str
    resolution: pd.Timedelta
    n_expected: int
    n_present: int
    n_missing: int
    max_gap: pd.Timedelta

    @property
    def completeness(self) -> float:
        return self.n_present / self.n_expected if self.n_expected else 1.0


def normalise(
    frame: pd.DataFrame,
    *,
    source_timezone: str | None = None,
) -> pd.DataFrame:
    """Coerce a raw frame to the internal schema.

    Parameters
    ----------
    source_timezone:
        Olson name of the timezone naive input, for example "Africa/Lagos". If
        the input timestamps are already timezone aware this is ignored. If they
        are naive and this is None, the timestamps are assumed to be UTC.
    """
    out = frame.copy()

    missing = {OPERATOR_ID, SITE_ID, TIMESTAMP} - set(out.columns)
    if missing:
        raise ValueError(f"input is missing required columns: {sorted(missing)}")

    ts = pd.to_datetime(out[TIMESTAMP], errors="raise", utc=False)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(source_timezone or "UTC")
    out[TIMESTAMP] = ts.dt.tz_convert("UTC")

    for col in (LOAD_KW, GENERATION_KW):
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out[OPERATOR_ID] = out[OPERATOR_ID].astype("string")
    out[SITE_ID] = out[SITE_ID].astype("string")

    # Duplicate timestamps within a site are a data error at the boundary.
    dupes = out.duplicated(subset=[OPERATOR_ID, SITE_ID, TIMESTAMP])
    if dupes.any():
        raise ValueError(f"{int(dupes.sum())} duplicate (operator, site, timestamp) rows")

    out = out.sort_values([OPERATOR_ID, SITE_ID, TIMESTAMP]).reset_index(drop=True)
    return schemas.validate(out)


def load_csv(
    path: str | Path,
    *,
    source_timezone: str | None = None,
    column_map: dict[str, str] | None = None,
    read_csv_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Load and validate a CSV of site load and generation.

    column_map renames operator specific headers onto the internal names before
    validation, for example {"kW": "load_kw", "ts": "timestamp"}.
    """
    raw = pd.read_csv(path, **(read_csv_kwargs or {}))
    if column_map:
        raw = raw.rename(columns=column_map)
    return normalise(raw, source_timezone=source_timezone)


def report_gaps(frame: pd.DataFrame, resolution: pd.Timedelta | None = None) -> list[GapReport]:
    """Detect and summarise missing intervals per site.

    Resolution is inferred per site from the median timestamp spacing when not
    given. Gaps are reported, not filled: filling strategy is a modelling
    decision made downstream, not silently at ingestion.
    """
    reports: list[GapReport] = []
    for (operator_id, site_id), group in frame.groupby([OPERATOR_ID, SITE_ID], sort=True):
        ts = pd.DatetimeIndex(group[TIMESTAMP]).sort_values()
        if len(ts) < 2:
            continue
        res = resolution or pd.Timedelta(ts.to_series().diff().dropna().median())
        expected_index = pd.date_range(ts[0], ts[-1], freq=res)
        n_expected = len(expected_index)
        present = ts.intersection(expected_index)
        max_gap = ts.to_series().diff().max()
        reports.append(
            GapReport(
                operator_id=str(operator_id),
                site_id=str(site_id),
                resolution=res,
                n_expected=n_expected,
                n_present=len(present),
                n_missing=max(0, n_expected - len(present)),
                max_gap=pd.Timedelta(max_gap),
            )
        )
    return reports
