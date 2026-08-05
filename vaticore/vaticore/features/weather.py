"""Weather enrichment.

Weather is a required feature source: solar generation and load both depend on
it. This wraps a provider (Open-Meteo to start) behind a small interface so the
rest of the system does not care which provider is live. Implementation lands
in build task 3.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class WeatherProvider(Protocol):
    """Interface every weather source implements."""

    def hourly(
        self,
        latitude: float,
        longitude: float,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Return hourly weather indexed by timezone aware UTC timestamp."""
        ...


class OpenMeteoProvider:
    """Open-Meteo client. Needs no API key to start.

    TODO(build task 3): call the Open-Meteo archive and forecast endpoints with
    httpx, normalise timestamps to UTC, and return temperature, cloud cover and
    shortwave radiation. Kept as a typed stub so callers can wire against the
    interface now.
    """

    def hourly(
        self,
        latitude: float,
        longitude: float,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        raise NotImplementedError("Open-Meteo integration lands in build task 3")
