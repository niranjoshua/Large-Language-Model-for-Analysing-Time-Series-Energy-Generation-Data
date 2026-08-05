"""Feature engineering: calendar, lags and weather enrichment."""

from vaticore.features.build import add_calendar_features, add_lag_features
from vaticore.features.weather import (
    OpenMeteoProvider,
    WeatherProvider,
    join_weather,
)

__all__ = [
    "OpenMeteoProvider",
    "WeatherProvider",
    "add_calendar_features",
    "add_lag_features",
    "join_weather",
]
