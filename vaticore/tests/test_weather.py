from __future__ import annotations

import httpx
import pandas as pd
import pytest

from vaticore.features import OpenMeteoProvider, join_weather
from vaticore.schemas import TIMESTAMP

# A canned Open-Meteo archive response, so the client is tested without network.
_FAKE_RESPONSE = {
    "latitude": 6.45,
    "longitude": 3.40,
    "hourly": {
        "time": ["2024-01-01T00:00", "2024-01-01T01:00", "2024-01-01T02:00"],
        "temperature_2m": [24.1, 23.8, 23.5],
        "shortwave_radiation": [0.0, 0.0, 0.0],
        "cloud_cover": [40, 55, 60],
    },
}


def _mock_client() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "archive-api.open-meteo.com" in str(request.url)
        assert request.url.params["timezone"] == "UTC"
        return httpx.Response(200, json=_FAKE_RESPONSE)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_hourly_parses_to_utc_frame() -> None:
    provider = OpenMeteoProvider(client=_mock_client())
    frame = provider.hourly(6.45, 3.40, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01"))
    assert len(frame) == 3
    assert str(frame[TIMESTAMP].dtype) == "datetime64[ns, UTC]"
    assert list(frame.columns) == [
        TIMESTAMP,
        "temperature_2m",
        "shortwave_radiation",
        "cloud_cover",
    ]
    assert frame["temperature_2m"].iloc[0] == pytest.approx(24.1)


def test_http_error_propagates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    provider = OpenMeteoProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(httpx.HTTPStatusError):
        provider.hourly(6.45, 3.40, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01"))


def test_join_weather_aligns_on_timestamp() -> None:
    provider = OpenMeteoProvider(client=_mock_client())
    weather = provider.hourly(6.45, 3.40, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01"))
    site = pd.DataFrame(
        {
            TIMESTAMP: pd.to_datetime(["2024-01-01T00:00", "2024-01-01T01:00"], utc=True),
            "load_kw": [100.0, 105.0],
        }
    )
    merged = join_weather(site, weather)
    assert len(merged) == 2
    assert "shortwave_radiation" in merged.columns
    assert merged["temperature_2m"].notna().all()
