"""Forecast real solar resource data from Open-Meteo.

This is real data, no customer required. It pulls a year of hourly shortwave
radiation (the dominant driver of solar generation) for a real location, then
backtests the quantile GBM model against the persistence baseline and reports
the skill improvement. The number this prints is a genuine, defensible result
you can put in a pitch deck or grant application.

Requires outbound HTTPS to open-meteo.com (no API key). Run it where the network
is open:

    uv run python examples/real_solar_forecast.py
"""

from __future__ import annotations

import warnings

import pandas as pd

from vaticore.engine import run_backtest
from vaticore.features import OpenMeteoProvider

# Lagos, Nigeria: a representative mini-grid and C&I solar market.
LATITUDE, LONGITUDE = 6.45, 3.40
TARGET = "shortwave_radiation"


def main() -> None:
    warnings.filterwarnings("ignore")

    provider = OpenMeteoProvider()
    print(f"Fetching one year of hourly weather for ({LATITUDE}, {LONGITUDE})...")
    weather = provider.hourly(
        LATITUDE,
        LONGITUDE,
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-12-31"),
    )
    provider.close()
    print(f"Got {len(weather)} hourly observations of {TARGET}.")

    print("\nBacktesting quantile GBM against persistence on real solar data...")
    result = run_backtest(
        weather,
        TARGET,
        horizon=24,
        initial=24 * 300,
        step=24 * 7,
        model="quantile_gbm",
    )
    summary = result.summary()
    print(summary.round(3).to_string())

    gbm = summary.loc["quantile_gbm", "pinball"]
    base = summary.loc["persistence", "pinball"]
    print(f"\nReal-data pinball improvement over baseline: {(1 - gbm / base) * 100:.1f}%")
    print("This is a genuine result on real solar resource data.")


if __name__ == "__main__":
    main()
