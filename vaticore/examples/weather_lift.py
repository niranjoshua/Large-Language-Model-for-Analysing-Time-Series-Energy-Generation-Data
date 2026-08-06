"""Show how much weather features improve the solar generation forecast.

Self-contained (no network): it uses synthetic data whose generation genuinely
depends on radiation and cloud cover, then backtests the quantile GBM model with
and without weather features against the persistence baseline. The gap between
the two is the value weather adds.

    uv run python examples/weather_lift.py
"""

from __future__ import annotations

import warnings

from vaticore.datasets import make_synthetic_site
from vaticore.engine import run_backtest
from vaticore.schemas import GENERATION_KW

WEATHER = ("shortwave_radiation", "cloud_cover", "temperature_2m")


def main() -> None:
    warnings.filterwarnings("ignore")

    site = make_synthetic_site("demo-op", "demo-site", days=50, seed=5, with_weather=True)
    common = dict(horizon=24, initial=24 * 35, step=72, model="quantile_gbm")

    without = run_backtest(site, GENERATION_KW, **common).summary()
    with_weather = run_backtest(site, GENERATION_KW, exog=WEATHER, **common).summary()

    baseline = without.loc["persistence", "pinball"]
    plain = without.loc["quantile_gbm", "pinball"]
    weather = with_weather.loc["quantile_gbm", "pinball"]

    print("Pinball loss (lower is better):")
    print(f"  persistence baseline : {baseline:.3f}")
    print(f"  GBM without weather  : {plain:.3f}   ({(1 - plain / baseline) * 100:4.1f}% better)")
    print(
        f"  GBM with weather     : {weather:.3f}   ({(1 - weather / baseline) * 100:4.1f}% better)"
    )
    print(f"\nWeather features add a {(1 - weather / plain) * 100:.1f}% reduction over the model")
    print("without them. On real data the lift is smaller (weather forecasts have")
    print("error), but the mechanism and the plumbing are the same.")


if __name__ == "__main__":
    main()
