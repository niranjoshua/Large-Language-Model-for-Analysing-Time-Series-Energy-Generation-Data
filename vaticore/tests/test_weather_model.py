from __future__ import annotations

import warnings

import pandas as pd
import pytest

from vaticore.datasets import make_synthetic_site
from vaticore.engine import run_backtest
from vaticore.forecasting import QuantileGBMForecaster
from vaticore.forecasting.base import quantile_column
from vaticore.schemas import GENERATION_KW, TIMESTAMP

pytestmark = pytest.mark.filterwarnings("ignore")

WEATHER = ("shortwave_radiation", "cloud_cover", "temperature_2m")


def _weather_site(days: int = 40, seed: int = 5) -> pd.DataFrame:
    return make_synthetic_site("op", "s", days=days, seed=seed, gap_fraction=0.0, with_weather=True)


def test_fit_and_predict_with_exog_shape() -> None:
    site = _weather_site(days=30)
    model = QuantileGBMForecaster(target=GENERATION_KW, exog_features=WEATHER).fit(site)
    future = site.tail(24)[[TIMESTAMP, *WEATHER]]
    forecast = model.predict_quantiles(horizon=24, future_exog=future)
    assert len(forecast) == 24
    assert list(forecast.columns) == [quantile_column(q) for q in (0.1, 0.5, 0.9)]


def test_future_exog_changes_the_forecast() -> None:
    site = _weather_site(days=30)
    model = QuantileGBMForecaster(target=GENERATION_KW, exog_features=WEATHER).fit(site)
    base_index = site["timestamp"].max() + pd.Timedelta(hours=1)
    idx = pd.date_range(base_index, periods=12, freq="h", name=TIMESTAMP)

    sunny = pd.DataFrame(
        {TIMESTAMP: idx, "shortwave_radiation": 900.0, "cloud_cover": 5.0, "temperature_2m": 30.0}
    )
    overcast = pd.DataFrame(
        {TIMESTAMP: idx, "shortwave_radiation": 50.0, "cloud_cover": 95.0, "temperature_2m": 22.0}
    )
    sunny_fc = model.predict_quantiles(horizon=12, future_exog=sunny)
    overcast_fc = model.predict_quantiles(horizon=12, future_exog=overcast)
    # More radiation should not forecast less generation on the whole.
    assert sunny_fc[quantile_column(0.5)].sum() > overcast_fc[quantile_column(0.5)].sum()


def test_weather_features_beat_persistence_on_generation() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        site = _weather_site(days=50, seed=5)
        result = run_backtest(
            site,
            GENERATION_KW,
            horizon=24,
            initial=24 * 35,
            step=72,
            model="quantile_gbm",
            exog=WEATHER,
        )
    summary = result.summary()
    gbm = summary.loc["quantile_gbm", "pinball"]
    persistence = summary.loc["persistence", "pinball"]
    assert gbm < persistence, f"weather GBM {gbm:.3f} did not beat persistence {persistence:.3f}"
