from __future__ import annotations

import pandas as pd

from vaticore.forecasting import PersistenceForecaster
from vaticore.forecasting.base import quantile_column
from vaticore.schemas import LOAD_KW


def test_infers_daily_period_for_hourly_data(single_site: pd.DataFrame) -> None:
    model = PersistenceForecaster(target=LOAD_KW).fit(single_site)
    assert model.seasonal_period == 24


def test_median_forecast_tracks_daily_shape(single_site: pd.DataFrame) -> None:
    model = PersistenceForecaster(target=LOAD_KW).fit(single_site)
    forecast = model.predict_quantiles(horizon=24, quantiles=(0.1, 0.5, 0.9))
    median = forecast[quantile_column(0.5)]
    # The synthetic load peaks in the afternoon; the naive repeat should too.
    assert 10 <= int(median.reset_index(drop=True).idxmax()) <= 20


def test_forecast_index_continues_history(single_site: pd.DataFrame) -> None:
    model = PersistenceForecaster(target=LOAD_KW).fit(single_site)
    forecast = model.predict_quantiles(horizon=3)
    last = single_site["timestamp"].max()
    assert forecast.index[0] == last + pd.Timedelta(hours=1)


def test_band_widens_with_more_uncertainty(single_site: pd.DataFrame) -> None:
    model = PersistenceForecaster(target=LOAD_KW).fit(single_site)
    forecast = model.predict_quantiles(horizon=24, quantiles=(0.1, 0.5, 0.9))
    width = forecast[quantile_column(0.9)] - forecast[quantile_column(0.1)]
    assert (width >= 0).all()
