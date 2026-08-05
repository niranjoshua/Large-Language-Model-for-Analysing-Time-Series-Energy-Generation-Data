from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from vaticore.datasets import make_synthetic_site
from vaticore.engine import run_backtest
from vaticore.forecasting import Forecaster, QuantileGBMForecaster
from vaticore.forecasting.base import NotFittedError, quantile_column
from vaticore.schemas import LOAD_KW

# LightGBM emits benign warnings on small data; keep test output clean.
pytestmark = pytest.mark.filterwarnings("ignore")


def test_gbm_is_a_forecaster() -> None:
    assert issubclass(QuantileGBMForecaster, Forecaster)


def test_output_shape_columns_and_index(single_site: pd.DataFrame) -> None:
    quantiles = (0.1, 0.5, 0.9)
    model = QuantileGBMForecaster(target=LOAD_KW, quantiles=quantiles).fit(single_site)
    forecast = model.predict_quantiles(horizon=24, quantiles=quantiles)

    assert len(forecast) == 24
    assert list(forecast.columns) == [quantile_column(q) for q in quantiles]
    assert str(forecast.index.dtype) == "datetime64[ns, UTC]"
    assert forecast.index[0] == single_site["timestamp"].max() + pd.Timedelta(hours=1)


def test_quantiles_do_not_cross(single_site: pd.DataFrame) -> None:
    model = QuantileGBMForecaster(target=LOAD_KW).fit(single_site)
    forecast = model.predict_quantiles(horizon=48)
    assert np.all(np.diff(forecast.to_numpy(), axis=1) >= -1e-9)


def test_predict_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        QuantileGBMForecaster(target=LOAD_KW).predict_quantiles(horizon=24)


def test_predict_unfitted_quantile_raises(single_site: pd.DataFrame) -> None:
    model = QuantileGBMForecaster(target=LOAD_KW, quantiles=(0.1, 0.5, 0.9)).fit(single_site)
    with pytest.raises(ValueError, match="not fitted for quantiles"):
        model.predict_quantiles(horizon=6, quantiles=(0.5, 0.95))


def test_gbm_beats_persistence_on_pinball() -> None:
    """The point of the model: it must beat the baseline it is measured against."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        site = make_synthetic_site("op", "s", days=50, seed=3)
        result = run_backtest(
            site,
            LOAD_KW,
            horizon=24,
            initial=24 * 35,
            step=72,
            model="quantile_gbm",
        )
    summary = result.summary()
    gbm = summary.loc["quantile_gbm", "pinball"]
    persistence = summary.loc["persistence", "pinball"]
    assert gbm < persistence, f"GBM pinball {gbm:.3f} did not beat persistence {persistence:.3f}"
