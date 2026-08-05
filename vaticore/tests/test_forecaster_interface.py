from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vaticore.forecasting import Forecaster, PersistenceForecaster
from vaticore.forecasting.base import NotFittedError, quantile_column
from vaticore.schemas import LOAD_KW


def test_persistence_is_a_forecaster() -> None:
    assert issubclass(PersistenceForecaster, Forecaster)


def test_output_shape_and_columns(single_site: pd.DataFrame) -> None:
    model = PersistenceForecaster(target=LOAD_KW).fit(single_site)
    quantiles = (0.1, 0.5, 0.9)
    forecast = model.predict_quantiles(horizon=24, quantiles=quantiles)

    assert len(forecast) == 24
    assert list(forecast.columns) == [quantile_column(q) for q in quantiles]
    assert str(forecast.index.dtype) == "datetime64[ns, UTC]"


def test_quantiles_do_not_cross(single_site: pd.DataFrame) -> None:
    model = PersistenceForecaster(target=LOAD_KW).fit(single_site)
    forecast = model.predict_quantiles(horizon=48, quantiles=(0.1, 0.5, 0.9))
    values = forecast.to_numpy()
    # Each row must be monotonically non decreasing across quantile columns.
    assert np.all(np.diff(values, axis=1) >= -1e-9)


def test_predict_before_fit_raises() -> None:
    model = PersistenceForecaster(target=LOAD_KW)
    with pytest.raises(NotFittedError):
        model.predict_quantiles(horizon=24)
