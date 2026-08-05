"""Engine facade: one orchestration path for forecasting, backtesting and advice.

The API and the dashboard both go through here, so there is a single, tested
route from a site's history to a forecast, a backtest, or an operator advisory.
Nothing above this layer touches model classes directly.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from vaticore.decisions import Advisory, battery_and_genset_advisory
from vaticore.evaluation.backtest import BacktestResult, backtest_site
from vaticore.forecasting import Forecaster, PersistenceForecaster, QuantileGBMForecaster
from vaticore.forecasting.base import DEFAULT_QUANTILES
from vaticore.schemas import GENERATION_KW, LOAD_KW, OPERATOR_ID, SITE_ID, TIMESTAMP

# Registry of model builders keyed by public name. Add new models here once
# they conform to the Forecaster interface.
ModelFactory = Callable[[str, tuple[float, ...]], Forecaster]

_MODELS: dict[str, ModelFactory] = {
    "persistence": lambda target, quantiles: PersistenceForecaster(target=target),
    "quantile_gbm": lambda target, quantiles: QuantileGBMForecaster(
        target=target, quantiles=quantiles
    ),
}


def available_models() -> list[str]:
    return list(_MODELS)


def _build(model: str, target: str, quantiles: tuple[float, ...]) -> Forecaster:
    if model not in _MODELS:
        raise ValueError(f"unknown model {model!r}; available: {available_models()}")
    return _MODELS[model](target, quantiles)


def select_site(fleet: pd.DataFrame, operator_id: str, site_id: str) -> pd.DataFrame:
    """Return one site's rows. Multi tenant scoping lives here, once."""
    site = fleet[(fleet[OPERATOR_ID] == operator_id) & (fleet[SITE_ID] == site_id)]
    if site.empty:
        raise KeyError(f"no data for operator {operator_id!r} site {site_id!r}")
    return site.sort_values(TIMESTAMP).reset_index(drop=True)


def forecast_site(
    history: pd.DataFrame,
    target: str,
    *,
    horizon: int,
    model: str = "quantile_gbm",
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Fit the chosen model on one site's history and forecast the horizon."""
    forecaster = _build(model, target, quantiles).fit(history)
    return forecaster.predict_quantiles(horizon=horizon, quantiles=quantiles)


def run_backtest(
    history: pd.DataFrame,
    target: str,
    *,
    horizon: int,
    initial: int,
    step: int | None = None,
    model: str = "quantile_gbm",
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> BacktestResult:
    """Backtest a model against persistence on one site."""
    return backtest_site(
        history,
        target=target,
        make_model=lambda: _build(model, target, quantiles),
        horizon=horizon,
        initial=initial,
        step=step,
        quantiles=quantiles,
        model_name=model,
    )


def advisory_for_site(
    history: pd.DataFrame,
    *,
    horizon: int,
    usable_battery_kwh: float,
    step_hours: float = 1.0,
    model: str = "quantile_gbm",
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> Advisory:
    """Forecast load and generation, then derive a battery and genset advisory."""
    load_fc = forecast_site(history, LOAD_KW, horizon=horizon, model=model, quantiles=quantiles)
    gen_fc = forecast_site(
        history, GENERATION_KW, horizon=horizon, model=model, quantiles=quantiles
    )
    reserve_q = max(quantiles)
    return battery_and_genset_advisory(
        load_fc,
        gen_fc,
        usable_battery_kwh=usable_battery_kwh,
        step_hours=step_hours,
        reserve_quantile=reserve_q,
    )
