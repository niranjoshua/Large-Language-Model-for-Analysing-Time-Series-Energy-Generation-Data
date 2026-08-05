"""Seasonal persistence baseline.

This is the reference model. It always exists and always runs, and every other
model is measured against it. It is a genuine probabilistic forecaster, not a
placeholder: the point forecast is a seasonal naive repeat of the last daily
cycle, and the quantile bands come from the empirical distribution of the
model's own in sample seasonal residuals. That keeps the baseline honest and
gives later models a real bar to clear on pinball loss, not just on MAE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vaticore.forecasting.base import (
    DEFAULT_QUANTILES,
    Forecaster,
    InsufficientHistoryError,
    NotFittedError,
    quantile_column,
)
from vaticore.schemas import TIMESTAMP


class PersistenceForecaster(Forecaster):
    """Seasonal naive persistence with empirical residual quantiles.

    Parameters
    ----------
    target:
        Column to forecast, for example "load_kw" or "generation_kw".
    seasonal_period:
        Number of steps in one seasonal cycle. If None, it is inferred as the
        number of steps per day from the median timestamp spacing.
    """

    def __init__(self, target: str, seasonal_period: int | None = None) -> None:
        self.target = target
        self.seasonal_period = seasonal_period

        self._freq: pd.Timedelta | None = None
        self._last_timestamp: pd.Timestamp | None = None
        self._base_cycle: np.ndarray | None = None
        self._residuals: np.ndarray | None = None

    def fit(self, history: pd.DataFrame) -> PersistenceForecaster:
        if self.target not in history.columns:
            raise ValueError(f"target column {self.target!r} not present in history")

        series = (
            history[[TIMESTAMP, self.target]]
            .dropna(subset=[TIMESTAMP])
            .sort_values(TIMESTAMP)
            .reset_index(drop=True)
        )
        if len(series) < 2:
            raise InsufficientHistoryError("need at least two observations to infer spacing")

        timestamps = pd.DatetimeIndex(series[TIMESTAMP])
        self._freq = self._infer_freq(timestamps)
        self._last_timestamp = timestamps[-1]

        period = self.seasonal_period or self._infer_seasonal_period(self._freq)
        self.seasonal_period = period

        values = series[self.target].to_numpy(dtype=float)
        if len(values) < period:
            raise InsufficientHistoryError(
                f"need at least one full seasonal cycle ({period} steps), got {len(values)}"
            )

        # Last full cycle, gap filled, becomes the repeating point forecast.
        base = pd.Series(values[-period:]).interpolate(limit_direction="both")
        if base.isna().all():
            raise InsufficientHistoryError("last seasonal cycle is entirely missing")
        base = base.fillna(np.nanmean(values))
        self._base_cycle = base.to_numpy(dtype=float)

        # In sample seasonal residuals: y[t] - y[t - period].
        resid = values[period:] - values[:-period]
        self._residuals = resid[~np.isnan(resid)]
        return self

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        if self._base_cycle is None or self._last_timestamp is None or self._freq is None:
            raise NotFittedError("call fit before predict_quantiles")
        if horizon < 1:
            raise ValueError("horizon must be a positive integer")

        period = len(self._base_cycle)
        steps = np.arange(horizon)
        point = self._base_cycle[steps % period]

        index = pd.date_range(
            start=self._last_timestamp + self._freq,
            periods=horizon,
            freq=self._freq,
            name=TIMESTAMP,
        )

        columns: dict[str, np.ndarray] = {}
        for q in quantiles:
            offset = self._residual_offset(q)
            columns[quantile_column(q)] = np.clip(point + offset, a_min=0.0, a_max=None)

        forecast = pd.DataFrame(columns, index=index)
        # Enforce non crossing quantiles row by row.
        forecast.loc[:, :] = np.sort(forecast.to_numpy(), axis=1)
        return forecast

    # -- internals ---------------------------------------------------------

    def _residual_offset(self, q: float) -> float:
        if self._residuals is None or self._residuals.size == 0:
            return 0.0
        return float(np.quantile(self._residuals, q))

    @staticmethod
    def _infer_freq(timestamps: pd.DatetimeIndex) -> pd.Timedelta:
        deltas = timestamps.to_series().diff().dropna()
        if deltas.empty:
            raise InsufficientHistoryError("cannot infer spacing from a single timestamp")
        median = deltas.median()
        if median <= pd.Timedelta(0):
            raise ValueError("non increasing timestamps in history")
        return pd.Timedelta(median)

    @staticmethod
    def _infer_seasonal_period(freq: pd.Timedelta) -> int:
        steps_per_day = pd.Timedelta("1D") / freq
        return max(1, round(steps_per_day))
