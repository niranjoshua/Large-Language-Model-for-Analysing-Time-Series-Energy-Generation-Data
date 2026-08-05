"""Quantile gradient boosting forecaster: the workhorse model.

One LightGBM regressor is trained per quantile with the quantile (pinball)
objective, over calendar and lagged features engineered from a single site's
history. Multi step forecasts are produced recursively: the median trajectory
feeds the lag features forward, and every quantile model predicts its band
around that trajectory at each step.

Design choices, and why:
  - Per quantile models with the native quantile objective give genuine
    probabilistic output, which is what the product sells and what pinball loss
    rewards. Quantiles are sorted per row at the end so they never cross.
  - Recursive multi horizon (rather than one model per horizon step) keeps the
    model count small and the code simple, which matters more than a marginal
    accuracy gain at this stage.
  - Features are engineered from timestamp and target only, so the model works
    inside the backtest harness, which passes just those two columns.
  - Training is deterministic (fixed seed, single threaded) so backtests and CI
    are reproducible. Accuracy claims must be reproducible to be credible.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from vaticore.forecasting.base import (
    DEFAULT_QUANTILES,
    Forecaster,
    InsufficientHistoryError,
    NotFittedError,
    quantile_column,
)
from vaticore.schemas import TIMESTAMP

# Conservative, general purpose defaults. Tuned per deployment later, not here.
_LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": 1,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}


def _calendar_row(ts: pd.Timestamp) -> dict[str, float]:
    """Calendar features for one timestamp. Cyclical encodings for periodicity."""
    return {
        "hour": float(ts.hour),
        "dayofweek": float(ts.dayofweek),
        "month": float(ts.month),
        "is_weekend": float(ts.dayofweek >= 5),
        "hour_sin": float(np.sin(2 * np.pi * ts.hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * ts.hour / 24)),
        "dow_sin": float(np.sin(2 * np.pi * ts.dayofweek / 7)),
        "dow_cos": float(np.cos(2 * np.pi * ts.dayofweek / 7)),
    }


class QuantileGBMForecaster(Forecaster):
    """One LightGBM model per quantile over calendar and lag features."""

    def __init__(
        self,
        target: str,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        lags: tuple[int, ...] = (1, 2, 3, 24, 48, 168),
    ) -> None:
        self.target = target
        self.quantiles = quantiles
        self.lags = tuple(sorted(lags))

        self._models: dict[float, LGBMRegressor] = {}
        self._feature_names: list[str] = []
        self._freq: pd.Timedelta | None = None
        self._last_timestamp: pd.Timestamp | None = None
        # Regular grid history, gap filled, for recursive lag lookups.
        self._grid: pd.Series | None = None

    # -- fitting -----------------------------------------------------------

    def fit(self, history: pd.DataFrame) -> QuantileGBMForecaster:
        if self.target not in history.columns:
            raise ValueError(f"target column {self.target!r} not present in history")

        grid = self._to_regular_grid(history)
        self._grid = grid
        grid_index = pd.DatetimeIndex(grid.index)
        self._freq = pd.Timedelta(grid_index.to_series().diff().dropna().median())
        self._last_timestamp = grid_index[-1]

        features = self._build_training_frame(grid)
        max_lag = max(self.lags)
        if features.dropna().shape[0] < max(2 * max_lag, 50):
            raise InsufficientHistoryError(
                f"not enough usable rows after building lag {max_lag} features; "
                "provide more history"
            )

        self._feature_names = [c for c in features.columns if c != self.target]
        train = features.dropna()
        x = train[self._feature_names]
        y = train[self.target]

        for q in self.quantiles:
            model = LGBMRegressor(objective="quantile", alpha=q, **_LGBM_PARAMS)
            model.fit(x, y)
            self._models[q] = model
        return self

    def _to_regular_grid(self, history: pd.DataFrame) -> pd.Series:
        """Reindex to a regular time grid so lags are well defined.

        Gaps become NaN. A separate interpolated copy is used only for lag
        lookups; training never invents target values.
        """
        series = (
            history[[TIMESTAMP, self.target]]
            .dropna(subset=[TIMESTAMP])
            .drop_duplicates(subset=[TIMESTAMP])
            .sort_values(TIMESTAMP)
            .set_index(TIMESTAMP)[self.target]
            .astype(float)
        )
        if len(series) < 2:
            raise InsufficientHistoryError("need at least two observations to infer spacing")
        freq = pd.Timedelta(series.index.to_series().diff().dropna().median())
        if freq <= pd.Timedelta(0):
            raise ValueError("non increasing timestamps in history")
        grid_index = pd.date_range(series.index[0], series.index[-1], freq=freq, name=TIMESTAMP)
        return series.reindex(grid_index)

    def _build_training_frame(self, grid: pd.Series) -> pd.DataFrame:
        cal = pd.DataFrame(
            [_calendar_row(ts) for ts in grid.index],
            index=grid.index,
        )
        frame = cal.copy()
        for lag in self.lags:
            frame[f"lag{lag}"] = grid.shift(lag).to_numpy()
        frame[self.target] = grid.to_numpy()
        return frame

    # -- prediction --------------------------------------------------------

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        if not self._models or self._grid is None or self._freq is None:
            raise NotFittedError("call fit before predict_quantiles")
        if horizon < 1:
            raise ValueError("horizon must be a positive integer")
        missing = [q for q in quantiles if q not in self._models]
        if missing:
            raise ValueError(f"model was not fitted for quantiles {missing}")

        assert self._last_timestamp is not None
        index = pd.date_range(
            start=self._last_timestamp + self._freq,
            periods=horizon,
            freq=self._freq,
            name=TIMESTAMP,
        )

        # Known trajectory for lag lookups: interpolated history, extended with
        # the median forecast as we roll forward.
        known = self._grid.interpolate(limit_direction="both")
        known = known.fillna(float(np.nanmean(self._grid.to_numpy())))
        trajectory: dict[pd.Timestamp, float] = dict(
            zip(known.index, known.to_numpy(), strict=True)
        )

        rows: list[dict[str, float]] = []
        median_q = min(self._models, key=lambda q: abs(q - 0.5))
        for ts in index:
            feats = _calendar_row(ts)
            for lag in self.lags:
                ref = ts - lag * self._freq
                feats[f"lag{lag}"] = trajectory.get(ref, known.iloc[-1])
            x = pd.DataFrame([feats])[self._feature_names]

            row = {quantile_column(q): float(self._models[q].predict(x)[0]) for q in quantiles}
            rows.append(row)
            # Feed the median forecast forward for future lags.
            trajectory[ts] = float(self._models[median_q].predict(x)[0])

        forecast = pd.DataFrame(rows, index=index)
        forecast = forecast.clip(lower=0.0)
        # Enforce non crossing quantiles row by row.
        forecast.loc[:, :] = np.sort(forecast.to_numpy(), axis=1)
        return forecast
