"""Quantile gradient boosting forecaster: the workhorse model.

One LightGBM regressor is trained per quantile with the quantile (pinball)
objective, over calendar, lagged and optional exogenous features engineered from
a single site's history. Multi step forecasts are produced recursively: the
median trajectory feeds the lag features forward, and every quantile model
predicts its band around that trajectory at each step.

Exogenous covariates (weather) are contemporaneous: the weather at time t is a
feature for the target at time t. Because weather forecasts for the horizon are
available in production (and the true weather is available in a backtest), the
model consumes future weather at prediction time through `future_exog`.

Design choices, and why:
  - Per quantile models with the native quantile objective give genuine
    probabilistic output, which is what the product sells and what pinball loss
    rewards. Quantiles are sorted per row at the end so they never cross.
  - Recursive multi horizon keeps the model count small and the code simple.
  - Features are engineered from timestamp, target and named exogenous columns,
    so the model works inside the backtest harness.
  - Training is deterministic (fixed seed, single threaded) so backtests and CI
    are reproducible.
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
    """One LightGBM model per quantile over calendar, lag and weather features."""

    def __init__(
        self,
        target: str,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        lags: tuple[int, ...] = (1, 2, 3, 24, 48, 168),
        exog_features: tuple[str, ...] = (),
    ) -> None:
        self.target = target
        self.quantiles = quantiles
        self.lags = tuple(sorted(lags))
        self.exog_features = tuple(exog_features)

        self._models: dict[float, LGBMRegressor] = {}
        self._feature_names: list[str] = []
        self._freq: pd.Timedelta | None = None
        self._last_timestamp: pd.Timestamp | None = None
        # Regular grid history (target + exog), gap filled, for recursive lookups.
        self._grid: pd.DataFrame | None = None
        self._last_exog: dict[str, float] = {}

    # -- fitting -----------------------------------------------------------

    def fit(self, history: pd.DataFrame) -> QuantileGBMForecaster:
        if self.target not in history.columns:
            raise ValueError(f"target column {self.target!r} not present in history")
        missing = [c for c in self.exog_features if c not in history.columns]
        if missing:
            raise ValueError(f"exogenous columns missing from history: {missing}")

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

        # Last observed exogenous values, for fallback when future weather is absent.
        if self.exog_features:
            last = grid[list(self.exog_features)].ffill().iloc[-1]
            self._last_exog = {c: float(last[c]) for c in self.exog_features}
        return self

    def _to_regular_grid(self, history: pd.DataFrame) -> pd.DataFrame:
        """Reindex target and exogenous columns to a regular time grid."""
        cols = [self.target, *self.exog_features]
        frame = (
            history[[TIMESTAMP, *cols]]
            .dropna(subset=[TIMESTAMP])
            .drop_duplicates(subset=[TIMESTAMP])
            .sort_values(TIMESTAMP)
            .set_index(TIMESTAMP)
        )
        if len(frame) < 2:
            raise InsufficientHistoryError("need at least two observations to infer spacing")
        freq = pd.Timedelta(frame.index.to_series().diff().dropna().median())
        if freq <= pd.Timedelta(0):
            raise ValueError("non increasing timestamps in history")
        grid_index = pd.date_range(frame.index[0], frame.index[-1], freq=freq, name=TIMESTAMP)
        return frame.reindex(grid_index).astype(float)

    def _build_training_frame(self, grid: pd.DataFrame) -> pd.DataFrame:
        target_series = grid[self.target]
        cal = pd.DataFrame([_calendar_row(ts) for ts in grid.index], index=grid.index)
        frame = cal.copy()
        for lag in self.lags:
            frame[f"lag{lag}"] = target_series.shift(lag).to_numpy()
        for col in self.exog_features:
            frame[col] = grid[col].to_numpy()
        frame[self.target] = target_series.to_numpy()
        return frame

    # -- prediction --------------------------------------------------------

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        future_exog: pd.DataFrame | None = None,
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
        exog_lookup = self._exog_lookup(future_exog)

        # Known target trajectory for lag lookups: interpolated history, extended
        # with the median forecast as we roll forward.
        known = self._grid[self.target].interpolate(limit_direction="both")
        known = known.fillna(float(np.nanmean(self._grid[self.target].to_numpy())))
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
            for col in self.exog_features:
                feats[col] = exog_lookup(col, ts)
            x = pd.DataFrame([feats])[self._feature_names]

            row = {quantile_column(q): float(self._models[q].predict(x)[0]) for q in quantiles}
            rows.append(row)
            trajectory[ts] = float(self._models[median_q].predict(x)[0])

        forecast = pd.DataFrame(rows, index=index)
        forecast = forecast.clip(lower=0.0)
        # Enforce non crossing quantiles row by row.
        forecast.loc[:, :] = np.sort(forecast.to_numpy(), axis=1)
        return forecast

    def _exog_lookup(self, future_exog: pd.DataFrame | None):  # type: ignore[no-untyped-def]
        """Return a function (column, timestamp) -> value for exogenous features.

        Uses supplied future weather when available, else the last observed value
        (forward filled), so a missing forecast degrades gracefully rather than
        failing.
        """
        table: dict[pd.Timestamp, dict[str, float]] = {}
        if future_exog is not None and self.exog_features and TIMESTAMP in future_exog.columns:
            cols = [c for c in self.exog_features if c in future_exog.columns]
            stamps = pd.DatetimeIndex(future_exog[TIMESTAMP])
            for i, ts in enumerate(stamps):
                table[ts] = {c: float(future_exog.iloc[i][c]) for c in cols}

        def lookup(column: str, ts: pd.Timestamp) -> float:
            row = table.get(ts)
            if row is not None and column in row and not np.isnan(row[column]):
                return row[column]
            return self._last_exog.get(column, 0.0)

        return lookup
