"""Rolling origin backtesting harness.

This is real, tested code and the place accuracy claims are made. Splits are
strictly chronological so there is no look ahead leakage. Every run scores the
candidate model and the persistence baseline on identical folds, so a model
that fails to beat persistence is reported honestly.

A single call evaluates one site. Fleet level evaluation iterates sites and is
built on top of this, never by assuming a single site inside the harness.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vaticore.evaluation.metrics import coverage, mae, mean_pinball_loss, rmse
from vaticore.forecasting.base import DEFAULT_QUANTILES, Forecaster, quantile_column
from vaticore.forecasting.baseline import PersistenceForecaster
from vaticore.schemas import TIMESTAMP

ForecasterFactory = Callable[[], Forecaster]


@dataclass(frozen=True)
class FoldMetrics:
    """Metrics for one model on one fold."""

    model: str
    fold: int
    n_scored: int
    pinball: float
    mae: float
    rmse: float
    coverage: float | None


@dataclass
class BacktestResult:
    """Per fold metrics for the candidate model and the baseline."""

    target: str
    horizon: int
    quantiles: tuple[float, ...]
    folds: list[FoldMetrics] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.folds)

    def summary(self) -> pd.DataFrame:
        """Mean metrics per model, plus whether the candidate beats persistence."""
        frame = self.to_frame()
        agg = frame.groupby("model")[["pinball", "mae", "rmse"]].mean().sort_values("pinball")
        return agg


def _rolling_origins(n: int, initial: int, horizon: int, step: int) -> list[int]:
    """Return the training cutoff index for each fold.

    Fold k trains on rows [0, cutoff) and tests on [cutoff, cutoff + horizon).
    """
    origins = []
    cutoff = initial
    while cutoff + horizon <= n:
        origins.append(cutoff)
        cutoff += step
    return origins


def backtest_site(
    history: pd.DataFrame,
    target: str,
    make_model: ForecasterFactory,
    *,
    horizon: int,
    initial: int,
    step: int | None = None,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    model_name: str = "candidate",
) -> BacktestResult:
    """Backtest one candidate model against persistence on a single site.

    Parameters
    ----------
    history:
        Validated single site frame, sorted or unsorted, containing target.
    make_model:
        Zero argument factory returning a fresh, unfitted Forecaster. A factory
        (not an instance) is required so each fold trains a clean model.
    horizon:
        Steps ahead to forecast and score per fold.
    initial:
        Number of rows in the first training window.
    step:
        Rows to advance the origin between folds. Defaults to horizon
        (non overlapping test windows).
    """
    step = step or horizon
    data = (
        history[[TIMESTAMP, target]]
        .dropna(subset=[TIMESTAMP])
        .sort_values(TIMESTAMP)
        .reset_index(drop=True)
    )
    origins = _rolling_origins(len(data), initial=initial, horizon=horizon, step=step)
    if not origins:
        raise ValueError(
            f"not enough history: need at least initial + horizon = {initial + horizon} rows, "
            f"got {len(data)}"
        )

    result = BacktestResult(target=target, horizon=horizon, quantiles=quantiles)

    for fold, cutoff in enumerate(origins):
        train = data.iloc[:cutoff]
        test = data.iloc[cutoff : cutoff + horizon]
        actual = test[target].to_numpy(dtype=float)

        candidate = _score_one(make_model(), model_name, fold, train, actual, horizon, quantiles)
        baseline = _score_one(
            PersistenceForecaster(target=target),
            "persistence",
            fold,
            train,
            actual,
            horizon,
            quantiles,
        )
        result.folds.extend([candidate, baseline])

    return result


def _score_one(
    model: Forecaster,
    name: str,
    fold: int,
    train: pd.DataFrame,
    actual: np.ndarray,
    horizon: int,
    quantiles: tuple[float, ...],
) -> FoldMetrics:
    model.fit(train)
    forecast = model.predict_quantiles(horizon=horizon, quantiles=quantiles)

    by_quantile = {q: forecast[quantile_column(q)].to_numpy(dtype=float) for q in quantiles}
    median_q = min(quantiles, key=lambda q: abs(q - 0.5))
    median = by_quantile[median_q]

    cov: float | None = None
    if len(quantiles) >= 2:
        lo = forecast[quantile_column(min(quantiles))].to_numpy(dtype=float)
        hi = forecast[quantile_column(max(quantiles))].to_numpy(dtype=float)
        cov = coverage(actual, lo, hi)

    return FoldMetrics(
        model=name,
        fold=fold,
        n_scored=int(np.sum(~np.isnan(actual))),
        pinball=mean_pinball_loss(actual, by_quantile),
        mae=mae(actual, median),
        rmse=rmse(actual, median),
        coverage=cov,
    )
