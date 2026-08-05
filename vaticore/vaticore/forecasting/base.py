"""The forecaster interface: the most important contract in the system.

Every model in this package subclasses Forecaster. This is what makes models
swappable and ensemble-able. Do not deviate from it.

Rules:
  - Output is always quantiles, even for the baseline. A naive model may return
    the same value across quantiles.
  - Models must not silently fail on missing data. Handle gaps explicitly or
    raise a clear, typed error.
  - The persistence baseline is the reference every other model is measured
    against. It always exists and always runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


def quantile_column(q: float) -> str:
    """Canonical column name for a quantile, for example 0.1 -> 'q0.1'."""
    return f"q{q:g}"


class ForecasterError(Exception):
    """Base class for typed forecaster errors."""


class InsufficientHistoryError(ForecasterError):
    """Raised when a model does not have enough history to produce a forecast."""


class NotFittedError(ForecasterError):
    """Raised when predict is called before fit."""


class Forecaster(ABC):
    """All Vaticore forecasting models implement this interface.

    A model instance is scoped to a single site's history once fitted. The
    target column (load_kw or generation_kw) is chosen at construction time so
    the same class can forecast either series.
    """

    @abstractmethod
    def fit(self, history: pd.DataFrame) -> Forecaster:
        """Fit on a site's history.

        Must tolerate gaps and irregular timestamps. Returns self so calls can
        be chained.
        """
        ...

    @abstractmethod
    def predict_quantiles(
        self,
        horizon: int,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        """Produce a probabilistic forecast.

        Returns a frame indexed by forecast timestamp (timezone aware UTC) with
        one column per quantile, named by quantile_column(). Quantiles are
        returned non crossing (monotonically non decreasing across columns).
        """
        ...
