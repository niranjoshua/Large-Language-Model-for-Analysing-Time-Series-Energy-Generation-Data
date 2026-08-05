"""Forecast metrics.

Primary metric is pinball (quantile) loss, because the product is probabilistic.
Secondary metrics are MAE, RMSE and quantile calibration/coverage. Everything
here operates on plain numpy arrays so it is fast and trivially testable, and
every function ignores positions where the actual value is missing.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]


def _aligned(actual: npt.ArrayLike, predicted: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if a.shape != p.shape:
        raise ValueError(f"shape mismatch: actual {a.shape} vs predicted {p.shape}")
    mask = ~np.isnan(a) & ~np.isnan(p)
    if not mask.any():
        raise ValueError("no overlapping non missing values to score")
    return a[mask], p[mask]


def pinball_loss(actual: npt.ArrayLike, predicted: npt.ArrayLike, quantile: float) -> float:
    """Pinball loss for a single quantile forecast. Lower is better."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between 0 and 1")
    a, p = _aligned(actual, predicted)
    error = a - p
    loss = np.maximum(quantile * error, (quantile - 1.0) * error)
    return float(np.mean(loss))


def mean_pinball_loss(
    actual: npt.ArrayLike,
    predicted_by_quantile: Mapping[float, npt.ArrayLike],
) -> float:
    """Average pinball loss across a set of quantiles."""
    if not predicted_by_quantile:
        raise ValueError("predicted_by_quantile is empty")
    losses = [pinball_loss(actual, p, q) for q, p in predicted_by_quantile.items()]
    return float(np.mean(losses))


def mae(actual: npt.ArrayLike, predicted: npt.ArrayLike) -> float:
    """Mean absolute error, scored against the median (P50) forecast."""
    a, p = _aligned(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual: npt.ArrayLike, predicted: npt.ArrayLike) -> float:
    """Root mean squared error, scored against the median (P50) forecast."""
    a, p = _aligned(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def coverage(actual: npt.ArrayLike, lower: npt.ArrayLike, upper: npt.ArrayLike) -> float:
    """Empirical coverage: fraction of actuals falling within [lower, upper]."""
    a = np.asarray(actual, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(lo) & ~np.isnan(hi)
    if not mask.any():
        raise ValueError("no overlapping non missing values to score")
    inside = (a[mask] >= lo[mask]) & (a[mask] <= hi[mask])
    return float(np.mean(inside))


def quantile_calibration(actual: npt.ArrayLike, predicted: npt.ArrayLike, quantile: float) -> float:
    """Fraction of actuals at or below the quantile forecast.

    A well calibrated q quantile forecast should sit close to q. Returning the
    observed fraction lets callers compare it against the nominal level.
    """
    a, p = _aligned(actual, predicted)
    return float(np.mean(a <= p))
