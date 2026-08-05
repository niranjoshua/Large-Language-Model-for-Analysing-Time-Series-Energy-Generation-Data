"""Calendar and lag features.

These are real and safe to use now: they are the inputs the quantile gradient
boosting model will consume. Lags respect the no look ahead rule, they only
reference past values, and they are computed per site so one site's history
never leaks into another.
"""

from __future__ import annotations

import pandas as pd

from vaticore.schemas import OPERATOR_ID, SITE_ID, TIMESTAMP


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add hour of day, day of week and month features from the UTC timestamp.

    Cyclical encodings (sin/cos) are added so tree and linear models both get a
    usable representation of periodicity.
    """
    out = frame.copy()
    ts = pd.DatetimeIndex(out[TIMESTAMP])
    out["hour"] = ts.hour
    out["dayofweek"] = ts.dayofweek
    out["month"] = ts.month
    out["is_weekend"] = (ts.dayofweek >= 5).astype(int)

    import numpy as np

    out["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
    return out


def add_lag_features(
    frame: pd.DataFrame,
    target: str,
    lags: tuple[int, ...] = (1, 24, 168),
) -> pd.DataFrame:
    """Add lagged values of target, computed per site to avoid cross site leakage.

    Lags are in steps, not hours. The default (1, 24, 168) suits hourly data.
    """
    out = frame.sort_values([OPERATOR_ID, SITE_ID, TIMESTAMP]).copy()
    grouped = out.groupby([OPERATOR_ID, SITE_ID], sort=False)[target]
    for lag in lags:
        out[f"{target}_lag{lag}"] = grouped.shift(lag)
    return out
