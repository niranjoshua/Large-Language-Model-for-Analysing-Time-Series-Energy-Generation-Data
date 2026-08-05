"""Forecast to decision translation.

Operators do not act on quantiles, they act on litres of diesel and battery
cycles. This module turns a probabilistic load and generation forecast into a
simple, explainable battery reserve and genset advisory with an uncertainty
range. It is advisory only. It never touches an operator's actual dispatch.
That is a safety and liability boundary, not a feature to add quietly later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vaticore.forecasting.base import quantile_column


@dataclass(frozen=True)
class Advisory:
    """A per horizon operational recommendation with an uncertainty range.

    All energy figures are in kWh over the forecast horizon. net_load is load
    minus generation: positive means the battery or genset must serve it.
    """

    horizon_hours: float
    expected_net_load_kwh: float
    conservative_net_load_kwh: float
    recommended_reserve_kwh: float
    genset_recommended: bool
    note: str


def battery_and_genset_advisory(
    load_forecast: pd.DataFrame,
    generation_forecast: pd.DataFrame,
    *,
    usable_battery_kwh: float,
    step_hours: float,
    reserve_quantile: float = 0.9,
    genset_threshold_kwh: float = 0.0,
) -> Advisory:
    """Derive a battery reserve and genset advisory from quantile forecasts.

    The reserve sizes against the conservative tail (default P90 load, P10
    generation), so the recommendation holds up on a bad day rather than an
    average one. A genset is advised when even a full usable battery cannot
    cover the conservative net load over the horizon.
    """
    lo_q = quantile_column(1.0 - reserve_quantile)
    hi_q = quantile_column(reserve_quantile)
    p50 = quantile_column(0.5)

    for name, frame in (("load", load_forecast), ("generation", generation_forecast)):
        for col in (lo_q, hi_q, p50):
            if col not in frame.columns:
                raise ValueError(f"{name} forecast is missing column {col!r}")

    expected_net = float(
        np.nansum(load_forecast[p50].to_numpy() - generation_forecast[p50].to_numpy()) * step_hours
    )
    # Conservative: high load, low generation.
    conservative_net = float(
        np.nansum(load_forecast[hi_q].to_numpy() - generation_forecast[lo_q].to_numpy())
        * step_hours
    )

    recommended_reserve = float(min(max(conservative_net, 0.0), usable_battery_kwh))
    shortfall = conservative_net - usable_battery_kwh
    genset = shortfall > genset_threshold_kwh

    if genset:
        note = (
            f"Conservative net load {conservative_net:.1f} kWh exceeds usable battery "
            f"{usable_battery_kwh:.1f} kWh by {shortfall:.1f} kWh. Plan genset runtime."
        )
    else:
        note = (
            f"Battery can cover the conservative net load ({conservative_net:.1f} kWh). "
            "No genset expected."
        )

    return Advisory(
        horizon_hours=len(load_forecast) * step_hours,
        expected_net_load_kwh=expected_net,
        conservative_net_load_kwh=conservative_net,
        recommended_reserve_kwh=recommended_reserve,
        genset_recommended=genset,
        note=note,
    )
