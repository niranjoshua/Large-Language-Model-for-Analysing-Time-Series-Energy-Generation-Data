from __future__ import annotations

import pandas as pd

from vaticore.decisions import battery_and_genset_advisory
from vaticore.forecasting.base import quantile_column


def _flat_forecast(value: float, n: int = 24) -> pd.DataFrame:
    index = pd.date_range("2024-02-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            quantile_column(0.1): [value * 0.9] * n,
            quantile_column(0.5): [value] * n,
            quantile_column(0.9): [value * 1.1] * n,
        },
        index=index,
    )


def test_genset_advised_when_battery_too_small() -> None:
    load = _flat_forecast(100.0)
    generation = _flat_forecast(10.0)
    advisory = battery_and_genset_advisory(
        load, generation, usable_battery_kwh=50.0, step_hours=1.0
    )
    assert advisory.genset_recommended is True
    assert advisory.conservative_net_load_kwh > advisory.expected_net_load_kwh


def test_no_genset_when_battery_covers_load() -> None:
    load = _flat_forecast(10.0)
    generation = _flat_forecast(8.0)
    advisory = battery_and_genset_advisory(
        load, generation, usable_battery_kwh=1000.0, step_hours=1.0
    )
    assert advisory.genset_recommended is False
    assert advisory.recommended_reserve_kwh <= 1000.0


def test_missing_quantile_column_raises() -> None:
    load = _flat_forecast(10.0).drop(columns=[quantile_column(0.9)])
    generation = _flat_forecast(8.0)
    try:
        battery_and_genset_advisory(load, generation, usable_battery_kwh=100.0, step_hours=1.0)
    except ValueError as exc:
        assert "missing column" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing quantile column")
