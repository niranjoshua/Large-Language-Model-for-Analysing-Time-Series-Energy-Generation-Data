"""End to end quickstart.

Runs the whole pipeline on synthetic demo data: ingest a site, forecast load
with the quantile GBM model, backtest it against the persistence baseline, and
produce a battery and genset advisory. This is the fastest way to see what
Vaticore does.

    uv run python examples/quickstart.py
"""

from __future__ import annotations

import warnings

from vaticore import engine
from vaticore.datasets import make_synthetic_fleet
from vaticore.schemas import LOAD_KW


def main() -> None:
    warnings.filterwarnings("ignore")

    fleet = make_synthetic_fleet(days=75, seed=3)
    site = engine.select_site(fleet, "lagos-energy", "ikeja-minigrid")
    print(f"Site history: {len(site)} hourly rows")

    print("\nForecasting next 24 hours of load (quantile GBM)...")
    forecast = engine.forecast_site(site, LOAD_KW, horizon=24, model="quantile_gbm")
    print(forecast.head().round(1).to_string())

    print("\nBacktesting quantile GBM against persistence baseline...")
    result = engine.run_backtest(
        site, LOAD_KW, horizon=24, initial=24 * 45, step=72, model="quantile_gbm"
    )
    summary = result.summary()
    print(summary.round(3).to_string())
    gbm = summary.loc["quantile_gbm", "pinball"]
    base = summary.loc["persistence", "pinball"]
    print(f"\nPinball loss improvement over baseline: {(1 - gbm / base) * 100:.1f}%")

    print("\nBattery and genset advisory (next 24h, 600 kWh usable battery)...")
    advisory = engine.advisory_for_site(
        site, horizon=24, usable_battery_kwh=600.0, model="quantile_gbm"
    )
    print(f"  Genset recommended: {advisory.genset_recommended}")
    print(f"  Recommended reserve: {advisory.recommended_reserve_kwh:.0f} kWh")
    print(f"  {advisory.note}")


if __name__ == "__main__":
    main()
