from __future__ import annotations

import pytest

from vaticore import engine
from vaticore.datasets import make_synthetic_fleet
from vaticore.schemas import LOAD_KW

pytestmark = pytest.mark.filterwarnings("ignore")


def test_available_models_includes_core() -> None:
    models = engine.available_models()
    assert "persistence" in models
    assert "quantile_gbm" in models


def test_select_site_scopes_by_operator_and_site() -> None:
    fleet = make_synthetic_fleet(days=10, seed=1)
    site = engine.select_site(fleet, "lagos-energy", "ikeja-minigrid")
    assert set(site["operator_id"].unique()) == {"lagos-energy"}
    assert set(site["site_id"].unique()) == {"ikeja-minigrid"}


def test_select_missing_site_raises() -> None:
    fleet = make_synthetic_fleet(days=5, seed=1)
    with pytest.raises(KeyError):
        engine.select_site(fleet, "nope", "nope")


def test_forecast_site_via_facade() -> None:
    fleet = make_synthetic_fleet(days=20, seed=1)
    site = engine.select_site(fleet, "lagos-energy", "ikeja-minigrid")
    forecast = engine.forecast_site(site, LOAD_KW, horizon=12, model="persistence")
    assert len(forecast) == 12


def test_advisory_for_site_via_facade() -> None:
    fleet = make_synthetic_fleet(days=20, seed=1)
    site = engine.select_site(fleet, "lagos-energy", "ikeja-minigrid")
    advisory = engine.advisory_for_site(
        site, horizon=24, usable_battery_kwh=500.0, model="persistence"
    )
    assert advisory.horizon_hours == 24
    assert advisory.recommended_reserve_kwh >= 0.0


def test_unknown_model_raises() -> None:
    fleet = make_synthetic_fleet(days=10, seed=1)
    site = engine.select_site(fleet, "lagos-energy", "ikeja-minigrid")
    with pytest.raises(ValueError, match="unknown model"):
        engine.forecast_site(site, LOAD_KW, horizon=6, model="does_not_exist")
