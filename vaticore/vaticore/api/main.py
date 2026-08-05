"""FastAPI application: the programmatic surface of the product.

Endpoints are thin. All forecasting, backtesting and advisory logic lives in the
engine facade, so the API and the dashboard always agree. The data source is
injected (get_fleet), backed by synthetic demo data for now and swapped for a
real repository against operator feeds without touching the handlers.

Run with:
    uv sync --extra service
    uv run uvicorn vaticore.api.main:app --reload
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import pandas as pd

from vaticore import __version__, engine
from vaticore.api.schemas import (
    AdvisoryRequest,
    AdvisoryResponse,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    SiteInfo,
)
from vaticore.datasets import make_synthetic_fleet
from vaticore.forecasting.base import quantile_column
from vaticore.schemas import GENERATION_KW, LOAD_KW, OPERATOR_ID, SITE_ID, TIMESTAMP

if TYPE_CHECKING:
    from fastapi import FastAPI

    from vaticore.storage import DuckDBRepository

_VALID_TARGETS = {LOAD_KW, GENERATION_KW}


@lru_cache(maxsize=1)
def get_repository() -> DuckDBRepository:
    """Cached repository, seeded with demo data if the store is empty.

    Seeding means a fresh deployment is never blank. Replace the seed with a real
    ingestion feed and this same store serves live operator data.
    """
    from vaticore.storage import get_repository as build_repository

    repo = build_repository()
    if repo.count() == 0:
        repo.upsert(make_synthetic_fleet(days=90, seed=1))
    return repo


def get_fleet() -> pd.DataFrame:
    """Data source dependency: read the full fleet from the repository."""
    return get_repository().read_fleet()


def create_app() -> FastAPI:
    """Application factory."""
    from fastapi import Depends, FastAPI, HTTPException

    app = FastAPI(
        title="Vaticore",
        version=__version__,
        summary="Probabilistic energy forecasting for distributed energy operators.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/models")
    def models() -> list[str]:
        return engine.available_models()

    @app.get("/sites", response_model=list[SiteInfo])
    def sites(fleet: pd.DataFrame = Depends(get_fleet)) -> list[SiteInfo]:
        out: list[SiteInfo] = []
        for (operator_id, site_id), group in fleet.groupby([OPERATOR_ID, SITE_ID], sort=True):
            out.append(
                SiteInfo(
                    operator_id=str(operator_id),
                    site_id=str(site_id),
                    n_observations=len(group),
                    first_timestamp=group[TIMESTAMP].min(),
                    last_timestamp=group[TIMESTAMP].max(),
                )
            )
        return out

    def _site_history(fleet: pd.DataFrame, operator_id: str, site_id: str) -> pd.DataFrame:
        try:
            return engine.select_site(fleet, operator_id, site_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(
        request: ForecastRequest,
        fleet: pd.DataFrame = Depends(get_fleet),
    ) -> ForecastResponse:
        if request.target not in _VALID_TARGETS:
            raise HTTPException(status_code=400, detail=f"target must be one of {_VALID_TARGETS}")
        history = _site_history(fleet, request.operator_id, request.site_id)
        try:
            frame = engine.forecast_site(
                history,
                request.target,
                horizon=request.horizon,
                model=request.model,
                quantiles=tuple(request.quantiles),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        index = pd.DatetimeIndex(frame.index)
        points = [
            ForecastPoint(
                timestamp=ts.to_pydatetime(),
                quantiles={
                    quantile_column(q): float(frame.iloc[i][quantile_column(q)])
                    for q in request.quantiles
                },
            )
            for i, ts in enumerate(index)
        ]
        return ForecastResponse(
            operator_id=request.operator_id,
            site_id=request.site_id,
            target=request.target,
            model=request.model,
            points=points,
        )

    @app.post("/advisory", response_model=AdvisoryResponse)
    def advisory(
        request: AdvisoryRequest,
        fleet: pd.DataFrame = Depends(get_fleet),
    ) -> AdvisoryResponse:
        history = _site_history(fleet, request.operator_id, request.site_id)
        try:
            result = engine.advisory_for_site(
                history,
                horizon=request.horizon,
                usable_battery_kwh=request.usable_battery_kwh,
                step_hours=request.step_hours,
                model=request.model,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return AdvisoryResponse(
            operator_id=request.operator_id,
            site_id=request.site_id,
            horizon_hours=result.horizon_hours,
            expected_net_load_kwh=result.expected_net_load_kwh,
            conservative_net_load_kwh=result.conservative_net_load_kwh,
            recommended_reserve_kwh=result.recommended_reserve_kwh,
            genset_recommended=result.genset_recommended,
            note=result.note,
        )

    return app


app = create_app()
