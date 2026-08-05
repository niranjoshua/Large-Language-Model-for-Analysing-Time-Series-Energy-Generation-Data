"""Request and response models for the API.

These are the public contract of the service. They are deliberately separate
from the internal pandas/dataclass types so the wire format can stay stable
while internals evolve.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from vaticore.schemas import GENERATION_KW, LOAD_KW

Target = str  # constrained in the request model below


class ForecastRequest(BaseModel):
    operator_id: str
    site_id: str
    target: str = Field(default=LOAD_KW, description=f"{LOAD_KW} or {GENERATION_KW}")
    horizon: int = Field(default=24, ge=1, le=168)
    model: str = Field(default="quantile_gbm")
    quantiles: list[float] = Field(default=[0.1, 0.5, 0.9])


class ForecastPoint(BaseModel):
    timestamp: datetime
    quantiles: dict[str, float]


class ForecastResponse(BaseModel):
    operator_id: str
    site_id: str
    target: str
    model: str
    points: list[ForecastPoint]


class AdvisoryRequest(BaseModel):
    operator_id: str
    site_id: str
    horizon: int = Field(default=24, ge=1, le=168)
    usable_battery_kwh: float = Field(gt=0.0)
    step_hours: float = Field(default=1.0, gt=0.0)
    model: str = Field(default="quantile_gbm")


class AdvisoryResponse(BaseModel):
    operator_id: str
    site_id: str
    horizon_hours: float
    expected_net_load_kwh: float
    conservative_net_load_kwh: float
    recommended_reserve_kwh: float
    genset_recommended: bool
    note: str


class SiteInfo(BaseModel):
    operator_id: str
    site_id: str
    n_observations: int
    first_timestamp: datetime
    last_timestamp: datetime
