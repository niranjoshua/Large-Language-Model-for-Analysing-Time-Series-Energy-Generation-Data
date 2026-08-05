"""Shared fixtures.

Small, fast, deterministic synthetic data with a known daily shape, so tests can
assert on behaviour with a known answer. Anything stochastic is seeded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vaticore.schemas import (
    GENERATION_KW,
    LOAD_KW,
    OPERATOR_ID,
    SITE_ID,
    TIMESTAMP,
)


def _daily_load(hours: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hour_of_day = hours % 24
    base = 100.0 + 40.0 * np.sin((hour_of_day - 6) / 24 * 2 * np.pi)
    noise = rng.normal(0.0, 3.0, size=hours.shape)
    return np.clip(base + noise, a_min=0.0, a_max=None)


def _daily_generation(hours: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hour_of_day = hours % 24
    # Solar bell around midday, zero at night.
    solar = np.maximum(0.0, np.sin((hour_of_day - 6) / 12 * np.pi)) * 80.0
    solar[(hour_of_day < 6) | (hour_of_day > 18)] = 0.0
    noise = rng.normal(0.0, 2.0, size=hours.shape)
    return np.clip(solar + noise, a_min=0.0, a_max=None)


def _make_site(operator_id: str, site_id: str, days: int, seed: int) -> pd.DataFrame:
    n = days * 24
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    hours = np.arange(n)
    return pd.DataFrame(
        {
            OPERATOR_ID: operator_id,
            SITE_ID: site_id,
            TIMESTAMP: index,
            LOAD_KW: _daily_load(hours, seed),
            GENERATION_KW: _daily_generation(hours, seed + 1),
        }
    )


@pytest.fixture
def single_site() -> pd.DataFrame:
    """One site, 30 days of clean hourly data."""
    return _make_site("op1", "siteA", days=30, seed=7)


@pytest.fixture
def multi_site() -> pd.DataFrame:
    """Two operators/sites, so tests catch single site assumptions."""
    a = _make_site("op1", "siteA", days=20, seed=7)
    b = _make_site("op2", "siteB", days=20, seed=99)
    return pd.concat([a, b], ignore_index=True)
