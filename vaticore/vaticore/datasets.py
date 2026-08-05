"""Synthetic demo data.

Realistic enough to demo and to exercise the pipeline, deterministic so results
are reproducible. This is not a substitute for real operator data, it is a
stand in for demos, examples and the dashboard until real feeds are connected.

The generator models the structure that matters for forecasting: a daily and
weekly load cycle, a weather driven solar profile, and occasional data gaps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vaticore import schemas
from vaticore.schemas import (
    GENERATION_KW,
    LOAD_KW,
    OPERATOR_ID,
    SITE_ID,
    TIMESTAMP,
)


def make_synthetic_site(
    operator_id: str,
    site_id: str,
    *,
    days: int = 90,
    seed: int = 0,
    base_load_kw: float = 120.0,
    solar_peak_kw: float = 90.0,
    gap_fraction: float = 0.01,
) -> pd.DataFrame:
    """Generate one site of hourly load and solar generation.

    Parameters
    ----------
    gap_fraction:
        Fraction of rows to blank out as missing, so ingestion and models are
        exercised against gaps rather than perfect data.
    """
    rng = np.random.default_rng(seed)
    n = days * 24
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()

    # Load: daily bell peaking late afternoon, lower on weekends, AR(1) noise.
    daily = np.sin((hour - 6) / 24 * 2 * np.pi)
    weekly = np.where(dow >= 5, 0.85, 1.0)
    load = base_load_kw * (1.0 + 0.35 * daily) * weekly
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.7 * noise[t - 1] + rng.normal(0.0, 4.0)
    load = np.clip(load + noise, a_min=0.0, a_max=None)

    # Solar: bell during daylight, scaled by a slowly varying cloud factor.
    daylight = np.clip(np.sin((hour - 6) / 12 * np.pi), 0.0, None)
    daylight[(hour < 6) | (hour > 18)] = 0.0
    cloud = np.clip(0.75 + 0.25 * np.sin(np.arange(n) / 47.0) + rng.normal(0.0, 0.1, n), 0.1, 1.0)
    generation = np.clip(solar_peak_kw * daylight * cloud, a_min=0.0, a_max=None)

    frame = pd.DataFrame(
        {
            OPERATOR_ID: operator_id,
            SITE_ID: site_id,
            TIMESTAMP: index,
            LOAD_KW: load,
            GENERATION_KW: generation,
        }
    )

    if gap_fraction > 0:
        n_gaps = int(n * gap_fraction)
        gap_rows = rng.choice(n, size=n_gaps, replace=False)
        frame.loc[gap_rows, [LOAD_KW, GENERATION_KW]] = np.nan

    return frame


def make_synthetic_fleet(days: int = 90, seed: int = 0) -> pd.DataFrame:
    """A small multi operator, multi site fleet for demos and the dashboard."""
    sites = [
        ("lagos-energy", "ikeja-minigrid", 120.0, 90.0),
        ("lagos-energy", "lekki-ci-solar", 220.0, 180.0),
        ("accra-power", "tema-industrial", 300.0, 140.0),
    ]
    frames = [
        make_synthetic_site(
            op, site, days=days, seed=seed + i, base_load_kw=base, solar_peak_kw=solar
        )
        for i, (op, site, base, solar) in enumerate(sites)
    ]
    fleet = pd.concat(frames, ignore_index=True)
    return schemas.validate(fleet)
