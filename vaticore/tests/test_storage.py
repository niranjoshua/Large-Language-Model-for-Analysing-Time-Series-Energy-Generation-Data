from __future__ import annotations

import pandas as pd
import pytest

from vaticore.datasets import make_synthetic_fleet, make_synthetic_site
from vaticore.storage import DuckDBRepository, get_repository
from vaticore.storage.duckdb_repository import _path_from_url


@pytest.fixture
def repo() -> DuckDBRepository:
    return DuckDBRepository(":memory:")


def test_upsert_then_read_roundtrip(repo: DuckDBRepository) -> None:
    site = make_synthetic_site("op1", "siteA", days=5, seed=1, gap_fraction=0.0)
    written = repo.upsert(site)
    assert written == len(site)
    history = repo.read_history("op1", "siteA")
    assert len(history) == len(site)


def test_scoping_isolates_sites(repo: DuckDBRepository) -> None:
    repo.upsert(make_synthetic_fleet(days=4, seed=1))
    a = repo.read_history("lagos-energy", "ikeja-minigrid")
    assert set(a["operator_id"].unique()) == {"lagos-energy"}
    assert set(a["site_id"].unique()) == {"ikeja-minigrid"}


def test_upsert_is_idempotent(repo: DuckDBRepository) -> None:
    site = make_synthetic_site("op1", "siteA", days=3, seed=2, gap_fraction=0.0)
    repo.upsert(site)
    repo.upsert(site)  # same primary keys, must not duplicate
    assert repo.count() == len(site)


def test_upsert_updates_existing_value(repo: DuckDBRepository) -> None:
    site = make_synthetic_site("op1", "siteA", days=2, seed=3, gap_fraction=0.0)
    repo.upsert(site)
    changed = site.copy()
    changed["load_kw"] = 999.0
    repo.upsert(changed)
    history = repo.read_history("op1", "siteA")
    assert (history["load_kw"] == 999.0).all()


def test_time_range_filter(repo: DuckDBRepository) -> None:
    site = make_synthetic_site("op1", "siteA", days=10, seed=4, gap_fraction=0.0)
    repo.upsert(site)
    start = pd.Timestamp("2024-01-05", tz="UTC")
    end = pd.Timestamp("2024-01-06", tz="UTC")
    window = repo.read_history("op1", "siteA", start=start, end=end)
    assert window["timestamp"].min() >= start
    assert window["timestamp"].max() <= end


def test_list_sites_summary(repo: DuckDBRepository) -> None:
    repo.upsert(make_synthetic_fleet(days=4, seed=1))
    sites = repo.list_sites()
    assert len(sites) == 3
    assert "n_observations" in sites.columns


def test_read_empty_returns_valid_empty_frame(repo: DuckDBRepository) -> None:
    empty = repo.read_history("nobody", "nowhere")
    assert len(empty) == 0
    assert list(empty.columns)[:3] == ["operator_id", "site_id", "timestamp"]


def test_path_parsing() -> None:
    assert _path_from_url("duckdb:///vaticore.duckdb") == "vaticore.duckdb"
    assert _path_from_url("duckdb:///:memory:") == ":memory:"
    assert _path_from_url("/tmp/x.duckdb") == "/tmp/x.duckdb"


def test_factory_rejects_unknown_scheme() -> None:
    from vaticore.config import Settings

    settings = Settings(database_url="mysql://localhost/db")
    with pytest.raises(ValueError, match="unsupported database_url scheme"):
        get_repository(settings)


def test_factory_builds_duckdb() -> None:
    from vaticore.config import Settings

    repo = get_repository(Settings(database_url="duckdb:///:memory:"))
    assert isinstance(repo, DuckDBRepository)
