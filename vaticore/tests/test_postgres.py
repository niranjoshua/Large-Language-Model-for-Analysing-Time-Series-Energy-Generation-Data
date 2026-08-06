"""Postgres backend conformance tests.

These run only when a test database is configured via VATICORE_TEST_POSTGRES_URL
(for example the docker-compose database, or a Postgres service in CI). Without
it they skip, so the default suite stays fast and dependency-free while the same
checks still validate Postgres wherever a database is available.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from vaticore.datasets import make_synthetic_fleet, make_synthetic_site

_URL = os.environ.get("VATICORE_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not _URL, reason="set VATICORE_TEST_POSTGRES_URL to run Postgres conformance tests"
)


@pytest.fixture
def repo():  # type: ignore[no-untyped-def]
    pytest.importorskip("psycopg")
    from vaticore.storage.postgres_repository import PostgresRepository

    assert _URL is not None
    repository = PostgresRepository(_URL)
    # Start each test from a clean table.
    repository._conn.execute("TRUNCATE observations")
    yield repository
    repository._conn.execute("TRUNCATE observations")
    repository.close()


def test_upsert_then_read_roundtrip(repo) -> None:  # type: ignore[no-untyped-def]
    site = make_synthetic_site("op1", "siteA", days=5, seed=1, gap_fraction=0.0)
    assert repo.upsert(site) == len(site)
    assert len(repo.read_history("op1", "siteA")) == len(site)


def test_upsert_is_idempotent(repo) -> None:  # type: ignore[no-untyped-def]
    site = make_synthetic_site("op1", "siteA", days=3, seed=2, gap_fraction=0.0)
    repo.upsert(site)
    repo.upsert(site)
    assert repo.count() == len(site)


def test_scoping_and_time_range(repo) -> None:  # type: ignore[no-untyped-def]
    repo.upsert(make_synthetic_fleet(days=5, seed=1))
    scoped = repo.read_history("lagos-energy", "ikeja-minigrid")
    assert set(scoped["operator_id"].unique()) == {"lagos-energy"}

    start = pd.Timestamp("2024-01-03", tz="UTC")
    end = pd.Timestamp("2024-01-04", tz="UTC")
    window = repo.read_history("lagos-energy", "ikeja-minigrid", start=start, end=end)
    assert window["timestamp"].min() >= start
    assert window["timestamp"].max() <= end


def test_list_sites(repo) -> None:  # type: ignore[no-untyped-def]
    repo.upsert(make_synthetic_fleet(days=4, seed=1))
    sites = repo.list_sites()
    assert len(sites) == 3
