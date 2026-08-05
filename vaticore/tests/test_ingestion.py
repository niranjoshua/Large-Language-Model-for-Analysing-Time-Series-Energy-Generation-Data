from __future__ import annotations

import pandas as pd
import pytest

from vaticore.ingestion import load_csv, normalise, report_gaps
from vaticore.schemas import LOAD_KW, OPERATOR_ID, SITE_ID, TIMESTAMP


def _raw_naive() -> pd.DataFrame:
    return pd.DataFrame(
        {
            OPERATOR_ID: ["op1", "op1", "op1"],
            SITE_ID: ["siteA", "siteA", "siteA"],
            TIMESTAMP: ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00"],
            LOAD_KW: [10.0, 11.0, 12.0],
        }
    )


def test_normalise_localises_naive_timestamps() -> None:
    out = normalise(_raw_naive(), source_timezone="Africa/Lagos")
    assert str(out[TIMESTAMP].dtype) == "datetime64[ns, UTC]"
    # Lagos is UTC+1, so 00:00 local becomes 23:00 UTC the previous day.
    assert out[TIMESTAMP].iloc[0] == pd.Timestamp("2023-12-31 23:00", tz="UTC")


def test_normalise_rejects_duplicate_timestamps() -> None:
    raw = _raw_naive()
    dup = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        normalise(dup)


def test_normalise_missing_columns_raises() -> None:
    raw = _raw_naive().drop(columns=[SITE_ID])
    with pytest.raises(ValueError, match="missing required columns"):
        normalise(raw)


def test_report_gaps_detects_missing_interval(single_site: pd.DataFrame) -> None:
    gapped = single_site.drop(index=range(10, 15)).reset_index(drop=True)
    reports = report_gaps(gapped)
    assert len(reports) == 1
    assert reports[0].n_missing == 5
    assert reports[0].completeness < 1.0


def test_load_csv_roundtrip(tmp_path, single_site: pd.DataFrame) -> None:
    path = tmp_path / "site.csv"
    single_site.to_csv(path, index=False)
    loaded = load_csv(path)
    assert len(loaded) == len(single_site)
    assert str(loaded[TIMESTAMP].dtype) == "datetime64[ns, UTC]"
