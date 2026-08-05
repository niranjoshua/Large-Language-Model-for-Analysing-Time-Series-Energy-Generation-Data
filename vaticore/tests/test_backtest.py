from __future__ import annotations

import pandas as pd

from vaticore.evaluation.backtest import backtest_site
from vaticore.forecasting import PersistenceForecaster
from vaticore.schemas import LOAD_KW


def test_backtest_reports_candidate_and_baseline(single_site: pd.DataFrame) -> None:
    result = backtest_site(
        single_site,
        target=LOAD_KW,
        make_model=lambda: PersistenceForecaster(target=LOAD_KW),
        horizon=24,
        initial=24 * 14,
        step=24,
        model_name="persistence_copy",
    )
    summary = result.summary()
    assert "persistence" in summary.index
    assert "persistence_copy" in summary.index
    # Every fold produced finite pinball loss.
    assert result.to_frame()["pinball"].notna().all()


def test_backtest_raises_on_insufficient_history(single_site: pd.DataFrame) -> None:
    tiny = single_site.iloc[:10]
    try:
        backtest_site(
            tiny,
            target=LOAD_KW,
            make_model=lambda: PersistenceForecaster(target=LOAD_KW),
            horizon=24,
            initial=24 * 14,
        )
    except ValueError as exc:
        assert "not enough history" in str(exc)
    else:
        raise AssertionError("expected ValueError on insufficient history")


def test_folds_are_chronological_and_non_overlapping(single_site: pd.DataFrame) -> None:
    result = backtest_site(
        single_site,
        target=LOAD_KW,
        make_model=lambda: PersistenceForecaster(target=LOAD_KW),
        horizon=24,
        initial=24 * 10,
        step=24,
    )
    folds = sorted({f.fold for f in result.folds})
    assert folds == list(range(len(folds)))
