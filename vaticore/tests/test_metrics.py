from __future__ import annotations

import numpy as np
import pytest

from vaticore.evaluation.metrics import (
    coverage,
    mae,
    mean_pinball_loss,
    pinball_loss,
    rmse,
)


def test_pinball_known_value() -> None:
    # actual=10, predicted=8, q=0.5 -> 0.5 * |2| = 1.0
    assert pinball_loss([10.0], [8.0], 0.5) == pytest.approx(1.0)


def test_pinball_penalises_under_and_over_asymmetrically() -> None:
    # q=0.9 penalises under prediction (actual above forecast) heavily.
    under = pinball_loss([10.0], [8.0], 0.9)
    over = pinball_loss([6.0], [8.0], 0.9)
    assert under > over


def test_pinball_ignores_nan_actuals() -> None:
    value = pinball_loss([10.0, np.nan], [8.0, 8.0], 0.5)
    assert value == pytest.approx(1.0)


def test_mean_pinball_averages_quantiles() -> None:
    preds = {0.1: [8.0], 0.5: [8.0], 0.9: [8.0]}
    assert mean_pinball_loss([10.0], preds) > 0


def test_mae_and_rmse() -> None:
    assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 5.0]) == pytest.approx(2.0 / 3.0)
    assert rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx(np.sqrt(12.5))


def test_coverage_fraction() -> None:
    actual = [1.0, 5.0, 9.0, 2.0]
    lower = [0.0, 0.0, 0.0, 0.0]
    upper = [4.0, 4.0, 4.0, 4.0]
    # 1, 2 inside; 5, 9 outside -> 0.5
    assert coverage(actual, lower, upper) == pytest.approx(0.5)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        pinball_loss([1.0, 2.0], [1.0], 0.5)
