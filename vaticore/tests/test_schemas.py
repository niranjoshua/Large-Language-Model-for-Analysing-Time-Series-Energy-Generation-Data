from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError

from vaticore import schemas
from vaticore.schemas import LOAD_KW, TIMESTAMP


def test_validate_accepts_clean_frame(single_site: pd.DataFrame) -> None:
    validated = schemas.validate(single_site)
    assert len(validated) == len(single_site)
    assert str(validated[TIMESTAMP].dtype) == "datetime64[ns, UTC]"


def test_validate_rejects_negative_load(single_site: pd.DataFrame) -> None:
    bad = single_site.copy()
    bad.loc[0, LOAD_KW] = -5.0
    with pytest.raises(SchemaError):
        schemas.validate(bad)


def test_validate_allows_missing_targets(single_site: pd.DataFrame) -> None:
    gapped = single_site.copy()
    gapped.loc[5:10, LOAD_KW] = pd.NA
    validated = schemas.validate(gapped)
    assert validated[LOAD_KW].isna().sum() == 6
