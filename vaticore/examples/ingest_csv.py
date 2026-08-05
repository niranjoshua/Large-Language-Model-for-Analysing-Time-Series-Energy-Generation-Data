"""Ingest an operator CSV into the store.

The path from a real operator's meter or inverter export to a queryable,
validated, multi-tenant store, in one script. Point it at a CSV, map its columns
onto the internal names, and it validates and upserts every row.

    uv run python examples/ingest_csv.py path/to/site.csv \
        --operator lagos-energy --site ikeja-minigrid \
        --timestamp-col ts --load-col kw --timezone Africa/Lagos

If your CSV already uses the internal column names (operator_id, site_id,
timestamp, load_kw, generation_kw), the mapping flags are optional.
"""

from __future__ import annotations

import argparse

import pandas as pd

from vaticore.ingestion import normalise
from vaticore.schemas import LOAD_KW, OPERATOR_ID, SITE_ID, TIMESTAMP
from vaticore.storage import get_repository


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest an operator CSV into the store.")
    parser.add_argument("csv", help="Path to the CSV file.")
    parser.add_argument("--operator", help="operator_id to assign if not in the CSV.")
    parser.add_argument("--site", help="site_id to assign if not in the CSV.")
    parser.add_argument("--timestamp-col", help="Source column holding the timestamp.")
    parser.add_argument("--load-col", help="Source column holding load in kW.")
    parser.add_argument("--generation-col", help="Source column holding generation in kW.")
    parser.add_argument("--timezone", help="Source timezone, e.g. Africa/Lagos.")
    args = parser.parse_args()

    column_map: dict[str, str] = {}
    if args.timestamp_col:
        column_map[args.timestamp_col] = TIMESTAMP
    if args.load_col:
        column_map[args.load_col] = LOAD_KW
    if args.generation_col:
        column_map[args.generation_col] = "generation_kw"

    raw = pd.read_csv(args.csv)
    raw = raw.rename(columns=column_map)
    if args.operator and OPERATOR_ID not in raw.columns:
        raw[OPERATOR_ID] = args.operator
    if args.site and SITE_ID not in raw.columns:
        raw[SITE_ID] = args.site

    frame = normalise(raw, source_timezone=args.timezone)

    repo = get_repository()
    written = repo.upsert(frame)
    print(f"Ingested and stored {written} rows for {frame[OPERATOR_ID].nunique()} operator(s).")
    print(repo.list_sites().to_string(index=False))


if __name__ == "__main__":
    main()
