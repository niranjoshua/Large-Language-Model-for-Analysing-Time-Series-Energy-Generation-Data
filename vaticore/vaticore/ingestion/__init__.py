"""Ingestion: CSV/API intake, validation and gap handling."""

from vaticore.ingestion.loader import GapReport, load_csv, normalise, report_gaps

__all__ = ["GapReport", "load_csv", "normalise", "report_gaps"]
