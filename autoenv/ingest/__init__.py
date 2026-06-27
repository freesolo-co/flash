"""Paper ingestion: dataset resolution (and, later, LLM-assisted manifest extraction)."""

from __future__ import annotations

from autoenv.ingest.sources import DatasetUnavailable, can_fetch, fetch_rows

__all__ = ["DatasetUnavailable", "can_fetch", "fetch_rows"]
