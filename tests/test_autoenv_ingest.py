"""Dataset ref resolution: local fetch works offline; unsupported schemes fail clearly."""

from __future__ import annotations

from pathlib import Path

import pytest

import autoenv
from autoenv.ingest import can_fetch, fetch_rows
from autoenv.ingest.sources import DatasetUnavailable

DATA = Path(autoenv.__file__).parent / "cases" / "data" / "arithmetic_smoke_train.jsonl"


def test_fetch_local_jsonl_canonicalizes_rows():
    rows = fetch_rows(str(DATA))
    assert rows
    assert set(rows[0]) == {"input", "output"}


def test_can_fetch_probe_on_local_file():
    ok, detail = can_fetch(str(DATA))
    assert ok
    assert "resolved" in detail


@pytest.mark.parametrize(
    "ref",
    ["github:owner/repo", "git+https://github.com/owner/repo", "s3://bucket/data.jsonl"],
)
def test_unsupported_scheme_raises_clear_error(ref):
    with pytest.raises(DatasetUnavailable, match="unsupported dataset ref scheme"):
        fetch_rows(ref)


def test_missing_local_file_raises():
    with pytest.raises(DatasetUnavailable, match="not found"):
        fetch_rows("/no/such/dataset.jsonl")
