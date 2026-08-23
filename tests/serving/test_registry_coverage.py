"""Registry edge coverage pins naive timestamps, unknown tombstones, and alias path handling.

The tests use in-memory records only and never download adapter artifacts.
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

from flash.serving.src.registry import AdapterRegistry, _parse_iso
from flash.serving.src.schemas import AdapterRecord


def _alias() -> AdapterRecord:
    return AdapterRecord.model_validate(
        {
            "adapter_id": "run",
            "repo_id": "org/run",
            "base_model": "Qwen/Qwen3.5-0.8B",
            "org_id": "org-1",
            "thinking": False,
            "metadata": {
                "record_type": "alias",
                "run_id": "run",
                "alias_of": "run@final." + "a" * 40,
            },
        }
    )


def test_parse_iso_assigns_utc_to_naive_timestamp() -> None:
    parsed = _parse_iso("2026-07-14T12:34:56")

    assert parsed is not None
    assert parsed.tzinfo is UTC
    assert parsed.isoformat() == "2026-07-14T12:34:56+00:00"


def test_remove_unknown_id_creates_and_preserves_tombstone() -> None:
    registry = AdapterRegistry()

    assert registry.remove("missing") is None
    first = registry._tombstones["missing"]
    assert first is not None

    assert registry.remove("missing") is None
    assert registry._tombstones["missing"] == first


def test_alias_local_path_is_never_stale() -> None:
    registry = AdapterRegistry()
    cached_record = AdapterRecord.model_validate(
        {
            "adapter_id": "run",
            "repo_id": "org/old-run",
            "base_model": "Qwen/Qwen3.5-0.8B",
            "org_id": "org-1",
            "thinking": False,
            "metadata": {"hf_revision": "b" * 40},
        }
    )
    registry.set_local_path(cached_record, Path("/tmp/old-run"))

    assert registry.local_path_is_stale(_alias()) is False
