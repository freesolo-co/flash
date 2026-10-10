"""Registry edge coverage pins naive timestamps and tenant-scoped tombstones."""

from __future__ import annotations

from datetime import UTC

from flash.serving.src.store.registry import AdapterRegistry, _parse_iso


def test_parse_iso_assigns_utc_to_naive_timestamp() -> None:
    parsed = _parse_iso("2026-07-14T12:34:56")

    assert parsed is not None
    assert parsed.tzinfo is UTC
    assert parsed.isoformat() == "2026-07-14T12:34:56+00:00"


def test_remove_unknown_id_creates_and_preserves_tombstone() -> None:
    registry = AdapterRegistry()

    key = ("org-1", "missing/final")
    assert registry.remove(*key) is None
    first = registry._tombstones[key]
    assert first is not None

    assert registry.remove(*key) is None
    assert registry._tombstones[key] == first
