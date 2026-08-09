"""Hermetic coverage for worker-metric sanitization fallbacks and persistence."""

from __future__ import annotations

import json

import flash.engine.result.accounting as accounting


def test_sanitizer_falls_back_safely_when_nested_adapter_parser_raises(monkeypatch) -> None:
    """Nested adapter fields must still redact colon-bearing refs when the canonical parser fails."""
    import flash.schema as schema

    monkeypatch.setattr(
        schema,
        "parse_adapter_storage_ref",
        lambda value: (_ for _ in ()).throw(RuntimeError("parser failed")),
    )

    assert accounting.sanitize_worker_metrics(
        {"init_from_adapter": "owner/repo:rl/run", "plain": "visible"}
    ) == {
        "init_from_adapter": accounting._PRIVATE_SOURCE_MARKER,
        "plain": "visible",
    }


def test_sanitizer_recurses_through_lists_and_tuples() -> None:
    """Container normalization must sanitize every nested element and convert tuples to JSON arrays."""
    value = [
        {"hf_repo": "private/repo"},
        ({"init_from_adapter_revision": "a" * 40}, "plain"),
    ]

    assert accounting.sanitize_worker_metrics(value) == [
        {"hf_repo": accounting._PRIVATE_SOURCE_MARKER},
        [{"init_from_adapter_revision": accounting._PRIVATE_SOURCE_MARKER}, "plain"],
    ]


def test_sanitizer_redacts_direct_adapter_ref_strings() -> None:
    """Direct strings that parse as storage refs must not escape outside named adapter fields."""
    assert (
        accounting.sanitize_worker_metrics("owner/repo:rl/run") == accounting._PRIVATE_SOURCE_MARKER
    )


def test_sanitizer_preserves_direct_strings_when_parser_raises(monkeypatch) -> None:
    """Ambiguous direct strings must remain unchanged when the canonical parser itself is unavailable."""
    import flash.schema as schema

    monkeypatch.setattr(
        schema,
        "parse_adapter_storage_ref",
        lambda value: (_ for _ in ()).throw(RuntimeError("parser failed")),
    )

    assert accounting.sanitize_worker_metrics("owner/repo:rl/run") == "owner/repo:rl/run"


def test_run_metrics_save_writes_sanitized_json(tmp_path) -> None:
    """Metrics persistence must write valid sanitized JSON through the public save boundary."""
    path = tmp_path / "metrics.json"
    metrics = accounting.RunMetrics(
        phase="sft",
        train_tokens=12,
        notes={"hf_repo": "private/repo"},
    )

    metrics.save(str(path))

    saved = json.loads(path.read_text())
    assert saved["phase"] == "sft"
    assert saved["train_tokens"] == 12
    assert saved["notes"]["hf_repo"] == accounting._PRIVATE_SOURCE_MARKER
