"""Hermetic delegation and capacity-error coverage for the Vast provider facade."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from flash.providers.core.base import AllocationConstraints, CapacityLookupError, JobHandle
from flash.providers.vast.execution.provider import VastProvider


def test_vast_provider_delegates_credentials_pricing_gc_and_orphan_sweep(monkeypatch) -> None:
    """Thin provider methods must preserve arguments and return values from their Vast helpers."""
    from flash.providers.vast import jobs
    from flash.providers.vast.client import preflight, pricing

    provider = VastProvider()
    calls = []
    monkeypatch.setattr(
        preflight,
        "missing_credentials",
        lambda require_hf: calls.append(("credentials", require_hf)) or ["missing"],
    )
    monkeypatch.setattr(
        pricing,
        "hourly_rate",
        lambda gpu: calls.append(("rate", gpu)) or 1.25,
    )
    monkeypatch.setattr(
        jobs,
        "destroy_run_instances",
        lambda run_id: calls.append(("gc", run_id)),
    )
    monkeypatch.setattr(
        jobs,
        "sweep_orphans",
        lambda **kwargs: calls.append(("sweep", kwargs)) or [11],
    )

    assert provider._missing_credentials(False) == ["missing"]
    assert provider._hourly_rate("H100") == 1.25
    assert provider._gc("flash-1") is None
    assert provider._sweep_orphans(active_labels={"live"}, known_labels={"live", "done"}) == [11]
    assert calls == [
        ("credentials", False),
        ("rate", "H100"),
        ("gc", "flash-1"),
        ("sweep", {"active_labels": {"live"}, "known_labels": {"live", "done"}}),
    ]


def test_live_candidates_returns_empty_when_no_gpu_class_fits(monkeypatch) -> None:
    """A VRAM request above every Vast class must avoid a pointless market lookup."""
    from flash.providers.vast.client import pricing

    provider = VastProvider()
    monkeypatch.setattr(
        provider,
        "gpu_classes",
        lambda: [SimpleNamespace(name="small", vram_gb=8)],
    )
    monkeypatch.setattr(
        pricing,
        "live_candidate_rates",
        lambda *args, **kwargs: pytest.fail("market lookup must be skipped"),
    )

    assert provider.live_candidates(16, AllocationConstraints()) == []


def test_live_candidates_wraps_market_failures_as_capacity_errors(monkeypatch) -> None:
    """Transient Vast market failures must remain retryable allocator capacity errors."""
    from flash.providers.vast.client import pricing

    provider = VastProvider()
    monkeypatch.setattr(
        provider,
        "gpu_classes",
        lambda: [SimpleNamespace(name="H100", vram_gb=80)],
    )
    monkeypatch.setattr(
        pricing,
        "live_candidate_rates",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("market down")),
    )

    with pytest.raises(CapacityLookupError, match="vast live capacity lookup failed"):
        provider.live_candidates(40, AllocationConstraints(disk_gb=100, max_wall_seconds=60))


def _generic_handle() -> JobHandle:
    return JobHandle.from_dict(
        {
            "provider": "vast",
            "instance_id": 17,
            "offer_id": 18,
            "machine_id": 19,
            "label": "flash-1-a0",
            "gpu": "H100",
            "hourly_usd": 1.5,
            "attempt": 0,
            "started_ts": 1.0,
        }
    )


def test_cancel_converts_to_strict_handle_and_delegates_serialized_payload(monkeypatch) -> None:
    """Cancellation must validate persisted Vast identity before handing jobs a canonical payload."""
    from flash.providers.vast import jobs

    payloads = []
    monkeypatch.setattr(jobs, "cancel", payloads.append)

    VastProvider().cancel(_generic_handle())

    assert payloads == [_generic_handle().to_dict()]


def test_run_instances_remaining_delegates_the_run_id(monkeypatch) -> None:
    """Billing confirmation must return the exact instance ids reported by the Vast jobs layer."""
    from flash.providers.vast import jobs

    seen = []
    monkeypatch.setattr(
        jobs,
        "run_instances_remaining",
        lambda run_id: seen.append(run_id) or [7, 8],
    )

    assert VastProvider().run_instances_remaining("flash-1") == [7, 8]
    assert seen == ["flash-1"]
