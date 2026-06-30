"""Provider registry + ``base.Provider`` interface coverage (CPU-only, offline).

RunPod (always on) and the instance-based complements — Lambda (opt-in via LAMBDA_API_KEY) and
Vast (opt-in via VAST_API_KEY) — all implement the SAME ``base.Provider`` interface
behind the SAME module layout, so the orchestrator/allocator treat them interchangeably."""

from __future__ import annotations

import importlib

import pytest

PROVIDER_MODULES = ("api", "auth", "pricing", "gpus", "jobs", "train", "preflight")
PROVIDER_METHODS = (
    "is_configured",
    "preflight",
    "gpu_classes",
    "hourly_rate",
    "submit_run",
    "poll",
    "cancel",
    "destroy",
    "gc",
    "sweep_orphans",
)


_PKG = {"runpod": "runpod", "lambda": "lambdalabs", "vast": "vast"}


def test_registry_lists_all_providers():
    from flash.providers import PROVIDER_NAMES, get_provider

    assert PROVIDER_NAMES == ("runpod", "lambda", "vast")
    assert get_provider("RunPod ").name == "runpod"
    assert get_provider(" Lambda ").name == "lambda"  # case/space-insensitive
    assert get_provider(" Vast ").name == "vast"
    with pytest.raises(KeyError):
        get_provider("hyperstack")  # removed; not registered


@pytest.mark.parametrize("provider", ["runpod", "lambda", "vast"])
def test_provider_implements_the_interface(provider):
    from flash.providers import get_provider
    from flash.providers.base import Provider

    prov = get_provider(provider)
    assert isinstance(prov, Provider)
    for meth in PROVIDER_METHODS:
        assert callable(getattr(prov, meth)), f"{provider} missing {meth}"


@pytest.mark.parametrize("provider", ["runpod", "lambda", "vast"])
def test_module_layout(provider):
    """Every provider subpackage exposes the SAME public module set (the symmetry contract)."""
    for mod in PROVIDER_MODULES:
        importlib.import_module(f"flash.providers.{_PKG[provider]}.{mod}")
    pkg = importlib.import_module(f"flash.providers.{_PKG[provider]}")
    assert hasattr(pkg, "PROVIDER")


@pytest.mark.parametrize("provider", ["lambda", "vast"])
def test_method_signatures_match_runpod(provider):
    """The interface methods take the same parameters on every provider (swappable)."""
    import inspect

    from flash.providers import get_provider

    rp = get_provider("runpod")
    other = get_provider(provider)
    for meth in PROVIDER_METHODS:
        rs = list(inspect.signature(getattr(rp, meth)).parameters)
        os_ = list(inspect.signature(getattr(other, meth)).parameters)
        assert rs == os_, f"{meth} param mismatch: runpod={rs} {provider}={os_}"


@pytest.mark.parametrize("provider", ["runpod", "lambda", "vast"])
def test_setup_vs_training_gate_is_the_one_canonical_helper(provider):
    """Every poll loop (runpod, lambda, vast) must draw the setup-vs-training stall boundary from the
    SAME canonical is_training_heartbeat helper in _poll (so the rule can't drift between providers). The
    helper keeps the cold-start pings — including model_prefetching / *_initializing — under the wide
    setup grace, flips to the tight window only on a COMPLETED-step rl_step/sft_step, and also flips on
    POST-training stages (so a hung teardown isn't left under setup grace)."""
    from flash.providers._poll import (
        SETUP_HEARTBEAT_STAGES,
        STEP_GATED_STAGES,
        is_training_heartbeat,
    )

    jobs = importlib.import_module(f"flash.providers.{_PKG[provider]}.jobs")
    # The provider must reference the shared helper, not a private copy that can drift.
    assert jobs.is_training_heartbeat is is_training_heartbeat

    # The slow cold-start pings must count as setup (kept under the wide setup grace).
    for stage in ("model_prefetching", "model_prefetched", "sft_initializing", "rl_initializing"):
        assert stage in SETUP_HEARTBEAT_STAGES, f"{stage} must be treated as setup, not training"
        assert is_training_heartbeat(stage, 9) is False
    # Per-step training heartbeats are step-gated: NOT setup, but only flip the window at step >= 1.
    assert sorted(STEP_GATED_STAGES) == ["rl_step", "sft_step"]
    assert is_training_heartbeat("rl_step", 0) is False  # cold first step keeps setup grace
    assert is_training_heartbeat("rl_step", 1) is True
    # Post-training stages flip to the tight window even without a step field.
    assert is_training_heartbeat("sft_trained", None) is True


def test_runpod_provider_implements_the_interface():
    from flash.providers import get_provider
    from flash.providers.base import Provider

    prov = get_provider("runpod")
    assert isinstance(prov, Provider)
    for meth in PROVIDER_METHODS:
        assert callable(getattr(prov, meth)), f"runpod missing {meth}"


def test_runpod_module_layout():
    for mod in PROVIDER_MODULES:
        importlib.import_module(f"flash.providers.runpod.{mod}")
    pkg = importlib.import_module("flash.providers.runpod")
    assert hasattr(pkg, "PROVIDER")


def test_gpu_classes_match_runpod_rows():
    from flash.providers import get_provider
    from flash.providers.base import GPU_INFO

    rp = {g.name for g in get_provider("runpod").gpu_classes()}
    assert rp == {g.name for g in GPU_INFO.values() if g.enum_member}


def test_sweep_orphans_is_part_of_the_protocol():
    from flash.providers import get_provider
    from flash.providers.base import Provider

    assert hasattr(get_provider("runpod"), "sweep_orphans")
    assert get_provider("runpod").sweep_orphans(active_labels={"flash-x"}) == []
    assert "sweep_orphans" in dir(Provider)


def test_run_instances_remaining_is_optional_not_required_by_protocol():
    # Copilot: run_instances_remaining is an OPTIONAL capability (Vast enumerates billable instances by
    # run label; RunPod serverless self-reaps). It must NOT be on the @runtime_checkable Provider
    # Protocol, or isinstance(runpod_provider, Provider) would go False and break the symmetry checks
    # above. Vast implements it; RunPod does not; both still satisfy Provider. Detected via getattr.
    from flash.providers import get_provider
    from flash.providers.base import Provider

    assert "run_instances_remaining" not in dir(Provider)  # not a required Protocol member
    assert hasattr(get_provider("vast"), "run_instances_remaining")  # Vast provides the capability
    assert not hasattr(get_provider("runpod"), "run_instances_remaining")  # RunPod opts out
    assert isinstance(get_provider("runpod"), Provider)  # still a Provider despite opting out


def test_static_pricing():
    pricing = importlib.import_module("flash.providers.runpod.pricing")
    assert pricing.hourly_rate("RTX 5090") == pytest.approx(0.99)


def test_allocator_picks_runpod_candidate(monkeypatch):
    from flash.providers.allocator import allocate

    a = allocate("Qwen/Qwen3.5-0.8B", "sft")
    assert a.provider == "runpod"
    assert a.gpu == "RTX 4090"


def test_allocation_summary_formats_runpod_choice(monkeypatch):
    from flash.providers.allocator import allocation_summary
    from flash.providers.base import Allocation, Candidate

    cand = Candidate("runpod", "RTX 4090", 0.69, 24)
    alloc = Allocation(
        provider="runpod",
        gpu="RTX 4090",
        hourly_usd=0.69,
        min_vram_gb=24,
        candidates=(cand,),
    )
    out = allocation_summary(alloc)
    assert "RTX 4090 on runpod" in out


def test_jobhandle_roundtrip_tags_provider():
    from flash.providers.base import JobHandle

    h = JobHandle(provider="runpod", data={"endpoint_id": "ep", "job_id": "j"})
    d = h.to_dict()
    assert d["provider"] == "runpod"
    back = JobHandle.from_dict(d)
    assert back.provider == "runpod"
    assert back.data == {"endpoint_id": "ep", "job_id": "j"}
    assert JobHandle.from_dict({"endpoint_id": "ep", "job_id": "j"}).provider == "runpod"


def test_provider_cancel_destroy_dispatch(monkeypatch):
    from flash.providers import get_provider
    from flash.providers.base import JobHandle
    from flash.providers.runpod import api as rp_api

    cancelled, deleted = [], []
    monkeypatch.setattr(rp_api, "cancel_job", lambda e, j: cancelled.append((e, j)))
    monkeypatch.setattr(rp_api, "delete_endpoint", lambda e: deleted.append(e) or True)
    handle = JobHandle("runpod", {"endpoint_id": "ep", "job_id": "j"})
    get_provider("runpod").cancel(handle)
    get_provider("runpod").destroy(handle)
    assert cancelled == [("ep", "j")]
    assert deleted == ["ep"]
