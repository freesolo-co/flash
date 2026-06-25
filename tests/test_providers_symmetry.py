"""Provider registry + ``base.Provider`` interface coverage (CPU-only, offline).

RunPod (always on) and Lambda (instance-based complement, opt-in via LAMBDA_API_KEY) both
implement the SAME ``base.Provider`` interface behind the SAME module layout, so the
orchestrator/allocator treat them interchangeably."""

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


def test_registry_lists_runpod_and_lambda():
    from flash.providers import PROVIDER_NAMES, get_provider

    assert PROVIDER_NAMES == ("runpod", "lambda")
    assert get_provider("RunPod ").name == "runpod"
    assert get_provider(" Lambda ").name == "lambda"  # case/space-insensitive
    with pytest.raises(KeyError):
        get_provider("vast")  # removed; not registered


@pytest.mark.parametrize("provider", ["runpod", "lambda"])
def test_provider_implements_the_interface(provider):
    from flash.providers import get_provider
    from flash.providers.base import Provider

    prov = get_provider(provider)
    assert isinstance(prov, Provider)
    for meth in PROVIDER_METHODS:
        assert callable(getattr(prov, meth)), f"{provider} missing {meth}"


@pytest.mark.parametrize("provider", ["runpod", "lambda"])
def test_module_layout(provider):
    """Both subpackages expose the SAME public module set (the symmetry contract)."""
    pkg_name = {"runpod": "runpod", "lambda": "lambdalabs"}[provider]
    for mod in PROVIDER_MODULES:
        importlib.import_module(f"flash.providers.{pkg_name}.{mod}")
    pkg = importlib.import_module(f"flash.providers.{pkg_name}")
    assert hasattr(pkg, "PROVIDER")


def test_method_signatures_match_across_providers():
    """The interface methods take the same parameters on both providers (swappable)."""
    import inspect

    from flash.providers import get_provider

    rp, la = get_provider("runpod"), get_provider("lambda")
    for meth in PROVIDER_METHODS:
        rs = list(inspect.signature(getattr(rp, meth)).parameters)
        ls = list(inspect.signature(getattr(la, meth)).parameters)
        assert rs == ls, f"{meth} param mismatch: runpod={rs} lambda={ls}"


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


def test_static_pricing():
    pricing = importlib.import_module("flash.providers.runpod.pricing")
    assert pricing.hourly_rate("RTX 5090") == pytest.approx(0.99)


def test_allocator_picks_runpod_candidate(monkeypatch):
    from flash.providers.allocator import allocate

    a = allocate("Qwen/Qwen3.5-0.8B", "sft")
    assert a.provider == "runpod"
    assert a.gpu == "RTX 3090"


def test_allocation_summary_formats_runpod_choice(monkeypatch):
    from flash.providers.allocator import allocation_summary
    from flash.providers.base import Allocation, Candidate

    cand = Candidate("runpod", "RTX 3090", 0.46, 24)
    alloc = Allocation(
        provider="runpod",
        gpu="RTX 3090",
        hourly_usd=0.46,
        min_vram_gb=24,
        candidates=(cand,),
    )
    out = allocation_summary(alloc)
    assert "RTX 3090 on runpod" in out


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
