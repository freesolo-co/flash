"""Provider registry + ``base.Provider`` interface coverage (CPU-only, offline).

RunPod (always on) and the instance-based complements — Lambda (opt-in via LAMBDA_API_KEY) and
Vast (opt-in via VAST_API_KEY) — all implement the SAME ``base.Provider`` interface, so the
orchestrator/allocator treat them interchangeably.

Every provider carries the universal per-substrate client surface in ``CLIENT_MODULES`` plus the
provider-specific jobs module returned by ``_jobs_module``. Modules that only SOME providers need
are divergence-driven, not part of the contract: ``gpus`` exists only
where a provider translates friendly names to substrate ids (RunPod validated classes, Lambda's
``instance_type_for``) — instance providers otherwise draw ``gpu_classes`` from the shared
``base.gpu_classes_for``; ``train`` exists only for RunPod's serverless endpoint-deploy machinery
(instance providers submit through ``jobs`` directly)."""

from __future__ import annotations

import importlib

import pytest

_RUNPOD_FINGERPRINT = "rpk-" + "0" * 64

# The universal per-provider surface: concerns every substrate must implement itself (no shared
# default fits). ``gpus``/``train`` are intentionally NOT here — they're per-provider by necessity.
# the per-substrate client surface now lives under each provider's `client` subpackage;
# `jobs` stays at the provider root (runpod groups its heavier job modules in `execution`).
CLIENT_MODULES = ("api", "auth", "pricing", "preflight")
PROVIDER_METHODS = (
    "is_configured",
    "preflight",
    "gpu_classes",
    "hourly_rate",
    "submit_attempt",
    "poll_attempt",
    "cancel",
    "destroy",
    "gc",
    "sweep_orphans",
)


_PKG = {"runpod": "runpod", "lambda": "lambda_", "vast": "vast"}


def test_registry_lists_all_providers():
    from flash.providers.core.registry import PROVIDER_NAMES, get_provider

    assert PROVIDER_NAMES == ("runpod", "lambda", "vast")
    assert get_provider("RunPod ").name == "runpod"
    assert get_provider(" Lambda ").name == "lambda"  # case/space-insensitive
    assert get_provider(" Vast ").name == "vast"
    with pytest.raises(KeyError):
        get_provider("hyperstack")  # removed; not registered


@pytest.mark.parametrize("provider", ["runpod", "lambda", "vast"])
def test_provider_implements_the_interface(provider):
    from flash.providers.core.base import Provider
    from flash.providers.core.registry import get_provider

    prov = get_provider(provider)
    assert isinstance(prov, Provider)
    for meth in PROVIDER_METHODS:
        assert callable(getattr(prov, meth)), f"{provider} missing {meth}"


def _jobs_module(pkg: str) -> str:
    """runpod groups job execution under `execution`; the others keep `jobs` at the root."""
    return (
        f"flash.providers.{pkg}.execution.jobs"
        if pkg == "runpod"
        else f"flash.providers.{pkg}.jobs"
    )


@pytest.mark.parametrize("provider", ["runpod", "lambda", "vast"])
def test_module_layout(provider):
    """Every provider subpackage exposes the SAME public module set (the symmetry contract)."""
    for mod in CLIENT_MODULES:
        importlib.import_module(f"flash.providers.{_PKG[provider]}.client.{mod}")
    importlib.import_module(_jobs_module(_PKG[provider]))
    provider_module = importlib.import_module(
        f"flash.providers.{_PKG[provider]}.execution.provider"
    )
    assert hasattr(provider_module, "PROVIDER")


@pytest.mark.parametrize("provider", ["lambda", "vast"])
def test_method_signatures_match_runpod(provider):
    """The interface methods take the same parameters on every provider (swappable)."""
    import inspect

    from flash.providers.core.registry import get_provider

    rp = get_provider("runpod")
    other = get_provider(provider)
    for meth in PROVIDER_METHODS:
        rs = list(inspect.signature(getattr(rp, meth)).parameters)
        os_ = list(inspect.signature(getattr(other, meth)).parameters)
        assert rs == os_, f"{meth} param mismatch: runpod={rs} {provider}={os_}"


def test_setup_vs_training_gate_contract():
    """define the canonical setup-vs-training classifier contract used by provider polling."""
    from flash.providers._lifecycle.instances.poll import (
        SETUP_HEARTBEAT_STAGES,
        STEP_GATED_STAGES,
        is_training_heartbeat,
    )

    # The slow cold-start pings must count as setup (kept under the wide setup grace).
    for stage in ("model_prefetching", "model_prefetched", "sft_initializing", "rl_initializing"):
        assert stage in SETUP_HEARTBEAT_STAGES, f"{stage} must be treated as setup, not training"
        assert is_training_heartbeat(stage, 9) is False
    # Per-step training heartbeats are step-gated: NOT setup, but only flip the window at step >= 1.
    assert sorted(STEP_GATED_STAGES) == ["opd_step", "rl_step", "sft_step"]
    assert is_training_heartbeat("rl_step", 0) is False  # cold first step keeps setup grace
    assert is_training_heartbeat("rl_step", 1) is True
    # opd_step is gated the same way: its mid-step ping reports opt_steps (0 during the first
    # optimizer step), which must keep setup grace, not trip the tight training window.
    assert is_training_heartbeat("opd_step", 0) is False
    assert is_training_heartbeat("opd_step", 1) is True
    # Post-training stages flip to the tight window even without a step field.
    assert is_training_heartbeat("sft_trained", None) is True


def test_format_heartbeat_includes_rl_step_metrics():
    from flash.providers._lifecycle.instances.poll import _format_heartbeat

    rendered = _format_heartbeat(
        {
            "stage": "rl_step",
            "step": 4,
            "reward": 0.75,
            "reward_std": 0.12,
            "grad_norm": 1.5,
            "kl": 0.03,
            "entropy": 0.82,
            "frac_reward_zero_std": 0.25,
            "mean_completion_tokens": 48.5,
            "truncation_rate": 0.125,
            "max_completion_tokens": 256,
        }
    )

    for field in (
        "step=4",
        "reward=0.750",
        "reward_std=0.120",
        "grad_norm=1.500",
        "kl=0.030",
        "entropy=0.820",
        "frac_reward_zero_std=0.250",
        "mean_completion_tokens=48.500",
        "truncation_rate=0.125",
        "max_completion_tokens=256",
    ):
        assert field in rendered


def test_runpod_provider_implements_the_interface():
    from flash.providers.core.base import Provider
    from flash.providers.core.registry import get_provider

    prov = get_provider("runpod")
    assert isinstance(prov, Provider)
    for meth in PROVIDER_METHODS:
        assert callable(getattr(prov, meth)), f"runpod missing {meth}"


def test_runpod_module_layout():
    for mod in CLIENT_MODULES:
        importlib.import_module(f"flash.providers.runpod.client.{mod}")
    importlib.import_module(_jobs_module("runpod"))
    provider_module = importlib.import_module("flash.providers.runpod.execution.provider")
    assert hasattr(provider_module, "PROVIDER")


def test_gpu_classes_match_runpod_rows():
    from flash.providers.core.base import GPU_INFO
    from flash.providers.core.registry import get_provider

    rp = {g.name for g in get_provider("runpod").gpu_classes()}
    assert rp == {g.name for g in GPU_INFO.values() if g.enum_member}


def test_sweep_orphans_is_part_of_the_protocol():
    from flash.providers.core.base import Provider
    from flash.providers.core.registry import get_provider

    assert hasattr(get_provider("runpod"), "sweep_orphans")
    assert get_provider("runpod").sweep_orphans(active_labels={"flash-x"}) == []
    assert "sweep_orphans" in dir(Provider)


def test_run_instances_remaining_is_optional_not_required_by_protocol():
    # Copilot: run_instances_remaining is an OPTIONAL capability (Vast enumerates billable instances by
    # run label; RunPod serverless self-reaps). It must NOT be on the @runtime_checkable Provider
    # Protocol, or isinstance(runpod_provider, Provider) would go False and break the symmetry checks
    # above. Vast implements it; RunPod does not; both still satisfy Provider. Detected via getattr.
    from flash.providers.core.base import Provider
    from flash.providers.core.registry import get_provider

    assert "run_instances_remaining" not in dir(Provider)  # not a required Protocol member
    assert hasattr(get_provider("vast"), "run_instances_remaining")  # Vast provides the capability
    assert not hasattr(get_provider("runpod"), "run_instances_remaining")  # RunPod opts out
    assert isinstance(get_provider("runpod"), Provider)  # still a Provider despite opting out


def test_static_pricing():
    pricing = importlib.import_module("flash.providers.runpod.client.pricing")
    assert pricing.hourly_rate("RTX 5090") == pytest.approx(0.99)


def _stub_candidates(monkeypatch, *, runpod=(), lambda_=(), vast=()):
    """Pin allocate()'s three provider candidate lists so ranking can be tested in isolation."""
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate
    from flash.providers.core.registry import get_provider

    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod", "lambda", "vast"))
    # allocate() sources candidates via provider.live_candidates(need, constraints); pin each provider's.
    monkeypatch.setattr(
        get_provider("runpod"),
        "live_candidates",
        lambda need, constraints: [Candidate(*c) for c in runpod],
    )
    monkeypatch.setattr(
        get_provider("lambda"),
        "live_candidates",
        lambda need, constraints: [Candidate(*c) for c in lambda_],
    )
    monkeypatch.setattr(
        get_provider("vast"),
        "live_candidates",
        lambda need, constraints: [Candidate(*c) for c in vast],
    )


def test_vast_competes_purely_on_price(monkeypatch):
    """Vast is ranked by cost alongside runpod/lambda: the cheapest offer wins, whoever owns it."""
    from flash.providers.core.allocator import allocate

    # Vast is the cheapest fitting offer -> Vast is selected.
    _stub_candidates(
        monkeypatch,
        runpod=[("runpod", "RTX 4090", 0.69, 48)],
        lambda_=[("lambda", "A10", 0.60, 48)],
        vast=[("vast", "RTX 4090", 0.47, 48)],
    )
    a = allocate("Qwen/Qwen3.5-9B", "sft")
    assert a.provider == "vast"
    assert a.hourly_usd == pytest.approx(0.47)

    # When Vast is the pricier offer for the SAME class it does NOT win -> no structural advantage
    # either way. RunPod's 4090 beats Lambda's nominally-cheaper A10 here because ranking is on
    # what the JOB costs: the A10 sustains ~125 TFLOPS against the 4090's ~165, so the run takes
    # long enough on it to more than erase the $0.09/hr saving.
    _stub_candidates(
        monkeypatch,
        runpod=[("runpod", "RTX 4090", 0.69, 48)],
        lambda_=[("lambda", "A10", 0.60, 48)],
        vast=[("vast", "RTX 4090", 0.80, 48)],
    )
    b = allocate("Qwen/Qwen3.5-9B", "sft")
    assert b.provider == "runpod"
    assert b.hourly_usd == pytest.approx(0.69)


def test_gpu_name_breaks_price_vram_tie_before_provider_order(monkeypatch):
    """gpu name breaks a price-and-VRAM tie before stable provider order; full-key ties retain order."""
    from flash.providers.core.allocator import allocate

    # identical price and VRAM; Vast's class name sorts before RunPod's, so Vast wins the tie.
    _stub_candidates(
        monkeypatch,
        runpod=[("runpod", "RTX 4090", 0.50, 48)],
        vast=[("vast", "A100", 0.50, 48)],
    )
    assert allocate("Qwen/Qwen3.5-9B", "sft").provider == "vast"  # "A100" < "RTX 4090"

    # flip which provider owns the name that sorts first, and the other provider wins. when all key
    # fields match, Python's stable sort preserves registry and provider-local order instead.
    _stub_candidates(
        monkeypatch,
        runpod=[("runpod", "A100", 0.50, 48)],
        vast=[("vast", "RTX 4090", 0.50, 48)],
    )
    assert allocate("Qwen/Qwen3.5-9B", "sft").provider == "runpod"  # "A100" < "RTX 4090"


def test_allocation_summary_formats_runpod_choice():
    from flash.providers.core.allocator import allocation_summary
    from flash.providers.core.base import Allocation, Candidate

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
    from flash.providers.core.base import JobHandle

    h = JobHandle(provider="runpod", data={"endpoint_id": "ep", "job_id": "j"})
    d = h.to_dict()
    assert d["provider"] == "runpod"
    back = JobHandle.from_dict(d)
    assert back.provider == "runpod"
    assert back.data == {"endpoint_id": "ep", "job_id": "j"}
    with pytest.raises(ValueError, match="provider identity is missing or invalid"):
        JobHandle.from_dict({"endpoint_id": "ep", "job_id": "j"})


def test_provider_cancel_destroy_dispatch(monkeypatch):
    from flash.providers.core.base import JobHandle
    from flash.providers.core.registry import get_provider
    from flash.providers.runpod.client import api as rp_api

    cancelled, deleted = [], []
    monkeypatch.setattr(
        rp_api,
        "cancel_job",
        lambda e, j, **_kw: cancelled.append((e, j)) or {"id": j, "status": "CANCELLED"},
    )
    monkeypatch.setattr(
        rp_api,
        "delete_endpoint_for_fingerprint",
        lambda e, _fingerprint: deleted.append(e) or True,
    )
    handle = JobHandle(
        "runpod",
        {
            "endpoint_id": "ep",
            "endpoint_name": "n",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "j",
            "attempt": 0,
            "started_ts": 1.0,
        },
    )
    get_provider("runpod").cancel(handle)
    get_provider("runpod").destroy(handle)
    assert cancelled == [("ep", "j")]
    assert deleted == ["ep"]


def test_runpod_destroy_accepts_exact_owner_404_confirmation(monkeypatch):
    from flash.providers.core.base import JobHandle
    from flash.providers.core.registry import get_provider
    from flash.providers.runpod.client import api as rp_api

    lookups = []
    monkeypatch.setattr(
        rp_api,
        "delete_endpoint_for_fingerprint",
        lambda endpoint_id, _fingerprint: False,
    )
    monkeypatch.setattr(
        rp_api,
        "endpoint_absent_for_fingerprint",
        lambda endpoint_id, fingerprint: lookups.append((endpoint_id, fingerprint)) or True,
    )
    handle = JobHandle(
        "runpod",
        {
            "endpoint_id": "ep-unconfirmed",
            "endpoint_name": "n",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "j",
            "attempt": 0,
            "started_ts": 1.0,
        },
    )

    get_provider("runpod").destroy(handle)

    assert lookups == [("ep-unconfirmed", _RUNPOD_FINGERPRINT)]


def test_runpod_destroy_rejects_non_authoritative_absence_result(monkeypatch):
    from flash.providers.core.base import JobHandle
    from flash.providers.core.registry import get_provider
    from flash.providers.runpod.client import api as rp_api

    monkeypatch.setattr(
        rp_api,
        "delete_endpoint_for_fingerprint",
        lambda endpoint_id, _fingerprint: False,
    )
    monkeypatch.setattr(
        rp_api,
        "endpoint_absent_for_fingerprint",
        lambda endpoint_id, fingerprint: False,
    )
    handle = JobHandle(
        "runpod",
        {
            "endpoint_id": "ep-unconfirmed",
            "endpoint_name": "n",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "j",
            "attempt": 0,
            "started_ts": 1.0,
        },
    )

    with pytest.raises(rp_api.RunpodApiError, match="unconfirmed"):
        get_provider("runpod").destroy(handle)


@pytest.mark.parametrize(
    ("provider", "env_var"),
    [("runpod", "RUNPOD_API_KEY"), ("lambda", "LAMBDA_API_KEY"), ("vast", "VAST_API_KEY")],
)
def test_the_key_sent_on_the_wire_is_the_key_preflight_accepted(monkeypatch, provider, env_var):
    """A padded provider key must not pass preflight and then go out unstripped.

    ``load_provider_key`` strips, so ``is_configured()`` and the startup preflight accept a key
    carrying a stray newline from an env file. The outbound REST client used to re-read
    ``os.environ`` raw, so the request went out as ``Authorization: Bearer <key>\\n`` -- a
    Lambda-only or Vast-only plane reported a clean preflight while every capacity lookup and
    allocation was rejected, with nothing pointing at the credential.

    Asserted on the ``Authorization`` HEADER, not on ``api_key()``: the header is what the
    provider sees, and it is the only place the two halves can be observed disagreeing.
    """
    import urllib.request

    from flash.providers._lifecycle.net.auth import load_provider_key
    from flash.providers._lifecycle.net.http import RestClient

    padded = "\n  key-with-padding  \n"
    monkeypatch.setenv(env_var, padded)
    assert load_provider_key(env_var) == "key-with-padding"  # what preflight judges

    client = RestClient(env_var=env_var, error_cls=RuntimeError, base_url="https://api.invalid")
    seen: dict[str, str] = {}

    class _Resp:
        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _capture(req, timeout=None):
        seen.update(req.headers)
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _capture)
    client.request("/whatever")

    auth = seen.get("Authorization") or seen.get("Authorization".title())
    assert auth == "Bearer key-with-padding", f"padded credential reached the wire: {auth!r}"
