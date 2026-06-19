"""Serving modes: deploy kwargs, per-run endpoints, undeploy (CPU-only)."""

from __future__ import annotations

import pytest

from flash.serve.deploy import (
    MODES,
    Deployment,
    deploy_adapter,
    resolve_serve_deps,
    serve_endpoint_name,
    undeploy_adapter,
)


def test_serve_endpoint_name_is_per_run():
    a = serve_endpoint_name("RTX 5090", "flash-123-abcd1234")
    b = serve_endpoint_name("RTX 5090", "flash-456-ffff0000")
    assert a != b
    assert a.startswith("flash-serve-5090-")


def test_deploy_dry_run_modes():
    dev = deploy_adapter("r1", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0", "RTX 4090", dry_run=True)
    assert dev.mode == "dev"
    assert dev.est_idle_cost_usd_per_day == 0.0
    warm = deploy_adapter(
        "r1",
        "Qwen/Qwen3.5-0.8B",
        "repo",
        "rl/r1/seed0",
        "RTX 4090",
        mode="always-on",
        dry_run=True,
    )
    assert warm.mode == "always-on"
    from flash.providers.runpod.pricing import hourly_rate

    assert warm.est_idle_cost_usd_per_day == pytest.approx(hourly_rate("RTX 4090") * 24, rel=0.01)


def test_deploy_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        deploy_adapter("r1", "m", "repo", "p", "RTX 4090", mode="warmish", dry_run=True)
    assert set(MODES) == {"dev", "always-on"}


def test_resolve_serve_deps(monkeypatch):
    monkeypatch.delenv("AUTOSLM_SERVE_DEPS", raising=False)
    assert any("vllm==0.19" in d for d in resolve_serve_deps())  # the single pinned stack
    monkeypatch.setenv("AUTOSLM_SERVE_DEPS", "vllm==9.9")  # explicit override wins
    assert resolve_serve_deps() == ["vllm==9.9"]


def test_resolve_serve_deps_preserves_comma_ranges(monkeypatch):
    # A PEP 440 range like transformers>=5.6,<5.11 must NOT be comma-split into two
    # pip args. Whitespace-separated string and JSON list both keep it intact.
    monkeypatch.setenv("AUTOSLM_SERVE_DEPS", "torch==2.10.0 transformers>=5.6,<5.11 vllm==0.19.1")
    assert resolve_serve_deps() == ["torch==2.10.0", "transformers>=5.6,<5.11", "vllm==0.19.1"]
    monkeypatch.setenv("AUTOSLM_SERVE_DEPS", '["transformers>=5.6,<5.11", "vllm==0.19.1"]')
    assert resolve_serve_deps() == ["transformers>=5.6,<5.11", "vllm==0.19.1"]


def test_undeploy_uses_rest(monkeypatch):
    from flash.providers.runpod import api as runpod
    from flash.serve import deploy as deploy_mod

    monkeypatch.setattr(
        runpod,
        "find_endpoints_by_name",
        lambda name: [{"id": "ep1", "name": name}],
    )
    deleted_ids = []
    monkeypatch.setattr(runpod, "delete_endpoint", lambda eid: deleted_ids.append(eid) or True)
    # A stale cache entry for this run+gpu must be dropped on undeploy so a later
    # redeploy builds a fresh endpoint instead of reusing the deleted one.
    name = deploy_mod.serve_endpoint_name("RTX 5090", "flash-1-abc")
    deploy_mod._ENDPOINT_CACHE[f"{name}:dev:300"] = object()
    out = undeploy_adapter("flash-1-abc", gpu_name="RTX 5090")
    assert deleted_ids == ["ep1"]
    assert len(out) == 1
    assert not [k for k in deploy_mod._ENDPOINT_CACHE if k.startswith(f"{name}:")]


def test_undeploy_skips_substring_collision(monkeypatch):
    # find_endpoints_by_name is a substring filter: another run's endpoint whose
    # name merely CONTAINS the target as a substring must NOT be deleted, and must
    # not appear in the returned `deleted` list.
    from flash.providers.runpod import api as runpod
    from flash.serve import deploy as deploy_mod

    name = deploy_mod.serve_endpoint_name("RTX 5090", "flash-1-abc")
    collision = name + "-other"  # a different endpoint that contains `name`
    monkeypatch.setattr(
        runpod,
        "find_endpoints_by_name",
        lambda n: [{"id": "exact", "name": name}, {"id": "collide", "name": collision}],
    )
    deleted_ids = []
    monkeypatch.setattr(runpod, "delete_endpoint", lambda eid: deleted_ids.append(eid) or True)
    out = undeploy_adapter("flash-1-abc", gpu_name="RTX 5090")
    assert deleted_ids == ["exact"], "only the exact-name endpoint may be deleted"
    assert out == [name]


def test_deployment_roundtrip_dict():
    d = Deployment(
        run_id="r",
        model="m",
        adapter_hf_prefix="p",
        gpu="RTX 4090",
        openai_model="flash-r",
        endpoint_name="e",
        mode="dev",
    )
    assert d.to_dict()["mode"] == "dev"
