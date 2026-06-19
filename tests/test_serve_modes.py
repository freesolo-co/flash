"""Serving modes + the freesolo-serving client surface (CPU-only, no network)."""

from __future__ import annotations

import pytest

from flash.serve.deploy import (
    MODES,
    Deployment,
    deploy_adapter,
    serve_endpoint_name,
    serving_base_url,
    undeploy_adapter,
)


def test_serving_base_url_default_and_override(monkeypatch):
    from flash.serve.deploy import DEFAULT_FREESOLO_SERVING_URL

    monkeypatch.delenv("FREESOLO_SERVING_URL", raising=False)
    assert serving_base_url() == DEFAULT_FREESOLO_SERVING_URL
    # Override wins; a trailing slash is stripped so f"{base}/adapters" is well-formed.
    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example/")
    assert serving_base_url() == "https://serve.example"


def test_deploy_dry_run_modes():
    dev = deploy_adapter("r1", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0", "RTX 4090", dry_run=True)
    assert dev.mode == "dev"
    # freesolo serving scales to zero per base model — no flash-side idle billing, in any mode.
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
    assert warm.est_idle_cost_usd_per_day == 0.0


def test_deploy_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        deploy_adapter("r1", "m", "repo", "p", "RTX 4090", mode="warmish", dry_run=True)
    assert set(MODES) == {"dev", "always-on"}


def test_undeploy_calls_freesolo_delete(monkeypatch):
    """undeploy issues a single DELETE to the serving app keyed by run_id and returns the
    removed id; an unrelated run is unaffected (the serving app keys by adapterId)."""
    import flash.serve.deploy as deploy_mod

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    deleted_urls = []

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

    def fake_delete(url, headers=None, timeout=None):
        deleted_urls.append(url)
        return _Resp()

    monkeypatch.setattr(deploy_mod.httpx, "delete", fake_delete)
    out = undeploy_adapter("flash-1-abc", gpu_name="RTX 5090")
    assert out == ["flash-1-abc"]
    assert deleted_urls == ["https://serve.example/adapters/flash-1-abc"]


def test_undeploy_404_is_clean(monkeypatch):
    """A 404 (adapter already deregistered) is a clean no-op, returning []."""
    import flash.serve.deploy as deploy_mod

    class _Resp:
        status_code = 404

        def raise_for_status(self):  # pragma: no cover - must not be called on 404
            raise AssertionError("404 must not raise")

    monkeypatch.setattr(deploy_mod.httpx, "delete", lambda *a, **k: _Resp())
    assert undeploy_adapter("flash-1-gone", gpu_name="RTX 5090") == []


def test_serve_endpoint_name_is_cosmetic_label():
    # serve_endpoint_name is retained for back-compat (importers/CLI); the freesolo app
    # serves all adapters on one endpoint, so this is just a stable per-run label now.
    a = serve_endpoint_name("RTX 5090", "flash-123-abcd1234")
    assert a.startswith("flash-serve-5090-")


def test_deployment_roundtrip_dict():
    d = Deployment(
        run_id="r",
        model="m",
        adapter_hf_prefix="p",
        gpu="RTX 4090",
        openai_model="r",
        endpoint_name="https://serve.example",
        mode="dev",
    )
    assert d.to_dict()["mode"] == "dev"
    assert d.to_dict()["est_idle_cost_usd_per_day"] == 0.0
