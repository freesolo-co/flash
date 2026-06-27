"""Serving client contract and token pricing."""

from __future__ import annotations

import pytest

from flash.serve.deploy import (
    Deployment,
    deploy_adapter,
    serving_base_url,
    undeploy_adapter,
)


def test_serving_base_url_default_and_override(monkeypatch):
    from flash.serve.deploy import DEFAULT_FREESOLO_SERVING_URL

    monkeypatch.delenv("FREESOLO_SERVING_URL", raising=False)
    assert serving_base_url() == DEFAULT_FREESOLO_SERVING_URL
    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example/")
    assert serving_base_url() == "https://serve.example"


def test_deploy_dry_run_has_no_user_facing_mode():
    dep = deploy_adapter("r1", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0", "RTX 4090", dry_run=True)
    data = dep.to_dict()
    assert data["state"] == "dry_run"
    assert data["gpu"] == "RTX 4090"
    assert "mode" not in data
    assert "idle_timeout_s" not in data
    assert "est_idle_cost_usd_per_day" not in data


def test_real_deploy_translates_serving_5xx_to_serving_error(monkeypatch):
    import httpx

    import flash.serve.deploy as deploy_mod
    from flash.serve.deploy import ServingError

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    req = httpx.Request("POST", "https://serve.example/adapters")
    resp = httpx.Response(500, text="no base-model engines loaded", request=req)
    monkeypatch.setattr(deploy_mod.httpx, "post", lambda *a, **k: resp)

    with pytest.raises(ServingError) as ei:
        deploy_adapter("flash-1-abc", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0", "RTX 4090")
    assert ei.value.status_code == 500
    assert "500" in str(ei.value)
    assert "no base-model engines loaded" in str(ei.value)
    assert "operator must check" in str(ei.value)


def test_real_deploy_4xx_hint_points_at_client_not_serving_outage(monkeypatch):
    import httpx

    import flash.serve.deploy as deploy_mod
    from flash.serve.deploy import ServingError

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    req = httpx.Request("POST", "https://serve.example/adapters")
    resp = httpx.Response(401, text="invalid internal key", request=req)
    monkeypatch.setattr(deploy_mod.httpx, "post", lambda *a, **k: resp)

    with pytest.raises(ServingError) as ei:
        deploy_adapter("flash-1-abc", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0", "RTX 4090")
    msg = str(ei.value)
    assert ei.value.status_code == 401
    assert "401" in msg
    assert "FREESOLO_INTERNAL_KEY" in msg
    assert "no engine" not in msg
    assert "operator must check" not in msg


def test_real_deploy_translates_unreachable_serving_to_serving_error(monkeypatch):
    import httpx

    import flash.serve.deploy as deploy_mod
    from flash.serve.deploy import ServingError

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")

    def fake_post(url, *a, **k):
        raise httpx.ConnectError("connection refused", request=httpx.Request("POST", url))

    monkeypatch.setattr(deploy_mod.httpx, "post", fake_post)
    with pytest.raises(ServingError) as ei:
        deploy_adapter("flash-1-abc", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0", "RTX 4090")
    assert ei.value.status_code is None
    assert "could not reach" in str(ei.value)


def test_undeploy_calls_freesolo_delete(monkeypatch):
    import flash.serve.deploy as deploy_mod

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    deleted_urls = []

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

    def fake_delete(url, headers=None, timeout=None, follow_redirects=None):
        deleted_urls.append(url)
        return _Resp()

    monkeypatch.setattr(deploy_mod.httpx, "delete", fake_delete)
    out = undeploy_adapter("flash-1-abc")
    assert out == ["flash-1-abc"]
    assert deleted_urls == ["https://serve.example/adapters/flash-1-abc"]


def test_undeploy_404_is_clean(monkeypatch):
    import flash.serve.deploy as deploy_mod

    class _Resp:
        status_code = 404

        def raise_for_status(self):  # pragma: no cover
            raise AssertionError("404 must not raise")

    monkeypatch.setattr(deploy_mod.httpx, "delete", lambda *a, **k: _Resp())
    assert undeploy_adapter("flash-1-gone") == []


def test_deployment_roundtrip_dict():
    d = Deployment(
        run_id="r",
        model="m",
        adapter_hf_prefix="p",
        gpu="RTX 4090",
        openai_model="r",
        endpoint_name="https://serve.example",
    )
    data = d.to_dict()
    assert data["run_id"] == "r"
    assert "mode" not in data


def test_serving_prices_cover_catalog_and_apply_markup():
    from flash.catalog import MODELS
    from flash.serve.pricing import SERVING_MARKUP, SERVING_PRICES, serving_price_rows

    assert set(SERVING_PRICES) == set(MODELS)
    assert pytest.approx(1.20) == SERVING_MARKUP
    rows = serving_price_rows()
    assert len(rows) == len(MODELS)
    for row in rows:
        assert row["billed_input_usd_per_mtok"] == pytest.approx(
            row["base_input_usd_per_mtok"] * SERVING_MARKUP
        )
        assert row["billed_output_usd_per_mtok"] == pytest.approx(
            row["base_output_usd_per_mtok"] * SERVING_MARKUP
        )
