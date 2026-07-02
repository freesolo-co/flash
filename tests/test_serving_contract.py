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
    monkeypatch.setattr(deploy_mod, "adapter_artifact_lora_rank", lambda *a, **k: 32)
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
    monkeypatch.setattr(deploy_mod, "adapter_artifact_lora_rank", lambda *a, **k: 32)
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

    monkeypatch.setattr(deploy_mod, "adapter_artifact_lora_rank", lambda *a, **k: 32)
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


def test_serving_prices_cover_catalog():
    from flash.catalog import MODELS
    from flash.serve.pricing import SERVING_MARKUP, SERVING_PRICES, serving_price_rows

    assert set(SERVING_PRICES) == set(MODELS)
    assert pytest.approx(1.20) == SERVING_MARKUP
    rows = serving_price_rows()
    assert len(rows) == len(MODELS)
    for row in rows:
        assert row["typical_input_usd_per_mtok"] > 0
        assert row["typical_output_usd_per_mtok"] > 0
        assert row["typical_cached_input_usd_per_mtok"] > 0
        assert row["billed_input_usd_per_mtok"] > 0
        assert row["billed_output_usd_per_mtok"] > 0
        assert row["billed_cached_input_usd_per_mtok"] > 0
        assert row["billed_cached_input_usd_per_mtok"] < row["billed_input_usd_per_mtok"]
        assert row["billed_input_usd_per_mtok"] == pytest.approx(
            row["typical_input_usd_per_mtok"] * SERVING_MARKUP
        )
        assert row["billed_output_usd_per_mtok"] == pytest.approx(
            row["typical_output_usd_per_mtok"] * SERVING_MARKUP
        )
        assert row["billed_cached_input_usd_per_mtok"] == pytest.approx(
            row["typical_cached_input_usd_per_mtok"] * SERVING_MARKUP
        )


def test_serving_prices_pin_public_rates_plus_markup():
    from flash.serve.pricing import SERVING_MARKUP, SERVING_PRICES

    typical = {
        "openbmb/MiniCPM5-1B": (0.01, 0.05, 0.002),
        "Qwen/Qwen3.5-0.8B": (0.01, 0.05, 0.002),
        "Qwen/Qwen3.5-2B": (0.02, 0.10, 0.004),
        "Qwen/Qwen3.5-4B": (0.03, 0.15, 0.006),
        "Qwen/Qwen3.5-9B": (0.10, 0.15, 0.020),
        "Qwen/Qwen3.6-35B-A3B": (0.15, 1.00, 0.050),
    }
    for model_id, (input_rate, output_rate, cached_rate) in typical.items():
        price = SERVING_PRICES[model_id]
        assert price.typical_input_usd_per_mtok == pytest.approx(input_rate)
        assert price.typical_output_usd_per_mtok == pytest.approx(output_rate)
        assert price.typical_cached_input_usd_per_mtok == pytest.approx(cached_rate)
        assert price.billed_input_usd_per_mtok == pytest.approx(input_rate * SERVING_MARKUP)
        assert price.billed_output_usd_per_mtok == pytest.approx(output_rate * SERVING_MARKUP)
        assert price.billed_cached_input_usd_per_mtok == pytest.approx(cached_rate * SERVING_MARKUP)


def test_resolve_deploy_step_rejects_malformed_step_as_400():
    """A malformed ``step`` must raise HTTPException(400), never a 500. Regression for ``"--5"``:
    ``str.lstrip("-").isdigit()`` accepted it, then ``int("--5")`` raised an uncaught ValueError.
    The 400 path raises before any checkpoint lookup, so the spec/app args are unused here."""
    pytest.importorskip("fastapi")

    from fastapi import HTTPException

    from flash.server.routes.serving import _resolve_deploy_step

    for bad in ("--5", "40.9", "+5", "5-", "abc", "-", "", "   ", "0x5"):
        with pytest.raises(HTTPException) as ei:
            _resolve_deploy_step("flash-7-abcd", object(), bad)
        assert ei.value.status_code == 400, bad


def _registry_resp(records):
    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True, "adapters": records}

    return _Resp()


def test_deploy_reads_registry_back_before_ready(monkeypatch):
    """POST /adapters returning 2xx is not enough: ready must be registry-backed."""
    import flash.serve.deploy as deploy_mod

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setattr(deploy_mod, "adapter_artifact_lora_rank", lambda *a, **k: 32)

    class _PostResp:
        status_code = 200

        def raise_for_status(self):
            return None

    gets: list[str] = []

    def fake_get(url, **k):
        gets.append(url)
        return _registry_resp([{"adapter_id": "flash-1-abc", "subfolder": "rl/r1/seed0/adapter"}])

    monkeypatch.setattr(deploy_mod.httpx, "post", lambda *a, **k: _PostResp())
    monkeypatch.setattr(deploy_mod.httpx, "get", fake_get)

    dep = deploy_adapter("flash-1-abc", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0", "RTX 4090")
    assert dep.state == "ready"
    assert gets == ["https://serve.example/adapters", "https://serve.example/adapters"]


def test_deploy_registry_readback_falls_back_to_camel_adapter_id(monkeypatch):
    import flash.serve.deploy as deploy_mod

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setattr(deploy_mod, "adapter_artifact_lora_rank", lambda *a, **k: 32)

    class _PostResp:
        status_code = 200

        def raise_for_status(self):
            return None

    monkeypatch.setattr(deploy_mod.httpx, "post", lambda *a, **k: _PostResp())
    monkeypatch.setattr(
        deploy_mod.httpx,
        "get",
        lambda *a, **k: _registry_resp(
            [{"adapter_id": None, "adapterId": "flash-1-abc", "subfolder": "rl/r1/seed0/adapter"}]
        ),
    )

    dep = deploy_adapter("flash-1-abc", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0", "RTX 4090")
    assert dep.state == "ready"


def test_deploy_fails_when_adapter_never_appears_in_registry(monkeypatch):
    import flash.serve.deploy as deploy_mod
    from flash.serve.deploy import ServingError

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setattr(deploy_mod, "adapter_artifact_lora_rank", lambda *a, **k: 32)

    class _PostResp:
        status_code = 200

        def raise_for_status(self):
            return None

    monkeypatch.setattr(deploy_mod.httpx, "post", lambda *a, **k: _PostResp())
    monkeypatch.setattr(deploy_mod.httpx, "get", lambda *a, **k: _registry_resp([]))

    with pytest.raises(ServingError) as ei:
        deploy_adapter("flash-1-abc", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0", "RTX 4090")
    msg = str(ei.value)
    assert "never appeared" in msg
    assert "Unknown adapter id" in msg


def test_deploy_fails_when_registry_keeps_prior_checkpoint(monkeypatch):
    """A checkpoint swap that leaves the OLD subfolder registered must fail loudly."""
    import flash.serve.deploy as deploy_mod
    from flash.serve.deploy import ServingError

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setattr(deploy_mod, "adapter_artifact_lora_rank", lambda *a, **k: 32)

    class _PostResp:
        status_code = 200

        def raise_for_status(self):
            return None

    monkeypatch.setattr(deploy_mod.httpx, "post", lambda *a, **k: _PostResp())
    monkeypatch.setattr(
        deploy_mod.httpx,
        "get",
        lambda *a, **k: _registry_resp(
            [{"adapter_id": "flash-1-abc", "subfolder": "rl/r1/seed0/checkpoints/step_100/adapter"}]
        ),
    )

    with pytest.raises(ServingError) as ei:
        deploy_adapter("flash-1-abc", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0", "RTX 4090")
    assert "previously deployed checkpoint" in str(ei.value)


def test_deploy_5xx_recovers_when_registry_shows_requested_checkpoint(monkeypatch):
    """An ambiguous POST failure (timeout/5xx) is resolved by reading the registry back."""
    import httpx

    import flash.serve.deploy as deploy_mod

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setattr(deploy_mod, "adapter_artifact_lora_rank", lambda *a, **k: 32)
    req = httpx.Request("POST", "https://serve.example/adapters")
    resp = httpx.Response(502, text="bad gateway", request=req)
    monkeypatch.setattr(deploy_mod.httpx, "post", lambda *a, **k: resp)
    monkeypatch.setattr(
        deploy_mod.httpx,
        "get",
        lambda *a, **k: _registry_resp(
            [{"adapter_id": "flash-1-abc", "subfolder": "rl/r1/seed0/adapter"}]
        ),
    )

    dep = deploy_adapter("flash-1-abc", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0", "RTX 4090")
    assert dep.state == "ready"


def test_deploy_5xx_recovers_when_new_registry_record_omits_subfolder(monkeypatch):
    """Older serving builds omit subfolder; accept it only when there was no prior deployment."""
    import httpx

    import flash.serve.deploy as deploy_mod

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setattr(deploy_mod, "adapter_artifact_lora_rank", lambda *a, **k: 32)
    req = httpx.Request("POST", "https://serve.example/adapters")
    resp = httpx.Response(502, text="bad gateway", request=req)
    monkeypatch.setattr(deploy_mod.httpx, "post", lambda *a, **k: resp)
    responses = iter([_registry_resp([]), _registry_resp([{"adapter_id": "flash-1-abc"}])])
    monkeypatch.setattr(deploy_mod.httpx, "get", lambda *a, **k: next(responses))

    dep = deploy_adapter("flash-1-abc", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0", "RTX 4090")
    assert dep.state == "ready"


def test_deploy_5xx_rejects_subfolderless_record_after_prior_deployment(monkeypatch):
    """A checkpoint swap must not accept a subfolder-less record after an ambiguous POST failure."""
    import httpx

    import flash.serve.deploy as deploy_mod
    from flash.serve.deploy import ServingError

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setattr(deploy_mod, "adapter_artifact_lora_rank", lambda *a, **k: 32)
    req = httpx.Request("POST", "https://serve.example/adapters")
    resp = httpx.Response(502, text="bad gateway", request=req)
    monkeypatch.setattr(deploy_mod.httpx, "post", lambda *a, **k: resp)
    responses = iter(
        [
            _registry_resp([{"adapter_id": "flash-1-abc"}]),
            _registry_resp([{"adapter_id": "flash-1-abc"}]),
        ]
    )
    monkeypatch.setattr(deploy_mod.httpx, "get", lambda *a, **k: next(responses))

    with pytest.raises(ServingError) as ei:
        deploy_adapter("flash-1-abc", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0", "RTX 4090")
    assert "cannot confirm" in str(ei.value)


def test_deployment_dict_carries_openai_v1_url(monkeypatch):
    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    dep = deploy_adapter("r1", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0", "RTX 4090", dry_run=True)
    data = dep.to_dict()
    assert data["endpoint_name"] == "https://serve.example"
    assert data["url"] == "https://serve.example/v1"
