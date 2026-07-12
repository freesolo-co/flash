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
    dep = deploy_adapter("r1", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0", dry_run=True)
    data = dep.to_dict()
    assert data["state"] == "dry_run"
    assert "gpu" not in data
    assert "mode" not in data
    assert "idle_timeout_s" not in data
    assert "est_idle_cost_usd_per_day" not in data
    assert "previous_registry_snapshot" not in data
    assert "registry_revision" not in data


def test_undeploy_absent_snapshot_is_clean(monkeypatch):
    import flash.serve.deploy as deploy_mod

    monkeypatch.setattr(deploy_mod, "snapshot_adapter_record", lambda _run_id: None)
    monkeypatch.setattr(deploy_mod, "_read_registry_snapshot", lambda _run_id: None)
    monkeypatch.setattr(
        deploy_mod.httpx, "delete", lambda *a, **k: pytest.fail("absent adapter must not delete")
    )
    assert undeploy_adapter("flash-1-gone") == []


def test_deployment_roundtrip_dict():
    d = Deployment(
        run_id="r",
        model="m",
        adapter_hf_prefix="p",
        openai_model="r",
        endpoint_name="https://serve.example",
    )
    data = d.to_dict()
    assert data["run_id"] == "r"
    assert "gpu" not in data
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


def test_deployment_dict_carries_openai_v1_url(monkeypatch):
    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    dep = deploy_adapter("r1", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0", dry_run=True)
    data = dep.to_dict()
    assert data["endpoint_name"] == "https://serve.example"
    assert data["url"] == "https://serve.example/v1"
