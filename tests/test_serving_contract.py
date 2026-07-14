"""Serving client contract and token pricing."""

from __future__ import annotations

import pytest

from flash.serve.deploy import (
    Deployment,
    deploy_adapter,
    serving_base_url,
    undeploy_adapter,
)

MUTATION_ID = "00000000-0000-4000-8000-000000000001"


def test_serving_base_url_default_and_override(monkeypatch):
    from flash.serve.deploy import DEFAULT_FREESOLO_SERVING_URL

    monkeypatch.delenv("FREESOLO_SERVING_URL", raising=False)
    assert serving_base_url() == DEFAULT_FREESOLO_SERVING_URL
    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example/")
    assert serving_base_url() == "https://serve.example"
    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example/v1/")
    assert serving_base_url() == "https://serve.example"


def test_deploy_dry_run_has_no_user_facing_mode():
    dep = deploy_adapter(
        "r1", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0", mutation_id=MUTATION_ID, dry_run=True
    )
    data = dep.to_dict()
    assert data == {
        "run_id": "r1",
        "model": "Qwen/Qwen3.5-0.8B",
        "adapter_hf_prefix": "rl/r1/seed0/adapter",
        "openai_model": "r1",
        "endpoint_name": serving_base_url(),
        "openai_base_url": f"{serving_base_url()}/v1",
        "state": "dry_run",
    }


def test_undeploy_absent_record_is_clean(monkeypatch):
    import flash.serve.deploy as deploy_mod

    monkeypatch.setattr(deploy_mod, "read_adapter_record", lambda _run_id: None)
    monkeypatch.setattr(
        deploy_mod,
        "_serving_request",
        lambda *a, **k: pytest.fail("absent adapter must not delete"),
    )
    assert undeploy_adapter("flash-1-gone") == []


def test_undeploy_disables_exact_registry_identity(monkeypatch):
    import flash.serve.deploy as deploy_mod

    monkeypatch.setattr(
        deploy_mod,
        "read_adapter_record",
        lambda _run_id: {"registry_revision": 7, "mutation_id": "m7", "status": "ready"},
    )
    disabled = []
    monkeypatch.setattr(
        deploy_mod, "disable_owned_adapter", lambda *args: disabled.append(args)
    )
    assert undeploy_adapter("flash-7-live") == ["flash-7-live"]
    assert disabled == [("flash-7-live", 7, "m7")]


@pytest.mark.parametrize(
    "record",
    [
        pytest.param({"registry_revision": 0, "mutation_id": "m7"}, id="nonpositive-revision"),
        pytest.param({"registry_revision": "7", "mutation_id": "m7"}, id="string-revision"),
        pytest.param({"registry_revision": 7, "mutation_id": "  "}, id="blank-mutation"),
        pytest.param({"registry_revision": 7}, id="missing-mutation"),
    ],
)
def test_undeploy_malformed_identity_raises_controlled_error(monkeypatch, record):
    # A malformed persisted identity must raise a controlled ServingError (mapped to a 502 by the
    # undeploy route) rather than a raw KeyError/ValueError that would surface as an opaque 500.
    import flash.serve.deploy as deploy_mod

    monkeypatch.setattr(deploy_mod, "read_adapter_record", lambda _run_id: {**record, "status": "ready"})
    monkeypatch.setattr(
        deploy_mod,
        "disable_owned_adapter",
        lambda *_args: pytest.fail("a malformed registry identity must not be disabled"),
    )
    with pytest.raises(deploy_mod.ServingError, match="registry identity was malformed"):
        undeploy_adapter("flash-7-bad")


def test_deployment_roundtrip_dict():
    d = Deployment(
        run_id="r",
        model="m",
        adapter_hf_prefix="p",
        openai_model="r",
        endpoint_name="https://serve.example",
        openai_base_url="https://serve.example/v1",
    )
    data = d.to_dict()
    assert data["run_id"] == "r"
    assert data["openai_base_url"] == "https://serve.example/v1"
    assert "url" not in data
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
    dep = deploy_adapter(
        "r1", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0", mutation_id=MUTATION_ID, dry_run=True
    )
    data = dep.to_dict()
    assert data["endpoint_name"] == "https://serve.example"
    assert data["openai_base_url"] == "https://serve.example/v1"
    assert "url" not in data


def test_new_deployment_does_not_duplicate_existing_v1_suffix(monkeypatch):
    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example/v1/")
    dep = deploy_adapter(
        "r1", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0", mutation_id=MUTATION_ID, dry_run=True
    )
    data = dep.to_dict()
    assert data["endpoint_name"] == "https://serve.example"
    assert data["openai_base_url"] == "https://serve.example/v1"
    assert "url" not in data
