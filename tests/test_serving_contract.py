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
    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example/v1/")
    assert serving_base_url() == "https://serve.example"


def test_deploy_dry_run_has_no_user_facing_mode():
    dep = deploy_adapter("r1", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0", dry_run=True)
    data = dep.to_dict()
    assert data["state"] == "dry_run"
    assert "gpu" not in data
    assert "mode" not in data
    assert "idle_timeout_s" not in data
    assert "est_idle_cost_usd_per_day" not in data


def _stub_deploy_preconditions(monkeypatch, deploy_mod) -> None:
    monkeypatch.setattr(deploy_mod, "resolve_hf_revision", lambda repo: "a" * 40)
    monkeypatch.setattr(deploy_mod, "adapter_artifact_lora_rank", lambda *a, **k: 32)
    monkeypatch.setattr(
        deploy_mod,
        "_require_serving_capabilities",
        lambda **_kwargs: {
            "immutable_adapter_revisions",
            "alias_compare_and_swap",
            "revision_provenance",
        },
    )


def test_real_deploy_translates_serving_5xx_to_serving_error(monkeypatch):
    import httpx

    import flash.serve.deploy as deploy_mod
    from flash.serve.deploy import ServingError

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    req = httpx.Request("POST", "https://serve.example/adapters")
    resp = httpx.Response(500, text="no base-model engines loaded", request=req)
    _stub_deploy_preconditions(monkeypatch, deploy_mod)
    monkeypatch.setattr(deploy_mod.httpx, "post", lambda *a, **k: resp)

    with pytest.raises(ServingError) as ei:
        deploy_adapter("flash-1-abc", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0")
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
    _stub_deploy_preconditions(monkeypatch, deploy_mod)
    monkeypatch.setattr(deploy_mod.httpx, "post", lambda *a, **k: resp)

    with pytest.raises(ServingError) as ei:
        deploy_adapter("flash-1-abc", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0")
    msg = str(ei.value)
    assert ei.value.status_code == 401
    assert "401" in msg
    assert "FREESOLO_INTERNAL_KEY" in msg
    assert "no engine" not in msg
    assert "operator must check" not in msg


def _stub_healthz(monkeypatch, deploy_mod, capabilities: list[str]) -> None:
    """Stub the serving /healthz GET so _require_serving_capabilities sees `capabilities`."""

    class _Resp:
        def json(self):
            return {"capabilities": list(capabilities)}

    monkeypatch.setattr(deploy_mod, "_serving_request", lambda method, url, **k: _Resp())


def test_require_capabilities_provenance_is_preferred_not_required(monkeypatch):
    # The production serving backend advertises the two safety-critical caps + the deferred
    # thinking/structured-outputs cap, but NOT `revision_provenance`. That must NOT block deploys
    # (it only gates the rare ambiguous-registration recovery path).
    import flash.serve.deploy as deploy_mod

    _stub_healthz(
        monkeypatch,
        deploy_mod,
        [
            "immutable_adapter_revisions",
            "alias_compare_and_swap",
            "thinking_structured_outputs_deferred_v1",
        ],
    )
    # Does not raise, with or without the thinking/structured-outputs requirement.
    deploy_mod._require_serving_capabilities()
    deploy_mod._require_serving_capabilities(thinking_structured_outputs=True)


def test_require_capabilities_still_fails_on_missing_safety_critical(monkeypatch):
    import flash.serve.deploy as deploy_mod
    from flash.serve.deploy import ServingError

    # Missing the atomic alias CAS (safety-critical) -> deploy MUST still be blocked.
    _stub_healthz(monkeypatch, deploy_mod, ["immutable_adapter_revisions", "revision_provenance"])
    with pytest.raises(ServingError) as ei:
        deploy_mod._require_serving_capabilities()
    assert "alias_compare_and_swap" in str(ei.value)
    assert "revision_provenance" not in str(ei.value)  # provenance is not the blocker


def test_require_capabilities_thinking_structured_outputs_required_when_used(monkeypatch):
    import flash.serve.deploy as deploy_mod
    from flash.serve.deploy import ServingError

    # Backend lacks the deferred thinking/structured-outputs cap -> a thinking+SO deploy is blocked.
    _stub_healthz(monkeypatch, deploy_mod, ["immutable_adapter_revisions", "alias_compare_and_swap"])
    deploy_mod._require_serving_capabilities()  # fine without SO
    with pytest.raises(ServingError):
        deploy_mod._require_serving_capabilities(thinking_structured_outputs=True)


def test_real_deploy_translates_unreachable_serving_to_serving_error(monkeypatch):
    import httpx

    import flash.serve.deploy as deploy_mod
    from flash.serve.deploy import ServingError

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")

    def fake_post(url, *a, **k):
        raise httpx.ConnectError("connection refused", request=httpx.Request("POST", url))

    _stub_deploy_preconditions(monkeypatch, deploy_mod)
    monkeypatch.setattr(deploy_mod.httpx, "post", fake_post)
    with pytest.raises(ServingError) as ei:
        deploy_adapter("flash-1-abc", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0")
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

        def json(self):
            return {
                "run_id": "flash-1-abc",
                "disabled_aliases": ["flash-1-abc"],
                "disabled_revisions": ["flash-1-abc@final." + "a" * 40],
            }

    def fake_delete(url, headers=None, timeout=None, follow_redirects=None):
        deleted_urls.append(url)
        return _Resp()

    monkeypatch.setattr(deploy_mod.httpx, "delete", fake_delete)
    out = undeploy_adapter("flash-1-abc")
    assert out["run_id"] == "flash-1-abc"
    assert out["disabled_aliases"] == ["flash-1-abc"]
    assert out["serving_deregistered"] is True
    assert deleted_urls == ["https://serve.example/adapters/flash-1-abc"]


def test_undeploy_404_is_clean(monkeypatch):
    import flash.serve.deploy as deploy_mod

    class _Resp:
        status_code = 404

        def raise_for_status(self):  # pragma: no cover
            raise AssertionError("404 must not raise")

    monkeypatch.setattr(deploy_mod.httpx, "delete", lambda *a, **k: _Resp())
    assert undeploy_adapter("flash-1-gone") == {
        "run_id": "flash-1-gone",
        "disabled_aliases": [],
        "disabled_revisions": [],
        "serving_deregistered": False,
    }


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
    from flash.serve.pricing import SERVING_MARKUP, SERVING_PRICES

    assert set(SERVING_PRICES) == set(MODELS)
    ordered_prices = sorted(SERVING_PRICES.items())
    assert [model_id for model_id, _price in ordered_prices] == sorted(MODELS)
    assert pytest.approx(1.20) == SERVING_MARKUP
    for model_id, price in ordered_prices:
        assert price.model_id == model_id
        assert price.typical_input_usd_per_mtok > 0
        assert price.typical_output_usd_per_mtok > 0
        assert price.typical_cached_input_usd_per_mtok > 0
        assert price.billed_input_usd_per_mtok > 0
        assert price.billed_output_usd_per_mtok > 0
        assert price.billed_cached_input_usd_per_mtok > 0
        assert price.billed_cached_input_usd_per_mtok < price.billed_input_usd_per_mtok
        assert price.billed_input_usd_per_mtok == pytest.approx(
            price.typical_input_usd_per_mtok * SERVING_MARKUP
        )
        assert price.billed_output_usd_per_mtok == pytest.approx(
            price.typical_output_usd_per_mtok * SERVING_MARKUP
        )
        assert price.billed_cached_input_usd_per_mtok == pytest.approx(
            price.typical_cached_input_usd_per_mtok * SERVING_MARKUP
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
    assert data["openai_base_url"] == "https://serve.example/v1"
    assert "url" not in data


def test_immutable_revision_identifier_uses_full_hub_sha():
    from flash.serve.deploy import adapter_revision_id

    sha = "a1" * 20
    assert adapter_revision_id("flash-1-abc", 32, sha) == f"flash-1-abc@step-32.{sha}"
    assert adapter_revision_id("flash-1-abc", None, sha) == f"flash-1-abc@final.{sha}"


def test_deploy_registers_pinned_revision_then_smokes_then_cas(monkeypatch):
    import flash.serve.deploy as deploy

    sha = "b2" * 20
    revision = f"flash-1-abc@step-20.{sha}"
    events = []
    requests = []

    class Resp:
        status_code = 200

        def __init__(self, payload=None):
            self.payload = payload or {}
            self.headers = {}

        def json(self):
            return self.payload

    monkeypatch.setattr(deploy, "resolve_hf_revision", lambda repo: events.append("sha") or sha)

    def artifact_rank(repo, subfolder, *, hf_revision):
        assert hf_revision == sha
        events.append("verify")
        return 32

    monkeypatch.setattr(deploy, "adapter_artifact_lora_rank", artifact_rank)
    def _caps(**_kwargs):
        events.append("capabilities")
        return {
            "immutable_adapter_revisions",
            "alias_compare_and_swap",
            "revision_provenance",
        }

    monkeypatch.setattr(deploy, "_require_serving_capabilities", _caps)

    def request(method, url, *, json=None, ok_statuses=()):
        requests.append((method, url, json))
        if url.endswith("/activate"):
            events.append("activate")
            return Resp(
                {
                    "adapter_id": "flash-1-abc",
                    "target_adapter_revision": revision,
                    "previous_adapter_revision": "flash-1-abc@step-10." + "c3" * 20,
                    "checkpoint": "flash-1-abc/step-20",
                    "updated_at": "2026-07-12T12:00:15Z",
                }
            )
        events.append("register")
        return Resp()

    monkeypatch.setattr(deploy, "_serving_request", request)

    def wait_ready(
        adapter_revision,
        subfolder,
        *,
        expected_identity=None,
        require_provenance=True,
    ):
        assert adapter_revision == revision
        assert expected_identity["metadata"]["hf_revision"] == sha
        events.append("ready")
        return {}

    monkeypatch.setattr(deploy, "_wait_revision_ready", wait_ready)

    def smoke(adapter_revision, checkpoint):
        assert adapter_revision == revision
        assert checkpoint == "flash-1-abc/step-20"
        events.append("smoke")

    previous = "flash-1-abc@step-10." + "c3" * 20
    result = deploy.deploy_adapter(
        "flash-1-abc",
        "Qwen/Qwen3.5-0.8B",
        "org/repo",
        "sft/flash-1-abc/checkpoints/step_20",
        checkpoint_step=20,
        expected_adapter_revision=previous,
        before_activate=smoke,
    )

    assert events == ["sha", "verify", "capabilities", "register", "ready", "smoke", "activate"]
    registration = requests[0][2]
    assert registration["adapter_id"] == revision
    # the client does NOT send a "status" -- the serving backend sets it and rejects it as an
    # extra field (see deploy._register body).
    assert "status" not in registration
    assert registration["metadata"]["hf_revision"] == sha
    assert requests[1][2] == {"expected_adapter_revision": previous}
    assert result.adapter_revision == revision
    assert result.openai_model == "flash-1-abc"
    assert result.verified_at is None


def test_registration_conflict_is_not_masked_by_existing_revision(monkeypatch):
    import flash.serve.deploy as deploy

    _stub_deploy_preconditions(monkeypatch, deploy)
    monkeypatch.setattr(
        deploy,
        "_serving_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            deploy.ServingError("revision conflict", status_code=409)
        ),
    )
    monkeypatch.setattr(
        deploy,
        "_registered_adapter",
        lambda adapter_id: pytest.fail(
            "a 409 conflict must not be reconciled as transport ambiguity"
        ),
    )

    with pytest.raises(deploy.ServingError, match="revision conflict"):
        deploy.deploy_adapter(
            "flash-1",
            "Qwen/Qwen3.5-0.8B",
            "org/repo",
            "sft/flash-1/seed0",
        )


def test_ambiguous_registration_requires_matching_immutable_identity(monkeypatch):
    import flash.serve.deploy as deploy

    _stub_deploy_preconditions(monkeypatch, deploy)
    monkeypatch.setattr(
        deploy,
        "_serving_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(deploy.ServingError("timeout")),
    )
    monkeypatch.setattr(
        deploy,
        "_registered_adapter",
        lambda adapter_id: {
            "adapter_id": adapter_id,
            "repo_id": "other/repo",
            "repo_type": "dataset",
            "subfolder": "sft/flash-1/seed0/adapter",
            "base_model": "Qwen/Qwen3.5-0.8B",
            "checkpoint": "flash-1",
            "thinking": False,
            "metadata": {
                "record_type": "revision",
                "run_id": "flash-1",
                "checkpoint_step": None,
                "hf_revision": "a" * 40,
            },
        },
    )

    with pytest.raises(deploy.ServingError, match="timeout"):
        deploy.deploy_adapter(
            "flash-1",
            "Qwen/Qwen3.5-0.8B",
            "org/repo",
            "sft/flash-1/seed0",
        )


def test_revision_poll_rejects_mismatched_immutable_identity(monkeypatch):
    import flash.serve.deploy as deploy

    revision = "flash-1@final." + "a" * 40
    expected = {
        "adapter_id": revision,
        "repo_id": "org/repo",
        "repo_type": "dataset",
        "subfolder": "sft/flash-1/seed0/adapter",
        "base_model": "Qwen/Qwen3.5-0.8B",
        "checkpoint": "flash-1",
        "thinking": False,
        "metadata": {
            "record_type": "revision",
            "run_id": "flash-1",
            "checkpoint_step": None,
            "hf_revision": "a" * 40,
        },
    }
    mismatched = {**expected, "repo_id": "other/repo"}
    monkeypatch.setattr(
        deploy,
        "_registered_adapter",
        lambda adapter_id, **_kwargs: mismatched,
    )

    with pytest.raises(deploy.ServingError, match="different immutable identity"):
        deploy._wait_revision_ready(
            revision,
            expected["subfolder"],
            expected_identity=expected,
        )


def test_revision_poll_tolerates_absent_provenance_when_not_advertised(monkeypatch):
    import flash.serve.deploy as deploy

    revision = "flash-1@final." + "a" * 40
    expected = {
        "adapter_id": revision,
        "repo_id": "org/repo",
        "repo_type": "dataset",
        "subfolder": "sft/flash-1/seed0/adapter",
        "base_model": "Qwen/Qwen3.5-0.8B",
        "checkpoint": "flash-1",
        "thinking": False,
        "metadata": {
            "record_type": "revision",
            "run_id": "flash-1",
            "checkpoint_step": None,
            "hf_revision": "a" * 40,
        },
    }
    record = {**expected, "metadata": {"lifecycle_state": "ready"}}
    readback = {"record": record}
    monkeypatch.setattr(deploy, "READBACK_DELAY_SECONDS", 0)
    monkeypatch.setattr(
        deploy,
        "_registered_adapter",
        lambda adapter_id, **_kwargs: readback["record"],
    )

    assert (
        deploy._wait_revision_ready(
            revision,
            expected["subfolder"],
            expected_identity=expected,
            require_provenance=False,
            budget_s=0.1,
        )
        == record
    )

    readback["record"] = {**record, "repo_id": "other/repo"}
    with pytest.raises(deploy.ServingError, match="different immutable identity"):
        deploy._wait_revision_ready(
            revision,
            expected["subfolder"],
            expected_identity=expected,
            require_provenance=False,
            budget_s=0.1,
        )


def test_activation_transport_ambiguity_reconciles_authoritative_alias(monkeypatch):
    import flash.serve.deploy as deploy

    revision = "flash-1@final." + "a" * 40
    monkeypatch.setattr(
        deploy,
        "_serving_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(deploy.ServingError("timeout")),
    )
    monkeypatch.setattr(
        deploy,
        "_registered_adapter",
        lambda adapter_id: {
            "adapter_id": "flash-1",
            "updated_at": "2026-07-12T12:00:15Z",
            "metadata": {"record_type": "alias", "alias_of": revision},
        },
    )
    out = deploy._activate_revision("flash-1", revision, "flash-1", expected_adapter_revision=None)
    assert out["target_adapter_revision"] == revision
    assert out["updated_at"] == "2026-07-12T12:00:15Z"


def test_activation_readback_rejects_disabled_alias_with_stale_target(monkeypatch):
    import flash.serve.deploy as deploy

    revision = "flash-1@final." + "a" * 40
    monkeypatch.setattr(deploy, "READBACK_DELAY_SECONDS", 0)
    monkeypatch.setattr(
        deploy,
        "_serving_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(deploy.ServingError("timeout")),
    )
    monkeypatch.setattr(
        deploy,
        "_registered_adapter",
        lambda adapter_id: {
            "adapter_id": adapter_id,
            "status": "disabled",
            "metadata": {"record_type": "alias", "alias_of": revision},
        },
    )

    with pytest.raises(deploy.ActivationOutcomeUnknown, match="alias_activation_unknown"):
        deploy._activate_revision("flash-1", revision, "flash-1", expected_adapter_revision=None)


def test_activation_commit_survives_lost_response_and_transient_readback_failure(monkeypatch):
    import flash.serve.deploy as deploy

    previous = "flash-1@step-10." + "b" * 40
    revision = "flash-1@final." + "a" * 40
    monkeypatch.setattr(deploy, "READBACK_DELAY_SECONDS", 0)
    monkeypatch.setattr(
        deploy,
        "_serving_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(deploy.ServingError("response lost")),
    )
    readbacks = iter(
        [
            deploy.ServingError("readback unavailable"),
            {
                "adapter_id": "flash-1",
                "metadata": {"record_type": "alias", "alias_of": previous},
            },
            {
                "adapter_id": "flash-1",
                "updated_at": "2026-07-12T12:00:15Z",
                "metadata": {"record_type": "alias", "alias_of": revision},
            },
        ]
    )

    def read_alias(adapter_id):
        value = next(readbacks)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(deploy, "_registered_adapter", read_alias)

    out = deploy._activate_revision(
        "flash-1", revision, "flash-1", expected_adapter_revision=previous
    )

    assert out["target_adapter_revision"] == revision
    assert out["updated_at"] == "2026-07-12T12:00:15Z"


def test_activation_lost_response_and_readback_remains_explicitly_unknown(monkeypatch):
    import flash.serve.deploy as deploy

    revision = "flash-1@final." + "a" * 40
    monkeypatch.setattr(deploy, "READBACK_DELAY_SECONDS", 0)
    monkeypatch.setattr(
        deploy,
        "_serving_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(deploy.ServingError("response lost")),
    )
    monkeypatch.setattr(
        deploy,
        "_registered_adapter",
        lambda adapter_id: (_ for _ in ()).throw(deploy.ServingError("readback unavailable")),
    )

    with pytest.raises(deploy.ActivationOutcomeUnknown, match="alias_activation_unknown"):
        deploy._activate_revision("flash-1", revision, "flash-1", expected_adapter_revision=None)


def test_activation_reconciliation_accepts_alias_without_updated_at(monkeypatch):
    import flash.serve.deploy as deploy

    revision = "flash-1@final." + "a" * 40
    monkeypatch.setattr(
        deploy,
        "_serving_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(deploy.ServingError("timeout")),
    )
    monkeypatch.setattr(
        deploy,
        "_registered_adapter",
        lambda adapter_id: {
            "adapter_id": adapter_id,
            "metadata": {"record_type": "alias", "alias_of": revision},
        },
    )

    out = deploy._activate_revision("flash-1", revision, "flash-1", expected_adapter_revision=None)

    assert out["target_adapter_revision"] == revision
    assert out["updated_at"] is None


def test_first_activation_missing_alias_target_remains_ambiguous(monkeypatch):
    import flash.serve.deploy as deploy

    revision = "flash-1@final." + "a" * 40
    monkeypatch.setattr(deploy, "READBACK_DELAY_SECONDS", 0)
    monkeypatch.setattr(
        deploy,
        "_serving_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(deploy.ServingError("timeout")),
    )
    monkeypatch.setattr(
        deploy,
        "_registered_adapter",
        lambda adapter_id: {"adapter_id": adapter_id, "metadata": {"record_type": "alias"}},
    )

    with pytest.raises(deploy.ServingError, match="outcome is ambiguous"):
        deploy._activate_revision("flash-1", revision, "flash-1", expected_adapter_revision=None)


def test_activation_divergence_requires_reconciliation(monkeypatch):
    import flash.serve.deploy as deploy

    previous = "flash-1@step-10." + "b" * 40
    revision = "flash-1@step-20." + "a" * 40
    divergent = "flash-1@step-30." + "c" * 40
    monkeypatch.setattr(
        deploy,
        "_serving_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(deploy.ServingError("response lost")),
    )
    monkeypatch.setattr(
        deploy,
        "_registered_adapter",
        lambda adapter_id: {
            "adapter_id": adapter_id,
            "metadata": {"record_type": "alias", "alias_of": divergent},
        },
    )

    with pytest.raises(deploy.ActivationOutcomeUnknown, match="activation diverged"):
        deploy._activate_revision(
            "flash-1", revision, "flash-1/step-20", expected_adapter_revision=previous
        )


def test_activation_response_mismatch_reconciles_authoritative_previous_alias(monkeypatch):
    import flash.serve.deploy as deploy

    previous = "flash-1@step-10." + "b" * 40
    monkeypatch.setattr(deploy, "READBACK_DELAY_SECONDS", 0)
    revision = "flash-1@step-20." + "a" * 40

    class Resp:
        def json(self):
            return {
                "adapter_id": "flash-1",
                "target_adapter_revision": revision,
                "previous_adapter_revision": "wrong",
                "checkpoint": "flash-1/step-20",
            }

    monkeypatch.setattr(deploy, "_serving_request", lambda *args, **kwargs: Resp())
    monkeypatch.setattr(
        deploy,
        "_registered_adapter",
        lambda adapter_id: {
            "adapter_id": adapter_id,
            "updated_at": "2026-07-12T12:00:15Z",
            "metadata": {"record_type": "alias", "alias_of": previous},
        },
    )
    with pytest.raises(deploy.ServingError, match="was not committed"):
        deploy._activate_revision(
            "flash-1", revision, "flash-1/step-20", expected_adapter_revision=previous
        )


def test_malformed_activation_response_reconciles_committed_alias(monkeypatch):
    import flash.serve.deploy as deploy

    previous = "flash-1@step-10." + "b" * 40
    revision = "flash-1@step-20." + "a" * 40

    class Resp:
        def json(self):
            return {"adapter_id": "flash-1", "target_adapter_revision": "wrong"}

    monkeypatch.setattr(deploy, "_serving_request", lambda *args, **kwargs: Resp())
    monkeypatch.setattr(
        deploy,
        "_registered_adapter",
        lambda adapter_id: {
            "adapter_id": adapter_id,
            "updated_at": "2026-07-12T12:00:15Z",
            "metadata": {"record_type": "alias", "alias_of": revision},
        },
    )

    out = deploy._activate_revision(
        "flash-1", revision, "flash-1/step-20", expected_adapter_revision=previous
    )
    assert out == {
        "adapter_id": "flash-1",
        "target_adapter_revision": revision,
        "previous_adapter_revision": previous,
        "checkpoint": "flash-1/step-20",
        "updated_at": "2026-07-12T12:00:15Z",
    }


def test_undeploy_returns_disabled_aliases_and_revisions(monkeypatch):
    import flash.serve.deploy as deploy

    class Resp:
        status_code = 200

        def json(self):
            return {
                "run_id": "flash-1",
                "disabled_aliases": ["flash-1"],
                "disabled_revisions": ["flash-1@final." + "a" * 40],
            }

    monkeypatch.setattr(deploy, "_serving_request", lambda *args, **kwargs: Resp())
    out = deploy.undeploy_adapter("flash-1")
    assert out["serving_deregistered"] is True
    assert out["disabled_aliases"] == ["flash-1"]
    assert len(out["disabled_revisions"]) == 1


def test_undeploy_rejects_malformed_disabled_id_lists(monkeypatch):
    import flash.serve.deploy as deploy

    class Resp:
        status_code = 200

        def json(self):
            return {
                "run_id": "flash-1",
                "disabled_aliases": "flash-1",
                "disabled_revisions": [],
            }

    monkeypatch.setattr(deploy, "_serving_request", lambda *args, **kwargs: Resp())
    with pytest.raises(deploy.ServingError, match="invalid disabled_aliases"):
        deploy.undeploy_adapter("flash-1")


def test_serving_capabilities_are_required(monkeypatch):
    import flash.serve.deploy as deploy

    class Resp:
        def json(self):
            return {"capabilities": ["immutable_adapter_revisions"]}

    monkeypatch.setattr(deploy, "_serving_request", lambda *args, **kwargs: Resp())
    with pytest.raises(deploy.ServingError, match="alias_compare_and_swap"):
        deploy._require_serving_capabilities()


def test_capability_preflight_reads_healthz(monkeypatch):
    # serving exposes GET /healthz only; a /health preflight 404s and fails every real deploy.
    import flash.serve.deploy as deploy

    seen = {}

    class Resp:
        def json(self):
            return {
                "capabilities": [
                    "immutable_adapter_revisions",
                    "alias_compare_and_swap",
                    "revision_provenance",
                ]
            }

    def request(method, url, **kwargs):
        seen["method"] = method
        seen["url"] = url
        return Resp()

    monkeypatch.setattr(deploy, "_serving_request", request)
    deploy._require_serving_capabilities()
    assert seen["method"] == "GET"
    assert seen["url"] == f"{deploy.serving_base_url()}/healthz"


def test_activation_conflict_preserves_expected_alias(monkeypatch):
    import flash.serve.deploy as deploy

    previous = "flash-1@step-10." + "b" * 40
    revision = "flash-1@step-20." + "a" * 40
    monkeypatch.setattr(
        deploy,
        "_serving_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(deploy.ServingError("conflict")),
    )
    monkeypatch.setattr(
        deploy,
        "_registered_adapter",
        lambda adapter_id: {"metadata": {"alias_of": previous}},
    )
    with pytest.raises(deploy.ServingError, match="was not committed"):
        deploy._activate_revision(
            "flash-1", revision, "flash-1/step-20", expected_adapter_revision=previous
        )


def test_new_deployment_does_not_duplicate_existing_v1_suffix(monkeypatch):
    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example/v1/")
    dep = deploy_adapter("r1", "Qwen/Qwen3.5-0.8B", "repo", "rl/r1/seed0", dry_run=True)
    data = dep.to_dict()
    assert data["endpoint_name"] == "https://serve.example"
    assert data["openai_base_url"] == "https://serve.example/v1"
    assert "url" not in data
