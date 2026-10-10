from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from flash.serving.loadtest.schema import Scenario, public_scenario_dict


def scenario_payload() -> dict:
    return {
        "schema_version": 1,
        "name": "fake hosted load",
        "endpoint": "https://example.invalid",
        "expected_deployment": {"sha": "94210a3", "deployment_id": "deploy-1"},
        "credential_env": "FAKE_SERVING_TOKEN",
        "required_capabilities": ["permanent_checkpoint_identity"],
        "discovery": {
            "enabled": True,
            "include": ["model-a", "model-b", "model-c"],
            "exclude": ["model-b"],
            "require": ["model-a"],
        },
        "targets": [
            {
                "name": "adapter-a",
                "kind": "adapter",
                "model": "adapter-a@final." + "a" * 40,
                "base_model": "model-a",
                "checkpoint": "adapter-a",
                "adapter_revision": "adapter-a@final." + "a" * 40,
                "hf_revision": "a" * 40,
            }
        ],
        "profiles": [
            {
                "name": "short",
                "messages": [{"role": "user", "content": "secret prompt"}],
                "max_tokens": 8,
            }
        ],
        "client": {"max_in_flight": 2, "max_scheduling_lag_ms": 10.0},
        "seed": 7,
        "fake": True,
        "phases": [
            {
                "name": "cold",
                "kind": "cold_burst",
                "requests": 2,
                "burst_window_seconds": 0.1,
                "cold_intent": "cold_scale_out",
            },
            {"name": "warm", "kind": "warm", "requests": 2, "concurrency": 1},
            {
                "name": "sustained",
                "kind": "sustained",
                "duration_seconds": 1.0,
                "rate_rps": 2.0,
            },
            {
                "name": "mixed",
                "kind": "mixed",
                "duration_seconds": 1.0,
                "rate_rps": 2.0,
            },
            {
                "name": "overload",
                "kind": "overload",
                "stages": [{"duration_seconds": 1.0, "rate_rps": 3.0}],
            },
        ],
    }


def test_schema_accepts_all_five_ordered_phase_kinds() -> None:
    scenario = Scenario.model_validate(scenario_payload())
    assert [phase.kind for phase in scenario.phases] == [
        "cold_burst",
        "warm",
        "sustained",
        "mixed",
        "overload",
    ]
    assert scenario.phases[0].cold_attestation == "cold_scale_out_intent_http_unattested"


def test_scenarios_may_omit_phase_kinds() -> None:
    payload = scenario_payload()
    payload["phases"] = [payload["phases"][1]]
    assert Scenario.model_validate(payload).phases[0].kind == "warm"


def test_cold_burst_must_be_first_inference_phase_when_present() -> None:
    payload = scenario_payload()
    payload["phases"] = [payload["phases"][1], payload["phases"][0]]
    with pytest.raises(ValidationError, match="cold_burst must be the first"):
        Scenario.model_validate(payload)


def test_unknown_fields_and_inapplicable_phase_fields_are_rejected() -> None:
    payload = scenario_payload()
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Scenario.model_validate(payload)

    payload = scenario_payload()
    payload["phases"][2]["requests"] = 10
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Scenario.model_validate(payload)


def test_sustained_and_mixed_are_duration_open_loop_only() -> None:
    payload = scenario_payload()
    payload["phases"][2].pop("duration_seconds")
    payload["phases"][2]["requests"] = 2
    with pytest.raises(ValidationError):
        Scenario.model_validate(payload)


def test_adapter_target_provenance_headers_are_optional_but_validated() -> None:
    """omitting a provenance header is allowed; supplying a malformed one is not.

    not every hosted deployment emits adapter-revision and hf-revision, so a scenario may verify
    checkpoint identity alone. that is different from accepting a bad value: a supplied
    hf_revision must still be a canonical hub commit sha.
    """
    payload = scenario_payload()
    payload["targets"][0].pop("hf_revision")
    scenario = Scenario.model_validate(payload)
    assert scenario.targets[0].hf_revision is None
    assert scenario.targets[0].checkpoint

    malformed = scenario_payload()
    malformed["targets"][0]["hf_revision"] = "not-a-sha"
    with pytest.raises(ValidationError, match="hf_revision"):
        Scenario.model_validate(malformed)


def test_adapter_checkpoint_is_always_required() -> None:
    payload = scenario_payload()
    payload["targets"][0].pop("checkpoint")
    with pytest.raises(ValidationError, match="checkpoint"):
        Scenario.model_validate(payload)


def test_discovery_requires_include_require_consistency() -> None:
    payload = scenario_payload()
    payload["discovery"]["require"] = ["model-z"]
    with pytest.raises(ValidationError, match="required models must be included"):
        Scenario.model_validate(payload)


def test_artifact_scenario_redacts_prompt_content() -> None:
    scenario = Scenario.model_validate(copy.deepcopy(scenario_payload()))
    public = public_scenario_dict(scenario)
    assert public["profiles"][0]["messages"][0]["content"] == "[redacted]"
    assert "secret prompt" not in str(public)
