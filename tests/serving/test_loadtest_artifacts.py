from __future__ import annotations

import json

import pytest

from flash.serving.loadtest.artifacts import (
    ArtifactError,
    ResultDirectory,
    load_events,
    verify_result_directory,
)
from flash.serving.loadtest.schema import (
    CLAIM_LIMITATIONS,
    BaseTarget,
    HealthSnapshot,
    ResolvedScenario,
    Scenario,
)
from tests.serving.test_loadtest_schema import scenario_payload


def _scenario() -> Scenario:
    return Scenario.model_validate(scenario_payload())


def _resolved(scenario: Scenario) -> ResolvedScenario:
    return ResolvedScenario(
        authored=scenario,
        health=HealthSnapshot(
            ok=True,
            accounting_ok=True,
            deployment_sha="94210a3",
            deployment_id="deploy-1",
            capabilities=["permanent_checkpoint_identity"],
            base_models=["model-a", "model-b", "model-c"],
        ),
        targets=[BaseTarget(name="model-a", model="model-a")],
        phase_cold_attestations={"cold": "cold_scale_out_intent_http_unattested"},
        phase_capacity_expectations={"overload": False},
        claim_limitations=CLAIM_LIMITATIONS,
    )


def test_result_directory_is_exclusive_and_complete_is_written_last(tmp_path) -> None:
    path = tmp_path / "result"
    result = ResultDirectory.create(path, _scenario())
    with pytest.raises(ArtifactError, match="already exists"):
        ResultDirectory.create(path, _scenario())
    result.write_resolved(_resolved(_scenario()))
    result.events.write({"type": "request_terminal", "request_id": "request-1"})
    assert not (path / "complete.json").exists()
    completion = result.complete({"schema_version": 1})
    assert completion["event_rows"] == 1
    assert verify_result_directory(path) == completion


def test_incomplete_run_remains_inspectable_but_invalid(tmp_path) -> None:
    path = tmp_path / "result"
    result = ResultDirectory.create(path, _scenario())
    result.events.write({"type": "phase_interrupted", "phase_name": "cold"})
    result.abort()
    assert load_events(path / "events.jsonl") == [
        {"phase_name": "cold", "type": "phase_interrupted"}
    ]
    with pytest.raises(ArtifactError, match="incomplete"):
        verify_result_directory(path)


def test_verification_detects_hash_and_row_count_tampering(tmp_path) -> None:
    path = tmp_path / "result"
    result = ResultDirectory.create(path, _scenario())
    result.write_resolved(_resolved(_scenario()))
    result.events.write({"type": "request_terminal", "request_id": "request-1"})
    result.complete({"schema_version": 1})
    with (path / "events.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps({"type": "tamper"}) + "\n")
    with pytest.raises(ArtifactError, match="hash mismatch"):
        verify_result_directory(path)


def test_authored_and_resolved_artifacts_never_store_prompt_content(tmp_path) -> None:
    path = tmp_path / "result"
    scenario = _scenario()
    result = ResultDirectory.create(path, scenario)
    result.write_resolved(_resolved(scenario))
    result.abort()
    combined = (path / "scenario.authored.json").read_text() + (
        path / "scenario.resolved.json"
    ).read_text()
    assert "secret prompt" not in combined
    assert "[redacted]" in combined
