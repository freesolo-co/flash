import json
import time
from types import SimpleNamespace

import pytest

from flash.benchmarks import managed
from flash.benchmarks.evaluation import evaluate_targets
from flash.benchmarks.evidence import Recorder, load_events, write_json
from flash.benchmarks.schema import Complete, Quality, Request


class FakeClient:
    api_key = "test"

    def __init__(self):
        self.submitted = []
        self.deployed = []
        self.undeployed = []
        self.cancelled = []
        self.state = "done"

    def create_run(self, spec):
        self.submitted.append(spec)
        return {"run_id": "run-1", "state": self.state}

    def get_run(self, run_id):
        return {"run_id": run_id, "state": self.state}

    def checkpoints(self, run_id):
        return [{"step": 4}, {"step": 8}]

    def deploy(self, checkpoint):
        self.deployed.append(checkpoint)
        return {"state": "ready", "checkpoint_id": checkpoint, "generation": 1}

    def undeploy(self, checkpoint):
        self.undeployed.append(checkpoint)

    def cancel_run(self, run_id):
        self.cancelled.append(run_id)

    def chat(self, adapter, messages, **kwargs):
        return {"choices": [{"message": {"content": "42"}}]}


def prepare_run(tmp_path, monkeypatch, case, request_data, *, evaluator=None):
    config = {
        "hooks": "test.hooks",
        "api_url": "https://example.org",
        "credential_env": "TEST_KEY",
        "episodes": [{"turns": [{"prompt": "6*7?"}], "expected": "42"}],
        "training_spec": {"algorithm": "sft", "model": case.revisions.model},
        "poll_s": 0.0,
    }
    case.configuration = config
    case.target.metric = "exact_match"
    request_data.update(
        case=case.model_dump(), synthetic=False, origin_monotonic_s=time.monotonic()
    )
    request_path = tmp_path / "request.json"
    write_json(request_path, request_data)
    calls = []
    hooks = SimpleNamespace(
        prepare=lambda context: calls.append("prepare"),
        cleanup=lambda context: calls.append("cleanup"),
        collect=lambda context: [],
    )
    if evaluator is not None:
        hooks.evaluate = evaluator
    monkeypatch.setattr(managed.importlib, "import_module", lambda name: hooks)
    client = FakeClient()
    monkeypatch.setattr(managed, "_client", lambda config: client)
    return request_path, client, calls


def test_native_training_evaluates_checkpoints_and_cleans_only_owned_resources(
    tmp_path, monkeypatch, case, request_data
):
    request_path, client, calls = prepare_run(tmp_path, monkeypatch, case, request_data)
    managed.run(request_path)
    events = load_events(tmp_path / "events.jsonl")
    assert client.submitted[0]["seed"] == 17
    assert client.deployed == ["run-1/step-4", "run-1/step-8", "run-1/final"]
    assert client.undeployed == client.deployed[::-1]
    assert calls == ["prepare", "cleanup"]
    assert [event.checkpoint for event in events if isinstance(event, Quality)] == client.deployed
    assert isinstance(events[-1], Complete)
    assert events[-1].outcome == "succeeded"
    assert json.loads((tmp_path / "resources.json").read_text())["run_id"] == "run-1"


def test_evaluator_failure_cancels_training_and_preserves_failed_completion(
    tmp_path, monkeypatch, case, request_data
):
    def fail(context, targets):
        raise RuntimeError("environment failed")

    request_path, client, _calls = prepare_run(
        tmp_path, monkeypatch, case, request_data, evaluator=fail
    )
    client.state = "running"
    with pytest.raises(RuntimeError, match="environment failed"):
        managed.run(request_path)
    assert client.cancelled == ["run-1"]
    assert client.undeployed == ["run-1/step-4"]
    assert load_events(tmp_path / "events.jsonl")[-1].outcome == "failed"


def test_agent_driver_refuses_generic_single_turn_evaluator(
    tmp_path, monkeypatch, case, request_data
):
    case.workload = "long_multi_turn"
    request_path, client, calls = prepare_run(tmp_path, monkeypatch, case, request_data)
    with pytest.raises(ValueError, match="environment-aware"):
        managed.run(request_path)
    assert not client.submitted
    assert not calls


def test_evaluation_records_per_customer_latency_and_multi_turn_history(tmp_path):
    histories = []

    class Client(FakeClient):
        def chat(self, adapter, messages, **kwargs):
            histories.append(list(messages))
            return super().chat(adapter, messages, **kwargs)

    recorder = Recorder(tmp_path / "events.jsonl", time.monotonic())
    episodes = [
        {"turns": [{"prompt": "Remember 42"}, {"prompt": "What number?"}], "expected": "42"}
    ]
    value, samples = evaluate_targets(
        [(Client(), "adapter-a", "alice"), (Client(), "adapter-b", "bob")], episodes, recorder
    )
    events = load_events(tmp_path / "events.jsonl")
    assert value == 1.0
    assert samples == 2
    assert len(events) == 4
    assert all(isinstance(event, Request) and event.success for event in events)
    assert {event.customer for event in events} == {"alice", "bob"}
    assert any(
        [message["role"] for message in history] == ["user", "assistant", "user"]
        for history in histories
    )


def test_evaluation_failure_is_recorded_and_not_scored_as_success(tmp_path):
    class Client(FakeClient):
        def chat(self, *args, **kwargs):
            raise RuntimeError("request failed")

    recorder = Recorder(tmp_path / "events.jsonl", time.monotonic())
    with pytest.raises(RuntimeError, match="request failed"):
        evaluate_targets(
            [(Client(), "adapter-a", "alice")],
            [{"turns": [{"prompt": "hi"}], "expected": "hello"}],
            recorder,
        )
    assert not load_events(tmp_path / "events.jsonl")[0].success


def test_ambiguous_submission_does_not_report_successful_cleanup(
    tmp_path, monkeypatch, case, request_data
):
    request_path, client, calls = prepare_run(tmp_path, monkeypatch, case, request_data)

    def ambiguous(spec):
        raise TimeoutError("submission response was lost")

    monkeypatch.setattr(client, "create_run", ambiguous)
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        managed.run(request_path)
    assert json.loads((tmp_path / "submission.json").read_text())["state"] == "pending"
    assert json.loads((tmp_path / "cleanup.json").read_text())["errors"] == [
        "unacknowledged_submission"
    ]
    assert load_events(tmp_path / "events.jsonl")[-1].outcome == "failed"
    assert calls == ["prepare", "cleanup"]
