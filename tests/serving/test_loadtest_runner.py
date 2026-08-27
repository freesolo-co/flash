from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from flash.serving.loadtest.artifacts import load_events, verify_result_directory
from flash.serving.loadtest.runner import run_scenario
from flash.serving.loadtest.schedule import FakeClock
from flash.serving.loadtest.schema import Scenario
from tests.serving.test_loadtest_schema import scenario_payload


def _health() -> dict:
    return {
        "ok": True,
        "accounting_ok": True,
        "deployment_sha": "94210a3",
        "deployment_id": "deploy-1",
        "capabilities": ["permanent_checkpoint_identity"],
        "base_models": ["model-a", "model-b", "model-c"],
    }


def _sse() -> bytes:
    values = [
        {"choices": [{"delta": {"reasoning_content": "hidden"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "generated-secret"}, "finish_reason": None}]},
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        },
    ]
    return (
        "".join(f"data: {json.dumps(value)}\n\n" for value in values).encode() + b"data: [DONE]\n\n"
    )


def _headers(request: httpx.Request) -> dict[str, str]:
    model = json.loads(request.content)["model"]
    if model.startswith("adapter-a@"):
        return {
            "X-Freesolo-Checkpoint": "adapter-a",
            "X-Freesolo-Adapter-Revision": model,
            "X-Freesolo-HF-Revision": "a" * 40,
        }
    return {}


def _factory(transport: httpx.AsyncBaseTransport):
    def build(**kwargs):
        return httpx.AsyncClient(transport=transport, **kwargs)

    return build


def test_subminute_all_five_phase_fake_end_to_end(tmp_path) -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.url.path == "/healthz":
            return httpx.Response(200, json=_health())
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer fake-secret"
        return httpx.Response(200, headers=_headers(request), content=_sse())

    scenario = Scenario.model_validate(scenario_payload())
    result_path = tmp_path / "result"
    summary = asyncio.run(
        run_scenario(
            scenario,
            "fake-secret",
            result_path,
            clock=FakeClock(),
            client_factory=_factory(httpx.MockTransport(handler)),
        )
    )
    verify_result_directory(result_path)
    events = load_events(result_path / "events.jsonl")
    terminals = [event for event in events if event["type"] == "request_terminal"]
    assert len(terminals) == 11
    assert {event["phase_kind"] for event in terminals} == {
        "cold_burst",
        "warm",
        "sustained",
        "mixed",
        "overload",
    }
    health_labels = {event["label"] for event in events if event["type"] == "health"}
    assert "phase:sustained:during" in health_labels
    assert "phase:mixed:during" in health_labels
    assert "phase:overload:during" in health_labels
    assert summary["fake"] is True
    assert summary["overall"]["retry_count"] == 0
    serialized = "".join(path.read_text() for path in result_path.iterdir())
    assert "fake-secret" not in serialized
    assert "secret prompt" not in serialized
    assert "generated-secret" not in serialized
    assert any(path == "/v1/chat/completions" for _, path in requests)


def test_open_loop_saturation_records_admission_misses_without_delayed_work(tmp_path) -> None:
    payload = scenario_payload()
    payload["discovery"] = {"enabled": False}
    payload["targets"] = [{"name": "base", "kind": "base_model", "model": "model-a"}]
    payload["client"] = {"max_in_flight": 1, "max_scheduling_lag_ms": 100.0}
    payload["phases"] = [
        {
            "name": "cold",
            "kind": "cold_burst",
            "requests": 4,
            "burst_window_seconds": 0.0,
            "cold_intent": "cold_scale_out",
        }
    ]
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.path == "/healthz":
            return httpx.Response(200, json=_health())
        calls += 1
        await asyncio.sleep(0.02)
        return httpx.Response(200, content=_sse())

    path = tmp_path / "result"
    asyncio.run(
        run_scenario(
            Scenario.model_validate(payload),
            "fake-secret",
            path,
            clock=FakeClock(),
            client_factory=_factory(httpx.MockTransport(handler)),
        )
    )
    terminals = [
        event for event in load_events(path / "events.jsonl") if event["type"] == "request_terminal"
    ]
    assert calls == 1
    assert sum(event["error_class"] == "client_admission_missed" for event in terminals) == 3
    assert all(event["retry_count"] == 0 for event in terminals)


def test_interruption_preserves_outcomes_already_observed(tmp_path) -> None:
    """an interrupt must not overwrite requests whose outcome the harness already saw.

    settling per request rather than returning a batch is what makes this true: only requests
    that never reached a terminal state become ``interrupted``. reporting completed successes as
    interrupted would understate throughput and hide real results.
    """
    payload = scenario_payload()
    payload["discovery"] = {"enabled": False}
    payload["targets"] = [{"name": "base", "kind": "base_model", "model": "model-a"}]
    payload["client"] = {"max_in_flight": 8, "max_scheduling_lag_ms": 1000.0}
    payload["phases"] = [
        {"name": "sustained", "kind": "sustained", "duration_seconds": 3.0, "rate_rps": 2.0}
    ]
    served = 0
    reached = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal served
        if request.url.path == "/healthz":
            return httpx.Response(200, json=_health())
        served += 1
        if served >= 3:
            reached.set()
            await asyncio.sleep(30)
        return httpx.Response(200, content=_sse())

    async def exercise() -> None:
        path = tmp_path / "result"
        task = asyncio.create_task(
            run_scenario(
                Scenario.model_validate(payload),
                "fake-secret",
                path,
                clock=FakeClock(),
                client_factory=_factory(httpx.MockTransport(handler)),
            )
        )
        await reached.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        terminals = [
            event
            for event in load_events(path / "events.jsonl")
            if event["type"] == "request_terminal"
        ]
        outcomes = [event["outcome"] for event in terminals]
        assert outcomes.count("success") == 2
        assert "interrupted" in outcomes
        # exactly one identified row per scheduled request: an interrupted phase must not flush
        # twice, and a row the harness synthesized still has to say which request it stands for
        request_ids = [event["request_id"] for event in terminals]
        assert len(terminals) == 6
        assert sorted(request_ids) == [f"request-{index:08d}" for index in range(6)]
        assert not (path / "complete.json").exists()

    asyncio.run(exercise())


def test_interruption_leaves_inspectable_invalid_result_without_completion(tmp_path) -> None:
    payload = scenario_payload()
    payload["discovery"] = {"enabled": False}
    payload["targets"] = [{"name": "base", "kind": "base_model", "model": "model-a"}]
    payload["phases"] = [{"name": "warm", "kind": "warm", "requests": 2, "concurrency": 1}]
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json=_health())
        entered.set()
        await release.wait()
        return httpx.Response(200, content=_sse())

    async def exercise() -> None:
        path = tmp_path / "result"
        task = asyncio.create_task(
            run_scenario(
                Scenario.model_validate(payload),
                "fake-secret",
                path,
                clock=FakeClock(),
                client_factory=_factory(httpx.MockTransport(handler)),
            )
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not (path / "complete.json").exists()
        events = load_events(path / "events.jsonl")
        terminals = [event for event in events if event["type"] == "request_terminal"]
        assert len(terminals) == 2
        assert {event["outcome"] for event in terminals} == {"interrupted"}
        assert any(event["type"] == "phase_interrupted" for event in events)

    asyncio.run(exercise())
