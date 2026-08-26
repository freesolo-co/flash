from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from flash.serving.src.accounting.usage import (
    build_usage_session,
    new_generation_id,
    principal_for_external_org,
)
from flash.serving.src.accounting.usage_outbox import RequestIdentity, UsageOutboxError
from flash.serving.src.engine import generation
from flash.serving.src.engine import support as engine_support
from flash.serving.src.http import inference_routes
from flash.serving.src.http.router import AdapterRouter, build_serving_app
from flash.serving.src.io.schemas import AdapterRecord, internal_adapter_payload
from tests.serving.conftest import RecordingUsageStore

BASE_MODEL = "Qwen/Qwen3.5-9B"


class _SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class _Engine:
    async def generate(self, *_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        choice = SimpleNamespace(
            index=0,
            text="ok",
            finish_reason="stop",
            token_ids=[21],
            logprobs=None,
        )
        yield SimpleNamespace(
            prompt_token_ids=[11, 12],
            num_cached_tokens=0,
            outputs=[choice],
        )


class _Owner:
    def __init__(self, record: AdapterRecord) -> None:
        self.record = record
        self.engine = _Engine()

    async def _lora_request(self, _adapter_id: str, _record_dict: Any) -> tuple[Any, Any]:
        return SimpleNamespace(lora_name=self.record.adapter_id), self.record

    @staticmethod
    def _lora_request_attestation(record: AdapterRecord, lora_request: Any) -> str:
        assert lora_request.lora_name == record.adapter_id
        return record.adapter_id

    def _enforce_expected_checkpoint(
        self, record: AdapterRecord, expected_checkpoint: str | None
    ) -> str:
        if expected_checkpoint is not None:
            assert expected_checkpoint == record.checkpoint
        assert record.checkpoint is not None
        return record.checkpoint

    @staticmethod
    def _thinking_default(_record: AdapterRecord, _payload: Any) -> bool:
        return False

    @staticmethod
    def _structured_outputs_state(
        _payload: Any, _record: AdapterRecord, _thinking: bool
    ) -> tuple[None, None, dict[str, Any]]:
        return None, None, {}

    @staticmethod
    async def _prepare_prompt_input(_payload: Any, _thinking: bool) -> dict[str, list[int]]:
        return {"prompt_token_ids": [11, 12]}

    @staticmethod
    def _self_heal_if_dead(_reason: str) -> None:
        return None

    @staticmethod
    def _close_prompt_images(_prompt_input: Any) -> None:
        return None

    @staticmethod
    def _replica_identifier() -> str:
        return "replica-1"


class _GenerationPool:
    def __init__(self, record: AdapterRecord) -> None:
        self.owner = _Owner(record)

    async def generate(
        self,
        _base_model: str,
        payload: Any,
        record: AdapterRecord,
        *,
        expected_checkpoint: str | None = None,
    ) -> dict[str, Any]:
        return await generation.generate(
            self.owner,
            payload.model_dump(mode="python"),
            internal_adapter_payload(record),
            expected_checkpoint,
            payload.generation_id,
        )

    async def stream_generate(
        self,
        _base_model: str,
        payload: Any,
        record: AdapterRecord,
        *,
        expected_checkpoint: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        async for event in generation.stream_generate(
            self.owner,
            payload.model_dump(mode="python"),
            internal_adapter_payload(record),
            expected_checkpoint,
            payload.generation_id,
        ):
            yield event

    async def register(self, _base_model: str, _record: AdapterRecord) -> None:
        return None

    async def unregister(
        self,
        _base_model: str,
        _adapter_id: str,
        _expected_generation: str | None = None,
    ) -> None:
        return None


def _revision() -> AdapterRecord:
    run_id = "duration"
    sha = hashlib.sha1(run_id.encode()).hexdigest()
    return AdapterRecord.model_validate(
        {
            "adapter_id": f"{run_id}@final.{sha}",
            "repo_id": "org/duration",
            "org_id": "org-1",
            "base_model": BASE_MODEL,
            "checkpoint": run_id,
            "thinking": False,
            "metadata": {
                "record_type": "revision",
                "run_id": run_id,
                "checkpoint_step": None,
                "hf_revision": sha,
            },
        }
    )


async def _authorize(_token: str, _adapter_id: str) -> str:
    return "org-1"


def _client(record: AdapterRecord, store: RecordingUsageStore) -> TestClient:
    return TestClient(
        build_serving_app(
            _GenerationPool(record),
            AdapterRouter([record]),
            usage_store=store,
            chat_authorizer=_authorize,
            deployment_id="deployment-1",
            deployment_sha="release-1",
        ),
        headers={"Authorization": "Bearer user-key"},
    )


def _install_clocks(monkeypatch: pytest.MonkeyPatch, wall_values: list[float]) -> None:
    wall_clock = _SequenceClock(wall_values)
    monotonic_clock = _SequenceClock([10.0, 10.1, 10.2, 10.3, 10.4])
    monkeypatch.setattr(
        generation,
        "time",
        SimpleNamespace(time=wall_clock, monotonic=monotonic_clock),
    )
    monkeypatch.setattr(
        inference_routes,
        "time",
        SimpleNamespace(time=wall_clock),
    )


@pytest.mark.parametrize(
    "wall_values",
    [
        pytest.param([100.0, 100.0, 90.0], id="rollback"),
        pytest.param([100.0, 100.0, 10_000.0], id="forward-jump"),
    ],
)
def test_nonstream_completed_generation_uses_monotonic_duration(
    monkeypatch: pytest.MonkeyPatch, wall_values: list[float]
) -> None:
    record = _revision()
    store = RecordingUsageStore()
    _install_clocks(monkeypatch, wall_values)

    with _client(record, store) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": record.adapter_id,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["created"] == 100
    assert len(store.finalized) == 1
    assert store.finalized[0].facts.generation_duration_seconds == pytest.approx(0.1)


@pytest.mark.parametrize(
    "wall_values",
    [
        pytest.param([100.0, 100.0, 101.0, 102.0, 103.0, 90.0], id="rollback"),
        pytest.param([100.0, 100.0, 101.0, 102.0, 103.0, 10_000.0], id="forward-jump"),
    ],
)
def test_stream_completed_generation_uses_monotonic_duration(
    monkeypatch: pytest.MonkeyPatch, wall_values: list[float]
) -> None:
    record = _revision()
    store = RecordingUsageStore()
    _install_clocks(monkeypatch, wall_values)

    with (
        _client(record, store) as client,
        client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": record.adapter_id,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as response,
    ):
        body = response.read()

    assert response.status_code == 200
    assert b'"type":"accounting_error"' not in body
    assert b"data: [DONE]" in body
    assert len(store.finalized) == 1
    assert store.finalized[0].facts.generation_duration_seconds == pytest.approx(0.4)


def test_support_usage_fields_use_monotonic_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    request_output = SimpleNamespace(prompt_token_ids=[11, 12], num_cached_tokens=0)

    def wall_clock_forbidden() -> float:
        raise AssertionError("elapsed inference duration read the wall clock")

    monkeypatch.setattr(
        engine_support,
        "time",
        SimpleNamespace(time=wall_clock_forbidden, monotonic=lambda: 10.25),
    )

    result = engine_support._stream_usage_fields(
        request_output,
        [21],
        start=10.0,
        request_id="fsgen-duration",
        engine_replica_id="replica-1",
        checkpoint="duration",
        thinking=False,
    )

    assert result["inference_time_seconds"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "invalid_duration",
    [
        pytest.param(-0.01, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
def test_invalid_duration_never_reaches_durable_finalization(invalid_duration: float) -> None:
    record = _revision()
    store = RecordingUsageStore()
    session = build_usage_session(
        store,
        identity=RequestIdentity(
            request_id=new_generation_id(),
            correlation_id="correlation-1",
        ),
        principal=principal_for_external_org("org-1"),
        requested=record,
        target=record,
        result={"checkpoint": record.checkpoint, "lora_request_adapter": record.adapter_id},
        deployment_id="deployment-1",
        serving_release="release-1",
        captured_at=datetime(2026, 8, 26, tzinfo=UTC),
    )
    result = {
        "prompt_token_ids": [11, 12],
        "completion_token_ids": [21],
        "reasoning_tokens": 0,
        "thinking": False,
        "inference_time_seconds": invalid_duration,
        "checkpoint": record.checkpoint,
        "lora_request_adapter": record.adapter_id,
    }

    with pytest.raises(UsageOutboxError, match="usage_duration_invalid"):
        asyncio.run(session.finalize(result))

    assert store.finalized == []
