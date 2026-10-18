"""Cancellable Modal stream protocol and offline transport behavior."""

from __future__ import annotations

import asyncio
import inspect
import queue as stdlib_queue
import time
import types
from collections.abc import AsyncIterator
from typing import Any

import modal
import pytest

from flash.serving.src.stream_channel import client
from flash.serving.src.stream_channel import engine as channel_engine
from flash.serving.src.stream_channel.engine import stream_generate_call
from flash.serving.src.stream_channel.protocol import (
    CALL_RESULT_MARGIN_SECONDS,
    CALL_RESULT_SECONDS,
    CLEANUP_SECONDS,
    CONTROL_PARTITION,
    DATA_PARTITION,
    ENGINE_CLEANUP_STEPS,
    ENGINE_CLEANUP_WAITS_PER_STEP,
    PROTOCOL_VERSION,
    ChannelErrorCode,
    ControlEnvelope,
    ControlSequenceValidator,
    DataEnvelope,
    DataSequenceValidator,
    StreamChannelError,
    TerminalManifest,
    validate_control,
    validate_data,
    validate_manifest,
)


class _AsyncMethod:
    def __init__(self, function: Any) -> None:
        self.aio = function


class _FakeQueue:
    def __init__(self, object_id: str = "qu-test", *, data_capacity: int = 0) -> None:
        self.object_id = object_id
        self._partitions = {
            CONTROL_PARTITION: asyncio.Queue(),
            DATA_PARTITION: asyncio.Queue(maxsize=data_capacity),
        }
        self.put = _AsyncMethod(self._put)
        self.get = _AsyncMethod(self._get)
        self.clear = _AsyncMethod(self._clear)
        self.clear_calls: list[str] = []
        self.put_values: list[tuple[str, Any]] = []
        self.pause_after_control_get = False
        self.control_get_taken = asyncio.Event()
        self.control_get_release = asyncio.Event()

    async def _put(
        self,
        value: Any,
        block: bool = True,
        timeout: float | None = None,  # noqa: ASYNC109
        *,
        partition: str | None = None,
        partition_ttl: int = 86400,
    ) -> None:
        del partition_ttl
        selected = partition or DATA_PARTITION
        self.put_values.append((selected, value))
        queue = self._partitions[selected]
        if not block:
            queue.put_nowait(value)
            return
        if timeout is None:
            await queue.put(value)
        else:
            await asyncio.wait_for(queue.put(value), timeout=timeout)

    async def _get(
        self,
        block: bool = True,
        timeout: float | None = None,  # noqa: ASYNC109
        *,
        partition: str | None = None,
    ) -> Any | None:
        queue = self._partitions[partition or DATA_PARTITION]
        if not block:
            try:
                return queue.get_nowait()
            except asyncio.QueueEmpty as exc:
                raise stdlib_queue.Empty from exc
        if timeout is None:
            value = await queue.get()
        else:
            try:
                value = await asyncio.wait_for(queue.get(), timeout=timeout)
            except TimeoutError as exc:
                raise stdlib_queue.Empty from exc
        if self.pause_after_control_get and partition == CONTROL_PARTITION:
            self.pause_after_control_get = False
            self.control_get_taken.set()
            await self.control_get_release.wait()
        return value

    async def _clear(self, *, partition: str | None = None, all: bool = False) -> None:
        del all
        selected = [partition] if partition is not None else list(self._partitions)
        for name in selected:
            self.clear_calls.append(name)
            queue = self._partitions[name]
            while True:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

    async def control(self, envelope: ControlEnvelope) -> None:
        await self._put(envelope.to_dict(), partition=CONTROL_PARTITION)

    async def data(self, envelope: DataEnvelope) -> None:
        await self._put(envelope.to_dict(), partition=DATA_PARTITION)


class _QueueContext:
    def __init__(self, queue: _FakeQueue) -> None:
        self.queue = queue
        self.exited = False

    async def __aenter__(self) -> _FakeQueue:
        return self.queue

    async def __aexit__(self, *_args: object) -> None:
        self.exited = True


class _Owner:
    def __init__(
        self,
        events: list[dict[str, Any]],
        *,
        block_after_ready: bool = False,
        pause_before_precheck: bool = False,
        first_event_delay: float = 0,
    ) -> None:
        self.events = events
        self.block_after_ready = block_after_ready
        self.first_event_delay = first_event_delay
        self.hydrated = 0
        self.started = 0
        self.finally_count = 0
        self.abort_ids: list[str] = []
        self.start_event = asyncio.Event()
        self.precheck_ready = asyncio.Event()
        self.precheck_release = asyncio.Event()
        if not pause_before_precheck:
            self.precheck_release.set()
        self.release = asyncio.Event()
        self.engine = types.SimpleNamespace(abort=self._abort)

    def _replica_identifier(self) -> str:
        return "replica-1"

    async def _abort(self, generation_id: str) -> None:
        self.abort_ids.append(generation_id)

    async def _stream_generate(
        self, *_args: Any, pre_generate_check: Any = None
    ) -> AsyncIterator[dict]:
        self.hydrated += 1
        self.precheck_ready.set()
        await self.precheck_release.wait()
        await pre_generate_check()
        self.started += 1
        self.start_event.set()
        try:
            if self.first_event_delay:
                await asyncio.sleep(self.first_event_delay)
            for index, event in enumerate(self.events):
                yield event
                if self.block_after_ready and index == 0:
                    await self.release.wait()
        finally:
            self.finally_count += 1


def _control(
    sequence: int,
    *,
    kind: str = "active",
    call_id: str | None = "fc-1",
    deadline: float | None = None,
) -> ControlEnvelope:
    return ControlEnvelope(
        kind=kind,
        generation_id="generation-1",
        invocation_nonce="nonce-1",
        function_call_id=call_id,
        sequence=sequence,
        lease_deadline_unix=deadline or (time.time() + 5),
    )


def _data(
    sequence: int,
    *,
    event: dict[str, Any] | None = None,
    call_id: str = "fc-1",
    replica_id: str = "replica-1",
    generation_id: str = "generation-1",
    nonce: str = "nonce-1",
    version: int = PROTOCOL_VERSION,
) -> dict[str, Any]:
    value = DataEnvelope(
        kind="event",
        generation_id=generation_id,
        invocation_nonce=nonce,
        function_call_id=call_id,
        engine_replica_id=replica_id,
        sequence=sequence,
        terminal=(event or {}).get("type") == "final",
        event=event or {"type": "delta", "text": "x"},
    ).to_dict()
    value["protocol_version"] = version
    return value


def _patch_modal(monkeypatch: pytest.MonkeyPatch, queue: _FakeQueue, call_id: str = "fc-1") -> None:
    monkeypatch.setattr(modal.Queue, "from_id", lambda queue_id: queue)
    monkeypatch.setattr(modal, "current_function_call_id", lambda: call_id)


def test_dispatch_context_suppresses_exit_failure_but_preserves_cancellation() -> None:
    class FailingExitContext:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *_args: object) -> None:
            raise RuntimeError("queue cleanup failed")

    class CancelledExitContext:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *_args: object) -> None:
            raise asyncio.CancelledError

    async def scenario() -> bool:
        context = client._DispatchDeadlineContext(FailingExitContext(), time.time() + 5)
        async with context:
            pass
        cancelled = client._DispatchDeadlineContext(CancelledExitContext(), time.time() + 5)
        with pytest.raises(asyncio.CancelledError):
            async with cancelled:
                pass
        return True

    assert asyncio.run(scenario())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.__setitem__("protocol_version", 2), "version"),
        (lambda value: value.__setitem__("generation_id", "wrong"), "generation"),
        (lambda value: value.__setitem__("invocation_nonce", "wrong"), "nonce"),
        (lambda value: value.__setitem__("function_call_id", "wrong"), "call"),
        (lambda value: value.__setitem__("engine_replica_id", "wrong"), "replica"),
        (lambda value: value.__setitem__("sequence", 2), "sequence"),
    ],
)
def test_data_validator_rejects_identity_version_and_order_failures(
    mutate: Any, message: str
) -> None:
    validator = DataSequenceValidator("generation-1", "nonce-1", "fc-1")
    if message == "replica":
        validator.accept(_data(0))
        value = _data(1)
    else:
        value = _data(0)
    mutate(value)
    with pytest.raises(StreamChannelError, match=message):
        validator.accept(value)


def test_data_validator_rejects_duplicate_gap_replay_and_post_terminal() -> None:
    duplicate = DataSequenceValidator("generation-1", "nonce-1", "fc-1")
    duplicate.accept(_data(0))
    with pytest.raises(StreamChannelError, match="sequence"):
        duplicate.accept(_data(0))

    gap = DataSequenceValidator("generation-1", "nonce-1", "fc-1")
    with pytest.raises(StreamChannelError, match="sequence"):
        gap.accept(_data(1))

    terminal = DataSequenceValidator("generation-1", "nonce-1", "fc-1")
    terminal.accept(_data(0, event={"type": "final", "ok": True}))
    with pytest.raises(StreamChannelError, match="after terminal"):
        terminal.accept(_data(1))


def test_terminal_manifest_must_match_exact_terminal() -> None:
    validator = DataSequenceValidator("generation-1", "nonce-1", "fc-1")
    validator.accept(_data(0, event={"type": "final", "ok": True}))
    manifest = TerminalManifest(
        generation_id="generation-1",
        invocation_nonce="nonce-1",
        function_call_id="fc-1",
        engine_replica_id="replica-1",
        final_sequence=0,
        terminal_kind="event",
        event_count=1,
    ).to_dict()
    validator.reconcile(manifest)
    manifest["event_count"] = 2
    with pytest.raises(StreamChannelError, match="count"):
        validator.reconcile(manifest)


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), float("-inf")])
def test_control_validator_rejects_nonfinite_lease_deadlines(deadline: float) -> None:
    control = _control(0, call_id=None).to_dict()
    control["lease_deadline_unix"] = deadline
    with pytest.raises(StreamChannelError, match="lease deadline"):
        validate_control(control)


@pytest.mark.parametrize("version", [True, 1.0])
def test_protocol_requires_exact_integer_version(version: Any) -> None:
    control = _control(0, call_id=None).to_dict()
    data = _data(0)
    manifest = TerminalManifest(
        "generation-1", "nonce-1", "fc-1", "replica-1", 0, "event", 1
    ).to_dict()
    for value, validator in (
        (control, validate_control),
        (data, validate_data),
        (manifest, validate_manifest),
    ):
        value["protocol_version"] = version
        with pytest.raises(StreamChannelError, match="protocol version"):
            validator(value)


def test_protocol_rejects_credential_fields_and_extra_fields() -> None:
    control = _control(0, call_id=None).to_dict()
    control["api_key"] = "not-allowed"
    with pytest.raises(StreamChannelError, match="fields"):
        validate_control(control)
    data = _data(0)
    data["event"]["authorization"] = "not-allowed"
    with pytest.raises(StreamChannelError, match="credential"):
        validate_data(data)
    manifest = TerminalManifest(
        "generation-1", "nonce-1", "fc-1", "replica-1", 0, "event", 1
    ).to_dict()
    manifest["provider_secret"] = "not-allowed"
    with pytest.raises(StreamChannelError, match="fields"):
        validate_manifest(manifest)


@pytest.mark.parametrize(
    "field",
    [
        "access_token",
        "refresh-token",
        "Bearer Token",
        "bearer_token_value",
        "apiKey",
        "api_secret",
        "authorization",
        "password",
        "SUPABASE_SERVICE_ROLE_KEY",
        "modal_token_secret",
        "hf-token",
        "external_inference_api_key",
    ],
)
def test_protocol_rejects_nested_canonical_credential_aliases(field: str) -> None:
    data = _data(0)
    data["event"] = {"type": "delta", "nested": [{field: "not-allowed"}]}
    with pytest.raises(StreamChannelError, match="credential"):
        validate_data(data)


@pytest.mark.parametrize(
    "field",
    ["token_ids", "prompt_tokens", "completion_tokens", "cached_tokens", "token_count"],
)
def test_protocol_allows_noncredential_token_metrics(field: str) -> None:
    data = _data(0)
    data["event"] = {"type": "delta", "usage": {field: 1}}
    assert validate_data(data).event == data["event"]


def test_control_validator_requires_bound_call_before_protected_work() -> None:
    validator = ControlSequenceValidator("generation-1", "nonce-1", "fc-1")
    validator.accept(_control(0, call_id=None).to_dict())
    with pytest.raises(StreamChannelError, match="call"):
        validator.accept(_control(1, call_id="wrong").to_dict())


def test_queued_cancel_refuses_before_hydration(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> tuple[int, int, list[dict[str, Any]]]:
        queue = _FakeQueue()
        _patch_modal(monkeypatch, queue)
        await queue.control(_control(0, call_id=None))
        await queue.control(_control(1, kind="cancel"))
        owner = _Owner([])
        manifest = await stream_generate_call(
            owner, {}, None, None, "generation-1", time.time() + 5, queue.object_id, "nonce-1"
        )
        terminal = [await queue._get(partition=DATA_PARTITION)]
        return owner.hydrated, owner.started, [manifest, *terminal]

    hydrated, started, output = asyncio.run(scenario())
    assert hydrated == 0
    assert started == 0
    assert output[0]["terminal_kind"] == "error"
    assert output[1]["error_code"] == ChannelErrorCode.CANCELLED


def test_cold_start_admission_drains_fresh_bound_lease_before_expiry_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[int, int, dict[str, Any]]:
        queue = _FakeQueue()
        _patch_modal(monkeypatch, queue)
        await queue.control(_control(0, call_id=None, deadline=time.time() - 1))
        await queue.control(_control(1, deadline=time.time() + 5))
        await queue.control(_control(2, deadline=time.time() + 5))
        owner = _Owner([{"type": "final", "ok": True}])
        manifest = await stream_generate_call(
            owner, {}, None, None, "generation-1", time.time() + 5, queue.object_id, "nonce-1"
        )
        return owner.hydrated, owner.started, manifest

    hydrated, started, manifest = asyncio.run(scenario())
    assert hydrated == 1
    assert started == 1
    assert manifest["terminal_kind"] == "event"


def test_running_lease_watch_drains_fresh_heartbeat_before_expiry_check() -> None:
    async def scenario() -> ChannelErrorCode:
        queue = _FakeQueue()
        watch = channel_engine._LeaseWatch(
            queue,
            generation_id="generation-1",
            invocation_nonce="nonce-1",
            function_call_id="fc-1",
        )
        watch._latest = watch._validator.accept(
            _control(0, call_id=None, deadline=time.time() - 1).to_dict()
        )
        watch._latest = watch._validator.accept(_control(1, deadline=time.time() - 1).to_dict())
        await queue.control(_control(2, deadline=time.time() + 5))
        task = asyncio.create_task(watch._run())
        await asyncio.sleep(0)
        await queue.control(_control(3, kind="cancel"))
        with pytest.raises(StreamChannelError) as exc_info:
            await asyncio.wait_for(task, timeout=1)
        return exc_info.value.code

    assert asyncio.run(scenario()) == ChannelErrorCode.CANCELLED


def test_queued_lease_expiry_refuses_before_hydration(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> tuple[int, int, dict[str, Any]]:
        queue = _FakeQueue()
        _patch_modal(monkeypatch, queue)
        expired = time.time() - 1
        await queue.control(_control(0, call_id=None, deadline=expired))
        await queue.control(_control(1, deadline=expired))
        owner = _Owner([])
        manifest = await stream_generate_call(
            owner, {}, None, None, "generation-1", time.time() + 5, queue.object_id, "nonce-1"
        )
        return owner.hydrated, owner.started, manifest

    hydrated, started, manifest = asyncio.run(scenario())
    assert hydrated == 0
    assert started == 0
    assert manifest["terminal_kind"] == "error"


def test_queued_cancel_arriving_during_hydration_refuses_before_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[int, int]:
        queue = _FakeQueue()
        _patch_modal(monkeypatch, queue)
        await queue.control(_control(0, call_id=None))
        await queue.control(_control(1))
        owner = _Owner([], pause_before_precheck=True)
        task = asyncio.create_task(
            stream_generate_call(
                owner,
                {},
                None,
                None,
                "generation-1",
                time.time() + 5,
                queue.object_id,
                "nonce-1",
            )
        )
        await owner.precheck_ready.wait()
        await queue.control(_control(2, kind="cancel"))
        owner.precheck_release.set()
        await asyncio.wait_for(task, timeout=1)
        return owner.hydrated, owner.started

    assert asyncio.run(scenario()) == (1, 0)


def test_pre_generation_refresh_waits_for_destructively_read_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[int, int]:
        queue = _FakeQueue()
        _patch_modal(monkeypatch, queue)
        await queue.control(_control(0, call_id=None))
        await queue.control(_control(1))
        owner = _Owner([], pause_before_precheck=True)
        task = asyncio.create_task(
            stream_generate_call(
                owner,
                {},
                None,
                None,
                "generation-1",
                time.time() + 5,
                queue.object_id,
                "nonce-1",
            )
        )
        await owner.precheck_ready.wait()
        queue.pause_after_control_get = True
        await queue.control(_control(2, kind="cancel"))
        await queue.control_get_taken.wait()
        owner.precheck_release.set()
        await asyncio.sleep(0)
        assert not task.done()
        queue.control_get_release.set()
        await asyncio.wait_for(task, timeout=1)
        return owner.hydrated, owner.started

    assert asyncio.run(scenario()) == (1, 0)


def test_dispatch_deadline_refuses_before_hydration(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> tuple[int, int, dict[str, Any]]:
        queue = _FakeQueue()
        _patch_modal(monkeypatch, queue)
        await queue.control(_control(0, call_id=None))
        await queue.control(_control(1))
        owner = _Owner([])
        manifest = await stream_generate_call(
            owner, {}, None, None, "generation-1", time.time() - 1, queue.object_id, "nonce-1"
        )
        return owner.hydrated, owner.started, manifest

    hydrated, started, manifest = asyncio.run(scenario())
    assert hydrated == 0
    assert started == 0
    assert manifest["terminal_kind"] == "error"


def test_dispatch_deadline_expiring_during_hydration_cannot_start_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[int, int, ChannelErrorCode]:
        queue = _FakeQueue()
        _patch_modal(monkeypatch, queue)
        await queue.control(_control(0, call_id=None))
        await queue.control(_control(1))
        owner = _Owner([], pause_before_precheck=True)
        task = asyncio.create_task(
            stream_generate_call(
                owner,
                {},
                None,
                None,
                "generation-1",
                time.time() + 0.02,
                queue.object_id,
                "nonce-1",
            )
        )
        await owner.precheck_ready.wait()
        await asyncio.sleep(0.03)
        owner.precheck_release.set()
        manifest = await asyncio.wait_for(task, timeout=1)
        terminal = await queue._get(partition=DATA_PARTITION)
        assert manifest["terminal_kind"] == "error"
        return owner.hydrated, owner.started, terminal["error_code"]

    assert asyncio.run(scenario()) == (1, 0, ChannelErrorCode.DISPATCH_DEADLINE)


def test_running_cancel_closes_and_aborts_exact_generation_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[_Owner, _Owner]:
        victim_queue = _FakeQueue()
        sibling_queue = _FakeQueue("qu-sibling")
        _patch_modal(monkeypatch, victim_queue)
        await victim_queue.control(_control(0, call_id=None))
        await victim_queue.control(_control(1))
        victim = _Owner([{"type": "ready"}], block_after_ready=True)
        task = asyncio.create_task(
            stream_generate_call(
                victim,
                {},
                None,
                None,
                "generation-1",
                time.time() + 5,
                victim_queue.object_id,
                "nonce-1",
            )
        )
        await victim.start_event.wait()
        await victim_queue.control(_control(2, kind="cancel"))
        await asyncio.wait_for(task, timeout=1)

        sibling = _Owner([{"type": "ready"}, {"type": "final", "ok": True}])
        monkeypatch.setattr(modal.Queue, "from_id", lambda queue_id: sibling_queue)
        await sibling_queue.control(_control(0, call_id=None))
        await sibling_queue.control(_control(1))
        await stream_generate_call(
            sibling,
            {},
            None,
            None,
            "generation-1",
            time.time() + 5,
            sibling_queue.object_id,
            "nonce-1",
        )
        return victim, sibling

    victim, sibling = asyncio.run(scenario())
    assert victim.started == 1
    assert victim.abort_ids == ["generation-1"]
    assert victim.finally_count == 1
    assert sibling.abort_ids == []
    assert sibling.finally_count == 1


def test_heartbeat_expiry_aborts_exact_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> _Owner:
        queue = _FakeQueue()
        _patch_modal(monkeypatch, queue)
        deadline = time.time() + 0.35
        await queue.control(_control(0, call_id=None, deadline=deadline))
        await queue.control(_control(1, deadline=deadline))
        owner = _Owner([{"type": "ready"}], block_after_ready=True)
        await asyncio.wait_for(
            stream_generate_call(
                owner,
                {},
                None,
                None,
                "generation-1",
                time.time() + 5,
                queue.object_id,
                "nonce-1",
            ),
            timeout=1,
        )
        return owner

    owner = asyncio.run(scenario())
    assert owner.abort_ids == ["generation-1"]
    assert owner.finally_count == 1


def test_closed_lease_rejects_retained_hydration_after_stream_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[str, int, bool]:
        monkeypatch.setattr(channel_engine, "CLEANUP_SECONDS", 0.02)
        queue = _FakeQueue()
        _patch_modal(monkeypatch, queue)
        await queue.control(_control(0, call_id=None))
        await queue.control(_control(1))
        owner = _Owner([])
        hydration_cancelled = asyncio.Event()
        hydration_release = asyncio.Event()
        hydration_finished = asyncio.Event()
        precheck_rejected = False

        async def delayed_hydration(
            *_args: Any, pre_generate_check: Any = None
        ) -> AsyncIterator[dict[str, Any]]:
            nonlocal precheck_rejected
            owner.hydrated += 1
            owner.precheck_ready.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                hydration_cancelled.set()
                await hydration_release.wait()
            try:
                await pre_generate_check()
            except StreamChannelError:
                precheck_rejected = True
                raise
            finally:
                hydration_finished.set()
            owner.started += 1
            yield {"type": "final", "ok": True}

        owner._stream_generate = delayed_hydration  # type: ignore[method-assign]
        task = asyncio.create_task(
            stream_generate_call(
                owner,
                {},
                None,
                None,
                "generation-1",
                time.time() + 5,
                queue.object_id,
                "nonce-1",
            )
        )
        await owner.precheck_ready.wait()
        await queue.control(_control(2, kind="cancel"))
        await asyncio.wait_for(hydration_cancelled.wait(), timeout=1)
        manifest = await asyncio.wait_for(task, timeout=1)
        hydration_release.set()
        await asyncio.wait_for(hydration_finished.wait(), timeout=1)
        return manifest["terminal_kind"], owner.started, precheck_rejected

    assert asyncio.run(scenario()) == ("error", 0, True)


def test_watchdog_idle_timeout_polls_do_not_fault_before_delayed_first_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[dict[str, Any], _Owner]:
        queue = _FakeQueue()
        _patch_modal(monkeypatch, queue)
        await queue.control(_control(0, call_id=None, deadline=time.time() + 2))
        await queue.control(_control(1, deadline=time.time() + 2))
        owner = _Owner(
            [{"type": "final", "ok": True}],
            first_event_delay=0.35,
        )
        manifest = await stream_generate_call(
            owner, {}, None, None, "generation-1", time.time() + 5, queue.object_id, "nonce-1"
        )
        return manifest, owner

    manifest, owner = asyncio.run(scenario())
    assert manifest["terminal_kind"] == "event"
    assert owner.started == 1
    assert owner.abort_ids == []


def test_guarded_committed_operation_wins_simultaneous_watchdog_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[str, list[tuple[int, bool]], int, bool]:
        queue = _FakeQueue()
        await queue.control(_control(0, call_id=None))
        await queue.control(_control(1))
        watch = channel_engine._LeaseWatch(
            queue,
            generation_id="generation-1",
            invocation_nonce="nonce-1",
            function_call_id="fc-1",
        )
        await watch.admit()
        await watch.close()
        watch._closed_error = None
        committed = asyncio.Event()
        watchdog_done = asyncio.Event()
        writes: list[tuple[int, bool]] = []
        sequence = 0
        terminal = False

        async def fail_watchdog() -> None:
            await committed.wait()
            watchdog_done.set()
            raise StreamChannelError(ChannelErrorCode.CANCELLED, "stream cancelled")

        async def commit_terminal() -> str:
            writes.append((sequence, True))
            committed.set()
            await watchdog_done.wait()
            return "committed"

        watch._task = asyncio.create_task(fail_watchdog())
        original_wait = asyncio.wait

        async def operation_last_wait(tasks: Any, **options: Any) -> Any:
            done, pending = await original_wait(tasks, **options)
            if watch._task in done:
                ordered = [watch._task, *(task for task in done if task is not watch._task)]
                monkeypatch.setattr(channel_engine.asyncio, "wait", original_wait)
                return ordered, pending
            return done, pending

        monkeypatch.setattr(channel_engine.asyncio, "wait", operation_last_wait)
        result = await watch.guarded(commit_terminal())
        sequence += 1
        terminal = True
        await asyncio.sleep(0)
        with pytest.raises(StreamChannelError, match="cancelled"):
            watch.check()
        await watch.close()
        return result, writes, sequence, terminal

    assert asyncio.run(scenario()) == ("committed", [(0, True)], 1, True)


def test_guarded_cancels_invalid_operation_before_awaiting_abort() -> None:
    async def scenario() -> tuple[bool, bool]:
        queue = _FakeQueue()
        watch = channel_engine._LeaseWatch(
            queue,
            generation_id="generation-1",
            invocation_nonce="nonce-1",
            function_call_id="fc-1",
        )
        watch._latest = watch._validator.accept(_control(0, call_id=None).to_dict())
        watch._latest = watch._validator.accept(_control(1).to_dict())
        watch._latest = watch._validator.accept(_control(2, kind="cancel").to_dict())
        operation_started = asyncio.Event()
        abort_observed_operation = False

        async def operation() -> None:
            operation_started.set()

        async def abort() -> None:
            nonlocal abort_observed_operation
            await asyncio.sleep(0)
            abort_observed_operation = operation_started.is_set()

        with pytest.raises(StreamChannelError, match="cancelled"):
            await watch.guarded(operation(), abort=abort)
        return operation_started.is_set(), abort_observed_operation

    assert asyncio.run(scenario()) == (False, False)


def test_guarded_aborts_before_bounded_join_of_noncooperative_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[bool, bool]:
        monkeypatch.setattr(channel_engine, "CLEANUP_SECONDS", 0.02)
        queue = _FakeQueue()
        await queue.control(_control(0, call_id=None))
        await queue.control(_control(1))
        watch = channel_engine._LeaseWatch(
            queue,
            generation_id="generation-1",
            invocation_nonce="nonce-1",
            function_call_id="fc-1",
        )
        await watch.admit()
        operation_started = asyncio.Event()
        release_operation = asyncio.Event()
        operation_finished = asyncio.Event()
        abort_called = asyncio.Event()

        async def noncooperative() -> None:
            operation_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release_operation.wait()
            finally:
                operation_finished.set()

        async def abort() -> None:
            abort_called.set()

        assert watch._task is not None
        watch._task.cancel()
        await asyncio.gather(watch._task, return_exceptions=True)
        fail_watchdog = asyncio.Event()

        async def failed_watchdog() -> None:
            await fail_watchdog.wait()
            raise StreamChannelError(ChannelErrorCode.CANCELLED, "stream cancelled")

        watch._task = asyncio.create_task(failed_watchdog())
        task = asyncio.create_task(watch.guarded(noncooperative(), abort=abort))
        await operation_started.wait()
        fail_watchdog.set()
        with pytest.raises(StreamChannelError, match="stream cancelled"):
            await asyncio.wait_for(task, timeout=0.2)
        state = abort_called.is_set(), operation_finished.is_set()
        release_operation.set()
        await asyncio.sleep(0)
        await watch.close()
        return state

    assert asyncio.run(scenario()) == (True, False)


def test_guarded_abort_timeout_retains_noncooperative_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[bool, bool]:
        monkeypatch.setattr(channel_engine, "CLEANUP_SECONDS", 0.01)
        queue = _FakeQueue()
        await queue.control(_control(0, call_id=None))
        await queue.control(_control(1))
        watch = channel_engine._LeaseWatch(
            queue,
            generation_id="generation-1",
            invocation_nonce="nonce-1",
            function_call_id="fc-1",
        )
        await watch.admit()
        assert watch._task is not None
        await watch.close()
        watch._closed_error = None
        fail_watchdog = asyncio.Event()
        release_abort = asyncio.Event()
        abort_cancelled = asyncio.Event()

        async def failed_watchdog() -> None:
            await fail_watchdog.wait()
            raise StreamChannelError(ChannelErrorCode.CANCELLED, "stream cancelled")

        async def operation() -> None:
            await asyncio.Event().wait()

        async def abort() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                abort_cancelled.set()
                await release_abort.wait()

        watch._task = asyncio.create_task(failed_watchdog())
        task = asyncio.create_task(watch.guarded(operation(), abort=abort))
        fail_watchdog.set()
        with pytest.raises(StreamChannelError, match="stream cancelled"):
            await asyncio.wait_for(task, timeout=0.1)
        await abort_cancelled.wait()
        retained = bool(channel_engine._BACKGROUND_TASKS)
        release_abort.set()
        for retained_task in tuple(channel_engine._BACKGROUND_TASKS):
            await retained_task
        await asyncio.sleep(0)
        await watch.close()
        return retained, not channel_engine._BACKGROUND_TASKS

    assert asyncio.run(scenario()) == (True, True)


def test_unsettled_data_write_does_not_publish_a_competing_terminal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[list[dict[str, Any]], bool]:
        monkeypatch.setattr(channel_engine, "CLEANUP_SECONDS", 0.01)
        queue = _FakeQueue()
        _patch_modal(monkeypatch, queue)
        await queue.control(_control(0, call_id=None))
        await queue.control(_control(1))
        data_put_started = asyncio.Event()
        data_put_cancelled = asyncio.Event()
        release_data_put = asyncio.Event()
        original_put = queue.put.aio

        async def noncooperative_put(
            value: Any,
            block: bool = True,
            timeout: float | None = None,  # noqa: ASYNC109
            *,
            partition: str | None = None,
            partition_ttl: int = 86400,
        ) -> None:
            if partition == DATA_PARTITION:
                data_put_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    data_put_cancelled.set()
                    await release_data_put.wait()
            await original_put(
                value,
                block=block,
                timeout=timeout,
                partition=partition,
                partition_ttl=partition_ttl,
            )

        queue.put.aio = noncooperative_put
        owner = _Owner([{"type": "final", "ok": True}])
        task = asyncio.create_task(
            stream_generate_call(
                owner,
                {},
                None,
                None,
                "generation-1",
                time.time() + 5,
                queue.object_id,
                "nonce-1",
            )
        )
        await data_put_started.wait()
        await queue.control(_control(2, kind="cancel"))
        with pytest.raises(StreamChannelError, match="cancelled"):
            await asyncio.wait_for(task, timeout=0.2)
        await data_put_cancelled.wait()
        returned_before_late_commit = queue._partitions[DATA_PARTITION].empty()
        release_data_put.set()
        for retained_task in tuple(channel_engine._BACKGROUND_TASKS):
            await retained_task
        values = list(queue._partitions[DATA_PARTITION]._queue)
        return values, returned_before_late_commit

    values, returned_before_late_commit = asyncio.run(scenario())
    assert returned_before_late_commit
    assert len(values) == 1
    assert values[0]["kind"] == "event"
    assert values[0]["sequence"] == 0
    assert values[0]["terminal"] is True


def test_data_write_settled_during_abort_does_not_publish_competing_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> list[dict[str, Any]]:
        queue = _FakeQueue()
        _patch_modal(monkeypatch, queue)
        await queue.control(_control(0, call_id=None))
        await queue.control(_control(1))
        data_put_started = asyncio.Event()
        data_put_cancelled = asyncio.Event()
        release_data_put = asyncio.Event()
        data_put_committed = asyncio.Event()
        original_put = queue.put.aio

        async def settling_put(
            value: Any,
            block: bool = True,
            timeout: float | None = None,  # noqa: ASYNC109
            *,
            partition: str | None = None,
            partition_ttl: int = 86400,
        ) -> None:
            if partition == DATA_PARTITION:
                data_put_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    data_put_cancelled.set()
                    await release_data_put.wait()
            await original_put(
                value,
                block=block,
                timeout=timeout,
                partition=partition,
                partition_ttl=partition_ttl,
            )
            if partition == DATA_PARTITION:
                data_put_committed.set()

        queue.put.aio = settling_put
        owner = _Owner([{"type": "final", "ok": True}])

        async def abort(generation_id: str) -> None:
            owner.abort_ids.append(generation_id)
            await data_put_cancelled.wait()
            release_data_put.set()
            await data_put_committed.wait()

        owner.engine.abort = abort
        task = asyncio.create_task(
            stream_generate_call(
                owner,
                {},
                None,
                None,
                "generation-1",
                time.time() + 5,
                queue.object_id,
                "nonce-1",
            )
        )
        await data_put_started.wait()
        await queue.control(_control(2, kind="cancel"))
        with pytest.raises(StreamChannelError, match="cancelled"):
            await asyncio.wait_for(task, timeout=0.2)
        return list(queue._partitions[DATA_PARTITION]._queue)

    values = asyncio.run(scenario())
    assert len(values) == 1
    assert values[0]["kind"] == "event"
    assert values[0]["sequence"] == 0
    assert values[0]["terminal"] is True


def test_backpressure_does_not_starve_control_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> _Owner:
        queue = _FakeQueue(data_capacity=1)
        _patch_modal(monkeypatch, queue)
        await queue.data(
            DataEnvelope(
                kind="event",
                generation_id="fill",
                invocation_nonce="fill",
                function_call_id="fill",
                engine_replica_id="fill",
                sequence=0,
                terminal=False,
                event={"type": "delta"},
            )
        )
        await queue.control(_control(0, call_id=None))
        await queue.control(_control(1))
        owner = _Owner([{"type": "ready"}], block_after_ready=True)
        task = asyncio.create_task(
            stream_generate_call(
                owner,
                {},
                None,
                None,
                "generation-1",
                time.time() + 5,
                queue.object_id,
                "nonce-1",
            )
        )
        await owner.start_event.wait()
        await queue.control(_control(2, kind="cancel"))
        with pytest.raises(StreamChannelError, match="cancelled"):
            await asyncio.wait_for(task, timeout=1)
        return owner

    owner = asyncio.run(scenario())
    assert owner.abort_ids == ["generation-1"]
    assert owner.finally_count == 1


def test_watcher_won_data_write_skips_terminal_error_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[_Owner, bool]:
        queue = _FakeQueue(data_capacity=1)
        _patch_modal(monkeypatch, queue)
        await queue.data(
            DataEnvelope(
                kind="event",
                generation_id="fill",
                invocation_nonce="fill",
                function_call_id="fill",
                engine_replica_id="fill",
                sequence=0,
                terminal=False,
                event={"type": "delta"},
            )
        )
        await queue.control(_control(0, call_id=None))
        await queue.control(_control(1))
        terminal_error_attempted = False
        original_put = queue.put.aio

        async def put(value: Any, **options: Any) -> None:
            nonlocal terminal_error_attempted
            if options.get("partition") == DATA_PARTITION and value.get("kind") == "error":
                terminal_error_attempted = True
            await original_put(value, **options)

        queue.put.aio = put
        owner = _Owner([{"type": "ready"}], block_after_ready=True)
        task = asyncio.create_task(
            stream_generate_call(
                owner,
                {},
                None,
                None,
                "generation-1",
                time.time() + 5,
                queue.object_id,
                "nonce-1",
            )
        )
        await owner.start_event.wait()
        await queue.control(_control(2, kind="cancel"))
        with pytest.raises(StreamChannelError, match="cancelled"):
            await asyncio.wait_for(task, timeout=1)
        return owner, terminal_error_attempted

    owner, terminal_error_attempted = asyncio.run(scenario())
    assert owner.abort_ids == ["generation-1"]
    assert owner.finally_count == 1
    assert not terminal_error_attempted


def test_engine_cleanup_timeout_retains_noncooperative_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[bool, bool]:
        monkeypatch.setattr(channel_engine, "CLEANUP_SECONDS", 0.01)
        release = asyncio.Event()
        cancellation_received = asyncio.Event()

        async def noncooperative_cleanup() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_received.set()
                await release.wait()

        await asyncio.wait_for(
            channel_engine._finish_cleanup(noncooperative_cleanup()),
            timeout=0.1,
        )
        await cancellation_received.wait()
        retained = bool(channel_engine._BACKGROUND_TASKS)
        release.set()
        for task in tuple(channel_engine._BACKGROUND_TASKS):
            await task
        await asyncio.sleep(0)
        return retained, not channel_engine._BACKGROUND_TASKS

    assert asyncio.run(scenario()) == (True, True)


def test_cancellation_during_cleanup_remains_cancelled_and_finishes_abort_and_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[_Owner, bool]:
        queue = _FakeQueue()
        _patch_modal(monkeypatch, queue)
        await queue.control(_control(0, call_id=None))
        await queue.control(_control(1))
        owner = _Owner([{"type": "ready"}], block_after_ready=True)
        close_started = asyncio.Event()
        close_release = asyncio.Event()
        close_finished = asyncio.Event()
        original_close = channel_engine._close_stream

        async def delayed_close(stream: Any) -> None:
            close_started.set()
            await close_release.wait()
            await original_close(stream)
            close_finished.set()

        monkeypatch.setattr(channel_engine, "_close_stream", delayed_close)
        task = asyncio.create_task(
            stream_generate_call(
                owner,
                {},
                None,
                None,
                "generation-1",
                time.time() + 5,
                queue.object_id,
                "nonce-1",
            )
        )
        await owner.start_event.wait()
        await queue.control(_control(2, kind="cancel"))
        await close_started.wait()
        task.cancel()
        close_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        return owner, close_finished.is_set()

    owner, close_finished = asyncio.run(scenario())
    assert close_finished
    assert owner.abort_ids == ["generation-1"]
    assert owner.finally_count == 1


def test_spawn_cancellation_race_obtains_handle_and_cancels() -> None:
    async def scenario() -> tuple[int, int]:
        spawn_started = asyncio.Event()
        release = asyncio.Event()
        cancelled: list[Any] = []
        handle = object()

        async def spawn() -> Any:
            spawn_started.set()
            await release.wait()
            return handle

        async def cancel(value: Any) -> None:
            cancelled.append(value)

        task = asyncio.create_task(
            client._spawn_cancellation_safe(
                spawn,
                cancel,
                dispatch_deadline_unix=time.time() + 5,
            )
        )
        await spawn_started.wait()
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        return len(cancelled), int(cancelled[0] is handle)

    assert asyncio.run(scenario()) == (1, 1)


def test_spawn_cancellation_is_bounded_when_handle_never_arrives() -> None:
    async def scenario() -> tuple[bool, bool]:
        spawn_started = asyncio.Event()
        spawn_finished = asyncio.Event()

        async def spawn() -> Any:
            spawn_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                spawn_finished.set()

        async def cancel(_value: Any) -> None:
            raise AssertionError("no handle exists to cancel")

        task = asyncio.create_task(
            client._spawn_cancellation_safe(
                spawn,
                cancel,
                dispatch_deadline_unix=time.time() + 0.02,
            )
        )
        await spawn_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.2)
        active = any(
            candidate is not asyncio.current_task()
            and not candidate.done()
            and getattr(candidate.get_coro(), "__name__", "") == "spawn"
            for candidate in asyncio.all_tasks()
        )
        return spawn_finished.is_set(), active

    assert asyncio.run(scenario()) == (True, False)


def test_spawn_wait_enforces_dispatch_deadline_without_outer_cancellation() -> None:
    async def scenario() -> tuple[bool, bool]:
        spawn_started = asyncio.Event()
        spawn_finished = asyncio.Event()

        async def spawn() -> Any:
            spawn_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                spawn_finished.set()

        async def cancel(_value: Any) -> None:
            raise AssertionError("no handle exists to cancel")

        task = asyncio.create_task(
            client._spawn_cancellation_safe(
                spawn,
                cancel,
                dispatch_deadline_unix=time.time() + 0.02,
            )
        )
        await spawn_started.wait()
        with pytest.raises(StreamChannelError) as exc_info:
            await asyncio.wait_for(task, timeout=0.2)
        return exc_info.value.code == ChannelErrorCode.DISPATCH_DEADLINE, spawn_finished.is_set()

    assert asyncio.run(scenario()) == (True, True)


def test_spawn_deadline_cancels_handle_returned_during_cleanup() -> None:
    async def scenario() -> tuple[ChannelErrorCode, int, bool]:
        spawn_started = asyncio.Event()
        handle = object()
        cancelled: list[Any] = []

        async def spawn() -> Any:
            spawn_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return handle

        async def cancel(value: Any) -> None:
            cancelled.append(value)

        task = asyncio.create_task(
            client._spawn_cancellation_safe(
                spawn,
                cancel,
                dispatch_deadline_unix=time.time() + 0.02,
            )
        )
        await spawn_started.wait()
        with pytest.raises(StreamChannelError) as exc_info:
            await asyncio.wait_for(task, timeout=0.2)
        return exc_info.value.code, len(cancelled), cancelled[0] is handle

    assert asyncio.run(scenario()) == (ChannelErrorCode.DISPATCH_DEADLINE, 1, True)


def test_deferred_spawn_cancel_retains_task_until_cleanup_finishes() -> None:
    async def scenario() -> tuple[bool, bool]:
        release = asyncio.Event()
        cancelled = asyncio.Event()
        handle = object()

        async def cancel(value: Any) -> None:
            assert value is handle
            cancelled.set()
            await release.wait()

        spawn_task = asyncio.create_task(asyncio.sleep(0, result=handle))
        await spawn_task
        client._schedule_spawn_result_cancel(spawn_task, cancel)
        await cancelled.wait()
        retained = bool(client._BACKGROUND_TASKS)
        release.set()
        for task in tuple(client._BACKGROUND_TASKS):
            await task
        await asyncio.sleep(0)
        return retained, not client._BACKGROUND_TASKS

    assert asyncio.run(scenario()) == (True, True)


def test_deferred_spawn_cancel_does_not_nest_call_result_budget() -> None:
    async def scenario() -> bool:
        handle = object()
        spawn_task = asyncio.create_task(asyncio.sleep(0, result=handle))
        await spawn_task
        cancelled = False

        async def cancel(value: Any) -> None:
            nonlocal cancelled
            assert value is handle
            await asyncio.sleep(0.02)
            cancelled = True

        await asyncio.wait_for(
            client._cancel_spawn_task_result(spawn_task, cancel),
            timeout=0.1,
        )
        return cancelled

    assert asyncio.run(scenario())


def test_spawn_cancellation_cancels_handle_returned_during_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[int, int]:
        monkeypatch.setattr(client, "CLEANUP_SECONDS", 0.1)
        spawn_started = asyncio.Event()
        cancellation_received = asyncio.Event()
        handle = object()
        cancelled: list[Any] = []

        async def spawn() -> Any:
            spawn_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_received.set()
                return handle

        async def cancel(value: Any) -> None:
            cancelled.append(value)

        task = asyncio.create_task(
            client._spawn_cancellation_safe(
                spawn,
                cancel,
                dispatch_deadline_unix=time.time() + 0.01,
            )
        )
        await spawn_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.2)
        await cancellation_received.wait()
        return len(cancelled), int(cancelled[0] is handle)

    assert asyncio.run(scenario()) == (1, 1)


def test_bounded_cancel_timeout_cancels_and_awaits_local_rpc_task() -> None:
    async def scenario() -> tuple[bool, bool]:
        started = asyncio.Event()
        finished = asyncio.Event()

        async def stalled_rpc() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()

        with pytest.raises(TimeoutError):
            await client._bounded_shield(stalled_rpc(), 0.01)
        await started.wait()
        active = any(
            task is not asyncio.current_task()
            and not task.done()
            and getattr(task.get_coro(), "__name__", "") == "stalled_rpc"
            for task in asyncio.all_tasks()
        )
        return finished.is_set(), active

    assert asyncio.run(scenario()) == (True, False)


def test_bounded_cancel_timeout_retains_noncooperative_rpc_until_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[bool, bool]:
        monkeypatch.setattr(client, "CLEANUP_SECONDS", 0.01)
        release = asyncio.Event()
        cancellation_received = asyncio.Event()

        async def noncooperative_rpc() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_received.set()
                await release.wait()

        with pytest.raises(TimeoutError):
            await client._bounded_shield(noncooperative_rpc(), 0.01)
        await cancellation_received.wait()
        retained = bool(client._BACKGROUND_TASKS)
        release.set()
        for task in tuple(client._BACKGROUND_TASKS):
            await task
        await asyncio.sleep(0)
        return retained, not client._BACKGROUND_TASKS

    assert asyncio.run(scenario()) == (True, True)


def test_bounded_cancel_timeout_does_not_join_noncooperative_rpc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[bool, bool]:
        monkeypatch.setattr(client, "CLEANUP_SECONDS", 0.01)
        cancellation_received = asyncio.Event()
        release = asyncio.Event()
        finished = asyncio.Event()

        async def noncooperative_rpc() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_received.set()
                await release.wait()
            finally:
                finished.set()

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                client._bounded_shield(noncooperative_rpc(), 0.01),
                timeout=0.1,
            )
        await cancellation_received.wait()
        returned_before_release = not finished.is_set()
        release.set()
        await asyncio.wait_for(finished.wait(), timeout=0.1)
        return returned_before_release, finished.is_set()

    assert asyncio.run(scenario()) == (True, True)


class _FakeCall:
    def __init__(self, object_id: str, result: Any) -> None:
        self.object_id = object_id
        self.cancel_count = 0
        self.get = _AsyncMethod(self._get)
        self.cancel = _AsyncMethod(self._cancel)
        self._result = result

    async def _get(self) -> Any:
        if inspect.isawaitable(self._result):
            return await self._result
        return self._result

    async def _cancel(self, terminate_containers: bool = False) -> None:
        assert terminate_containers is False
        self.cancel_count += 1


class _FakeSpawnMethod:
    def __init__(self, spawn: Any) -> None:
        self.spawn = _AsyncMethod(spawn)
        self.spawn_count = 0


async def _channel_with_preloaded_data(
    queue: _FakeQueue,
    envelopes: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], _FakeCall, _QueueContext, int]:
    call = _FakeCall("fc-1", manifest)
    method = _FakeSpawnMethod(None)

    async def spawn(*_args: Any) -> _FakeCall:
        method.spawn_count += 1
        for envelope in envelopes:
            await queue._put(envelope, partition=DATA_PARTITION)
        return call

    method.spawn.aio = spawn
    context = _QueueContext(queue)
    channel = client.CancellableStreamChannel(
        spawn_method=method,
        payload_dict={},
        record_dict=None,
        expected_checkpoint=None,
        generation_id="generation-1",
        dispatch_deadline_unix=time.time() + 5,
        invocation_nonce="nonce-1",
        queue_context=lambda: context,
    )
    events = [event async for event in channel]
    return events, call, context, method.spawn_count


def test_router_timed_empty_polls_allow_first_event_after_250ms() -> None:
    async def scenario() -> tuple[list[dict[str, Any]], int]:
        queue = _FakeQueue()
        event = {"type": "final", "ok": True}
        envelope = _data(0, event=event)
        manifest = TerminalManifest(
            "generation-1", "nonce-1", "fc-1", "replica-1", 0, "event", 1
        ).to_dict()
        call = _FakeCall("fc-1", manifest)
        method = _FakeSpawnMethod(None)

        async def publish_later() -> None:
            await asyncio.sleep(0.35)
            await queue._put(envelope, partition=DATA_PARTITION)

        publisher: asyncio.Task[None] | None = None

        async def spawn(*_args: Any) -> _FakeCall:
            nonlocal publisher
            method.spawn_count += 1
            publisher = asyncio.create_task(publish_later())
            return call

        method.spawn.aio = spawn
        channel = client.CancellableStreamChannel(
            spawn_method=method,
            payload_dict={},
            record_dict=None,
            expected_checkpoint=None,
            generation_id="generation-1",
            dispatch_deadline_unix=time.time() + 5,
            invocation_nonce="nonce-1",
            queue_context=lambda: _QueueContext(queue),
        )
        events = [value async for value in channel]
        assert publisher is not None
        await publisher
        return events, call.cancel_count

    events, cancel_count = asyncio.run(scenario())
    assert events == [{"type": "final", "ok": True}]
    assert cancel_count == 0


def test_post_terminal_empty_is_normal_and_final_is_completed() -> None:
    async def scenario() -> tuple[list[dict[str, Any]], int, list[str]]:
        queue = _FakeQueue()
        final = {"type": "final", "ok": True}
        manifest = TerminalManifest(
            "generation-1", "nonce-1", "fc-1", "replica-1", 0, "event", 1
        ).to_dict()
        events, call, _context, _spawn_count = await _channel_with_preloaded_data(
            queue,
            [_data(0, event=final)],
            manifest,
        )
        return events, call.cancel_count, queue.clear_calls

    events, cancel_count, clear_calls = asyncio.run(scenario())
    assert events == [{"type": "final", "ok": True}]
    assert cancel_count == 0
    assert clear_calls == [DATA_PARTITION, CONTROL_PARTITION]


def test_router_channel_preserves_fifo_multi_choice_events_and_cleans_partitions() -> None:
    async def scenario() -> Any:
        queue = _FakeQueue()
        events = [
            {"type": "ready", "request_id": "generation-1"},
            {"type": "delta", "index": 1, "text": "b"},
            {"type": "delta", "index": 0, "text": "a"},
            {"type": "choice_finished", "index": 0, "finish_reason": "stop"},
            {"type": "choice_finished", "index": 1, "finish_reason": "stop"},
            {"type": "final", "choices": [{"index": 0}, {"index": 1}]},
        ]
        envelopes = [_data(index, event=event) for index, event in enumerate(events)]
        manifest = TerminalManifest(
            "generation-1",
            "nonce-1",
            "fc-1",
            "replica-1",
            len(events) - 1,
            "event",
            len(events),
        ).to_dict()
        return await _channel_with_preloaded_data(queue, envelopes, manifest)

    events, call, context, spawn_count = asyncio.run(scenario())
    assert [event["type"] for event in events] == [
        "ready",
        "delta",
        "delta",
        "choice_finished",
        "choice_finished",
        "final",
    ]
    assert [event.get("index") for event in events[1:5]] == [1, 0, 0, 1]
    assert call.cancel_count == 0
    assert context.exited
    assert spawn_count == 1


@pytest.mark.parametrize(
    "bad_envelope",
    [
        _data(1),
        _data(0, call_id="wrong"),
        _data(0, replica_id="replica-1", version=2),
        _data(0, generation_id="wrong"),
        _data(0, nonce="wrong"),
    ],
)
def test_router_channel_faults_fail_closed_cancel_once_and_never_respawn(
    bad_envelope: dict[str, Any],
) -> None:
    async def scenario() -> tuple[int, int]:
        queue = _FakeQueue()
        manifest = TerminalManifest(
            "generation-1", "nonce-1", "fc-1", "replica-1", 0, "event", 1
        ).to_dict()
        call = _FakeCall("fc-1", manifest)
        method = _FakeSpawnMethod(None)

        async def spawn(*_args: Any) -> _FakeCall:
            method.spawn_count += 1
            await queue._put(bad_envelope, partition=DATA_PARTITION)
            return call

        method.spawn.aio = spawn
        channel = client.CancellableStreamChannel(
            spawn_method=method,
            payload_dict={},
            record_dict=None,
            expected_checkpoint=None,
            generation_id="generation-1",
            dispatch_deadline_unix=time.time() + 5,
            invocation_nonce="nonce-1",
            queue_context=lambda: _QueueContext(queue),
        )
        with pytest.raises(StreamChannelError):
            async for _ in channel:
                pass
        return call.cancel_count, method.spawn_count

    assert asyncio.run(scenario()) == (1, 1)


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("Unknown adapter id: adapter-1"),
        ValueError("checkpoint mismatch: expected immutable revision"),
        RuntimeError("remote infrastructure failure"),
    ],
)
def test_router_channel_preserves_original_function_exception(failure: Exception) -> None:
    async def scenario() -> tuple[int, int]:
        queue = _FakeQueue()
        terminal = DataEnvelope(
            kind="error",
            generation_id="generation-1",
            invocation_nonce="nonce-1",
            function_call_id="fc-1",
            engine_replica_id="replica-1",
            sequence=0,
            terminal=True,
            error_code=ChannelErrorCode.ENGINE_ERROR,
        ).to_dict()

        async def failed_result() -> Any:
            raise failure

        call = _FakeCall("fc-1", failed_result())
        method = _FakeSpawnMethod(None)

        async def spawn(*_args: Any) -> _FakeCall:
            method.spawn_count += 1
            await queue._put(terminal, partition=DATA_PARTITION)
            return call

        method.spawn.aio = spawn
        channel = client.CancellableStreamChannel(
            spawn_method=method,
            payload_dict={},
            record_dict=None,
            expected_checkpoint=None,
            generation_id="generation-1",
            dispatch_deadline_unix=time.time() + 5,
            invocation_nonce="nonce-1",
            queue_context=lambda: _QueueContext(queue),
        )
        with pytest.raises(type(failure), match=str(failure)):
            async for _ in channel:
                pass
        return call.cancel_count, method.spawn_count

    assert asyncio.run(scenario()) == (0, 1)


def test_terminal_error_result_timeout_cancels_exact_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[int, list[str]]:
        monkeypatch.setattr(client, "CALL_RESULT_SECONDS", 0.01)
        queue = _FakeQueue()
        terminal = DataEnvelope(
            kind="error",
            generation_id="generation-1",
            invocation_nonce="nonce-1",
            function_call_id="fc-1",
            engine_replica_id="replica-1",
            sequence=0,
            terminal=True,
            error_code=ChannelErrorCode.ENGINE_ERROR,
        ).to_dict()
        call = _FakeCall("fc-1", asyncio.Event().wait())
        method = _FakeSpawnMethod(None)

        async def spawn(*_args: Any) -> _FakeCall:
            method.spawn_count += 1
            await queue._put(terminal, partition=DATA_PARTITION)
            return call

        method.spawn.aio = spawn
        channel = client.CancellableStreamChannel(
            spawn_method=method,
            payload_dict={},
            record_dict=None,
            expected_checkpoint=None,
            generation_id="generation-1",
            dispatch_deadline_unix=time.time() + 5,
            invocation_nonce="nonce-1",
            queue_context=lambda: _QueueContext(queue),
        )
        with pytest.raises(StreamChannelError, match="result timed out"):
            async for _ in channel:
                pass
        return call.cancel_count, queue.clear_calls

    assert asyncio.run(scenario()) == (1, [DATA_PARTITION, CONTROL_PARTITION])


def test_terminal_error_manifest_must_match_envelope() -> None:
    async def scenario() -> int:
        queue = _FakeQueue()
        terminal = DataEnvelope(
            kind="error",
            generation_id="generation-1",
            invocation_nonce="nonce-1",
            function_call_id="fc-1",
            engine_replica_id="replica-1",
            sequence=0,
            terminal=True,
            error_code=ChannelErrorCode.CANCELLED,
        ).to_dict()
        manifest = TerminalManifest(
            "generation-1", "nonce-1", "fc-1", "wrong-replica", 0, "error", 1
        ).to_dict()
        call = _FakeCall("fc-1", manifest)
        method = _FakeSpawnMethod(None)

        async def spawn(*_args: Any) -> _FakeCall:
            await queue._put(terminal, partition=DATA_PARTITION)
            return call

        method.spawn.aio = spawn
        channel = client.CancellableStreamChannel(
            spawn_method=method,
            payload_dict={},
            record_dict=None,
            expected_checkpoint=None,
            generation_id="generation-1",
            dispatch_deadline_unix=time.time() + 5,
            invocation_nonce="nonce-1",
            queue_context=lambda: _QueueContext(queue),
        )
        with pytest.raises(StreamChannelError, match="manifest"):
            async for _ in channel:
                pass
        return call.cancel_count

    assert asyncio.run(scenario()) == 1


def test_router_channel_rejects_post_terminal_data_without_respawn() -> None:
    async def scenario() -> tuple[int, int]:
        queue = _FakeQueue()
        terminal = _data(0, event={"type": "final", "ok": True})
        replay = _data(1)
        manifest = TerminalManifest(
            "generation-1", "nonce-1", "fc-1", "replica-1", 0, "event", 1
        ).to_dict()
        call = _FakeCall("fc-1", manifest)
        method = _FakeSpawnMethod(None)

        async def spawn(*_args: Any) -> _FakeCall:
            method.spawn_count += 1
            await queue._put(terminal, partition=DATA_PARTITION)
            await queue._put(replay, partition=DATA_PARTITION)
            return call

        method.spawn.aio = spawn
        channel = client.CancellableStreamChannel(
            spawn_method=method,
            payload_dict={},
            record_dict=None,
            expected_checkpoint=None,
            generation_id="generation-1",
            dispatch_deadline_unix=time.time() + 5,
            invocation_nonce="nonce-1",
            queue_context=lambda: _QueueContext(queue),
        )
        with pytest.raises(StreamChannelError, match="after terminal"):
            async for _ in channel:
                pass
        return call.cancel_count, method.spawn_count

    assert asyncio.run(scenario()) == (1, 1)


def test_router_channel_rejects_manifest_mismatch_without_respawn() -> None:
    async def scenario() -> tuple[int, int]:
        queue = _FakeQueue()
        envelope = _data(0, event={"type": "final", "ok": True})
        manifest = TerminalManifest(
            "generation-1", "nonce-1", "fc-1", "replica-1", 0, "event", 1
        ).to_dict()
        manifest["engine_replica_id"] = "wrong"
        call = _FakeCall("fc-1", manifest)
        method = _FakeSpawnMethod(None)

        async def spawn(*_args: Any) -> _FakeCall:
            method.spawn_count += 1
            await queue._put(envelope, partition=DATA_PARTITION)
            return call

        method.spawn.aio = spawn
        channel = client.CancellableStreamChannel(
            spawn_method=method,
            payload_dict={},
            record_dict=None,
            expected_checkpoint=None,
            generation_id="generation-1",
            dispatch_deadline_unix=time.time() + 5,
            invocation_nonce="nonce-1",
            queue_context=lambda: _QueueContext(queue),
        )
        with pytest.raises(StreamChannelError, match="manifest"):
            async for _ in channel:
                pass
        return call.cancel_count, method.spawn_count

    assert asyncio.run(scenario()) == (1, 1)


def test_manifest_wait_covers_sequential_remote_cleanup_budget() -> None:
    async def scenario() -> tuple[list[dict[str, Any]], int]:
        queue = _FakeQueue()
        final = {"type": "final", "ok": True}
        manifest = TerminalManifest(
            "generation-1", "nonce-1", "fc-1", "replica-1", 0, "event", 1
        ).to_dict()
        cleanup_budget = ENGINE_CLEANUP_STEPS * ENGINE_CLEANUP_WAITS_PER_STEP * CLEANUP_SECONDS
        delay = cleanup_budget + 0.05
        assert cleanup_budget + CALL_RESULT_MARGIN_SECONDS == CALL_RESULT_SECONDS
        assert delay < CALL_RESULT_SECONDS

        async def delayed_manifest() -> dict[str, Any]:
            await asyncio.sleep(delay)
            return manifest

        call = _FakeCall("fc-1", delayed_manifest())
        method = _FakeSpawnMethod(None)

        async def spawn(*_args: Any) -> _FakeCall:
            method.spawn_count += 1
            await queue._put(_data(0, event=final), partition=DATA_PARTITION)
            return call

        method.spawn.aio = spawn
        channel = client.CancellableStreamChannel(
            spawn_method=method,
            payload_dict={},
            record_dict=None,
            expected_checkpoint=None,
            generation_id="generation-1",
            dispatch_deadline_unix=time.time() + 5,
            invocation_nonce="nonce-1",
            queue_context=lambda: _QueueContext(queue),
        )
        return [event async for event in channel], call.cancel_count

    events, cancel_count = asyncio.run(scenario())
    assert events == [{"type": "final", "ok": True}]
    assert cancel_count == 0


def test_function_success_gets_bounded_terminal_drain(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> tuple[int, int]:
        monkeypatch.setattr(client, "TERMINAL_DRAIN_SECONDS", 0.03)
        monkeypatch.setattr(client, "DATA_GET_SECONDS", 0.005)
        queue = _FakeQueue()
        manifest = TerminalManifest(
            "generation-1", "nonce-1", "fc-1", "replica-1", 0, "event", 1
        ).to_dict()
        call = _FakeCall("fc-1", manifest)
        method = _FakeSpawnMethod(None)

        async def spawn(*_args: Any) -> _FakeCall:
            method.spawn_count += 1
            return call

        method.spawn.aio = spawn
        channel = client.CancellableStreamChannel(
            spawn_method=method,
            payload_dict={},
            record_dict=None,
            expected_checkpoint=None,
            generation_id="generation-1",
            dispatch_deadline_unix=time.time() + 5,
            invocation_nonce="nonce-1",
            queue_context=lambda: _QueueContext(queue),
        )
        with pytest.raises(StreamChannelError, match="before terminal"):
            async for _ in channel:
                pass
        return call.cancel_count, method.spawn_count

    assert asyncio.run(scenario()) == (1, 1)


def test_terminal_drain_deadline_resets_after_each_buffered_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> list[dict[str, Any]]:
        monkeypatch.setattr(client, "TERMINAL_DRAIN_SECONDS", 0.01)
        queue = _FakeQueue()
        events = [
            {"type": "delta", "text": "a"},
            {"type": "delta", "text": "b"},
            {"type": "final", "ok": True},
        ]
        manifest = TerminalManifest(
            "generation-1", "nonce-1", "fc-1", "replica-1", 2, "event", 3
        ).to_dict()
        call = _FakeCall("fc-1", manifest)
        method = _FakeSpawnMethod(None)

        async def spawn(*_args: Any) -> _FakeCall:
            for index, event in enumerate(events):
                await queue._put(_data(index, event=event), partition=DATA_PARTITION)
            return call

        method.spawn.aio = spawn
        channel = client.CancellableStreamChannel(
            spawn_method=method,
            payload_dict={},
            record_dict=None,
            expected_checkpoint=None,
            generation_id="generation-1",
            dispatch_deadline_unix=time.time() + 5,
            invocation_nonce="nonce-1",
            queue_context=lambda: _QueueContext(queue),
        )
        observed = [await anext(channel), await anext(channel)]
        await asyncio.sleep(0.03)
        observed.append(await anext(channel))
        with pytest.raises(StopAsyncIteration):
            await anext(channel)
        return observed

    assert asyncio.run(scenario()) == [
        {"type": "delta", "text": "a"},
        {"type": "delta", "text": "b"},
        {"type": "final", "ok": True},
    ]


def test_channel_is_direct_async_iterator_with_one_spawn_and_idempotent_close() -> None:
    async def scenario() -> tuple[dict[str, Any], int, int, list[str], bool]:
        queue = _FakeQueue()
        pending = asyncio.Event()
        pending_result = asyncio.create_task(pending.wait())
        call = _FakeCall("fc-1", pending_result)
        method = _FakeSpawnMethod(None)
        deadline = time.time() + 5

        async def spawn(*args: Any) -> _FakeCall:
            method.spawn_count += 1
            assert args[4] == deadline
            await queue._put(_data(0, event={"type": "ready"}), partition=DATA_PARTITION)
            return call

        method.spawn.aio = spawn
        channel = client.CancellableStreamChannel(
            spawn_method=method,
            payload_dict={},
            record_dict=None,
            expected_checkpoint=None,
            generation_id="generation-1",
            dispatch_deadline_unix=deadline,
            invocation_nonce="nonce-1",
            queue_context=lambda: _QueueContext(queue),
        )
        assert channel.__aiter__() is channel
        event = await anext(channel)
        await channel.aclose()
        await channel.aclose()
        with pytest.raises(StopAsyncIteration):
            await anext(channel)
        return event, method.spawn_count, call.cancel_count, queue.clear_calls, channel._closed

    event, spawn_count, cancel_count, clear_calls, closed = asyncio.run(scenario())
    assert event == {"type": "ready"}
    assert spawn_count == 1
    assert cancel_count == 1
    assert clear_calls == [DATA_PARTITION, CONTROL_PARTITION]
    assert closed


def test_cancelled_anext_runs_channel_cleanup_before_idempotent_close() -> None:
    async def scenario() -> tuple[int, list[str], int]:
        queue = _FakeQueue()
        pending = asyncio.Event()
        pending_result = asyncio.create_task(pending.wait())
        call = _FakeCall("fc-1", pending_result)
        method = _FakeSpawnMethod(None)
        spawn_ready = asyncio.Event()

        async def spawn(*_args: Any) -> _FakeCall:
            method.spawn_count += 1
            spawn_ready.set()
            return call

        method.spawn.aio = spawn
        channel = client.CancellableStreamChannel(
            spawn_method=method,
            payload_dict={},
            record_dict=None,
            expected_checkpoint=None,
            generation_id="generation-1",
            dispatch_deadline_unix=time.time() + 5,
            invocation_nonce="nonce-1",
            queue_context=lambda: _QueueContext(queue),
        )
        next_task = asyncio.create_task(anext(channel))
        await spawn_ready.wait()
        await asyncio.sleep(0)
        next_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await next_task
        cancel_count_after_anext = call.cancel_count
        await channel.aclose()
        return cancel_count_after_anext, queue.clear_calls, call.cancel_count

    cancel_count, clear_calls, final_cancel_count = asyncio.run(scenario())
    assert cancel_count == 1
    assert final_cancel_count == 1
    assert clear_calls == [DATA_PARTITION, CONTROL_PARTITION]


def test_iterator_close_cancels_exact_call_before_stalled_control_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[int, bool]:
        monkeypatch.setattr(client, "CLEANUP_SECONDS", 0.01)
        queue = _FakeQueue()
        stalled_cancel = asyncio.Event()
        release_cancel = asyncio.Event()
        original_put = queue.put.aio

        async def put(value: Any, **options: Any) -> None:
            if options.get("partition") == CONTROL_PARTITION and value.get("kind") == "cancel":
                stalled_cancel.set()
                await release_cancel.wait()
            else:
                await original_put(value, **options)

        queue.put.aio = put
        pending_result = asyncio.create_task(asyncio.Event().wait())
        call = _FakeCall("fc-1", pending_result)
        method = _FakeSpawnMethod(None)

        async def spawn(*_args: Any) -> _FakeCall:
            await queue._put(_data(0, event={"type": "ready"}), partition=DATA_PARTITION)
            return call

        method.spawn.aio = spawn
        channel = client.CancellableStreamChannel(
            spawn_method=method,
            payload_dict={},
            record_dict=None,
            expected_checkpoint=None,
            generation_id="generation-1",
            dispatch_deadline_unix=time.time() + 5,
            invocation_nonce="nonce-1",
            queue_context=lambda: _QueueContext(queue),
        )
        await anext(channel)
        await asyncio.wait_for(channel.aclose(), timeout=0.2)
        observed = stalled_cancel.is_set()
        release_cancel.set()
        for task in tuple(client._BACKGROUND_TASKS):
            await task
        return call.cancel_count, observed

    assert asyncio.run(scenario()) == (1, True)


def test_router_cleanup_bounds_noncooperative_call_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[bool, bool]:
        monkeypatch.setattr(client, "CLEANUP_SECONDS", 0.01)
        queue = _FakeQueue()
        release = asyncio.Event()
        cancellation_received = asyncio.Event()

        async def noncooperative_result() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_received.set()
                await release.wait()

        result_task = asyncio.create_task(noncooperative_result())
        call = _FakeCall("fc-1", result_task)
        method = _FakeSpawnMethod(None)

        async def spawn(*_args: Any) -> _FakeCall:
            await queue._put(_data(0, event={"type": "ready"}), partition=DATA_PARTITION)
            return call

        method.spawn.aio = spawn
        channel = client.CancellableStreamChannel(
            spawn_method=method,
            payload_dict={},
            record_dict=None,
            expected_checkpoint=None,
            generation_id="generation-1",
            dispatch_deadline_unix=time.time() + 5,
            invocation_nonce="nonce-1",
            queue_context=lambda: _QueueContext(queue),
        )
        await anext(channel)
        await asyncio.wait_for(channel.aclose(), timeout=0.2)
        await cancellation_received.wait()
        retained = bool(client._BACKGROUND_TASKS)
        release.set()
        for task in tuple(client._BACKGROUND_TASKS):
            await task
        await asyncio.sleep(0)
        return retained, not client._BACKGROUND_TASKS

    assert asyncio.run(scenario()) == (True, True)


def test_iterator_close_sends_tombstone_and_exact_call_cancel() -> None:
    async def scenario() -> tuple[int, list[str]]:
        queue = _FakeQueue()
        pending = asyncio.Event()
        pending_result = asyncio.create_task(pending.wait())
        call = _FakeCall("fc-1", pending_result)
        method = _FakeSpawnMethod(None)

        async def spawn(*_args: Any) -> _FakeCall:
            method.spawn_count += 1
            await queue._put(_data(0, event={"type": "ready"}), partition=DATA_PARTITION)
            return call

        method.spawn.aio = spawn
        channel = client.CancellableStreamChannel(
            spawn_method=method,
            payload_dict={},
            record_dict=None,
            expected_checkpoint=None,
            generation_id="generation-1",
            dispatch_deadline_unix=time.time() + 5,
            invocation_nonce="nonce-1",
            queue_context=lambda: _QueueContext(queue),
        )
        iterator = channel.__aiter__()
        assert (await anext(iterator))["type"] == "ready"
        await iterator.aclose()
        return call.cancel_count, queue.clear_calls

    cancel_count, clear_calls = asyncio.run(scenario())
    assert cancel_count == 1
    assert clear_calls == [DATA_PARTITION, CONTROL_PARTITION]


def test_dispatch_deadline_bounds_channel_setup_and_closes_late_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[bool, bool]:
        monkeypatch.setattr(client, "CLEANUP_SECONDS", 0.01)
        queue = _FakeQueue()
        entered = asyncio.Event()
        release = asyncio.Event()
        exited = asyncio.Event()

        class SlowContext:
            async def __aenter__(self) -> _FakeQueue:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    entered.set()
                    await release.wait()
                    return queue

            async def __aexit__(self, *_args: object) -> None:
                exited.set()

        method = _FakeSpawnMethod(None)
        channel = client.CancellableStreamChannel(
            spawn_method=method,
            payload_dict={},
            record_dict=None,
            expected_checkpoint=None,
            generation_id="generation-1",
            dispatch_deadline_unix=time.time() + 0.01,
            invocation_nonce="nonce-1",
            queue_context=SlowContext,
        )
        with pytest.raises(StreamChannelError) as exc_info:
            await asyncio.wait_for(anext(channel), timeout=0.2)
        await entered.wait()
        release.set()
        await asyncio.wait_for(exited.wait(), timeout=0.2)
        return exc_info.value.code == ChannelErrorCode.DISPATCH_DEADLINE, method.spawn_count == 0

    assert asyncio.run(scenario()) == (True, True)


def test_cancelled_channel_setup_closes_late_ephemeral_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[bool, bool]:
        monkeypatch.setattr(client, "CLEANUP_SECONDS", 0.01)
        queue = _FakeQueue()
        enter_started = asyncio.Event()
        cancellation_received = asyncio.Event()
        release = asyncio.Event()
        exited = asyncio.Event()

        class SlowContext:
            async def __aenter__(self) -> _FakeQueue:
                enter_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancellation_received.set()
                    await release.wait()
                    return queue

            async def __aexit__(self, *_args: object) -> None:
                exited.set()

        method = _FakeSpawnMethod(None)
        channel = client.CancellableStreamChannel(
            spawn_method=method,
            payload_dict={},
            record_dict=None,
            expected_checkpoint=None,
            generation_id="generation-1",
            dispatch_deadline_unix=time.time() + 5,
            invocation_nonce="nonce-1",
            queue_context=SlowContext,
        )
        task = asyncio.create_task(anext(channel))
        await enter_started.wait()
        task.cancel()
        await cancellation_received.wait()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.2)
        retained = bool(client._BACKGROUND_TASKS)
        release.set()
        await asyncio.wait_for(exited.wait(), timeout=0.2)
        for retained_task in tuple(client._BACKGROUND_TASKS):
            await retained_task
        await asyncio.sleep(0)
        return retained, not client._BACKGROUND_TASKS

    assert asyncio.run(scenario()) == (True, True)


def test_dispatch_deadline_bounds_initial_lease_before_spawn() -> None:
    async def scenario() -> tuple[ChannelErrorCode, int]:
        queue = _FakeQueue()
        put_started = asyncio.Event()

        async def stalled_put(*_args: Any, **_options: Any) -> None:
            put_started.set()
            await asyncio.Event().wait()

        queue.put.aio = stalled_put
        method = _FakeSpawnMethod(None)
        channel = client.CancellableStreamChannel(
            spawn_method=method,
            payload_dict={},
            record_dict=None,
            expected_checkpoint=None,
            generation_id="generation-1",
            dispatch_deadline_unix=time.time() + 0.01,
            invocation_nonce="nonce-1",
            queue_context=lambda: _QueueContext(queue),
        )
        task = asyncio.create_task(anext(channel))
        await put_started.wait()
        with pytest.raises(StreamChannelError) as exc_info:
            await asyncio.wait_for(task, timeout=0.2)
        return exc_info.value.code, method.spawn_count

    assert asyncio.run(scenario()) == (ChannelErrorCode.DISPATCH_DEADLINE, 0)


def test_dispatch_deadline_bounds_bound_lease_and_cancels_spawned_call() -> None:
    async def scenario() -> tuple[ChannelErrorCode, int]:
        queue = _FakeQueue()
        original_put = queue.put.aio
        put_count = 0
        bound_put_started = asyncio.Event()

        async def put(value: Any, **options: Any) -> None:
            nonlocal put_count
            put_count += 1
            if put_count == 2:
                bound_put_started.set()
                await asyncio.Event().wait()
            await original_put(value, **options)

        queue.put.aio = put
        pending_result = asyncio.create_task(asyncio.Event().wait())
        call = _FakeCall("fc-1", pending_result)
        method = _FakeSpawnMethod(None)

        async def spawn(*_args: Any) -> _FakeCall:
            method.spawn_count += 1
            return call

        method.spawn.aio = spawn
        channel = client.CancellableStreamChannel(
            spawn_method=method,
            payload_dict={},
            record_dict=None,
            expected_checkpoint=None,
            generation_id="generation-1",
            dispatch_deadline_unix=time.time() + 0.03,
            invocation_nonce="nonce-1",
            queue_context=lambda: _QueueContext(queue),
        )
        task = asyncio.create_task(anext(channel))
        await bound_put_started.wait()
        with pytest.raises(StreamChannelError) as exc_info:
            await asyncio.wait_for(task, timeout=0.2)
        return exc_info.value.code, call.cancel_count

    assert asyncio.run(scenario()) == (ChannelErrorCode.DISPATCH_DEADLINE, 1)


def test_router_data_poll_cancellation_reaches_exact_call_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[int, bool, bool]:
        monkeypatch.setattr(client, "CLEANUP_SECONDS", 0.01)
        queue = _FakeQueue()
        get_started = asyncio.Event()
        cancellation_received = asyncio.Event()
        release = asyncio.Event()

        async def stalled_get(*_args: Any, **_options: Any) -> Any:
            get_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_received.set()
                await release.wait()

        queue.get.aio = stalled_get
        pending_result = asyncio.create_task(asyncio.Event().wait())
        call = _FakeCall("fc-1", pending_result)
        method = _FakeSpawnMethod(None)

        async def spawn(*_args: Any) -> _FakeCall:
            method.spawn_count += 1
            return call

        method.spawn.aio = spawn
        channel = client.CancellableStreamChannel(
            spawn_method=method,
            payload_dict={},
            record_dict=None,
            expected_checkpoint=None,
            generation_id="generation-1",
            dispatch_deadline_unix=time.time() + 5,
            invocation_nonce="nonce-1",
            queue_context=lambda: _QueueContext(queue),
        )
        task = asyncio.create_task(anext(channel))
        await get_started.wait()
        task.cancel()
        await cancellation_received.wait()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.2)
        retained = bool(client._BACKGROUND_TASKS)
        release.set()
        for retained_task in tuple(client._BACKGROUND_TASKS):
            await retained_task
        await asyncio.sleep(0)
        return call.cancel_count, retained, not client._BACKGROUND_TASKS

    assert asyncio.run(scenario()) == (1, True, True)


def test_engine_control_read_stall_fails_closed_on_local_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[ChannelErrorCode, bool, bool]:
        monkeypatch.setattr(channel_engine, "CLEANUP_SECONDS", 0.01)
        monkeypatch.setattr(channel_engine, "CONTROL_POLL_SECONDS", 0.01)
        queue = _FakeQueue()
        read_started = asyncio.Event()
        cancellation_received = asyncio.Event()
        release = asyncio.Event()

        async def stalled_get(*_args: Any, **_options: Any) -> Any:
            read_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_received.set()
                await release.wait()

        queue.get.aio = stalled_get
        watch = channel_engine._LeaseWatch(
            queue,
            generation_id="generation-1",
            invocation_nonce="nonce-1",
            function_call_id="fc-1",
        )
        watch._latest = watch._validator.accept(_control(0, call_id=None).to_dict())
        watch._task = asyncio.create_task(watch._run())
        await read_started.wait()
        with pytest.raises(StreamChannelError) as exc_info:
            await asyncio.wait_for(watch._task, timeout=0.2)
        await cancellation_received.wait()
        retained = bool(channel_engine._BACKGROUND_TASKS)
        release.set()
        for retained_task in tuple(channel_engine._BACKGROUND_TASKS):
            await retained_task
        await asyncio.sleep(0)
        return exc_info.value.code, retained, not channel_engine._BACKGROUND_TASKS

    assert asyncio.run(scenario()) == (ChannelErrorCode.CHANNEL_FAULT, True, True)


def test_router_context_exit_is_bounded_and_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[list[dict[str, Any]], bool, bool]:
        monkeypatch.setattr(client, "CLEANUP_SECONDS", 0.01)
        queue = _FakeQueue()
        cancellation_received = asyncio.Event()
        release = asyncio.Event()

        class SlowExitContext:
            async def __aenter__(self) -> _FakeQueue:
                return queue

            async def __aexit__(self, *_args: object) -> None:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancellation_received.set()
                    await release.wait()

        final = {"type": "final", "ok": True}
        manifest = TerminalManifest(
            "generation-1", "nonce-1", "fc-1", "replica-1", 0, "event", 1
        ).to_dict()
        call = _FakeCall("fc-1", manifest)
        method = _FakeSpawnMethod(None)

        async def spawn(*_args: Any) -> _FakeCall:
            method.spawn_count += 1
            await queue._put(_data(0, event=final), partition=DATA_PARTITION)
            return call

        method.spawn.aio = spawn
        channel = client.CancellableStreamChannel(
            spawn_method=method,
            payload_dict={},
            record_dict=None,
            expected_checkpoint=None,
            generation_id="generation-1",
            dispatch_deadline_unix=time.time() + 5,
            invocation_nonce="nonce-1",
            queue_context=SlowExitContext,
        )
        events = [event async for event in channel]
        await cancellation_received.wait()
        retained = bool(client._BACKGROUND_TASKS)
        release.set()
        for retained_task in tuple(client._BACKGROUND_TASKS):
            await retained_task
        await asyncio.sleep(0)
        return events, retained, not client._BACKGROUND_TASKS

    assert asyncio.run(scenario()) == ([{"type": "final", "ok": True}], True, True)


def test_router_partition_clear_retains_noncooperative_rpc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[bool, bool]:
        monkeypatch.setattr(client, "CLEANUP_SECONDS", 0.01)
        queue = _FakeQueue()
        cancellation_received = asyncio.Event()
        release = asyncio.Event()

        async def stalled_clear(*, partition: str | None = None) -> None:
            del partition
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_received.set()
                await release.wait()

        queue.clear.aio = stalled_clear
        await asyncio.wait_for(
            client._clear_partition(queue, DATA_PARTITION),
            timeout=0.1,
        )
        await cancellation_received.wait()
        retained = bool(client._BACKGROUND_TASKS)
        release.set()
        for task in tuple(client._BACKGROUND_TASKS):
            await task
        await asyncio.sleep(0)
        return retained, not client._BACKGROUND_TASKS

    assert asyncio.run(scenario()) == (True, True)


def test_channel_writes_only_protocol_envelopes_without_credentials() -> None:
    async def scenario() -> list[tuple[str, Any]]:
        queue = _FakeQueue()
        events = [
            {"type": "ready", "request_id": "generation-1"},
            {"type": "final", "ok": True},
        ]
        envelopes = [_data(index, event=event) for index, event in enumerate(events)]
        manifest = TerminalManifest(
            "generation-1", "nonce-1", "fc-1", "replica-1", 1, "event", 2
        ).to_dict()
        await _channel_with_preloaded_data(queue, envelopes, manifest)
        return queue.put_values

    values = asyncio.run(scenario())
    assert values
    for partition, value in values:
        if partition == CONTROL_PARTITION:
            validate_control(value)
        else:
            validate_data(value)
