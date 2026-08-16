"""Exercise the generated Modal app with Modal and the GPU engine stubbed."""

from __future__ import annotations

import ast
import asyncio
import copy
import json
import sys
import threading
import types

import pytest

from flash.content.multimodal import _IMAGE_BLOCK_TYPES
from flash.core.catalog import MODELS
from flash.serve.backend.generate import render_app

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

REVISION = "run-abc@step-10." + "a" * 40
SECOND_REVISION = "run-abc@step-20." + "b" * 40
RUN_ID = "run-abc"
BASE_MODEL = "Qwen/Qwen3.5-4B"
_MESSAGES = [{"role": "user", "content": "hi"}]
BAD_REPO = "bad/repo"
REGISTRATION = {
    "adapter_id": REVISION,
    "repo_id": "acme/artifacts",
    "base_model": BASE_MODEL,
    "subfolder": "sft/run-abc/adapter",
    "repo_type": "dataset",
    "checkpoint": "run-abc/step-10",
    "metadata": {
        "record_type": "revision",
        "run_id": RUN_ID,
        "checkpoint_step": 10,
        "hf_revision": "a" * 40,
        "settle_attempt": "caller-attempt",
    },
    "thinking": False,
}


class _Aio:
    def __init__(self, fn):
        self._fn = fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    async def aio(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeVolume:
    def __init__(self):
        self.remove_calls = []
        self.remove_failures = {}

    @classmethod
    def from_name(cls, *args, **kwargs):
        return cls()

    def _remove_file(self, path, *, recursive=False):
        self.remove_calls.append((path, recursive))
        failure = self.remove_failures.get(path)
        if failure is not None:
            raise failure

    @property
    def remove_file(self):
        return _Aio(self._remove_file)


class _FakeDict(dict):
    @classmethod
    def from_name(cls, *args, **kwargs):
        return cls()

    def _put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        super().__setitem__(key, copy.deepcopy(value))
        return True

    def _get(self, key, default=None):
        return copy.deepcopy(super().get(key, default))

    @property
    def put(self):
        return _Aio(self._put)

    @property
    def get(self):
        return _Aio(self._get)


def _run_awaitable(awaitable):
    result = []
    errors = []

    def target():
        try:
            result.append(asyncio.run(awaitable))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return result[0] if result else None


class _CoordinatorGate:
    def __init__(self):
        self.serial = threading.Lock()
        self._condition = threading.Condition()
        self.submissions = []

    def submit(self, method_name):
        with self._condition:
            self.submissions.append(method_name)
            self._condition.notify_all()

    def wait_for(self, method_name, count=1, timeout=2):
        with self._condition:
            return self._condition.wait_for(
                lambda: self.submissions.count(method_name) >= count,
                timeout=timeout,
            )


def _stub_modal(
    monkeypatch,
    engine_methods,
    spawned,
    spawn_queue,
    coordinator_gate,
    engine_classes,
    cls_options,
    fn_options,
):
    modal = types.ModuleType("modal")

    class _Named:
        @classmethod
        def from_name(cls, *args, **kwargs):
            return cls()

    class _Image(_Named):
        @classmethod
        def from_registry(cls, *args, **kwargs):
            return cls()

        def _chain(self, *args, **kwargs):
            return self

        apt_install = pip_install = env = _chain

    class _ClassMethod:
        def __init__(self, class_name, instance, name):
            self._class_name = class_name
            self._instance = instance
            self._name = name
            self.remote = types.SimpleNamespace(aio=self._call)
            self.remote_gen = types.SimpleNamespace(aio=self._generate)

        def _target(self):
            if self._class_name == "Engine":
                return engine_methods[self._name]
            return getattr(self._instance, self._name)

        async def _call(self, *args, **kwargs):
            if self._class_name != "LifecycleCoordinator":
                return await self._target()(*args, **kwargs)
            coordinator_gate.submit(self._name)
            await asyncio.to_thread(coordinator_gate.serial.acquire)
            try:
                return await self._target()(*args, **kwargs)
            finally:
                coordinator_gate.serial.release()

        async def _generate(self, *args, **kwargs):
            async for item in self._target()(*args, **kwargs):
                yield item

        def spawn(self, *args, **kwargs):
            copied_args = copy.deepcopy(args)
            copied_kwargs = copy.deepcopy(kwargs)
            spawned.append((f"{self._class_name}.{self._name}", copied_args, copied_kwargs))
            spawn_queue.append((self, copied_args, copied_kwargs))

    class _ClassHandle:
        def __init__(self, class_name, instance):
            self._class_name = class_name
            self._instance = instance
            self._methods = {}

        def __getattr__(self, name):
            return self._methods.setdefault(
                name, _ClassMethod(self._class_name, self._instance, name)
            )

    class _Spawnable:
        def __init__(self, fn):
            self._fn = fn

        def __call__(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

    class _App:
        def __init__(self, *args, **kwargs):
            pass

        def cls(self, *args, **kwargs):
            def decorate(klass):
                engine_classes.append(klass)
                cls_options[klass.__name__] = kwargs
                instance = klass.__new__(klass)
                return lambda *a, **k: _ClassHandle(klass.__name__, instance)

            return decorate

        def function(self, *args, **kwargs):
            def decorate(fn):
                fn_options[fn.__name__] = kwargs
                return _Spawnable(fn)

            return decorate

    modal.App = _App
    modal.Dict = _FakeDict
    modal.Volume = _FakeVolume
    modal.Secret = _Named
    modal.Image = _Image
    for hook in ("enter", "method", "concurrent", "asgi_app"):
        setattr(modal, hook, lambda *a, **k: lambda fn: fn)
    monkeypatch.setitem(sys.modules, "modal", modal)


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASH_SERVING_KEY", "test-serving-key")
    module_box = {}
    spawned = []
    spawn_queue = []
    coordinator_gate = _CoordinatorGate()
    engine_classes = []
    cls_options = {}
    fn_options = {}
    unregistered = []

    async def settle(record):
        if record.get("repo_id") == BAD_REPO:
            return {
                "ok": False,
                "incarnation": module_box["module"]._record_incarnation(record),
                "failure": "ValueError: adapter rank exceeds max_lora_rank",
            }
        return {
            "ok": True,
            "incarnation": module_box["module"]._record_incarnation(record),
        }

    async def unload_exact(adapter_id, incarnation):
        unregistered.append((adapter_id, incarnation))
        return {"ok": True, "evicted": True}

    async def generate(payload, record):
        return {
            "text": f"served by {record['adapter_id']}",
            "finish_reason": "stop",
            "prompt_tokens": 7,
            "completion_tokens": 4,
        }

    async def generate_stream(payload, record):
        for piece in ("served ", "by ", record["adapter_id"]):
            yield {"delta": piece, "finish_reason": None}
        yield {"delta": "", "finish_reason": "stop"}

    engine_methods = {
        "settle": settle,
        "unload_exact": unload_exact,
        "generate": generate,
        "generate_stream": generate_stream,
    }
    _stub_modal(
        monkeypatch,
        engine_methods,
        spawned,
        spawn_queue,
        coordinator_gate,
        engine_classes,
        cls_options,
        fn_options,
    )
    source = render_app(MODELS[BASE_MODEL])
    module = types.ModuleType("generated_serving_app")
    module_box["module"] = module
    exec(compile(source, str(tmp_path / "app.py"), "exec"), module.__dict__)
    test_client = TestClient(module.api())
    test_client.headers["X-Freesolo-Internal-Key"] = "test-serving-key"
    for name, value in {
        "generated_module": module,
        "engine_classes": engine_classes,
        "cls_options": cls_options,
        "fn_options": fn_options,
        "engine_methods": engine_methods,
        "spawned": spawned,
        "spawn_queue": spawn_queue,
        "coordinator_gate": coordinator_gate,
        "unregistered": unregistered,
    }.items():
        setattr(test_client.app.state, name, value)
    return test_client


@pytest.fixture
def engine_class(client):
    classes = client.app.state.engine_classes
    return next(klass for klass in classes if klass.__name__ == "Engine")


@pytest.fixture
def lifecycle_class(client):
    classes = client.app.state.engine_classes
    return next(klass for klass in classes if klass.__name__ == "LifecycleCoordinator")


@pytest.fixture
def generated_module(client):
    return client.app.state.generated_module


def _lifecycle(client, adapter_id):
    response = client.get(f"/adapters/{adapter_id}")
    return (response.json()["adapter"].get("metadata") or {}).get("lifecycle_state")


def _drain_spawn_queue(client):
    queue = client.app.state.spawn_queue
    while queue:
        method, args, kwargs = queue.pop(0)
        _run_awaitable(method._call(*args, **kwargs))


def _register_and_ready(client, registration=None):
    body = dict(REGISTRATION if registration is None else registration)
    body["metadata"] = dict(body["metadata"])
    response = client.post("/adapters", json=body)
    assert response.status_code == 202
    assert _lifecycle(client, body["adapter_id"]) == "registered"
    _drain_spawn_queue(client)
    assert _lifecycle(client, body["adapter_id"]) == "ready"
    return body


def _activate(client, revision=REVISION, expected=None):
    return client.post(
        f"/adapters/{revision}/activate",
        json={"expected_adapter_revision": expected},
    )


def _second_registration():
    return {
        **REGISTRATION,
        "adapter_id": SECOND_REVISION,
        "checkpoint": "run-abc/step-20",
        "metadata": {**REGISTRATION["metadata"], "checkpoint_step": 20, "hf_revision": "b" * 40},
    }


def _record(registration=REGISTRATION, *, status="registered", attempt=None):
    record = copy.deepcopy(registration)
    record["status"] = status
    record["metadata"]["lifecycle_state"] = status
    if attempt is not None:
        record["metadata"]["settle_attempt"] = attempt
    return record


def _engine(engine_class, *, runtime=None, registered=None):
    instance = engine_class.__new__(engine_class)
    instance._locks = {}
    instance._registered = dict(registered or {})
    if runtime is not None:
        instance._runtime = runtime
    return instance


def _request_for(engine_class, record, payload=None):
    instance = _engine(engine_class, registered={record["adapter_id"]: "incarnation"})

    async def no_op(*args):
        return "incarnation"

    instance._ensure_loaded = no_op
    instance._reap_terminal_residents = no_op
    body = {"messages": [{"role": "user", "content": "hi"}], **(payload or {})}
    return _run_awaitable(engine_class._request(instance, body, record))


def test_registration_returns_before_queued_settlement_runs(client):
    response = client.post("/adapters", json=REGISTRATION)
    assert response.status_code == 202
    assert _lifecycle(client, REVISION) == "registered"
    assert len(client.app.state.spawn_queue) == 1
    _drain_spawn_queue(client)
    assert _lifecycle(client, REVISION) == "ready"
    assert client.app.state.spawn_queue == []


def test_registration_rejects_thinking_with_structured_outputs_before_mutation(client):
    body = {
        **REGISTRATION,
        "metadata": dict(REGISTRATION["metadata"]),
        "thinking": True,
        "structured_outputs": {"choice": ["yes", "no"]},
    }
    assert client.post("/adapters", json=body).status_code == 422
    module = client.app.state.generated_module
    assert module._record_key(REVISION) not in module.adapter_records
    assert module._record_key(RUN_ID) not in module.adapter_records
    assert client.app.state.spawn_queue == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repo_id", "other/repo"),
        ("repo_type", "model"),
        ("subfolder", "other/path"),
        ("checkpoint", "other/checkpoint"),
        ("thinking", True),
        ("org_id", "other-org"),
        ("structured_outputs", {"regex": "x+"}),
    ],
)
def test_reregistering_different_content_under_one_revision_is_a_conflict(client, field, value):
    _register_and_ready(client)
    changed = {**REGISTRATION, field: value, "metadata": dict(REGISTRATION["metadata"])}
    assert client.post("/adapters", json=changed).status_code == 409


def test_a_failed_adapter_load_reports_failed_not_ready(client):
    body = {**REGISTRATION, "repo_id": BAD_REPO, "metadata": dict(REGISTRATION["metadata"])}
    assert client.post("/adapters", json=body).status_code == 202
    _drain_spawn_queue(client)
    record = client.get(f"/adapters/{REVISION}").json()["adapter"]
    assert record["status"] == "disabled"
    assert record["metadata"]["lifecycle_state"] == "failed"
    assert "max_lora_rank" in record["metadata"]["failure"]


def test_a_late_settle_exception_preserves_an_already_ready_record(client):
    module = client.app.state.generated_module

    async def committed_then_lost(record):
        current = await module._read(record["adapter_id"])
        current["status"] = "ready"
        current["metadata"] = {
            **(current.get("metadata") or {}),
            "lifecycle_state": "ready",
        }
        await module._write(current)
        raise RuntimeError("response lost")

    client.app.state.engine_methods["settle"] = committed_then_lost
    assert client.post("/adapters", json=REGISTRATION).status_code == 202
    _drain_spawn_queue(client)
    record = client.get(f"/adapters/{REVISION}").json()["adapter"]
    assert record["status"] == "ready"
    assert record["metadata"]["lifecycle_state"] == "ready"
    assert client.app.state.unregistered == []


def test_engine_response_error_unloads_exact_attempt_and_marks_it_failed(client):
    module = client.app.state.generated_module

    async def response_error(record):
        raise RuntimeError("engine response unavailable")

    client.app.state.engine_methods["settle"] = response_error
    assert client.post("/adapters", json=REGISTRATION).status_code == 202
    registered = client.get(f"/adapters/{REVISION}").json()["adapter"]
    incarnation = module._record_incarnation(registered)
    assert incarnation == f"{REVISION}@{registered['metadata']['settle_attempt']}"
    assert registered["status"] == "registered"
    assert module._lifecycle_state(registered) == "registered"
    _drain_spawn_queue(client)
    failed = client.get(f"/adapters/{REVISION}").json()["adapter"]
    assert client.app.state.unregistered == [(REVISION, incarnation)]
    assert failed["status"] == "disabled"
    assert module._lifecycle_state(failed) == "failed"
    assert (
        "the serving engine did not answer: engine response unavailable"
        in failed["metadata"]["failure"]
    )


def test_redeploying_the_same_checkpoint_after_undeploy_uses_a_fresh_incarnation(
    client, monkeypatch
):
    _register_and_ready(client)
    module = client.app.state.generated_module
    first = module.adapter_records[module._record_key(REVISION)]
    first_incarnation = module._record_incarnation(first)
    assert _activate(client).status_code == 200
    assert client.delete(f"/adapters/{RUN_ID}").status_code == 200
    pending = []
    monkeypatch.setattr(
        module.lifecycle.settle,
        "spawn",
        lambda record: pending.append(copy.deepcopy(record)),
    )
    response = client.post("/adapters", json=REGISTRATION)
    assert response.status_code == 202
    second = module.adapter_records[module._record_key(REVISION)]
    assert module._lifecycle_state(second) == "registered"
    assert module._record_incarnation(second) != first_incarnation
    assert module._record_incarnation(pending[0]) == module._record_incarnation(second)


def test_remove_cannot_be_overwritten_by_an_inflight_successful_settlement(
    client, lifecycle_class, monkeypatch
):
    module = client.app.state.generated_module
    pending = []
    monkeypatch.setattr(
        module.lifecycle.settle,
        "spawn",
        lambda record: pending.append(copy.deepcopy(record)),
    )
    assert client.post("/adapters", json=REGISTRATION).status_code == 202
    record = pending[0]
    assert client.delete(f"/adapters/{RUN_ID}").status_code == 200
    outcome = _run_awaitable(lifecycle_class.settle(module.lifecycle._instance, record))
    current = module.adapter_records[module._record_key(REVISION)]
    assert outcome == {"ok": True, "superseded": True}
    assert current["status"] == "disabled"
    assert module._lifecycle_state(current) == "disabled"


def test_coordinator_serializes_remove_behind_a_blocked_settlement(client):
    module = client.app.state.generated_module
    settle_entered = threading.Event()
    release_settle = threading.Event()
    remove_entered = threading.Event()
    failures = []
    remove_results = []

    async def blocked_settle(record):
        settle_entered.set()
        await asyncio.to_thread(release_settle.wait)
        return {
            "ok": True,
            "incarnation": module._record_incarnation(record),
        }

    original_remove = module.lifecycle._instance.remove

    async def observed_remove(adapter_id):
        remove_entered.set()
        return await original_remove(adapter_id)

    def drain():
        try:
            _drain_spawn_queue(client)
        except BaseException as exc:
            failures.append(exc)

    def remove():
        try:
            remove_results.append(_run_awaitable(module.lifecycle.remove.remote.aio(RUN_ID)))
        except BaseException as exc:
            failures.append(exc)

    assert client.post("/adapters", json=REGISTRATION).status_code == 202
    module.lifecycle._instance.remove = observed_remove
    client.app.state.engine_methods["settle"] = blocked_settle
    settle_thread = threading.Thread(target=drain)
    settle_thread.start()
    assert settle_entered.wait(2)
    remove_thread = threading.Thread(target=remove)
    remove_thread.start()
    assert client.app.state.coordinator_gate.wait_for("remove")
    try:
        assert remove_entered.is_set() is False
        assert _lifecycle(client, REVISION) == "registered"
    finally:
        release_settle.set()
        settle_thread.join(2)
        remove_thread.join(2)
    assert settle_thread.is_alive() is False
    assert remove_thread.is_alive() is False
    assert failures == []
    assert remove_results[0]["ok"] is True
    assert _lifecycle(client, REVISION) == "disabled"


def test_stale_settlement_cannot_overwrite_or_unload_a_newer_attempt(
    client, lifecycle_class, monkeypatch
):
    module = client.app.state.generated_module
    pending = []
    monkeypatch.setattr(
        module.lifecycle.settle,
        "spawn",
        lambda record: pending.append(copy.deepcopy(record)),
    )
    assert client.post("/adapters", json=REGISTRATION).status_code == 202
    stale = pending[-1]
    assert client.post("/adapters", json=REGISTRATION).status_code == 202
    newer = pending[-1]
    assert module._record_incarnation(stale) != module._record_incarnation(newer)
    outcome = _run_awaitable(lifecycle_class.settle(module.lifecycle._instance, stale))
    current = module.adapter_records[module._record_key(REVISION)]
    assert outcome == {"ok": True, "superseded": True}
    assert module._record_incarnation(current) == module._record_incarnation(newer)
    assert module._lifecycle_state(current) == "registered"
    assert client.app.state.unregistered == []


def test_rejected_settlement_unloads_only_the_attempt_it_loaded(
    client, lifecycle_class, monkeypatch
):
    module = client.app.state.generated_module
    pending = []
    monkeypatch.setattr(
        module.lifecycle.settle,
        "spawn",
        lambda record: pending.append(copy.deepcopy(record)),
    )
    assert client.post("/adapters", json=REGISTRATION).status_code == 202
    stale = pending[-1]
    newer = {
        **stale,
        "metadata": {
            **stale["metadata"],
            "settle_attempt": "newer-attempt",
        },
    }

    async def load_then_advance(record):
        await module._write(newer)
        return {"ok": True, "incarnation": module._record_incarnation(record)}

    client.app.state.engine_methods["settle"] = load_then_advance
    outcome = _run_awaitable(lifecycle_class.settle(module.lifecycle._instance, stale))
    assert outcome == {"ok": True, "superseded": True}
    assert module._record_incarnation(
        module.adapter_records[module._record_key(REVISION)]
    ) == module._record_incarnation(newer)
    assert client.app.state.unregistered == [(REVISION, module._record_incarnation(stale))]


def test_a_stale_compare_and_swap_is_rejected(client):
    _register_and_ready(client)
    assert _activate(client).status_code == 200
    assert _activate(client).status_code == 409


def test_chat_reports_which_immutable_revision_answered(client):
    _register_and_ready(client)
    assert _activate(client).status_code == 200
    response = client.post(
        "/v1/chat/completions",
        json={"model": RUN_ID, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == f"served by {REVISION}"
    assert payload["freesolo"] == {
        "adapter_revision": REVISION,
        "checkpoint": REGISTRATION["checkpoint"],
        "hf_revision": "a" * 40,
    }
    assert response.headers["X-Freesolo-Adapter-Revision"] == REVISION
    assert response.headers["X-Freesolo-Checkpoint"] == REGISTRATION["checkpoint"]
    assert response.headers["X-Freesolo-HF-Revision"] == "a" * 40


def test_a_streaming_request_gets_sse(client):
    _register_and_ready(client)
    assert _activate(client).status_code == 200
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": RUN_ID,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        assert "text/event-stream" in response.headers["content-type"]
        lines = [line for line in response.iter_lines() if line.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[5:]) for line in lines[:-1]]
    text = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
    assert f"served by {REVISION}" in text
    assert all(chunk["freesolo"]["adapter_revision"] == REVISION for chunk in chunks)


@pytest.mark.parametrize("block_type", sorted(_IMAGE_BLOCK_TYPES))
def test_an_image_request_is_refused_rather_than_answered_blind(client, block_type):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": RUN_ID,
            "messages": [{"role": "user", "content": [{"type": block_type, "image_url": "x"}]}],
        },
    )
    assert response.status_code == 400
    assert "text-only" in response.json()["detail"]


def test_generation_uses_the_record_grammar_not_the_request(engine_class, generated_module):
    module = generated_module
    record = {**REGISTRATION, "structured_outputs": {"choice": ["yes", "no"]}}
    spec = module._adapter_spec(record, "/tmp/adapter")
    assert spec.structured_outputs == {"choice": ["yes", "no"]}
    request = _request_for(engine_class, record, {"structured_outputs": {"regex": ".*"}})
    assert request.structured_outputs is None


def test_missing_hf_revision_fails_before_download_or_provenance_reconstruction(
    engine_class, generated_module, monkeypatch
):
    record = copy.deepcopy(REGISTRATION)
    record["metadata"].pop("hf_revision")
    downloads = []
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **kwargs: downloads.append(kwargs)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    instance = engine_class.__new__(engine_class)
    with pytest.raises(RuntimeError, match="hf_revision"):
        _run_awaitable(engine_class._adapter_path(instance, record))
    with pytest.raises(RuntimeError, match="hf_revision"):
        generated_module._provenance(record)
    assert downloads == []


def test_malformed_request_fails_before_resident_mutation(engine_class):
    record = copy.deepcopy(REGISTRATION)
    record["metadata"].pop("settle_attempt")
    reaped = []
    instance = engine_class.__new__(engine_class)

    async def reap():
        reaped.append(True)

    instance._reap_terminal_residents = reap
    with pytest.raises(RuntimeError, match="settle_attempt"):
        _run_awaitable(engine_class._request(instance, {"messages": _MESSAGES}, record))
    assert reaped == []


def test_stop_sequences_reach_the_runtime_request(engine_class):
    request = _request_for(engine_class, REGISTRATION, {"stop": ["END"]})
    assert request.stop == ("END",)
    assert _request_for(engine_class, REGISTRATION).stop == ()


def test_request_uses_the_incarnation_captured_while_loading(engine_class):
    instance = _engine(engine_class, registered={REVISION: "newer-incarnation"})

    async def no_reap():
        return None

    async def capture_then_replace(record):
        instance._registered.pop(record["adapter_id"])
        return "captured-incarnation"

    instance._reap_terminal_residents = no_reap
    instance._ensure_loaded = capture_then_replace
    request = _run_awaitable(engine_class._request(instance, {"messages": _MESSAGES}, REGISTRATION))
    assert request.expected_incarnation == "captured-incarnation"
    assert REVISION not in instance._registered


def test_malformed_undeploy_state_fails_before_durable_or_cache_mutation(client, lifecycle_class):
    module = client.app.state.generated_module
    record = copy.deepcopy(REGISTRATION)
    record["metadata"].pop("run_id")
    module.adapter_records[module._record_key(REVISION)] = record
    before = copy.deepcopy(dict(module.adapter_records))
    instance = lifecycle_class.__new__(lifecycle_class)
    with pytest.raises(RuntimeError, match="run_id"):
        _run_awaitable(lifecycle_class.remove(instance, REVISION))
    assert dict(module.adapter_records) == before
    assert module.cache_volume.remove_calls == []


def test_undeploy_disables_the_alias_and_its_revisions(client):
    _register_and_ready(client)
    assert _activate(client).status_code == 200
    response = client.delete(f"/adapters/{RUN_ID}")
    assert response.status_code == 200
    assert response.json()["disabled_aliases"] == [RUN_ID]
    assert response.json()["disabled_revisions"] == [REVISION]
    for adapter_id in (RUN_ID, REVISION):
        assert client.get(f"/adapters/{adapter_id}").json()["adapter"]["status"] == "disabled"
    chat = client.post(
        "/v1/chat/completions",
        json={"model": RUN_ID, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert chat.status_code == 404
    assert client.app.state.unregistered == []
    module = client.app.state.generated_module
    assert module.cache_volume.remove_calls == [
        (f"adapters/{module._adapter_digest(REVISION)}", True)
    ]


def test_undeploy_reads_only_the_runs_durable_index(client, monkeypatch):
    _register_and_ready(client)
    assert _activate(client).status_code == 200
    module = client.app.state.generated_module
    assert module.adapter_records[module._run_index_key(RUN_ID)] == [RUN_ID, REVISION]
    unrelated = {"adapter_id": "other-run", "status": "ready", "metadata": {"run_id": "other"}}
    module.adapter_records[module._record_key("other-run")] = unrelated

    def scanned_every_key(_records):
        raise AssertionError("undeploy scanned the app-wide modal.Dict instead of the run index")

    monkeypatch.setattr(type(module.adapter_records), "keys", property(scanned_every_key))
    response = client.delete(f"/adapters/{RUN_ID}")
    assert response.status_code == 200
    assert response.json()["disabled_aliases"] == [RUN_ID]
    assert response.json()["disabled_revisions"] == [REVISION]
    assert module.adapter_records[module._record_key("other-run")] == unrelated


def test_undeploy_cleanup_failure_is_retryable_without_reviving_records(client):
    _register_and_ready(client)
    assert _activate(client).status_code == 200
    module = client.app.state.generated_module
    path = f"adapters/{module._adapter_digest(REVISION)}"
    module.cache_volume.remove_failures[path] = RuntimeError("volume unavailable")
    failed = client.delete(f"/adapters/{RUN_ID}")
    assert failed.status_code == 503
    assert failed.json()["retryable"] is True
    assert failed.json()["disabled_aliases"] == [RUN_ID]
    assert failed.json()["disabled_revisions"] == [REVISION]
    for adapter_id in (RUN_ID, REVISION):
        assert client.get(f"/adapters/{adapter_id}").json()["adapter"]["status"] == "disabled"
    module.cache_volume.remove_failures.clear()
    retried = client.delete(f"/adapters/{RUN_ID}")
    assert retried.status_code == 200
    assert retried.json()["disabled_aliases"] == [RUN_ID]
    assert retried.json()["disabled_revisions"] == [REVISION]
    assert module.cache_volume.remove_calls == [(path, True), (path, True)]
    assert client.app.state.unregistered == []


def _state_record(registration=REGISTRATION, *, status="registered", attempt=None):
    record = copy.deepcopy(registration)
    record["status"] = status
    record["metadata"]["lifecycle_state"] = status
    if attempt:
        record["metadata"]["settle_attempt"] = attempt
    return record


def _engine_instance(engine_class, runtime, registered=None):
    instance = engine_class.__new__(engine_class)
    instance._locks = {}
    instance._registered = dict(registered or {})
    instance._runtime = runtime
    instance._adapter_path = lambda record: asyncio.sleep(0, result="/tmp/adapter")
    return instance


def test_warm_disabled_resident_unloads_before_later_settlement(client, engine_class):
    module = client.app.state.generated_module
    first = _state_record(status="disabled", attempt="old-attempt")
    second = _state_record(_second_registration(), attempt="new-attempt")
    module.adapter_records[module._record_key(REVISION)] = first
    module.adapter_records[module._record_key(SECOND_REVISION)] = second
    events = []

    async def unload_adapter(adapter_id, *, expected_incarnation):
        events.append(("unload", adapter_id, expected_incarnation))

    async def register_adapter(spec):
        events.append(("register", spec.adapter_id, spec.incarnation))

    instance = _engine_instance(
        engine_class,
        types.SimpleNamespace(unload_adapter=unload_adapter, register_adapter=register_adapter),
        {REVISION: module._record_incarnation(first)},
    )
    assert _run_awaitable(engine_class.settle(instance, second))["ok"] is True
    assert events == [
        ("unload", REVISION, module._record_incarnation(first)),
        ("register", SECOND_REVISION, module._record_incarnation(second)),
    ]


def test_failed_deferred_unload_stays_tracked_and_retries(client, engine_class):
    module = client.app.state.generated_module
    record = _state_record(status="disabled", attempt="attempt-1")
    module.adapter_records[module._record_key(REVISION)] = record
    incarnation = module._record_incarnation(record)
    attempts = []

    async def unload_adapter(adapter_id, *, expected_incarnation):
        attempts.append((adapter_id, expected_incarnation))
        if len(attempts) == 1:
            raise RuntimeError("busy")

    instance = _engine_instance(
        engine_class,
        types.SimpleNamespace(unload_adapter=unload_adapter),
        {REVISION: incarnation},
    )
    _run_awaitable(engine_class._reap_terminal_residents(instance))
    assert instance._registered == {REVISION: incarnation}
    _run_awaitable(engine_class._reap_terminal_residents(instance))
    assert instance._registered == {}
    assert attempts == [(REVISION, incarnation), (REVISION, incarnation)]


def test_concurrent_adapter_loads_register_each_adapter_once(client, engine_class):
    module = client.app.state.generated_module
    registered = []

    async def register_adapter(spec):
        await asyncio.sleep(0)
        registered.append(spec.adapter_id)
        return True

    instance = _engine_instance(
        engine_class, types.SimpleNamespace(register_adapter=register_adapter)
    )
    records = (REGISTRATION, _second_registration())
    for record in records:
        module.adapter_records[module._record_key(record["adapter_id"])] = _state_record(record)

    async def load():
        await asyncio.gather(
            *(engine_class._ensure_loaded(instance, record) for record in (*records, records[0]))
        )

    _run_awaitable(load())
    assert sorted(registered) == sorted(record["adapter_id"] for record in records)


def test_engine_resident_shortcut_compares_the_exact_incarnation(client, engine_class):
    module = client.app.state.generated_module
    record = _state_record(attempt="new-attempt")
    module.adapter_records[module._record_key(REVISION)] = record
    old = f"{REVISION}@old-attempt"
    events = []

    async def unload_adapter(adapter_id, *, expected_incarnation):
        events.append(("unload", expected_incarnation))

    async def register_adapter(spec):
        events.append(("register", spec.incarnation))

    instance = _engine_instance(
        engine_class,
        types.SimpleNamespace(unload_adapter=unload_adapter, register_adapter=register_adapter),
        {REVISION: old},
    )
    assert _run_awaitable(engine_class._load_lora_locked(instance, record))["ok"] is True
    assert events == [("unload", old), ("register", module._record_incarnation(record))]


def test_a_raising_post_load_read_still_unloads_what_it_just_registered(
    client, engine_class, monkeypatch
):
    module = client.app.state.generated_module
    unloaded = []

    async def register_adapter(spec):
        return True

    async def unload_adapter(adapter_id, *, expected_incarnation):
        unloaded.append((adapter_id, expected_incarnation))

    instance = _engine_instance(
        engine_class,
        types.SimpleNamespace(register_adapter=register_adapter, unload_adapter=unload_adapter),
    )
    reads = iter([_state_record()])

    async def read(adapter_id):
        try:
            return next(reads)
        except StopIteration:
            raise RuntimeError("dict unavailable") from None

    monkeypatch.setattr(module, "_read", read)
    with pytest.raises(RuntimeError, match="dict unavailable"):
        _run_awaitable(engine_class._load_lora_locked(instance, REGISTRATION))
    assert unloaded == [(REVISION, module._record_incarnation(REGISTRATION))]
    assert REVISION not in instance._registered


def test_a_failed_mid_load_eviction_keeps_the_resident_adapter_removable(
    client, engine_class, monkeypatch
):
    module = client.app.state.generated_module

    async def register_adapter(spec):
        return True

    async def failing_unload(adapter_id, *, expected_incarnation):
        raise RuntimeError("eviction failed")

    instance = _engine_instance(
        engine_class,
        types.SimpleNamespace(register_adapter=register_adapter, unload_adapter=failing_unload),
    )
    reads = iter([_state_record(), _state_record(status="disabled")])

    async def read(adapter_id):
        return next(reads)

    monkeypatch.setattr(module, "_read", read)
    assert (
        _run_awaitable(engine_class._load_lora_locked(instance, REGISTRATION))["superseded"] is True
    )
    incarnation = module._record_incarnation(REGISTRATION)
    unloaded = []

    async def unload_ok(adapter_id, *, expected_incarnation):
        unloaded.append((adapter_id, expected_incarnation))

    instance._runtime.unload_adapter = unload_ok
    assert _run_awaitable(engine_class._unload_locked(instance, REVISION, incarnation)) is True
    assert unloaded == [(REVISION, incarnation)]


@pytest.mark.parametrize(
    ("configured", "presented", "status_code"),
    [(" secret ", "secret", 404), ("secret", " wrong ", 401), ("   ", "wrong", 503)],
)
def test_the_serving_key_is_compared_after_stripping(
    client, monkeypatch, configured, presented, status_code
):
    monkeypatch.setenv("FLASH_SERVING_KEY", configured)
    response = client.get("/adapters/unknown", headers={"X-Freesolo-Internal-Key": presented})
    assert response.status_code == status_code


def test_serving_key_uses_constant_time_comparison(client, monkeypatch):
    module = client.app.state.generated_module
    seen = []
    monkeypatch.setenv("FLASH_SERVING_KEY", "secret")
    monkeypatch.setattr(module.hmac, "compare_digest", lambda a, b: seen.append((a, b)) or True)
    assert (
        client.get("/adapters/unknown", headers={"X-Freesolo-Internal-Key": ""}).status_code == 404
    )
    assert seen == [("", "secret")]


def test_only_the_lifecycle_coordinator_writes_durable_serving_state():
    tree = ast.parse(render_app(MODELS[BASE_MODEL]))
    coordinator = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LifecycleCoordinator"
    )
    engine = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Engine"
    )
    handlers = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name in {"register_adapter", "activate", "remove"}
    }

    def durable_writes(node):
        return [
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and (
                (isinstance(call.func, ast.Name) and call.func.id == "_write")
                or (
                    isinstance(call.func, ast.Attribute)
                    and call.func.attr == "aio"
                    and isinstance(call.func.value, ast.Attribute)
                    and call.func.value.attr == "put"
                )
            )
        ]

    assert durable_writes(coordinator)
    assert not durable_writes(engine)
    assert set(handlers) == {"register_adapter", "activate", "remove"}
    assert all(not durable_writes(handler) for handler in handlers.values())


def test_every_generated_function_is_within_the_production_limit():
    tree = ast.parse(render_app(MODELS[BASE_MODEL]))
    lengths = {
        node.name: node.end_lineno - node.lineno + 1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert max(lengths.values()) <= 150


def test_generated_app_is_under_the_production_file_limit():
    assert len(render_app(MODELS[BASE_MODEL]).splitlines()) <= 1000
