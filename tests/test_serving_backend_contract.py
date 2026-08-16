"""Exercise the generated Modal app with Modal and the GPU engine stubbed."""

from __future__ import annotations

import ast
import asyncio
import copy
import inspect
import json
import sys
import threading
import types
from pathlib import Path

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


class _AsyncIterAio(_Aio):
    async def aio(self, *args, **kwargs):  # type: ignore[override]
        for item in self._fn(*args, **kwargs):
            yield item


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

    def _pop(self, key, default=None):
        return copy.deepcopy(super().pop(key, default))

    @property
    def put(self):
        return _Aio(self._put)

    @property
    def get(self):
        return _Aio(self._get)

    @property
    def pop(self):
        return _Aio(self._pop)

    @property
    def keys(self):
        return _AsyncIterAio(lambda: list(super(_FakeDict, self).keys()))


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

        def apt_install(self, *args, **kwargs):
            return self

        def pip_install(self, *args, **kwargs):
            return self

        def env(self, *args, **kwargs):
            return self

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
            if self._class_name == "LifecycleCoordinator":
                spawn_queue.append((self, copied_args, copied_kwargs))
                return None
            result = self._target()(*copied_args, **copied_kwargs)
            if inspect.isawaitable(result):
                return _run_awaitable(result)
            return result

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

        @property
        def remote(self):
            return types.SimpleNamespace(aio=self._fn)

        def spawn(self, *args, **kwargs):
            spawned.append((self._fn.__name__, args, kwargs))
            result = self._fn(*args, **kwargs)
            if inspect.isawaitable(result):
                return _run_awaitable(result)
            return result

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
    test_client.app.state.generated_module = module
    test_client.app.state.engine_classes = engine_classes
    test_client.app.state.cls_options = cls_options
    test_client.app.state.fn_options = fn_options
    test_client.app.state.engine_methods = engine_methods
    test_client.app.state.spawned = spawned
    test_client.app.state.spawn_queue = spawn_queue
    test_client.app.state.coordinator_gate = coordinator_gate
    test_client.app.state.unregistered = unregistered
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
    """the executed generated app.

    the fixture execs the rendered source into a bare module without registering it in
    `sys.modules`, so module-level helpers have to be reached through the client rather than
    looked up by name.
    """
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
        "metadata": {
            **REGISTRATION["metadata"],
            "checkpoint_step": 20,
            "hf_revision": "b" * 40,
        },
    }


def _request_for(engine_class, record, payload=None):
    """the runtime request the generated app builds for one chat payload and durable record.

    the app no longer constructs vllm sampling params itself, so the meaningful boundary is the
    `GenerationRequest` handed to the runtime plus the adapter spec registered alongside it.
    """
    instance = engine_class.__new__(engine_class)
    adapter_id = record["adapter_id"]
    instance._locks = {}
    instance._registered = {adapter_id: "incarnation"}

    async def no_op(*args):
        return None

    instance._ensure_loaded = no_op
    instance._reap_terminal_residents = no_op
    body = {"messages": [{"role": "user", "content": "hi"}], **(payload or {})}
    return _run_awaitable(engine_class._request(instance, body, record))


def test_healthz_advertises_the_required_capabilities(client):
    payload = client.get("/healthz").json()
    assert payload["capabilities"] == [
        "immutable_adapter_revisions",
        "alias_compare_and_swap",
        "revision_provenance",
    ]


def test_engine_coordinator_and_api_are_pinned_to_one_container(client):
    assert client.app.state.cls_options["Engine"]["max_containers"] == 1
    coordinator = client.app.state.cls_options["LifecycleCoordinator"]
    assert coordinator["max_containers"] == 1
    assert "gpu" not in coordinator
    assert client.app.state.fn_options["api"]["max_containers"] == 1


def test_registration_is_accepted_and_reaches_ready(client):
    _register_and_ready(client)
    assert client.app.state.spawned[0][0] == "LifecycleCoordinator.settle"


def test_registration_returns_before_queued_settlement_runs(client):
    response = client.post("/adapters", json=REGISTRATION)

    assert response.status_code == 202
    assert _lifecycle(client, REVISION) == "registered"
    assert len(client.app.state.spawn_queue) == 1

    _drain_spawn_queue(client)

    assert _lifecycle(client, REVISION) == "ready"
    assert client.app.state.spawn_queue == []


def test_readback_carries_the_identity_the_client_cross_checks(client):
    _register_and_ready(client)
    record = client.get(f"/adapters/{REVISION}").json()["adapter"]
    for field in (
        "adapter_id",
        "repo_id",
        "repo_type",
        "subfolder",
        "base_model",
        "checkpoint",
        "thinking",
    ):
        assert record[field] == REGISTRATION[field]
    assert record["metadata"]["run_id"] == RUN_ID
    assert record["metadata"]["checkpoint_step"] == 10
    assert record["metadata"]["hf_revision"] == "a" * 40


def test_reregistering_identical_content_is_idempotent(client):
    _register_and_ready(client)
    response = client.post("/adapters", json=REGISTRATION)
    assert response.status_code == 202
    assert _lifecycle(client, REVISION) == "ready"


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


def test_changed_provenance_under_one_id_is_refused(client):
    _register_and_ready(client)
    changed = {**REGISTRATION, "metadata": {**REGISTRATION["metadata"], "run_id": "other"}}
    assert client.post("/adapters", json=changed).status_code == 422


def test_registration_requires_a_pinned_commit_sha(client):
    body = {
        **REGISTRATION,
        "adapter_id": "run-abc@step-10.main",
        "metadata": {**REGISTRATION["metadata"], "hf_revision": "main"},
    }
    assert client.post("/adapters", json=body).status_code == 422


def test_a_fractional_checkpoint_step_is_refused(client):
    body = {
        **REGISTRATION,
        "metadata": {**REGISTRATION["metadata"], "checkpoint_step": 10.5},
    }
    assert client.post("/adapters", json=body).status_code == 422


def test_padded_provenance_is_stored_canonically(client):
    body = {
        **REGISTRATION,
        "metadata": {
            **REGISTRATION["metadata"],
            "run_id": " run-abc ",
            "checkpoint_step": "10",
            "hf_revision": "  " + "a" * 40 + "  ",
        },
    }
    _register_and_ready(client, body)
    metadata = client.get(f"/adapters/{REVISION}").json()["adapter"]["metadata"]
    assert metadata["run_id"] == RUN_ID
    assert metadata["checkpoint_step"] == 10
    assert metadata["hf_revision"] == "a" * 40


def test_metadata_cannot_turn_a_revision_into_an_alias(client):
    body = {
        **REGISTRATION,
        "metadata": {**REGISTRATION["metadata"], "record_type": "alias"},
    }
    _register_and_ready(client, body)
    record = client.get(f"/adapters/{REVISION}").json()["adapter"]
    assert record["metadata"]["record_type"] == "revision"


def test_a_failed_adapter_load_reports_failed_not_ready(client):
    body = {**REGISTRATION, "repo_id": BAD_REPO, "metadata": dict(REGISTRATION["metadata"])}
    assert client.post("/adapters", json=body).status_code == 202
    _drain_spawn_queue(client)
    record = client.get(f"/adapters/{REVISION}").json()["adapter"]
    assert record["status"] == "disabled"
    assert record["metadata"]["lifecycle_state"] == "failed"
    assert "max_lora_rank" in record["metadata"]["failure"]


def test_a_lost_settle_response_does_not_overwrite_a_committed_ready_record(client):
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


def test_activation_returns_the_provenance_the_client_validates(client):
    _register_and_ready(client)
    response = _activate(client)
    assert response.status_code == 200
    payload = response.json()
    assert payload["adapter_id"] == RUN_ID
    assert payload["target_adapter_revision"] == REVISION
    assert payload["previous_adapter_revision"] is None
    assert payload["checkpoint"] == REGISTRATION["checkpoint"]
    assert payload["updated_at"]


def test_a_stale_compare_and_swap_is_rejected(client):
    _register_and_ready(client)
    assert _activate(client).status_code == 200
    assert _activate(client).status_code == 409


def test_a_non_null_expectation_replaces_the_alias(client):
    first = _register_and_ready(client)
    second = _register_and_ready(client, _second_registration())
    assert _activate(client, first["adapter_id"]).status_code == 200
    response = _activate(client, second["adapter_id"], first["adapter_id"])
    assert response.status_code == 200
    assert response.json()["previous_adapter_revision"] == first["adapter_id"]


def test_chat_is_refused_until_the_alias_is_activated(client):
    _register_and_ready(client)
    response = client.post(
        "/v1/chat/completions",
        json={"model": RUN_ID, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404


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


def test_chat_fails_closed_on_a_terminal_revision(client):
    _register_and_ready(client)
    assert _activate(client).status_code == 200
    module = client.app.state.generated_module
    record = module.adapter_records[module._record_key(REVISION)]
    record["status"] = "disabled"
    record["metadata"]["lifecycle_state"] = "failed"
    response = client.post(
        "/v1/chat/completions",
        json={"model": RUN_ID, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404


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


@pytest.mark.parametrize("limit", [0, -1, 1.5, True, "4"])
def test_an_invalid_max_tokens_is_rejected(client, limit):
    # the messages have to be valid, or the empty-messages guard answers 400 first and this
    # passes without ever reaching the max_tokens check.
    response = client.post(
        "/v1/chat/completions",
        json={"model": RUN_ID, "messages": _MESSAGES, "max_tokens": limit},
    )
    assert response.status_code == 400
    assert "max_tokens" in response.json()["detail"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", -0.1),
        ("temperature", True),
        ("temperature", float("inf")),
        ("top_p", 0),
        ("top_p", 1.1),
        ("top_p", float("nan")),
    ],
)
def test_invalid_sampling_values_are_rejected(client, field, value):
    # same reason as max_tokens above: filler empty messages would trip the emptiness guard and
    # return 400 without the sampling field being looked at.
    response = client.post(
        "/v1/chat/completions",
        content=json.dumps({"model": RUN_ID, "messages": _MESSAGES, field: value}),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert field in response.json()["detail"]


@pytest.mark.parametrize(
    "messages",
    [None, [], "hi", {"role": "user"}, ["hi"], [{"role": "user", "content": "hi"}, "hi"]],
)
def test_unusable_messages_are_refused_before_a_gpu_is_woken(client, messages, monkeypatch):
    """bad `messages` must 400 on the api container, not 500 from the gpu.

    `_request` builds the `GenerationRequest` inside the engine, so without an api-side guard
    these payloads reach `engine.generate.remote` first: modal wakes a gpu replica, and the
    runtime's own rejection surfaces as an unhandled 500. every other bad chat input answers 400
    without leaving the api container, so this asserts the gpu is never called at all.
    """
    # the adapter has to be ready and active, or the request 404s at model resolution and never
    # reaches the engine no matter what the guard does -- which would make `calls == []` below
    # true for the wrong reason.
    _register_and_ready(client)
    assert _activate(client).status_code == 200
    calls: list[object] = []
    module = client.app.state.generated_module
    monkeypatch.setattr(
        module.engine.generate,
        "remote",
        types.SimpleNamespace(aio=lambda *a, **k: calls.append(a)),
    )
    payload = {"model": RUN_ID}
    if messages is not None:
        payload["messages"] = messages

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert "non-empty list" in response.json()["detail"]
    assert calls == []


def test_a_failed_settlement_dispatch_cannot_disable_a_ready_record(client, lifecycle_class):
    _register_and_ready(client)
    module = client.app.state.generated_module
    record = module.adapter_records[module._record_key(REVISION)]
    assert record["status"] == "ready"
    coordinator = lifecycle_class.__new__(lifecycle_class)

    failed = _run_awaitable(
        lifecycle_class._fail_attempt(coordinator, record, "engine did not answer")
    )

    after = module.adapter_records[module._record_key(REVISION)]
    assert failed is False
    assert after["status"] == "ready"
    assert module._lifecycle_state(after) == "ready"
    assert "failure" not in after["metadata"]


def test_an_activated_alias_carries_both_state_fields(client):
    """The alias must record `lifecycle_state`, not just `status`.

    Records carry state in two fields and readers check either one. `activate` rebuilds the alias
    metadata from scratch, so omitting the lifecycle key leaves the alias as the only record whose
    state is legible from just one field -- and the only reason a `lifecycle_state`-based check
    does not then reject a healthy alias is that `_resolve_chat_record` rebinds to the alias
    target first. That makes a user-visible 200-vs-503 depend on statement ordering.
    """
    _register_and_ready(client)
    assert _activate(client).status_code == 200
    module = client.app.state.generated_module
    alias = module.adapter_records[module._record_key(RUN_ID)]

    assert alias["status"] == "ready"
    assert alias["metadata"]["lifecycle_state"] == "ready"
    assert module._is_terminal(alias) is False
    assert alias["metadata"]["alias_of"] == REVISION


def test_a_registered_grammar_reaches_the_adapter_spec(generated_module):
    record = {**REGISTRATION, "structured_outputs": {"regex": "[ab]+"}}
    spec = generated_module._adapter_spec(record, "/tmp/adapter")
    assert spec.structured_outputs == {"regex": "[ab]+"}


def test_generation_uses_the_record_grammar_not_the_request(engine_class, generated_module):
    """the registered grammar is authoritative; a request cannot widen or replace it."""
    module = generated_module
    record = {**REGISTRATION, "structured_outputs": {"choice": ["yes", "no"]}}
    spec = module._adapter_spec(record, "/tmp/adapter")
    assert spec.structured_outputs == {"choice": ["yes", "no"]}
    # the request carries no grammar at all, so a caller-supplied one cannot reach the runtime.
    request = _request_for(engine_class, record, {"structured_outputs": {"regex": ".*"}})
    assert request.structured_outputs is None


def test_an_omitted_max_tokens_gets_the_default(engine_class):
    assert _request_for(engine_class, REGISTRATION).max_tokens == 512


def test_stop_sequences_reach_the_runtime_request(engine_class):
    request = _request_for(engine_class, REGISTRATION, {"stop": ["END"]})
    assert request.stop == ("END",)
    assert _request_for(engine_class, REGISTRATION).stop == ()


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
    """undeploy must not scan every historical key in the app-wide modal dict.

    Records are never deleted, so a full scan grows without bound and makes one remote `get` per
    checkpoint ever registered by any run. The run index must name both the alias and its immutable
    revision, and `keys()` is made fatal here so a fallback scan cannot silently return later.
    """
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


def test_undeploying_an_unknown_run_is_a_clean_404(client):
    assert client.delete("/adapters/unknown-run").status_code == 404


def test_a_registration_for_another_base_model_is_refused(client):
    body = {**REGISTRATION, "base_model": "other/model", "metadata": dict(REGISTRATION["metadata"])}
    assert client.post("/adapters", json=body).status_code == 409


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


def test_warm_disabled_resident_unloads_before_later_settlement(client, engine_class):
    module = client.app.state.generated_module
    first = {
        **REGISTRATION,
        "status": "disabled",
        "metadata": {
            **REGISTRATION["metadata"],
            "lifecycle_state": "disabled",
            "settle_attempt": "old-attempt",
        },
    }
    second = {
        **_second_registration(),
        "status": "registered",
        "metadata": {
            **_second_registration()["metadata"],
            "lifecycle_state": "registered",
            "settle_attempt": "new-attempt",
        },
    }
    module.adapter_records[module._record_key(REVISION)] = first
    module.adapter_records[module._record_key(SECOND_REVISION)] = second
    events = []

    async def unload_adapter(adapter_id, *, expected_incarnation):
        events.append(("unload", adapter_id, expected_incarnation))

    async def register_adapter(spec):
        events.append(("register", spec.adapter_id, spec.incarnation))

    instance = engine_class.__new__(engine_class)
    instance._locks = {}
    instance._registered = {REVISION: module._record_incarnation(first)}
    instance._runtime = types.SimpleNamespace(
        unload_adapter=unload_adapter,
        register_adapter=register_adapter,
    )
    instance._adapter_path = lambda record: asyncio.sleep(0, result="/tmp/adapter")

    result = _run_awaitable(engine_class.settle(instance, second))

    assert result["ok"] is True
    assert events == [
        ("unload", REVISION, module._record_incarnation(first)),
        ("register", SECOND_REVISION, module._record_incarnation(second)),
    ]


def test_failed_deferred_unload_stays_tracked_and_retries(client, engine_class):
    module = client.app.state.generated_module
    record = {
        **REGISTRATION,
        "status": "disabled",
        "metadata": {
            **REGISTRATION["metadata"],
            "lifecycle_state": "disabled",
            "settle_attempt": "attempt-1",
        },
    }
    module.adapter_records[module._record_key(REVISION)] = record
    incarnation = module._record_incarnation(record)
    attempts = []

    async def unload_adapter(adapter_id, *, expected_incarnation):
        attempts.append((adapter_id, expected_incarnation))
        if len(attempts) == 1:
            raise RuntimeError("busy")

    instance = engine_class.__new__(engine_class)
    instance._locks = {}
    instance._registered = {REVISION: incarnation}
    instance._runtime = types.SimpleNamespace(unload_adapter=unload_adapter)

    _run_awaitable(engine_class._reap_terminal_residents(instance))
    assert instance._registered == {REVISION: incarnation}

    _run_awaitable(engine_class._reap_terminal_residents(instance))
    assert instance._registered == {}
    assert attempts == [(REVISION, incarnation), (REVISION, incarnation)]


def test_concurrent_adapter_loads_register_each_adapter_once(client, engine_class, monkeypatch):
    """two adapters loading at once must each register exactly once.

    lora slot identity now belongs to the runtime, so what the generated app still owns is the
    per-adapter lock that keeps a concurrent load from registering the same adapter twice.
    """
    module = client.app.state.generated_module
    instance = engine_class.__new__(engine_class)
    instance._locks = {}
    instance._registered = {}
    registered = []

    async def register_adapter(spec):
        await asyncio.sleep(0)
        registered.append(spec.adapter_id)
        return True

    instance._runtime = types.SimpleNamespace(register_adapter=register_adapter)
    instance._adapter_path = lambda record: asyncio.sleep(0, result="/tmp/adapter")
    first = dict(REGISTRATION)
    second = _second_registration()
    for record in (first, second):
        module.adapter_records[module._record_key(record["adapter_id"])] = {
            **record,
            "status": "registered",
        }

    async def load_both():
        await asyncio.gather(
            engine_class._ensure_loaded(instance, first),
            engine_class._ensure_loaded(instance, second),
            engine_class._ensure_loaded(instance, first),
        )

    _run_awaitable(load_both())
    assert sorted(registered) == sorted({first["adapter_id"], second["adapter_id"]})
    assert set(instance._registered) == {first["adapter_id"], second["adapter_id"]}


def test_engine_resident_shortcut_compares_the_exact_incarnation(client, engine_class):
    module = client.app.state.generated_module
    record = {
        **REGISTRATION,
        "status": "registered",
        "metadata": {
            **REGISTRATION["metadata"],
            "lifecycle_state": "registered",
            "settle_attempt": "new-attempt",
        },
    }
    module.adapter_records[module._record_key(REVISION)] = record
    old_incarnation = f"{REVISION}@old-attempt"
    events = []

    async def unload_adapter(adapter_id, *, expected_incarnation):
        events.append(("unload", expected_incarnation))

    async def register_adapter(spec):
        events.append(("register", spec.incarnation))

    instance = engine_class.__new__(engine_class)
    instance._locks = {}
    instance._registered = {REVISION: old_incarnation}
    instance._runtime = types.SimpleNamespace(
        unload_adapter=unload_adapter,
        register_adapter=register_adapter,
    )
    instance._adapter_path = lambda candidate: asyncio.sleep(0, result="/tmp/adapter")

    outcome = _run_awaitable(engine_class._load_lora_locked(instance, record))

    assert outcome["ok"] is True
    assert events == [
        ("unload", old_incarnation),
        ("register", module._record_incarnation(record)),
    ]
    assert instance._registered[REVISION] == module._record_incarnation(record)


def test_a_raising_post_load_read_still_unloads_what_it_just_registered(
    client, engine_class, monkeypatch, tmp_path
):
    """a durable read that raises must not leave the adapter resident.

    the disabled-record branch already undoes the load, but a transient Dict failure took the
    early-return path out of the function with the adapter still registered -- holding a LoRA slot
    and its cache with nothing left to reclaim them.
    """
    module = client.app.state.generated_module
    monkeypatch.setattr(module, "ADAPTER_DIR", str(tmp_path))
    instance = engine_class.__new__(engine_class)
    instance._locks = {}
    instance._registered = {}
    unloaded = []

    async def register_adapter(spec):
        return True

    async def unload_adapter(adapter_id, *, expected_incarnation):
        unloaded.append((adapter_id, expected_incarnation))
        return True

    instance._runtime = types.SimpleNamespace(
        register_adapter=register_adapter,
        unload_adapter=unload_adapter,
    )
    instance._adapter_path = lambda record: asyncio.sleep(0, result="/tmp/adapter")
    reads = iter([{**REGISTRATION, "status": "registered"}])

    async def read(adapter_id):
        try:
            return next(reads)
        except StopIteration:
            raise RuntimeError("dict unavailable") from None

    monkeypatch.setattr(module, "_read", read)
    with pytest.raises(RuntimeError, match="dict unavailable"):
        _run_awaitable(engine_class._load_lora_locked(instance, REGISTRATION))
    assert unloaded == [(REVISION, REVISION)]
    assert REVISION not in instance._registered


def test_a_failed_mid_load_eviction_keeps_the_resident_adapter_removable(
    client, engine_class, monkeypatch
):
    module = client.app.state.generated_module
    instance = engine_class.__new__(engine_class)
    instance._locks = {}
    instance._registered = {}

    async def register_adapter(spec):
        return True

    async def failing_unload(adapter_id, *, expected_incarnation):
        raise RuntimeError("eviction failed")

    instance._runtime = types.SimpleNamespace(
        register_adapter=register_adapter,
        unload_adapter=failing_unload,
    )
    instance._adapter_path = lambda record: asyncio.sleep(0, result="/tmp/adapter")
    registered = {**REGISTRATION, "status": "registered"}
    disabled = {
        **REGISTRATION,
        "status": "disabled",
        "metadata": {**REGISTRATION["metadata"], "lifecycle_state": "disabled"},
    }
    reads = iter([registered, disabled])

    async def read(adapter_id):
        return next(reads)

    monkeypatch.setattr(module, "_read", read)
    result = _run_awaitable(engine_class._load_lora_locked(instance, REGISTRATION))

    assert result["superseded"] is True
    assert instance._registered[REVISION] == REVISION
    unloaded = []

    async def unload_ok(adapter_id, *, expected_incarnation):
        unloaded.append((adapter_id, expected_incarnation))

    instance._runtime.unload_adapter = unload_ok
    assert _run_awaitable(engine_class._unload_locked(instance, REVISION, REVISION)) is True
    assert unloaded == [(REVISION, REVISION)]
    assert REVISION not in instance._registered


@pytest.mark.parametrize(
    ("configured", "presented", "status_code"),
    [(" secret ", "secret", 404), ("secret", " wrong ", 401), ("   ", "wrong", 404)],
)
def test_the_serving_key_is_compared_after_stripping(
    client, monkeypatch, configured, presented, status_code
):
    monkeypatch.setenv("FLASH_SERVING_KEY", configured)
    response = client.get(
        "/adapters/unknown",
        headers={"X-Freesolo-Internal-Key": presented},
    )
    assert response.status_code == status_code


def test_serving_key_uses_constant_time_comparison(client, monkeypatch):
    module = client.app.state.generated_module
    seen = []
    monkeypatch.setenv("FLASH_SERVING_KEY", "secret")
    monkeypatch.setattr(module.hmac, "compare_digest", lambda a, b: seen.append((a, b)) or True)
    assert client.get("/adapters/unknown").status_code == 404
    assert seen == [("", "secret")]


def test_durable_state_calls_use_the_async_modal_api():
    source = render_app(MODELS[BASE_MODEL])
    blocking = [
        line.strip()
        for line in source.splitlines()
        if "adapter_records." in line and not line.lstrip().startswith("#") and ".aio" not in line
    ]
    assert blocking == []


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
        writes = []
        for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
            if isinstance(call.func, ast.Name) and call.func.id == "_write":
                writes.append(call)
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "aio"
                and isinstance(call.func.value, ast.Attribute)
                and call.func.value.attr == "put"
            ):
                writes.append(call)
        return writes

    assert durable_writes(coordinator)
    assert durable_writes(engine) == []
    assert set(handlers) == {"register_adapter", "activate", "remove"}
    assert all(durable_writes(handler) == [] for handler in handlers.values())


def test_mutating_a_durable_read_without_put_does_not_persist(client):
    _register_and_ready(client)
    module = client.app.state.generated_module
    record = _run_awaitable(module._read(REVISION))

    record["status"] = "disabled"
    record["metadata"]["lifecycle_state"] = "failed"

    reread = _run_awaitable(module._read(REVISION))
    assert reread["status"] == "ready"
    assert reread["metadata"]["lifecycle_state"] == "ready"


def test_the_generated_app_has_no_cross_container_coordination_machinery():
    source = render_app(MODELS[BASE_MODEL])
    for removed in (
        "_lora_int_id",
        "_int_locks",
        "_int_ids",
        "lora id collision",
        "reclaim_adapter_cache",
        '"reclaim"',
        "_claim_lora_int_id",
        "_release_lora_int_id",
        "_engine_is_warm",
        "cache_reclaim_pending",
        "lora_release_pending",
        "_run_lock",
        "_members_key",
        "_UNDEPLOY_PASSES",
        "loraid:",
        "members:",
        "_RUN_LOCKS",
        "engine.unregister",
    ):
        assert removed not in source


def test_every_generated_function_is_within_the_production_limit():
    source = render_app(MODELS[BASE_MODEL])
    tree = ast.parse(source)
    lengths = {
        node.name: node.end_lineno - node.lineno + 1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert max(lengths.values()) <= 150, lengths


def test_generated_app_is_under_the_production_file_limit():
    source = render_app(MODELS[BASE_MODEL])
    assert len(source.splitlines()) <= 1000


def test_no_unexpected_test_artifacts_were_written(client):
    assert not Path("flash_serving_app.py").exists()
