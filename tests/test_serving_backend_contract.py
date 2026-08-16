"""Exercise the generated Modal app with Modal and the GPU engine stubbed."""

from __future__ import annotations

import ast
import asyncio
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


class _FakeDict(dict):
    @classmethod
    def from_name(cls, *args, **kwargs):
        return cls()

    def _put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = value
        return True

    @property
    def put(self):
        return _Aio(self._put)

    @property
    def get(self):
        return _Aio(super().get)

    @property
    def pop(self):
        return _Aio(super().pop)

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


def _stub_modal(monkeypatch, engine_methods, spawned, engine_classes, cls_options, fn_options):
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

    class _EngineHandle:
        def __getattr__(self, name):
            method = engine_methods[name]
            if inspect.isasyncgenfunction(method):
                return types.SimpleNamespace(remote_gen=types.SimpleNamespace(aio=method))
            return types.SimpleNamespace(remote=types.SimpleNamespace(aio=method))

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
                return lambda *a, **k: _EngineHandle()

            return decorate

        def function(self, *args, **kwargs):
            def decorate(fn):
                fn_options[fn.__name__] = kwargs
                return _Spawnable(fn)

            return decorate

    modal.App = _App
    modal.Dict = _FakeDict
    modal.Volume = _Named
    modal.Secret = _Named
    modal.Image = _Image
    for hook in ("enter", "method", "concurrent", "asgi_app"):
        setattr(modal, hook, lambda *a, **k: lambda fn: fn)
    monkeypatch.setitem(sys.modules, "modal", modal)


@pytest.fixture
def client(monkeypatch, tmp_path):
    module_box = {}
    spawned = []
    engine_classes = []
    cls_options = {}
    fn_options = {}
    unregistered = []

    async def settle(record):
        module = module_box["module"]
        current = await module._read(record["adapter_id"])
        if current is None:
            return {"ok": False}
        metadata = current.get("metadata") or {}
        if metadata.get("settle_attempt") != (record.get("metadata") or {}).get("settle_attempt"):
            return {"ok": True, "superseded": True}
        if current.get("status") == "disabled":
            return {"ok": True, "superseded": True}
        if record.get("repo_id") == BAD_REPO:
            current["status"] = "disabled"
            current["metadata"] = {
                **metadata,
                "lifecycle_state": "failed",
                "failure": "ValueError: adapter rank exceeds max_lora_rank",
            }
            await module._write(current)
            return {"ok": False}
        current["status"] = "ready"
        current["metadata"] = {**metadata, "lifecycle_state": "ready"}
        await module._write(current)
        return {"ok": True}

    async def unregister(adapter_id):
        unregistered.append(adapter_id)
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
        "unregister": unregister,
        "generate": generate,
        "generate_stream": generate_stream,
    }
    _stub_modal(
        monkeypatch,
        engine_methods,
        spawned,
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
    test_client.app.state.unregistered = unregistered
    return test_client


@pytest.fixture
def engine_class(client):
    classes = client.app.state.engine_classes
    assert len(classes) == 1
    return classes[0]


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


def _register_and_ready(client, registration=None):
    body = dict(REGISTRATION if registration is None else registration)
    body["metadata"] = dict(body["metadata"])
    response = client.post("/adapters", json=body)
    assert response.status_code == 202
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
    instance._registered = {adapter_id: "incarnation"}

    async def ensure_loaded(_record):
        return None

    instance._ensure_loaded = ensure_loaded
    body = {"messages": [{"role": "user", "content": "hi"}], **(payload or {})}
    return _run_awaitable(engine_class._request(instance, body, record))


def test_healthz_advertises_the_required_capabilities(client):
    payload = client.get("/healthz").json()
    assert payload["capabilities"] == [
        "immutable_adapter_revisions",
        "alias_compare_and_swap",
        "revision_provenance",
    ]


def test_engine_and_api_are_pinned_to_one_container(client):
    assert client.app.state.cls_options["Engine"]["max_containers"] == 1
    assert client.app.state.fn_options["api"]["max_containers"] == 1


def test_registration_is_accepted_and_reaches_ready(client):
    _register_and_ready(client)
    assert client.app.state.spawned[0][0] == "settle_adapter"


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
    record = client.get(f"/adapters/{REVISION}").json()["adapter"]
    assert record["status"] == "ready"
    assert record["metadata"]["lifecycle_state"] == "ready"


def test_redeploying_the_same_checkpoint_after_undeploy_works(client):
    _register_and_ready(client)
    assert _activate(client).status_code == 200
    assert client.delete(f"/adapters/{RUN_ID}").status_code == 200
    _register_and_ready(client)


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


def test_a_failed_settlement_dispatch_cannot_disable_a_ready_record(client):
    """Both dispatch paths fence identically, and neither may kill a record that reached ready.

    `settle_adapter` and `_spawn_settle` used to open-code the same fence with different guards:
    only one checked that the record was still `registered`. A settlement failure arriving after
    the adapter is serving would then disable a live adapter, which is exactly the case a
    stale-attempt fence exists to prevent.
    """
    _register_and_ready(client)
    module = client.app.state.generated_module
    record = module.adapter_records[module._record_key(REVISION)]
    assert record["status"] == "ready"

    # the failure carries the same settle_attempt, so only the ready check can reject it.
    _run_awaitable(module._fail_settlement(record, "engine did not answer"))

    after = module.adapter_records[module._record_key(REVISION)]
    assert after["status"] == "ready", "a settlement failure disabled an already-serving adapter"
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
    assert client.app.state.unregistered == [REVISION]


def test_undeploying_an_unknown_run_is_a_clean_404(client):
    assert client.delete("/adapters/unknown-run").status_code == 404


def test_a_registration_for_another_base_model_is_refused(client):
    body = {**REGISTRATION, "base_model": "other/model", "metadata": dict(REGISTRATION["metadata"])}
    assert client.post("/adapters", json=body).status_code == 409


def test_unload_deletes_cache_without_a_resident_request(
    client, engine_class, monkeypatch, tmp_path
):
    module = client.app.state.generated_module
    adapter_dir = tmp_path / module._adapter_digest(REVISION)
    adapter_dir.mkdir()
    (adapter_dir / "adapter.safetensors").write_text("weights")
    monkeypatch.setattr(module, "ADAPTER_DIR", str(tmp_path))
    instance = engine_class.__new__(engine_class)
    instance._registered = {}

    assert _run_awaitable(engine_class._unload_locked(instance, REVISION)) is True
    assert not adapter_dir.exists()


def test_unregister_preserves_cache_for_a_revived_revision(
    client, engine_class, monkeypatch, tmp_path
):
    _register_and_ready(client)
    module = client.app.state.generated_module
    adapter_dir = tmp_path / module._adapter_digest(REVISION)
    adapter_dir.mkdir()
    (adapter_dir / "adapter.safetensors").write_text("weights")
    monkeypatch.setattr(module, "ADAPTER_DIR", str(tmp_path))
    record = module.adapter_records[module._record_key(REVISION)]
    record["status"] = "registered"
    record["metadata"]["lifecycle_state"] = "registered"
    instance = engine_class.__new__(engine_class)
    instance._locks = {}
    instance._registered = {}

    result = _run_awaitable(engine_class.unregister(instance, REVISION))
    assert result == {"ok": True, "evicted": False, "revived": True}
    assert adapter_dir.exists()


def test_failed_load_deletes_cache_when_no_lora_is_resident(
    client, engine_class, monkeypatch, tmp_path
):
    module = client.app.state.generated_module
    monkeypatch.setattr(module, "ADAPTER_DIR", str(tmp_path))
    lora_request = types.ModuleType("vllm.lora.request")
    lora_request.LoRARequest = object
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.lora", types.ModuleType("vllm.lora"))
    monkeypatch.setitem(sys.modules, "vllm.lora.request", lora_request)
    record = {
        **REGISTRATION,
        "status": "registered",
        "metadata": {**REGISTRATION["metadata"], "settle_attempt": "attempt-1"},
    }
    module.adapter_records[module._record_key(REVISION)] = record
    adapter_dir = tmp_path / module._adapter_digest(REVISION)
    adapter_dir.mkdir()
    instance = engine_class.__new__(engine_class)
    instance._locks = {}
    instance._allocator_lock = asyncio.Lock()
    instance._next_int_id = 1
    instance._registered = {}

    async def fail_path(_record):
        raise RuntimeError("download failed")

    instance._adapter_path = fail_path
    result = _run_awaitable(engine_class.settle(instance, record))

    assert result["ok"] is False
    assert not adapter_dir.exists()
    current = module.adapter_records[module._record_key(REVISION)]
    assert current["status"] == "disabled"
    assert current["metadata"]["lifecycle_state"] == "failed"


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
    client, engine_class, monkeypatch, tmp_path
):
    """a failed unload must leave the adapter removable rather than orphaning it.

    the record flips to disabled while weights are loading, so the app unloads what it just
    registered. if that unload fails the adapter stays tracked, so a later unload can still
    reclaim both the slot and its cache.
    """
    module = client.app.state.generated_module
    monkeypatch.setattr(module, "ADAPTER_DIR", str(tmp_path))
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
    reads = iter(
        [
            {**REGISTRATION, "status": "registered"},
            {
                **REGISTRATION,
                "status": "disabled",
                "metadata": {**REGISTRATION["metadata"], "lifecycle_state": "disabled"},
            },
        ]
    )

    async def read(adapter_id):
        return next(reads)

    monkeypatch.setattr(module, "_read", read)
    with pytest.raises(RuntimeError, match="undeployed while it was loading"):
        _run_awaitable(engine_class._load_lora_locked(instance, REGISTRATION))
    # the failed unload must not drop the adapter, or its slot and cache would leak. this record
    # carries no settle attempt, so its incarnation is the bare adapter id.
    assert instance._registered[REVISION] == REVISION
    adapter_dir = tmp_path / module._adapter_digest(REVISION)
    adapter_dir.mkdir()

    unloaded = []

    async def unload_ok(adapter_id, *, expected_incarnation):
        unloaded.append((adapter_id, expected_incarnation))
        return True

    instance._runtime.unload_adapter = unload_ok
    assert _run_awaitable(engine_class._unload_locked(instance, REVISION)) is True
    assert unloaded == [(REVISION, REVISION)]
    assert REVISION not in instance._registered
    assert not adapter_dir.exists()


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
