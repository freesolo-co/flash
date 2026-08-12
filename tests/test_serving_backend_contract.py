"""Drive the generated Modal app's HTTP surface, with modal and the GPU engine stubbed.

Parsing the generated file proves it is valid python. It does not prove the routes mount, that the
handlers accept what flash actually sends, or that the lifecycle transitions the client waits on
ever happen. Those failures otherwise surface on a real GPU, after a multi-minute cold start.

One caught here already: `from __future__ import annotations` turns every annotation into a string,
and FastAPI resolves them against MODULE globals -- with the fastapi imports inside the app factory,
`request: Request` silently degraded into a required query parameter and every POST 422'd.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import sys
import threading
import time
import types

import pytest

from flash.core.catalog import MODELS
from flash.serve.backend.generate import render_app

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

REVISION = "run-abc@step-10." + "a" * 40
RUN_ID = "run-abc"
BASE_MODEL = "Qwen/Qwen3.5-4B"
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
    """Modal exposes the async form of every Dict method as a `.aio` attribute on the method.

    Modelled here because the app calls Dict exclusively through `.aio` -- a stub offering only the
    blocking form would pass every test while the deployed app raises AttributeError on the first
    request. The blocking form stays available for anything that still uses it.
    """

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    async def aio(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _AsyncIterAio(_Aio):
    """`keys()` is iterated with `async for`, so its `.aio` yields rather than returning."""

    async def aio(self, *args, **kwargs):  # type: ignore[override]
        for item in self._fn(*args, **kwargs):
            yield item


class _FakeDict(dict):
    """modal.Dict, including the atomic insert-if-absent the alias lock is built on."""

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
    """Drive a coroutine to completion from inside a running event loop.

    `spawn` is called from an async request handler, so the loop on this thread is already running
    and `asyncio.run` would refuse. A private loop on its own thread runs the work to completion
    before the caller continues, which keeps the test deterministic.
    """
    error: list[BaseException] = []

    def _target():
        try:
            asyncio.run(_await(awaitable))
        except BaseException as exc:  # surfaced below, never swallowed
            error.append(exc)

    async def _await(value):
        return await value

    thread = threading.Thread(target=_target)
    thread.start()
    thread.join()
    if error:
        raise error[0]


def _run_awaitable_result(awaitable):
    """`_run_awaitable` for a coroutine whose return value the test needs."""
    box: list = []

    async def _capture():
        box.append(await awaitable)

    _run_awaitable(_capture())
    return box[0]


def _stub_modal(monkeypatch, engine_methods, spawned=None, engine_classes=None):
    modal = types.ModuleType("modal")
    # Every `.spawn()` the app makes, so a test can assert that a settle was (or was not) driven.
    spawned = [] if spawned is None else spawned
    # The real classes `@app.cls` was applied to. The handle below stands in for the GPU, so the
    # class body is otherwise unreachable -- and it holds the vLLM call the engine tests assert on.
    engine_classes = [] if engine_classes is None else engine_classes

    class _Named:
        @classmethod
        def from_name(cls, *args, **kwargs):
            return cls()

    class _Image(_Named):
        @classmethod
        def from_registry(cls, *args, **kwargs):
            return cls()

        @classmethod
        def debian_slim(cls, *args, **kwargs):
            return cls()

        def apt_install(self, *args, **kwargs):
            return self

        def pip_install(self, *args, **kwargs):
            return self

        def env(self, *args, **kwargs):
            return self

    class _EngineHandle:
        """Modal exposes a generator method as `.remote_gen`, NOT `.remote`.

        Modelled because Modal decides generator-ness per function and raises InvalidError on the
        wrong one, so a stub offering `.remote` for everything would pass while the deployed app
        fails on the first streaming request.
        """

        def __getattr__(self, name):
            method = engine_methods[name]
            if inspect.isasyncgenfunction(method):
                return types.SimpleNamespace(remote_gen=types.SimpleNamespace(aio=method))
            return types.SimpleNamespace(remote=types.SimpleNamespace(aio=method))

    class _Spawnable:
        """A Modal function handle.

        `spawn` runs the work server-side and outlives the request that scheduled it; the local
        stand-in runs it to completion immediately, which is the same observable outcome for a
        client that polls. Modelling it at all matters: registration settles through `spawn`, so a
        stub exposing only `__call__` would make every lifecycle test pass against an app whose
        adapters never reach `ready` once deployed.
        """

        def __init__(self, fn):
            self._fn = fn

        def __call__(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

        def spawn(self, *args, **kwargs):
            spawned.append((self._fn.__name__, args, kwargs))
            return self.local(*args, **kwargs)

        def local(self, *args, **kwargs):
            """Run the function body here, the way `.local()` does on a real Modal handle."""
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
                return lambda *a, **k: _EngineHandle()

            return decorate

        def function(self, *args, **kwargs):
            return _Spawnable

    modal.App = _App
    modal.Dict = _FakeDict
    modal.Volume = _Named
    modal.Secret = _Named
    modal.Image = _Image
    modal.parameter = lambda **kwargs: None
    for hook in ("enter", "method", "concurrent", "asgi_app"):
        setattr(modal, hook, lambda *a, **k: lambda fn: fn)
    monkeypatch.setitem(sys.modules, "modal", modal)


@pytest.fixture
def client(monkeypatch, tmp_path):
    loaded: dict[str, dict] = {}

    async def register(record):
        if record.get("repo_id") == BAD_REPO:
            return {"ok": False, "failure": "ValueError: adapter rank 512 exceeds max_lora_rank"}
        loaded[record["adapter_id"]] = record
        return {"ok": True}

    async def unregister(adapter_id):
        loaded.pop(adapter_id, None)
        return {"ok": True}

    async def generate(payload, record):
        return {
            "text": f"served by {record['adapter_id']}",
            "finish_reason": "stop",
            "prompt_tokens": 7,
            "completion_tokens": 4,
        }

    async def generate_stream(payload, record):
        # Deltas, the way the real generator yields them: pieces plus a final finish_reason frame.
        for piece in ("served ", "by ", record["adapter_id"]):
            yield {"delta": piece, "finish_reason": None}
        yield {"delta": "", "finish_reason": "stop"}

    spawned: list = []
    engine_classes: list = []
    _stub_modal(
        monkeypatch,
        {
            "register": register,
            "unregister": unregister,
            "generate": generate,
            "generate_stream": generate_stream,
        },
        spawned=spawned,
        engine_classes=engine_classes,
    )
    source = render_app(MODELS[BASE_MODEL])
    module = types.ModuleType("generated_serving_app")
    exec(compile(source, str(tmp_path / "app.py"), "exec"), module.__dict__)
    module.spawned = spawned
    module.engine_classes = engine_classes
    test_client = TestClient(module.api())
    # Reach the generated module from a test: its Dict and its functions ARE the durable state,
    # so lifecycle races have to be driven through them rather than simulated alongside.
    test_client.app.state.generated_module = module
    return test_client


@pytest.fixture
def engine(client):
    """The real `Engine` class the generated app defines, not the fixture's stand-in handle.

    `@app.cls` replaces the class with a Modal handle, so the GPU-side code is unreachable through
    the HTTP client. Everything the class touches at generation time (vLLM, the tokenizer, the
    adapter load) is substituted per test, which leaves the method's own logic as what runs.
    """
    classes = client.app.state.generated_module.engine_classes
    assert len(classes) == 1, f"expected one @app.cls engine, got {len(classes)}"
    return classes[0]


def _generate_with(monkeypatch, engine_class, record, payload=None):
    """Run the real `Engine.generate` against a stubbed vLLM and return its SamplingParams."""
    seen: list = []

    class _SamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _StructuredOutputsParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = _SamplingParams
    sampling_params = types.ModuleType("vllm.sampling_params")
    sampling_params.StructuredOutputsParams = _StructuredOutputsParams
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)

    async def _generated(prompt, sampling, request_id, lora_request=None):
        seen.append(sampling)
        yield types.SimpleNamespace(
            outputs=[types.SimpleNamespace(text="ok", finish_reason="stop", token_ids=[1, 2])],
            prompt_token_ids=[1],
        )

    instance = engine_class.__new__(engine_class)
    instance.engine = types.SimpleNamespace(generate=_generated)
    instance.tokenizer = types.SimpleNamespace(apply_chat_template=lambda *a, **k: [1, 2, 3])

    async def _lora_request(_record):
        return None

    instance._lora_request = _lora_request
    _run_awaitable(engine_class.generate(instance, dict(payload or {}), record))
    assert seen, "the engine never reached vLLM"
    return seen[-1]


def _lifecycle(
    client, adapter_id: str, *, until=("ready", "failed"), tries: int = 100
) -> str | None:
    """Poll a record the way flash's client does, until it settles."""
    state = None
    for _ in range(tries):
        response = client.get(f"/adapters/{adapter_id}")
        record = response.json().get("adapter") or {}
        state = (record.get("metadata") or {}).get("lifecycle_state")
        if state in until:
            return state
        time.sleep(0.01)
    return state


def _register_and_ready(client) -> None:
    assert client.post("/adapters", json=REGISTRATION).status_code in (200, 202)
    assert _lifecycle(client, REVISION) == "ready"


def test_healthz_advertises_the_required_capabilities(client):
    payload = client.get("/healthz").json()
    assert {"immutable_adapter_revisions", "alias_compare_and_swap"} <= set(payload["capabilities"])


def test_registration_is_accepted_and_reaches_ready(client):
    """The POST must not 422.

    flash sends a JSON body plus its internal-key header and nothing in the query string, so a
    handler that requires a query parameter rejects every deploy before the GPU is ever touched.
    """
    _register_and_ready(client)


def test_readback_carries_the_identity_the_client_cross_checks(client):
    """flash re-reads the record after an ambiguous 5xx and compares it field by field.

    A record that drops or renames one of these looks like a DIFFERENT artifact under the same
    revision id, which the client treats as an immutability violation and refuses to deploy.
    """
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
        assert record.get(field) == REGISTRATION.get(field), field
    for field in ("record_type", "run_id", "checkpoint_step", "hf_revision"):
        assert record["metadata"].get(field) == REGISTRATION["metadata"][field], field


def test_reregistering_identical_content_is_idempotent(client):
    """A retried registration must not fail: flash retries after an ambiguous 5xx."""
    _register_and_ready(client)
    assert client.post("/adapters", json=REGISTRATION).status_code in (200, 202)


def test_reregistering_different_content_under_one_revision_is_a_conflict(client):
    """This IS `immutable_adapter_revisions`: one revision id, exactly one artifact, forever."""
    _register_and_ready(client)
    mutated = {**REGISTRATION, "repo_id": "acme/somewhere-else"}
    assert client.post("/adapters", json=mutated).status_code == 409


def test_chat_reports_which_immutable_revision_answered(client):
    """A plane smoke-tests the revision before flipping its alias, and rejects a response that
    does not say which artifact served it -- as a top-level `freesolo` object AND as the matching
    `X-Freesolo-*` headers. Without both, generation succeeds but every deploy fails at smoke.

    Reported from the resolved record, so asking by alias still names the revision.
    """
    _register_and_ready(client)
    assert (
        client.post(
            f"/adapters/{REVISION}/activate", json={"expected_adapter_revision": None}
        ).status_code
        == 200
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": RUN_ID, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
    )
    assert response.status_code == 200
    payload = response.json()
    provenance = payload.get("freesolo")
    assert isinstance(provenance, dict), "chat response carried no freesolo provenance object"
    assert provenance["adapter_revision"] == REVISION
    assert provenance["checkpoint"] == REGISTRATION["checkpoint"]
    assert provenance["hf_revision"] == REVISION.rsplit(".", 1)[-1]
    for header, expected in (
        ("X-Freesolo-Adapter-Revision", provenance["adapter_revision"]),
        ("X-Freesolo-Checkpoint", provenance["checkpoint"]),
        ("X-Freesolo-HF-Revision", provenance["hf_revision"]),
    ):
        assert response.headers.get(header) == expected, header


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ({"json": {"type": "object"}}, {"json": {"type": "object"}}),
        # Registered without a grammar: sampling must stay unconstrained rather than be handed an
        # empty constraint object, which vLLM would try to compile.
        (None, None),
        ({"json": None, "regex": None}, None),
    ],
)
def test_a_registered_grammar_reaches_the_sampling_params(monkeypatch, engine, stored, expected):
    """The run's declared output contract has to reach SamplingParams.

    Stored on the record at registration but never applied, the adapter serves unconstrained text
    while claiming to honour the constraint -- and a plane's deployment smoke validates the
    constraint before it activates the revision.

    This drives the real `Engine.generate`, not the fixture's stub: the constraint is built inside
    that method, so a test that goes through the stubbed engine cannot see it at all.
    """
    record = {**REGISTRATION, "structured_outputs": stored}
    sampling = _generate_with(monkeypatch, engine, record)
    structured = sampling.structured_outputs
    if expected is None:
        assert structured is None, "an absent grammar became a constraint"
    else:
        assert structured is not None, "the registered grammar did not reach the sampling params"
        assert structured.json == expected["json"]


def test_generation_is_constrained_by_the_record_not_the_request(monkeypatch, engine):
    """A caller must not be able to loosen (or impose) the run's declared output contract.

    The grammar comes off the stored record; a request field of the same name is not a channel for
    changing it, or a deployed adapter's contract is whatever the last caller asked for.
    """
    record = {**REGISTRATION, "structured_outputs": {"json": {"type": "object"}}}
    sampling = _generate_with(
        monkeypatch, engine, record, payload={"structured_outputs": {"regex": "^liberated$"}}
    )
    assert sampling.structured_outputs is not None
    assert sampling.structured_outputs.json == {"type": "object"}
    assert getattr(sampling.structured_outputs, "regex", None) is None


def test_reregistering_a_different_grammar_under_one_revision_is_a_conflict(client):
    """`structured_outputs` is identity too: the client cross-checks it on readback.

    Left out of the fingerprint, the same revision id could be re-registered with a different
    grammar and accepted as identical -- the client would then see a field it did not send.
    """
    _register_and_ready(client)
    mutated = {**REGISTRATION, "structured_outputs": '{"type": "object"}'}
    assert client.post("/adapters", json=mutated).status_code == 409


def test_undeploy_is_not_undone_by_a_load_that_finishes_afterwards(client):
    """A settle that outlives the DELETE must not resurrect the revision.

    The load runs server-side and can finish after undeploy has already written `disabled`. If
    settle overwrites that with `ready`, undeploy silently does not stick and the revision can be
    activated again.
    """
    _register_and_ready(client)
    assert client.delete(f"/adapters/{REVISION}").status_code == 200
    record = client.get(f"/adapters/{REVISION}").json()["adapter"]
    assert record["status"] == "disabled"

    # Re-run the settle the way a slow in-flight load would land, then assert it stood down.
    module = client.app.state.generated_module
    module.settle_adapter.local(dict(REGISTRATION))
    after = client.get(f"/adapters/{REVISION}").json()["adapter"]
    assert after["status"] == "disabled", "a late settle resurrected an undeployed revision"
    assert (after.get("metadata") or {}).get("lifecycle_state") == "disabled"


def test_a_registration_stuck_at_registered_is_retried_not_stranded(client):
    """Re-registering identical content must re-drive a load that never reached a terminal state.

    This is the client's only recovery path after an ambiguous failure: if the first settle died
    without writing `ready` or `failed`, an early return leaves the revision at `registered`
    forever and every retry is a no-op.
    """
    module = client.app.state.generated_module
    module.spawned.clear()
    assert client.post("/adapters", json=REGISTRATION).status_code in (200, 202)
    assert _lifecycle(client, REVISION) == "ready"

    # Force the stuck state a dead settle would leave behind, then retry.
    stuck = client.get(f"/adapters/{REVISION}").json()["adapter"]
    stuck["status"] = "registered"
    stuck["metadata"] = {**stuck["metadata"], "lifecycle_state": "registered"}
    module.adapter_records[module._record_key(REVISION)] = stuck
    module.spawned.clear()

    assert client.post("/adapters", json=REGISTRATION).status_code in (200, 202)
    assert module.spawned, "a stuck registration was not re-settled on retry"


def test_redeploying_the_same_checkpoint_after_undeploy_works(client):
    """Undeploy then deploy the same checkpoint again must succeed.

    A revision id is deterministic in (run_id, step, hf_revision), so redeploying an unchanged
    checkpoint reuses the exact id, and its record is `disabled` from the undeploy. Returning that
    record unchanged is a permanent dead end: the client reads `status: "disabled"` and raises a
    readiness failure (deploy.py:622), so the run can never be redeployed unless the artifact
    gains a new commit.

    This is distinct from a late settle racing an undeploy: that arrives with no fresh
    registration behind it, and must still stand down.
    """
    _register_and_ready(client)
    assert client.delete(f"/adapters/{REVISION}").status_code == 200
    assert client.get(f"/adapters/{REVISION}").json()["adapter"]["status"] == "disabled"

    assert client.post("/adapters", json=REGISTRATION).status_code in (200, 202)
    assert _lifecycle(client, REVISION) == "ready", (
        "an explicit redeploy of the same checkpoint stayed disabled forever"
    )
    assert (
        client.post(
            f"/adapters/{REVISION}/activate", json={"expected_adapter_revision": None}
        ).status_code
        == 200
    )


def test_a_terminal_record_is_not_resettled_on_reregistration(client):
    """The retry above must not reload a revision that already settled.

    Re-settling a `ready` record would reload a live adapter, and re-settling a `disabled` one
    would undo an undeploy.
    """
    _register_and_ready(client)
    module = client.app.state.generated_module
    module.spawned.clear()
    assert client.post("/adapters", json=REGISTRATION).status_code in (200, 202)
    assert not module.spawned, "a settled revision was re-settled"


def test_a_stale_settle_cannot_overwrite_a_newer_attempts_result(client):
    """Two settles for one revision must not commit out of order.

    Reachable without any crash: a user whose `models deploy` exceeded the client's five-minute
    readiness budget retries while the first load is still running, so two settles for the same
    revision are in flight on separate containers. Unguarded, a stale FAILURE landing after a fresh
    success pins the record to `failed` -- and re-registration treats a terminal state as something
    to reset rather than a load to retry, so the run needs a new artifact to recover.

    Driven by replaying the first attempt's record (carrying its now-superseded attempt id) after a
    second registration has already settled.
    """
    _register_and_ready(client)
    module = client.app.state.generated_module
    first_attempt = dict(client.get(f"/adapters/{REVISION}").json()["adapter"])
    # Re-register so a NEW attempt id is stamped and settles to ready, retiring the one above.
    module.adapter_records[module._record_key(REVISION)] = {
        **first_attempt,
        "status": "registered",
        "metadata": {**first_attempt["metadata"], "lifecycle_state": "registered"},
    }
    assert client.post("/adapters", json=REGISTRATION).status_code in (200, 202)
    assert _lifecycle(client, REVISION) == "ready"

    # The older attempt now reports the transient failure it hit, minutes late.
    async def _failing_register(record):
        return {"ok": False, "failure": "transient: connection reset"}

    module.Engine = lambda: types.SimpleNamespace(
        register=types.SimpleNamespace(remote=types.SimpleNamespace(aio=_failing_register)),
        unregister=types.SimpleNamespace(
            remote=types.SimpleNamespace(aio=lambda *a, **k: _noop_coroutine())
        ),
    )
    module.settle_adapter.local(first_attempt)
    assert client.get(f"/adapters/{REVISION}").json()["adapter"]["status"] == "ready", (
        "a retired settle attempt overwrote the current one's result"
    )


async def _noop_coroutine():
    return {"ok": True}


def test_activation_rechecks_readiness_inside_the_lock(client):
    """The revision's readiness must be re-read inside the critical section.

    Checked only outside it, an undeploy can land between the check and the lock. Undeploy leaves
    the alias `disabled`, so `previous` collapses to None and a first activation's
    `expected_adapter_revision: null` still matches -- the alias is written back to `ready`
    pointing at a revision that is disabled and evicted from the GPU. That is a 200 the client
    trusts, followed by a run that cannot answer a single request.

    Driven by disabling the revision while the activation is parked on the run lock.
    """
    _register_and_ready(client)
    module = client.app.state.generated_module
    read_barrier = threading.Event()
    released = threading.Event()
    original = module._run_lock

    @contextlib.asynccontextmanager
    async def _slow_lock(run_id):
        async with original(run_id):
            # Inside the lock, before the handler re-reads: stand in for the undeploy that
            # completed after the outer readiness check.
            read_barrier.set()
            await asyncio.get_running_loop().run_in_executor(None, released.wait, 5)
            yield

    def _undeploy_once():
        read_barrier.wait(5)
        record = module.adapter_records[module._record_key(REVISION)]
        record["status"] = "disabled"
        record["metadata"] = {**record["metadata"], "lifecycle_state": "disabled"}
        module.adapter_records[module._record_key(REVISION)] = record
        released.set()

    module._run_lock = _slow_lock
    worker = threading.Thread(target=_undeploy_once)
    worker.start()
    try:
        response = client.post(
            f"/adapters/{REVISION}/activate", json={"expected_adapter_revision": None}
        )
    finally:
        released.set()
        worker.join()
        module._run_lock = original
    assert response.status_code == 409, (
        f"activation returned {response.status_code} for a revision undeployed under the lock"
    )
    assert module.adapter_records[module._record_key(RUN_ID)]["status"] == "disabled", (
        "an undeployed run was made live again by a concurrent activation"
    )


def test_chat_fails_closed_on_a_terminal_revision(client):
    """A dead revision is a hard miss, not a slow load.

    The retryable 503 tells a plane's deployment smoke to keep polling. Returned for a revision
    that already reached `disabled` or `failed`, the smoke polls a permanently dead adapter until
    its budget expires and then reports a TIMEOUT -- hiding the load failure that actually
    happened behind a symptom that reads like a slow cold start.
    """
    _register_and_ready(client)
    module = client.app.state.generated_module
    client.post(f"/adapters/{REVISION}/activate", json={"expected_adapter_revision": None})
    record = module.adapter_records[module._record_key(REVISION)]
    record["metadata"] = {**record["metadata"], "lifecycle_state": "failed"}
    module.adapter_records[module._record_key(REVISION)] = record

    response = client.post(
        "/v1/chat/completions",
        json={"model": RUN_ID, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
    )
    assert response.status_code == 404, (
        f"a terminally failed revision answered {response.status_code}, which the smoke retries"
    )


def test_a_streaming_request_gets_sse_not_a_json_body(client):
    """`stream: true` on an OpenAI-compatible route must produce SSE.

    Ignored, the route returns a one-shot JSON body: a direct OpenAI client parses it as an event
    stream and fails outright, and long generations produce no incremental output. flash's own
    client carries a JSON fallback, which is exactly why this would look fine in-house while
    breaking every other client.
    """
    _register_and_ready(client)
    client.post(f"/adapters/{REVISION}/activate", json={"expected_adapter_revision": None})
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": RUN_ID,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream"), (
        f"stream: true returned {response.headers['content-type']}, which an OpenAI client "
        "cannot parse as events"
    )
    frames = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert response.text.rstrip().endswith("data: [DONE]"), "the stream never terminated"
    assert all(frame["object"] == "chat.completion.chunk" for frame in frames)
    # Deltas, not cumulative text: vLLM yields the whole answer so far each step, so forwarding it
    # unchanged makes a concatenating client render the response duplicated quadratically.
    content = "".join(frame["choices"][0]["delta"].get("content") or "" for frame in frames)
    assert content == f"served by {REVISION}", f"reassembled stream was {content!r}"
    assert frames[-1]["choices"][0]["finish_reason"] == "stop", "no chunk carried a finish reason"
    # Provenance has to reach a streaming caller too -- it never sees the JSON body, so otherwise
    # which artifact answered would depend on how the caller asked.
    assert frames[0]["freesolo"]["adapter_revision"] == REVISION
    assert response.headers["X-Freesolo-Adapter-Revision"] == REVISION


def test_an_image_request_is_refused_rather_than_answered_blind(client):
    """An image-bearing request must not be served as if it were text.

    `flash env eval` sends images as OpenAI `image_url` blocks and every catalog model is
    image-capable, so this arrives in ordinary use. The engine renders messages to token ids and
    passes nothing else: no image is fetched or decoded and no `multi_modal_data` reaches vLLM.
    Answered anyway, the model replies having never seen the image and the eval scores those
    replies as valid -- a silently invalid result, which is worse than a refusal.
    """
    _register_and_ready(client)
    client.post(f"/adapters/{REVISION}/activate", json={"expected_adapter_revision": None})
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": RUN_ID,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is in this image?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}},
                    ],
                }
            ],
            "max_tokens": 16,
        },
    )
    assert response.status_code == 400, (
        f"an image request was answered with {response.status_code} by a text-only backend"
    )
    assert "text-only" in response.json()["detail"]


def test_a_text_request_is_unaffected_by_the_image_guard(client):
    """The guard must not reject ordinary structured text content.

    OpenAI content blocks are a list for plain text too, so a guard keyed on "content is a list"
    rather than on the block type would refuse every request `flash models chat` sends.
    """
    _register_and_ready(client)
    client.post(f"/adapters/{REVISION}/activate", json={"expected_adapter_revision": None})
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": RUN_ID,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "max_tokens": 16,
        },
    )
    assert response.status_code == 200, "structured text content was refused as if it were an image"


def test_a_failed_adapter_load_reports_failed_not_ready(client):
    """A bad adapter must fail loudly at registration.

    Reporting ready and only failing on the user's first chat turns a clear deploy error into a
    mystery 500 well after the deploy claimed success.
    """
    bad_revision = "run-bad@final." + "b" * 40
    client.post(
        "/adapters",
        json={
            **REGISTRATION,
            "adapter_id": bad_revision,
            "repo_id": BAD_REPO,
            "checkpoint": "run-bad",
            "metadata": {
                "record_type": "revision",
                "run_id": "run-bad",
                "checkpoint_step": None,
                "hf_revision": "b" * 40,
            },
        },
    )
    assert _lifecycle(client, bad_revision) == "failed"


def test_chat_is_refused_until_the_alias_is_activated(client):
    """Registration alone must not take traffic; activation is what publishes a revision."""
    _register_and_ready(client)
    response = client.post(
        "/v1/chat/completions",
        json={"model": RUN_ID, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code in (404, 503)


def test_activation_returns_the_provenance_the_client_validates(client):
    """flash rejects the activation unless every one of these matches what it asked for."""
    _register_and_ready(client)
    response = client.post(
        f"/adapters/{REVISION}/activate", json={"expected_adapter_revision": None}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["adapter_id"] == RUN_ID
    assert payload["target_adapter_revision"] == REVISION
    assert payload["previous_adapter_revision"] is None
    assert payload["checkpoint"] == REGISTRATION["checkpoint"]
    assert isinstance(payload["updated_at"], str)
    assert payload["updated_at"].strip()


def test_a_stale_compare_and_swap_is_rejected(client):
    """The second activation carries an expectation the alias has already moved past.

    Accepting it would let two concurrent deploys of one run silently overwrite each other, which
    is exactly what advertising `alias_compare_and_swap` promises cannot happen.
    """
    _register_and_ready(client)
    first = client.post(f"/adapters/{REVISION}/activate", json={"expected_adapter_revision": None})
    assert first.status_code == 200
    stale = client.post(f"/adapters/{REVISION}/activate", json={"expected_adapter_revision": None})
    assert stale.status_code == 409


def test_releasing_the_run_lock_removes_only_this_holders_lease(client):
    """A lock release must be conditional on still owning the lease.

    Two waiters can read the same expired lease before either acts on it. With an unconditional
    delete, the second one removes the lease the FIRST has already acquired and then takes its
    own, putting both inside the critical section -- the lost update `alias_compare_and_swap`
    promises cannot happen. Modal Dicts are atomic per key, so the guard is a re-read of the
    value, not an expiry check. Driven here by letting another lease land while the lock is held.
    """
    module = client.app.state.generated_module
    key = module._lock_key(RUN_ID)
    newer = {"token": "another-holder", "expires_at": time.time() + 30}

    async def _release_over_a_newer_lease():
        async with module._run_lock(RUN_ID):
            module.adapter_records[key] = newer

    _run_awaitable(_release_over_a_newer_lease())
    assert module.adapter_records.get(key) == newer, "release deleted a lease it did not hold"


def test_undeploy_cannot_be_undone_by_a_concurrent_activation(client):
    """Undeploy must disable the alias under the SAME per-run lock `activate` takes.

    Outside it, an in-flight activation writes the alias back to `ready` after the delete disabled
    it, so undeploy returns success while the run stays callable by its alias. Driven by holding
    the run lock and observing that the DELETE waits for it rather than walking straight through.
    """
    _register_and_ready(client)
    # Activate first, so `ready` is the state a lock-less delete would visibly destroy. An alias
    # is created disabled, so asserting on `disabled` alone would pass without the delete running.
    assert (
        client.post(
            f"/adapters/{REVISION}/activate", json={"expected_adapter_revision": None}
        ).status_code
        == 200
    )
    module = client.app.state.generated_module
    key = module._lock_key(RUN_ID)
    # A live lease held by someone else. If remove takes the lock it cannot proceed while this is
    # here; if it does not, it disables the alias regardless.
    module.adapter_records[key] = {"token": "activation-in-flight", "expires_at": time.time() + 30}
    module.LOCK_TTL_SECONDS = 0.5

    response = client.delete(f"/adapters/{REVISION}")
    assert response.status_code == 409, (
        f"undeploy did not take the run lock (got {response.status_code}); a concurrent "
        f"activation could write the alias back to ready after this disabled it"
    )
    alias = client.get(f"/adapters/{RUN_ID}").json()["adapter"]
    assert alias["status"] == "ready", "the alias was mutated without holding the run lock"


def test_a_disabled_revision_is_dropped_from_the_engine(client):
    """Undeploy must remove the LoRA from vLLM, not just from the app's own dict.

    The engine's LoRA cache is bounded by max_loras, so adapters left resident across repeated
    deploy/undeploy cycles eventually evict live ones or fail new loads until the container
    restarts.
    """
    _register_and_ready(client)
    module = client.app.state.generated_module
    engine_class = module.engine_classes[0]
    removed: list[int] = []

    instance = engine_class.__new__(engine_class)
    instance._locks = {}
    instance._int_ids = {}
    instance._loaded = {REVISION: object()}

    async def _remove_lora(lora_id):
        removed.append(lora_id)
        return True

    instance.engine = types.SimpleNamespace(remove_lora=_remove_lora)
    _run_awaitable(engine_class.unregister(instance, REVISION))
    assert removed == [module._lora_int_id(REVISION)], "the LoRA was never removed from vLLM"
    assert REVISION not in instance._loaded


def test_a_run_lock_is_released_by_its_owner(client):
    """The guard above must not leave the lock held forever, which would wedge every later deploy.

    Paired with the test above deliberately: conditional-delete is only correct if the ordinary
    release still clears the key.
    """
    module = client.app.state.generated_module
    key = module._lock_key(RUN_ID)
    # An expired lease from a container that died mid-update is reclaimed, not waited on.
    module.adapter_records[key] = {"token": "dead-holder", "expires_at": time.time() - 60}

    async def _acquire_and_release():
        async with module._run_lock(RUN_ID):
            assert (module.adapter_records.get(key) or {}).get("token") != "dead-holder"

    _run_awaitable(_acquire_and_release())
    assert module.adapter_records.get(key) is None, "the lock outlived its holder"


class _StaleReadBarrier(_FakeDict):
    """The app's Dict, with a two-party rendezvous on every read that returns an expired lease.

    A lock race is not reachable by calling the lock twice in a row -- it needs two waiters pinned
    in the same window, and which window matters is exactly what a fix changes. Pairing on the
    OBSERVATION (a read that saw a stale lease) rather than on a call count keeps the barrier
    meaningful across implementations: it pins both waiters wherever they judge the same dead
    holder reclaimable, which is the only place a compare-less delete can lose an update.

    A waiter that never finds a partner fails the test on a timeout instead of hanging it.
    """

    @staticmethod
    def _is_stale(value) -> bool:
        expires_at = value.get("expires_at") if isinstance(value, dict) else None
        return isinstance(expires_at, (int, float)) and expires_at < time.time()

    @property
    def get(self):
        parked = self.__dict__.setdefault("_parked", [])

        async def _read(*args, **kwargs):
            value = dict.get(self, *args, **kwargs)
            if self._is_stale(value):
                if parked:
                    parked.pop().set_result(None)
                else:
                    waiter = asyncio.get_running_loop().create_future()
                    parked.append(waiter)
                    await asyncio.wait_for(waiter, timeout=5)
            return value

        # Only `.aio` is gated. The blocking form stays a plain read so a test can inspect the
        # lock without parking itself on the barrier it is trying to observe.
        gated = _Aio(lambda *args, **kwargs: dict.get(self, *args, **kwargs))
        gated.aio = _read
        return gated


def test_undeploy_does_not_scan_the_whole_keyspace_under_the_lock(client):
    """Work under the run lock must not grow with the number of adapters on the app.

    The lease expires on a fixed TTL and cannot be renewed atomically -- modal.Dict has no
    compare-and-update, and pop-then-put vacates the key between the two calls, so a waiter can win
    the insert in that gap and enter alongside the live holder. The TTL is therefore only safe if
    every critical section is bounded, which makes "no unbounded work under the lock" the real
    invariant rather than an optimization.

    Undeploy is where it was violated: it walked the entire Dict, so hold time grew with every
    adapter ever registered. Asserted by counting reads against a keyspace padded with unrelated
    runs -- a scan's cost tracks that padding, an indexed lookup's does not.
    """
    _register_and_ready(client)
    module = client.app.state.generated_module
    for i in range(50):
        module.adapter_records[module._record_key(f"other-run-{i}")] = {
            "adapter_id": f"other-run-{i}",
            "status": "ready",
            "metadata": {"record_type": "alias", "run_id": f"other-run-{i}"},
        }
    reads: list[str] = []
    records = module.adapter_records

    class _CountingDict(type(records)):
        @property
        def get(self):
            def _read(key, default=None):
                reads.append(key)
                return dict.get(self, key, default)

            return _Aio(_read)

    counting = _CountingDict()
    counting.update(records)
    module.adapter_records = counting
    assert client.delete(f"/adapters/{REVISION}").status_code == 200
    assert len(reads) < 20, (
        f"undeploy made {len(reads)} reads against a 50-run keyspace, so it is scanning rather "
        "than using the run index, and hold time grows with the app's total adapter count"
    )


def test_an_interrupted_first_registration_can_still_be_activated(client):
    """A revision whose alias write never landed must be repairable by re-registering.

    Registration writes the revision and the alias as two operations. A container that dies between
    them leaves a revision with no run alias -- and every retry took the identical-registration
    path and returned BEFORE the alias creation, so the revision could reach `ready` while
    activation failed forever with "run alias is missing". A permanently undeployable run,
    recoverable only by producing a new artifact.
    """
    _register_and_ready(client)
    module = client.app.state.generated_module
    # Exactly the state an interruption leaves: revision present, alias never written.
    del module.adapter_records[module._record_key(RUN_ID)]

    assert client.post("/adapters", json=REGISTRATION).status_code in (200, 202)
    assert (
        client.post(
            f"/adapters/{REVISION}/activate", json={"expected_adapter_revision": None}
        ).status_code
        == 200
    ), "a revision left without its alias could never be activated"


def test_an_undeploy_landing_mid_settle_is_not_overwritten(client):
    """Settle's decision and its write must be one atomic step, not a read then a write.

    `test_undeploy_is_not_undone_by_a_load_that_finishes_afterwards` covers the settle that starts
    after the DELETE has fully landed. It cannot see the narrower race: the DELETE arriving between
    settle's read of the record and its write of the result. Read-then-write judges `ready` against
    a record that was still live, and the write then lands over `disabled` -- undeploy returns
    success while the revision goes back to serving. Taking the run lock around both closes it,
    because undeploy holds that same lock for its whole disable pass.

    Driven by hooking settle's read of the record and landing the disable right after it -- but
    only when the run lock is unheld, which is precisely when a real undeploy could have got in.
    Under the fix the read happens inside the lock, so the lease is present, the undeploy is
    excluded, and the record legitimately reaches `ready`.
    """
    _register_and_ready(client)
    module = client.app.state.generated_module
    in_flight = dict(module.adapter_records[module._record_key(REVISION)])
    record_key = module._record_key(REVISION)
    lock_key = module._lock_key(RUN_ID)
    landed: list[str] = []
    records = module.adapter_records

    class _UndeployMidSettle(type(records)):
        @property
        def get(self):
            async def _read(key, default=None):
                value = dict.get(self, key, default)
                if key == record_key and not landed and not self._lock_is_held():
                    self._disable()
                    landed.append(key)
                return value

            gated = _Aio(lambda key, default=None: dict.get(self, key, default))
            gated.aio = _read
            return gated

        def _lock_is_held(self) -> bool:
            lease = dict.get(self, lock_key)
            expires_at = lease.get("expires_at") if isinstance(lease, dict) else None
            return isinstance(expires_at, (int, float)) and expires_at > time.time()

        def _disable(self) -> None:
            record = dict.get(self, record_key)
            self[record_key] = {
                **record,
                "status": "disabled",
                "metadata": {**record["metadata"], "lifecycle_state": "disabled"},
            }

    hooked = _UndeployMidSettle()
    hooked.update(records)
    module.adapter_records = hooked
    try:
        module.settle_adapter.local(in_flight)
        after = hooked[record_key]
    finally:
        module.adapter_records = records

    if landed:
        assert after["status"] == "disabled", (
            "an undeploy landed between settle's read and its write, and settle's `ready` "
            "overwrote it -- the revision is serving again after a successful undeploy"
        )
    else:
        assert after["status"] == "ready", (
            "the run lock excluded the undeploy, so this settle's result must stand"
        )


def test_a_cold_adapter_load_does_not_block_a_resident_one(client, monkeypatch):
    """Loading one adapter must not stall generations for adapters already resident.

    A first load downloads from HF and calls add_lora, which takes minutes. Held under one
    container-wide lock, every concurrent generation queues behind it -- including requests for
    adapters that are already loaded and only need to read a dict.
    """
    lora = types.ModuleType("vllm.lora.request")
    lora.LoRARequest = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.lora", types.ModuleType("vllm.lora"))
    monkeypatch.setitem(sys.modules, "vllm.lora.request", lora)

    engine_class = client.app.state.generated_module.engine_classes[0]
    instance = engine_class.__new__(engine_class)
    instance._locks = {}
    instance._int_ids = {}
    resident = object()
    instance._loaded = {REVISION: resident}
    started = threading.Event()

    async def _slow_path(record):
        started.set()
        await asyncio.sleep(5)
        return "/tmp/never"

    instance._adapter_path = _slow_path

    async def _resident_while_cold_loads():
        cold = asyncio.create_task(
            engine_class._lora_request(instance, {"adapter_id": "run-other@final." + "c" * 40})
        )
        await asyncio.get_running_loop().run_in_executor(None, started.wait, 5)
        # The cold load is mid-download. A resident adapter must still be served immediately.
        got = await asyncio.wait_for(engine_class._lora_request(instance, dict(REGISTRATION)), 1.0)
        cold.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cold
        return got

    assert _run_awaitable_result(_resident_while_cold_loads()) is resident


def test_two_adapters_sharing_one_lora_int_id_are_refused_not_aliased(client, monkeypatch):
    """A 31-bit LoRA id collision must fail the load, not silently serve the wrong weights.

    vLLM addresses a LoRA only by that int, so two revisions hashing to the same value are one
    adapter as far as `add_lora` and `remove_lora` are concerned: the second registration takes over
    the first's slot, and a chat against either run answers from whichever weights are resident.
    Wrong answers under a correct-looking `ready`, which is worse than a failed deploy.

    Driven by forcing the hash to collide, since a natural collision is birthday-bound at tens of
    thousands of revisions and cannot be produced in a test.
    """
    lora = types.ModuleType("vllm.lora.request")
    lora.LoRARequest = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.lora", types.ModuleType("vllm.lora"))
    monkeypatch.setitem(sys.modules, "vllm.lora.request", lora)

    module = client.app.state.generated_module
    engine_class = module.engine_classes[0]
    instance = engine_class.__new__(engine_class)
    instance._locks = {}
    instance._int_ids = {}
    instance._loaded = {}
    instance._int_ids = {}
    added: list = []

    async def _add_lora(request):
        added.append(request)

    instance.engine = types.SimpleNamespace(add_lora=_add_lora)

    async def _path(record):
        return f"/tmp/{record['adapter_id']}"

    instance._adapter_path = _path
    monkeypatch.setattr(module, "_lora_int_id", lambda adapter_id: 12345)

    first = {"adapter_id": "run-one@final." + "a" * 40}
    second = {"adapter_id": "run-two@final." + "b" * 40}
    _run_awaitable(engine_class._lora_request(instance, first))
    with pytest.raises(RuntimeError, match="collision"):
        _run_awaitable(engine_class._lora_request(instance, second))
    assert len(added) == 1, (
        "a colliding adapter was loaded over the one already holding its lora id"
    )

    # The registration path turns that into a `failed` record with the reason, not a crash.
    result = _run_awaitable_result(engine_class.register(instance, second))
    assert result["ok"] is False
    assert "collision" in result["failure"]


def test_two_adapters_do_not_share_one_download_directory(client):
    """The cache directory must be keyed injectively, not by the truncated LoRA int id.

    Keyed by the 31-bit id, two colliding revisions download into the SAME directory and the second
    overwrites the first's files -- so even a backend that refuses the colliding load has already
    corrupted the resident adapter's weights on disk.
    """
    module = client.app.state.generated_module
    first = "run-one@final." + "a" * 40
    second = "run-two@final." + "b" * 40
    assert module._adapter_digest(first) != module._adapter_digest(second)
    # And it must not be the colliding id itself, whatever that id happens to be.
    assert module._adapter_digest(first) != str(module._lora_int_id(first))


def test_two_cold_adapters_do_not_serialize_behind_each_other(client, monkeypatch):
    """Two different adapters loading at once must download concurrently.

    The lock-free fast path only helps adapters that are ALREADY resident, so it hides this: with
    one lock shared across the container, a second cold adapter waits out the first one's
    multi-minute HF download before its own even begins. That is the ordinary case of two runs
    deploying at the same time -- and the same lock also puts an undeploy of one adapter behind an
    unrelated adapter's load. Keying the lock by adapter id is what separates them.

    Driven by parking the first adapter inside its download and asserting the second still reaches
    its own; under a shared lock the second never gets there.
    """
    lora = types.ModuleType("vllm.lora.request")
    lora.LoRARequest = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.lora", types.ModuleType("vllm.lora"))
    monkeypatch.setitem(sys.modules, "vllm.lora.request", lora)

    engine_class = client.app.state.generated_module.engine_classes[0]
    instance = engine_class.__new__(engine_class)
    instance._locks = {}
    instance._int_ids = {}
    instance._loaded = {}
    first = "run-first@final." + "d" * 40
    second = "run-second@final." + "e" * 40
    reached: list[str] = []
    first_started = threading.Event()

    async def _hanging_download(record):
        reached.append(record["adapter_id"])
        if record["adapter_id"] == first:
            first_started.set()
        await asyncio.sleep(30)
        return "/tmp/never"

    instance._adapter_path = _hanging_download

    async def _both_cold():
        loads = [
            asyncio.create_task(engine_class._lora_request(instance, {"adapter_id": adapter_id}))
            for adapter_id in (first, second)
        ]
        await asyncio.get_running_loop().run_in_executor(None, first_started.wait, 5)
        # Long enough for the second load to be scheduled and run up to its own download -- unless
        # something is holding it back.
        await asyncio.sleep(0.25)
        for load in loads:
            load.cancel()
        for load in loads:
            with contextlib.suppress(asyncio.CancelledError):
                await load

    _run_awaitable(_both_cold())
    assert second in reached, (
        "a second adapter's load never started while an unrelated adapter was downloading, so one "
        "container-wide lock is serializing cold loads that have nothing to do with each other"
    )


def test_two_waiters_reclaiming_one_dead_lease_do_not_both_enter(client):
    """Reclaiming a crashed holder's lease must not let both reclaimers into the run.

    The lock is built on `put(skip_if_exists=True)` because modal.Dict has no compare-and-swap, so
    the release has to be conditional some other way. Read-then-delete is not: the value can change
    between the two calls, and the delete then destroys a lease the caller never inspected. `pop`
    removes and returns in one step, so a wrongly-taken lease is identifiable and gets put back.

    Both waiters are pinned on the same dead lease, so both judge it reclaimable. Whichever wins,
    the other must find the slot taken and wait rather than clear it -- otherwise two deploys of
    one run both commit, which is the lost update `alias_compare_and_swap` promises cannot happen.
    """
    module = client.app.state.generated_module
    key = module._lock_key(RUN_ID)
    records = _StaleReadBarrier()
    records.update(module.adapter_records)
    # A container that died mid-update: the lease is real, and long expired.
    records[key] = {"token": "dead-holder", "expires_at": time.time() - 60}
    module.adapter_records = records

    holders: list[str] = []
    concurrent: list[int] = []

    async def _waiter(name: str):
        async with module._run_lock(RUN_ID):
            holders.append(name)
            concurrent.append(len(holders))
            # Long enough for the other waiter to run its reclaim and a wait loop against a lock
            # this one demonstrably still holds.
            await asyncio.sleep(0.25)
            assert (records.get(key) or {}).get("token") not in (
                None,
                "dead-holder",
            ), "the holder's own lease was cleared while it was inside"
            holders.remove(name)

    async def _both():
        # Gathered inside the coroutine: `gather` binds its future to the loop running at CALL
        # time, which out here is the test's loop rather than the one `_run_awaitable` drives.
        await asyncio.gather(_waiter("a"), _waiter("b"))

    _run_awaitable(_both())
    assert concurrent == [1, 1], f"both reclaimers entered the run lock together: {concurrent}"
    assert records.get(key) is None, "the lock outlived both holders"


def test_chat_resolves_the_run_alias_to_its_immutable_revision(client):
    """Users chat with a run id; the weights must come from the revision it currently targets."""
    _register_and_ready(client)
    client.post(f"/adapters/{REVISION}/activate", json={"expected_adapter_revision": None})
    response = client.post(
        "/v1/chat/completions",
        json={"model": RUN_ID, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == f"served by {REVISION}"
    assert body["usage"]["total_tokens"] == 11


def test_undeploy_disables_the_alias_and_its_revisions(client):
    _register_and_ready(client)
    client.post(f"/adapters/{REVISION}/activate", json={"expected_adapter_revision": None})
    payload = client.delete(f"/adapters/{RUN_ID}").json()
    assert payload["run_id"] == RUN_ID
    assert RUN_ID in payload["disabled_aliases"]
    assert REVISION in payload["disabled_revisions"]


def test_undeploying_an_unknown_run_is_a_clean_404(client):
    """The client maps 404 to "nothing to undeploy" rather than an error."""
    assert client.delete("/adapters/run-never-existed").status_code == 404


def test_a_registration_for_another_base_model_is_refused(client):
    """One app serves one base model; its engine cannot load an adapter trained on a different one."""
    response = client.post("/adapters", json={**REGISTRATION, "base_model": "Qwen/Qwen3.6-27B"})
    assert response.status_code == 409
