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


def _stub_modal(monkeypatch, engine_methods, spawned=None, engine_classes=None, runners=None):
    modal = types.ModuleType("modal")
    # Every `.spawn()` the app makes, so a test can assert that a settle was (or was not) driven.
    spawned = [] if spawned is None else spawned
    # The real classes `@app.cls` was applied to. The handle below stands in for the GPU, so the
    # class body is otherwise unreachable -- and it holds the vLLM call the engine tests assert on.
    engine_classes = [] if engine_classes is None else engine_classes
    # How many Engine containers the app should believe are warm. A test flips this to 0 to model
    # a scaled-to-zero deployment, where an eviction would cold-start a GPU for nothing.
    runners = types.SimpleNamespace(count=1) if runners is None else runners

    async def _engine_stats():
        return types.SimpleNamespace(num_total_runners=runners.count, backlog=0)

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
            # `get_current_stats` alongside `remote`, because Modal's bound method IS a Function
            # and carries both. The app asks it how many containers are warm before paying for a
            # cold start to evict; a stub without it would AttributeError on every undeploy.
            # Defaults to one runner, so the eviction tests still exercise eviction.
            return types.SimpleNamespace(
                remote=types.SimpleNamespace(aio=method),
                get_current_stats=types.SimpleNamespace(aio=_engine_stats),
            )

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
    # Every adapter the app asked the GPU to evict. Reaching the engine at all is the assertion:
    # eviction is skipped when nothing is warm, so "was it called" is the behavior under test.
    unregistered: list[str] = []
    # Every revision whose cached download the app asked to reclaim. Separate from `unregistered`
    # because the two are deliberately different calls: cleanup after a failed load must not touch
    # the GPU, whose int-id claim that failure already released.
    discarded: list[str] = []

    async def register(record):
        if record.get("repo_id") == BAD_REPO:
            return {"ok": False, "failure": "ValueError: adapter rank 512 exceeds max_lora_rank"}
        loaded[record["adapter_id"]] = record
        return {"ok": True}

    async def unregister(adapter_id):
        unregistered.append(adapter_id)
        loaded.pop(adapter_id, None)
        return {"ok": True}

    async def discard_cache(adapter_id):
        discarded.append(adapter_id)
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
    runners = types.SimpleNamespace(count=1)
    _stub_modal(
        monkeypatch,
        {
            "register": register,
            "unregister": unregister,
            "discard_cache": discard_cache,
            "generate": generate,
            "generate_stream": generate_stream,
        },
        spawned=spawned,
        engine_classes=engine_classes,
        runners=runners,
    )
    source = render_app(MODELS[BASE_MODEL])
    module = types.ModuleType("generated_serving_app")
    exec(compile(source, str(tmp_path / "app.py"), "exec"), module.__dict__)
    module.spawned = spawned
    module.engine_classes = engine_classes
    module.runners = runners
    module.unregistered = unregistered
    module.discarded = discarded
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


@pytest.mark.parametrize("limit", [0, -1, "many", 1.5, True])
def test_an_invalid_max_tokens_is_rejected_rather_than_defaulted(client, limit):
    """`max_tokens: 0` must not silently become a 512-token generation.

    `int(payload.get("max_tokens") or 512)` cannot tell an omitted field from an explicit 0, so an
    invalid request became a long billable completion the caller never asked for. A negative went
    through to vLLM and came back as an opaque 500 rather than a client error.

    `True` is in the list because `bool` is an `int` subclass: `max_tokens: true` would otherwise
    pass an `isinstance` check and generate exactly one token.
    """
    _register_and_ready(client)
    client.post(f"/adapters/{REVISION}/activate", json={"expected_adapter_revision": None})
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": RUN_ID,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": limit,
        },
    )
    assert response.status_code == 400, (
        f"max_tokens={limit!r} was accepted with {response.status_code}, so an invalid limit "
        "either became a default-length billable generation or reached vLLM as a 500"
    )
    assert "max_tokens" in response.json()["detail"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", -0.5),
        ("temperature", "hot"),
        ("top_p", 0),
        ("top_p", 1.5),
        ("top_p", -1),
    ],
)
def test_invalid_sampling_values_are_rejected_at_the_boundary(client, field, value):
    """Sampling ranges must be checked before the engine round trip.

    `SamplingParams` raises inside the remote GPU method, so a bad value came back as an opaque 500
    after paying for the call. `top_p: 0` is worse than opaque: it is falsy, so `or 1.0` silently
    rewrote it to 1.0 and the request was served with sampling the caller never asked for.
    """
    _register_and_ready(client)
    client.post(f"/adapters/{REVISION}/activate", json={"expected_adapter_revision": None})
    response = client.post(
        "/v1/chat/completions",
        json={"model": RUN_ID, "messages": [{"role": "user", "content": "hi"}], field: value},
    )
    assert response.status_code == 400, (
        f"{field}={value!r} was accepted with {response.status_code}, so it either reached the "
        "engine as a 500 or was silently rewritten to a different sampling behavior"
    )
    assert field in response.json()["detail"]


@pytest.mark.parametrize(("field", "value"), [("temperature", 0), ("top_p", 1), ("top_p", 0.1)])
def test_valid_sampling_values_are_accepted(client, field, value):
    """The guard must reject bad ranges, not narrow the supported ones.

    `temperature: 0` is greedy decoding and the ordinary case for evaluation, so a guard keyed on
    truthiness rather than on range would reject the most common request there is.
    """
    _register_and_ready(client)
    client.post(f"/adapters/{REVISION}/activate", json={"expected_adapter_revision": None})
    response = client.post(
        "/v1/chat/completions",
        json={"model": RUN_ID, "messages": [{"role": "user", "content": "hi"}], field: value},
    )
    assert response.status_code == 200, f"{field}={value!r} is valid but was rejected"


def test_an_omitted_max_tokens_still_gets_the_default(client):
    """The validation above must reject explicit bad values, not require the field.

    `max_tokens` is optional in the OpenAI schema and flash's own chat path omits it, so a guard
    that demanded it would reject every ordinary request.
    """
    _register_and_ready(client)
    client.post(f"/adapters/{REVISION}/activate", json={"expected_adapter_revision": None})
    response = client.post(
        "/v1/chat/completions",
        json={"model": RUN_ID, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200, "an omitted max_tokens must keep its default"


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


def test_a_registration_landing_before_the_lock_is_still_undeployed(client):
    """The member list must be read inside the lock, not snapshotted before it.

    Read first, it is a snapshot: a registration that COMPLETES in the gap before the lock is taken
    adds its revision to the index, and this pass never sees it. Undeploy then returns 200 with the
    new revision left `ready`, resident on the GPU, and callable by its immutable id -- the exact
    thing undeploy promises to have stopped.

    Driven by registering a second revision at the moment the lock is acquired, which is inside the
    window a pre-lock read would have missed.
    """
    _register_and_ready(client)
    module = client.app.state.generated_module
    second = "run-abc@step-20." + "b" * 40
    original = module._run_lock
    landed: list[str] = []

    @contextlib.asynccontextmanager
    async def _register_then_lock(run_id):
        # Before the lock is held, so a real concurrent registration could genuinely be here.
        if not landed:
            landed.append(second)
            module.adapter_records[module._record_key(second)] = {
                "adapter_id": second,
                "status": "ready",
                "base_model": BASE_MODEL,
                "metadata": {
                    "record_type": "revision",
                    "run_id": RUN_ID,
                    "lifecycle_state": "ready",
                },
            }
            members = module.adapter_records.get(module._members_key(RUN_ID)) or []
            module.adapter_records[module._members_key(RUN_ID)] = [*members, second]
        async with original(run_id):
            yield

    module._run_lock = _register_then_lock
    try:
        response = client.delete(f"/adapters/{REVISION}")
    finally:
        module._run_lock = original
    assert response.status_code == 200
    assert module.adapter_records[module._record_key(second)]["status"] == "disabled", (
        "a revision registered just before the lock survived undeploy as `ready`, so it is still "
        "callable by its immutable id after undeploy reported success"
    )


def test_undeploy_works_when_the_member_index_is_missing(client):
    """A run with no `members:` key must still be disabled.

    Two ways to get there, neither exotic: an app upgraded from the earlier scan-based version has
    no index at all for runs registered before the upgrade, and a container that died between the
    revision write and the alias write left an incomplete one. If the index is treated as the only
    source of truth, `_run_members` returns [] and the disable loop never runs -- so undeploy
    answers 200 with empty `disabled_*` lists while the adapter keeps serving.
    """
    _register_and_ready(client)
    module = client.app.state.generated_module
    del module.adapter_records[module._members_key(RUN_ID)]

    response = client.delete(f"/adapters/{REVISION}")
    assert response.status_code == 200
    assert module.adapter_records[module._record_key(REVISION)]["status"] == "disabled", (
        "undeploy reported success without disabling the record it was handed, so the adapter is "
        "still serving"
    )
    assert response.json()["disabled_revisions"] == [REVISION]


def test_a_registration_landing_during_the_disable_pass_is_still_undeployed(client):
    """Membership must be re-read between passes, not snapshotted once inside the lock.

    Alias-last bounds the DAMAGE of losing the lease mid-pass, but not the omission: a registration
    that lands after the membership read appends to `members:`, and a pass working from the stale
    list never reaches that revision. It settles to `ready`, stays resident, and stays callable by
    its immutable id while undeploy answers 200 -- the same "success over a live run" this endpoint
    exists to rule out, arriving one step later.

    Driven by appending a member during the first pass's writes, which is exactly where a
    concurrent registration lands.
    """
    _register_and_ready(client)
    module = client.app.state.generated_module
    late = "run-abc@step-30." + "c" * 40
    records = module.adapter_records
    landed: list[str] = []

    class _RegisterDuringPass(type(records)):
        def _put(self, key, value, skip_if_exists=False):
            result = super()._put(key, value, skip_if_exists=skip_if_exists)
            # after the first record is disabled, a registration completes and indexes itself.
            if not landed and isinstance(value, dict) and value.get("status") == "disabled":
                landed.append(late)
                dict.__setitem__(
                    self,
                    module._record_key(late),
                    {
                        "adapter_id": late,
                        "status": "ready",
                        "base_model": BASE_MODEL,
                        "metadata": {
                            "record_type": "revision",
                            "run_id": RUN_ID,
                            "lifecycle_state": "ready",
                        },
                    },
                )
                members = dict.get(self, module._members_key(RUN_ID)) or []
                dict.__setitem__(self, module._members_key(RUN_ID), [*members, late])
            return result

    hooked = _RegisterDuringPass()
    hooked.update(records)
    module.adapter_records = hooked
    try:
        response = client.delete(f"/adapters/{RUN_ID}")
        after = hooked[module._record_key(late)]
    finally:
        module.adapter_records = records

    assert response.status_code == 200
    assert landed, "no registration landed during the pass, so this test did not exercise the race"
    assert after["status"] == "disabled", (
        "a revision registered during the disable pass was never revisited, so it is still ready, "
        "still resident, and still callable after undeploy reported success"
    )


def test_undeploying_an_idle_run_does_not_cold_start_the_gpu(client):
    """Eviction must be skipped when every container is scaled to zero.

    Eviction exists to free a `max_loras` slot on a RESIDENT engine. With nothing warm there is no
    slot to free, and `engine.unregister.remote` would start a container -- pulling the weights and
    compiling the base model, minutes of paid GPU time against a 30-minute request timeout, to
    remove an adapter that is not loaded anywhere. Scale-to-zero is the default, so undeploying a
    run nobody has called recently is the ORDINARY case, not an edge one.

    Safe to skip because a container that starts later loads from the durable records, and those
    are already `disabled` by then.
    """
    _register_and_ready(client)
    module = client.app.state.generated_module
    module.runners.count = 0

    response = client.delete(f"/adapters/{RUN_ID}")
    assert response.status_code == 200
    assert module.adapter_records[module._record_key(REVISION)]["status"] == "disabled", (
        "skipping eviction must not skip disabling the record"
    )
    assert module.unregistered == [], (
        "an idle deployment reached the engine to evict an adapter, so undeploy cold-started a "
        "GPU and paid for a weight load to remove something that was not resident"
    )


def test_undeploy_still_evicts_from_a_warm_engine(client):
    """The skip above must not become "never evict".

    A warm container is the case eviction is FOR: the adapter is resident, `max_loras` is bounded,
    and leaving it there across repeated deploy/undeploy cycles eventually evicts live adapters or
    fails new loads until the container recycles.
    """
    _register_and_ready(client)
    module = client.app.state.generated_module
    module.runners.count = 1

    assert client.delete(f"/adapters/{RUN_ID}").status_code == 200
    assert REVISION in module.unregistered, (
        "a warm engine was left holding the disabled adapter, so its max_loras slot never frees"
    )


def test_undeploy_by_run_id_disables_revisions_when_the_index_is_missing(client):
    """The client deletes by RUN id, so the missing-index repair must work on that path.

    `undeploy_adapter` builds its url from the run id and nothing else, so `adapter_id == run_id`
    on every real undeploy and unioning the two adds nothing. With no `members:` key the loop then
    sees only the alias: sibling revisions stay `ready`, stay resident on the GPU, and can be
    reactivated by their immutable ids while undeploy answers 200.

    The sibling test above deletes by REVISION id, where the union does carry the record through --
    which is exactly why it cannot catch this.
    """
    _register_and_ready(client)
    module = client.app.state.generated_module
    del module.adapter_records[module._members_key(RUN_ID)]

    response = client.delete(f"/adapters/{RUN_ID}")
    assert response.status_code == 200
    assert module.adapter_records[module._record_key(REVISION)]["status"] == "disabled", (
        "undeploying by run id left the revision `ready`, so it is still on the GPU and still "
        "callable by its immutable id after undeploy reported success"
    )
    assert REVISION in response.json()["disabled_revisions"]


def test_an_activation_racing_an_expired_undeploy_lease_cannot_leave_a_ready_alias(client):
    """Undeploy must disable the run alias LAST, so losing the lock mid-pass is not corrupting.

    The disable loop costs a Modal round trip per member while the lock's lease does not renew --
    it cannot be renewed atomically on modal.Dict -- so a run with enough members can hand the lock
    to a concurrent `activate` before the pass finishes. Disabling the alias first makes that
    handover corrupting: the activation writes the alias back to `ready` at a revision the pass has
    not reached, the pass then disables that revision and never revisits the alias, and undeploy
    returns 200 leaving a `ready` alias pointing at a disabled, evicted revision. Every request to
    the run 404s while its records claim it is serving.

    Driven by expiring the lease at the moment the alias is written and landing a real activation
    through the app's own endpoint. Alias-last puts that activation BEFORE the alias write, so the
    pass overwrites whatever it pointed at; alias-first puts it after, and nothing revisits it.
    """
    _register_and_ready(client)
    # the alias has to be live for undeploy to write it at all: a never-activated alias is already
    # `disabled` and the loop skips it, which would make this test vacuous in both orderings.
    client.post(f"/adapters/{REVISION}/activate", json={"expected_adapter_revision": None})
    module = client.app.state.generated_module
    second = "run-abc@step-20." + "b" * 40
    module.adapter_records[module._record_key(second)] = {
        "adapter_id": second,
        "status": "ready",
        "base_model": BASE_MODEL,
        "checkpoint": "step-20",
        "metadata": {
            "record_type": "revision",
            "run_id": RUN_ID,
            "lifecycle_state": "ready",
        },
    }
    members = module.adapter_records.get(module._members_key(RUN_ID)) or []
    module.adapter_records[module._members_key(RUN_ID)] = [*members, second]
    alias_key = module._record_key(RUN_ID)
    lock_key = module._lock_key(RUN_ID)
    records = module.adapter_records
    landed: list[str] = []

    class _ActivateOnAliasWrite(type(records)):
        def _put(self, key, value, skip_if_exists=False):
            # the moment the pass reaches the alias, drop its lease and let a real activation in --
            # exactly what a member loop that outruns its TTL hands the next waiter.
            if key == alias_key and not landed:
                landed.append(key)
                dict.pop(self, lock_key, None)
                client.post(
                    f"/adapters/{second}/activate",
                    json={"expected_adapter_revision": None},
                )
            return super()._put(key, value, skip_if_exists=skip_if_exists)

    hooked = _ActivateOnAliasWrite()
    hooked.update(records)
    module.adapter_records = hooked
    try:
        response = client.delete(f"/adapters/{REVISION}")
        alias = hooked[alias_key]
        other = hooked[module._record_key(second)]
    finally:
        module.adapter_records = records

    assert response.status_code == 200
    assert landed, "the alias was never written, so this test did not exercise the race"
    assert other["status"] == "disabled", "a member revision survived undeploy as `ready`"
    assert alias["status"] == "disabled", (
        "an activation landed after undeploy released the alias, and undeploy never revisited "
        "it -- the run reports `ready` while pointing at a disabled, evicted revision"
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


def test_a_revision_that_settles_failed_releases_its_lora_id_claim(client):
    """A claim taken before the download must not survive the revision going terminal.

    Claiming early is what makes the download window safe, and it is also what makes a leak
    permanent: settle records the revision `failed`, undeploy skips already-disabled records, so
    nothing later clears the claim. The id is then refused forever for an adapter that was never
    resident -- and because the id is a hash of the adapter id, the run cannot get a different one
    by retrying.
    """
    module = client.app.state.generated_module
    record = dict(REGISTRATION)
    key = module._lora_id_key(module._lora_int_id(REVISION))
    module.adapter_records[key] = REVISION

    async def _failing_register(_record):
        return {"ok": False, "failure": "RuntimeError: hf download failed"}

    module.Engine = lambda: types.SimpleNamespace(
        register=types.SimpleNamespace(remote=types.SimpleNamespace(aio=_failing_register)),
        unregister=types.SimpleNamespace(
            remote=types.SimpleNamespace(aio=lambda *a, **k: _noop_coroutine())
        ),
    )
    assert client.post("/adapters", json=record).status_code in (200, 202)
    assert _lifecycle(client, REVISION) == "failed"
    assert dict.get(module.adapter_records, key) is None, (
        "a revision that settled `failed` kept its lora id claim, so that id is refused forever "
        "for an adapter that never became resident"
    )


def test_a_failed_load_does_not_release_a_claim_another_container_is_using(client, monkeypatch):
    """One container's failed load must not release the shared claim.

    A load runs PER CONTAINER but the claim is global. `Engine` scales horizontally and a retried
    `models deploy` starts a second settle for the same revision, so a container failing says
    nothing about whether another already has the adapter resident -- and `_claim_lora_int_id`
    deliberately lets a second loader through when the recorded owner already matches, so the
    failing container reaches the release holding a claim that is not exclusively its own.

    Released there, the id goes free while the peer is still serving under it, and a colliding
    adapter can then claim and load it -- the exact outcome the claim exists to prevent, since vLLM
    addresses a LoRA solely by the int and would answer from the wrong run's weights. Only
    `settle_adapter`, under the run lock and behind the `settle_attempt` guard, can tell "this
    revision failed" from "this container's attempt at it failed".
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

    async def _boom(record):
        raise RuntimeError("hf download failed")

    instance._adapter_path = _boom
    instance.engine = types.SimpleNamespace(add_lora=lambda request: None)

    retried = {"adapter_id": "run-retried@final." + "e" * 40}
    key = module._lora_id_key(module._lora_int_id(retried["adapter_id"]))
    # The peer container that already loaded this revision successfully holds the claim.
    module.adapter_records[key] = retried["adapter_id"]

    with pytest.raises(RuntimeError, match="download failed"):
        _run_awaitable(engine_class._lora_request(instance, retried))

    assert dict.get(module.adapter_records, key) == retried["adapter_id"], (
        "a failed load released the shared lora id claim while another container still had the "
        "adapter resident under it, leaving the id free for a colliding adapter to take"
    )


def test_unregister_evicts_before_releasing_its_claim(client, monkeypatch):
    """The claim must be held across the eviction, not dropped before it.

    The claim is what keeps a colliding newcomer out, and the two adapters hold DIFFERENT
    per-adapter locks -- so releasing first lets the newcomer claim the id and load into it, and
    this adapter's own `remove_lora(int_id)` then evicts the newcomer, whose record still reads
    `ready` while nothing is resident.

    Driven by having a newcomer claim the id at the moment the outgoing adapter releases it, which
    is only reachable when the release happens first.
    """
    module = client.app.state.generated_module
    engine_class = module.engine_classes[0]
    adapter = "run-out@final." + "e" * 40
    int_id = module._lora_int_id(adapter)
    key = module._lora_id_key(int_id)
    dict.__setitem__(module.adapter_records, key, adapter)
    order: list[str] = []

    records = module.adapter_records

    class _NewcomerRaces(type(records)):
        @property
        def pop(self):
            async def _pop(k, default=None):
                if k == key:
                    order.append("release")
                return dict.pop(self, k, default)

            gated = _Aio(lambda k, default=None: dict.pop(self, k, default))
            gated.aio = _pop
            return gated

    instance = engine_class.__new__(engine_class)
    instance._locks = {}
    instance._int_ids = {int_id: adapter}
    instance._loaded = {adapter: object()}

    async def _remove(lora_id):
        order.append("evict")

    instance.engine = types.SimpleNamespace(remove_lora=_remove)
    hooked = _NewcomerRaces()
    hooked.update(records)
    module.adapter_records = hooked
    try:
        _run_awaitable(engine_class.unregister(instance, adapter))
    finally:
        module.adapter_records = records

    assert order == ["evict", "release"], (
        f"the claim was released before the eviction (order: {order}), so a colliding newcomer "
        "can claim and load the id in between and then be evicted by this call"
    )


def test_a_lora_claim_whose_owner_vanishes_is_recontested(client):
    """A vacated claim must be re-contested, not assumed.

    `put(skip_if_exists=True)` and the follow-up `get` are two calls, so the holder can be
    undeployed between them and the read returns None. Answering with the caller's own id there
    reports a claim that was never RECORDED -- the slot stays free, the old owner can re-register
    into it while this adapter downloads, and both end up resident under one vLLM id, which is the
    aliasing the collision guard exists to prevent.

    Driven by emptying the slot between the insert and the read, which is exactly that window.
    """
    module = client.app.state.generated_module
    records = module.adapter_records
    key = module._lora_id_key(777)
    dict.__setitem__(records, key, "run-old@final." + "a" * 40)
    mine = "run-new@final." + "b" * 40
    vacated: list[str] = []

    class _OwnerVanishes(type(records)):
        @property
        def get(self):
            async def _read(k, default=None):
                value = dict.get(self, k, default)
                if k == key and not vacated:
                    vacated.append(k)
                    dict.pop(self, k, None)
                    return None
                return value

            gated = _Aio(lambda k, default=None: dict.get(self, k, default))
            gated.aio = _read
            return gated

    hooked = _OwnerVanishes()
    hooked.update(records)
    module.adapter_records = hooked
    try:
        owner = _run_awaitable_result(module._claim_lora_int_id(777, mine))
    finally:
        module.adapter_records = records

    assert vacated, "the owner never vanished, so this test did not exercise the window"
    assert owner == mine, "a vacated slot must be claimable"
    assert dict.get(hooked, key) == mine, (
        "the claim was reported without being recorded, so the slot is still free for another "
        "adapter to take while this one downloads"
    )


def test_a_failed_settle_handoff_is_recorded_as_failed(client):
    """A spawn that never queues must not leave a durable `registered` record.

    The client's ambiguous-registration recovery re-reads the record after a 5xx: a matching
    identity in `registered` reads as "the registration landed, settling is in progress", so it
    polls for the full readiness budget instead of repeating the idempotent POST. The deploy then
    fails as a timeout, which points at a slow GPU rather than at a queue that refused the work.
    """
    module = client.app.state.generated_module
    original = module.settle_adapter

    def _refuse(record):
        raise RuntimeError("queue unavailable")

    module.settle_adapter = types.SimpleNamespace(spawn=_refuse)
    try:
        # TestClient re-raises rather than rendering the 500 a deployed app would return; the
        # failure surfacing at all is the point, and what it leaves behind is what is asserted.
        with pytest.raises(RuntimeError, match="queue unavailable"):
            client.post("/adapters", json=REGISTRATION)
    finally:
        module.settle_adapter = original

    record = module.adapter_records[module._record_key(REVISION)]
    assert record["metadata"]["lifecycle_state"] == "failed", (
        "the record was left `registered` with nothing to advance it, so the client polls it to "
        "timeout instead of re-registering"
    )
    assert "enqueue" in record["metadata"]["failure"]


@pytest.mark.parametrize(
    ("configured", "presented"),
    [("sekrit\n", "sekrit"), (" sekrit ", "sekrit"), ("sekrit", "sekrit\n")],
    ids=["trailing-newline-in-secret", "padded-secret", "padded-header"],
)
def test_the_serving_key_is_compared_after_stripping(client, monkeypatch, configured, presented):
    """Both sides must normalize the same way, or an identical key 401s.

    The client strips before building the header (`flash/serve/urls.py:internal_key_header`), so a
    secret stored with surrounding whitespace -- a heredoc or a copied line carries a trailing
    newline -- makes an operator who supplied the same raw value on both sides get a 401 on every
    request, while `/healthz` reports `requires_key: true`.
    """
    monkeypatch.setenv("FLASH_SERVING_KEY", configured)
    response = client.get("/healthz", headers={"X-Freesolo-Internal-Key": presented})
    assert response.status_code == 200
    ready = client.post(
        "/adapters", json=REGISTRATION, headers={"X-Freesolo-Internal-Key": presented}
    )
    assert ready.status_code != 401, (
        f"a key configured as {configured!r} rejected the same key presented as {presented!r}"
    )


def test_a_whitespace_only_serving_key_is_not_advertised_as_authentication(client, monkeypatch):
    """`requires_key` must report what `_require_key` actually enforces.

    A whitespace-only value authenticates nothing, so advertising `true` for it tells
    `flash serve setup` the URL is guarded when it is wide open -- suppressing the one warning
    that would have caught an unauthenticated public endpoint.
    """
    monkeypatch.setenv("FLASH_SERVING_KEY", "   ")
    assert client.get("/healthz").json()["requires_key"] is False
    assert client.post("/adapters", json=REGISTRATION).status_code != 401, (
        "requires_key and _require_key disagree about whether this app authenticates"
    )


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


def test_a_failed_eviction_keeps_the_lora_claim(client, monkeypatch):
    """A `remove_lora` that raises must not release the claim.

    The exception is suppressed because eviction must never fail an undeploy, but suppressing it
    does not make the adapter gone: it is still resident under this int id. Releasing anyway lets a
    colliding adapter claim the id and load while the old weights still occupy it, which is the
    wrong-run's-weights outcome the claim exists to prevent.
    """
    module = client.app.state.generated_module
    engine_class = module.engine_classes[0]
    instance = engine_class.__new__(engine_class)
    instance._locks = {}
    instance._loaded = {REVISION: object()}
    int_id = module._lora_int_id(REVISION)
    instance._int_ids = {int_id: REVISION}

    async def _refuse(_int_id):
        raise RuntimeError("engine is wedged")

    instance.engine = types.SimpleNamespace(remove_lora=_refuse)
    key = module._lora_id_key(int_id)
    module.adapter_records[key] = REVISION

    _run_awaitable_result(engine_class.unregister(instance, REVISION))

    assert dict.get(module.adapter_records, key) == REVISION, (
        "an eviction that failed still released the lora id claim, so a colliding adapter can "
        "claim the id while the old weights are still resident under it"
    )


def test_an_engine_rpc_failure_settles_the_record_as_failed(client):
    """A dead Engine container must not leave the record at `registered`.

    The raise happens outside `Engine.register`'s own handler, and settle runs detached so nothing
    observes it. Unhandled, the record stays `registered` forever and the client polls its whole
    readiness budget, reporting a timeout that reads as a slow GPU rather than an engine that never
    answered.
    """
    module = client.app.state.generated_module

    async def _die(_record):
        raise RuntimeError("engine container exited during startup")

    module.Engine = lambda: types.SimpleNamespace(
        register=types.SimpleNamespace(remote=types.SimpleNamespace(aio=_die)),
        unregister=types.SimpleNamespace(
            remote=types.SimpleNamespace(aio=lambda *a, **k: _noop_coroutine())
        ),
    )
    assert client.post("/adapters", json=REGISTRATION).status_code in (200, 202)

    assert _lifecycle(client, REVISION) == "failed", (
        "an engine rpc that never answered left the record at `registered`, so the client polls "
        "its full readiness budget and reports a timeout instead of the real failure"
    )
    failure = client.get(f"/adapters/{REVISION}").json()["adapter"]["metadata"]["failure"]
    assert "did not answer" in failure


def test_a_stale_enqueue_failure_does_not_overwrite_a_newer_attempt(client):
    """Only this request's attempt may be marked `failed` when its handoff fails.

    Two identical registrations overlap by design. If the first one's spawn fails after the second
    has already stamped a new attempt and queued it, an unconditional write restores the first's
    stale record as `failed` -- and the live settle then finds a token mismatch and cannot commit
    its result at all, pinning a run whose load succeeded to `failed` permanently.
    """
    _register_and_ready(client)
    module = client.app.state.generated_module
    ready = dict(client.get(f"/adapters/{REVISION}").json()["adapter"])
    # Put the record back to `registered` so a re-registration reaches the spawn at all: a `ready`
    # record is a genuine no-op that never enqueues, so the failure under test is unreachable
    # without this. The token stays the CURRENT one, which the failing request will not carry.
    module.adapter_records[module._record_key(REVISION)] = {
        **ready,
        "status": "registered",
        "metadata": {**ready["metadata"], "lifecycle_state": "registered"},
    }

    def _refuse(_record):
        raise RuntimeError("queue unavailable")

    # The re-registration stamps a NEW token, writes it, and then fails to enqueue. Meanwhile the
    # concurrent settle this simulates has already committed `ready` under a different token: the
    # durable record is rewritten to that state between the stamp and the spawn.
    spawned_with: list = []

    def _refuse_after_a_newer_attempt_lands(record):
        spawned_with.append(record)
        module.adapter_records[module._record_key(REVISION)] = {
            **ready,
            "status": "ready",
            "metadata": {**ready["metadata"], "lifecycle_state": "ready"},
        }
        raise RuntimeError("queue unavailable")

    original = module.settle_adapter
    module.settle_adapter = types.SimpleNamespace(spawn=_refuse_after_a_newer_attempt_lands)
    try:
        # TestClient re-raises what a deployed app returns as a 500, so the raise itself is expected.
        with pytest.raises(RuntimeError, match="queue unavailable"):
            client.post("/adapters", json=REGISTRATION)
    finally:
        module.settle_adapter = original

    assert spawned_with, "the registration never reached the enqueue, so nothing was under test"
    record = dict.get(module.adapter_records, module._record_key(REVISION))
    assert record["status"] == "ready", (
        "a superseded request's failed enqueue overwrote the newer attempt's `ready` record, so "
        "the settle that actually succeeded can no longer commit its result"
    )


def test_undeploy_reports_when_registrations_outran_its_passes(client):
    """A run that keeps receiving registrations must not report a complete undeploy.

    The pass budget is bounded, so a revision appended during the final pass is read but never
    processed. Returning 200 there leaves it `ready` and directly callable by its immutable id
    while the client believes the run is gone.
    """
    module = client.app.state.generated_module
    _register_and_ready(client)

    members_key = module._members_key(RUN_ID)
    original_read = module._run_members
    calls = {"n": 0}

    async def _always_growing(run_id):
        calls["n"] += 1
        existing = list(dict.get(module.adapter_records, members_key) or [])
        fresh = f"{RUN_ID}@final.{calls['n']:040d}"
        module.adapter_records[module._record_key(fresh)] = {
            "adapter_id": fresh,
            "status": "ready",
            "metadata": {"run_id": RUN_ID, "record_type": "revision", "lifecycle_state": "ready"},
        }
        return [*existing, fresh]

    module._run_members = _always_growing
    try:
        response = client.delete(f"/adapters/{RUN_ID}")
    finally:
        module._run_members = original_read

    assert response.status_code == 409, (
        f"undeploy returned {response.status_code} while revisions it never disabled were still "
        f"ready and callable by their immutable ids"
    )
    # And the 409 must not cost the eviction of what this call DID disable. Raised before the
    # eviction loop, those revisions are already `disabled`, so the retry passes over them and
    # their max_loras slots stay occupied until the container recycles.
    assert REVISION in module.unregistered, (
        "the straggler 409 skipped the gpu eviction, so revisions this call disabled stay resident "
        "and a retry will pass over them as already disabled"
    )


def test_a_colliding_load_evicts_a_stale_local_resident(client, monkeypatch):
    """A replica that missed the undeploy eviction must not keep two adapters on one int id.

    Undeploy's eviction is one remote call and Modal routes it to a single replica, so another
    replica can still hold the old adapter under this int. The claim was released, so the id is
    legitimately re-claimable -- loading over it without evicting first leaves two adapters mapped
    to one int, which is exactly what vLLM cannot represent.
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
    instance._loaded = {}
    instance._int_ids = {}

    removed: list[int] = []

    async def _remove(int_id):
        removed.append(int_id)

    async def _path(record):
        return "/cache/adapter"

    async def _add(request):
        return None

    instance._adapter_path = _path
    instance.engine = types.SimpleNamespace(add_lora=_add, remove_lora=_remove)
    monkeypatch.setattr(module, "_lora_int_id", lambda adapter_id: 4242)

    # The old adapter is still resident on THIS replica; its claim was already released elsewhere.
    old = "run-old@final." + "a" * 40
    instance._loaded[old] = object()
    instance._int_ids[4242] = old

    newcomer = {"adapter_id": "run-new@final." + "b" * 40}
    _run_awaitable(engine_class._lora_request(instance, newcomer))

    assert removed == [4242], (
        "the newcomer loaded over a stale local resident without evicting it, so two adapters are "
        "mapped to one vllm int id on this container"
    )
    assert old not in instance._loaded


def test_an_unclaimable_lora_id_fails_the_registration(client):
    """Exhausting the claim attempts must raise, not report ownership that was never recorded.

    Answering with the caller's own id looks conservative -- it lets a legitimate adapter proceed
    over a slot nobody holds -- but it hands back a claim that does not exist, which is the exact
    unclaimed-load path the function exists to remove: a colliding registration can take the id
    while this one downloads, and both end up resident under one vLLM id.
    """
    module = client.app.state.generated_module

    class _NeverSettles(_FakeDict):
        """The claim key is always taken on insert and always gone by the time it is read.

        The real shape of this is a holder that is undeployed in the gap between the failed insert
        and the read, on every attempt.
        """

        def _put(self, key, value, skip_if_exists=False):
            if skip_if_exists and key.startswith("loraid:"):
                return False
            return super()._put(key, value, skip_if_exists=skip_if_exists)

    original = module.adapter_records
    module.adapter_records = _NeverSettles()
    try:
        with pytest.raises(RuntimeError, match="could not establish ownership"):
            _run_awaitable_result(module._claim_lora_int_id(7777, "run-x@final." + "c" * 40))
    finally:
        module.adapter_records = original


def test_a_load_that_loses_its_claim_mid_download_undoes_itself(client, monkeypatch):
    """A superseded load must not stay resident after the claim changes hands.

    The download takes minutes and the claim can legitimately move inside that window: a retried
    deploy makes this attempt superseded, and if the newer authoritative one then fails, settle
    releases the claim while this load is still running. A colliding adapter takes the id from
    there, and this `add_lora` lands on top of it -- two adapters under one vLLM int.
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
    instance._loaded = {}
    instance._int_ids = {}

    mine = "run-superseded@final." + "f" * 40
    int_id = module._lora_int_id(mine)
    removed: list[int] = []

    async def _remove(evicted_id):
        removed.append(evicted_id)

    async def _path(record):
        # The claim is RELEASED during the download, exactly as settle deciding this attempt was
        # superseded would leave it: by the time the load finishes, this adapter holds nothing.
        module.adapter_records.pop(module._lora_id_key(int_id), None)
        return "/cache/adapter"

    async def _add(request):
        return None

    instance._adapter_path = _path
    instance.engine = types.SimpleNamespace(add_lora=_add, remove_lora=_remove)

    with pytest.raises(RuntimeError, match="changed hands"):
        _run_awaitable(engine_class._lora_request(instance, {"adapter_id": mine}))

    assert removed == [int_id], (
        "a load that lost its claim mid-download stayed resident, so it shares a vllm int id with "
        "whichever adapter now owns the claim"
    )
    assert mine not in instance._loaded


def test_a_superseded_load_leaves_no_orphan_when_a_peer_took_its_id(client, monkeypatch):
    """Undoing a superseded load must not leave a LoRA resident that nothing can find.

    An earlier version skipped the eviction whenever a peer held the claim, to avoid kicking the
    winner off the GPU. That traded one fault for a worse one: the LoRA this container had just
    loaded stayed resident with no `_int_ids` entry, and the collision sweep only consults
    `_int_ids` -- so the winner's own load could not find the orphan and its `add_lora` failed on
    this replica. Neither ordering is fixable here, because `_locks` is keyed by ADAPTER id while
    the contested resource is the INT id, so two colliding adapters never serialize.

    Evicting is the recoverable side: the durable claim is atomic, the winner still owns the id,
    and its next load re-adds the adapter. An orphan has no path back.
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
    instance._loaded = {}
    instance._int_ids = {}

    mine = "run-superseded@final." + "e" * 40
    winner = "run-winner@final." + "7" * 40
    int_id = module._lora_int_id(mine)
    removed: list[int] = []

    async def _remove(evicted_id):
        removed.append(evicted_id)

    async def _path(record):
        # A COLLIDER takes the id mid-download and is now the live owner of this int.
        module.adapter_records[module._lora_id_key(int_id)] = winner
        return "/cache/adapter"

    async def _add(request):
        return None

    instance._adapter_path = _path
    instance.engine = types.SimpleNamespace(add_lora=_add, remove_lora=_remove)

    with pytest.raises(RuntimeError, match="changed hands"):
        _run_awaitable(engine_class._lora_request(instance, {"adapter_id": mine}))

    assert removed == [int_id], (
        "the superseded load left its lora resident, and with no _int_ids entry the winner's own "
        "load cannot find the orphan -- its add_lora then fails on this replica"
    )
    assert int_id not in instance._int_ids, (
        "the undo left an _int_ids entry pointing at the superseded adapter"
    )
    assert mine not in instance._loaded
    assert module.adapter_records[module._lora_id_key(int_id)] == winner, (
        "the superseded load disturbed the winner's claim"
    )


def test_undeploy_deletes_the_downloaded_adapter(client, monkeypatch, tmp_path):
    """Retired revisions must not accumulate on the persistent volume forever.

    Every immutable revision downloads into its own digest directory, so a backend serving
    successive checkpoints keeps every one it has ever loaded -- paid storage growing without
    bound until the volume fills and new adapters cannot download at all.
    """
    module = client.app.state.generated_module
    engine_class = module.engine_classes[0]
    instance = engine_class.__new__(engine_class)
    instance._locks = {}
    instance._loaded = {}
    instance._int_ids = {}

    async def _remove(_int_id):
        return None

    instance.engine = types.SimpleNamespace(remove_lora=_remove)

    downloaded = tmp_path / module._adapter_digest(REVISION)
    downloaded.mkdir()
    (downloaded / "adapter_model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(module, "ADAPTER_DIR", str(tmp_path))
    # Disabled first, which is the state undeploy has already written durably by the time it calls
    # the engine. The delete is gated on it: a revision that is `ready` again has been redeployed
    # and its files are in use.
    module.adapter_records[module._record_key(REVISION)] = {
        "adapter_id": REVISION,
        "status": "disabled",
    }

    _run_awaitable_result(engine_class.unregister(instance, REVISION))

    assert not downloaded.exists(), (
        "undeploy left the downloaded adapter on the volume, so retired revisions accumulate "
        "until storage runs out"
    )


def test_a_redeployed_revision_keeps_its_downloaded_adapter(client, monkeypatch, tmp_path):
    """Cleanup must not delete weights a concurrent redeploy has already brought back.

    The engine call is awaited OUTSIDE the run lock and the per-adapter lock is per-container, so
    `models deploy` can re-register this exact revision and settle it `ready` while the eviction is
    in flight. The volume is shared by every replica, so deleting then removes the files under a
    live load on another container: a `ready` record whose weights are gone, which fails at the
    next cold start rather than here.
    """
    module = client.app.state.generated_module
    engine_class = module.engine_classes[0]
    instance = engine_class.__new__(engine_class)
    instance._locks = {}
    instance._loaded = {}
    instance._int_ids = {}

    async def _remove(_int_id):
        return None

    instance.engine = types.SimpleNamespace(remove_lora=_remove)

    downloaded = tmp_path / module._adapter_digest(REVISION)
    downloaded.mkdir()
    (downloaded / "adapter_model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(module, "ADAPTER_DIR", str(tmp_path))
    # Redeployed: the record is `ready` again by the time cleanup reads it.
    module.adapter_records[module._record_key(REVISION)] = {
        "adapter_id": REVISION,
        "status": "ready",
    }

    _run_awaitable_result(engine_class.unregister(instance, REVISION))

    assert downloaded.exists(), (
        "cleanup deleted the weights of a revision that had been redeployed, so a ready record "
        "now points at files that are gone"
    )


def test_undeploy_redisables_a_revision_that_was_re_registered_mid_pass(client):
    """A member re-registered after being visited must still end disabled.

    The lock's lease does not renew, so a long undeploy can hand the lock to a concurrent
    registration. Re-registering an EXISTING revision resets it to `registered` without adding a
    new member id, so a loop that skipped already-visited members would never look at it again --
    its settle turns it back to `ready` and DELETE answers 200 for a run that is still directly
    callable by that revision's immutable id.
    """
    module = client.app.state.generated_module
    _register_and_ready(client)

    original_read = module._run_members
    calls = {"n": 0}

    async def _revives_the_same_member(run_id):
        # `_run_members` is called twice BEFORE the loop (the repair probe and the membership read)
        # and then once at the end of each pass, so the revive has to wait for the call that
        # follows the first pass. At that point the revision has been disabled, and a concurrent
        # re-registration of that same immutable id brings it back `ready`. The membership does NOT
        # grow -- re-registering an existing revision adds no new member id -- so only a
        # status-based convergence check can notice.
        calls["n"] += 1
        if calls["n"] == 3:
            record = dict(module.adapter_records[module._record_key(REVISION)])
            record["status"] = "ready"
            record["metadata"] = {**(record.get("metadata") or {}), "lifecycle_state": "ready"}
            module.adapter_records[module._record_key(REVISION)] = record
        return await original_read(run_id)

    module._run_members = _revives_the_same_member
    try:
        response = client.delete(f"/adapters/{RUN_ID}")
    finally:
        module._run_members = original_read

    assert response.status_code == 200, response.text
    final = module.adapter_records[module._record_key(REVISION)]
    assert final["status"] == "disabled", (
        "a revision re-registered after its pass stayed ready, so undeploy returned success for a "
        "run that is still callable by that revision's immutable id"
    )


@pytest.mark.parametrize(
    ("field", "literal"),
    [
        ("temperature", "1e400"),
        ("temperature", "NaN"),
        ("top_p", "NaN"),
        ("top_p", "Infinity"),
    ],
)
def test_a_non_finite_sampling_value_is_rejected(client, field, literal):
    """inf and NaN must not reach the GPU as sampling parameters.

    json accepts `1e400` and the bare literals `Infinity` and `NaN`. The first two parse to inf,
    which passes `temperature >= 0` because temperature has no upper bound; NaN defeats every
    ordered comparison at once, so `nan <= 0.0` and `nan > 1.0` are both False and it passes even a
    two-sided bound. Both then reach vLLM as sampling nobody asked for.
    """
    _register_and_ready(client)
    response = client.post(
        "/v1/chat/completions",
        # Raw body, not `json=`: the whole point is a literal python's json parser turns into inf
        # or nan, and `json.dumps` cannot emit `1e400` or a bare `NaN` through a float round-trip.
        content=(
            f'{{"model": "{RUN_ID}", "messages": [{{"role": "user", "content": "hi"}}], '
            f'"{field}": {literal}}}'
        ),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400, (
        f"{field}={literal} was accepted with {response.status_code}, so a non-finite sampling "
        f"value reaches the gpu"
    )
    assert "finite" in response.text


def test_undeploy_rechecks_after_the_membership_read_that_revived_a_member(client):
    """A revive landing on the CONFIRMATION read must not slip through as convergence.

    `disabled_here` is counted before the membership refresh, and a revive adds no member id, so
    `seen.issuperset(members)` cannot see one either. Both convergence signals are stale by exactly
    one Dict round trip: a re-registration landing on that read leaves the revision `ready` with an
    empty straggler list, and DELETE returns 200 for a run that is live again.
    """
    module = client.app.state.generated_module
    _register_and_ready(client)

    original_read = module._run_members
    calls = {"n": 0}

    async def _revives_on_the_confirmation_read(run_id):
        # Call 1 and 2 are the pre-loop repair probe and membership read; call 3 ends pass one and
        # is where the earlier regression test revives. Call 4 is the read that CONFIRMS
        # convergence -- reviving there is what this test exists for.
        calls["n"] += 1
        if calls["n"] == 4:
            record = dict(module.adapter_records[module._record_key(REVISION)])
            record["status"] = "ready"
            record["metadata"] = {**(record.get("metadata") or {}), "lifecycle_state": "ready"}
            module.adapter_records[module._record_key(REVISION)] = record
        return await original_read(run_id)

    module._run_members = _revives_on_the_confirmation_read
    try:
        response = client.delete(f"/adapters/{RUN_ID}")
    finally:
        module._run_members = original_read

    final = module.adapter_records[module._record_key(REVISION)]
    assert final["status"] == "disabled" or response.status_code == 409, (
        f"undeploy returned {response.status_code} with the revision still "
        f"{final['status']!r}: a revive on the confirmation read was reported as a clean undeploy"
    )


def test_a_failed_eviction_keeps_the_superseded_load_findable(client, monkeypatch):
    """A `remove_lora` that FAILS must leave the `_int_ids` entry behind.

    The entry is recorded before the eviction precisely so a failure stays findable. Popping it
    unconditionally undoes that for the one case it exists to cover: the LoRA is still resident and
    nothing can locate it, which is the orphan the ordering was written to prevent.
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
    instance._loaded = {}
    instance._int_ids = {}

    mine = "run-superseded@final." + "d" * 40
    int_id = module._lora_int_id(mine)

    async def _remove(_evicted_id):
        raise RuntimeError("engine refused the eviction")

    async def _path(record):
        module.adapter_records.pop(module._lora_id_key(int_id), None)
        return "/cache/adapter"

    async def _add(request):
        return None

    instance._adapter_path = _path
    instance.engine = types.SimpleNamespace(add_lora=_add, remove_lora=_remove)

    with pytest.raises(RuntimeError, match="changed hands"):
        _run_awaitable(engine_class._lora_request(instance, {"adapter_id": mine}))

    assert instance._int_ids.get(int_id) == mine, (
        "a failed eviction dropped the _int_ids entry, so the still-resident lora is invisible to "
        "the collision sweep that would otherwise evict it before loading over the id"
    )


def test_provenance_must_agree_with_the_revision_id(client):
    """The id IS the identity, so metadata sent alongside it cannot contradict it.

    `run_id` decides which run's alias and membership this revision joins, so a mismatch files the
    artifact under a DIFFERENT run -- undeploying that run then reports success while this revision
    keeps serving under its own id. `checkpoint_step` is echoed back as provenance the client
    cross-checks on its 5xx recovery path. Neither is repairable afterwards.

    Checked on the FIRST registration, which is what widening the fingerprint alone cannot do: by
    the second one the alias has already been written to the wrong run.
    """
    module = client.app.state.generated_module
    for bad_metadata, label in (
        ({"run_id": "some-other-run"}, "run_id"),
        ({"checkpoint_step": 99}, "checkpoint_step"),
        ({"checkpoint_step": None}, "checkpoint_step dropped"),
        ({"checkpoint_step": "ten"}, "checkpoint_step not an int"),
        ({"hf_revision": "b" * 40}, "hf_revision"),
    ):
        response = client.post(
            "/adapters",
            json={**REGISTRATION, "metadata": {**REGISTRATION["metadata"], **bad_metadata}},
        )
        assert response.status_code == 422, (
            f"registration accepted {label} disagreeing with the revision id "
            f"({response.status_code}); the record's provenance then contradicts its own id"
        )
        assert module._record_key(REVISION) not in module.adapter_records, (
            f"the record was stored despite {label} contradicting the id"
        )
    # A non-revision id cannot be registered at all: every downstream path parses identity out of
    # it, and there is nothing to reconcile the metadata against.
    assert (
        client.post("/adapters", json={**REGISTRATION, "adapter_id": "not-a-revision"}).status_code
        == 422
    )
    # And the matching payload still registers.
    assert client.post("/adapters", json=REGISTRATION).status_code in (200, 202)


def test_changed_provenance_under_one_id_is_a_conflict(client):
    """Re-registering an id with different provenance is different content, so it must 409.

    The fingerprint took only `hf_revision` from metadata, so a re-registration that changed
    `run_id` or `checkpoint_step` compared equal and was accepted as an identical retry.
    """
    module = client.app.state.generated_module
    _register_and_ready(client)
    stored = module.adapter_records[module._record_key(REVISION)]
    for field, value in (("run_id", "other-run"), ("checkpoint_step", 999)):
        mutated = {**stored, "metadata": {**stored["metadata"], field: value}}
        assert module._fingerprint(mutated) != module._fingerprint(stored), (
            f"a record differing only in {field} fingerprinted identically, so a re-registration "
            f"that changes it is accepted as an unchanged retry instead of conflicting"
        )


def test_registration_requires_a_pinned_commit_sha(client):
    """A mutable ref cannot back a revision this app advertises as immutable.

    `metadata.hf_revision` is what `_adapter_path` hands to `snapshot_download`, so a branch name
    makes one immutable id serve whatever that branch points at today: undeploy, re-register the
    same id, and different weights load while `_fingerprint` sees no change at all.
    """
    # ONLY `hf_revision` varies. The id, `run_id` and `checkpoint_step` are held in agreement, so
    # nothing else can supply the 422 -- an earlier version of this test posted an `@final` id
    # while inheriting `REGISTRATION`'s `run_id` and `checkpoint_step: 10`, and those mismatches
    # forced the 422 by themselves. It passed with the sha check deleted outright.
    for bad in ("main", "", "a" * 39, "A" * 40, "refs/heads/main"):
        response = client.post(
            "/adapters",
            json={
                **REGISTRATION,
                "metadata": {**REGISTRATION["metadata"], "hf_revision": bad},
            },
        )
        assert response.status_code == 422, (
            f"registration accepted hf_revision={bad!r} with "
            f"{response.status_code}: a moved ref can then serve different weights under an id "
            f"the client is told is immutable"
        )
    # A sha of the right SHAPE that names a different commit than the id does. This is the case a
    # shape-only check passes and a reconciliation catches.
    assert (
        client.post(
            "/adapters",
            json={
                **REGISTRATION,
                "metadata": {**REGISTRATION["metadata"], "hf_revision": "b" * 40},
            },
        ).status_code
        == 422
    ), "a well-formed sha that disagrees with the id was accepted"
    # And the good case still registers, so the guard is not simply refusing everything.
    assert client.post("/adapters", json=REGISTRATION).status_code in (200, 202)


def test_metadata_cannot_turn_a_revision_into_an_alias(client):
    """`record_type` is this endpoint's to state, not the caller's to supply.

    A record registered as an alias loads onto the GPU but can never be activated (`/activate`
    refuses non-revisions) and undeploy classifies it as an alias, so it skips both eviction and
    cache cleanup. An identical retry cannot repair it either: `_fingerprint` does not cover
    `record_type`, so the retry is a no-op conflict-free re-registration of the same broken record.
    """
    module = client.app.state.generated_module
    assert client.post(
        "/adapters",
        json={
            **REGISTRATION,
            "metadata": {**REGISTRATION["metadata"], "record_type": "alias"},
        },
    ).status_code in (200, 202)
    record = module.adapter_records[module._record_key(REVISION)]
    assert record["metadata"]["record_type"] == "revision", (
        "a caller relabelled its own revision as an alias, producing a record that can never be "
        "activated and that undeploy will not evict or clean up"
    )


def test_a_failed_load_reclaims_its_downloaded_weights(client):
    """A terminally failed revision must not keep its download forever.

    `_adapter_path` runs before `add_lora`, so a rejected adapter is one that downloaded fine and
    then failed to load. Nothing else ever collects it: DELETE skips records that are already
    `disabled`, and settling the failure is what made it so. The digest is a hash of the adapter
    id, so every distinct bad revision leaves its own directory behind permanently.
    """
    module = client.app.state.generated_module
    bad_revision = "run-badcache@final." + "e" * 40
    client.post(
        "/adapters",
        json={
            **REGISTRATION,
            "adapter_id": bad_revision,
            "repo_id": BAD_REPO,
            "checkpoint": "run-badcache",
            "metadata": {
                "record_type": "revision",
                "run_id": "run-badcache",
                "checkpoint_step": None,
                "hf_revision": "e" * 40,
            },
        },
    )
    assert _lifecycle(client, bad_revision) == "failed"
    assert bad_revision in module.discarded, (
        "a load that failed after downloading left its weights on the shared volume, and no other "
        "path ever collects them: delete skips records that are already disabled"
    )
    assert bad_revision not in module.unregistered, (
        "cleanup went through unregister, which evicts by int id -- but the failure already "
        "released that claim, so it can evict an adapter that has since taken the id"
    )


def test_a_failed_load_does_not_cold_start_a_gpu_to_reclaim_its_cache(client):
    """Reclaiming disk must not boot a GPU against a deploy that already failed.

    The volume is only reachable from a container that mounts it, so with everything scaled to zero
    this call starts one -- minutes of paid time to run `rmtree`. And the dominant way a load goes
    terminal is `register.remote` never answering because no container could start, which is
    exactly when nothing is warm: unguarded, every engine outage bought a cold start per failed
    registration.
    """
    module = client.app.state.generated_module
    module.runners.count = 0
    bad_revision = "run-idlecache@final." + "7" * 40
    client.post(
        "/adapters",
        json={
            **REGISTRATION,
            "adapter_id": bad_revision,
            "repo_id": BAD_REPO,
            "checkpoint": "run-idlecache",
            "metadata": {
                "record_type": "revision",
                "run_id": "run-idlecache",
                "checkpoint_step": None,
                "hf_revision": "7" * 40,
            },
        },
    )
    assert _lifecycle(client, bad_revision) == "failed", (
        "skipping the cache reclaim must not skip recording the failure the client polls for"
    )
    assert module.discarded == [], (
        "a failed registration reached the engine with nothing warm, so it cold-started a gpu "
        "purely to delete files -- paid boot time charged against a deploy that already failed"
    )


def test_a_failed_load_still_reclaims_its_cache_on_a_warm_engine(client):
    """The skip above must not become "never reclaim".

    A warm container is the case this is FOR: the files are on a mounted volume, no cold start is
    needed, and nothing else ever collects them because DELETE skips already-disabled records.
    """
    module = client.app.state.generated_module
    module.runners.count = 1
    bad_revision = "run-warmcache@final." + "8" * 40
    client.post(
        "/adapters",
        json={
            **REGISTRATION,
            "adapter_id": bad_revision,
            "repo_id": BAD_REPO,
            "checkpoint": "run-warmcache",
            "metadata": {
                "record_type": "revision",
                "run_id": "run-warmcache",
                "checkpoint_step": None,
                "hf_revision": "8" * 40,
            },
        },
    )
    assert _lifecycle(client, bad_revision) == "failed"
    assert bad_revision in module.discarded, (
        "a warm engine left the failed revision's download on the volume, and no other path "
        "collects it: delete skips records that are already disabled"
    )


def test_a_cold_failed_load_is_reclaimed_by_the_next_undeploy(client):
    """The skip must DEFER the reclaim, not cancel it.

    Skipping the cold-start reclaim is only defensible if something later collects the directory.
    The obvious candidate is undeploy -- but the same settle that skipped the reclaim also marked
    the record `disabled`, and undeploy's loop skips records that are already disabled, so it never
    reaches this one. Left there, every failed-load-while-scaled-to-zero leaks its download onto
    the volume permanently, and the volume is what the user pays for.

    Nothing here is warm at any point: this is the cold path end to end, which is exactly the case
    the guard was written for.
    """
    module = client.app.state.generated_module
    module.runners.count = 0
    bad_revision = "run-coldreclaim@final." + "9" * 40
    client.post(
        "/adapters",
        json={
            **REGISTRATION,
            "adapter_id": bad_revision,
            "repo_id": BAD_REPO,
            "checkpoint": "run-coldreclaim",
            "metadata": {
                "record_type": "revision",
                "run_id": "run-coldreclaim",
                "checkpoint_step": None,
                "hf_revision": "9" * 40,
            },
        },
    )
    assert _lifecycle(client, bad_revision) == "failed"
    assert module.discarded == [], "the cold path must not have reclaimed inline"

    response = client.delete("/adapters/run-coldreclaim")
    assert response.status_code == 200, f"undeploy returned {response.status_code}"
    assert bad_revision in module.discarded, (
        "undeploy did not reclaim the download the cold settle deliberately left behind. the "
        "record was already `disabled` (the failed settle set it), so undeploy's skip-if-disabled "
        "branch passed over it and nothing ever collects the directory -- the volume grows without "
        "bound across failed deploys"
    )
    # Reported as what this call disabled, and it disabled nothing: the revision was already
    # `disabled` before the request arrived. Claiming it here would tell the client the undeploy
    # took a live revision out of service when it only swept a leftover.
    assert bad_revision not in (response.json().get("disabled_revisions") or []), (
        "undeploy reported an already-disabled revision as one it disabled"
    )

    # Once collected, the marker is gone: a second undeploy of the same run must not re-run the
    # reclaim on every call for the life of the record.
    module.discarded.clear()
    again = client.delete("/adapters/run-coldreclaim")
    assert again.status_code in (200, 404), f"repeat undeploy returned {again.status_code}"
    assert module.discarded == [], (
        f"a repeated undeploy reclaimed {module.discarded} again; the pending marker was never "
        f"cleared, so every future delete re-runs the rmtree"
    )


def test_a_stale_resident_that_cannot_be_evicted_refuses_the_load(client, monkeypatch):
    """Suppressing a failed stale-resident eviction loses the adapter and loads over it anyway.

    Clearing the maps first made the still-resident LoRA untracked, so no later sweep could find
    it; proceeding to `add_lora` then bound a second adapter to an int id vLLM still has mapped to
    the first. Refusing keeps the resident findable and the claim held.
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
    instance._loaded = {}
    instance._int_ids = {}

    mine = "run-newcomer@final." + "f" * 40
    int_id = module._lora_int_id(mine)
    stale = "run-stale@final." + "0" * 40
    # A leftover from an adapter this replica served before the id was released and re-claimed.
    instance._int_ids[int_id] = stale
    instance._loaded[stale] = object()

    added: list = []

    async def _remove(_int_id):
        raise RuntimeError("engine refused the eviction")

    async def _add(request):
        added.append(request)

    instance._adapter_path = _path_returning("/cache/adapter")
    instance.engine = types.SimpleNamespace(add_lora=_add, remove_lora=_remove)

    # Not `pytest.raises`: the raise is the fix, but the DAMAGE is what this test has to see. A
    # short-circuit on "it raised" would skip every assertion below, and those are the ones that
    # describe the state the suppressed version left behind.
    with contextlib.suppress(RuntimeError):
        _run_awaitable(engine_class._lora_request(instance, {"adapter_id": mine}))

    assert not added, (
        "the load went ahead over an int id vllm still has bound to the stale adapter, which is "
        "the two-adapters-one-id state the claim exists to prevent"
    )
    assert instance._int_ids.get(int_id) == stale, (
        "a failed eviction dropped the stale resident's map entry, so the lora it left on the gpu "
        "is invisible to the sweep that would otherwise evict it"
    )
    assert module.adapter_records.get(module._lora_id_key(int_id)) == mine, (
        "the claim was released while the stale adapter is still resident, so a collider can take "
        "the id and load on top of the old weights"
    )


def _path_returning(path: str):
    async def _path(_record):
        return path

    return _path


def test_undeploy_reports_conflict_when_it_never_saw_a_clean_pass(client, monkeypatch):
    """Exhausting the pass budget is itself a failure, even with nothing left unvisited.

    Re-registering an ALREADY SEEN revision grows neither the membership nor the unseen set, so a
    run that keeps reviving the same id runs out of passes with an empty straggler list. Reporting
    200 there tells the operator the run is down while that revision settles back to `ready` and
    stays directly callable by its immutable id.
    """
    module = client.app.state.generated_module
    _register_and_ready(client)
    client.post(
        f"/adapters/{REVISION}/activate",
        json={"run_id": RUN_ID, "expected_adapter_revision": None},
    )

    original_read = module._run_members
    calls = types.SimpleNamespace(count=0)

    async def _reviving_members(run_id):
        # Revive on every read AFTER the first, modelling a deploy that keeps landing inside the
        # undeploy. The id is already in `members` and already in `seen`, so neither convergence
        # signal notices; only "did a pass actually re-read everything and find nothing live"
        # does.
        calls.count += 1
        if calls.count > 1:
            record = module.adapter_records.get(module._record_key(REVISION))
            if isinstance(record, dict):
                record["status"] = "ready"
                record["metadata"] = {**record["metadata"], "lifecycle_state": "ready"}
                module.adapter_records[module._record_key(REVISION)] = record
        return await original_read(run_id)

    module._run_members = _reviving_members
    try:
        response = client.delete(f"/adapters/{RUN_ID}")
    finally:
        module._run_members = original_read

    assert response.status_code == 409, (
        f"undeploy returned {response.status_code} after exhausting its passes without one clean "
        f"sweep: the revision is still ready and callable by its immutable id while delete "
        f"reported the run down"
    )


def test_undeploy_still_reports_success_when_it_converges(client):
    """The 409 above must come from a real failure to converge, not from every undeploy.

    Without this, tightening the conflict check to "always 409" would satisfy the test above while
    making ordinary undeploy unusable.
    """
    _register_and_ready(client)
    client.post(
        f"/adapters/{REVISION}/activate",
        json={"run_id": RUN_ID, "expected_adapter_revision": None},
    )
    response = client.delete(f"/adapters/{RUN_ID}")
    assert response.status_code == 200, (
        f"a quiet undeploy reported {response.status_code}; the convergence check is refusing "
        f"runs that did settle"
    )
    assert REVISION in response.json()["disabled_revisions"]
