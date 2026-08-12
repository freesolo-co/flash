"""Drive the generated Modal app's HTTP surface, with modal and the GPU engine stubbed.

Parsing the generated file proves it is valid python. It does not prove the routes mount, that the
handlers accept what flash actually sends, or that the lifecycle transitions the client waits on
ever happen. Those failures otherwise surface on a real GPU, after a multi-minute cold start.

One caught here already: `from __future__ import annotations` turns every annotation into a string,
and FastAPI resolves them against MODULE globals -- with the fastapi imports inside the app factory,
`request: Request` silently degraded into a required query parameter and every POST 422'd.
"""

from __future__ import annotations

import sys
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


class _FakeDict(dict):
    """modal.Dict, including the atomic insert-if-absent the alias lock is built on."""

    @classmethod
    def from_name(cls, *args, **kwargs):
        return cls()

    def put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = value
        return True


def _stub_modal(monkeypatch, engine_methods):
    modal = types.ModuleType("modal")

    class _Named:
        @classmethod
        def from_name(cls, *args, **kwargs):
            return cls()

    class _Image(_Named):
        @classmethod
        def debian_slim(cls, *args, **kwargs):
            return cls()

        def pip_install(self, *args, **kwargs):
            return self

        def env(self, *args, **kwargs):
            return self

    class _EngineHandle:
        def __getattr__(self, name):
            return types.SimpleNamespace(remote=types.SimpleNamespace(aio=engine_methods[name]))

    class _App:
        def __init__(self, *args, **kwargs):
            pass

        def cls(self, *args, **kwargs):
            return lambda klass: lambda *a, **k: _EngineHandle()

        def function(self, *args, **kwargs):
            return lambda fn: fn

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

    _stub_modal(
        monkeypatch,
        {"register": register, "unregister": unregister, "generate": generate},
    )
    source = render_app(MODELS[BASE_MODEL])
    module = types.ModuleType("generated_serving_app")
    exec(compile(source, str(tmp_path / "app.py"), "exec"), module.__dict__)
    return TestClient(module.api())


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
