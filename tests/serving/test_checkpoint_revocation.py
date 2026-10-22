"""A ready cache on another router must not survive a durable checkpoint disable."""

from contextlib import ExitStack

import pytest

from flash.serving.src.http.routing import AdapterRouter
from tests.serving.checkpoint_fixtures import checkpoint_record
from tests.serving.test_router import QWEN, FakePool, _serve


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("disabled_state", ["disabled", "failed", None])
def test_remote_disable_blocks_cached_checkpoint_before_dispatch(stream, disabled_state):
    ready = checkpoint_record("run-1", QWEN)
    stored = ready

    def lookup(org_id, model_id):
        assert (org_id, model_id) == (ready.org_id, ready.model_id)
        return stored

    pools = [FakePool(), FakePool()]
    body = {
        "model": ready.model_id,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
    }
    with ExitStack() as stack:
        clients = [
            stack.enter_context(_serve(pool, AdapterRouter([ready]), lookup_record=lookup))
            for pool in pools
        ]
        for client in clients:
            assert client.post("/v1/chat/completions", json=body).status_code == 200
        # Simulate an undeploy committed by a third router; neither local cache is invalidated.
        stored = ready.model_copy(update={"status": disabled_state}) if disabled_state else None
        for client in clients:
            assert client.post("/v1/chat/completions", json=body).status_code == 404
    assert [len(pool.generated) for pool in pools] == [1, 1]


def test_persisted_checkpoint_read_failure_does_not_serve_cached_ready_record():
    ready = checkpoint_record("run-1", QWEN)

    def unavailable(*_args):
        raise RuntimeError("database unavailable")

    pool = FakePool()
    with _serve(pool, AdapterRouter([ready]), lookup_record=unavailable) as client:
        response = client.post("/generate", json={"model_id": ready.model_id, "prompt": "hi"})
    assert response.status_code == 503
    assert response.json() == {"detail": "adapter storage is unavailable"}
    assert pool.generated == []


def test_lifecycle_read_observes_remote_disable_despite_ready_cache():
    ready = checkpoint_record("run-1", QWEN)
    disabled = ready.model_copy(update={"status": "disabled", "deployment_generation": None})
    with _serve(
        FakePool(),
        AdapterRouter([ready]),
        lookup_record=lambda org_id, model_id: disabled,
        internal_key="test-internal-key",
    ) as client:
        response = client.get(
            "/adapters/" + ready.model_id,
            headers={"X-Freesolo-Internal-Key": "test-internal-key"},
        )
    assert response.status_code == 200
    assert response.json()["adapter"]["status"] == "disabled"


def test_new_checkpoint_resolves_directly_from_persistence_without_bulk_reload():
    ready = checkpoint_record("run-1", QWEN)
    pool = FakePool()
    with _serve(pool, AdapterRouter(), lookup_record=lambda org_id, model_id: ready) as client:
        response = client.post("/generate", json={"model_id": ready.model_id, "prompt": "hi"})
    assert response.status_code == 200
    assert pool.generated == [(QWEN, ready.model_id)]
