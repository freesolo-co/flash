"""Run the permanent checkpoint contract against an explicitly configured live backend."""

from __future__ import annotations

import contextlib
import json
import time

import pytest

from flash.schema import format_checkpoint_ref
from flash.serve.contract.protocol import PERMANENT_CHECKPOINT_IDENTITY_CAPABILITY
from flash.serve.deployment.adapter_check import adapter_artifact_metadata

CONFORMANCE_ORG_ID = "conformance-org"
_UNAUTHORIZED_RUN_ID = "conformance-unauthorized"
_CLIENT_READBACK_BASE_DELAY = 0.5
_CLIENT_READBACK_MAX_DELAY = 2.0


def _record(payload: object) -> dict:
    assert isinstance(payload, dict)
    inner = payload.get("adapter")
    return inner if isinstance(inner, dict) else payload


def _lifecycle_state(record: dict) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return str(metadata.get("lifecycle_state") or record.get("lifecycle_state") or "registered")


def _readback_delay(attempt: int, retry_after: str | None = None) -> float:
    if retry_after is not None:
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            pass
        else:
            if delay > 0 and delay == delay and delay != float("inf"):
                return min(delay, _CLIENT_READBACK_MAX_DELAY)
    return min(_CLIENT_READBACK_BASE_DELAY * (2**attempt), _CLIENT_READBACK_MAX_DELAY)


def _registration(run_id: str, source: dict, *, step: int | None = 10) -> dict:
    checkpoint_id = format_checkpoint_ref(run_id, step)
    metadata = adapter_artifact_metadata(
        source["repo_id"],
        source["subfolder"],
        artifact_revision=source["hf_revision"],
    )
    return {
        "adapter_id": checkpoint_id,
        "repo_id": source["repo_id"],
        "base_model": source["base_model"],
        "subfolder": source["subfolder"],
        "repo_type": source["repo_type"],
        "checkpoint": checkpoint_id,
        "org_id": CONFORMANCE_ORG_ID,
        "run_id": run_id,
        "checkpoint_step": step,
        "artifact_revision": source["hf_revision"],
        "artifact_digest": metadata.artifact_digest,
        "artifact_fingerprint": metadata.artifact_digest,
        "lora_rank": metadata.lora_rank,
        "thinking": False,
    }


def _register(http, body: dict):
    response = http.post("/adapters", json=body)
    assert response.status_code in (200, 202), response.text[:400]
    return response


def _wait_ready(http, checkpoint_id: str, timeout: float, expected: dict | None = None) -> dict:
    deadline = time.monotonic() + timeout
    attempt = 0
    last = "registered"
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        response = http.get(f"/adapters/{checkpoint_id}", timeout=min(60.0, remaining))
        if response.status_code == 200:
            record = _record(response.json())
            if expected is not None:
                for field in (
                    "adapter_id",
                    "repo_id",
                    "repo_type",
                    "subfolder",
                    "base_model",
                    "checkpoint",
                    "thinking",
                ):
                    assert record.get(field) == expected.get(field)
            last = _lifecycle_state(record)
            if last == "failed" or record.get("status") == "disabled":
                pytest.fail(f"checkpoint {checkpoint_id} reported {last}")
            if last == "ready":
                return record
        elif response.status_code < 500 and response.status_code != 404:
            pytest.fail(
                f"checkpoint readback returned {response.status_code}: {response.text[:400]}"
            )
        delay = _readback_delay(attempt, response.headers.get("Retry-After"))
        attempt += 1
        time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
    pytest.fail(f"checkpoint {checkpoint_id} never reached ready within {timeout}s, last={last!r}")


@pytest.fixture
def deployed(http, adapter_source, run_id, ready_timeout):
    body = _registration(run_id, adapter_source)
    try:
        _register(http, body)
        _wait_ready(http, body["adapter_id"], ready_timeout, expected=body)
        yield body
    finally:
        with contextlib.suppress(Exception):
            http.delete(f"/adapters/{body['adapter_id']}")


def test_healthz_advertises_permanent_checkpoint_identity(http) -> None:
    response = http.get("/healthz")
    assert response.status_code == 200
    capabilities = response.json().get("capabilities")
    assert isinstance(capabilities, list)
    assert PERMANENT_CHECKPOINT_IDENTITY_CAPABILITY in capabilities


def test_readback_echoes_the_exact_checkpoint_identity(http, deployed) -> None:
    record = _record(http.get(f"/adapters/{deployed['adapter_id']}").json())
    assert record["adapter_id"] == deployed["adapter_id"]
    assert record["checkpoint"] == deployed["checkpoint"]
    assert record["adapter_id"] == record["checkpoint"]
    serialized = json.dumps(record)
    assert "adapter_revision" not in serialized
    assert "hf_revision" not in serialized


def test_identical_reregistration_is_idempotent(http, deployed) -> None:
    _register(http, deployed)
    record = _record(http.get(f"/adapters/{deployed['adapter_id']}").json())
    assert record["adapter_id"] == deployed["adapter_id"]


def test_changed_immutable_binding_conflicts(http, deployed) -> None:
    response = http.post("/adapters", json={**deployed, "artifact_digest": "0" * 64})
    assert response.status_code == 409


def test_bare_and_composite_model_identities_are_rejected(http, deployed) -> None:
    run_id = deployed["run_id"]
    invalid = (run_id, run_id + "@final." + "a" * 40)
    for model in invalid:
        response = http.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
        )
        assert response.status_code in (400, 404, 422)


def test_exact_checkpoint_chat_returns_checkpoint_only_provenance(
    http, deployed, chat_timeout
) -> None:
    checkpoint_id = deployed["adapter_id"]
    response = http.post(
        "/v1/chat/completions",
        json={
            "model": checkpoint_id,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 16,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=chat_timeout,
    )
    assert response.status_code == 200, response.text[:400]
    payload = response.json()
    assert payload.get("freesolo") == {"checkpoint_id": checkpoint_id}
    assert response.headers.get("X-Freesolo-Checkpoint") == checkpoint_id
    serialized = json.dumps(payload)
    assert "adapter_revision" not in serialized
    assert "hf_revision" not in serialized


def test_sibling_undeploy_preserves_the_other_checkpoint(
    http, adapter_source, run_id, ready_timeout
) -> None:
    first = _registration(run_id, adapter_source, step=10)
    second = _registration(run_id, adapter_source, step=20)
    try:
        for body in (first, second):
            _register(http, body)
            _wait_ready(http, body["adapter_id"], ready_timeout, expected=body)
        response = http.delete(f"/adapters/{first['adapter_id']}")
        assert response.status_code == 200
        payload = response.json()
        assert payload.get("checkpoint_id") == first["adapter_id"]
        assert payload.get("disabled_checkpoints") == [first["adapter_id"]]
        sibling = _record(http.get(f"/adapters/{second['adapter_id']}").json())
        assert sibling.get("status") == "ready"
        assert _lifecycle_state(sibling) == "ready"
    finally:
        for body in (first, second):
            with contextlib.suppress(Exception):
                http.delete(f"/adapters/{body['adapter_id']}")


def test_streaming_uses_the_exact_checkpoint(http, deployed, chat_timeout) -> None:
    body = {
        "model": deployed["adapter_id"],
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 16,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
    }
    deltas: list[str] = []
    saw_done = False
    with http.stream("POST", "/v1/chat/completions", json=body, timeout=chat_timeout) as response:
        assert response.status_code == 200
        assert response.headers.get("X-Freesolo-Checkpoint") == deployed["adapter_id"]
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                saw_done = True
                break
            delta = (json.loads(data).get("choices") or [{}])[0].get("delta") or {}
            if isinstance(delta.get("content"), str):
                deltas.append(delta["content"])
    assert saw_done
    assert "".join(deltas).strip()


def test_a_wrong_serving_key_is_rejected(
    serving_client_factory, serving_key, adapter_source
) -> None:
    if not serving_key:
        pytest.skip("no FLASH_SERVING_KEY configured; the backend is intentionally open")
    body = _registration(_UNAUTHORIZED_RUN_ID, adapter_source, step=None)
    probes = (
        ("POST", "/adapters", body),
        ("GET", f"/adapters/{body['adapter_id']}", None),
        (
            "POST",
            "/v1/chat/completions",
            {"model": body["adapter_id"], "messages": [{"role": "user", "content": "hi"}]},
        ),
        ("DELETE", f"/adapters/{body['adapter_id']}", None),
    )
    unprotected: list[str] = []
    with serving_client_factory() as client:
        client.headers["Authorization"] = f"Bearer {serving_key}-wrong"
        for method, path, payload in probes:
            response = client.request(method, path, json=payload)
            if response.status_code not in (401, 403):
                unprotected.append(f"{method} {path} -> {response.status_code}")
    assert not unprotected
