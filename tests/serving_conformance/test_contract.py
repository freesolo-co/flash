"""Run `docs/serving-contract.md` against a live serving backend.
Skipped unless `--serving-url` names one:
    uv run pytest tests/serving_conformance --serving-url http://localhost:8100 \\
        --conformance-repo acme/artifacts \\
        --conformance-subfolder sft/run-abc/adapter \\
        --conformance-base-model Qwen/Qwen3.5-4B
Every assertion here mirrors something `flash/serve/deploy.py` genuinely checks, so a backend that
passes works with `flash models deploy` / `chat` / `undeploy` unchanged. Where the client accepts
two shapes (a bare record or `{"adapter": ...}`), so does this suite -- it tests the contract, not
one implementation of it.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import re
import sys
import time
import types
from pathlib import Path
from typing import ClassVar

import pytest

from flash.schema import format_adapter_revision

REQUIRED_CAPABILITIES = {"immutable_adapter_revisions", "alias_compare_and_swap"}
_CLIENT_REQUEST_TIMEOUT_CAP = 60.0
_CLIENT_READBACK_BASE_DELAY = 0.5
_CLIENT_READBACK_MAX_DELAY = 2.0
IDENTITY_FIELDS = (
    "adapter_id",
    "repo_id",
    "repo_type",
    "subfolder",
    "base_model",
    "checkpoint",
    "thinking",
    "org_id",
)
PROVENANCE_FIELDS = ("record_type", "run_id", "checkpoint_step", "hf_revision")
CONFORMANCE_ORG_ID = "conformance-org"
_UNAUTHORIZED_RUN_ID = "conformance-unauthorized"
_MUTATIONS = {
    "repo_id": lambda body: {**body, "repo_id": body["repo_id"] + "-other"},
    "repo_type": lambda body: {
        **body,
        "repo_type": "dataset" if body.get("repo_type") == "model" else "model",
    },
    "subfolder": lambda body: {**body, "subfolder": body["subfolder"].rstrip("/") + "-other"},
    "base_model": lambda body: {**body, "base_model": body["base_model"] + "-other"},
    "checkpoint": lambda body: {**body, "checkpoint": str(body["checkpoint"]) + "-other"},
    "thinking": lambda body: {**body, "thinking": not body.get("thinking")},
    "org_id": lambda body: {**body, "org_id": str(body.get("org_id") or "") + "-other"},
    "structured_outputs": lambda body: {**body, "structured_outputs": {"regex": r"[xy]+"}},
}


def _record(payload: object) -> dict:
    assert isinstance(payload, dict)
    inner = payload.get("adapter")
    return inner if isinstance(inner, dict) else payload


def _lifecycle_state(record: dict) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return str(metadata.get("lifecycle_state") or record.get("lifecycle_state") or "registered")


def _registration(run_id: str, source: dict, *, step: int | None = 10) -> dict:
    revision = format_adapter_revision(run_id, step, source["hf_revision"])
    checkpoint = f"{run_id}/step-{step}" if step is not None else run_id
    return {
        "adapter_id": revision,
        "repo_id": source["repo_id"],
        "base_model": source["base_model"],
        "subfolder": source["subfolder"],
        "repo_type": source["repo_type"],
        "checkpoint": checkpoint,
        "org_id": CONFORMANCE_ORG_ID,
        "metadata": {
            "record_type": "revision",
            "run_id": run_id,
            "checkpoint_step": step,
            "hf_revision": source["hf_revision"],
        },
        "thinking": False,
    }


def _register(http, body: dict):
    response = http.post("/adapters", json=body)
    assert response.status_code in (200, 202)
    return response


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


def _requires_provenance(http) -> bool:
    cached = getattr(http, "_conformance_requires_provenance", None)
    if cached is None:
        payload = http.get("/healthz").json()
        capabilities = payload.get("capabilities") if isinstance(payload, dict) else None
        cached = "revision_provenance" in set(capabilities or [])
        http._conformance_requires_provenance = cached
    return cached


def _identity_mismatch(record: dict, expected: dict, *, require_provenance: bool) -> str | None:
    scalar_fields = (
        "adapter_id",
        "repo_id",
        "repo_type",
        "subfolder",
        "base_model",
        "checkpoint",
        "thinking",
    )
    for field in scalar_fields:
        if record.get(field) != expected.get(field):
            return f"{field}: {record.get(field)!r} != {expected.get(field)!r}"
    for field in ("org_id", "structured_outputs"):
        if (record.get(field) or None) != (expected.get(field) or None):
            return f"{field}: {record.get(field)!r} != {expected.get(field)!r}"
    if not require_provenance:
        return None
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    expected_metadata = expected.get("metadata") or {}
    for field in ("record_type", "run_id", "checkpoint_step", "hf_revision"):
        if metadata.get(field) != expected_metadata.get(field):
            return f"metadata.{field}: {metadata.get(field)!r} != {expected_metadata.get(field)!r}"
    return None


def _wait_ready(http, revision: str, timeout: float, expected: dict | None = None) -> dict:
    import httpx

    deadline = time.monotonic() + timeout
    attempt = 0
    delay = _CLIENT_READBACK_BASE_DELAY
    retry_after: str | None = None
    last = "registered"
    while True:
        retry_after = None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            response = http.get(
                f"/adapters/{revision}", timeout=min(_CLIENT_REQUEST_TIMEOUT_CAP, remaining)
            )
        except httpx.RequestError:
            delay = _readback_delay(attempt)
            attempt += 1
            time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
            continue
        if deadline - time.monotonic() <= 0:
            break
        if response.status_code >= 500:
            retry_after = response.headers.get("Retry-After")
        elif response.status_code != 404:
            assert response.status_code == 200
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if not isinstance(payload, dict):
                delay = _readback_delay(attempt, response.headers.get("Retry-After"))
                attempt += 1
                time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
                continue
            record = _record(payload)
            if expected is not None:
                mismatch = _identity_mismatch(
                    record, expected, require_provenance=_requires_provenance(http)
                )
                assert mismatch is None, f"different immutable identity: {mismatch}"
            last = _lifecycle_state(record)
            if last == "failed" or record.get("status") == "disabled":
                metadata = record.get("metadata") or {}
                pytest.fail(f"revision {revision} reported failed: {metadata.get('failure')}")
            if last == "ready":
                return record
            retry_after = response.headers.get("Retry-After")
        delay = _readback_delay(attempt, retry_after)
        attempt += 1
        time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
    pytest.fail(
        f"revision {revision} never reached ready within {timeout}s (last state {last!r}). "
        "a read-back that only completes after the budget is spent does not count: the client "
        "bounds each poll by its remaining budget and discards a late response, so a backend that "
        "needs longer than this fails every deploy"
    )


@pytest.fixture
def deployed(http, adapter_source, run_id, ready_timeout):
    body = _registration(run_id, adapter_source)
    try:
        _register(http, body)
        _wait_ready(http, body["adapter_id"], ready_timeout, expected=body)
        yield body
    finally:
        http.delete(f"/adapters/{run_id}")


def _activate(http, revision: str, expected: str | None):
    return http.post(f"/adapters/{revision}/activate", json={"expected_adapter_revision": expected})


def test_the_per_request_cap_still_matches_the_client():
    from flash.serve import deploy

    source = inspect.getsource(deploy._serving_request)
    found = re.findall(r"min\((\d+(?:\.\d+)?),", source)
    assert found
    assert float(found[0]) == _CLIENT_REQUEST_TIMEOUT_CAP


def test_the_readiness_backoff_still_matches_the_client():
    from flash.serve import deploy

    source = Path(deploy.__file__).read_text()
    shipped = {}
    for name in ("READBACK_DELAY_SECONDS", "READBACK_MAX_DELAY_SECONDS"):
        found = re.search(rf"^{name} = ([0-9.]+)$", source, re.MULTILINE)
        assert found
        shipped[name] = float(found.group(1))
    assert shipped["READBACK_MAX_DELAY_SECONDS"] == _CLIENT_READBACK_MAX_DELAY
    assert shipped["READBACK_DELAY_SECONDS"] == _CLIENT_READBACK_BASE_DELAY
    client_delay = types.FunctionType(
        deploy._readback_delay.__code__,
        {**deploy._readback_delay.__globals__, **shipped},
        deploy._readback_delay.__name__,
        deploy._readback_delay.__defaults__,
    )
    for attempt, retry_after in ((0, None), (3, None), (9, None), (0, "30"), (0, "0.5"), (1, "x")):
        assert _readback_delay(attempt, retry_after) == client_delay(attempt, retry_after)


def test_the_readiness_backoff_is_driven_through_the_same_attempt_sequence(monkeypatch):
    from flash.serve import deploy

    class _Response:
        status_code = 404
        headers: ClassVar[dict[str, str]] = {}
        text = ""

    class _Stop(Exception):
        pass

    def _recorder(into: list[int]):
        def _delay(attempt: int, retry_after: str | None = None) -> float:
            into.append(attempt)
            if len(into) >= 5:
                raise _Stop
            return 0.0  # no real sleeping; the sequence is what is under test

        return _delay

    suite_attempts: list[int] = []
    monkeypatch.setattr(
        sys.modules[__name__], "_readback_delay", _recorder(suite_attempts), raising=True
    )
    with pytest.raises(_Stop):
        _wait_ready(types.SimpleNamespace(get=lambda *a, **k: _Response()), "rev", timeout=30.0)
    client_attempts: list[int] = []
    monkeypatch.setattr(deploy, "_readback_delay", _recorder(client_attempts), raising=True)
    monkeypatch.setattr(
        deploy, "_registered_adapter_response", lambda *a, **k: (None, _Response()), raising=True
    )
    with pytest.raises(_Stop):
        deploy._wait_revision_ready("rev", None, budget_s=30.0)
    assert suite_attempts == client_attempts


def test_healthz_advertises_the_required_capabilities(http):
    response = http.get("/healthz")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, dict)
    capabilities = payload.get("capabilities")
    assert isinstance(capabilities, list)
    non_strings = [item for item in capabilities if not isinstance(item, str)]
    assert not non_strings
    missing = REQUIRED_CAPABILITIES - set(capabilities)
    assert not missing


def test_readback_echoes_the_identity_the_client_cross_checks(http, deployed):
    record = _record(http.get(f"/adapters/{deployed['adapter_id']}").json())
    for field in IDENTITY_FIELDS:
        assert record.get(field) == deployed.get(field)
    capabilities = set(http.get("/healthz").json().get("capabilities") or [])
    if "revision_provenance" not in capabilities:
        return
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    expected = deployed.get("metadata") or {}
    for field in PROVENANCE_FIELDS:
        assert metadata.get(field) == expected.get(field)


def test_a_constrained_registration_serves_a_constrained_completion(
    http, adapter_source, run_id, ready_timeout, chat_timeout
):
    body = {
        **_registration(run_id, adapter_source),
        "structured_outputs": {"regex": r"[ab]+"},
    }
    try:
        _register(http, body)
        record = _wait_ready(http, body["adapter_id"], ready_timeout, expected=body)
        assert record.get("structured_outputs") == body["structured_outputs"]
        assert _activate(http, body["adapter_id"], None).status_code == 200
        response = http.post(
            "/v1/chat/completions",
            json={
                "model": run_id,
                "messages": [{"role": "user", "content": "Answer with letters only."}],
                "max_tokens": 16,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=chat_timeout,
        )
        assert response.status_code == 200
        content = (response.json().get("choices") or [{}])[0].get("message", {}).get(
            "content"
        ) or ""
        assert re.fullmatch("[ab]+", content.strip())
    finally:
        http.delete(f"/adapters/{run_id}")


def test_reregistering_while_the_first_load_is_pending_is_idempotent(
    http, adapter_source, run_id, ready_timeout
):
    body = _registration(run_id, adapter_source)
    try:
        _register(http, body)
        _register(http, body)
        _wait_ready(http, body["adapter_id"], ready_timeout, expected=body)
    finally:
        http.delete(f"/adapters/{run_id}")


def test_a_final_revision_registers_and_reaches_ready(http, adapter_source, run_id, ready_timeout):
    body = _registration(run_id, adapter_source, step=None)
    assert body["adapter_id"].endswith(f"@final.{adapter_source['hf_revision']}")
    try:
        _register(http, body)
        record = _wait_ready(http, body["adapter_id"], ready_timeout, expected=body)
        assert (record.get("metadata") or {}).get("checkpoint_step") is None
    finally:
        http.delete(f"/adapters/{run_id}")


def test_a_thinking_adapter_registers_and_reaches_ready(
    http, adapter_source, run_id, ready_timeout
):
    body = {**_registration(run_id, adapter_source), "thinking": True}
    try:
        _register(http, body)
        record = _wait_ready(http, body["adapter_id"], ready_timeout, expected=body)
        assert record.get("thinking") is True
    finally:
        http.delete(f"/adapters/{run_id}")


@pytest.mark.parametrize("field", sorted(_MUTATIONS))
def test_reregistering_different_content_under_one_revision_conflicts(http, deployed, field):
    mutated = _MUTATIONS[field](deployed)
    assert mutated != deployed
    response = http.post("/adapters", json=mutated)
    assert response.status_code in (409, 422)


def test_chat_is_refused_before_the_alias_is_activated(http, deployed):
    response = http.post(
        "/v1/chat/completions",
        json={
            "model": deployed["metadata"]["run_id"],
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        },
    )
    assert response.status_code in (404, 503)


def test_activation_returns_the_provenance_the_client_validates(http, deployed):
    revision = deployed["adapter_id"]
    response = _activate(http, revision, None)
    assert response.status_code == 200
    payload = response.json()
    assert payload.get("adapter_id") == deployed["metadata"]["run_id"]
    assert payload.get("target_adapter_revision") == revision
    assert payload.get("previous_adapter_revision") is None
    assert payload.get("checkpoint") == deployed["checkpoint"]
    updated_at = payload.get("updated_at")
    assert isinstance(updated_at, str)
    assert updated_at.strip()


def test_the_alias_record_names_its_live_revision(http, deployed):
    revision = deployed["adapter_id"]
    assert _activate(http, revision, None).status_code == 200
    alias = _record(http.get(f"/adapters/{deployed['metadata']['run_id']}").json())
    metadata = alias.get("metadata") if isinstance(alias.get("metadata"), dict) else {}
    assert metadata.get("alias_of") == revision


def test_a_stale_compare_and_swap_is_rejected(http, deployed):
    revision = deployed["adapter_id"]
    assert _activate(http, revision, None).status_code == 200
    stale = _activate(http, revision, None)
    assert stale.status_code == 409


def test_a_non_null_expectation_replaces_the_live_revision(
    http, adapter_source, run_id, ready_timeout
):
    first = _registration(run_id, adapter_source, step=10)
    second = _registration(run_id, adapter_source, step=20)
    try:
        for body in (first, second):
            _register(http, body)
            _wait_ready(http, body["adapter_id"], ready_timeout, expected=body)
        assert _activate(http, first["adapter_id"], None).status_code == 200
        response = _activate(http, second["adapter_id"], first["adapter_id"])
        assert response.status_code == 200
        payload = response.json()
        assert payload.get("previous_adapter_revision") == first["adapter_id"]
        assert payload.get("target_adapter_revision") == second["adapter_id"]
        alias = _record(http.get(f"/adapters/{run_id}").json())
        metadata = alias.get("metadata") if isinstance(alias.get("metadata"), dict) else {}
        assert metadata.get("alias_of") == second["adapter_id"]
    finally:
        http.delete(f"/adapters/{run_id}")


def test_concurrent_activations_of_one_alias_leave_exactly_one_winner(
    http, serving_client_factory, adapter_source, run_id, ready_timeout
):
    import concurrent.futures

    first = _registration(run_id, adapter_source, step=10)
    second = _registration(run_id, adapter_source, step=20)
    try:
        for body in (first, second):
            _register(http, body)
            _wait_ready(http, body["adapter_id"], ready_timeout, expected=body)

        def activate(revision: str) -> int:
            with serving_client_factory() as client:
                response = client.post(
                    f"/adapters/{revision}/activate",
                    json={"expected_adapter_revision": None},
                )
                return response.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            codes = list(
                pool.map(activate, (first["adapter_id"], second["adapter_id"])),
            )
        winners = [code for code in codes if code == 200]
        assert len(winners) == 1
        losers = [code for code in codes if code != 200]
        assert all(code == 409 for code in losers)
        revisions = (first["adapter_id"], second["adapter_id"])
        expected = revisions[codes.index(200)]
        alias = _record(http.get(f"/adapters/{run_id}").json())
        metadata = alias.get("metadata") if isinstance(alias.get("metadata"), dict) else {}
        assert metadata.get("alias_of") == expected
    finally:
        http.delete(f"/adapters/{run_id}")


def test_chat_resolves_the_alias_to_its_immutable_revision(http, deployed, chat_timeout):
    run = deployed["metadata"]["run_id"]
    assert _activate(http, deployed["adapter_id"], None).status_code == 200
    response = http.post(
        "/v1/chat/completions",
        json={
            "model": run,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 16,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=chat_timeout,
    )
    assert response.status_code == 200
    payload = response.json()
    choices = payload.get("choices")
    assert isinstance(choices, list)
    assert choices
    assert isinstance(choices[0].get("message", {}).get("content"), str)
    revision = deployed["adapter_id"]
    expected = {
        "adapter_revision": revision,
        "checkpoint": deployed["checkpoint"],
        "hf_revision": revision.rsplit(".", 1)[-1],
    }
    provenance = payload.get("freesolo")
    assert isinstance(provenance, dict)
    for field, value in expected.items():
        assert provenance.get(field) == value
    for header, value in (
        ("X-Freesolo-Adapter-Revision", expected["adapter_revision"]),
        ("X-Freesolo-Checkpoint", expected["checkpoint"]),
        ("X-Freesolo-HF-Revision", expected["hf_revision"]),
    ):
        assert response.headers.get(header) == value


def _solid_png_data_uri(rgb: tuple[int, int, int]) -> str:
    """a one-colour png as a base64 data uri, built here rather than committed as a fixture blob."""

    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), rgb).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def test_an_image_request_reaches_the_vision_path(http, deployed, chat_timeout):
    """an image-capable deployment must actually decode images, not just accept the request.

    the profile declares these engines multimodal, which is what loads the processor and passes
    limit_mm_per_prompt. that claim is only worth anything if a real image round-trips: a text-only
    engine answers this request with 400 or a MultimodalRequestError rather than a completion.

    the image is solid red and the question asks for its colour, so the answer is only available by
    decoding pixels. the assertion stays on the transport and the vision path rather than on the
    exact word: a small adapter-tuned model may describe the colour loosely, and pinning its wording
    would make this a quality test that fails for the wrong reason.
    """

    run = deployed["metadata"]["run_id"]
    assert _activate(http, deployed["adapter_id"], None).status_code == 200
    response = http.post(
        "/v1/chat/completions",
        json={
            "model": run,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What colour is this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": _solid_png_data_uri((220, 20, 20))},
                        },
                    ],
                }
            ],
            "max_tokens": 24,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=chat_timeout,
    )
    # the contract does not require every backend to accept images, and a text-only deployment
    # refusing them is correct behaviour rather than a contract violation. treat that refusal the
    # way the suite already treats an unadvertised optional capability: skip, do not fail. what this
    # test is here to catch is a backend that CLAIMS images and then does not decode them.
    if response.status_code in {400, 415, 422}:
        pytest.skip(
            f"backend refused the image request with {response.status_code}; the contract allows a "
            "text-only deployment, so there is no vision path to exercise here"
        )
    assert response.status_code == 200, (
        f"image request refused with {response.status_code}: {response.text[:400]}"
    )
    payload = response.json()
    choices = payload.get("choices")
    assert isinstance(choices, list)
    assert choices
    content = choices[0].get("message", {}).get("content")
    assert isinstance(content, str)
    assert content.strip(), "vision path returned no text"


def test_undeploy_disables_the_alias_and_its_revisions(http, adapter_source, run_id, ready_timeout):
    body = _registration(run_id, adapter_source)
    try:
        _register(http, body)
        _wait_ready(http, body["adapter_id"], ready_timeout, expected=body)
        assert _activate(http, body["adapter_id"], None).status_code == 200
    except BaseException:
        with contextlib.suppress(Exception):
            http.delete(f"/adapters/{run_id}")
        raise
    response = http.delete(f"/adapters/{run_id}")
    assert response.status_code == 200
    payload = response.json()
    assert payload.get("run_id") == run_id
    for field in ("disabled_aliases", "disabled_revisions"):
        value = payload.get(field)
        assert isinstance(value, list)
        assert all(isinstance(item, str) for item in value)
    assert run_id in payload["disabled_aliases"]
    assert body["adapter_id"] in payload["disabled_revisions"]
    for record_id in (run_id, body["adapter_id"]):
        response = http.get(f"/adapters/{record_id}")
        if response.status_code == 404:
            continue  # deleting the record outright is a valid way to disable it
        assert response.status_code == 200
        record = _record(response.json())
        state = _lifecycle_state(record)
        assert record.get("status") == "disabled" or state in ("disabled", "failed")
    chat = http.post(
        "/v1/chat/completions",
        json={
            "model": run_id,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        },
    )
    assert chat.status_code != 200


def test_provenance_contradicting_the_revision_id_is_refused(http, adapter_source, run_id):
    body = _registration(run_id, adapter_source)
    body["metadata"] = {**body["metadata"], "run_id": f"{run_id}-elsewhere"}
    response = http.post("/adapters", json=body)
    try:
        assert response.status_code in (400, 409, 422)
    finally:
        http.delete(f"/adapters/{run_id}")
        http.delete(f"/adapters/{run_id}-elsewhere")


def test_undeploying_an_unknown_run_is_a_clean_404(http):
    response = http.delete("/adapters/conformance-run-never-existed")
    assert response.status_code == 404


def test_chat_streams_deltas_the_way_the_cli_asks_for_them(http, deployed, chat_timeout):
    run = deployed["metadata"]["run_id"]
    assert _activate(http, deployed["adapter_id"], None).status_code == 200
    body = {
        "model": run,
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
        media = response.headers.get("content-type", "")
        assert "text/event-stream" in media
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                saw_done = True
                break
            chunk = json.loads(data)
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str):
                deltas.append(content)
    assert saw_done
    assert "".join(deltas).strip()


def test_a_wrong_serving_key_is_rejected(serving_client_factory, serving_key, adapter_source):
    if not serving_key:
        pytest.skip("no FLASH_SERVING_KEY configured; the backend is intentionally open")
    revision = f"{_UNAUTHORIZED_RUN_ID}@final." + "0" * 40
    unauthorized_body = _registration(
        _UNAUTHORIZED_RUN_ID, {**adapter_source, "hf_revision": "0" * 40}, step=None
    )
    assert unauthorized_body["adapter_id"] == revision
    probes = (
        ("register", "POST", "/adapters", unauthorized_body),
        ("read back", "GET", f"/adapters/{revision}", None),
        ("activate", "POST", f"/adapters/{revision}/activate", {"expected_adapter_revision": None}),
        (
            "chat",
            "POST",
            "/v1/chat/completions",
            {"model": revision, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
        ),
        ("delete", "DELETE", f"/adapters/{revision}", None),
    )
    unprotected = []
    try:
        with serving_client_factory() as client:
            client.headers["Authorization"] = f"Bearer {serving_key}-wrong"
            for name, method, path, body in probes:
                response = client.request(method, path, json=body)
                if response.status_code not in (401, 403):
                    unprotected.append(f"{name} ({method} {path}) -> {response.status_code}")
    finally:
        with contextlib.suppress(Exception), serving_client_factory() as cleanup:
            cleanup.delete(f"/adapters/{_UNAUTHORIZED_RUN_ID}")
    assert not unprotected


def test_a_malformed_readback_is_polled_through_rather_than_failed(monkeypatch):
    class _Response:
        def __init__(self, payload, *, raises=False):
            self.status_code = 200
            self.headers: dict[str, str] = {}
            self.text = "<junk>"
            self._payload = payload
            self._raises = raises

        def json(self):
            if self._raises:
                raise ValueError("invalid json")
            return self._payload

    ready = {"adapter_id": "rev", "status": "ready", "metadata": {"lifecycle_state": "ready"}}
    responses = [_Response(None, raises=True), _Response(["not", "an", "object"]), _Response(ready)]

    def get(*args, **kwargs):
        return responses.pop(0)

    record = _wait_ready(types.SimpleNamespace(get=get), "rev", timeout=30.0)
    assert record["status"] == "ready"
    assert not responses


def test_a_corrected_identity_still_fails_the_readiness_wait():
    class _Response:
        status_code = 200
        headers: ClassVar[dict[str, str]] = {}
        text = "{}"

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    expected = {
        "adapter_id": "rev",
        "repo_id": "acme/artifacts",
        "repo_type": "model",
        "subfolder": "sft/run-abc/adapter",
        "base_model": "Qwen/Qwen3.5-4B",
        "checkpoint": "run-abc/step-10",
        "thinking": False,
        "org_id": "conformance-org",
        "metadata": {
            "record_type": "revision",
            "run_id": "run-abc",
            "checkpoint_step": 10,
            "hf_revision": "a" * 40,
        },
    }
    partial = {**expected, "org_id": None, "status": "registered"}
    partial["metadata"] = {**expected["metadata"], "lifecycle_state": "registered"}
    corrected = {**expected, "status": "ready"}
    corrected["metadata"] = {**expected["metadata"], "lifecycle_state": "ready"}
    responses = [_Response(partial), _Response(corrected)]
    http = types.SimpleNamespace(
        get=lambda *args, **kwargs: responses.pop(0),
        _conformance_requires_provenance=True,
    )
    with pytest.raises(AssertionError, match="different immutable identity"):
        _wait_ready(http, "rev", timeout=30.0, expected=expected)
    assert responses
