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
import json
import time

import pytest

from flash.schema import format_adapter_revision

REQUIRED_CAPABILITIES = {"immutable_adapter_revisions", "alias_compare_and_swap"}

# Echoed back verbatim on read-back; the client compares each one to decide whether a revision id
# still names the artifact it registered.
IDENTITY_FIELDS = (
    "adapter_id",
    "repo_id",
    "repo_type",
    "subfolder",
    "base_model",
    "checkpoint",
    "thinking",
    # Compared by the client too, on every readiness poll. Only meaningful because registrations
    # here carry a nonempty org: with both sides None the check cannot fail.
    "org_id",
)

# The provenance metadata `_matches_revision_identity` cross-checks when the backend advertises
# `revision_provenance`. Asserted only in that case, matching the client's own `require_provenance`
# -- a backend that does not claim the capability is not expected to echo it.
PROVENANCE_FIELDS = ("record_type", "run_id", "checkpoint_step", "hf_revision")

# Any nonempty value works: the contract is that whatever was sent comes back unchanged.
CONFORMANCE_ORG_ID = "conformance-org"


def _record(payload: object) -> dict:
    """Unwrap a read-back the way the client does: bare record or under `adapter`."""
    assert isinstance(payload, dict), f"adapter read-back is not a JSON object: {payload!r}"
    inner = payload.get("adapter")
    return inner if isinstance(inner, dict) else payload


def _lifecycle_state(record: dict) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return str(metadata.get("lifecycle_state") or record.get("lifecycle_state") or "registered")


def _registration(run_id: str, source: dict, *, step: int | None = 10) -> dict:
    """Exactly the body `deploy_adapter` sends, minus the optional keys."""
    revision = format_adapter_revision(run_id, step, source["hf_revision"])
    checkpoint = f"{run_id}/step-{step}" if step is not None else run_id
    return {
        "adapter_id": revision,
        "repo_id": source["repo_id"],
        "base_model": source["base_model"],
        "subfolder": source["subfolder"],
        "repo_type": source["repo_type"],
        "checkpoint": checkpoint,
        # A NONEMPTY org, because the client compares `(record.get("org_id") or None)` against the
        # same on the request. Registering without one makes both sides None and the comparison
        # passes for a backend that drops the field entirely -- which then rejects every ordinary
        # managed-plane deploy, where the org is always set, as a different immutable identity.
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
    assert response.status_code in (200, 202), (
        f"registration returned {response.status_code}; the client accepts only 200/202. "
        f"body: {response.text[:400]}"
    )
    return response


def _wait_ready(http, revision: str, timeout: float) -> dict:
    """Poll to a settled state, honoring Retry-After exactly as the client does."""
    deadline = time.monotonic() + timeout
    delay = 1.0
    last = "registered"
    while time.monotonic() < deadline:
        response = http.get(f"/adapters/{revision}")
        if response.status_code != 404:
            assert response.status_code == 200, (
                f"read-back returned {response.status_code}: {response.text[:400]}"
            )
            record = _record(response.json())
            last = _lifecycle_state(record)
            if last == "failed" or record.get("status") == "disabled":
                metadata = record.get("metadata") or {}
                pytest.fail(f"revision {revision} reported failed: {metadata.get('failure')}")
            if last == "ready":
                return record
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                with contextlib.suppress(ValueError):
                    delay = max(0.5, min(float(retry_after), 30.0))
        time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
        delay = min(delay * 2, 15.0)
    pytest.fail(f"revision {revision} never reached ready (last state {last!r})")


@pytest.fixture
def deployed(http, adapter_source, run_id, ready_timeout):
    """Register one revision, wait for ready, and always undeploy the run afterwards."""
    body = _registration(run_id, adapter_source)
    try:
        _register(http, body)
        _wait_ready(http, body["adapter_id"], ready_timeout)
        yield body
    finally:
        # cleanup runs even when the test failed mid-way: these records are permanent on the target.
        http.delete(f"/adapters/{run_id}")


def _activate(http, revision: str, expected: str | None):
    return http.post(f"/adapters/{revision}/activate", json={"expected_adapter_revision": expected})


def test_healthz_advertises_the_required_capabilities(http):
    """The client reads this before every deploy and refuses to proceed without both strings."""
    response = http.get("/healthz")
    assert response.status_code == 200, f"/healthz returned {response.status_code}"
    payload = response.json()
    assert isinstance(payload, dict), "/healthz must return a JSON object"
    capabilities = set(payload.get("capabilities") or [])
    missing = REQUIRED_CAPABILITIES - capabilities
    assert not missing, f"/healthz does not advertise {sorted(missing)}; deploys will be refused"


def test_a_revision_registers_and_reaches_ready(deployed):
    """Covers the 422 class of failure: a schema that rejects what flash actually sends."""
    assert deployed["adapter_id"]


def test_readback_echoes_the_identity_the_client_cross_checks(http, deployed):
    """A dropped or renamed field reads as a different artifact under the same revision id."""
    record = _record(http.get(f"/adapters/{deployed['adapter_id']}").json())
    for field in IDENTITY_FIELDS:
        assert record.get(field) == deployed.get(field), (
            f"read-back {field}={record.get(field)!r}, registered {deployed.get(field)!r}"
        )
    # Provenance only when it is claimed. The client gates the same cross-check on the capability,
    # so requiring it unconditionally would fail a backend that is behaving correctly -- and
    # skipping it for one that DOES advertise it certifies a backend the client will refuse.
    capabilities = set(http.get("/healthz").json().get("capabilities") or [])
    if "revision_provenance" not in capabilities:
        return
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    expected = deployed.get("metadata") or {}
    for field in PROVENANCE_FIELDS:
        assert metadata.get(field) == expected.get(field), (
            f"this backend advertises revision_provenance, but read-back metadata.{field}="
            f"{metadata.get(field)!r} while {expected.get(field)!r} was registered. the client "
            f"cross-checks these on every readiness poll and refuses the deploy on a mismatch."
        )


def test_a_constrained_registration_is_echoed_back_unchanged(http, adapter_source, run_id):
    """`structured_outputs` must survive the round trip byte for byte.

    The client compares it exactly (`flash/serve/deploy.py`, alongside the scalar identity fields),
    and it is the one identity-bearing field that is OPTIONAL -- so a backend can accept the
    registration, drop the constraint, and look completely healthy. Every other test here registers
    without one, which means a suite that stops at those would certify a backend that fails every
    constrained deploy: `_wait_revision_ready` reads the constraint back as `None`, calls that a
    different artifact, and refuses. Constrained runs are ordinary, not an edge case.
    """
    body = {
        **_registration(run_id, adapter_source),
        "structured_outputs": {"regex": r"[ab]+"},
    }
    try:
        _register(http, body)
        record = _record(http.get(f"/adapters/{body['adapter_id']}").json())
        assert record.get("structured_outputs") == body["structured_outputs"], (
            f"read-back structured_outputs={record.get('structured_outputs')!r}, registered "
            f"{body['structured_outputs']!r}. The client compares this exactly, so every "
            f"constrained deploy against this backend will be refused as a different artifact."
        )
    finally:
        http.delete(f"/adapters/{run_id}")


def test_reregistering_identical_content_is_idempotent(http, deployed):
    """The client retries registration after an ambiguous 5xx; a retry must not fail."""
    _register(http, deployed)


def test_reregistering_different_content_under_one_revision_conflicts(http, deployed):
    """This IS `immutable_adapter_revisions`: one revision id, exactly one artifact, forever."""
    mutated = {**deployed, "subfolder": deployed["subfolder"].rstrip("/") + "-other"}
    response = http.post("/adapters", json=mutated)
    assert response.status_code == 409, (
        f"mutated re-registration returned {response.status_code}, expected 409; the backend "
        "advertises immutable_adapter_revisions but lets a revision id change artifact"
    )


def test_chat_is_refused_before_the_alias_is_activated(http, deployed):
    """Registration alone must not take traffic; activation is what publishes a revision."""
    response = http.post(
        "/v1/chat/completions",
        json={
            "model": deployed["metadata"]["run_id"],
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        },
    )
    assert response.status_code in (404, 503), (
        f"chat on an unactivated alias returned {response.status_code}, expected 404 or 503"
    )


def test_activation_returns_the_provenance_the_client_validates(http, deployed):
    """The client fails the deploy on any one of these five fields mismatching."""
    revision = deployed["adapter_id"]
    response = _activate(http, revision, None)
    assert response.status_code == 200, (
        f"activation returned {response.status_code}: {response.text[:400]}"
    )
    payload = response.json()
    assert payload.get("adapter_id") == deployed["metadata"]["run_id"]
    assert payload.get("target_adapter_revision") == revision
    assert payload.get("previous_adapter_revision") is None
    assert payload.get("checkpoint") == deployed["checkpoint"]
    updated_at = payload.get("updated_at")
    assert isinstance(updated_at, str), f"updated_at must be a string, got {updated_at!r}"
    assert updated_at.strip(), "updated_at must not be blank"


def test_the_alias_record_names_its_live_revision(http, deployed):
    """`metadata.alias_of` is how the client reconciles an activation whose response was lost."""
    revision = deployed["adapter_id"]
    assert _activate(http, revision, None).status_code == 200
    alias = _record(http.get(f"/adapters/{deployed['metadata']['run_id']}").json())
    metadata = alias.get("metadata") if isinstance(alias.get("metadata"), dict) else {}
    assert metadata.get("alias_of") == revision, (
        f"alias points at {metadata.get('alias_of')!r}, expected {revision!r}"
    )


def test_a_stale_compare_and_swap_is_rejected(http, deployed):
    """Two concurrent deploys of one run must not be able to overwrite each other silently."""
    revision = deployed["adapter_id"]
    assert _activate(http, revision, None).status_code == 200
    stale = _activate(http, revision, None)
    assert stale.status_code == 409, (
        f"a repeated expected_adapter_revision=None activation returned {stale.status_code}, "
        "expected 409; the alias has already moved past that expectation"
    )


def test_chat_resolves_the_alias_to_its_immutable_revision(http, deployed):
    """Users chat with a run id; the weights must come from the revision it currently targets."""
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
    )
    assert response.status_code == 200, (
        f"chat returned {response.status_code}: {response.text[:400]}"
    )
    payload = response.json()
    choices = payload.get("choices")
    assert isinstance(choices, list), f"chat returned a non-list choices: {choices!r}"
    assert choices, "chat returned no choices"
    assert isinstance(choices[0].get("message", {}).get("content"), str)


def test_undeploy_disables_the_alias_and_its_revisions(http, adapter_source, run_id, ready_timeout):
    """Not using the `deployed` fixture: this test IS the teardown, so it owns the whole run."""
    body = _registration(run_id, adapter_source)
    _register(http, body)
    _wait_ready(http, body["adapter_id"], ready_timeout)
    assert _activate(http, body["adapter_id"], None).status_code == 200

    response = http.delete(f"/adapters/{run_id}")
    assert response.status_code == 200, f"undeploy returned {response.status_code}"
    payload = response.json()
    assert payload.get("run_id") == run_id
    for field in ("disabled_aliases", "disabled_revisions"):
        value = payload.get(field)
        assert isinstance(value, list), f"{field} must be a list, got {value!r}"
        assert all(isinstance(item, str) for item in value), (
            f"{field} must contain only strings, got {value!r}"
        )
    assert run_id in payload["disabled_aliases"]
    assert body["adapter_id"] in payload["disabled_revisions"]


def test_undeploying_an_unknown_run_is_a_clean_404(http):
    """The client maps 404 to "nothing to undeploy" rather than an error."""
    response = http.delete("/adapters/conformance-run-never-existed")
    assert response.status_code == 404, (
        f"undeploying an unknown run returned {response.status_code}, expected 404"
    )


def test_chat_streams_deltas_the_way_the_cli_asks_for_them(http, deployed):
    """`flash models chat` streams; a backend that only serves non-streaming fails the CLI.

    `flash/serve/streaming.py` sends `stream: true` and decodes SSE `data:` frames, taking the
    text from `choices[0].delta.content` and stopping at `[DONE]`. A backend that ignores the flag
    and returns one JSON body, or emits frames without deltas, passes every other test here and
    then produces an empty answer in the shipped client.
    """
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
    with http.stream("POST", "/v1/chat/completions", json=body) as response:
        assert response.status_code == 200, f"streaming chat returned {response.status_code}"
        media = response.headers.get("content-type", "")
        assert "text/event-stream" in media, (
            f"streaming chat answered with content-type {media!r}; the client decodes SSE frames "
            f"and a plain JSON body yields no deltas"
        )
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
    assert saw_done, "the stream never sent the [DONE] sentinel the client stops on"
    assert "".join(deltas).strip(), (
        "the stream carried no delta content, so `flash models chat` would print nothing"
    )


def test_a_wrong_serving_key_is_rejected(http, serving_url, internal_key):
    """A backend that ignores the key passes every other test while being publicly writable.

    The rest of the suite only ever sends the CORRECT key, so it cannot tell "checks the key" from
    "ignores the header". This sends a deliberately wrong one at a mutating endpoint: registration
    is what an unauthenticated caller would abuse to load arbitrary adapters onto the GPU.

    Skipped when no key is configured, because a backend with no key set is legitimately open (a
    laptop, a private network) and the contract does not require one.
    """
    if not internal_key:
        pytest.skip("no FREESOLO_INTERNAL_KEY configured; the backend is intentionally open")
    import httpx

    with httpx.Client(base_url=serving_url, timeout=30.0) as client:
        response = client.post(
            "/adapters",
            json={"adapter_id": "conformance-unauthorized@final." + "0" * 40},
            headers={"X-Freesolo-Internal-Key": internal_key + "-wrong"},
        )
    assert response.status_code in (401, 403), (
        f"registration with a WRONG serving key returned {response.status_code}; the backend does "
        f"not check the key, so registration, activation, chat, and deletion are publicly callable"
    )
