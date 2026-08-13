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
import re
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

# One mutation per identity-bearing field CLASS, each changing something the client compares on
# read-back. A backend that pins only the artifact path passes a subfolder-only check while still
# letting a revision id resolve to different weights, rendering, tenancy, or grammar.
#
# `adapter_id` is deliberately absent: changing it is a DIFFERENT revision, not a mutation of this
# one. The provenance fields are covered by their own test, which asserts the stronger property
# that they must agree with the id itself.
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
    """The client reads this before every deploy and refuses to proceed without both strings.

    The WIRE SHAPE is asserted before the contents. `set()` accepts any iterable, so a backend that
    returns `capabilities` as an object mapping names to booleans -- or as a bare string -- yields
    the right names here and passes, while the client's own parse
    (`flash/serve/deploy.py`: `isinstance(capabilities, list)` then every element a `str`) raises
    `serving_contract_unsupported` and refuses every deploy. A green suite has to mean the client
    accepts the payload, not just that the names appear somewhere in it.
    """
    response = http.get("/healthz")
    assert response.status_code == 200, f"/healthz returned {response.status_code}"
    payload = response.json()
    assert isinstance(payload, dict), "/healthz must return a JSON object"
    capabilities = payload.get("capabilities")
    assert isinstance(capabilities, list), (
        f"/healthz returned capabilities as {type(capabilities).__name__}, not a list; the client "
        f"requires a list of strings and refuses the deploy on anything else"
    )
    non_strings = [item for item in capabilities if not isinstance(item, str)]
    assert not non_strings, (
        f"/healthz capabilities contains non-string entries {non_strings!r}; the client requires "
        f"every element to be a string"
    )
    missing = REQUIRED_CAPABILITIES - set(capabilities)
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


def test_a_constrained_registration_serves_a_constrained_completion(
    http, adapter_source, run_id, ready_timeout
):
    """`structured_outputs` must survive the round trip byte for byte AND reach generation.

    The client compares it exactly (`flash/serve/deploy.py`, alongside the scalar identity fields),
    and it is the one identity-bearing field that is OPTIONAL -- so a backend can accept the
    registration, drop the constraint, and look completely healthy. Every other test here registers
    without one, which means a suite that stops at those would certify a backend that fails every
    constrained deploy: `_wait_revision_ready` reads the constraint back as `None`, calls that a
    different artifact, and refuses. Constrained runs are ordinary, not an edge case.

    Driven all the way through GENERATION, not just read back. Echoing the field costs a backend
    nothing; the failure this catches is a constraint that is accepted and stored but never handed
    to the engine, or one the engine rejects at load time so the revision settles `failed` -- and
    reading the record immediately after an asynchronous registration sees neither.
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
        # Reaching `ready` is itself the assertion for the load-time rejection case: a constraint
        # the engine will not accept settles the revision `failed`, and `_wait_ready` fails on it.
        _wait_ready(http, body["adapter_id"], ready_timeout)
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
        )
        assert response.status_code == 200, (
            f"chat on a constrained revision returned {response.status_code}: {response.text[:400]}"
        )
        content = (response.json().get("choices") or [{}])[0].get("message", {}).get(
            "content"
        ) or ""
        assert re.fullmatch(r"[ab]+", content.strip()), (
            f"the revision registered a structured_outputs regex of [ab]+ but generated "
            f"{content.strip()[:120]!r}; the constraint was accepted and stored but never applied "
            f"at generation, so every constrained deploy produces unconstrained output"
        )
    finally:
        http.delete(f"/adapters/{run_id}")


def test_reregistering_identical_content_is_idempotent(http, deployed):
    """The client retries registration after an ambiguous 5xx; a retry must not fail."""
    _register(http, deployed)


@pytest.mark.parametrize("field", sorted(_MUTATIONS))
def test_reregistering_different_content_under_one_revision_conflicts(http, deployed, field):
    """This IS `immutable_adapter_revisions`: one revision id, exactly one artifact, forever.

    Every identity-bearing field class, not just `subfolder`. The client compares all of these on
    its 5xx-recovery read-back and treats any difference as a different artifact, so a backend that
    pins only the path still lets one revision id resolve to different weights (`repo_id`),
    different artifact kind (`repo_type`), different prompt rendering (`thinking`), different
    tenancy (`org_id`), a different grammar (`structured_outputs`), or different provenance -- and
    an ambiguous registration retry then silently changes what the id names.
    """
    mutated = _MUTATIONS[field](deployed)
    assert mutated != deployed, f"mutation for {field} did not change the registration body"
    response = http.post("/adapters", json=mutated)
    assert response.status_code in (409, 422), (
        f"re-registering {deployed['adapter_id']} with a different {field} returned "
        f"{response.status_code}, expected 409 (or 422 when the field is pinned to the revision "
        f"id); the backend advertises immutable_adapter_revisions but lets a revision id change "
        f"artifact"
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


def test_concurrent_activations_of_one_alias_leave_exactly_one_winner(
    http, serving_url, internal_key, adapter_source, run_id, ready_timeout
):
    """`alias_compare_and_swap` is an ATOMICITY claim, and only concurrency can test it.

    The sequential test above returns 200 then 409 against an unlocked read/check/write too: the
    second call simply observes the first call's result. Two SIMULTANEOUS deploys of one run are
    the case that matters -- both read `None`, both find their expectation satisfied, both write,
    and one silently overwrites the other while reporting success. That is exactly the race the
    capability promises cannot happen.

    Not a proof of atomicity (no finite number of attempts is), but it fails a backend whose
    activation holds no lock often enough to be worth running, and it cannot false-fail a correct
    one: exactly one winner is the contract.
    """
    import concurrent.futures

    import httpx

    first = _registration(run_id, adapter_source, step=10)
    second = _registration(run_id, adapter_source, step=20)
    try:
        for body in (first, second):
            _register(http, body)
            _wait_ready(http, body["adapter_id"], ready_timeout)

        headers = {"X-Freesolo-Internal-Key": internal_key} if internal_key else {}

        def activate(revision: str) -> int:
            with httpx.Client(base_url=serving_url, timeout=60.0, headers=headers) as client:
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
        assert len(winners) == 1, (
            f"two concurrent activations of {run_id} with the same expectation returned {codes}; "
            f"exactly one must win. more than one success means the compare-and-swap is not "
            f"atomic and two deploys can silently overwrite each other"
        )
        losers = [code for code in codes if code != 200]
        assert all(code == 409 for code in losers), (
            f"the losing concurrent activation returned {losers}, expected 409"
        )

        # The winner is what the alias actually points at -- a backend could return one 200 and one
        # 409 while landing the loser's write. Accepting EITHER revision here would pass exactly
        # that backend, so the expected target is derived from which activation got the 200:
        # `pool.map` yields results in input order, so `codes[i]` belongs to `revisions[i]`.
        revisions = (first["adapter_id"], second["adapter_id"])
        expected = revisions[codes.index(200)]
        alias = _record(http.get(f"/adapters/{run_id}").json())
        metadata = alias.get("metadata") if isinstance(alias.get("metadata"), dict) else {}
        assert metadata.get("alias_of") == expected, (
            f"the activation of {expected} returned 200, but after the race the alias points at "
            f"{metadata.get('alias_of')!r}. the backend told one caller it won and then committed "
            f"the other write, so a deploy reports success while a different revision serves"
        )
    finally:
        http.delete(f"/adapters/{run_id}")


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

    # The provenance is the POINT of this test: it is what proves the alias resolved to the
    # revision it names rather than to whatever the engine happened to have loaded. Validating only
    # the content certifies a backend that answers from the wrong weights -- and the deployment
    # smoke (`flash/server/routes/serving_smoke.py`) rejects a missing or mismatched value in both
    # the body and the headers, so every deploy against such a backend fails after generation.
    revision = deployed["adapter_id"]
    expected = {
        "adapter_revision": revision,
        "checkpoint": deployed["checkpoint"],
        "hf_revision": revision.rsplit(".", 1)[-1],
    }
    provenance = payload.get("freesolo")
    assert isinstance(provenance, dict), (
        f"chat response omitted the `freesolo` provenance object (got {provenance!r}); the "
        f"deployment smoke requires it and fails the deploy without it"
    )
    for field, value in expected.items():
        assert provenance.get(field) == value, (
            f"chat provenance {field}={provenance.get(field)!r}, expected {value!r}; the response "
            f"does not prove it came from the revision the alias names"
        )
    for header, value in (
        ("X-Freesolo-Adapter-Revision", expected["adapter_revision"]),
        ("X-Freesolo-Checkpoint", expected["checkpoint"]),
        ("X-Freesolo-HF-Revision", expected["hf_revision"]),
    ):
        assert response.headers.get(header) == value, (
            f"chat response header {header}={response.headers.get(header)!r}, expected {value!r}; "
            f"the deployment smoke compares all three headers and fails the deploy on a mismatch"
        )


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

    # READ BACK, do not trust the response. A DELETE that returns the requested run and a
    # fabricated pair of lists satisfies every assertion above while the alias and revision stay
    # ready and callable -- and the shipped client performs no read-back of its own, so
    # `flash models undeploy` would report success over a run that keeps serving and keeps
    # billing. The authoritative state is what the backend answers on the next read.
    for record_id in (run_id, body["adapter_id"]):
        response = http.get(f"/adapters/{record_id}")
        if response.status_code == 404:
            continue  # deleting the record outright is a valid way to disable it
        assert response.status_code == 200, (
            f"read-back of {record_id} after undeploy returned {response.status_code}"
        )
        record = _record(response.json())
        state = _lifecycle_state(record)
        assert record.get("status") == "disabled" or state in ("disabled", "failed"), (
            f"after undeploy, {record_id} still reads back status={record.get('status')!r} "
            f"lifecycle_state={state!r}; undeploy reported success without disabling it, so the "
            f"run keeps serving and keeps costing money"
        )

    chat = http.post(
        "/v1/chat/completions",
        json={
            "model": run_id,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        },
    )
    assert chat.status_code != 200, (
        "chat still succeeds after undeploy; the alias was reported disabled but is live"
    )


def test_provenance_contradicting_the_revision_id_is_refused(http, adapter_source, run_id):
    """A backend must not store provenance that disagrees with the id it is filed under.

    The revision id already encodes run, step, and commit. `run_id` is the damaging one: it decides
    which run's alias and membership the revision joins, so a backend that trusts a contradicting
    value files the artifact under a DIFFERENT run -- undeploying that run then reports success
    while the revision keeps serving under its own immutable id, which no client call can detect.

    Refused, not merely fingerprinted: the first registration has nothing to compare against, and
    by the second one the alias has already been written to the wrong run.
    """
    body = _registration(run_id, adapter_source)
    body["metadata"] = {**body["metadata"], "run_id": f"{run_id}-elsewhere"}
    response = http.post("/adapters", json=body)
    # Cleaned up whatever the backend decided, because a backend that DID accept it has now created
    # records under a run this test never otherwise touches.
    try:
        assert response.status_code in (400, 409, 422), (
            f"registering metadata.run_id that contradicts the revision id returned "
            f"{response.status_code}; the revision is now filed under a different run than its id "
            f"names, so undeploying that run reports success while this revision keeps serving"
        )
    finally:
        http.delete(f"/adapters/{run_id}")
        http.delete(f"/adapters/{run_id}-elsewhere")


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
