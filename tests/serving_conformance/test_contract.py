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

import inspect
import json
import re
import time
import types
from pathlib import Path

import pytest

from flash.schema import format_adapter_revision

REQUIRED_CAPABILITIES = {"immutable_adapter_revisions", "alias_compare_and_swap"}

# The longest the client will ever wait for ONE request, whatever budget it is given:
# `flash/serve/deploy.py` clamps every call to `min(60.0, timeout_s)`. Mirrored rather than
# imported so the suite stays runnable against a backend without flash's serving module installed --
# and asserted against the real value in `test_the_per_request_cap_still_matches_the_client`, so
# the copy cannot drift into passing backends the client would time out on.
_CLIENT_REQUEST_TIMEOUT_CAP = 60.0

# The readiness backoff the client actually uses: `_readback_delay` starts at
# READBACK_DELAY_SECONDS and doubles, but caps EVERY delay -- including one taken from a
# `Retry-After` header -- at READBACK_MAX_DELAY_SECONDS. Mirrored here for the same reason as the
# timeout cap, and guarded the same way by `test_the_readiness_backoff_still_matches_the_client`:
# a suite that sleeps longer than the client polls fewer times inside one budget, so a revision the
# client would have seen go ready near the end of its window is missed here and a conforming
# backend fails.
_CLIENT_READBACK_BASE_DELAY = 0.5
_CLIENT_READBACK_MAX_DELAY = 2.0

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


def _readback_delay(attempt: int, retry_after: str | None = None) -> float:
    """`flash/serve/deploy.py:_readback_delay`, mirrored.

    Same two rules as the original: a usable `Retry-After` wins over the backoff, and BOTH are
    clamped to the same ceiling. The clamp on the header is the part that is easy to drop and the
    part that matters -- a backend answering `Retry-After: 30` would otherwise get 30s of silence
    here and 2s from the client.
    """
    if retry_after is not None:
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            pass
        else:
            if delay > 0 and delay == delay and delay != float("inf"):
                return min(delay, _CLIENT_READBACK_MAX_DELAY)
    return min(_CLIENT_READBACK_BASE_DELAY * (2**attempt), _CLIENT_READBACK_MAX_DELAY)


def _wait_ready(http, revision: str, timeout: float) -> dict:
    """Poll to a settled state, honoring Retry-After exactly as the client does.

    The deadline bounds each REQUEST and is re-checked after it, not only before. Checked only
    beforehand, a backend that long-polls the read-back is accepted for a `ready` that arrived after
    the budget was gone -- and the shipped client would not accept it: `_wait_revision_ready` passes
    the remaining budget as the per-request timeout and breaks on a read that completes past the
    deadline. A backend with slow status reads would otherwise pass conformance and then fail every
    real deploy, which is the one thing this suite exists to rule out.

    The per-request bound is the remaining budget capped at `_CLIENT_REQUEST_TIMEOUT_CAP`, because
    the budget alone is not what the client allows: `_serving_request` clamps whatever it is handed
    to 60s. Passing the full remaining budget here would accept a backend whose read-back takes 90s
    -- inside the suite's overall budget, but past what every real poll permits.

    A request that times out is RETRIED, not raised, which is the other half of matching the
    client: `_wait_revision_ready` re-raises only a `status_code < 500`, and a transport timeout
    carries no status at all, so it falls through to the retry. Letting httpx's exception escape
    here would fail conformance for a backend `flash models deploy` handles -- and on a
    scale-to-zero app the first read-back is exactly where a cold start makes one poll exceed 60s
    and the next answer immediately. The overall deadline is what ends the wait, not any single
    slow read.

    The BACKOFF mirrors `_readback_delay` rather than inventing its own, and the ceiling is the
    part that matters. The client caps every readiness delay at 2s -- including one it took from a
    `Retry-After` header, which it clamps rather than obeys. Sleeping the header's value here (up
    to 30s, growing to 15s on the fallback) makes this suite poll far fewer times inside the same
    budget, so a revision that goes ready near the end of the window is seen by a real deploy and
    missed here: conformance fails a backend that works. Honoring `Retry-After` at all is still
    right, since it is what the client reads; honoring it UNCAPPED is what was wrong.
    """
    # Imported here, not at module scope, matching the rest of the suite: httpx is only needed on
    # the paths that talk to a backend, and a hard import would break collection wherever the
    # `--serving-url` tests are skipped anyway.
    import httpx

    deadline = time.monotonic() + timeout
    attempt = 0
    delay = _CLIENT_READBACK_BASE_DELAY
    # Reset every pass rather than carried: a 404 read-back sets no header, and the client likewise
    # re-reads it from the response it just got instead of reusing a previous poll's value.
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
        except httpx.TimeoutException:
            # Transient by the client's own rules. Backing off rather than hammering a backend that
            # is plainly still warming up. Same compute-then-sleep order as the tail of the loop; a
            # timed-out request carries no response and so no `Retry-After`.
            attempt += 1
            delay = _readback_delay(attempt)
            time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
            continue
        if deadline - time.monotonic() <= 0:
            # Completed, but not in time. Not honored, for the same reason the client does not.
            break
        if response.status_code >= 500:
            # TRANSIENT, exactly as the client treats it: `_wait_revision_ready` re-raises only a
            # `status_code < 500` and lets everything else fall through to another poll. A backend
            # that answers 503 while its engine container starts is the ordinary scale-to-zero
            # case, and failing the run on the first one would reject a backend every real deploy
            # tolerates. The overall deadline still ends the wait.
            retry_after = response.headers.get("Retry-After")
        elif response.status_code != 404:
            assert response.status_code == 200, (
                f"read-back returned {response.status_code}: {response.text[:400]}. a non-404 4xx "
                f"is a hard failure for the client too -- it re-raises any status below 500"
            )
            record = _record(response.json())
            last = _lifecycle_state(record)
            if last == "failed" or record.get("status") == "disabled":
                metadata = record.get("metadata") or {}
                pytest.fail(f"revision {revision} reported failed: {metadata.get('failure')}")
            if last == "ready":
                return record
            retry_after = response.headers.get("Retry-After")
        # Computed from THIS response, then slept -- not slept-then-computed. The client derives
        # each delay from the response it just received before its next poll, so computing after
        # the sleep applied the PREVIOUS response's delay: a first `Retry-After: 2` was answered
        # with a 0.5s sleep here and a 2s one there. Near the deadline that lets the suite make a
        # poll the real client never would and certify a backend the client cannot drive.
        attempt += 1
        delay = _readback_delay(attempt, retry_after)
        time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
    pytest.fail(
        f"revision {revision} never reached ready within {timeout}s (last state {last!r}). "
        "a read-back that only completes after the budget is spent does not count: the client "
        "bounds each poll by its remaining budget and discards a late response, so a backend that "
        "needs longer than this fails every deploy"
    )


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


def test_the_per_request_cap_still_matches_the_client():
    """The mirrored 60s cap must not drift from the client's real one.

    `_wait_ready` bounds each poll by `_CLIENT_REQUEST_TIMEOUT_CAP` because that is what the client
    genuinely permits per request, not what the overall budget allows. Copied rather than imported
    so the suite runs against a backend without flash's serving module installed -- which is exactly
    the arrangement where a copy rots silently. If `_serving_request` is ever relaxed to 120s, this
    fails and says so, rather than leaving the suite quietly stricter than the client; and if it is
    tightened, the suite would otherwise start passing backends that every deploy times out on.

    Takes no fixtures ON PURPOSE. `serving_url` is what skips this suite without `--serving-url`, so
    a test that requested it would never run in CI -- and a drift guard that only runs when someone
    has a live backend is not a guard.
    """
    from flash.serve import deploy

    source = inspect.getsource(deploy._serving_request)
    found = re.findall(r"min\((\d+(?:\.\d+)?),", source)
    assert found, (
        "could not find the per-request timeout clamp in `_serving_request`; the conformance "
        "suite's mirrored cap can no longer be checked against the client and may be silently wrong"
    )
    assert float(found[0]) == _CLIENT_REQUEST_TIMEOUT_CAP, (
        f"the client now clamps each request to {found[0]}s but this suite still bounds its polls "
        f"by {_CLIENT_REQUEST_TIMEOUT_CAP}s. a suite that allows longer than the client passes "
        f"backends whose read-backs time out on every real deploy"
    )


def test_the_readiness_backoff_still_matches_the_client():
    """The mirrored polling delays must not drift either, and the CEILING is what matters.

    Same reasoning as the per-request cap, and the same failure shape in reverse: a suite that
    sleeps LONGER than the client polls fewer times inside one budget, so a revision the client
    would have caught going ready near the end of its window is missed here and a working backend
    fails conformance. The `Retry-After` clamp is the specific value worth pinning -- the client
    treats that header as an upper hint it may shorten, not as an instruction, and a suite that
    obeys it verbatim hands a slow backend the power to make itself untestable.

    Takes no fixtures, for the same reason as the cap guard: it must run in CI, where no
    `--serving-url` is set and every live test skips.
    """
    from flash.serve import deploy

    # Read out of the SOURCE, not off the module. `tests/conftest.py` installs an autouse fixture
    # that zeroes these very constants so the offline suite does not sleep, so a runtime attribute
    # read here would compare against 0.0 and assert on the test harness rather than on what flash
    # ships. The cap guard reads source for the same reason -- there the clamp is a literal.
    source = Path(deploy.__file__).read_text()
    shipped = {}
    for name in ("READBACK_DELAY_SECONDS", "READBACK_MAX_DELAY_SECONDS"):
        found = re.search(rf"^{name} = ([0-9.]+)$", source, re.MULTILINE)
        assert found, (
            f"could not find `{name}` in flash/serve/deploy.py; this suite's mirrored readiness "
            f"backoff can no longer be checked against the client and may be silently wrong"
        )
        shipped[name] = float(found.group(1))

    assert shipped["READBACK_MAX_DELAY_SECONDS"] == _CLIENT_READBACK_MAX_DELAY, (
        f"the client now caps readiness delays at {shipped['READBACK_MAX_DELAY_SECONDS']}s but "
        f"this suite still uses {_CLIENT_READBACK_MAX_DELAY}s. a suite that waits longer than the "
        f"client polls fewer times in the same budget and fails backends the client accepts"
    )
    assert shipped["READBACK_DELAY_SECONDS"] == _CLIENT_READBACK_BASE_DELAY, (
        f"the client's readiness backoff now starts at {shipped['READBACK_DELAY_SECONDS']}s but "
        f"this suite still starts at {_CLIENT_READBACK_BASE_DELAY}s"
    )
    # The mirrored helper must also AGREE with the real one, not merely share its constants: the
    # clamp on `Retry-After` lives in the code, not in a constant, and dropping it would leave both
    # assertions above passing. Compared against a copy of `_readback_delay` rebound to the SHIPPED
    # constants, since the live one is running under the zeroing fixture too.
    client_delay = types.FunctionType(
        deploy._readback_delay.__code__,
        {**deploy._readback_delay.__globals__, **shipped},
        deploy._readback_delay.__name__,
        deploy._readback_delay.__defaults__,
    )
    for attempt, retry_after in ((0, None), (3, None), (9, None), (0, "30"), (0, "0.5"), (1, "x")):
        assert _readback_delay(attempt, retry_after) == client_delay(attempt, retry_after), (
            f"the mirrored readiness backoff disagrees with the client for "
            f"attempt={attempt} retry_after={retry_after!r}"
        )


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
        # Asserted on the record `_wait_ready` returns, NOT on an immediate read. Registration may
        # answer 202 before the record is visible, and the client accommodates that -- it polls,
        # treating 404 as "not registered yet". Reading straight after the POST demands
        # read-after-write visibility the contract never required, so an asynchronous backend that
        # `flash models deploy` drives fine would fail here on a legitimate 404.
        #
        # Reaching `ready` is itself the assertion for the load-time rejection case: a constraint
        # the engine will not accept settles the revision `failed`, and `_wait_ready` fails on it.
        record = _wait_ready(http, body["adapter_id"], ready_timeout)
        assert record.get("structured_outputs") == body["structured_outputs"], (
            f"read-back structured_outputs={record.get('structured_outputs')!r}, registered "
            f"{body['structured_outputs']!r}. The client compares this exactly, so every "
            f"constrained deploy against this backend will be refused as a different artifact."
        )
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


def test_reregistering_while_the_first_load_is_pending_is_idempotent(
    http, adapter_source, run_id, ready_timeout
):
    """Idempotence is required in the PENDING state, which is where the retry actually happens.

    Asserting it only after `ready` tests the easy half. The case that matters is the recovery
    path: a cold load exceeds the client's five-minute readiness budget, the operator reruns
    `models deploy`, and the identical registration arrives while the first load is still
    `registered`. A backend that accepts duplicates only once loading has finished answers 409
    there and strands exactly the retry the client is built to make.

    So the second POST goes in immediately after the first is accepted, with no wait between them.
    """
    body = _registration(run_id, adapter_source)
    try:
        _register(http, body)
        # No `_wait_ready` in between -- the pending window IS the state under test.
        _register(http, body)
        # And the run must still converge afterwards: accepting the duplicate is not enough if it
        # wedged the first load or left two settles fighting over one record.
        _wait_ready(http, body["adapter_id"], ready_timeout)
    finally:
        http.delete(f"/adapters/{run_id}")


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


def test_a_non_null_expectation_replaces_the_live_revision(
    http, adapter_source, run_id, ready_timeout
):
    """The upgrade path: activating step-20 while naming step-10 as the expectation.

    Every activation test above passes `expected_adapter_revision: None`, which is only what the
    FIRST deploy of a run sends. On every subsequent deploy the managed plane computes the alias's
    live revision (`flash/server/routes/serving.py:_activation_predecessor`) and sends THAT, and
    the client then verifies the response echoes it back as `previous_adapter_revision`. So a
    backend that implements only the null expectation -- treating any non-null value as a mismatch,
    or ignoring it and answering `null` -- passed this entire suite and then failed every
    checkpoint upgrade its users attempted. That is precisely the class of gap conformance exists
    to close, and no null-expectation test can reach it.

    Asserts the two things the client genuinely acts on: the 200 with the correct
    `previous_adapter_revision` (mismatched, it raises `mismatched previous alias revision`), and
    the alias record actually moving, since a backend could echo the right provenance while
    leaving the alias where it was.
    """
    first = _registration(run_id, adapter_source, step=10)
    second = _registration(run_id, adapter_source, step=20)
    for body in (first, second):
        _register(http, body)
        _wait_ready(http, body["adapter_id"], ready_timeout)

    assert _activate(http, first["adapter_id"], None).status_code == 200, (
        "the initial null-expectation activation must succeed before the upgrade can be tested"
    )

    response = _activate(http, second["adapter_id"], first["adapter_id"])
    assert response.status_code == 200, (
        f"activating {second['adapter_id']} with {first['adapter_id']} as the expectation returned "
        f"{response.status_code}: {response.text[:400]}. this is the ordinary upgrade path -- "
        f"every deploy after a run's first one names the live revision as its expectation, so a "
        f"backend that rejects it cannot serve a second checkpoint"
    )
    payload = response.json()
    assert payload.get("previous_adapter_revision") == first["adapter_id"], (
        f"activation echoed previous_adapter_revision="
        f"{payload.get('previous_adapter_revision')!r}, expected {first['adapter_id']!r}; the "
        f"client fails the deploy on exactly this comparison"
    )
    assert payload.get("target_adapter_revision") == second["adapter_id"], (
        f"activation echoed target_adapter_revision={payload.get('target_adapter_revision')!r}, "
        f"expected {second['adapter_id']!r}"
    )

    alias = _record(http.get(f"/adapters/{run_id}").json())
    metadata = alias.get("metadata") if isinstance(alias.get("metadata"), dict) else {}
    assert metadata.get("alias_of") == second["adapter_id"], (
        f"the activation reported success but the alias still names "
        f"{metadata.get('alias_of')!r}; a backend that returns the right provenance without "
        f"moving the alias serves the OLD checkpoint to every request after an upgrade"
    )


def test_concurrent_activations_of_one_alias_leave_exactly_one_winner(
    http, serving_client_factory, adapter_source, run_id, ready_timeout
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

    first = _registration(run_id, adapter_source, step=10)
    second = _registration(run_id, adapter_source, step=20)
    try:
        for body in (first, second):
            _register(http, body)
            _wait_ready(http, body["adapter_id"], ready_timeout)

        def activate(revision: str) -> int:
            # One client PER THREAD, but built by the same factory the `http` fixture uses. A
            # thread needs its own so the test measures the backend's locking rather than httpx's
            # connection pool; it needs the factory's shape so a Modal 303 is followed and the
            # internal key is stripped across an off-origin redirect, exactly as a real deploy
            # does.
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
