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

# The run the wrong-key probe posts under. FIXED rather than per-run unique, so that a run killed
# between the probe and its cleanup leaves a single known name a later run reclaims, instead of one
# more orphan per attempt.
_UNAUTHORIZED_RUN_ID = "conformance-unauthorized"

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


def _requires_provenance(http) -> bool:
    """Whether this backend advertises `revision_provenance`, asked once and cached.

    The client gates its provenance cross-check on exactly this capability
    (`_require_serving_capabilities` treats it as PREFERRED, not required), so a mirror that
    compares provenance unconditionally would fail a backend the client drives happily. Cached on
    the client object because `_wait_ready` polls in a loop and this cannot become a per-poll
    request.
    """
    cached = getattr(http, "_conformance_requires_provenance", None)
    if cached is None:
        payload = http.get("/healthz").json()
        capabilities = payload.get("capabilities") if isinstance(payload, dict) else None
        cached = "revision_provenance" in set(capabilities or [])
        http._conformance_requires_provenance = cached
    return cached


def _identity_mismatch(record: dict, expected: dict, *, require_provenance: bool) -> str | None:
    """`flash/serve/deploy.py:_matches_revision_identity`, mirrored, naming the field that differs.

    Returns None when the record matches. The client compares these fields on EVERY readiness poll,
    not once at the end, and raises immediately on the first mismatch -- so a backend that exposes a
    partial `registered` record and fills the identity in before `ready` fails every real deploy
    while a suite that only inspects the final record certifies it.

    Compared with a plain `!=`, including `thinking`, because that is what the client uses. Coercing
    with `bool()` here would be LOOSER than the client: a backend that omits the field answers None,
    which the client calls a different identity and this would have let pass.
    """
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
    # Normalized on both sides, exactly as the client does: it compares `(x or None)`, so a backend
    # may answer "" or omit the field for a request that did not set one.
    for field in ("org_id", "structured_outputs"):
        if (record.get(field) or None) != (expected.get(field) or None):
            return f"{field}: {record.get(field)!r} != {expected.get(field)!r}"
    if not require_provenance:
        # Matches the client's own `require_provenance=False` early return: a backend that does not
        # advertise `revision_provenance` is not expected to echo this metadata, and the immutable
        # adapter_id already pins the artifact.
        return None
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    expected_metadata = expected.get("metadata") or {}
    for field in ("record_type", "run_id", "checkpoint_step", "hf_revision"):
        if metadata.get(field) != expected_metadata.get(field):
            return f"metadata.{field}: {metadata.get(field)!r} != {expected_metadata.get(field)!r}"
    return None


def _wait_ready(http, revision: str, timeout: float, expected: dict | None = None) -> dict:
    """Poll to a settled state, honoring Retry-After exactly as the client does.

    `expected` is the registration body, and passing it applies the client's identity check to
    every visible record rather than only the final one. `_wait_revision_ready` calls
    `_matches_revision_identity` on each non-404 read BEFORE inspecting its lifecycle and raises on
    the first mismatch, so an asynchronous backend that first exposes a record missing `org_id` or
    carrying the wrong provenance, then corrects it on the way to `ready`, fails every real deploy.
    Checked only at the end, the suite sees the corrected record and passes it. Optional because
    two call sites drive stub objects rather than a registration.

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
        except httpx.RequestError:
            # Transient by the client's own rules. Backing off rather than hammering a backend that
            # is plainly still warming up. Same compute-then-sleep order as the tail of the loop; a
            # failed request carries no response and so no `Retry-After`.
            #
            # The FULL transport-error class, not just timeouts. `_serving_request` converts every
            # `httpx.RequestError` into a `ServingError` carrying no status code, and
            # `_wait_revision_ready` re-raises only `status_code < 500`, so the client retries all
            # of them until its deadline. Catching timeouts alone let a connection reset or a
            # dropped keep-alive -- the ordinary shape of a scale-to-zero backend bringing a
            # container up -- fail conformance on the first occurrence, rejecting a backend every
            # real deploy tolerates. `TimeoutException` is a subclass of `RequestError`, so this
            # strictly widens what was already handled; the deadline still bounds the wait.
            delay = _readback_delay(attempt)
            attempt += 1
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
            # Decoded defensively, because the CLIENT treats both failures as transient. A 200
            # carrying truncated JSON or a non-object payload makes
            # `_registered_adapter_response` raise a `ServingError` with NO status code, and
            # `_wait_revision_ready` re-raises only `status_code < 500` -- so it falls through and
            # polls again until the deadline. Asserting on the decode here failed the whole run on
            # one malformed readback and rejected a backend the real deploy drives successfully
            # once its next read is valid.
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if not isinstance(payload, dict):
                # Backed off, not spun on. Falling straight to `continue` would skip the sleep at
                # the tail of the loop and hammer a backend answering malformed 200s as fast as the
                # network allows for the whole readiness budget. This mirrors the transport-error
                # branch above: compute from THIS response, sleep, then poll again.
                delay = _readback_delay(attempt, response.headers.get("Retry-After"))
                attempt += 1
                time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
                continue
            record = _record(payload)
            # Identity BEFORE lifecycle, matching the client's order. It raises on a mismatch
            # without ever looking at the state, so a record that is both wrong and `ready` is an
            # identity failure to the client and must be one here.
            if expected is not None:
                mismatch = _identity_mismatch(
                    record, expected, require_provenance=_requires_provenance(http)
                )
                assert mismatch is None, (
                    f"read-back of {revision} exposed a record with a different immutable identity "
                    f"({mismatch}). the client checks this on EVERY poll and raises at once, so a "
                    f"record that is corrected before it reaches ready still fails every deploy"
                )
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
        #
        # And computed with the CURRENT `attempt`, incrementing after -- not with the incremented
        # one. The client counts reads and backs off on `attempt - 1`, so its first no-header sleep
        # is 0.5s and its second 1s; computing on the post-increment value here started at 1s and
        # stayed one step ahead forever. That is the same defect as the ordering above, in the same
        # direction: fewer polls inside one budget than the client makes, so a revision that goes
        # ready late is seen by a real deploy and missed here. `Retry-After` short-circuits before
        # the exponent, so only the no-header path drifted.
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
    """Register one revision, wait for ready, and always undeploy the run afterwards."""
    body = _registration(run_id, adapter_source)
    try:
        _register(http, body)
        _wait_ready(http, body["adapter_id"], ready_timeout, expected=body)
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


def test_the_readiness_backoff_is_driven_through_the_same_attempt_sequence(monkeypatch):
    """Sharing the helper is not enough; the two loops must FEED it the same attempts.

    The guard above compares `_readback_delay` against the client's copy for a fixed set of
    arguments, and that is blind to the defect this test exists for: both loops can hold identical
    constants and an identical clamp while passing different `attempt` values, and the mirrored
    backoff is then wrong at every poll. Which is what happened -- the client counts reads and
    backs off on `attempt - 1`, this suite briefly computed on the post-increment value, and its
    first no-header sleep was 1s against the client's 0.5s and stayed one step ahead.

    Recorded as ARGUMENTS rather than as delays, deliberately. `tests/conftest.py` zeroes the
    backoff constants so the offline suite does not sleep, so comparing sleep durations would
    compare 0.0 with 0.0 and pass for any sequence at all.

    Takes no live fixtures, so it runs in CI where every `--serving-url` test skips.
    """
    from flash.serve import deploy

    class _Response:
        status_code = 404
        headers: ClassVar[dict[str, str]] = {}
        text = ""

    class _Stop(Exception):
        """Ends a loop that would otherwise poll until its budget expires."""

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

    assert suite_attempts == client_attempts, (
        f"the suite backs off through attempts {suite_attempts} where the client uses "
        f"{client_attempts}. the delays are computed by the same helper, so a different sequence "
        f"means a different backoff at every poll -- and a suite that sleeps longer polls fewer "
        f"times inside one budget, missing a revision the client would have seen go ready"
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
    http, adapter_source, run_id, ready_timeout, chat_timeout
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
        record = _wait_ready(http, body["adapter_id"], ready_timeout, expected=body)
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
            timeout=chat_timeout,
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
    path: a cold load exceeds the client's model-scaled readiness budget, the operator reruns
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
        _wait_ready(http, body["adapter_id"], ready_timeout, expected=body)
    finally:
        http.delete(f"/adapters/{run_id}")


def test_a_final_revision_registers_and_reaches_ready(http, adapter_source, run_id, ready_timeout):
    """The ordinary finished-run shape, which every other test here skips.

    `deploy_adapter` emits `<run>@final.<sha>` with `checkpoint_step: null` whenever no checkpoint
    step is selected, which is what a completed run deploys as -- yet every registration in this
    suite passes an integer step, so a backend that parses only `@step-N` passes conformance and
    then fails every normal deploy. The null step is the specific part worth exercising: a backend
    that coerces it, or that requires the field to be an int, rejects the id its own grammar
    accepts.
    """
    body = _registration(run_id, adapter_source, step=None)
    assert body["adapter_id"].endswith(f"@final.{adapter_source['hf_revision']}"), (
        "the helper stopped producing the final-revision shape, so this test no longer covers it"
    )
    try:
        _register(http, body)
        record = _wait_ready(http, body["adapter_id"], ready_timeout, expected=body)
        assert (record.get("metadata") or {}).get("checkpoint_step") is None, (
            "a final revision came back with a checkpoint step; the client compares provenance "
            "exactly on its recovery readback and would call this a different immutable identity"
        )
    finally:
        http.delete(f"/adapters/{run_id}")


def test_a_thinking_adapter_registers_and_reaches_ready(
    http, adapter_source, run_id, ready_timeout
):
    """`thinking: true` must be storable, not merely rejectable.

    Every successful registration in this suite hardcodes `thinking: false`, and the immutability
    matrix only proves that flipping it CONFLICTS. A backend that stores the field unconditionally
    as false satisfies both: the identity readback matches, and the mutation still conflicts. The
    control plane sends `thinking: true` for runs trained in thinking mode, where that backend
    either fails the client's identity check or renders prompts in the wrong mode -- a silent
    quality failure rather than an error.
    """
    body = {**_registration(run_id, adapter_source), "thinking": True}
    try:
        _register(http, body)
        record = _wait_ready(http, body["adapter_id"], ready_timeout, expected=body)
        assert record.get("thinking") is True, (
            f"a thinking adapter read back as {record.get('thinking')!r}; the client compares this "
            f"field exactly, so the deploy fails, and a backend that ignores it renders prompts in "
            f"the wrong mode"
        )
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
    # Undeployed on every path, like the other multi-revision tests here. This one registers two
    # revisions and leaves the alias ACTIVE, so without the cleanup each suite run against a live
    # backend leaves a serving run behind -- holding its cached artifacts, its `max_loras` slots,
    # and (on a scale-to-zero deployment) a reason to keep answering traffic. The suite is meant to
    # be safe to re-run against a real deployment, which it is not if it accumulates runs.
    try:
        for body in (first, second):
            _register(http, body)
            _wait_ready(http, body["adapter_id"], ready_timeout, expected=body)

        assert _activate(http, first["adapter_id"], None).status_code == 200, (
            "the initial null-expectation activation must succeed before the upgrade can be tested"
        )

        response = _activate(http, second["adapter_id"], first["adapter_id"])
        assert response.status_code == 200, (
            f"activating {second['adapter_id']} with {first['adapter_id']} as the expectation "
            f"returned {response.status_code}: {response.text[:400]}. this is the ordinary upgrade "
            f"path -- every deploy after a run's first one names the live revision as its "
            f"expectation, so a backend that rejects it cannot serve a second checkpoint"
        )
        payload = response.json()
        assert payload.get("previous_adapter_revision") == first["adapter_id"], (
            f"activation echoed previous_adapter_revision="
            f"{payload.get('previous_adapter_revision')!r}, expected {first['adapter_id']!r}; the "
            f"client fails the deploy on exactly this comparison"
        )
        assert payload.get("target_adapter_revision") == second["adapter_id"], (
            f"activation echoed target_adapter_revision="
            f"{payload.get('target_adapter_revision')!r}, expected {second['adapter_id']!r}"
        )

        alias = _record(http.get(f"/adapters/{run_id}").json())
        metadata = alias.get("metadata") if isinstance(alias.get("metadata"), dict) else {}
        assert metadata.get("alias_of") == second["adapter_id"], (
            f"the activation reported success but the alias still names "
            f"{metadata.get('alias_of')!r}; a backend that returns the right provenance without "
            f"moving the alias serves the OLD checkpoint to every request after an upgrade"
        )
    finally:
        http.delete(f"/adapters/{run_id}")


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
            _wait_ready(http, body["adapter_id"], ready_timeout, expected=body)

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


def test_chat_resolves_the_alias_to_its_immutable_revision(http, deployed, chat_timeout):
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
        timeout=chat_timeout,
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
    try:
        _register(http, body)
        _wait_ready(http, body["adapter_id"], ready_timeout, expected=body)
        assert _activate(http, body["adapter_id"], None).status_code == 200
    except BaseException:
        # The registration LANDED and something after it failed -- a readiness timeout, a
        # nonconforming activation. Those failures are exactly what this suite exists to diagnose
        # on a live backend, so they are the likely outcome, not the rare one; without this the
        # run's revision stays registered, its artifact stays cached, and its LoRA can stay
        # resident on somebody's GPU after a failed conformance run.
        #
        # Every other test in this file has a `finally` that deletes its run, and this one is the
        # exception only because the DELETE below is its subject. So the cleanup is on the failure
        # path alone: on success the assertions own the teardown and must observe it themselves.
        with contextlib.suppress(Exception):
            http.delete(f"/adapters/{run_id}")
        raise

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


def test_chat_streams_deltas_the_way_the_cli_asks_for_them(http, deployed, chat_timeout):
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
    with http.stream("POST", "/v1/chat/completions", json=body, timeout=chat_timeout) as response:
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


def test_a_wrong_serving_key_is_rejected(serving_client_factory, internal_key, adapter_source):
    """A backend that ignores the key passes every other test while being publicly writable.

    The rest of the suite only ever sends the CORRECT key, so it cannot tell "checks the key" from
    "ignores the header". This sends a deliberately wrong one at a mutating endpoint: registration
    is what an unauthenticated caller would abuse to load arbitrary adapters onto the GPU.

    EVERY protected route, not registration alone. Authentication is per-endpoint, so a backend
    that guards its writes and leaves the rest open passes a registration-only probe while exposing
    adapter metadata, letting an anonymous caller switch or disable a run's alias, and serving
    public GPU-backed generation. Each of those is reachable independently, so each is asked
    directly rather than inferred from a global policy the backend never actually has.

    A 404 counts as unauthenticated exposure here, not as a pass: it means the handler ran and
    reported on a resource before any key was checked.

    Skipped when no key is configured, because a backend with no key set is legitimately open (a
    laptop, a private network) and the contract does not require one.
    """
    if not internal_key:
        pytest.skip("no FREESOLO_INTERNAL_KEY configured; the backend is intentionally open")

    revision = f"{_UNAUTHORIZED_RUN_ID}@final." + "0" * 40
    # Bodies are deliberately well-formed: a 422 would prove nothing about authentication, since
    # rejecting a malformed payload does not tell an anonymous caller apart from an authorized one.
    #
    # The register probe therefore sends the FULL registration shape, not a bare `adapter_id`. It
    # used to send only the id while the comment above claimed otherwise: a backend whose schema
    # validation runs before its auth dependency answers 422 to that, which is neither 401 nor 403,
    # so the suite reported the registration route as unauthenticated on a backend that rejects
    # every wrongly-keyed valid request. `_registration` builds the same body `deploy_adapter`
    # sends; only the run id differs, and nothing is ever created because the key is wrong.
    # Built to be internally CONSISTENT, not merely well-formed. Overriding `adapter_id` after the
    # helper returned left `checkpoint` and `metadata` describing `@step-10.<real sha>` while the id
    # said `@final.<zero sha>` -- and a backend that checks the id against its own provenance before
    # authenticating answers 422 to that, which is the same false pass the malformed-body fix
    # removed, reintroduced one line later. The whole registration is therefore built from the
    # final-revision provenance, so the ONLY thing wrong with this request is the key.
    unauthorized_body = _registration(
        _UNAUTHORIZED_RUN_ID, {**adapter_source, "hf_revision": "0" * 40}, step=None
    )
    assert unauthorized_body["adapter_id"] == revision, (
        "the unauthorized probe's body no longer describes the revision it posts to; a backend that "
        "validates identity before authenticating would answer 422 and this test would report an "
        "authenticated route as unprotected"
    )
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
        # Deletion last: an authenticated backend refuses it, and an unauthenticated one has
        # already failed this test before it could delete anything real.
        ("delete", "DELETE", f"/adapters/{revision}", None),
    )
    unprotected = []
    # Built through the shared factory rather than a bare `httpx.Client`, for two reasons the
    # standalone client got wrong. It did not follow redirects, so a backend that answers a
    # protected route with Modal's 303 async-result redirect recorded 303 here -- neither 401 nor
    # 403 -- and a conforming backend was reported as unprotected. And it carried no
    # `_strip_key_off_origin` hook, so a redirect to another origin would have forwarded the
    # `internal_key + "-wrong"` header off-origin, from which the real key is trivially recovered.
    # Overriding just the header on the shipped shape keeps both properties.
    try:
        with serving_client_factory() as client:
            client.headers["X-Freesolo-Internal-Key"] = internal_key + "-wrong"
            for name, method, path, body in probes:
                response = client.request(method, path, json=body)
                if response.status_code not in (401, 403):
                    unprotected.append(f"{name} ({method} {path}) -> {response.status_code}")
    finally:
        # Undo whatever the probes managed to create, using the CORRECT key.
        #
        # A backend that fails this test is by definition one that accepted at least one of these
        # calls, and the first probe is a full valid registration -- so a failing run downloads an
        # adapter onto the operator's volume, writes a durable revision record, and creates a run
        # alias, all under a name the suite never cleans up. That is a real mutation of somebody's
        # backend performed by a read-only-looking conformance check, and it persists across runs.
        #
        # Deleted by RUN id, not by revision: that is the id flash's own `undeploy_adapter` uses,
        # and it sweeps the alias and every member revision in one call, whereas deleting the
        # revision alone would leave the alias the registration created behind.
        #
        # Failures here are swallowed on purpose. The interesting result is the assertion below,
        # and a 404 (nothing was created, which is the conforming case) or an unreachable backend
        # must not replace a precise "these routes are unprotected" report with a cleanup error.
        with contextlib.suppress(Exception), serving_client_factory() as cleanup:
            cleanup.delete(f"/adapters/{_UNAUTHORIZED_RUN_ID}")

    assert not unprotected, (
        f"these routes answered a WRONG serving key without rejecting it: {'; '.join(unprotected)}. "
        f"each one is independently reachable, so an anonymous caller can use exactly the ones "
        f"listed -- read adapter metadata, switch or disable a run's alias, or run generation on "
        f"your GPU -- while the backend still advertises key protection"
    )


def test_a_malformed_readback_is_polled_through_rather_than_failed(monkeypatch):
    """A 200 carrying junk must be treated as transient, exactly as the shipped client treats it.

    `_registered_adapter_response` raises a `ServingError` with NO status code for invalid JSON and
    for a non-object payload, and `_wait_revision_ready` re-raises only `status_code < 500` -- so
    both fall through and poll again until the deadline. The suite asserted on the decode instead,
    so one malformed readback failed the whole conformance run and rejected a backend that
    `flash models deploy` drives successfully once its next read is valid.

    Takes no live fixtures, so it runs in CI where every `--serving-url` test skips.
    """

    class _Response:
        def __init__(self, payload, *, raises=False):
            self.status_code = 200
            self.headers: dict[str, str] = {}
            self.text = "<junk>"
            self._payload = payload
            self._raises = raises

        def json(self):
            if self._raises:
                raise ValueError("Expecting value: line 1 column 1 (char 0)")
            return self._payload

    ready = {
        "adapter_id": "rev",
        "status": "ready",
        "metadata": {"lifecycle_state": "ready"},
    }
    # Truncated body, then a non-object body, then the real record. A suite that fails on either of
    # the first two never sees the third.
    responses = [
        _Response(None, raises=True),
        _Response(["not", "an", "object"]),
        _Response(ready),
    ]

    def _get(*args, **kwargs):
        return responses.pop(0)

    record = _wait_ready(types.SimpleNamespace(get=_get), "rev", timeout=30.0)

    assert record.get("status") == "ready", (
        "the suite gave up on a malformed read-back instead of polling through it, so a backend "
        "that answers one truncated 200 mid-deploy fails conformance while the shipped client "
        "retries and succeeds"
    )
    assert not responses, "the loop stopped early and never reached the valid record"


def test_a_corrected_identity_still_fails_the_readiness_wait():
    """A record whose identity is wrong on an EARLY poll must fail, even if it is fixed by `ready`.

    `_wait_revision_ready` calls `_matches_revision_identity` on every non-404 record BEFORE looking
    at its lifecycle, and raises at once on a mismatch. So an asynchronous backend that first
    exposes a partial `registered` record -- `org_id` missing, or the wrong provenance -- and fills
    it in on the way to `ready` fails every real deploy. Inspecting only the final record certified
    exactly that backend.

    Takes no live fixtures, so it runs in CI where every `--serving-url` test skips.
    """

    class _Response:
        def __init__(self, payload):
            self.status_code = 200
            self.headers: dict[str, str] = {}
            self.text = "{}"
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
    # The org is dropped on the first read and restored on the second, which is the exact shape the
    # client rejects: it compares `(record.get("org_id") or None)` and raises without ever reaching
    # the lifecycle field.
    partial = {**expected, "org_id": None, "status": "registered"}
    partial["metadata"] = {**expected["metadata"], "lifecycle_state": "registered"}
    corrected = {**expected, "status": "ready"}
    corrected["metadata"] = {**expected["metadata"], "lifecycle_state": "ready"}
    responses = [_Response(partial), _Response(corrected)]

    def _get(*args, **kwargs):
        return responses.pop(0)

    # `revision_provenance` advertised, so the mirror applies the same provenance cross-check the
    # client does against this backend.
    http = types.SimpleNamespace(get=_get, _conformance_requires_provenance=True)
    with pytest.raises(AssertionError, match="different immutable identity"):
        _wait_ready(http, "rev", timeout=30.0, expected=expected)
    assert responses, (
        "the wait polled past the mismatched record instead of failing on it, so a backend that "
        "corrects a wrong identity before it reaches ready passes conformance and then fails "
        "every real deploy"
    )
