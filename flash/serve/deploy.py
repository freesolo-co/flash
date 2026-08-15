"""Thin client for the freesolo multi-LoRA serving app (Modal); no flash-side vLLM."""

from __future__ import annotations

import atexit
import contextlib
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from urllib.parse import quote

import httpx

from flash._internal.channel import CHANNEL
from flash._internal.logging import get_logger
from flash.content.structured_outputs import parse_structured_outputs
from flash.envs.loader import is_commit_sha
from flash.schema import format_adapter_revision
from flash.serve.errors import (  # noqa: F401 -- re-exported: callers import these from here
    ActivationOutcomeUnknown,
    AdapterConfigMissing,
    AdapterTensorMissing,
    RetryableServingUnavailable,
    ServingError,
)
from flash.serve.urls import is_freesolo_hosted_url, openai_base_url, serving_control_url

logger = get_logger(__name__)

PROD_FREESOLO_SERVING_URL = "https://serve.freesolo.co"
DEV_FREESOLO_SERVING_URL = "https://serve-dev.freesolo.co"


def default_serving_url(channel: str = CHANNEL) -> str:
    """Default serving control root for the given release channel.

    Serving and control planes use separate per-channel databases; mixing them causes org FK 23503.
    Dev serving is ``serve-dev.freesolo.co``.
    """
    return DEV_FREESOLO_SERVING_URL if channel == "dev" else PROD_FREESOLO_SERVING_URL


DEFAULT_FREESOLO_SERVING_URL = default_serving_url()
READBACK_DELAY_SECONDS = 0.5
READBACK_MAX_DELAY_SECONDS = 2.0
# how long to wait for serving to report a newly registered revision ready. the wait covers a COLD
# engine: serving pulls the base model, starts the engine, then loads the adapter, and none of that
# is proportional to the adapter, which is megabytes. so the budget scales with the BASE model.
#
# the floor is NOT the old 5 minutes: a 4B deploy was observed timing out on a cold engine and then
# succeeding in ~4.6 minutes against the now-warm one, so 5 minutes did not even cover the warm case
# with margin. the floor is doubled to 10 and the per-B term covers a bigger base on top.
#
# the cap is the real constraint, and readiness is only one leg of the attempt. the same deploy also
# spends time BEFORE this wait (resolving the hub revision, downloading the adapter config to read
# its rank, the capability check, registration) and AFTER it (`_SMOKE_BUDGET_SECONDS` = 600s of
# smoke, then activation). the whole attempt must finish inside BOTH `_DEPLOYMENT_STALE_SECONDS`
# (1800s, when the plane declares an in-flight attempt abandoned) and the CLI's 1800s default
# `--wait`, or a deploy that is still progressing is reaped or reported as failed.
#
# so the cap leaves real slack rather than just clearing smoke: 900 + 600 = 1500, keeping 300s for
# the surrounding hub reads, registration, activation, and poll latency, none of which share a
# wall-clock bound with this one.
#
# a longer budget costs little: an adapter serving REJECTS raises as soon as the revision reports
# `failed`, so only a revision that is genuinely still loading waits out the clock.
REVISION_READY_MIN_BUDGET_SECONDS = 10 * 60.0
REVISION_READY_MAX_BUDGET_SECONDS = 15 * 60.0
REVISION_READY_SECONDS_PER_PARAM_B = 20.0
ACTIVATION_READBACK_ATTEMPTS = 3
ACTIVATION_READBACK_DELAY_SECONDS = 2.0
# smoke-retry fallback when a 503 carries no usable Retry-After: keep the prior 2s default rather
# than the 0.5s readiness backoff base, so cold-start smoke retries don't hammer serving.
SMOKE_RETRY_FALLBACK_DELAY_SECONDS = 2.0
THINKING_STRUCTURED_OUTPUTS_CAPABILITY = "thinking_structured_outputs_deferred_v1"
REVISION_PROVENANCE_CAPABILITY = "revision_provenance"
_RETRYABLE_SMOKE_503_CODES = frozenset({"adapter_loading", "engine_unavailable"})
_INTERNAL_KEY_HEADER = "X-Freesolo-Internal-Key"
# modal 303-redirects a slow request to an async-result poll url on the same origin, once per poll
# cycle until the result is ready, so the cap must cover a full 30-minute chat window of polls
# (~150s apart). it guards against redirect loops, not credential leakage: the request hook strips
# the internal key on every off-origin hop regardless of chain length.
_MAX_REDIRECTS = 100
_HTTP_CLIENT: httpx.Client | None = None
_CHAT_HTTP_CLIENT: httpx.Client | None = None
_STREAM_HTTP_CLIENT: httpx.Client | None = None
_HTTP_CLIENT_LOCK = threading.Lock()


def _url_origin(url: httpx.URL) -> tuple[str, str, int | None]:
    # scheme + host + port identify an origin; httpx normalizes default ports to None.
    return (url.scheme.lower(), (url.host or "").rstrip(".").lower(), url.port)


def _configured_serving_origin() -> tuple[str, str, int | None] | None:
    """Origin of the configured serving backend, or None when it cannot be parsed.

    Mirrors ``serving_base_url()``'s url resolution without its standalone guard: this runs
    inside an httpx event hook, so it must never raise, and any request carrying the key was
    already built from a ``serving_base_url()`` that vetted the configuration.
    """
    configured = (os.environ.get("FREESOLO_SERVING_URL") or "").strip()
    base = serving_control_url(configured or DEFAULT_FREESOLO_SERVING_URL)
    try:
        url = httpx.URL(base)
    except Exception:
        return None
    if not url.host:
        return None
    return _url_origin(url)


def _strip_internal_key_off_origin(request: httpx.Request) -> None:
    """Drop the plane credential from any request that leaves the serving origin.

    httpx strips only ``Authorization`` and ``Cookie`` when a redirect changes origin; the
    internal key rides a custom header, so without this hook a single 302 from the serving host
    would forward the credential that controls this plane to an arbitrary origin. The hook runs
    once per redirect hop, so same-origin redirects (modal's async-result polls) keep the key.
    """
    if _INTERNAL_KEY_HEADER not in request.headers:
        return
    origin = _configured_serving_origin()
    if origin is None or _url_origin(request.url) != origin:
        del request.headers[_INTERNAL_KEY_HEADER]


def _new_serving_client(**kwargs) -> httpx.Client:
    """An httpx client for the serving backend: bounded redirects, credential-scoping hook."""
    return httpx.Client(
        follow_redirects=True,
        max_redirects=_MAX_REDIRECTS,
        event_hooks={"request": [_strip_internal_key_off_origin]},
        **kwargs,
    )


def _http_client() -> httpx.Client:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        with _HTTP_CLIENT_LOCK:
            if _HTTP_CLIENT is None:
                _HTTP_CLIENT = _new_serving_client()
    return _HTTP_CLIENT


def _chat_http_client() -> httpx.Client:
    global _CHAT_HTTP_CLIENT
    if _CHAT_HTTP_CLIENT is None:
        with _HTTP_CLIENT_LOCK:
            if _CHAT_HTTP_CLIENT is None:
                _CHAT_HTTP_CLIENT = _new_serving_client(
                    limits=httpx.Limits(max_connections=None, max_keepalive_connections=100),
                )
    return _CHAT_HTTP_CLIENT


def _stream_http_client() -> httpx.Client:
    global _STREAM_HTTP_CLIENT
    if _STREAM_HTTP_CLIENT is None:
        with _HTTP_CLIENT_LOCK:
            if _STREAM_HTTP_CLIENT is None:
                _STREAM_HTTP_CLIENT = _new_serving_client(
                    limits=httpx.Limits(max_connections=None, max_keepalive_connections=100),
                )
    return _STREAM_HTTP_CLIENT


def _close_http_client() -> None:
    global _CHAT_HTTP_CLIENT, _HTTP_CLIENT, _STREAM_HTTP_CLIENT
    with _HTTP_CLIENT_LOCK:
        clients = (_HTTP_CLIENT, _CHAT_HTTP_CLIENT, _STREAM_HTTP_CLIENT)
        _HTTP_CLIENT = None
        _CHAT_HTTP_CLIENT = None
        _STREAM_HTTP_CLIENT = None
    for client in clients:
        if client is not None:
            client.close()


atexit.register(_close_http_client)


def _serving_request(
    method: str,
    url: str,
    *,
    json: dict | None = None,
    ok_statuses: tuple[int, ...] = (),
    timeout_s: float | None = None,
) -> httpx.Response:
    """Issue a request to the serving backend; translates failures into ServingError."""
    # follow_redirects: modal 303-redirects slow requests to an async-result poll url.
    timeout = 60.0 if timeout_s is None else min(60.0, max(0.0, float(timeout_s)))
    kwargs: dict = {"headers": _internal_key_header(), "timeout": timeout, "follow_redirects": True}
    if json is not None:
        kwargs["json"] = json
    try:
        resp = _http_client().request(method, url, **kwargs)
        if resp.status_code in ok_statuses:
            return resp
        resp.raise_for_status()
        return resp
    except httpx.HTTPStatusError as exc:
        raise _serving_status_error(url, exc) from exc
    except httpx.RequestError as exc:
        raise ServingError(f"could not reach the serving backend at {url}: {exc}") from exc


def _serving_status_error(url: str, exc: httpx.HTTPStatusError) -> ServingError:
    """Build a ServingError from an upstream HTTP failure with a tailored hint."""
    resp = exc.response
    status = resp.status_code if resp is not None else None
    detail = ((resp.text if resp is not None else "") or "").strip()[:500]
    msg = f"serving backend error for {url}"
    if status is not None:
        msg += f" (HTTP {status})"
    if detail:
        msg += f": {detail}"
    if status is not None and status < 500:
        msg += (
            " — the serving backend rejected the request (4xx); check FREESOLO_INTERNAL_KEY "
            "and the request payload (this is a client/auth error, not a serving outage)"
        )
    else:
        msg += (
            " — the serving backend is unavailable or has no engine for this base model; "
            "an operator must check the freesolo serving deployment"
        )
    headers = getattr(resp, "headers", {}) if resp is not None else {}
    retry_after = headers.get("Retry-After")
    return ServingError(msg, status_code=status, retry_after=retry_after)


def serving_base_url() -> str:
    """Env-overridable serving control root.

    Standalone planes must target a backend they operate because every request carries the plane's
    ``FREESOLO_INTERNAL_KEY``. Reject hosted URLs whether supplied explicitly or by fallback.
    """
    # imported lazily: flash.serve is the CLIENT side, and a module-level import would pull
    # flash.server into every CLI invocation.
    from flash.server.platform.auth import standalone

    configured = (os.environ.get("FREESOLO_SERVING_URL") or "").strip()
    if standalone() and (not configured or is_freesolo_hosted_url(configured)):
        raise ServingError(
            f"FREESOLO_SERVING_URL is {'not set' if not configured else 'a Freesolo-hosted URL'}. "
            "A standalone plane has no serving backend of its own, and using the hosted one would "
            "send FREESOLO_INTERNAL_KEY - the key that controls this plane - to a service you do "
            "not operate. Point FREESOLO_SERVING_URL at your own multi-LoRA deployment, or export "
            "the adapter and serve it yourself (see SELF_HOSTING.md). Training does not require "
            "this."
        )
    return serving_control_url(configured or DEFAULT_FREESOLO_SERVING_URL)


def serving_openai_base_url() -> str:
    """OpenAI-compatible base URL for the configured serving backend."""
    return openai_base_url(serving_base_url())


def _internal_key_header() -> dict[str, str]:
    # strip exactly as authenticate does: newlines are invalid headers and spaces change the key.
    # blank values omit the header.
    key = (os.environ.get("FREESOLO_INTERNAL_KEY") or "").strip()
    return {"X-Freesolo-Internal-Key": key} if key else {}


@dataclass
class Deployment:
    run_id: str
    model: str
    adapter_hf_prefix: str
    openai_model: str
    endpoint_name: str
    openai_base_url: str
    adapter_revision: str | None = None
    checkpoint_step: int | None = None
    verified_at: float | None = None
    state: str = "ready"

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_hf_revision(hf_repo: str) -> str:
    """Resolve the full immutable commit SHA for the uploaded adapter repository."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover
        raise ServingError(
            "could not resolve adapter revision: huggingface_hub is not installed"
        ) from exc
    try:
        revision = str(
            HfApi()
            .repo_info(
                repo_id=hf_repo,
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN"),
            )
            .sha
            or ""
        ).strip()
    except Exception as exc:
        raise ServingError(f"could not resolve adapter revision for {hf_repo}: {exc}") from exc
    if not is_commit_sha(revision):
        raise ServingError(f"could not resolve full Hub commit SHA for {hf_repo}")
    return revision.lower()


def deployment_record(
    run_id: str,
    model: str,
    adapter_prefix: str,
    *,
    state: str = "ready",
    checkpoint_step: int | None = None,
    adapter_revision: str | None = None,
) -> Deployment:
    subfolder = f"{adapter_prefix}/adapter"
    base = serving_base_url()
    openai_url = serving_openai_base_url()
    return Deployment(
        run_id=run_id,
        model=model,
        adapter_hf_prefix=subfolder,
        openai_model=run_id,
        adapter_revision=adapter_revision,
        checkpoint_step=checkpoint_step,
        endpoint_name=base,
        openai_base_url=openai_url,
        state=state,
    )


def deploy_adapter(
    run_id: str,
    model: str,
    hf_repo: str,
    adapter_prefix: str,
    *,
    dry_run: bool = False,
    lora_rank: int = 32,
    thinking: bool = False,
    structured_outputs: str = "",
    org_id: str | None = None,
    checkpoint_step: int | None = None,
    expected_adapter_revision: str | None = None,
    before_activate: Callable[[str, str], None] | None = None,
) -> Deployment:
    """Register, verify, and atomically activate one immutable adapter revision.

    Thinking adapters with structured outputs require serving to advertise deferred constraint
    support before the immutable revision is registered.
    """
    validate_serving_lora_rank(model, lora_rank, rank_source="configured train.lora_rank")
    subfolder = f"{adapter_prefix}/adapter"
    dep = deployment_record(
        run_id,
        model,
        adapter_prefix,
        state="dry_run" if dry_run else "queued",
        checkpoint_step=checkpoint_step,
    )
    if dry_run:
        return dep

    hf_revision = resolve_hf_revision(hf_repo)
    validate_serving_lora_rank(
        model,
        adapter_artifact_lora_rank(hf_repo, subfolder, hf_revision=hf_revision),
        rank_source="adapter artifact",
    )
    revision = format_adapter_revision(run_id, checkpoint_step, hf_revision)
    dep.adapter_revision = revision
    checkpoint = f"{run_id}/step-{checkpoint_step}" if checkpoint_step is not None else run_id
    so_default = parse_structured_outputs(structured_outputs) if structured_outputs else None
    advertised = _require_serving_capabilities(
        thinking_structured_outputs=thinking and so_default is not None
    )
    require_provenance = REVISION_PROVENANCE_CAPABILITY in advertised

    body = {
        "adapter_id": revision,
        "repo_id": hf_repo,
        "base_model": model,
        "subfolder": subfolder,
        "repo_type": "dataset",
        "checkpoint": checkpoint,
        # NOTE: do NOT send a client-chosen "status" -- the serving backend sets the record status
        # itself on registration and rejects an incoming "status" as an extra field (HTTP 422
        # extra_forbidden). Sending it blocked every deploy against the deployed serving backend.
        "metadata": {
            "record_type": "revision",
            "run_id": run_id,
            "checkpoint_step": checkpoint_step,
            "hf_revision": hf_revision,
        },
        "thinking": bool(thinking),
    }
    if so_default is not None:
        body["structured_outputs"] = so_default
    normalized_org_id = (org_id or "").strip()
    if normalized_org_id:
        body["org_id"] = normalized_org_id

    try:
        registration = _serving_request("POST", f"{serving_base_url()}/adapters", json=body)
        if registration.status_code not in {200, 202}:
            raise ServingError(
                f"serving returned unexpected adapter registration status {registration.status_code}"
            )
    except ServingError as exc:
        if exc.status_code is not None and exc.status_code < 500:
            raise
        try:
            record = _registered_adapter(revision)
        except ServingError as read_exc:
            raise exc from read_exc
        if record is None or not _matches_revision_identity(
            record, body, require_provenance=require_provenance
        ):
            raise exc
        logger.warning(
            "adapter registration response was ambiguous; revision %s exists with matching identity",
            revision,
        )

    _wait_revision_ready(
        revision,
        subfolder,
        expected_identity=body,
        require_provenance=require_provenance,
        budget_s=revision_ready_budget_seconds(model),
    )
    if before_activate is not None:
        before_activate(revision, checkpoint)
    activation = _activate_revision(
        run_id,
        revision,
        checkpoint,
        expected_adapter_revision=expected_adapter_revision,
    )
    dep.state = "ready"
    dep.openai_model = str(activation.get("adapter_id") or run_id)
    logger.info("activated adapter %s revision %s", run_id, revision)
    return dep


def _adapter_url(adapter_id: str) -> str:
    return f"{serving_base_url()}/adapters/{quote(adapter_id, safe='')}"


def _registered_adapter_response(
    adapter_id: str, *, timeout_s: float | None = None
) -> tuple[dict | None, httpx.Response]:
    """Read one authoritative adapter record and retain its polling headers."""
    resp = _serving_request(
        "GET",
        _adapter_url(adapter_id),
        ok_statuses=(404,),
        timeout_s=timeout_s,
    )
    if resp.status_code == 404:
        return None, resp
    try:
        payload = resp.json()
    except ValueError as exc:
        raise ServingError(
            f"serving returned invalid status JSON for adapter {adapter_id}"
        ) from exc
    if not isinstance(payload, dict):
        raise ServingError(f"serving returned invalid status data for adapter {adapter_id}")
    record = payload.get("adapter") if isinstance(payload.get("adapter"), dict) else payload
    if not isinstance(record, dict):
        raise ServingError(f"serving returned no adapter record for {adapter_id}")
    return record, resp


def _registered_adapter(adapter_id: str, *, timeout_s: float | None = None) -> dict | None:
    """Read one authoritative adapter record, including disabled records."""
    record, _ = _registered_adapter_response(adapter_id, timeout_s=timeout_s)
    return record


def _matches_revision_identity(
    record: dict, expected: dict, *, require_provenance: bool = True
) -> bool:
    scalar_fields = (
        "adapter_id",
        "repo_id",
        "repo_type",
        "subfolder",
        "base_model",
        "checkpoint",
        "thinking",
    )
    if any(record.get(field) != expected.get(field) for field in scalar_fields):
        return False
    if (record.get("org_id") or None) != (expected.get("org_id") or None):
        return False
    if (record.get("structured_outputs") or None) != (expected.get("structured_outputs") or None):
        return False
    if not require_provenance:
        # backends without revision_provenance do not echo provenance metadata; the immutable
        # adapter_id already pins the artifact, so this cross-check is best effort here.
        return True
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    expected_metadata = expected["metadata"]
    return all(
        metadata.get(field) == expected_metadata.get(field)
        for field in ("record_type", "run_id", "checkpoint_step", "hf_revision")
    )


def _require_serving_capabilities(*, thinking_structured_outputs: bool = False) -> set[str]:
    # SAFETY-CRITICAL capabilities the deploy correctness contract genuinely depends on: immutable
    # revisions (a revision id always maps to one artifact) and an atomic alias compare-and-swap
    # (the alias flip can't race). These are hard-required.
    required = {
        "immutable_adapter_revisions",
        "alias_compare_and_swap",
    }
    # PREFERRED (not required): `revision_provenance` only lets the serving backend echo back the
    # run/checkpoint/hf_revision metadata, which is used ONLY on the rare 5xx-during-registration
    # recovery path (`_matches_revision_identity`). Hard-requiring it blocked EVERY deploy org-wide
    # whenever the serving build lagged on advertising it, even though the happy path never uses it.
    # So its absence is a logged warning, not a deploy-blocking error.
    preferred = {REVISION_PROVENANCE_CAPABILITY}
    if thinking_structured_outputs:
        # Genuinely required for this run: thinking + structured outputs needs the serving backend's
        # deferred-constraint support (grammar applied after </think>) or served output is invalid.
        required.add(THINKING_STRUCTURED_OUTPUTS_CAPABILITY)
    url = f"{serving_base_url()}/healthz"
    response = _serving_request("GET", url)
    try:
        payload = response.json()
    except ValueError as exc:
        raise ServingError(
            f"serving_contract_unsupported: serving health check at {url} did not return valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise ServingError(
            f"serving_contract_unsupported: serving health check at {url} returned a non-object payload"
        )
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, list):
        raise ServingError(
            f"serving_contract_unsupported: serving health check at {url} must return a list field "
            "named capabilities"
        )
    if not all(isinstance(capability, str) for capability in capabilities):
        raise ServingError(
            f"serving_contract_unsupported: serving health check at {url} capabilities must be "
            "strings"
        )
    advertised = set(capabilities)
    missing = sorted(required - advertised)
    if missing:
        raise ServingError(
            "serving_contract_unsupported: serving is missing required capabilities "
            + ", ".join(missing)
        )
    missing_preferred = sorted(preferred - advertised)
    if missing_preferred:
        # Not fatal: the happy-path deploy does not use these; only the ambiguous-registration
        # recovery path degrades (best-effort identity check). Surface it so an operator can ship
        # the serving build that advertises them, without blocking customers meanwhile.
        logger.warning(
            "serving backend does not advertise preferred capabilities %s; deploying anyway "
            "(ambiguous-registration recovery will be best-effort)",
            ", ".join(missing_preferred),
        )
    return advertised


def revision_ready_budget_seconds(model: str) -> float:
    """Readiness budget for a cold serving engine holding ``model``, scaled by base-model size.

    An unknown model keeps the floor: a fork can add a catalog entry, and a revision-pinned id need
    not be a catalog key, so a lookup miss must not fail a deploy that would otherwise succeed.
    """
    from flash.core.catalog import MODELS

    info = MODELS.get(str(model or "").strip())
    if info is None:
        return REVISION_READY_MIN_BUDGET_SECONDS
    # total params, not active: an MoE loads every expert into VRAM even though a token routes
    # through few, so the cold-start cost tracks the full checkpoint.
    scaled = REVISION_READY_MIN_BUDGET_SECONDS + REVISION_READY_SECONDS_PER_PARAM_B * max(
        0.0, float(info.params_b)
    )
    return min(scaled, REVISION_READY_MAX_BUDGET_SECONDS)


def _readback_delay(attempt: int, retry_after: str | None = None) -> float:
    if retry_after is not None:
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            pass
        else:
            if math.isfinite(delay) and delay > 0:
                return min(delay, READBACK_MAX_DELAY_SECONDS)
    return min(READBACK_DELAY_SECONDS * (2**attempt), READBACK_MAX_DELAY_SECONDS)


def _wait_revision_ready(
    revision: str,
    subfolder: str,
    *,
    expected_identity: dict | None = None,
    require_provenance: bool = True,
    budget_s: float = REVISION_READY_MIN_BUDGET_SECONDS,
) -> dict:
    budget = max(0.0, float(budget_s))
    deadline = time.monotonic() + budget
    last_state = "registered"
    last_read_error: ServingError | None = None
    # the loader's own complaint, kept even when serving reports it WITHOUT moving the revision to
    # `failed`. without this a stuck load times out reporting only the state, and the one piece of
    # evidence that says which subsystem is at fault is dropped on the floor.
    last_failure: str | None = None
    observed_record = False
    first_read = True
    attempt = 0
    retry_after: str | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if not first_read:
            delay = _readback_delay(attempt - 1, retry_after)
            time.sleep(min(delay, remaining))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
        first_read = False
        try:
            record, response = _registered_adapter_response(revision, timeout_s=remaining)
        except ServingError as exc:
            if exc.status_code is not None and exc.status_code < 500:
                raise
            last_read_error = exc
            retry_after = exc.retry_after
            attempt += 1
            continue
        # a read that only completes once the whole budget is spent was not confirmed ready in
        # time; do not honor it, and leave last_state untouched so the timeout is reported.
        if deadline - time.monotonic() <= 0:
            break
        retry_after = response.headers.get("Retry-After")
        attempt += 1
        # a record fetched within the deadline is still inspected for readiness before giving up.
        if record is None:
            continue
        last_read_error = None
        if expected_identity is not None and not _matches_revision_identity(
            record, expected_identity, require_provenance=require_provenance
        ):
            raise ServingError(
                f"adapter revision {revision} resolved to a different immutable identity"
            )
        observed_record = True
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        last_state = str(
            metadata.get("lifecycle_state") or record.get("lifecycle_state") or "registered"
        )
        # track the LATEST record's failure, not the first one seen: a record that later omits or
        # clears `failure` has withdrawn that complaint, and keeping the stale string would make the
        # timeout below prescribe "fix the artifact" for what is now a plain cold-engine timeout.
        failure = metadata.get("failure")
        last_failure = str(failure) if failure else None
        if last_state == "failed" or record.get("status") == "disabled":
            raise ServingError(
                f"serving failed to load adapter revision {revision}: {failure or 'unknown error'}"
            )
        if last_state == "ready":
            value = record.get("subfolder")
            record_subfolder = str(value) if value is not None else None
            if record_subfolder == subfolder:
                return record
    if last_read_error is not None:
        raise ServingError(
            f"adapter revision {revision} readiness could not be confirmed after transient "
            f"serving errors: {last_read_error}"
        ) from last_read_error
    # a TIMEOUT, not a rejection. the two are distinguishable in code (a rejected adapter raises
    # "serving failed to load adapter revision" above) but the old message said only that the
    # revision "remained 'registered'", which reads as a serving fault and sent readers to the wrong
    # subsystem. say which of the two happened, what the clock actually was, and that a retry is the
    # correct response to THIS one.
    details = [f"waited {budget:g}s"]
    if observed_record:
        details.append(f"last state {last_state!r}")
    else:
        # serving never returned a record: every completed poll 404ed. that is registration
        # visibility, not a slow load, and it points at a different failure than a stuck loader.
        # no lifecycle state was ever observed, so do not quote the initial one as if it were read.
        details.append("serving never returned the revision record (every completed poll 404ed)")
    if last_failure:
        details.append(f"loader reported: {last_failure}")
    # the advice depends on WHICH timeout this is. a loader that named a failure while leaving the
    # revision un-failed has already told us the artifact or config is wrong, and that survives a
    # warm engine, so prescribing a retry there would send the reader into a futile loop -- the same
    # wrong-direction problem this message exists to fix.
    if last_failure:
        remedy = (
            "serving reported that failure without failing the revision, so this is unlikely to be "
            "a cold-engine timeout: fix what the loader reported before retrying, because a retry "
            "against a warm engine will hit the same artifact."
        )
    elif not observed_record:
        # nothing was ever read back, so there is no evidence of a loading engine to retry against.
        # the fault is that the revision never became visible, which a warm engine does not fix.
        remedy = (
            "no engine or loader state was ever observed, so this is a registration-visibility "
            "problem rather than a slow load: check that the revision was registered against this "
            "serving backend before retrying."
        )
    else:
        remedy = (
            "serving may still be loading, so retrying this deploy is the correct response: a cold "
            "engine loading a large base model can exceed the budget, and the retry usually "
            "succeeds against the now-warm engine."
        )
    raise ServingError(
        f"revision_ready_timeout: adapter revision {revision} did not become ready in time "
        f"({'; '.join(details)}). the previous alias remains available and {remedy} this is NOT "
        "the same as serving rejecting the adapter, which fails the deployment with 'serving "
        "failed to load adapter revision'."
    )


def _active_alias_target(record: dict | None) -> str | None:
    if not isinstance(record, dict) or record.get("status") == "disabled":
        return None
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("alias_of"), str):
        return metadata["alias_of"]
    return None


def adapter_alias_target(run_id: str) -> str | None:
    """Read the authoritative immutable revision targeted by a mutable run alias."""
    record = _registered_adapter(run_id)
    if record is None or record.get("status") == "disabled":
        return None
    target = _active_alias_target(record)
    if target is None:
        raise ServingError(
            f"serving alias {run_id} is not an immutable alias record; legacy aliases are unsupported"
        )
    return target


def _validate_activation_response(
    response: object,
    *,
    run_id: str,
    revision: str,
    checkpoint: str,
    expected_adapter_revision: str | None,
) -> dict:
    if not isinstance(response, dict):
        raise ServingError("serving returned an invalid alias activation response")
    if response.get("adapter_id") != run_id or response.get("target_adapter_revision") != revision:
        raise ServingError("serving returned mismatched committed alias activation provenance")
    if response.get("previous_adapter_revision") != expected_adapter_revision:
        raise ServingError("serving returned mismatched previous alias revision")
    if response.get("checkpoint") != checkpoint:
        raise ServingError("serving returned mismatched committed alias checkpoint")
    if not isinstance(response.get("updated_at"), str) or not response["updated_at"].strip():
        raise ServingError("serving returned committed alias activation without updated_at")
    return response


def _activate_revision(
    run_id: str,
    revision: str,
    checkpoint: str,
    *,
    expected_adapter_revision: str | None,
) -> dict:
    body = {"expected_adapter_revision": expected_adapter_revision}
    try:
        response = _serving_request(
            "POST",
            f"{_adapter_url(revision)}/activate",
            json=body,
        ).json()
        return _validate_activation_response(
            response,
            run_id=run_id,
            revision=revision,
            checkpoint=checkpoint,
            expected_adapter_revision=expected_adapter_revision,
        )
    except (ServingError, ValueError) as exc:
        alias = None
        read_error: ServingError | None = None
        target = None
        for attempt in range(ACTIVATION_READBACK_ATTEMPTS):
            if attempt:
                time.sleep(ACTIVATION_READBACK_DELAY_SECONDS)
            try:
                alias = _registered_adapter(run_id)
                read_error = None
            except ServingError as read_exc:
                read_error = read_exc
                continue
            target = _active_alias_target(alias)
            if target == revision:
                return {
                    "adapter_id": run_id,
                    "target_adapter_revision": revision,
                    "previous_adapter_revision": expected_adapter_revision,
                    "checkpoint": checkpoint,
                    "updated_at": alias.get("updated_at") if alias else None,
                }
            if target not in (None, expected_adapter_revision):
                break
        if read_error is not None:
            raise ActivationOutcomeUnknown(run_id, revision) from read_error
        if expected_adapter_revision is not None and target == expected_adapter_revision:
            raise ServingError(
                f"alias activation was not committed; {run_id} still targets {target!r}"
            ) from exc
        if target is None:
            raise ActivationOutcomeUnknown(
                run_id,
                revision,
                detail=(
                    "alias activation outcome is ambiguous because authoritative readback exposed "
                    "no target"
                ),
            ) from exc
        raise ActivationOutcomeUnknown(
            run_id,
            revision,
            detail=f"alias activation diverged because authoritative readback targets {target!r}",
        ) from exc


def undeploy_adapter(run_id: str) -> dict:
    """Disable the run alias and all immutable revisions without engine eviction."""
    response = _serving_request(
        "DELETE",
        _adapter_url(run_id),
        ok_statuses=(404,),
    )
    if response.status_code == 404:
        return {
            "run_id": run_id,
            "disabled_aliases": [],
            "disabled_revisions": [],
            "serving_deregistered": False,
        }
    try:
        payload = response.json()
    except ValueError as exc:
        raise ServingError("serving returned an invalid undeploy response") from exc
    if not isinstance(payload, dict) or payload.get("run_id") != run_id:
        raise ServingError("serving returned a mismatched undeploy response")
    for field in ("disabled_aliases", "disabled_revisions"):
        value = payload.setdefault(field, [])
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ServingError(f"serving returned invalid {field} in undeploy response")
    payload["serving_deregistered"] = bool(
        payload["disabled_aliases"] or payload["disabled_revisions"]
    )
    return payload


def _retryable_smoke_unavailable(
    response: httpx.Response,
    *,
    requested_model: str,
) -> RetryableServingUnavailable | None:
    if response.status_code != 503:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("error"), dict):
        return None
    error = payload["error"]
    code = error.get("code")
    if (
        error.get("type") != "adapter_unavailable"
        or error.get("retryable") is not True
        or code not in _RETRYABLE_SMOKE_503_CODES
        or error.get("requested_model") != requested_model
        or error.get("adapter_revision") != requested_model
    ):
        return None
    raw_delay = response.headers.get("Retry-After") or error.get("retry_after_seconds")
    try:
        retry_after_seconds = float(raw_delay)
    except (TypeError, ValueError):
        retry_after_seconds = SMOKE_RETRY_FALLBACK_DELAY_SECONDS
    if not math.isfinite(retry_after_seconds) or retry_after_seconds <= 0:
        retry_after_seconds = SMOKE_RETRY_FALLBACK_DELAY_SECONDS
    return RetryableServingUnavailable(str(code), retry_after_seconds)


def chat(
    run_id: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 512,
    thinking: bool = False,
    expected_checkpoint: str | None = None,
    timeout_s: float | None = None,
    retry_unavailable: bool = False,
    stop: list[str] | None = None,
) -> dict:
    """Send an OpenAI-style chat request for the run's adapter to freesolo serving.

    ``timeout_s`` overrides the default 30-minute request timeout. deployment smoke also enables
    recognized unavailable-envelope classification so its caller can retry within one deadline.
    ``stop`` carries the run's own stop sequences so a model trained to terminate on a delimiter
    rather than EOS finishes on ``stop`` instead of running to ``max_tokens``.
    """
    base = serving_openai_base_url()
    body = {
        "model": run_id,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }
    if stop:
        body["stop"] = [str(value) for value in stop]
    # follow_redirects: modal 303-redirects slow cold-start requests across many poll cycles
    # before the result is ready (bounded by _MAX_REDIRECTS, all on the serving origin).
    headers = _internal_key_header()
    if expected_checkpoint:
        headers["X-Freesolo-Expected-Checkpoint"] = expected_checkpoint
    timeout = 30 * 60.0 if timeout_s is None else max(0.0, float(timeout_s))
    client_context = (
        _new_serving_client() if retry_unavailable else contextlib.nullcontext(_chat_http_client())
    )
    with client_context as client:
        resp = client.post(f"{base}/chat/completions", json=body, headers=headers, timeout=timeout)
        if retry_unavailable:
            retryable_error = _retryable_smoke_unavailable(resp, requested_model=run_id)
            if retryable_error is not None:
                raise retryable_error
        resp.raise_for_status()
        payload = resp.json()
        _balance_thinking_payload(payload, thinking=thinking)
        if expected_checkpoint and isinstance(payload, dict):
            payload["_freesolo_headers"] = {
                "adapter_revision": resp.headers.get("X-Freesolo-Adapter-Revision"),
                "checkpoint": resp.headers.get("X-Freesolo-Checkpoint"),
                "hf_revision": resp.headers.get("X-Freesolo-HF-Revision"),
            }
        return payload


# Adapter artifact validation lives in `flash.serve.adapter_check`, which imports the exception
# types above. Re-exported so `deploy_adapter` and the serving tests keep resolving them here.
from flash.serve.adapter_check import (  # noqa: E402,F401
    _is_adapter_tensor_filename,
    _is_hf_not_found_error,
    _verify_adapter_artifact_tensors,
    adapter_artifact_lora_rank,
    validate_serving_lora_rank,
)

# `chat_stream` and its SSE decoder live in `flash.serve.streaming`, which imports the client
# helpers above. Re-exported so `from flash.serve.deploy import chat_stream` (flash.server.app,
# and the CLI through the client) keeps resolving.
from flash.serve.streaming import (  # noqa: E402,F401
    _openai_stream_content,
    chat_stream,
)

# The reasoning-delimiter parsing lives in `flash.serve.thinking`, which imports the tag
# constants above. Re-exported so the existing `deploy._balanced_thinking_content` call sites
# and tests keep resolving.
from flash.serve.thinking import (  # noqa: E402,F401
    _TAG_CLOSE,
    _TAG_OPEN,
    _balance_thinking_payload,
    _balanced_thinking_content,
    _delimiter_may_complete,
    _duplicates_reasoning,
    _find_delimiter,
    _inline_reasoning_block,
    _is_only_retained_delimiter,
    _is_sampled_delimiter,
    _is_terminal_reasoning_repeat,
    _retained_delimiter_end,
    _strip_retained_close,
)
