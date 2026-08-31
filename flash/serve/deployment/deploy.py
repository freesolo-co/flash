"""Thin client for the freesolo multi-LoRA serving app (Modal); no flash-side vLLM."""

from __future__ import annotations

import contextlib
import math
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import quote

import httpx

import flash.serve.contract.errors as serving_errors
import flash.serve.contract.urls as serving_urls
from flash._internal.logging import get_logger
from flash.content.structured_outputs import parse_structured_outputs
from flash.envs.loading.loader import is_commit_sha
from flash.schema import format_checkpoint_ref, parse_checkpoint_ref
from flash.serve.contract.protocol import (
    PREFERRED_SERVING_CAPABILITIES,
    REQUIRED_SERVING_CAPABILITIES,
    THINKING_STRUCTURED_OUTPUTS_CAPABILITY,
    ServingHealthError,
    parse_serving_health,
)
from flash.serve.contract.provenance import immutable_binding_fingerprint
from flash.serve.contract.responses import (
    matches_revision_identity as _matches_revision_identity,
)
from flash.serve.deployment import adapter_check, readiness
from flash.serve.request import streaming as streaming_support
from flash.serve.request import thinking as thinking_support
from flash.serve.request import transport
from flash.serve.runtime.sampling import (
    validate_choice_count,
    validate_logprobs,
    validate_top_logprobs,
)

logger = get_logger(__name__)

READBACK_DELAY_SECONDS = 0.5
READBACK_MAX_DELAY_SECONDS = 2.0
SMOKE_RETRY_FALLBACK_DELAY_SECONDS = 2.0


@dataclass
class Deployment:
    run_id: str
    model: str
    adapter_hf_prefix: str
    openai_model: str
    endpoint_name: str
    openai_base_url: str
    checkpoint_id: str | None = None
    checkpoint_step: int | None = None
    verified_at: float | None = None
    state: str = "ready"

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_artifact_revision(hf_repo: str) -> str:
    """Resolve the full immutable commit SHA for the uploaded adapter repository."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover
        raise serving_errors.ServingError(
            "could not resolve checkpoint: huggingface_hub is not installed"
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
        raise serving_errors.ServingError(
            f"could not resolve checkpoint for {hf_repo}: {exc}"
        ) from exc
    if not is_commit_sha(revision):
        raise serving_errors.ServingError(f"could not resolve full Hub commit SHA for {hf_repo}")
    return revision.lower()


def deployment_record(
    run_id: str,
    model: str,
    adapter_prefix: str,
    *,
    state: str = "ready",
    checkpoint_step: int | None = None,
) -> Deployment:
    subfolder = f"{adapter_prefix}/adapter"
    base = serving_urls.serving_base_url()
    openai_url = transport.serving_openai_base_url()
    checkpoint_id = format_checkpoint_ref(run_id, checkpoint_step)
    return Deployment(
        run_id=run_id,
        model=model,
        adapter_hf_prefix=subfolder,
        openai_model=checkpoint_id,
        checkpoint_id=checkpoint_id,
        checkpoint_step=checkpoint_step,
        endpoint_name=base,
        openai_base_url=openai_url,
        state=state,
    )


def _call_before_ready(
    callback: Callable[..., None],
    revision: str,
    checkpoint: str,
    advertised: frozenset[str],
    *,
    adapter_targets_images: bool,
) -> None:
    """Invoke the pre-activation callback with the deployment facts it declares."""
    import inspect

    try:
        parameters = inspect.signature(callback).parameters
    except (TypeError, ValueError):  # builtins and C callables are not introspectable
        parameters = {}
    accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    kwargs = {}
    if accepts_kwargs or "advertised_capabilities" in parameters:
        kwargs["advertised_capabilities"] = advertised
    if accepts_kwargs or "adapter_targets_images" in parameters:
        kwargs["adapter_targets_images"] = adapter_targets_images
    callback(revision, checkpoint, **kwargs)


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
    # called with the immutable target plus keyword-only deployment facts already read here. handing
    # them down avoids re-fetching either health or adapter metadata from inside the paid smoke path.
    # keywords preserve two-argument callbacks that do not need those facts.
    before_ready: Callable[..., None] | None = None,
) -> Deployment:
    """register, load, and verify one permanent checkpoint identity.

    Thinking adapters with structured outputs require serving to advertise deferred constraint
    support before the immutable revision is registered.
    """
    from flash.serving.src.engine.model_config import is_supported_base_model

    if not is_supported_base_model(model):
        raise ValueError(f"model {model!r} is not active in hosted serving")
    adapter_check.validate_serving_lora_rank(
        model, lora_rank, rank_source="configured train.lora_rank"
    )
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

    checkpoint_id = format_checkpoint_ref(run_id, checkpoint_step)
    dep.checkpoint_id = checkpoint_id
    from flash.server.platform.auth import serving_org_id

    normalized_org_id = serving_org_id(org_id)
    bound_record = (
        _registered_adapter(normalized_org_id, checkpoint_id) if normalized_org_id else None
    )
    bound_revision = (
        str(bound_record.get("artifact_revision") or "").strip().lower()
        if isinstance(bound_record, dict)
        else ""
    )
    artifact_revision = (
        bound_revision if is_commit_sha(bound_revision) else resolve_artifact_revision(hf_repo)
    )
    artifact_metadata = adapter_check.adapter_artifact_metadata(
        hf_repo, subfolder, artifact_revision=artifact_revision
    )
    adapter_check.validate_serving_lora_rank(
        model,
        artifact_metadata.lora_rank,
        rank_source="adapter artifact",
    )
    so_default = parse_structured_outputs(structured_outputs) if structured_outputs else None
    advertised = _require_serving_capabilities(
        thinking_structured_outputs=thinking and so_default is not None
    )

    body = {
        "adapter_id": checkpoint_id,
        "repo_id": hf_repo,
        "base_model": model,
        "subfolder": subfolder,
        "repo_type": "dataset",
        "checkpoint": checkpoint_id,
        "run_id": run_id,
        "checkpoint_step": checkpoint_step,
        "artifact_revision": artifact_revision,
        "artifact_digest": artifact_metadata.artifact_digest,
        "lora_rank": artifact_metadata.lora_rank,
        "thinking": bool(thinking),
    }
    if so_default is not None:
        body["structured_outputs"] = so_default
    if not normalized_org_id:
        raise ValueError("org_id is required for hosted checkpoint deployment")
    body["org_id"] = normalized_org_id
    body["artifact_fingerprint"] = immutable_binding_fingerprint(body)

    try:
        registration = transport.serving_request(
            "POST",
            f"{serving_urls.serving_base_url()}/adapters",
            json=body,
            org_id=normalized_org_id,
        )
        if registration.status_code not in {200, 202}:
            raise serving_errors.ServingError(
                f"serving returned unexpected adapter registration status {registration.status_code}"
            )
    except serving_errors.ServingError as exc:
        if exc.status_code is not None and exc.status_code < 500:
            raise
        try:
            record = _registered_adapter(normalized_org_id, checkpoint_id)
        except serving_errors.ServingError as read_exc:
            raise exc from read_exc
        if record is None or not _matches_revision_identity(record, body):
            raise exc
        logger.warning(
            "adapter registration response was ambiguous; checkpoint %s exists with matching identity",
            checkpoint_id,
        )

    _wait_checkpoint_ready(
        normalized_org_id,
        checkpoint_id,
        subfolder,
        expected_identity=body,
        budget_s=readiness.revision_ready_budget_seconds(model),
    )
    if before_ready is not None:
        _call_before_ready(
            before_ready,
            checkpoint_id,
            checkpoint_id,
            frozenset(advertised),
            adapter_targets_images=artifact_metadata.targets_images,
        )
    dep.state = "ready"
    dep.openai_model = checkpoint_id
    logger.info("deployed checkpoint %s", checkpoint_id)
    return dep


def _adapter_url(adapter_id: str) -> str:
    return f"{serving_urls.serving_base_url()}/adapters/{quote(adapter_id, safe='')}"


def _registered_adapter_response(
    org_id: str, adapter_id: str, *, timeout_s: float | None = None
) -> tuple[dict | None, httpx.Response]:
    """Read one authoritative adapter record and retain its polling headers."""
    resp = transport.serving_request(
        "GET",
        _adapter_url(adapter_id),
        ok_statuses=(404,),
        timeout_s=timeout_s,
        org_id=org_id,
    )
    if resp.status_code == 404:
        return None, resp
    try:
        payload = resp.json()
    except ValueError as exc:
        raise serving_errors.ServingError(
            f"serving returned invalid status JSON for adapter {adapter_id}"
        ) from exc
    if not isinstance(payload, dict):
        raise serving_errors.ServingError(
            f"serving returned invalid status data for adapter {adapter_id}"
        )
    record = payload.get("adapter") if isinstance(payload.get("adapter"), dict) else payload
    if not isinstance(record, dict):
        raise serving_errors.ServingError(f"serving returned no adapter record for {adapter_id}")
    return record, resp


def _registered_adapter(
    org_id: str, adapter_id: str, *, timeout_s: float | None = None
) -> dict | None:
    """read one tenant-scoped authoritative adapter record, including disabled records."""

    record, _ = _registered_adapter_response(org_id, adapter_id, timeout_s=timeout_s)
    return record


def _require_serving_capabilities(*, thinking_structured_outputs: bool = False) -> set[str]:
    # the hosted backend must advertise the permanent checkpoint identity contract.
    required = set(REQUIRED_SERVING_CAPABILITIES)
    preferred = PREFERRED_SERVING_CAPABILITIES
    if thinking_structured_outputs:
        # Genuinely required for this run: thinking + structured outputs needs the serving backend's
        # deferred-constraint support (grammar applied after </think>) or served output is invalid.
        required.add(THINKING_STRUCTURED_OUTPUTS_CAPABILITY)
    url = f"{serving_urls.serving_base_url()}/healthz"
    response = transport.serving_request("GET", url)
    try:
        payload = response.json()
    except ValueError as exc:
        raise serving_errors.ServingError(
            f"serving_contract_unsupported: serving health check at {url} did not return valid JSON"
        ) from exc
    try:
        health = parse_serving_health(payload)
    except ServingHealthError as exc:
        if exc.code == "non_object":
            detail = "returned a non-object payload"
        elif exc.code == "capabilities_not_list":
            detail = "must return a list field named capabilities"
        else:
            detail = "capabilities must be strings"
        raise serving_errors.ServingError(
            f"serving_contract_unsupported: serving health check at {url} {detail}"
        ) from exc
    if health.ok is False:
        raise serving_errors.ServingError(
            f"serving_contract_unsupported: serving health check at {url} reported ok=false"
        )
    advertised = set(health.capabilities)
    missing = sorted(required - advertised)
    if missing:
        raise serving_errors.ServingError(
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


def _wait_checkpoint_ready(
    org_id: str,
    revision: str,
    subfolder: str,
    *,
    expected_identity: dict | None = None,
    budget_s: float = readiness.REVISION_READY_MIN_BUDGET_SECONDS,
) -> dict:
    budget = max(0.0, float(budget_s))
    deadline = time.monotonic() + budget
    last_state = "registered"
    last_read_error: serving_errors.ServingError | None = None
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
            record, response = _registered_adapter_response(org_id, revision, timeout_s=remaining)
        except serving_errors.ServingError as exc:
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
            record, expected_identity
        ):
            raise serving_errors.ServingError(
                f"checkpoint {revision} resolved to a different immutable identity"
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
            raise serving_errors.ServingError(
                f"serving failed to load checkpoint {revision}: {failure or 'unknown error'}"
            )
        if last_state == "ready":
            value = record.get("subfolder")
            record_subfolder = str(value) if value is not None else None
            if record_subfolder == subfolder:
                return record
    if last_read_error is not None:
        # a transient read error is not an authoritative record, so it withdraws nothing. when an
        # earlier authoritative record named a loader failure, that complaint still stands and is
        # the more actionable half: reporting only the read error would send the reader to retry
        # against serving when serving already said the artifact itself is wrong.
        message = (
            f"checkpoint {revision} readiness could not be confirmed after transient "
            f"serving errors: {last_read_error}"
        )
        if last_failure:
            message += (
                f". before those errors the loader reported: {last_failure} -- that is not "
                "transient and survives a warm engine, so fix it before retrying"
            )
        raise serving_errors.ServingError(message) from last_read_error
    # a TIMEOUT, not a rejection. the two are distinguishable in code (a rejected adapter raises
    # "serving failed to load checkpoint" above) but the old message said only that the
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
        details.append("serving never returned the checkpoint record (every completed poll 404ed)")
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
    raise serving_errors.ServingError(
        f"checkpoint_ready_timeout: checkpoint {revision} did not become ready in time "
        f"({'; '.join(details)}). {remedy} this is NOT "
        "the same as serving rejecting the adapter, which fails the deployment with 'serving "
        "failed to load checkpoint'."
    )


def undeploy_adapter(checkpoint_id: str, *, org_id: str) -> dict:
    """disable one exact permanent checkpoint."""
    response = transport.serving_request(
        "DELETE",
        _adapter_url(checkpoint_id),
        ok_statuses=(404,),
        org_id=org_id,
    )
    if response.status_code == 404:
        return {
            "checkpoint_id": checkpoint_id,
            "disabled_checkpoints": [],
            "serving_deregistered": False,
        }
    try:
        payload = response.json()
    except ValueError as exc:
        raise serving_errors.ServingError("serving returned an invalid undeploy response") from exc
    if not isinstance(payload, dict) or payload.get("checkpoint_id") != checkpoint_id:
        raise serving_errors.ServingError("serving returned a mismatched undeploy response")
    for field in ("disabled_checkpoints",):
        value = payload.setdefault(field, [])
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise serving_errors.ServingError(
                f"serving returned invalid {field} in undeploy response"
            )
    payload["serving_deregistered"] = bool(payload["disabled_checkpoints"])
    return payload


def _retryable_smoke_unavailable(
    response: httpx.Response,
    *,
    requested_model: str,
    expected_checkpoint_id: str,
) -> serving_errors.RetryableServingUnavailable | None:
    return transport.retryable_smoke_unavailable(
        response,
        requested_model=requested_model,
        expected_checkpoint_id=expected_checkpoint_id,
        fallback_delay_seconds=SMOKE_RETRY_FALLBACK_DELAY_SECONDS,
    )


def _openai_stream_content(lines: Iterator[str], *, thinking: bool) -> Iterator[str]:
    return streaming_support._openai_stream_content(
        lines, thinking=thinking, find_delimiter=thinking_support._find_delimiter
    )


def chat_sse(
    run_id: str,
    messages: list[dict],
    *,
    org_id: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    thinking: bool = False,
    top_p: float = 0.95,
    stop: list[str] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    structured_outputs: dict[str, Any] | None = None,
    stream_options: dict[str, bool] | None = None,
    n: int = 1,
    seed: int | None = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    logprobs: bool = False,
    top_logprobs: int = 0,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    parallel_tool_calls: bool | None = None,
) -> transport.OpenAIStreamResponse:
    """open a raw openai stream while preserving status, headers, and sse bytes."""

    if parse_checkpoint_ref(run_id) is None:
        raise ValueError("chat target must be `<run_id>/final` or `<run_id>/step-N`")
    return transport.request_chat_sse(
        transport._chat_http_client(),
        url=f"{transport.serving_openai_base_url()}/chat/completions",
        headers=transport._internal_key_header(org_id=org_id),
        run_id=run_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
        top_p=top_p,
        n=n,
        seed=seed,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        stop=stop,
        chat_template_kwargs=chat_template_kwargs,
        structured_outputs=structured_outputs,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        stream_options=stream_options,
        frame_bytes=streaming_support._complete_sse_frames,
    )


def chat_stream(
    run_id: str,
    messages: list[dict],
    *,
    org_id: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    thinking: bool = False,
    stop: list[str] | None = None,
    n: int = 1,
    seed: int | None = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    logprobs: bool = False,
    top_logprobs: int = 0,
    tools: list[dict[str, Any]] | None = None,
) -> Iterator[str]:
    """yield one decoded choice while preserving eager open and cleanup semantics."""

    if tools is not None:
        raise ValueError("text-only chat_stream does not support tools")
    n = validate_choice_count(n)
    logprobs = validate_logprobs(logprobs)
    top_logprobs = validate_top_logprobs(top_logprobs)
    if n != 1:
        raise ValueError("text-only chat_stream requires n=1")
    if logprobs or top_logprobs:
        raise ValueError("text-only chat_stream does not expose logprobs")

    def decode_body(upstream: transport.OpenAIStreamResponse, enabled: bool) -> Iterator[str]:
        return streaming_support._streamed_body(
            upstream,
            thinking=enabled,
            find_delimiter=thinking_support._find_delimiter,
        )

    if parse_checkpoint_ref(run_id) is None:
        raise ValueError("chat target must be `<run_id>/final` or `<run_id>/step-N`")
    return transport.request_chat_stream(
        transport._chat_http_client(),
        url=f"{transport.serving_openai_base_url()}/chat/completions",
        headers=transport._internal_key_header(org_id=org_id),
        run_id=run_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
        n=n,
        seed=seed,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        stop=stop,
        frame_bytes=streaming_support._complete_sse_frames,
        decode_body=decode_body,
    )


def chat(
    run_id: str,
    messages: list[dict],
    *,
    org_id: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    thinking: bool = False,
    expected_checkpoint: str | None = None,
    timeout_s: float | None = None,
    retry_unavailable: bool = False,
    stop: list[str] | None = None,
    structured_outputs: dict | None = None,
    top_p: float = 0.95,
    chat_template_kwargs: dict | None = None,
    n: int = 1,
    seed: int | None = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    logprobs: bool = False,
    top_logprobs: int = 0,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    parallel_tool_calls: bool | None = None,
) -> dict:
    """Send an OpenAI-style chat request for the run's adapter to freesolo serving.

    the target must be an explicit permanent checkpoint id. ``timeout_s`` overrides the default
    30-minute request timeout. deployment smoke also enables
    recognized unavailable-envelope classification so its caller can retry within one deadline.
    ``stop`` carries the run's own stop sequences so a model trained to terminate on a delimiter
    rather than EOS finishes on ``stop`` instead of running to ``max_tokens``.
    """
    # follow_redirects: modal 303-redirects slow cold-start requests across many poll cycles
    # before the result is ready, bounded by the transport redirect limit on the serving origin.
    if parse_checkpoint_ref(run_id) is None:
        raise ValueError("chat target must be `<run_id>/final` or `<run_id>/step-N`")
    headers = transport._internal_key_header(org_id=org_id)
    if expected_checkpoint:
        headers["X-Freesolo-Expected-Checkpoint"] = expected_checkpoint
    timeout = 30 * 60.0 if timeout_s is None else max(0.0, float(timeout_s))
    client_context = (
        transport._new_serving_client()
        if retry_unavailable
        else contextlib.nullcontext(transport._chat_http_client())
    )

    def classify_unavailable(response: httpx.Response) -> None:
        if not retry_unavailable:
            return
        retryable_error = _retryable_smoke_unavailable(
            response,
            requested_model=run_id,
            expected_checkpoint_id=expected_checkpoint or run_id,
        )
        if retryable_error is not None:
            raise retryable_error

    return transport.request_chat_json(
        client_context,
        url=f"{transport.serving_openai_base_url()}/chat/completions",
        headers=headers,
        run_id=run_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
        top_p=top_p,
        n=n,
        seed=seed,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        stop=stop,
        chat_template_kwargs=chat_template_kwargs,
        structured_outputs=structured_outputs,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        timeout=timeout,
        before_raise=classify_unavailable,
        balance_payload=lambda payload, enabled: thinking_support._balance_thinking_payload(
            payload, thinking=enabled
        ),
        expected_checkpoint=expected_checkpoint,
    )
