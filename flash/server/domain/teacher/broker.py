"""Closed managed-teacher broker for Parasail completion scoring."""

from __future__ import annotations

import contextlib
import http.client
import json
import math
import os
import sqlite3
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any

from flash._internal.fileio import reject_duplicate_keys
from flash.core.spec import PUBLIC_URL_ENV, TEACHER_CAPABILITY_ENV, JobSpec
from flash.engine.plan.recipe import RECIPE, resolve_teacher
from flash.server.domain.teacher.errors import TeacherBrokerError, ValidatedCompletionRequest
from flash.server.domain.teacher.requests import (
    CAPABILITY_PATTERN,
    MAX_REQUEST_BODY_BYTES,
    REQUEST_ID_PATTERN,
    _canonical_json,
    _reject_nonfinite,
    parse_strict_json,
    request_fingerprint,
    validate_capability_token,
    validate_chat_completion_request,
    validate_completion_request,
    validate_request_id,
)

# the validation half moved to `teacher_requests`, but routes and tests reach these names through
# THIS module (`teacher_broker.validate_completion_request`, and patch seams like
# `teacher_broker._provider_chat_post`). re-export so the split stays invisible to callers; the
# names below are unused in this file by design, which is what __all__ tells ruff.
__all__ = [
    "CAPABILITY_PATTERN",
    "MAX_REQUEST_BODY_BYTES",
    "REQUEST_ID_PATTERN",
    "TeacherBrokerError",
    "ValidatedCompletionRequest",
    "authenticate_teacher_capability",
    "capability_limits_for_spec",
    "complete_teacher_chat_request",
    "complete_teacher_request",
    "issue_teacher_capability",
    "parse_strict_json",
    "request_fingerprint",
    "require_teacher_broker_configuration",
    "resolve_public_url",
    "teacher_attempt_transport",
    "validate_capability_token",
    "validate_chat_completion_request",
    "validate_completion_request",
    "validate_public_url",
    "validate_request_id",
]
from flash.server.platform import db
from flash.teacher.limits import (
    OPD_TEACHER_SCORING_CONCURRENCY,
    configured_opd_turn_limit,
    opd_teacher_request_multiplier,
)

PARASAIL_PROVIDER = "parasail"
PARASAIL_ORIGIN = "https://api.parasail.io"
PARASAIL_COMPLETIONS_PATH = "/v1/completions"
PARASAIL_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
PARASAIL_SCORING_MODE = "supplied_token_completion_v1"
PARASAIL_API_KEY_ENV = "PARASAIL_API_KEY"

MAX_RESPONSE_BODY_BYTES = 32 * 1024 * 1024
MAX_PROVIDER_ERROR_BODY_BYTES = 64 * 1024
# per-logical-request upstream attempt budget, enforced by mark_teacher_request_started.
# retries reuse the request_id, so they spend this budget rather than the request/score-item
# quotas, and the no-signal replacement multiplier composes with it instead of multiplying it.
# kept equal to the worker client's max_retries so neither side retries past the other.
MAX_UPSTREAM_ATTEMPTS = 4
MAX_REQUEST_TOKENS = 131_072
MAX_TOTAL_SCORE_ITEMS = 1_000_000
MAX_TOTAL_REQUESTS = 1_000_000
MAX_TOTAL_TOKENS = 2_000_000_000
MAX_CAPABILITY_LIFETIME_S = 24 * 60 * 60
MAX_PROVIDER_TIMEOUT_S = 90.0


class TeacherBrokerConfigurationError(RuntimeError):
    """A submit-time managed-teacher gate rejection whose message is safe to show the submitter.

    The generic run-failure handler redacts ``str(exc)`` on purpose: an arbitrary exception can name
    internal storage paths, provider internals, or upstream bodies. These messages are authored
    here, contain no such detail, and are the only thing that distinguishes a plane misconfiguration
    from a bad spec -- so this type opts them back in to being surfaced rather than swallowed.

    ``plane_fault`` carries which side is at fault. The submitter cannot fix an unset plane-side
    credential by editing their config, so reporting it as a client error would re-create the same
    conflation one layer up: a spec the user must change and an outage they can only wait out would
    again be indistinguishable.
    """

    def __init__(self, message: str, *, plane_fault: bool = False) -> None:
        super().__init__(message)
        self.plane_fault = plane_fault


@dataclass(frozen=True)
class ProviderResponse:
    body: dict[str, Any]
    input_tokens: int
    output_tokens: int


def validate_public_url(value: str) -> str:
    url = str(value or "").strip().rstrip("/")
    parsed = urllib.parse.urlsplit(url)
    # urlsplit defers port parsing to attribute access, so a malformed, out-of-range, or zero port
    # still yields a hostname and would pass every check below. force it here, because the worker
    # reads parsed.port when it opens the broker connection: an unparseable one raises there, after
    # the gpu is already allocated, and a zero one is in range but falsy, so the `parsed.port or
    # 443` at engine/worker/teacher/client.py silently dials 443 instead of the configured port.
    invalid_port = f"{PUBLIC_URL_ENV} must be a worker-reachable https URL with a valid port"
    # these messages name the env var and the shape rule only -- never the configured value, which
    # can embed credentials -- so they stay safe to surface to the submitter verbatim. every one is
    # plane_fault: the only production caller is resolve_public_url, reading the plane's own env.
    try:
        port = parsed.port
    except ValueError as error:
        raise TeacherBrokerConfigurationError(invalid_port, plane_fault=True) from error
    if port == 0:
        raise TeacherBrokerConfigurationError(invalid_port, plane_fault=True)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise TeacherBrokerConfigurationError(
            f"{PUBLIC_URL_ENV} must be a worker-reachable https URL without credentials, "
            "query parameters, or a fragment",
            plane_fault=True,
        )
    return url


def resolve_public_url() -> str:
    """Resolve this plane's worker-reachable origin from the environment.

    reads os.environ directly and does not fall back to the cli's FLASH_API_URL. the two are the
    same string on a plane addressed publicly at the name its cli dials, but a self-hosted plane
    reached over a tunnel, a vpn, or localhost has a client url no rented worker can resolve, and
    falling back would turn that into a failure after the gpu is allocated instead of at submit.

    validation constrains shape and port validity, not reachability, so an https://localhost origin
    or a public origin belonging to a different plane still passes here and is caught only when the
    worker fails to present its capability.
    """
    return validate_public_url(os.environ.get(PUBLIC_URL_ENV, ""))


def require_teacher_broker_configuration(
    spec: JobSpec,
    *,
    deadline_at: float | None = None,
    now: float | None = None,
) -> str:
    if spec.algorithm != "opd":
        raise TeacherBrokerConfigurationError(
            "teacher broker configuration is only valid for opd runs"
        )
    public_url = resolve_public_url()
    if not os.environ.get(PARASAIL_API_KEY_ENV, "").strip():
        raise TeacherBrokerConfigurationError(
            f"{PARASAIL_API_KEY_ENV} is required on the control plane for managed opd teachers",
            plane_fault=True,
        )
    teacher = resolve_teacher(spec.train.teacher_model)
    if teacher.alias not in {
        "kimi-k3",
        "glm-5.2",
        "qwen3.5-397b-a17b",
        "deepseek-v4-pro",
        "qwen3-vl-235b",
    }:
        raise TeacherBrokerConfigurationError(
            "the selected managed teacher is not supported by the Parasail broker"
        )
    capability_limits_for_spec(spec)
    if deadline_at is not None:
        current = time.time() if now is None else float(now)
        deadline = float(deadline_at)
        if not math.isfinite(current) or not math.isfinite(deadline) or deadline <= current:
            raise TeacherBrokerConfigurationError(
                "the run deadline leaves no valid teacher capability lifetime"
            )
        if deadline - current > MAX_CAPABILITY_LIFETIME_S:
            raise TeacherBrokerConfigurationError(
                "managed opd teacher capabilities are limited to a 24-hour run deadline"
            )
    return public_url


def preflight_validate_managed_teacher(spec: JobSpec) -> None:
    """Validate the managed-teacher gate at submit, before a run record exists.

    The same checks run again before allocation (that is what actually protects a paid worker), but
    running them here too means a `--dry-run` reports a plane misconfiguration or an unsupported
    teacher/shape instead of validating clean and then failing seconds after a real submit.

    Deadline-dependent policy is deliberately excluded: the run deadline is derived from the status
    record this runs ahead of. `capability_limits_for_spec` already rejects a `gpu.max_wall_seconds`
    over the capability lifetime, so the only gap is the queue allowance, which is not knowable yet.
    """
    if spec.algorithm != "opd":
        return
    require_teacher_broker_configuration(spec)


def capability_limits_for_spec(spec: JobSpec) -> dict[str, int]:
    if spec.algorithm != "opd":
        raise ValueError("teacher capabilities require an opd spec")
    wall_seconds = spec.gpu.max_wall_seconds
    if (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or not math.isfinite(wall_seconds)
        or wall_seconds <= 0
    ):
        raise ValueError("opd gpu.max_wall_seconds must be a finite positive duration")
    if wall_seconds > MAX_CAPABILITY_LIFETIME_S:
        raise ValueError(
            "managed opd teacher capabilities require gpu.max_wall_seconds of 24 hours or less"
        )
    from flash.cost.spec import spec_steps
    from flash.engine.plan.vram import opd_completion_len

    steps = int(spec_steps(spec))
    prompts_per_step = int(spec.train.prompts_per_step or RECIPE.opd.prompts_per_step)
    group_size = int(spec.train.group_size or RECIPE.opd.group_size)
    base_items = steps * max(1, prompts_per_step) * max(1, group_size)
    multi_turn, max_turns = configured_opd_turn_limit(spec.environment)
    request_multiplier = opd_teacher_request_multiplier(
        multi_turn=multi_turn,
        max_turns=max_turns,
    )
    score_items = base_items * request_multiplier
    requests = score_items
    completion_tokens = opd_completion_len(spec.train.max_completion_tokens or 0, spec.thinking)
    context_tokens = int(
        spec.train.max_context_tokens or (RECIPE.opd.max_prompt_len + completion_tokens)
    )
    request_tokens = min(MAX_REQUEST_TOKENS, max(4096, context_tokens * 4))
    total_tokens = score_items * request_tokens
    if score_items <= 0 or score_items > MAX_TOTAL_SCORE_ITEMS:
        raise ValueError(
            f"planned opd shape exceeds the broker score-item limit of {MAX_TOTAL_SCORE_ITEMS}"
        )
    if requests > MAX_TOTAL_REQUESTS:
        raise ValueError(
            f"planned opd shape exceeds the broker request limit of {MAX_TOTAL_REQUESTS}"
        )
    if total_tokens > MAX_TOTAL_TOKENS:
        raise ValueError(f"planned opd shape exceeds the broker token limit of {MAX_TOTAL_TOKENS}")
    return {
        "max_requests": requests,
        "max_score_items": score_items,
        "max_request_bytes": MAX_REQUEST_BODY_BYTES,
        "max_response_bytes": MAX_RESPONSE_BODY_BYTES,
        "max_concurrency": OPD_TEACHER_SCORING_CONCURRENCY,
        "max_upstream_attempts": MAX_UPSTREAM_ATTEMPTS,
        "max_request_tokens": request_tokens,
        "max_total_tokens": total_tokens,
    }


def issue_teacher_capability(
    spec: JobSpec,
    *,
    attempt: int,
    deadline_at: float,
    now: float | None = None,
) -> tuple[str, str]:
    issued_at = time.time() if now is None else float(now)
    public_url = require_teacher_broker_configuration(
        spec,
        deadline_at=deadline_at,
        now=issued_at,
    )
    expiry = min(float(deadline_at), issued_at + MAX_CAPABILITY_LIFETIME_S)
    teacher = resolve_teacher(spec.train.teacher_model)
    token = db.issue_teacher_capability(
        run_id=spec.run_id,
        attempt=int(attempt),
        teacher_alias=teacher.alias,
        provider=PARASAIL_PROVIDER,
        model=teacher.model_id,
        scoring_mode=PARASAIL_SCORING_MODE,
        expires_at=expiry,
        limits=capability_limits_for_spec(spec),
        now=issued_at,
    )
    return public_url, token


@contextlib.contextmanager
def teacher_attempt_transport(
    spec: JobSpec,
    *,
    attempt: int,
    deadline_at: float,
):
    if spec.algorithm != "opd":
        yield {}
        return
    public_url, capability = issue_teacher_capability(
        spec,
        attempt=attempt,
        deadline_at=deadline_at,
    )
    try:
        yield {
            PUBLIC_URL_ENV: public_url,
            TEACHER_CAPABILITY_ENV: capability,
        }
    finally:
        db.revoke_teacher_capability(capability)


_reject_duplicate_keys = reject_duplicate_keys(
    lambda _key: TeacherBrokerError("duplicate_json_key", status_code=400)
)


def _require_current_attempt(capability: dict[str, Any]) -> None:
    from flash.runner.lifecycle.attempts import latest_reserved_attempt
    from flash.runner.lifecycle.state import _internal_spec_from_status
    from flash.runner.lifecycle.status import get_status

    try:
        status = get_status(capability["run_id"])
    except FileNotFoundError as exc:
        raise TeacherBrokerError("run_not_found", status_code=401) from exc
    if status.state != "running":
        raise TeacherBrokerError("run_not_active", status_code=401)
    if latest_reserved_attempt(capability["run_id"]) != capability["attempt"]:
        raise TeacherBrokerError("attempt_replaced", status_code=401)
    try:
        spec = _internal_spec_from_status(status)
    except (TypeError, ValueError) as exc:
        raise TeacherBrokerError("run_scope_invalid", status_code=401) from exc
    if spec.algorithm != "opd" or spec.run_id != capability["run_id"]:
        raise TeacherBrokerError("run_scope_invalid", status_code=401)
    teacher = resolve_teacher(spec.train.teacher_model)
    if (
        teacher.alias != capability["teacher_alias"]
        or teacher.model_id != capability["model"]
        or capability["provider"] != PARASAIL_PROVIDER
        or capability["scoring_mode"] != PARASAIL_SCORING_MODE
    ):
        raise TeacherBrokerError("capability_scope_mismatch", status_code=403)


def _map_ledger_error(error: db.TeacherLedgerError, request_id: str) -> TeacherBrokerError:
    unauthorized = {
        "invalid_capability",
        "revoked_capability",
        "expired_capability",
        "capability_fenced",
    }
    conflict = {
        "request_body_changed",
        "request_in_progress",
        "replay_unavailable",
        "outcome_unknown",
        "provider_rejected",
        "provider_contract_error",
    }
    if error.code in unauthorized:
        status = 401
    elif error.code in conflict:
        status = 409
    elif error.code == "broker_busy":
        status = 503
    else:
        status = 429
    return TeacherBrokerError(
        error.code,
        status_code=status,
        retryable=error.retryable,
        request_id=request_id,
        provider_status=error.provider_status,
    )


def authenticate_teacher_capability(
    *, capability_token: str, request_id: str
) -> tuple[str, str, dict[str, Any]]:
    request_id = validate_request_id(request_id)
    capability_token = validate_capability_token(capability_token)
    try:
        capability = db.active_teacher_capability(capability_token)
    except db.TeacherLedgerError as exc:
        raise _map_ledger_error(exc, request_id) from exc
    except sqlite3.OperationalError as exc:
        raise TeacherBrokerError(
            "broker_busy",
            status_code=503,
            retryable=True,
            request_id=request_id,
        ) from exc
    _require_current_attempt(capability)
    return request_id, capability_token, capability


def _provider_timeout(capability: dict[str, Any]) -> float:
    remaining = float(capability["expires_at"]) - time.time()
    if remaining <= 0:
        raise TeacherBrokerError("expired_capability", status_code=401)
    return min(MAX_PROVIDER_TIMEOUT_S, remaining)


def _provider_post_path(path: str, body: bytes, api_key: str, timeout: float) -> tuple[int, bytes]:
    parsed = urllib.parse.urlsplit(PARASAIL_ORIGIN)
    if parsed.hostname is None:
        raise RuntimeError("Parasail origin is missing a hostname")
    connection = http.client.HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=timeout)
    try:
        connection.request(
            "POST",
            path,
            body=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        response = connection.getresponse()
        limit = (
            MAX_RESPONSE_BODY_BYTES
            if 200 <= response.status < 300
            else MAX_PROVIDER_ERROR_BODY_BYTES
        )
        payload = response.read(limit + 1)
        if len(payload) > limit:
            if 200 <= response.status < 300:
                raise TeacherBrokerError("provider_response_too_large", status_code=502)
            payload = b""
        return int(response.status), payload
    finally:
        connection.close()


def _provider_post(body: bytes, api_key: str, timeout: float) -> tuple[int, bytes]:
    return _provider_post_path(PARASAIL_COMPLETIONS_PATH, body, api_key, timeout)


def _provider_chat_post(body: bytes, api_key: str, timeout: float) -> tuple[int, bytes]:
    return _provider_post_path(PARASAIL_CHAT_COMPLETIONS_PATH, body, api_key, timeout)


def _validated_provider_response(
    raw: bytes, *, score_items: int, capability: dict[str, Any]
) -> ProviderResponse:
    if len(raw) > capability["max_response_bytes"]:
        raise TeacherBrokerError("provider_response_too_large", status_code=502)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except TeacherBrokerError as exc:
        raise TeacherBrokerError("provider_contract_error", status_code=502) from exc
    except (TypeError, ValueError) as exc:
        raise TeacherBrokerError("provider_contract_error", status_code=502) from exc
    if not isinstance(value, dict) or score_items != 1:
        raise TeacherBrokerError("provider_contract_error", status_code=502)
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise TeacherBrokerError("provider_contract_error", status_code=502)
    choice = choices[0]
    prompt_ids = choice.get("prompt_token_ids")
    output_ids = choice.get("token_ids")
    if (
        not isinstance(prompt_ids, list)
        or not prompt_ids
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in prompt_ids
        )
        or not isinstance(output_ids, list)
        or len(output_ids) != 1
        or isinstance(output_ids[0], bool)
        or not isinstance(output_ids[0], int)
        or output_ids[0] < 0
    ):
        raise TeacherBrokerError("provider_contract_error", status_code=502)
    prompt_logprobs = choice.get("prompt_logprobs")
    positional = choice.get("logprobs")
    positional_scores = positional.get("token_logprobs") if isinstance(positional, dict) else None
    if not (
        (isinstance(prompt_logprobs, list) and len(prompt_logprobs) == len(prompt_ids))
        or (isinstance(positional_scores, list) and len(positional_scores) == len(prompt_ids) + 1)
    ):
        raise TeacherBrokerError("provider_contract_error", status_code=502)
    usage = value.get("usage")
    if not isinstance(usage, dict):
        raise TeacherBrokerError("provider_contract_error", status_code=502)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        count = usage.get(key)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise TeacherBrokerError("provider_contract_error", status_code=502)
    input_tokens = int(usage["prompt_tokens"])
    output_tokens = int(usage["completion_tokens"])
    total_tokens = int(usage["total_tokens"])
    if (
        input_tokens != len(prompt_ids)
        or output_tokens != 1
        or total_tokens != input_tokens + output_tokens
    ):
        raise TeacherBrokerError("provider_contract_error", status_code=502)
    if total_tokens > capability["max_request_tokens"]:
        raise TeacherBrokerError("provider_token_limit_exceeded", status_code=502)
    return ProviderResponse(
        body=value,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _validated_chat_provider_response(
    raw: bytes, *, score_items: int, capability: dict[str, Any]
) -> ProviderResponse:
    if len(raw) > capability["max_response_bytes"]:
        raise TeacherBrokerError("provider_response_too_large", status_code=502)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except TeacherBrokerError as exc:
        raise TeacherBrokerError("provider_contract_error", status_code=502) from exc
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise TeacherBrokerError("provider_contract_error", status_code=502) from exc
    if not isinstance(value, dict) or score_items != 1:
        raise TeacherBrokerError("provider_contract_error", status_code=502)
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise TeacherBrokerError("provider_contract_error", status_code=502)
    prompt_ids = value.get("prompt_token_ids")
    prompt_logprobs = value.get("prompt_logprobs")
    output_ids = choices[0].get("token_ids")
    if (
        not isinstance(prompt_ids, list)
        or not prompt_ids
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in prompt_ids
        )
        or not isinstance(prompt_logprobs, list)
        or len(prompt_logprobs) != len(prompt_ids)
        or not isinstance(output_ids, list)
        or len(output_ids) != 1
        or isinstance(output_ids[0], bool)
        or not isinstance(output_ids[0], int)
        or output_ids[0] < 0
    ):
        raise TeacherBrokerError("provider_contract_error", status_code=502)
    usage = value.get("usage")
    if not isinstance(usage, dict):
        raise TeacherBrokerError("provider_contract_error", status_code=502)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        count = usage.get(key)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise TeacherBrokerError("provider_contract_error", status_code=502)
    input_tokens = int(usage["prompt_tokens"])
    output_tokens = int(usage["completion_tokens"])
    total_tokens = int(usage["total_tokens"])
    if (
        input_tokens != len(prompt_ids)
        or output_tokens != 1
        or total_tokens != input_tokens + output_tokens
    ):
        raise TeacherBrokerError("provider_contract_error", status_code=502)
    if total_tokens > capability["max_request_tokens"]:
        raise TeacherBrokerError("provider_token_limit_exceeded", status_code=502)
    return ProviderResponse(
        body=value,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _reserve_teacher_request(
    *, capability_token: str, request_id: str, request, capability: dict
) -> dict:
    """Reserve the ledger row for this request, mapping ledger faults onto broker errors."""
    fingerprint = request_fingerprint(capability_token, request.canonical_body)
    try:
        return db.reserve_teacher_request(
            token=capability_token,
            request_id=request_id,
            request_fingerprint=fingerprint,
            request_bytes=len(request.canonical_body),
            score_items=request.score_items,
            expected_run_id=capability["run_id"],
            expected_attempt=capability["attempt"],
        )
    except db.TeacherLedgerError as exc:
        raise _map_ledger_error(exc, request_id) from exc
    except sqlite3.OperationalError as exc:
        raise TeacherBrokerError(
            "broker_busy",
            status_code=503,
            retryable=True,
            request_id=request_id,
        ) from exc


def _prepare_teacher_dispatch(
    *, capability_id: int, request_id: str, capability: dict, request
) -> tuple[str, bytes]:
    """Resolve the provider credential, encode the upstream body, and fence the dispatch.

    Returns ``(api_key, encoded_body)``. Every failure here readmits the request for retry before
    anything is dispatched, so no upstream call can have happened.
    """
    api_key = os.environ.get(PARASAIL_API_KEY_ENV, "").strip()
    if not api_key:
        with contextlib.suppress(Exception):
            db.retry_teacher_request_before_dispatch(
                capability_id,
                request_id,
                error_class="provider_credential_unavailable",
            )
        raise TeacherBrokerError(
            "provider_unavailable",
            status_code=503,
            retryable=True,
            request_id=request_id,
        )
    upstream_body = dict(request.body)
    upstream_body["model"] = capability["model"]
    encoded = _canonical_json(upstream_body)
    try:
        db.mark_teacher_request_started(capability_id, request_id)
    except db.TeacherLedgerError as exc:
        with contextlib.suppress(Exception):
            db.retry_teacher_request_before_dispatch(
                capability_id,
                request_id,
                error_class="dispatch_fenced",
            )
        raise _map_ledger_error(exc, request_id) from exc
    except sqlite3.OperationalError as exc:
        with contextlib.suppress(Exception):
            db.retry_teacher_request_before_dispatch(
                capability_id,
                request_id,
                error_class="ledger_contention_before_dispatch",
            )
        raise TeacherBrokerError(
            "broker_busy",
            status_code=503,
            retryable=True,
            request_id=request_id,
        ) from exc
    return api_key, encoded


def _dispatch_to_teacher_provider(
    *,
    capability_id: int,
    request_id: str,
    capability: dict,
    api_key: str,
    encoded: bytes,
    chat: bool,
) -> tuple[int, bytes]:
    """Post to the provider and return ``(status, response_body)`` for a 2xx answer only.

    ``chat`` selects the upstream endpoint: image scoring has to go to chat/completions because
    only that route accepts image parts, while text scoring stays on the completions route.
    """
    provider_post = _provider_chat_post if chat else _provider_post
    try:
        status, response_body = provider_post(encoded, api_key, _provider_timeout(capability))
    except TeacherBrokerError as exc:
        with contextlib.suppress(Exception):
            db.complete_teacher_request(
                capability_id,
                request_id,
                state="provider_contract_error",
                error_class=exc.code,
            )
        raise TeacherBrokerError(
            exc.code,
            status_code=exc.status_code,
            request_id=request_id,
        ) from exc
    except (OSError, http.client.HTTPException) as exc:
        # the provider may have accepted work before the connection failed. without an upstream
        # idempotency key, redispatch could execute and bill the same logical request twice, so the
        # ambiguous outcome is terminal even though flash cannot settle provider usage internally.
        with contextlib.suppress(Exception):
            db.complete_teacher_request(
                capability_id,
                request_id,
                state="outcome_unknown",
                error_class="provider_transport_ambiguous",
            )
        raise TeacherBrokerError(
            "outcome_unknown",
            status_code=502,
            request_id=request_id,
        ) from exc
    if status < 200 or status >= 300:
        # only a conventional 429 proves the provider rejected the request before execution. a 408 or
        # 5xx can arrive after work started, so those outcomes are terminal until the provider offers
        # an idempotency contract that makes redispatch safe.
        rate_limited = status == 429
        ambiguous = status == 408 or 500 <= status <= 599
        state = "outcome_unknown" if ambiguous else "provider_rejected"
        with contextlib.suppress(Exception):
            db.complete_teacher_request(
                capability_id,
                request_id,
                state=state,
                provider_status=status,
                error_class="transient" if rate_limited else "permanent",
            )
        raise TeacherBrokerError(
            state,
            status_code=502,
            retryable=rate_limited,
            request_id=request_id,
            provider_status=status,
        )
    return status, response_body


def _settle_teacher_response(
    *,
    capability_id: int,
    request_id: str,
    capability: dict,
    request,
    status: int,
    response_body: bytes,
    chat: bool,
) -> dict[str, Any]:
    """Validate the provider answer against the capability, then close the ledger row.

    ``chat`` picks the validator: the two routes return different shapes, and the chat one is the
    only one that can carry an image-scored answer.
    """
    response_validator = _validated_chat_provider_response if chat else _validated_provider_response
    try:
        response = response_validator(
            response_body,
            score_items=request.score_items,
            capability=capability,
        )
    except TeacherBrokerError as exc:
        with contextlib.suppress(Exception):
            db.complete_teacher_request(
                capability_id,
                request_id,
                state="provider_contract_error",
                provider_status=status,
                error_class=exc.code,
            )
        raise TeacherBrokerError(
            exc.code,
            status_code=exc.status_code,
            request_id=request_id,
        ) from exc
    try:
        db.complete_teacher_request(
            capability_id,
            request_id,
            state="succeeded",
            provider_status=status,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            response_body=_canonical_json(response.body),
        )
    except Exception as exc:
        # a validated provider response is already in hand. if ledger settlement did not commit, the
        # result cannot be replayed, but redispatch would be pure duplicate provider work. if it did
        # commit before raising, the normal succeeded replay path remains available.
        with contextlib.suppress(Exception):
            db.complete_teacher_request(
                capability_id,
                request_id,
                state="outcome_unknown",
                error_class="ledger_completion_failed",
            )
        raise TeacherBrokerError(
            "outcome_unknown",
            status_code=503,
            retryable=True,
            request_id=request_id,
        ) from exc
    return response.body


def _complete_teacher_request(
    *,
    capability_token: str,
    request_id: str,
    raw_body: bytes | bytearray,
    chat: bool,
) -> dict[str, Any]:
    """Run one teacher request end to end on either upstream route.

    ``chat`` is threaded rather than duplicated into two near-identical bodies: the routes differ
    only in which request validator, provider endpoint, and response validator apply, and every
    ledger and fencing step between them must stay identical or the two paths drift apart.
    """
    request_id, capability_token, capability = authenticate_teacher_capability(
        capability_token=capability_token,
        request_id=request_id,
    )
    body = parse_strict_json(raw_body)
    request = (
        validate_chat_completion_request(body, capability)
        if chat
        else validate_completion_request(body, capability)
    )
    reservation = _reserve_teacher_request(
        capability_token=capability_token,
        request_id=request_id,
        request=request,
        capability=capability,
    )
    capability = reservation["capability"]
    replay_body = reservation.get("response_body")
    response_validator = _validated_chat_provider_response if chat else _validated_provider_response
    if isinstance(replay_body, bytes):
        return response_validator(
            replay_body,
            score_items=request.score_items,
            capability=capability,
        ).body
    capability_id = int(capability["id"])
    api_key, encoded = _prepare_teacher_dispatch(
        capability_id=capability_id,
        request_id=request_id,
        capability=capability,
        request=request,
    )
    status, response_body = _dispatch_to_teacher_provider(
        capability_id=capability_id,
        request_id=request_id,
        capability=capability,
        api_key=api_key,
        encoded=encoded,
        chat=chat,
    )
    return _settle_teacher_response(
        capability_id=capability_id,
        request_id=request_id,
        capability=capability,
        request=request,
        status=status,
        response_body=response_body,
        chat=chat,
    )


def complete_teacher_request(
    *,
    capability_token: str,
    request_id: str,
    raw_body: bytes | bytearray,
) -> dict[str, Any]:
    return _complete_teacher_request(
        capability_token=capability_token,
        request_id=request_id,
        raw_body=raw_body,
        chat=False,
    )


def complete_teacher_chat_request(
    *,
    capability_token: str,
    request_id: str,
    raw_body: bytes | bytearray,
) -> dict[str, Any]:
    return _complete_teacher_request(
        capability_token=capability_token,
        request_id=request_id,
        raw_body=raw_body,
        chat=True,
    )
