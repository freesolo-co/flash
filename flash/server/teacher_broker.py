"""Closed managed-teacher broker for Parasail completion scoring."""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import http.client
import json
import math
import os
import re
import sqlite3
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any

from flash.engine.recipe import RECIPE, resolve_teacher
from flash.opd_limits import (
    OPD_TEACHER_SCORING_CONCURRENCY,
    configured_opd_turn_limit,
    opd_teacher_request_multiplier,
)
from flash.server import db
from flash.spec import CONTROL_PANEL_URL_ENV, TEACHER_CAPABILITY_ENV, JobSpec

PARASAIL_PROVIDER = "parasail"
PARASAIL_ORIGIN = "https://api.parasail.io"
PARASAIL_COMPLETIONS_PATH = "/v1/completions"
PARASAIL_SCORING_MODE = "supplied_token_completion_v1"
PARASAIL_API_KEY_ENV = "PARASAIL_API_KEY"

MAX_REQUEST_BODY_BYTES = 48 * 1024 * 1024
MAX_RESPONSE_BODY_BYTES = 32 * 1024 * 1024
MAX_PROVIDER_ERROR_BODY_BYTES = 64 * 1024
MAX_UPSTREAM_ATTEMPTS = 1
MAX_REQUEST_TOKENS = 131_072
MAX_TOTAL_SCORE_ITEMS = 1_000_000
MAX_TOTAL_REQUESTS = 1_000_000
MAX_TOTAL_TOKENS = 2_000_000_000
MAX_CAPABILITY_LIFETIME_S = 24 * 60 * 60
MAX_PROVIDER_TIMEOUT_S = 90.0
REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._~-]{16,128}\Z")
CAPABILITY_PATTERN = re.compile(r"[A-Za-z0-9_-]{40,128}\Z")


class TeacherBrokerError(RuntimeError):
    def __init__(
        self,
        code: str,
        *,
        status_code: int,
        retryable: bool = False,
        request_id: str | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable
        self.request_id = request_id

    def payload(self) -> dict[str, Any]:
        error: dict[str, Any] = {
            "code": self.code,
            "classification": "transient" if self.retryable else "permanent",
        }
        if self.request_id is not None:
            error["request_id"] = self.request_id
        return {"error": error}


@dataclass(frozen=True)
class ValidatedCompletionRequest:
    body: dict[str, Any]
    canonical_body: bytes
    score_items: int


@dataclass(frozen=True)
class ProviderResponse:
    body: dict[str, Any]
    input_tokens: int
    output_tokens: int


def validate_control_panel_url(value: str) -> str:
    url = str(value or "").strip().rstrip("/")
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(
            f"{CONTROL_PANEL_URL_ENV} must be a worker-reachable https URL without credentials, "
            "query parameters, or a fragment"
        )
    return url


def require_teacher_broker_configuration(
    spec: JobSpec,
    *,
    deadline_at: float | None = None,
    now: float | None = None,
) -> str:
    if spec.algorithm != "opd":
        raise RuntimeError("teacher broker configuration is only valid for opd runs")
    control_panel_url = validate_control_panel_url(os.environ.get(CONTROL_PANEL_URL_ENV, ""))
    if not os.environ.get(PARASAIL_API_KEY_ENV, "").strip():
        raise RuntimeError(
            f"{PARASAIL_API_KEY_ENV} is required on the control plane for managed opd teachers"
        )
    teacher = resolve_teacher(spec.train.teacher_model)
    if teacher.alias not in {"kimi-k3", "glm-5.2", "qwen3.5-397b-a17b", "deepseek-v4-pro"}:
        raise RuntimeError("the selected managed teacher is not supported by the Parasail broker")
    capability_limits_for_spec(spec)
    if deadline_at is not None:
        current = time.time() if now is None else float(now)
        deadline = float(deadline_at)
        if not math.isfinite(current) or not math.isfinite(deadline) or deadline <= current:
            raise RuntimeError("the run deadline leaves no valid teacher capability lifetime")
        if deadline - current > MAX_CAPABILITY_LIFETIME_S:
            raise RuntimeError(
                "managed opd teacher capabilities are limited to a 24-hour run deadline"
            )
    return control_panel_url


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
    from flash.engine.vram import opd_completion_len

    steps = int(spec_steps(spec))
    prompts_per_step = int(spec.train.batch_size or RECIPE.opd.prompts_per_step)
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
    control_panel_url = require_teacher_broker_configuration(
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
    return control_panel_url, token


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
    control_panel_url, capability = issue_teacher_capability(
        spec,
        attempt=attempt,
        deadline_at=deadline_at,
    )
    try:
        yield {
            CONTROL_PANEL_URL_ENV: control_panel_url,
            TEACHER_CAPABILITY_ENV: capability,
        }
    finally:
        db.revoke_teacher_capability(capability)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise TeacherBrokerError("duplicate_json_key", status_code=400)
        output[key] = value
    return output


def _reject_nonfinite(_value: str) -> None:
    raise TeacherBrokerError("non_finite_number", status_code=400)


def parse_strict_json(raw: bytes | bytearray) -> dict[str, Any]:
    if len(raw) > MAX_REQUEST_BODY_BYTES:
        raise TeacherBrokerError("request_too_large", status_code=413)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except TeacherBrokerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise TeacherBrokerError("invalid_json", status_code=400) from exc
    if not isinstance(value, dict):
        raise TeacherBrokerError("request_must_be_object", status_code=400)
    return value


def _canonical_json(value: dict[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TeacherBrokerError("invalid_request_value", status_code=400) from exc


def validate_request_id(value: str) -> str:
    request_id = str(value or "").strip()
    if not REQUEST_ID_PATTERN.fullmatch(request_id):
        raise TeacherBrokerError("invalid_request_id", status_code=400)
    return request_id


def validate_capability_token(value: str) -> str:
    capability = str(value or "").strip()
    if not CAPABILITY_PATTERN.fullmatch(capability):
        raise TeacherBrokerError("invalid_capability", status_code=401)
    return capability


def validate_completion_request(
    body: dict[str, Any], capability: dict[str, Any]
) -> ValidatedCompletionRequest:
    required = {
        "model",
        "prompt",
        "max_tokens",
        "echo",
        "logprobs",
        "prompt_logprobs",
        "return_token_ids",
        "temperature",
        "top_p",
        "seed",
    }
    if set(body) - required:
        raise TeacherBrokerError("extra_request_fields", status_code=400)
    if set(body) != required:
        raise TeacherBrokerError("missing_request_fields", status_code=400)
    if body["model"] != capability["model"]:
        raise TeacherBrokerError("model_scope_mismatch", status_code=403)
    if (
        not isinstance(body["prompt"], str)
        or not body["prompt"]
        or isinstance(body["max_tokens"], bool)
        or body["max_tokens"] != 1
        or body["echo"] is not True
        or isinstance(body["logprobs"], bool)
        or body["logprobs"] != 1
        or isinstance(body["prompt_logprobs"], bool)
        or body["prompt_logprobs"] != 1
        or body["return_token_ids"] is not True
        or isinstance(body["temperature"], bool)
        or body["temperature"] != 0
        or isinstance(body["top_p"], bool)
        or body["top_p"] != 1
        or isinstance(body["seed"], bool)
        or body["seed"] != 0
    ):
        raise TeacherBrokerError("unsupported_scoring_parameters", status_code=400)
    canonical = _canonical_json(body)
    if len(canonical) > capability["max_request_bytes"]:
        raise TeacherBrokerError("request_too_large", status_code=413)
    return ValidatedCompletionRequest(
        body=dict(body),
        canonical_body=canonical,
        score_items=1,
    )


def request_fingerprint(capability: str, canonical_body: bytes) -> str:
    return hmac.new(capability.encode("utf-8"), canonical_body, hashlib.sha256).hexdigest()


def _require_current_attempt(capability: dict[str, Any]) -> None:
    from flash.runner import _internal_spec_from_status, _latest_reserved_attempt, get_status

    try:
        status = get_status(capability["run_id"])
    except FileNotFoundError as exc:
        raise TeacherBrokerError("run_not_found", status_code=401) from exc
    if status.state != "running":
        raise TeacherBrokerError("run_not_active", status_code=401)
    if _latest_reserved_attempt(capability["run_id"]) != capability["attempt"]:
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


def _provider_post(body: bytes, api_key: str, timeout: float) -> tuple[int, bytes]:
    parsed = urllib.parse.urlsplit(PARASAIL_ORIGIN)
    if parsed.hostname is None:
        raise RuntimeError("Parasail origin is missing a hostname")
    connection = http.client.HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=timeout)
    try:
        connection.request(
            "POST",
            PARASAIL_COMPLETIONS_PATH,
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
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
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


def complete_teacher_request(
    *,
    capability_token: str,
    request_id: str,
    raw_body: bytes | bytearray,
) -> dict[str, Any]:
    request_id, capability_token, capability = authenticate_teacher_capability(
        capability_token=capability_token,
        request_id=request_id,
    )
    request = validate_completion_request(parse_strict_json(raw_body), capability)
    fingerprint = request_fingerprint(capability_token, request.canonical_body)
    try:
        reservation = db.reserve_teacher_request(
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
    capability = reservation["capability"]
    replay_body = reservation.get("response_body")
    if isinstance(replay_body, bytes):
        return _validated_provider_response(
            replay_body,
            score_items=request.score_items,
            capability=capability,
        ).body
    capability_id = int(capability["id"])
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
    try:
        status, response_body = _provider_post(encoded, api_key, _provider_timeout(capability))
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
    except (TimeoutError, OSError, http.client.HTTPException) as exc:
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
        # every provider reject is terminal for this logical request, including a 429. that is a
        # property of the ledger, not a classification choice here: MAX_UPSTREAM_ATTEMPTS is 1, so
        # db.mark_teacher_request_started refuses a second upstream call with
        # upstream_attempt_quota_exhausted, and db.reserve_teacher_request re-admits only rows in
        # state 'retryable' -- a completed 'provider_rejected' row cannot be reopened whatever its
        # error_class says. marking a 429 retryable here would only buy the worker a round trip that
        # comes back permanent. recovering shed load needs an attempt budget above 1 and a
        # re-admission path for a 'started' row, which is broker lifecycle work, not this change.
        with contextlib.suppress(Exception):
            db.complete_teacher_request(
                capability_id,
                request_id,
                state="provider_rejected",
                provider_status=status,
                error_class="permanent",
            )
        raise TeacherBrokerError(
            "provider_rejected",
            status_code=502,
            request_id=request_id,
        )
    try:
        response = _validated_provider_response(
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
            request_id=request_id,
        ) from exc
    return response.body
