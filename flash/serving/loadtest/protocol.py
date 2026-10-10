"""public http and strict sse protocol for hosted inference load tests."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from flash.serving.loadtest.schedule import Clock
from flash.serving.loadtest.schema import (
    AdapterTarget,
    DeploymentExpectation,
    HealthSnapshot,
    RequestProfile,
    Target,
)

_ALLOWED_RESPONSE_HEADERS = {
    "content-type",
    "retry-after",
    "x-freesolo-adapter-revision",
    "x-freesolo-checkpoint",
    "x-freesolo-hf-revision",
}
_MAX_ERROR_CHARS = 512
# a machine error code is a bounded slug. anything else is server prose that may echo the
# request, so it is never persisted verbatim.
_ERROR_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


class ProtocolError(RuntimeError):
    pass


@dataclass
class RequestObservation:
    """what one dispatched request was observed to do.

    deliberately carries no request id: the schedule already owns that identity, and duplicating
    it here would leave synthesized observations (an interrupt, an admission miss) with no id to
    supply. the event writer stamps the id from the scheduled request instead.
    """

    outcome: str = "protocol_error"
    http_status: int | None = None
    error_class: str | None = None
    error_detail: str | None = None
    response_headers: dict[str, str] = field(default_factory=dict)
    dispatch_ns: int | None = None
    headers_ns: int | None = None
    first_generated_ns: int | None = None
    first_visible_ns: int | None = None
    completed_ns: int | None = None
    finish_reasons: list[str] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    done_count: int = 0
    retry_count: int = 0


async def get_health(
    client: httpx.AsyncClient,
    endpoint: str,
    expected: DeploymentExpectation,
    required_capabilities: list[str],
) -> HealthSnapshot:
    response = await client.get(f"{endpoint}/healthz")
    if response.status_code != 200:
        raise ProtocolError(f"healthz returned http {response.status_code}")
    try:
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("health body is not an object")
        health = HealthSnapshot.model_validate(
            {name: body.get(name) for name in HealthSnapshot.model_fields}
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError("healthz returned an invalid identity response") from exc
    if not health.ok:
        raise ProtocolError("healthz reports the deployment is not ready")
    if health.accounting_ok is False:
        raise ProtocolError("healthz reports accounting is not ready for attributed inference")
    if health.deployment_sha != expected.sha:
        raise ProtocolError("healthz deployment sha does not match the scenario")
    if health.deployment_id != expected.deployment_id:
        raise ProtocolError("healthz deployment id does not match the scenario")
    missing = sorted(set(required_capabilities) - set(health.capabilities))
    if missing:
        raise ProtocolError(f"healthz is missing required capabilities: {', '.join(missing)}")
    return health


def resolve_discovered_models(
    base_models: list[str], include: list[str], exclude: list[str], require: list[str]
) -> list[str]:
    available = set(base_models)
    missing = sorted(set(require) - available)
    if missing:
        raise ProtocolError(f"required base models are unavailable: {', '.join(missing)}")
    selected = available if not include else available & set(include)
    selected -= set(exclude)
    if not selected:
        raise ProtocolError("model discovery resolved no base models")
    return sorted(selected)


async def stream_chat(
    client: httpx.AsyncClient,
    endpoint: str,
    credential: str,
    target: Target,
    profile: RequestProfile,
    clock: Clock,
) -> RequestObservation:
    observation = RequestObservation(dispatch_ns=clock.monotonic_ns())
    headers = {"Authorization": f"Bearer {credential}"}
    if isinstance(target, AdapterTarget):
        headers["X-Freesolo-Expected-Checkpoint"] = target.checkpoint
    payload = {
        "model": target.model,
        "messages": [message.model_dump(mode="json") for message in profile.messages],
        "max_tokens": profile.max_tokens,
        "temperature": profile.temperature,
        "top_p": profile.top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    try:
        async with client.stream(
            "POST",
            f"{endpoint}/v1/chat/completions",
            headers=headers,
            json=payload,
        ) as response:
            observation.headers_ns = clock.monotonic_ns()
            observation.http_status = response.status_code
            observation.response_headers = _allowed_headers(response.headers)
            if response.status_code != 200:
                await _classify_http_error(response, observation)
                return observation
            _validate_provenance(response.headers, target)
            await _consume_sse(response, observation, clock)
    except ProtocolError as exc:
        observation.error_class = "protocol_error"
        observation.error_detail = _sanitize(str(exc))
    except httpx.TimeoutException as exc:
        observation.error_class = "client_timeout"
        observation.error_detail = _sanitize(type(exc).__name__)
    except httpx.HTTPError as exc:
        observation.error_class = "client_transport_error"
        observation.error_detail = _sanitize(type(exc).__name__)
    finally:
        observation.completed_ns = observation.completed_ns or clock.monotonic_ns()
    if observation.error_class is None:
        observation.outcome = "success"
    elif observation.error_class in {
        "exact_capacity_503",
        "other_503",
        "http_429",
        "http_error",
    }:
        observation.outcome = "http_error"
    elif observation.error_class.startswith("client_"):
        observation.outcome = "client_error"
    return observation


async def _consume_sse(
    response: httpx.Response, observation: RequestObservation, clock: Clock
) -> None:
    done = False
    saw_data = False
    async for line in response.aiter_lines():
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            raise ProtocolError("sse contained a non-data field")
        saw_data = True
        payload = line[5:].strip()
        if done:
            raise ProtocolError("sse contained data after [done]")
        if payload == "[DONE]":
            observation.done_count += 1
            done = True
            continue
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProtocolError("sse data payload was not valid json") from exc
        if not isinstance(value, dict):
            raise ProtocolError("sse data payload must be a json object")
        _observe_chunk(value, observation, clock.monotonic_ns())
    if not saw_data:
        raise ProtocolError("sse response contained no data")
    if observation.done_count != 1:
        raise ProtocolError("sse response must contain exactly one [done]")
    if not observation.finish_reasons:
        raise ProtocolError("sse response omitted terminal finish reasons")


def _observe_chunk(value: dict[str, Any], observation: RequestObservation, now_ns: int) -> None:
    choices = value.get("choices")
    if choices is not None and not isinstance(choices, list):
        raise ProtocolError("sse choices must be a list")
    for choice in choices or []:
        if not isinstance(choice, dict):
            raise ProtocolError("sse choice must be an object")
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            raise ProtocolError("sse delta must be an object")
        # both checks test for generated text, not for key presence: an opening chunk carrying
        # only a role with an empty content string would otherwise stamp ttft at header time and
        # report a first token the server had not produced.
        if (delta.get("content") or delta.get("reasoning_content")) and (
            observation.first_generated_ns is None
        ):
            observation.first_generated_ns = now_ns
        if delta.get("content") and observation.first_visible_ns is None:
            observation.first_visible_ns = now_ns
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            if not isinstance(finish_reason, str) or not finish_reason:
                raise ProtocolError("terminal finish reason must be a nonempty string")
            observation.finish_reasons.append(finish_reason)
    usage = value.get("usage")
    if usage is not None:
        if not isinstance(usage, dict):
            raise ProtocolError("usage must be an object")
        observation.prompt_tokens = _usage_integer(usage, "prompt_tokens")
        observation.completion_tokens = _usage_integer(usage, "completion_tokens")
        details = usage.get("prompt_tokens_details") or {}
        if not isinstance(details, dict):
            raise ProtocolError("prompt token details must be an object")
        cached = details.get("cached_tokens")
        if cached is not None:
            observation.cached_tokens = _strict_nonnegative_integer(cached, "cached_tokens")


def _usage_integer(usage: dict[str, Any], key: str) -> int:
    if key not in usage:
        raise ProtocolError(f"usage omitted {key}")
    return _strict_nonnegative_integer(usage[key], key)


def _strict_nonnegative_integer(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ProtocolError(f"{label} must be a non-negative integer")
    return value


async def _classify_http_error(response: httpx.Response, observation: RequestObservation) -> None:
    """classify a non-200 response without retaining any server-supplied prose.

    only a machine error code is read out of the body, and only when it looks like a code rather
    than a message. an error body can echo the request back, so a free-text ``message`` is never
    persisted and a ``code`` that is not a slug is recorded as unrecognized instead of stored.
    """
    code = None
    try:
        body = await response.aread()
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, dict):
                candidate = error.get("code")
                if isinstance(candidate, str):
                    code = candidate if _ERROR_CODE_RE.fullmatch(candidate) else "unrecognized_code"
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    retry_after = response.headers.get("Retry-After")
    if (
        response.status_code == 503
        and code == "serving_capacity_unavailable"
        and retry_after == "1"
    ):
        observation.error_class = "exact_capacity_503"
    elif response.status_code == 503:
        observation.error_class = "other_503"
    elif response.status_code == 429:
        observation.error_class = "http_429"
    else:
        observation.error_class = "http_error"
    observation.error_detail = _sanitize(
        f"http {response.status_code} {code or observation.error_class}"
    )


def _validate_provenance(headers: httpx.Headers, target: Target) -> None:
    """assert every provenance header the target declared, and only those.

    a declared header that the response omits is a mismatch, not a skipped check, so a
    deployment that stops emitting provenance cannot quietly pass.
    """
    if not isinstance(target, AdapterTarget):
        return
    expected = {
        "X-Freesolo-Checkpoint": target.checkpoint,
        "X-Freesolo-Adapter-Revision": target.adapter_revision,
        "X-Freesolo-HF-Revision": target.hf_revision,
    }
    for name, value in expected.items():
        if value is None:
            continue
        if headers.get(name) != value:
            raise ProtocolError(f"response {name.lower()} did not match the immutable target")


def _allowed_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        name.lower(): value
        for name, value in headers.items()
        if name.lower() in _ALLOWED_RESPONSE_HEADERS
    }


def _sanitize(value: str) -> str:
    return " ".join(value.split())[:_MAX_ERROR_CHARS]
