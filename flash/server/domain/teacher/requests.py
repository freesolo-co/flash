"""Request-shape validation for the managed-teacher broker.

Split out of `teacher_broker` to keep that module under the file-size gate. Everything here is
pure: it inspects a decoded request body against the capability that authorized it and raises,
with no ledger row, no provider socket, and no clock. The dispatch and settlement logic that does
touch those stays in the parent, which re-exports these names so `teacher_broker.<name>` keeps
working for callers and for tests that patch through that module.

The scoring contract is exact-parameter, not best-effort: a request that differs from the one
shape we score is rejected rather than normalized, because a silently adjusted parameter would
produce logprobs for something other than the tokens the caller supplied.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from typing import Any

from flash._internal.fileio import reject_duplicate_keys
from flash.server.domain.teacher.errors import (
    TeacherBrokerError,
    ValidatedCompletionRequest,
)

MAX_REQUEST_BODY_BYTES = 48 * 1024 * 1024
REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._~-]{16,128}\Z")
CAPABILITY_PATTERN = re.compile(r"[A-Za-z0-9_-]{40,128}\Z")

_reject_duplicate_keys = reject_duplicate_keys(
    lambda _key: TeacherBrokerError("duplicate_json_key", status_code=400)
)


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
    except (TypeError, ValueError) as exc:
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


def _validate_chat_messages(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise TeacherBrokerError("unsupported_scoring_parameters", status_code=400)
    for message in value:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise TeacherBrokerError("invalid_chat_messages", status_code=400)
        if not isinstance(message["role"], str) or not message["role"]:
            raise TeacherBrokerError("invalid_chat_messages", status_code=400)
        content = message["content"]
        if isinstance(content, str):
            continue
        if not isinstance(content, list) or not content:
            raise TeacherBrokerError("invalid_chat_messages", status_code=400)
        for block in content:
            if not isinstance(block, dict) or not isinstance(block.get("type"), str):
                raise TeacherBrokerError("invalid_chat_content_block", status_code=400)
            block_type = block["type"]
            if block_type == "text":
                if set(block) != {"type", "text"} or not isinstance(block["text"], str):
                    raise TeacherBrokerError("invalid_chat_content_block", status_code=400)
            elif block_type == "image_url":
                image_url = block.get("image_url")
                if (
                    set(block) != {"type", "image_url"}
                    or not isinstance(image_url, dict)
                    or set(image_url) != {"url"}
                    or not isinstance(image_url["url"], str)
                    or not image_url["url"].startswith("data:image/")
                ):
                    raise TeacherBrokerError("invalid_chat_content_block", status_code=400)
            else:
                raise TeacherBrokerError("unknown_chat_content_block_type", status_code=400)


def validate_chat_completion_request(
    body: dict[str, Any], capability: dict[str, Any]
) -> ValidatedCompletionRequest:
    required = {
        "model",
        "messages",
        "max_tokens",
        "temperature",
        "seed",
        "prompt_logprobs",
        "return_token_ids",
    }
    if set(body) - required:
        raise TeacherBrokerError("extra_request_fields", status_code=400)
    if set(body) != required:
        raise TeacherBrokerError("missing_request_fields", status_code=400)
    if body["model"] != capability["model"]:
        raise TeacherBrokerError("model_scope_mismatch", status_code=403)
    if (
        isinstance(body["max_tokens"], bool)
        or body["max_tokens"] != 1
        or isinstance(body["temperature"], bool)
        or body["temperature"] != 0
        or isinstance(body["seed"], bool)
        or body["seed"] != 0
        or isinstance(body["prompt_logprobs"], bool)
        or body["prompt_logprobs"] != 1
        or body["return_token_ids"] is not True
    ):
        raise TeacherBrokerError("unsupported_scoring_parameters", status_code=400)
    _validate_chat_messages(body["messages"])
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
