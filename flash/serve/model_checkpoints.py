"""control-plane client and validation for interpolated full-model checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import re
import uuid
from typing import Any

import httpx

from flash.serve.deploy import ServingError, serving_base_url
from flash.spec import SUPPORTED_MODEL_INTERPOLATION_PAIR

_CHECKPOINT_KEY_ENV = "FREESOLO_CHECKPOINT_INTERNAL_KEY"
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REPO_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
)
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
INTERPOLATION_METADATA_SCHEMA = "flash.model-interpolation"
INTERPOLATION_METADATA_VERSION = 1
INTERPOLATED_CHECKPOINT_INTENT_SCHEMA = "flash.interpolated_checkpoint_intent"
INTERPOLATED_CHECKPOINT_ACTIVATION_SCHEMA = "flash.interpolated_checkpoint_activation"
FLASH_INTERPOLATED_CHECKPOINT_NAMESPACE = uuid.UUID("57ca34bc-b4d8-5f0e-9e6c-60cbbd56f6af")

_INTENT_FIELDS = (
    "schema",
    "version",
    "model_id",
    "base_model",
    "model_repo_id",
    "model_revision",
    "tokenizer_repo_id",
    "tokenizer_revision",
    "thinking",
    "structured_outputs",
    "private",
    "metadata",
    "output_fingerprint",
    "interpolation_output_fingerprint",
)


def _checkpoint_headers() -> dict[str, str]:
    key = (os.environ.get(_CHECKPOINT_KEY_ENV) or "").strip()
    if not key:
        raise ServingError(
            f"checkpoint-backed serving is unavailable: {_CHECKPOINT_KEY_ENV} is not configured"
        )
    return {"X-Freesolo-Checkpoint-Internal-Key": key}


def _validate_model_id(model_id: Any) -> str:
    if not isinstance(model_id, str) or not _MODEL_ID_RE.fullmatch(model_id):
        raise ValueError("model_id must be a safe flat serving id")
    return model_id


def _validate_repo(repo_id: Any) -> str:
    if not isinstance(repo_id, str) or not _REPO_RE.fullmatch(repo_id) or ".." in repo_id:
        raise ValueError("checkpoint repository must be a strict Hugging Face namespace/name id")
    return repo_id


def _validate_sha(revision: Any) -> str:
    if not isinstance(revision, str) or not _SHA_RE.fullmatch(revision):
        raise ValueError("checkpoint revision must be an immutable lowercase 40-character commit sha")
    return revision


def _validate_fingerprint(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _FINGERPRINT_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 64-character sha256")
    return value


def canonical_interpolation_metadata(
    *,
    canonical_model: str,
    interpolation_manifest: dict[str, Any],
    trained_tree_fingerprint: str,
) -> dict[str, Any]:
    """build the versioned metadata that serving must echo exactly."""
    spec = interpolation_manifest.get("spec") or {}
    required = {
        "formula": interpolation_manifest.get("formula"),
        "parents": interpolation_manifest.get("parents"),
        "output_fingerprint": interpolation_manifest.get("output_fingerprint"),
        "tree_fingerprint": interpolation_manifest.get("tree_fingerprint"),
        "alpha": spec.get("alpha"),
    }
    missing = sorted(key for key, value in required.items() if value is None)
    if missing or not trained_tree_fingerprint:
        raise ValueError(f"interpolation metadata is incomplete: {missing}")
    return {
        "schema": INTERPOLATION_METADATA_SCHEMA,
        "version": INTERPOLATION_METADATA_VERSION,
        "canonical_model": canonical_model,
        "formula": interpolation_manifest.get("formula"),
        "alpha": spec.get("alpha"),
        "parents": interpolation_manifest.get("parents"),
        "tokenizer_config_source": interpolation_manifest.get("tokenizer_config_source"),
        "interpolation_output_fingerprint": interpolation_manifest.get("output_fingerprint"),
        "interpolation_tree_fingerprint": interpolation_manifest.get("tree_fingerprint"),
        "trained_tree_fingerprint": trained_tree_fingerprint,
    }


def validate_interpolated_checkpoint_intent(value: Any, *, run_id: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(_INTENT_FIELDS):
        raise ValueError("interpolated checkpoint intent has missing or unknown fields")
    intent = dict(value)
    if intent["schema"] != INTERPOLATED_CHECKPOINT_INTENT_SCHEMA or intent["version"] != 1:
        raise ValueError("interpolated checkpoint intent must use protocol schema version 1")
    if _validate_model_id(intent["model_id"]) != run_id:
        raise ValueError("interpolated checkpoint intent model_id must equal run_id")
    if not isinstance(intent["base_model"], str) or not intent["base_model"].strip():
        raise ValueError("interpolated checkpoint intent base_model is required")
    intent["model_repo_id"] = _validate_repo(intent["model_repo_id"])
    expected_repo = f"Freesolo-Co/flash-checkpoint-{run_id}"
    if intent["model_repo_id"] != expected_repo:
        raise ValueError("interpolated checkpoint intent repository is not the managed run repository")
    intent["model_revision"] = _validate_sha(intent["model_revision"])
    tokenizer_repo = intent["tokenizer_repo_id"]
    tokenizer_revision = intent["tokenizer_revision"]
    if (tokenizer_repo is None) != (tokenizer_revision is None):
        raise ValueError("tokenizer repository and revision must both be null or both be set")
    if tokenizer_repo is not None:
        intent["tokenizer_repo_id"] = _validate_repo(tokenizer_repo)
        intent["tokenizer_revision"] = _validate_sha(tokenizer_revision)
    if not isinstance(intent["thinking"], bool):
        raise ValueError("interpolated checkpoint intent thinking must be a boolean")
    structured = intent["structured_outputs"]
    if structured is not None and not isinstance(structured, str):
        raise ValueError("interpolated checkpoint intent structured_outputs must be null or a string")
    if intent["private"] is not True:
        raise ValueError("interpolated checkpoint intent must be private")
    metadata = intent["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("interpolated checkpoint intent metadata must be an object")
    metadata_fields = {
        "schema",
        "version",
        "canonical_model",
        "formula",
        "alpha",
        "parents",
        "tokenizer_config_source",
        "interpolation_output_fingerprint",
        "interpolation_tree_fingerprint",
        "trained_tree_fingerprint",
    }
    if set(metadata) != metadata_fields:
        raise ValueError("interpolated checkpoint metadata has missing or unknown fields")
    if metadata.get("schema") != INTERPOLATION_METADATA_SCHEMA or metadata.get("version") != 1:
        raise ValueError("interpolated checkpoint metadata schema is invalid")
    if metadata.get("canonical_model") != intent["base_model"]:
        raise ValueError("interpolated checkpoint canonical model does not match base_model")
    if metadata.get("formula") != "W=(1-alpha)*W_base+alpha*W_instruct":
        raise ValueError("interpolated checkpoint metadata formula is unsupported")
    alpha = metadata.get("alpha")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, numbers.Real)
        or not math.isfinite(alpha)
        or not 0.0 <= alpha <= 1.0
    ):
        raise ValueError("interpolated checkpoint metadata alpha is invalid")
    if metadata.get("tokenizer_config_source") not in {"base", "instruct"}:
        raise ValueError("interpolated checkpoint tokenizer source is invalid")
    parents = metadata.get("parents")
    if not isinstance(parents, dict) or set(parents) != {"base", "instruct"}:
        raise ValueError("interpolated checkpoint metadata parents are required")
    for name, expected_model in zip(
        ("base", "instruct"), SUPPORTED_MODEL_INTERPOLATION_PAIR, strict=True
    ):
        parent = parents[name]
        if not isinstance(parent, dict) or set(parent) != {
            "model",
            "requested_revision",
            "commit",
            "config_fingerprint",
        }:
            raise ValueError("interpolated checkpoint metadata parent is malformed")
        if parent.get("model") != expected_model:
            raise ValueError("interpolated checkpoint metadata parent pair is unsupported")
        _validate_sha(parent.get("commit"))
        requested_revision = parent.get("requested_revision")
        if requested_revision is not None:
            _validate_sha(requested_revision)
        _validate_fingerprint(parent.get("config_fingerprint"), "parent config fingerprint")
    _validate_fingerprint(
        metadata.get("interpolation_tree_fingerprint"), "interpolation tree fingerprint"
    )
    intent["output_fingerprint"] = _validate_fingerprint(
        intent["output_fingerprint"], "output_fingerprint"
    )
    intent["interpolation_output_fingerprint"] = _validate_fingerprint(
        intent["interpolation_output_fingerprint"], "interpolation_output_fingerprint"
    )
    if metadata.get("trained_tree_fingerprint") != intent["output_fingerprint"]:
        raise ValueError("trained output fingerprint does not match interpolation metadata")
    if (
        metadata.get("interpolation_output_fingerprint")
        != intent["interpolation_output_fingerprint"]
    ):
        raise ValueError("interpolation output fingerprint does not match metadata")
    return intent


def checkpoint_payload_hash(intent: dict[str, Any]) -> str:
    canonical = {key: intent[key] for key in _INTENT_FIELDS}
    payload = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def checkpoint_deployment_token(payload_hash: str) -> str:
    _validate_fingerprint(payload_hash, "payload_hash")
    return str(uuid.uuid5(FLASH_INTERPOLATED_CHECKPOINT_NAMESPACE, payload_hash))


def build_checkpoint_outbox(intent: dict[str, Any], *, run_id: str, now: float) -> dict[str, Any]:
    normalized = validate_interpolated_checkpoint_intent(intent, run_id=run_id)
    payload_hash = checkpoint_payload_hash(normalized)
    return {
        **normalized,
        "schema": INTERPOLATED_CHECKPOINT_ACTIVATION_SCHEMA,
        "payload_hash": payload_hash,
        "deployment_token": checkpoint_deployment_token(payload_hash),
        "expected_active_token": None,
        "activation_state": "pending",
        "activation_attempts": 0,
        "activation_error": None,
        "activation_updated_at": now,
        "activation_next_retry_at": None,
        "activated_at": None,
        "backend_mirror_state": "pending",
        "backend_mirror_error": None,
        "backend_mirrored_at": None,
    }


def checkpoint_registration_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_version": 1,
        "schema": INTERPOLATED_CHECKPOINT_INTENT_SCHEMA,
        "version": checkpoint["version"],
        "deployment_token": checkpoint["deployment_token"],
        "expected_active_token": checkpoint["expected_active_token"],
        "payload_hash": checkpoint["payload_hash"],
        "model_id": checkpoint["model_id"],
        "base_model": checkpoint["base_model"],
        "model_repo_id": checkpoint["model_repo_id"],
        "model_revision": checkpoint["model_revision"],
        "tokenizer_repo_id": checkpoint["tokenizer_repo_id"],
        "tokenizer_revision": checkpoint["tokenizer_revision"],
        "thinking": checkpoint["thinking"],
        "structured_outputs": checkpoint["structured_outputs"],
        "private": checkpoint["private"],
        "metadata": checkpoint["metadata"],
        "output_fingerprint": checkpoint["output_fingerprint"],
        "interpolation_output_fingerprint": checkpoint[
            "interpolation_output_fingerprint"
        ],
    }


def validate_active_checkpoint_response(body: Any, checkpoint: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ServingError("checkpoint activation returned a non-object response")
    expected = {
        "protocol_version": 1,
        "schema": INTERPOLATED_CHECKPOINT_INTENT_SCHEMA,
        "version": checkpoint["version"],
        "deployment_token": checkpoint["deployment_token"],
        "payload_hash": checkpoint["payload_hash"],
        "model_id": checkpoint["model_id"],
        "base_model": checkpoint["base_model"],
        "model_repo_id": checkpoint["model_repo_id"],
        "model_revision": checkpoint["model_revision"],
        "tokenizer_repo_id": checkpoint["tokenizer_repo_id"],
        "tokenizer_revision": checkpoint["tokenizer_revision"],
        "thinking": checkpoint["thinking"],
        "structured_outputs": checkpoint["structured_outputs"],
        "private": checkpoint["private"],
        "metadata": checkpoint["metadata"],
        "output_fingerprint": checkpoint["output_fingerprint"],
        "interpolation_output_fingerprint": checkpoint[
            "interpolation_output_fingerprint"
        ],
        "status": "ready",
    }
    missing = sorted(set(expected) - set(body))
    mismatches = {
        key: (body[key], value)
        for key, value in expected.items()
        if key in body and body[key] != value
    }
    if missing or mismatches:
        raise ServingError(
            f"checkpoint activation identity mismatch: missing={missing}, mismatches={mismatches}",
            status_code=409,
        )
    return body


def checkpoint_request(method: str, path: str, **kwargs: Any) -> httpx.Response:
    url = f"{serving_base_url()}{path}"
    try:
        return httpx.request(
            method,
            url,
            headers=_checkpoint_headers(),
            timeout=kwargs.pop("timeout", 1800.0),
            follow_redirects=True,
            **kwargs,
        )
    except httpx.RequestError as exc:
        raise ServingError(f"checkpoint-backed serving is unavailable at {url}: {exc}") from exc
