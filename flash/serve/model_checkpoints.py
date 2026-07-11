"""internal client for exact full-model checkpoint evaluation serving."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from flash.serve.deploy import ServingError, serving_base_url

_CHECKPOINT_KEY_ENV = "FREESOLO_CHECKPOINT_INTERNAL_KEY"
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REPO_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
)
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
INTERPOLATION_METADATA_SCHEMA = "flash.model-interpolation"
INTERPOLATION_METADATA_VERSION = 1
_READBACK_ATTEMPTS = 3
_READBACK_DELAY_SECONDS = 1.0


@dataclass(frozen=True)
class RegisteredModelCheckpoint:
    model_id: str
    base_model: str
    model_repo_id: str
    model_revision: str
    tokenizer_repo_id: str
    tokenizer_revision: str
    deployment_token: str


def _checkpoint_headers() -> dict[str, str]:
    key = (os.environ.get(_CHECKPOINT_KEY_ENV) or "").strip()
    if not key:
        raise ServingError(
            f"checkpoint-backed serving is unavailable: {_CHECKPOINT_KEY_ENV} is not configured"
        )
    return {"X-Freesolo-Checkpoint-Internal-Key": key}


def _validate_model_id(model_id: str) -> str:
    if not _MODEL_ID_RE.fullmatch(model_id) or model_id.strip() != model_id:
        raise ValueError("model_id must be a safe flat serving id")
    return model_id


def _validate_repo(repo_id: str) -> str:
    if not _REPO_RE.fullmatch(repo_id) or ".." in repo_id:
        raise ValueError("checkpoint repository must be a strict Hugging Face namespace/name id")
    return repo_id


def _validate_sha(revision: str) -> str:
    value = revision.strip().lower()
    if not _SHA_RE.fullmatch(value):
        raise ValueError("checkpoint revision must be an immutable 40-character commit sha")
    return value


def _request(method: str, path: str, **kwargs: Any) -> httpx.Response:
    url = f"{serving_base_url()}{path}"
    try:
        response = httpx.request(
            method,
            url,
            headers=_checkpoint_headers(),
            timeout=kwargs.pop("timeout", 1800.0),
            follow_redirects=True,
            **kwargs,
        )
        response.raise_for_status()
        return response
    except httpx.HTTPStatusError as exc:
        detail = (exc.response.text or "")[:500]
        raise ServingError(
            f"checkpoint-backed serving rejected {path} with HTTP "
            f"{exc.response.status_code}: {detail}",
            status_code=exc.response.status_code,
        ) from exc
    except httpx.RequestError as exc:
        raise ServingError(f"checkpoint-backed serving is unavailable at {url}: {exc}") from exc


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


def register_evaluation_checkpoint(
    *,
    model_id: str,
    canonical_base_model: str,
    model_repo_id: str,
    model_revision: str,
    tokenizer_repo_id: str | None = None,
    tokenizer_revision: str | None = None,
    thinking: bool,
    metadata: dict[str, Any],
) -> RegisteredModelCheckpoint:
    """register one exact published checkpoint and verify the active record."""
    model_id = _validate_model_id(model_id)
    model_repo_id = _validate_repo(model_repo_id)
    model_revision = _validate_sha(model_revision)
    if not canonical_base_model.strip():
        raise ValueError("canonical_base_model must not be empty")
    if metadata.get("schema") != INTERPOLATION_METADATA_SCHEMA or metadata.get(
        "version"
    ) != INTERPOLATION_METADATA_VERSION:
        raise ValueError("checkpoint metadata must use the canonical interpolation metadata schema")
    if tokenizer_repo_id is not None:
        tokenizer_repo_id = _validate_repo(tokenizer_repo_id)
        if tokenizer_revision is None:
            raise ValueError("tokenizer_revision is required with tokenizer_repo_id")
        tokenizer_revision = _validate_sha(tokenizer_revision)
    elif tokenizer_revision is not None:
        raise ValueError("tokenizer_revision requires tokenizer_repo_id")

    payload: dict[str, Any] = {
        "model_id": model_id,
        "base_model": canonical_base_model.strip(),
        "model_repo_id": model_repo_id,
        "model_revision": model_revision,
        "thinking": bool(thinking),
        "private": True,
        "metadata": dict(metadata),
    }
    if tokenizer_repo_id is not None:
        payload["tokenizer_repo_id"] = tokenizer_repo_id
        payload["tokenizer_revision"] = tokenizer_revision

    response = _request("POST", "/model-checkpoints", json=payload)
    body = response.json()
    expected_tokenizer_repo = tokenizer_repo_id or model_repo_id
    expected_tokenizer_revision = tokenizer_revision or model_revision
    expected = {
        "model_id": model_id,
        "base_model": canonical_base_model.strip(),
        "model_repo_id": model_repo_id,
        "model_revision": model_revision,
        "tokenizer_repo_id": expected_tokenizer_repo,
        "tokenizer_revision": expected_tokenizer_revision,
        "status": "ready",
        "metadata": dict(metadata),
    }
    mismatches = {
        key: (body.get(key), value)
        for key, value in expected.items()
        if body.get(key) != value
    }
    deployment_token = str(body.get("deployment_token") or "")
    if mismatches or not deployment_token:
        raise ServingError(
            "checkpoint registration did not activate the exact requested model: "
            f"mismatches={mismatches}, deployment_token={deployment_token!r}"
        )

    active = None
    last_error: ServingError | None = None
    for attempt in range(_READBACK_ATTEMPTS):
        try:
            active = _request(
                "GET",
                f"/model-checkpoints/{model_id}",
                params={"expected_deployment_token": deployment_token},
                timeout=60.0,
            ).json()
            break
        except ServingError as exc:
            last_error = exc
            if exc.status_code not in {None, 502, 503, 504} or attempt + 1 >= _READBACK_ATTEMPTS:
                raise
            time.sleep(_READBACK_DELAY_SECONDS)
    if active is None:
        raise ServingError(f"authoritative checkpoint readback failed: {last_error}")
    if active.get("deployment_token") != deployment_token or any(
        active.get(key) != value for key, value in expected.items()
    ):
        raise ServingError(
            "authoritative checkpoint readback did not match the exact active checkpoint; refusing "
            "canonical-base plus LoRA fallback"
        )

    return RegisteredModelCheckpoint(
        model_id=model_id,
        base_model=canonical_base_model.strip(),
        model_repo_id=model_repo_id,
        model_revision=model_revision,
        tokenizer_repo_id=expected_tokenizer_repo,
        tokenizer_revision=expected_tokenizer_revision,
        deployment_token=deployment_token,
    )
