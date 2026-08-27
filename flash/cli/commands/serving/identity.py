"""portable immutable identity for existing customer-owned serving deployments.

The Modal plan validates more than deterministic resource names: it binds the exact manifest and
spec into app tags. Re-resolving mutable Hub repositories is therefore neither sufficient nor
necessary for status or teardown. This codec carries the already-resolved, credential-free bundle
without persisting it.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from collections.abc import Mapping

from flash.serve.control._canonical import canonical_json

_IDENTITY_SCHEMA = "flash.cli.serving.deployment-identity"
_IDENTITY_VERSION = 2
_MAX_IDENTITY_BYTES = 128 * 1024
_BASE64URL_RE = re.compile(r"[A-Za-z0-9_-]+")


def _placement_payload(placement) -> dict[str, object]:
    from flash.serve.control import ModalPlacement

    if type(placement) is not ModalPlacement:
        raise TypeError("deployment identity requires an exact ModalPlacement")
    return {
        "environment": placement.environment,
        "gpu": placement.gpu,
        "gpu_count": placement.gpu_count,
        "region": placement.region,
        "web_suffix": placement.web_suffix,
        "workspace_name": placement.workspace_name,
    }


def encode_deployment_identity(bundle) -> str:
    """encode one exact credential-free bundle for later status or teardown."""

    from flash.serve.provisioning.common.records import (
        DeploymentBundle,
        encode_manifest_environment,
    )

    if type(bundle) is not DeploymentBundle:
        raise TypeError("deployment identity requires an exact DeploymentBundle")
    bundle.__post_init__()
    payload = {
        "image_reference": bundle.image.reference,
        "manifest": encode_manifest_environment(bundle.manifest),
        "placement": _placement_payload(bundle.spec.placement),
        "provider": bundle.spec.provider,
        "schema": _IDENTITY_SCHEMA,
        "version": _IDENTITY_VERSION,
    }
    raw = canonical_json(payload).encode("utf-8")
    if len(raw) > _MAX_IDENTITY_BYTES:
        raise ValueError("deployment identity exceeds its byte limit")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("deployment identity contains a duplicate json key")
        value[key] = item
    return value


def _decode_payload(value: str) -> dict[str, object]:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError("--deployment-identity must be a nonempty unpadded string")
    if len(value) > _MAX_IDENTITY_BYTES * 2 or _BASE64URL_RE.fullmatch(value) is None:
        raise ValueError("--deployment-identity must be canonical base64url")
    padding = "=" * (-len(value) % 4)
    try:
        raw = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("--deployment-identity must be canonical base64url") from exc
    if len(raw) > _MAX_IDENTITY_BYTES:
        raise ValueError("deployment identity exceeds its byte limit")
    if base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=") != value:
        raise ValueError("--deployment-identity must be canonical base64url")
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("--deployment-identity must contain canonical utf-8 json") from exc
    if type(payload) is not dict or canonical_json(payload).encode("utf-8") != raw:
        raise ValueError("--deployment-identity json must be canonical")
    expected = {"image_reference", "manifest", "placement", "provider", "schema", "version"}
    if set(payload) != expected:
        raise ValueError("--deployment-identity fields are not exact")
    if payload["schema"] != _IDENTITY_SCHEMA or payload["version"] != _IDENTITY_VERSION:
        raise ValueError("--deployment-identity schema is not supported")
    return payload


def _mapping(value: object, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"deployment identity {name} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"deployment identity {name} fields are not exact")


def _placement(provider: object, value: object):
    from flash.serve.control import ModalPlacement

    if provider != "modal":
        raise ValueError("deployment identity provider must be modal")
    payload = _mapping(value, "placement")
    _exact_keys(
        payload,
        {"environment", "gpu", "gpu_count", "region", "web_suffix", "workspace_name"},
        "modal placement",
    )
    return ModalPlacement(**payload)


def _resolved_adapters(manifest) -> tuple[object, ...]:
    from flash.serve.control import ResolvedAdapter
    from flash.serve.control._serialization import canonical_adapter_sort_key

    adapters = []
    for adapter in manifest.adapters:
        structured = (
            None
            if adapter.structured_outputs_default is None
            else canonical_json(dict(adapter.structured_outputs_default))
        )
        adapters.append(
            ResolvedAdapter(
                run_id=adapter.run_id,
                checkpoint_id=adapter.checkpoint_id,
                artifact_repo_id=adapter.repo_id,
                artifact_repo_type=adapter.repo_type,
                artifact_revision=adapter.source_revision,
                artifact_digest=adapter.aggregate_sha256,
                artifact_subfolder=adapter.source_subfolder,
                base_model=adapter.base_model,
                base_model_revision=adapter.base_model_revision,
                lora_rank=adapter.lora_rank,
                thinking_default=adapter.thinking_default,
                structured_outputs_default_json=structured,
            )
        )
    return tuple(sorted(adapters, key=canonical_adapter_sort_key))


def decode_deployment_identity(value: str):
    """decode and fully revalidate one exact deployment bundle without network access."""

    from flash.serve.control import DeploymentSpec
    from flash.serve.provisioning import ServingImage
    from flash.serve.provisioning.common.records import (
        DeploymentBundle,
        decode_manifest_environment,
    )

    payload = _decode_payload(value)
    if payload["provider"] != "modal":
        raise ValueError("deployment identity provider must be modal")
    manifest_value = payload["manifest"]
    image_reference = payload["image_reference"]
    if type(manifest_value) is not str or type(image_reference) is not str:
        raise ValueError("deployment identity manifest and image reference must be strings")
    manifest = decode_manifest_environment(manifest_value)
    placement = _placement(payload["provider"], payload["placement"])
    spec = DeploymentSpec(
        deployment_id=manifest.deployment_id,
        generation=manifest.generation,
        provider=payload["provider"],
        placement=placement,
        engine=manifest.engine,
        adapters=_resolved_adapters(manifest),
    )
    image = ServingImage(reference=image_reference, digest=manifest.expected_oci_digest)
    return DeploymentBundle(spec=spec, manifest=manifest, image=image)


def _mismatch(flag: str) -> None:
    raise ValueError(f"--deployment-identity does not match --{flag}")


def _validate_authored_identity(args, bundle) -> None:
    from flash.serve.control import ModalPlacement

    spec = bundle.spec
    if len(spec.adapters) != 1:
        raise ValueError("--deployment-identity must contain exactly one adapter for this command")
    adapter = spec.adapters[0]
    from flash.schema import format_checkpoint_ref

    expected_checkpoint = format_checkpoint_ref(
        adapter.run_id, getattr(args, "checkpoint_step", None)
    )
    comparisons = (
        ("provider", spec.provider, args.provider),
        ("model", adapter.base_model, args.model),
        ("run", adapter.run_id, args.run),
        ("deployment-id", spec.deployment_id, args.deployment_id),
        ("generation", spec.generation, args.generation),
        ("image", bundle.image.reference, args.image),
        ("artifact-repo", adapter.artifact_repo_id, args.artifact_repo),
        ("artifact-repo-type", adapter.artifact_repo_type, args.artifact_repo_type),
        ("artifact-subfolder", adapter.artifact_subfolder, args.artifact_subfolder),
        ("lora-rank", adapter.lora_rank, args.lora_rank),
        ("checkpoint-step", adapter.checkpoint_id, expected_checkpoint),
        ("thinking", adapter.thinking_default, bool(args.thinking)),
    )
    for flag, actual, expected in comparisons:
        if actual != expected:
            _mismatch(flag)

    placement = spec.placement
    if type(placement) is ModalPlacement:
        modal = (
            ("modal-workspace", placement.workspace_name, args.modal_workspace),
            ("modal-environment", placement.environment, args.modal_environment),
            ("modal-web-suffix", placement.web_suffix, args.modal_web_suffix or None),
            ("modal-region", placement.region, args.modal_region),
        )
        for flag, actual, expected in modal:
            if actual != expected:
                _mismatch(flag)
    else:  # pragma: no cover - DeploymentSpec validation already excludes this
        raise TypeError("deployment identity placement is invalid")


def existing_deployment_bundle(args):
    """require the immutable identity created with the provider resources."""

    identity = getattr(args, "deployment_identity", "") or ""
    if not identity:
        raise ValueError(
            "--deployment-identity is required; pass the value printed by `flash serve deploy` so "
            "the original immutable bundle is used instead of resolving current Hub state"
        )
    bundle = decode_deployment_identity(identity)
    _validate_authored_identity(args, bundle)
    return bundle
