"""strict data-only execution manifest for one immutable serving deployment."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Literal

from flash.schema import parse_checkpoint_ref
from flash.serve.control import DeploymentSpec, EngineIdentity
from flash.serve.control._canonical import (
    canonical_json,
    canonical_mapping,
    canonical_mapping_fingerprint,
)
from flash.serve.control._serialization import serialize_engine
from flash.serve.control.types import validate_deployment_spec

MANIFEST_SCHEMA = "flash.serving.manifest"
MANIFEST_VERSION = 2
_REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
_HEX_40_RE = re.compile(r"[0-9a-f]{40}")
_HEX_64_RE = re.compile(r"[0-9a-f]{64}")
_IMAGE_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_REPO_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
_SAFE_PART_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_ALLOWED_ENGINE_ARGS = frozenset({"enforce_eager"})
_ALLOWED_TOKENIZER_KWARGS = frozenset({"use_fast"})
_ALLOWED_PROCESSOR_KWARGS = frozenset({"max_pixels"})


class ManifestError(ValueError):
    """the execution manifest is malformed or not bound to its control spec."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ManifestError(f"{path} keys are not exact; missing={missing}, unknown={unknown}")


def _mapping(value: object, path: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ManifestError(f"{path} must be an object")
    if any(type(key) is not str for key in value):
        raise ManifestError(f"{path} keys must be strings")
    return value


def _sequence(value: object, path: str) -> list[Any]:
    if type(value) is not list:
        raise ManifestError(f"{path} must be an array")
    return value


def _string(value: object, path: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ManifestError(f"{path} must be a nonempty unpadded string")
    return value


def _digest(value: object, path: str) -> str:
    text = _string(value, path)
    if _HEX_64_RE.fullmatch(text) is None:
        raise ManifestError(f"{path} must be a lowercase sha-256 digest")
    return text


def _positive_int(value: object, path: str) -> int:
    if type(value) is not int or value <= 0:
        raise ManifestError(f"{path} must be a positive integer")
    return value


def _safe_relative_path(value: object, path: str) -> str:
    text = _string(value, path)
    if "\\" in text or text.startswith("/") or text.endswith("/"):
        raise ManifestError(f"{path} must be a safe relative posix path")
    parsed = PurePosixPath(text)
    if str(parsed) != text or any(
        part in {".", ".."} or _SAFE_PART_RE.fullmatch(part) is None for part in parsed.parts
    ):
        raise ManifestError(f"{path} must be a safe relative posix path")
    return text


def _frozen_json_mapping(value: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    try:
        normalized = canonical_mapping(value)
    except ValueError as exc:
        raise ManifestError(f"{path} must contain only json-native values") from exc
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    """one exact regular file declared by path, size, and digest."""

    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _safe_relative_path(self.path, "artifact file path"))
        object.__setattr__(self, "size", _positive_int(self.size, "artifact file size"))
        object.__setattr__(self, "sha256", _digest(self.sha256, "artifact file sha256"))

    def payload(self) -> dict[str, object]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class AdapterExecutionInput:
    """execution-only file declarations for one control-authorized adapter."""

    checkpoint_id: str
    files: tuple[ArtifactFile, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_id", _string(self.checkpoint_id, "checkpoint_id"))
        object.__setattr__(self, "files", tuple(self.files))
        _validate_file_table(self.files)


@dataclass(frozen=True, slots=True)
class ExecutionInputs:
    """json-native runtime kwargs and exact adapter file declarations."""

    expected_oci_digest: str
    engine_args: Mapping[str, Any]
    tokenizer_kwargs: Mapping[str, Any]
    processor_kwargs: Mapping[str, Any]
    adapters: tuple[AdapterExecutionInput, ...]

    def __post_init__(self) -> None:
        digest = _string(self.expected_oci_digest, "expected_oci_digest")
        if _IMAGE_DIGEST_RE.fullmatch(digest) is None:
            raise ManifestError("expected_oci_digest must be an exact sha256 image digest")
        object.__setattr__(self, "expected_oci_digest", digest)
        object.__setattr__(
            self, "engine_args", _frozen_json_mapping(self.engine_args, "engine_args")
        )
        object.__setattr__(
            self,
            "tokenizer_kwargs",
            _frozen_json_mapping(self.tokenizer_kwargs, "tokenizer_kwargs"),
        )
        object.__setattr__(
            self,
            "processor_kwargs",
            _frozen_json_mapping(self.processor_kwargs, "processor_kwargs"),
        )
        object.__setattr__(self, "adapters", tuple(self.adapters))
        revisions = [entry.checkpoint_id for entry in self.adapters]
        if len(revisions) != len(set(revisions)):
            raise ManifestError("execution inputs contain duplicate checkpoint identities")
        _validate_runtime_options(
            self.engine_args,
            self.tokenizer_kwargs,
            self.processor_kwargs,
        )


@dataclass(frozen=True, slots=True)
class ManifestAdapter:
    """one immutable adapter declaration in a validated serving manifest."""

    run_id: str
    checkpoint_id: str
    repo_id: str
    repo_type: Literal["model", "dataset"]
    source_revision: str
    source_subfolder: str
    base_model: str
    base_model_revision: str
    lora_rank: int
    thinking_default: bool
    structured_outputs_default: Mapping[str, Any] | None
    files: tuple[ArtifactFile, ...]
    aggregate_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "checkpoint_id",
            "repo_id",
            "source_revision",
            "base_model",
            "base_model_revision",
        ):
            object.__setattr__(self, name, _string(getattr(self, name), name))
        parsed_checkpoint = parse_checkpoint_ref(self.checkpoint_id)
        if parsed_checkpoint is None or parsed_checkpoint[0] != self.run_id:
            raise ManifestError("checkpoint_id must be permanent and belong to run_id")
        if _REPO_ID_RE.fullmatch(self.repo_id) is None:
            raise ManifestError("repo_id must be an exact owner/name repository id")
        if _HEX_40_RE.fullmatch(self.source_revision) is None:
            raise ManifestError("source_revision must be an exact lowercase revision")
        if _HEX_40_RE.fullmatch(self.base_model_revision) is None:
            raise ManifestError("base_model_revision must be an exact lowercase revision")
        if type(self.repo_type) is not str or self.repo_type not in {"model", "dataset"}:
            raise ManifestError("repo_type must be model or dataset")
        object.__setattr__(
            self,
            "source_subfolder",
            _safe_relative_path(self.source_subfolder, "source_subfolder"),
        )
        object.__setattr__(self, "lora_rank", _positive_int(self.lora_rank, "lora_rank"))
        if type(self.thinking_default) is not bool:
            raise ManifestError("thinking_default must be a boolean")
        if self.structured_outputs_default is not None:
            object.__setattr__(
                self,
                "structured_outputs_default",
                _frozen_json_mapping(
                    self.structured_outputs_default,
                    "structured_outputs_default",
                ),
            )
        object.__setattr__(self, "files", tuple(self.files))
        _validate_file_table(self.files)
        aggregate = _digest(self.aggregate_sha256, "aggregate_sha256")
        if aggregate != aggregate_file_digest(self.files):
            raise ManifestError("adapter aggregate_sha256 does not match its exact file table")
        object.__setattr__(self, "aggregate_sha256", aggregate)

    def payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "checkpoint_id": self.checkpoint_id,
            "source": {
                "repo_id": self.repo_id,
                "repo_type": self.repo_type,
                "revision": self.source_revision,
                "subfolder": self.source_subfolder,
            },
            "base_model": self.base_model,
            "base_model_revision": self.base_model_revision,
            "lora_rank": self.lora_rank,
            "thinking_default": self.thinking_default,
            "structured_outputs_default": (
                None
                if self.structured_outputs_default is None
                else dict(self.structured_outputs_default)
            ),
            "files": [entry.payload() for entry in self.files],
            "aggregate_sha256": self.aggregate_sha256,
        }


@dataclass(frozen=True, slots=True)
class ServingManifest:
    """strict manifest bound to one control spec and one expected image digest."""

    manifest_id: str
    spec_id: str
    deployment_id: str
    generation: int
    expected_oci_digest: str
    logical_base_model: str
    logical_base_revision: str
    engine: EngineIdentity
    engine_args: Mapping[str, Any]
    tokenizer_kwargs: Mapping[str, Any]
    processor_kwargs: Mapping[str, Any]
    adapters: tuple[ManifestAdapter, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_id", _digest(self.manifest_id, "manifest_id"))
        object.__setattr__(self, "spec_id", _digest(self.spec_id, "spec_id"))
        object.__setattr__(self, "deployment_id", _string(self.deployment_id, "deployment_id"))
        object.__setattr__(self, "generation", _positive_int(self.generation, "generation"))
        if type(self.engine) is not EngineIdentity:
            raise ManifestError("engine must be an exact EngineIdentity")
        if self.engine.trust_remote_code:
            raise ManifestError("serving manifests cannot enable trust_remote_code")
        expected = _string(self.expected_oci_digest, "expected_oci_digest")
        if _IMAGE_DIGEST_RE.fullmatch(expected) is None or expected != self.engine.image_digest:
            raise ManifestError("expected_oci_digest must exactly match the engine image digest")
        object.__setattr__(self, "expected_oci_digest", expected)
        object.__setattr__(
            self, "logical_base_model", _string(self.logical_base_model, "logical_base_model")
        )
        object.__setattr__(
            self,
            "logical_base_revision",
            _string(self.logical_base_revision, "logical_base_revision"),
        )
        if _HEX_40_RE.fullmatch(self.logical_base_revision) is None:
            raise ManifestError("logical_base_revision must be an exact lowercase revision")
        object.__setattr__(
            self, "engine_args", _frozen_json_mapping(self.engine_args, "engine_args")
        )
        object.__setattr__(
            self,
            "tokenizer_kwargs",
            _frozen_json_mapping(self.tokenizer_kwargs, "tokenizer_kwargs"),
        )
        object.__setattr__(
            self,
            "processor_kwargs",
            _frozen_json_mapping(self.processor_kwargs, "processor_kwargs"),
        )
        _validate_runtime_options(
            self.engine_args,
            self.tokenizer_kwargs,
            self.processor_kwargs,
        )
        _validate_fingerprints(
            self.engine, self.engine_args, self.tokenizer_kwargs, self.processor_kwargs
        )
        object.__setattr__(self, "adapters", tuple(self.adapters))
        if not self.adapters:
            raise ManifestError("manifest requires at least one adapter")
        checkpoint_ids = [adapter.checkpoint_id for adapter in self.adapters]
        if checkpoint_ids != sorted(checkpoint_ids) or len(checkpoint_ids) != len(
            set(checkpoint_ids)
        ):
            raise ManifestError(
                "manifest adapters must be unique and sorted by checkpoint identity"
            )
        if len(self.adapters) > self.engine.max_cpu_loras:
            raise ManifestError("manifest adapter count exceeds engine capacity")
        if any(adapter.lora_rank > self.engine.max_lora_rank for adapter in self.adapters):
            raise ManifestError("manifest adapter rank exceeds engine rank ceiling")
        if any(
            (adapter.base_model, adapter.base_model_revision)
            != (self.logical_base_model, self.logical_base_revision)
            for adapter in self.adapters
        ):
            raise ManifestError("manifest adapter logical base does not match logical_base")
        payload_id = hashlib.sha256(canonical_json(self.payload(False)).encode()).hexdigest()
        if payload_id != self.manifest_id:
            raise ManifestError("manifest_id does not match the canonical manifest payload")

    def payload(self, include_manifest_id: bool = True) -> dict[str, object]:
        engine = serialize_engine(self.engine)
        engine.update(
            {
                "engine_id": self.engine.engine_id,
                "engine_args": dict(self.engine_args),
                "tokenizer_kwargs": dict(self.tokenizer_kwargs),
                "processor_kwargs": dict(self.processor_kwargs),
            }
        )
        payload: dict[str, object] = {
            "schema": MANIFEST_SCHEMA,
            "version": MANIFEST_VERSION,
            "spec_id": self.spec_id,
            "deployment_id": self.deployment_id,
            "generation": self.generation,
            "expected_oci_digest": self.expected_oci_digest,
            "logical_base": {
                "model": self.logical_base_model,
                "revision": self.logical_base_revision,
            },
            "engine": engine,
            "adapters": [adapter.payload() for adapter in self.adapters],
        }
        if include_manifest_id:
            return {"manifest_id": self.manifest_id, **payload}
        return payload

    def canonical_json(self) -> str:
        return canonical_json(self.payload())


def _validate_file_table(table: Sequence[ArtifactFile]) -> None:
    if type(table) is not tuple or any(type(entry) is not ArtifactFile for entry in table):
        raise ManifestError("artifact files must be exact ArtifactFile records")
    paths = tuple(entry.path for entry in table)
    if paths != _REQUIRED_ADAPTER_FILES:
        raise ManifestError(
            "adapter file table must contain exactly config and one safetensors file"
        )


def aggregate_file_digest(files: Sequence[ArtifactFile]) -> str:
    """return the canonical aggregate identity of an exact ordered file table."""

    payload = [entry.payload() for entry in files]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_runtime_options(
    engine_args: Mapping[str, Any],
    tokenizer_kwargs: Mapping[str, Any],
    processor_kwargs: Mapping[str, Any],
) -> None:
    option_sets = (
        ("engine_args", engine_args, _ALLOWED_ENGINE_ARGS),
        ("tokenizer_kwargs", tokenizer_kwargs, _ALLOWED_TOKENIZER_KWARGS),
        ("processor_kwargs", processor_kwargs, _ALLOWED_PROCESSOR_KWARGS),
    )
    for name, options, allowed in option_sets:
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ManifestError(f"{name} contains unsupported keys: " + ", ".join(unknown))
    if "enforce_eager" in engine_args and type(engine_args["enforce_eager"]) is not bool:
        raise ManifestError("engine_args.enforce_eager must be a boolean")
    if "use_fast" in tokenizer_kwargs and type(tokenizer_kwargs["use_fast"]) is not bool:
        raise ManifestError("tokenizer_kwargs.use_fast must be a boolean")
    if "max_pixels" in processor_kwargs:
        _positive_int(processor_kwargs["max_pixels"], "processor_kwargs.max_pixels")


def _validate_fingerprints(
    engine: EngineIdentity,
    engine_args: Mapping[str, Any],
    tokenizer_kwargs: Mapping[str, Any],
    processor_kwargs: Mapping[str, Any],
) -> None:
    actual = (
        canonical_mapping_fingerprint(engine_args),
        canonical_mapping_fingerprint(tokenizer_kwargs),
        canonical_mapping_fingerprint(processor_kwargs),
    )
    expected = (
        engine.engine_args_fingerprint,
        engine.tokenizer_kwargs_fingerprint,
        engine.processor_kwargs_fingerprint,
    )
    if actual != expected:
        raise ManifestError("runtime kwargs fingerprints do not match the control engine identity")


def build_serving_manifest(
    spec: DeploymentSpec, execution_inputs: ExecutionInputs
) -> ServingManifest:
    """prove exact control binding and build one canonical data-only execution manifest."""

    validate_deployment_spec(spec)
    if type(execution_inputs) is not ExecutionInputs:
        raise ManifestError("execution_inputs must be an exact ExecutionInputs record")
    if spec.engine.trust_remote_code:
        raise ManifestError("serving manifests cannot enable trust_remote_code")
    if execution_inputs.expected_oci_digest != spec.engine.image_digest:
        raise ManifestError("execution image digest does not match the control engine")
    _validate_fingerprints(
        spec.engine,
        execution_inputs.engine_args,
        execution_inputs.tokenizer_kwargs,
        execution_inputs.processor_kwargs,
    )
    input_by_checkpoint = {entry.checkpoint_id: entry for entry in execution_inputs.adapters}
    expected_checkpoints = {adapter.checkpoint_id for adapter in spec.adapters}
    if set(input_by_checkpoint) != expected_checkpoints:
        raise ManifestError("execution adapter inputs do not exactly match the control spec")

    manifest_adapters: list[ManifestAdapter] = []
    for adapter in spec.adapters:
        entry = input_by_checkpoint[adapter.checkpoint_id]
        aggregate = aggregate_file_digest(entry.files)
        if aggregate != adapter.artifact_digest:
            raise ManifestError("adapter file aggregate does not match the control artifact digest")
        structured = (
            None
            if adapter.structured_outputs_default_json is None
            else json.loads(adapter.structured_outputs_default_json)
        )
        manifest_adapters.append(
            ManifestAdapter(
                run_id=adapter.run_id,
                checkpoint_id=adapter.checkpoint_id,
                repo_id=adapter.artifact_repo_id,
                repo_type=adapter.artifact_repo_type,
                source_revision=adapter.artifact_revision,
                source_subfolder=adapter.artifact_subfolder,
                base_model=adapter.base_model,
                base_model_revision=adapter.base_model_revision,
                lora_rank=adapter.lora_rank,
                thinking_default=adapter.thinking_default,
                structured_outputs_default=structured,
                files=entry.files,
                aggregate_sha256=aggregate,
            )
        )
    manifest_adapters.sort(key=lambda value: value.checkpoint_id)
    logical = spec.adapters[0]
    values = {
        "manifest_id": "0" * 64,
        "spec_id": spec.spec_id,
        "deployment_id": spec.deployment_id,
        "generation": spec.generation,
        "expected_oci_digest": execution_inputs.expected_oci_digest,
        "logical_base_model": logical.base_model,
        "logical_base_revision": logical.base_model_revision,
        "engine": spec.engine,
        "engine_args": execution_inputs.engine_args,
        "tokenizer_kwargs": execution_inputs.tokenizer_kwargs,
        "processor_kwargs": execution_inputs.processor_kwargs,
        "adapters": tuple(manifest_adapters),
    }
    provisional = object.__new__(ServingManifest)
    for field in fields(ServingManifest):
        object.__setattr__(provisional, field.name, values[field.name])
    manifest_id = hashlib.sha256(canonical_json(provisional.payload(False)).encode()).hexdigest()
    values["manifest_id"] = manifest_id
    return ServingManifest(**values)


def load_serving_manifest(raw: str | bytes | Mapping[str, Any]) -> ServingManifest:
    """parse and strictly validate one manifest without accepting unknown keys."""

    if isinstance(raw, bytes | str):
        try:
            payload = json.loads(raw, object_pairs_hook=_reject_duplicate_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ManifestError("serving manifest is not valid json") from exc
    elif type(raw) is dict:
        payload = raw
    else:
        raise ManifestError("serving manifest must be json text or an object")
    root = _mapping(payload, "manifest")
    _exact_keys(
        root,
        {
            "manifest_id",
            "schema",
            "version",
            "spec_id",
            "deployment_id",
            "generation",
            "expected_oci_digest",
            "logical_base",
            "engine",
            "adapters",
        },
        "manifest",
    )
    if root["schema"] != MANIFEST_SCHEMA or root["version"] != MANIFEST_VERSION:
        raise ManifestError("unsupported serving manifest schema or version")
    logical = _mapping(root["logical_base"], "logical_base")
    _exact_keys(logical, {"model", "revision"}, "logical_base")
    engine_payload = _mapping(root["engine"], "engine")
    identity_names = {field.name for field in fields(EngineIdentity)}
    _exact_keys(
        engine_payload,
        identity_names | {"engine_id", "engine_args", "tokenizer_kwargs", "processor_kwargs"},
        "engine",
    )
    identity = EngineIdentity(**{name: engine_payload[name] for name in identity_names})
    if engine_payload["engine_id"] != identity.engine_id:
        raise ManifestError("engine_id does not match the exact engine identity")
    adapters = tuple(
        _parse_adapter(entry, index)
        for index, entry in enumerate(_sequence(root["adapters"], "adapters"))
    )
    return ServingManifest(
        manifest_id=root["manifest_id"],
        spec_id=root["spec_id"],
        deployment_id=root["deployment_id"],
        generation=root["generation"],
        expected_oci_digest=root["expected_oci_digest"],
        logical_base_model=logical["model"],
        logical_base_revision=logical["revision"],
        engine=identity,
        engine_args=_mapping(engine_payload["engine_args"], "engine.engine_args"),
        tokenizer_kwargs=_mapping(engine_payload["tokenizer_kwargs"], "engine.tokenizer_kwargs"),
        processor_kwargs=_mapping(engine_payload["processor_kwargs"], "engine.processor_kwargs"),
        adapters=adapters,
    )


def _parse_adapter(value: object, index: int) -> ManifestAdapter:
    path = f"adapters[{index}]"
    payload = _mapping(value, path)
    _exact_keys(
        payload,
        {
            "run_id",
            "checkpoint_id",
            "source",
            "base_model",
            "base_model_revision",
            "lora_rank",
            "thinking_default",
            "structured_outputs_default",
            "files",
            "aggregate_sha256",
        },
        path,
    )
    source = _mapping(payload["source"], f"{path}.source")
    _exact_keys(source, {"repo_id", "repo_type", "revision", "subfolder"}, f"{path}.source")
    files = tuple(
        _parse_file(entry, index, file_index)
        for file_index, entry in enumerate(_sequence(payload["files"], f"{path}.files"))
    )
    structured = payload["structured_outputs_default"]
    if structured is not None:
        structured = _mapping(structured, f"{path}.structured_outputs_default")
    return ManifestAdapter(
        run_id=payload["run_id"],
        checkpoint_id=payload["checkpoint_id"],
        repo_id=source["repo_id"],
        repo_type=source["repo_type"],
        source_revision=source["revision"],
        source_subfolder=source["subfolder"],
        base_model=payload["base_model"],
        base_model_revision=payload["base_model_revision"],
        lora_rank=payload["lora_rank"],
        thinking_default=payload["thinking_default"],
        structured_outputs_default=structured,
        files=files,
        aggregate_sha256=payload["aggregate_sha256"],
    )


def _parse_file(value: object, adapter_index: int, file_index: int) -> ArtifactFile:
    path = f"adapters[{adapter_index}].files[{file_index}]"
    payload = _mapping(value, path)
    _exact_keys(payload, {"path", "size", "sha256"}, path)
    return ArtifactFile(**payload)


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate json key {key!r}")
        result[key] = value
    return result
