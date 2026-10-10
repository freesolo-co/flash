"""Adapter artifact validation for serving deployments.

Before an adapter is registered with the serving backend, its files on the Hub must be
checked: the tensor file has to exist, and the rank recorded in `adapter_config.json` has
to be one the serving engine was built for. That validation is independent of the
deployment state machine, so it lives here.

Split out of `flash.serve.deployment.deploy` to keep that module under the file-size limit.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

from flash.adapters.artifacts import has_loadable_adapter_weights, is_adapter_weight_filename
from flash.adapters.lora_rank import rank_from_adapter_config
from flash.adapters.targets import config_targets_images
from flash.serve.contract.errors import AdapterConfigMissing, AdapterTensorMissing, ServingError


@dataclass(frozen=True)
class AdapterArtifactMetadata:
    lora_rank: int
    targets_images: bool
    artifact_digest: str


def _is_hf_not_found_error(exc: Exception) -> bool:
    try:
        import huggingface_hub.errors as hf_errors  # type: ignore[import-not-found]

        not_found_types = tuple(
            cls
            for name in (
                "EntryNotFoundError",
                "LocalEntryNotFoundError",
                "RepositoryNotFoundError",
                "RevisionNotFoundError",
            )
            if isinstance((cls := getattr(hf_errors, name, None)), type)
        )
        if not_found_types and isinstance(exc, not_found_types):
            return True
    except Exception:
        pass
    return getattr(getattr(exc, "response", None), "status_code", None) == 404


def validate_serving_lora_rank(model: str, lora_rank: int, *, rank_source: str = "adapter") -> None:
    """Fail before registration when a trained adapter rank exceeds serving capacity."""
    from flash.core.catalog import serving_lora_rank_cap

    max_lora_rank = serving_lora_rank_cap(model)
    if max_lora_rank is None:
        return
    if int(lora_rank) > max_lora_rank:
        raise ValueError(
            f"{model} serving supports max_lora_rank={max_lora_rank}; "
            f"{rank_source} has rank {int(lora_rank)} and cannot be deployed"
        )


def _verify_adapter_artifact_tensors(
    hf_repo: str, subfolder: str, *, artifact_revision: str
) -> tuple[dict[str, object], ...]:
    """Confirm the adapter has tensor weights before registering it with serving."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - package extra is present in supported installs
        raise ServingError(
            "could not verify adapter tensors: huggingface_hub is not installed"
        ) from exc
    try:
        entries = list(
            HfApi().list_repo_tree(
                repo_id=hf_repo,
                path_in_repo=subfolder.rstrip("/"),
                repo_type="dataset",
                recursive=False,
                revision=artifact_revision,
                token=os.environ.get("HF_TOKEN"),
            )
        )
    except Exception as exc:
        message = f"could not verify adapter tensors: failed to list {hf_repo}:{subfolder}"
        if _is_hf_not_found_error(exc):
            raise AdapterTensorMissing(message) from exc
        raise ServingError(message) from exc

    listed_paths: list[str] = []
    tensor_paths: list[str] = []
    file_facts: list[dict[str, object]] = []
    zero_byte_tensor_paths: list[str] = []
    for entry in entries:
        path = str(getattr(entry, "path", "") or "")
        if not path:
            continue
        # every listed name, because whether the shards form a loadable set is decided by the index
        # beside them -- and the index is not itself a weight file.
        listed_paths.append(path)
        lfs = getattr(entry, "lfs", None)
        file_facts.append(
            {
                "path": path,
                "size": int(getattr(entry, "size", 0) or 0),
                "oid": str(
                    getattr(lfs, "sha256", None)
                    or getattr(lfs, "oid", None)
                    or getattr(entry, "blob_id", None)
                    or ""
                ),
            }
        )
        if not is_adapter_weight_filename(path):
            continue
        tensor_paths.append(path)
        size = getattr(entry, "size", None)
        try:
            if size is not None and int(size) <= 0:
                zero_byte_tensor_paths.append(path)
                continue
        except (TypeError, ValueError):
            pass

    location = f"{hf_repo}:{subfolder}"
    if zero_byte_tensor_paths:
        raise AdapterTensorMissing(
            f"could not verify adapter tensors: {location} has zero-byte adapter tensor "
            f"file(s): {', '.join(zero_byte_tensor_paths)}"
        )
    if not tensor_paths:
        raise AdapterTensorMissing(
            f"could not verify adapter tensors: {location} has no adapter_model tensor file"
        )
    # Present is not loadable. peft discovers the sharded representation only through
    # `adapter_model.<ext>.index.json`, so an interrupted upload that left one
    # `adapter_model-00001-of-00002.safetensors` behind passes the check above while carrying
    # nothing peft can bind. Registering it deploys a base model the user benchmarks as their
    # adapter, and peft reports that as a UserWarning rather than an error -- this refusal is the
    # only signal that ever reaches them.
    if not has_loadable_adapter_weights(listed_paths):
        raise AdapterTensorMissing(
            f"could not verify adapter tensors: {location} has adapter_model shard(s) but no "
            "complete index-referenced set, so there is nothing peft can load"
        )
    return tuple(sorted(file_facts, key=lambda item: str(item["path"])))


def _load_adapter_config(
    hf_repo: str, subfolder: str, *, artifact_revision: str
) -> tuple[dict, str]:
    filename = f"{subfolder.rstrip('/')}/adapter_config.json"
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - package extra is present in supported installs
        raise ServingError(
            "could not verify adapter rank: huggingface_hub is not installed"
        ) from exc
    try:
        local = hf_hub_download(
            repo_id=hf_repo,
            filename=filename,
            repo_type="dataset",
            revision=artifact_revision,
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as exc:
        message = f"could not verify adapter rank: failed to read {hf_repo}:{filename}"
        if _is_hf_not_found_error(exc):
            raise AdapterConfigMissing(message) from exc
        raise ServingError(message) from exc
    try:
        with open(local, encoding="utf-8") as f:
            config = json.load(f)
    except Exception as exc:
        raise ValueError(
            f"could not verify adapter rank: invalid JSON in {hf_repo}:{filename}"
        ) from exc
    if not isinstance(config, dict):
        raise ValueError(
            f"could not verify adapter rank: {hf_repo}:{filename} is not a JSON object"
        )
    return config, filename


def adapter_artifact_metadata(
    hf_repo: str, subfolder: str, *, artifact_revision: str
) -> AdapterArtifactMetadata:
    """Read adapter metadata and verify that tensor weights exist."""
    config, filename = _load_adapter_config(hf_repo, subfolder, artifact_revision=artifact_revision)
    files = _verify_adapter_artifact_tensors(
        hf_repo, subfolder, artifact_revision=artifact_revision
    )
    content = json.dumps(
        {
            "config": config,
            "files": files,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return AdapterArtifactMetadata(
        lora_rank=rank_from_adapter_config(config, source=f"{hf_repo}:{filename}"),
        # non-fatal by construction: an unmarked or malformed marker reads as text-only, so modality
        # uncertainty weakens the smoke rather than stranding an otherwise usable deployment.
        targets_images=config_targets_images(config),
        artifact_digest=hashlib.sha256(content).hexdigest(),
    )
