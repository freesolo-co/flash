"""Adapter artifact validation for serving deployments.

Before an adapter is registered with the serving backend, its files on the Hub must be
checked: the tensor file has to exist, and the rank recorded in `adapter_config.json` has
to be one the serving engine was built for. That validation is independent of the
deployment state machine, so it lives here.

Split out of `flash.serve.deploy` to keep that module under the file-size limit.
"""

from __future__ import annotations

import json
import os

from flash.adapters.artifacts import ADAPTER_WEIGHT_FILES
from flash.adapters.lora_rank import rank_from_adapter_config
from flash.serve.errors import AdapterConfigMissing, AdapterTensorMissing, ServingError


def _is_adapter_tensor_filename(filename: str) -> bool:
    name = filename.rsplit("/", 1)[-1]
    if name in ADAPTER_WEIGHT_FILES:
        return True
    return name.startswith("adapter_model-") and name.endswith((".safetensors", ".bin"))


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


def _verify_adapter_artifact_tensors(hf_repo: str, subfolder: str, *, hf_revision: str) -> None:
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
                revision=hf_revision,
                token=os.environ.get("HF_TOKEN"),
            )
        )
    except Exception as exc:
        message = f"could not verify adapter tensors: failed to list {hf_repo}:{subfolder}"
        if _is_hf_not_found_error(exc):
            raise AdapterTensorMissing(message) from exc
        raise ServingError(message) from exc

    tensor_paths: list[str] = []
    zero_byte_tensor_paths: list[str] = []
    for entry in entries:
        path = str(getattr(entry, "path", "") or "")
        if not _is_adapter_tensor_filename(path):
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
    if tensor_paths:
        return
    raise AdapterTensorMissing(
        f"could not verify adapter tensors: {location} has no adapter_model tensor file"
    )


def adapter_artifact_lora_rank(hf_repo: str, subfolder: str, *, hf_revision: str) -> int:
    """Read rank metadata and verify the adapter has tensor weights."""
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
            revision=hf_revision,
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
    _verify_adapter_artifact_tensors(hf_repo, subfolder, hf_revision=hf_revision)
    return rank_from_adapter_config(config, source=f"{hf_repo}:{filename}")
