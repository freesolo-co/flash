"""Build a private checkpoint archive with the server's artifact-storage credential."""

from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from pathlib import Path, PurePosixPath

from flash.adapters.artifacts import is_adapter_weight_filename
from flash.serve.contract.errors import ServingError
from flash.serve.deployment.export import (
    _adapter_weight_paths,
    _normalize_export_adapter_keys,
    _normalize_export_targeting,
    _repair_export_metadata,
)

MAX_ARCHIVE_BYTES = 8 * 1024**3
SCHEMA = "flash-adapter-download-v1"


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _fetch_adapter(source_repo: str, subfolder: str, destination: Path) -> tuple[Path, str]:
    from huggingface_hub import HfApi, RepoFile, hf_hub_download

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise ServingError("server artifact-storage credential is unavailable")
    prefix = PurePosixPath(subfolder)
    if prefix.is_absolute() or ".." in prefix.parts or str(prefix) != subfolder:
        raise ValueError("invalid checkpoint artifact directory")
    api = HfApi(token=token)
    try:
        revision = api.repo_info(repo_id=source_repo, repo_type="dataset").sha
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ServingError("could not pin checkpoint artifact revision")
        entries = api.list_repo_tree(
            repo_id=source_repo,
            repo_type="dataset",
            revision=revision,
            path_in_repo=subfolder,
            recursive=False,
        )
        size = 0
        for entry in entries:
            if not isinstance(entry, RepoFile):
                continue
            path = PurePosixPath(entry.path)
            if path.parent != prefix:
                raise ServingError("unexpected checkpoint artifact path")
            if path.name not in {
                "adapter_config.json",
                "adapter_model.safetensors.index.json",
            } and not (is_adapter_weight_filename(path.name)):
                continue
            if type(entry.size) is not int or entry.size < 0:
                raise ServingError("invalid checkpoint artifact size")
            size += entry.size
            if size > MAX_ARCHIVE_BYTES:
                raise ServingError("checkpoint adapter exceeds download size limit")
            hf_hub_download(
                repo_id=source_repo,
                repo_type="dataset",
                revision=revision,
                filename=entry.path,
                local_dir=destination,
                token=token,
            )
    except ServingError:
        raise
    except Exception:
        # hub exceptions can contain private repository names and signed URLs.
        raise ServingError("could not read checkpoint adapter from artifact storage") from None
    return destination / subfolder, revision


def build_archive(
    *,
    source_repo: str,
    source_subfolder: str,
    run_id: str,
    checkpoint_id: str,
    base_model: str,
    base_model_revision: str,
    directory: Path,
) -> tuple[Path, str]:
    adapter, revision = _fetch_adapter(source_repo, source_subfolder, directory / "source")
    if not (adapter / "adapter_config.json").is_file() or not _adapter_weight_paths(adapter):
        raise ValueError("checkpoint has no complete, loadable LoRA adapter")
    _repair_export_metadata(adapter, base_model, base_model_revision)
    namespace = _normalize_export_adapter_keys(adapter)
    _normalize_export_targeting(adapter, namespace)
    weights = _adapter_weight_paths(adapter)
    files = [adapter / "adapter_config.json", *weights]
    if not (adapter / "adapter_model.safetensors").is_file():
        files.append(adapter / "adapter_model.safetensors.index.json")
    if any(path.is_symlink() or path.parent != adapter for path in files):
        raise ValueError("checkpoint adapter contains unsupported artifact paths")
    if sum(path.stat().st_size for path in files) > MAX_ARCHIVE_BYTES:
        raise ValueError("checkpoint adapter exceeds download size limit")
    manifest = {
        "schema": SCHEMA,
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "base_model": base_model,
        "base_model_revision": base_model_revision,
        "source_revision": revision,
        "files": {
            path.name: {"sha256": file_digest(path), "bytes": path.stat().st_size} for path in files
        },
    }
    target = directory / "adapter.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
        for path in files:
            archive.write(path, arcname=path.name)
    return target, file_digest(target)
