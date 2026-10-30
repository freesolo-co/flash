"""Session credentials and backend-specific resume validation for managed custom runs."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import replace
from pathlib import Path


def session_token(run_id, attempt):
    secret = os.environ.get("HF_TOKEN", "")
    if not secret:
        raise RuntimeError("managed custom sessions require artifact credentials")
    message = json.dumps(["flash-custom-session-v1", run_id, attempt]).encode()
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def resolve_resume(spec, *, owner_org_id, owner_key_id):
    from huggingface_hub import HfApi, hf_hub_download

    from flash.engine.support.checkpoint_transport import MANIFEST, parse_manifest
    from flash.runner.lifecycle import preparation, status
    from flash.runner.results.checkpoints import (
        _adapter_folder_has_required_files,
        checkpoint_adapter_prefix,
    )
    from flash.schema import parse_checkpoint_ref

    source_id, step = parse_checkpoint_ref(spec.custom["resume_from"])
    source_status = status.get_status(source_id)
    if not preparation._warmstart_source_is_authorized(
        source_status,
        source_id,
        owner_org_id=owner_org_id,
        owner_key_id=owner_key_id,
    ):
        raise ValueError("custom resume checkpoint is not accessible")
    source = status.effective_spec_from_status(source_status)
    if source.algorithm != "custom" or source.model != spec.model:
        raise ValueError("custom resume requires a custom checkpoint of the same base model")
    if (source.train.lora_rank, source.train.lora_alpha) != (
        spec.train.lora_rank,
        spec.train.lora_alpha,
    ):
        raise ValueError("custom resume requires the source LoRA rank and alpha")
    repo = source.train.hf_repo
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    revision = api.repo_info(repo, repo_type="dataset").sha
    path = checkpoint_adapter_prefix(source, step)
    required = {f"{path}/{name}" for name in ("metadata.json", "algorithm.json")}
    files = api.list_repo_files(repo, repo_type="dataset", revision=revision)
    native_state = f"{path}/state.pt" in files
    if not native_state and f"{path}/{MANIFEST}" in files:
        manifest = hf_hub_download(
            repo,
            f"{path}/{MANIFEST}",
            repo_type="dataset",
            revision=revision,
            token=os.environ.get("HF_TOKEN"),
        )
        entry = parse_manifest(Path(manifest).read_bytes()).get("state.pt")
        native_state = entry is not None and all(
            f"{path}/{part['path']}" in files for part in entry["parts"]
        )
    if (
        not required <= set(files)
        or not native_state
        or not _adapter_folder_has_required_files(files, path)
    ):
        raise ValueError("custom checkpoint is incomplete")
    metadata_file = hf_hub_download(
        repo,
        f"{path}/metadata.json",
        repo_type="dataset",
        revision=revision,
        token=os.environ.get("HF_TOKEN"),
    )
    metadata = json.loads(Path(metadata_file).read_text())
    if metadata.get("format") != "flash-custom-v1" or metadata.get("step") != step:
        raise ValueError("checkpoint has no compatible custom training state")
    if metadata.get("model_revision") != source.model_revision:
        raise ValueError("checkpoint base revision does not match the source run")
    if spec.train.max_steps <= step:
        raise ValueError("train.max_steps must exceed the resumed optimizer step")
    return replace(
        spec,
        model_revision=source.model_revision,
        model_revision_auto=True,
        custom={
            **spec.custom,
            "resolved_resume": {"repo": repo, "revision": revision, "path": path},
        },
    )
