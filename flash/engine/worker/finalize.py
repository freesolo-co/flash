"""Train-metadata + run-metrics finalize for the worker."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from flash.engine.accounting import RunMetrics
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.perf import gpu_diagnostics

_FULL_MODEL_NAMESPACE = "Freesolo-Co"
_FULL_MODEL_REPO_PREFIX = "flash-checkpoint-"
_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _full_model_repo_id(run_id: str) -> str:
    if not _SAFE_RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run id is not safe for full-model checkpoint publication")
    name = f"{_FULL_MODEL_REPO_PREFIX}{run_id}"
    if len(name) > 96:
        raise ValueError("run id is too long for full-model checkpoint publication")
    return f"{_FULL_MODEL_NAMESPACE}/{name}"


def _remove_model_weights(root: Path) -> None:
    for pattern in (
        "*.safetensors",
        "*.safetensors.index.json",
        "pytorch_model*.bin",
        "pytorch_model*.bin.index.json",
    ):
        for path in root.glob(pattern):
            if path.is_file():
                path.unlink()


def _publish_private_full_model(root: Path, repo_id: str) -> str:
    if not repo_id.startswith(f"{_FULL_MODEL_NAMESPACE}/{_FULL_MODEL_REPO_PREFIX}"):
        raise ValueError("full-model checkpoint repository is outside the approved namespace")
    api = _w.hf_api()
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    api.update_repo_settings(repo_id=repo_id, repo_type="model", private=True)
    repo_info = api.repo_info(repo_id=repo_id, repo_type="model")
    if getattr(repo_info, "private", None) is not True:
        raise RuntimeError("full-model checkpoint repository is not private")
    parent_commit = repo_info.sha
    commit = api.upload_folder(
        folder_path=str(root),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"publish exact flash checkpoint {_w.RUN_ID}",
        delete_patterns=["*"],
        parent_commit=parent_commit,
    )
    revision = str(getattr(commit, "oid", "") or "").lower()
    if not _SHA_RE.fullmatch(revision):
        raise RuntimeError("full-model checkpoint upload did not return an immutable Hub commit")
    for attempt in range(3):
        active = str(api.repo_info(repo_id=repo_id, repo_type="model").sha or "").lower()
        if active == revision:
            return revision
        if attempt < 2:
            time.sleep(attempt + 1)
    raise RuntimeError("full-model checkpoint repository did not resolve to the uploaded Hub commit")


def finalize_interpolated_training(model: Any, tokenizer: Any) -> dict[str, Any] | None:
    """merge, publish, verify, and describe the exact trained interpolation."""
    resolved = getattr(_w, "RESOLVED_MODEL_SOURCE", None)
    if resolved is None or resolved.interpolation is None:
        return None

    from flash.engine.worker.model_interpolation import (
        ensure_safetensors_index,
        validate_materialized_interpolation,
        validate_output_tree,
    )
    from flash.serve.model_checkpoints import (
        INTERPOLATED_CHECKPOINT_INTENT_SCHEMA,
        canonical_interpolation_metadata,
    )

    source = Path(resolved.source)
    manifest = resolved.interpolation
    validate_materialized_interpolation(source, manifest)
    repo_id = _full_model_repo_id(_w.RUN_ID)
    with tempfile.TemporaryDirectory(prefix=f"flash-final-{_w.RUN_ID}-") as tmp:
        output = Path(tmp) / "model"
        shutil.copytree(source, output)
        _remove_model_weights(output)
        if not hasattr(model, "merge_and_unload"):
            raise RuntimeError("interpolated training model cannot merge its final LoRA")
        merged = model.merge_and_unload()
        merged.save_pretrained(output, safe_serialization=True, max_shard_size="1GB")
        tokenizer.save_pretrained(output)
        ensure_safetensors_index(output)
        trained_tree_fingerprint = validate_output_tree(output)
        revision = _publish_private_full_model(output, repo_id)

    metadata = canonical_interpolation_metadata(
        canonical_model=resolved.canonical_model,
        interpolation_manifest=manifest,
        trained_tree_fingerprint=trained_tree_fingerprint,
    )
    structured_outputs = getattr(getattr(_w, "JOB_SPEC", None), "train", None)
    structured_outputs = getattr(structured_outputs, "structured_outputs", "") or None
    return {
        "schema": INTERPOLATED_CHECKPOINT_INTENT_SCHEMA,
        "version": 1,
        "model_id": _w.RUN_ID,
        "base_model": resolved.canonical_model,
        "model_repo_id": repo_id,
        "model_revision": revision,
        "tokenizer_repo_id": None,
        "tokenizer_revision": None,
        "thinking": bool(_w.THINKING),
        "structured_outputs": structured_outputs,
        "private": True,
        "metadata": metadata,
        "output_fingerprint": metadata["trained_tree_fingerprint"],
        "interpolation_output_fingerprint": metadata[
            "interpolation_output_fingerprint"
        ],
    }


def write_train_meta(
    phase,
    adapter_dir,
    model_id,
    train_wall,
    setup_seconds,
    train_tokens,
    generated_tokens,
    notes,
    *,
    step=None,
    interpolated_checkpoint_intent=None,
):
    env = _w.require_active_env()
    resolved = getattr(_w, "RESOLVED_MODEL_SOURCE", None)
    source_metadata = (
        {
            "model_source": resolved.source,
            "model_interpolation": resolved.interpolation,
        }
        if resolved is not None
        else {"model_source": model_id, "model_interpolation": None}
    )
    finalized_notes = dict(notes or {})
    if interpolated_checkpoint_intent is not None:
        finalized_notes["interpolated_checkpoint_intent"] = interpolated_checkpoint_intent
    meta = {
        "phase": phase,
        "adapter_dir": adapter_dir,
        "model_id": model_id,
        "train_wall": train_wall,
        "setup_seconds": setup_seconds,
        "train_tokens": train_tokens,
        "generated_tokens": generated_tokens,
        **source_metadata,
        "notes": finalized_notes,
    }
    with open("/tmp/train_meta.json", "w") as f:
        json.dump(meta, f)
    _w.hf_upload_file("/tmp/train_meta.json", "train_meta.json", required=True)
    # Carry the completed optimizer ``step`` (when the caller supplies it) so this final pre-DONE
    # heartbeat doesn't clobber the last stepped training ping with a stepless one -- a cancel between
    # here and DONE would otherwise re-price a fully-trained run to 0 steps (codex[bot]).
    _step_field = {"step": int(step)} if isinstance(step, (int, float)) and step > 0 else {}
    _w.heartbeat(
        f"{phase}_train_done",
        **_step_field,
        **{k: meta[k] for k in ("train_wall", "train_tokens", "generated_tokens")},
        gpu=gpu_diagnostics(),
    )
    m = RunMetrics(
        arm=os.environ.get("FLASH_ARM", "runpod"),
        phase=phase,
        # Completed optimizer updates (opd passes step=opt_steps; sft/rl omit it -> None). _finalize
        # reads metrics.step to carry the true step onto the terminal `done` heartbeat.
        step=step,
        seed=_w.SEED,
        model_id=model_id,
        wall_seconds=train_wall,
        setup_seconds=setup_seconds,
        train_throughput_toks_per_s=(
            (generated_tokens or train_tokens) / train_wall if train_wall else 0.0
        ),
        train_tokens=train_tokens,
        generated_tokens=generated_tokens,
        notes={
            **finalized_notes,
            "renderer": "flash_env",
            "thinking": _w.THINKING,
            "train_wall": train_wall,
            "model_id": model_id,
            "environment": env.id,
            "job_spec": _w.JOB_SPEC.to_dict() if _w.JOB_SPEC else None,
            **source_metadata,
        },
    )
    _w._finalize(m)
