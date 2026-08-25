"""LoRA config + warm-start adapter loading for the fine-tuning worker."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from typing import Any

import flash.engine.worker.runtime.state as _worker_state
from flash.adapters.fused_experts import (
    has_complete_fused_expert_tensors,
    lora_target_parameters,
    validate_fused_expert_adapter_config,
)
from flash.adapters.lora_rank import resolve_adapter_ref
from flash.adapters.targets import LoraTargeting, require_modality_marker, resolve_lora_targeting
from flash.engine.plan.recipe import RECIPE
from flash.engine.worker.io.hf import (
    RetriableInfraError,
    _has_deployable_adapter,
    _prefetch_error_is_retriable,
    _require_hf_deadline_allowance,
    _sleep_with_hf_deadline,
)
from flash.engine.worker.model.lora import (
    _read_adapter_tensor_keys,
    _read_adapter_tensor_metadata,
)

_ADAPTER_DOWNLOAD_RETRIES = 4
_ADAPTER_DOWNLOAD_BACKOFF_S = 5.0


def validate_warmstart_adapter(
    config: Mapping[str, Any],
    model_id: str,
    adapter_dir: str,
    targeting: LoraTargeting | None = None,
) -> None:
    """Validate a downloaded warm-start adapter without changing its config or files."""
    # the same rejection the control plane already made at submit time, repeated on the bytes this
    # worker actually downloaded. not redundant: the two never share a call stack, so only this
    # call sees the config the trainer will load.
    require_modality_marker(config, source="warm-start adapter")
    tensors: Mapping[str, tuple[int, ...]] | None = None
    source_is_multimodal = config.get("exclude_modules") is None
    if targeting is not None:
        run_is_multimodal = targeting.exclude_modules is None
        if source_is_multimodal != run_is_multimodal:
            source_modality = "multimodal (image-trained)" if source_is_multimodal else "text-only"
            run_modality = "multimodal" if run_is_multimodal else "text-only"
            raise ValueError(
                f"warm-start modality mismatch: a {run_modality} run cannot continue a "
                f"{source_modality} adapter; start a fresh run or use a matching-modality source adapter"
            )
    validate_fused_expert_adapter_config(config, model_id)
    if not lora_target_parameters(model_id):
        return
    if tensors is None:
        tensors = _read_adapter_tensor_metadata(adapter_dir) or {}
    if not has_complete_fused_expert_tensors(tensors, config, model_id):
        raise ValueError(
            f"warm-start adapter for {model_id} does not contain complete fused expert LoRA weights"
        )


def make_lora(model_id: str, *, algorithm: str = "sft", multimodal: bool = False):
    """build the model's modality-correct serve-compatible lora target set."""
    from peft import LoraConfig

    targeting = resolve_lora_targeting(model_id, algorithm=algorithm, multimodal=multimodal)
    rank = _worker_state.JOB_SPEC.train.lora_rank if _worker_state.JOB_SPEC else RECIPE.lora.rank
    alpha = _worker_state.JOB_SPEC.train.lora_alpha if _worker_state.JOB_SPEC else RECIPE.lora.alpha
    model_revision = (
        getattr(_worker_state.JOB_SPEC, "model_revision", "") if _worker_state.JOB_SPEC else ""
    )
    kwargs = {
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": RECIPE.lora.dropout,
        "target_modules": targeting.target_modules,
        "exclude_modules": targeting.exclude_modules,
        "task_type": "CAUSAL_LM",
        "revision": model_revision or None,
    }
    if targeting.target_parameters:
        kwargs["target_parameters"] = targeting.target_parameters
    # pissa removed: it mutates the base, so its adapter corrupts serve + warm-start on the unmodified base.
    kwargs["init_lora_weights"] = True
    print(
        "[lora] init_lora_weights=True (standard zero-B; PiSSA removed for serve/warm-start safety)"
    )
    # rsLoRA removed: ~5.6x effective LR inflation with catalog LRs causes SFT divergence + serve corruption.
    kwargs["use_rslora"] = False
    return LoraConfig(**kwargs)


def _warmstart_adapter_is_loadable(adir: str) -> bool:
    """return true only for a structurally complete adapter config and weight file."""
    if not _has_deployable_adapter(adir):
        return False
    try:
        with open(os.path.join(adir, "adapter_config.json"), encoding="utf-8") as config_file:
            config = json.load(config_file)
        if not isinstance(config, dict) or str(config.get("peft_type", "")).upper() != "LORA":
            return False
        return bool(_read_adapter_tensor_keys(adir))
    except Exception:
        return False


def _download_adapter(adapter_prefix: str | None) -> str | None:
    """Download an init_from_adapter LoRA to /tmp/evdl/<prefix>/adapter and return its dir.

    ``adapter_prefix`` is the internal storage ref resolved by the control plane:
    ``<owner>/<repo>:<phase>/<run_id>[/checkpoints/step-N]``.
    """
    if not adapter_prefix:
        return None
    resolved = resolve_adapter_ref(adapter_prefix)
    if not resolved:
        return None
    repo, prefix = resolved
    from huggingface_hub import snapshot_download

    adir = os.path.join("/tmp/evdl", prefix, "adapter")
    # start from a clean path so the loadable-check can only ever accept files THIS download
    # materialized -- leftover materialization from an earlier worker subprocess, attempt, or a
    # different run sharing the same prefix must not satisfy the post-exception loadable check and
    # mask a terminal 404/403/429 for the current repo/revision.
    shutil.rmtree(adir, ignore_errors=True)
    for attempt in range(_ADAPTER_DOWNLOAD_RETRIES):
        _require_hf_deadline_allowance()
        try:
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                allow_patterns=[f"{prefix}/adapter/*"],
                local_dir="/tmp/evdl",
                token=os.environ.get("HF_TOKEN"),
                revision=(
                    _worker_state.JOB_SPEC.train.init_from_adapter_revision
                    if _worker_state.JOB_SPEC
                    else None
                )
                or None,
            )
        except Exception as error:
            # a later nonessential sidecar may fail after the config and weights are already complete.
            if _warmstart_adapter_is_loadable(adir):
                return adir
            if not _prefetch_error_is_retriable(error):
                raise RuntimeError(
                    "the prepared warm-start source adapter could not be downloaded"
                ) from None
        else:
            # a returned snapshot can still be incomplete: an interrupted transfer, or hf falling
            # back to a partial local_dir when a throttled metadata call cannot confirm the file set.
            if _warmstart_adapter_is_loadable(adir):
                return adir
        # discard partial local_dir materialization so the next attempt cannot reuse stale files.
        shutil.rmtree(adir, ignore_errors=True)
        if attempt + 1 < _ADAPTER_DOWNLOAD_RETRIES:
            try:
                if not _sleep_with_hf_deadline(_ADAPTER_DOWNLOAD_BACKOFF_S * (attempt + 1)):
                    break
            except Exception:
                break
    raise RetriableInfraError(
        "the prepared warm-start source adapter could not be downloaded after transient failures"
    ) from None
