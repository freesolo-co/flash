"""sft training through OpenRLHF's out-of-process DeepSpeed trainer.

flash prepares exact whole-conversation token ids and completion-only masks in the parent process.
a generated child runtime replaces OpenRLHF's generic role-masking dataset with a dataset that returns
those tensors verbatim, adds multimodal tensors to the actor forward, installs LoRA+, and preserves
flash checkpoint and resume boundaries without importing the cuda 13 stack into the parent.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time

from flash.engine.recipe import RECIPE
from flash.engine.steps import (
    final_save_due,
    resolve_update_horizon,
    sft_update_steps,
    validate_save_steps,
)
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.heartbeat import liveness_heartbeat
from flash.engine.worker.hf import _is_commit_sha, resolve_cached_model_commit
from flash.engine.worker.openrlhf_common import (
    _hf_snapshot_identity,
    export_openrlhf_adapter,
    resolve_openrlhf_python,
    run_openrlhf_training,
)
from flash.engine.worker.packing import completion_mask_from_ids, model_is_gdn_hybrid
from flash.engine.worker.perf.attn import _attn_impl_for_capability
from flash.engine.worker.sft import (
    _model_arch_dims,
    _prepare_sft_examples,
    _pretokenize_completion_only,
    _reject_image_completion,
    _select_indexed_sft_examples,
    sft_completed_train_tokens,
    sft_under_ran,
)

_SFT_LORAPLUS_RATIO = 16.0
_LORAPLUS_READY_MARKER = "FLASH_OPENRLHF_LORAPLUS_READY"
_STEP_PREFIX = "FLASH_OPENRLHF_SFT_STEP="
_FINAL_STATE_FILE = "flash_sft_final_state.json"
_RUNTIME_CONFIG_FILE = "flash_sft_runtime.json"
_REQUIRED_ARG_KEYS = (
    "checkpoint_dir",
    "dataset_path",
    "epochs",
    "gradient_checkpointing",
    "gradient_checkpointing_reentrant",
    "learning_rate",
    "lora_alpha",
    "lora_rank",
    "max_length",
    "max_num_checkpoints",
    "micro_batch_size",
    "model_path",
    "output_dir",
    "resume_enabled",
    "seed",
    "train_batch_size",
)


def build_sft_openrlhf_args(cfg: dict) -> list[str]:
    """map a validated flash sft config to OpenRLHF's dotted argparse surface."""
    missing = [key for key in _REQUIRED_ARG_KEYS if key not in cfg]
    if missing:
        raise KeyError(f"build_sft_openrlhf_args missing required cfg keys: {missing}")
    if int(cfg["lora_rank"]) <= 0:
        raise ValueError("OpenRLHF SFT requires a positive LoRA rank")
    if int(cfg["train_batch_size"]) < int(cfg["micro_batch_size"]):
        raise ValueError("OpenRLHF train batch size must be at least the micro batch size")

    args = [
        "--ckpt.output_dir",
        str(cfg["output_dir"]),
        "--ckpt.path",
        str(cfg["checkpoint_dir"]),
        "--ckpt.save_steps",
        "1",
        "--ckpt.save_hf",
        "--ckpt.max_num",
        str(cfg["max_num_checkpoints"]),
        "--logger.logging_steps",
        "1",
        "--eval.steps",
        "-1",
        "--train.micro_batch_size",
        str(cfg["micro_batch_size"]),
        "--train.batch_size",
        str(cfg["train_batch_size"]),
        "--train.max_epochs",
        str(cfg["epochs"]),
        "--train.seed",
        str(cfg["seed"]),
        "--ds.zero_stage",
        "3",
        "--ds.param_dtype",
        "bf16",
        "--ds.attn_implementation",
        str(cfg.get("attn_implementation") or "eager"),
        "--model.model_name_or_path",
        str(cfg["model_path"]),
        "--ds.lora.rank",
        str(cfg["lora_rank"]),
        "--ds.lora.alpha",
        str(cfg["lora_alpha"]),
        "--data.dataset",
        str(cfg["dataset_path"]),
        "--data.max_samples",
        str(cfg.get("row_count", 1_000_000)),
        "--data.max_len",
        str(cfg["max_length"]),
        "--data.dataloader_num_workers",
        str(cfg.get("num_workers", 4)),
        "--optim",
        "adam",
        "--adam.lr",
        str(cfg["learning_rate"]),
        "--adam.betas",
        str(cfg.get("adam_beta1", 0.9)),
        str(cfg.get("adam_beta2", 0.999)),
        "--adam.eps",
        str(cfg.get("adam_epsilon", 1e-8)),
        "--adam.weight_decay",
        str(cfg.get("weight_decay", 0.0)),
        "--lr_scheduler",
        "linear",
        "--lr_warmup_ratio",
        str(cfg.get("warmup_ratio", RECIPE.sft.warmup_frac)),
        "--max_norm",
        str(cfg.get("max_norm", 1.0)),
    ]
    if cfg["resume_enabled"]:
        args.append("--ckpt.load_enable")
    if cfg["gradient_checkpointing"]:
        args.append("--model.gradient_checkpointing_enable")
    if cfg["gradient_checkpointing_reentrant"]:
        args.append("--model.gradient_checkpointing_reentrant")
    if cfg.get("wandb_enabled"):
        args.extend(
            [
                "--logger.wandb.key",
                "use-env",
                "--logger.wandb.project",
                str(cfg.get("wandb_project") or "flash"),
                "--logger.wandb.run_name",
                str(cfg.get("wandb_run_name") or "sft"),
            ]
        )
        if cfg.get("wandb_org"):
            args.extend(["--logger.wandb.org", str(cfg["wandb_org"])])
        if cfg.get("wandb_group"):
            args.extend(["--logger.wandb.group", str(cfg["wandb_group"])])
    return args


def _serialize_multimodal_inputs(values: dict) -> bytes:
    if not values:
        return b""
    import numpy as np

    arrays = {}
    for key, value in values.items():
        if value is None:
            continue
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arrays[key] = np.asarray(value)
    if not arrays:
        return b""
    payload = io.BytesIO()
    np.savez(payload, **arrays)
    return payload.getvalue()


def _openrlhf_dataset_features():
    from datasets import Features, Sequence, Value

    return Features(
        {
            "input_ids": Sequence(Value("int64")),
            "loss_mask": Sequence(Value("int8")),
            "multimodal_inputs": Value("binary"),
        }
    )


def _write_openrlhf_dataset(rows: list[dict], path: str) -> None:
    from datasets import Dataset

    Dataset.from_list(rows, features=_openrlhf_dataset_features()).save_to_disk(path)


def _multimodal_messages_with_images(messages: list[dict], images: list[object]) -> list[dict]:
    image_iter = iter(images)
    prepared = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, list):
            blocks = []
            for block in content:
                block = dict(block)
                if block.get("type") == "image":
                    block["image"] = next(image_iter)
                blocks.append(block)
            copied["content"] = blocks
        prepared.append(copied)
    try:
        next(image_iter)
    except StopIteration:
        return prepared
    raise ValueError("unused decoded image while preparing multimodal sft tokens")


def _processor_tokenized_row(
    processor,
    prompt_messages: list[dict],
    completion_messages: list[dict],
    images: list[object],
    *,
    max_length: int,
    thinking: bool,
) -> tuple[list[int], list[int], bytes]:
    prepared_prompt = _multimodal_messages_with_images(prompt_messages, images)
    full_messages = [*prepared_prompt, *completion_messages]
    common = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "enable_thinking": thinking,
    }
    full = dict(
        processor.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            **common,
        )
    )
    prompt = dict(
        processor.apply_chat_template(
            prepared_prompt,
            add_generation_prompt=True,
            **common,
        )
    )

    def ids(value) -> list[int]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if value and isinstance(value[0], list):
            value = value[0]
        return [int(item) for item in value]

    full_input_ids = ids(full.pop("input_ids"))
    full.pop("attention_mask", None)
    full.pop("token_type_ids", None)
    full.pop("mm_token_type_ids", None)
    if images and len(full_input_ids) > max_length:
        raise ValueError(
            "multimodal SFT example exceeds sft_max_len; truncating image tokens would desynchronize "
            "the processor features, so increase sft_max_len or shorten the example"
        )
    input_ids = full_input_ids[:max_length]
    prompt_ids = ids(prompt["input_ids"])[:max_length]
    loss_mask = completion_mask_from_ids(prompt_ids, input_ids)
    return input_ids, loss_mask, _serialize_multimodal_inputs(full)


def _has_real_target(row: dict, special_ids: set[int]) -> bool:
    return any(
        mask and token_id not in special_ids
        for token_id, mask in zip(row["input_ids"], row["loss_mask"], strict=True)
    )


def filter_openrlhf_sft_rows(rows: list[dict], special_ids: set[int]) -> tuple[list[dict], int]:
    """drop rows without a trainable completion token and fail if no signal remains."""
    kept = [row for row in rows if _has_real_target(row, special_ids)]
    dropped = len(rows) - len(kept)
    if not kept:
        raise ValueError(
            "every SFT example has an empty completion after sft_max_len truncation "
            "(nothing to train on); increase sft_max_len or shorten the prompts"
        )
    return kept, dropped


def build_text_openrlhf_rows(
    text_specs: list[dict], tokenizer, max_length: int
) -> tuple[list[dict], int]:
    """turn flash-rendered text rows into the exact tensors consumed by OpenRLHF."""
    kept_specs, tokenized, dropped = _pretokenize_completion_only(text_specs, tokenizer, max_length)
    del kept_specs
    rows = [
        {
            "input_ids": row["input_ids"],
            "loss_mask": row["completion_mask"],
            "multimodal_inputs": b"",
        }
        for row in tokenized
    ]
    return rows, dropped


def _warmstart_provenance_revision(adapter_dir: str, model_id: str) -> str:
    path = os.path.join(adapter_dir, "base_model_provenance.json")
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, encoding="utf-8") as file:
            provenance = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("the prepared SFT warm-start adapter has invalid provenance") from error
    if str(provenance.get("model_id") or "").strip() != model_id:
        raise ValueError("SFT warm-start adapter provenance model does not match the target model")
    return str(provenance.get("resolved_commit") or "").strip()


def validate_openrlhf_warmstart_adapter(
    adapter_dir: str,
    *,
    model_id: str,
    model_revision: str,
    expected_rank: int,
) -> None:
    """validate warm-start identity before the paid child process starts."""
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    try:
        with open(config_path, encoding="utf-8") as file:
            config = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("the prepared SFT warm-start adapter has no valid config") from error
    if not model_revision:
        raise ValueError("SFT warm-start validation requires an immutable target model revision")
    if str(config.get("peft_type") or "").upper() != "LORA":
        raise ValueError("SFT warm-start source must be a LoRA adapter")
    rank = int(config.get("r") or 0)
    if rank != expected_rank:
        raise ValueError(
            f"SFT warm-start adapter rank {rank} does not match the prepared train.lora_rank "
            f"{expected_rank}; rank changes are not supported"
        )
    base = str(config.get("base_model_name_or_path") or "").strip()
    if not base:
        raise ValueError("SFT warm-start adapter does not declare its base model")
    snapshot = _hf_snapshot_identity(base)
    snapshot_revision = ""
    if snapshot is not None:
        base_model, snapshot_revision = snapshot
        if base_model != model_id:
            raise ValueError("SFT warm-start adapter base model does not match the target model")
    elif base != model_id:
        raise ValueError("SFT warm-start adapter base model does not match the target model")
    revision = str(config.get("revision") or "").strip()
    declared_revision = revision if _is_commit_sha(revision) else snapshot_revision
    if not declared_revision:
        declared_revision = _warmstart_provenance_revision(adapter_dir, model_id)
    if declared_revision != model_revision:
        raise ValueError("SFT warm-start adapter revision does not match the target model revision")
    if _is_commit_sha(revision) and snapshot_revision and revision != snapshot_revision:
        raise ValueError("SFT warm-start adapter carries conflicting base revisions")
    if not any(
        os.path.isfile(os.path.join(adapter_dir, name))
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise RuntimeError("the prepared SFT warm-start adapter has no weights")


def _warmstart_adapter_path(model_id: str, model_revision: str, expected_rank: int) -> str | None:
    spec = _w.JOB_SPEC
    source = spec.train.init_from_adapter if spec else ""
    if not source:
        return None
    adapter_dir = _w._download_adapter(source)
    if not adapter_dir:
        raise RuntimeError("the prepared SFT warm-start adapter could not be downloaded")
    validate_openrlhf_warmstart_adapter(
        adapter_dir,
        model_id=model_id,
        model_revision=model_revision,
        expected_rank=expected_rank,
    )
    return adapter_dir


def _resolve_immutable_model_revision(model_id: str, requested_revision: str) -> str:
    """bind the prefetched model snapshot to the immutable commit used for all paid work."""
    resolved = resolve_cached_model_commit(model_id, requested_revision)
    if not resolved:
        message = (
            f"could not resolve the cached model snapshot for {model_id!r} to an immutable commit"
        )
        if not requested_revision:
            raise _w.RetriableInfraError(message)
        raise RuntimeError(message)
    return resolved


def _cached_model_path(model_id: str, model_revision: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=model_id,
        revision=model_revision,
        local_files_only=True,
    )


def _validate_gdn_realized_length(
    model_id: str,
    model_revision: str,
    realized_max_length: int,
) -> None:
    """fail loudly only when tokenized GDN rows actually reach the unvalidated 32k path."""
    if model_is_gdn_hybrid(model_id, revision=model_revision) and realized_max_length >= 32768:
        raise ValueError(
            "OpenRLHF 32k GDN SFT is not validated: the hybrid full-attention layers remain "
            "memory-intensive, and 32k execution needs matched real-GPU validation of eager "
            "attention plus ZeRO-3 fit before use; no validated sequence-parallel path exists"
        )


def _copy_processing_sidecars(source_dir: str, adapter_dir: str) -> None:
    if not os.path.isdir(source_dir):
        return
    for name in os.listdir(source_dir):
        source = os.path.join(source_dir, name)
        target = os.path.join(adapter_dir, name)
        if os.path.isdir(source):
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)


def _export_checkpoint_adapter(
    checkpoint_dir: str,
    adapter_dir: str,
    *,
    processing_dir: str,
    model_id: str,
    model_revision: str,
    python_bin: str,
) -> None:
    export_openrlhf_adapter(
        checkpoint_dir,
        adapter_dir,
        model_id,
        model_revision,
        python_bin,
    )
    _copy_processing_sidecars(processing_dir, adapter_dir)
    _w.write_base_model_provenance(adapter_dir, model_id, model_revision)


def _restore_openrlhf_resume(checkpoint_dir: str) -> int:
    resume = _w.hf_resume_checkpoint()
    if not resume:
        return 0
    match = re.fullmatch(r"checkpoint-(\d+)", os.path.basename(resume))
    if match is None:
        raise RuntimeError(f"invalid SFT resume checkpoint path {resume!r}")
    step = int(match.group(1))
    tag = f"global_step{step}"
    target = os.path.join(checkpoint_dir, tag)
    shutil.copytree(resume, target, dirs_exist_ok=True)
    with open(os.path.join(checkpoint_dir, "latest"), "w", encoding="utf-8") as file:
        file.write(tag)
    return step


def _durable_required_save_steps(required_steps: tuple[int, ...], resume_step: int) -> set[int]:
    candidates = [step for step in required_steps if step <= resume_step]
    if not candidates:
        return set()
    if not _w.HF_REPO:
        raise RuntimeError("required SFT saves have no artifact repository")
    durable: set[int] = set()
    for step in candidates:
        marker = f"{_w.hf_prefix()}/checkpoints/step-{step}/adapter/adapter_config.json"
        try:
            exists = _w.hf_api().file_exists(
                repo_id=_w.HF_REPO,
                filename=marker,
                repo_type="dataset",
            )
        except Exception as error:
            raise _w.RetriableInfraError(
                f"could not verify required SFT save step {step} on hf"
            ) from error
        if exists:
            durable.add(step)
    return durable


def _required_save_marker(checkpoint_dir: str, step: int, *, failed: bool = False) -> str:
    suffix = ".failed" if failed else ".done"
    return os.path.join(checkpoint_dir, f".flash-required-step-{step}{suffix}")


def _hf_export_ready_marker(checkpoint_dir: str, step: int) -> str:
    return os.path.join(checkpoint_dir, f".flash-hf-step-{step}.ready")


def _checkpoint_upload_marker(checkpoint_dir: str, step: int, *, failed: bool = False) -> str:
    suffix = ".failed" if failed else ".done"
    return os.path.join(checkpoint_dir, f".flash-upload-step-{step}{suffix}")


class _OpenRLHFCheckpointWatcher:
    """publish completed DeepSpeed checkpoints and their matching PEFT exports."""

    def __init__(
        self,
        *,
        checkpoint_dir: str,
        export_root: str,
        processing_dir: str,
        python_bin: str,
        model_id: str,
        model_revision: str,
        required_steps: tuple[int, ...],
        max_num_checkpoints: int = 1,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.export_root = export_root
        self.processing_dir = processing_dir
        self.python_bin = python_bin
        self.model_id = model_id
        self.model_revision = model_revision
        self.required_steps = frozenset(required_steps)
        self.max_num_checkpoints = max(1, int(max_num_checkpoints))
        self.processed_steps: set[int] = set()
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("OpenRLHF checkpoint watcher failed") from self._error

    def stop(self, *, require_complete: bool) -> None:
        self._stop.set()
        self._thread.join()
        self.raise_if_failed()
        if require_complete:
            missing = sorted(self.required_steps - self.processed_steps)
            if missing:
                raise RuntimeError(f"required saves were not durably published: {missing}")

    def _completed_checkpoints(self) -> list[tuple[int, str, str]]:
        found = []
        try:
            names = os.listdir(self.checkpoint_dir)
        except OSError:
            return found
        for name in names:
            match = re.fullmatch(r"global_step(\d+)_hf", name)
            if match is None:
                continue
            step = int(match.group(1))
            if step in self.processed_steps:
                continue
            hf_dir = os.path.join(self.checkpoint_dir, name)
            ds_dir = os.path.join(self.checkpoint_dir, f"global_step{step}")
            has_config = os.path.isfile(os.path.join(hf_dir, "adapter_config.json"))
            has_authoritative_weights = os.path.isfile(os.path.join(hf_dir, "adapter_model.bin"))
            export_ready = os.path.isfile(_hf_export_ready_marker(self.checkpoint_dir, step))
            if os.path.isdir(ds_dir) and has_config and has_authoritative_weights and export_ready:
                found.append((step, ds_dir, hf_dir))
        return sorted(found)

    def _prune_uploaded_checkpoints(self) -> None:
        retained = sorted(
            step
            for step in self.processed_steps
            if os.path.isdir(os.path.join(self.checkpoint_dir, f"global_step{step}"))
            or os.path.isdir(os.path.join(self.checkpoint_dir, f"global_step{step}_hf"))
        )
        for step in retained[: -self.max_num_checkpoints]:
            ds_dir = os.path.join(self.checkpoint_dir, f"global_step{step}")
            hf_dir = os.path.join(self.checkpoint_dir, f"global_step{step}_hf")
            markers = (
                _hf_export_ready_marker(self.checkpoint_dir, step),
                _checkpoint_upload_marker(self.checkpoint_dir, step),
                _checkpoint_upload_marker(self.checkpoint_dir, step, failed=True),
            )
            if os.path.isdir(ds_dir):
                shutil.rmtree(ds_dir)
            if os.path.isdir(hf_dir):
                shutil.rmtree(hf_dir)
            for marker in markers:
                if os.path.isfile(marker):
                    os.remove(marker)

    def _publish(self, step: int, ds_dir: str, hf_dir: str) -> None:
        required = step in self.required_steps
        try:
            adapter_dir = os.path.join(self.export_root, f"step-{step}")
            _export_checkpoint_adapter(
                hf_dir,
                adapter_dir,
                processing_dir=self.processing_dir,
                model_id=self.model_id,
                model_revision=self.model_revision,
                python_bin=self.python_bin,
            )

            def publish_adapter() -> None:
                _w.publish_deployable_checkpoint(
                    adapter_dir,
                    step,
                    required=required,
                    _provenance_ready=True,
                )

            uploaded = _w.upload_resume_checkpoint(
                step,
                ds_dir,
                before_upload=publish_adapter,
            )
            if not uploaded:
                raise RuntimeError(f"save step {step} full-state checkpoint was not published")
            self.processed_steps.add(step)
            if required:
                marker = _required_save_marker(self.checkpoint_dir, step)
                with open(marker, "w", encoding="utf-8") as file:
                    file.write("durable\n")
            self._prune_uploaded_checkpoints()
            marker = _checkpoint_upload_marker(self.checkpoint_dir, step)
            with open(marker, "w", encoding="utf-8") as file:
                file.write("uploaded\n")
        except BaseException as error:
            marker = _checkpoint_upload_marker(self.checkpoint_dir, step, failed=True)
            with open(marker, "w", encoding="utf-8") as file:
                file.write(f"{type(error).__name__}: {error}\n")
            if required:
                marker = _required_save_marker(self.checkpoint_dir, step, failed=True)
                with open(marker, "w", encoding="utf-8") as file:
                    file.write(f"{type(error).__name__}: {error}\n")
            raise

    def _run(self) -> None:
        try:
            while True:
                completed = self._completed_checkpoints()
                for step, ds_dir, hf_dir in completed:
                    self._publish(step, ds_dir, hf_dir)
                if self._stop.is_set() and not self._completed_checkpoints():
                    return
                time.sleep(0.25)
        except BaseException as error:
            self._error = error


_CHILD_ENV_EXACT = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "TZ",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "NVIDIA_VISIBLE_DEVICES",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_HUB_OFFLINE",
        "HF_HUB_ENABLE_HF_TRANSFER",
        "HF_HUB_DISABLE_XET",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
    }
)
_CHILD_ENV_PREFIXES = (
    "CUDA_",
    "NCCL_",
    "TORCH_",
    "PYTORCH_",
    "DEEPSPEED_",
    "OMP_",
    "MKL_",
    "OPENBLAS_",
    "FLA_",
    "LC_",
)


def build_openrlhf_sft_child_env(*, shim_dir: str, wandb_enabled: bool) -> dict[str, str]:
    """build a minimal child environment without provider or artifact credentials."""
    child = {
        key: value
        for key, value in os.environ.items()
        if key in _CHILD_ENV_EXACT or key.startswith(_CHILD_ENV_PREFIXES)
    }
    if wandb_enabled:
        child.update({key: value for key, value in os.environ.items() if key.startswith("WANDB_")})
    child["PYTHONPATH"] = os.pathsep.join(
        item for item in (shim_dir, os.environ.get("PYTHONPATH", "")) if item
    )
    child["PYTHONUNBUFFERED"] = "1"
    child["HF_HUB_OFFLINE"] = "1"
    child["TRANSFORMERS_OFFLINE"] = "1"
    child["TOKENIZERS_PARALLELISM"] = "false"
    child["FLASH_OPENRLHF_SFT_CONFIG"] = os.path.join(shim_dir, _RUNTIME_CONFIG_FILE)
    return child


def _render_sitecustomize() -> str:
    return "from flash_openrlhf_sft_runtime import apply_flash_openrlhf_sft_patches\n\napply_flash_openrlhf_sft_patches()\n"


def render_openrlhf_sft_runtime() -> str:
    """return the standalone child patch module loaded through sitecustomize."""
    return r"""from __future__ import annotations

import io
import json
import math
import os

_CONFIG_PATH = os.environ["FLASH_OPENRLHF_SFT_CONFIG"]
with open(_CONFIG_PATH, encoding="utf-8") as _config_file:
    CONFIG = json.load(_config_file)


def _apply_blackwell_fla_safety():
    import importlib.util

    import torch

    if not torch.cuda.is_available():
        return
    capability = torch.cuda.get_device_capability()
    if capability == (10, 0):
        os.environ.setdefault("FLA_TILELANG", "0")
    if capability[0] not in (10, 12) or importlib.util.find_spec("fla") is None:
        return

    from fla.ops.gated_delta_rule import wy_fast

    tuner = getattr(wy_fast, "prepare_wy_repr_bwd_kernel", None)
    for _ in range(8):
        if tuner is None or hasattr(tuner, "configs"):
            break
        tuner = getattr(tuner, "fn", None)
    configs = getattr(tuner, "configs", None)
    if not configs:
        raise RuntimeError("fla GDN backward autotuner is unavailable on Blackwell")
    validated = [
        config
        for config in configs
        if getattr(config, "num_warps", None) == 2
        and getattr(config, "num_stages", None) == 4
    ]
    if not validated:
        raise RuntimeError("fla GDN backward has no validated Blackwell autotune config")
    tuner.configs = validated


class FlashTokenizedSFTDataset:
    def __init__(
        self,
        dataset,
        tokenizer,
        max_length,
        strategy,
        input_template=None,
        pretrain_mode=False,
        num_processors=8,
        multiturn=False,
    ):
        del input_template, num_processors, multiturn
        if pretrain_mode:
            raise ValueError("flash OpenRLHF SFT does not use pretrain mode")
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = int(max_length)
        if len(dataset) <= 0:
            raise ValueError("flash OpenRLHF SFT dataset is empty")

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _list(value):
        if hasattr(value, "to_pylist"):
            value = value.to_pylist()
        elif hasattr(value, "tolist"):
            value = value.tolist()
        return [int(item) for item in value]

    @staticmethod
    def _multimodal_inputs(payload):
        if payload is None:
            return {}
        if hasattr(payload, "as_py"):
            payload = payload.as_py()
        payload = bytes(payload)
        if not payload:
            return {}
        import numpy as np
        import torch

        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            return {key: torch.from_numpy(arrays[key].copy()) for key in arrays.files}

    def __getitem__(self, index):
        import torch

        row = self.dataset[int(index)]
        input_ids = self._list(row["input_ids"])
        loss_mask = self._list(row["loss_mask"])
        if len(input_ids) != len(loss_mask):
            raise ValueError("input_ids and loss_mask must have identical lengths")
        if len(input_ids) > self.max_length:
            raise ValueError("pretokenized row exceeds data.max_len")
        if not any(loss_mask):
            raise ValueError("pretokenized row has no completion target")
        inputs = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        attention = torch.ones_like(inputs)
        mask = torch.tensor(loss_mask, dtype=torch.float32).unsqueeze(0)
        mm_inputs = self._multimodal_inputs(row.get("multimodal_inputs"))
        return inputs, attention, mask, mm_inputs

    def collate_fn(self, item_list):
        import torch
        from openrlhf.utils.utils import zero_pad_sequences

        inputs = zero_pad_sequences([item[0] for item in item_list], "right", self.tokenizer.pad_token_id)
        attention = zero_pad_sequences([item[1] for item in item_list], "right")
        masks = zero_pad_sequences([item[2] for item in item_list], "right")
        multimodal_rows = [item[3] for item in item_list]
        multimodal_keys = {key for row in multimodal_rows for key in row}
        mm_batch = {}
        for key in multimodal_keys:
            template = next(row[key] for row in multimodal_rows if key in row)
            values = [
                row[key] if key in row else template.narrow(0, 0, 0)
                for row in multimodal_rows
            ]
            mm_batch[key] = torch.cat(values, dim=0)
        return inputs, attention, masks, mm_batch


def _install_dataset_patch():
    import openrlhf.datasets

    openrlhf.datasets.SFTDataset = FlashTokenizedSFTDataset


def assert_lora_applied(model, model_id):
    count = sum(
        1
        for name, _ in model.named_modules()
        if name.endswith("lora_A.default") or name.endswith("lora_B.default")
    )
    if count == 0:
        raise RuntimeError(
            f"warm-start adapter for {model_id} loaded zero LoRA modules; the adapter was not applied"
        )
    return count


def assert_adapter_load_clean(load_result, model_id):
    def lora_only(keys):
        return [key for key in (keys or []) if "lora_" in key]

    missing = lora_only(getattr(load_result, "missing_keys", None))
    unexpected = lora_only(getattr(load_result, "unexpected_keys", None))
    if missing or unexpected:
        raise RuntimeError(
            f"warm-start adapter for {model_id} did not load cleanly: "
            f"missing={missing[:3]} unexpected={unexpected[:3]}"
        )


def assert_adapter_delta_nonzero(model, model_id):
    seen = 0
    nonzero = 0
    for name, module in model.named_modules():
        if not name.endswith("lora_B.default"):
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        seen += 1
        if bool(weight.detach().ne(0).any()):
            nonzero += 1
    if seen and nonzero == 0:
        raise RuntimeError(
            f"warm-start adapter for {model_id} has all-zero lora_B weights; "
            "the adapter delta is an identity no-op"
        )
    return nonzero


def _install_warmstart_actor_patch():
    warmstart = str(CONFIG.get("warmstart_adapter") or "")
    if not warmstart:
        return
    import openrlhf.models
    from openrlhf.models.actor import Actor as OriginalActor
    from peft import PeftModel

    class FlashWarmstartActor(OriginalActor):
        def __init__(self, pretrain_or_model, *args, lora_rank=0, **kwargs):
            if int(lora_rank) != int(CONFIG["lora_rank"]):
                raise ValueError("OpenRLHF warm-start LoRA rank changed before model construction")
            super().__init__(pretrain_or_model, *args, lora_rank=0, **kwargs)
            base = self.model
            model = PeftModel.from_pretrained(base, warmstart, is_trainable=True)
            model.enable_input_require_grads()
            key_mapping = getattr(base, "_checkpoint_conversion_mapping", None)
            load_result = model.load_adapter(
                warmstart,
                adapter_name="default",
                is_trainable=True,
                key_mapping=key_mapping,
            )
            model_id = str(CONFIG["model_id"])
            assert_adapter_load_clean(load_result, model_id)
            assert_lora_applied(model, model_id)
            assert_adapter_delta_nonzero(model, model_id)
            self.model = model

    openrlhf.models.Actor = FlashWarmstartActor


def _install_attention_patch():
    if not bool(CONFIG.get("force_cudnn_sdpa")):
        return
    import openrlhf.models
    from torch.nn.attention import SDPBackend, sdpa_kernel

    original_forward = openrlhf.models.Actor.forward

    def forward(self, *args, **kwargs):
        with sdpa_kernel(
            [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ],
            set_priority=True,
        ):
            return original_forward(self, *args, **kwargs)

    openrlhf.models.Actor.forward = forward


def _install_dataloader_and_scheduler_patches():
    from openrlhf.utils.deepspeed.deepspeed import DeepspeedStrategy

    original_setup = DeepspeedStrategy.setup_dataloader
    original_prepare = DeepspeedStrategy.prepare
    original_load_ckpt = DeepspeedStrategy.load_ckpt

    def setup_dataloader(self, *args, **kwargs):
        args = list(args)
        kwargs["drop_last"] = False
        if len(args) >= 3:
            args[2] = True
            kwargs.pop("pin_memory", None)
        else:
            kwargs["pin_memory"] = True
        return original_setup(self, *args, **kwargs)

    def prepare(self, *args):
        patched = []
        for arg in args:
            if isinstance(arg, tuple):
                model, cfg = arg
                cfg = dict(cfg)
                cfg["scheduler_steps"] = int(CONFIG["total_steps"])
                patched.append((model, cfg))
            else:
                patched.append(arg)
        return original_prepare(self, *patched)

    def load_ckpt(self, *args, **kwargs):
        load_path, states = original_load_ckpt(self, *args, **kwargs)
        CONFIG["_resume_states"] = states or {}
        return load_path, states

    DeepspeedStrategy.setup_dataloader = setup_dataloader
    DeepspeedStrategy.prepare = prepare
    DeepspeedStrategy.load_ckpt = load_ckpt


def _install_loraplus_patch():
    import deepspeed
    from peft.optimizers import create_loraplus_optimizer

    original_initialize = deepspeed.initialize

    def initialize(*args, **kwargs):
        if kwargs.get("optimizer") is not None:
            raise RuntimeError("flash OpenRLHF SFT expected DeepSpeed to construct no optimizer")
        model = kwargs.get("model")
        if model is None:
            raise RuntimeError("flash OpenRLHF SFT received no model for LoRA+")
        optimizer_name = str(CONFIG["optimizer_name"])
        if optimizer_name == "paged_adamw_8bit":
            import bitsandbytes as bnb

            optimizer_cls = bnb.optim.PagedAdamW8bit
        elif optimizer_name == "adamw":
            import torch

            optimizer_cls = torch.optim.AdamW
        else:
            raise ValueError(f"unknown flash OpenRLHF optimizer {optimizer_name!r}")
        optimizer_kwargs = {
            "lr": float(CONFIG["learning_rate"]),
            "betas": tuple(float(value) for value in CONFIG["adam_betas"]),
            "eps": float(CONFIG["adam_epsilon"]),
            "weight_decay": float(CONFIG["weight_decay"]),
        }
        try:
            optimizer = create_loraplus_optimizer(
                model=model,
                optimizer_cls=optimizer_cls,
                optimizer_kwargs=optimizer_kwargs,
                loraplus_lr_ratio=float(CONFIG["loraplus_ratio"]),
                loraplus_weight_decay=float(CONFIG["weight_decay"]),
            )
        except TypeError:
            optimizer = create_loraplus_optimizer(
                model=model,
                optimizer_cls=optimizer_cls,
                lr=float(CONFIG["learning_rate"]),
                loraplus_lr_ratio=float(CONFIG["loraplus_ratio"]),
                loraplus_weight_decay=float(CONFIG["weight_decay"]),
                betas=tuple(float(value) for value in CONFIG["adam_betas"]),
                eps=float(CONFIG["adam_epsilon"]),
                weight_decay=float(CONFIG["weight_decay"]),
            )
        config = dict(kwargs["config"])
        config.pop("optimizer", None)
        config["zero_allow_untested_optimizer"] = True
        kwargs["config"] = config
        kwargs["optimizer"] = optimizer
        kwargs["model_parameters"] = None
        print(
            "FLASH_OPENRLHF_LORAPLUS_READY "
            f"ratio={float(CONFIG['loraplus_ratio']):g} optimizer={optimizer_cls.__name__}",
            flush=True,
        )
        return original_initialize(*args, **kwargs)

    deepspeed.initialize = initialize


def _global_count(local_count, device):
    import torch
    import torch.distributed as dist

    value = torch.tensor(float(local_count), device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return int(value.item())


def _accumulation_window_scale(configured_size, realized_size):
    configured_size = max(1, int(configured_size))
    realized_size = int(realized_size)
    if realized_size <= 0 or realized_size > configured_size:
        raise ValueError("invalid gradient accumulation window size")
    return configured_size / realized_size


def _iter_windows(iterable, size):
    window = []
    for item in iterable:
        window.append(item)
        if len(window) == size:
            yield window
            window = []
    if window:
        yield window


def _mark_hf_export_ready(checkpoint_dir, step):
    marker = os.path.join(checkpoint_dir, f".flash-hf-step-{step}.ready")
    temporary = f"{marker}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        file.write("ready\n")
    os.replace(temporary, marker)


def _wait_for_checkpoint_upload(checkpoint_dir, step):
    import time

    done = os.path.join(checkpoint_dir, f".flash-upload-step-{step}.done")
    failed = os.path.join(checkpoint_dir, f".flash-upload-step-{step}.failed")
    while not os.path.isfile(done):
        if os.path.isfile(failed):
            with open(failed, encoding="utf-8") as file:
                detail = file.read().strip()
            raise RuntimeError(f"save step {step} upload failed: {detail}")
        time.sleep(0.25)


def _resume_training_state(consumed_samples):
    resume_states = dict(CONFIG.get("_resume_states") or {})
    global_step = int(CONFIG.get("resume_step", 0))
    state_step = int(resume_states.get("global_step", global_step) or 0)
    if state_step != global_step:
        raise RuntimeError("OpenRLHF resume checkpoint step does not match its directory tag")
    loss_curve = [float(value) for value in resume_states.get("loss_curve", [])]
    token_count = int(resume_states.get("token_count", 0) or 0)
    restored_samples = int(resume_states.get("consumed_samples", consumed_samples) or 0)
    return global_step, restored_samples, loss_curve, token_count


def _install_trainer_patch():
    import torch
    import torch.distributed as dist
    from openrlhf.models import SFTLoss
    from openrlhf.trainer.sft_trainer import SFTTrainer
    from openrlhf.utils.distributed_sampler import DistributedSampler
    from openrlhf.utils.loss_utils import _optimizer_step_loss_norm

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        del num_update_steps_per_epoch
        total_steps = int(CONFIG["total_steps"])
        global_step, consumed_total, loss_curve, token_count = _resume_training_state(
            consumed_samples
        )
        gas = max(1, int(self.strategy.accumulated_gradient))
        sampler = getattr(self.train_dataloader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            epoch_samples = int(sampler.total_size)
        else:
            epoch_samples = len(self.train_dataloader.dataset)
        epoch_samples = max(1, epoch_samples)
        start_epoch = consumed_total // epoch_samples
        consumed_in_epoch = consumed_total % epoch_samples
        loss_fn = SFTLoss()
        self.model.train()
        device = next(self.model.parameters()).device

        for epoch in range(start_epoch, self.epochs):
            if global_step >= total_steps:
                break
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch, consumed_samples=consumed_in_epoch if epoch == start_epoch else 0)
            for window in _iter_windows(self.train_dataloader, gas):
                if global_step >= total_steps:
                    break
                window_size = len(window)
                masks = [batch[2].squeeze(1)[:, 1:] for batch in window]
                dp_group = self.strategy.ds_device_mesh["dp"].get_group()
                dp_size = dist.get_world_size(group=dp_group)
                loss_info = _optimizer_step_loss_norm(
                    masks, dp_group, dp_size, window_size
                )
                gradient_scale = _accumulation_window_scale(gas, window_size)
                step_loss = 0.0
                step_tokens = 0
                step_examples = 0
                engine = self.model.model
                if not hasattr(engine, "set_gradient_accumulation_boundary"):
                    raise RuntimeError(
                        "OpenRLHF DeepSpeed cannot preserve exact SFT update boundaries"
                    )
                for micro_index, batch in enumerate(window):
                    inputs, attention_masks, loss_masks, mm_inputs = batch
                    inputs = inputs.to(device, non_blocking=True).squeeze(1)
                    attention_mask = attention_masks.to(device, non_blocking=True).squeeze(1)
                    loss_mask = loss_masks.to(device, non_blocking=True).squeeze(1)
                    mm_inputs = {
                        key: value.to(device, non_blocking=True)
                        for key, value in mm_inputs.items()
                    }
                    engine.set_gradient_accumulation_boundary(micro_index + 1 == len(window))
                    per_token_log_probs, output = self.model(
                        inputs,
                        attention_mask=attention_mask,
                        return_output=True,
                        return_logprobs=True,
                        ring_attn_group=self.strategy.ring_attn_group,
                        **mm_inputs,
                    )
                    aux_loss = output.aux_loss if self.aux_loss else 0
                    gpt_loss = loss_fn(per_token_log_probs, loss_mask[:, 1:], **loss_info)
                    loss = (
                        gpt_loss + aux_loss * self.args.model.aux_loss_coef
                    ) * gradient_scale
                    self.strategy.backward(loss, self.model, self.optimizer)
                    self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                    step_loss += float(gpt_loss.item())
                    step_tokens += int(attention_mask.sum().item())
                    step_examples += int(inputs.shape[0])
                global_step += 1
                global_examples = _global_count(step_examples, device)
                global_tokens = _global_count(step_tokens, device)
                consumed_total += global_examples
                consumed_in_epoch += global_examples
                token_count += global_tokens
                logs = self.strategy.all_reduce(
                    {
                        "gpt_loss": step_loss / window_size,
                        "lr": self.scheduler.get_last_lr()[0],
                        "grad_norm": self.strategy.get_grad_norm(self.model),
                    }
                )
                loss_curve.append(float(logs["gpt_loss"]))
                if self._wandb is not None and self.strategy.is_rank_0():
                    self._wandb.log(
                        {
                            "train/global_step": global_step,
                            "train/gpt_loss": logs["gpt_loss"],
                            "train/loss_mean": logs["gpt_loss"],
                            "train/lr": logs["lr"],
                            "train/grad_norm": logs["grad_norm"],
                            "train/num_tokens": token_count,
                        }
                    )
                if self.strategy.is_rank_0():
                    payload = {
                        "step": global_step,
                        "loss": float(logs["gpt_loss"]),
                        "grad_norm": float(logs["grad_norm"]),
                        "lr": float(logs["lr"]),
                        "tokens": token_count,
                    }
                    print("FLASH_OPENRLHF_SFT_STEP=" + json.dumps(payload, sort_keys=True), flush=True)
                required = {int(step) for step in CONFIG["save_at_steps"]}
                save_due = global_step in required if required else global_step % int(CONFIG["save_every"]) == 0
                if save_due:
                    client_state = {
                        "consumed_samples": consumed_total,
                        "global_step": global_step,
                        "loss_curve": loss_curve,
                        "token_count": token_count,
                    }
                    tag = f"global_step{global_step}"
                    self.strategy.save_ckpt(
                        self.model.model,
                        args.ckpt.path,
                        tag,
                        2**31 - 1,
                        float("inf"),
                        client_state,
                    )
                    self.strategy.save_model(
                        self.model,
                        self.tokenizer,
                        os.path.join(args.ckpt.path, f"{tag}_hf"),
                    )
                    if self.strategy.is_rank_0():
                        _mark_hf_export_ready(args.ckpt.path, global_step)
                    _wait_for_checkpoint_upload(args.ckpt.path, global_step)
            consumed_in_epoch = 0

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()
        if self.strategy.is_rank_0():
            state = {
                "step": global_step,
                "consumed_samples": consumed_total,
                "tokens": token_count,
                "loss_curve": loss_curve,
            }
            with open(CONFIG["final_state_path"], "w", encoding="utf-8") as state_file:
                json.dump(state, state_file, sort_keys=True)

    SFTTrainer.fit = fit


def apply_flash_openrlhf_sft_patches():
    _apply_blackwell_fla_safety()
    _install_dataset_patch()
    _install_warmstart_actor_patch()
    _install_attention_patch()
    _install_dataloader_and_scheduler_patches()
    _install_loraplus_patch()
    _install_trainer_patch()
"""


def _probe_gpu_in_subprocess(
    python_bin: str, requested_gpu: str | None, exact_type: str = ""
) -> dict:
    script = r"""
import json
import sys

from flash.engine.worker.perf.lifecycle import wait_for_gpu

requested, exact = json.loads(sys.argv[1])
wait_for_gpu(requested, exact_type=exact)
import torch
try:
    from transformers.utils import is_flash_attn_2_available, is_flash_attn_3_available
    fa2_available = bool(is_flash_attn_2_available())
    fa3_available = bool(is_flash_attn_3_available())
except Exception:
    fa2_available = False
    fa3_available = False
print("FLASH_GPU_PROBE=" + json.dumps({
    "name": torch.cuda.get_device_name(0),
    "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
    "capability": list(torch.cuda.get_device_capability(0)),
    "fa2_available": fa2_available,
    "fa3_available": fa3_available,
}), flush=True)
"""
    try:
        result = subprocess.run(
            [python_bin, "-c", script, json.dumps([requested_gpu, exact_type])],
            capture_output=True,
            text=True,
            timeout=150,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise _w.RetriableInfraError("gpu readiness probe failed in its subprocess") from error
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    if result.returncode != 0:
        raise _w.RetriableInfraError(f"gpu readiness probe exited with status {result.returncode}")
    for line in result.stdout.splitlines():
        if line.startswith("FLASH_GPU_PROBE="):
            return json.loads(line.split("=", 1)[1])
    raise _w.RetriableInfraError("gpu readiness probe returned no device metadata")


class _NvidiaSmiPeakSampler:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak_mib = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=8,
                )
                if result.returncode == 0:
                    values = [
                        float(value.strip())
                        for value in result.stdout.splitlines()
                        if value.strip()
                    ]
                    if values:
                        self.peak_mib = max(self.peak_mib, max(values))
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass

    def start(self):
        self._thread.start()
        return self

    def stop_gb(self) -> float:
        self._stop.set()
        self._thread.join(timeout=2)
        return round(self.peak_mib / 1024, 3)


def _parse_step_line(line: str) -> dict | None:
    index = line.find(_STEP_PREFIX)
    if index < 0:
        return None
    payload = line[index + len(_STEP_PREFIX) :].strip()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _attention_implementation(gpu_probe: dict) -> str:
    capability = tuple(gpu_probe.get("capability") or ())
    if len(capability) != 2:
        return "sdpa"
    implementation = _attn_impl_for_capability(
        int(capability[0]),
        int(capability[1]),
        fa2_available=bool(gpu_probe.get("fa2_available")),
        fa3_available=bool(gpu_probe.get("fa3_available")),
    )
    return implementation or "sdpa"


def _training_batch_shape(
    *,
    row_count: int,
    effective_batch: int,
    per_device_limit: int,
    gpu_count: int,
) -> tuple[int, int, int]:
    gpu_count = max(1, int(gpu_count))
    local_rows = max(1, math.ceil(int(row_count) / gpu_count))
    micro_batch = max(1, min(int(per_device_limit), local_rows))
    max_accum = max(1, math.ceil(local_rows / micro_batch))
    accumulation = min(
        max_accum,
        max(1, math.ceil(int(effective_batch) / (micro_batch * gpu_count))),
    )
    train_batch = micro_batch * gpu_count * accumulation
    return micro_batch, accumulation, train_batch


def run_sft_openrlhf(spec=None) -> None:
    """run flash sft through OpenRLHF's DeepSpeed SFT trainer."""
    from flash.catalog import MODELS, resolve_vocab_size
    from flash.engine.vram import sft_grad_accum

    spec = spec or _w.JOB_SPEC
    env = _w.require_active_env()
    started_at = time.time()
    _w.heartbeat("sft_start", gpu=_w.gpu_diagnostics(include_torch=False))
    python_bin = resolve_openrlhf_python("")
    gpu_probe = _probe_gpu_in_subprocess(
        python_bin,
        spec.gpu.type if spec else None,
        exact_type=spec.gpu.exact_type if spec else "",
    )

    model_id = spec.model if spec else RECIPE.hf_model_id
    requested_model_revision = getattr(spec, "model_revision", "") if spec else ""
    download_seconds = _w.prefetch_model(model_id, revision=requested_model_revision)
    model_revision = _resolve_immutable_model_revision(model_id, requested_model_revision)
    train_spec = spec.train if spec else None

    def train_opt(name, default):
        value = getattr(train_spec, name, None) if train_spec else None
        return value if value is not None else default

    max_length = int(
        train_opt(
            "max_context_tokens",
            RECIPE.sft.max_seq_len_thinking if _w.THINKING else RECIPE.sft.max_seq_len,
        )
    )
    epochs = int(train_opt("epochs", RECIPE.sft.num_epochs))
    learning_rate = float(train_opt("learning_rate", RECIPE.sft.learning_rate))
    effective_batch = int(train_opt("batch_size", RECIPE.sft.effective_batch))
    max_examples = int(train_opt("max_examples", 0) or 0)
    max_steps = int(train_opt("max_steps", 0) or 0)
    save_at_steps = tuple(getattr(train_spec, "save_at_steps", ()) or ())
    save_every = int(train_opt("save_every", 50))
    gpu_count = int(getattr(getattr(spec, "gpu", None), "count", 1) or 1)

    workdir = os.path.join("/tmp", "flash-sft-openrlhf", _w.RUN_ID, f"seed-{_w.SEED}")
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir, exist_ok=True)
    dataset_path = os.path.join(workdir, "dataset")
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    output_dir = os.path.join(workdir, "output")
    export_root = os.path.join(workdir, "checkpoint-adapters")
    processing_dir = os.path.join(workdir, "processing")
    shim_dir = os.path.join(workdir, "shim")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(export_root, exist_ok=True)
    os.makedirs(processing_dir, exist_ok=True)
    os.makedirs(shim_dir, exist_ok=True)

    with liveness_heartbeat("sft_data_loading"):
        from transformers import AutoProcessor

        from flash.multimodal import (
            decode_image_descriptors,
            normalize_prompt_images,
            record_has_images,
            text_only_prompt_messages,
            validate_multimodal_training,
        )

        indexed_train = _select_indexed_sft_examples(env.dataset(), max_examples, _w.SEED)
        selected = [example for _, example in indexed_train]
        prefix_indices = [index for index, _ in indexed_train]
        prompt_rows = [
            (example, env.prompt_messages(example), env.sft_completion(example))
            for example in selected
        ]
        package_root = getattr(env, "package_root", None)
        multimodal = any(
            record_has_images(example, prompt_messages)
            for example, prompt_messages, _completion in prompt_rows
        )
        processor = None
        if multimodal:
            validate_multimodal_training(model_id, "sft")
            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                **_w.model_revision_kwargs(model_revision),
            )
            tokenizer = processor.tokenizer
        else:
            tokenizer = _w.load_tokenizer(model_id, revision=model_revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        (processor or tokenizer).save_pretrained(processing_dir)

        for _example, _prompt, completion in prompt_rows:
            _reject_image_completion(completion)

        rows: list[dict] = []
        dropped = 0
        sampled_texts: list[str] = []
        multiturn_targets = sum(
            1 for _example, _prompt, completion in prompt_rows if len(completion) > 1
        )
        if not multimodal:
            texts, tokenized, dropped, _multiturn, cache_hit = _prepare_sft_examples(
                env,
                selected,
                tokenizer,
                env_resolved_sha=(spec.environment.resolved_sha if spec else ""),
                seed=_w.SEED,
                model_revision=model_revision,
                thinking=_w.THINKING,
                max_length=max_length,
                prefix_indices=prefix_indices,
                prompt_completion_rows=[
                    (prompt_messages, completion)
                    for _example, prompt_messages, completion in prompt_rows
                ],
            )
            print(f"[sft] tokenized data cache: {'hit' if cache_hit else 'miss'}")
            sampled_texts = [row["text"] for row in texts]
            rows = [
                {
                    "input_ids": row["input_ids"],
                    "loss_mask": row["completion_mask"],
                    "multimodal_inputs": b"",
                }
                for row in tokenized
            ]
        else:
            row_by_index: dict[int, dict] = {}
            text_specs: list[dict] = []
            for row_index, (example, prompt_messages, completion) in enumerate(prompt_rows):
                if record_has_images(example, prompt_messages):
                    assert processor is not None
                    normalized = normalize_prompt_images(example, prompt_messages, package_root)
                    completion = text_only_prompt_messages(completion)
                    decoded_images = decode_image_descriptors(normalized.descriptors, package_root)
                    input_ids, loss_mask, mm_inputs = _processor_tokenized_row(
                        processor,
                        normalized.messages,
                        completion,
                        decoded_images,
                        max_length=max_length,
                        thinking=bool(_w.THINKING),
                    )
                    row_by_index[row_index] = {
                        "input_ids": input_ids,
                        "loss_mask": loss_mask,
                        "multimodal_inputs": mm_inputs,
                    }
                    sampled_texts.append(
                        tokenizer.apply_chat_template(
                            [*normalized.messages, *completion],
                            tokenize=False,
                            add_generation_prompt=False,
                            enable_thinking=_w.THINKING,
                        )
                    )
                else:
                    text = tokenizer.apply_chat_template(
                        [*prompt_messages, *completion],
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=_w.THINKING,
                    )
                    prompt_text = tokenizer.apply_chat_template(
                        prompt_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=_w.THINKING,
                    )
                    sampled_texts.append(text)
                    text_specs.append(
                        {"text": text, "prompt_text": prompt_text, "row_index": row_index}
                    )
            if text_specs:
                kept_specs, tokenized, text_dropped = _pretokenize_completion_only(
                    text_specs, tokenizer, max_length
                )
                dropped += text_dropped
                for spec_row, tokenized_row in zip(kept_specs, tokenized, strict=True):
                    row_by_index[spec_row["row_index"]] = {
                        "input_ids": tokenized_row["input_ids"],
                        "loss_mask": tokenized_row["completion_mask"],
                        "multimodal_inputs": b"",
                    }
            rows = [row_by_index[index] for index in sorted(row_by_index)]

        rows, extra_dropped = filter_openrlhf_sft_rows(
            rows,
            set(getattr(tokenizer, "all_special_ids", None) or []),
        )
        dropped += extra_dropped
        if dropped:
            print(
                f"[sft] dropped {dropped} rows with no real completion target "
                "(sft_max_len truncated away the whole completion, or it was content-free)"
            )
        if multiturn_targets:
            print(
                f"[sft] multi-turn SFT: {multiturn_targets}/{len(selected)} rows train on a full "
                "target transcript"
            )
        elif getattr(env, "multi_turn", False):
            print(
                "[sft][warn] this is a multi-turn environment but no row ships a multi-turn "
                "target completion"
            )
        if _w.THINKING and not any("<think>" in text for text in sampled_texts[:256]):
            print(
                "WARN: thinking mode is ON but no sampled SFT target contains a <think> trace; "
                "training on non-reasoning targets teaches the model to skip thinking"
            )
        masked_tokens = sum(row["loss_mask"].count(0) for row in rows)
        total_tokens_per_epoch = sum(len(row["input_ids"]) for row in rows)
        realized_max_length = max(len(row["input_ids"]) for row in rows)
        runtime_max_length = realized_max_length
        _validate_gdn_realized_length(model_id, model_revision, realized_max_length)
        print(
            f"[sft] completion-only loss: masking {masked_tokens}/{total_tokens_per_epoch} "
            f"({masked_tokens / total_tokens_per_epoch:.0%}) prompt tokens"
        )
        _write_openrlhf_dataset(rows, dataset_path)

    setup_seconds = time.time() - started_at
    _w.heartbeat(
        "sft_model_load",
        setup_seconds=setup_seconds,
        gpu=_w.gpu_diagnostics(include_torch=False),
    )

    lora_config = _w.make_lora(model_id)
    lora_rank = int(lora_config.r)
    lora_alpha = int(lora_config.lora_alpha)
    warmstart_adapter = _warmstart_adapter_path(model_id, model_revision, lora_rank)

    vocab_size = resolve_vocab_size(model_id, model_revision)
    per_device_limit, _unused_grad_accum = sft_grad_accum(
        effective_batch,
        seq_len=runtime_max_length,
        vocab=vocab_size,
        fused=False,
    )
    micro_batch, gradient_accumulation, train_batch_size = _training_batch_shape(
        row_count=len(rows),
        effective_batch=effective_batch,
        per_device_limit=per_device_limit,
        gpu_count=gpu_count,
    )
    derived_steps = sft_update_steps(
        epochs=epochs,
        example_count=len(rows),
        examples_per_update=train_batch_size,
    )
    update_horizon = resolve_update_horizon(derived_steps, max_steps)
    validate_save_steps(save_at_steps, update_horizon)
    steps_per_epoch = max(1, math.ceil(len(rows) / train_batch_size))
    loop_epochs = max(epochs, math.ceil(update_horizon / steps_per_epoch))

    card_vram_gb = float(gpu_probe.get("memory_gb") or 0.0)
    raw_capability = gpu_probe.get("capability")
    capability = tuple(raw_capability) if raw_capability else None
    hidden, layers = _model_arch_dims(model_id, revision=model_revision)
    info = MODELS.get(model_id)
    active_params_b = float(getattr(info, "active_params_b", 0.0) or 0.0) or None
    gradient_checkpointing = _w.grad_checkpointing_on(
        model_id,
        runtime_max_length,
        allow_disable=True,
        card_vram_gb=card_vram_gb,
        capability=capability,
        active_params_b=active_params_b,
        hidden=hidden,
        num_layers=layers,
        fused_ce=False,
        per_device_bs=micro_batch,
        lora_rank=lora_rank,
        revision=model_revision,
    )
    reentrant_gradient_checkpointing = bool(
        gradient_checkpointing and _w.grpo_use_reentrant(model_id)
    )

    model_path = _cached_model_path(model_id, model_revision)
    resume_step = _restore_openrlhf_resume(checkpoint_dir)
    if resume_step > update_horizon:
        print(
            f"[sft] resume checkpoint step {resume_step} is past the requested horizon "
            f"{update_horizon}; loading it and performing no new updates"
        )

    wandb_enabled = bool(os.environ.get("WANDB_API_KEY"))
    wandb_project = (spec.wandb.project if spec and spec.wandb else None) or "flash"
    wandb_run_name = _w.wandb_run_name()
    final_state_path = os.path.join(workdir, _FINAL_STATE_FILE)
    attn_implementation = _attention_implementation(gpu_probe)
    capability = tuple(gpu_probe.get("capability") or ())
    runtime_config = {
        "adam_betas": [0.9, 0.999],
        "adam_epsilon": 1e-8,
        "attn_implementation": attn_implementation,
        "force_cudnn_sdpa": bool(capability and int(capability[0]) in (10, 12)),
        "final_state_path": final_state_path,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "loraplus_ratio": _SFT_LORAPLUS_RATIO,
        "model_id": model_id,
        "optimizer_name": "paged_adamw_8bit",
        "resume_step": resume_step,
        "save_at_steps": list(save_at_steps),
        "save_every": save_every,
        "total_steps": update_horizon,
        "warmstart_adapter": warmstart_adapter,
        "weight_decay": 0.0,
    }
    with open(os.path.join(shim_dir, _RUNTIME_CONFIG_FILE), "w", encoding="utf-8") as file:
        json.dump(runtime_config, file, sort_keys=True)
    with open(
        os.path.join(shim_dir, "flash_openrlhf_sft_runtime.py"), "w", encoding="utf-8"
    ) as file:
        file.write(render_openrlhf_sft_runtime())
    with open(os.path.join(shim_dir, "sitecustomize.py"), "w", encoding="utf-8") as file:
        file.write(_render_sitecustomize())

    arg_config = {
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "attn_implementation": attn_implementation,
        "checkpoint_dir": checkpoint_dir,
        "dataset_path": dataset_path,
        "epochs": loop_epochs,
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_checkpointing_reentrant": reentrant_gradient_checkpointing,
        "learning_rate": learning_rate,
        "lora_alpha": lora_alpha,
        "lora_rank": lora_rank,
        "max_length": max_length,
        "max_num_checkpoints": 1,
        "max_norm": 1.0,
        "micro_batch_size": micro_batch,
        "model_path": model_path,
        "num_workers": 4,
        "output_dir": output_dir,
        "resume_enabled": resume_step > 0,
        "row_count": len(rows),
        "seed": _w.backend_seed(_w.SEED),
        "train_batch_size": train_batch_size,
        "wandb_enabled": wandb_enabled,
        "wandb_group": None,
        "wandb_org": None,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "warmup_ratio": RECIPE.sft.warmup_frac,
        "weight_decay": 0.0,
    }
    entry_args = build_sft_openrlhf_args(arg_config)
    child_env = build_openrlhf_sft_child_env(
        shim_dir=shim_dir,
        wandb_enabled=wandb_enabled,
    )

    watcher = _OpenRLHFCheckpointWatcher(
        checkpoint_dir=checkpoint_dir,
        export_root=export_root,
        processing_dir=processing_dir,
        python_bin=python_bin,
        model_id=model_id,
        model_revision=model_revision,
        required_steps=save_at_steps,
        max_num_checkpoints=arg_config["max_num_checkpoints"],
    )
    watcher.processed_steps.update(_durable_required_save_steps(save_at_steps, resume_step))

    progress = {"step": resume_step, "loss": None, "grad_norm": None, "lr": None}
    loss_curve: list[float] = []
    loraplus_applied = False

    def on_line(line: str) -> None:
        nonlocal loraplus_applied
        watcher.raise_if_failed()
        if _LORAPLUS_READY_MARKER in line:
            loraplus_applied = True
        payload = _parse_step_line(line)
        if payload is None:
            return
        if not loraplus_applied:
            raise RuntimeError("OpenRLHF reached an optimizer step before LoRA+ was installed")
        progress["step"] = int(payload.get("step") or progress["step"] or 0)
        for name in ("loss", "grad_norm", "lr"):
            if payload.get(name) is not None:
                progress[name] = float(payload[name])
        if progress["loss"] is not None:
            loss_curve.append(round(float(progress["loss"]), 4))

    def on_step(step: int) -> None:
        progress["step"] = step
        payload = {
            "step": step,
            "loss": progress["loss"],
            "grad_norm": progress["grad_norm"],
            "learning_rate": progress["lr"],
        }
        _w.heartbeat(
            "sft_step", **{key: value for key, value in payload.items() if value is not None}
        )

    def child_heartbeat() -> None:
        watcher.raise_if_failed()
        _w.heartbeat("sft_step", liveness=True, step=int(progress["step"] or 0))

    gpu_sampler = _NvidiaSmiPeakSampler().start()
    train_started_at = time.time()
    watcher.start()
    return_code = -1
    try:
        with liveness_heartbeat(
            "sft_step",
            progress=lambda: int(progress["step"] or 0),
            progress_step=True,
        ):
            return_code = run_openrlhf_training(
                python_bin,
                entry_args,
                env=child_env,
                entrypoint="openrlhf.cli.train_sft",
                on_step=on_step,
                on_line=on_line,
                heartbeat=child_heartbeat,
                step_pattern=r"FLASH_OPENRLHF_SFT_STEP=.*?\"step\":\s*(\d+)",
                torchrun_args=[
                    "--standalone",
                    "--nnodes=1",
                    f"--nproc-per-node={gpu_count}",
                ],
            )
    finally:
        try:
            watcher.stop(require_complete=return_code == 0)
        finally:
            device_peak_gpu_gb = gpu_sampler.stop_gb()
    train_wall = time.time() - train_started_at
    if return_code != 0:
        raise RuntimeError(f"OpenRLHF SFT subprocess exited with status {return_code}")
    if not loraplus_applied:
        raise RuntimeError("required OpenRLHF LoRA+ shim did not emit its success marker")
    try:
        with open(final_state_path, encoding="utf-8") as file:
            final_state = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("OpenRLHF SFT did not write its final training state") from error
    final_step = int(final_state.get("step") or 0)
    if sft_under_ran(final_step, update_horizon, max_steps):
        raise RuntimeError(
            f"sft completed {final_step}/{update_horizon} requested optimizer updates"
        )
    if final_step < update_horizon:
        raise RuntimeError(f"sft completed {final_step}/{update_horizon} optimizer updates")
    state_loss_curve = [round(float(value), 4) for value in final_state.get("loss_curve", [])]
    if state_loss_curve:
        loss_curve = state_loss_curve

    train_tokens = sft_completed_train_tokens(
        total_tokens_per_epoch,
        epochs,
        derived_steps,
        final_step,
    )
    with liveness_heartbeat(
        "sft_finalizing",
        progress=lambda: final_step,
        progress_step=True,
        keepalive=True,
    ):
        adapter_dir = os.path.join(workdir, "adapter")
        _export_checkpoint_adapter(
            output_dir,
            adapter_dir,
            processing_dir=processing_dir,
            model_id=model_id,
            model_revision=model_revision,
            python_bin=python_bin,
        )
        _w.hf_upload_folder(adapter_dir, "adapter", required=True)
        if final_save_due(final_step, save_at_steps) and final_step not in watcher.processed_steps:
            _w.publish_deployable_checkpoint(adapter_dir, final_step)

    _w.heartbeat(
        "sft_trained",
        train_wall=train_wall,
        step=final_step,
        gpu=_w.gpu_diagnostics(include_torch=False),
    )
    _w.write_train_meta(
        phase="sft",
        adapter_dir=adapter_dir,
        model_id=model_id,
        train_wall=train_wall,
        setup_seconds=setup_seconds,
        train_tokens=train_tokens,
        generated_tokens=0,
        step=final_step,
        notes={
            "epochs": epochs,
            "resumed": bool(resume_step),
            "warm_started": bool(warmstart_adapter),
            "download_seconds": download_seconds,
            "hf_transfer": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", ""),
            "thinking": _w.THINKING,
            "multimodal": multimodal,
            "gradient_checkpointing": gradient_checkpointing,
            "gradient_checkpointing_reentrant": reentrant_gradient_checkpointing,
            "configured_max_length": max_length,
            "realized_max_length": realized_max_length,
            "runtime_max_length": runtime_max_length,
            "per_device_train_batch_size": micro_batch,
            "gradient_accumulation_steps": gradient_accumulation,
            "packing": "unpacked_openrlhf",
            "loss_curve": loss_curve[:400],
            "peak_gpu_gb": device_peak_gpu_gb,
            "device_peak_gpu_gb": device_peak_gpu_gb,
            "loraplus_optim": "PagedAdamW8bit",
            "loraplus_applied": loraplus_applied,
            "chalk_kernels": None,
            "openrlhf_backend": "deepspeed_zero3",
            "wandb_project": wandb_project if wandb_enabled else None,
            "wandb_run_name": wandb_run_name if wandb_enabled else None,
        },
    )
