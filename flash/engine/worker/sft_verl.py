"""sft training via verl in a separate interpreter.

flash prepares the exact whole-conversation token ids and completion-only loss mask, writes them to
parquet, and injects a custom verl dataset that returns those tensors verbatim. the parent streams
progress and checkpoints without holding a cuda context while torchrun owns the training devices.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from functools import reduce
from math import gcd
from pathlib import Path

from flash.engine.recipe import RECIPE
from flash.engine.steps import (
    final_save_due,
    resolve_update_horizon,
    sft_update_steps,
    validate_save_steps,
)
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.heartbeat import liveness_heartbeat
from flash.engine.worker.packing import completion_mask_from_ids
from flash.engine.worker.sft import (
    _model_arch_dims,
    _pretokenize_completion_only,
    _reject_image_completion,
    _select_indexed_sft_examples,
    sft_completed_train_tokens,
    sft_under_ran,
)
from flash.engine.worker.verl_common import (
    export_peft_adapter,
    latest_global_step_dir,
    resolve_verl_python,
    run_verl_training,
    stamp_adapter_dir_provenance,
)

# todo: run the two-gpu sft smoke on the exact runpod image and command assembled below.
_SFT_LORAPLUS_RATIO = 16.0
_LORAPLUS_READY_MARKER = "FLASH_LORAPLUS_READY"
_VERL_STEP_RE = re.compile(r"(?:^|\s)step:(\d+)(?:\s|$)")
_VERL_METRIC_RE = re.compile(r"(?:^| - )(?P<name>[^:]+):(?P<value>[^ ]+)")

_REQUIRED_OVERRIDE_KEYS = (
    "train_files",
    "val_files",
    "train_batch_size",
    "max_length",
    "micro_batch",
    "max_token_len_per_gpu",
    "custom_dataset_path",
    "model_path",
    "lora_rank",
    "lora_alpha",
    "target_modules",
    "ulysses_sp_size",
    "lr",
    "warmup_ratio",
    "optimizer_impl",
    "optimizer_name",
    "local_dir",
    "save_freq",
    "n_gpus_per_node",
    "seed",
    "project_name",
    "experiment_name",
    "loop_epochs",
)


def _hydra_val(value) -> str:
    """render a python value as a hydra override rhs."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # fixed-point keeps hydra off scientific notation, but 12 places truncates lrs below
        # 1e-12 to zero; fall back to repr for those (hydra parses e-notation fine when quoted).
        if value != 0 and abs(value) < 1e-12:
            return repr(value)
        rendered = f"{value:.12f}".rstrip("0")
        return rendered + "0" if rendered.endswith(".") else rendered
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{_hydra_val(item)}" for key, item in value.items()) + "}"
    if isinstance(value, (list, tuple, set, frozenset)):
        return "[" + ",".join(_hydra_val(item) for item in value) + "]"
    text = str(value)
    # quote strings containing hydra/shell-special characters so paths and ids survive parsing
    # (e.g. commas or '=' in a dataset path would split the override).
    if any(ch in text for ch in (",", "=", " ", "[", "]", "{", "}", "(", ")", ":", "'", '"')):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def _optimizer_override_config(cfg: dict) -> dict:
    override = dict(cfg.get("optimizer_kwargs") or {})
    override.setdefault("eps", cfg.get("eps", 1e-8))
    return override


def build_sft_verl_overrides(cfg: dict) -> list[str]:
    """build hydra overrides for verl's ``sft_trainer_engine.yaml`` config."""
    missing = [key for key in _REQUIRED_OVERRIDE_KEYS if key not in cfg]
    if missing:
        raise KeyError(f"build_sft_verl_overrides missing required cfg keys: {missing}")
    steps = cfg.get("total_training_steps")
    epochs = cfg.get("total_epochs")
    if bool(steps) == bool(epochs):
        raise ValueError("set exactly one of cfg['total_training_steps'] or cfg['total_epochs']")
    optimizer_override = _optimizer_override_config(cfg)

    overrides = [
        f"data.train_files={_hydra_val(cfg['train_files'])}",
        f"data.val_files={_hydra_val(cfg['val_files'])}",
        f"data.train_batch_size={_hydra_val(cfg['train_batch_size'])}",
        f"data.max_length={_hydra_val(cfg['max_length'])}",
        f"data.micro_batch_size_per_gpu={_hydra_val(cfg['micro_batch'])}",
        "data.use_dynamic_bsz=true",
        f"data.max_token_len_per_gpu={_hydra_val(cfg['max_token_len_per_gpu'])}",
        "data.truncation=right",
        f"data.num_workers={_hydra_val(cfg.get('num_workers', 4))}",
        "data.ignore_input_ids_mismatch=false",
        f"data.custom_cls.path={_hydra_val(cfg['custom_dataset_path'])}",
        "data.custom_cls.name=FlashTokenizedSFTDataset",
        f"model.path={_hydra_val(cfg['model_path'])}",
        "model.trust_remote_code=true",
        f"model.lora_rank={_hydra_val(cfg['lora_rank'])}",
        f"model.lora_alpha={_hydra_val(cfg['lora_alpha'])}",
        f"model.target_modules={_hydra_val(cfg['target_modules'])}",
        f"model.lora_adapter_path={_hydra_val(cfg.get('lora_adapter_path'))}",
        "model.use_remove_padding=true",
        # 32k contexts: the fused linear-CE forward never materializes the [tokens, vocab] logits
        # tensor (~130 GB at 32k on a 248k vocab), computing loss from hidden states + lm_head in
        # chunks instead. torch backend = numerically exact CE, no extra deps.
        "model.use_fused_kernels=true",
        "model.fused_kernel_options.impl_backend=torch",
        f"model.use_liger={_hydra_val(cfg.get('use_liger', True))}",
        f"model.enable_gradient_checkpointing={_hydra_val(cfg.get('gradient_checkpointing', True))}",
        f"engine.strategy={_hydra_val(cfg.get('strategy', 'fsdp2'))}",
        "engine.model_dtype=bfloat16",
        f"engine.seed={_hydra_val(cfg['seed'])}",
        f"engine.ulysses_sequence_parallel_size={_hydra_val(cfg['ulysses_sp_size'])}",
        f"optim.lr={_hydra_val(cfg['lr'])}",
        f"optim.lr_warmup_steps_ratio={_hydra_val(cfg['warmup_ratio'])}",
        f"optim.optimizer_impl={_hydra_val(cfg['optimizer_impl'])}",
        f"optim.optimizer={_hydra_val(cfg['optimizer_name'])}",
        f"optim.weight_decay={_hydra_val(cfg.get('weight_decay', 0.0))}",
        f"optim.betas={_hydra_val(cfg.get('betas', [0.9, 0.999]))}",
        f"optim.override_optimizer_config={_hydra_val(optimizer_override)}",
        f"trainer.default_local_dir={_hydra_val(cfg['local_dir'])}",
        f"trainer.save_freq={_hydra_val(cfg['save_freq'])}",
        f"trainer.n_gpus_per_node={_hydra_val(cfg['n_gpus_per_node'])}",
        "trainer.nnodes=1",
        f"trainer.seed={_hydra_val(cfg['seed'])}",
        f"trainer.logger={_hydra_val(cfg.get('loggers', ['console']))}",
        f"trainer.project_name={_hydra_val(cfg['project_name'])}",
        f"trainer.experiment_name={_hydra_val(cfg['experiment_name'])}",
        f"trainer.total_epochs={_hydra_val(cfg['loop_epochs'])}",
        "trainer.test_freq=-1",
        "trainer.resume_mode=auto",
        # retain only the latest full-state checkpoint on the pod: each global_step_N holds model
        # +optimizer shards (GBs); unbounded retention fills the container disk on long runs. flash
        # publishes required interval checkpoints to HF durably, so on-pod history is redundant.
        "trainer.max_ckpt_to_keep=1",
    ]
    if steps:
        overrides.append(f"trainer.total_training_steps={_hydra_val(steps)}")
    else:
        overrides.append("trainer.total_training_steps=null")
    return overrides


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


def _sft_parquet_features():
    from datasets import Features, Sequence, Value

    return Features(
        {
            "input_ids": Sequence(Value("int64")),
            "loss_mask": Sequence(Value("int8")),
            "images": Sequence(Value("string")),
            "multimodal_inputs": Value("binary"),
        }
    )


def _write_sft_parquet(rows: list[dict], path: str) -> None:
    from datasets import Dataset

    Dataset.from_list(rows, features=_sft_parquet_features()).to_parquet(path)


def _render_sft_dataset_module() -> str:
    """return the standalone custom dataset loaded by verl 0.8's data.custom_cls hook."""
    return '''from __future__ import annotations

import io

import numpy as np
import pandas as pd


class FlashTokenizedSFTDataset:
    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        if not isinstance(parquet_files, (list, tuple)):
            parquet_files = [parquet_files]
        frames = [pd.read_parquet(path, dtype_backend="pyarrow") for path in parquet_files]
        self.dataframe = pd.concat(frames, ignore_index=True)
        if max_samples > 0:
            self.dataframe = self.dataframe.iloc[:max_samples]
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = int(config.get("max_length", 1024))
        self.truncation = config.get("truncation", "error")
        if config.get("ignore_input_ids_mismatch", False):
            raise ValueError("flash tokenized sft requires input-id mismatch checks to stay enabled")
        if self.truncation not in {"error", "right"}:
            raise ValueError("flash tokenized sft supports error or right truncation only")

    def __len__(self):
        return len(self.dataframe)

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
        import torch

        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            return {key: torch.from_numpy(arrays[key].copy()) for key in arrays.files}

    def __getitem__(self, index):
        import torch

        row = self.dataframe.iloc[index]
        input_ids = self._list(row["input_ids"])
        loss_mask = self._list(row["loss_mask"])
        if len(input_ids) != len(loss_mask):
            raise ValueError("input_ids and loss_mask must have identical lengths")
        if len(input_ids) > self.max_length:
            if self.truncation == "error":
                raise ValueError("pretokenized row exceeds data.max_length")
            input_ids = input_ids[: self.max_length]
            loss_mask = loss_mask[: self.max_length]
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        loss_mask_tensor = torch.tensor(loss_mask, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids_tensor)
        multi_modal_inputs = self._multimodal_inputs(row["multimodal_inputs"])
        if (
            self.processor is not None
            and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
        ):
            from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids_tensor,
                image_grid_thw=multi_modal_inputs.get("image_grid_thw"),
                video_grid_thw=multi_modal_inputs.get("video_grid_thw"),
                second_per_grid_ts=multi_modal_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask,
            )
            text_position_ids = torch.arange(len(input_ids), dtype=torch.long).unsqueeze(0)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
        else:
            position_ids = torch.arange(len(input_ids), dtype=torch.long)
        return {
            "input_ids": input_ids_tensor,
            "position_ids": position_ids,
            "loss_mask": loss_mask_tensor,
            "multi_modal_inputs": multi_modal_inputs,
        }
'''


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

    input_ids = ids(full.pop("input_ids"))[:max_length]
    prompt_ids = ids(prompt["input_ids"])[:max_length]
    loss_mask = completion_mask_from_ids(prompt_ids, input_ids)
    full.pop("attention_mask", None)
    return input_ids, loss_mask, _serialize_multimodal_inputs(full)


def _has_real_target(row: dict, special_ids: set[int]) -> bool:
    return any(
        mask and token_id not in special_ids
        for token_id, mask in zip(row["input_ids"], row["loss_mask"], strict=True)
    )


def render_loraplus_shim(ratio: float) -> str:
    """return the sitecustomize source that adds LoRA+ to verl's fsdp engine."""
    ratio = float(ratio)
    if ratio <= 1:
        return ""
    return f'''
from importlib import import_module as _flash_import_module
from peft.optimizers import create_loraplus_optimizer as _flash_create_loraplus_optimizer
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine as _FlashFSDPEngine

def _flash_build_loraplus_optimizer(self, module):
    config = self.optimizer_config
    optimizer_cls = getattr(_flash_import_module(config.optimizer_impl), config.optimizer)
    optimizer_kwargs = {{"lr": config.lr, "weight_decay": config.weight_decay}}
    if "adam" in config.optimizer.lower() or "ademamix" in config.optimizer.lower():
        optimizer_kwargs["betas"] = config.betas
    if config.override_optimizer_config:
        optimizer_kwargs.update(config.override_optimizer_config)
    try:
        optimizer = _flash_create_loraplus_optimizer(
            model=module,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            loraplus_lr_ratio={ratio!r},
            loraplus_weight_decay=config.weight_decay,
        )
    except TypeError:
        optimizer = _flash_create_loraplus_optimizer(
            model=module,
            optimizer_cls=optimizer_cls,
            lr=config.lr,
            loraplus_lr_ratio={ratio!r},
            loraplus_weight_decay=config.weight_decay,
            **{{key: value for key, value in optimizer_kwargs.items() if key != "lr"}},
        )
    self._flash_loraplus_applied = True
    print("{_LORAPLUS_READY_MARKER} ratio={ratio:g} optimizer=" + optimizer_cls.__name__, flush=True)
    return optimizer

_FlashFSDPEngine._build_optimizer = _flash_build_loraplus_optimizer
'''


def _render_sft_sitecustomize(
    *,
    seed: int,
    loraplus_ratio: float,
    save_at_steps: tuple[int, ...],
    total_steps: int,
    reentrant_gradient_checkpointing: bool,
) -> str:
    required_steps = tuple(int(step) for step in save_at_steps)
    source = f'''# generated flash sft runtime patches for verl 0.8
import random as _flash_random

import numpy as _flash_numpy
import torch as _flash_torch
from transformers import get_linear_schedule_with_warmup as _flash_linear_schedule
from verl.trainer import sft_trainer as _flash_sft_trainer
from verl.utils.checkpoint.checkpoint_handler import CheckpointHandler as _FlashCheckpointHandler
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine as _FlashFSDPEngine

_flash_seed = {int(seed)}
_flash_random.seed(_flash_seed)
_flash_numpy.random.seed(_flash_seed % (2**32))
_flash_torch.manual_seed(_flash_seed)
_flash_torch.set_float32_matmul_precision("high")
_flash_torch.backends.cuda.matmul.allow_tf32 = True
_flash_torch.backends.cudnn.allow_tf32 = True

_flash_original_loader = _flash_sft_trainer.StatefulDataLoader

def _flash_non_dropping_loader(*args, **kwargs):
    kwargs["drop_last"] = False
    return _flash_original_loader(*args, **kwargs)

_flash_sft_trainer.StatefulDataLoader = _flash_non_dropping_loader

_flash_original_build_dataloader = _flash_sft_trainer.SFTTrainer._build_dataloader

def _flash_seeded_build_dataloader(self):
    result = _flash_original_build_dataloader(self)
    self.train_sampler.seed = _flash_seed
    if self.val_sampler is not None:
        self.val_sampler.seed = _flash_seed
    return result

_flash_sft_trainer.SFTTrainer._build_dataloader = _flash_seeded_build_dataloader

def _flash_build_lr_scheduler(self, optimizer):
    config = self.optimizer_config
    warmup_steps = config.lr_warmup_steps
    if warmup_steps <= 0:
        warmup_steps = int(config.lr_warmup_steps_ratio * config.total_training_steps)
    return _flash_linear_schedule(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=config.total_training_steps,
    )

_FlashFSDPEngine._build_lr_scheduler = _flash_build_lr_scheduler

_flash_required_save_steps = frozenset({required_steps!r})
_flash_total_steps = {int(total_steps)}
_flash_original_save_checkpoint = _FlashCheckpointHandler.save_checkpoint

def _flash_save_exact_checkpoint(self, step):
    if _flash_required_save_steps and step not in _flash_required_save_steps and step != _flash_total_steps:
        return None
    return _flash_original_save_checkpoint(self, step)

_FlashCheckpointHandler.save_checkpoint = _flash_save_exact_checkpoint
'''
    if reentrant_gradient_checkpointing:
        source += '''
_flash_original_build_module = _FlashFSDPEngine._build_module

def _flash_build_reentrant_module(self):
    module = _flash_original_build_module(self)
    module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    return module

_FlashFSDPEngine._build_module = _flash_build_reentrant_module
'''
    source += render_loraplus_shim(loraplus_ratio)
    return source


def _metric_value(line: str, name: str) -> float | None:
    for match in _VERL_METRIC_RE.finditer(line):
        if match.group("name") != name:
            continue
        try:
            return float(match.group("value"))
        except ValueError:
            return None
    return None


def _copy_processing_sidecars(actor_dir: str, adapter_dir: str) -> None:
    source = os.path.join(actor_dir, "huggingface")
    if not os.path.isdir(source):
        return
    prefixes = (
        "added_tokens",
        "chat_template",
        "merges",
        "preprocessor_config",
        "processor_config",
        "special_tokens_map",
        "tokenizer",
        "video_preprocessor_config",
        "vocab",
    )
    for name in os.listdir(source):
        if name.startswith(prefixes):
            src = os.path.join(source, name)
            dst = os.path.join(adapter_dir, name)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)


def _export_checkpoint_adapter(
    actor_dir: str,
    adapter_dir: str,
    *,
    model_id: str,
    model_revision: str,
    python_bin: str,
) -> None:
    shutil.rmtree(adapter_dir, ignore_errors=True)
    export_peft_adapter(
        actor_dir,
        adapter_dir,
        base_model_id=model_id,
        python_bin=python_bin,
    )
    _copy_processing_sidecars(actor_dir, adapter_dir)
    stamp_adapter_dir_provenance(adapter_dir, model_id, model_revision)
    _w.write_base_model_provenance(adapter_dir, model_id, model_revision)


class _VerlCheckpointWatcher:
    """watch verl's completion marker and publish each completed flash checkpoint."""

    def __init__(
        self,
        *,
        local_dir: str,
        export_root: str,
        python_bin: str,
        model_id: str,
        model_revision: str,
        required_steps: tuple[int, ...],
    ) -> None:
        self.local_dir = local_dir
        self.export_root = export_root
        self.python_bin = python_bin
        self.model_id = model_id
        self.model_revision = model_revision
        self.required_steps = frozenset(required_steps)
        self.processed_steps: set[int] = set()
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("verl checkpoint watcher failed") from self._error

    def stop(self, *, require_complete: bool) -> None:
        self._stop.set()
        self._thread.join(timeout=600)
        if self._thread.is_alive():
            raise RuntimeError("verl checkpoint watcher did not stop")
        self.raise_if_failed()
        if require_complete:
            missing = sorted(self.required_steps - self.processed_steps)
            if missing:
                raise RuntimeError(f"required saves were not durably published: {missing}")

    def _completed_step(self) -> int:
        tracker = os.path.join(self.local_dir, "latest_checkpointed_iteration.txt")
        try:
            with open(tracker) as file:
                return int(file.read().strip())
        except (FileNotFoundError, OSError, ValueError):
            return 0

    def _step_dirs(self, completed_step: int) -> list[tuple[int, str]]:
        found: list[tuple[int, str]] = []
        try:
            names = os.listdir(self.local_dir)
        except OSError:
            return found
        for name in names:
            match = re.fullmatch(r"global_step_(\d+)", name)
            if match is None:
                continue
            step = int(match.group(1))
            path = os.path.join(self.local_dir, name)
            if step <= completed_step and step not in self.processed_steps and os.path.isdir(path):
                found.append((step, path))
        return sorted(found)

    def _should_publish(self, step: int) -> bool:
        return not self.required_steps or step in self.required_steps

    def _publish(self, step: int, checkpoint_dir: str) -> None:
        if not self._should_publish(step):
            self.processed_steps.add(step)
            return
        actor_dir = os.path.join(checkpoint_dir, "actor")
        adapter_dir = os.path.join(self.export_root, f"step-{step}")
        _export_checkpoint_adapter(
            actor_dir,
            adapter_dir,
            model_id=self.model_id,
            model_revision=self.model_revision,
            python_bin=self.python_bin,
        )

        def publish_adapter() -> None:
            _w.publish_deployable_checkpoint(
                adapter_dir,
                step,
                required=step in self.required_steps,
                _provenance_ready=True,
            )

        uploaded = _w.upload_resume_checkpoint(
            step,
            checkpoint_dir,
            before_upload=publish_adapter,
        )
        if step in self.required_steps and not uploaded:
            raise RuntimeError(f"required save step {step} full-state checkpoint was not published")
        self.processed_steps.add(step)

    def _run(self) -> None:
        try:
            while True:
                completed_step = self._completed_step()
                for step, checkpoint_dir in self._step_dirs(completed_step):
                    self._publish(step, checkpoint_dir)
                if self._stop.is_set():
                    final_completed = self._completed_step()
                    remaining = self._step_dirs(final_completed)
                    if not remaining:
                        return
                time.sleep(0.5)
        except BaseException as error:
            self._error = error


def _cached_model_path(model_id: str, model_revision: str) -> str:
    from huggingface_hub import snapshot_download

    from flash.engine.worker.hf import _shared_weight_cache_dir
    from flash.engine.worker.perf import RetriableInfraError

    # prefetch lands weights on the shared volume when attached; resolve from the same cache first.
    for cache_dir in (_shared_weight_cache_dir(), None):
        try:
            return snapshot_download(
                repo_id=model_id,
                revision=model_revision or None,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        except Exception:
            continue
    # the prefetch step reported success but the cache has no resolvable snapshot: a transient
    # volume/link glitch, not a config error — retriable so the run lands on a healthy worker.
    raise RetriableInfraError(
        f"model {model_id} not resolvable from the local HF cache after prefetch"
    )


def _warmstart_adapter_path(model_id: str, model_revision: str, expected_rank: int) -> str | None:
    spec = _w.JOB_SPEC
    source = spec.train.init_from_adapter if spec else ""
    if not source:
        return None
    adapter_dir = _w._download_adapter(source)
    if not adapter_dir:
        raise RuntimeError("the prepared SFT warm-start adapter could not be downloaded")
    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as file:
        config = json.load(file)
    rank = int(config.get("r") or 0)
    if rank != expected_rank:
        raise ValueError(
            f"SFT warm-start adapter rank {rank} does not match the prepared train.lora_rank "
            f"{expected_rank}; rank changes are not supported"
        )
    base = str(config.get("base_model_name_or_path") or "").strip()
    if base and base != model_id:
        raise ValueError("SFT warm-start adapter base model does not match the target model")
    revision = str(config.get("revision") or "").strip()
    if revision and model_revision and revision != model_revision:
        raise ValueError("SFT warm-start adapter revision does not match the target model revision")
    return adapter_dir


def _message_contains_text(messages: list[dict], needle: str) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and needle in content:
            return True
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and needle in str(block.get("text") or ""):
                    return True
    return False


def _verl_image_message_content(content) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        raise ValueError("multimodal message content must be text or content blocks")
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            raise ValueError("multimodal message content blocks must be objects")
        if block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
        elif block.get("type") == "image":
            parts.append("<image>")
        else:
            raise ValueError(f"unsupported multimodal SFT content block {block.get('type')!r}")
    return "".join(parts)


def _materialize_verl_images(descriptors: list[str], package_root, image_dir: str, row_index: int) -> list[str]:
    from flash.multimodal import decode_image_descriptors

    os.makedirs(image_dir, exist_ok=True)
    images = decode_image_descriptors(descriptors, package_root)
    rows: list[str] = []
    for image_index, image in enumerate(images):
        path = Path(image_dir, f"row-{row_index}-image-{image_index}.png").resolve()
        image.save(path, format="PNG")
        rows.append(path.as_uri())
    return rows


def _restore_verl_resume(local_dir: str) -> int:
    resume = _w.hf_resume_checkpoint()
    if not resume:
        return 0
    match = re.fullmatch(r"checkpoint-(\d+)", os.path.basename(resume))
    if match is None:
        raise RuntimeError(f"invalid SFT resume checkpoint path {resume!r}")
    step = int(match.group(1))
    target = os.path.join(local_dir, f"global_step_{step}")
    shutil.copytree(resume, target, dirs_exist_ok=True)
    with open(os.path.join(local_dir, "latest_checkpointed_iteration.txt"), "w") as file:
        file.write(str(step))
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
        "FLASH_VERL_PYTHON",
    }
)
_CHILD_ENV_PREFIXES = (
    "CUDA_",
    "NCCL_",
    "TORCH_",
    "PYTORCH_",
    "VERL_",
    "OMP_",
    "MKL_",
    "OPENBLAS_",
    "LC_",
)


def _build_verl_child_env(*, shim_dir: str, wandb_enabled: bool) -> dict[str, str]:
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
    child["HYDRA_FULL_ERROR"] = "1"
    child["HF_HUB_OFFLINE"] = "1"
    child["TRANSFORMERS_OFFLINE"] = "1"
    return child


def _probe_gpu_in_subprocess(requested_gpu: str | None, exact_type: str = "") -> dict:
    script = r'''
import json
import sys

from flash.engine.worker.perf.lifecycle import wait_for_gpu

requested, exact = json.loads(sys.argv[1])
wait_for_gpu(requested, gpu_type=exact)
import torch
print("FLASH_GPU_PROBE=" + json.dumps({
    "name": torch.cuda.get_device_name(0),
    "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
    "capability": list(torch.cuda.get_device_capability(0)),
}), flush=True)
'''
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, json.dumps([requested_gpu, exact_type])],
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
        raise _w.RetriableInfraError(
            f"gpu readiness probe exited with status {result.returncode}"
        )
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
                    values = [float(value.strip()) for value in result.stdout.splitlines() if value.strip()]
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


def run_sft_verl(spec=None) -> None:
    """run flash sft through verl's out-of-process fsdp trainer."""
    from flash.catalog import MODELS, resolve_vocab_size
    from flash.engine.vram import sft_grad_accum

    spec = spec or _w.JOB_SPEC
    env = _w.require_active_env()
    started_at = time.time()
    _w.heartbeat("sft_start", gpu=_w.gpu_diagnostics(include_torch=False))
    gpu_probe = _probe_gpu_in_subprocess(
        spec.gpu.type if spec else None,
        exact_type=spec.gpu.type if spec else "",
    )

    model_id = spec.model if spec else RECIPE.hf_model_id
    model_revision = getattr(spec, "model_revision", "") if spec else ""
    download_seconds = _w.prefetch_model(model_id, revision=model_revision)
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

    workdir = os.path.join("/tmp", "flash-sft-verl", _w.RUN_ID, f"seed-{_w.SEED}")
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir, exist_ok=True)
    data_dir = os.path.join(workdir, "data")
    image_dir = os.path.join(workdir, "images")
    local_dir = os.path.join(workdir, "checkpoints")
    export_root = os.path.join(workdir, "checkpoint-adapters")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(local_dir, exist_ok=True)

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

        row_by_index: dict[int, dict] = {}
        text_specs: list[dict] = []
        sampled_texts: list[str] = []
        multiturn_targets = 0
        for row_index, (example, prompt_messages, completion_messages) in enumerate(prompt_rows):
            _reject_image_completion(completion_messages)
            if len(completion_messages) > 1:
                multiturn_targets += 1
            if record_has_images(example, prompt_messages):
                assert processor is not None
                normalized = normalize_prompt_images(example, prompt_messages, package_root)
                completion_messages = text_only_prompt_messages(completion_messages)
                decoded_images = decode_image_descriptors(normalized.descriptors, package_root)
                input_ids, loss_mask, multimodal_inputs = _processor_tokenized_row(
                    processor,
                    normalized.messages,
                    completion_messages,
                    decoded_images,
                    max_length=max_length,
                    thinking=bool(_w.THINKING),
                )
                row_by_index[row_index] = {
                    "input_ids": input_ids,
                    "loss_mask": loss_mask,
                    "images": _materialize_verl_images(
                        normalized.descriptors,
                        package_root,
                        image_dir,
                        row_index,
                    ),
                    "multimodal_inputs": multimodal_inputs,
                }
                sampled_texts.append(
                    tokenizer.apply_chat_template(
                        [*normalized.messages, *completion_messages],
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=_w.THINKING,
                    )
                )
            else:
                text = tokenizer.apply_chat_template(
                    [*prompt_messages, *completion_messages],
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

        dropped = 0
        if text_specs:
            kept_specs, tokenized_rows, text_dropped = _pretokenize_completion_only(
                text_specs, tokenizer, max_length
            )
            dropped += text_dropped
            for spec_row, tokenized in zip(kept_specs, tokenized_rows, strict=True):
                row_by_index[spec_row["row_index"]] = {
                    "input_ids": tokenized["input_ids"],
                    "loss_mask": tokenized["completion_mask"],
                    "images": [],
                    "multimodal_inputs": b"",
                }

        special_ids = set(getattr(tokenizer, "all_special_ids", None) or [])
        rows = []
        for row_index in sorted(row_by_index):
            row = row_by_index[row_index]
            if _has_real_target(row, special_ids):
                rows.append(row)
            else:
                dropped += 1
        if not rows:
            raise ValueError(
                "every SFT example has an empty completion after sft_max_len truncation "
                "(nothing to train on); increase sft_max_len or shorten the prompts"
            )
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

        masked_tokens = sum(mask.count(0) for mask in (row["loss_mask"] for row in rows))
        total_tokens_per_epoch = sum(len(row["input_ids"]) for row in rows)
        realized_max_length = max(len(row["input_ids"]) for row in rows)
        print(
            f"[sft] completion-only loss: masking {masked_tokens}/{total_tokens_per_epoch} "
            f"({masked_tokens / total_tokens_per_epoch:.0%}) prompt tokens"
        )
        train_file = os.path.join(data_dir, "train.parquet")
        val_file = os.path.join(data_dir, "val.parquet")
        _write_sft_parquet(rows, train_file)
        _write_sft_parquet([rows[0]], val_file)

    setup_seconds = time.time() - started_at
    _w.heartbeat(
        "sft_model_load",
        setup_seconds=setup_seconds,
        gpu=_w.gpu_diagnostics(include_torch=False),
    )

    lora_config = _w.make_lora(model_id)
    lora_rank = int(lora_config.r)
    lora_alpha = int(lora_config.lora_alpha)
    target_modules = lora_config.target_modules
    if isinstance(target_modules, set | frozenset):
        target_modules = sorted(target_modules)
    warmstart_adapter = _warmstart_adapter_path(model_id, model_revision, lora_rank)

    vocab_size = resolve_vocab_size(model_id, model_revision)
    per_device_batch, _ = sft_grad_accum(
        effective_batch,
        seq_len=max_length,
        vocab=vocab_size,
        fused=False,
    )
    train_batch_size = min(effective_batch, len(rows))
    micro_batch = max(1, min(per_device_batch, train_batch_size))
    steps_per_epoch = max(1, math.ceil(len(rows) / train_batch_size))
    derived_steps = sft_update_steps(
        epochs=epochs,
        example_count=len(rows),
        examples_per_update=train_batch_size,
    )
    update_horizon = resolve_update_horizon(derived_steps, max_steps)
    validate_save_steps(save_at_steps, update_horizon)
    loop_epochs = max(epochs, math.ceil(update_horizon / steps_per_epoch))
    save_freq = reduce(gcd, save_at_steps) if save_at_steps else save_every

    card_vram_gb = float(gpu_probe.get("memory_gb") or 0.0)
    raw_capability = gpu_probe.get("capability")
    capability = tuple(raw_capability) if raw_capability else None
    hidden, layers = _model_arch_dims(model_id, revision=model_revision)
    info = MODELS.get(model_id)
    active_params_b = float(getattr(info, "active_params_b", 0.0) or 0.0) or None
    gradient_checkpointing = _w.grad_checkpointing_on(
        model_id,
        max_length,
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

    optimizer_cls, optimizer_kwargs = _w.loraplus_optimizer_cls(_w.fused_optim_name())
    python_bin = resolve_verl_python(workdir)
    model_path = _cached_model_path(model_id, model_revision)
    loggers = ["console"]
    if os.environ.get("WANDB_API_KEY"):
        loggers.append("wandb")
    project_name = (
        (spec.wandb.project if spec and spec.wandb else None) or "flash"
    )
    experiment_name = _w.wandb_run_name()
    shim_dir = os.path.join(workdir, "shim")
    os.makedirs(shim_dir, exist_ok=True)
    custom_dataset_path = os.path.join(shim_dir, "flash_verl_sft_dataset.py")

    config = {
        "train_files": train_file,
        "val_files": val_file,
        "train_batch_size": train_batch_size,
        "max_length": max_length,
        "micro_batch": micro_batch,
        "max_token_len_per_gpu": max_length * micro_batch,
        "custom_dataset_path": custom_dataset_path,
        "model_path": model_path,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules,
        "lora_adapter_path": warmstart_adapter,
        "ulysses_sp_size": gpu_count,
        "lr": learning_rate,
        "warmup_ratio": RECIPE.sft.warmup_frac,
        "optimizer_impl": optimizer_cls.__module__,
        "optimizer_name": optimizer_cls.__name__,
        "optimizer_kwargs": optimizer_kwargs or None,
        "local_dir": local_dir,
        "save_freq": save_freq,
        "n_gpus_per_node": gpu_count,
        "seed": _w.backend_seed(_w.SEED),
        "project_name": project_name,
        "experiment_name": experiment_name,
        "loop_epochs": loop_epochs,
        "loggers": loggers,
        "use_liger": True,
        "gradient_checkpointing": gradient_checkpointing and not reentrant_gradient_checkpointing,
        "total_training_steps": update_horizon if max_steps > 0 else None,
        "total_epochs": epochs if max_steps <= 0 else None,
    }
    overrides = build_sft_verl_overrides(config)

    shim_source = _render_sft_sitecustomize(
        seed=config["seed"],
        loraplus_ratio=_SFT_LORAPLUS_RATIO,
        save_at_steps=save_at_steps,
        total_steps=update_horizon,
        reentrant_gradient_checkpointing=reentrant_gradient_checkpointing,
    )
    with open(os.path.join(shim_dir, "sitecustomize.py"), "w", encoding="utf-8") as file:
        file.write(shim_source)
    with open(custom_dataset_path, "w", encoding="utf-8") as file:
        file.write(_render_sft_dataset_module())

    resume_step = _restore_verl_resume(local_dir)
    watcher = _VerlCheckpointWatcher(
        local_dir=local_dir,
        export_root=export_root,
        python_bin=python_bin,
        model_id=model_id,
        model_revision=model_revision,
        required_steps=save_at_steps,
    )
    watcher.processed_steps.update(_durable_required_save_steps(save_at_steps, resume_step))
    if resume_step >= update_horizon:
        missing = sorted(watcher.required_steps - watcher.processed_steps)
        if missing:
            raise RuntimeError(f"required saves were not durably published: {missing}")

    child_env = _build_verl_child_env(
        shim_dir=shim_dir,
        wandb_enabled="wandb" in loggers,
    )
    command = [
        python_bin,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={gpu_count}",
        "-m",
        "verl.trainer.sft_trainer",
        *overrides,
    ]

    progress = {"step": resume_step, "loss": None, "grad_norm": None, "lr": None}
    loss_curve: list[float] = []
    train_tokens = sft_completed_train_tokens(
        total_tokens_per_epoch,
        epochs,
        derived_steps,
        resume_step,
    )
    loraplus_applied = resume_step >= update_horizon

    def on_line(line: str) -> None:
        nonlocal loraplus_applied
        watcher.raise_if_failed()
        if _LORAPLUS_READY_MARKER in line:
            loraplus_applied = True
        step_match = _VERL_STEP_RE.search(line)
        if step_match is None:
            return
        if _SFT_LORAPLUS_RATIO > 1 and not loraplus_applied:
            raise RuntimeError("verl reached an optimizer step before the required lora+ shim succeeded")
        loss = _metric_value(line, "train/loss")
        grad_norm = _metric_value(line, "train/grad_norm")
        learning_rate_value = _metric_value(line, "train/lr")
        if loss is not None:
            loss_curve.append(round(loss, 4))
            progress["loss"] = loss
        if grad_norm is not None:
            progress["grad_norm"] = grad_norm
        if learning_rate_value is not None:
            progress["lr"] = learning_rate_value

    def on_step(step: int) -> None:
        progress["step"] = step
        payload = {
            "step": step,
            "loss": progress["loss"],
            "grad_norm": progress["grad_norm"],
            "learning_rate": progress["lr"],
        }
        _w.heartbeat("sft_step", **{key: value for key, value in payload.items() if value is not None})

    def child_heartbeat() -> None:
        _w.heartbeat("sft_step", liveness=True, step=int(progress["step"] or 0))

    gpu_sampler = _NvidiaSmiPeakSampler().start()
    train_started_at = time.time()
    return_code = 0
    if resume_step < update_horizon:
        watcher.start()
        try:
            with liveness_heartbeat(
                "sft_step",
                progress=lambda: int(progress["step"] or 0),
                progress_step=True,
            ):
                return_code = run_verl_training(
                    command,
                    env=child_env,
                    on_step=on_step,
                    on_line=on_line,
                    heartbeat=child_heartbeat,
                )
        finally:
            watcher.stop(require_complete=return_code == 0)
    train_wall = time.time() - train_started_at
    device_peak_gpu_gb = gpu_sampler.stop_gb()
    if return_code != 0:
        raise RuntimeError(f"verl SFT subprocess exited with status {return_code}")
    if _SFT_LORAPLUS_RATIO > 1 and not loraplus_applied:
        raise RuntimeError("required lora+ shim did not emit its success marker")

    actor_dir, final_step = latest_global_step_dir(local_dir)
    if sft_under_ran(final_step, update_horizon, max_steps):
        raise RuntimeError(f"sft completed {final_step}/{update_horizon} requested optimizer updates")
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
            actor_dir,
            adapter_dir,
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
            "runtime_max_length": max_length,
            "per_device_train_batch_size": micro_batch,
            "gradient_accumulation_steps": math.ceil(train_batch_size / micro_batch),
            "packing": "verl_remove_padding",
            "loss_curve": loss_curve[:400],
            "peak_gpu_gb": device_peak_gpu_gb,
            "device_peak_gpu_gb": device_peak_gpu_gb,
            "loraplus_optim": optimizer_cls.__name__,
            "loraplus_applied": loraplus_applied,
            "chalk_kernels": None,
            "verl_backend": "fsdp2",
            "ulysses_sequence_parallel_size": gpu_count,
            "wandb_project": project_name if "wandb" in loggers else None,
            "wandb_run_name": experiment_name if "wandb" in loggers else None,
        },
    )
