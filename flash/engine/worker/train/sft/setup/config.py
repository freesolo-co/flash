"""Render every file and hydra override the verl sft child is launched with.

`build_sft_overrides` turns a resolved config dict into the hydra argv verl's sft trainer takes, and
the `_render_*`/`render_*` helpers emit the child-side python that goes with it: the dataset module
that reads flash's parquet, and the sitecustomize that patches verl in-process (tf32, gdn varlen,
exact dataloader order, LoRA+). It is all string and dict construction -- no torch, no subprocess.

Split out of `flash.engine.worker.train.entry.sft_train` to keep that module under the file-size limit.
"""

from __future__ import annotations

# printed by the LoRA+ shim once it has installed itself, and matched on the child's stdout by the
# parent: the shim runs inside verl, so this line is the only proof the optimizer grouping took.
_LORAPLUS_READY_MARKER = "FLASH_LORAPLUS_READY"

# every key `build_sft_overrides` requires its input to carry. an override the child never receives
# is a silently-defaulted hyperparameter, so a missing key raises rather than falling back.
_REQUIRED_OVERRIDE_KEYS = (
    "train_files",
    "fused_ce_backend",
    "train_batch_size",
    "max_length",
    "micro_batch",
    "max_token_len_per_gpu",
    "custom_dataset_path",
    "model_path",
    "lora_rank",
    "lora_alpha",
    "target_modules",
    "fsdp_generation",
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


# verl imports this optimizer by name in the torch-bearing worker process.
# use fp32 adamw: bitsandbytes cannot update fsdp2 dtensors. fsdp1 is not an alternative because flattened
# names make peft's lora+ grouping treat every parameter as one-dimensional and silently mis-group it.
_VERL_OPTIMIZER_IMPL = "torch.optim"
_VERL_OPTIMIZER_NAME = "AdamW"


def _optimizer_override_config(cfg: dict) -> dict:
    override = dict(cfg.get("optimizer_kwargs") or {})
    override.setdefault("eps", cfg.get("eps", 1e-8))
    return override


def build_sft_overrides(cfg: dict) -> list[str]:
    """build hydra overrides for verl's ``sft_trainer_engine.yaml`` config."""
    missing = [key for key in _REQUIRED_OVERRIDE_KEYS if key not in cfg]
    if missing:
        raise KeyError(f"build_sft_overrides missing required cfg keys: {missing}")
    steps = cfg.get("total_training_steps")
    epochs = cfg.get("total_epochs")
    if bool(steps) == bool(epochs):
        raise ValueError("set exactly one of cfg['total_training_steps'] or cfg['total_epochs']")
    optimizer_override = _optimizer_override_config(cfg)

    overrides = [
        f"data.train_files={_hydra_val(cfg['train_files'])}",
        # HARDCODED null, not a cfg key. flash never reads verl's val/loss, and a val dataloader
        # that merely EXISTS buys a full inference forward on the last step (see trainer.test_freq
        # below). there is no supported value other than null, so do not accept one.
        "data.val_files=null",
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
        f"model.exclude_modules={_hydra_val(cfg.get('exclude_modules'))}",
        *(
            [f"++model.target_parameters={_hydra_val(cfg['target_parameters'])}"]
            if cfg.get("target_parameters")
            else []
        ),
        f"model.lora_adapter_path={_hydra_val(cfg.get('lora_adapter_path'))}",
        # remove-padding concatenates the micro-batch into one (1, total_nnz) row -- real packing.
        # a GDN hybrid only gets packed NEIGHBOURS when the profile says "packed", which it never
        # does for gdn (see the caller): those runs train one example per update, so there is no
        # neighbour whose linear-attention residue could bleed in. the flag itself must stay on
        # regardless -- it selects the nested-tensor layout verl's no_padding sft_loss requires.
        f"model.use_remove_padding={_hydra_val(cfg.get('use_remove_padding', True))}",
        # 32k contexts: the fused linear-CE forward never materializes the [tokens, vocab] logits
        # tensor (~130 GB at 32k on a 248k vocab), computing loss from hidden states + lm_head in
        # chunks instead. the backend is chosen per card -- see fused_ce_backend for the measured
        # table; triton wins on sm90+ and loses below it.
        "model.use_fused_kernels=true",
        f"model.fused_kernel_options.impl_backend={cfg['fused_ce_backend']}",
        f"model.use_liger={_hydra_val(cfg.get('use_liger', False))}",
        f"model.enable_gradient_checkpointing={_hydra_val(cfg.get('gradient_checkpointing', True))}",
        f"engine.strategy={'fsdp2' if cfg['fsdp_generation'] == 2 else 'fsdp'}",
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
        # -1 does NOT disable validation on its own. the child's gate is
        # `is_last_step and self.val_dataloader is not None or (self.test_freq > 0 and is_valid_step)`
        # (verl/trainer/sft_trainer.py), and `and` binds tighter than `or`, so a val dataloader that
        # merely EXISTS buys a full inference forward on the last step whatever test_freq says. a
        # null `data.val_files` is what removes the dataloader; this stays so no periodic pass runs.
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
    return """from __future__ import annotations

import io

import numpy as np
import pandas as pd


class FlashTokenizedSFTDataset:
    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        # verl hands data.train_files straight through from hydra, so this arrives as an omegaconf
        # ListConfig, not a list. an isinstance (list, tuple) check misses it and wraps the sequence
        # into [ListConfig([...])], which pandas rejects with "cannot construct a FileSource from
        # [...]" before the first step. branch on str instead: a bare path is the only scalar shape.
        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]
        frames = [pd.read_parquet(str(path), dtype_backend="pyarrow") for path in parquet_files]
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
"""
