"""Render Hydra overrides and parquet data for the packaged SFT trainer.

`build_sft_overrides` turns resolved execution settings into upstream trainer configuration.
The packaged dataset and Flash engine own dataset loading and model policy.
This module constructs configuration and data without launching a trainer.

Split out of `flash.engine.worker.train.entry.sft_train` to keep that module under the file-size limit.
"""

from __future__ import annotations

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
    "ulysses_sp_size",
    "lr",
    "warmup_ratio",
    "local_dir",
    "save_freq",
    "n_gpus_per_node",
    "seed",
    "project_name",
    "experiment_name",
    "loop_epochs",
    "total_training_steps",
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
# use fp32 adamw: bitsandbytes cannot update fsdp2 dtensors.
_VERL_OPTIMIZER_IMPL = "torch.optim"
_VERL_OPTIMIZER_NAME = "AdamW"


def build_sft_overrides(cfg: dict) -> list[str]:
    """build hydra overrides for verl's ``sft_trainer_engine.yaml`` config."""
    missing = [key for key in _REQUIRED_OVERRIDE_KEYS if key not in cfg]
    if missing:
        raise KeyError(f"build_sft_overrides missing required cfg keys: {missing}")
    steps = cfg["total_training_steps"]
    if type(steps) is not int or steps <= 0:
        raise ValueError("total_training_steps must be a positive integer")

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
        # liger severs lora gradients under fsdp2; fused CE uses use_fused_kernels instead.
        "model.use_liger=false",
        f"model.enable_gradient_checkpointing={_hydra_val(cfg.get('gradient_checkpointing', True))}",
        "engine.strategy=fsdp2",
        "engine.model_dtype=bfloat16",
        f"engine.seed={_hydra_val(cfg['seed'])}",
        f"engine.ulysses_sequence_parallel_size={_hydra_val(cfg['ulysses_sp_size'])}",
        f"optim.lr={_hydra_val(cfg['lr'])}",
        f"optim.lr_warmup_steps_ratio={_hydra_val(cfg['warmup_ratio'])}",
        f"optim.optimizer_impl={_VERL_OPTIMIZER_IMPL}",
        f"optim.optimizer={_VERL_OPTIMIZER_NAME}",
        "optim.weight_decay=0.0",
        "optim.betas=[0.9,0.999]",
        "optim.override_optimizer_config={eps:0.00000001}",
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
    overrides.append(f"trainer.total_training_steps={steps}")
    return overrides


def _sft_parquet_features():
    from datasets import Features, Sequence, Value

    return Features(
        {
            "input_ids": Sequence(Value("int64")),
            "loss_mask": Sequence(Value("int8")),
            "images": Sequence(Value("string")),
            # 64-bit offsets: a 32-bit binary column overflows past 2 GiB per dataset.
            "multimodal_inputs": Value("large_binary"),
        }
    )


def _write_sft_parquet(rows: list[dict], path: str) -> None:
    from datasets import Dataset

    Dataset.from_list(rows, features=_sft_parquet_features()).to_parquet(path)
