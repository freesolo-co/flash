"""sft training via verl (out-of-process), replacing the trl sft path.

flash emits verl `messages` parquet rows plus a hydra override list for
``python -m verl.trainer.sft_trainer`` (config sft_trainer_engine.yaml), then reuses verl_common to
run the subprocess and export the resulting lora adapter. verl's sft trainer masks by message role,
so completion-only supervision is expressed as ``[{user prompt}, {assistant completion}]`` messages;
that this reproduces flash's completion_mask is verified at the tokenizer level by a separate test.
"""

from __future__ import annotations

# required cfg keys the caller must resolve from the flash recipe + job spec before building
# overrides. kept explicit so a missing knob fails loudly here rather than as a confusing verl error.
_REQUIRED_OVERRIDE_KEYS = (
    "train_files",
    "val_files",
    "max_length",
    "micro_batch",
    "max_token_len_per_gpu",
    "model_path",
    "lora_rank",
    "lora_alpha",
    "target_modules",
    "ulysses_sp_size",
    "lr",
    "local_dir",
    "save_freq",
    "n_gpus_per_node",
    "project_name",
    "experiment_name",
)


def _hydra_val(v) -> str:
    """render a python value as a hydra override rhs (lowercase bools, bracketed lists).

    floats are rendered fixed-point (e.g. 5e-5 -> "0.00005") so hydra parses them as floats
    regardless of magnitude; str() would emit scientific notation ("5e-05") for common small lrs.
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        s = f"{v:.12f}".rstrip("0")
        return s + "0" if s.endswith(".") else s
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(str(x) for x in v) + "]"
    return str(v)


def build_sft_verl_overrides(cfg: dict) -> list[str]:
    """build the hydra override list for verl.trainer.sft_trainer from flash-resolved ``cfg``.

    keys map to sft_trainer_engine.yaml nodes (data/model/engine/optim/trainer). packing
    (model.use_remove_padding) is required for ulysses sequence parallel
    (engine.ulysses_sequence_parallel_size); liger fused-ce (model.use_liger) keeps 32k from
    materializing full logits. exactly one of total_training_steps / total_epochs must be set.
    """
    missing = [k for k in _REQUIRED_OVERRIDE_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"build_sft_verl_overrides missing required cfg keys: {missing}")
    steps = cfg.get("total_training_steps")
    epochs = cfg.get("total_epochs")
    if bool(steps) == bool(epochs):
        raise ValueError("set exactly one of cfg['total_training_steps'] or cfg['total_epochs']")

    ov = [
        # data: messages-based (verl computes the role mask); dynamic batching bounds per-gpu tokens.
        f"data.train_files={_hydra_val(cfg['train_files'])}",
        f"data.val_files={_hydra_val(cfg['val_files'])}",
        f"data.max_length={_hydra_val(cfg['max_length'])}",
        "data.messages_key=messages",
        f"data.micro_batch_size_per_gpu={_hydra_val(cfg['micro_batch'])}",
        "data.use_dynamic_bsz=true",
        f"data.max_token_len_per_gpu={_hydra_val(cfg['max_token_len_per_gpu'])}",
        # model: lora on the immutable base (offline-cached commit); packing + liger for long context.
        f"model.path={_hydra_val(cfg['model_path'])}",
        f"model.lora_rank={_hydra_val(cfg['lora_rank'])}",
        f"model.lora_alpha={_hydra_val(cfg['lora_alpha'])}",
        f"model.target_modules={_hydra_val(cfg['target_modules'])}",
        "model.use_remove_padding=true",
        f"model.use_liger={_hydra_val(cfg.get('use_liger', True))}",
        f"model.enable_gradient_checkpointing={_hydra_val(cfg.get('gradient_checkpointing', True))}",
        # engine (fsdp): sequence-parallel degree lives here, not on the model node.
        f"engine.strategy={_hydra_val(cfg.get('strategy', 'fsdp2'))}",
        f"engine.ulysses_sequence_parallel_size={_hydra_val(cfg['ulysses_sp_size'])}",
        f"optim.lr={_hydra_val(cfg['lr'])}",
        # trainer: n_gpus_per_node is the per-job gpu.count; single node.
        f"trainer.default_local_dir={_hydra_val(cfg['local_dir'])}",
        f"trainer.save_freq={_hydra_val(cfg['save_freq'])}",
        f"trainer.n_gpus_per_node={_hydra_val(cfg['n_gpus_per_node'])}",
        "trainer.nnodes=1",
        f"trainer.logger={_hydra_val(cfg.get('loggers', ['console']))}",
        f"trainer.project_name={_hydra_val(cfg['project_name'])}",
        f"trainer.experiment_name={_hydra_val(cfg['experiment_name'])}",
    ]
    if steps:
        ov.append(f"trainer.total_training_steps={_hydra_val(steps)}")
    else:
        ov.append(f"trainer.total_epochs={_hydra_val(epochs)}")
    return ov


def build_sft_verl_messages_rows(prompt_completion_rows) -> list[dict]:
    """turn flash sft examples into verl `messages` parquet rows (structural mapping).

    each flash example is ``(prompt_messages, completion_messages)`` - both lists of {role, content}
    dicts (env.prompt_messages / env.sft_completion). verl's sft trainer takes a single `messages`
    list per row and trains the assistant-role tokens. rows with an empty completion are dropped,
    matching flash's "no real completion target" drop.

    PARITY NOTE: verl trains ALL assistant-role tokens; flash's completion_mask trains the completion
    span. these agree for single-turn (prompt has no assistant turn) but can DIVERGE for multi-turn
    prompts that already contain assistant history. this builder only guarantees the structural
    mapping; the tokenizer-level mask-parity test (flash completion_mask vs verl role-mask) is what
    verifies the masks actually match per example, and gates whether a multi-turn special case is
    needed.
    """
    rows: list[dict] = []
    for prompt_messages, completion_messages in prompt_completion_rows:
        if not completion_messages:
            continue
        rows.append({"messages": [*prompt_messages, *completion_messages]})
    return rows
