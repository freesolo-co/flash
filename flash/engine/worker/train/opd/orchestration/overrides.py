"""Hydra overrides, child environment, and parquet writing for the OPD trainer.

`build_opd_overrides` renders the exact verl 0.8.0 synchronous-PPO and distillation config surface
the child is launched with; the rest prepare what that child needs on disk and in its environment
(the resolved env vars and rollout parquet).

Split out of `flash.engine.worker.train.entry.opd_train` to keep that module under the file-size limit.
"""

from __future__ import annotations

import contextlib
import json
import os

from flash.content.structured_outputs import reasoning_parser_for
from flash.engine.worker.train.entry.backend_common import (
    actor_fsdp_strategy_overrides,
    agent_loop_workers,
    ray_num_cpus,
    rollout_layered_summon_overrides,
    rollout_mm_processor_cache_overrides,
    rollout_resident_overrides,
    shim_marker_file,
    trainer_dtype_overrides,
)
from flash.engine.worker.train.entry.sft_train import _build_verl_child_env, _hydra_val
from flash.engine.worker.verl.capabilities import rollout_max_num_seqs
from flash.teacher.limits import OPD_NO_SIGNAL_ATTEMPTS

_REQUIRED_OVERRIDE_KEYS = (
    "train_files",
    "val_files",
    "train_batch_size",
    "max_prompt_length",
    "max_response_length",
    "max_sequence_length",
    "model_path",
    "lora_rank",
    "lora_alpha",
    "target_modules",
    "fsdp_generation",
    "learning_rate",
    "local_dir",
    "save_freq",
    "n_gpus_per_node",
    "gpu_mem_util",
    "ulysses_sequence_parallel_size",
    "seed",
    "project_name",
    "experiment_name",
    "total_training_steps",
    "group_size",
    "bridge_url",
    "bridge_token",
    "kl_penalty_coef",
    "reward_path",
)


def _algorithm_overrides() -> list[str]:
    return [
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=false",
        "algorithm.norm_adv_by_std_in_grpo=false",
    ]


def _data_input_overrides(config: dict) -> list[str]:
    return [
        f"data.train_files={_hydra_val(config['train_files'])}",
        f"data.val_files={_hydra_val(config['val_files'])}",
        f"data.train_batch_size={_hydra_val(config['train_batch_size'])}",
        f"data.max_prompt_length={_hydra_val(config['max_prompt_length'])}",
        f"data.max_response_length={_hydra_val(config['max_response_length'])}",
        "data.filter_overlong_prompts=true",
        "data.truncation=error",
        "data.shuffle=false",
        f"data.seed={_hydra_val(config.get('seed', 42))}",
    ]


def _actor_rollout_seed_overrides(config: dict) -> list[str]:
    return [
        # set the seed through ++actor_rollout_ref.rollout.engine_kwargs.seed: RolloutConfig has no
        # rollout.seed, and engine_kwargs wins after verl's own seed. requests are seeded
        # separately.
        f"++actor_rollout_ref.rollout.engine_kwargs.vllm.seed={_hydra_val(config.get('seed', 42))}",
    ]


def _data_runtime_overrides(config: dict) -> list[str]:
    return [
        "data.dataloader_num_workers=0",
        "data.image_key=images",
        "data.return_raw_chat=true",
        "data.return_multi_modal_inputs=false",
        # `++`, not a bare key: data.apply_chat_template_kwargs exists but holds an empty struct
        # dict, so assigning a whole dict to it is rejected as an unknown sub-key.
        "++data.apply_chat_template_kwargs={enable_thinking:"
        + _hydra_val(config.get("thinking", False))
        + ",preserve_thinking:false}",
    ]


def _actor_rollout_overrides(config: dict, *, max_tokens: int) -> list[str]:
    return [
        f"actor_rollout_ref.model.path={_hydra_val(config['model_path'])}",
        "actor_rollout_ref.model.trust_remote_code=true",
        # packing the micro-batch into one row is only safe for a gdn hybrid when the child can
        # reset linear-attention state at the example boundaries -- and the caller's gdn gate now
        # RAISES when it cannot, because the padded alternative cannot complete a step alongside the
        # fused kernels below. always true by the time we get here.
        "actor_rollout_ref.model.use_remove_padding=true",
        # 32k contexts: the fused linear-CE forward computes per-token log_probs from hidden states
        # + lm_head in chunks (FusedLinearForPPO), never materializing the [tokens, vocab] logits
        # tensor (~130 GB at 32k on a 248k vocab). the distillation loss consumes exactly
        # model_output["log_probs"], so the fused path is loss-equivalent. the backend is chosen
        # per card; see fused_ce_backend for the measured table.
        "actor_rollout_ref.model.use_fused_kernels=true",
        f"actor_rollout_ref.model.fused_kernel_options.impl_backend={config['fused_ce_backend']}",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        f"actor_rollout_ref.model.lora_rank={_hydra_val(config['lora_rank'])}",
        f"actor_rollout_ref.model.lora_alpha={_hydra_val(config['lora_alpha'])}",
        f"actor_rollout_ref.model.target_modules={_hydra_val(config['target_modules'])}",
        f"actor_rollout_ref.model.exclude_modules={_hydra_val(config.get('exclude_modules'))}",
        *(
            [
                "++actor_rollout_ref.model.target_parameters="
                + _hydra_val(config["target_parameters"])
            ]
            if config.get("target_parameters")
            else []
        ),
        f"actor_rollout_ref.model.lora_adapter_path={_hydra_val(config.get('lora_adapter_path'))}",
        "actor_rollout_ref.actor.use_kl_loss=false",
        "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean",
        "actor_rollout_ref.actor.use_dynamic_bsz=true",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={_hydra_val(config['train_batch_size'])}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={_hydra_val(max_tokens)}",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.shuffle=false",
        "actor_rollout_ref.actor.use_torch_compile=true",
        f"actor_rollout_ref.actor.optim.lr={_hydra_val(config['learning_rate'])}",
        "actor_rollout_ref.actor.optim.weight_decay=0.0",
        f"actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size={_hydra_val(config['ulysses_sequence_parallel_size'])}",
        # zero-2 vs zero-3, resolved by the allocator's fit model in the caller. lora makes verl
        # reuse the actor module for the ref policy (`ref_in_actor`), so there is no second weight
        # copy and no matching `ref.fsdp_config` key to set.
        # absent -> True, verl's own default: a config built without the gate must render the
        # lower-memory strategy, never fall into zero-2 by omission.
        f"actor_rollout_ref.actor.fsdp_config.reshard_after_forward={_hydra_val(bool(config.get('reshard_after_forward', True)))}",
        # dense opd stays on the live-validated fsdp1 path. fused-expert target_parameters requires
        # fsdp2 so peft can parametrize named tensors. reshard_after_forward remains independent.
        *actor_fsdp_strategy_overrides(config["fsdp_generation"]),
        # store the frozen base in bf16, not verl's fp32 yaml default. shared with the rl driver.
        *trainer_dtype_overrides(),
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        # safetensors is required for lora rollout: the dummy sync appends .base_layer to vision
        # weights that plain vllm does not expose. keep vllm authoritative for base weights.
        "actor_rollout_ref.rollout.load_format=safetensors",
        # rollout.enforce_eager is a real verl field, so this is a plain override, not a '+' append.
        # the caller resolves it from the device capability; absent/false keeps verl's default.
        *(["actor_rollout_ref.rollout.enforce_eager=True"] if config.get("enforce_eager") else []),
        # blackwell attention pins, resolved by the caller and absent off blackwell. both are real
        # AsyncEngineArgs fields in the pinned vllm 0.19.1 and verl spreads engine_kwargs.vllm
        # straight into them, so '+' appends under the existing struct exactly as on the grpo path.
        *(
            [
                (
                    "+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend="
                    f"{config['attention_backend']}"
                )
            ]
            if config.get("attention_backend")
            else []
        ),
        *(
            [
                (
                    "+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_attn_backend="
                    f"{config['mm_encoder_attn_backend']}"
                )
            ]
            if config.get("mm_encoder_attn_backend")
            else []
        ),
        # fused-expert lora models gather the weight sync one submodule at a time, as the grpo path
        # does; opd shares verl's collect_lora_params. requires load_format=safetensors above.
        *rollout_layered_summon_overrides(config.get("target_parameters")),
        *rollout_mm_processor_cache_overrides(),
        # keep the rollout engine RESIDENT for models whose vLLM wake/reload HANGS (catalog
        # sleep_unsupported), exactly as the grpo path does. opd is NOT exempt: main_ppo_sync calls
        # checkpoint_manager.sleep_replicas() during init_workers and again around validation, which
        # lands in the same vllm_async_server sleep(). the flagged model declares algos including
        # opd and the parse-time gate admits it, so without this an opd run on it wedges.
        *rollout_resident_overrides(bool(config.get("sleep_unsupported"))),
        # size the colocated executor budget from this run's geometry, as the grpo path does.
        # verl's default is 0.5, i.e. half the CARD, claimed on every wake no matter what the
        # trainer already holds -- and `wake_up` re-acquires released physical pages, so an
        # overcommit is a hard cumem_allocator OOM rather than a smaller pool.
        f"actor_rollout_ref.rollout.gpu_memory_utilization={_hydra_val(config['gpu_mem_util'])}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={_hydra_val(config['n_gpus_per_node'])}",
        f"actor_rollout_ref.rollout.n={_hydra_val(config['group_size'])}",
        # `++`, not a bare key: limit_images is a real RolloutConfig field but is absent from the
        # composed rollout node, so hydra rejects a bare assignment.
        "++actor_rollout_ref.rollout.limit_images=4",
        f"actor_rollout_ref.rollout.max_model_len={_hydra_val(max_tokens)}",
        f"actor_rollout_ref.rollout.temperature={_hydra_val(config.get('temperature', 1.0))}",
        f"actor_rollout_ref.rollout.top_p={_hydra_val(config.get('top_p', 1.0))}",
        "actor_rollout_ref.rollout.top_k=-1",
        "actor_rollout_ref.rollout.calculate_log_probs=false",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={_hydra_val(max_tokens)}",
        "actor_rollout_ref.rollout.agent.default_agent_loop="
        + ("flash_multi_turn" if config.get("multi_turn") else "flash_single_turn"),
        # opd runs async rollout through verl's AgentLoopManager, which chunks the rollout batch
        # across agent.num_workers and asserts an exact split. the batch is
        # train_batch_size * group_size, so verl's default of 8 aborts the run before the first step
        # whenever that product is not a multiple of 8. size the pool to the batch instead.
        (
            "actor_rollout_ref.rollout.agent.num_workers="
            f"{agent_loop_workers(int(config['train_batch_size']) * int(config['group_size']), cap=4 if config.get('multimodal') else 8)}"
        ),
        # one opd step submits exactly train_batch_size * group_size student generations. verl's
        # 1024 default makes vllm reserve cuda-graph capture sizes -- and, on a gdn/mamba hybrid,
        # one recurrent state block per decode slot -- for a batch this run never submits. both are
        # allocated before the first token, so the over-provisioning OOMs a card that stays mostly
        # free. `rollout.max_num_seqs` is a real verl field, so this is a plain override.
        (
            "actor_rollout_ref.rollout.max_num_seqs="
            f"{rollout_max_num_seqs(int(config['train_batch_size']) * int(config['group_size']))}"
        ),
    ]


def _transfer_queue_overrides() -> list[str]:
    return [
        # verl force-enables TransferQueue and waits forever for every cpu bundle.
        # one storage unit fits flash's single-node trainer regardless of ray cluster sizing.
        "transfer_queue.backend.SimpleStorage.num_data_storage_units=1",
    ]


def _ray_overrides(config: dict) -> list[str]:
    return [
        # ray autodetects the HOST's cpu count inside a rented pod and eagerly forks one idle worker
        # per core. on a 1x4090 pod that is 48 forks nothing asked for, which oom-killed the actor
        # that mattered (VERL-123). size the pool to the container instead. this also keeps the
        # storage-unit reservation above satisfiable, since it stays well under the cpu floor.
        f"ray_kwargs.ray_init.num_cpus={ray_num_cpus(config['n_gpus_per_node'])}",
        # num_gpus is absent from verl's generated ray_init node, so hydra requires add-key syntax.
        f"+ray_kwargs.ray_init.num_gpus={config['n_gpus_per_node']}",
    ]


def _critic_overrides() -> list[str]:
    return ["critic.enable=false"]


def _reward_overrides(config: dict) -> list[str]:
    return [
        "reward.reward_model.enable=false",
        # disabling the reward MODEL does not disable reward SCORING. with no custom function the
        # loop still calls the default rule-based scorer, which dispatches on data_source and raises
        # NotImplementedError for "flash_opd" (VERL-153). point it at the generated zero function.
        # both key forms are required: the legacy top-level key is migrated in the main process but
        # is not visible to the RewardLoopWorker actor, which reads reward.custom_reward_function.
        f"custom_reward_function.path={_hydra_val(config['reward_path'])}",
        "custom_reward_function.name=compute_score",
        f"reward.custom_reward_function.path={_hydra_val(config['reward_path'])}",
        "reward.custom_reward_function.name=compute_score",
    ]


def _distillation_overrides(config: dict) -> list[str]:
    return [
        "distillation._target_=flash_opd_plugin.FlashRemoteDistillationConfig",
        "distillation.enabled=true",
        "distillation.n_gpus_per_node=0",
        "distillation.nnodes=0",
        "distillation.teacher_key=index",
        f"+distillation.bridge_url={_hydra_val(config['bridge_url'])}",
        f"+distillation.bridge_token={_hydra_val(config['bridge_token'])}",
        f"+distillation.kl_penalty_coef={_hydra_val(config['kl_penalty_coef'])}",
        "distillation.distillation_loss.loss_mode=flash_groupwise_reverse_kl",
        "distillation.distillation_loss.topk=null",
        "distillation.distillation_loss.use_task_rewards=false",
        "distillation.distillation_loss.use_policy_gradient=false",
        "distillation.distillation_loss.loss_max_clamp=null",
        "distillation.distillation_loss.log_prob_min_clamp=null",
    ]


def _trainer_overrides(config: dict) -> list[str]:
    return [
        f"trainer.default_local_dir={_hydra_val(config['local_dir'])}",
        f"trainer.save_freq={_hydra_val(config['save_freq'])}",
        f"trainer.n_gpus_per_node={_hydra_val(config['n_gpus_per_node'])}",
        "trainer.nnodes=1",
        f"trainer.project_name={_hydra_val(config['project_name'])}",
        f"trainer.experiment_name={_hydra_val(config['experiment_name'])}",
        f"trainer.logger={_hydra_val(config.get('loggers', ['console']))}",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={_hydra_val(config['total_training_steps'])}",
        "trainer.val_before_train=false",
        "trainer.test_freq=-1",
        "trainer.resume_mode=auto",
        "trainer.max_actor_ckpt_to_keep=null",
    ]


def build_opd_overrides(config: dict) -> list[str]:
    """Render the exact verl 0.8.0 synchronous PPO and distillation config surface."""
    missing = [key for key in _REQUIRED_OVERRIDE_KEYS if key not in config]
    if missing:
        raise KeyError(f"build_opd_overrides missing required config keys: {missing}")
    # the full sequence length the engine is sized for. the caller derives max_prompt_length by
    # carving max_response_length out of this same value, so the token budget, the prompt filter,
    # and the engine always agree.
    max_tokens = int(config["max_sequence_length"])
    overrides = [
        *_algorithm_overrides(),
        *_data_input_overrides(config),
        *_actor_rollout_seed_overrides(config),
        # use fp8 kv where resolved because vram.py sizes against it; bf16 would double the reserved
        # cache and OOM. gdn and unsupported devices leave this unset.
        *(
            ["+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8"]
            if config.get("fp8_kv")
            else []
        ),
        *_data_runtime_overrides(config),
        *_actor_rollout_overrides(config, max_tokens=max_tokens),
        *_transfer_queue_overrides(),
        *_ray_overrides(config),
        *_critic_overrides(),
        *_reward_overrides(config),
        *_distillation_overrides(config),
        *_trainer_overrides(config),
    ]
    if config.get("multi_turn"):
        overrides.append(f"actor_rollout_ref.rollout.prompt_length={_hydra_val(max_tokens)}")
    structured_outputs = config.get("structured_outputs")
    if structured_outputs:
        structured_outputs_config = {
            "backend": "xgrammar",
            "disable_any_whitespace": bool(structured_outputs.get("disable_any_whitespace", False)),
        }
        reasoning_parser = reasoning_parser_for(
            thinking=bool(config.get("thinking", False)),
            structured_outputs=structured_outputs,
        )
        if reasoning_parser is not None:
            structured_outputs_config["reasoning_parser"] = reasoning_parser
        overrides.append(
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.structured_outputs_config="
            + _hydra_val(structured_outputs_config)
        )
    return overrides


def _build_opd_plugin_config(
    *,
    shim_dir: str,
    save_at_steps,
    total_steps: int,
    gdn_model_type: str | None,
    loggers,
    lora_language_prefix: str = "",
) -> str:
    """serialize the non-secret OPD runtime patch configuration."""
    return json.dumps(
        {
            "marker_file": shim_marker_file(shim_dir),
            "no_signal_attempts": OPD_NO_SIGNAL_ATTEMPTS,
            "save_at_steps": list(save_at_steps),
            "total_steps": int(total_steps),
            **({"lora_language_prefix": lora_language_prefix} if lora_language_prefix else {}),
            "gdn_model_type": gdn_model_type,
            "wandb": "wandb" in loggers,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _build_opd_child_env(
    *,
    shim_dir: str,
    wandb_enabled: bool,
    bridge_url: str,
    bridge_token: str,
    seed: int,
    stop_sequences: tuple,
    eos_token_ids: frozenset[int],
    structured_outputs: dict | None,
    model_vocab_size: int,
    thinking: bool,
    multi_turn: bool = False,
    max_turns: int = 0,
    max_model_len: int = 32768,
    mutation_failure_path: str = "",
    score_delivery_failure_path: str = "",
    rollout_failure_path: str = "",
    abandonment_failure_path: str = "",
    resample_failure_path: str = "",
    cycle_commit_failure_path: str = "",
    plugin_config: str = "",
) -> dict[str, str]:
    child = _build_verl_child_env(shim_dir=shim_dir, wandb_enabled=wandb_enabled)
    child.update(
        {
            "VERL_USE_EXTERNAL_MODULES": "flash_opd_plugin",
            "FLASH_OPD_BRIDGE_URL": bridge_url,
            "FLASH_OPD_BRIDGE_TOKEN": bridge_token,
            "FLASH_OPD_SEED": str(int(seed)),
            "FLASH_OPD_STOP_SEQUENCES": json.dumps(list(stop_sequences)),
            "FLASH_OPD_EOS_TOKEN_IDS": json.dumps(sorted(eos_token_ids)),
        }
    )
    if mutation_failure_path:
        child["FLASH_OPD_MUTATION_FAILURE_PATH"] = mutation_failure_path
    if score_delivery_failure_path:
        child["FLASH_OPD_SCORE_DELIVERY_FAILURE_PATH"] = score_delivery_failure_path
    if rollout_failure_path:
        child["FLASH_OPD_ROLLOUT_FAILURE_PATH"] = rollout_failure_path
    if abandonment_failure_path:
        child["FLASH_OPD_ABANDONMENT_FAILURE_PATH"] = abandonment_failure_path
    if resample_failure_path:
        child["FLASH_OPD_RESAMPLE_FAILURE_PATH"] = resample_failure_path
    if cycle_commit_failure_path:
        child["FLASH_OPD_CYCLE_COMMIT_FAILURE_PATH"] = cycle_commit_failure_path
    if multi_turn:
        child.update(
            {
                "FLASH_OPD_THINKING": "1" if thinking else "0",
                "FLASH_OPD_MAX_TURNS": str(int(max_turns)),
                "FLASH_OPD_MAX_MODEL_LEN": str(int(max_model_len)),
                "FLASH_OPD_ENV_CAPABILITIES": json.dumps(
                    [
                        "new_rollout_state",
                        "record_model_turn",
                        "env_reply",
                        "rollout_done",
                    ]
                ),
            }
        )
    if structured_outputs:
        child.update(
            {
                "FLASH_OPD_STRUCTURED_OUTPUTS": json.dumps(
                    structured_outputs, sort_keys=True, separators=(",", ":")
                ),
                "FLASH_OPD_MODEL_VOCAB_SIZE": str(int(model_vocab_size)),
                "FLASH_OPD_THINKING": "1" if thinking else "0",
            }
        )
    if plugin_config:
        child["FLASH_OPD_PLUGIN_CONFIG"] = plugin_config
    return child


def _opd_parquet_features(*, multimodal: bool, prompt: dict):
    from datasets import Features, Value

    features = {
        "prompt": [prompt],
        "data_source": Value("string"),
        "reward_model": {
            "style": Value("string"),
            "ground_truth": Value("string"),
        },
        "extra_info": {"index": Value("int64")},
    }
    if multimodal:
        features["images"] = [{"image": Value("string")}]
    return Features(features)


# arrow duplicates shared prompt references, so one table scales host ram with the full horizon.
# fixed batches keep the peak flat; this was empirically measured and will drift with prompt shape.
_OPD_PARQUET_WRITE_BATCH_ROWS = 2000


def _write_opd_parquet(rows: list[dict], path: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        raise ValueError("refusing to write an empty OPD parquet")
    from flash.engine.worker.train.entry.prompt_rows import prompt_message_features

    multimodal = any("images" in row for row in rows)
    # derive one explicit schema from the full row set before opening the writer. this keeps optional
    # reasoning authored after the first batch and the multimodal images struct stable across batches.
    schema = _opd_parquet_features(
        multimodal=multimodal,
        prompt=prompt_message_features(rows),
    ).arrow_schema
    # write to a sibling temp file and rename only once every batch landed. closing a partially
    # written ParquetWriter still emits a valid footer, so failing in place would leave a READABLE
    # short file at `path` -- a truncated horizon that trains silently instead of raising.
    partial = f"{path}.partial"
    try:
        writer = pq.ParquetWriter(partial, schema)
        try:
            for start in range(0, len(rows), _OPD_PARQUET_WRITE_BATCH_ROWS):
                writer.write_table(
                    pa.Table.from_pylist(
                        rows[start : start + _OPD_PARQUET_WRITE_BATCH_ROWS], schema=schema
                    )
                )
        finally:
            writer.close()
        os.replace(partial, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(partial)
        raise
