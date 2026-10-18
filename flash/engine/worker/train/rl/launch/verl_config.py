"""verl config rendering for the GRPO trainer.

Everything that turns a resolved flash job into the exact verl 0.8.0 surface: the parquet rows the
child reads, the hydra override list, the GPU memory-utilization split between the training and
rollout engines, and the human-readable notes attached to the run.

Split out of `flash.engine.worker.train.entry.rl_train` to keep that module under the file-size limit.
"""

from __future__ import annotations

import copy
import math
import os

import flash.engine.worker.runtime.state as _worker_state
from flash.adapters.targets import resolve_lora_targeting
from flash.content.multimodal import messages_with_decoded_images
from flash.content.structured_outputs import reasoning_parser_for
from flash.content.thinking import messages_for_chat_template
from flash.engine.worker.train.entry.backend_common import (
    actor_fsdp_strategy_overrides,
    agent_loop_workers,
    ray_num_cpus,
    rollout_layered_summon_overrides,
    rollout_mm_processor_cache_overrides,
    rollout_resident_overrides,
    rollout_sleep_unsupported,
    trainer_dtype_overrides,
)
from flash.engine.worker.train.entry.prompt_rows import (
    canonical_prompt_messages,
    prompt_message_features,
)
from flash.engine.worker.train.entry.sft_train import _hydra_val
from flash.engine.worker.verl.capabilities import rollout_max_num_seqs
from flash.engine.worker.verl.parallelism import (
    ULYSSES_SEQUENCE_PARALLEL_SIZE,
    resolve_reshard_after_forward,
)

# the data_source every flash-generated row carries. verl routes scoring by this key, so it must
# match what the reward shim registers under.
DATA_SOURCE = "flash_env"


def _verl_epochs_for_horizon(
    *, epochs: int, prompt_count: int, prompts_per_step: int, steps: int
) -> int:
    if prompt_count <= 0:
        raise ValueError("prompt_count must be positive")
    if prompts_per_step <= 0:
        raise ValueError("prompts_per_step must be positive")
    if prompt_count < prompts_per_step:
        raise ValueError("prompt_count must be at least prompts_per_step")
    # verl hardcodes drop_last=True, so each epoch serves floor(prompt_count / batch_size) steps.
    # the flag cannot be configured off, and ceil here would still under-serve the requested horizon.
    steps_per_epoch = prompt_count // prompts_per_step
    return max(epochs, math.ceil(steps / steps_per_epoch))


def build_verl_dataset_rows(
    message_prompts: list[list[dict]],
    example_indices: list[int],
    ground_truths: list[str],
    image_uris: list[list[str]] | None = None,
) -> list[dict]:
    """convert flash chat-message prompts into verl parquet rows.

    ``extra_info.index`` carries the flash example index to the reward bridge.

    multimodal rows must pair flattened ``<image>`` placeholders with an ``images`` column because
    verl re-expands them and asserts equal counts. ``<image>``, ``<video>``, and ``<audio>`` are
    reserved on every multimodal-job row; literal uses can be mistaken for media and abort loading.
    """
    if not (len(message_prompts) == len(example_indices) == len(ground_truths)):
        raise ValueError("message_prompts / example_indices / ground_truths length mismatch")
    if image_uris is not None and len(image_uris) != len(message_prompts):
        raise ValueError("image_uris length mismatch")
    rows = []
    for position, (messages, idx, gt) in enumerate(
        zip(message_prompts, example_indices, ground_truths, strict=True)
    ):
        prompt = canonical_prompt_messages(messages, multimodal=image_uris is not None)
        if image_uris is not None:
            placeholders = sum(message["content"].count("<image>") for message in prompt)
            if placeholders != len(image_uris[position]):
                raise ValueError(
                    f"multimodal prompt for example {int(idx)} has {placeholders} <image> "
                    f"placeholder(s) but {len(image_uris[position])} image(s); a literal "
                    "'<image>' in prompt text is reserved by verl's dataset loader"
                )
            for reserved in ("<video>", "<audio>"):
                if any(reserved in str(message["content"]) for message in prompt):
                    raise ValueError(
                        f"multimodal prompt for example {int(idx)} contains a literal "
                        f"'{reserved}', which verl's dataset loader reserves as a media "
                        "placeholder; flash writes no such column"
                    )
        row = {
            "data_source": DATA_SOURCE,
            "prompt": prompt,
            "ability": "flash",
            "reward_model": {"style": "rule", "ground_truth": str(gt)},
            # verl's example index is the flash rollout_examples index; the reward bridge keys on it.
            "extra_info": {"split": "train", "index": int(idx)},
        }
        if image_uris is not None:
            row["images"] = [{"image": uri} for uri in image_uris[position]]
        rows.append(row)
    return rows


def _processor_expanded_prompt(
    processor,
    messages: list[dict],
    image_descriptors: tuple[str, ...],
    package_root: str | None,
    *,
    enable_thinking: bool,
) -> tuple[list[int], str]:
    """count a multimodal prompt after processor image-token expansion.

    a bare tokenizer undercounts image rows and can admit prompts the engine rejects. returns the
    expanded ids and rendered text so the caller can also inspect the chat template's think span.
    """
    from flash.content.multimodal import decode_image_descriptors

    images = decode_image_descriptors(list(image_descriptors), package_root)
    prepared = messages_with_decoded_images(messages, images)
    rendered = processor.apply_chat_template(
        messages_for_chat_template(prepared),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        preserve_thinking=False,
    )
    model_inputs = processor(
        text=[rendered], images=images or None, videos=None, return_tensors="pt"
    )
    input_ids = model_inputs["input_ids"]
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list | tuple):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids], rendered


def _verl_grpo_parquet_features(*, multimodal: bool, prompt: dict):
    """provide one explicit arrow schema for the full grpo row set."""
    from datasets import Features, Value

    features = {
        "data_source": Value("string"),
        "prompt": [prompt],
        "ability": Value("string"),
        "reward_model": {"style": Value("string"), "ground_truth": Value("string")},
        "extra_info": {"split": Value("string"), "index": Value("int64")},
    }
    if multimodal:
        # mixed jobs can otherwise infer an all-empty images column as null, which verl cannot read.
        features["images"] = [{"image": Value("string")}]
    return Features(features)


def write_verl_grpo_parquet(rows: list[dict], path: str) -> None:
    """write grpo rows with one explicit schema derived from the full row set."""
    from datasets import Dataset

    features = _verl_grpo_parquet_features(
        multimodal=any("images" in row for row in rows),
        prompt=prompt_message_features(rows),
    )
    Dataset.from_list(rows, features=features).to_parquet(path)


def _algorithm_overrides(cfg: dict) -> list[str]:
    """preserve the objective and rollout-correction recipe."""
    return [
        "algorithm.adv_estimator=grpo",
        # dr-grpo recipe: group-mean-centered advantages with NO std normalization, and a
        # constant-length loss aggregation (no per-response length bias). matches the retired trl path's
        # scale_rewards=none + loss_type=dr_grpo.
        "algorithm.norm_adv_by_std_in_grpo=False",
        f"actor_rollout_ref.actor.loss_agg_mode={cfg['loss_agg_mode']}",
        "algorithm.use_kl_in_reward=False",
        # truncated importance sampling (token-level, cap 2.0): corrects the vllm-rollout vs
        # fsdp-train policy mismatch (token_truncate, c_max=2.0). verl otherwise defaults to
        # sequence-level tis, so pin token.
        "algorithm.rollout_correction.rollout_is=token",
        "algorithm.rollout_correction.rollout_is_threshold=2.0",
        # required for rollout importance-sampling correction. verl only writes rollout_log_probs
        # when calculate_log_probs is true (agent_loop.py:124,501), and ray_trainer.py:1608 gates
        # correction on that key. without it these overrides silently do nothing, including multi-turn.
        "actor_rollout_ref.rollout.calculate_log_probs=True",
    ]


def _data_overrides(cfg: dict) -> list[str]:
    """shape parquet loading and prompt rendering for training."""
    return [
        f"data.train_files={cfg['train_files']}",
        f"data.val_files={cfg['val_files']}",
        f"data.train_batch_size={cfg['prompts_per_step']}",
        f"data.max_prompt_length={cfg['max_prompt_len']}",
        # rollout.response_length interpolates from this key, and it is BOTH the response tensor's
        # width and the default per-request generation cap. on multi-turn it is the episode budget,
        # not the per-turn cap -- the agent loop passes an explicit per-turn max_tokens, so the
        # wider default is never used as a single turn's limit.
        f"data.max_response_length={cfg['max_response_len']}",
        "data.prompt_key=prompt",
        # rollout prompt parity: verl renders raw messages with the tokenizer's chat template;
        # thread flash's thinking mode so the rollout sees the same prompt the retired trl path saw.
        f"+data.apply_chat_template_kwargs.enable_thinking={str(bool(cfg.get('thinking', False))).lower()}",
        "+data.apply_chat_template_kwargs.preserve_thinking=false",
        f"data.seed={cfg['seed']}",
        "data.dataloader_num_workers=0",
        # set the rollout seed through engine_kwargs: verl 0.8.0 has no RolloutConfig.seed, and
        # direct keys either fail hydra or the dataclass conversion. engine_kwargs is declared and
        # overrides verl's own vllm seed; `++` is required because this sub-key is absent.
        f"++actor_rollout_ref.rollout.engine_kwargs.vllm.seed={cfg['seed']}",
        # multimodal parquet carries placeholders plus file:// image uris; verl re-expands them.
        # return_raw_chat makes the trainer build a processor, not a tokenizer. flash pre-filters with
        # the same processor, while verl's filter remains a guard because it raises on over-budget
        # multimodal prompts instead of truncating them (agent_loop.py).
        *(
            [
                "data.image_key=images",
                "data.return_raw_chat=true",
                # the agent loop recomputes multi-modal inputs itself; materializing them in the
                # dataloader as well doubles the pixel tensors held per row for no consumer.
                "data.return_multi_modal_inputs=false",
                "data.filter_overlong_prompts=true",
                "data.truncation=error",
                "actor_rollout_ref.model.trust_remote_code=true",
            ]
            if cfg.get("multimodal")
            else []
        ),
    ]


def _actor_model_overrides(cfg: dict) -> list[str]:
    """configure the trainable model and adapter loading."""
    return [
        f"actor_rollout_ref.model.path={cfg['model_path']}",
        f"actor_rollout_ref.model.lora_rank={cfg['lora_rank']}",
        f"actor_rollout_ref.model.lora_alpha={cfg['lora_alpha']}",
        f"actor_rollout_ref.model.target_modules={cfg['target_modules']}",
        f"actor_rollout_ref.model.exclude_modules={_hydra_val(cfg.get('exclude_modules'))}",
        *(
            ["++actor_rollout_ref.model.target_parameters=" + _hydra_val(cfg["target_parameters"])]
            if cfg.get("target_parameters")
            else []
        ),
        # memory: match the retired trl path's gradient checkpointing.
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        # fused linear-ce avoids materializing the roughly 130 gb [tokens, vocab] logits tensor at
        # 32k context. torch is exact and adds no dependency. packed gdn hybrids require child support
        # for seq_idx + cu_seqlens; run_rl_train raises when that boundary reset is unavailable.
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.use_fused_kernels=True",
        # backend chosen per card; see fused_ce_backend for the measured table.
        f"actor_rollout_ref.model.fused_kernel_options.impl_backend={cfg['fused_ce_backend']}",
        *(
            [f"actor_rollout_ref.model.lora_adapter_path={cfg['warmstart_adapter']}"]
            if cfg.get("warmstart_adapter")
            else []
        ),
    ]


def _actor_overrides(cfg: dict) -> list[str]:
    """configure optimization and distributed actor execution."""
    return [
        f"actor_rollout_ref.actor.optim.lr={cfg['lr']}",
        # only the lora adapter is trainable; decaying its weights toward zero is not part of grpo.
        "actor_rollout_ref.actor.optim.weight_decay=0.0",
        # 0 warmup -> verl's warmup+constant scheduler holds lr flat, matching the retired constant recipe.
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={cfg['prompts_per_step']}",
        # bound backward by tokens, not sequences: fixed sequence counts are unsafe at 32k and
        # waste capacity on short rows. this is a PER-GPU budget and ulysses is pinned off below, so
        # verl's `max_token_len_per_gpu * sp_size` (engine/utils.py) is the budget itself. one
        # full-length sequence is the floor -- see `max_token_len_per_gpu` in the caller -- which
        # satisfies verl's `max_token_len >= max_seq_len` assert without a sequence-parallel
        # multiplier. token balancing across the dp ranks is what `use_dynamic_bsz` does, and it
        # also skips verl's strict batch-divisibility validation, so no rank can be starved to zero.
        "actor_rollout_ref.actor.use_dynamic_bsz=true",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={cfg['max_token_len_per_gpu']}",
        # ppo_epochs multiplies verl's update loop, so its default of 1 preserves the requested update
        # count and on-policy baseline: verl samples a fresh rollout for every update, with no reuse.
        f"actor_rollout_ref.actor.ppo_epochs={cfg['ppo_epochs']}",
        # shard by DATA, not by sequence -- see ULYSSES_SEQUENCE_PARALLEL_SIZE for why. ref inherits
        # this via dp_ref.yaml, and use_remove_padding is required either way.
        f"actor_rollout_ref.actor.ulysses_sequence_parallel_size={ULYSSES_SEQUENCE_PARALLEL_SIZE}",
        # zero-2 vs zero-3. NOT decided here: `zero2_enabled` is the allocator's own gate, and the
        # value is resolved once in `build_verl_cfg` so the shape this run was admitted on is the
        # shape it trains under. the ref policy needs no matching key -- lora makes verl reuse the
        # actor module with the adapter disabled (`ref_in_actor`), so there is no second copy.
        # absent -> True, verl's own default: a cfg built without the gate must render the
        # lower-memory strategy, never fall into zero-2 by omission.
        f"actor_rollout_ref.actor.fsdp_config.reshard_after_forward={_hydra_val(bool(cfg.get('reshard_after_forward', True)))}",
        # grpo requires fsdp2 for dense and fused-expert actors. reshard_after_forward above still
        # independently selects zero-2 or zero-3, so the allocator's memory decision is unchanged.
        *actor_fsdp_strategy_overrides(cfg["fsdp_generation"]),
        # store the frozen base in bf16, not verl's fp32 yaml default. shared with the opd driver.
        *trainer_dtype_overrides(),
    ]


def _rollout_overrides(cfg: dict) -> list[str]:
    """configure generation, batching, and rollout engine behavior."""
    return [
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.n={cfg['group_size']}",
        # hydra's composed rollout node omits this real rollout config field.
        "++actor_rollout_ref.rollout.limit_images=4",
        # verl 0.8.0 chunks the rollout batch across agent workers with exact divisibility
        # (agent_loop.py:1111 -> protocol.py:874). choose the largest divisor of
        # prompts_per_step * group_size up to the cap, so small or final short batches cannot abort.
        # a multimodal run halves the cap: every agent worker is a ray actor holding its own
        # processor copy, and on an image run that processor is a full image processor living
        # beside the vllm engine, enginecore, load balancer and http server in one container. at
        # eight workers one actor dies with `libgomp: Thread creation failed`, and because it dies
        # mid-grading the rollouts generate but never get graded, so the run hangs until the
        # child-silence watchdog kills it 1200s later. capping the pool leaves group_size and
        # prompts_per_step exactly as authored -- only scheduling parallelism narrows.
        f"actor_rollout_ref.rollout.agent.num_workers={agent_loop_workers(int(cfg['prompts_per_step']) * int(cfg['group_size']), cap=4 if cfg.get('multimodal') else 8)}",
        # one grpo step submits exactly prompts_per_step * group_size sequences. left unset verl
        # keeps its 1024 default, and vllm sizes cuda-graph capture and (on a gdn/mamba hybrid)
        # per-slot recurrent state from that number, both up front -- so a 32-sequence run paid a
        # 1024-sequence reservation and OOMed during capture with most of the card still free.
        f"actor_rollout_ref.rollout.max_num_seqs={rollout_max_num_seqs(int(cfg['prompts_per_step']) * int(cfg['group_size']))}",
        # multi-turn runs flash's own agent loop, registered in the child by flash_grpo_plugin. the
        # stock single_turn_agent generates once and returns; it has no notion of an environment
        # reply, so a multi-turn env on the default loop would train on first turns only.
        *(
            ["actor_rollout_ref.rollout.agent.default_agent_loop=flash_grpo_multi_turn"]
            if cfg["multi_turn"]
            else []
        ),
        # fork-only rollout field, so `++` (append-or-override): it exists in the fork's rollout.yaml
        # but not in stock verl 0.8.0's, where a bare key or `+` would break. omitted entirely when
        # false, since not masking is already stock behavior and stock verl rejects the unknown key.
        *(
            [
                (
                    "++actor_rollout_ref.rollout.mask_truncated_completions="
                    f"{str(bool(cfg['mask_truncated_completions'])).lower()}"
                )
            ]
            if cfg["mask_truncated_completions"]
            else []
        ),
        # safetensors load format is required for lora rollout on vllm.
        "actor_rollout_ref.rollout.load_format=safetensors",
        # fused-expert lora models gather the weight sync one submodule at a time; the whole-model
        # summon materializes every layer's expert stack at once. requires load_format=safetensors
        # above, which verl checks as base_sync_done before allowing the layered walk.
        *rollout_layered_summon_overrides(cfg.get("target_parameters")),
        *rollout_mm_processor_cache_overrides(),
        # keep the rollout engine RESIDENT for models whose vLLM wake/reload HANGS (catalog
        # sleep_unsupported). shared with the opd driver, which runs the same verl sleep path.
        *rollout_resident_overrides(bool(cfg.get("sleep_unsupported"))),
        f"actor_rollout_ref.rollout.gpu_memory_utilization={cfg['gpu_mem_util']}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={cfg['n_gpus']}",
        f"actor_rollout_ref.rollout.temperature={cfg['temperature']}",
        f"actor_rollout_ref.rollout.top_p={cfg['top_p']}",
        # size vllm's kv cache for the job's context, not the architecture's. left unset, verl
        # substitutes the model's full max_position_embeddings and hands that straight to the
        # engine, so a 4k job on a 40k-capable model reserves ten times the kv cache it can ever
        # use -- and on a long-context model that reservation alone can exhaust the
        # gpu_memory_utilization budget before a single rollout runs.
        f"actor_rollout_ref.rollout.max_model_len={cfg['max_model_len']}",
        # verl recomputes rollout log-probs for the importance ratio regardless of the kl term.
        # the engine asserts actor.use_dynamic_bsz == rollout.log_prob_use_dynamic_bsz, so the
        # log-prob pass switches to token budgeting with the actor, never independently.
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={cfg['max_token_len_per_gpu']}",
    ]


def _reward_overrides(cfg: dict) -> list[str]:
    """route custom scoring into both verl reward entry points."""
    return [
        f"custom_reward_function.path={cfg['reward_path']}",
        f"custom_reward_function.name={cfg['reward_name']}",
        # verl trainer-v1 reward loop reads reward.custom_reward_function (the legacy top-level key
        # is migrated in the main process but not visible to the RewardLoopWorker actor); emit both.
        f"reward.custom_reward_function.path={cfg['reward_path']}",
        f"reward.custom_reward_function.name={cfg['reward_name']}",
    ]


def _ray_overrides(cfg: dict) -> list[str]:
    """bound ray resources to the rented container."""
    return [
        # ray autodetects the HOST's cpu count inside a rented pod and eagerly forks one idle worker
        # per core, which has killed real runs two ways (host-ram oom, and a fatal raylet fork
        # failure that takes every actor with it). size the pool to the container instead.
        f"ray_kwargs.ray_init.num_cpus={ray_num_cpus(cfg['n_gpus'])}",
        # num_gpus is absent from verl's generated ray_init node, so hydra requires add-key syntax.
        f"+ray_kwargs.ray_init.num_gpus={cfg['n_gpus']}",
    ]


def _trainer_overrides(cfg: dict) -> list[str]:
    """control training horizon, checkpoints, and run metadata."""
    return [
        f"trainer.n_gpus_per_node={cfg['n_gpus']}",
        "trainer.nnodes=1",
        f"trainer.total_epochs={cfg['total_epochs']}",
        # honor [train].max_steps: total_training_steps caps the optimizer-update horizon.
        f"trainer.total_training_steps={cfg['steps']}",
        f"trainer.save_freq={cfg['save_freq']}",
        f"trainer.max_actor_ckpt_to_keep={cfg['ckpt_to_keep']}",
        # resume from whatever _restore_verl_resume staged into default_local_dir. auto is a no-op
        # when nothing was staged, so a fresh run is unaffected; without it a preempted run silently
        # restarts at step 0 and re-bills the whole budget.
        "trainer.resume_mode=auto",
        "trainer.test_freq=-1",
        "trainer.val_before_train=False",
        f"trainer.logger={_hydra_val(cfg['loggers'])}",
        # the run's own wandb project/name, exactly as the sft and opd verl backends resolve them. a
        # hardcoded pair would land every grpo run in one wandb project under one experiment name, so
        # concurrent runs overwrite each other's curves and an explicit [wandb] config is ignored.
        f"trainer.project_name={_hydra_val(cfg['project_name'])}",
        f"trainer.experiment_name={_hydra_val(cfg['experiment_name'])}",
        f"trainer.default_local_dir={cfg['local_dir']}",
    ]


def build_verl_overrides(cfg: dict) -> list[str]:
    """build hydra overrides for verl grpo with lora and vllm.

    preserves flash's dr-grpo objective, kl coefficient, constant lr, seed, top_p, and one ppo epoch.
    verl samples a fresh rollout per optimizer update.
    """
    kl_on = float(cfg["kl_coef"]) > 0
    o = (
        _algorithm_overrides(cfg)
        + _data_overrides(cfg)
        + _actor_model_overrides(cfg)
        + _actor_overrides(cfg)
        + _rollout_overrides(cfg)
        + _reward_overrides(cfg)
        + _ray_overrides(cfg)
        + _trainer_overrides(cfg)
    )
    if kl_on:
        # a reference policy is active -> add the token-level kl loss. the ref worker needs no
        # batching keys of its own: ref.yaml defaults log_prob_use_dynamic_bsz and
        # log_prob_max_token_len_per_gpu to oc.select on the matching actor keys, which the actor
        # block above sets, and the engine pops them onto the ref's own config.
        o += [
            "actor_rollout_ref.actor.use_kl_loss=True",
            f"actor_rollout_ref.actor.kl_loss_coef={cfg['kl_coef']}",
        ]
    else:
        # flash default: dr-grpo with no kl term, so no reference policy.
        o.append("actor_rollout_ref.actor.use_kl_loss=False")
    if cfg.get("entropy_quantile") is not None:
        # the top-entropy shim thresholds on per-token entropy, which verl computes only when asked:
        # calculate_entropy defaults to false and entropy_coeff is 0 on the flash recipe, so without
        # this the shim would receive no entropy and raise rather than silently skip the masking.
        o.append("actor_rollout_ref.actor.calculate_entropy=True")
    if cfg.get("fp8_kv"):
        # fp8 kv cache on ada/hopper+ (cc>=8.9), matching engine/vram.py's sizing; tis (above) covers
        # the extra rollout-vs-train mismatch fp8 introduces. '+' appends the key under the existing
        # engine_kwargs.vllm struct (it is not a default field).
        o.append("+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8")
    if cfg.get("enforce_eager"):
        # this card's vllm 0.19.1 graph capture degenerates (see resolve_rollout_enforce_eager).
        # rollout.enforce_eager is a real verl field, so this is a plain override, not a '+' append.
        o.append("actor_rollout_ref.rollout.enforce_eager=True")
    # blackwell attention pins (see resolve_blackwell_attention_backends for why each default is
    # wrong). both are real AsyncEngineArgs fields in the pinned vllm 0.19.1, and verl spreads
    # engine_kwargs.vllm straight into them, so a plain override reaches the engine. '+' appends
    # under the existing engine_kwargs.vllm struct, as kv_cache_dtype does above.
    if cfg.get("attention_backend"):
        o.append(
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend="
            f"{cfg['attention_backend']}"
        )
    if cfg.get("mm_encoder_attn_backend"):
        o.append(
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_attn_backend="
            f"{cfg['mm_encoder_attn_backend']}"
        )
    _reasoning_parser = reasoning_parser_for(
        thinking=bool(cfg.get("thinking", False)),
        structured_outputs=cfg.get("structured_outputs"),
    )
    if _reasoning_parser:
        # the engine half of structured outputs: hold the guided grammar until </think> closes, so a
        # thinking model reasons freely and only its answer is constrained. on the retired trl path this is
        # injected into the colocate llm kwargs; verl spreads engine_kwargs.vllm straight into
        # AsyncEngineArgs, where reasoning_parser is a real field, so a plain override suffices.
        # '+' appends under the existing engine_kwargs.vllm struct, as kv_cache_dtype does above.
        o.append(
            f"+actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser={_reasoning_parser}"
        )
    return o


_DEFAULT_GPU_MEM_UTIL = 0.5


def resolve_gpu_mem_util(
    inp: dict,
    *,
    gpu_type: str,
    n_gpus: int,
    fp8_kv: bool,
    sleep_unsupported: bool,
    preserve_legacy_floor: bool = False,
) -> float:
    """size vllm's colocated executor budget from this run's geometry.

    ``gpu_memory_utilization`` covers the second bf16 weight copy, lora adapter, and kv pool. using
    ``colocate_kv_util`` keeps worker allocation aligned with preflight admission.

    for multi-gpu runs, vllm shards its bf16 weight copy across tensor-parallel ranks. its default
    lora layout shards only one factor of each projection's a/b pair, so the rank-local adapter uses
    the larger safe orientation from catalog geometry rather than dividing the full adapter evenly.
    the kv term remains unsharded here because vllm can replicate kv heads when tensor parallelism is
    wider than the model's kv-head count. the adapter lives inside vllm's reservation, and the result
    is explicitly capped at the old 0.5 constant, so this path can only lower a rank's reservation
    and cannot create a new over-reservation.

    the flat constant is kept where the model does not apply, because a wrong number is worse than
    a conservative one:

    - unknown card (empty/unmanaged ``gpu.type``): the budget is a fraction of the card, so without
      its size there is nothing to take a fraction of.
    - pinned revision with no resolvable parameter count: the weight term dominates, so a guessed
      size is not worth acting on.
    - any sizing exception: launch falls back to the previous constant instead of failing.
    """
    if not (gpu_type or "").strip():
        return _DEFAULT_GPU_MEM_UTIL
    try:
        from flash.core.catalog import MODELS
        from flash.engine.plan.vram import colocate_kv_util, resolve_params_b
        from flash.providers.core.base import get_gpu_info

        total_vram_gb = float(get_gpu_info(gpu_type).vram_gb)
        if total_vram_gb <= 0:
            return _DEFAULT_GPU_MEM_UTIL
        model_id = str(inp["model_id"])
        revision = str(inp.get("model_revision") or "")
        catalog_info = MODELS.get(model_id)
        # a pinned revision is sized on generic architecture geometry:
        # the commit's real geometry is not validated, so the curated kv/lora shape may not describe
        # it. the parameter count still comes from the pinned config.
        info = None if revision else catalog_info
        # the ONE way the worker, the preflight and the cost estimator agree on a model's size:
        # curated catalog params_b, else the pinned commit's real HF count. the catalog answers
        # FIRST and locally, so the common path adds no network call to launch; the HF lookup is
        # reached only for a pinned revision, already warm in the hub cache from the upstream
        # config fetch for this same model.
        params_b = float(
            (getattr(catalog_info, "params_b", 0.0) or 0.0)
            if (catalog_info is not None and not revision)
            else (resolve_params_b(model_id, revision=revision) or 0.0)
        )
        if params_b <= 0:
            return _DEFAULT_GPU_MEM_UTIL
        sized = colocate_kv_util(
            params_b,
            int(inp["engine_len"]),
            total_vram_gb,
            # the engine stays RESIDENT for a model whose vLLM wake/reload hangs, and the sleep
            # branch budgets a 1.5x larger pool on the grounds that the engine is offloaded during
            # the backward. crediting an offload that never happens would size the pool against a
            # peak the run does not have.
            sleep_mode=not sleep_unsupported,
            num_generations=int(inp["group_size"]),
            active_params_b=float(getattr(info, "active_params_b", 0.0) or 0.0) or None,
            fp8_kv=fp8_kv,
            model_info=info,
            preserve_legacy_floor=preserve_legacy_floor,
            tensor_parallel=n_gpus,
            lora_rank=int(inp["lora_rank"]),
        )
        # multi-rank sizing is strictly one-directional against the previous worker contract: it may
        # free memory but can never claim more than the flat reservation a working run already used.
        return min(_DEFAULT_GPU_MEM_UTIL, sized) if n_gpus > 1 else sized
    except Exception as e:  # sizing must never be what stops a run from launching
        print(
            f"[rl-verl] gpu_memory_utilization sizing failed ({e}); using {_DEFAULT_GPU_MEM_UTIL}"
        )
        return _DEFAULT_GPU_MEM_UTIL


def _build_verl_training_cfg(
    inp: dict,
    *,
    train_files: str,
    val_files: str,
    # the local snapshot dir verl loads weights from, not the hf repo id. anything needing the
    # catalog id reads inp["model_id"].
    model_path: str,
    thinking: bool,
    loggers: list[str],
    fp8_kv: bool,
    enforce_eager: bool,
    attention_backend: str | None,
    mm_encoder_attn_backend: str | None,
    reward_path: str,
    local_dir: str,
    project_name: str,
    experiment_name: str,
    gpu_type: str = "",
    n_gpus: int = 1,
    # the authored [train] table, used only to size the zero-2 gate against the same knobs the
    # allocator sized the shape with. absent -> the gate falls closed to verl's zero-3 default.
    train_spec=None,
    # NOT named fused_ce_backend: that is the imported resolver, and a parameter of the same name
    # would shadow it inside this function so a later `fused_ce_backend(caps)` call silently
    # returned a string. required, since every caller resolves it from the capability probe.
    ce_backend: str,
) -> dict:
    engine_len = int(inp["engine_len"])
    sleep_unsupported = rollout_sleep_unsupported(inp["model_id"])
    targeting = resolve_lora_targeting(
        inp["model_id"], algorithm="grpo", multimodal=bool(inp.get("multimodal"))
    )
    return {
        "fused_ce_backend": ce_backend,
        "train_files": train_files,
        "val_files": val_files,
        "model_path": model_path,
        "lora_rank": inp["lora_rank"],
        "lora_alpha": inp["lora_alpha"],
        "target_modules": targeting.target_modules,
        "exclude_modules": None,
        # the catalog id, never model_path: fused routed-expert parameters are part of the same
        # resolved target surface and must not be derived from the local snapshot path.
        "target_parameters": targeting.target_parameters,
        "fsdp_generation": inp["fsdp_generation"],
        "multimodal": bool(inp.get("multimodal")),
        "lr": inp["lr"],
        "group_size": inp["group_size"],
        "prompts_per_step": inp["prompts_per_step"],
        "mask_truncated_completions": inp["mask_truncated_completions"],
        # already clamped to the model's limit by the resolver, which derives max_prompt_len from
        # the same value -- so the engine, the prompt filter, and the token budget cannot disagree.
        "max_model_len": engine_len,
        # one full-length sequence is the floor: a budget below engine_len cannot schedule the
        # longest sequence the engine will ever produce, and verl would fail to place it in any
        # micro-batch. this is the same sequence_length-derived budget sft and opd use.
        "max_token_len_per_gpu": engine_len,
        "max_prompt_len": inp["max_prompt_len"],
        "max_completion": inp["max_completion"],
        "max_response_len": inp["max_response_len"],
        "multi_turn": bool(inp["multi_turn"]),
        "temperature": inp["temperature"],
        "top_p": inp["top_p"],
        "kl_coef": inp["kl_coef"],
        "entropy_quantile": inp["entropy_quantile"],
        "stop_sequences": inp["stop_sequences"],
        "structured_outputs": inp["structured_outputs"],
        "thinking": thinking,
        "loss_agg_mode": "seq-mean-token-sum-norm",
        "seed": inp["seed"],
        "ppo_epochs": inp["ppo_epochs"],
        "steps": int(inp["steps"]),
        "warmstart_adapter": inp["warmstart_adapter"],
        "gpu_mem_util": resolve_gpu_mem_util(
            inp,
            gpu_type=gpu_type,
            n_gpus=n_gpus,
            fp8_kv=fp8_kv,
            sleep_unsupported=sleep_unsupported,
        ),
        # `train` is the AUTHORED spec table, never `inp`: the sizer reads `max_context_tokens` /
        # `max_completion_tokens`, which `inp` carries under resolved names (`engine_len`), so
        # passing `inp` would size the gate against recipe defaults and answer a different question
        # than the allocator asked.
        "reshard_after_forward": resolve_reshard_after_forward(
            model_id=inp["model_id"],
            algorithm="grpo",
            gpu_type=gpu_type,
            n_gpus=n_gpus,
            train=train_spec,
            thinking=thinking,
            model_revision=str(inp.get("model_revision") or ""),
        ),
        "n_gpus": n_gpus,
        "loggers": loggers,
        "fp8_kv": fp8_kv,
        "enforce_eager": enforce_eager,
        "attention_backend": attention_backend,
        "mm_encoder_attn_backend": mm_encoder_attn_backend,
        "sleep_unsupported": sleep_unsupported,
        "reward_path": reward_path,
        "reward_name": "compute_score",
        "total_epochs": inp["verl_total_epochs"],
        # already the gcd of any exact save steps, so verl's modulo save lands on every one of them.
        "save_freq": inp["save_freq"],
        "ckpt_to_keep": inp["ckpt_to_keep"],
        "local_dir": local_dir,
        "project_name": project_name,
        "experiment_name": experiment_name,
    }


def _build_verl_train_notes(
    inp: dict,
    *,
    steps_run: int,
    retained_prompts: int,
    reward_history: list[float],
    loss_curve: list[float],
    resumed: bool = False,
    download_seconds: float = 0.0,
    device_peak_gpu_gb: float | None = None,
    fp8_kv: bool = False,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    wandb_url: str | None = None,
    wandb_id: str | None = None,
    reward_bridge_batching: bool = False,
    gdn_boundary_resets: bool | None = None,
    host_census: dict | None = None,
    rollout_identity_evidence: dict | None = None,
    advantage_spread_history: list[float] | None = None,
    advantage_bounds: list[dict] | None = None,
    grad_norm_evidence: list[dict] | None = None,
    multi_turn_accounting: dict | None = None,
) -> dict:
    return {
        "backend": "verl",
        "steps": steps_run,
        "epochs": inp["epochs"],
        "retained_prompts": retained_prompts,
        # matches the retired trl path: without it a resumed run is indistinguishable from a fresh one.
        "resumed": resumed,
        "group_size": inp["group_size"],
        "reward_history": reward_history,
        "loss_curve": loss_curve,
        "download_seconds": download_seconds,
        "hf_transfer": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", ""),
        # verl trains out-of-process, so torch's in-process allocator counter would read ~0 here and
        # miss the trainer entirely. nvidia-smi is the only reading that sees the child, so both keys
        # carry the same device-level figure. (on the retired in-process trl path peak_gpu_gb was the
        # torch-allocated subset of device_peak_gpu_gb; consumers must not assume that split here.)
        "peak_gpu_gb": device_peak_gpu_gb,
        "device_peak_gpu_gb": device_peak_gpu_gb,
        # verl derives generated tokens from the response lengths it actually observed, not from a
        # padded upper bound. the flag is published so a consumer comparing against historical runs
        # (which counted the padded bound) does not read the two as equivalent.
        "gen_tokens_is_upper_bound": False,
        "thinking": _worker_state.THINKING,
        # the console is uploaded only on failure, so a SUCCESSFUL run has no other record that fp8 kv
        # engaged. this is resolved per-card (cc>=8.9, and never for gdn hybrids), so it is a property
        # of the run rather than of the config.
        "vllm_kv_cache_dtype": "fp8" if fp8_kv else None,
        # persist whether the child can reset gdn state across packed examples. this is known only
        # from a child probe and otherwise disappears with successful-run console logs; none means the
        # model is not gdn.
        "gdn_boundary_resets": gdn_boundary_resets,
        # an explicit vllm prefill-batch pin is only needed when the caller hardcodes 4096; this path sets no
        # such override, so the engine keeps its own default. None records "not pinned by flash"
        # rather than asserting a number flash never chose.
        "vllm_max_num_batched_tokens": None,
        "max_completion_len": inp["max_completion"],
        "prompts_per_step": inp["prompts_per_step"],
        # one optimizer step consumes exactly this many completions. still exact under data
        # parallelism: `use_dynamic_bsz` token-balances the SAME global batch across the dp ranks
        # rather than dropping a remainder, so splitting it across cards does not change the count.
        "generations_per_step": inp["prompts_per_step"] * inp["group_size"],
        # the retired trl path fixed a per-device SEQUENCE count and carried the rest in grad accumulation. verl
        # bounds the backward pass by TOKENS instead (use_dynamic_bsz + ppo_max_token_len_per_gpu),
        # so a micro-batch holds however many sequences fit the budget and varies step to step.
        # there is no constant to report, and reporting a fabricated one would read as comparable.
        "per_device_train_batch_size": None,
        "gradient_accumulation_steps": None,
        "ppo_max_token_len_per_gpu": inp["engine_len"],
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        # the sdk's link_wandb reads notes["wandb_url"]; an in-process trainer gets it from the parent's live
        # wandb.run, verl from the child marker (see backend_common.render_wandb_link_shim).
        "wandb_url": wandb_url,
        "wandb_id": wandb_id,
        "reward_bridge_batching": bool(reward_bridge_batching),
        "host_census": copy.deepcopy(host_census or {}),
        "rollout_identity_evidence": copy.deepcopy(
            rollout_identity_evidence or {"steps": [], "validation": []}
        ),
        "advantage_spread_history": list(advantage_spread_history or []),
        "advantage_bounds": copy.deepcopy(advantage_bounds or []),
        "grad_norm_evidence": copy.deepcopy(grad_norm_evidence or []),
        # episode and turn totals for a multi-turn run; None for single-turn, which has no
        # episode loop to account for. published in the durable notes rather than only on the
        # heartbeat because this is the evidence that separates "multi-turn was configured" from
        # "multi-turn actually iterated", and a one-turn collapse passes every other gate.
        "multi_turn_accounting": copy.deepcopy(multi_turn_accounting),
        "grpo_recipe": {
            "kl_coef": inp["kl_coef"],
            "entropy_quantile": inp["entropy_quantile"],
            "stop_sequences": list(inp["stop_sequences"]),
            "structured_outputs": inp["structured_outputs"],
            "temperature": inp["temperature"],
            "top_p": inp["top_p"],
            "ppo_epochs": inp["ppo_epochs"],
            "verl_total_epochs": inp["verl_total_epochs"],
            "seed": inp["seed"],
            "loss_agg_mode": "seq-mean-token-sum-norm",
            "norm_adv_by_std_in_grpo": False,
        },
    }
