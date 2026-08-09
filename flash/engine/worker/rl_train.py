"""verl-backed grpo training path for the fine-tuning worker.

verl runs in a separate interpreter because its torch/vllm pins conflict with flash. rewards
cross a localhost rpc bridge into flash's live env.

supports non-tool, single/multi-turn, text/multimodal grpo. function-calling tool envs raise.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import threading
import time
from collections.abc import Callable
from functools import reduce
from http.server import BaseHTTPRequestHandler
from math import gcd

from flash.content.structured_outputs import (
    describe_structured_outputs,
    parse_structured_outputs,
    reasoning_parser_for,
)
from flash.core.spec import DEFAULT_CREDIT_ASSIGNMENT, gpu_count_of
from flash.engine.plan.recipe import RECIPE
from flash.engine.plan.steps import (
    final_save_due,
    on_policy_steps,
    resolve_update_horizon,
    validate_save_steps,
)
from flash.engine.profiling.sft_workload import (
    _materialize_verl_images,
    _multimodal_messages_with_images,
)
from flash.engine.result.rollout_samples import (
    sample_completion_text,
    sanitize_rollout_text,
)
from flash.engine.worker.backend_common import (
    _ORPHANED_PIPE_GRACE_S,
    _TEARDOWN_GRACE_S,
    VERL_REQUIREMENT,
    BoundedThreadingHTTPServer,
    ChildOutputTail,
    _ChildExitWatchdog,
    adopt_orphaned_descendants,
    agent_loop_workers,
    append_step_metrics,
    clamp_engine_len,
    completed_checkpoint_step,
    export_peft_adapter,
    fused_ce_backend,
    gdn_probe_module,
    kill_process_group,
    latest_global_step_dir,
    model_max_position_embeddings,
    parse_verl_metric,
    parse_verl_step_metrics,
    parse_wandb_link,
    probe_verl_capabilities,
    raise_for_classified_verl_exit,
    ray_num_cpus,
    reap_stragglers,
    render_gdn_varlen_shim,
    render_tf32_shim,
    render_wandb_link_shim,
    require_gdn_boundary_resets,
    resolve_blackwell_attention_backends,
    resolve_rollout_enforce_eager,
    resolve_verl_loggers,
    resolve_verl_python,
    rollout_resident_overrides,
    rollout_sleep_unsupported,
    stage_verl_resume,
    stamp_adapter_dir_provenance,
    trainer_dtype_overrides,
    unprocessed_checkpoint_dirs,
    verl_declares_rollout_field,
    verl_device_capability,
)
from flash.engine.worker.io.heartbeat import (
    GRPO_METRIC_HISTORY_LIMIT,
    LATEST_GRPO_METRICS_LAST,
    RewardObservabilityBuffer,
    join_while_draining,
    liveness_heartbeat,
)
from flash.engine.worker.io.hf import _deployable_adapter_on_hf
from flash.engine.worker.model.packing import model_is_gdn_hybrid
from flash.engine.worker.perf import gpu_diagnostics, wait_for_gpu
from flash.engine.worker.runtime.pkg_proxy import W as _w
from flash.engine.worker.runtime.rng import backend_seed, seed_training_rngs
from flash.engine.worker.score_batcher import ScoreBatcher
from flash.engine.worker.sft_train import (
    _build_verl_child_env,
    _cached_model_path,
    _hydra_val,
    _NvidiaSmiPeakSampler,
    _verl_image_message_content,
)
from flash.engine.worker.train.core.child.glue import validate_glue_template
from flash.engine.worker.train.opd.gkd import generation_eos_from_cached_config
from flash.engine.worker.train.rl.scoring import RolloutScoreRequest, score_rollouts
from flash.engine.worker.train.rl.shims import (
    render_entropy_quantile_shim,
    render_exact_save_steps_shim,
    render_image_pad_ban_shim,
    render_kl_ref_adapter_shim,
    render_per_turn_credit_shim,
    render_reentrant_checkpointing_shim,
    render_reward_module,
    render_stop_sequences_shim,
    render_structured_outputs_shim,
)

DATA_SOURCE = "flash_env"

# how many concurrently-finished episodes the multi-turn bridge scores in ONE env call. a whole
# generation is prompts_per_step * group_size episodes and they finish at different turn counts,
# so the batch is whatever has landed when the first waiter's grace period expires rather than a
# barrier over the step. the grace period is short because the cost being amortised is a judge
# round-trip: an episode should never wait longer than one is worth.
_MULTI_TURN_SCORE_BATCH_SIZE = 64
_MULTI_TURN_SCORE_FLUSH_WAIT_S = 0.1
_MULTI_TURN_SCORE_SHUTDOWN_WAIT_S = 5.0
# reap sessions abandoned by dead ray actors, which otherwise retain env state forever.
# the timeout must cover a live turn's full generation and env reply; late dead-session cleanup only
# holds memory, while premature cleanup breaks a working rollout.
_MULTI_TURN_SESSION_LEASE_S = 1800.0
_SINGLE_TURN_SCORE_BATCH_SIZE = 64
_SINGLE_TURN_SCORE_FLUSH_WAIT_S = 0.1
_SINGLE_TURN_SCORE_SHUTDOWN_WAIT_S = 5.0

# size the listen backlog for the full prompts_per_step * group_size connection burst.
# socketserver's default of 5 resets overflowed clients, and bridge_post intentionally does not retry.
# a fixed backlog only moves the cliff; linux may clamp this value to somaxconn.
_REWARD_BRIDGE_MIN_REQUEST_BACKLOG = 128


# --------------------------------------------------------------------------------------------
# pure helpers (no verl, no gpu, no network) -- unit tested directly.
# --------------------------------------------------------------------------------------------
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
        prompt: list[dict] | list
        if image_uris is None:
            prompt = messages
        else:
            prompt = [
                {
                    "role": str(message.get("role") or ""),
                    "content": _verl_image_message_content(message.get("content")),
                }
                for message in messages
            ]
            placeholders = sum(str(message["content"]).count("<image>") for message in prompt)
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
    prepared = _multimodal_messages_with_images(messages, images)
    rendered = processor.apply_chat_template(
        prepared, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
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


def _verl_grpo_parquet_features():
    """provide an explicit arrow schema for multimodal grpo rows.

    mixed text/image jobs can infer an all-empty ``images`` column as null, which verl cannot read
    as a struct. the pinned schema keeps every row compatible.
    """
    from datasets import Features, Value

    return Features(
        {
            "data_source": Value("string"),
            "prompt": [{"role": Value("string"), "content": Value("string")}],
            "images": [{"image": Value("string")}],
            "ability": Value("string"),
            "reward_model": {"style": Value("string"), "ground_truth": Value("string")},
            "extra_info": {"split": Value("string"), "index": Value("int64")},
        }
    )


def write_verl_grpo_parquet(rows: list[dict], path: str) -> None:
    """write grpo rows to parquet, pinning the schema when the job is multimodal."""
    from datasets import Dataset

    multimodal = any("images" in row for row in rows)
    features = _verl_grpo_parquet_features() if multimodal else None
    Dataset.from_list(rows, features=features).to_parquet(path)


def build_verl_overrides(cfg: dict) -> list[str]:
    """build hydra overrides for verl grpo with lora and vllm.

    preserves flash's dr-grpo objective, kl coefficient, constant lr, seed, top_p, and one ppo epoch.
    verl samples a fresh rollout per optimizer update.
    """
    kl_on = float(cfg["kl_coef"]) > 0
    o = [
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
        f"data.seed={cfg['seed']}",
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
                # the processor's image loader is not fork-safe under verl's default workers.
                "data.dataloader_num_workers=0",
                "actor_rollout_ref.model.trust_remote_code=true",
            ]
            if cfg.get("multimodal")
            else []
        ),
        f"actor_rollout_ref.model.path={cfg['model_id']}",
        f"actor_rollout_ref.model.lora_rank={cfg['lora_rank']}",
        f"actor_rollout_ref.model.lora_alpha={cfg['lora_alpha']}",
        f"actor_rollout_ref.model.target_modules={cfg['target_modules']}",
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
        f"actor_rollout_ref.actor.optim.lr={cfg['lr']}",
        # only the lora adapter is trainable; decaying its weights toward zero is not part of grpo.
        "actor_rollout_ref.actor.optim.weight_decay=0.0",
        # 0 warmup -> verl's warmup+constant scheduler holds lr flat, matching the retired constant recipe.
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={cfg['prompts_per_step']}",
        # bound backward by tokens, not sequences: fixed sequence counts are unsafe at 32k and
        # waste capacity on short rows. verl multiplies this per-gpu budget by sp_size, so do not
        # pre-divide it by the ulysses width.
        "actor_rollout_ref.actor.use_dynamic_bsz=true",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={cfg['max_token_len_per_gpu']}",
        # ppo_epochs multiplies verl's update loop, so its default of 1 preserves the requested update
        # count and on-policy baseline: verl samples a fresh rollout for every update, with no reuse.
        f"actor_rollout_ref.actor.ppo_epochs={cfg['ppo_epochs']}",
        # shard sequence, not batch, so sp == n_gpus keeps dp == 1 and preserves the exact global
        # batch and gradient. vllm uses the same width for tensor parallelism; activations then divide
        # across cards. ref inherits this via dp_ref.yaml, and use_remove_padding is required.
        f"actor_rollout_ref.actor.ulysses_sequence_parallel_size={cfg['n_gpus']}",
        # store the frozen base in bf16, not verl's fp32 yaml default. shared with the opd driver.
        *trainer_dtype_overrides(),
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.n={cfg['group_size']}",
        # verl 0.8.0 chunks the rollout batch across agent workers with exact divisibility
        # (agent_loop.py:1111 -> protocol.py:874). choose the largest divisor of
        # prompts_per_step * group_size up to 8, so small or final short batches cannot abort.
        f"actor_rollout_ref.rollout.agent.num_workers={agent_loop_workers(int(cfg['prompts_per_step']) * int(cfg['group_size']))}",
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
                "++actor_rollout_ref.rollout.mask_truncated_completions="
                f"{str(bool(cfg['mask_truncated_completions'])).lower()}"
            ]
            if cfg["mask_truncated_completions"]
            else []
        ),
        # safetensors load format is required for lora rollout on vllm.
        "actor_rollout_ref.rollout.load_format=safetensors",
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
        f"custom_reward_function.path={cfg['reward_path']}",
        f"custom_reward_function.name={cfg['reward_name']}",
        # verl trainer-v1 reward loop reads reward.custom_reward_function (the legacy top-level key
        # is migrated in the main process but not visible to the RewardLoopWorker actor); emit both.
        f"reward.custom_reward_function.path={cfg['reward_path']}",
        f"reward.custom_reward_function.name={cfg['reward_name']}",
        # ray autodetects the HOST's cpu count inside a rented pod and eagerly forks one idle worker
        # per core, which has killed real runs two ways (host-ram oom, and a fatal raylet fork
        # failure that takes every actor with it). size the pool to the container instead.
        f"ray_kwargs.ray_init.num_cpus={ray_num_cpus(cfg['n_gpus'])}",
        # num_gpus is absent from verl's generated ray_init node, so hydra requires add-key syntax.
        f"+ray_kwargs.ray_init.num_gpus={cfg['n_gpus']}",
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
) -> float:
    """size vllm's colocated executor budget from this run's geometry.

    ``gpu_memory_utilization`` covers the second bf16 weight copy plus kv pool. using
    ``colocate_kv_util`` keeps worker allocation aligned with preflight admission.

    the flat constant is kept where the model does NOT apply, because a wrong number is worse than
    a conservative one:

    - UNKNOWN CARD (empty/unmanaged ``gpu.type``): the budget is a fraction of the card, so without
      its size there is nothing to take a fraction of.
    - MULTI-GPU (``n_gpus > 1``): the rollout is tensor-parallel, so vLLM's weight copy is sharded
      ACROSS cards while ``colocate_kv_util`` sizes one whole copy against one card. a single-card
      number would over-reserve per rank on exactly the shapes already tight.
    - PINNED REVISION with no resolvable parameter count: the weight term dominates, so a guessed
      size is not worth acting on.
    """
    if n_gpus > 1 or not (gpu_type or "").strip():
        return _DEFAULT_GPU_MEM_UTIL
    try:
        from flash.core.catalog import MODELS
        from flash.engine.plan.vram import colocate_kv_util, resolve_params_b
        from flash.providers.base import get_gpu_info

        total_vram_gb = float(get_gpu_info(gpu_type).vram_gb)
        if total_vram_gb <= 0:
            return _DEFAULT_GPU_MEM_UTIL
        model_id = str(inp["model_id"])
        revision = str(inp.get("model_revision") or "")
        catalog_info = MODELS.get(model_id)
        # a pinned revision is sized on generic architecture geometry, matching grpo_fits_resident:
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
        return colocate_kv_util(
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
        )
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
    model_id: str,
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
    # NOT named fused_ce_backend: that is the imported resolver, and a parameter of the same name
    # would shadow it inside this function so a later `fused_ce_backend(caps)` call silently
    # returned a string. required, since every caller resolves it from the capability probe.
    ce_backend: str,
) -> dict:
    engine_len = int(inp["engine_len"])
    sleep_unsupported = rollout_sleep_unsupported(inp["model_id"])
    return {
        "fused_ce_backend": ce_backend,
        "train_files": train_files,
        "val_files": val_files,
        "model_id": model_id,
        "lora_rank": inp["lora_rank"],
        "lora_alpha": inp["lora_alpha"],
        "target_modules": "all-linear",
        "target_parameters": _w.lora_target_parameters(model_id),
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


def _step_intervals(step_line_times: list[float]) -> list[float]:
    """measure step walls from consecutive metric arrival times.

    n lines bound n-1 steps; startup before the first and partial work after the last are excluded.
    """
    return [
        later - earlier for earlier, later in itertools.pairwise(step_line_times) if later > earlier
    ]


def _measured_idle_fraction(
    reward_profile,
    *,
    completions_per_step: int,
    step_intervals: list[float],
) -> float | None:
    """measure the gpu's grading-idle share from this run's own timings.

    combine profiled grading latency with the median gap between consecutive step metric lines.
    ``train_wall / steps_run`` is invalid because it includes startup/upload and mixes session wall
    with an absolute resumed step count. the median resists checkpoint and lazy-compilation outliers.

    return none when the profile or completed-step evidence is missing, or grading consumes the
    entire measured step and leaves no trustworthy gpu-bound remainder.
    """
    if reward_profile is None or not reward_profile.trustworthy:
        return None
    completions = max(0, int(completions_per_step))
    intervals = [float(gap) for gap in step_intervals if gap > 0]
    if completions <= 0 or not intervals:
        return None
    step_wall = statistics.median(intervals)
    reward_s = reward_profile.seconds_per_completion * completions
    gpu_seconds = step_wall - reward_s
    if gpu_seconds <= 0:
        return None
    from flash.engine.profiling.reward_profile import gpu_idle_fraction

    return gpu_idle_fraction(reward_profile.seconds_per_completion, completions, gpu_seconds)


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
    reward_profile=None,
    step_intervals: list[float] | None = None,
    reward_bridge_batching: bool = False,
    gdn_boundary_resets: bool | None = None,
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
        "thinking": _w.THINKING,
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
        # one optimizer step consumes exactly this many completions: ulysses shards along the
        # sequence, so dp stays 1 and the global batch is not split across cards.
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
        # scalar grader latency remains useful after batching, but multiplying it by every completion
        # no longer measures the reward wall. publish no idle fraction until the batch-aware estimator
        # has a measured batch denominator rather than presenting the old serial projection as fact.
        "reward_bridge_batching": bool(reward_bridge_batching),
        "reward_seconds_per_completion": (
            reward_profile.seconds_per_completion
            if reward_profile is not None and reward_profile.trustworthy
            else None
        ),
        "reward_gpu_idle_fraction": (
            None
            if reward_bridge_batching
            else _measured_idle_fraction(
                reward_profile,
                completions_per_step=inp["prompts_per_step"] * inp["group_size"],
                step_intervals=step_intervals or [],
            )
        ),
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


def _single_turn_scoring_state(
    solution_str: str, *, thinking: bool, prompt_opened_thinking: bool
) -> tuple[str, dict | None]:
    graded = _w.graded_text(solution_str, prompt_opened_thinking=prompt_opened_thinking)
    state = (
        {
            "raw": solution_str,
            "completion": graded,
            "thinking": _w.thinking_text(
                solution_str, prompt_opened_thinking=prompt_opened_thinking
            ),
        }
        if thinking
        else None
    )
    return graded, state


def _finalize_single_turn_reward(
    reward: float,
    solution_str: str,
    *,
    tok,
    thinking: bool,
    prompt_opened_thinking: bool,
    think_penalty: float,
    raise_on_error: bool = False,
) -> float:
    if think_penalty > 0 and thinking:
        reward -= think_penalty * _w.think_token_count(
            solution_str, tok, prompt_opened_thinking=prompt_opened_thinking
        )
    reward = float(reward)
    # verl computes the grpo baseline with plain mean/std, so one non-finite row poisons every
    # advantage in its group. match the scalar and batched paths by masking it before the bridge.
    if not math.isfinite(reward):
        if raise_on_error:
            raise ValueError(f"env scoring returned a non-finite reward: {reward}")
        print(f"[rl-verl] env scored {reward}; unscorable, scoring 0.0", flush=True)
        return 0.0
    return reward


def score_single_turn(
    env,
    solution_str: str,
    ex,
    *,
    tok,
    thinking: bool,
    prompt_opened_thinking: bool,
    think_penalty: float,
    raise_on_error: bool = False,
    breakdowns: list[dict[str, float] | None] | None = None,
) -> float:
    """score one single-turn text completion against flash's live env.

    mirrors the training path, including optional thinking penalty. training errors become 0.0;
    ``raise_on_error`` lets profiling distinguish a real zero from failed grading.

    append named breakdowns only when provided. grading failures append none so reported component
    means count them as zero without inventing metrics for scalar-only envs.
    """
    # optimistic until the probe answers: a probe that RAISES is itself a failed grading, and the
    # except branch has to be able to record it -- omitting it instead would drop this completion
    # from the denominator and bias every named metric high. narrowed the instant the probe returns.
    collect_breakdown = breakdowns is not None
    try:
        # probe INSIDE the guard: `scores_breakdown` may be a property or a proxy whose lookup
        # raises something other than AttributeError, and out here that escapes into the reward http
        # handler -- the verl child reads a bridge failure and aborts the run, when this function's
        # whole contract is to turn an env scoring fault into 0.0.
        has_breakdown = hasattr(env, "scores_breakdown")
        collect_breakdown = collect_breakdown and has_breakdown
        graded, state = _single_turn_scoring_state(
            solution_str,
            thinking=thinking,
            prompt_opened_thinking=prompt_opened_thinking,
        )
        if has_breakdown:
            breakdown = env.scores_breakdown(graded, ex, state)
            # convert total FIRST: a breakdown whose total is not a number is a failed grading, and
            # recording its named components would credit metrics to a completion that scored 0.0.
            r = float(breakdown.get("total", 0.0))
            if collect_breakdown:
                breakdowns.append(breakdown)
        else:
            r = float(env.reward(graded, ex, state))
    except Exception as exc:  # env scoring must not kill the run
        if raise_on_error:
            raise
        if collect_breakdown:
            breakdowns.append(None)
        print(
            f"[rl-verl] env scoring raised ({type(exc).__name__}: {exc}); scoring 0.0", flush=True
        )
        return 0.0
    return _finalize_single_turn_reward(
        r,
        solution_str,
        tok=tok,
        thinking=thinking,
        prompt_opened_thinking=prompt_opened_thinking,
        think_penalty=think_penalty,
        raise_on_error=raise_on_error,
    )


def score_single_turn_batch(
    env,
    requests: list[tuple[str, dict]],
    *,
    tok,
    thinking: bool,
    prompt_opened_thinking: bool,
    think_penalty: float,
) -> list[tuple[float, list[dict[str, float] | None]]]:
    """score a batch of single-turn completions while preserving scalar failure semantics."""
    if not requests:
        return []

    def score_one_serially(solution_str, ex):
        # score_single_turn finalizes internally, so what it returns is already a FINAL row -- the
        # trailing finalize loop must not touch it again.
        breakdowns: list[dict[str, float] | None] = []
        score = score_single_turn(
            env,
            solution_str,
            ex,
            tok=tok,
            thinking=thinking,
            prompt_opened_thinking=prompt_opened_thinking,
            think_penalty=think_penalty,
            breakdowns=breakdowns,
        )
        return score, breakdowns

    def score_serially():
        return [score_one_serially(solution_str, ex) for solution_str, ex in requests]

    try:
        prepared = []
        for solution_str, ex in requests:
            graded, state = _single_turn_scoring_state(
                solution_str,
                thinking=thinking,
                prompt_opened_thinking=prompt_opened_thinking,
            )
            batch_state = {"response_text": graded}
            if state is not None:
                batch_state.update(state)
            prepared.append((solution_str, ex, batch_state))

        scores_breakdown_many = getattr(env, "scores_breakdown_many", None)
        reward_many = getattr(env, "reward_many", None)
        items = [(ex, state) for _, ex, state in prepared]
        if callable(scores_breakdown_many):
            batch_values = list(scores_breakdown_many(items))
            use_breakdown = True
        elif callable(reward_many):
            batch_values = list(reward_many(items))
            use_breakdown = False
        else:
            return score_serially()
    except Exception as exc:
        # the batch call itself blew up, so NOTHING came back: no env work is credited and a full
        # serial pass is the only way to score this batch.
        print(
            f"[rl-verl] env batch scoring raised ({type(exc).__name__}: {exc}); retrying serially",
            flush=True,
        )
        return score_serially()

    # the batch scorer RETURNED, so its env calls already happened -- and for a paid or
    # side-effecting grader they are already billed. re-running every item (the old behaviour on any
    # malformed payload) charged the whole batch twice to recover a single bad row. keep every
    # well-formed row and repair only the rest serially.
    results: list[tuple[float, list[dict[str, float] | None]] | None] = [None] * len(prepared)
    repair: list[int] = []
    # a wrong-length payload cannot be trusted to be positionally aligned even where it does have
    # entries, so it condemns every row rather than just the missing tail.
    aligned = len(batch_values) == len(prepared)
    for idx, (solution_str, _, _) in enumerate(prepared):
        if not aligned:
            repair.append(idx)
            continue
        try:
            value = batch_values[idx]
            if use_breakdown:
                score, breakdowns = float(value.get("total", 0.0)), [value]
            else:
                score, breakdowns = float(value), []
        except Exception:
            # unusable row: a non-mapping breakdown, a missing/unparseable `total`, or a reward that
            # will not survive float(). recover this one item, not the batch.
            repair.append(idx)
            continue
        results[idx] = (
            _finalize_single_turn_reward(
                score,
                solution_str,
                tok=tok,
                thinking=thinking,
                prompt_opened_thinking=prompt_opened_thinking,
                think_penalty=think_penalty,
            ),
            breakdowns,
        )

    if repair:
        detail = (
            f"returned {len(repair)} unusable row(s)"
            if aligned
            else f"returned {len(batch_values)} rows for {len(prepared)} items"
        )
        print(
            f"[rl-verl] env batch scoring {detail}; re-scoring {len(repair)} of {len(prepared)} "
            f"serially",
            flush=True,
        )
        for idx in repair:
            solution_str, ex, _ = prepared[idx]
            results[idx] = score_one_serially(solution_str, ex)

    # every index is filled by one path or the other. return one row per request, in request order:
    # the caller zips these back onto its completions, so dropping a row would silently misalign the
    # whole batch rather than fail.
    return [row if row is not None else (0.0, []) for row in results]


# the total startup delay this hook is allowed to add, covering reference extraction AND timing.
# both call user code, so one shared ceiling is the only number that means anything to a caller.
_PROFILE_BUDGET_S = 30.0


def _log_reward_profile(env, score_one, rollout_examples: list, completions_per_step: int):
    """measure this env's grading latency once before training.

    return and log the profile because latency affects cost and placement. references use each
    example's real completion; blank text measures an early return rather than the grader.

    ``_PROFILE_BUDGET_S`` bounds reference gathering and grading together because both call user code.
    skip envs with ``reward_thread_safe = False``: profiling off-thread could mutate scorer state or
    touch a thread-affine handle. the flag covers ``sft_completion`` here for the same reason.

    advisory only and never fatal.
    """
    try:
        from flash.content.multimodal import assistant_completion_text
        from flash.engine.profiling.reward_profile import call_bounded, profile_reward_latency

        if not getattr(env, "reward_thread_safe", True):
            print(
                "[rl-verl] reward profiling skipped: env declares reward_thread_safe = False, "
                "so grading it early could change the rewards training sees",
                flush=True,
            )
            return None

        # sft_completion reaches user code (FreesoloEnvAdapter delegates to the env's own hook), so
        # it can block on i/o exactly like a grader can. it shares the profiler's deadline instead
        # of running before it: bounding only the timing phase would leave the startup delay this
        # hook adds unbounded, which is the ceiling the docstring promises.
        deadline = time.perf_counter() + _PROFILE_BUDGET_S
        samples: list[tuple[int, str]] = []
        for index, example in enumerate(rollout_examples[:4]):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            # the env's own assistant-text semantics: takes the last assistant turn and
            # flattens openai-style text blocks. hand-rolling this stringifies block dicts
            # into a python repr and mixes in user/system turns.
            ok, _, messages = call_bounded(
                lambda ex=example: assistant_completion_text(env.sft_completion(ex)), remaining
            )
            if ok is None:
                # report usable references gathered before the shared deadline expired. failed calls
                # keep index placeholders and blank results are dropped, so counting either would claim
                # samples that cannot be profiled and misdiagnose the env hook.
                usable = sum(1 for _, text in samples if text and text.strip())
                print(
                    "[rl-verl] reward profiling skipped: env.sft_completion did not return within "
                    f"{_PROFILE_BUDGET_S:.0f}s ({usable} usable reference completion(s) gathered "
                    "before the deadline)",
                    flush=True,
                )
                return None
            samples.append((index, messages if ok and isinstance(messages, str) else ""))
        if not samples:
            return None
        profile = profile_reward_latency(
            score_one, samples, budget_s=max(0.0, deadline - time.perf_counter())
        )
        print(f"[rl-verl] {profile.describe()}", flush=True)
        if profile.trustworthy and completions_per_step > 0:
            print(
                f"[rl-verl] scalar reward profile sampled "
                f"{profile.seconds_per_completion:.3f}s per completion; runtime batching may "
                f"overlap up to {completions_per_step} completions per step",
                flush=True,
            )
        return profile
    except Exception as exc:
        print(f"[rl-verl] reward profiling skipped: {exc}", flush=True)
        return None


# --------------------------------------------------------------------------------------------
# reward rpc bridge: verl subprocess -> flash live env.
# --------------------------------------------------------------------------------------------
class _ScoreBatcher(ScoreBatcher):
    """coalesce concurrent verl requests into one ordered env scoring call.

    the env still receives one top-level call at a time; concurrency lives inside its own batched
    scorer. errors propagate as raised, since classifying an env scoring failure belongs to the
    caller and the reward routes already translate it for the child.

    ``recheck_closed_after_wait`` stays off: an env batch assembled during the flush window costs
    nothing external to finish, so scoring it lets the last requests of a run answer normally
    instead of failing on a shutdown that arrived while they were queued.
    """

    def __init__(
        self,
        score_batch,
        *,
        max_batch_size: int,
        flush_wait_s: float,
        label: str,
        thread_name: str,
    ) -> None:
        super().__init__(
            score_batch,
            max_batch_size=max_batch_size,
            flush_wait_s=flush_wait_s,
            label=f"{label} score batcher",
            thread_name=thread_name,
        )

    def score(self, request):
        return self.submit(request)


class MultiTurnBridge:
    """hold parent-side env state for the child's multi-turn loop.

    the child owns tokens and generation; this bridge owns the flash env and treats its rollout state
    as authoritative for turn limits, doneness, transcript, and scoring. the callback receives each
    scored episode. multi-turn scoring is scalar, so no component breakdown exists.
    """

    def __init__(
        self,
        env,
        examples: list[dict],
        *,
        env_prompts: list[list[dict]],
        max_turns: int,
        per_turn_credit: bool = False,
        on_episode_scored: Callable[[object, object, float], None] | None = None,
        score_batch_size: int = _MULTI_TURN_SCORE_BATCH_SIZE,
        score_flush_wait_s: float = _MULTI_TURN_SCORE_FLUSH_WAIT_S,
        session_lease_s: float = _MULTI_TURN_SESSION_LEASE_S,
    ) -> None:
        if len(env_prompts) != len(examples):
            raise ValueError("multi-turn env prompts must align one-to-one with examples")
        self._env = env
        self._examples = examples
        self._env_prompts = env_prompts
        self._max_turns = int(max_turns)
        self._per_turn_credit = bool(per_turn_credit)
        self._on_episode_scored = on_episode_scored
        # the flash env is not required to be thread-safe, and verl runs many rollouts at once.
        # every stateful episode touch below happens under this lock.
        self._lock = threading.Lock()
        self._sessions: dict[str, dict] = {}
        self._session_lease_s = float(session_lease_s)
        # score many episodes under one lock-held env call so judge work can use the env's own
        # max_score_concurrency. keep the lock: reward_thread_safe permits scorer/scorer concurrency,
        # not racing scoring against env_reply.
        self._scorer = _ScoreBatcher(
            self._score_batch,
            max_batch_size=int(score_batch_size),
            flush_wait_s=float(score_flush_wait_s),
            label="multi-turn episode",
            thread_name="flash-grpo-episode-scorer",
        )

    def routes(self) -> dict:
        return {
            "/multiturn/start": self.start,
            "/multiturn/step": self.step,
            "/multiturn/score": self.score,
            "/multiturn/close": self.close,
        }

    def shutdown(self) -> None:
        """stop the scoring thread. distinct from the ``/multiturn/close`` route, which ends one
        episode. any episode still waiting is failed rather than left blocked on its event."""
        self._scorer.close(_MULTI_TURN_SCORE_SHUTDOWN_WAIT_S)

    def _session(self, payload: dict) -> dict:
        session_id = str(payload["session_id"])
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"unknown multi-turn session {session_id}")
        # every touch renews the lease, so only a session nobody is driving can ever age out. all
        # callers already hold the lock.
        session["touched_at"] = time.monotonic()
        return session

    def _reap_abandoned_sessions(self) -> list[str]:
        """Drop sessions whose owner stopped driving them. Caller holds the lock.

        `monotonic` rather than wall time: a clock step must not expire a live episode, nor keep a
        dead one alive.
        """
        if self._session_lease_s <= 0:
            return []
        cutoff = time.monotonic() - self._session_lease_s
        stale = [sid for sid, s in self._sessions.items() if s.get("touched_at", 0.0) <= cutoff]
        for session_id in stale:
            self._sessions.pop(session_id, None)
        return stale

    def start(self, payload: dict) -> dict:
        index = int(payload["index"])
        if index < 0 or index >= len(self._examples):
            raise IndexError(
                f"multi-turn example index {index} is outside [0, {len(self._examples)})"
            )
        session_id = str(payload["session_id"])
        example = self._examples[index]
        with self._lock:
            # swept here rather than on a timer thread: a session is only ever abandoned by an
            # actor that stopped calling, and the actors that replace it announce themselves
            # exactly here. no extra thread, and nothing to shut down.
            reaped = self._reap_abandoned_sessions()
            if session_id in self._sessions:
                raise KeyError(f"duplicate multi-turn session {session_id}")
            state = self._env.new_rollout_state(example)
            # new_rollout_state calls start_episode again after dataset preparation. adopt the dataset's
            # prompt so randomized envs do not generate for episode a and score episode b; keep the
            # remaining state created by the env.
            env_prompt = [dict(message) for message in self._env_prompts[index]]
            state["prompt"] = env_prompt
            state["messages"] = [dict(message) for message in env_prompt]
            self._sessions[session_id] = {
                "example": example,
                "state": state,
                "touched_at": time.monotonic(),
            }
        if reaped:
            # loud, because it means rollout actors are dying mid-episode. the leak it prevents is
            # silent, so the reap must not be.
            print(
                f"[rl-verl] reaped {len(reaped)} abandoned multi-turn session(s) after "
                f"{self._session_lease_s:.0f}s idle: {', '.join(sorted(reaped))}",
                flush=True,
            )
        # the per-example budget wins over the batch-wide cap, same precedence as rollout_done.
        episode_turns = state.get("max_episode_turns")
        turns = self._max_turns if episode_turns is None else int(episode_turns)
        return {"max_turns": max(1, min(self._max_turns, turns))}

    def step(self, payload: dict) -> dict:
        with self._lock:
            session = self._session(payload)
            state = session["state"]
            # an unusable turn is terminal and must NOT be shown to the env: recording it would
            # append a truncated or empty assistant message to the transcript that gets scored.
            # the child stops on the same condition, so this only decides what the env sees.
            if bool(payload.get("truncated")) or str(payload.get("skip_reason") or ""):
                # it is still the turn the model generated and the child trained on, so it is kept
                # for the DIAGNOSTIC transcript. dropping it entirely would publish an empty
                # completion for a first-turn truncation -- the one sample worth reading, since it
                # is the failure being diagnosed.
                session["aborted_turn"] = {
                    "role": "assistant",
                    "content": str(payload.get("completion_text") or ""),
                }
                return {"terminal": True, "messages": []}
            self._env.record_model_turn(state, str(payload.get("completion_text") or ""))
            if self._env.rollout_done(state, self._max_turns):
                return {"terminal": True, "messages": []}
            replies = self._env.env_reply(list(state.get("messages") or ()), state)
            terminal = bool(self._env.rollout_done(state, self._max_turns))
        return {
            "terminal": terminal,
            "messages": [
                {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
                for message in replies
            ],
        }

    def _score_batch(self, requests: list) -> list:
        """score a whole batch of terminal episodes in ONE env call. runs on the batcher thread.

        takes the same lock every other env touch takes, so scoring never overlaps a concurrent
        episode's ``env_reply``. the win is that one lock acquisition now covers a whole batch.
        """
        with self._lock:
            return score_rollouts(self._env, requests)

    def score(self, payload: dict) -> dict:
        turn_count = int(payload["turn_count"])
        with self._lock:
            session = self._session(payload)
            state = session["state"]
        # queued OUTSIDE the lock so concurrent episodes can coalesce into one env call; the
        # batcher thread reacquires it to do the scoring. safe to read this session's state
        # unlocked because the episode is terminal -- the child sends /score only after its turn
        # loop has ended, so nothing else mutates this session.
        reward = self._scorer.score(
            RolloutScoreRequest(
                example=session["example"],
                state=state,
                turn_count=turn_count,
            )
        )
        with self._lock:
            # snapshot under the same lock that guards the session: `step` mutates this list in
            # place, and a concurrent episode's turn would otherwise be read mid-append.
            prompt = list(state.get("prompt") or ())
            # the episode transcript, NOT the whole message list: new_rollout_state seeds `messages`
            # with a copy of `prompt` and appends turns onto it, so publishing it whole would repeat
            # the prompt inside `completion` when it already rides the sample as `prompt_tail`.
            # slice by length rather than by equality -- an env may legitimately produce a turn that
            # matches a prompt message, and dropping it would silently truncate the episode.
            transcript = list(state.get("messages") or ())[len(prompt) :]
            # the aborted turn never entered `messages` -- the env must not score it -- but the
            # child trained on its tokens, so the sample shows what was actually generated.
            aborted = session.get("aborted_turn")
            if aborted is not None:
                transcript.append(aborted)
        # nan is score_rollouts' unscorable marker; verl has no equivalent, and a nan would
        # propagate into the group baseline and poison every other rollout in the group, so an
        # unscorable episode scores 0.0 the way a failed single-turn grading does.
        episode = float(reward.episode)
        if not math.isfinite(episode):
            print("[rl-verl] multi-turn episode unscorable; scoring 0.0", flush=True)
            episode = 0.0
        if self._on_episode_scored is not None:
            # outside the lock: the callback is the caller's buffer, which has its own lock, and
            # nesting them in this order would invert the single-turn path's acquisition order.
            self._on_episode_scored(prompt, transcript, episode)
        if not self._per_turn_credit:
            return {"score": episode}
        # score_rollouts already validated the vector against turn_count and canonicalised an
        # unusable one to None (multiturn_reward_scoring), so nothing further is checked here.
        # None means this episode falls back to episode credit; the shim widens that fallback to
        # the whole group so a group is never centred on a mix of the two.
        turns = None if reward.turns is None else [float(value) for value in reward.turns]
        return {"score": episode, "turns": turns}

    def close(self, payload: dict) -> dict:
        with self._lock:
            self._sessions.pop(str(payload["session_id"]), None)
        return {"closed": True}

    def open_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)


# the three stdlib-only modules the child needs beside its shim, mapped to the flat names it
# imports them under. flat and `flash_`-prefixed because the child imports them as top-level
# modules, not as a package: flash itself is NOT importable in the verl interpreter (incompatible
# torch/vllm pins), which is why they are copied rather than imported. same mechanism the opd verl
# path uses for its own loop.
# (path relative to flash/engine/worker/, flat name the child imports it under). the child code
# lives under train/, so these are package-relative paths rather than bare siblings.
MULTI_TURN_CHILD_MODULES = (
    (os.path.join("train", "core", "child", "glue.py"), "flash_multiturn_glue.py"),
    (os.path.join("train", "rl", "child", "multiturn.py"), "flash_grpo_multiturn.py"),
    (os.path.join("train", "rl", "child", "plugin.py"), "flash_grpo_plugin.py"),
)


def copy_multi_turn_child_modules(shim_dir: str) -> tuple[str, ...]:
    """copy the child-side agent loop next to the shim; returns the paths written."""
    written = []
    for source_name, child_name in MULTI_TURN_CHILD_MODULES:
        target = os.path.join(shim_dir, child_name)
        shutil.copy2(os.path.join(os.path.dirname(__file__), source_name), target)
        written.append(target)
    return tuple(written)


def multi_turn_child_env(inp: dict, *, reward_url: str, thinking: bool) -> dict[str, str]:
    """return only the child env vars consumed by the multi-turn loop.

    the verl interpreter cannot import flash, so the parent resolves generation-config eos values,
    turn limits, halting, and glue rendering into strings.
    """
    return {
        # verl imports this at the end of `verl/__init__` (import_external_libs), which is what
        # registers the loop under the name the agent-loop override selects. without it the
        # override names a loop that was never registered and the child dies at rollout build.
        "VERL_USE_EXTERNAL_MODULES": "flash_grpo_plugin",
        "FLASH_VERL_MULTITURN_URL": reward_url,
        "FLASH_VERL_MAX_TURNS": str(int(inp["max_turns"])),
        "FLASH_VERL_MAX_MODEL_LEN": str(int(inp["engine_len"])),
        # the PER-TURN cap. distinct from the two episode-wide budgets the child already derives:
        # `[train].max_completion_tokens` bounds ONE assistant turn, while max_model_len and the
        # response tensor bound the whole transcript. without it the first turn may consume the
        # entire episode budget, which is both a cost and a behaviour change from the retired
        # driver's `_turn_budget` (per_turn_max_tokens=max_completion).
        "FLASH_VERL_MAX_COMPLETION_TOKENS": str(int(inp["max_completion"])),
        "FLASH_VERL_STOP_SEQUENCES": json.dumps(list(inp["stop_sequences"])),
        # sorted so the child sees a stable set regardless of frozenset iteration order.
        "FLASH_VERL_EOS_TOKEN_IDS": json.dumps(sorted(inp["eos_token_ids"])),
        "FLASH_VERL_THINKING": "1" if thinking else "0",
    }


def start_reward_server(
    score_by_index,
    *,
    example_count: int,
    multi_turn_bridge=None,
    rollout_batch: int = 0,
    score_batch=None,
):
    """start the localhost reward server and return ``(server, base_url)``.

    call ``server.shutdown()`` when done. multi-turn adds four episode routes beside ``/score``;
    ``rollout_batch`` sizes the accept queue for the simultaneous episode burst.
    """
    # the scalar compatibility path serializes top-level env calls. training supplies score_batch,
    # which keeps that same one-call-at-a-time boundary while coalescing concurrent verl requests into
    # the env's own batched scorer.
    score_lock = threading.Lock()
    score_batcher = (
        _ScoreBatcher(
            score_batch,
            max_batch_size=_SINGLE_TURN_SCORE_BATCH_SIZE,
            flush_wait_s=_SINGLE_TURN_SCORE_FLUSH_WAIT_S,
            label="single-turn reward",
            thread_name="flash-grpo-reward-scorer",
        )
        if callable(score_batch)
        else None
    )

    def _score_route(payload: dict) -> dict:
        index = int(payload["index"])
        if index < 0 or index >= example_count:
            raise IndexError(f"reward example index {index} is outside [0, {example_count})")
        solution_str = payload.get("solution_str", "")
        if score_batcher is not None:
            return {"score": float(score_batcher.score((index, solution_str)))}
        with score_lock:
            return {"score": float(score_by_index(index, solution_str))}

    routes = {"/score": _score_route}
    if multi_turn_bridge is not None:
        routes.update(multi_turn_bridge.routes())

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            route = routes.get(self.path)
            if route is None:
                self.send_response(404)
                self.end_headers()
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n).decode("utf-8"))
                result = route(payload)
            except Exception as exc:
                print(f"[rl-verl] reward server request failed: {exc}", flush=True)
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
            else:
                body = json.dumps(result).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class _RewardBridgeHTTPServer(BoundedThreadingHTTPServer):
        request_queue_size = max(_REWARD_BRIDGE_MIN_REQUEST_BACKLOG, int(rollout_batch))

        def shutdown(self):
            if score_batcher is not None:
                score_batcher.close(_SINGLE_TURN_SCORE_SHUTDOWN_WAIT_S)
            super().shutdown()

    server = _RewardBridgeHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    return server, f"http://127.0.0.1:{port}"


# --------------------------------------------------------------------------------------------
# verl interpreter + checkpoint export.
# --------------------------------------------------------------------------------------------
class _VerlResumeUploader:
    """stream completed verl checkpoints to hf for resume and required saves.

    all checkpoints upload resume state. exact ``save_at_steps`` also exports deployable adapters.
    verl advances ``latest_checkpointed_iteration.txt`` only after writing ``global_step_N``, so the
    marker prevents half-written uploads.
    """

    def __init__(
        self,
        local_dir: str,
        *,
        resume_step: int,
        required_steps: tuple[int, ...] = (),
        export_root: str = "",
        python_bin: str = "",
        model_id: str = "",
        model_revision: str = "",
        preprocessor=None,
        had_gradient: Callable[[], bool] | None = None,
    ) -> None:
        self.local_dir = local_dir
        self.required_steps = frozenset(required_steps)
        # seed uploaded resume steps because hf already has them, but do not seed processed required
        # steps: a prior worker may have uploaded resume state while the gradient gate withheld its
        # deployable adapter, which this worker must still stage.
        self.processed_steps: set[int] = (
            {resume_step} if resume_step and resume_step not in self.required_steps else set()
        )
        # tracked separately from processed_steps: a required resume step is left unprocessed so it
        # can be staged, but its resume state is already on hf and must not be re-uploaded.
        self.uploaded_steps: set[int] = {resume_step} if resume_step else set()
        # required step -> staged adapter directory, populated the sweep a checkpoint appears and
        # drained by publication once the gradient gate opens. this is what decouples publication
        # from verl's checkpoint retention.
        self.staged_steps: dict[int, str] = {}
        self.export_root = export_root
        self.python_bin = python_bin
        self.model_id = model_id
        self.model_revision = model_revision
        # the processor on a multimodal job, else the tokenizer: an image model cannot be served
        # from an adapter dir that carries only tokenizer files, since the runtime needs the
        # preprocessor config to turn pixels back into tokens.
        self.preprocessor = preprocessor
        # gate deployable publication on real gradient evidence so zero-gradient runs cannot leave
        # servable per-step adapters before the final guard fails. resume uploads remain allowed because
        # they are internal retry state, not servable artifacts.
        self.had_gradient = had_gradient
        # populated by credit_durable_required_steps() before start(): crediting a required step
        # needs an hf lookup, which does not belong in a constructor.
        self.published_steps: set[int] = set()
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def credit_durable_required_steps(self, resume_step: int) -> None:
        """credit required saves already published before resume.

        verify the adapter on hf rather than trusting the step counter; a preempted worker can pass a
        required step without publishing its deployable artifact.
        """
        for step in sorted(self.required_steps):
            if step <= int(resume_step) and _deployable_adapter_on_hf(step):
                self.published_steps.add(step)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        # idempotent: the clean path stops the drain early to surface a missing required save, and
        # the finally block stops it again on every path.
        self._stop.set()
        # bounded by lack of progress, not wall clock: the uploads run on this thread, so a big
        # model's full-state drain can legitimately outlast any constant deadline (VERL-131).
        join_while_draining(self._thread, "verl resume uploader")

    def raise_if_incomplete(self) -> None:
        """fail the run when a required save never became durable.

        called only after training finished cleanly: on a crash or cancel the missing steps are a
        symptom of the real failure, and raising here would mask it.
        """
        if self._error is not None:
            raise RuntimeError("verl resume uploader failed") from self._error
        missing = sorted(self.required_steps - self.published_steps)
        if missing:
            raise RuntimeError(f"required saves were not durably published: {missing}")

    def _deployable_allowed(self) -> bool:
        """return whether gradient evidence permits deployable publication.

        no callback means no gate. callback errors fail closed because this controls durable serving.
        """
        if self.had_gradient is None:
            return True
        try:
            return bool(self.had_gradient())
        except Exception:
            return False

    def _completed_step(self) -> int:
        return completed_checkpoint_step(self.local_dir)

    def _pending(self) -> list[tuple[int, str]]:
        """the completed checkpoint dirs this uploader has not handled yet, oldest first."""
        return unprocessed_checkpoint_dirs(
            self.local_dir, self._completed_step(), self.processed_steps
        )

    def _stage_deployable(self, step: int, checkpoint_dir: str) -> str:
        """stage a required checkpoint as a servable adapter under ``export_root``.

        stage before the gradient gate because verl may prune ``checkpoint_dir`` while the gate is
        closed. ``export_root`` outlives verl retention; only later publication makes it servable.
        """
        actor_dir = os.path.join(checkpoint_dir, "actor")
        adapter_dir = os.path.join(self.export_root, f"step-{step}")
        shutil.rmtree(adapter_dir, ignore_errors=True)
        os.makedirs(adapter_dir, exist_ok=True)
        export_peft_adapter(
            actor_dir, adapter_dir, base_model_id=self.model_id, python_bin=self.python_bin
        )
        # the served adapter needs its preprocessor alongside it, exactly as the final publish does:
        # the processor on a multimodal job (tokenizer + image preprocessor), else the tokenizer.
        self.preprocessor.save_pretrained(adapter_dir)
        stamp_adapter_dir_provenance(adapter_dir, self.model_id, self.model_revision)
        _w.write_base_model_provenance(adapter_dir, self.model_id, self.model_revision)
        return adapter_dir

    def _publish_staged(self, step: int, adapter_dir: str) -> None:
        """make an already-staged adapter durable and servable."""
        _w.publish_deployable_checkpoint(adapter_dir, step, required=True, _provenance_ready=True)
        self.published_steps.add(step)

    def _publish_ready(self) -> None:
        """publish permitted staged steps oldest first.

        staged exports remain publishable after verl prunes their checkpoints. read the gate once so
        evidence appearing during export applies on the same pass.
        """
        if not self._deployable_allowed():
            return
        for step in sorted(self.staged_steps):
            if step not in self.published_steps:
                self._publish_staged(step, self.staged_steps[step])

    def _run(self) -> None:
        # a failed resume upload must not fail the run: the policy is still trained and published at
        # the end, and the only loss is having to restart from an earlier step after a preemption.
        # a failed REQUIRED save is different -- the customer asked for a deployable at that step, so
        # it propagates and raise_if_incomplete() turns it into a run failure.
        try:
            while True:
                # sampled before the sweep so a stop() arriving mid-sweep still gets one full pass
                # over the checkpoints it made visible. verl advances
                # latest_checkpointed_iteration.txt right up to the moment the child exits, so the
                # newest resume checkpoint routinely appears in that window; exiting without
                # sweeping it would drop durable work a preemption then has to redo.
                stopping = self._stop.is_set()
                for step, path in self._pending():
                    # stage while verl's checkpoint still exists; publication happens below, once
                    # the gate opens. the two have different deadlines -- the source expires with
                    # verl's retention, the permission arrives with the first varying-reward group --
                    # so a step is staged on the sweep that finds it and published whenever it may be.
                    if (
                        step in self.required_steps
                        and step not in self.published_steps
                        and step not in self.staged_steps
                    ):
                        self.staged_steps[step] = self._stage_deployable(step, path)
                        # published here, not once the sweep ends: staging a later step can raise,
                        # and the resume upload below can be interrupted by a preemption. either
                        # would leave an already-exported adapter local-only, so each step is made
                        # durable as soon as it is both staged and permitted.
                        self._publish_ready()
                    # resume uploads are ungated internal retry state and may be the only checkpoints before
                    # gradient evidence appears. mark attempts before upload so a permanent failure cannot
                    # spin every 0.5s, preserving the existing non-fatal upload semantics.
                    if step not in self.uploaded_steps:
                        self.uploaded_steps.add(step)
                        try:
                            _w.upload_resume_checkpoint(step, path)
                        except Exception as error:
                            print(
                                f"[rl-verl] resume checkpoint upload failed at step {step}: {error}",
                                flush=True,
                            )
                    self.processed_steps.add(step)
                # a step staged while the gate was shut is published by a later sweep, so this runs
                # every pass and not only when something was staged: the gate opens on the first
                # varying-reward group, which is usually a sweep that finds no new checkpoint.
                self._publish_ready()
                # `stopping`, not a fresh read: the sweep above is the full pass over everything
                # stop() had made visible, and only after running it may the loop exit. staged
                # steps still awaiting the gate cannot hold the loop open -- with the gate shut they
                # never clear, and waiting would hang stop(); raise_if_incomplete() reports them.
                if stopping and not self._pending():
                    return
                time.sleep(0.5)
        except BaseException as error:
            self._error = error


def _restore_verl_resume(local_dir: str) -> int:
    """stage this run's streamed resume checkpoint into local_dir; return the step it resumes at.

    returns 0 when there is nothing to resume, which is the ordinary fresh-run path.
    """
    resume = _w.hf_resume_checkpoint()
    if not resume:
        return 0
    return stage_verl_resume(resume, local_dir, job_label="GRPO")


def _check_grpo_had_a_gradient(
    reward_history: list[float],
    adv_spread_history: list[float],
    *,
    resumed: bool = False,
    already_complete: bool = False,
) -> None:
    """raise unless grpo rewards produced a nonzero policy gradient.

    grpo mean-centres rewards within each group, so equal rewards yield zero advantages and a zero
    gradient despite valid checkpoints and reward means. advantage spread is the decisive signal;
    pg_loss can also be near zero on a valid first step.

    abstain for resumed or already-complete runs because this worker did not observe the earlier
    productive steps. the guard still catches degenerate envs on their first, unresumed attempt.
    """
    if already_complete:
        # no step ran, so there is no metric stream to judge and nothing the parse could have
        # dropped. the restored policy's evidence lives in the worker that produced it.
        return
    if not reward_history:
        raise RuntimeError(
            "verl reported no reward metrics for the whole run — the flash reward bridge was "
            "never consulted (wiring regression); refusing to publish a policy trained on "
            "default rewards"
        )
    if not adv_spread_history:
        # advantages ride the same log line as critic/rewards/mean (both come from one
        # compute_data_metrics dict), so reward metrics without advantage metrics means the parse
        # broke, not that verl chose not to report. treat that as a regression rather than let the
        # spread check below degrade into a guard that cannot fire.
        raise RuntimeError(
            "verl reported reward metrics but no advantage metrics for any step — "
            "critic/advantages/max and /min could not be parsed (metric-format regression); "
            "refusing to publish because the zero-gradient check cannot run"
        )
    if resumed:
        # see the docstring: this worker's history is a suffix of the run's training, so an all-zero
        # suffix is not evidence the run never had a gradient. the parse checks above still apply --
        # they catch a wiring/format regression regardless of where training started.
        return
    # deliberately "no step ever had spread" rather than "some step had none": a run that genuinely
    # converges, or that draws one unlucky all-equal group, legitimately reports zero spread on
    # individual steps, and rejecting those would fail correct runs.
    if not any(spread > 0.0 for spread in adv_spread_history):
        raise RuntimeError(
            f"grpo saw zero advantage spread on all {len(adv_spread_history)} steps — every group's "
            "rewards were identical, so every advantage was 0 and the gradient was exactly 0; the "
            "environment's reward does not discriminate between completions. refusing to publish an "
            "adapter identical to its initialization"
        )


# --------------------------------------------------------------------------------------------
# orchestration.
# --------------------------------------------------------------------------------------------
def _resolve_grpo_inputs():
    """reproduce run_rl's front-half config + dataset prep for text, multimodal, and multi-turn."""
    env = _w.require_active_env()
    if getattr(env, "is_tool_env", False):
        raise RuntimeError(
            "verl grpo does not support openai function-calling tool environments; this env "
            "reports is_tool_env"
        )
    multi_turn = bool(getattr(env, "multi_turn", False))
    if multi_turn:
        # the bridge drives the env through exactly these four calls, so a missing one would only
        # surface mid-rollout on the first episode. same gate the opd verl path applies.
        missing = [
            name
            for name in ("new_rollout_state", "record_model_turn", "env_reply", "rollout_done")
            if not callable(getattr(env, name, None))
        ]
        if missing:
            raise RuntimeError(
                f"multi-turn grpo environment is missing required rollout methods: {missing}"
            )
        if int(getattr(env, "max_turns", 0) or 0) <= 0:
            raise RuntimeError("multi-turn grpo environment requires a positive bounded turn limit")
    seed_training_rngs(_w.SEED)
    model_id = _w.JOB_SPEC.model if _w.JOB_SPEC else RECIPE.hf_model_id
    model_revision = getattr(_w.JOB_SPEC, "model_revision", "") if _w.JOB_SPEC else ""

    rl = RECIPE.rl
    _t = _w.JOB_SPEC.train if _w.JOB_SPEC else None
    # fail loudly on unsupported verl features. structured-output constraints travel in per-sample
    # params via render_structured_outputs_shim; the reasoning parser is a hydra override. malformed
    # payloads raise instead of training unconstrained.
    structured_outputs = (
        parse_structured_outputs(getattr(_t, "structured_outputs", "") if _t else "") or None
    )
    if structured_outputs:
        print(
            f"[rl-verl] structured outputs: every rollout constrained to "
            f"{describe_structured_outputs(structured_outputs)}"
        )
    # stop_sequences: on the retired trl backend these ride generation_kwargs["stop"] into vllm's
    # SamplingParams. verl builds its sampling params without a stop field, so a sitecustomize shim
    # inserts the key; see render_stop_sequences_shim. note grpo_mask_truncated_completions below
    # gates itself OFF when stop_sequences is set, on either backend.
    stop_sequences = tuple(str(s) for s in (getattr(_t, "stop_sequences", ()) or ())) if _t else ()
    # entropy_quantile keeps the loss to the top-entropy tokens of each completion. verl has no
    # such knob, so a sitecustomize shim adds it; see render_entropy_quantile_shim. a value >= 1.0
    # (or unset) masks nothing and needs no shim.
    _eq = getattr(_t, "entropy_quantile", None) if _t else None
    entropy_quantile = float(_eq) if _eq is not None and float(_eq) < 1.0 else None
    # per-turn credit is equivalent to per-episode credit for a single-turn env, so accept it there.
    # on multi-turn, centre each turn against sibling turns and rewrite the stock advantage tensor;
    # the agent loop records token spans and the bridge returns per-turn rewards. see
    # render_per_turn_credit_shim.
    credit_assignment = getattr(_t, "credit_assignment", DEFAULT_CREDIT_ASSIGNMENT) if _t else None
    per_turn_credit = bool(
        credit_assignment and credit_assignment != DEFAULT_CREDIT_ASSIGNMENT and multi_turn
    )
    if credit_assignment and credit_assignment != DEFAULT_CREDIT_ASSIGNMENT and not multi_turn:
        print(
            f"[rl-verl] credit assignment: {credit_assignment} is equivalent to "
            f"{DEFAULT_CREDIT_ASSIGNMENT} for single-turn environments",
            flush=True,
        )
    # flash's lora dropout is fixed at 0.0; verl matches (peft default). guard defensively so a future
    # non-zero recipe value can never be silently ignored.
    if float(RECIPE.lora.dropout) != 0.0:
        raise RuntimeError(
            f"RECIPE.lora.dropout={RECIPE.lora.dropout} is not wired on the verl backend; "
            "add actor_rollout_ref.model.lora_dropout support before setting it non-zero."
        )
    gcfg = _w.grpo_overrides()
    prompts_per_step = int(
        _t.batch_size if _t and _t.batch_size is not None else rl.prompts_per_step
    )
    group_size = int(gcfg.get("group_size") or rl.group_size)
    _gcfg_temp = gcfg.get("temperature")
    temperature = float(_gcfg_temp if _gcfg_temp is not None else rl.sampling_temperature)
    think_penalty = float(gcfg.get("thinking_length_penalty_coef") or 0.0)
    # flash defaults kl_penalty_coef to 0 (dr-grpo, no kl term); honor it rather than forcing a kl.
    kl_coef = float(gcfg.get("kl_penalty_coef") or 0.0)
    mask_truncated_completions = _w.grpo_mask_truncated_completions(_t)
    learning_rate = float(
        _t.learning_rate if _t and _t.learning_rate is not None else rl.learning_rate
    )
    # warm-start forbids lora_rank, so a set init_from_adapter already raised above; read rank/alpha
    # from the job spec (falling back to the recipe) exactly like the retired trl path's lora config.
    lora_rank = int(_t.lora_rank) if (_t and _t.lora_rank) else int(RECIPE.lora.rank)
    lora_alpha = int(_t.lora_alpha) if (_t and _t.lora_alpha) else int(RECIPE.lora.alpha)
    # warm-start: continue the sft adapter in place (verl lora_adapter_path). uses the SOURCE
    # adapter's rank/alpha (flash forbids a child lora_rank on warm-start).
    warmstart_adapter = ""
    if _t and getattr(_t, "init_from_adapter", ""):
        from flash.engine.worker.model.adapter import _download_adapter

        # a multi-GB adapter pull emits nothing of its own, so it must run under a liveness wrap or
        # the provider judges the silence as a stall.
        with liveness_heartbeat("rl_adapter_loading"):
            warmstart_adapter = _download_adapter(_t.init_from_adapter)
        if not warmstart_adapter:
            raise RuntimeError(
                "warm-start source adapter could not be downloaded; refusing to start from the base."
            )
        with open(os.path.join(warmstart_adapter, "adapter_config.json")) as f:
            _src_cfg = json.load(f)
        _w.validate_lora_target_parameters(_src_cfg, model_id)
        # a patterned adapter trains some modules at higher rank than the base `r`; verl allocates
        # one uniform rank, so it must cover the MAXIMUM prepared rank or the load truncates.
        _ranks = [int(_src_cfg.get("r", lora_rank))]
        _ranks += [int(v) for v in (_src_cfg.get("rank_pattern") or {}).values()]
        lora_rank = max(_ranks)
        _alphas = [int(_src_cfg.get("lora_alpha", lora_alpha))]
        _alphas += [int(v) for v in (_src_cfg.get("alpha_pattern") or {}).values()]
        lora_alpha = max(_alphas)
        print(
            f"[rl-verl] warm-start: continuing source adapter (r={lora_rank}, alpha={lora_alpha}) "
            f"from {_t.init_from_adapter}",
            flush=True,
        )

    train = env.dataset()
    _max_examples = getattr(_t, "max_examples", None) if _t else None
    if _max_examples:
        train = train[: int(_max_examples)]
    rng = random.Random(_w.SEED)
    rng.shuffle(train)
    message_prompts = [env.prompt_messages(ex) for ex in train]

    from flash.content.multimodal import (
        normalize_prompt_images,
        record_has_images,
        resolve_image_pad_token_id,
        validate_multimodal_training,
    )

    multimodal = any(
        record_has_images(ex, messages) for ex, messages in zip(train, message_prompts, strict=True)
    )
    package_root = getattr(env, "package_root", None)
    processor = None
    image_pad_token_id = None
    if multimodal:
        # the model must actually support image training; this raises for a text-only checkpoint
        # rather than letting the processor silently drop the pixels.
        validate_multimodal_training(model_id, "grpo")
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, **_w.model_revision_kwargs(model_revision)
        )
        tok = processor.tokenizer
        image_pad_token_id = resolve_image_pad_token_id(processor, tok)
    else:
        tok = _w.load_tokenizer(model_id, revision=model_revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    max_completion = int(
        gcfg.get("max_tokens")
        or (rl.max_completion_len_thinking if _w.THINKING else rl.max_completion_len)
    )
    train_ctx = _t.max_context_tokens if (_t and _t.max_context_tokens) else 0
    requested_len = int(train_ctx or max(1024, rl.max_prompt_len + max_completion))
    # clamp to the architecture BEFORE deriving the prompt budget, so every downstream length agrees.
    # clamping only the engine would admit prompts up to the unclamped budget and then fail them at
    # rollout, and would let the token budget pack more than the one-sequence memory floor intends.
    vllm_max_len = clamp_engine_len(
        requested_len, model_max_position_embeddings(model_id, model_revision)
    )
    if vllm_max_len < requested_len:
        print(
            f"[rl-verl] max_context_tokens {requested_len} exceeds the {model_id} context limit; "
            f"training at {vllm_max_len}",
            flush=True,
        )
    prompt_budget = vllm_max_len - max_completion
    if prompt_budget <= 0:
        raise ValueError(
            "engine length leaves no room for the completion; raise max_context_tokens"
        )
    if multi_turn:
        # the child derives inter-turn glue by probing this template. a template that cannot
        # round-trip assistant content would fail on the first environment reply, after the rollout
        # has already burned gpu time -- fail here instead, before anything is launched.
        validate_glue_template(tok, thinking=bool(_w.THINKING))

    prompts = []
    if multimodal:
        # verl raises rather than truncating over-budget multimodal prompts because truncation corrupts
        # feature alignment, so pre-filter them. every mixed-job row uses the same processor as the
        # child; tokenizer-only counts miss image expansion by hundreds of tokens.
        for ex, messages in zip(train, message_prompts, strict=True):
            normalized = normalize_prompt_images(ex, messages, package_root)
            expanded, rendered = _processor_expanded_prompt(
                processor,
                normalized.messages,
                tuple(normalized.descriptors),
                package_root,
                enable_thinking=bool(_w.THINKING),
            )
            if 0 < len(expanded) <= prompt_budget:
                prompts.append(
                    {
                        "prompt": normalized.messages,
                        # the pre-normalization messages, i.e. exactly what this example's
                        # start_episode returned. see the text branch below for why the bridge
                        # needs them.
                        "env_prompt": messages,
                        "images": list(normalized.descriptors),
                        "rendered": rendered,
                        "example": ex,
                        "prompt_len": len(expanded),
                    }
                )
    else:
        for ex, messages in zip(train, message_prompts, strict=True):
            rendered = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=_w.THINKING
            )
            prompt_len = len(tok(rendered, add_special_tokens=False).input_ids)
            if 0 < prompt_len <= prompt_budget:
                prompts.append(
                    {
                        "prompt": messages,
                        # reuse the messages that built this prompt. calling start_episode again can randomize
                        # a different episode, making the bridge score different state from the generated
                        # transcript.
                        "env_prompt": messages,
                        "rendered": rendered,
                        "example": ex,
                        "prompt_len": prompt_len,
                    }
                )
    if len(prompts) < len(train):
        print(
            f"[rl-verl] dropped {len(train) - len(prompts)} prompts over the "
            f"{prompt_budget}-token prompt budget",
            flush=True,
        )
    if not prompts:
        raise ValueError(f"every training prompt exceeds the {prompt_budget}-token prompt budget")

    # single-turn responses are max_completion wide. multi-turn responses contain the whole episode,
    # and verl silently right-truncates beyond data.max_response_length. size to engine budget minus
    # the shortest admitted prompt so no possible episode is cut mid-turn; max_completion remains the
    # per-turn cap.
    max_response_len = (
        max(max_completion, vllm_max_len - min(int(p["prompt_len"]) for p in prompts))
        if multi_turn
        else max_completion
    )

    prompts_per_step = _w.resolve_grpo_prompts_per_step(prompts_per_step, len(prompts))
    prompt_opened_thinking = bool(_w.THINKING) and _w.prompt_opens_thinking(prompts[0]["rendered"])
    # hand the same derived flag to the multi-turn grading path. the env cannot derive it itself --
    # it has no tokenizer and never sees a rendered prompt -- and the template opens the block in
    # EVERY assistant generation prompt (the glue tokenizer renders the next turn's header the same
    # way), so one run-level value covers every turn, exactly as it does single-turn.
    if hasattr(env, "prompt_opens_thinking"):
        env.prompt_opens_thinking = prompt_opened_thinking

    # optimizer-update horizon, honoring [train].max_steps exactly.
    epochs = int(_t.epochs) if (_t and _t.epochs is not None) else int(rl.num_epochs)
    derived_steps = on_policy_steps(
        epochs=epochs, prompt_count=len(prompts), prompts_per_step=prompts_per_step
    )
    steps = resolve_update_horizon(derived_steps, getattr(_t, "max_steps", None) if _t else None)
    # verl stops at both total_epochs and total_training_steps. give its hardcoded drop_last
    # dataloader enough epoch capacity to serve the horizon, but preserve the configured epoch count
    # for user-facing metadata. total_training_steps prevents the capacity headroom from over-training.
    verl_total_epochs = _verl_epochs_for_horizon(
        epochs=epochs,
        prompt_count=len(prompts),
        prompts_per_step=prompts_per_step,
        steps=int(steps),
    )
    save_every = int(_t.save_every) if (_t and _t.save_every) else 20
    # exact save steps: verl only saves when global_step % save_freq == 0, so it cannot hit an
    # arbitrary set directly. the gcd is the largest interval every required step is a multiple of,
    # so verl saves a superset and the uploader publishes the deployables at exactly the required
    # ones. same derivation the sft verl backend already uses.
    save_at_steps = tuple(int(step) for step in (getattr(_t, "save_at_steps", ()) or ()))
    validate_save_steps(save_at_steps, int(steps))
    save_freq = reduce(gcd, save_at_steps) if save_at_steps else save_every
    # keep checkpoints long enough for asynchronous export. verl prunes only after the next save, but
    # consecutive required steps can leave a short merger window; a small history prevents a slow
    # export from losing a requested step.
    ckpt_to_keep = 3 if save_at_steps else 1

    return {
        "env": env,
        "tok": tok,
        "processor": processor,
        "multimodal": multimodal,
        "multi_turn": multi_turn,
        "max_turns": int(getattr(env, "max_turns", 0) or 0) if multi_turn else 0,
        # the child decides whether an assistant turn ended naturally or was cut off, and it has no
        # access to the model config -- so the full halting set is resolved here and handed over.
        # the union of the tokenizer's eos and the model's generation_config eos, because a model can
        # stop on a secondary id its tokenizer never exposes.
        "eos_token_ids": (
            generation_eos_from_cached_config(model_id, model_revision, tok)
            if multi_turn
            else frozenset()
        ),
        "package_root": package_root,
        "image_pad_token_id": image_pad_token_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "prompts": prompts,
        "prompts_per_step": prompts_per_step,
        "group_size": group_size,
        "mask_truncated_completions": mask_truncated_completions,
        "temperature": temperature,
        "top_p": float(rl.sampling_top_p),
        "think_penalty": think_penalty,
        "kl_coef": kl_coef,
        "entropy_quantile": entropy_quantile,
        # verl always checkpoints (enable_gradient_checkpointing=True) and always asks for
        # non-reentrant recompute, which the MoE router and GDN chunk-scan die on. resolved here so
        # the child shim can put the flag back to what the model needs.
        "reentrant_checkpointing": bool(_w.grpo_use_reentrant(model_id)),
        "per_turn_credit": per_turn_credit,
        "stop_sequences": stop_sequences,
        "structured_outputs": structured_outputs,
        "max_completion": max_completion,
        # the response TENSOR's width. equals max_completion on a single-turn job; on multi-turn it
        # holds the whole episode (see the derivation above).
        "max_response_len": max_response_len,
        "max_prompt_len": prompt_budget,
        # the engine's full sequence length (prompt + completion), already clamped to the model's
        # own limit. sizes vllm's kv cache and the training token budget, and prompt_budget above is
        # derived from this same value so all three lengths agree.
        "engine_len": vllm_max_len,
        "prompt_opened_thinking": prompt_opened_thinking,
        "lr": learning_rate,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "warmstart_adapter": warmstart_adapter,
        "epochs": epochs,
        "verl_total_epochs": verl_total_epochs,
        "steps": int(steps),
        "save_freq": save_freq,
        "save_at_steps": save_at_steps,
        "ckpt_to_keep": ckpt_to_keep,
        # verl's default preserves the update horizon and on-policy baseline; with no generation reuse, each
        # update gets a fresh rollout batch.
        "ppo_epochs": 1,
        "seed": int(backend_seed(_w.SEED)),
    }


class _GrpoSubprocessStream:
    """one grpo child stream and the evidence latched from that same stream."""

    def __init__(self, proc) -> None:
        self._proc = proc
        # the caller uses start_new_session, so the leader pid remains the group's stable identity.
        self._process_group_id = proc.pid
        self._tail = ChildOutputTail()
        self._terminated = False
        self._orphaned_pipe = False

    def __iter__(self):
        assert self._proc.stdout is not None
        # the child's exit is watched independently of pipe EOF: verl's vllm EngineCore grandchild
        # inherits this same pipe, so a trainer that dies while it lives leaves a pipe nobody will
        # close and this loop would run forever on a paid gpu. the watchdog tears the group down,
        # which frees the cuda context AND closes the pipe, ending this loop.
        with _ChildExitWatchdog(
            self._proc, process_group_id=self._process_group_id, grace_s=_ORPHANED_PIPE_GRACE_S
        ) as watchdog:
            for line in self._proc.stdout:
                # held ACROSS the yield. this is a generator, so the consumer's work for a line runs
                # while suspended here -- counting only the arrival would make a consumer inside one
                # long step look exactly like a reader blocked on a dead pipe.
                with watchdog.handling_line():
                    self._tail.record(line)
                    yield line
        if watchdog.tore_down:
            self._orphaned_pipe = True
            # the group is already gone, so teardown must not be attempted a second time.
            self._terminated = True

    def terminate(self) -> None:
        if self._terminated:
            return
        kill_process_group(self._proc, process_group_id=self._process_group_id)
        self._terminated = True

    def wait_and_classify(self) -> int:
        # BOUNDED. the iterator above ends on stdout EOF, which is not the direct child's exit: verl
        # spawns vllm's EngineCore as a grandchild holding this same merged pipe, so a trainer that
        # dies while the EngineCore lives keeps the pipe open and an unbounded wait parks the
        # attempt on a paid gpu. a child that has not exited within the grace after its own stdout
        # closed is not going to, so tear the group down; `terminate` waits on it.
        try:
            return_code = int(self._proc.wait(timeout=_TEARDOWN_GRACE_S))
        except subprocess.TimeoutExpired:
            self.terminate()
            # `terminate` waits, so the child is normally collected by now. it is still not
            # guaranteed -- a member wedged in uninterruptible io outlives even the SIGKILL -- and a
            # survivor is a failure however the exit reads, so an uncollected child is reported as
            # one rather than defaulted to zero.
            collected = self._proc.returncode
            return_code = int(collected) if collected is not None else 1
        try:
            raise_for_classified_verl_exit(return_code, self._tail)
        except BaseException:
            self.terminate()
            raise
        if return_code != 0:
            # an unclassified nonzero exit RETURNS from the classifier rather than raising, so this
            # is the failing path that reached no teardown. the direct child is gone but its group
            # need not be, and a surviving EngineCore strands the gpu for the next attempt.
            self.terminate()
        if self._orphaned_pipe and return_code == 0:
            # the trainer exited 0 but a descendant held the pipe open past the grace, so the group
            # was killed to release it. the trainer's status says nothing about that descendant, and
            # returning 0 here would publish a partial run as a completed one.
            raise RuntimeError(
                f"verl subprocess {self._proc.pid} exited 0 but a descendant held its output pipe "
                f"open for {_ORPHANED_PIPE_GRACE_S:.0f}s; the process group was torn down to "
                "release the gpu"
            )
        return return_code


def run_rl_train():
    """grpo training on verl, output-compatible with run_rl. see module docstring for scope."""
    t_start = time.time()
    _w.heartbeat("rl_start", gpu=gpu_diagnostics())
    wait_for_gpu(
        _w.JOB_SPEC.gpu.type if _w.JOB_SPEC else None,
        gpu_type=_w.JOB_SPEC.gpu.type if _w.JOB_SPEC else "",
    )
    # no setup_perf_backends() here: torch's tf32 flags are per-process state that a subprocess does
    # not inherit, and this process trains nothing -- verl does, out of process. the child opts in
    # from its own sitecustomize instead (render_tf32_shim, wired into shim_source below).

    # env load, prompt render and tokenization run for minutes on a large split and emit nothing of
    # their own; without the wrap the provider sees silence from rl_start until rl_train_start.
    # (the warm-start adapter pull nested inside carries its own rl_adapter_loading wrap.)
    with liveness_heartbeat("rl_data_loading"):
        inp = _resolve_grpo_inputs()
    env, tok = inp["env"], inp["tok"]
    # what gets saved next to a published adapter. a multimodal adapter is unservable without its
    # image preprocessor, so save the whole processor there; a text run saves the tokenizer alone.
    preprocessor = inp["processor"] or tok
    prompts = inp["prompts"]

    # cache the base model before launching verl, then run verl fully offline so its vllm /
    # transformers never hit hf's (rate-limited) api. flash already owns model prefetch; the verl
    # subprocess simply reuses that cache.
    if inp["model_revision"]:
        download_seconds = _w.prefetch_model(inp["model_id"], revision=inp["model_revision"])
    else:
        download_seconds = _w.prefetch_model(inp["model_id"])
    # verl resolves model.path with HF_HUB_OFFLINE=1, so pass a real snapshot directory. bare ids can
    # miss pinned revisions or cache symlinks and fail after gpu rental. _cached_model_path handles
    # both and raises RetriableInfraError so the run can move to a healthy worker.
    model_path_for_verl = _cached_model_path(inp["model_id"], inp["model_revision"])

    # stable int index -> rollout example, exactly as the retired trl path (reward maps back via this).
    ds_rows, rollout_examples = _w.build_grpo_prompt_dataset(prompts)
    message_prompts = [p["prompt"] for p in prompts]
    indices = [int(r["example_idx"]) for r in ds_rows]
    # ground_truth is a verl-schema placeholder only; the reward bridge scores by example_idx
    # against the live env and never reads it.
    ground_truths = [
        str(ex.get("answer", "") or "") if isinstance(ex, dict) else "" for ex in rollout_examples
    ]

    workdir = f"/tmp/rl_train_seed{_w.SEED}"
    os.makedirs(workdir, exist_ok=True)
    local_dir = os.path.join(workdir, "ckpt")
    # a retry reuses the pod workdir; stale global_step_N dirs from a prior attempt would satisfy
    # latest_global_step_dir and publish an old policy as if this attempt trained it.
    shutil.rmtree(local_dir, ignore_errors=True)
    # restore after the wipe, never before: the wipe is what makes a stale local dir safe, and the
    # resume checkpoint is the one global_step_N this attempt is entitled to start from.
    os.makedirs(local_dir, exist_ok=True)
    resume_step = _restore_verl_resume(local_dir)
    train_pq = os.path.join(workdir, "train.parquet")
    val_pq = os.path.join(workdir, "val.parquet")
    reward_py = os.path.join(workdir, "reward.py")

    # multimodal: decode each prompt's images to png on disk and carry file:// uris in the parquet.
    # verl's dataset loads them through qwen_vl_utils.fetch_image, which reads file:// natively, so
    # the pixels never have to round-trip through arrow. same contract the opd verl path writes.
    image_uris = None
    if inp["multimodal"]:
        image_dir = os.path.join(workdir, "images")
        shutil.rmtree(image_dir, ignore_errors=True)
        image_uris = [
            _materialize_verl_images(
                list(prompt.get("images") or []), inp["package_root"], image_dir, index
            )
            for index, prompt in enumerate(prompts)
        ]

    rows = build_verl_dataset_rows(message_prompts, indices, ground_truths, image_uris)
    write_verl_grpo_parquet(rows, train_pq)
    write_verl_grpo_parquet(rows[: max(1, min(4, len(rows)))], val_pq)
    with open(reward_py, "w") as f:
        f.write(render_reward_module())

    # runtime patches for the verl interpreter. a stale shim from a prior attempt would otherwise
    # keep patching this one, so the file is rewritten every time.
    shim_dir = os.path.join(workdir, "shim")
    os.makedirs(shim_dir, exist_ok=True)
    shim_py = os.path.join(shim_dir, "sitecustomize.py")
    # one sitecustomize holds every patch: python imports it once, so a second file would never be
    # loaded. each feature renderer returns "" when its feature is off; the tf32 fragment is
    # unconditional, so this source is never empty.
    shim_source = "".join(
        part
        for part in (
            # first: torch's matmul flags are process-wide state, and reading them back is how the
            # rest of the child sees the choice. nothing below depends on it, but a later fragment
            # that raised would otherwise cost the whole run its tensor-core throughput.
            render_tf32_shim(),
            render_reentrant_checkpointing_shim(
                inp["reentrant_checkpointing"], multimodal=bool(inp["multimodal"])
            ),
            render_entropy_quantile_shim(inp["entropy_quantile"]),
            render_per_turn_credit_shim(inp["per_turn_credit"]),
            render_stop_sequences_shim(inp["stop_sequences"]),
            render_image_pad_ban_shim(inp["image_pad_token_id"]),
            render_structured_outputs_shim(inp["structured_outputs"]),
            render_exact_save_steps_shim(inp["save_at_steps"], inp["steps"]),
            render_kl_ref_adapter_shim(
                bool(inp["warmstart_adapter"]) and float(inp["kl_coef"]) > 0
            ),
            # gated on the key rather than the resolved logger list: that list needs python_bin,
            # which is resolved after this file is written. the shim is inert either way -- it only
            # fires when verl actually calls wandb.init, which requires wandb in the logger list.
            render_wandb_link_shim() if os.environ.get("WANDB_API_KEY") else "",
        )
        if part
    )
    with open(shim_py, "w") as f:
        f.write(shim_source)

    # multi-turn: copy the child-side agent loop next to the shim so the verl interpreter can
    # import it (see copy_multi_turn_child_modules for why it is a copy and not an import).
    if inp["multi_turn"]:
        copy_multi_turn_child_modules(shim_dir)

    # the localhost bridge carries rollouts and #607 reward components. generation_size closes each
    # generation when its last scoring call finishes, before a later stdout step line can mix in the
    # next generation. test_freq=-1 and val_before_train=false make every bridged completion training
    # data.
    observability = RewardObservabilityBuffer(
        generation_size=int(inp["prompts_per_step"]) * int(inp["group_size"]),
    )
    # filled from the child's marker line; stays empty when wandb is off (see render_wandb_link_shim).
    wandb_link: dict[str, str | None] = {}

    def _score_batch(requests: list[tuple[int, str]]) -> list[float]:
        # grade the whole batch before touching the observability lock. the env's scorer may block on
        # judge i/o, while record is intentionally a short per-result critical section.
        scored = score_single_turn_batch(
            env,
            [(solution_str, rollout_examples[int(index)]) for index, solution_str in requests],
            tok=tok,
            thinking=bool(_w.THINKING),
            prompt_opened_thinking=inp["prompt_opened_thinking"],
            think_penalty=inp["think_penalty"],
        )
        results = []
        for (index, solution_str), (score, breakdowns) in zip(requests, scored, strict=True):
            observability.record(message_prompts[int(index)], solution_str, score, breakdowns)
            results.append(score)
        return results

    def _score_for_profile(index: int, solution_str: str) -> float:
        """score for profiling without training observability side effects.

        do not seed rollout buffers. errors propagate so a broken grader is not measured as fast.
        """
        return score_single_turn(
            env,
            solution_str,
            rollout_examples[int(index)],
            tok=tok,
            thinking=bool(_w.THINKING),
            prompt_opened_thinking=inp["prompt_opened_thinking"],
            think_penalty=inp["think_penalty"],
            raise_on_error=True,
        )

    # the profiler times the single-turn grading path (env.reward / env.scores_breakdown on one
    # completion). a multi-turn env scores a terminal episode instead, so that timing would neither
    # describe nor even validly reach its reward path.
    reward_profile = (
        None
        if inp["multi_turn"]
        else _log_reward_profile(
            env,
            _score_for_profile,
            rollout_examples,
            int(inp["prompts_per_step"]) * int(inp["group_size"]),
        )
    )

    multi_turn_bridge = (
        MultiTurnBridge(
            env,
            rollout_examples,
            # index-aligned with rollout_examples: build_grpo_prompt_dataset preserves order.
            env_prompts=[p["env_prompt"] for p in prompts],
            max_turns=int(inp["max_turns"]),
            per_turn_credit=bool(inp["per_turn_credit"]),
            on_episode_scored=observability.record,
        )
        if inp["multi_turn"]
        else None
    )
    server, reward_url = start_reward_server(
        _score_for_profile,
        example_count=len(rollout_examples),
        multi_turn_bridge=multi_turn_bridge,
        rollout_batch=int(inp["prompts_per_step"]) * int(inp["group_size"]),
        score_batch=None if inp["multi_turn"] else _score_batch,
    )
    # bound before the try so the finally can always ask whether it was started.
    resume_uploader: _VerlResumeUploader | None = None
    # verl trains out-of-process, so torch's in-process allocator counter never sees the trainer.
    # nvidia-smi is the only reading that covers the child, and it must be sampled while the child
    # runs; stopping it in the finally keeps the reading even when the run crashes or is cancelled.
    gpu_sampler = _NvidiaSmiPeakSampler().start()
    device_peak_gpu_gb: float | None = None
    try:
        # provisioning and the cold verl capability probe can take minutes without step output. keep
        # liveness running so the stall watchdog does not fail healthy setup (#442). there is no
        # monotonic progress counter here, only the keepalive.
        with liveness_heartbeat("rl_configuring"):
            python_bin = resolve_verl_python(
                workdir, install_wandb=bool(os.environ.get("WANDB_API_KEY"))
            )
            # gdn boundary resets need the CHILD to have fla + causal_conv1d: without them the
            # kwargs are accepted and discarded, so packed examples bleed state into each other.
            # the modeling module is resolved HERE, in the parent, because it needs a hub/cache read
            # the child must not repeat; "" skips the question for a non-hybrid.
            gdn_hybrid = model_is_gdn_hybrid(inp["model_id"], inp["model_revision"])
            gdn_module = (
                gdn_probe_module(inp["model_id"], inp["model_revision"]) if gdn_hybrid else ""
            )
            # ONE child answers every independent capability question. each used to cost its own
            # interpreter, and the torch/verl import -- not the question -- was the price.
            caps = probe_verl_capabilities(python_bin, gdn_module)
        # masking truncated completions is a fork-only rollout field. FLASH_VERL_PYTHON can point at a
        # stock verl, and hydra would compose the unknown key only to abort in dataclass conversion.
        # fail here with the cause instead, and never silently train on truncated completions.
        if inp["mask_truncated_completions"] and not verl_declares_rollout_field(
            caps, "mask_truncated_completions"
        ):
            raise RuntimeError(
                f"grpo requested mask_truncated_completions but the verl at {python_bin} does not "
                "support it. the worker image's baked interpreter predates the freesolo fork; "
                f"rebuild the worker image with '{VERL_REQUIREMENT}' installed."
            )
        # the shim is appended here rather than with the shims above because the answer needs
        # python_bin, resolved in the block above; sitecustomize is imported by the child, which has
        # not started yet.
        # raises when a gdn child cannot honor resets: the padded fallback that used to handle that
        # case cannot complete a step on verl's fsdp engine. see require_gdn_boundary_resets.
        gdn_reset_arch = require_gdn_boundary_resets(caps, gdn_module)
        if gdn_reset_arch is not None:
            with open(shim_py, "a") as f:
                f.write(render_gdn_varlen_shim(gdn_reset_arch))

        expected_steps = int(inp["steps"])
        # verl logs from its own interpreter; gate wandb on that env (see resolve_verl_loggers).
        loggers = resolve_verl_loggers(caps)
        _spec = _w.JOB_SPEC
        project_name = (_spec.wandb.project if _spec and _spec.wandb else None) or "flash"
        experiment_name = _w.wandb_run_name()
        # fp8 kv cache on ada/hopper+ (cc>=8.9), matching the sizing math in engine/vram.py. NOT for
        # hybrid linear-attention (GDN) models: vllm's fp8-kv wake path (init_fp8_kv_scales) assumes a
        # plain kv tensor and crashes on the hybrid cache ('list' has no zero_) under verl sleep/wake.
        try:
            import torch as _torch_cc

            _cc_ok = bool(
                _torch_cc.cuda.is_available() and _torch_cc.cuda.get_device_capability() >= (8, 9)
            )
        except Exception:  # no cuda / probe failure -> conservative bf16 kv
            _cc_ok = False
        # reuse the gdn answer resolved above rather than re-probing; see gdn_hybrid.
        fp8_kv = _cc_ok and not gdn_hybrid
        # one capability probe, both rollout decisions below. asked of the verl interpreter, whose
        # torch/vllm stack is the one that has to run the rollout.
        verl_cc = verl_device_capability(caps)
        # blackwell needs both rollout attention backends pinned; vllm 0.19.1's own defaults pick
        # flash-attn, which is PTX-unreliable on sm120 (silent empty rollouts) and routes the ViT
        # into an unimportable CUTE kernel on sm100/sm120. no-op off blackwell.
        attention_backend, mm_encoder_attn_backend = resolve_blackwell_attention_backends(
            caps, verl_cc
        )
        # sm86 is the one arch whose vllm 0.19.1 graph capture is a measured failure (completions
        # repeat to the token cap without emitting EOS), so only it runs the rollout eagerly. see
        # resolve_rollout_enforce_eager for the per-arch evidence and why one knob is enough.
        enforce_eager = resolve_rollout_enforce_eager(verl_cc)
        cfg = _build_verl_training_cfg(
            inp,
            train_files=train_pq,
            val_files=val_pq,
            model_id=model_path_for_verl,
            thinking=bool(_w.THINKING),
            loggers=loggers,
            fp8_kv=fp8_kv,
            enforce_eager=enforce_eager,
            attention_backend=attention_backend,
            mm_encoder_attn_backend=mm_encoder_attn_backend,
            reward_path=reward_py,
            local_dir=local_dir,
            project_name=project_name,
            experiment_name=experiment_name,
            gpu_type=(_w.JOB_SPEC.gpu.type if _w.JOB_SPEC else ""),
            n_gpus=gpu_count_of(_w.JOB_SPEC),
            # resolved from the out-of-process capability probe, never by opening cuda in this
            # parent -- see fused_ce_backend.
            ce_backend=fused_ce_backend(caps),
        )
        overrides = build_verl_overrides(cfg)
        # the executor budget is sized per run now, so print what this one actually asked for --
        # a vllm init failure reports the demand against the free memory, and the demand is
        # otherwise invisible in the log.
        print(
            f"[rl-verl] rollout gpu_memory_utilization={cfg['gpu_mem_util']:.4f}",
            flush=True,
        )

        setup_seconds = time.time() - t_start
        _w.heartbeat("rl_train_start", setup_seconds=setup_seconds, gpu=gpu_diagnostics())
        _w.heartbeat("rl_step", step=0, initial=True)
        t_train = time.time()
        # equal within-group rewards produce zero advantages and gradients. collect per-step spread;
        # reward mean and pg_loss cannot prove a signal. declare this before the uploader because its
        # publication gate closes over the histories.
        adv_spread_history: list[float] = []
        resume_uploader = _VerlResumeUploader(
            local_dir,
            resume_step=resume_step,
            required_steps=inp["save_at_steps"],
            export_root=os.path.join(workdir, "exports"),
            python_bin=python_bin,
            model_id=inp["model_id"],
            model_revision=inp["model_revision"],
            preprocessor=preprocessor,
            # a resumed run's restored weights already carry the earlier steps' updates, so this
            # worker's own spread history cannot speak for them; let it publish as before and leave
            # the verdict to the same abstention _check_grpo_had_a_gradient makes.
            had_gradient=(
                None if resume_step else lambda: any(spread > 0.0 for spread in adv_spread_history)
            ),
        )
        resume_uploader.credit_durable_required_steps(resume_step)
        resume_uploader.start()
        step_box = [0]

        def _progress():
            return step_box[0]

        # allowlist the child env because ray fans it to every actor; scoring stays in the parent, so
        # the child needs no platform hf token, github token, or user secrets. FLA_ kernel settings
        # still cross through _CHILD_ENV_PREFIXES.
        env_for_verl = _build_verl_child_env(shim_dir=shim_dir, wandb_enabled="wandb" in loggers)
        env_for_verl["FLASH_VERL_REWARD_URL"] = reward_url
        # the model is prefetched above; keep the subprocess off hf's rate-limited api.
        env_for_verl["HF_HUB_OFFLINE"] = "1"
        env_for_verl["TRANSFORMERS_OFFLINE"] = "1"
        env_for_verl["HF_HUB_DISABLE_XET"] = "1"
        if inp["multi_turn"]:
            # the plugin named here and the loop it builds live in shim_dir, so PYTHONPATH must
            # carry it -- see below.
            env_for_verl.update(
                multi_turn_child_env(inp, reward_url=reward_url, thinking=bool(_w.THINKING))
            )
        # python imports sitecustomize automatically at startup, so the shim patches verl before
        # main_ppo runs. prepend rather than replace: an inherited PYTHONPATH may carry the verl
        # install itself, and ray workers inherit this env so every actor gets the same patch.
        # multi-turn needs the same entry for its copied-in agent loop modules. unconditional: the
        # shim always carries at least the tf32 fragment, so a skipped entry would silently drop it.
        env_for_verl["PYTHONPATH"] = os.pathsep.join(
            item for item in (shim_dir, os.environ.get("PYTHONPATH", "")) if item
        )
        step_re = re.compile(r"step:\s*(\d+)")
        reward_history: list[float] = []
        resp_len_history: list[float] = []
        loss_curve: list[float] = []
        last_dump_step = [-1]
        # consecutive LocalLogger metric arrival gaps are complete optimizer-step walls. do not use
        # train_wall/steps_run: it includes setup/upload and divides session wall by an absolute resumed
        # checkpoint step.
        step_line_times: list[float] = []
        # per-step backlog for `flash runs log -f`, rebuilt from verl's own step lines because its
        # trainer runs out of process and cannot host an in-process trainer callback. read by the liveness thread
        # below, so mutate it in place (append_step_metrics) rather than rebinding.
        metrics_last: list[dict] = []
        sent_first_metrics = False

        def _reward_observability() -> dict:
            """return reward metrics and sampled completions for one heartbeat.

            verl is out of process, so liveness/stdout threads read the bridge's buffer. this is
            read-only: only generation completion drains it, and mid-generation heartbeats repeat the
            last complete reading.
            """
            return observability.heartbeat_fields()

        with liveness_heartbeat(
            "rl_step",
            progress=_progress,
            fields=lambda: {"metrics_last": list(metrics_last), **_reward_observability()},
            progress_step=True,
        ):
            # claimed before the child exists, so a grandchild it orphans reparents here and can be
            # reaped at teardown. this process is not pid 1 (the runpod handler is), so without it
            # every wait answers ChildProcessError for a zombie nobody will collect.
            adopt_orphaned_descendants()
            proc = subprocess.Popen(
                [python_bin, "-m", "verl.trainer.main_ppo", *overrides],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env_for_verl,
                start_new_session=True,
            )
            child_stream = _GrpoSubprocessStream(proc)
            try:
                for line in child_stream:
                    print(f"[verl] {line}", end="", flush=True)
                    link = parse_wandb_link(line)
                    if link is not None:
                        wandb_link.update(link)
                    m = step_re.search(line)
                    if m:
                        step_box[0] = int(m.group(1))
                        # dump one sample completion per new step to the flash log (#607).
                        if step_box[0] != last_dump_step[0]:
                            # the generation boundary: verl logs this line once its step is scored,
                            # so everything the reward bridge buffered since the last one is that
                            # step's complete output. seal it before the preview reads `latest`, so
                            # both the log line and the heartbeat describe the same generation.
                            observability.close_generation(step_box[0])
                            # asks for THIS step's rows, not merely the newest: when the line is
                            # spent on a generation the queue already dropped, nothing is published
                            # and the previous generation's text would print under this step.
                            samp = observability.latest_for_step(step_box[0])
                            if samp:
                                last_dump_step[0] = step_box[0]
                                _, completion, reward = samp
                                text = sanitize_rollout_text(sample_completion_text(completion))
                                preview = " ".join(text[:300].split())
                                print(
                                    f"[rl-verl] step {step_box[0]} sample (reward={reward:.3f}): {preview}",
                                    flush=True,
                                )
                    step_metrics = parse_verl_step_metrics(line)
                    if step_metrics is not None:
                        step_line_times.append(time.time())
                        # a run constant rather than a verl metric, so it is stamped here from
                        # the resolved run config.
                        step_metrics["max_completion_tokens"] = inp["max_completion"]
                        append_step_metrics(
                            metrics_last, step_metrics, limit=GRPO_METRIC_HISTORY_LIMIT
                        )
                        # the worker's error path reads this global, so a run that dies mid-training
                        # still reports the steps it did complete (worker/__init__.py:_err_metrics).
                        LATEST_GRPO_METRICS_LAST[:] = metrics_last
                        # rl_train_start arms a 900s throttle, so force the first backlog through like
                        # heartbeat.py force_first_samples and retry until committed. later emissions remain
                        # throttled to protect the hf commit cap.
                        if not sent_first_metrics:
                            sent_first_metrics = _w.heartbeat(
                                "rl_step",
                                force=True,
                                step=step_metrics["step"],
                                metrics_last=list(metrics_last),
                                **_reward_observability(),
                                gpu=gpu_diagnostics(include_torch=False),
                            )
                        # per-step series for train_meta observability parity. these live on the same
                        # line as everything else: verl's only console metric sink is LocalLogger,
                        # which always prints "step:N - ..." (verl/utils/logger/aggregate_logger.py),
                        # so a line without a step carries no metric to collect.
                        for verl_key, sink in (
                            ("critic/rewards/mean", reward_history),
                            ("actor/pg_loss", loss_curve),
                            ("response_length/mean", resp_len_history),
                        ):
                            value = parse_verl_metric(line, verl_key)
                            if value is not None:
                                sink.append(value)
                        # advantages/max and /min are emitted for every step outside verl's
                        # use_critic branch (trainer/ppo/metric_utils.py), so they are present under
                        # grpo even though the key is namespaced critic/.
                        adv_max = parse_verl_metric(line, "critic/advantages/max")
                        adv_min = parse_verl_metric(line, "critic/advantages/min")
                        if adv_max is not None and adv_min is not None:
                            adv_spread_history.append(adv_max - adv_min)
                rc = child_stream.wait_and_classify()
            except BaseException:
                # the stream loop died (upload error, cancel, oom in the parent): a still-running
                # verl child would keep burning the gpu unattended, so kill its whole process group.
                # this escalates to SIGKILL after the grace period, which a bare SIGTERM does not: a
                # vllm EngineCore that ignores the term keeps its cuda context and strands the gpu
                # for every later job on a reusable worker.
                child_stream.terminate()
                raise
        if rc != 0:
            raise RuntimeError(
                f"verl.trainer.main_ppo exited {rc}; see the flash log for the traceback"
            )
        # the gradient verdict runs here, ahead of required-save completeness, because a zero-spread
        # run withholds every required deployable BY DESIGN: checking completeness first would raise
        # on artifacts the gate is deliberately holding and report a checkpoint-publication failure
        # -- the symptom -- instead of the constant reward signal that caused it. raising inside the
        # try still runs the finally below, so the reward server and gpu sampler shut down either way.
        _check_grpo_had_a_gradient(
            reward_history,
            adv_spread_history,
            resumed=bool(resume_step),
            # a resume already at the target runs zero steps and emits zero metrics; that is a
            # complete policy, not a broken reward bridge. steps_run below still has to reach
            # expected_steps, so this cannot excuse a run that stopped short.
            already_complete=bool(resume_step) and resume_step >= expected_steps,
        )
        # training finished cleanly, so a missing required save is a real defect rather than a
        # side effect of a crash. stop here (not in finally, which suppresses) to surface it.
        # only when exact saves were requested: without them the drain stays best-effort, and
        # letting a slow resume upload raise here would fail an otherwise-successful run.
        if resume_uploader is not None and resume_uploader.required_steps:
            resume_uploader.stop()
            resume_uploader.raise_if_incomplete()
    finally:
        # drain before the reward server goes down: on a cancel or crash the last completed
        # checkpoint is exactly the one a retry needs, so it is worth uploading on the way out.
        if resume_uploader is not None:
            with contextlib.suppress(Exception):
                resume_uploader.stop()
        with contextlib.suppress(Exception):
            device_peak_gpu_gb = gpu_sampler.stop_gb()
        # bridge first: the scoring thread is what the server's routes block on, so stopping the
        # server before it would strand a scoring episode on an event nothing will ever set.
        if multi_turn_bridge is not None:
            with contextlib.suppress(Exception):
                multi_turn_bridge.shutdown()
        server.shutdown()
        # every job boundary, not just the failing ones. `kill_process_group` runs on exceptions
        # alone here, so a straggler an earlier teardown SIGKILLed but could not drain in time would
        # otherwise be collected only by the next FAILING job: a reusable worker running successful
        # grpo jobs after one late straggler keeps that zombie for life.
        with contextlib.suppress(Exception):
            reap_stragglers()

    # collect verl's lora checkpoint -> flash-servable peft adapter, then reuse flash finalize.
    out_dir = f"/tmp/rl_seed{_w.SEED}"
    adapter_dir = f"{out_dir}/adapter"
    shutil.rmtree(adapter_dir, ignore_errors=True)
    os.makedirs(adapter_dir, exist_ok=True)
    train_wall = time.time() - t_train
    # the zero-gradient verdict already ran inside the try above, ahead of required-save
    # completeness, so that a withheld deployable reports the reward cause rather than the
    # publication symptom.
    actor_dir, steps_run = latest_global_step_dir(local_dir)
    if steps_run < expected_steps:
        raise RuntimeError(
            f"grpo completed {steps_run}/{expected_steps} requested optimizer updates"
        )

    with liveness_heartbeat(
        "rl_finalizing",
        progress=lambda: steps_run,
        fields=lambda: {"metrics_last": list(metrics_last)},
        progress_step=True,
        keepalive=True,
    ):
        export_peft_adapter(
            actor_dir, adapter_dir, base_model_id=inp["model_id"], python_bin=python_bin
        )
        preprocessor.save_pretrained(adapter_dir)
        stamp_adapter_dir_provenance(adapter_dir, inp["model_id"], inp["model_revision"])
        _w.write_base_model_provenance(adapter_dir, inp["model_id"], inp["model_revision"])
        _w.hf_upload_folder(adapter_dir, "adapter", required=True)
        # preserve the final checkpoint only when exact save steps are not configured: with
        # save_at_steps set the customer asked for those steps and nothing else.
        if final_save_due(steps_run, inp["save_at_steps"]):
            _w.publish_deployable_checkpoint(adapter_dir, steps_run)

    _w.heartbeat(
        "rl_trained",
        train_wall=train_wall,
        step=steps_run,
        gpu=gpu_diagnostics(),
        metrics_last=list(metrics_last),
    )
    _w.write_train_meta(
        phase="rl",
        adapter_dir=adapter_dir,
        model_id=inp["model_id"],
        train_wall=train_wall,
        setup_seconds=setup_seconds,
        train_tokens=0,
        # carry the completed step, as the sft and opd finalizers already do. without it the
        # rl_train_done and done heartbeats land stepless and overwrite the stepped rl_trained ping
        # above, so a cancel arriving between here and DONE reprices a fully trained run at zero
        # steps (actual_steps_run returns 0 for a non-training stage with no step).
        step=steps_run,
        heartbeat_fields={"metrics_last": list(metrics_last)},
        generated_tokens=int(
            sum(resp_len_history)
            / max(1, len(resp_len_history))
            * steps_run
            * inp["prompts_per_step"]
            * inp["group_size"]
        )
        if resp_len_history
        else 0,
        notes=_build_verl_train_notes(
            inp,
            steps_run=steps_run,
            retained_prompts=len(prompts),
            reward_history=reward_history,
            loss_curve=loss_curve,
            resumed=bool(resume_step),
            download_seconds=download_seconds,
            device_peak_gpu_gb=device_peak_gpu_gb,
            fp8_kv=fp8_kv,
            wandb_project=project_name if "wandb" in loggers else None,
            wandb_run_name=experiment_name if "wandb" in loggers else None,
            wandb_url=wandb_link.get("wandb_url"),
            wandb_id=wandb_link.get("wandb_id"),
            reward_profile=reward_profile,
            step_intervals=_step_intervals(step_line_times),
            reward_bridge_batching=not inp["multi_turn"],
            gdn_boundary_resets=gdn_hybrid or None,
        ),
    )
