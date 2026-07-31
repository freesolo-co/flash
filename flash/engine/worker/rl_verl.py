"""verl-backed grpo training path for the fine-tuning worker.

selected by the private worker env FLASH_RL_BACKEND=verl. this path exists so grpo can run on
verl (volcengine hybridflow) instead of trl, as the first step toward multi-node rl. verl's
torch/vllm pins are incompatible with flash's, so flash NEVER imports verl in-process: verl runs
as a subprocess against a separate interpreter (FLASH_VERL_PYTHON, or a venv provisioned on the
pod). reward parity with the trl path is provided by a localhost rpc bridge that scores each
completion against flash's live env, so verl and trl compute identical rewards.

scope: non-tool grpo, single- or multi-turn, text or multimodal. a multi-turn env is driven by
flash's own verl agent loop (grpo_multiturn.py), which runs in the verl interpreter and calls back
into this process for every environment reply. openai function-calling tool envs are still out of
scope and raise rather than silently training on a different contract.
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import gcd

from flash.engine.multiturn_reward_scoring import RolloutScoreRequest, score_rollouts
from flash.engine.recipe import RECIPE
from flash.engine.steps import (
    final_save_due,
    on_policy_steps,
    resolve_update_horizon,
    validate_save_steps,
)
from flash.engine.structured_outputs import (
    describe_structured_outputs,
    parse_structured_outputs,
    reasoning_parser_for,
)
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.heartbeat import (
    GRPO_METRIC_HISTORY_LIMIT,
    LATEST_GRPO_METRICS_LAST,
    RewardObservabilityBuffer,
    liveness_heartbeat,
)
from flash.engine.worker.hf import _deployable_adapter_on_hf
from flash.engine.worker.multiturn_glue import validate_glue_template
from flash.engine.worker.opd_gkd import generation_eos_from_cached_config
from flash.engine.worker.perf import gpu_diagnostics, setup_perf_backends, wait_for_gpu
from flash.engine.worker.rng import backend_seed, seed_training_rngs
from flash.engine.worker.rollout_samples import (
    sample_completion_text,
    sanitize_rollout_text,
)
from flash.engine.worker.sft_train import (
    _hydra_val,
    _materialize_verl_images,
    _multimodal_messages_with_images,
    _NvidiaSmiPeakSampler,
    _verl_image_message_content,
)
from flash.engine.worker.verl_common import (
    VERL_REQUIREMENT,
    agent_loop_workers,
    append_step_metrics,
    clamp_engine_len,
    model_max_position_embeddings,
    parse_verl_metric,
    parse_verl_step_metrics,
    parse_wandb_link,
    render_wandb_link_shim,
    resolve_blackwell_attention_backends,
    resolve_verl_loggers,
    resolve_verl_python,
    verl_supports_rollout_field,
)
from flash.spec import DEFAULT_CREDIT_ASSIGNMENT, gpu_count_of

DATA_SOURCE = "flash_env"


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

    verl passes ``extra_info`` through to the reward fn; we carry the flash ``example_idx`` in
    ``extra_info.index`` so the reward bridge can map a completion back to its rollout example.

    on a multimodal job every row also carries an ``images`` column of ``file://`` uris and its
    message content is flattened to a ``<image>``-bearing string: verl's RLHFDataset re-expands
    those placeholders back into content blocks (``_build_messages``) and asserts the placeholder
    count equals ``len(images)``, so the two must be produced together. this is the same contract
    the sft and opd verl paths already write.

    ``<image>``, ``<video>`` and ``<audio>`` are therefore reserved substrings, not ordinary text,
    on every row of a multimodal job -- including a text-only row in a mixed dataset, because verl
    splits on all three whenever ANY modality column is non-empty for that row. a prompt that merely
    talks about one of these tokens would be re-expanded as real media and abort dataset loading
    inside verl with a bare offset assertion (and for video/audio we never write a column at all, so
    a single literal occurrence asserts against an empty list). reject it here instead, where the
    message is still attributable to its example.
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
    """count a multimodal prompt the way the rollout will actually see it.

    an image is one content block on the way in and hundreds of placeholder tokens on the way out,
    and only the processor knows the expansion (it depends on the image's resolution). counting
    with the bare tokenizer would therefore under-measure an image row by most of its length, so
    the budget filter would admit prompts the engine then rejects.

    returns the expanded ids alongside the rendered text, because the caller needs the same text to
    decide whether the chat template left an open ``<think>`` span.
    """
    from flash.multimodal import decode_image_descriptors

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
    """explicit arrow schema for multimodal grpo rows.

    ``Dataset.from_list`` infers one type per column across all rows. a mixed text/image job has
    rows whose ``images`` list is empty, and inference on an all-empty column picks a null type
    that verl's dataset then cannot read back as a struct. pinning the schema keeps every row the
    same shape regardless of how many images it happens to carry -- the same reason the opd verl
    writer pins its own.
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
    """build the hydra override list for `python -m verl.trainer.main_ppo` (grpo + lora + vllm).

    carries the flash grpo recipe: dr-grpo advantages (no std norm, constant-length loss), the
    job's kl coefficient (flash default 0 = no kl term), constant lr, seed, sampling top_p, and one
    ppo epoch so total_training_steps counts optimizer updates. unlike trl's generation-batch reuse,
    verl samples a fresh rollout per update, so rollout volume and policy staleness still differ.
    """
    kl_on = float(cfg["kl_coef"]) > 0
    o = [
        "algorithm.adv_estimator=grpo",
        # dr-grpo recipe: group-mean-centered advantages with NO std normalization, and a
        # constant-length loss aggregation (no per-response length bias). matches the trl path's
        # scale_rewards=none + loss_type=dr_grpo.
        "algorithm.norm_adv_by_std_in_grpo=False",
        f"actor_rollout_ref.actor.loss_agg_mode={cfg['loss_agg_mode']}",
        "algorithm.use_kl_in_reward=False",
        # truncated importance sampling (token-level, cap 2.0): corrects the vllm-rollout vs
        # fsdp-train policy mismatch. matches the trl path's tis recipe (token_truncate, c_max=2.0);
        # verl otherwise defaults to sequence-level tis, so pin token to match flash.
        "algorithm.rollout_correction.rollout_is=token",
        "algorithm.rollout_correction.rollout_is_threshold=2.0",
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
        # thread flash's thinking mode so the rollout sees the same prompt as the trl path.
        f"+data.apply_chat_template_kwargs.enable_thinking={str(bool(cfg.get('thinking', False))).lower()}",
        f"data.seed={cfg['seed']}",
        # rollout sampling seed. NOT `rollout.seed`: verl 0.8.0's RolloutConfig declares no such
        # field, so a bare key fails hydra composition and a `+`/`++` prefix composes but then dies
        # in omega_conf_to_dataclass with an unexpected-kwarg TypeError. engine_kwargs is a declared
        # dict on both 0.8.0 and 0.9.x and is spread into the vllm engine args *after* verl's own
        # "seed" entry, so it wins. `++` because the sub-key is absent from the composed node.
        # single replica here (tp == n_gpus, nnodes=1), so verl's replica_rank offset is always 0.
        f"++actor_rollout_ref.rollout.engine_kwargs.vllm.seed={cfg['seed']}",
        # multimodal: hand verl the images column and let it own prompt expansion. the parquet
        # carries `<image>` placeholders plus a top-level `images` list of file:// uris, which
        # RLHFDataset re-expands into content blocks and qwen_vl_utils.fetch_image loads. the
        # trainer must build a processor rather than a bare tokenizer, hence trust_remote_code and
        # return_raw_chat. filter_overlong_prompts is verl's own budget drop; flash pre-filters
        # with the SAME processor above, so it should find nothing left to drop -- it is a
        # belt-and-braces guard because verl RAISES rather than truncating an over-budget
        # multimodal prompt (agent_loop.py), which would kill the run mid-rollout.
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
        # memory: match the trl path's gradient checkpointing.
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        # 32k contexts: fused linear-CE computes logprobs/entropy from hidden states + lm_head in
        # chunks (FusedLinearForPPO), never materializing the [tokens, vocab] logits tensor
        # (~130 GB at 32k on a 248k vocab). torch backend = numerically exact, no extra deps.
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.use_fused_kernels=True",
        "actor_rollout_ref.model.fused_kernel_options.impl_backend=torch",
        *(
            [f"actor_rollout_ref.model.lora_adapter_path={cfg['warmstart_adapter']}"]
            if cfg.get("warmstart_adapter")
            else []
        ),
        f"actor_rollout_ref.actor.optim.lr={cfg['lr']}",
        # only the lora adapter is trainable; decaying its weights toward zero is not part of grpo.
        "actor_rollout_ref.actor.optim.weight_decay=0.0",
        # 0 warmup -> verl's warmup+constant scheduler holds lr flat, matching the trl constant recipe.
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={cfg['prompts_per_step']}",
        # 32k contexts: bound the backward pass by TOKENS, not by sequence count. a fixed
        # micro_batch_size_per_gpu of N sequences is N*engine_len tokens in the worst case, so the
        # only count that is safe at 32k is 1 -- which then wastes most of the budget on the short
        # sequences that make up a typical batch. dynamic bsz packs each micro-batch up to
        # max_token_len instead, so long sequences get their own pass and short ones share.
        # verl's engine multiplies this per-gpu budget by sp_size itself, so it must NOT be
        # pre-divided by the ulysses width.
        "actor_rollout_ref.actor.use_dynamic_bsz=true",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={cfg['max_token_len_per_gpu']}",
        # ppo_epochs multiplies verl's update loop, so its default of 1 preserves the requested update
        # count and on-policy baseline. unlike trl reuse, verl samples a fresh rollout for every update.
        f"actor_rollout_ref.actor.ppo_epochs={cfg['ppo_epochs']}",
        # multi-gpu shape: shard every card along the sequence (ulysses) rather than the batch, and
        # give vllm the same width for tensor parallelism. verl builds mesh_shape=(dp, sp) from this,
        # so sp == n_gpus keeps dp == 1: the optimizer still sees ONE global batch of exactly
        # prompts_per_step * group_size, identical to the single-gpu recipe. splitting the batch
        # instead would silently change the gradient (dp shards ppo_mini_batch_size). sequence
        # sharding is also what makes long contexts fit, since activations divide by n_gpus.
        # ref (when kl is on) inherits this through dp_ref.yaml's oc.select on the actor key.
        # requires use_remove_padding, which this recipe already sets above.
        f"actor_rollout_ref.actor.ulysses_sequence_parallel_size={cfg['n_gpus']}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.n={cfg['group_size']}",
        # verl 0.8.0 hardcodes async rollout (ray_trainer.py:914), and AgentLoopManager splits the
        # rollout batch across agent.num_workers with DataProto.chunk, which asserts EXACT
        # divisibility (agent_loop.py:1111 -> protocol.py:874). the batch it chunks is
        # prompts_per_step * group_size, so verl's default of 8 workers aborts the run before the
        # first step whenever that product is not a multiple of 8 - e.g. the last short batch of an
        # epoch, or any small job. flash's own defaults (64 x 8) happen to divide, but nothing
        # constrains a user's [train].batch_size or group_size to keep that true. size the worker
        # pool to the batch instead: the largest divisor of the rollout batch that is <= 8, which is
        # always valid and still parallelizes whenever the batch allows it.
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
        # sleep_unsupported). verl defaults free_cache_engine and enable_sleep_mode BOTH True and
        # offloads between every step, so without this a flagged model wedges at the first step
        # boundary instead of failing fast. free_cache_engine is the one that actually gates the
        # sleep()/wake_up() rpcs (vllm_rollout.resume/release, engine_workers resume path);
        # enable_sleep_mode is what builds the sleep-capable engine in the first place, so both go.
        # safe because the parse-time gate sizes a flagged job on its RESIDENT peak and rejects a
        # config that cannot fit -- staying resident cannot admit an OOM the sleep path avoided.
        *(
            [
                "actor_rollout_ref.rollout.free_cache_engine=false",
                "actor_rollout_ref.rollout.enable_sleep_mode=false",
            ]
            if cfg.get("sleep_unsupported")
            else []
        ),
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
        f"trainer.logger=[{cfg['loggers']}]",
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
        # fp8 kv cache on ada/hopper+ (cc>=8.9), matching the trl colocate path; tis (above) covers
        # the extra rollout-vs-train mismatch fp8 introduces. '+' appends the key under the existing
        # engine_kwargs.vllm struct (it is not a default field).
        o.append("+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8")
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
        # thinking model reasons freely and only its answer is constrained. on the trl path this is
        # injected into the colocate llm kwargs; verl spreads engine_kwargs.vllm straight into
        # AsyncEngineArgs, where reasoning_parser is a real field, so a plain override suffices.
        # '+' appends under the existing engine_kwargs.vllm struct, as kv_cache_dtype does above.
        o.append(
            f"+actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser={_reasoning_parser}"
        )
    return o


def _sleep_unsupported(model_id: str) -> bool:
    """Whether vLLM's sleep/wake cycle is non-functional for this model.

    A few models HANG on wake/reload rather than erroring (live-confirmed, every attempt), so the
    catalog flags them and GRPO must keep the rollout engine RESIDENT instead. The parse-time gate in
    flash.engine.vram already sizes such a job on its resident peak and rejects a config too long to
    fit, so staying resident cannot admit anything that would OOM.
    """
    from flash.catalog import MODELS

    info = MODELS.get(model_id)
    return bool(info is not None and getattr(info, "sleep_unsupported", False))


def _build_verl_training_cfg(
    inp: dict,
    *,
    train_files: str,
    val_files: str,
    model_id: str,
    thinking: bool,
    loggers: str,
    fp8_kv: bool,
    attention_backend: str | None,
    mm_encoder_attn_backend: str | None,
    reward_path: str,
    local_dir: str,
    project_name: str,
    experiment_name: str,
    n_gpus: int = 1,
) -> dict:
    engine_len = int(inp["engine_len"])
    return {
        "train_files": train_files,
        "val_files": val_files,
        "model_id": model_id,
        "lora_rank": inp["lora_rank"],
        "lora_alpha": inp["lora_alpha"],
        "target_modules": "all-linear",
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
        "gpu_mem_util": 0.5,
        "n_gpus": n_gpus,
        "loggers": loggers,
        "fp8_kv": fp8_kv,
        "attention_backend": attention_backend,
        "mm_encoder_attn_backend": mm_encoder_attn_backend,
        "sleep_unsupported": _sleep_unsupported(inp["model_id"]),
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
    """Wall-clock length of each step, from the times its metric lines arrived.

    N step lines bound N-1 whole steps. The span BEFORE the first line is not one of them: it holds
    engine init, weight load and cudagraph capture, which a later step never pays again. Nothing is
    known about the span after the last line either, since the run may have been cancelled mid-step.
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
    """Share of a step the gpu spent waiting on grading, from this run's OWN numbers.

    Both inputs are measured rather than modelled: the per-completion latency comes from the
    warm-up profile against the real env, and the gpu-bound half is the observed step wall minus
    the grading it contains. A modelled step time would report the estimator's opinion of the run
    back to the estimator, which cannot then be used to check it.

    ``step_intervals`` are the wall-clock gaps between consecutive step metric lines, so each one
    is a real step and nothing else. ``train_wall / steps_run`` is NOT a step wall and cannot
    substitute: it charges subprocess startup and the checkpoint-upload drain to the numerator,
    while ``steps_run`` is the absolute checkpoint step, so a run resuming at 90 and training to
    100 would divide ten steps of wall time by a hundred.

    The MEDIAN, not the mean: any single step can absorb a checkpoint save or a lazily compiled
    kernel, and one such step would drag a mean far enough to change the verdict. A median needs
    most steps to be slow before it moves.

    None whenever the reading would not mean anything -- no trustworthy profile, no completed step,
    or grading that accounts for the entire step wall (which makes the gpu-bound remainder zero or
    negative, i.e. the step wall and the profile disagree and neither can arbitrate).
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
    from flash.engine.reward_profile import gpu_idle_fraction

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
) -> dict:
    return {
        "backend": "verl",
        "steps": steps_run,
        "epochs": inp["epochs"],
        "retained_prompts": retained_prompts,
        # matches the trl path: without it a resumed run is indistinguishable from a fresh one.
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
        # chalk installs against an in-process trainer.model; verl's model lives in the subprocess, so
        # no kernel can engage on this path. None (not an empty list) matches the sft verl backend.
        "chalk_kernels": None,
        # verl derives generated tokens from the response lengths it actually observed, not from a
        # padded upper bound. the flag is published so a consumer comparing against historical runs
        # (which counted the padded bound) does not read the two as equivalent.
        "gen_tokens_is_upper_bound": False,
        "thinking": _w.THINKING,
        # the console is uploaded only on failure, so a SUCCESSFUL run has no other record that fp8 kv
        # engaged. this is resolved per-card (cc>=8.9, and never for gdn hybrids), so it is a property
        # of the run rather than of the config.
        "vllm_kv_cache_dtype": "fp8" if fp8_kv else None,
        # trl pins vllm's prefill batch explicitly because it hardcodes 4096; the verl path sets no
        # such override, so the engine keeps its own default. None records "not pinned by flash"
        # rather than asserting a number flash never chose.
        "vllm_max_num_batched_tokens": None,
        "max_completion_len": inp["max_completion"],
        "prompts_per_step": inp["prompts_per_step"],
        # one optimizer step consumes exactly this many completions: ulysses shards along the
        # sequence, so dp stays 1 and the global batch is not split across cards.
        "generations_per_step": inp["prompts_per_step"] * inp["group_size"],
        # trl fixes a per-device SEQUENCE count and carries the rest in grad accumulation. verl
        # bounds the backward pass by TOKENS instead (use_dynamic_bsz + ppo_max_token_len_per_gpu),
        # so a micro-batch holds however many sequences fit the budget and varies step to step.
        # there is no constant to report, and reporting a fabricated one would read as comparable.
        "per_device_train_batch_size": None,
        "gradient_accumulation_steps": None,
        "ppo_max_token_len_per_gpu": inp["engine_len"],
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        # the sdk's link_wandb reads notes["wandb_url"]; trl gets it from the parent's live
        # wandb.run, verl from the child marker (see verl_common.render_wandb_link_shim).
        "wandb_url": wandb_url,
        "wandb_id": wandb_id,
        # what this env's grader actually cost, and what share of a step the gpu spent waiting on
        # it. measured, not modelled: the cost model prices every grader at one global average
        # (1.0 s/completion) spanning regex graders to llm judges, and those two sit at opposite
        # ends of the utilization range. None when the profile was skipped or untrustworthy --
        # recording a number with nothing behind it is worse than recording no number.
        "reward_seconds_per_completion": (
            reward_profile.seconds_per_completion
            if reward_profile is not None and reward_profile.trustworthy
            else None
        ),
        "reward_gpu_idle_fraction": _measured_idle_fraction(
            reward_profile,
            completions_per_step=inp["prompts_per_step"] * inp["group_size"],
            step_intervals=step_intervals or [],
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


_ENTROPY_QUANTILE_MARKER = "[flash-verl] top-entropy token masking active"
_STOP_SEQUENCES_MARKER = "[flash-verl] rollout stop strings active"
_STRUCTURED_OUTPUTS_MARKER = "[flash-verl] rollout structured outputs active"
_EXACT_SAVE_STEPS_MARKER = "[flash-verl] exact save steps active"
_IMAGE_PAD_BAN_MARKER = "[flash-verl] image-pad token banned from rollouts"
_KL_REF_ADAPTER_MARKER = "[flash-verl] kl reference anchored to the warm-start adapter"


def render_kl_ref_adapter_shim(warmstart: bool) -> str:
    """return the sitecustomize source that anchors verl's kl reference to the warm-start adapter.

    verl computes reference logprobs on the actor whenever lora is active (``ref_in_actor`` in
    ray_trainer.py is ``lora_rank > 0 or lora_adapter_path is not None``, always true on flash) and
    marks that call ``no_lora_adapter=True``, which engine_workers.py turns into
    ``engine.disable_adapter()`` -- the BARE BASE. for a fresh-start run that is correct. for a
    warm-started run it is not: the kl term would pull the policy back toward the base and undo the
    sft adapter the run was told to continue, which is why the warm-start + kl combination was
    refused until now. trl instead snapshots a frozen reference adapter and evaluates the reference
    under it (rl.py's grpo path); this ports that behavior.

    the snapshot is registered as NON-PERSISTENT BUFFERS rather than as a second peft adapter's
    parameters, which is what keeps it out of every downstream consumer:

    - ``named_parameters()`` never sees it, so fsdp does not flatten it and the optimizer cannot
      train it (a trainable reference would drift with the policy and anchor nothing).
    - ``state_dict()`` never sees it, so it stays out of the saved shards. that matters more than
      it looks: verl's merger does not call ``save_pretrained`` for the adapter, it hand-builds one
      from every state-dict key containing ``lora_`` and derives ``target_modules`` from
      ``key.split(".")[-3]`` (base_model_merger.save_lora_adapter). a second adapter's keys do not
      match its ``.default.weight`` rewrite, so they would resolve to ``lora_A``/``lora_B`` and ship
      a deliverable adapter with bogus target modules.
    - a resumed run reloads the same shards with ``strict=True``; absent keys would fail that load.
      the snapshot is rebuilt from ``lora_adapter_path``, which flash passes on every warm-start
      run including a resume, so the anchor re-forms identically rather than being restored.

    the swap is ``BaseTunerLayer._active_adapter``, not peft's ``set_adapter``: ``set_adapter``
    flips ``requires_grad`` on both adapters, and verl wraps fsdp1 with ``use_orig_params=false``
    where that breaks flat-param uniformity. writing ``_active_adapter`` changes zero flags and
    restores the policy forward bit-exactly.
    """
    if not warmstart:
        return ""
    return f'''
from contextlib import contextmanager as _flash_ref_contextmanager

import torch.nn as _flash_ref_nn
from peft.tuners.tuners_utils import BaseTunerLayer as _flash_ref_tuner_layer
from verl.workers.engine.fsdp import transformer_impl as _flash_ref_impl

_FLASH_REF_ADAPTER = "flash_kl_ref"


def _flash_ref_snapshot(module):
    """freeze a copy of the warm-start adapter under a second, non-trainable adapter name."""
    if _FLASH_REF_ADAPTER in getattr(module, "peft_config", {{}}):
        return module
    module.add_adapter(_FLASH_REF_ADAPTER, module.peft_config[module.active_adapter])
    for name, param in module.named_parameters():
        if ".default." in name:
            twin = module.get_parameter(name.replace(".default.", f".{{_FLASH_REF_ADAPTER}}."))
            twin.data.copy_(param.data)
    # demote the snapshot from parameters to non-persistent buffers. lora_A/lora_B are ModuleDicts
    # keyed by adapter name, so the snapshot's leaves are exactly the ones under our key.
    demoted = 0
    for container in module.modules():
        if isinstance(container, _flash_ref_nn.ModuleDict) and _FLASH_REF_ADAPTER in container:
            leaf = container[_FLASH_REF_ADAPTER]
            for attr, value in list(leaf.named_parameters(recurse=False)):
                frozen = value.detach().clone()
                delattr(leaf, attr)
                leaf.register_buffer(attr, frozen, persistent=False)
                demoted += 1
    if not demoted:
        raise RuntimeError("flash kl reference snapshot found no adapter weights to freeze")
    print({_KL_REF_ADAPTER_MARKER!r} + " " + repr(demoted), flush=True)
    return module


_flash_ref_original_build_lora = _flash_ref_impl.FSDPEngine._build_lora_module


def _flash_ref_build_lora_module(self, module):
    # after the warm-start adapter is loaded, before _build_fsdp_module wraps it.
    return _flash_ref_snapshot(_flash_ref_original_build_lora(self, module))


@_flash_ref_contextmanager
def _flash_ref_use_ref_adapter(module):
    """activate the frozen snapshot without touching any requires_grad flag."""
    layers = [m for m in module.modules() if isinstance(m, _flash_ref_tuner_layer)]
    if not layers:
        raise RuntimeError("flash kl reference: no lora layers on the actor module")
    saved = [layer._active_adapter for layer in layers]
    for layer in layers:
        layer._active_adapter = [_FLASH_REF_ADAPTER]
    try:
        yield
    finally:
        for layer, previous in zip(layers, saved, strict=True):
            layer._active_adapter = previous


def _flash_ref_disable_adapter(self):
    # this shim is only written for a warm start, so the snapshot must exist. falling back to the
    # stock disable_adapter() here would silently anchor the kl term to the base -- the exact
    # defect this patch removes -- and the run would look healthy while training the wrong thing.
    module = self.module
    inner = getattr(module, "_fsdp_wrapped_module", module)
    if _FLASH_REF_ADAPTER not in getattr(inner, "peft_config", {{}}):
        raise RuntimeError(
            "flash kl reference adapter missing: expected " + _FLASH_REF_ADAPTER + " on the actor"
        )
    return _flash_ref_use_ref_adapter(module)


_flash_ref_impl.FSDPEngine._build_lora_module = _flash_ref_build_lora_module
_flash_ref_impl.FSDPEngine.disable_adapter = _flash_ref_disable_adapter
'''


def render_structured_outputs_shim(structured_outputs: dict | None) -> str:
    """return the sitecustomize source that constrains verl's rollout to a guided grammar.

    the sampling half of ``train.structured_outputs``. it rides the same per-sample dict as the stop
    strings, so the mechanism is identical to render_stop_sequences_shim; only the value differs.
    the engine half (``reasoning_parser``, applied when thinking is also on) is a plain hydra
    override and needs no shim -- see _build_verl_overrides.

    the value MUST be wrapped in ``StructuredOutputsParams``. vllm accepts a raw dict here, passes
    ``_verify_args()``, and then stores a plain dict with no ``.json`` attribute -- constraining
    nothing, with no error and no log line. trl avoids this by wrapping in its colocate path
    (generation/vllm_generation.py), which is exactly why flash's trl path passes the spec as a
    plain dict; on verl nothing wraps it, so the shim must.

    the object survives the worker -> server hop: that hop is ``server.generate.remote(...)``, a ray
    actor rpc (cloudpickle), not http/json, so it arrives as the same dataclass it left as.
    """
    if not structured_outputs:
        return ""
    return f'''
from verl.experimental.agent_loop import agent_loop as _flash_so_agent_loop
from vllm.sampling_params import StructuredOutputsParams as _FlashStructuredOutputsParams

_flash_structured_outputs = {structured_outputs!r}


def _flash_patch_structured_outputs():
    """add ``structured_outputs`` to the per-sample sampling params on their way into the loop.

    patched on ``_run_agent_loop`` for the same reason as the stop strings: it receives the
    per-sample dict after verl's validate/greedy overrides, so the constraint also applies to
    validation rollouts -- matching trl, where it lives in generation_kwargs and is not swapped out
    for eval.
    """
    original = _flash_so_agent_loop.AgentLoopWorker._run_agent_loop

    async def _run_agent_loop(self, sampling_params, *args, **kwargs):
        params = dict(sampling_params)
        # build a fresh params object per request: vllm mutates sampling params in place (it
        # resolves the structured-outputs backend on first use and caches it on the instance), so a
        # shared one would leak that resolution across requests.
        params["structured_outputs"] = _FlashStructuredOutputsParams(**_flash_structured_outputs)
        return await original(self, params, *args, **kwargs)

    _flash_so_agent_loop.AgentLoopWorker._run_agent_loop = _run_agent_loop


if not getattr(
    _flash_so_agent_loop.AgentLoopWorker._run_agent_loop, "_flash_so_patched", False
):
    _flash_patch_structured_outputs()
    _flash_so_agent_loop.AgentLoopWorker._run_agent_loop._flash_so_patched = True
    print({_STRUCTURED_OUTPUTS_MARKER!r} + " " + repr(_flash_structured_outputs), flush=True)
'''


def render_exact_save_steps_shim(save_at_steps: tuple[int, ...], total_steps: int) -> str:
    """return the sitecustomize source that suppresses verl's superset checkpoint writes.

    verl only saves when ``global_steps % save_freq == 0``, so it cannot hit an arbitrary set of
    steps. the resolver picks the gcd of the required steps, which makes verl save a SUPERSET and
    the uploader publish deployables at exactly the required ones. correct, but the gcd can be
    tiny -- save_at_steps=(7, 13) gives gcd 1, a full checkpoint written every single step -- and
    each write is a full-state dump of a multi-billion-parameter policy.

    this drops the writes flash never asked for, so only the required steps (and the last step,
    which the run's final publish needs) reach disk. the sft verl backend already does exactly this
    (sft_train.py); this is the same suppression on the ppo driver.

    ``RayPPOTrainer._save_checkpoint`` takes no step argument -- it reads ``self.global_steps`` --
    so the filter reads it off the instance rather than a parameter. returning early is safe
    because the method's only other effect is advancing latest_checkpointed_iteration.txt, and a
    step with no checkpoint on disk must not be advertised as resumable: the uploader gates on that
    marker precisely so it never uploads a half-written or absent directory.
    """
    if not save_at_steps:
        return ""
    return f'''
from verl.trainer.ppo import ray_trainer as _flash_save_ray_trainer

_flash_required_save_steps = frozenset({tuple(sorted(save_at_steps))!r})
_flash_total_steps = {int(total_steps)}

def _flash_patch_exact_save_steps():
    """save only the steps flash asked for, plus the final step.

    the gcd interval makes verl save a superset; every extra write is a full-state dump that is
    never published and is pruned again a few steps later. the last step stays because the run's
    final publish reads the checkpoint verl writes there.
    """
    original = _flash_save_ray_trainer.RayPPOTrainer._save_checkpoint

    def _save_checkpoint(self):
        step = int(self.global_steps)
        if step not in _flash_required_save_steps and step != _flash_total_steps:
            return None
        return original(self)

    _flash_save_ray_trainer.RayPPOTrainer._save_checkpoint = _save_checkpoint

if not getattr(
    _flash_save_ray_trainer.RayPPOTrainer._save_checkpoint, "_flash_save_patched", False
):
    _flash_patch_exact_save_steps()
    _flash_save_ray_trainer.RayPPOTrainer._save_checkpoint._flash_save_patched = True
    print(
        {_EXACT_SAVE_STEPS_MARKER!r} + " " + repr(sorted(_flash_required_save_steps))
        + " final=" + repr(_flash_total_steps),
        flush=True,
    )
'''


def render_stop_sequences_shim(stop_sequences: tuple[str, ...]) -> str:
    """return the sitecustomize source that gives verl's rollout trl's stop-string behavior.

    on the trl backend flash puts ``stop`` into ``generation_kwargs``, which reaches vllm's
    ``SamplingParams`` unchanged. verl builds its sampling params as a literal dict in
    ``AgentLoopWorker.generate_sequences`` (agent_loop.py) with no stop field and no passthrough, so
    the key has to be inserted there. the value then rides the existing dict all the way into
    ``SamplingParams(max_tokens=..., **sampling_params)`` in the vllm server, which accepts it.

    the patch lands on ``_run_agent_loop`` rather than the vllm server because the worker owns the
    dict: patching further down would have to reconstruct which request the params belong to, and
    the tool/multi-turn loops pass the same dict through untouched.

    token-level semantics match trl exactly. vllm truncates ``output_text`` at a stop-string match
    but leaves ``token_ids`` intact, and both backends read ``output.token_ids`` -- so the trained
    tokens are the same on either backend, including the trailing delimiter tokens.
    """
    if not stop_sequences:
        return ""
    return f'''
from verl.experimental.agent_loop import agent_loop as _flash_agent_loop

_flash_stop_sequences = {list(stop_sequences)!r}


def _flash_patch_run_agent_loop():
    """add ``stop`` to the per-sample sampling params on their way into the agent loop.

    ``_run_agent_loop`` receives the fully-built dict for one sample, after verl has applied its
    validation/greedy overrides. patching here rather than at dict construction means the stop
    strings survive those overrides and apply to validation rollouts too, matching trl, where the
    stop list lives in generation_kwargs and is not swapped out for eval.
    """
    original = _flash_agent_loop.AgentLoopWorker._run_agent_loop

    async def _run_agent_loop(self, sampling_params, *args, **kwargs):
        params = dict(sampling_params)
        params["stop"] = list(_flash_stop_sequences)
        return await original(self, params, *args, **kwargs)

    _flash_agent_loop.AgentLoopWorker._run_agent_loop = _run_agent_loop


# patch once. wrapping twice would be harmless here (the key is overwritten, not appended) but the
# guard keeps the behavior obvious and matches the entropy shim.
if not getattr(_flash_agent_loop.AgentLoopWorker._run_agent_loop, "_flash_stop_patched", False):
    _flash_patch_run_agent_loop()
    _flash_agent_loop.AgentLoopWorker._run_agent_loop._flash_stop_patched = True
    print({_STOP_SEQUENCES_MARKER!r} + " " + repr(_flash_stop_sequences), flush=True)
'''


def render_image_pad_ban_shim(image_pad_token_id: int | None) -> str:
    """return the sitecustomize source that stops a multimodal rollout emitting the image-pad token.

    the vision placeholder token is a real vocabulary entry, so an unconstrained sampler can emit
    it inside a *completion*. nothing there expands it back into pixels, so the trained sequence
    then contains a token the model can only have produced by hallucinating an image -- and on the
    next forward pass the processor's image/text alignment counts a placeholder with no image
    behind it. trl bans it through ``generation_kwargs["logit_bias"]`` (rl.py); verl builds its
    sampling params as a literal dict, so the key is inserted the same way the stop-strings shim
    inserts ``stop``.

    unconditional rather than gated on the row: this shim is only written for a multimodal job, and
    a text-only row in such a job still must not invent a placeholder. -100.0 matches trl's bias.
    """
    if image_pad_token_id is None:
        return ""
    return f'''
from verl.experimental.agent_loop import agent_loop as _flash_image_agent_loop

_flash_image_pad_token_id = {int(image_pad_token_id)!r}


def _flash_patch_image_pad_ban():
    original = _flash_image_agent_loop.AgentLoopWorker._run_agent_loop

    async def _run_agent_loop(self, sampling_params, *args, **kwargs):
        params = dict(sampling_params)
        logit_bias = dict(params.get("logit_bias") or {{}})
        logit_bias[_flash_image_pad_token_id] = -100.0
        params["logit_bias"] = logit_bias
        return await original(self, params, *args, **kwargs)

    _flash_image_agent_loop.AgentLoopWorker._run_agent_loop = _run_agent_loop


# patch once, and independently of the stop-strings patch: both wrap the same method, so each
# needs its own marker attribute or the second would be skipped by the first one's flag.
if not getattr(
    _flash_image_agent_loop.AgentLoopWorker._run_agent_loop, "_flash_image_pad_patched", False
):
    _flash_patch_image_pad_ban()
    _flash_image_agent_loop.AgentLoopWorker._run_agent_loop._flash_image_pad_patched = True
    print({_IMAGE_PAD_BAN_MARKER!r} + " " + repr(_flash_image_pad_token_id), flush=True)
'''


def render_per_turn_credit_shim(per_turn_credit: bool) -> str:
    """return the sitecustomize source that gives verl trl's per-turn group-relative credit.

    verl credits a whole episode: ``compute_grpo_outcome_advantage`` centres one scalar per rollout
    against its group and broadcasts it across every response token. trl's per-turn mode centres
    each TURN against the same turn of its group siblings, so a good turn inside a bad episode
    still gets positive advantage.

    this wraps ``compute_advantage`` rather than registering a custom estimator. a registered
    estimator would be the tidier hook, but ``compute_advantage`` forwards ``non_tensor_batch`` to
    exactly one estimator by name (``if adv_estimator in (AdvantageEstimator.GDPO, "gdpo")``), so a
    custom one could never see the spans it needs. wrapping keeps stock grpo as the baseline and
    overwrites only the token axis, which is the same shape as trl's GRPOPerTurnTrainer overriding
    its parent's scalar advantages.

    the fallback is per GROUP, not per row: grpo centres each rollout against its group, so a group
    holding a mix of per-turn and episode credit would compare quantities of different scales. one
    unusable row therefore drops its whole group to episode credit.
    """
    if not per_turn_credit:
        return ""
    return '''
import torch as _flash_pt_torch
from verl.trainer.ppo import ray_trainer as _flash_pt_ray_trainer

_flash_pt_original_compute_advantage = _flash_pt_ray_trainer.compute_advantage
_flash_pt_logged = False


def _flash_pt_rows(non_tensor_batch, batch_size):
    """per-row (spans, turns) or None when this batch carries no usable per-turn metadata."""
    spans_column = non_tensor_batch.get("flash_turn_spans")
    rewards_column = non_tensor_batch.get("flash_turn_rewards")
    if spans_column is None or rewards_column is None:
        return None
    if len(spans_column) != batch_size or len(rewards_column) != batch_size:
        return None
    rows = []
    for spans, turns in zip(spans_column, rewards_column):
        if spans is None or turns is None or len(spans) != len(turns):
            # a row the loop could not align. keep the row so its group can be identified and
            # dropped whole, rather than silently centring the rest against a smaller sample.
            rows.append(None)
            continue
        rows.append(
            (
                tuple((int(start), int(end)) for start, end in spans),
                tuple(float(value) for value in turns),
            )
        )
    return rows


def _flash_pt_per_turn_advantages(rows, index, episode_advantages):
    """centre each turn against the same turn of its group; returns [B, width].

    starts from stock grpo's own output and overwrites only the rows of groups that earned
    per-turn credit. a group that falls back therefore keeps the exact tensor grpo produced rather
    than a reconstruction of it -- there is no scalar to recover, so nothing can drift.
    """
    advantages = episode_advantages.clone()
    groups = {}
    for row_index, uid in enumerate(index):
        groups.setdefault(uid, []).append(row_index)
    for member_indexes in groups.values():
        if any(rows[row_index] is None for row_index in member_indexes):
            continue
        for row_index in member_indexes:
            advantages[row_index] = 0.0
        turn_total = max(len(rows[row_index][1]) for row_index in member_indexes)
        for turn_index in range(turn_total):
            scoring = [
                row_index
                for row_index in member_indexes
                if turn_index < len(rows[row_index][1])
                and rows[row_index][0][turn_index][1] > rows[row_index][0][turn_index][0]
            ]
            if not scoring:
                # every member emitted nothing for this turn; an empty turn carries no signal and
                # must not skew the baseline for the members that did emit one.
                continue
            baseline = sum(rows[row_index][1][turn_index] for row_index in scoring) / len(scoring)
            for row_index in scoring:
                start, end = rows[row_index][0][turn_index]
                advantages[row_index, start:end] = rows[row_index][1][turn_index] - baseline
    return advantages


def _flash_pt_compute_advantage(data, *args, **kwargs):
    global _flash_pt_logged
    data = _flash_pt_original_compute_advantage(data, *args, **kwargs)
    episode = data.batch.get("advantages")
    if episode is None or episode.dim() != 2:
        return data
    batch_size, width = episode.shape
    rows = _flash_pt_rows(data.non_tensor_batch, batch_size)
    if rows is None or all(row is None for row in rows):
        return data
    index = data.non_tensor_batch.get("uid")
    if index is None or len(index) != batch_size:
        return data
    for row in rows:
        if row is None:
            continue
        for start, end in row[0]:
            if not 0 <= start <= end <= width:
                raise ValueError(
                    f"turn span [{start}, {end}) exceeds the response width {width}"
                )
    advantages = _flash_pt_per_turn_advantages(rows, index, episode)
    if not bool(_flash_pt_torch.isfinite(advantages).all()):
        raise ValueError("per-turn advantages must be finite")
    # keep the response mask authoritative: glue tokens sit inside a turn span only when the
    # environment reply was appended mid-turn, and they must never carry gradient.
    response_mask = data.batch.get("response_mask")
    if response_mask is not None:
        advantages = advantages * response_mask.to(dtype=advantages.dtype)
    data.batch["advantages"] = advantages
    # returns feeds the critic, which grpo does not use; stock grpo sets it to the same tensor.
    data.batch["returns"] = advantages
    if not _flash_pt_logged:
        print("[rl-verl] multi-turn per-turn group-relative credit is active", flush=True)
        _flash_pt_logged = True
    return data


_flash_pt_ray_trainer.compute_advantage = _flash_pt_compute_advantage
'''


def render_entropy_quantile_shim(entropy_quantile: float | None) -> str:
    """return the sitecustomize source that adds trl's top-entropy token masking to verl.

    trl's GRPOTrainer keeps only the top ``entropy_quantile`` fraction of response tokens in the
    policy-gradient term (``get_high_entropy_mask``); verl has no such knob. the mask is expressible
    as an extra factor on ``response_mask``, so this patches ``ppo_loss`` rather than registering a
    custom policy loss: ``ppo_loss`` computes per-token entropy but never forwards it to
    ``policy_loss_fn``, so a registered loss could not see the entropy it needs to threshold on.

    the mask applies to the policy-gradient term ONLY, which is what trl does -- it multiplies
    ``per_token_loss`` by the mask and then adds the kl term, so kl and the entropy bonus stay on the
    full response mask. equivalence also needs a mask-independent denominator: flash pins
    ``seq-mean-token-sum-norm``, which divides by ``global_batch_size * loss_scale_factor``, so
    dropping tokens from the numerator does not rescale the remaining ones (the same property that
    makes trl's dr_grpo normalizer ``per_token_loss.size(0) * max_completion_length``).
    """
    if entropy_quantile is None or float(entropy_quantile) >= 1.0:
        return ""
    threshold = 1.0 - float(entropy_quantile)
    return f'''
import threading as _flash_threading

import torch as _flash_torch
import torch.distributed as _flash_dist
from verl.workers.utils import losses as _flash_losses
from verl.workers.utils.padding import no_padding_2_padding as _flash_no_padding_2_padding

_flash_entropy_threshold_q = {threshold!r}
_flash_original_ppo_loss = _flash_losses.ppo_loss
_flash_original_get_policy_loss_fn = _flash_losses.get_policy_loss_fn
# carries this micro-batch's padded entropy from ppo_loss down to the policy loss, which verl
# never passes it. thread-local rather than a module global so the handoff cannot leak across
# concurrent loss calls, and always cleared in a finally.
_flash_entropy_state = _flash_threading.local()


def _flash_high_entropy_mask(entropy, response_mask):
    """trl get_high_entropy_mask: keep tokens at or above the global entropy quantile."""
    local = entropy[response_mask.bool()].float().reshape(-1)
    # the quantile is over the whole global batch, not the local shard, so every rank thresholds
    # identically. trl gathers via accelerate; verl's actor shards over the default process group.
    if _flash_dist.is_available() and _flash_dist.is_initialized() and _flash_dist.get_world_size() > 1:
        sizes = [_flash_torch.zeros(1, dtype=_flash_torch.long, device=local.device)
                 for _ in range(_flash_dist.get_world_size())]
        _flash_dist.all_gather(sizes, _flash_torch.tensor([local.numel()], dtype=_flash_torch.long, device=local.device))
        largest = int(max(int(s.item()) for s in sizes))
        if largest == 0:
            return _flash_torch.zeros_like(entropy, dtype=_flash_torch.bool)
        # pad with a negative sentinel: entropy is non-negative, so it can never collide with a
        # real value and is dropped after the gather (trl uses the same -1e9 trick).
        padded = _flash_torch.full((largest,), -1e9, dtype=_flash_torch.float32, device=local.device)
        padded[: local.numel()] = local
        buckets = [_flash_torch.empty_like(padded) for _ in range(_flash_dist.get_world_size())]
        _flash_dist.all_gather(buckets, padded)
        gathered = _flash_torch.cat(buckets)
        gathered = gathered[gathered != -1e9]
    else:
        gathered = local
    if gathered.numel() == 0:
        return _flash_torch.zeros_like(entropy, dtype=_flash_torch.bool)
    cutoff = _flash_torch.quantile(gathered, _flash_entropy_threshold_q)
    return ((entropy * response_mask.float()) >= cutoff) & response_mask.bool()


def _flash_masked_policy_loss_fn(loss_mode):
    """wrap the resolved policy loss so it sees the entropy-masked response mask.

    masking here rather than in ppo_loss keeps the mask on the policy-gradient term only: ppo_loss
    aggregates the kl and entropy-bonus terms against its own unmasked response_mask, exactly as
    trl adds its kl term after multiplying per_token_loss by the entropy mask.
    """
    inner = _flash_original_get_policy_loss_fn(loss_mode)

    def _masked(*args, **kwargs):
        entropy = getattr(_flash_entropy_state, "entropy", None)
        response_mask = kwargs.get("response_mask", None)
        if entropy is not None and response_mask is not None:
            kwargs["response_mask"] = _flash_high_entropy_mask(entropy, response_mask)
        return inner(*args, **kwargs)

    return _masked


def _flash_entropy_masked_ppo_loss(config, model_output, data, dp_group=None):
    entropy = model_output.get("entropy", None)
    if entropy is None:
        raise RuntimeError(
            "train.entropy_quantile needs per-token entropy, but verl produced none; "
            "actor.calculate_entropy must be true."
        )
    # ppo_loss re-derives this internally; convert here too so the mask is padded to the same
    # (bsz, response_len) shape the policy loss receives its response_mask in.
    _flash_entropy_state.entropy = _flash_no_padding_2_padding(entropy, data)
    try:
        return _flash_original_ppo_loss(config, model_output, data, dp_group)
    finally:
        _flash_entropy_state.entropy = None


# patch once. python imports sitecustomize a single time per interpreter, so this should not come
# up -- but wrapping an already-wrapped loss would mask the top quantile OF the top quantile and
# train on a fraction of the requested tokens, with nothing in the logs to show for it.
if not getattr(_flash_original_ppo_loss, "_flash_entropy_masked", False):
    _flash_entropy_masked_ppo_loss._flash_entropy_masked = True
    _flash_losses.get_policy_loss_fn = _flash_masked_policy_loss_fn
    _flash_losses.ppo_loss = _flash_entropy_masked_ppo_loss
    print({_ENTROPY_QUANTILE_MARKER!r} + " quantile={entropy_quantile:g}", flush=True)
'''


def render_reward_module(url_env: str = "FLASH_VERL_REWARD_URL") -> str:
    """source for the verl custom reward module.

    runs INSIDE the verl interpreter, so it must be self-contained (stdlib only, no flash import).
    it forwards (index, solution_str) to the flash reward bridge and returns the float score.
    """
    return (
        '"""flash reward bridge shim (generated). posts each completion to the flash worker."""\n'
        "import json\n"
        "import os\n"
        "import urllib.error\n"
        "import urllib.request\n"
        "\n"
        f"_URL = os.environ.get({url_env!r}, '')\n"
        "\n"
        "\n"
        "def compute_score(data_source, solution_str, ground_truth, extra_info=None):\n"
        "    idx = (extra_info or {}).get('index')\n"
        "    if idx is None:\n"
        "        raise RuntimeError('flash reward bridge received no example index')\n"
        "    if not _URL:\n"
        "        raise RuntimeError('flash reward bridge url is not configured')\n"
        "    if isinstance(idx, bool) or getattr(getattr(idx, 'dtype', None), 'kind', None) == 'b':\n"
        "        raise RuntimeError('flash reward bridge received an invalid example index: %r' % idx)\n"
        "    try:\n"
        "        exact_idx = int(idx)\n"
        "    except (TypeError, ValueError, OverflowError) as exc:\n"
        "        raise RuntimeError('flash reward bridge received an invalid example index: %r' % idx) from exc\n"
        "    if exact_idx != idx:\n"
        "        raise RuntimeError('flash reward bridge received an invalid example index: %r' % idx)\n"
        "    idx = exact_idx\n"
        "    body = json.dumps({'index': idx, 'solution_str': solution_str or ''}).encode()\n"
        "    req = urllib.request.Request(\n"
        "        _URL.rstrip('/') + '/score', data=body, headers={'Content-Type': 'application/json'}\n"
        "    )\n"
        "    try:\n"
        "        with urllib.request.urlopen(req, timeout=120) as r:\n"
        "            payload = json.loads(r.read().decode())\n"
        "            return float(payload['score'])\n"
        "    except urllib.error.URLError as exc:\n"
        "        raise RuntimeError('flash reward bridge request failed: %s' % exc) from exc\n"
        "    except Exception as exc:\n"
        "        raise RuntimeError('flash reward bridge returned an invalid response: %s' % exc) from exc\n"
    )


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
    """score one single-turn text completion exactly as the trl reward path does.

    mirrors run_rl.reward_fn's single-turn text branch: graded text -> env.scores_breakdown
    (preferred) or env.reward, minus the optional thinking-length penalty. env scoring errors
    are swallowed to 0.0 so a bad completion never kills the run.

    ``raise_on_error`` re-raises instead, for callers that must tell a real 0.0 score apart from a
    failed grading. training never sets it; the latency profiler does, because a swallowed error
    turns a grader that is failing on every call into a fast, confident, wrong reading.

    ``breakdowns`` collects this completion's per-name reward components for the ``reward_metrics``
    heartbeat field. appended to only when the env actually exposes ``scores_breakdown`` -- an env
    with a plain scalar ``reward`` has no components, and appending an empty dict for it would put a
    real denominator under no numerators and publish a flat 0 for a run that simply has no named
    metrics. a failed grading appends ``None``, which the mean counts as a zero for every name the
    other completions did report, exactly as trl does; the caller owns the list's lock and bound.
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
    if think_penalty > 0 and thinking:
        r -= think_penalty * _w.think_token_count(
            solution_str, tok, prompt_opened_thinking=prompt_opened_thinking
        )
    return float(r)


# the total startup delay this hook is allowed to add, covering reference extraction AND timing.
# both call user code, so one shared ceiling is the only number that means anything to a caller.
_PROFILE_BUDGET_S = 30.0


def _log_reward_profile(env, score_one, rollout_examples: list, completions_per_step: int):
    """Measure this env's real grading latency once, before training starts, and report it.

    Returns the ``RewardProfile`` when one was measured, else None. The reading is RETURNED as
    well as printed because a log line cannot be read by the cost model or by a scheduler: this
    env's latency is what separates a compute-bound run from a latency-bound one, and that is a
    placement input, not just an observability line.

    Single-turn grading is serial, so it is pure gpu-idle wall time -- at the default 64x8 shape a
    1s grader idles the gpu for ~80% of every step. The cost model has to guess that number from
    one global average spanning regex graders to llm judges; this prints what it actually is for
    THIS env, so a mispriced run is visible in the log instead of only in the bill.

    Profiles against each example's own reference completion rather than blank text: an empty
    string does not exercise a grader, so a blank-text profile measures the early-return.

    Bounded by ``_PROFILE_BUDGET_S`` end to end. Both halves call into user code -- gathering a
    reference completion runs the env's own ``sft_completion``, grading runs its scorer -- so they
    share one deadline. Bounding only the half that gets timed would still let this hook stall
    startup forever on the half that does not.

    SKIPPED for an env that declares ``reward_thread_safe = False``. That flag means the scorer
    keeps mutable or thread-bound state (flash/envs/adapter.py documents it), so grading extra
    completions before training starts could advance a counter, consume a quota or warm a cache,
    and change the rewards training then receives. A measurement must not be able to move the
    thing it measures, and the profiler also needs a worker thread to bound a hung call, which
    such an env must not be given. The cost of skipping is one unmeasured latency in the log.

    Note the flag is read here as covering ``sft_completion`` too, though the adapter documents it
    against ``reward()``. This hook is the only caller of ``sft_completion`` in a grpo run, so state
    it advances starves nothing downstream; what the flag is standing in for is the narrower case of
    a hook that lazily builds a thread-affine handle shared with the scorer. One declared signal for
    "do not touch me off-thread" covers both, and no env currently distinguishes them.

    Advisory only, and never fatal -- it must not be able to break a run it is only measuring.
    """
    try:
        from flash.engine.reward_profile import call_bounded, profile_reward_latency
        from flash.multimodal import assistant_completion_text

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
                # the shared deadline is spent either way, so there is no timing budget left and
                # this returns regardless. report how far it got: "no reference completion could be
                # gathered" is false once an earlier example has already yielded one, and a skip
                # reason that misstates what happened sends a reader after the wrong env hook.
                #
                # count only the USABLE ones, by the same test the profiler itself applies. a failed
                # call still appends its empty placeholder to keep example indices aligned, and a
                # succeeded-but-blank one is dropped downstream by ``text.strip()``; counting either
                # would claim references were gathered that nothing could be profiled from -- exactly
                # as misleading as the message this replaced, in the opposite direction.
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
            per_step_s = profile.seconds_per_completion * completions_per_step
            print(
                f"[rl-verl] serial reward grading costs ~{per_step_s:.0f}s per step "
                f"({completions_per_step} completions x {profile.seconds_per_completion:.3f}s), "
                "all of it gpu-idle.",
                flush=True,
            )
        return profile
    except Exception as exc:
        print(f"[rl-verl] reward profiling skipped: {exc}", flush=True)
        return None


# --------------------------------------------------------------------------------------------
# reward rpc bridge: verl subprocess -> flash live env.
# --------------------------------------------------------------------------------------------
class MultiTurnBridge:
    """parent-side episode state for the child's multi-turn agent loop.

    the child owns tokens and the engine; this owns the flash env. one session per in-flight
    episode, keyed by the id the child mints. the env's own rollout state is the source of truth
    for turn budget, doneness, and the terminal episode that gets scored -- exactly the state the
    trl driver accumulates, so both backends score the same object.

    ``on_episode_scored(prompt, transcript, reward)`` receives each scored episode so the caller can
    surface it as a rollout sample. multi-turn has no per-completion breakdown to report -- the env
    scores a whole episode through ``rollout_rewards_many``, which returns a scalar -- so only the
    sample side of the reward observability pair applies here, exactly as on the trl path.
    """

    def __init__(
        self,
        env,
        examples: list[dict],
        *,
        max_turns: int,
        per_turn_credit: bool = False,
        on_episode_scored: Callable[[object, object, float], None] | None = None,
    ) -> None:
        self._env = env
        self._examples = examples
        self._max_turns = int(max_turns)
        self._per_turn_credit = bool(per_turn_credit)
        self._on_episode_scored = on_episode_scored
        # the flash env is not required to be thread-safe, and verl runs many rollouts at once.
        # every env touch below happens under this lock, matching the single-turn path's contract.
        self._lock = threading.Lock()
        self._sessions: dict[str, dict] = {}

    def routes(self) -> dict:
        return {
            "/multiturn/start": self.start,
            "/multiturn/step": self.step,
            "/multiturn/score": self.score,
            "/multiturn/close": self.close,
        }

    def _session(self, payload: dict) -> dict:
        session_id = str(payload["session_id"])
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"unknown multi-turn session {session_id}")
        return session

    def start(self, payload: dict) -> dict:
        index = int(payload["index"])
        if index < 0 or index >= len(self._examples):
            raise IndexError(
                f"multi-turn example index {index} is outside [0, {len(self._examples)})"
            )
        session_id = str(payload["session_id"])
        example = self._examples[index]
        with self._lock:
            if session_id in self._sessions:
                raise KeyError(f"duplicate multi-turn session {session_id}")
            state = self._env.new_rollout_state(example)
            self._sessions[session_id] = {"example": example, "state": state}
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
                # is the failure being diagnosed (codex[bot]).
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

    def score(self, payload: dict) -> dict:
        turn_count = int(payload["turn_count"])
        with self._lock:
            session = self._session(payload)
            state = session["state"]
            reward = score_rollouts(
                self._env,
                [
                    RolloutScoreRequest(
                        example=session["example"],
                        state=state,
                        turn_count=turn_count,
                    )
                ],
            )[0]
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
MULTI_TURN_CHILD_MODULES = (
    ("multiturn_glue.py", "flash_multiturn_glue.py"),
    ("grpo_multiturn.py", "flash_grpo_multiturn.py"),
    ("grpo_plugin.py", "flash_grpo_plugin.py"),
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
    """the child env vars the multi-turn agent loop reads, and nothing else.

    the loop runs in the verl interpreter and cannot import flash, so every decision it makes about
    turn budget, halting and glue rendering has to arrive as a string here. the parent is the only
    process that can resolve them: the eos set needs the model's generation_config and the turn
    limit needs the live env.
    """
    return {
        # verl imports this at the end of `verl/__init__` (import_external_libs), which is what
        # registers the loop under the name the agent-loop override selects. without it the
        # override names a loop that was never registered and the child dies at rollout build.
        "VERL_USE_EXTERNAL_MODULES": "flash_grpo_plugin",
        "FLASH_VERL_MULTITURN_URL": reward_url,
        "FLASH_VERL_MAX_TURNS": str(int(inp["max_turns"])),
        "FLASH_VERL_MAX_MODEL_LEN": str(int(inp["engine_len"])),
        "FLASH_VERL_STOP_SEQUENCES": json.dumps(list(inp["stop_sequences"])),
        # sorted so the child sees a stable set regardless of frozenset iteration order.
        "FLASH_VERL_EOS_TOKEN_IDS": json.dumps(sorted(inp["eos_token_ids"])),
        "FLASH_VERL_THINKING": "1" if thinking else "0",
    }


def start_reward_server(score_by_index, *, example_count: int, multi_turn_bridge=None):
    """start a localhost http reward server. score_by_index(index, solution_str) -> float.

    returns (server, base_url). the server runs in a daemon thread; call server.shutdown() when
    done. single-turn scoring lives at ``<base_url>/score``; when ``multi_turn_bridge`` is given,
    its four episode routes are served alongside it from the same thread pool.
    """
    # serialize scoring so the flash env sees sequential calls, matching the trl reward path's
    # contract; verl's reward manager may otherwise call the reward with several workers at once.
    score_lock = threading.Lock()

    def _score_route(payload: dict) -> dict:
        index = int(payload["index"])
        if index < 0 or index >= example_count:
            raise IndexError(f"reward example index {index} is outside [0, {example_count})")
        with score_lock:
            return {"score": float(score_by_index(index, payload.get("solution_str", "")))}

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

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    return server, f"http://127.0.0.1:{port}"


# --------------------------------------------------------------------------------------------
# verl interpreter + checkpoint export.
# --------------------------------------------------------------------------------------------
def _resolve_verl_loggers(python_bin: str) -> str:
    """verl's ``trainer.logger`` list, as the comma-joined form this module's override emits."""
    return ",".join(resolve_verl_loggers(python_bin))


def _export_peft_adapter(
    ckpt_actor_dir: str,
    out_adapter_dir: str,
    *,
    base_model_id: str,
    python_bin: str,
) -> None:
    """turn verl's saved lora checkpoint into a flash-servable peft adapter dir.

    verl saves fsdp-sharded checkpoints under ``<local_dir>/global_step_N/actor`` (model/optim
    shards + a ``huggingface/`` config+tokenizer subfolder). ``verl.model_merger merge`` writes a
    standard peft adapter (adapter_config.json + adapter_model.safetensors) to a ``lora_adapter/``
    subfolder of its target; we copy just that adapter into flash's adapter dir (the co-produced
    merged full model is discarded -- flash serves the lora on the immutable base).

    verified against verl 0.8 on an h100 (r2): merger emits ``<target>/lora_adapter/{adapter_config
    .json,adapter_model.safetensors}`` with ``base_model_name_or_path: null``.
    """
    os.makedirs(out_adapter_dir, exist_ok=True)
    merge_out = out_adapter_dir.rstrip("/") + "_merge"
    shutil.rmtree(merge_out, ignore_errors=True)
    merge_env = dict(os.environ)
    # the base config/tokenizer are already cached; keep the merger off hf's rate-limited api.
    merge_env["HF_HUB_OFFLINE"] = "1"
    merge_env["TRANSFORMERS_OFFLINE"] = "1"
    merge_env["HF_HUB_DISABLE_XET"] = "1"
    subprocess.run(
        [
            python_bin,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            ckpt_actor_dir,
            "--target_dir",
            merge_out,
        ],
        check=True,
        env=merge_env,
    )
    lora_dir = os.path.join(merge_out, "lora_adapter")
    if not os.path.exists(os.path.join(lora_dir, "adapter_config.json")):
        raise RuntimeError(
            f"verl model_merger did not produce a peft adapter at {lora_dir} (no adapter_config.json); "
            "the merger output layout must be adjusted for this verl version."
        )
    for name in os.listdir(lora_dir):
        shutil.copy2(os.path.join(lora_dir, name), os.path.join(out_adapter_dir, name))
    shutil.rmtree(merge_out, ignore_errors=True)


class _VerlResumeUploader:
    """stream each completed verl checkpoint to hf so a preempted grpo run can resume from it.

    without exact save steps grpo publishes only a final deployable adapter (both backends), so this
    uploads the resume state alone. with save_at_steps set it also exports and publishes a servable
    adapter at each required step, which is what makes those steps deployable rather than merely
    resumable. verl writes global_step_N under default_local_dir and only then advances
    latest_checkpointed_iteration.txt, so gating on that marker never uploads a half-written dir.
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
        # whatever this run resumed from is already durable as RESUME state, so re-uploading it
        # would waste the upload slot on state hf already holds. that is tracked in uploaded_steps
        # below; processed_steps is deliberately NOT seeded for a required resume step, because its
        # DEPLOYABLE can still be missing -- a previous worker resume-uploads a checkpoint while
        # withholding its adapter behind the gradient gate -- and _pending must yield that step once
        # so its adapter gets staged from the checkpoint this run restored.
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
        # gates DEPLOYABLE publication (not resume upload) on the run having produced a real
        # gradient. these publishes land while training is still running, so without the gate a
        # degenerate-reward run makes untrained adapters durable and servable minutes before
        # _check_grpo_had_a_gradient fails the run -- the guard would reject the final adapter while
        # the per-step ones stayed published. resume state is deliberately still uploaded: it is
        # internal retry scaffolding, not something a customer can serve.
        self.had_gradient = had_gradient
        # populated by credit_durable_required_steps() before start(): crediting a required step
        # needs an hf lookup, which does not belong in a constructor.
        self.published_steps: set[int] = set()
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def credit_durable_required_steps(self, resume_step: int) -> None:
        """Credit required saves a pre-resume worker already published.

        A resumed run never re-saves a step it trained past, so without this every required step
        below the resume point would be reported missing and fail an otherwise-successful retry.
        Crediting is gated on the adapter being verified on hf rather than inferred from the step
        counter: a preempted worker can advance past a required step without its deployable ever
        landing, and that step must stay uncredited so completeness still catches it.
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
        self._thread.join(timeout=600)
        if self._thread.is_alive():
            raise RuntimeError("verl resume uploader did not stop")

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
        """whether gradient evidence has appeared yet, so a deployable may be published.

        no callback means no gate (the resume-only configuration and the tests that predate it).
        a raising callback is treated as CLOSED: this decides whether an artifact becomes durable
        and servable, so an unreadable signal must not be read as permission.
        """
        if self.had_gradient is None:
            return True
        try:
            return bool(self.had_gradient())
        except Exception:
            return False

    def _completed_step(self) -> int:
        tracker = os.path.join(self.local_dir, "latest_checkpointed_iteration.txt")
        try:
            with open(tracker) as file:
                return int(file.read().strip())
        except (FileNotFoundError, OSError, ValueError):
            return 0

    def _pending(self, completed_step: int) -> list[tuple[int, str]]:
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

    def _stage_deployable(self, step: int, checkpoint_dir: str) -> str:
        """export a required step's checkpoint into a servable adapter under export_root.

        staging is deliberately NOT gated on gradient evidence, while publication is. verl owns
        ``checkpoint_dir`` and prunes it once max_actor_ckpt_to_keep newer saves exist, so an export
        deferred until the gate opens can find its source already deleted -- with four or more
        required steps ahead of the first varying-reward group, the earliest ones would then be
        unpublishable and fail an otherwise valid run. export_root is flash's own workdir, so the
        staged adapter outlives verl's retention. it is not servable until _publish_staged uploads
        it, which is what keeps the gate's guarantee intact.
        """
        actor_dir = os.path.join(checkpoint_dir, "actor")
        adapter_dir = os.path.join(self.export_root, f"step-{step}")
        shutil.rmtree(adapter_dir, ignore_errors=True)
        os.makedirs(adapter_dir, exist_ok=True)
        _export_peft_adapter(
            actor_dir, adapter_dir, base_model_id=self.model_id, python_bin=self.python_bin
        )
        # the served adapter needs its preprocessor alongside it, exactly as the final publish does:
        # the processor on a multimodal job (tokenizer + image preprocessor), else the tokenizer.
        self.preprocessor.save_pretrained(adapter_dir)
        _stamp_adapter_dir_provenance(adapter_dir, self.model_id, self.model_revision)
        _w.write_base_model_provenance(adapter_dir, self.model_id, self.model_revision)
        return adapter_dir

    def _publish_staged(self, step: int, adapter_dir: str) -> None:
        """make an already-staged adapter durable and servable."""
        _w.publish_deployable_checkpoint(adapter_dir, step, required=True, _provenance_ready=True)
        self.published_steps.add(step)

    def _publish_ready(self) -> None:
        """publish every staged step the gate now permits, oldest first.

        driven off staged_steps rather than the pending sweep, so a step whose verl checkpoint has
        since been pruned is still publishable: everything the upload needs already lives under
        export_root. the gate is read once per call, so a spread recorded during a slow export is
        honoured on the same pass that produced it.
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
                for step, path in self._pending(self._completed_step()):
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
                    # the resume upload is NOT gated on gradient evidence: it is internal retry
                    # scaffolding rather than a servable artifact, and with exact save_at_steps these
                    # are often the only on-disk checkpoints, so skipping it would leave a run
                    # preempted before the first nonzero spread nothing to resume from. marked before
                    # the attempt, so one permanently failing upload cannot spin every 0.5s -- which
                    # is also the pre-existing non-fatal semantics for this upload.
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
                if stopping and not self._pending(self._completed_step()):
                    return
                time.sleep(0.5)
        except BaseException as error:
            self._error = error


def _restore_verl_resume(local_dir: str) -> int:
    """stage this run's streamed resume checkpoint into local_dir; return the step it resumes at.

    the resume artifact is keyed on the run prefix, not the job type, so the control plane hands
    grpo the same ``checkpoint-N`` layout it hands sft. verl finds it via
    latest_checkpointed_iteration.txt under trainer.default_local_dir once resume_mode=auto.
    returns 0 when there is nothing to resume, which is the ordinary fresh-run path.
    """
    resume = _w.hf_resume_checkpoint()
    if not resume:
        return 0
    match = re.fullmatch(r"checkpoint-(\d+)", os.path.basename(resume))
    if match is None:
        raise RuntimeError(f"invalid GRPO resume checkpoint path {resume!r}")
    step = int(match.group(1))
    target = os.path.join(local_dir, f"global_step_{step}")
    shutil.copytree(resume, target, dirs_exist_ok=True)
    with open(os.path.join(local_dir, "latest_checkpointed_iteration.txt"), "w") as file:
        file.write(str(step))
    return step


def _check_grpo_had_a_gradient(
    reward_history: list[float],
    adv_spread_history: list[float],
    *,
    resumed: bool = False,
    already_complete: bool = False,
) -> None:
    """raise unless the run's rewards actually produced a nonzero policy gradient.

    every other completion check a grpo run passes -- terminal state, a written checkpoint, an
    exported adapter, a populated reward history -- is also passed by a run that trained on nothing.
    grpo mean-centres its advantage within each group (A_i = r_i - mean(r_group)), so an environment
    whose reward does not discriminate between completions gives every sample advantage exactly 0,
    hence pg_loss exactly 0 and an adapter identical to its initialization. the spread of the
    advantages is the only one of these series that can tell the two apart: the reward mean cannot
    (a constant reward has a perfectly healthy mean) and neither can pg_loss, which is near zero at
    step 1 for a genuinely training run too, because the importance ratio is still exactly 1.

    ``resumed`` disables the spread verdict, because these series only cover the steps THIS worker
    observed. a run resuming at step 9 of 10 sees one step, and if that step's group happens to tie
    the history is all-zero even though the restored weights already carry nine steps of productive
    updates -- rejecting it would discard a correctly trained policy. the evidence needed to judge a
    resumed run lives in the steps a previous worker ran, which is not recoverable from this
    worker's stdout, so the honest move is to abstain rather than guess. the degenerate-environment
    case this guard exists for is unaffected: such a run fails on its first, unresumed attempt.

    ``already_complete`` is the narrower case where the resume checkpoint ALREADY sits at the target
    step, so verl's loop body never executes: ``current_epoch = global_steps // len(dataloader)``
    equals total_epochs, the epoch range is empty, and the child exits 0 having emitted no metric
    lines at all. both histories are then empty for a fully-trained policy, which the empty-history
    check below would report as a reward-bridge wiring regression. abstain instead -- the steps that
    trained this policy were observed by an earlier worker, exactly as for ``resumed``.
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


def _latest_global_step_dir(local_dir: str) -> tuple[str, int]:
    """return (actor_dir, step) for the highest global_step_N checkpoint verl wrote."""
    best_step, best = -1, ""
    if os.path.isdir(local_dir):
        for name in os.listdir(local_dir):
            m = re.fullmatch(r"global_step_(\d+)", name)
            if m and int(m.group(1)) > best_step:
                best_step = int(m.group(1))
                best = os.path.join(local_dir, name, "actor")
    if best_step < 0:
        raise RuntimeError(f"no global_step_N checkpoint found under {local_dir}")
    return best, best_step


def _stamp_adapter_dir_provenance(
    adapter_dir: str, model_id: str, model_revision: str = ""
) -> None:
    """stamp the saved adapter's immutable base identity into adapter_config.json.

    dir-based analogue of _w.stamp_adapter_provenance (which needs an in-memory peft model). same
    validation + fields, applied to the json verl produced.
    """
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    current_base = str(cfg.get("base_model_name_or_path", "") or "").strip()
    if current_base and current_base != model_id:
        raise RuntimeError(
            f"adapter base model {current_base!r} does not match validated target {model_id!r}"
        )
    current_rev = str(cfg.get("revision", "") or "").strip()
    if current_rev and model_revision and current_rev != model_revision:
        raise RuntimeError("adapter base revision does not match the validated target commit")
    cfg["base_model_name_or_path"] = model_id
    cfg["revision"] = model_revision or None
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)


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
    # fail loud on grpo features the verl backend does not yet honor, so a migration never silently
    # trains without a requested behavior.
    # structured_outputs: two halves, mirroring the trl path. the constraint itself rides the
    # per-sample sampling params via a shim (render_structured_outputs_shim); the reasoning_parser
    # that defers the grammar past </think> is an engine arg, applied as a plain hydra override in
    # _build_verl_overrides. parse raises on a corrupt payload rather than training unconstrained.
    structured_outputs = (
        parse_structured_outputs(getattr(_t, "structured_outputs", "") if _t else "") or None
    )
    if structured_outputs:
        print(
            f"[rl-verl] structured outputs: every rollout constrained to "
            f"{describe_structured_outputs(structured_outputs)}"
        )
    # stop_sequences: on the trl backend these ride generation_kwargs["stop"] into vllm's
    # SamplingParams. verl builds its sampling params without a stop field, so a sitecustomize shim
    # inserts the key; see render_stop_sequences_shim. note grpo_mask_truncated_completions below
    # gates itself OFF when stop_sequences is set, on either backend.
    stop_sequences = tuple(str(s) for s in (getattr(_t, "stop_sequences", ()) or ())) if _t else ()
    # entropy_quantile drives trl GRPOTrainer's internal top-entropy token masking
    # (top_entropy_quantile). verl has no such knob, so a sitecustomize shim adds it; see
    # render_entropy_quantile_shim. a value >= 1.0 (or unset) masks nothing and needs no shim.
    _eq = getattr(_t, "entropy_quantile", None) if _t else None
    entropy_quantile = float(_eq) if _eq is not None and float(_eq) < 1.0 else None
    # credit_assignment: per_turn only changes the objective when there is more than one assistant
    # turn to credit -- trl reaches GRPOPerTurnTrainer solely via use_rollout_func, which requires
    # is_multi_turn (rl.py select_grpo_trainer). single-turn envs cannot get there, and trl says so
    # itself while accepting the key ("per-turn is equivalent to per-episode for single-turn
    # environments", rl.py). match that: log it, do not reject.
    #
    # on multi-turn it centres each turn against the same turn of its group siblings instead of
    # broadcasting one episode advantage across the transcript. the agent loop records a token span
    # per turn, the bridge returns the env's per-turn vector alongside the episode reward, and a
    # shim rewrites the advantage tensor after stock grpo has run; see
    # render_per_turn_credit_shim.
    credit_assignment = getattr(_t, "credit_assignment", DEFAULT_CREDIT_ASSIGNMENT) if _t else None
    per_turn_credit = bool(
        credit_assignment
        and credit_assignment != DEFAULT_CREDIT_ASSIGNMENT
        and multi_turn
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
    # advantage_clip is recorded but not applied (the trl path centers advantages without a value
    # clip; verl's grpo advantage is likewise group-centered). log it for parity, do not apply.
    if float(gcfg.get("advantage_clip") or 0.0) > 0:
        print(
            f"[rl-verl] advantage_clip={gcfg['advantage_clip']} recorded; verl centers grpo "
            "advantages (no value clip), matching the trl path",
            flush=True,
        )
    mask_truncated_completions = _w.grpo_mask_truncated_completions(_t)
    learning_rate = float(
        _t.learning_rate if _t and _t.learning_rate is not None else rl.learning_rate
    )
    # warm-start forbids lora_rank, so a set init_from_adapter already raised above; read rank/alpha
    # from the job spec (falling back to the recipe) exactly like the trl path's lora config.
    lora_rank = int(_t.lora_rank) if (_t and _t.lora_rank) else int(RECIPE.lora.rank)
    lora_alpha = int(_t.lora_alpha) if (_t and _t.lora_alpha) else int(RECIPE.lora.alpha)
    # warm-start: continue the sft adapter in place (verl lora_adapter_path). uses the SOURCE
    # adapter's rank/alpha (flash forbids a child lora_rank on warm-start).
    warmstart_adapter = ""
    if _t and getattr(_t, "init_from_adapter", ""):
        from flash.engine.worker.adapter import _download_adapter

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

    from flash.multimodal import (
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
        validate_multimodal_training(model_id, "grpo", multi_turn=multi_turn)
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
        # verl RAISES on an over-budget multimodal prompt instead of truncating it (agent_loop.py:
        # "truncating multimodal token sequences corrupts vision/audio feature alignment"), so this
        # pre-filter is what keeps a long prompt from killing the run mid-rollout rather than just
        # being dropped.
        #
        # every row goes through the PROCESSOR, including the text-only rows of a mixed job: the
        # verl child tokenizes the whole dataset through the multimodal path, so a row measured
        # with the bare tokenizer here would be measured against a different expansion there. an
        # image expands to hundreds of placeholder tokens, so a tokenizer-only count on an image
        # row is not merely imprecise, it is off by most of the prompt.
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

    # single-turn: one completion IS the response, so the response tensor is max_completion wide and
    # prompt + response exactly fills the engine.
    #
    # multi-turn: the whole EPISODE is the response -- every assistant turn plus every environment
    # glue span concatenated. verl right-pads response_ids to data.max_response_length and drops
    # whatever is longer (_pad_token_ids), so a max_completion-wide tensor would silently cut a
    # transcript mid-turn and train on the fragment. the tightest width that can hold any episode
    # this dataset can produce is the engine budget minus the SHORTEST admitted prompt: the child
    # stops generating once the prefix reaches max_model_len, so no episode can exceed that.
    # max_completion stays the PER-TURN cap, which is exactly what the trl driver does
    # (per_turn_max_tokens=max_completion, episode budget=engine_len-prompt_len-slack).
    max_response_len = (
        max(max_completion, vllm_max_len - min(int(p["prompt_len"]) for p in prompts))
        if multi_turn
        else max_completion
    )

    prompts_per_step = _w.resolve_grpo_prompts_per_step(prompts_per_step, len(prompts))
    prompt_opened_thinking = bool(_w.THINKING) and _w.prompt_opens_thinking(prompts[0]["rendered"])

    # optimizer-update horizon, honoring [train].max_steps exactly like the trl path.
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
    # retention has to outlive the export. verl prunes a checkpoint only once the NEXT one finishes
    # writing, so keeping 1 leaves the uploader one save interval to run model_merger before its
    # source directory is deleted. with exact saves the suppression shim drops every write that is
    # not required, so that interval is the gap between two REQUIRED steps rather than the gcd -- but
    # keep a small history anyway: consecutive required steps (e.g. 7 and 8) still leave a short
    # window, and a slow export must not lose a step the customer asked for.
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
        # verl's default preserves the update horizon and on-policy baseline; unlike trl reuse, each
        # update gets a fresh rollout batch.
        "ppo_epochs": 1,
        "seed": int(backend_seed(_w.SEED)),
    }


def run_rl_verl():
    """grpo training on verl, output-compatible with run_rl. see module docstring for scope."""
    t_start = time.time()
    _w.heartbeat("rl_start", gpu=gpu_diagnostics())
    wait_for_gpu(
        _w.JOB_SPEC.gpu.type if _w.JOB_SPEC else None,
        gpu_type=_w.JOB_SPEC.gpu.type if _w.JOB_SPEC else "",
    )
    setup_perf_backends()

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
    model_path_for_verl = inp["model_id"]
    if inp["model_revision"]:
        download_seconds = _w.prefetch_model(inp["model_id"], revision=inp["model_revision"])
        # verl resolves model.path offline against the HF cache; a bare repo id would pick the
        # cached "main" ref, not the pin. hand verl the pinned revision's snapshot dir instead.
        from huggingface_hub import snapshot_download as _snap

        from flash.engine.worker.hf import _shared_weight_cache_dir

        # prefetch_model lands pinned weights on the shared volume when attached; resolve from the
        # same cache_dir it used, else fall back to the ephemeral default cache.
        _shared = _shared_weight_cache_dir()
        try:
            model_path_for_verl = _snap(
                inp["model_id"],
                revision=inp["model_revision"],
                cache_dir=_shared,
                local_files_only=True,
            )
        except Exception:
            model_path_for_verl = _snap(
                inp["model_id"], revision=inp["model_revision"], local_files_only=True
            )
    else:
        download_seconds = _w.prefetch_model(inp["model_id"])

    # stable int index -> rollout example, exactly as the trl path (reward maps back via this).
    ds_rows, rollout_examples = _w.build_grpo_prompt_dataset(prompts)
    message_prompts = [p["prompt"] for p in prompts]
    indices = [int(r["example_idx"]) for r in ds_rows]
    # ground_truth is a verl-schema placeholder only; the reward bridge scores by example_idx
    # against the live env and never reads it.
    ground_truths = [
        str(ex.get("answer", "") or "") if isinstance(ex, dict) else "" for ex in rollout_examples
    ]

    workdir = f"/tmp/rl_verl_seed{_w.SEED}"
    os.makedirs(workdir, exist_ok=True)
    local_dir = os.path.join(workdir, "ckpt")
    # a retry reuses the pod workdir; stale global_step_N dirs from a prior attempt would satisfy
    # _latest_global_step_dir and publish an old policy as if this attempt trained it.
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

    # runtime patches for the verl interpreter. only written when something needs patching, so the
    # default path puts nothing on the child's import path. a stale shim from a prior attempt would
    # otherwise keep patching this one, so the file is removed when no patch is wanted.
    shim_dir = os.path.join(workdir, "shim")
    os.makedirs(shim_dir, exist_ok=True)
    shim_py = os.path.join(shim_dir, "sitecustomize.py")
    # one sitecustomize holds every patch: python imports it once, so a second file would never be
    # loaded. each renderer returns "" when its feature is off, so the default path writes nothing.
    shim_source = "".join(
        part
        for part in (
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
    if shim_source:
        with open(shim_py, "w") as f:
            f.write(shim_source)
    elif os.path.exists(shim_py):
        os.remove(shim_py)

    # multi-turn: copy the child-side agent loop next to the shim so the verl interpreter can
    # import it (see copy_multi_turn_child_modules for why it is a copy and not an import).
    if inp["multi_turn"]:
        copy_multi_turn_child_modules(shim_dir)

    # reward bridge: verl (out of process) -> flash live env, identical to trl scoring. the buffer
    # carries what trl's reward TrainerCallback carries -- recent rollouts for the per-step log dump
    # and `sampled_completions`, plus per-name components for `reward_metrics` (#607).
    # the generation size lets the buffer close each generation on the scoring thread that finishes
    # it, rather than when the child's `step:N` line reaches this process: those are the same
    # completions but not the same instant, and the next generation scores in between. verl is run
    # with test_freq=-1 and val_before_train=False, so every completion reaching the bridge is a
    # training rollout and the count is exact.
    observability = RewardObservabilityBuffer(
        generation_size=int(inp["prompts_per_step"]) * int(inp["group_size"]),
    )
    # filled from the child's marker line; stays empty when wandb is off (see render_wandb_link_shim).
    wandb_link: dict[str, str | None] = {}

    def _score(index: int, solution_str: str) -> float:
        ex = rollout_examples[int(index)]
        # graded BEFORE the buffer is touched: grading calls user code and can block on i/o for
        # seconds while verl scores many rollouts at once, and RewardObservabilityBuffer.record
        # takes a lock every grading in the run would then serialize behind.
        breakdowns: list[dict[str, float] | None] = []
        score = score_single_turn(
            env,
            solution_str,
            ex,
            tok=tok,
            thinking=bool(_w.THINKING),
            prompt_opened_thinking=inp["prompt_opened_thinking"],
            think_penalty=inp["think_penalty"],
            breakdowns=breakdowns,
        )
        observability.record(message_prompts[int(index)], solution_str, score, breakdowns)
        return score

    def _score_for_profile(index: int, solution_str: str) -> float:
        """the same grading path as _score, minus training's own bookkeeping.

        deliberately NOT _score: that buffers the rollout, so profiling would seed the
        per-step completion dump with gradings that never came from a rollout. it also propagates
        scoring errors instead of scoring them 0.0, so a grader failing on every call reports as a
        failure rather than as a fast latency.
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
            max_turns=int(inp["max_turns"]),
            per_turn_credit=bool(inp["per_turn_credit"]),
            on_episode_scored=observability.record,
        )
        if inp["multi_turn"]
        else None
    )
    server, reward_url = start_reward_server(
        _score,
        example_count=len(rollout_examples),
        multi_turn_bridge=multi_turn_bridge,
    )
    # bound before the try so the finally can always ask whether it was started.
    resume_uploader: _VerlResumeUploader | None = None
    # verl trains out-of-process, so torch's in-process allocator counter never sees the trainer.
    # nvidia-smi is the only reading that covers the child, and it must be sampled while the child
    # runs; stopping it in the finally keeps the reading even when the run crashes or is cancelled.
    gpu_sampler = _NvidiaSmiPeakSampler().start()
    device_peak_gpu_gb: float | None = None
    try:
        python_bin = resolve_verl_python(
            workdir, install_wandb=bool(os.environ.get("WANDB_API_KEY"))
        )
        # masking truncated completions is a fork-only rollout field. FLASH_VERL_PYTHON can point at a
        # stock verl, and hydra would compose the unknown key only to abort in dataclass conversion.
        # fail here with the cause instead, and never silently train on truncated completions.
        if inp["mask_truncated_completions"] and not verl_supports_rollout_field(
            python_bin, "mask_truncated_completions"
        ):
            raise RuntimeError(
                f"grpo requested mask_truncated_completions but the verl at {python_bin} does not "
                "support it. that verl predates the freesolo fork; point FLASH_VERL_PYTHON at an "
                f"interpreter with '{VERL_REQUIREMENT}' installed, or set it EMPTY under "
                '[worker_env] as FLASH_VERL_PYTHON = "" to provision one.'
            )
        expected_steps = int(inp["steps"])
        # verl logs from its own interpreter; gate wandb on that env (see _resolve_verl_loggers).
        loggers = _resolve_verl_loggers(python_bin)
        _spec = _w.JOB_SPEC
        project_name = (_spec.wandb.project if _spec and _spec.wandb else None) or "flash"
        experiment_name = _w.wandb_run_name()
        # fp8 kv cache on ada/hopper+ (cc>=8.9), exactly like the trl colocate path — but NOT for
        # hybrid linear-attention (GDN) models: vllm's fp8-kv wake path (init_fp8_kv_scales) assumes a
        # plain kv tensor and crashes on the hybrid cache ('list' has no zero_) under verl sleep/wake.
        try:
            import torch as _torch_cc

            from flash.engine.worker.packing import model_is_gdn_hybrid

            _cc_ok = bool(
                _torch_cc.cuda.is_available() and _torch_cc.cuda.get_device_capability() >= (8, 9)
            )
            fp8_kv = _cc_ok and not model_is_gdn_hybrid(
                inp["model_id"], revision=inp["model_revision"]
            )
        except Exception:  # no cuda / probe failure -> conservative bf16 kv
            fp8_kv = False
        # blackwell needs both rollout attention backends pinned; vllm 0.19.1's own defaults pick
        # flash-attn, which is PTX-unreliable on sm120 (silent empty rollouts) and routes the ViT
        # into an unimportable CUTE kernel on sm100/sm120. no-op off blackwell.
        attention_backend, mm_encoder_attn_backend = resolve_blackwell_attention_backends(
            python_bin
        )
        cfg = _build_verl_training_cfg(
            inp,
            train_files=train_pq,
            val_files=val_pq,
            model_id=model_path_for_verl,
            thinking=bool(_w.THINKING),
            loggers=loggers,
            fp8_kv=fp8_kv,
            attention_backend=attention_backend,
            mm_encoder_attn_backend=mm_encoder_attn_backend,
            reward_path=reward_py,
            local_dir=local_dir,
            project_name=project_name,
            experiment_name=experiment_name,
            n_gpus=gpu_count_of(_w.JOB_SPEC),
        )
        overrides = build_verl_overrides(cfg)

        setup_seconds = time.time() - t_start
        _w.heartbeat("rl_train_start", setup_seconds=setup_seconds, gpu=gpu_diagnostics())
        _w.heartbeat("rl_step", step=0, initial=True)
        t_train = time.time()
        # grpo's advantage is within-group and mean-centred (A_i = r_i - mean(r_group)), so a group
        # whose rewards are all equal yields advantage exactly 0 for every sample, hence a zero
        # gradient. the spread of the advantages is what says whether the optimizer got a signal;
        # neither the reward mean nor pg_loss can, so collect max/min per step and check below.
        # declared here, ahead of the uploader, because the uploader's publication gate closes over
        # it -- the metric loop that fills it starts further down.
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

        env_for_verl = dict(os.environ)
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
        if shim_source or inp["multi_turn"]:
            # python imports sitecustomize automatically at startup, so the shim patches verl before
            # main_ppo runs. prepend rather than replace: an inherited PYTHONPATH may carry the verl
            # install itself, and ray workers inherit this env so every actor gets the same patch.
            # multi-turn needs the same entry for its copied-in agent loop modules.
            env_for_verl["PYTHONPATH"] = os.pathsep.join(
                item for item in (shim_dir, os.environ.get("PYTHONPATH", "")) if item
            )
        step_re = re.compile(r"step:\s*(\d+)")
        reward_history: list[float] = []
        resp_len_history: list[float] = []
        loss_curve: list[float] = []
        last_dump_step = [-1]
        # when each step metric line arrived. verl's LocalLogger emits exactly one such line per
        # optimizer step, so the gaps between them are whole steps and nothing else -- the only
        # honest denominator for the reward idle fraction. train_wall/steps_run is not: it charges
        # subprocess startup and the checkpoint-upload drain to the numerator, and steps_run is the
        # ABSOLUTE checkpoint step while the wall is session-only, so a resumed run divides this
        # session's seconds by every step the checkpoint ever took.
        step_line_times: list[float] = []
        # per-step backlog for `flash runs log -f`, rebuilt from verl's own step lines because its
        # trainer runs out of process and cannot host trl's callback. read by the liveness thread
        # below, so mutate it in place (append_step_metrics) rather than rebinding.
        metrics_last: list[dict] = []
        sent_first_metrics = False

        def _reward_observability() -> dict:
            """the `reward_metrics` / `sampled_completions` fields for one heartbeat emission.

            trl publishes both from a TrainerCallback on the trainer's own thread. verl's trainer is
            out of process, so this is called from the liveness thread and from the stdout loop
            instead -- reading the buffer the reward bridge fills on its server threads.

            read-only: the generation boundary below owns the drain, so a 30s liveness tick landing
            mid-generation republishes the last complete reading instead of publishing whichever
            completions happened to be graded by then.
            """
            return observability.heartbeat_fields()

        with liveness_heartbeat(
            "rl_step",
            progress=_progress,
            fields=lambda: {"metrics_last": list(metrics_last), **_reward_observability()},
            progress_step=True,
        ):
            proc = subprocess.Popen(
                [python_bin, "-m", "verl.trainer.main_ppo", *overrides],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env_for_verl,
                start_new_session=True,
            )
            try:
                for line in proc.stdout:
                    print(f"[verl] {line}", end="", flush=True)
                    link = parse_wandb_link(line)
                    if link is not None:
                        wandb_link.update(link)
                    m = step_re.search(line)
                    if m:
                        step_box[0] = int(m.group(1))
                        # dump one sample completion per new step to the flash log (matches trl #607).
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
                        # a run constant rather than a verl metric, so it is stamped here the way
                        # trl's callback reads it off args.max_completion_length.
                        step_metrics["max_completion_tokens"] = inp["max_completion"]
                        append_step_metrics(
                            metrics_last, step_metrics, limit=GRPO_METRIC_HISTORY_LIMIT
                        )
                        # the worker's error path reads this global, so a run that dies mid-training
                        # still reports the steps it did complete (worker/__init__.py:_err_metrics).
                        LATEST_GRPO_METRICS_LAST[:] = metrics_last
                        # the rl_train_start ping arms the 900s rl_step throttle, and the liveness
                        # daemon never forces, so the first backlog would stay invisible for 15
                        # minutes. force it through the way trl does (heartbeat.py
                        # force_first_samples), and keep retrying until one commits: the daemon may
                        # claim this step first, which is fine because its payload carries the same
                        # backlog. only the FIRST is forced, so the hf commit cap stays protected.
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
                rc = proc.wait()
            except BaseException:
                # the stream loop died (upload error, cancel, oom in the parent): a still-running
                # verl child would keep burning the gpu unattended — kill its whole process group.
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    os.killpg(os.getpgid(proc.pid), 15)
                with contextlib.suppress(Exception):
                    proc.wait(timeout=10)
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
        server.shutdown()

    # collect verl's lora checkpoint -> flash-servable peft adapter, then reuse flash finalize.
    out_dir = f"/tmp/rl_seed{_w.SEED}"
    adapter_dir = f"{out_dir}/adapter"
    shutil.rmtree(adapter_dir, ignore_errors=True)
    os.makedirs(adapter_dir, exist_ok=True)
    train_wall = time.time() - t_train
    # the zero-gradient verdict already ran inside the try above, ahead of required-save
    # completeness, so that a withheld deployable reports the reward cause rather than the
    # publication symptom.
    actor_dir, steps_run = _latest_global_step_dir(local_dir)
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
        _export_peft_adapter(
            actor_dir, adapter_dir, base_model_id=inp["model_id"], python_bin=python_bin
        )
        preprocessor.save_pretrained(adapter_dir)
        _stamp_adapter_dir_provenance(adapter_dir, inp["model_id"], inp["model_revision"])
        _w.write_base_model_provenance(adapter_dir, inp["model_id"], inp["model_revision"])
        _w.hf_upload_folder(adapter_dir, "adapter", required=True)
        # preserve the final checkpoint only when exact save steps are not configured, matching the
        # trl path: with save_at_steps set the customer asked for those steps and nothing else.
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
        ),
    )
