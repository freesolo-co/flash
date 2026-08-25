"""Resolve every grpo input the verl launch needs, before any subprocess exists.

`_resolve_grpo_inputs` is the front half of `run_rl_train`: it reads the active env and job spec,
rejects the shapes verl cannot run, and returns the fully-resolved dict the dataset writer, the
hydra override builder, and the child launcher all read from. It is pure config work -- no verl,
no gpu, no subprocess -- which is why so many tests call it directly.

Split out of `flash.engine.worker.train.entry.rl_train` to keep that module under the file-size limit.
"""

from __future__ import annotations

import json
import os
import random
from functools import reduce
from math import gcd

import flash.engine.worker.io.hf as _worker_hf
import flash.engine.worker.model.adapter as _worker_adapter
import flash.engine.worker.model.decoding as _worker_decoding
import flash.engine.worker.perf as _worker_perf
import flash.engine.worker.runtime.state as _worker_state
import flash.engine.worker.train.rl.launch.config as _worker_config
from flash.adapters.targets import LoraTargeting, resolve_lora_targeting
from flash.content.structured_outputs import (
    describe_structured_outputs,
    parse_structured_outputs,
)
from flash.content.thinking import messages_for_chat_template
from flash.core.spec import DEFAULT_CREDIT_ASSIGNMENT, gpu_count_of
from flash.engine.plan.recipe import RECIPE
from flash.engine.plan.steps import (
    on_policy_steps,
    resolve_update_horizon,
    rl_data_parallel_cards,
    validate_save_steps,
)
from flash.engine.support.verl_policy import _resolve_fsdp_generation
from flash.engine.worker.io.heartbeat import liveness_heartbeat
from flash.engine.worker.runtime.rng import backend_seed, seed_training_rngs
from flash.engine.worker.train.core.child.glue import validate_glue_template
from flash.engine.worker.train.entry.backend_common import (
    clamp_engine_len,
    model_max_position_embeddings,
)
from flash.engine.worker.train.opd.orchestration.gkd import generation_eos_from_cached_config
from flash.engine.worker.train.rl.launch.verl_config import (
    _processor_expanded_prompt,
    _verl_epochs_for_horizon,
)


def _validate_grpo_environment(env):
    if getattr(env, "is_tool_env", False):
        raise RuntimeError(
            "verl grpo does not support openai function-calling tool environments; this env "
            "reports is_tool_env"
        )
    multi_turn = bool(getattr(env, "multi_turn", False))
    if not multi_turn:
        return False

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
    return True


def _resolve_grpo_options(train_spec, rl, multi_turn):
    # fail loudly on unsupported verl features. structured-output constraints travel in per-sample
    # params via render_structured_outputs_shim; the reasoning parser is a hydra override. malformed
    # payloads raise instead of training unconstrained.
    structured_outputs = (
        parse_structured_outputs(
            getattr(train_spec, "structured_outputs", "") if train_spec else ""
        )
        or None
    )
    if structured_outputs:
        print(
            f"[rl-verl] structured outputs: every rollout constrained to "
            f"{describe_structured_outputs(structured_outputs)}"
        )
    # stop_sequences: on the retired trl backend these ride generation_kwargs["stop"] into vllm's
    # SamplingParams. verl builds its sampling params without a stop field, so the deferred child
    # plugin inserts the key. note grpo_mask_truncated_completions below gates itself OFF when
    # stop_sequences is set, on either backend.
    stop_sequences = (
        tuple(str(s) for s in (getattr(train_spec, "stop_sequences", ()) or ()))
        if train_spec
        else ()
    )
    # per-turn credit is equivalent to per-episode credit for a single-turn env, so accept it there.
    # on multi-turn, centre each turn against sibling turns and rewrite the stock advantage tensor;
    # the agent loop records token spans and the bridge returns per-turn rewards through the deferred
    # child plugin.
    credit_assignment = (
        getattr(train_spec, "credit_assignment", DEFAULT_CREDIT_ASSIGNMENT) if train_spec else None
    )
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
    gcfg = _worker_config.grpo_overrides()
    prompts_per_step = int(
        train_spec.prompts_per_step
        if train_spec and train_spec.prompts_per_step is not None
        else rl.prompts_per_step
    )
    configured_group_size = gcfg.get("group_size")
    group_size = int(rl.group_size if configured_group_size is None else configured_group_size)
    gcfg_temp = gcfg.get("temperature")
    temperature = float(gcfg_temp if gcfg_temp is not None else rl.sampling_temperature)
    think_penalty = float(gcfg.get("thinking_length_penalty_coef") or 0.0)
    # flash defaults kl_penalty_coef to 0 (dr-grpo, no kl term); honor it rather than forcing a kl.
    kl_coef = float(gcfg.get("kl_penalty_coef") or 0.0)
    learning_rate = float(
        train_spec.learning_rate
        if train_spec and train_spec.learning_rate is not None
        else rl.learning_rate
    )
    # warm-start forbids lora_rank, so a set init_from_adapter already raised above; read rank/alpha
    # from the job spec (falling back to the recipe) exactly like the retired trl path's lora config.
    lora_rank = (
        int(train_spec.lora_rank) if train_spec and train_spec.lora_rank else int(RECIPE.lora.rank)
    )
    lora_alpha = (
        int(train_spec.lora_alpha)
        if train_spec and train_spec.lora_alpha
        else int(RECIPE.lora.alpha)
    )
    return {
        "structured_outputs": structured_outputs,
        "stop_sequences": stop_sequences,
        "per_turn_credit": per_turn_credit,
        "gcfg": gcfg,
        "prompts_per_step": prompts_per_step,
        "group_size": group_size,
        "temperature": temperature,
        "think_penalty": think_penalty,
        "kl_coef": kl_coef,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
    }


def _resolve_warmstart_config(
    train_spec,
    model_id,
    adapter_path,
    lora_rank,
    lora_alpha,
    targeting: LoraTargeting,
):
    with open(os.path.join(adapter_path, "adapter_config.json")) as f:
        source_config = json.load(f)
    _worker_adapter.validate_warmstart_adapter(source_config, model_id, adapter_path, targeting)
    # a patterned adapter trains some modules at higher rank than the base `r`; verl allocates
    # one uniform rank, so it must cover the MAXIMUM prepared rank or the load truncates.
    ranks = [int(source_config.get("r", lora_rank))]
    ranks += [int(v) for v in (source_config.get("rank_pattern") or {}).values()]
    lora_rank = max(ranks)
    alphas = [int(source_config.get("lora_alpha", lora_alpha))]
    alphas += [int(v) for v in (source_config.get("alpha_pattern") or {}).values()]
    lora_alpha = max(alphas)
    print(
        f"[rl-verl] warm-start: continuing source adapter (r={lora_rank}, alpha={lora_alpha}) "
        f"from {train_spec.init_from_adapter}",
        flush=True,
    )
    return lora_rank, lora_alpha


def _load_training_records(env, train_spec):
    train = env.dataset()
    max_examples = getattr(train_spec, "max_examples", None) if train_spec else None
    if max_examples:
        train = train[: int(max_examples)]
    rng = random.Random(_worker_state.SEED)
    rng.shuffle(train)
    return train, [env.prompt_messages(ex) for ex in train]


def _grpo_is_multimodal(env, train, message_prompts):
    from flash.content.multimodal import record_has_images

    return bool(getattr(env, "image_observations", False)) or any(
        record_has_images(ex, messages) for ex, messages in zip(train, message_prompts, strict=True)
    )


def _resolve_sequence_lengths(model_id, model_revision, train_spec, rl, gcfg, tok, multi_turn):
    from flash.engine.plan.vram import grpo_completion_len, grpo_rollout_seq_len

    thinking = bool(_worker_state.THINKING)
    max_tokens = gcfg.get("max_tokens")
    max_completion = grpo_completion_len(max_tokens, thinking)
    train_ctx = (
        train_spec.max_context_tokens if (train_spec and train_spec.max_context_tokens) else 0
    )
    requested_len = grpo_rollout_seq_len(train_ctx, max_tokens, thinking)
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
        validate_glue_template(tok, thinking=bool(_worker_state.THINKING))
    return {
        "max_completion": max_completion,
        "vllm_max_len": vllm_max_len,
        "prompt_budget": prompt_budget,
    }


def _build_grpo_prompts(
    train, message_prompts, multimodal, processor, tok, package_root, prompt_budget
):
    from flash.content.multimodal import normalize_prompt_images

    prompts = []
    if multimodal:
        # verl raises rather than truncating over-budget multimodal prompts because truncation corrupts
        # feature alignment, so pre-filter them. every mixed-job row uses the same processor as the
        # child; tokenizer-only counts miss image expansion by hundreds of tokens.
        for ex, messages in zip(train, message_prompts, strict=True):
            normalized = normalize_prompt_images(ex, messages, package_root)
            template_messages = messages_for_chat_template(normalized.messages)
            expanded, rendered = _processor_expanded_prompt(
                processor,
                template_messages,
                tuple(normalized.descriptors),
                package_root,
                enable_thinking=bool(_worker_state.THINKING),
            )
            if 0 < len(expanded) <= prompt_budget:
                prompts.append(
                    {
                        "prompt": template_messages,
                        # use the same canonical message shape the processor and child authenticate.
                        # this also carries top-level record images that were absent from `messages`.
                        "env_prompt": template_messages,
                        "images": list(normalized.descriptors),
                        "rendered": rendered,
                        "prompt_ids": list(expanded),
                        "example": ex,
                        "prompt_len": len(expanded),
                    }
                )
    else:
        for ex, messages in zip(train, message_prompts, strict=True):
            template_messages = messages_for_chat_template(messages)
            rendered = tok.apply_chat_template(
                template_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=_worker_state.THINKING,
                preserve_thinking=False,
            )
            prompt_ids = list(tok(rendered, add_special_tokens=False).input_ids)
            prompt_len = len(prompt_ids)
            if 0 < prompt_len <= prompt_budget:
                prompts.append(
                    {
                        "prompt": template_messages,
                        # reuse the messages that built this prompt. calling start_episode again can randomize
                        # a different episode, making the bridge score different state from the generated
                        # transcript.
                        "env_prompt": template_messages,
                        "rendered": rendered,
                        "prompt_ids": prompt_ids,
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
    return prompts


def _resolve_grpo_schedule(train_spec, rl, prompts, prompts_per_step, lengths, multi_turn):
    max_completion = lengths["max_completion"]
    vllm_max_len = lengths["vllm_max_len"]
    # single-turn responses are max_completion wide. multi-turn responses contain the whole episode,
    # and verl silently right-truncates beyond data.max_response_length. size to engine budget minus
    # the shortest admitted prompt so no possible episode is cut mid-turn; max_completion remains the
    # per-turn cap.
    max_response_len = (
        max(max_completion, vllm_max_len - min(int(p["prompt_len"]) for p in prompts))
        if multi_turn
        else max_completion
    )
    prompts_per_step = _worker_config.resolve_grpo_prompts_per_step(prompts_per_step, len(prompts))
    # optimizer-update horizon, honoring [train].max_steps exactly.
    epochs = (
        int(train_spec.epochs)
        if train_spec and train_spec.epochs is not None
        else int(rl.num_epochs)
    )
    derived_steps = on_policy_steps(
        epochs=epochs, prompt_count=len(prompts), prompts_per_step=prompts_per_step
    )
    steps = resolve_update_horizon(
        derived_steps, getattr(train_spec, "max_steps", None) if train_spec else None
    )
    # verl stops at both total_epochs and total_training_steps. give its hardcoded drop_last
    # dataloader enough epoch capacity to serve the horizon, but preserve the configured epoch count
    # for user-facing metadata. total_training_steps prevents the capacity headroom from over-training.
    verl_total_epochs = _verl_epochs_for_horizon(
        epochs=epochs,
        prompt_count=len(prompts),
        prompts_per_step=prompts_per_step,
        steps=int(steps),
    )
    save_every = int(train_spec.save_every) if (train_spec and train_spec.save_every) else 20
    # periodic saves land on multiples of save_freq. clamp the default interval to the horizon so a
    # short derived run still lands a periodic save on its final step. exact save steps can be
    # arbitrary, so their gcd produces a superset that includes every requested step; the uploader
    # publishes only those deployables. pinned verl additionally saves the last step.
    save_at_steps = tuple(int(step) for step in (getattr(train_spec, "save_at_steps", ()) or ()))
    validate_save_steps(save_at_steps, int(steps))
    save_freq = reduce(gcd, save_at_steps) if save_at_steps else max(1, min(save_every, int(steps)))
    # keep checkpoints long enough for asynchronous export. verl prunes only after the next save, but
    # consecutive required steps can leave a short merger window; a small history prevents a slow
    # export from losing a requested step.
    ckpt_to_keep = 3 if save_at_steps else 1
    return {
        "max_response_len": max_response_len,
        "prompts_per_step": prompts_per_step,
        "epochs": epochs,
        "verl_total_epochs": verl_total_epochs,
        "steps": int(steps),
        "save_freq": save_freq,
        "save_at_steps": save_at_steps,
        "ckpt_to_keep": ckpt_to_keep,
    }


def _assemble_grpo_inputs(
    *,
    env,
    tok,
    processor,
    multimodal,
    multi_turn,
    package_root,
    image_pad_token_id,
    model_id,
    model_revision,
    prompts,
    options,
    lengths,
    schedule,
    prompt_opened_thinking,
    warmstart_adapter,
    fsdp_generation,
    pinned,
):
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
        "prompts_per_step": schedule["prompts_per_step"],
        "group_size": options["group_size"],
        # the ranks this attempt launches verl at, resolved ONCE here because two phases ask: the
        # child config (`_configure_rl_child`) and the resume probe, which discards a checkpoint whose
        # shard count disagrees with the width it is handed. deriving it twice from the raw knobs let
        # them drift -- resume compared the RENTED count while the child launched the clamped one, so
        # a checkpoint written at the executed width read as the wrong topology and every retry
        # restarted from step 0. opd binds one value for both phases; this is grpo's.
        "dp_cards": rl_data_parallel_cards(
            gpu_count_of(_worker_state.JOB_SPEC),
            int(schedule["prompts_per_step"]) * int(options["group_size"]),
        ),
        "mask_truncated_completions": options["mask_truncated_completions"],
        "temperature": options["temperature"],
        "top_p": float(RECIPE.rl.sampling_top_p),
        "think_penalty": options["think_penalty"],
        "kl_coef": options["kl_coef"],
        **pinned,
        "stop_sequences": options["stop_sequences"],
        "structured_outputs": options["structured_outputs"],
        "max_completion": lengths["max_completion"],
        # the response TENSOR's width. equals max_completion on a single-turn job; on multi-turn it
        # holds the whole episode (see the derivation above).
        "max_response_len": schedule["max_response_len"],
        "max_prompt_len": lengths["prompt_budget"],
        # the engine's full sequence length (prompt + completion), already clamped to the model's
        # own limit. sizes vllm's kv cache and the training token budget, and prompt_budget above is
        # derived from this same value so all three lengths agree.
        "engine_len": lengths["vllm_max_len"],
        "prompt_opened_thinking": prompt_opened_thinking,
        "lr": options["learning_rate"],
        "lora_rank": options["lora_rank"],
        "lora_alpha": options["lora_alpha"],
        "warmstart_adapter": warmstart_adapter,
        "fsdp_generation": fsdp_generation,
        "epochs": schedule["epochs"],
        "verl_total_epochs": schedule["verl_total_epochs"],
        "steps": schedule["steps"],
        "save_freq": schedule["save_freq"],
        "save_at_steps": schedule["save_at_steps"],
        "ckpt_to_keep": schedule["ckpt_to_keep"],
        # verl's default preserves the update horizon and on-policy baseline; with no generation reuse, each
        # update gets a fresh rollout batch.
        "ppo_epochs": 1,
        "seed": int(backend_seed(_worker_state.SEED)),
    }


def _resolve_grpo_inputs():
    """reproduce run_rl's front-half config + dataset prep for text, multimodal, and multi-turn."""
    env = _worker_state.require_active_env()
    multi_turn = _validate_grpo_environment(env)
    seed_training_rngs(_worker_state.SEED)
    model_id = _worker_state.JOB_SPEC.model if _worker_state.JOB_SPEC else RECIPE.hf_model_id
    model_revision = (
        getattr(_worker_state.JOB_SPEC, "model_revision", "") if _worker_state.JOB_SPEC else ""
    )
    rl = RECIPE.rl
    _t = _worker_state.JOB_SPEC.train if _worker_state.JOB_SPEC else None
    options = _resolve_grpo_options(_t, rl, multi_turn)

    # entropy_quantile keeps the loss to the top-entropy tokens of each completion. verl has no
    # such knob, so a sitecustomize shim adds it; see render_entropy_quantile_shim. a value >= 1.0
    # (or unset) masks nothing and needs no shim.
    configured_quantile = getattr(_t, "entropy_quantile", None) if _t else None
    entropy_quantile = (
        float(configured_quantile)
        if configured_quantile is not None and float(configured_quantile) < 1.0
        else None
    )

    mask_truncated_completions = _worker_config.grpo_mask_truncated_completions(_t)
    options["mask_truncated_completions"] = mask_truncated_completions
    warmstart_adapter = ""
    if _t and getattr(_t, "init_from_adapter", ""):
        from flash.engine.worker.model.adapter import _download_adapter

        # warm-start: continue the sft adapter in place (verl lora_adapter_path). uses the SOURCE
        # adapter's rank/alpha (flash forbids a child lora_rank on warm-start).
        # a multi-GB adapter pull emits nothing of its own, so it must run under a liveness wrap or
        # the provider judges the silence as a stall.
        with liveness_heartbeat("rl_adapter_loading"):
            warmstart_adapter = _download_adapter(_t.init_from_adapter)
        if not warmstart_adapter:
            raise RuntimeError(
                "warm-start source adapter could not be downloaded; refusing to start from the base."
            )

    train, message_prompts = _load_training_records(env, _t)
    multimodal = _grpo_is_multimodal(env, train, message_prompts)
    targeting = resolve_lora_targeting(model_id, algorithm="grpo", multimodal=multimodal)
    if warmstart_adapter:
        options["lora_rank"], options["lora_alpha"] = _resolve_warmstart_config(
            _t,
            model_id,
            warmstart_adapter,
            options["lora_rank"],
            options["lora_alpha"],
            targeting,
        )
    package_root = getattr(env, "package_root", None)
    processor = None
    image_pad_token_id = None
    if multimodal:
        from flash.content.multimodal import (
            resolve_image_pad_token_id,
            validate_multimodal_training,
        )

        # the model must actually support image training; this raises for a text-only checkpoint
        # rather than letting the processor silently drop the pixels.
        validate_multimodal_training(
            model_id,
            "grpo",
            getattr(_t, "teacher_model", None),
        )
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, **_worker_hf.model_revision_kwargs(model_revision)
        )
        tok = processor.tokenizer
        image_pad_token_id = resolve_image_pad_token_id(processor, tok)
    else:
        tok = _worker_hf.load_tokenizer(model_id, revision=model_revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    lengths = _resolve_sequence_lengths(
        model_id, model_revision, _t, rl, options["gcfg"], tok, multi_turn
    )
    prompts = _build_grpo_prompts(
        train,
        message_prompts,
        multimodal,
        processor,
        tok,
        package_root,
        lengths["prompt_budget"],
    )
    prompt_opened_thinking = bool(
        _worker_state.THINKING
    ) and _worker_decoding.prompt_opens_thinking(prompts[0]["rendered"])
    # hand the same derived flag to the multi-turn grading path. the env cannot derive it itself --
    # it has no tokenizer and never sees a rendered prompt -- and the template opens the block in
    # EVERY assistant generation prompt (the glue tokenizer renders the next turn's header the same
    # way), so one run-level value covers every turn, exactly as it does single-turn.
    if hasattr(env, "prompt_opens_thinking"):
        env.prompt_opens_thinking = prompt_opened_thinking

    per_turn_credit = options["per_turn_credit"]
    schedule = _resolve_grpo_schedule(
        _t, rl, prompts, options["prompts_per_step"], lengths, multi_turn
    )
    pinned = {
        "entropy_quantile": entropy_quantile,
        # verl always checkpoints (enable_gradient_checkpointing=True) and always asks for
        # non-reentrant recompute, which the MoE router and GDN chunk-scan die on. resolved here so
        # the child shim can put the flag back to what the model needs.
        "reentrant_checkpointing": bool(_worker_perf.grpo_use_reentrant(model_id)),
        "per_turn_credit": per_turn_credit,
    }
    return _assemble_grpo_inputs(
        env=env,
        tok=tok,
        processor=processor,
        multimodal=multimodal,
        multi_turn=multi_turn,
        package_root=package_root,
        image_pad_token_id=image_pad_token_id,
        model_id=model_id,
        model_revision=model_revision,
        prompts=prompts,
        options=options,
        lengths=lengths,
        schedule=schedule,
        prompt_opened_thinking=prompt_opened_thinking,
        warmstart_adapter=warmstart_adapter,
        fsdp_generation=_resolve_fsdp_generation("grpo", targeting.target_parameters),
        pinned=pinned,
    )
