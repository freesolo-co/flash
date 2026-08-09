"""Resolve every grpo input the verl launch needs, before any subprocess exists.

`_resolve_grpo_inputs` is the front half of `run_rl_train`: it reads the active env and job spec,
rejects the shapes verl cannot run, and returns the fully-resolved dict the dataset writer, the
hydra override builder, and the child launcher all read from. It is pure config work -- no verl,
no gpu, no subprocess -- which is why so many tests call it directly.

Split out of `flash.engine.worker.rl_train` to keep that module under the file-size limit.
"""

from __future__ import annotations

import json
import os
import random
from functools import reduce
from math import gcd

from flash.content.structured_outputs import (
    describe_structured_outputs,
    parse_structured_outputs,
)
from flash.core.spec import DEFAULT_CREDIT_ASSIGNMENT
from flash.engine.plan.recipe import RECIPE
from flash.engine.plan.steps import (
    on_policy_steps,
    resolve_update_horizon,
    validate_save_steps,
)
from flash.engine.worker.backend_common import clamp_engine_len
from flash.engine.worker.io.heartbeat import liveness_heartbeat
from flash.engine.worker.runtime.pkg_proxy import W as _w
from flash.engine.worker.runtime.rng import backend_seed
from flash.engine.worker.train.core.child.glue import validate_glue_template
from flash.engine.worker.train.rl.verl_config import (
    _processor_expanded_prompt,
    _verl_epochs_for_horizon,
)


def _rl_train():
    """The orchestrator module, imported lazily because it imports this one.

    `seed_training_rngs`, `model_max_position_embeddings` and `generation_eos_from_cached_config`
    are patched on `rl_train` by the resolver tests (the last two keep the context-limit probe and
    the halting-set read off the hub, which a clean runner has no cache for). Binding them here
    with a `from ... import` would capture the originals at import time, so the patch would rebind
    an object this module never reads and the resolver would run the real ones.
    """
    from flash.engine.worker import rl_train

    return rl_train


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
    _rl_train().seed_training_rngs(_w.SEED)
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
        requested_len, _rl_train().model_max_position_embeddings(model_id, model_revision)
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
            _rl_train().generation_eos_from_cached_config(model_id, model_revision, tok)
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
