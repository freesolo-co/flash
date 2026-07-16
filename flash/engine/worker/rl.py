"""GRPO / RL training path (TRL GRPOTrainer + colocated vLLM) for the fine-tuning worker."""

from __future__ import annotations

import os
import random
import time

from flash.engine.chalk_kernels import active_kernels, install_chalk_kernels
from flash.engine.recipe import RECIPE
from flash.engine.steps import (
    configure_trainer_save_schedule,
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
from flash.engine.worker.grpo import resolve_grpo_sleep_mode
from flash.engine.worker.heartbeat import liveness_heartbeat
from flash.engine.worker.perf import (
    _GpuPeakSampler,
    _metric_curve,
    _peak_gpu_gb,
    _reset_peak_gpu,
    _sdpa_cudnn_ctx,
    free_gpu,
    fused_optim_name,
    gpu_diagnostics,
    grad_checkpointing_on,
    grpo_use_reentrant,
    optimal_attn_impl,
    setup_perf_backends,
    wait_for_gpu,
)
from flash.engine.worker.rng import backend_seed, seed_training_rngs


def grpo_under_ran(steps_run: int, steps: int) -> bool:
    """return true when grpo completes fewer optimizer updates than requested."""
    return int(steps_run) < int(steps)


def run_rl():
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    seed_training_rngs(_w.SEED)
    env = _w.require_active_env()
    t_start = time.time()
    _w.heartbeat("rl_start", gpu=gpu_diagnostics())
    is_tool_env = getattr(env, "is_tool_env", False)
    is_multi_turn = getattr(env, "multi_turn", False)
    conversational = is_multi_turn
    if is_multi_turn:
        # Multi-turn completions are variable-length; keep dynamo failures non-fatal for any compiled
        # helper path that sees those dynamic shapes.
        try:
            import torch._dynamo

            torch._dynamo.config.suppress_errors = True
            print(
                "[rl] multi-turn: torch._dynamo suppress_errors=True (dynamic-shape compiled helpers fall back)"
            )
        except Exception as exc:
            print(f"[rl] could not set torch._dynamo.suppress_errors: {exc!r}")
    wait_for_gpu(
        _w.JOB_SPEC.gpu.type if _w.JOB_SPEC else None,
        exact_type=_w.JOB_SPEC.gpu.exact_type if _w.JOB_SPEC else "",
    )
    setup_perf_backends()
    model_id = _w.JOB_SPEC.model if _w.JOB_SPEC else RECIPE.hf_model_id
    model_revision = getattr(_w.JOB_SPEC, "model_revision", "") if _w.JOB_SPEC else ""
    download_seconds = (
        _w.prefetch_model(model_id, revision=model_revision)
        if model_revision
        else _w.prefetch_model(model_id)
    )
    rl = RECIPE.rl
    _t = _w.JOB_SPEC.train if _w.JOB_SPEC else None
    gcfg = _w.grpo_overrides()
    prompts_per_step = int(
        _t.batch_size if _t and _t.batch_size is not None else rl.prompts_per_step
    )
    group_size = int(gcfg.get("group_size") or rl.group_size)
    # explicit None check, NOT `or` — a configured 0.0 (greedy) must be honored.
    _gcfg_temp = gcfg.get("temperature")
    _temperature = float(_gcfg_temp if _gcfg_temp is not None else rl.sampling_temperature)
    _kl_beta = float(gcfg.get("kl_penalty_coef") or 0.0)
    _adv_clip = float(gcfg.get("advantage_clip") or 0.0)
    _think_penalty = float(gcfg.get("thinking_length_penalty_coef") or 0.0)
    sleep_mode, _grpo_ctx, _card_vram_gb, _fp8_kv = resolve_grpo_sleep_mode()
    print(
        f"[rl] vLLM sleep mode = {sleep_mode} "
        f"(model={model_id}, ctx={_grpo_ctx}, card={_card_vram_gb:.0f}GB)"
    )
    # vLLM colocate LLM overrides actually applied (recorded in train_meta for observability — the
    # console is only uploaded on failure, so a SUCCESSFUL run otherwise can't confirm fp8 KV engaged).
    print("[rl] rollout backend: colocated vLLM")
    if model_revision:
        _w.patch_trl_colocate_llm_kwargs(revision=model_revision)
    from flash.catalog import MODELS as _CATALOG

    _info = _CATALOG.get(model_id)
    # tokenizer + dataset download + per-prompt budget tokenization of the whole dataset can run
    # for minutes with no heartbeat in between; keep the channel visibly fresh.
    with liveness_heartbeat("rl_data_loading"):
        tok = _w.load_tokenizer(model_id, revision=model_revision)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        train = env.dataset()
        _max_examples = getattr(_t, "max_examples", None) if _t else None
        max_examples = int(_max_examples or 0) if _max_examples is not None else 0
        if max_examples > 0:
            train = train[:max_examples]
        rng = random.Random(_w.SEED)
        rng.shuffle(train)
        if conversational:
            prompts = [{"prompt": env.prompt_messages(ex), "example": ex} for ex in train]
        else:
            prompts = [{"prompt": _w.render_prompt(tok, ex), "example": ex} for ex in train]
        _max_completion = int(
            gcfg.get("max_tokens")
            or (rl.max_completion_len_thinking if _w.THINKING else rl.max_completion_len)
        )
        _train_ctx = _t.max_context_tokens if (_t and _t.max_context_tokens) else 0
        vllm_max_len = int(_train_ctx or max(1024, rl.max_prompt_len + _max_completion))
        # Engine must fit completion + some prompt; fail fast rather than OOM mid-rollout.
        if vllm_max_len <= _max_completion:
            raise ValueError(
                f"engine length {vllm_max_len} leaves no room for the {_max_completion}-token "
                "completion; raise [train].max_context_tokens or lower "
                "[train].max_completion_tokens"
            )
        prompt_budget = vllm_max_len - _max_completion

        # TRL 1.5's GRPOConfig doesn't truncate prompts, so drop over-budget prompts up front (applies
        # to both string and conversational prompts) before the paid worker rolls out.
        _oai_tools = getattr(getattr(env, "_env", None), "oai_tools", None) if is_tool_env else None

        def _render_for_budget(p) -> str:
            """Render a prompt to text EXACTLY as the rollout does (incl. tool schemas)."""
            if not conversational:
                return p["prompt"]
            kw = {"tools": _oai_tools} if _oai_tools else {}
            try:
                return tok.apply_chat_template(
                    p["prompt"],
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=_w.THINKING,
                    **kw,
                )
            except Exception as exc:
                raise RuntimeError(
                    "failed to render a conversational prompt with this model's chat template "
                    f"(fix the model/template or the env's prompts): {exc}"
                ) from exc

        def _prompt_tokens(p) -> int:
            return len(tok(_render_for_budget(p), add_special_tokens=False).input_ids)

        kept = [p for p in prompts if 0 < _prompt_tokens(p) <= prompt_budget]
    if len(kept) < len(prompts):
        print(
            f"[rl] dropped {len(prompts) - len(kept)} prompts over the {prompt_budget}-token "
            f"prompt budget (engine {vllm_max_len} - completion {_max_completion})"
        )
    if not kept:
        raise ValueError(
            f"every training prompt exceeds the {prompt_budget}-token prompt budget (engine "
            f"{vllm_max_len} - completion {_max_completion}); raise [train].max_context_tokens, "
            "lower [train].max_completion_tokens, or shorten the environment's prompts"
        )
    prompts = kept
    resolved_prompts_per_step = _w.resolve_grpo_prompts_per_step(prompts_per_step, len(prompts))
    if resolved_prompts_per_step != prompts_per_step:
        print(
            f"[rl] lowering prompts_per_step from {prompts_per_step} to "
            f"{resolved_prompts_per_step}: only {len(prompts)} prompt(s) fit after filtering"
        )
        prompts_per_step = resolved_prompts_per_step
    # Carry a stable int index, not the rich record, so PyArrow can't crash on mixed-type info.
    ds_rows, rollout_examples = _w.build_grpo_prompt_dataset(prompts)
    ds = Dataset.from_list(ds_rows)

    # Derived from a real rendered prompt, NOT _w.THINKING: a template that ignores enable_thinking
    # must not have its tagless answers treated as unterminated reasoning (over-penalized).
    _prompt_opens_thinking = (
        bool(_w.THINKING)
        and bool(prompts)
        and _w.prompt_opens_thinking(_render_for_budget(prompts[0]))
    )
    if _w.THINKING:
        print(f"[rl] prompt_opens_thinking={_prompt_opens_thinking}")

    def reward_fn(completions, **kwargs):
        # rollout_func (multi-turn) reward is computed by the env and forwarded as "reward".
        if kwargs.get("reward") is not None:
            return [float(r) for r in kwargs["reward"]]
        # Fail LOUD if TRL stops forwarding example_idx: defaulting to [] would zip to zero
        # examples -> empty rewards -> silent broken training (#206 / #210).
        example_idx = kwargs.get("example_idx")
        if example_idx is None:
            raise RuntimeError(
                "GRPO reward_fn received no 'example_idx' column from TRL — the reward cannot be "
                "mapped back to its training example, so every reward would be empty/misaligned "
                f"(got kwargs keys {sorted(kwargs)}). This usually means TRL dropped the dataset "
                "column (remove_unused_columns / a TRL version change); the run is aborted rather "
                "than silently training on no signal."
            )
        if len(example_idx) != len(completions):
            raise RuntimeError(
                f"GRPO reward_fn example_idx/completions length mismatch "
                f"({len(example_idx)} vs {len(completions)}) — rewards would be misaligned with "
                "the sampled completions; aborting rather than training on a shifted reward signal."
            )
        examples = [rollout_examples[int(i)] for i in example_idx]
        rewards = []
        debug_rows = []
        for idx, (comp, ex) in enumerate(zip(completions, examples, strict=False)):
            try:
                if isinstance(comp, list):
                    # Conversational transcript (list of messages): score the whole transcript.
                    r = env.reward_from_messages(comp, ex)
                    rewards.append(r)
                    continue
                graded = _w.graded_text(comp, prompt_opened_thinking=_prompt_opens_thinking)
                state = (
                    {
                        "raw": comp,
                        "completion": graded,
                        "thinking": _w.thinking_text(
                            comp, prompt_opened_thinking=_prompt_opens_thinking
                        ),
                    }
                    if _w.THINKING
                    else None
                )
                breakdown = None
                if hasattr(env, "scores_breakdown"):
                    breakdown = env.scores_breakdown(graded, ex, state)
                    r = float(breakdown.get("total", 0.0))
                else:
                    r = env.reward(graded, ex, state)
            except Exception as _reward_exc:
                # The user's env raised during scoring: score 0 rather than killing the run.
                # Scoped to the env calls only — flash's own logic below stays outside.
                print(
                    f"[reward_fn] env scoring raised for completion {idx} "
                    f"({type(_reward_exc).__name__}: {_reward_exc}); scoring as 0.0",
                    flush=True,
                )
                rewards.append(0.0)
                continue
            if _think_penalty > 0 and _w.THINKING:
                # Gated on _prompt_opens_thinking so a template that ignores enable_thinking
                # doesn't get its tagless answers counted as reasoning.
                r -= _think_penalty * _w.think_token_count(
                    comp, tok, prompt_opened_thinking=_prompt_opens_thinking
                )
            rewards.append(r)
            if idx < 8:
                debug_rows.append(
                    {
                        "ts": time.time(),
                        "attempt": _w.ATTEMPT,
                        "run_id": _w.RUN_ID,
                        "mode": _w.RUN_MODE,
                        "seed": _w.SEED,
                        "reward": r,
                        "breakdown": breakdown,
                        "completion_prefix": str(comp or "")[:1000],
                        "graded_prefix": str(graded or "")[:1000],
                        "example_id": (ex or {}).get("id") if isinstance(ex, dict) else None,
                        "example_input": (ex or {}).get("input") if isinstance(ex, dict) else None,
                    }
                )
        _w.upload_debug_jsonl("reward_debug.jsonl", debug_rows)
        return rewards

    from flash.engine.vram import resolve_params_b

    _params_b = resolve_params_b(model_id)
    from flash.catalog import vocab_size_for

    # Multi-turn accumulates a full transcript up to the engine context, so size the fp32 logits
    # cap against the worst-case (engine context), not the per-turn _max_completion, or it OOMs.
    _cap_completion_len = vllm_max_len if is_multi_turn else _max_completion
    # chalk 0.5.0 ships a GRPO fused-logprob op (chalk.ops.grpo) that streams selected-token logprobs
    # without materializing [B,T,V]; WIRING it to flip fused_logits=True is a tracked follow-up (needs a
    # GRPO end-to-end A/B first). Until then every GRPO path keeps the conservative full-logits budget
    # cap (fused_logits=False below), so long completions are BUDGETED to fit — slower than the old Liger
    # fused loss, but not an OOM. Feature-detect GRPOConfig fields once here; the TIS and num_iterations
    # knobs are set later, so this must stay outside the vLLM engine-kwargs block.
    _grpo_fields = set(getattr(GRPOConfig, "__dataclass_fields__", {}))
    _grpo_has_num_iter = "num_iterations" in _grpo_fields
    per_device_comps = _w.rl_per_device_comps(
        _cap_completion_len,
        vocab=vocab_size_for(model_id),
        use_vllm=True,
        params_b=_params_b,
        active_params_b=(float(getattr(_info, "active_params_b", 0.0) or 0.0) or None),
        seq_len=vllm_max_len,
        # Conservative until chalk's GRPO fused-logprob op (chalk 0.5.0, chalk.ops.grpo) is wired in:
        # keep the full-logits budget cap so the unfused path is BUDGETED to fit, never OOMs.
        fused_logits=False,
    )
    if is_multi_turn and _cap_completion_len != _max_completion:
        print(
            f"[rl] multi-turn: sizing the per-device logits cap against the full transcript length "
            f"{_cap_completion_len} (engine context), not the per-turn budget {_max_completion}"
        )
    batching = _w.compute_grpo_batching(prompts_per_step, group_size, per_device_comps)
    if not batching["divisible_by_group"]:
        print(
            "WARN: generation batch not divisible by group size; check prompts_per_step/group_size"
        )
    epochs = int(_t.epochs) if _t and _t.epochs is not None else RECIPE.rl.num_epochs
    derived_steps = on_policy_steps(
        epochs=epochs,
        prompt_count=len(prompts),
        prompts_per_step=batching["unique_prompts_per_step"],
    )
    configured_max_steps = getattr(_t, "max_steps", None) if _t else None
    steps = resolve_update_horizon(derived_steps, configured_max_steps)
    save_at_steps = tuple(getattr(_t, "save_at_steps", ()) or ())
    validate_save_steps(save_at_steps, steps)
    print(
        f"[rl] epochs={epochs} over {len(prompts)} retained prompt(s) at "
        f"{batching['unique_prompts_per_step']} unique_prompts/step -> "
        f"derived_steps={derived_steps} update_horizon={steps}"
    )
    print(
        f"[rl] GRPO batching: per_device={batching['per_device_train_batch_size']} "
        f"grad_accum={batching['gradient_accumulation_steps']} "
        f"generations/step={batching['generations_per_step']} "
        f"unique_prompts/step={batching['unique_prompts_per_step']} "
        f"(target prompts/step={prompts_per_step}, group={group_size}, sleep={sleep_mode})"
    )
    out_dir = f"/tmp/rl_seed{_w.SEED}"
    resume_ckpt = _w.hf_resume_checkpoint()

    grpo_kwargs = {
        "output_dir": out_dir,
        "learning_rate": (
            _t.learning_rate if _t and _t.learning_rate is not None else rl.learning_rate
        ),
        "per_device_train_batch_size": batching["per_device_train_batch_size"],
        "gradient_accumulation_steps": batching["gradient_accumulation_steps"],
        "num_generations": group_size,
        "max_completion_length": _max_completion,
        "max_steps": steps,
        "temperature": _temperature,
        "top_p": rl.sampling_top_p,
        "use_vllm": True,
        "logging_steps": 1,
        "save_steps": _t.save_every if _t and _t.save_every is not None else 20,
        "save_total_limit": 1,
        # Keep optimizer/scheduler/RNG state with the adapter so a preempted run resumes intact.
        "save_only_model": False,
        "bf16": True,
        "report_to": _w.wandb_report_to(),
        "run_name": _w.wandb_run_name(),
        "seed": backend_seed(_w.SEED),
        "gradient_checkpointing": grad_checkpointing_on(
            model_id,
            vllm_max_len,
            revision=model_revision,
        ),
        # MoE needs REENTRANT recompute: its router re-dispatches tokens on the backward recompute,
        # so non-reentrant's metadata-equality assert fires on the first backward and kills the run
        # (Qwen3.6-35B-A3B). Dense models keep the faster non-reentrant path. See grpo_use_reentrant.
        "gradient_checkpointing_kwargs": {"use_reentrant": grpo_use_reentrant(model_id)},
        # Pin a stable GRPO recipe instead of TRL's defaults (which suppress the lift on short runs):
        # constant LR, group-mean-centered advantages (no std scaling), no length-norm; beta = KL coef.
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "beta": _kl_beta,
        "scale_rewards": "none",
        "loss_type": "dr_grpo",
        # Exclude truncated (no-EOS) completions from the loss: including them biases the policy
        # gradient and destabilizes GRPO. GATED OFF when stop_sequences is set (a stop-string
        # rollout's last token isn't EOS, so TRL would mask every completion).
        "mask_truncated_completions": _w.grpo_mask_truncated_completions(_t),
        # 8-bit paged AdamW: colocated GRPO is memory-tight, so int8 state paged to host RAM.
        "optim": fused_optim_name(),
    }
    configure_trainer_save_schedule(grpo_kwargs, save_at_steps)
    if "use_liger_kernel" in _grpo_fields:
        grpo_kwargs["use_liger_kernel"] = False
    # sm120: pin a PTX-independent vLLM attention backend before TRL builds the engine, else
    # the rollout can silently produce no completions (flash-attn PTX JIT failure).
    _w.force_vllm_backend_for_sm120()
    # Blackwell (sm100 B200 / sm120): force the VL model's ViT attention to TORCH_SDPA — vLLM
    # 0.19.1's CUTE flash-attn ViT path is unimportable vs every nvidia-cutlass-dsl and crashes
    # every B200 rollout (no version pin fixes it). No-op off Blackwell / on non-VL models.
    _w.force_vit_sdpa_on_blackwell()
    # colocate_kv_util sizes vLLM's KV pool from flash's per-model estimate; the old blanket
    # 0.45 over-reserved (e.g. 36 GB on an 80 GB A100) and dominated the step peak.
    try:
        import torch as _torch_vram

        from flash.catalog import MODELS
        from flash.engine.vram import colocate_kv_util

        # MoE: size the KV pool on the ACTIVE backbone (matches grpo_fits_resident's resident-fit
        # gate), so the budget and the sleep-mode decision count the SAME KV. Dense -> 0 -> total.
        _active_b = float(getattr(MODELS.get(model_id), "active_params_b", 0.0) or 0.0)
        _total_vram_gb = _torch_vram.cuda.get_device_properties(0).total_memory / 1e9
        _vllm_gpu_mem_util = colocate_kv_util(
            _params_b,
            vllm_max_len,
            _total_vram_gb,
            sleep_mode,
            num_generations=group_size,
            active_params_b=_active_b,
            fp8_kv=_fp8_kv,
        )
    except Exception:
        _vllm_gpu_mem_util = 0.45 if sleep_mode else 0.10
    grpo_kwargs.update(
        vllm_mode="colocate",
        vllm_max_model_length=vllm_max_len,
        vllm_gpu_memory_utilization=_vllm_gpu_mem_util,
        vllm_enable_sleep_mode=sleep_mode,
    )

    def _set_vllm_field(names, value, label):
        for _f in names:
            if _f in _grpo_fields:
                grpo_kwargs[_f] = value
                print(f"[rl] {label} ({_f}={value})")
                return True
        return False

    try:
        import torch as _torch

        _cc = _torch.cuda.get_device_capability()
        _card_gb = _torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        _cc, _card_gb = (0, 0), 0.0
    _kv_dtype = "fp8" if _cc >= (8, 9) else None
    _mnbt = max(8192, vllm_max_len) if _card_gb >= 140 else None
    if _kv_dtype or _mnbt:
        _w.patch_trl_colocate_llm_kwargs(kv_cache_dtype=_kv_dtype, max_num_batched_tokens=_mnbt)
    _set_vllm_field(
        ("vllm_enable_prefix_caching", "enable_prefix_caching"),
        True,
        "vLLM prefix caching (shared GRPO prompt KV reuse)",
    )
    _set_vllm_field(
        ("vllm_enable_chunked_prefill", "enable_chunked_prefill"),
        True,
        "vLLM chunked prefill",
    )
    # vLLM > 0.19.0 regressed the Triton slot-mapping kernel into an illegal memory access on
    # the first generation step (CUDA graph compilation triggers it); skip FULL_AND_PIECEWISE.
    _cudagraph_safe = True
    try:
        import vllm as _vllm_mod

        _ver_base = _vllm_mod.__version__.split("+")[0]
        _vllm_ver = tuple(int(x) for x in _ver_base.split(".")[:3])
        if _vllm_ver > (0, 19, 0):
            _cudagraph_safe = False
            print(
                f"[rl][warn] vLLM {_vllm_mod.__version__} > 0.19.0: skipping "
                "FULL_AND_PIECEWISE CUDA graph compilation (Triton slot-mapping "
                "crash workaround; update vLLM to a TRL-supported version to re-enable)"
            )
            try:
                import torch as _t_cc

                _cc = _t_cc.cuda.get_device_capability()
            except Exception:
                _cc = (0, 0)
            _is_b200 = _cc == (10, 0)  # sm100, the only arch validated for cudagraphs here
            if not _is_b200:
                _w.patch_trl_colocate_llm_kwargs(enforce_eager=True)
                print(
                    f"[rl][warn] enforce_eager=True on the colocate rollout (cc={_cc[0]}.{_cc[1]} "
                    "-> prevent 0.19.1 aot_compile/slot-mapping crash; the removed "
                    "VLLM_TORCH_COMPILE_LEVEL env no longer applies on this vLLM)"
                )
            else:
                print(
                    f"[rl] cc={_cc[0]}.{_cc[1]} (B200/sm100): keeping vLLM CUDA graphs for the "
                    "rollout (validated; MoE-decode speedup)"
                )
    except Exception:
        pass
    if _cudagraph_safe:
        _set_vllm_field(
            ("vllm_compilation_config", "compilation_config"),
            {"cudagraph_mode": "FULL_AND_PIECEWISE"},
            "vLLM cudagraph_mode (verl rollout default)",
        )
    # Continue the SFT adapter when train.init_from_adapter is set, else a fresh LoRA on the id.
    # The warm-start path downloads the adapter and loads it as the trainable adapter on the base
    # (VL checkpoints load the full multimodal model, still just an adapter — no on-disk merge copy).
    seed_training_rngs(_w.SEED)
    with liveness_heartbeat("rl_adapter_loading"):
        init_model, init_peft = _w._init_adapter_model(model_id)
    # _init_adapter_model returns peft_config=None only for a warm-start (the prior adapter is loaded
    # as the trainable "default"); a fresh run returns a LoraConfig for TRL to build.
    is_warm_start = init_peft is None
    # chalk kernels are applied below against trainer.model (the authoritative target); TRL may
    # rebuild/wrap the PeftModel, and the fresh-LoRA path only passes the model-id string here.
    if init_peft is not None:
        _attn = optimal_attn_impl()
        # Force bf16 (TRL string-loading can fall back to fp32 and double VRAM).
        grpo_kwargs["model_init_kwargs"] = {
            "dtype": "bfloat16",
            **_w.model_revision_kwargs(model_revision),
        }
        if _attn:
            grpo_kwargs["model_init_kwargs"]["attn_implementation"] = _attn
    else:
        _attn = optimal_attn_impl()
    # vLLM sampler stop: truncate each rollout at the delimiter so the reward sees the same text.
    _gen_kwargs: dict = {}
    if _t and _t.stop_sequences:
        _gen_kwargs["stop"] = list(_t.stop_sequences)
    # [train] structured_outputs: pass the spec as a plain dict — TRL's colocate path wraps it
    # into StructuredOutputsParams itself, so GRPOConfig (and its wandb/config serialization)
    # never holds a vLLM object.
    _so_spec = parse_structured_outputs(getattr(_t, "structured_outputs", "") if _t else "")
    if _so_spec:
        _gen_kwargs["structured_outputs"] = _so_spec
        print(
            f"[rl] structured outputs: every rollout turn constrained to "
            f"{describe_structured_outputs(_so_spec)}"
        )
    # thinking + a constraint: gate the grammar on </think> so the rollout reasons freely before its
    # answer is constrained. Engine-level (EngineArgs.reasoning_parser) — TRL/GRPOConfig has no field
    # for it, so inject via the colocate-LLM patch, which MUST run before GRPOTrainer builds the engine.
    _reasoning_parser = reasoning_parser_for(thinking=_w.THINKING, structured_outputs=_so_spec)
    if _reasoning_parser:
        _w.patch_trl_colocate_llm_kwargs(reasoning_parser=_reasoning_parser)
        print(
            f"[rl] structured outputs under thinking: grammar applied only after </think> "
            f"(reasoning_parser={_reasoning_parser})"
        )
    if _gen_kwargs:
        grpo_kwargs["generation_kwargs"] = _gen_kwargs
    # TRL has no advantage-value clip knob (it clips the importance ratio); just note the request.
    if _adv_clip > 0:
        print(f"[rl] advantage_clip={_adv_clip} recorded; TRL centers advantages (no value clip)")
    if _grpo_has_num_iter:
        grpo_kwargs["num_iterations"] = 2
        print("[rl] rollout amortization: num_iterations=2 (reuse each generation batch)")
    # Truncated importance sampling: adopt the verl recipe (token-level, c_max=2.0) over TRL's
    # defaults (sequence_mask / 3.0), feature-detected against this TRL's GRPOConfig fields.
    if "vllm_importance_sampling_mode" in _grpo_fields:
        grpo_kwargs["vllm_importance_sampling_mode"] = "token_truncate"
        print("[rl] tis mode=token_truncate (token-level truncated importance sampling)")
    _tis_c = 2.0
    _tis_clip_field = next(
        (
            f
            for f in ("vllm_importance_sampling_clip_max", "vllm_importance_sampling_cap")
            if f in _grpo_fields
        ),
        None,
    )
    if _tis_clip_field:
        grpo_kwargs[_tis_clip_field] = _tis_c
        print(f"[rl] tis clip c_max={_tis_c} ({_tis_clip_field})")
    else:
        print(
            "[rl] tis: trl default importance-sampling correction in effect; no clip field on this trl"
        )
    cfg = GRPOConfig(**grpo_kwargs)
    setup_seconds = time.time() - t_start
    _w.heartbeat("rl_train_start", setup_seconds=setup_seconds, gpu=gpu_diagnostics())

    # VL checkpoints roll out on the FULL multimodal engine (vision tower included) — the colocated
    # vLLM loads Qwen3_5ForConditionalGeneration and its own hf_to_vllm_mapper maps the trainer's
    # ``model.language_model.*`` weight-sync names, so no flash-side remap is needed.
    hb_cb = _w.make_reward_heartbeat_callback()
    # Tool envs hand TRL the tool callables; pure multi-turn envs hand TRL a rollout_func.
    extra_trainer_kwargs: dict = {}
    tools = env.tools() if is_tool_env else []
    # A tool env with no tools would degrade to single-shot — drive it through rollout_func instead.
    if is_tool_env and not tools:
        print("[rl][warn] tool env exposes no tools — using the multi-turn rollout_func path")
    use_rollout_func = is_multi_turn and not (is_tool_env and tools)
    _w.require_vllm_for_rollout_func(use_rollout_func, True, model_id)
    if is_tool_env and tools:
        extra_trainer_kwargs["tools"] = tools
        print(f"[rl] tool env: handing {len(tools)} tool(s) to TRL's native tool loop")
    if use_rollout_func:
        from flash.engine.multiturn_rollout import (
            build_examples_index,
            build_rollout_func,
            index_collisions,
            resolve_rollout_request_timeout_seconds,
        )

        _rollout_request_timeout = resolve_rollout_request_timeout_seconds(vllm_max_len)
        _rollout_request_max_attempts = 2
        examples_by_key = build_examples_index(train, env.prompt_messages)
        ncol = index_collisions(train, env.prompt_messages)
        if ncol:
            print(
                f"[rl][warn] {ncol} duplicate prompt(s) collide in the reward index; the shared "
                "prompt scores against the last example's answer/info"
            )
        extra_trainer_kwargs["rollout_func"] = build_rollout_func(
            active_env=env,
            tok=tok,
            examples_by_key=examples_by_key,
            max_completion=_max_completion,
            max_turns=getattr(env, "max_turns", 10),
            temperature=_temperature,
            top_p=rl.sampling_top_p,
            stop=(list(_t.stop_sequences) if _t and _t.stop_sequences else None),
            thinking=_w.THINKING,
            engine_max_len=vllm_max_len,
            structured_outputs=_so_spec,
            request_timeout_seconds=_rollout_request_timeout,
            request_max_attempts=_rollout_request_max_attempts,
        )
        print(
            f"[rl] multi-turn env: driving the turn loop via rollout_func; each physical vLLM "
            f"request has timeout={_rollout_request_timeout:.1f}s and "
            f"max_attempts={_rollout_request_max_attempts}"
        )
    # GRPOTrainer.__init__ blocks 10-20 min on first use (vLLM build + FA2 compile); the side-thread
    # heartbeat must use the nvidia-smi-only path (include_torch=False) — torch.cuda calls would
    # serialize on the CUDA/allocator locks held by the init thread and false-flag a hang.
    with liveness_heartbeat("rl_initializing"):
        trainer_model = init_model
        if init_peft is not None:
            model_init_kwargs = dict(grpo_kwargs.get("model_init_kwargs") or {})
            model_init_kwargs["device_map"] = None
            trainer_model = _w.prepare_fresh_lora_base(
                init_model,
                model_id,
                model_init_kwargs,
                phase="rl",
                model_revision=model_revision,
            )
            if not isinstance(trainer_model, str):
                cfg.model_init_kwargs = None
        trainer = GRPOTrainer(
            model=trainer_model,
            args=cfg,
            train_dataset=ds,
            reward_funcs=reward_fn,
            peft_config=init_peft,
            processing_class=tok,
            callbacks=[
                hb_cb,
                _w.make_checkpoint_upload_callback(save_at_steps),
            ],
            **extra_trainer_kwargs,
        )
        # Apply chalk's standalone kernels on trainer.model (the authoritative target).
        # inside the liveness wrap: chalk's kernel install can JIT-compile, silent for minutes.
        _chalk_report = install_chalk_kernels(getattr(trainer, "model", None))
    # Warm-start + KL anchor: continuing the one SFT adapter means TRL derives the KL reference by
    # snapshotting a frozen "ref" adapter from the loaded SFT default (grpo_trainer: is_peft_model and
    # beta != 0) — so KL anchors to SFT, NOT the bare base. Assert the snapshot exists whenever KL is
    # on, so a TRL change that silently fell back to adapter-disable (ref == bare base, which pulls the
    # policy back to base — the #296 collapse re-expressed as a loss term) fails LOUDLY here instead of
    # quietly degrading a warm-started run. No-op when KL is off (beta == 0, flash's default) or fresh.
    if is_warm_start and _kl_beta > 0:
        _peft_cfg = getattr(getattr(trainer, "model", None), "peft_config", None) or {}
        if "ref" not in _peft_cfg:
            raise RuntimeError(
                "GRPO warm-start with kl_penalty_coef>0 expected TRL to snapshot a frozen 'ref' "
                "adapter from the continued SFT adapter (so the KL reference is SFT), but none was "
                "created — the KL term would anchor to the bare base and walk the policy back to base. "
                "Set kl_penalty_coef=0, or use a TRL build that creates the 'ref' adapter for PEFT."
            )
    # Mid-run eval is intentionally skipped: held-out eval happens deploy-side, keeping training pure.
    _reset_peak_gpu()
    _gpu_sampler = _GpuPeakSampler().start()
    t_train = time.time()
    # The cold first GRPO step (~17 min) emits no rl_step; liveness pings keep the stall detector
    # quiet while the real per-step rl_step callback remains the progress signal.
    # progress + progress_step: step advances emit REAL heartbeats (and stamp the step), so the
    # daemon can never starve the provider's stall clock by winning the throttled upload slot with
    # a bare liveness ping while training is healthy.
    with (
        liveness_heartbeat(
            "rl_step",
            progress=lambda: int(getattr(trainer.state, "global_step", 0) or 0),
            progress_step=True,
        ),
        _sdpa_cudnn_ctx(_attn),
    ):
        trainer.train(resume_from_checkpoint=resume_ckpt)
    train_wall = time.time() - t_train
    rl_peak_gpu_gb = _peak_gpu_gb()
    rl_device_peak_gpu_gb = _gpu_sampler.stop_gb()
    reward_history = list(getattr(hb_cb, "reward_history", []))
    # An empty reward_history means the reward path never ran (rollout scored nothing) — a no-op
    # run, not a success. An all-zero-reward env still appends 0.0s, so empty is unambiguous.
    _steps_run = int(getattr(trainer.state, "global_step", 0) or 0)
    # A resume that already reached target steps does zero new steps (and has an empty history)
    # though the policy is fully trained — finalize those instead of failing.
    _resumed_complete = _w._grpo_resume_already_complete(resume_ckpt, steps, _steps_run)
    if _w._grpo_is_no_op_failure(reward_history, resume_ckpt, steps, _steps_run):
        if _steps_run == 0:
            raise RuntimeError(
                "GRPO trainer completed zero optimizer steps before any reward was scored. "
                f"retained_prompts={len(prompts)}, prompts_per_step={prompts_per_step}, "
                f"generations_per_step={batching['generations_per_step']}. This usually means "
                "TRL built an empty dataloader; add training examples, lower [train].batch_size, "
                "or reduce prompt length/max_completion_tokens so more examples fit."
            )
        raise RuntimeError(
            f"GRPO scored no reward in {train_wall:.1f}s over {_steps_run} step(s) — the rollout "
            "produced no completions, so the policy was never actually trained. Failing loudly "
            "instead of reporting a no-op run as done (seen on RTX 5090/sm120 vLLM rollout)."
        )
    if not reward_history and _resumed_complete:
        print(
            f"[resume] no new reward in this worker but resumed checkpoint already reached "
            f"{_steps_run}/{steps} step(s) — finalizing the completed policy instead of failing."
        )
    # only a genuine under-run fails: a resume already at/past target does zero new steps yet holds a
    # fully-trained policy (_steps_run >= steps, _resumed_complete above), matching opd (opt_steps < steps).
    if grpo_under_ran(_steps_run, steps):
        raise RuntimeError(f"grpo completed {_steps_run}/{steps} requested optimizer updates")
    # adapter save + required upload can take minutes on a 35B; keep the heartbeat fresh through the
    # whole finalize. keepalive=True:
    # _steps_run is CONSTANT here (training is done), so without it every finalize ping is a bare
    # liveness that does NOT advance the provider's stall clock — a finalize outlasting STALL_AFTER_S
    # (1500s) would be killed at the finish line. keepalive forces a REAL heartbeat each tick.
    # progress_step stamps the final step on every finalize heartbeat so a cancel landing in this
    # window still bills the actual steps trained (actual_steps_run reads last_heartbeat.step).
    with liveness_heartbeat(
        "rl_finalizing", progress=lambda: _steps_run, progress_step=True, keepalive=True
    ):
        adapter_dir = f"{out_dir}/adapter"
        _w.stamp_adapter_provenance(trainer.model, model_id, model_revision)
        trainer.model.save_pretrained(adapter_dir)
        tok.save_pretrained(adapter_dir)
        _w.write_base_model_provenance(adapter_dir, model_id, model_revision)
        # Warm-start CONTINUES the one SFT adapter in place, so the saved adapter already carries
        # SFT+GRPO on the original catalog base and deploys as-is — no recombine step (fresh-LoRA runs
        # likewise deploy their single adapter directly).
        _w.hf_upload_folder(adapter_dir, "adapter", required=True)
        # preserve the final checkpoint only when exact save steps are not configured.
        if final_save_due(_steps_run, save_at_steps):
            _w.publish_deployable_checkpoint(adapter_dir, _steps_run)
    _w.heartbeat("rl_trained", train_wall=train_wall, step=_steps_run, gpu=gpu_diagnostics())

    # Upper bound on generated tokens (over-counts; used only for a rough throughput).
    gen_tokens = steps * batching["unique_prompts_per_step"] * group_size * _max_completion
    _w.write_train_meta(
        phase="rl",
        adapter_dir=adapter_dir,
        model_id=model_id,
        train_wall=train_wall,
        setup_seconds=setup_seconds,
        train_tokens=0,
        generated_tokens=gen_tokens,
        notes={
            "steps": steps,
            "epochs": epochs,
            "retained_prompts": len(prompts),
            "resumed": bool(resume_ckpt),
            "download_seconds": download_seconds,
            "hf_transfer": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", ""),
            "reward_history": reward_history,
            # vLLM colocate-rollout overrides actually applied (the trl-LLM monkeypatch): confirms
            # fp8 KV / the raised prefill batch engaged on a SUCCESSFUL run, without the console.
            "vllm_kv_cache_dtype": _kv_dtype,
            "vllm_max_num_batched_tokens": _mnbt,
            "loss_curve": _metric_curve(trainer, "loss"),
            # peak_gpu_gb = torch-allocated; device_peak_gpu_gb = true device footprint (incl. vLLM).
            "peak_gpu_gb": rl_peak_gpu_gb,
            "device_peak_gpu_gb": rl_device_peak_gpu_gb,
            # Which chalk kernels actually engaged (None = not installed / all fell back).
            "chalk_kernels": active_kernels(_chalk_report) or None,
            **_w.wandb_run_info(),
            "gen_tokens_is_upper_bound": True,
            "thinking": _w.THINKING,
            "max_completion_len": _max_completion,
            "prompts_per_step": batching["unique_prompts_per_step"],
            "generations_per_step": batching["generations_per_step"],
            "group_size": group_size,
            "per_device_train_batch_size": batching["per_device_train_batch_size"],
            "gradient_accumulation_steps": batching["gradient_accumulation_steps"],
            "grpo_recipe": {
                "lr_scheduler": "constant",
                "beta": _kl_beta,
                "scale_rewards": "none",
                "loss_type": "dr_grpo",
                "temperature": _temperature,
                "advantage_clip": _adv_clip,
                "thinking_length_penalty_coef": _think_penalty,
                "init_from_adapter": _w.JOB_SPEC.train.init_from_adapter if _w.JOB_SPEC else "",
            },
        },
    )
    free_gpu(trainer)
