"""GRPO / RL training path (TRL GRPOTrainer + colocated vLLM) for the fine-tuning worker."""

from __future__ import annotations

import dataclasses
import os
import random
import time

from flash.engine.chalk_kernels import active_kernels, install_chalk_kernels
from flash.engine.recipe import RECIPE
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.grpo import resolve_grpo_sleep_mode
from flash.engine.worker.heartbeat import liveness_heartbeat
from flash.engine.worker.lora import (
    _LM_SYNC_REMAP_ON,
    disable_liger_grpo_torch_compile,
    is_vl_checkpoint,
    patch_grpo_mask_aware_lm_head,
    patch_vllm_language_model_only,
    patch_vllm_lm_weight_sync,
)
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
    liger_on,
    optimal_attn_impl,
    setup_perf_backends,
    wait_for_gpu,
)


def run_rl():
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    env = _w.require_active_env()
    t_start = time.time()
    _w.heartbeat("rl_start", gpu=gpu_diagnostics())
    is_tool_env = getattr(env, "is_tool_env", False)
    is_multi_turn = getattr(env, "multi_turn", False)
    conversational = is_multi_turn
    if is_multi_turn:
        # Liger fused GRPO loss torch.compiles; on variable-length multi-turn completions its
        # dynamo guard trips a torch 2.10 symbol_to_source crash — fall back to eager, don't raise.
        try:
            import torch._dynamo

            torch._dynamo.config.suppress_errors = True
            print("[rl] multi-turn: torch._dynamo suppress_errors=True (Liger loss falls back to eager on dynamic shapes)")
        except Exception as exc:
            print(f"[rl] could not set torch._dynamo.suppress_errors: {exc!r}")
    wait_for_gpu(_w.JOB_SPEC.gpu.type if _w.JOB_SPEC else None)
    setup_perf_backends()
    model_id = _w.JOB_SPEC.model if _w.JOB_SPEC else RECIPE.hf_model_id
    download_seconds = _w.prefetch_model(model_id)
    rl = RECIPE.rl
    steps = int(
        _w.JOB_SPEC.train.steps if _w.JOB_SPEC and _w.JOB_SPEC.train.steps is not None else rl.num_steps
    )
    gcfg = _w.grpo_overrides()
    _t = _w.JOB_SPEC.train if _w.JOB_SPEC else None
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
    use_vllm = True
    # vLLM colocate LLM overrides actually applied (recorded in train_meta for observability — the
    # console is only uploaded on failure, so a SUCCESSFUL run otherwise can't confirm fp8 KV engaged).
    _kv_dtype = None
    _mnbt = None
    print("[rl] rollout backend: colocated vLLM")
    from flash.catalog import MODELS as _CATALOG

    _info = _CATALOG.get(model_id)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train = env.dataset()
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
    _train_ctx = _t.max_length if (_t and _t.max_length) else 0
    vllm_max_len = int(_train_ctx or max(1024, rl.max_prompt_len + _max_completion))
    # Engine must fit completion + some prompt; fail fast rather than OOM mid-rollout.
    if vllm_max_len <= _max_completion:
        raise ValueError(
            f"engine length {vllm_max_len} leaves no room for the {_max_completion}-token "
            "completion; raise [train].max_length or lower [train].max_tokens"
        )
    prompt_budget = vllm_max_len - _max_completion

    # TRL 1.5's GRPOConfig doesn't truncate prompts, so drop over-budget prompts up front (applies
    # to both string and conversational prompts) before the paid worker rolls out.
    _oai_tools = (
        getattr(getattr(env, "_env", None), "oai_tools", None) if is_tool_env else None
    )

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
            f"{vllm_max_len} - completion {_max_completion}); raise [train].max_length, lower "
            "[train].max_tokens, or shorten the environment's prompts"
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
    # num_iterations / KL decide whether the trainer runs EXTRA unfused logprob forwards beyond the
    # Liger-fused loss: num_iterations>1 caches old_per_token_logps via a separate full-logits forward,
    # and kl_penalty_coef>0 adds a ref_per_token_logps forward. Liger fuses only the LOSS, so those
    # passes still materialize [pd, completion, vocab] logits and the 6 GB logits cap must bind for
    # them. So the fused loss is the SOLE logits pass (cap safe to drop) only when the RESOLVED mu == 1
    # (no old_per_token_logps forward) AND KL is off. The full GRPOConfig field set is feature-
    # detected ONCE here at function scope (not inside the vLLM-only block below) because the TIS
    # rollout-correction knobs read it unconditionally -- they run even when use_vllm is False, so a
    # vLLM-gated definition would NameError on the CPU/non-vLLM path. getattr(__dataclass_fields__)
    # is safe on any class (-> empty set on a non-dataclass TRL, so num_iterations stays 1 and every
    # feature-gated kwarg is simply skipped).
    _grpo_fields = set(getattr(GRPOConfig, "__dataclass_fields__", {}))
    _grpo_has_num_iter = "num_iterations" in _grpo_fields
    # mu (num_iterations) is fixed at 2 when the field exists (the standard GRPO config -- reuse each
    # rollout for 2 optimizer steps; set at the canonical block below). So mu==1 only when the field is
    # ABSENT (old / non-dataclass TRL -> implicit 1): that is the one case the Liger-fused loss can be
    # the sole logits pass, so the cap-drop keys off field existence.
    _mu_one = not _grpo_has_num_iter
    # ...but TRL's COLOCATED-vLLM path ALSO runs a separate old_per_token_logps forward for its
    # importance-sampling (TIS) correction whenever vllm_importance_sampling_correction is enabled —
    # TRL's DEFAULT, which we only tune (mode/clip, below) and never disable — even at mu==1. That
    # unfused [pd, completion, vocab] forward materializes full logits the 6 GB cap must still bound
    # (else a long-completion / multi-turn run sized to a larger per-device batch OOMs in it). Detect
    # whether it runs: prefer the explicit correction field's DECLARED DEFAULT (we never set it in
    # grpo_kwargs, so the default IS the effective value); fall back to the presence of the TIS
    # mode/clip fields (older TRL where the correction is implicitly on for the vLLM path). Only the
    # vLLM rollout path runs this forward, so gate on use_vllm too.
    _corr_field = getattr(GRPOConfig, "__dataclass_fields__", {}).get(
        "vllm_importance_sampling_correction"
    )
    if _corr_field is not None and _corr_field.default is not dataclasses.MISSING:
        _tis_correction_on = bool(_corr_field.default)
    else:
        _tis_correction_on = any(
            f in _grpo_fields
            for f in (
                "vllm_importance_sampling_mode",
                "vllm_importance_sampling_clip_max",
                "vllm_importance_sampling_cap",
            )
        )
    _vllm_is_logprob_forward = use_vllm and _tis_correction_on
    per_device_comps = _w.rl_per_device_comps(
        _cap_completion_len,
        vocab=vocab_size_for(model_id),
        use_vllm=use_vllm,
        params_b=_params_b,
        active_params_b=(float(getattr(_info, "active_params_b", 0.0) or 0.0) or None),
        seq_len=vllm_max_len,
        # The 6 GB logits cap is droppable only when the Liger-fused loss is the SOLE logits-
        # materializing pass: liger fuses the LOSS, but num_iterations>1 (old_per_token_logps), kl>0
        # (ref_per_token_logps), AND TRL's vLLM TIS correction (a per-step old_per_token_logps forward,
        # on by default even at mu==1) each add an unfused full-logits forward the cap must still bound
        # (else a long multi-turn transcript OOMs that forward at pd>1). mu is fixed at 2 (>1) whenever
        # the field exists, so _mu_one is only True on an old TRL with no num_iterations field; combined
        # with _vllm_is_logprob_forward (TIS on by default), the cap is in practice always kept on the
        # colocated-vLLM path. liger_on (True) is also False off-GPU/without the liger wheel -> cap stays.
        fused_logits=(
            liger_on(True) and _mu_one and _kl_beta == 0 and not _vllm_is_logprob_forward
        ),
    )
    if is_multi_turn and _cap_completion_len != _max_completion:
        print(
            f"[rl] multi-turn: sizing the per-device logits cap against the full transcript length "
            f"{_cap_completion_len} (engine context), not the per-turn budget {_max_completion}"
        )
    batching = _w.compute_grpo_batching(prompts_per_step, group_size, per_device_comps)
    # A FORCED per_device (FLASH_RL_PER_DEVICE_COMPS) can differ from the value TRL actually runs for
    # TWO distinct reasons in compute_grpo_batching, and an MFU sweep that logs/probes a per_device it
    # never ran needs to know WHICH: (1) overshoot -- a forced value ABOVE the per-step completion
    # batch (prompts_per_step*group_size) is clamped DOWN to it via min(); (2) non-divisibility -- a
    # forced value at/below the target that doesn't divide it is shrunk to the largest divisor <= it.
    if os.environ.get("FLASH_RL_PER_DEVICE_COMPS", "").strip():
        _used_pd = batching["per_device_train_batch_size"]
        if _used_pd != per_device_comps:
            _target = prompts_per_step * group_size
            if per_device_comps > _target:
                _why = (
                    f"exceeds the per-step completion batch "
                    f"(prompts_per_step*group_size={_target}) and was clamped down to it"
                )
            else:
                _why = (
                    f"does not divide prompts_per_step*group_size={_target} and was shrunk to the "
                    f"largest divisor <= it"
                )
            print(
                f"WARN: forced FLASH_RL_PER_DEVICE_COMPS={per_device_comps} {_why}; TRL will run "
                f"per_device={_used_pd}. Pick a per_device that divides {_target} (and is <= it) "
                f"to probe the exact value."
            )
    if not batching["divisible_by_group"]:
        print(
            "WARN: generation batch not divisible by group size; check prompts_per_step/group_size"
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
        "use_vllm": use_vllm,
        "logging_steps": 1,
        "save_steps": _t.save_every if _t and _t.save_every is not None else 20,
        "save_total_limit": 1,
        # Keep optimizer/scheduler/RNG state with the adapter so a preempted run resumes intact.
        "save_only_model": False,
        "bf16": True,
        "report_to": _w.wandb_report_to(),
        "run_name": _w.wandb_run_name(),
        "seed": _w.SEED,
        "gradient_checkpointing": grad_checkpointing_on(model_id, vllm_max_len),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
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
    # Liger fused GRPO loss: fuses lm_head + logprob so the fp32 248k-vocab logits never
    # materialize. DEFAULT ON regardless of size — without it even 0.8B OOMs a 24 GB card.
    if liger_on(True):
        grpo_kwargs["use_liger_kernel"] = True
        print("[rl] liger fused GRPO loss enabled")
    if use_vllm:
        # sm120: pin a PTX-independent vLLM attention backend before TRL builds the engine, else
        # the rollout can silently produce no completions (flash-attn PTX JIT failure).
        _w.force_vllm_backend_for_sm120()
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
            _w.patch_trl_colocate_llm_kwargs(
                kv_cache_dtype=_kv_dtype, max_num_batched_tokens=_mnbt
            )
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
    init_model, init_peft = _w._init_adapter_model(model_id)
    # chalk kernels are applied below against trainer.model (the authoritative target); TRL may
    # rebuild/wrap the PeftModel, and the fresh-LoRA path only passes the model-id string here.
    if init_peft is not None:
        _attn = optimal_attn_impl()
        # Force bf16 (TRL string-loading can fall back to fp32 and double VRAM).
        grpo_kwargs["model_init_kwargs"] = {"dtype": "bfloat16"}
        if _attn:
            grpo_kwargs["model_init_kwargs"]["attn_implementation"] = _attn
    else:
        _attn = optimal_attn_impl()
    # vLLM sampler stop: truncate each rollout at the delimiter so the reward sees the same text.
    if _t and _t.stop_sequences:
        grpo_kwargs["generation_kwargs"] = {"stop": list(_t.stop_sequences)}
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
        print("[rl] tis: trl default importance-sampling correction in effect; no clip field on this trl")
    cfg = GRPOConfig(**grpo_kwargs)
    setup_seconds = time.time() - t_start
    _w.heartbeat("rl_train_start", setup_seconds=setup_seconds, gpu=gpu_diagnostics())

    # VL checkpoints train text-only: make the colocated rollout engine skip the vision tower.
    if use_vllm:
        patch_vllm_language_model_only(model_id)
        # Install (but don't yet activate) the TRL->vLLM weight-sync name remap for VL checkpoints;
        # activated below, after the trainer's initial checkpoint load is built.
        patch_vllm_lm_weight_sync(model_id)
    hb_cb = _w.make_reward_heartbeat_callback()
    # Tool envs hand TRL the tool callables; pure multi-turn envs hand TRL a rollout_func.
    extra_trainer_kwargs: dict = {}
    tools = env.tools() if is_tool_env else []
    # A tool env with no tools would degrade to single-shot — drive it through rollout_func instead.
    if is_tool_env and not tools:
        print("[rl][warn] tool env exposes no tools — using the multi-turn rollout_func path")
    use_rollout_func = is_multi_turn and not (is_tool_env and tools)
    _w.require_vllm_for_rollout_func(use_rollout_func, use_vllm, model_id)
    if is_tool_env and tools:
        extra_trainer_kwargs["tools"] = tools
        print(f"[rl] tool env: handing {len(tools)} tool(s) to TRL's native tool loop")
    if use_rollout_func:
        from flash.engine.multiturn_rollout import (
            build_examples_index,
            build_rollout_func,
            index_collisions,
        )

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
        )
        print("[rl] multi-turn env: driving the turn loop via rollout_func")
    # GRPOTrainer.__init__ blocks 10-20 min on first use (vLLM build + FA2 compile); the side-thread
    # heartbeat must use the nvidia-smi-only path (include_torch=False) — torch.cuda calls would
    # serialize on the CUDA/allocator locks held by the init thread and false-flag a hang.
    with liveness_heartbeat("rl_initializing"):
        trainer = GRPOTrainer(
            model=init_model,
            args=cfg,
            train_dataset=ds,
            reward_funcs=reward_fn,
            peft_config=init_peft,
            processing_class=tok,
            callbacks=[hb_cb, _w.make_checkpoint_upload_callback()],
            **extra_trainer_kwargs,
        )
    # Apply chalk's gap-filling kernels on trainer.model (the authoritative target).
    _chalk_report = install_chalk_kernels(getattr(trainer, "model", None))
    # Collapse the Liger fused-loss chunk_size to one invocation over the whole micro-batch
    # (TRL's default 1 runs the cycle once per sequence). Numerically identical. Must run BEFORE
    # the mask-aware wrap below, which replaces liger_grpo_loss with a chunk_size-less closure.
    _liger_loss = getattr(trainer, "liger_grpo_loss", None)
    if _liger_loss is not None and hasattr(_liger_loss, "chunk_size"):
        _cs = max(1, int(getattr(trainer.args, "per_device_train_batch_size", 1)))
        if _cs > int(getattr(_liger_loss, "chunk_size", 1)):
            _liger_loss.chunk_size = _cs
            print(f"[rl] liger fused-loss chunk_size -> {_cs} (one invocation, not one per sequence)")
    # Run Liger's fused loss eager: drop only its torch.compile (broken on torch 2.10), keep the
    # chunked memory path. Must run BEFORE the mask-aware wrap below.
    if disable_liger_grpo_torch_compile(trainer):
        print(
            "[rl] liger GRPO loss: torch.compile DISABLED (eager loss math; chunked memory path "
            "retained) — dodges the torch 2.10 dynamo guard-gen crash (symbol_to_source IndexError)"
        )
    # Mask-aware lm_head: skip the 248k-vocab projection at masked completion positions (loss-
    # preserving; no-op when nothing is masked).
    if grpo_kwargs.get("use_liger_kernel") and patch_grpo_mask_aware_lm_head(trainer):
        _masked_kind = "env + padding" if use_rollout_func else "padding"
        print(f"[rl] mask-aware lm_head: skipping masked ({_masked_kind}) positions in the GRPO loss")
    # Activate the weight-sync remap only now, after the initial checkpoint load is built.
    if use_vllm:
        _LM_SYNC_REMAP_ON["on"] = True
        if is_vl_checkpoint(model_id):
            print("[vllm] LM weight-sync remap activated for training syncs")
    # Mid-run eval is intentionally skipped: held-out eval happens deploy-side, keeping training pure.
    _reset_peak_gpu()
    _gpu_sampler = _GpuPeakSampler().start()
    t_train = time.time()
    # The cold first GRPO step (~17 min) emits no rl_step; liveness pings keep the stall detector
    # quiet while the real per-step rl_step callback remains the progress signal.
    with liveness_heartbeat("rl_step"), _sdpa_cudnn_ctx(_attn):
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
                "or reduce prompt length/max_tokens so more examples fit."
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
    adapter_dir = f"{out_dir}/adapter"
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    # VL merge-into-base warm-start (#296) saves a GRPO-ONLY LoRA trained on the SFT-merged base;
    # deployed on the catalog base it drops the SFT and the served model collapses to ~base. Stack
    # the original SFT LoRA back in so the DEPLOYED adapter reproduces base+SFT+GRPO on the original
    # base (no-op for the continued-adapter / fresh-LoRA paths, which already carry the SFT). Both
    # the default `<prefix>/adapter` upload and the `--step <final>` deployable below ship the
    # recombined adapter; the resume checkpoints (`checkpoint/**`) are untouched — they reattach to
    # the re-merged base on resume.
    recombined = _w.recombined_warmstart_adapter_dir(adapter_dir)
    deploy_dir = recombined or adapter_dir
    try:
        _w.hf_upload_folder(deploy_dir, "adapter", required=True)
        # Guarantee the FINAL training step is always a deployable checkpoint, not just an unlabeled
        # `<prefix>/adapter`. The per-save callback only publishes per-step snapshots at save_steps
        # boundaries (and on_train_end re-flushes the latest such boundary), so a final step that
        # doesn't land on one would have NO `flash deploy --step` entry even though it IS the served
        # default adapter. Publish the just-saved final adapter here, keyed by the true final
        # global_step: same bytes as `<prefix>/adapter`, so `--step <final>` always resolves to
        # exactly the deployed default. Idempotent (content-addressed path) when the step already
        # aligned, and best-effort (never fails a paid run).
        if _steps_run:
            _w.publish_deployable_checkpoint(deploy_dir, _steps_run)
    finally:
        # recombined_warmstart_adapter_dir returns a fresh temp dir (flash_recomb_adapter_*); remove
        # it after the uploads so a finalize that runs more than once per process (tests/refactors)
        # doesn't grow /tmp. adapter_dir (the real save) is untouched. Mirrors the per-step cleanup.
        if recombined:
            import shutil

            shutil.rmtree(recombined, ignore_errors=True)
    _w.heartbeat("rl_trained", train_wall=train_wall, gpu=gpu_diagnostics())

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
