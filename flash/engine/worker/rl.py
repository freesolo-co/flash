"""GRPO / RL training path (TRL GRPOTrainer + colocated vLLM) for the fine-tuning worker.

Run-scoped state (``JOB_SPEC``/``SEED``/``THINKING``/``RUN_ID``/``RUN_MODE``/``ATTEMPT``) and the
worker-namespace helpers (``heartbeat``/``render_prompt``/``grpo_overrides``/``rl_per_device_comps``
/``_init_adapter_model``/``write_train_meta`` ...) are read THROUGH the worker package
(``_w.<name>``) at CALL time so the module-level monkeypatch contract holds. The pure
perf/lora/kernel probes are imported directly (they read no run state).
"""

from __future__ import annotations

import os
import random
import time

from flash.engine.chalk_kernels import active_kernels, install_chalk_kernels
from flash.engine.recipe import RECIPE
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.heartbeat import init_liveness_heartbeat, train_liveness_heartbeat
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
    grpo_sleep_mode,
    liger_on,
    optimal_attn_impl,
    setup_perf_backends,
    wait_for_gpu,
)


def run_rl():
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    env = _w.require_active_env()  # fail loudly (not AttributeError: NoneType) on the no-JobSpec path
    t_start = time.time()
    _w.heartbeat("rl_start", gpu=gpu_diagnostics())
    # GRPO rollout strategy by env shape (trl 1.6 adds the hooks these need):
    #   * single-turn          -> TRL single-shot generation + per-completion reward (below);
    #   * tool (ToolEnv & subs:
    #     Stateful/Sandbox/Python) -> TRL drives the tool-call loop natively via
    #     GRPOTrainer(tools=...) (it parses tool calls, executes the tools, and masks the
    #     tool-result tokens itself); the reward scores the full transcript;
    #   * pure multi-turn      -> a custom rollout_func (flash.engine.multiturn_rollout)
    #     drives THIS env's turn loop on the colocate engine and returns the interleaved
    #     token sequence with an env_mask so only the model's tokens are trained.
    is_tool_env = getattr(env, "is_tool_env", False)
    is_multi_turn = getattr(env, "multi_turn", False)
    conversational = is_multi_turn  # message-list prompts (tool + pure multi-turn) vs strings
    if is_multi_turn:
        # The Liger fused GRPO loss (use_liger_kernel, kept ON to avoid the 248k-vocab fp32-logits
        # OOM) torch.compiles, and on the VARIABLE-length multi-turn completions its dynamo guard
        # build trips a torch 2.10 bug (symbol_to_source IndexError) that crashes the first
        # training step. Let dynamo FALL BACK TO EAGER for the offending function instead of
        # raising. This is NOT `TORCHDYNAMO_DISABLE` (which would also break the colocate vLLM
        # engine's required compilation) — dynamo stays enabled; only erroring graphs run eager.
        try:
            import torch._dynamo

            torch._dynamo.config.suppress_errors = True
            print("[rl] multi-turn: torch._dynamo suppress_errors=True (Liger loss falls back to eager on dynamic shapes)")
        except Exception as exc:  # never let a torch internals change block the run
            print(f"[rl] could not set torch._dynamo.suppress_errors: {exc!r}")
    wait_for_gpu(_w.JOB_SPEC.gpu.type if _w.JOB_SPEC else None)
    setup_perf_backends()
    model_id = _w.JOB_SPEC.model if _w.JOB_SPEC else RECIPE.hf_model_id
    download_seconds = _w.prefetch_model(model_id)
    rl = RECIPE.rl
    # Steps come from the run's [train] steps (already in JOB_SPEC), else the recipe default.
    steps = int(
        _w.JOB_SPEC.train.steps if _w.JOB_SPEC and _w.JOB_SPEC.train.steps is not None else rl.num_steps
    )
    # Throughput/quality knobs: the number of prompts optimized per step, completions per
    # prompt, and whether vLLM offloads weights between steps. Sleep mode frees memory for the
    # optimizer but reloads ~weights each step (a large per-step cost); it's gated OFF by model
    # size when both the policy and rollout engine fit resident.
    gcfg = _w.grpo_overrides()
    _t = _w.JOB_SPEC.train if _w.JOB_SPEC else None
    # batch_size = prompts per optimizer step for GRPO.
    # prompts per optimizer step = the run config's [train].batch_size (recipe default otherwise).
    prompts_per_step = int(
        _t.batch_size if _t and _t.batch_size is not None else rl.prompts_per_step
    )
    group_size = int(gcfg.get("group_size") or rl.group_size)
    # temperature: explicit None check, NOT `or` — a configured 0.0 (greedy/deterministic
    # rollouts) must be honored, not fall back to the recipe sampling temperature.
    _gcfg_temp = gcfg.get("temperature")
    _temperature = float(_gcfg_temp if _gcfg_temp is not None else rl.sampling_temperature)
    _kl_beta = float(gcfg.get("kl_penalty_coef") or 0.0)
    _adv_clip = float(gcfg.get("advantage_clip") or 0.0)
    _think_penalty = float(gcfg.get("thinking_length_penalty_coef") or 0.0)
    # vLLM sleep mode offloads the rollout engine's weights between steps to free memory for the
    # optimizer, but reloading each step is a large per-step cost (PR #174 measured ~2-2.6x faster
    # GRPO with it OFF on models that fit) AND on the large-model GRPO path the sleep/wake cycle
    # STALLS the colocated rollout (the rollout emits unparseable completions, then the worker
    # hangs mid-training). So enable sleep only when the run genuinely can't fit RESIDENT on THIS
    # card: large/long-context AND the policy + colocated rollout engine + training peak don't fit
    # on the live GPU. When they fit (the common allocator-sized case), skip sleep entirely.
    _grpo_ctx = int(_t.max_length if _t and _t.max_length else 0)
    _card_vram_gb = 0.0
    try:
        import torch as _torch_card

        if _torch_card.cuda.is_available():
            # Decimal GB (/1e9), matching grpo_fits_resident's comparison target: estimate_vram_gb
            # measures peak in decimal GB (its byte-counting logits terms divide by 1e9), and
            # rl_per_device_comps sizes the micro-batch in decimal GB too. Binary GiB here would
            # UNDER-report the card ~7% vs the estimate, so a card that genuinely fits resident
            # could be told it doesn't (or the micro-batch could assume headroom the gate denied);
            # one unit everywhere keeps the resident-fit decision and the micro-batch cap consistent.
            _card_vram_gb = _torch_card.cuda.get_device_properties(0).total_memory / 1e9
    except Exception as _e:
        print("[rl] card VRAM probe failed (sleep-mode gate falls back to size/context):", _e)
    _lora_rank = int(_t.lora_rank) if _t and _t.lora_rank else 32
    sleep_mode = grpo_sleep_mode(
        model_id,
        max_length=_grpo_ctx,
        group_size=group_size,
        max_tokens=gcfg.get("max_tokens"),
        lora_rank=_lora_rank,
        thinking=_w.THINKING,
        card_vram_gb=_card_vram_gb,
    )
    print(
        f"[rl] vLLM sleep mode = {sleep_mode} "
        f"(model={model_id}, ctx={_grpo_ctx}, card={_card_vram_gb:.0f}GB)"
    )
    # Rollout backend: always colocated vLLM (fast). The whole supported catalog runs GRPO with
    # colocated vLLM; there is no transformers-generation fallback.
    use_vllm = True
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
        # Message-list prompts so the chat template applies roles + (for tool envs) the tool
        # schemas; per-turn length is managed by the tool loop / rollout_func, not a flat budget.
        prompts = [{"prompt": env.prompt_messages(ex), "example": ex} for ex in train]
    else:
        prompts = [{"prompt": _w.render_prompt(tok, ex), "example": ex} for ex in train]
    # The colocated vLLM engine's model length is the hard cap on prompt+completion at
    # rollout. Size it from [train].max_length and derive the prompt budget from it so a
    # bigger engine or a smaller completion automatically admits longer prompts (rather than
    # a fixed rl.max_prompt_len that no env override could lift).
    _max_completion = int(
        gcfg.get("max_tokens")
        or (rl.max_completion_len_thinking if _w.THINKING else rl.max_completion_len)
    )
    # Engine context = the run's [train].max_length (so a long-context GRPO config sized/paid for
    # by the allocator actually RUNS at that length), else the recipe default. Without the
    # train.max_length fallback the allocator provisions a big GPU for the long context but the
    # engine runs short — paying for headroom we never use.
    _train_ctx = _t.max_length if (_t and _t.max_length) else 0
    vllm_max_len = int(_train_ctx or max(1024, rl.max_prompt_len + _max_completion))
    # The engine must fit completion + at least some prompt. If [train].max_length is below the
    # completion budget, no prompt can ever fit — fail fast here rather than passing a 1-token
    # budget that lets prompts through and then OOMs/overflows mid-rollout.
    if vllm_max_len <= _max_completion:
        raise ValueError(
            f"engine length {vllm_max_len} leaves no room for the {_max_completion}-token "
            "completion; raise [train].max_length or lower [train].max_tokens"
        )
    prompt_budget = vllm_max_len - _max_completion

    # TRL 1.5's GRPOConfig has no max_prompt_length and does NOT truncate prompts, so a prompt
    # that leaves no room for the completion within the engine length would fail mid-rollout
    # AFTER the paid worker is provisioned. Drop prompts that don't fit the budget up front.
    # render_prompt returns an apply_chat_template(tokenize=False) string that already carries
    # the special tokens, so tokenize with add_special_tokens=False (the default re-adds
    # BOS/EOS and over-counts).
    # Drop prompts that leave no room for the completion within the engine length — applies to
    # BOTH single-turn (string prompts) and conversational (message-list) prompts, so a tool /
    # multi-turn rollout can't overflow the colocate engine mid-generation. Conversational
    # prompts are length-checked via the chat template (with the generation prompt).
    # Tool schemas TRL injects into the prompt for native tools= GRPO — include them in the
    # budget for a tool env so a prompt isn't undercounted at filter time vs. rollout time.
    _oai_tools = (
        getattr(getattr(env, "_env", None), "oai_tools", None) if is_tool_env else None
    )

    def _prompt_tokens(p) -> int:
        if conversational:
            # Render to text then tokenize — the SAME path the rollout uses — so the filter
            # count matches the rollout's count (avoids a tokenize=True vs text mismatch).
            kw = {"tools": _oai_tools} if _oai_tools else {}
            try:
                text = tok.apply_chat_template(
                    p["prompt"],
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=_w.THINKING,
                    **kw,
                )
            except Exception as exc:
                # Fail fast WITH context: a tokenizer/template incompatibility would render every
                # prompt uncountable and otherwise surface as a misleading "all prompts exceed
                # budget" — raise so the model/template can be fixed before a paid run trains on
                # a degenerate dataset.
                raise RuntimeError(
                    "failed to render a conversational prompt with this model's chat template "
                    f"(fix the model/template or the env's prompts): {exc}"
                ) from exc
            return len(tok(text, add_special_tokens=False).input_ids)
        return len(tok(p["prompt"], add_special_tokens=False).input_ids)

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
    # Carry a stable integer index instead of the rich record so PyArrow can't crash on an env whose
    # per-row info/metadata legitimately mixes types (see build_grpo_prompt_dataset). reward_fn maps
    # the index back to the original example object below.
    ds_rows, rollout_examples = _w.build_grpo_prompt_dataset(prompts)
    ds = Dataset.from_list(ds_rows)

    def reward_fn(completions, **kwargs):
        # rollout_func (pure multi-turn) path: the per-rollout reward is computed by the env
        # during the rollout and forwarded as the "reward" extra field — pass it through.
        if kwargs.get("reward") is not None:
            return [float(r) for r in kwargs["reward"]]
        # Score the <think>-stripped text (graded_text), then — datums parity — deduct
        # the thinking-length penalty computed from the RAW completion's <think> span.
        # The dataset carries example_idx (not the record); map each back to its original object.
        # Fail LOUD if TRL stops forwarding example_idx (column pruning / a TRL change): defaulting to
        # [] would zip to ZERO examples -> empty rewards -> silent no-op / broken training (issues
        # #206 / #210). A reward over the wrong/empty examples is far worse than crashing the run.
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
                    # Tool / conversational transcript (TRL passes a list of messages): score the
                    # whole transcript via the environment reward (no <think> stripping —
                    # multi-turn content).
                    r = env.reward_from_messages(comp, ex)
                    rewards.append(r)
                    continue
                graded = _w.graded_text(comp)
                breakdown = None
                if hasattr(env, "scores_breakdown"):
                    breakdown = env.scores_breakdown(graded, ex)
                    r = float(breakdown.get("total", 0.0))
                else:
                    r = env.reward(graded, ex)
            except Exception as _reward_exc:
                # The user's environment raised during scoring (e.g. a transient network error
                # calling an LLM judge). Treat this sample as 0 reward rather than crashing the
                # entire RL run — a single bad score call should not kill the worker. Scoped
                # deliberately to the env calls above: flash's own logic (the think penalty below)
                # stays outside so a tokenizer/config bug fails loud instead of hiding as a 0.0.
                print(
                    f"[reward_fn] env scoring raised for completion {idx} "
                    f"({type(_reward_exc).__name__}: {_reward_exc}); scoring as 0.0",
                    flush=True,
                )
                rewards.append(0.0)
                continue
            # Thinking-length penalty is computed from flash's own tokenizer (not the user env), so it
            # lives outside the try/except above — an internal failure here should crash the run, not
            # be silently swallowed into a 0.0 reward.
            if _think_penalty > 0 and _w.THINKING:
                r -= _think_penalty * _w.think_token_count(comp, tok)
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

    # TRL's per_device_train_batch_size counts COMPLETIONS, not prompts. Size grad-accum so
    # the global completion batch = prompts_per_step * group_size, i.e. each optimizer step
    # actually optimizes `prompts_per_step` prompts. The per-device *completion* micro-batch
    # is the VRAM knob (thinking-aware; see rl_per_device_comps).
    from flash.engine.vram import resolve_params_b

    # Open-model (uncataloged) GRPO: size the colocate activation cap from the catalog stat, else
    # the HF safetensors metadata (no download). Without a real count a large open model falls back
    # to the ~2B-width default in rl_per_device_comps and gets too LOOSE a per-device cap ->
    # colocate OOM. Best-effort: stays None offline, keeping prior behavior.
    _params_b = resolve_params_b(model_id)
    from flash.catalog import vocab_size_for

    # Per-device completion-logits cap: a multi-turn rollout accumulates a FULL transcript (model
    # turns + masked env tokens) up to the engine context — far longer than the single-turn per-turn
    # budget `_max_completion` — and the trainer's logprob forward processes that whole completion.
    # So size the fp32 [per_device, completion, vocab] cap against the WORST-CASE multi-turn
    # completion length (the engine context) instead of `_max_completion`, or a long multi-turn run
    # OOMs the trainer forward. Single-turn keeps `_max_completion` (its true completion length).
    _cap_completion_len = vllm_max_len if is_multi_turn else _max_completion
    per_device_comps = _w.rl_per_device_comps(
        _cap_completion_len,
        vocab=vocab_size_for(model_id),
        use_vllm=use_vllm,
        params_b=_params_b,
        # The trainer forward processes prompt+completion up to the engine context, so the
        # activation/VRAM cap is sized against the worst-case training sequence length.
        seq_len=vllm_max_len,
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
        # NB: GRPOConfig has no max_prompt_length field (TRL 1.5) and does not truncate
        # prompts; the dataset is pre-filtered above to prompts that fit prompt_budget
        # (vllm_max_len - completion), so every prompt fits the engine sized here.
        "max_completion_length": _max_completion,
        "max_steps": steps,
        "temperature": _temperature,
        "top_p": rl.sampling_top_p,
        "use_vllm": use_vllm,
        "logging_steps": 1,
        "save_steps": _t.save_every if _t and _t.save_every is not None else 20,
        "save_total_limit": 1,
        # Resumable checkpoints: keep the optimizer/scheduler/RNG state with the LoRA adapter so a
        # preempted GRPO run resumed via resume_from_checkpoint(hf_resume_checkpoint()) continues
        # with intact optimizer state + step instead of a fresh optimizer. For LoRA this state is
        # small (trainable adapter params only). The deployable per-step snapshot strips it
        # separately, so serving still gets adapter-only files.
        "save_only_model": False,
        "bf16": True,
        "report_to": _w.wandb_report_to(),  # W&B when WANDB_API_KEY present (restored post-flash-migration)
        "run_name": _w.wandb_run_name(),
        "seed": _w.SEED,
        "gradient_checkpointing": grad_checkpointing_on(model_id, vllm_max_len),
        # Non-reentrant checkpointing: the modern path that composes correctly with autograd
        # saved-tensor hooks and avoids the reentrant path's extra graph retention. (verl #3629.)
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        # Pin a stable, well-conditioned GRPO recipe instead of inheriting TRL's defaults
        # (which on a short run suppress the lift): constant LR (TRL default 'linear' decays
        # to 0 over the run), advantages centered by group mean only (no std scaling, which
        # biases by difficulty/length — matches datums.centered_advantages), and no
        # length-normalized loss. beta is the KL-to-reference coef (datums kl_masks ->
        # kl_penalty_coef).
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "beta": _kl_beta,
        "scale_rewards": "none",
        "loss_type": "dr_grpo",
        # Exclude TRUNCATED completions (ran to max_completion_length without an EOS) from the
        # loss. TRL's default is False, which trains on these incomplete rollouts — and because a
        # truncated completion is not a real sample from the policy's distribution over FINISHED
        # sequences, including it gives a biased policy gradient (TRL/DAPO: truncated completions
        # are "incorrectly penalized and introduce noise during training"). On long-completion or
        # multi-turn envs that frequently hit the budget this destabilizes GRPO and can DEGRADE the
        # model below its SFT start (observed: runs with clipped_ratio 0.6-0.99 stalled at ~0 reward
        # while the vLLM-vs-trainer sampling_logp_difference exploded). dr_grpo normalizes by the
        # constant batch*max_completion_length (not mask.sum()), so masking every completion in a
        # batch yields a 0 loss / 0 gradient — safe, never a divide-by-zero. TRL>=1.6 also applies
        # this mask to the multi-turn tool/env mask, so the rollout_func path is covered too.
        # GATED OFF when stop_sequences is set (see grpo_mask_truncated_completions): TRL flags
        # truncation by "last token != EOS/PAD", but a stop-string rollout's last token is not EOS.
        "mask_truncated_completions": _w.grpo_mask_truncated_completions(_t),
        # Optimizer: 8-bit paged AdamW (int8 state paged to host RAM -> fits a smaller GPU);
        # colocated GRPO (trainer + vLLM on one GPU) is memory-tight, so this is the right default.
        "optim": fused_optim_name(),
    }
    # Liger fused GRPO loss: fuses the lm_head + per-token logprob so the fp32
    # [batch, seq, ~248k vocab] logits never materialize — the documented GRPO OOM driver.
    # TRL 1.6's GRPOConfig flag is `use_liger_kernel` (NOT `use_liger_loss`, which doesn't
    # exist in 1.6). DEFAULT ON for the GRPO path regardless of model size: MEASURED that
    # WITHOUT it even Qwen3.5-0.8B GRPO OOMs a 24 GB (and 32 GB) card because the per-completion
    # logits over the 248k vocab dominate — the small-scale JIT cost is far cheaper than the OOM.
    # (This differs from SFT, where Liger is gated by size since 1B-class SFT can be net-negative.)
    if liger_on(True):
        grpo_kwargs["use_liger_kernel"] = True
        print("[rl] liger fused GRPO loss enabled")
    if use_vllm:
        # RTX 5090 / sm120: pin a PTX-independent vLLM attention backend (FLASHINFER) BEFORE TRL
        # builds the colocated engine — else the rollout can silently produce no completions on
        # old-driver Blackwell hosts (flash-attn PTX JIT failure). No-op off sm120 / if pinned.
        _w.force_vllm_backend_for_sm120()
        # Colocate shares one GPU between the policy model and the vLLM rollout engine.
        # vllm_max_model_length bounds the KV cache to what GRPO needs (else vLLM sizes for
        # the model's FULL context and won't start on a consumer GPU).
        # vllm_gpu_memory_utilization sizes vLLM's KV pool. The blanket sleep-path 0.45 was a
        # misjudgement: on an 80 GB A100 it reserves 0.45 x 80 = 36 GB of KV, but a GRPO rollout only
        # holds ~num_generations x context tokens. MEASURED (Qwen3.5-4B colocate): that 36 GB
        # reservation is the dominant resident allocation and sets the step peak (~46 GB) — exactly why
        # trainer-side optimisations (mask-aware lm_head, fused layers) moved nothing. colocate_kv_util
        # sizes both paths from flash's per-model KV estimate instead (vram.py); MEASURED 4B/80 GB peak
        # 46 -> 26 GB, reward byte-identical, train_wall neutral.
        try:
            import torch as _torch_vram

            from flash.engine.vram import colocate_kv_util

            _total_vram_gb = _torch_vram.cuda.get_device_properties(0).total_memory / 1e9
            _vllm_gpu_mem_util = colocate_kv_util(
                _params_b, vllm_max_len, _total_vram_gb, sleep_mode, num_generations=group_size
            )
        except Exception:
            _vllm_gpu_mem_util = 0.45 if sleep_mode else 0.10  # safe fallback to the old constants
        grpo_kwargs.update(
            vllm_mode="colocate",
            vllm_max_model_length=vllm_max_len,
            vllm_gpu_memory_utilization=_vllm_gpu_mem_util,
            vllm_enable_sleep_mode=sleep_mode,
        )
        # Rollout-memory + throughput knobs, applied ONLY if this TRL exposes the field (so an
        # older TRL never crashes on an unknown kwarg). All verl-validated for GRPO colocate (#174).
        _grpo_fields = set(getattr(GRPOConfig, "__dataclass_fields__", {}))

        def _set_vllm_field(names, value, label):
            for _f in names:
                if _f in _grpo_fields:
                    grpo_kwargs[_f] = value
                    print(f"[rl] {label} ({_f}={value})")
                    return True
            return False

        # fp8 KV cache only where the silicon has native fp8 (compute capability >= 8.9: Ada /
        # Hopper / Blackwell) — ~halves the rollout KV pool. Ampere (A100/A6000/3090) lacks
        # fp8, so it stays fp16 there (forcing it on would error / silently emulate).
        try:
            import torch as _torch

            _want_fp8 = _torch.cuda.get_device_capability() >= (8, 9)
        except Exception:
            _want_fp8 = False
        if _want_fp8:
            _set_vllm_field(("vllm_kv_cache_dtype", "kv_cache_dtype"), "fp8", "fp8 KV cache")
        # PREFIX CACHING: every GRPO group of `num_generations` rollouts shares the SAME prompt
        # prefix, so caching the prompt KV computes it once and reuses it — the dominant rollout win
        # on one GPU. CHUNKED PREFILL interleaves prefill with decode so a long prompt doesn't stall
        # the batch. CUDAGRAPH MODE sets verl's full-graph-decode + piecewise-fallback rollout mode.
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
        # vLLM 0.19.1 regressed the Triton _compute_slot_mapping_kernel: it launches
        # (num_reqs + 1) thread blocks but the block table only has num_reqs rows, so the
        # extra block causes an illegal memory access (cudaErrorIllegalAddress) on the first
        # generation step. CUDA graph compilation triggers this path. Skip FULL_AND_PIECEWISE
        # for vLLM versions outside TRL's supported range (0.12.0-0.19.0) until a fix lands.
        _cudagraph_safe = True
        try:
            import vllm as _vllm_mod

            _ver_base = _vllm_mod.__version__.split("+")[0]  # strip PEP440 local (e.g. +cu121)
            _vllm_ver = tuple(int(x) for x in _ver_base.split(".")[:3])
            if _vllm_ver > (0, 19, 0):
                _cudagraph_safe = False
                print(
                    f"[rl][warn] vLLM {_vllm_mod.__version__} > 0.19.0: skipping "
                    "FULL_AND_PIECEWISE CUDA graph compilation (Triton slot-mapping "
                    "crash workaround; update vLLM to a TRL-supported version to re-enable)"
                )
                # vLLM 0.19.1 ALSO hits `RuntimeError: aot_compile is not supported by the
                # current configuration` through its DEFAULT torch.compile path on some GPU
                # arches (Ampere sm_86: A6000, A100) — it fires from vllm/compilation/wrapper.py
                # when torch._dynamo.is_compiling() is False inside the CUDA-graph capture path.
                # Skipping FULL_AND_PIECEWISE above is not enough (the vllm_compilation_config
                # GRPOConfig field doesn't exist in this TRL, so that _set_vllm_field is a no-op).
                # VLLM_TORCH_COMPILE_LEVEL=0 (NO_COMPILATION) forces vLLM to execute the model
                # eagerly, preventing the AOT path entirely. Official vLLM env var (vllm/envs.py);
                # a no-op on a vLLM that doesn't define it. Don't override an operator-set value.
                if "VLLM_TORCH_COMPILE_LEVEL" not in os.environ:
                    os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "0"
                    print("[rl][warn] VLLM_TORCH_COMPILE_LEVEL=0 (prevent aot_compile on vLLM 0.19.1)")
        except Exception:
            pass
        if _cudagraph_safe:
            _set_vllm_field(
                ("vllm_compilation_config", "compilation_config"),
                {"cudagraph_mode": "FULL_AND_PIECEWISE"},
                "vLLM cudagraph_mode (verl rollout default)",
            )
    # Adapter init: continue training the SFT adapter (peft_config=None, model is the
    # loaded PeftModel) when train.init_from_adapter is set, else a fresh LoRA on the
    # string model id (model_init_kwargs forces bf16 — TRL string-loading can fall back
    # to fp32 and double VRAM).
    init_model, init_peft = _w._init_adapter_model(model_id)
    # chalk's kernels are applied AFTER construction (below) against trainer.model: chalk's apply
    # patches the LIVE nn.Module, so there is nothing to install pre-build. On the fresh-LoRA path
    # init_model is just the model-id string (TRL builds the module), and even on the
    # continue-adapter path TRL may rebuild/wrap the PeftModel, so trainer.model is the
    # authoritative target.
    if init_peft is not None:
        # Fresh LoRA: TRL loads the string model id with these kwargs, then attaches the
        # adapter. Force bf16 (TRL string-loading can fall back to fp32 and double VRAM).
        _attn = optimal_attn_impl()  # arch-aware FlashAttention (Kernels Hub) / SDPA
        grpo_kwargs["model_init_kwargs"] = {"dtype": "bfloat16"}
        if _attn:
            grpo_kwargs["model_init_kwargs"]["attn_implementation"] = _attn
    else:
        _attn = optimal_attn_impl()
    # stop_sequences: TRL forwards generation_kwargs to the (vLLM) sampler, whose
    # SamplingParams.stop truncates each rollout at the requested delimiter — so the reward
    # sees the same completion the config intends, instead of generating to max_completion.
    if _t and _t.stop_sequences:
        grpo_kwargs["generation_kwargs"] = {"stop": list(_t.stop_sequences)}
    # advantage_clip>0 is the datums centered-advantage clamp; TRL has no advantage-value
    # clip knob (it clips the importance ratio), so honor the default (clip off ==
    # centered) and surface a note when a config asks for an explicit clamp.
    if _adv_clip > 0:
        print(f"[rl] advantage_clip={_adv_clip} recorded; TRL centers advantages (no value clip)")
    # num_iterations (the one promoted GRPO speed lever, measured 1.38x faster) is feature-detected
    # so an older TRL that lacks the field is simply skipped (GRPOConfig rejects unknown kwargs).
    # Generation dominates GRPO wall-clock, so reusing each rollout batch for 2 optimizer steps is
    # the cheapest large speedup; mu=2 is the standard GRPO config and TRL's importance-sampling
    # correction (on by default) keeps the step stable. (The GSPO/DAPO A/B levers were dropped: the
    # framework-scan in gpu-bench/RESEARCH_FINDINGS.md measured no robust win over baseline.)
    import dataclasses as _dc

    try:
        _grpo_fields = {f.name for f in _dc.fields(GRPOConfig)}
    except TypeError:
        _grpo_fields = set()  # not a dataclass on this TRL -> skip the feature-detected knob
    if "num_iterations" in _grpo_fields:
        grpo_kwargs["num_iterations"] = 2
        print("[rl] rollout amortization: num_iterations=2 (reuse each generation batch)")
    # truncated importance sampling (tis): trl's grpo applies an importance-sampling correction by
    # default, but with mode="sequence_mask" and clip_max=3.0. the verl/openrlhf recipe for the
    # rollout(vllm)-vs-training token-distribution mismatch is TOKEN-LEVEL truncated is with the
    # per-token ratio clipped at c=2 (verl rollout_is_threshold=2.0). adopt that recipe here:
    # token_truncate + c_max=2.0. feature-detected against this trl's GRPOConfig fields (canonical
    # clip field first, then the pre-2.0 deprecated alias), so a trl that lacks a field is skipped.
    # note: this deliberately changes trl's defaults (sequence_mask / 3.0) to the recipe values.
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

    # VL checkpoints (Qwen3.5/3.6) train text-only: make TRL's colocated rollout
    # engine skip the vision tower (VRAM + 5090 PTX-compat; see the patch docstring).
    # Only relevant when vLLM drives rollouts; transformers generation uses the trainer
    # model (already text-only via the LoRA target/exclude config).
    if use_vllm:
        patch_vllm_language_model_only(model_id)
        # Install (but do NOT yet activate) the TRL->vLLM weight-sync name remap for Qwen3.5/3.6:
        # the trainer pushes ``model.*`` names but the VL engine's LM params live under
        # ``language_model.*``, so the first sync_weights() would raise without this. Activated
        # below, after the trainer + its initial checkpoint load are built.
        patch_vllm_lm_weight_sync(model_id)
    hb_cb = _w.make_reward_heartbeat_callback()
    # Multi-turn / tool wiring (trl 1.6): tool envs hand TRL the tool callables so it runs the
    # tool-call loop natively; pure multi-turn envs hand TRL a rollout_func that drives the
    # env's own turn loop on the colocate engine (env_mask masks the non-model tokens).
    extra_trainer_kwargs: dict = {}
    tools = env.tools() if is_tool_env else []
    # A tool env exposing NO tools would silently degrade to single-shot under tools=[]; drive
    # it through the rollout_func turn loop instead so it isn't mis-trained as single-turn.
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
    # GRPOTrainer.__init__ blocks during model/vLLM init + FA2 kernel compilation (can be
    # 10-20 min on first use). Background heartbeats keep the stall detector quiet.
    #
    # CRITICAL: this heartbeat runs on a SIDE THREAD while the main thread is deep inside the
    # blocking GRPOTrainer.__init__ (vLLM colocate engine build + weight load) — a long, CUDA- and
    # allocator-busy section. gpu_diagnostics(include_torch=True) makes torch.cuda calls
    # (mem_get_info / memory_allocated / memory_reserved / get_device_name) that serialize on the
    # CUDA driver lock and PyTorch's caching-allocator mutex, both held by the init thread. Those
    # calls can then BLOCK for the whole init -> the heartbeat thread freezes -> the control plane
    # sees no heartbeat and false-flags a HANG on a run that is merely doing a slow (but live) cold
    # init (observed on consumer GPUs: RTX 4090/5090/A6000, where cold init is longest). Use the
    # nvidia-smi-only path (include_torch=False): it runs out-of-process with an 8s timeout and
    # releases the GIL during the subprocess wait, so it keeps ticking through a CUDA-busy init. The
    # torch memory numbers are meaningless here anyway (the model isn't built yet). If even THIS
    # stops ticking, the main thread is holding the GIL in a C extension (a true wedge) -> no heartbeat
    # lands at all and the provider's stall detection catches it (init_liveness_heartbeat, heartbeat.py).
    with init_liveness_heartbeat("rl_initializing"):
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
    # Apply chalk's gap-filling kernels (RoPE/LoRA-delta/embedding, like Liger) on the module
    # GRPOTrainer actually optimizes (trainer.model) — the fresh-LoRA path only passes the model-id
    # string to TRL, so trainer.model is the authoritative target. chalk composes on top of Liger.
    # Capture the install report so the engaged kernels land in metrics (active_kernels below).
    _chalk_report = install_chalk_kernels(getattr(trainer, "model", None))
    # Liger fused-loss chunk_size: TRL leaves it at the default 1, so the fused GRPO loss runs its
    # whole detach -> chunk_forward -> compiled-loss -> autograd.grad cycle ONCE PER SEQUENCE
    # (per_device_train_batch_size times) — Python/kernel-launch/compile-guard overhead that
    # dominates at small-model scale where the GEMMs are tiny. Collapse it to ONE invocation over the
    # whole per-device micro-batch. Numerically identical (every loss_type normalizes by the GLOBAL
    # token count, not the chunk-local size, and chunk losses are summed). Must run BEFORE the
    # mask-aware wrap below, which replaces trainer.liger_grpo_loss with a closure that has no
    # chunk_size attribute.
    _liger_loss = getattr(trainer, "liger_grpo_loss", None)
    if _liger_loss is not None and hasattr(_liger_loss, "chunk_size"):
        _cs = max(1, int(getattr(trainer.args, "per_device_train_batch_size", 1)))
        if _cs > int(getattr(_liger_loss, "chunk_size", 1)):
            _liger_loss.chunk_size = _cs
            print(f"[rl] liger fused-loss chunk_size -> {_cs} (one invocation, not one per sequence)")
    # Run liger's fused GRPO loss EAGER: drop ONLY its torch.compile (BROKEN on torch 2.10 — its
    # dynamo guard-gen trips a symbol_to_source IndexError that crashes the first GRPO step on every
    # path), keep the chunked memory path that prevents the 248k-vocab fp32-logit OOM. Must run BEFORE
    # the mask-aware wrap below, which replaces trainer.liger_grpo_loss with a closure. See the helper.
    if disable_liger_grpo_torch_compile(trainer):
        print(
            "[rl] liger GRPO loss: torch.compile DISABLED (eager loss math; chunked memory path "
            "retained) — dodges the torch 2.10 dynamo guard-gen crash (symbol_to_source IndexError)"
        )
    # Mask-aware lm_head: skip the 248k-vocab projection at MASKED completion positions in the GRPO
    # loss — its most expensive op, and the trainer step dominates train_wall. For MULTI-TURN that
    # masked set is the ~half-to-most of the transcript that is env/tool text; for SINGLE-TURN it is
    # the right-PADDING (GRPO samples variable-length completions, padded to the batch max). Either
    # way those positions add zero loss/gradient but pay full FLOPs. Loss-preserving; applies to ALL
    # GRPO with the Liger fused loss; no-op when nothing is masked (uniform-length single-turn).
    if grpo_kwargs.get("use_liger_kernel") and patch_grpo_mask_aware_lm_head(trainer):
        _masked_kind = "env + padding" if use_rollout_func else "padding"
        print(f"[rl] mask-aware lm_head: skipping masked ({_masked_kind}) positions in the GRPO loss")
    # The trainer (and its colocated vLLM engine + initial checkpoint load) is now built. Activate
    # the TRL->vLLM weight-sync name remap ONLY now (see patch_vllm_lm_weight_sync) so the initial
    # checkpoint load stayed untouched while the train-time syncs get remapped. No-op unless the VL
    # patch above was installed.
    if use_vllm:
        _LM_SYNC_REMAP_ON["on"] = True
        if is_vl_checkpoint(model_id):
            print("[vllm] LM weight-sync remap activated for training syncs")
    # Mid-run eval is intentionally NOT run during training: held-out evaluation happens on the
    # deploy/serving side (against the trained adapter), keeping training pure (no eval-phase cost
    # or eval-boundary stalls). Training streams only the per-step reward heartbeat.
    _reset_peak_gpu()  # peak_gpu_gb reflects the train loop (verifies the micro-batch headroom)
    _gpu_sampler = _GpuPeakSampler().start()  # true device peak incl. vLLM colocate + bnb pages
    t_train = time.time()
    # Gap-filler liveness around the train loop: the cold FIRST GRPO step (vLLM rollout warmup +
    # backward, ~17 min observed on a consumer GPU) emits no rl_step until it completes and would look
    # like a hang. train_liveness_heartbeat fills the gap, gated on global_step so a genuinely stuck
    # train() still trips the stall path. See heartbeat.liveness_heartbeat.
    with train_liveness_heartbeat(
        "rl_step", lambda: getattr(getattr(trainer, "state", None), "global_step", 0)
    ), _sdpa_cudnn_ctx(_attn):  # force cuDNN SDPA on sm120 (no-op otherwise)
        trainer.train(resume_from_checkpoint=resume_ckpt)
    train_wall = time.time() - t_train
    rl_peak_gpu_gb = _peak_gpu_gb()
    rl_device_peak_gpu_gb = _gpu_sampler.stop_gb()
    reward_history = list(getattr(hb_cb, "reward_history", []))
    # A GRPO run that finishes WITHOUT the reward callback ever firing (empty reward_history)
    # produced NO real training — the rollout scored nothing (e.g. vLLM generation silently
    # returning no completions, observed on RTX 5090 / sm120: ~1.4 s wall, empty reward + loss
    # curves, but the run otherwise "succeeds"). That is a FAILURE, not a success: a no-op run with
    # an unchanged adapter must not be reported as done — fail loudly so the operator/agent doesn't
    # trust it. (An env returning all-zero rewards still appends 0.0s, so an EMPTY history uniquely
    # means the reward path never ran.)
    _steps_run = int(getattr(trainer.state, "global_step", 0) or 0)
    # A resume that already reached the target steps legitimately performs ZERO new optimizer
    # steps: the previous worker uploaded the final checkpoint (and scored its rewards) but died
    # before writing metrics/DONE, so this worker's fresh hb_cb has an empty reward_history even
    # though the policy IS fully trained. Don't fail those — finalize from the resumed state. The
    # no-op guard below is only for a run that genuinely trained nothing (no resume, or the resume
    # didn't reach the target steps).
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
    _w.hf_upload_folder(adapter_dir, "adapter", required=True)
    # Guarantee the FINAL training step is always a deployable checkpoint, not just an unlabeled
    # `<prefix>/adapter`. The per-save callback only publishes per-step snapshots at save_steps
    # boundaries (and on_train_end re-flushes the latest such boundary), so a final step that
    # doesn't land on one would have NO `flash deploy --step` entry even though it IS the served
    # default adapter. Publish the just-saved final adapter here, keyed by the true final
    # global_step: same bytes as `<prefix>/adapter`, so `--step <final>` always resolves to exactly
    # the deployed default. Idempotent (content-addressed path) when the step already aligned, and
    # best-effort (never fails a paid run).
    if _steps_run:
        _w.publish_deployable_checkpoint(adapter_dir, _steps_run)
    _w.heartbeat("rl_trained", train_wall=train_wall, gpu=gpu_diagnostics())

    # Upper bound on generated tokens: completions actually optimized (the intended
    # prompts_per_step after the batch fix) x the max completion length. Over-counts (most
    # completions are shorter); reported as an upper bound, used only for a rough throughput.
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
            "loss_curve": _metric_curve(trainer, "loss"),
            # Peak torch-allocated GPU memory during the GRPO train loop (excludes bnb managed
            # pages). device_peak_gpu_gb is the TRUE device footprint (total-free, incl. the vLLM
            # colocate engine + bnb pages): the headline for verifying the per-device micro-batch
            # left the card with headroom (no OOM) at the sized batch.
            "peak_gpu_gb": rl_peak_gpu_gb,
            "device_peak_gpu_gb": rl_device_peak_gpu_gb,
            # Which chalk gap-filling kernels actually ENGAGED (None = chalk not installed or every
            # kernel fell back) — verifies the chalk stack on a GRPO run without the console.
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
