"""SFT training path (TRL SFTTrainer) for the fine-tuning worker."""

from __future__ import annotations

import math
import os
import random
import time

from flash.engine.chalk_kernels import active_kernels, install_chalk_kernels
from flash.engine.recipe import RECIPE
from flash.engine.steps import (
    configure_trainer_save_schedule,
    final_save_due,
    resolve_update_horizon,
    sft_update_steps,
    validate_save_steps,
)
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.heartbeat import liveness_heartbeat
from flash.engine.worker.packing import (
    BlockDiagonalCollator,
    completion_mask_from_ids,
    gdn_packing_available,
    model_is_gdn_hybrid,
    model_is_pure_attention,
    pack_token_ids,
    packing_efficiency,
    tokenize_for_packing,
)
from flash.engine.worker.perf import (
    _flash_attn_available,
    _GpuPeakSampler,
    _memory_mode,
    _metric_curve,
    _peak_gpu_gb,
    _reset_peak_gpu,
    _sdpa_cudnn_ctx,
    free_gpu,
    fused_optim_name,
    gpu_diagnostics,
    grad_checkpointing_on,
    grpo_use_reentrant,
    loraplus_optimizer_cls,
    optimal_attn_impl,
    setup_perf_backends,
    wait_for_gpu,
)
from flash.engine.worker.rng import backend_seed, seed_training_rngs


def _pretokenize_completion_only(texts, tokenizer, max_length):
    """Pre-tokenize SFT rows into ``{input_ids, completion_mask}``, dropping rows with no real completion target.

    Returns ``(kept_texts, pretok, n_dropped)``.
    """
    full_ids = tokenize_for_packing([t["text"] for t in texts], tokenizer, max_length)
    prompt_ids = tokenizer(
        [t["prompt_text"] for t in texts], truncation=True, max_length=max_length
    )["input_ids"]
    pretok = [
        {"input_ids": ids, "completion_mask": completion_mask_from_ids(pids, ids)}
        for ids, pids in zip(full_ids, prompt_ids, strict=True)
    ]
    special_ids = set(getattr(tokenizer, "all_special_ids", None) or [])

    def _has_real_target(row) -> bool:
        return any(
            m and tid not in special_ids
            for tid, m in zip(row["input_ids"], row["completion_mask"], strict=True)
        )

    kept = [(t, r) for t, r in zip(texts, pretok, strict=True) if _has_real_target(r)]
    return [t for t, _ in kept], [r for _, r in kept], len(pretok) - len(kept)


def _model_arch_dims(model_id: str, revision: str = "") -> tuple[int, int]:
    """``(hidden_size, num_hidden_layers)`` used to size the GC-off activation estimate.

    Prefer the CURATED catalog geometry (deterministic, no network/parse risk) for known models — a
    live B200 SFT showed the runtime ``AutoConfig`` probe returning (0, 0) on the 35B-A3B's
    multimodal-nested config, which silently kept GC on. For open-model-policy ids (no catalog dims)
    fall back to the HF config, handling the ``text_config`` nesting (config.json is already cached by
    the tokenizer load). Best-effort: ``(0, 0)`` if neither is available -> the GC-off gate
    conservatively keeps gradient checkpointing on."""
    from flash.catalog import MODELS

    info = MODELS.get(model_id)
    c_hidden = int(getattr(info, "hidden_size", 0) or 0) if info else 0
    c_layers = int(getattr(info, "num_layers", 0) or 0) if info else 0
    if c_hidden and c_layers and not revision:
        return c_hidden, c_layers
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
            **_w.model_revision_kwargs(revision),
        )
        tc = getattr(cfg, "text_config", None) or cfg
        layers = int(
            getattr(tc, "num_hidden_layers", 0) or getattr(cfg, "num_hidden_layers", 0) or 0
        )
        hidden = int(getattr(tc, "hidden_size", 0) or getattr(cfg, "hidden_size", 0) or 0)
        # Only a NONZERO probe dim that disagrees with the catalog is a real revision mismatch. A (0, 0)
        # probe means "couldn't parse" (the 35B-A3B multimodal-nested config does exactly this), not
        # "differs" -- treat it like the unpinned path and fall back to catalog dims below, otherwise a
        # revision pin would spuriously fail such models after the GPU is already rented.
        if revision and (
            (c_hidden and hidden and hidden != c_hidden)
            or (c_layers and layers and layers != c_layers)
        ):
            raise ValueError("revision architecture does not match catalog geometry")
        return hidden or c_hidden, layers or c_layers
    except Exception as e:
        if revision:
            raise RuntimeError("could not validate revision-specific model architecture") from e
        print(f"[sft] arch-dims probe failed ({e}); GC decision stays conservative (keep GC on)")
        return c_hidden, c_layers


def select_sft_examples(train, max_examples, seed):
    """Pick the SFT sample: the first ``max_examples`` rows of the dataset (file order), shuffled.

    The slice happens BEFORE the shuffle so ``max_examples`` is a deterministic prefix fence,
    not a random subsample. A train.jsonl that carries fully-labeled SFT rows first and
    prompt-only (empty-output) GRPO rows after can cap SFT to the labeled head — an empty
    completion can never be shuffled into the SFT sample and teach the model to emit nothing.
    """
    if max_examples > 0:
        train = train[:max_examples]
    rng = random.Random(seed)
    rng.shuffle(train)
    return train


def sft_completed_train_tokens(
    tokens_per_epoch: int,
    epochs: int,
    derived_steps: int,
    completed_steps: int,
) -> int:
    """Estimate tokens processed from completed updates while preserving epoch accounting at parity."""
    epoch_tokens = max(0, int(tokens_per_epoch)) * max(0, int(epochs))
    completed = max(0, int(completed_steps))
    derived = max(1, int(derived_steps))
    if completed == derived:
        return epoch_tokens
    if completed == 0 or epoch_tokens == 0:
        return 0
    return max(1, round(epoch_tokens * completed / derived))


def sft_under_ran(final_step: int, update_horizon: int, max_steps: int) -> bool:
    """True when a max_steps-authoritative run completed fewer updates than requested.

    with max_steps authoritative, trl's max_steps override lands a fresh run exactly on the horizon,
    and a resume from a checkpoint past a lowered horizon does zero new steps yet holds a fully-trained
    adapter (final_step >= horizon). fail loudly only on a genuine under-run, mirroring grpo
    (steps_run < steps) and opd (opt_steps < steps).
    """
    return int(max_steps) > 0 and int(final_step) < int(update_horizon)


def run_sft():
    from datasets import Dataset
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    seed_training_rngs(_w.SEED)
    env = _w.require_active_env()
    t_start = time.time()
    _w.heartbeat("sft_start", gpu=gpu_diagnostics())
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

    _t = _w.JOB_SPEC.train if _w.JOB_SPEC else None

    def _train_opt(name, default):
        val = getattr(_t, name, None) if _t else None
        return val if val is not None else default

    # tokenizer + dataset download + O(N) chat-template render can run for minutes on a big
    # dataset with no heartbeat in between; keep the channel visibly fresh.
    with liveness_heartbeat("sft_data_loading"):
        tok = _w.load_tokenizer(model_id, revision=model_revision)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        train = select_sft_examples(env.dataset(), int(_train_opt("max_examples", 0) or 0), _w.SEED)
        texts = []
        multiturn_targets = 0
        for ex in train:
            completion = env.sft_completion(ex)
            if len(completion) > 1:
                multiturn_targets += 1
            prompt_messages = env.prompt_messages(ex)
            msgs = [*prompt_messages, *completion]
            texts.append(
                {
                    "text": tok.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=_w.THINKING,
                    ),
                    "prompt_text": tok.apply_chat_template(
                        prompt_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=_w.THINKING,
                    ),
                }
            )
    if multiturn_targets:
        print(
            f"[sft] multi-turn SFT: {multiturn_targets}/{len(train)} rows train on a full target transcript"
        )
    elif getattr(env, "multi_turn", False):
        print(
            "[sft][warn] this is a multi-turn Freesolo environment but no row ships a multi-turn "
            "target completion; SFT collapses to a single assistant turn per row (tool/env turns "
            'ignored). Provide target transcripts (output={"messages": [...]}) for proper multi-turn SFT.'
        )
    if _w.THINKING and not any("<think>" in t["text"] for t in texts[:256]):
        print(
            "WARN: thinking mode is ON but no sampled SFT target contains a <think> "
            "trace — training on non-reasoning targets teaches the model to SKIP "
            "thinking. Use a dataset with reasoning traces, or set thinking = false."
        )
    setup_seconds = time.time() - t_start
    _w.heartbeat("sft_model_load", setup_seconds=setup_seconds, gpu=gpu_diagnostics())

    epochs = int(
        _w.JOB_SPEC.train.epochs
        if _w.JOB_SPEC and _w.JOB_SPEC.train.epochs is not None
        else RECIPE.sft.num_epochs
    )
    from flash.catalog import MODELS, resolve_vocab_size
    from flash.engine.vram import sft_grad_accum

    sft_lr = _train_opt("learning_rate", RECIPE.sft.learning_rate)
    # Total SFT sequence budget (prompt + completion), honoring [train] max_context_tokens like the
    # cost/preflight path (flash.cost.spec._sft_seq_len); unset -> the tuned recipe cap. Read the
    # RENAMED knob: reading a stale "max_length" here made getattr silently return the 1024 default.
    sft_max_len = _train_opt(
        "max_context_tokens",
        RECIPE.sft.max_seq_len_thinking if _w.THINKING else RECIPE.sft.max_seq_len,
    )
    with liveness_heartbeat("sft_pretokenizing"):
        texts, _pretok, _dropped = _pretokenize_completion_only(texts, tok, sft_max_len)
    if _dropped:
        print(
            f"[sft] dropped {_dropped} rows with no real completion target "
            "(sft_max_len truncated away the whole completion, or an empty/content-free completion)"
        )
    if not _pretok:
        raise ValueError(
            "every SFT example has an empty completion after sft_max_len truncation (nothing to "
            "train on); increase sft_max_len or shorten the prompts"
        )
    ds = Dataset.from_list(_pretok)
    _masked_tok = sum(m.count(0) for m in (r["completion_mask"] for r in _pretok))
    _total_tok = sum(len(r["input_ids"]) for r in _pretok)
    if _total_tok:
        print(
            f"[sft] completion-only loss: masking {_masked_tok}/{_total_tok} "
            f"({_masked_tok / _total_tok:.0%}) prompt tokens; training on the completion only"
        )
    effective_batch = _train_opt("batch_size", RECIPE.sft.effective_batch)
    # Large-vocab logits sizing: the trl SFT path DISABLES chalk fused CE (install_chalk_kernels(
    # fused_ce=False) below — the #421/#431 logits=None fix) and Liger is off, so SFTTrainer ALWAYS
    # materialises [micro_batch, seq, vocab] fp32 logits. Size the micro-batch / grad-accum / grad-
    # checkpointing (and the allocator, vram.py) for that UNFUSED path UP FRONT — cap the micro-batch and
    # raise grad-accum to hold the effective batch — instead of sizing fused and fixing it up AFTER the
    # trainer's Accelerator is built (which left the accelerator's grad-accum stale; codex[bot]).
    # revision-aware vocab so the worker sizes the same batch the cost quote priced
    # (cost/spec.py _sft_realized_batch uses resolve_vocab_size on the same (model, revision)).
    _sft_vocab = resolve_vocab_size(model_id, model_revision)
    _sft_fused = False
    per_device_bs, grad_accum = sft_grad_accum(
        effective_batch, seq_len=sft_max_len, vocab=_sft_vocab, fused=_sft_fused
    )
    if not _sft_fused and per_device_bs < min(effective_batch, 4):
        print(
            f"[sft] large-vocab logits cap: per_device={per_device_bs} grad_accum={grad_accum} "
            f"(seq={sft_max_len}, vocab={_sft_vocab}; realized batch "
            f"{per_device_bs * grad_accum} >= requested {effective_batch})"
        )
    sft_save_default = _train_opt("save_every", 50)
    out_dir = f"/tmp/sft_seed{_w.SEED}"
    resume_ckpt = _w.hf_resume_checkpoint()

    _gc_card_gb, _gc_cap = 0.0, None
    try:
        import torch as _torch_gc

        if _torch_gc.cuda.is_available():
            _gc_card_gb = _torch_gc.cuda.get_device_properties(0).total_memory / 1e9
            _gc_cap = _torch_gc.cuda.get_device_capability(0)
    except Exception:
        _gc_card_gb, _gc_cap = 0.0, None
    _gc_hidden, _gc_layers = _model_arch_dims(model_id, revision=model_revision)
    _gc_active_b = float(getattr(MODELS.get(model_id), "active_params_b", 0.0) or 0.0) or None
    _gc_lora_rank = int(_t.lora_rank if _t and _t.lora_rank else RECIPE.lora.rank)
    _grad_ckpt = grad_checkpointing_on(
        model_id,
        sft_max_len,
        allow_disable=True,
        card_vram_gb=_gc_card_gb,
        capability=_gc_cap,
        active_params_b=_gc_active_b,
        hidden=_gc_hidden,
        num_layers=_gc_layers,
        fused_ce=_sft_fused,
        per_device_bs=per_device_bs,
        lora_rank=_gc_lora_rank,
        revision=model_revision,
    )

    max_steps = int(_t.max_steps or 0) if _t else 0
    save_at_steps = tuple(getattr(_t, "save_at_steps", ()) or ())
    cfg_kwargs = {
        "output_dir": out_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": per_device_bs,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": sft_lr,
        "warmup_ratio": RECIPE.sft.warmup_frac,
        "logging_steps": 10,
        "save_steps": sft_save_default,
        "save_total_limit": 1,
        # save_only_model=False: saves optimizer/scheduler state so a resumed worker truly continues
        # instead of re-initializing Adam moments. The deployable snapshot strips trainer state separately.
        "save_only_model": False,
        "max_length": sft_max_len,
        "bf16": True,
        "report_to": _w.wandb_report_to(),
        "run_name": _w.wandb_run_name(),
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": True,
        "seed": backend_seed(_w.SEED),
        "gradient_checkpointing": _grad_ckpt,
        # MoE / GatedDeltaNet hybrids re-dispatch tokens (MoE router) or lay out custom-kernel
        # saved tensors differently on recompute, so non-reentrant checkpointing's metadata-equality
        # assert fires on the FIRST backward (Qwen3.6-35B-A3B). Mirror the GRPO path (rl.py, #429/#432)
        # and pick REENTRANT recompute for those; dense models keep the faster non-reentrant path.
        "gradient_checkpointing_kwargs": {"use_reentrant": grpo_use_reentrant(model_id)},
        "completion_only_loss": True,
        # remove_unused_columns=False: HF Trainer would otherwise drop completion_mask before
        # collation, silently reverting all paths to full-transcript loss.
        "remove_unused_columns": False,
        "optim": fused_optim_name(),
    }
    _sft_config_fields = set(getattr(TRLSFTConfig, "__dataclass_fields__", {}))
    if "use_liger_kernel" in _sft_config_fields:
        cfg_kwargs["use_liger_kernel"] = False
    if max_steps > 0:
        cfg_kwargs["max_steps"] = max_steps
    configure_trainer_save_schedule(cfg_kwargs, save_at_steps)
    # TRL 'bfd' packing: boundary-correct only under FA2/FA3 varlen (SDPA cross-contaminates).
    # GDN hybrids can't use bfd (no seq_idx to reset causal conv); they pack via the varlen collator.
    _pure_attn = model_is_pure_attention(model_id, revision=model_revision)
    _gdn = model_is_gdn_hybrid(model_id, revision=model_revision)
    _bfd_block_count: int | None = None
    _fa_ok = _flash_attn_available()
    if _fa_ok and _pure_attn:
        cfg_kwargs["packing"] = True
        print("[sft] example packing enabled (FA2 varlen)")
    elif _fa_ok and _gdn:
        print(
            "[sft] TRL bfd packing NOT used for the GatedDeltaNet hybrid (bfd can't reset the conv); "
            "the cu_seqlens/seq_idx varlen collator handles its packing when both kernels are present."
        )
    else:
        _bfd_why = (
            "flash_attn not importable" if not _fa_ok else "arch not bfd-safe under FA2 varlen"
        )
        print(
            f"[sft] TRL bfd (FA2) packing not used ({_bfd_why}); the SDPA-mask path decides packing below."
        )
    if _memory_mode(model_id, sft_max_len, revision=model_revision):
        print("[sft] chalk standalone fused kernels scheduled after trainer build")
    _attn = optimal_attn_impl()
    # When bfd packing is on, ensure a varlen-capable flash impl; sdpa cross-contaminates packed examples.
    # _attn=="sdpa" (Blackwell): disable bfd — don't force FA2 (unverified SASS). SDPA-mask path below still packs.
    # _attn is None (Hopper without FA3): force FA2 if available, else drop packing.
    if cfg_kwargs.get("packing"):
        if _attn in ("flash_attention_2", "flash_attention_3"):
            print(f"[sft] attn_implementation={_attn} (packing boundary-correct varlen)")
        elif _attn == "sdpa":
            cfg_kwargs["packing"] = False
            print(
                "[sft] packing disabled: selected attn_implementation=sdpa (no varlen flash backend)"
            )
        elif _fa_ok:
            _attn = "flash_attention_2"
            print("[sft] attn_implementation=flash_attention_2 (packing boundary-correct varlen)")
        else:
            cfg_kwargs["packing"] = False
            print(
                "[sft] packing disabled: no varlen flash backend (FA2/FA3) available -> plain SDPA"
            )
    if cfg_kwargs.get("packing"):
        _bfd_ids = [r["input_ids"] for r in _pretok]
        _bfd_block_count = max(1, len(pack_token_ids(_bfd_ids, sft_max_len)))
        _bfd_ex = len(_bfd_ids) / _bfd_block_count
        cfg_kwargs["gradient_accumulation_steps"] = max(
            1, math.ceil(effective_batch / max(1.0, per_device_bs * _bfd_ex))
        )
        print(
            f"[sft] bfd packing: ~{_bfd_ex:.1f} ex/block -> pd={per_device_bs} "
            f"ga={cfg_kwargs['gradient_accumulation_steps']} (effective batch kept ~{effective_batch} ex)"
        )

    # 4D block-diagonal SDPA mask packing: for pure-attn models on plain SDPA (e.g. sm120 RTX 5090).
    # Flash varlen would silently IGNORE the 4D mask — so we downgrade to sdpa when using this path.
    # GDN hybrids need the next branch (mask alone can't reset their causal conv state).
    _collator = None
    # Cap at 16384: dense [B,1,T,T] mask is O(T^2) memory; above this packing gains little anyway.
    _PACK_MASK_MAX_LEN = 16384
    _mask_pack_ok = sft_max_len <= _PACK_MASK_MAX_LEN
    _sdpa_pack = bool(not cfg_kwargs.get("packing") and _pure_attn and _mask_pack_ok)
    if _sdpa_pack:
        if _attn in ("flash_attention_2", "flash_attention_3"):
            print(
                f"[sft] packing under SDPA: downgrading {_attn} -> sdpa (a flash kernel ignores the 4D mask)"
            )
        _attn = "sdpa"
        cfg_kwargs["packing"] = False  # we own the packing; TRL must not also pack
        _dk = dict(cfg_kwargs.get("dataset_kwargs") or {})
        _dk["skip_prepare_dataset"] = True
        cfg_kwargs["dataset_kwargs"] = _dk
        _ids = [r["input_ids"] for r in _pretok]
        _cmask = [r["completion_mask"] for r in _pretok]
        _packed_rows = pack_token_ids(_ids, sft_max_len, completion_masks=_cmask)
        ds = Dataset.from_list(_packed_rows)
        _collator = BlockDiagonalCollator(pad_token_id=tok.pad_token_id)
        _pd_pack, _ = sft_grad_accum(
            effective_batch,
            seq_len=sft_max_len,
            vocab=_sft_vocab,  # reuse the revision-aware vocab resolved above (single source of truth)
            fused=_sft_fused,
        )
        # Cap pd so the dense [pd,1,T,T] mask stays <=512MB (only bites past ~12k tokens).
        _pd_pack = max(1, min(_pd_pack, (512 * 1024 * 1024) // (sft_max_len * sft_max_len)))
        _ex_per_block = len(_ids) / max(1, len(_packed_rows))
        cfg_kwargs["per_device_train_batch_size"] = _pd_pack
        cfg_kwargs["gradient_accumulation_steps"] = max(
            1, math.ceil(effective_batch / max(1.0, _pd_pack * _ex_per_block))
        )
        print(
            "[sft] true token packing ENABLED (4D block-diagonal SDPA mask): "
            f"{len(_ids)} examples -> {len(_packed_rows)} blocks (~{_ex_per_block:.1f} ex/block, "
            f"{packing_efficiency(_packed_rows, sft_max_len):.0%} dense) of <= {sft_max_len} tok; "
            f"pd={_pd_pack} ga={cfg_kwargs['gradient_accumulation_steps']} (effective batch kept "
            f"~{effective_batch} ex); no flash-attn / no flex_attention"
        )
    elif (
        not cfg_kwargs.get("packing")
        and _gdn
        and gdn_packing_available(model_id, revision=model_revision)
        and _mask_pack_ok
    ):
        # GDN hybrid: 4D mask for full-attn layers + cu_seqlens/seq_idx to reset DeltaNet recurrence.
        # Flash varlen would ignore the 4D mask — downgrade to sdpa for the full-attn layers.
        if _attn in ("flash_attention_2", "flash_attention_3"):
            print(
                f"[sft] GDN packing under SDPA: downgrading {_attn} -> sdpa for the full-attn layers"
            )
        _attn = "sdpa"
        cfg_kwargs["packing"] = False
        _dk = dict(cfg_kwargs.get("dataset_kwargs") or {})
        _dk["skip_prepare_dataset"] = True
        cfg_kwargs["dataset_kwargs"] = _dk
        _ids = [r["input_ids"] for r in _pretok]
        _cmask = [r["completion_mask"] for r in _pretok]
        _packed_rows = pack_token_ids(_ids, sft_max_len, completion_masks=_cmask)
        ds = Dataset.from_list(_packed_rows)
        _collator = BlockDiagonalCollator(pad_token_id=tok.pad_token_id, emit_varlen=True)
        # cu_seqlens spans one block -> per-device=1; re-derive grad_accum to keep effective batch in examples.
        _ex_per_block = len(_ids) / max(1, len(_packed_rows))
        cfg_kwargs["per_device_train_batch_size"] = 1
        cfg_kwargs["gradient_accumulation_steps"] = max(
            1, math.ceil(effective_batch / max(1.0, _ex_per_block))
        )
        print(
            "[sft] true token packing ENABLED for GatedDeltaNet hybrid (4D mask + cu_seqlens/seq_idx "
            f"varlen): {len(_ids)} examples -> {len(_packed_rows)} blocks (~{_ex_per_block:.1f} "
            f"ex/block, {packing_efficiency(_packed_rows, sft_max_len):.0%} dense) of <= {sft_max_len} "
            f"tok; pd=1 ga={cfg_kwargs['gradient_accumulation_steps']} (effective batch kept ~{effective_batch} ex)"
        )
    elif not cfg_kwargs.get("packing") and (_pure_attn or _gdn) and not _mask_pack_ok:
        print(
            f"[sft] packing stays OFF: max_length {sft_max_len} > {_PACK_MASK_MAX_LEN} — the dense "
            "O(T^2) block-diagonal mask gets too large at long context (unpacked is more memory-"
            "efficient there, and long rows already fill a block)."
        )
    elif not cfg_kwargs.get("packing") and not _pure_attn:
        _why = (
            "hybrid GatedDeltaNet but the fla/causal_conv1d varlen kernels aren't both importable"
            if _gdn
            else "non-full-attention arch (e.g. sliding-window) a block-diagonal mask can't pack"
        )
        print(
            f"[sft] packing stays OFF: {_why}. (Pure full-attention models pack via the SDPA mask.)"
        )
    # Explicit bf16 + device_map=None: transformers-5 string loading otherwise falls back to fp32
    # (2x VRAM) or accelerate-offloads to meta ("expected device meta but got cuda:0" in backward).
    model_init_kwargs = {
        "dtype": "bfloat16",
        "device_map": None,
        **_w.model_revision_kwargs(model_revision),
    }
    if _attn:
        model_init_kwargs["attn_implementation"] = _attn
    cfg_kwargs["model_init_kwargs"] = model_init_kwargs
    examples_per_update = int(cfg_kwargs["per_device_train_batch_size"]) * int(
        cfg_kwargs["gradient_accumulation_steps"]
    )
    derived_steps = sft_update_steps(
        epochs=epochs,
        example_count=len(ds),
        examples_per_update=examples_per_update,
        packed_block_count=_bfd_block_count if cfg_kwargs.get("packing") else None,
    )
    update_horizon = resolve_update_horizon(derived_steps, max_steps)
    validate_save_steps(save_at_steps, update_horizon)
    cfg = TRLSFTConfig(**cfg_kwargs)

    # LoRA+ (arXiv 2402.12354): B-matrix LR ratio=16, measured -52% train loss. Must override
    # create_optimizer because TRL builds the model inside __init__ (can't pre-build the optimizer).
    _lp_ratio = 16
    _SFT = SFTTrainer
    if _lp_ratio > 1:

        class _SFT(SFTTrainer):  # local LoRA+ subclass
            _loraplus_applied = False  # True only once the LoRA+ grouping actually installs

            def create_optimizer(self):
                if self.optimizer is None:
                    try:
                        from peft.optimizers import create_loraplus_optimizer

                        # Use .value not str(): OptimizerNames enum str() includes the class name.
                        opt_cls, extra = loraplus_optimizer_cls(
                            getattr(self.args.optim, "value", self.args.optim)
                        )
                        # Explicitly forward betas/eps/weight_decay — PEFT doesn't read TrainingArguments.
                        fwd = dict(extra)
                        _betas = (
                            getattr(self.args, "adam_beta1", None),
                            getattr(self.args, "adam_beta2", None),
                        )
                        if None not in _betas:
                            fwd.setdefault("betas", _betas)
                        _eps = getattr(self.args, "adam_epsilon", None)
                        if _eps is not None:
                            fwd.setdefault("eps", _eps)
                        lp_extra: dict[str, object] = {}
                        _wd = getattr(self.args, "weight_decay", None)
                        if _wd is not None:
                            lp_extra["loraplus_weight_decay"] = _wd
                        # lr kwarg name shifted across PEFT versions; fall back to top-level lr=.
                        try:
                            self.optimizer = create_loraplus_optimizer(
                                model=self.model,
                                optimizer_cls=opt_cls,
                                optimizer_kwargs={"lr": self.args.learning_rate, **fwd},
                                loraplus_lr_ratio=_lp_ratio,
                                **lp_extra,
                            )
                        except TypeError:
                            self.optimizer = create_loraplus_optimizer(
                                model=self.model,
                                optimizer_cls=opt_cls,
                                lr=self.args.learning_rate,
                                loraplus_lr_ratio=_lp_ratio,
                                **fwd,
                                **lp_extra,
                            )
                        self._loraplus_applied = True
                        print(
                            f"[lora+] optimizer enabled (B-matrix LR ratio={_lp_ratio}, "
                            f"cls={opt_cls.__name__})"
                        )
                        return self.optimizer
                    except Exception as e:  # never block training on LoRA+ wiring failure
                        print("[lora+] setup failed, falling back to default optimizer:", e)
                return super().create_optimizer()

    # SFTTrainer.__init__ can block 10-15 min (FA2 JIT). liveness_heartbeat keeps the control plane
    # from recycling the worker. include_torch=False: side-thread torch.cuda telemetry serializes on
    # the CUDA/allocator lock held by the init thread and can freeze the heartbeat itself.
    with liveness_heartbeat("sft_initializing"):
        seed_training_rngs(_w.SEED)
        sft_model = _w.prepare_fresh_lora_base(
            model_id,
            model_id,
            model_init_kwargs,
            phase="sft",
            model_revision=model_revision,
        )
        if not isinstance(sft_model, str):
            cfg.model_init_kwargs = None
        trainer = _SFT(
            model=sft_model,
            args=cfg,
            train_dataset=ds,
            peft_config=_w.make_lora(model_id),
            processing_class=tok,
            data_collator=_collator,
            callbacks=[
                _w.make_sft_heartbeat_callback(),
                _w.make_checkpoint_upload_callback(save_at_steps),
            ],
        )
        # fused_ce=False: flce returns logits=None, but trl's SFTTrainer.compute_loss reads outputs.logits
        # (it only skips them under use_liger_kernel=True, which would make trl apply Liger and clash with
        # chalk). So the trl SFT path keeps flce OFF and materialises [micro_batch, seq, vocab] logits —
        # otherwise every large-vocab Qwen3.5 SFT crashes with "'NoneType' object is not subscriptable" once
        # chalk actually applies flce (#421). Because flce is ALWAYS off here, _sft_fused is False above and
        # the micro-batch / grad-accum / grad-checkpointing (and the allocator, vram.py) were already sized
        # for the materialised-logits path UP FRONT — no post-init batch/grad-accum fixup is needed, which
        # would otherwise mutate grad_accum after the trainer's Accelerator was built from the old value
        # (codex[bot]). The custom GRPO/opd loops read the fused loss directly, so they keep flce on.
        # inside the liveness wrap: chalk's kernel install can JIT-compile, silent for minutes.
        _chalk_report = install_chalk_kernels(getattr(trainer, "model", None), fused_ce=False)
    _chalk_active = active_kernels(_chalk_report)

    _reset_peak_gpu()
    _gpu_sampler = _GpuPeakSampler().start()
    t_train = time.time()
    # progress + progress_step: step advances emit REAL heartbeats (and stamp the step), so the
    # daemon can never starve the provider's stall clock by winning the throttled upload slot with
    # a bare liveness ping while training is healthy.
    with (
        liveness_heartbeat(
            "sft_step",
            progress=lambda: int(getattr(trainer.state, "global_step", 0) or 0),
            progress_step=True,
        ),
        _sdpa_cudnn_ctx(_attn),
    ):
        trainer.train(resume_from_checkpoint=resume_ckpt)
    train_wall = time.time() - t_train
    sft_peak_gpu_gb = _peak_gpu_gb()
    sft_device_peak_gpu_gb = _gpu_sampler.stop_gb()

    _final_step = int(getattr(trainer.state, "global_step", 0) or 0)
    if sft_under_ran(_final_step, update_horizon, max_steps):
        raise RuntimeError(
            f"sft completed {_final_step}/{update_horizon} requested optimizer updates"
        )
    # adapter save + required upload can take minutes on a slow HF; keep the heartbeat fresh.
    # keepalive=True: _final_step is CONSTANT here (training is done), so without it every finalize
    # ping is a bare liveness that does NOT advance the provider's stall clock — a finalize outlasting
    # STALL_AFTER_S (1500s) would be killed at the finish line. keepalive forces a REAL heartbeat/tick.
    # progress_step stamps the final step on every finalize heartbeat so a cancel landing in this
    # window still bills the actual steps trained (actual_steps_run reads last_heartbeat.step).
    with liveness_heartbeat(
        "sft_finalizing", progress=lambda: _final_step, progress_step=True, keepalive=True
    ):
        adapter_dir = f"{out_dir}/adapter"
        _w.stamp_adapter_provenance(trainer.model, model_id, model_revision)
        trainer.model.save_pretrained(adapter_dir)
        tok.save_pretrained(adapter_dir)
        _w.write_base_model_provenance(adapter_dir, model_id, model_revision)
        _w.hf_upload_folder(adapter_dir, "adapter", required=True)
        # preserve the final checkpoint only when exact save steps are not configured.
        if final_save_due(_final_step, save_at_steps):
            _w.publish_deployable_checkpoint(adapter_dir, _final_step)
    _w.heartbeat("sft_trained", train_wall=train_wall, step=_final_step, gpu=gpu_diagnostics())

    train_tokens = sft_completed_train_tokens(
        _total_tok,
        epochs,
        derived_steps,
        _final_step,
    )
    _w.write_train_meta(
        phase="sft",
        adapter_dir=adapter_dir,
        model_id=model_id,
        train_wall=train_wall,
        setup_seconds=setup_seconds,
        train_tokens=train_tokens,
        generated_tokens=0,
        notes={
            "epochs": epochs,
            "resumed": bool(resume_ckpt),
            "download_seconds": download_seconds,
            "hf_transfer": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", ""),
            "thinking": _w.THINKING,
            "gradient_checkpointing": _grad_ckpt,
            "loss_curve": _metric_curve(trainer, "loss"),
            "peak_gpu_gb": sft_peak_gpu_gb,
            # device_peak_gpu_gb includes bnb managed optimizer pages; peak_gpu_gb does not.
            "device_peak_gpu_gb": sft_device_peak_gpu_gb,
            # Unwrap AcceleratedOptimizer (transformers 5.x) to get the underlying class name.
            "loraplus_optim": (
                type(getattr(trainer.optimizer, "optimizer", trainer.optimizer)).__name__
                if getattr(trainer, "optimizer", None) is not None
                else loraplus_optimizer_cls(fused_optim_name())[0].__name__
            ),
            "loraplus_applied": getattr(trainer, "_loraplus_applied", False),
            "chalk_kernels": _chalk_active or None,
            **_w.wandb_run_info(),
        },
    )
    free_gpu(trainer)
