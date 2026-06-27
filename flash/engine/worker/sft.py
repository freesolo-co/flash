"""SFT training path (TRL SFTTrainer) for the fine-tuning worker."""

from __future__ import annotations

import math
import os
import random
import time

from flash.engine.chalk_kernels import active_kernels, install_chalk_kernels
from flash.engine.recipe import RECIPE
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
    liger_on,
    loraplus_optimizer_cls,
    optimal_attn_impl,
    setup_perf_backends,
    wait_for_gpu,
)


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


def run_sft():
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    env = _w.require_active_env()
    t_start = time.time()
    _w.heartbeat("sft_start", gpu=gpu_diagnostics())
    wait_for_gpu(_w.JOB_SPEC.gpu.type if _w.JOB_SPEC else None)
    setup_perf_backends()
    model_id = _w.JOB_SPEC.model if _w.JOB_SPEC else RECIPE.hf_model_id
    download_seconds = _w.prefetch_model(model_id)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    _t = _w.JOB_SPEC.train if _w.JOB_SPEC else None

    def _train_opt(name, default):
        val = getattr(_t, name, None) if _t else None
        return val if val is not None else default

    train = env.dataset()
    rng = random.Random(_w.SEED)
    rng.shuffle(train)
    max_examples = int(_train_opt("max_examples", 0) or 0)
    if max_examples > 0:
        train = train[:max_examples]
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
                    msgs, tokenize=False, add_generation_prompt=False, enable_thinking=_w.THINKING
                ),
                "prompt_text": tok.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=_w.THINKING
                ),
            }
        )
    if multiturn_targets:
        print(f"[sft] multi-turn SFT: {multiturn_targets}/{len(train)} rows train on a full target transcript")
    elif getattr(env, "multi_turn", False):
        print(
            "[sft][warn] this is a multi-turn Freesolo environment but no row ships a multi-turn "
            "target completion; SFT collapses to a single assistant turn per row (tool/env turns "
            "ignored). Provide target transcripts (output={\"messages\": [...]}) for proper multi-turn SFT."
        )
    if _w.THINKING and not any("<think>" in t["text"] for t in texts[:256]):
        print(
            "WARN: thinking mode is ON but no sampled SFT target contains a <think> "
            "trace — training on non-reasoning targets teaches the model to SKIP "
            "thinking. Use a dataset with reasoning traces, or set thinking = false."
        )
    setup_seconds = time.time() - t_start
    _w.heartbeat("sft_model_load", setup_seconds=setup_seconds, gpu=gpu_diagnostics())

    epochs = int(_train_opt("epochs", RECIPE.sft.num_epochs))
    from flash.catalog import vocab_size_for
    from flash.engine.vram import resolve_params_b, sft_grad_accum, sft_logits_fused

    sft_lr = _train_opt("learning_rate", RECIPE.sft.learning_rate)
    sft_max_len = _train_opt(
        "max_length", RECIPE.sft.max_seq_len_thinking if _w.THINKING else RECIPE.sft.max_seq_len
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
    # Large-vocab OOM guard: without fused CE, SFTTrainer materializes [per_device, seq, vocab] fp32
    # logits; cap micro-batch so they fit, raise grad-accum to keep effective batch unchanged.
    _sft_params_b = resolve_params_b(model_id)
    _sft_vocab = vocab_size_for(model_id)
    _sft_fused = sft_logits_fused(_sft_params_b, sft_max_len) and liger_on(
        _memory_mode(model_id, sft_max_len)
    )
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

    max_steps = int(_train_opt("max_steps", 0) or 0)
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
        "seed": _w.SEED,
        "gradient_checkpointing": grad_checkpointing_on(model_id, sft_max_len),
        # use_reentrant=False: required by TRL for correct grad flow through LoRA adapters.
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "completion_only_loss": True,
        # remove_unused_columns=False: HF Trainer would otherwise drop completion_mask before
        # collation, silently reverting all paths to full-transcript loss.
        "remove_unused_columns": False,
        "optim": fused_optim_name(),
    }
    if max_steps > 0:
        cfg_kwargs["max_steps"] = max_steps
    # TRL 'bfd' packing: boundary-correct only under FA2/FA3 varlen (SDPA cross-contaminates).
    # GDN hybrids can't use bfd (no seq_idx to reset causal conv); they pack via the varlen collator.
    _pure_attn = model_is_pure_attention(model_id)
    _gdn = model_is_gdn_hybrid(model_id)
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
        _bfd_why = "flash_attn not importable" if not _fa_ok else "arch not bfd-safe under FA2 varlen"
        print(f"[sft] TRL bfd (FA2) packing not used ({_bfd_why}); the SDPA-mask path decides packing below.")
    if liger_on(_memory_mode(model_id, sft_max_len)):
        cfg_kwargs["use_liger_kernel"] = True
        print("[sft] liger fused kernels enabled")
    _attn = optimal_attn_impl()
    # When bfd packing is on, ensure a varlen-capable flash impl; sdpa cross-contaminates packed examples.
    # _attn=="sdpa" (Blackwell): disable bfd — don't force FA2 (unverified SASS). SDPA-mask path below still packs.
    # _attn is None (Hopper without FA3): force FA2 if available, else drop packing.
    if cfg_kwargs.get("packing"):
        if _attn in ("flash_attention_2", "flash_attention_3"):
            print(f"[sft] attn_implementation={_attn} (packing boundary-correct varlen)")
        elif _attn == "sdpa":
            cfg_kwargs["packing"] = False
            print("[sft] packing disabled: selected attn_implementation=sdpa (no varlen flash backend)")
        elif _fa_ok:
            _attn = "flash_attention_2"
            print("[sft] attn_implementation=flash_attention_2 (packing boundary-correct varlen)")
        else:
            cfg_kwargs["packing"] = False
            print("[sft] packing disabled: no varlen flash backend (FA2/FA3) available -> plain SDPA")
    if cfg_kwargs.get("packing"):
        _bfd_ids = [r["input_ids"] for r in _pretok]
        _bfd_ex = len(_bfd_ids) / max(1, len(pack_token_ids(_bfd_ids, sft_max_len)))
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
            print(f"[sft] packing under SDPA: downgrading {_attn} -> sdpa (a flash kernel ignores the 4D mask)")
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
            effective_batch, seq_len=sft_max_len, vocab=vocab_size_for(model_id),
            fused=bool(cfg_kwargs.get("use_liger_kernel")),
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
    elif not cfg_kwargs.get("packing") and _gdn and gdn_packing_available(model_id) and _mask_pack_ok:
        # GDN hybrid: 4D mask for full-attn layers + cu_seqlens/seq_idx to reset DeltaNet recurrence.
        # Flash varlen would ignore the 4D mask — downgrade to sdpa for the full-attn layers.
        if _attn in ("flash_attention_2", "flash_attention_3"):
            print(f"[sft] GDN packing under SDPA: downgrading {_attn} -> sdpa for the full-attn layers")
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
        cfg_kwargs["gradient_accumulation_steps"] = max(1, math.ceil(effective_batch / max(1.0, _ex_per_block)))
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
        print(f"[sft] packing stays OFF: {_why}. (Pure full-attention models pack via the SDPA mask.)")
    # Explicit bf16 + device_map=None: transformers-5 string loading otherwise falls back to fp32
    # (2x VRAM) or accelerate-offloads to meta ("expected device meta but got cuda:0" in backward).
    mik = {"dtype": "bfloat16", "device_map": None}
    if _attn:
        mik["attn_implementation"] = _attn
    cfg_kwargs["model_init_kwargs"] = mik
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
        trainer = _SFT(
            model=model_id,
            args=cfg,
            train_dataset=ds,
            peft_config=_w.make_lora(model_id),
            processing_class=tok,
            data_collator=_collator,
            callbacks=[_w.make_sft_heartbeat_callback(), _w.make_checkpoint_upload_callback()],
        )
    _chalk_report = install_chalk_kernels(getattr(trainer, "model", None))

    _reset_peak_gpu()
    _gpu_sampler = _GpuPeakSampler().start()
    t_train = time.time()
    with liveness_heartbeat("sft_step"), _sdpa_cudnn_ctx(_attn):
        trainer.train(resume_from_checkpoint=resume_ckpt)
    train_wall = time.time() - t_train
    sft_peak_gpu_gb = _peak_gpu_gb()
    sft_device_peak_gpu_gb = _gpu_sampler.stop_gb()

    adapter_dir = f"{out_dir}/adapter"
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    _w.hf_upload_folder(adapter_dir, "adapter", required=True)
    # Ensure `flash deploy --step <final>` always resolves: save_steps may not align with the last step.
    _final_step = int(getattr(trainer.state, "global_step", 0) or 0)
    if _final_step:
        _w.publish_deployable_checkpoint(adapter_dir, _final_step)
    _w.heartbeat("sft_trained", train_wall=train_wall, gpu=gpu_diagnostics())

    train_tokens = _total_tok * epochs
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
            # Persist loss curve: trainer_state.json is only written on save_step boundaries.
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
            "chalk_kernels": active_kernels(_chalk_report) or None,
            **_w.wandb_run_info(),
        },
    )
    free_gpu(trainer)
