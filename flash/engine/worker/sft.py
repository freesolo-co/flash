"""SFT training path (TRL SFTTrainer) for the fine-tuning worker.

Run-scoped state (``JOB_SPEC``/``SEED``/``THINKING``) and the worker-namespace helpers
(``heartbeat``/``prefetch_model``/``make_lora``/``wandb_*``/``hf_*``/``write_train_meta`` ...) are
read THROUGH the worker package (``_w.<name>``) at CALL time so the module-level monkeypatch
contract holds. The pure perf/lora/kernel probes are imported directly (they read no run state).
"""

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


def run_sft():
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    env = _w.require_active_env()  # fail loudly (not AttributeError: NoneType) on the no-JobSpec path
    t_start = time.time()
    _w.heartbeat("sft_start", gpu=gpu_diagnostics())
    # SFT on a multi-turn env: rows whose target completion is a full trajectory train on the whole
    # transcript (proper multi-turn SFT, handled below); rows with a single-turn target completion
    # collapse to one assistant turn. Warn only for the collapsing case (computed during the
    # dataset build below), not unconditionally.
    wait_for_gpu(_w.JOB_SPEC.gpu.type if _w.JOB_SPEC else None)
    setup_perf_backends()
    model_id = _w.JOB_SPEC.model if _w.JOB_SPEC else RECIPE.hf_model_id
    download_seconds = _w.prefetch_model(model_id)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Build SFT text dataset (seeded shuffle for reproducibility)
    train = env.dataset()
    rng = random.Random(_w.SEED)
    rng.shuffle(train)
    max_examples = int(
        _w.JOB_SPEC.train.max_examples or 0
        if _w.JOB_SPEC and _w.JOB_SPEC.train and _w.JOB_SPEC.train.max_examples is not None
        else 0
    )
    if max_examples > 0:
        train = train[:max_examples]
    texts = []
    multiturn_targets = 0
    for ex in train:
        # The env (via the freesolo-sdk Environment.sft_completion) owns the target completion: the
        # full multi-turn target trajectory (assistant turns + tool calls + tool results + replies)
        # when the row ships one, else a single target assistant turn. Training on the whole
        # transcript is what makes SFT actually multi-turn (the tool-call protocol + replies) — the
        # warm start the GRPO recipe expects. A >1-message completion is a multi-turn trajectory.
        completion = env.sft_completion(ex)
        if len(completion) > 1:  # a multi-turn target trajectory (vs a single assistant turn)
            multiturn_targets += 1
        msgs = [*env.prompt_messages(ex), *completion]
        texts.append(
            {
                "text": tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False, enable_thinking=_w.THINKING
                )
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
    ds = Dataset.from_list(texts)

    setup_seconds = time.time() - t_start
    _w.heartbeat("sft_model_load", setup_seconds=setup_seconds, gpu=gpu_diagnostics())

    # Epochs come from the run's [train] epochs (already in JOB_SPEC), else the recipe default.
    epochs = int(
        _w.JOB_SPEC.train.epochs
        if _w.JOB_SPEC and _w.JOB_SPEC.train.epochs is not None
        else RECIPE.sft.num_epochs
    )
    # SDK [train] knobs override the recipe default.
    from flash.catalog import vocab_size_for
    from flash.engine.vram import resolve_params_b, sft_grad_accum, sft_logits_fused

    _t = _w.JOB_SPEC.train if _w.JOB_SPEC else None
    sft_lr = _t.learning_rate if _t and _t.learning_rate is not None else RECIPE.sft.learning_rate
    sft_max_len = (
        _t.max_length
        if _t and _t.max_length is not None
        else (RECIPE.sft.max_seq_len_thinking if _w.THINKING else RECIPE.sft.max_seq_len)
    )
    # batch_size is the GLOBAL/effective batch; sft_grad_accum sizes the per-device micro-batch +
    # grad-accum to realize it (shared with the cost estimator's step count, see engine.vram).
    effective_batch = (
        _t.batch_size if _t and _t.batch_size is not None else RECIPE.sft.effective_batch
    )
    # Large-vocab OOM guard: when the fused CE (Liger) is OFF, the SFTTrainer materializes the full
    # [per_device, seq, vocab] fp32 logits + grad — at Qwen3.5's ~248k vocab a 0.8B SFT OOM'd a
    # 24 GB card in backward. Cap the per-device micro-batch by the real model vocab + seq so those
    # logits stay within the logits budget; grad-accum rises to keep the effective batch unchanged
    # (the SFT mirror of rl_per_device_comps' GRPO cap). fused mirrors liger_on(_memory_mode(...))
    # below, so the cap binds exactly when the worker won't fuse the CE.
    _sft_params_b = resolve_params_b(model_id)  # catalog stat else HF safetensors (open models)
    _sft_vocab = vocab_size_for(model_id)
    # Actual fused-CE decision == what `use_liger_kernel` is set from below (line ~879). sft_logits_fused
    # is the offline size/ctx mirror; liger_on(...) adds the runtime CUDA + liger_kernel-importable
    # check, so the cap binds exactly when the fused CE is NOT really taken.
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
    sft_save_default = _t.save_every if _t and _t.save_every is not None else 50
    out_dir = f"/tmp/sft_seed{_w.SEED}"
    resume_ckpt = _w.hf_resume_checkpoint()

    # [train].max_steps>0 caps optimizer steps (used by the cheap pre-flight smoke).
    max_steps = int(_t.max_steps or 0 if _t and _t.max_steps is not None else 0)
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
        # Resumable checkpoints: save the optimizer / scheduler / RNG state alongside the (small)
        # LoRA adapter. We DO resume mid-run — make_checkpoint_upload_callback streams each save to
        # HF and a replacement worker calls resume_from_checkpoint(hf_resume_checkpoint()) after a
        # preemption — so without this the resumed run would re-initialize the optimizer (Adam
        # moments) and LR schedule instead of truly continuing. For LoRA the optimizer state is tiny
        # (it covers only the trainable adapter params), so the save spike is negligible. The
        # deployable per-step snapshot (publish_deployable_checkpoint) strips this trainer state
        # separately, so serving still gets adapter-only files.
        "save_only_model": False,
        "max_length": sft_max_len,
        "bf16": True,
        "report_to": _w.wandb_report_to(),  # W&B when WANDB_API_KEY present (restored post-flash-migration)
        "run_name": _w.wandb_run_name(),
        # Dataloader parallelism: overlap host-side collation/tokenization with GPU compute so a
        # real (large) training set isn't dataloader-bound. Pure throughput, zero quality change.
        # Negligible on the tiny benchmark (pre-tokenized, in-memory); a real win at production
        # dataset sizes.
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": True,
        "seed": _w.SEED,
        "gradient_checkpointing": grad_checkpointing_on(model_id, sft_max_len),
        # Non-reentrant checkpointing: composes cleanly with autograd hooks (verl #3629) and is
        # required by TRL for correct grad flow through the LoRA adapters.
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "completion_only_loss": False,
        # Optimizer: 8-bit paged AdamW (int8 state paged to host RAM -> fits a smaller GPU).
        "optim": fused_optim_name(),
    }
    if max_steps > 0:
        cfg_kwargs["max_steps"] = max_steps
    # Example packing: concatenate short examples into full max_length sequences so a batch isn't
    # mostly pad tokens — PR #174 measured a 4.4-10.7x SFT speedup (h100 8.2x, 4090 10.7x) because
    # instruction targets are far shorter than max_seq_len; unpacked batches waste most of their
    # FLOPs on padding. TRL's 'bfd' strategy makes padding-free batches whose example boundaries are
    # honored ONLY by an attention impl that reads them — under plain SDPA packed examples
    # cross-contaminate (silent quality loss). The boundary-correct backend is FlashAttention-2
    # varlen (reads position_ids), which the worker image bakes in best-effort: Dockerfile.worker
    # installs FLASH_ATTN_SPEC (a community cu128/torch2.10/cp312 wheel preferred, source build as a
    # fallback) and tolerates a build failure -> SDPA. So _fa_ok is True whenever that install landed;
    # packing is ON then (varlen keeps 'bfd' example boundaries correct). If the best-effort install
    # failed, _fa_ok is False and we SKIP packing — without a boundary-correct attn backend examples
    # would cross-contaminate under SDPA.
    # Pure full-attention vs GatedDeltaNet hybrid (Qwen3.5/3.6) — probed ONCE here and reused across
    # the whole packing decision (each probe reads the cached HF config). TRL 'bfd' packing keeps
    # example boundaries via position_ids that a varlen attn honors, but it provides NO seq_idx, so it
    # can't reset a GDN hybrid's causal conv -> bfd-packing a GDN model silently cross-contaminates its
    # linear-attention layers. So bfd is enabled for PURE full-attention models only; GDN hybrids pack
    # via the cu_seqlens/seq_idx varlen collator branch below (when their kernels are present).
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
        # FA2 bfd packing not enabled here — either flash_attn isn't importable, or it is but the arch
        # isn't bfd-safe (e.g. sliding-window). This is NOT the final word: the SDPA block-diagonal /
        # GDN-varlen block below may still turn packing on for a pure-attention or GDN-hybrid model.
        _bfd_why = "flash_attn not importable" if not _fa_ok else "arch not bfd-safe under FA2 varlen"
        print(f"[sft] TRL bfd (FA2) packing not used ({_bfd_why}); the SDPA-mask path decides packing below.")
    # Liger fused CE/RMSNorm/RoPE kernels, gated by model size (_memory_mode). The fused linear
    # cross-entropy is the big large-vocab (Qwen3.5 ~248k) memory/throughput win.
    if liger_on(_memory_mode(model_id, sft_max_len)):
        cfg_kwargs["use_liger_kernel"] = True
        print("[sft] liger fused kernels enabled")
    _attn = optimal_attn_impl()  # arch-best FlashAttention (FA3 Hopper / FA2 Ampere·Ada) or SDPA
    # Packing correctness: 'bfd' packed batches are boundary-correct ONLY under a varlen-capable attn
    # (FA2 and FA3 both expose flash_attn_varlen_func; plain SDPA cross-contaminates packed examples).
    # Use the ARCH-BEST flash impl optimal_attn_impl already picked (so Hopper packs under FA3, not
    # FA2). Cases when it did NOT pick a flash impl:
    #   * _attn == "sdpa" (Blackwell sm120 5090/RTX Pro + sm100 B200, the deliberate no-flash archs):
    #     DISABLE TRL bfd packing — they stay plain SDPA; do NOT force FA2 (its Blackwell-SASS coverage
    #     is unverified). The SDPA-mask packing block below still repacks pure-attention models here.
    #   * _attn is None (Hopper without FA3): force FA2 for boundary-correct varlen IF the wheel is
    #     importable; else drop packing rather than silently cross-contaminate.
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

    # --- True token packing via a 4D block-diagonal SDPA mask (no flash-attn / no flex) ---------
    # When the run lands on plain SDPA (no varlen flash backend) the block above left packing OFF —
    # notably on sm120 (RTX 5090, flash's DEFAULT GPU), and anywhere the best-effort flash-attn
    # build didn't land. For a PURE full-attention model we can still pack: concatenate examples
    # into max_length blocks and feed a 4D block-diagonal causal mask SDPA honors natively, so
    # packed examples never attend across boundaries (boundary-correct, numerically identical to
    # unpacked — verified on a tiny Qwen3/Llama: |packed-separate| logits ~1e-7). This reclaims the packing
    # throughput win on the default GPU with neither flash-attn nor flex_attention. GatedDeltaNet
    # hybrids (Qwen3.5/3.6) take the NEXT branch instead — a mask alone can't reset their linear-
    # attention state, so they also need the cu_seqlens/seq_idx varlen kwargs.
    _collator = None
    # The mask paths materialize a dense [B, 1, T, T] mask — O(T^2) memory. At very long context that
    # tax (hundreds of MB to >1 GB) can OOM a run that previously fit under memory-efficient SDPA, and
    # packing buys little there anyway (long rows already fill a block). Above this cap, leave packing
    # off (train unpacked, as today). 16384: the dense bf16/bool mask stays <=~256 MB at bsz=1.
    _PACK_MASK_MAX_LEN = 16384
    _mask_pack_ok = sft_max_len <= _PACK_MASK_MAX_LEN
    _sdpa_pack = bool(not cfg_kwargs.get("packing") and _pure_attn and _mask_pack_ok)
    if _sdpa_pack:
        # The 4D mask requires a MASK-READING attn (SDPA). DOWNGRADE any flash impl optimal_attn_impl
        # picked — e.g. FA3 on a Hopper worker whose FA2 wheel didn't build — to SDPA: a flash varlen
        # kernel SILENTLY IGNORES the 4D mask, so packed examples would attend across boundaries. (A
        # bare ``_attn or "sdpa"`` would leave the truthy flash string in place — the bug this avoids.)
        if _attn in ("flash_attention_2", "flash_attention_3"):
            print(f"[sft] packing under SDPA: downgrading {_attn} -> sdpa (a flash kernel ignores the 4D mask)")
        _attn = "sdpa"
        cfg_kwargs["packing"] = False  # we own the packing; TRL must not also pack
        # Hand TRL pre-tokenized, pre-packed rows + our collator: skip its dataset prep and stop the
        # signature-based column pruning from dropping our seq_lengths column before collation.
        _dk = dict(cfg_kwargs.get("dataset_kwargs") or {})
        _dk["skip_prepare_dataset"] = True
        cfg_kwargs["dataset_kwargs"] = _dk
        cfg_kwargs["remove_unused_columns"] = False

        # Tokenize EXACTLY like TRL's non-packed prep (EOS-append parity so the model still learns to
        # stop; batched; truncate to max_length) then bin-pack into <= max_length blocks.
        _tokenized = tokenize_for_packing([t["text"] for t in texts], tok, sft_max_len)
        _packed_rows = pack_token_ids(_tokenized, sft_max_len)
        ds = Dataset.from_list(_packed_rows)
        _collator = BlockDiagonalCollator(pad_token_id=tok.pad_token_id)
        # Memory: re-size the per-device micro-batch (in BLOCKS) for the full-block [pd, max_length,
        # vocab] fp32 logits budget — a no-op under Liger's fused CE. Quality: each block holds
        # ~ex_per_block examples, so KEEP the effective batch in EXAMPLES at the configured value by
        # re-deriving grad_accum from the block count. Without this, packing balloons the effective
        # batch ~ex_per_block-fold (fewer, larger updates -> mild undertraining at the same epochs:
        # an A/B measured +5.2% held-out loss vs unpacked, closed to +0.1% once matched).
        _pd_pack, _ = sft_grad_accum(
            effective_batch, seq_len=sft_max_len, vocab=vocab_size_for(model_id),
            fused=bool(cfg_kwargs.get("use_liger_kernel")),
        )
        # The dense [pd, 1, T, T] bool mask is pd*T^2 bytes — under Liger the logits cap doesn't bind
        # so pd can be 4, and at long context that mask alone is GBs. Cap pd so the mask stays <=512MB
        # (a no-op at short ctx: at T=2048 it allows pd up to ~125; it only bites past ~12k tokens).
        _pd_pack = max(1, min(_pd_pack, (512 * 1024 * 1024) // (sft_max_len * sft_max_len)))
        _ex_per_block = len(_tokenized) / max(1, len(_packed_rows))
        cfg_kwargs["per_device_train_batch_size"] = _pd_pack
        cfg_kwargs["gradient_accumulation_steps"] = max(
            1, math.ceil(effective_batch / max(1.0, _pd_pack * _ex_per_block))
        )
        print(
            "[sft] true token packing ENABLED (4D block-diagonal SDPA mask): "
            f"{len(_tokenized)} examples -> {len(_packed_rows)} blocks (~{_ex_per_block:.1f} ex/block, "
            f"{packing_efficiency(_packed_rows, sft_max_len):.0%} dense) of <= {sft_max_len} tok; "
            f"pd={_pd_pack} ga={cfg_kwargs['gradient_accumulation_steps']} (effective batch kept "
            f"~{effective_batch} ex); no flash-attn / no flex_attention"
        )
    elif not cfg_kwargs.get("packing") and _gdn and gdn_packing_available(model_id) and _mask_pack_ok:
        # GatedDeltaNet hybrid (Qwen3.5/3.6, flash's flagship tier): the 4D block-diagonal mask makes
        # the FULL-attention layers boundary-correct, and the linear-attention (DeltaNet) layers reset
        # their recurrence + causal conv at example boundaries via cu_seq_lens_q (fla kernel) + seq_idx
        # (causal_conv1d). GPU-validated on Qwen3.5-0.8B (RTX 5090): a packed example's output is
        # byte-identical regardless of its neighbors' content (ZERO cross-example leakage); the only
        # diff vs unpacked is benign bf16 GDN-kernel tiling numerics (~0.3 on logits). Gated on BOTH
        # kernels being importable (gdn_packing_available) so a worker without them stays unpacked.
        # Pin SDPA for the full-attn layers (downgrade any flash impl, e.g. FA3 on Hopper — it would
        # ignore the 4D mask); the DeltaNet layers are unaffected (they use cu_seqlens/seq_idx).
        if _attn in ("flash_attention_2", "flash_attention_3"):
            print(f"[sft] GDN packing under SDPA: downgrading {_attn} -> sdpa for the full-attn layers")
        _attn = "sdpa"
        cfg_kwargs["packing"] = False
        _dk = dict(cfg_kwargs.get("dataset_kwargs") or {})
        _dk["skip_prepare_dataset"] = True
        cfg_kwargs["dataset_kwargs"] = _dk
        cfg_kwargs["remove_unused_columns"] = False
        # EOS-append parity + batched + truncated tokenization (same as the unpacked path), then pack.
        _tokenized = tokenize_for_packing([t["text"] for t in texts], tok, sft_max_len)
        _packed_rows = pack_token_ids(_tokenized, sft_max_len)
        ds = Dataset.from_list(_packed_rows)
        _collator = BlockDiagonalCollator(pad_token_id=tok.pad_token_id, emit_varlen=True)
        # cu_seqlens spans ONE packed block, so per-device is a single block; keep the effective batch
        # in EXAMPLES at the configured value via grad-accum (each block holds ~ex_per_block examples —
        # without this the effective batch would balloon ~ex_per_block-fold -> undertraining).
        _ex_per_block = len(_tokenized) / max(1, len(_packed_rows))
        cfg_kwargs["per_device_train_batch_size"] = 1
        cfg_kwargs["gradient_accumulation_steps"] = max(1, math.ceil(effective_batch / max(1.0, _ex_per_block)))
        print(
            "[sft] true token packing ENABLED for GatedDeltaNet hybrid (4D mask + cu_seqlens/seq_idx "
            f"varlen): {len(_tokenized)} examples -> {len(_packed_rows)} blocks (~{_ex_per_block:.1f} "
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
    # Explicit bf16 + no auto device-map: TRL/transformers-5 string loading can
    # otherwise fall back to fp32 (2x VRAM; observed 18.6 GB for a 4.66B model) or
    # accelerate-offload large models to meta ("expected device meta but got
    # cuda:0" in backward on the 9B).
    mik = {"dtype": "bfloat16", "device_map": None}
    if _attn:
        mik["attn_implementation"] = _attn
    cfg_kwargs["model_init_kwargs"] = mik
    cfg = TRLSFTConfig(**cfg_kwargs)

    # LoRA+ (convergence lever, arXiv 2402.12354; always-on: measured -52% train loss in A/B
    # (gpu-bench)): give the LoRA B matrices a higher LR than A (ratio 16). Reported ~2x fewer steps
    # to target at identical per-step FLOPs. TRL builds the model from a string inside __init__, so
    # the optimizer (which needs the instantiated params) can't be pre-built — override
    # create_optimizer to construct it from self.model once it exists.
    _lp_ratio = 16
    _SFT = SFTTrainer
    if _lp_ratio > 1:

        class _SFT(SFTTrainer):  # local LoRA+ subclass
            _loraplus_applied = False  # True only once the LoRA+ grouping actually installs

            def create_optimizer(self):
                if self.optimizer is None:
                    try:
                        from peft.optimizers import create_loraplus_optimizer

                        # Mirror the configured `optim` so LoRA+ and the 8-bit paged optimizer state
                        # coexist (instead of silently forcing fp32 AdamW); see loraplus_optimizer_cls.
                        # .value (not str()): self.args.optim is a TRL OptimizerNames enum whose
                        # str() is "OptimizerNames.PAGED_ADAMW_8BIT"; pass the raw value
                        # ("paged_adamw_8bit") so the 8-bit match works.
                        opt_cls, extra = loraplus_optimizer_cls(
                            getattr(self.args.optim, "value", self.args.optim)
                        )
                        # Forward the TrainingArguments optimizer config that the default HF
                        # create_optimizer path would have applied. Building the optimizer
                        # ourselves means we must replicate it explicitly, or LoRA+ runs would
                        # silently use the optimizer class's own defaults instead of the
                        # configured betas/eps/weight_decay. betas/eps go straight to the optimizer
                        # constructor (alongside any `extra` from loraplus_optimizer_cls);
                        # weight_decay is handled separately below.
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
                        # PEFT does NOT read args.weight_decay; it applies decay via its own LoRA+
                        # param groups, keyed off the loraplus_weight_decay kwarg (which it pops
                        # before constructing the optimizer). Pass it as a top-level kwarg so it
                        # isn't forwarded into the optimizer constructor.
                        lp_extra: dict[str, object] = {}
                        _wd = getattr(self.args, "weight_decay", None)
                        if _wd is not None:
                            lp_extra["loraplus_weight_decay"] = _wd
                        # PEFT's create_loraplus_optimizer forwards extra kwargs to the optimizer;
                        # the lr keyword name has shifted across PEFT versions, so pass it via
                        # optimizer_kwargs (the stable form) and fall back to a top-level lr=.
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
                    except Exception as e:  # never block training on the LoRA+ wiring
                        print("[lora+] setup failed, falling back to default optimizer:", e)
                return super().create_optimizer()

    # Pass model as a string id + tokenizer as processing_class so TRL takes the
    # text/causal-LM path (not the VLM processor path) for this multimodal checkpoint.
    # SFTTrainer.__init__ blocks for 10-15 min on first use (FA2 CUDA kernel JIT compilation);
    # without a heartbeat the control plane can't distinguish this from a real hang and may
    # recycle the worker. A daemon thread pings every 30s so the stall detector stays quiet.
    #
    # include_torch=False: this side-thread heartbeat runs while the main thread is blocked in
    # SFTTrainer.__init__ (CUDA- and allocator-busy). torch.cuda telemetry from a side thread
    # serializes on the CUDA driver / allocator locks held by the init thread and can freeze the
    # heartbeat for the whole init -> false hang. The nvidia-smi-only path (out-of-process, 8s
    # timeout, GIL released during the wait) keeps ticking. Mirrors run_rl's rl_initializing fix.
    with liveness_heartbeat("sft_initializing"):
        trainer = _SFT(
            model=model_id,
            args=cfg,
            train_dataset=ds,
            peft_config=_w.make_lora(model_id),
            processing_class=tok,
            # Our block-diagonal collator on the SDPA-packing path; None elsewhere == TRL default.
            data_collator=_collator,
            callbacks=[_w.make_sft_heartbeat_callback(), _w.make_checkpoint_upload_callback()],
        )
    # Apply chalk's gap-filling kernels (RoPE/LoRA-delta/embedding, like Liger) on the materialized
    # SFT trainer.model — chalk's apply patches the LIVE module, so it must run AFTER TRL builds the
    # model (chalk composes on top of TRL's Liger). No-op unless a FLASH_* kernel flag selects it and
    # freesolo-chalk is installed.
    _chalk_report = install_chalk_kernels(getattr(trainer, "model", None))

    _reset_peak_gpu()  # so peak_gpu_gb reflects the train loop (optimizer-state A/B is measurable)
    _gpu_sampler = _GpuPeakSampler().start()  # true device peak incl. bnb managed optimizer pages
    t_train = time.time()
    # Gap-filler liveness around the train loop: make_sft_heartbeat_callback only emits on on_log
    # (every N steps), so the first step(s) can be silent for minutes. liveness_heartbeat emits
    # best-effort liveness pings for operator visibility/stack dumps; pollers ignore liveness for stalls.
    # See heartbeat.liveness_heartbeat.
    with liveness_heartbeat("sft_step"), _sdpa_cudnn_ctx(_attn):  # cuDNN SDPA on sm120 (no-op else)
        trainer.train(resume_from_checkpoint=resume_ckpt)
    train_wall = time.time() - t_train
    sft_peak_gpu_gb = _peak_gpu_gb()
    sft_device_peak_gpu_gb = _gpu_sampler.stop_gb()

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
    _final_step = int(getattr(trainer.state, "global_step", 0) or 0)
    if _final_step:
        _w.publish_deployable_checkpoint(adapter_dir, _final_step)
    _w.heartbeat("sft_trained", train_wall=train_wall, gpu=gpu_diagnostics())

    train_tokens = int(sum(len(tok(t["text"])["input_ids"]) for t in texts) * epochs)

    # Write train metadata + the completion sentinel (metrics.json/DONE) for this phase.
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
            # Persist the loss curve so a CONVERGENCE A/B (PiSSA / LoRA+ init, etc.) is measurable
            # without a checkpoint: trainer_state.json is only written on a save_step, and the
            # console is only uploaded on failure, so a short successful run otherwise drops its
            # loss history entirely.
            "loss_curve": _metric_curve(trainer, "loss"),
            # Peak torch-allocated GPU memory during the train loop (excludes bnb managed pages, so
            # it overstates the 8-bit saving — use device_peak_gpu_gb for the true footprint).
            "peak_gpu_gb": sft_peak_gpu_gb,
            # True peak device memory (total-free, incl. bnb managed optimizer pages): the honest
            # headline for the fp32-vs-8-bit LoRA+ optimizer A/B.
            "device_peak_gpu_gb": sft_device_peak_gpu_gb,
            # Report the optimizer ACTUALLY built on the trainer, not the planned class: if the
            # LoRA+ create_optimizer override failed, training falls back to TRL's configured
            # optimizer without LoRA+ grouping. loraplus_applied records which path actually ran.
            # Accelerate wraps the optimizer (AcceleratedOptimizer) under transformers 5.x, so unwrap
            # via `.optimizer` to record the underlying PagedAdamW8bit/AdamW the A/B cares about, not
            # the wrapper name.
            "loraplus_optim": (
                type(getattr(trainer.optimizer, "optimizer", trainer.optimizer)).__name__
                if getattr(trainer, "optimizer", None) is not None
                else loraplus_optimizer_cls(fused_optim_name())[0].__name__
            ),
            "loraplus_applied": getattr(trainer, "_loraplus_applied", False),
            # Which chalk gap-filling kernels actually ENGAGED (empty/None = chalk not installed or
            # every kernel fell back) — verifies the chalk stack without the console.
            "chalk_kernels": active_kernels(_chalk_report) or None,
            **_w.wandb_run_info(),
        },
    )
    free_gpu(trainer)
