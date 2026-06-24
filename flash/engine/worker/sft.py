"""SFT training path (TRL SFTTrainer) for the fine-tuning worker.

Run-scoped state (``JOB_SPEC``/``SEED``/``THINKING``) and the worker-namespace helpers
(``heartbeat``/``prefetch_model``/``make_lora``/``wandb_*``/``hf_*``/``write_train_meta`` ...) are
read THROUGH the worker package (``_w.<name>``) at CALL time so the module-level monkeypatch
contract holds. The pure perf/lora/kernel probes are imported directly (they read no run state).
"""

from __future__ import annotations

import os
import random
import threading
import time

from flash.engine.chalk_kernels import active_kernels, install_chalk_kernels
from flash.engine.recipe import RECIPE
from flash.engine.worker._pkg import W as _w
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
    wait_for_gpu()
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
    _fa_ok = _flash_attn_available()
    if _fa_ok:
        cfg_kwargs["packing"] = True
        print("[sft] example packing enabled (FA2 varlen)")
    else:
        print(
            "[sft] packing SKIPPED: flash_attn not importable (best-effort image build failed) "
            "— no boundary-correct attn backend, falling back to SDPA without packing."
        )
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
    #   * _attn == "sdpa" (sm120, the deliberate no-flash exception): DISABLE packing — consumer
    #     Blackwell stays plain SDPA; do NOT force FA2 (its sm120 kernel coverage is unverified).
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
    _sft_init_done = threading.Event()

    def _sft_init_heartbeat() -> None:
        while not _sft_init_done.wait(30.0):
            _w.heartbeat("sft_initializing", gpu=gpu_diagnostics())

    _sft_init_hb = threading.Thread(target=_sft_init_heartbeat, daemon=True)
    _sft_init_hb.start()
    try:
        trainer = _SFT(
            model=model_id,
            args=cfg,
            train_dataset=ds,
            peft_config=_w.make_lora(model_id),
            processing_class=tok,
            callbacks=[_w.make_sft_heartbeat_callback(), _w.make_checkpoint_upload_callback()],
        )
    finally:
        _sft_init_done.set()
    # Apply chalk's gap-filling kernels (RoPE/LoRA-delta/embedding, like Liger) on the materialized
    # SFT trainer.model — chalk's apply patches the LIVE module, so it must run AFTER TRL builds the
    # model (chalk composes on top of TRL's Liger). No-op unless a FLASH_* kernel flag selects it and
    # freesolo-chalk is installed.
    _chalk_report = install_chalk_kernels(getattr(trainer, "model", None))

    _reset_peak_gpu()  # so peak_gpu_gb reflects the train loop (optimizer-state A/B is measurable)
    _gpu_sampler = _GpuPeakSampler().start()  # true device peak incl. bnb managed optimizer pages
    t_train = time.time()
    with _sdpa_cudnn_ctx(_attn):  # force cuDNN SDPA on sm120 (no-op otherwise)
        trainer.train(resume_from_checkpoint=resume_ckpt)
    train_wall = time.time() - t_train
    sft_peak_gpu_gb = _peak_gpu_gb()
    sft_device_peak_gpu_gb = _gpu_sampler.stop_gb()

    adapter_dir = f"{out_dir}/adapter"
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    _w.hf_upload_folder(adapter_dir, "adapter", required=True)
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
