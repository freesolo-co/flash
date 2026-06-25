"""Pure GRPO batch-sizing + no-op-failure helpers for the RL worker path.

These translate an intended prompts/step into a TRL completion-batch config, cap the
per-device micro-batch to fit VRAM, and decide whether a finished GRPO run actually trained.
The only run-scoped state read here is ``THINKING`` (via the worker package at CALL time, so a
test's ``monkeypatch.setattr(worker, "JOB_SPEC", ...)`` + reload reaches ``rl_per_device_comps``)
and ``JOB_SPEC`` (``grpo_overrides``)."""

from __future__ import annotations

from flash.engine.worker._pkg import W as _w


def compute_grpo_batching(prompts_per_step: int, group_size: int, per_device_comps: int) -> dict:
    """Translate an intended ``prompts_per_step`` into a TRL GRPO batch configuration.

    TRL's GRPO batch sizing is denominated in **completions (prompt-completion pairs), not
    prompts**. The number of *unique prompts* optimized per step is

        (per_device_train_batch_size * gradient_accumulation_steps * num_processes)
        / num_generations

    So to actually optimize ``prompts_per_step`` prompts per step, the global *completion*
    batch must equal ``prompts_per_step * group_size``. We keep ``per_device`` small (it,
    not grad-accum, sets peak VRAM) and put the rest in gradient accumulation.

    The bug this fixes: ``grad_accum = prompts_per_step // per_device`` treated
    ``per_device_train_batch_size`` as a *prompt* count, omitting the ``* group_size``
    factor, so a run intended as 64 prompts/step actually optimized only
    ``64 / group_size = 8`` prompts/step (an 8x smaller effective batch).
    """
    group_size = max(1, int(group_size))
    prompts_per_step = max(1, int(prompts_per_step))
    per_device = max(1, int(per_device_comps))
    target_comps = prompts_per_step * group_size  # total completions / optimizer step
    # Never let the per-device completion micro-batch exceed the target completion batch:
    # a small prompts_per_step would otherwise overshoot it (mirrors run_sft's
    # `min(per_device_bs, effective_batch)`). No-op at the default (prompts_per_step=64).
    per_device = max(1, min(per_device, target_comps))
    # per_device is the fixed VRAM knob, but when it does NOT divide target_comps neither floor
    # nor ceil of grad_accum is right: floor (the old bug) silently optimizes FEWER prompts than
    # requested, while ceil over-shoots and asks TRL for MORE unique prompts than the (already
    # dataset-capped) prompts_per_step -- which, on a small retained dataset, yields no batches
    # after the paid worker is provisioned. Instead shrink per_device to the largest divisor of
    # target_comps that is <= the requested per_device: that lowers (never raises) peak VRAM and
    # makes per_device * grad_accum == target_comps EXACTLY, so unique prompts == prompts_per_step
    # with no over/under-shoot. (per_device=16, target_comps=40 -> 10 -> grad_accum=4 -> 40 comps
    # = exactly 5 prompts. A divisor always exists since 1 divides everything.)
    while target_comps % per_device != 0:
        per_device -= 1
    grad_accum = max(1, target_comps // per_device)
    # The global completion batch (per_device * grad_accum == target_comps) is divisible by
    # num_generations (= group_size) by construction, since target_comps = prompts_per_step *
    # group_size; TRL's divisibility requirement is satisfied with no further rounding.
    generations_per_step = per_device * grad_accum
    unique_prompts_per_step = generations_per_step // group_size
    return {
        "per_device_train_batch_size": per_device,
        "gradient_accumulation_steps": grad_accum,
        "generations_per_step": generations_per_step,
        "unique_prompts_per_step": unique_prompts_per_step,
        # TRL requires the global completion batch be divisible by num_generations.
        "divisible_by_group": (generations_per_step % group_size == 0),
    }


def resolve_grpo_prompts_per_step(requested: int, available_prompts: int) -> int:
    """Cap GRPO's prompt batch to the retained dataset size.

    TRL's GRPO dataloader can yield zero batches when the configured prompt batch is larger
    than the dataset that remains after prompt-budget filtering. That surfaces late as
    "There seems not to be a single sample in your epoch_iterator" and then our no-reward guard
    reports the wrong cause. Small smoke envs should still train; use every retained prompt per
    step instead of asking TRL for an impossible larger batch.
    """
    requested = max(1, int(requested))
    available_prompts = int(available_prompts)
    if available_prompts <= 0:
        raise ValueError("GRPO needs at least one retained training prompt")
    return min(requested, available_prompts)


def build_grpo_prompt_dataset(prompts: list[dict]) -> tuple[list[dict], list]:
    """Arrow-safe GRPO rollout rows + the parallel example lookup ``reward_fn`` maps back through.

    ``Dataset.from_list`` lets PyArrow infer ONE column type per (nested) field across ALL rows, so
    embedding the rich per-example record makes a *valid* env whose per-row ``info``/``metadata``
    legitimately mixes types crash dataset construction with ``ArrowInvalid`` — and the whole RL
    phase dies at startup, AFTER the paid GPU is provisioned, on input that passed offline
    single-example validation. (Observed with ifeval-lite: ``metadata.param`` is an int target word
    count for some rows and a required-word string ``'gentle'`` for others; Arrow infers ``int64``
    from the leading rows then fails on the first string.)

    Fix: keep the dataset columns trivially typed — the TRL-required ``prompt`` plus a stable integer
    ``example_idx`` — and return the original example objects in a parallel list. ``reward_fn`` maps
    the index back, so the env still sees its EXACT record (no JSON/Arrow round-trip, no type
    coercion). ``rows[i]["example_idx"] == i`` and ``examples[i]`` is that row's record.
    """
    examples = [p["example"] for p in prompts]
    rows = [{"prompt": p["prompt"], "example_idx": i} for i, p in enumerate(prompts)]
    return rows, examples


# Hard ceiling on the per-device completion micro-batch when growing on a SHORT-seq run. MEASURED
# (RunPod, Qwen3.5-0.8B GRPO, group8, gsm8k, seq1024, 6 steps): trainer throughput rises from
# per_device 4 -> 8 (~+12%) and plateaus 8..16 (A100 80GB: 375/407/411 tok/s at pd 4/8/16), then
# REGRESSES at pd 32 (326 tok/s, -20%) as the larger forward stops buying MFU. So we never grow
# past the top of that plateau, even on a card with VRAM to spare. (Reward histories at pd 4 and
# 16 were identical -> per_device is a pure speed/VRAM knob, not an optimization change.)
_RL_PER_DEVICE_MAX = 16
# Reference sequence length the activation/VRAM divisor is calibrated at. The colocate activation
# peak grows with the training sequence length; the cap is scaled by seq_len/_RL_ACT_SEQ_REF so a
# short-seq run (the underfed regime) is allowed a proportionally bigger micro-batch.
_RL_ACT_SEQ_REF = 2048.0
# VRAM-per-(micro-batch element) divisor at the reference seq, normalized to ~2B width (1.41).
# MEASURED: Qwen3.5-2B group8 seq2048 OOMs a 32 GB card at per_device=8 but trains at 4.
# vram_gb is now decimal GB (/1e9) to match the rest of the VRAM logic, so the divisor is the
# old GiB-calibrated 7.5 scaled by 1024**3/1e9 (=7.5*1.0737=8.053). A physical 32 GiB card reports
# 34.36 decimal GB -> 34.36 / (8.053 * 1.0 * 1.0) = 4, byte-for-byte the old GiB result (32/7.5=4);
# the scaling makes the unit switch a pure no-op on real cards. (Unchanged historical colocate cap,
# so at/above the reference seq the value is byte-for-byte the old one — no regression.)
_RL_ACT_DIVISOR = 8.053
# Floor on the seq scale: caps how far a short sequence may grow the micro-batch. Set so the
# underfed case that motivated this — Qwen3.5-0.8B GRPO on a 24 GB card at seq<=1024 — lands on
# the MEASURED-SAFE per_device 8 (RunPod RTX 4090 24 GB: pd8 fits at 19.0 GB and is +12.6% over
# pd4, while the old seq-independent cap under-fed it at ~5; pd16 there would need ~27 GB -> OOM).
# A physical 24 GiB card reports 25.77 decimal GB: 25.77 / (8.053 * (0.894/1.41) * 0.63) = 8.0
# (the old GiB form 24/(7.5*...)=8.0, unchanged by the unit switch). Bounds short-seq growth to
# ~1.6x the reference cap.
_RL_ACT_SEQ_SCALE_FLOOR = 0.63
# Clamp the seq scale at 1.0 (never ABOVE the reference). Combined with the short_seq growth gate,
# this makes a seq>=reference run byte-for-byte the old value: seq_scale==1.0 -> vram_cap == the
# old colocate cap, and the ceiling falls back to the historical default, so min(default, ...) is
# exactly what the old code returned. We deliberately do NOT tighten long-seq below the historical
# value (grad checkpointing makes activations sub-linear in seq there, so the linear model would
# over-cap), nor grow above it (unvalidated — the regression is in tokens-in-flight = pd x seq).
_RL_ACT_SEQ_SCALE_CEIL = 1.0


def rl_per_device_comps(
    completion_len: int = 0,
    vocab: int = 248_320,
    *,
    use_vllm: bool = True,
    params_b: float | None = None,
    seq_len: int = 0,
) -> int:
    """Per-device *completion* micro-batch for GRPO (TRL counts completions, not prompts).

    This, not grad-accum, sets peak trainer VRAM AND the trainer step's MFU: a bigger
    micro-batch means bigger, fewer GEMMs (less launch overhead, fuller tensor cores) at the
    same effective batch (compute_grpo_batching pushes the remainder into grad-accum, so the
    optimization is identical — only speed/VRAM change). MEASURED on RunPod (Qwen3.5-0.8B GRPO,
    group8, seq1024): the old seq-independent colocate cap under-fed a 24 GB card at per_device ~5,
    while per_device 8 fits (19.0 GB) and is +12.6% throughput; on an 80 GB card throughput
    plateaus at per_device 8..16 and regresses by per_device 32. So on a SHORT-seq run we grow the
    micro-batch into the card's measured VRAM headroom up to the plateau ceiling.

    Growth is GATED to short sequences (seq < the reference). At/above the reference seq the value
    is byte-for-byte the historical one — bigger per_device at long context is unvalidated and the
    regression is driven by tokens-in-flight (per_device x seq), which a fixed-per_device ceiling
    would not catch.

    Two upper bounds cap the growth:

    * **logits budget (6 GB)** — a HARD correctness cap. The logprob pass can materialize fp32
      logits of shape [per_device, completion_len, vocab]; at Qwen3.5's ~248k vocab a long
      completion is enormous (per_device 8 x 4096 tok x 248k x 4 B = ~30 GiB -> OOMs a small
      card). Liger normally fuses these away, but this stays a safety net for the fallback path.

    * **activation/VRAM cap** — the per-device forward holds the model's attention/activation
      memory (the Qwen3.5 GDN/FLA kernels peak per micro-batch even with grad checkpointing),
      which the logits term can't see and which Liger does NOT touch. Calibrated against the live
      card's VRAM, model width (~sqrt(params)), and — unlike the old seq-independent cap — the
      training sequence length: activations scale ~linearly with seq, so a SHORT-seq run gets a
      proportionally bigger cap. MEASURED at seq_ref=2048: Qwen3.5-2B (width ~1.41) group8 OOMs a
      32 GB card at per_device=8 but trains at 4 -> 34.36 / 8.053 = 4 (decimal GB; the old GiB
      form 32/7.5=4, unchanged by the unit switch).

    Off a live card (allocator / unit tests) there is no VRAM signal, so we fall back to the
    conservative historical default (8, or 2 with thinking) bounded by the logits budget — the
    allocator already provisions for that floor, and the worker only ever grows INTO the spare
    VRAM the chosen card actually reports, so it cannot over-fill the card it was routed to.
    """
    default = 2 if _w.THINKING else 8

    # Logits budget: hard upper bound on the fp32 [per_device, completion, vocab] logprob tensor —
    # a single forward's logits must fit a ~6 GB ceiling.
    logits_cap = _RL_PER_DEVICE_MAX
    if completion_len > 0:
        logits_cap = max(1, int(6.0e9 / (max(1, completion_len) * vocab * 4)))

    # Growth is gated to SHORT sequences (seq < the reference). At/above the reference seq the
    # micro-batch is left exactly as the historical code computed it: bigger per_device at long
    # context is unvalidated and risky — the measured throughput regression is driven by
    # tokens-in-flight (per_device x seq), so per_device 16 at seq 2048 (~the regression-zone
    # per_device 32 at seq 1024) could regress, and a fixed-per_device ceiling would not catch it.
    short_seq = (seq_len or _RL_ACT_SEQ_REF) < _RL_ACT_SEQ_REF

    # Activation/VRAM cap — only computable on a live card. It both caps DOWN (big model / small
    # card / long seq) and, on a SHORT-seq run, lets the micro-batch GROW into spare VRAM.
    vram_cap = None
    if use_vllm:
        try:
            import torch

            if torch.cuda.is_available():
                # Decimal GB (/1e9), matching the rest of the VRAM sizing logic
                # (flash.engine.vram + gpu_setup.finalize_alloc_conf_for_sleep) so the
                # divisor/scale thresholds and calibration comments stay on one unit.
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                width = (max(float(params_b), 0.1) ** 0.5) if params_b else 1.41
                seq_scale = min(
                    _RL_ACT_SEQ_SCALE_CEIL,
                    max(_RL_ACT_SEQ_SCALE_FLOOR, (seq_len or _RL_ACT_SEQ_REF) / _RL_ACT_SEQ_REF),
                )
                vram_cap = max(
                    1, int(vram_gb / (_RL_ACT_DIVISOR * (width / 1.41) * seq_scale))
                )
        except Exception as e:
            print("rl_per_device_comps colocate cap probe failed (keeping logits cap):", e)

    if vram_cap is None:
        # No live card (allocator / offline / unit tests): conservative default, logits-bounded.
        return max(1, min(default, logits_cap))
    # Short seq -> grow into measured VRAM headroom up to the plateau ceiling. At/above the
    # reference seq the ceiling is the historical default, and seq_scale is clamped to 1.0 so
    # vram_cap == the old colocate cap -> the result is byte-for-byte the old value (no regression,
    # no unvalidated long-seq growth).
    #
    # THINKING runs are EXCLUDED from the growth path: they emit long completions whose
    # activation/logprob cost the prompt-only `seq_len` gate cannot see, so letting short-seq
    # growth raise the ceiling to _RL_PER_DEVICE_MAX would silently override the conservative
    # thinking default (2) and risk OOM / unstable training. They keep `default` as the ceiling,
    # i.e. byte-for-byte the historical value.
    ceiling = _RL_PER_DEVICE_MAX if (short_seq and not _w.THINKING) else default
    return max(1, min(ceiling, logits_cap, vram_cap))


def grpo_overrides() -> dict:
    """The GRPO recipe knobs, read off the job spec's ``[train]`` table (``TrainSpec``).
    A field left unset (None) is omitted here so the recipe default applies downstream.

    Knobs: group_size, temperature, max_tokens (completion budget), kl_penalty_coef (the KL
    beta), advantage_clip (centered-advantage clip), and thinking_length_penalty_coef
    (a per-<think>-token reward deduction). These live in ``[train]`` — NOT in
    ``[environment.params]``, which is forwarded verbatim to the Freesolo env loader."""
    if not _w.JOB_SPEC:
        return {}
    train = _w.JOB_SPEC.train
    cfg = {
        "group_size": train.group_size,
        "temperature": train.temperature,
        "max_tokens": train.max_tokens,
        "kl_penalty_coef": train.kl_penalty_coef,
        "advantage_clip": train.advantage_clip,
        "thinking_length_penalty_coef": train.thinking_length_penalty_coef,
    }
    return {k: v for k, v in cfg.items() if v is not None}


def _grpo_resume_already_complete(resume_ckpt, target_steps: int, steps_run: int) -> bool:
    """True when this worker resumed a checkpoint that already reached the target step count.

    Such a resume legitimately performs ZERO new optimizer steps (so the fresh hb_cb has an empty
    reward_history) yet the policy IS fully trained — it must NOT be flagged as a no-op failure.
    """
    return bool(resume_ckpt) and target_steps > 0 and steps_run >= target_steps


def _grpo_is_no_op_failure(reward_history, resume_ckpt, target_steps: int, steps_run: int) -> bool:
    """True when a GRPO run trained NOTHING and must fail loudly instead of reporting as done.

    An empty ``reward_history`` means the reward callback never fired — the rollout scored nothing
    (e.g. vLLM silently returning no completions), so no real training happened. The sole exception
    is a resume that already reached the target steps (see ``_grpo_resume_already_complete``): that
    has an empty fresh history but a fully-trained policy, so it is NOT a failure.
    """
    if reward_history:
        return False
    return not _grpo_resume_already_complete(resume_ckpt, target_steps, steps_run)
