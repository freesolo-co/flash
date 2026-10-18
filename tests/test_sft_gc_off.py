"""SFT gradient-checkpointing-OFF gate for the 35B-A3B MoE on a big card (CPU-only, no GPU).

GC recomputes the forward in the backward (~+33% compute) to save activation memory. For a big MoE
with a cheap ~3B active backbone + fused CE (Liger FLCE drops the [B,T,vocab] logit spike), the
no-recompute activations were assumed to FIT an H200 / B200 with margin — so GC is pure tax and we turn
it OFF. The gate is conservative: it only fires with the SFT signals (``allow_disable=True``), keeps
GC on for the 80 GB H100 / long context / unknown dims, and never touches the colocated GRPO trainer.
"""

from __future__ import annotations

from flash.engine.plan.vram import sft_gc_off_peak_gb, sft_grad_checkpoint_can_disable
from flash.engine.worker.perf.memory import grad_checkpointing_on


def _moe_info():
    """The catalog entry production passes as ``model_info``.

    Without it ``_lora_memory_gb`` falls back to the legacy floor, which understates the adapter by
    59 GB at rank 64 (1.68 vs 61.00) because it cannot see the two fused routed-expert tensors.
    The production path (``grad_checkpointing_on`` -> ``sft_grad_checkpoint_can_disable``) always
    forwards it, so a fixture that omits it is sizing a model this platform never trains.
    """
    from flash.core.catalog import MODELS

    return MODELS["Qwen/Qwen3.6-35B-A3B"]


# Qwen3.6-35B-A3B real architecture dims (from its config.json text_config).
_MOE = {
    "active_params_b": 3.0,
    "hidden": 2048,
    "num_layers": 40,
    "batch": 4,
    "lora_rank": 16,
    "model_info": _moe_info(),
}


def test_active_params_resolution_is_null_safe():
    # The GC-off gate reads active_params_b from the catalog to size the MoE backbone. It must use
    # MODELS.get (None for an unknown id), NOT get_model (which RAISES): the gate runs on the
    # allocation path, where a stale id should degrade to the dense default rather than abort.
    import pytest

    from flash.core.catalog import MODELS, get_model

    unknown_id = "some/unknown-model"
    with pytest.raises(ValueError, match="unsupported model"):
        get_model(unknown_id)  # the raising path the gate must NOT take
    # the gate's actual (null-safe) pattern -> dense default
    assert (float(getattr(MODELS.get(unknown_id), "active_params_b", 0.0) or 0.0) or None) is None
    # a cataloged MoE still resolves its active count through the same expression
    assert (
        float(getattr(MODELS.get("Qwen/Qwen3.6-35B-A3B"), "active_params_b", 0.0) or 0.0) or None
    ) == 3.0


def test_gc_off_gate_is_lora_rank_aware():
    # the GC-off gate must size on the run's REAL LoRA rank, not the default 32. A higher rank grows
    # the LoRA optimizer/adapter memory, so on a borderline card a high-rank run that the
    # default-rank estimate would have disabled GC for (then OOM'd) must KEEP GC on.
    p32 = sft_gc_off_peak_gb(35.0, seq_len=711, **{**_MOE, "batch": 1, "lora_rank": 32})
    p512 = sft_gc_off_peak_gb(35.0, seq_len=711, **{**_MOE, "batch": 1, "lora_rank": 512})
    assert p512 > p32  # the peak grows with rank
    # end-to-end through grad_checkpointing_on at the measured 1404-token step, on a B200 (torch
    # reports 178.35 GiB as ~191.5 decimal GB, which is the expression that feeds card_vram_gb):
    # rank 8 disables GC, rank 64 keeps it on. this IS the rank-8 escape -- the same workload OOM'd
    # at rank 64 and trained at rank 8, because rank drives ~53 GB of Adam state, not activations.
    kw = {
        "model_id": "Qwen/Qwen3.6-35B-A3B",
        "max_length": 1404,
        "allow_disable": True,
        "active_params_b": 3.0,
        "hidden": 2048,
        "num_layers": 40,
        "fused_ce": True,
        "per_device_bs": 1,
        "capability": (9, 0),
    }
    assert grad_checkpointing_on(card_vram_gb=191.5, lora_rank=8, **kw) is False  # GC off (fits)
    assert grad_checkpointing_on(card_vram_gb=191.5, lora_rank=64, **kw) is True  # GC kept on


def test_gc_off_peak_scales_linearly_with_seq():
    p2k = sft_gc_off_peak_gb(35.0, seq_len=2368, **_MOE)
    p8k = sft_gc_off_peak_gb(35.0, seq_len=8192, **_MOE)
    # base (weights+overhead) is constant; the activation delta is linear in seq.
    assert p8k > p2k
    # weights term alone (~70 GB) dominates the floor.
    assert p2k > 70.0


def test_dense_activation_constant_matches_the_live_rtx5090_peak():
    """The dense K is a live measurement, not the MoE's safety pad.

    Measured on an RTX 5090 (31.37 GB usable), a sub-1B dense checkpoint (hidden 1024, 24 layers),
    LoRA rank 32 all-linear, bs 4, bf16, and chunked dense-logit-free CE matching
    ``model.use_fused_kernels=true``:
    seq 1024 -> 15.09 GB, seq 2048 -> 28.18 GB, seq 4096 -> OOM. The shared 18.0 predicted 9.89 GB at
    seq 1024 and 20.76 GB at seq 4096 -- it called a run that OOMs a FIT, which is the failure this
    pins. A safety gate must over-reserve, so the estimate has to sit ABOVE each live peak.
    """
    dense = {"active_params_b": None, "hidden": 1024, "num_layers": 24, "batch": 4, "lora_rank": 32}
    for seq, live_peak in ((1024, 15.09), (2048, 28.18)):
        est = sft_gc_off_peak_gb(0.8, seq_len=seq, **dense)
        assert est > live_peak, f"seq {seq}: estimate {est:.2f} under-reserves vs live {live_peak}"
        # ...but not so far above that the gate becomes useless -- 2x the live peak is the ceiling.
        assert est < 2.0 * live_peak, f"seq {seq}: estimate {est:.2f} wildly over-reserves"
    # seq 4096 OOMed a 31.37 GB card, so the estimate must exceed the card and keep GC ON.
    assert sft_gc_off_peak_gb(0.8, seq_len=4096, **dense) > 31.37
    assert sft_grad_checkpoint_can_disable(0.8, seq_len=4096, card_vram_gb=31.37, **dense) is False
    # the 18 GB margin is what blocks the seq lengths that DO fit bare (15.09 GB at seq 1024 leaves
    # only 16.28 GB), so a 32 GB card never disables GC for this model at any production context.
    for seq in (1024, 2048):
        assert (
            sft_grad_checkpoint_can_disable(0.8, seq_len=seq, card_vram_gb=31.37, **dense) is False
        )


def test_dense_and_moe_activation_constants_are_separate():
    """One constant cannot describe both geometries -- and the MoE is the EXPENSIVE one.

    The old 18.0 encoded the intuition that ~3B active params imply small activations. Routing is
    per-token, so all 40 layers still materialize activations and the wide expert stack makes each
    one large: measured, the MoE's per-layer cost is ~3x the dense fit, not a third of it. The two
    constants stay separate because they are separate measurements, on separate hardware.
    """
    from flash.engine.plan.vram import _GC_OFF_ACT_K_DENSE, _GC_OFF_ACT_K_MOE

    assert _GC_OFF_ACT_K_MOE > _GC_OFF_ACT_K_DENSE
    # same geometry, same params: only the MoE signal differs -> the MoE estimate must be larger.
    geom = {"hidden": 2048, "num_layers": 40, "batch": 4, "lora_rank": 16, "seq_len": 2368}
    as_moe = sft_gc_off_peak_gb(35.0, active_params_b=3.0, **geom)
    as_dense = sft_gc_off_peak_gb(35.0, active_params_b=None, **geom)
    assert as_moe > as_dense


def test_moe_activation_constant_matches_the_live_b200_peak():
    """The MoE K is calibrated to a live OOM boundary, and must reject the run that hit it.

    Run ``flash-1786414384-2273e8d6``, 1x B200, Qwen3.6-35B-A3B, LoRA rank 64, fused CE, at the
    SMALLEST step this platform can express (micro_batch=1, max_token_len_per_gpu=1404):

        torch.OutOfMemoryError ... Of the allocated memory 167.91 GiB is allocated by PyTorch

    167.91 GiB is 180.3 decimal GB, and this estimator predicts in decimal GB: the card reports
    178.35 GiB == 191.5 GB via ``torch.cuda.get_device_properties(0).total_memory / 1e9``, the exact
    expression that feeds ``card_vram_gb``. Because the run DIED there, 180.3 GB is a lower bound on
    what the step needed, not a peak it achieved -- so the estimator's TOTAL is pinned to it as a
    fit-gate boundary, and the gate must call this configuration a NON-fit. At 18.0 it predicted
    139.1 GB, declared "GC-off peak fits", and the run died at step 0.

    Do NOT re-derive this from "allocated minus After-FSDP": that baseline is logged before the
    optimizer exists and in GiB despite its "(GB)" label, so the subtraction mixes units and counts
    optimizer state as activation. See the note on ``_GC_OFF_ACT_K_MOE``.
    """
    moe64 = {**_MOE, "batch": 1, "lora_rank": 64}
    peak = sft_gc_off_peak_gb(35.0, seq_len=1404, **moe64)
    assert 178.0 < peak < 183.0, f"estimate {peak:.1f} GB drifted from the 180.3 GB OOM boundary"
    # the decisive assertion: this exact configuration must NOT be allowed to run GC-off.
    assert sft_grad_checkpoint_can_disable(35.0, seq_len=1404, card_vram_gb=191.5, **moe64) is False
    # and the H200 it was first scheduled onto (139.80 GiB == 150.1 GB) is rejected too.
    assert sft_grad_checkpoint_can_disable(35.0, seq_len=1404, card_vram_gb=150.1, **moe64) is False
    # shrinking the batch must NOT restore false confidence -- every reachable step stays rejected.
    for batch in (1, 2, 4):
        assert (
            sft_grad_checkpoint_can_disable(
                35.0, seq_len=1404, card_vram_gb=191.5, **{**_MOE, "batch": batch, "lora_rank": 64}
            )
            is False
        )


def test_rank8_single_card_run_still_disables_gc():
    """The rank-8 escape must survive the recalibration.

    ``flash-1786422649-0d04e8e9`` trained on ONE B200 at rank 8 with GC off (dataset max row 711
    tokens). Rank 8 frees ~53 GB of Adam state versus rank 64, which is what made the same
    activations fit. A constant tuned only to reject the OOM would be free to reject this too;
    pinning the working run keeps the fix from degenerating into "always checkpoint".
    """
    moe8 = {**_MOE, "batch": 1, "lora_rank": 8}
    assert sft_grad_checkpoint_can_disable(35.0, seq_len=711, card_vram_gb=191.5, **moe8) is True


def test_gc_off_unknown_dims_is_inf():
    assert sft_gc_off_peak_gb(
        35.0, active_params_b=3.0, seq_len=2368, hidden=0, num_layers=40
    ) == float("inf")
    assert sft_gc_off_peak_gb(
        35.0, active_params_b=3.0, seq_len=2368, hidden=2048, num_layers=0
    ) == float("inf")


def test_can_disable_needs_a_short_step_not_just_a_big_card():
    # at the measured MoE cost a 2368-token step at micro-batch 4 does not fit ANY card we rent:
    # the H200 (150 GB) and the B200 (191.5 GB) both keep GC on. card size alone is not the lever.
    assert sft_grad_checkpoint_can_disable(35.0, seq_len=2368, card_vram_gb=150.1, **_MOE) is False
    assert sft_grad_checkpoint_can_disable(35.0, seq_len=2368, card_vram_gb=191.5, **_MOE) is False
    # 80 GB H100: 70 GB weights leave no room for the no-recompute activations -> keep GC on.
    assert sft_grad_checkpoint_can_disable(35.0, seq_len=2368, card_vram_gb=80.0, **_MOE) is False
    # a short step on a big card still legitimately disables GC (the rank-8 regime).
    short = {**_MOE, "batch": 1, "lora_rank": 8}
    assert sft_grad_checkpoint_can_disable(35.0, seq_len=711, card_vram_gb=191.5, **short) is True


def test_can_disable_keeps_gc_at_long_context():
    # A long (8k) SFT context pushes the no-recompute activations past the H200 -> keep GC on.
    assert sft_grad_checkpoint_can_disable(35.0, seq_len=8192, card_vram_gb=150.1, **_MOE) is False


def test_can_disable_conservative_on_unknown_card_or_dims():
    assert sft_grad_checkpoint_can_disable(35.0, seq_len=2368, card_vram_gb=0.0, **_MOE) is False
    bad = {**_MOE, "hidden": 0}
    assert sft_grad_checkpoint_can_disable(35.0, seq_len=2368, card_vram_gb=150.1, **bad) is False


def test_gate_turns_gc_off_for_35b_on_b200():
    # the speed win survives, but only for a step that actually fits: rank 8, micro-batch 1, and
    # this dataset's real 711-token rows -- the configuration that trained to step 150 on one B200.
    off = grad_checkpointing_on(
        "Qwen/Qwen3.6-35B-A3B",
        711,
        allow_disable=True,
        card_vram_gb=191.5,
        capability=(9, 0),
        active_params_b=3.0,
        hidden=2048,
        num_layers=40,
        fused_ce=True,
        per_device_bs=1,
        lora_rank=8,
    )
    assert off is False  # GC OFF (the speed win)


def test_gate_keeps_gc_on_for_the_step_that_oomed_the_b200():
    # the regression this whole change exists for: rank 64, micro-batch 1, 1404 tokens on a B200.
    # the gate previously printed "GC-off peak fits" and the run died at step 0.
    on = grad_checkpointing_on(
        "Qwen/Qwen3.6-35B-A3B",
        1404,
        allow_disable=True,
        card_vram_gb=191.5,
        capability=(9, 0),
        active_params_b=3.0,
        hidden=2048,
        num_layers=40,
        fused_ce=True,
        per_device_bs=1,
        lora_rank=64,
    )
    assert on is True


def test_gate_keeps_gc_on_for_35b_on_h200():
    # the 150.1 GB H200 (139.80 GiB) can no longer hold the GC-off peak once the experts are trained, so the gate
    # must keep gradient checkpointing ON there rather than disabling it into an OOM.
    on = grad_checkpointing_on(
        "Qwen/Qwen3.6-35B-A3B",
        2368,
        allow_disable=True,
        card_vram_gb=150.1,
        capability=(9, 0),
        active_params_b=3.0,
        hidden=2048,
        num_layers=40,
        fused_ce=True,
        per_device_bs=4,
    )
    assert on is True


def test_gate_keeps_gc_on_h100_80gb():
    on = grad_checkpointing_on(
        "Qwen/Qwen3.6-35B-A3B",
        2368,
        allow_disable=True,
        card_vram_gb=80.0,
        capability=(9, 0),
        active_params_b=3.0,
        hidden=2048,
        num_layers=40,
        fused_ce=True,
        per_device_bs=4,
    )
    assert on is True


def test_gate_keeps_gc_on_when_fused_ce_off():
    # Without fused CE the [B,T,vocab] logit spike returns -> GC must stay on regardless of card.
    on = grad_checkpointing_on(
        "Qwen/Qwen3.6-35B-A3B",
        2368,
        allow_disable=True,
        card_vram_gb=150.1,
        capability=(9, 0),
        active_params_b=3.0,
        hidden=2048,
        num_layers=40,
        fused_ce=False,
        per_device_bs=4,
    )
    assert on is True


def test_gate_unchanged_without_disable_signal():
    # No allow_disable -> memory-mode default: big model -> GC on.
    assert grad_checkpointing_on("Qwen/Qwen3.6-35B-A3B", 2368) is True
    # Even passing GPU signals positionally-free: without allow_disable they're ignored.
    assert (
        grad_checkpointing_on(
            "Qwen/Qwen3.6-35B-A3B", 2368, card_vram_gb=150.1, capability=(9, 0), fused_ce=True
        )
        is True
    )


def test_grpo_use_reentrant_true_for_moe():
    # MoE (Qwen3.6-35B-A3B, active 3B < total 35B) MUST use reentrant GC recompute under GRPO:
    # non-reentrant asserts recompute-metadata equality, which the MoE router violates on the first
    # backward (fwd 28192 vs recompute 3524 == group_size x) and crashes the run before step 1.
    from flash.engine.worker.perf.memory import grpo_use_reentrant

    assert grpo_use_reentrant("Qwen/Qwen3.6-35B-A3B") is True


def test_grpo_use_reentrant_true_for_gdn_hybrid():
    # cataloged gated-deltanet hybrids need reentrant gc based on architecture, not display name: fa2 +
    # the fused gdn chunk-scan + the fused triton kernels save data-dependent tensors the non-reentrant
    # metadata-equality assert cannot reconcile (forward packed [1636,..] vs recompute padded [1024,..]),
    # crashing at step 0 exactly like moe. live-confirmed on a qwen3.5 gdn grpo run.
    from flash.engine.worker.perf.memory import grpo_use_reentrant

    for gdn_id in ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.8-27B"):
        assert grpo_use_reentrant(gdn_id) is True, gdn_id


def test_grpo_use_reentrant_uses_catalog_geometry_not_model_name(monkeypatch):
    from dataclasses import replace

    from flash.core.catalog import MODELS
    from flash.engine.worker.perf.memory import grpo_use_reentrant

    renamed = replace(MODELS["Qwen/Qwen3.8-27B"], id="acme/renamed-hybrid")
    pure_attention = replace(
        MODELS["Qwen/Qwen3.8-27B"],
        id="Qwen/Qwen3.8-name-only",
        num_linear_attention_layers=0,
    )
    monkeypatch.setitem(MODELS, renamed.id, renamed)
    monkeypatch.setitem(MODELS, pure_attention.id, pure_attention)

    assert grpo_use_reentrant(renamed.id) is True
    assert grpo_use_reentrant(pure_attention.id) is False


def test_grpo_use_reentrant_false_for_non_gdn_dense():
    # uncataloged non-gdn dense models keep the faster non-reentrant path because standard
    # transformer layers recompute deterministically without metadata divergence.
    from flash.engine.worker.perf.memory import grpo_use_reentrant

    assert grpo_use_reentrant("meta-llama/Llama-3.2-1B") is False
    # an uncataloged open non-qwen model is treated as non-gdn dense (null-safe, no crash).
    assert grpo_use_reentrant("some/non-qwen-dense-model") is False


def test_is_moe_property():
    from flash.core.catalog import MODELS

    assert MODELS["Qwen/Qwen3.6-35B-A3B"].is_moe is True
    # every dense entry: active_params_b defaults to 0.0
    assert all(m.is_moe is False for m in MODELS.values() if m.active_params_b == 0.0)


def test_size_gate_reads_the_catalog_not_a_dense_param_estimate():
    """The GC/liger size gate must not size a MoE with a dense parameter formula.

    ``_estimate_params`` reconstructs ``embeddings + 12 h^2 per layer``, which has no term for an
    expert stack. For Qwen3.6-35B-A3B (2048 hidden, 40 layers, 248320 vocab, untied) that is
    3.03B against a real 35B -- an 11.5x under-count that lands 1% above the 3.0B threshold. The
    gate's answer was therefore correct by luck, on a model that clears the bar 11x over, and a
    small catalog change (a narrower vocab, fewer layers) would silently flip it to "small model,
    no gradient checkpointing needed" on a 35B MoE.

    This test only characterizes the formula -- every assertion below holds with or without the
    fix. ``test_size_gate_is_offline_and_fail_safe_for_cataloged_models`` owns the behavior.
    """
    from flash.core.catalog import MODELS
    from flash.engine.worker.perf.liger import _LIGER_MIN_PARAMS, _estimate_params

    class _Cfg:
        hidden_size = 2048
        vocab_size = 248_320
        num_hidden_layers = 40
        tie_word_embeddings = False

    estimate = _estimate_params(_Cfg())
    assert estimate < 3.1e9, "dense formula no longer under-counts the MoE; revisit this test"
    assert MODELS["Qwen/Qwen3.6-35B-A3B"].params_b == 35.0
    # the margin the old path depended on: 1% of the threshold, on an 11.5x-larger model.
    assert 1.0 < estimate / _LIGER_MIN_PARAMS < 1.02


def test_size_gate_is_offline_and_fail_safe_for_cataloged_models(monkeypatch):
    """A cataloged model must not need a network probe to decide gradient checkpointing.

    ``_liger_default_for_model`` returns False on ANY exception, and False means "small model" ->
    ``_memory_mode`` False -> gradient checkpointing OFF. So a rate-limited or offline HF read used
    to disable checkpointing on a 35B model. The catalog is local and exact, so it answers first.

    The failure is injected at ``AutoConfig.from_pretrained`` rather than at ``_estimate_params``:
    that call IS the network boundary, and it runs first. Patching the estimator instead would let
    a regressed short-circuit reach the real HF read before hitting anything this test controls.
    """
    import sys
    import types

    import flash.engine.worker.perf.liger as liger_mod
    from flash.core.catalog import MODELS, ModelInfo
    from flash.engine.worker.perf.memory import _memory_mode

    def _explode(*a, **k):
        raise AssertionError("cataloged model fell through to the HF config probe")

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = types.SimpleNamespace(from_pretrained=_explode)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    synthetic_id = "test/sub-3b-model"
    monkeypatch.setitem(
        MODELS,
        synthetic_id,
        ModelInfo(
            id=synthetic_id,
            display_name="synthetic sub-3b model",
            params="2.3B",
            algos=("sft",),
            min_vram_gb=24,
            params_b=2.3,
        ),
    )

    # every cataloged model answers from params_b, with the probe guaranteed to fail if consulted.
    assert liger_mod._liger_default_for_model("Qwen/Qwen3.6-35B-A3B") is True
    assert liger_mod._liger_default_for_model("Qwen/Qwen3.8-27B") is True
    assert liger_mod._liger_default_for_model("Qwen/Qwen3.5-9B") is True
    assert liger_mod._liger_default_for_model(synthetic_id) is False
    # and the decision that matters: a short-context 35b step keeps gradient checkpointing on.
    assert _memory_mode("Qwen/Qwen3.6-35B-A3B", 711) is True
