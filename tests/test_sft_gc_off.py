"""SFT gradient-checkpointing-OFF gate for the 35B-A3B MoE on a big card (CPU-only, no GPU).

GC recomputes the forward in the backward (~+33% compute) to save activation memory. For a big MoE
with a cheap ~3B active backbone + fused CE (Liger FLCE drops the [B,T,vocab] logit spike), the
no-recompute activations FIT a 141 GB H200 / 180 GB B200 with margin — so GC is pure tax and we turn
it OFF. The gate is conservative: it only fires with the SFT signals (``allow_disable=True``), keeps
GC on for the 80 GB H100 / long context / unknown dims, and never touches the colocated GRPO trainer.
"""

from __future__ import annotations

from flash.engine.vram import sft_gc_off_peak_gb, sft_grad_checkpoint_can_disable
from flash.engine.worker.perf.memory import grad_checkpointing_on

# Qwen3.6-35B-A3B real architecture dims (from its config.json text_config).
_MOE = {"active_params_b": 3.0, "hidden": 2048, "num_layers": 40, "batch": 4, "lora_rank": 16}


def test_open_model_active_params_resolution_is_null_safe():
    # The GC-off gate reads active_params_b from the catalog to size the MoE backbone. It must use
    # MODELS.get (None for an uncataloged id), NOT get_model (which RAISES), so an open-model SFT run
    # (model_policy="allow") doesn't abort here. Regression for the Cursor "Open-model SFT crashes on
    # get_model" finding.
    import pytest

    from flash.catalog import MODELS, get_model

    open_id = "some/uncataloged-open-model"
    with pytest.raises(ValueError, match="unsupported model"):
        get_model(open_id)  # the raising path the gate must NOT take
    # the gate's actual (null-safe) pattern -> dense default, since an open model's MoE active count
    # is unknown
    assert (float(getattr(MODELS.get(open_id), "active_params_b", 0.0) or 0.0) or None) is None
    # a cataloged MoE still resolves its active count through the same expression
    assert (
        float(getattr(MODELS.get("Qwen/Qwen3.6-35B-A3B"), "active_params_b", 0.0) or 0.0) or None
    ) == 3.0


def test_gc_off_gate_is_lora_rank_aware():
    # Codex P2 (Mrx13): the GC-off gate must size on the run's REAL LoRA rank, not the default 32.
    # A higher rank grows the LoRA optimizer/adapter memory, so on a borderline card a high-rank run
    # that the default-rank estimate would have disabled GC for (then OOM'd) must KEEP GC on.
    p32 = sft_gc_off_peak_gb(35.0, seq_len=2368, **{**_MOE, "lora_rank": 32})
    p512 = sft_gc_off_peak_gb(35.0, seq_len=2368, **{**_MOE, "lora_rank": 512})
    assert p512 > p32  # the peak grows with rank
    # end-to-end through grad_checkpointing_on: at a 125 GB card the default rank disables GC (returns
    # False) but rank 512 keeps it on (returns True) -> the configured rank changes the decision.
    kw = {
        "model_id": "Qwen/Qwen3.6-35B-A3B",
        "max_length": 2368,
        "allow_disable": True,
        "active_params_b": 3.0,
        "hidden": 2048,
        "num_layers": 40,
        "fused_ce": True,
        "per_device_bs": 4,
        "capability": (9, 0),
    }
    assert grad_checkpointing_on(card_vram_gb=125.0, lora_rank=32, **kw) is False  # GC off (fits)
    assert grad_checkpointing_on(card_vram_gb=125.0, lora_rank=512, **kw) is True  # GC kept on


def test_gc_off_peak_scales_linearly_with_seq():
    p2k = sft_gc_off_peak_gb(35.0, seq_len=2368, **_MOE)
    p8k = sft_gc_off_peak_gb(35.0, seq_len=8192, **_MOE)
    # base (weights+overhead) is constant; the activation delta is linear in seq.
    assert p8k > p2k
    # weights term alone (~70 GB) dominates the floor.
    assert p2k > 70.0


def test_gc_off_unknown_dims_is_inf():
    assert sft_gc_off_peak_gb(
        35.0, active_params_b=3.0, seq_len=2368, hidden=0, num_layers=40
    ) == float("inf")
    assert sft_gc_off_peak_gb(
        35.0, active_params_b=3.0, seq_len=2368, hidden=2048, num_layers=0
    ) == float("inf")


def test_can_disable_fits_h200_not_h100_at_default_ctx():
    # 141 GB H200: the 35B-A3B GC-off peak (~102 GB at seq 2368) + margin fits -> can disable.
    assert sft_grad_checkpoint_can_disable(35.0, seq_len=2368, card_vram_gb=141.0, **_MOE) is True
    # 80 GB H100: 70 GB weights leave no room for the no-recompute activations -> keep GC on.
    assert sft_grad_checkpoint_can_disable(35.0, seq_len=2368, card_vram_gb=80.0, **_MOE) is False


def test_can_disable_keeps_gc_at_long_context():
    # A long (8k) SFT context pushes the no-recompute activations past the H200 -> keep GC on.
    assert sft_grad_checkpoint_can_disable(35.0, seq_len=8192, card_vram_gb=141.0, **_MOE) is False


def test_can_disable_conservative_on_unknown_card_or_dims():
    assert sft_grad_checkpoint_can_disable(35.0, seq_len=2368, card_vram_gb=0.0, **_MOE) is False
    bad = {**_MOE, "hidden": 0}
    assert sft_grad_checkpoint_can_disable(35.0, seq_len=2368, card_vram_gb=141.0, **bad) is False


def test_gate_turns_gc_off_for_35b_on_h200():
    off = grad_checkpointing_on(
        "Qwen/Qwen3.6-35B-A3B",
        2368,
        allow_disable=True,
        card_vram_gb=141.0,
        capability=(9, 0),
        active_params_b=3.0,
        hidden=2048,
        num_layers=40,
        fused_ce=True,
        per_device_bs=4,
    )
    assert off is False  # GC OFF (the speed win)


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
        card_vram_gb=141.0,
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
            "Qwen/Qwen3.6-35B-A3B", 2368, card_vram_gb=141.0, capability=(9, 0), fused_ce=True
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
    # Dense Qwen3.5/3.6 GatedDeltaNet hybrids ALSO need reentrant GC under GRPO: FA2 varlen-unpad +
    # the fused GDN chunk-scan + chalk's fused kernels save data-dependent tensors the non-reentrant
    # metadata-equality assert can't reconcile (forward packed [1636,..] vs recompute padded [1024,..]),
    # crashing at step 0 exactly like MoE. Live-confirmed on Qwen3.5-0.8B GRPO / RTX 4090.
    from flash.engine.worker.perf.memory import grpo_use_reentrant

    for gdn_id in ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"):
        assert grpo_use_reentrant(gdn_id) is True, gdn_id


def test_grpo_use_reentrant_false_for_non_gdn_dense():
    # Non-GDN dense models (MiniCPM = plain Llama attention) keep the faster non-reentrant path:
    # standard transformer layers recompute deterministically, no metadata divergence.
    from flash.engine.worker.perf.memory import grpo_use_reentrant

    assert grpo_use_reentrant("openbmb/MiniCPM5-1B") is False
    # An uncataloged / open non-Qwen model is treated as non-GDN dense (null-safe, no crash).
    assert grpo_use_reentrant("some/uncataloged-open-model") is False


def test_is_moe_property():
    from flash.catalog import MODELS

    assert MODELS["Qwen/Qwen3.6-35B-A3B"].is_moe is True
    # every dense entry: active_params_b defaults to 0.0
    assert all(m.is_moe is False for m in MODELS.values() if m.active_params_b == 0.0)
