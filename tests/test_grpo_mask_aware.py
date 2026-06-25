"""CPU test for the mask-aware lm_head patch (skip masked completion positions in the GRPO loss).

Verifies the wrapper is EXACTLY loss-preserving and that every per-token tensor is gathered with one
consistent index (a misaligned gather would change the loss), using a fake ``liger_grpo_loss`` whose
loss depends on all per-token inputs the way TRL's dr_grpo does (numerator = sum over UNMASKED
positions; normalizer = a constant independent of the sequence length).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from flash.engine.worker.lora import patch_grpo_mask_aware_lm_head


class _FakeTrainer:
    pass


def _make_fake_loss(record):
    # Numerator depends on EVERY per-token tensor (so a misaligned gather of any one would change the
    # result); normalizer is constant (mirrors dr_grpo's B * max_completion_length, not seq length).
    NORM = 7.0

    def fake_loss(*, _input, lin_weight, selected_token_ids, attention_mask, advantages,
                  old_per_token_logps=None, ref_per_token_logps=None, vllm_is_ratio=None, **_):
        record["T"] = int(selected_token_ids.size(1))
        per_token = (
            selected_token_ids.float() * _input.sum(-1)
            + old_per_token_logps
            - ref_per_token_logps
        )
        loss = (per_token * attention_mask).sum() / NORM
        return loss, {"reward": advantages.mean()}

    return fake_loss


def _inputs():
    torch.manual_seed(0)
    b, t, h = 2, 6, 3
    return {
        "_input": torch.randn(b, t, h),
        "lin_weight": torch.randn(8, h),
        "selected_token_ids": torch.randint(1, 100, (b, t)),
        # row0: 3 unmasked, row1: 4 unmasked -> T' = 4 < 6
        "attention_mask": torch.tensor([[1, 0, 1, 1, 0, 0], [1, 1, 0, 1, 0, 1]]),
        "advantages": torch.randn(b),
        "old_per_token_logps": torch.randn(b, t),
        "ref_per_token_logps": torch.randn(b, t),
    }


def test_mask_aware_lm_head_is_loss_preserving_and_aligned():
    rec_full, rec_masked = {}, {}
    kw = _inputs()

    # baseline loss with the full-length tensors (no patch)
    loss_full, _ = _make_fake_loss(rec_full)(**kw)

    trainer = _FakeTrainer()
    trainer.liger_grpo_loss = _make_fake_loss(rec_masked)
    assert patch_grpo_mask_aware_lm_head(trainer) is True
    loss_masked, metrics = trainer.liger_grpo_loss(**kw)

    assert rec_full["T"] == 6  # the fake saw full length without the patch
    assert rec_masked["T"] == 4  # ...and GATHERED length (max unmasked = 4) with it
    # exactly loss-preserving — and since the numerator uses sel/_input/old/ref, this only holds if
    # every per-token tensor was gathered with the SAME index (a misaligned gather would diverge).
    assert torch.allclose(loss_full, loss_masked)
    assert "reward" in metrics  # passes through the loss object's metrics dict


def test_mask_aware_lm_head_noop_when_nothing_masked():
    # Single-turn / fully-unmasked: T' == T, so the wrapper must pass through untouched.
    rec = {}
    kw = _inputs()
    kw["attention_mask"] = torch.ones_like(kw["attention_mask"])
    trainer = _FakeTrainer()
    trainer.liger_grpo_loss = _make_fake_loss(rec)
    patch_grpo_mask_aware_lm_head(trainer)
    trainer.liger_grpo_loss(**kw)
    assert rec["T"] == 6  # untouched (no gather)


def test_mask_aware_lm_head_single_turn_padding():
    # Single-turn GRPO has no tool_mask, but completions are padded to the batch max -> the
    # completion_mask carries trailing right-padding zeros. The patch must skip those too, still
    # loss-preserving. Row0: 3 real + 3 pad, row1: 5 real + 1 pad -> T' = 5.
    rec_full, rec_masked = {}, {}
    kw = _inputs()
    kw["attention_mask"] = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]])
    loss_full, _ = _make_fake_loss(rec_full)(**kw)
    trainer = _FakeTrainer()
    trainer.liger_grpo_loss = _make_fake_loss(rec_masked)
    patch_grpo_mask_aware_lm_head(trainer)
    loss_masked, _ = trainer.liger_grpo_loss(**kw)
    assert rec_masked["T"] == 5  # gathered to the deepest real length (skips padding)
    assert torch.allclose(loss_full, loss_masked)  # exactly loss-preserving for single-turn padding


def test_mask_aware_lm_head_returns_false_without_loss_object():
    assert patch_grpo_mask_aware_lm_head(_FakeTrainer()) is False
