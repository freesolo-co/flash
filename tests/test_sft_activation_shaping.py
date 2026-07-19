"""focused regressions for sft activation shaping."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
from datasets import Dataset

from flash.engine.vram import sft_grad_accum
from flash.engine.worker.packing import BlockDiagonalCollator, pack_token_ids
from flash.engine.worker.sft import (
    _configure_unpacked_length_sampling,
    _safe_realized_sft_max_length,
    _sft_local_token_count,
    _sft_quality_metrics_due,
    _sft_runtime_max_length,
    _SFTTokenCountingCollator,
)


def test_realized_length_sizing_uses_safe_max_and_raises_microbatch():
    rows = [
        {"input_ids": list(range(64))},
        {"input_ids": list(range(512))},
        {"input_ids": list(range(128))},
    ]
    configured_cap = 8192
    realized_max = _safe_realized_sft_max_length(rows, configured_cap)

    assert realized_max == 512
    capped = sft_grad_accum(8, seq_len=configured_cap, vocab=150_000, fused=False)
    realized = sft_grad_accum(8, seq_len=realized_max, vocab=150_000, fused=False)
    assert capped == (1, 8)
    assert realized == (4, 2)
    assert realized[0] * realized[1] >= 8


def test_realized_length_rejects_rows_past_configured_cap():
    rows = [{"input_ids": list(range(513))}]
    with pytest.raises(ValueError, match="exceeds configured max"):
        _safe_realized_sft_max_length(rows, 512)


def test_sdpa_runtime_sizing_includes_collator_padding():
    realized_max = 625
    runtime_max = _sft_runtime_max_length(realized_max, pad_to_multiple_of=8)

    assert runtime_max == 632
    raw = sft_grad_accum(4, seq_len=realized_max, vocab=150_000, fused=False)
    padded = sft_grad_accum(4, seq_len=runtime_max, vocab=150_000, fused=False)
    assert raw == (4, 1)
    assert padded == (3, 2)


def test_packed_token_count_is_linear_and_not_attention_edge_count():
    pytest.importorskip("torch")
    rows = pack_token_ids([list(range(4)), list(range(10, 13))], max_length=8)
    collator = _SFTTokenCountingCollator(
        BlockDiagonalCollator(pad_token_id=99, pad_to_multiple_of=8)
    )
    batch = collator(rows)

    token_count = _sft_local_token_count(batch)
    edge_count = batch["attention_mask"].sum()
    assert token_count.item() == 7
    assert edge_count.item() == 17
    assert token_count.item() != edge_count.item()


def test_group_by_length_is_gated_to_unpacked_only():
    base = Dataset.from_list(
        [
            {"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]},
            {"input_ids": [4, 5], "completion_mask": [0, 1]},
        ]
    )

    unpacked_cfg = {}
    unpacked = _configure_unpacked_length_sampling(unpacked_cfg, base, packed=False)
    assert unpacked["length"] == [3, 2]
    assert unpacked_cfg == {
        "train_sampling_strategy": "group_by_length",
        "length_column_name": "length",
    }

    packed_cfg = {}
    packed = _configure_unpacked_length_sampling(packed_cfg, base, packed=True)
    assert packed is base
    assert "length" not in packed.column_names
    assert packed_cfg == {}


def test_quality_metrics_run_only_on_logging_microbatch():
    args = SimpleNamespace(logging_first_step=False, logging_steps=10)
    state = SimpleNamespace(global_step=9)
    assert _sft_quality_metrics_due(state, args, SimpleNamespace(sync_gradients=False)) is False
    assert _sft_quality_metrics_due(state, args, SimpleNamespace(sync_gradients=True)) is True
    state.global_step = 8
    assert _sft_quality_metrics_due(state, args, SimpleNamespace(sync_gradients=True)) is False


def test_non_blocking_h2d_is_passed_to_sft_config():
    from flash.engine.worker import sft

    source = inspect.getsource(sft.run_sft)
    assert '"accelerator_config": {"non_blocking": True}' in source

    trl = pytest.importorskip("trl")
    cfg = trl.SFTConfig(
        output_dir="/tmp/sft-activation-shaping-test",
        use_cpu=True,
        bf16=False,
        loss_type="nll",
        accelerator_config={"non_blocking": True},
    )
    assert cfg.accelerator_config.non_blocking is True
