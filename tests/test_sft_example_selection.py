"""``max_examples`` must fence SFT to the HEAD of train.jsonl, slicing BEFORE the shuffle.

A dataset may carry fully-labeled SFT rows first and prompt-only (empty-output) GRPO rows
after; with shuffle-then-slice an empty completion gets sampled into SFT and trains the
model to emit nothing. Regression for the shuffle-before-slice ordering in run_sft.
"""

from __future__ import annotations

import pytest

from flash.engine.worker.sft import (
    reasoning_target_coverage,
    select_sft_examples,
    validate_sft_gold_source,
)


def test_max_examples_is_a_prefix_fence_not_a_random_subsample():
    # rows 0..9 are labeled SFT rows; rows 10..19 are prompt-only GRPO rows (output="")
    labeled = [{"input": f"q{i}", "output": f"a{i}"} for i in range(10)]
    prompt_only = [{"input": f"p{i}", "output": ""} for i in range(10)]
    picked = select_sft_examples(labeled + prompt_only, 10, seed=0)
    assert len(picked) == 10
    assert sorted(r["input"] for r in picked) == sorted(r["input"] for r in labeled)
    # the property that actually matters: no empty completion leaks into the SFT sample
    assert all(r["output"] for r in picked)


def test_selection_is_seed_deterministic_but_still_shuffled():
    rows = list(range(100))
    a = select_sft_examples(list(rows), 50, seed=7)
    b = select_sft_examples(list(rows), 50, seed=7)
    assert a == b  # same seed -> same training order
    assert sorted(a) == list(range(50))  # exactly the head of the file...
    assert a != list(range(50))  # ...but shuffled for training


def test_zero_or_unset_max_examples_keeps_the_full_dataset():
    rows = list(range(20))
    out = select_sft_examples(list(rows), 0, seed=3)
    assert sorted(out) == rows


def test_reasoning_target_coverage_counts_closed_assistant_blocks_only():
    completions = [
        [{"role": "assistant", "content": "<think>a</think>answer"}],
        [{"role": "assistant", "content": "answer only"}],
        [{"role": "user", "content": "literal <think>user</think>"}],
        [{"role": "tool", "content": "literal <think>tool</think>"}],
        [{"role": "assistant", "content": "<think>unclosed"}],
        [{"role": "assistant", "content": "</think>backwards<think>"}],
        [
            {"role": "assistant", "content": "tool call"},
            {"role": "assistant", "content": "<think>b</think>done"},
        ],
    ]
    assert reasoning_target_coverage(completions) == (2, 7, pytest.approx(2 / 7))
    assert reasoning_target_coverage([]) == (0, 0, 0.0)


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3.5-4B-Base", "Qwen/Qwen3-4B-Base"])
def test_worker_defense_rejects_raw_base_self_gold(model_id):
    with pytest.raises(ValueError, match=r"raw base.*self-generated"):
        validate_sft_gold_source(model_id, "self")
    with pytest.raises(ValueError, match=r"existing.*cannot prove oracle"):
        validate_sft_gold_source(model_id, "existing")
    validate_sft_gold_source(model_id, "oracle")


def test_worker_allows_instruct_and_unknown_self_gold():
    validate_sft_gold_source("Qwen/Qwen3.5-4B", "self")
    validate_sft_gold_source("uncatalogued/instruct", "self")
