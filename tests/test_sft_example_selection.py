"""``max_examples`` must fence SFT to the HEAD of train.jsonl, slicing BEFORE the shuffle.

A dataset may carry fully-labeled SFT rows first and prompt-only (empty-output) GRPO rows
after; with shuffle-then-slice an empty completion gets sampled into SFT and trains the
model to emit nothing. Regression for the shuffle-before-slice ordering in run_sft.
"""

from __future__ import annotations

from flash.engine.worker.entry.sft import select_sft_examples


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


def test_max_examples_slices_before_enumerating_the_dataset():
    class PrefixOnlyDataset:
        def __getitem__(self, index):
            assert isinstance(index, slice)
            assert index.start is None
            assert index.step is None
            return list(range(index.stop))

        def __iter__(self):
            raise AssertionError("full dataset must not be enumerated")

    out = select_sft_examples(PrefixOnlyDataset(), 10, seed=3)
    assert sorted(out) == list(range(10))


def test_zero_or_unset_max_examples_keeps_the_full_dataset():
    rows = list(range(20))
    out = select_sft_examples(list(rows), 0, seed=3)
    assert sorted(out) == rows
