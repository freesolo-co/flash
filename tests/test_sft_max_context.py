"""SFT context length must honor ``train.max_context_tokens`` (regression for the silent 1024 cap).

The SFT worker resolved its context length from a non-existent ``max_length`` train field, so
``max_context_tokens`` was ignored and non-thinking SFT was hard-capped at ``RECIPE.sft.max_seq_len``
(1024): a >1024-token system prompt filled the whole window, every completion was truncated to empty,
and the run aborted with "every SFT example has an empty completion after sft_max_len truncation".

The worker reads the knob through ``_train_opt(name, default)`` -> ``getattr(train, name, None)``,
which CANNOT distinguish "user did not set it" from "I asked for a field that does not exist" — both
return the default. So the regression is guarded two ways: the string key the worker passes must be a
real ``TrainSpec`` field (not the renamed-away ``max_length``), and the resolved length must stay in
lockstep with the cost/preflight path (``flash.cost.spec._sft_seq_len``).
"""

from __future__ import annotations

import pathlib
from dataclasses import fields

import flash.engine.worker.sft_train as sft_train_mod
from flash.engine.recipe import RECIPE
from flash.spec import JobSpec, TrainSpec


def test_trainspec_exposes_max_context_tokens_and_not_the_stale_max_length():
    names = {f.name for f in fields(TrainSpec)}
    assert "max_context_tokens" in names
    # `max_length` was renamed to `max_context_tokens` (#464). If it ever returns as a real field,
    # a stale `_train_opt("max_length", ...)` would silently mask it again instead of crashing.
    assert "max_length" not in names


def test_sft_worker_reads_max_context_tokens_not_max_length():
    # The worker resolves sft_max_len via `train_opt("<key>", <recipe default>)`. The key MUST be a
    # real TrainSpec field, else getattr(_t, key, None) silently returns the recipe default — the
    # non-thinking 1024 hard-cap that dropped every row of a >1024-token-prompt dataset. Compared
    # whitespace-insensitively so the multi-line call formatting doesn't matter.
    src = "".join(pathlib.Path(sft_train_mod.__file__).read_text().split())
    assert 'train_opt("max_context_tokens"' in src
    assert 'train_opt("max_length"' not in src


def test_sft_context_length_matches_cost_preflight_resolution():
    # The trainer must resolve the SAME context length the cost/preflight path budgeted (else a run
    # is billed/preflighted for one window and trained at another). This mirrors the exact fallback
    # the worker applies: [train] max_context_tokens when set, else the recipe cap for the mode.
    from flash.cost.spec import _sft_seq_len

    for thinking in (False, True):
        recipe_cap = RECIPE.sft.max_seq_len_thinking if thinking else RECIPE.sft.max_seq_len
        for mct in (None, 2048, 3072):
            spec = JobSpec(
                model="Qwen/Qwen3.5-4B",
                algorithm="sft",
                thinking=thinking,
                model_policy="catalog",
                train=TrainSpec(max_context_tokens=mct),
            )
            expected = mct if mct is not None else recipe_cap
            assert _sft_seq_len(spec) == expected
