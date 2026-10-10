"""SFT context length must honor ``train.max_context_tokens`` (regression for the silent 1024 cap).

The SFT worker resolved its context length from a non-existent ``max_length`` train field, so
``max_context_tokens`` was ignored and non-thinking SFT was hard-capped at ``RECIPE.sft.max_seq_len``
(1024): a >1024-token system prompt filled the whole window, every completion was truncated to empty,
and the run aborted with "every SFT example has an empty completion after sft_max_len truncation".

The knob is read through ``getattr(train, name, None)``, which CANNOT distinguish "user did not set
it" from "I asked for a field that does not exist": both return the default. So the regression is
guarded two ways: the string key the resolver passes must be a real ``TrainSpec`` field (not the
renamed-away ``max_length``), and the trainer and the quote must resolve the same length.

The second guard used to compare two independent derivations, one in the worker and one in
``flash.cost.spec._sft_seq_len``. There is now only one: the workload profile measures the rows at
its own ``max_length``, the quote prices ``profile.max_length``, and the worker reads the same field
off the profile it trains from. That is why the lockstep assertion below is about the resolver being
the single producer rather than about two numbers agreeing: agreement between two derivations was
only ever a proxy for there being one.
"""

from __future__ import annotations

import pathlib
from dataclasses import fields

import flash.engine.profiling.sft_workload as sft_workload_mod
import flash.engine.worker.train.entry.sft_train as sft_train_mod
from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec
from flash.engine.plan.recipe import RECIPE
from flash.engine.profiling.sft_workload import sft_max_length as _measured_max_length


def test_trainspec_exposes_max_context_tokens_and_not_the_stale_max_length():
    names = {f.name for f in fields(TrainSpec)}
    assert "max_context_tokens" in names
    # `max_length` was renamed to `max_context_tokens` (#464). If it ever returns as a real field,
    # a stale `_train_opt("max_length", ...)` would silently mask it again instead of crashing.
    assert "max_length" not in names


def test_sft_preprocessing_reads_max_context_tokens_not_max_length():
    # The resolver reads `spec.train.max_context_tokens`. The name MUST be a real TrainSpec field,
    # else getattr silently returns the recipe default -- the non-thinking 1024 hard-cap that dropped
    # every row of a >1024-token-prompt dataset. Compared whitespace-insensitively so the multi-line
    # expression formatting doesn't matter.
    src = "".join(pathlib.Path(sft_workload_mod.__file__).read_text().split())
    assert "spec.train.max_context_tokens" in src
    assert "spec.train.max_length" not in src


def test_sft_context_length_resolves_the_authored_window_or_the_mode_cap():
    """The measured window is the authored cap when set, else the recipe cap for the mode.

    this is the resolution the 1024-cap regression got wrong. it now happens once, inside the
    preprocessing used by both control-plane estimation and training, so asserting it here covers
    the quote too: the quote reads ``profile.max_length``, which is this value.
    """
    for thinking in (False, True):
        recipe_cap = RECIPE.sft.max_seq_len_thinking if thinking else RECIPE.sft.max_seq_len
        for mct in (None, 2048, 3072):
            spec = JobSpec(
                model="Qwen/Qwen3.5-9B",
                algorithm="sft",
                thinking=thinking,
                environment=EnvironmentSpec(id="team/example", resolved_sha="b" * 40),
                model_revision="a" * 40,
                train=TrainSpec(max_context_tokens=mct, max_examples=2),
            )
            expected = mct if mct is not None else recipe_cap
            assert _measured_max_length(spec) == expected


def test_trainer_and_quote_read_the_one_measured_window():
    """Neither side may re-derive the window: both read the profile the rows were truncated at.

    A second derivation is invisible to the worker's parity check, which compares two values the
    workload module produced. So the guard is that no other derivation exists -- the trainer takes
    ``profile.max_length`` and the quote takes the same field off the same artifact.
    """
    from flash.cost.spec import runconfig_from_spec

    trainer_src = "".join(pathlib.Path(sft_train_mod.__file__).read_text().split())
    assert "max_length=profile.max_length" in trainer_src
    assert 'train_opt("max_context_tokens"' not in trainer_src

    from tests._helpers.profile import attach_sft_profile

    spec = attach_sft_profile(
        JobSpec(
            model="Qwen/Qwen3.5-9B",
            algorithm="sft",
            train=TrainSpec(max_context_tokens=3072, max_examples=8),
        )
    )
    assert runconfig_from_spec(spec).seq_len == spec.workload_profile["max_length"] == 3072
