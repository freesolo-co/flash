"""Cost estimator: measured-run ground truth (spec -> RunConfig, measured wrapper).

No network -- uses a fixture control-plane status payload shaped like `slm status`.
"""

from __future__ import annotations

import pytest

from flash.cost.experiment import decaying_stub_factory, run_experiment
from flash.cost.measured import (
    measured_as_estimate,
    measured_from_status,
    measured_ground_truth_fn,
    runconfig_from_status,
)

GRPO_STATUS = {
    "run_id": "flash-1-abc",
    "state": "done",
    "cost_usd": 0.0473,
    "created_at": 1000.0,
    "updated_at": 1367.0,  # wall = 367 - (started offset)
    "spec": {
        "model": "openbmb/MiniCPM5-1B",
        "algorithm": "grpo",
        "thinking": False,
        "environment": {"id": "freesolo-co/flash-bench"},
        "gpu": {"type": "RTX 5090"},
        "train": {
            "steps": 20,
            "group_size": 4,
            "batch_size": 16,
            "max_tokens": 256,
            "max_length": 768,
            "lora_rank": 32,
        },
    },
    "remote": {
        "allocated_gpu": "RTX 5090",
        "provider": "vast",
        "hourly_usd": 0.4689,
        "started_ts": 1000.7,
    },
}

SFT_STATUS = {
    "run_id": "flash-2-def",
    "state": "done",
    "cost_usd": 0.02,
    "created_at": 0.0,
    "updated_at": 300.0,
    "spec": {
        "model": "openbmb/MiniCPM5-1B",
        "algorithm": "sft",
        "thinking": False,
        "environment": {"id": "freesolo-co/flash-bench"},
        "gpu": {"type": "RTX 5090"},
        "train": {"epochs": 1, "max_examples": 512, "batch_size": 16, "max_length": 768},
    },
    "remote": {"allocated_gpu": "RTX 5090", "provider": "vast", "hourly_usd": 0.47},
}


def test_grpo_runconfig_from_status():
    cfg = runconfig_from_status(GRPO_STATUS)
    assert cfg.model_id == "openbmb/MiniCPM5-1B"
    assert cfg.method == "grpo"
    assert cfg.steps == 20
    assert cfg.group_size == 4
    assert cfg.completion_len == 256
    assert cfg.gpu == "RTX 5090"
    assert cfg.environment == "freesolo-co/flash-bench"


def test_sft_steps_derived_from_epochs_and_examples():
    cfg = runconfig_from_status(SFT_STATUS)
    assert cfg.method == "sft"
    assert cfg.steps == 32  # ceil(512 / 16) * 1 epoch
    assert cfg.completion_len is None


def test_grpo_steps_not_capped_by_max_steps():
    # max_steps is an SFT-only optimizer-step cap; GRPO runs all the steps it carries, so a
    # stray max_steps in a GRPO spec must NOT clamp the reconstructed (and priced) count.
    status = {
        **GRPO_STATUS,
        "spec": {**GRPO_STATUS["spec"], "train": {**GRPO_STATUS["spec"]["train"], "max_steps": 5}},
    }
    cfg = runconfig_from_status(status)
    assert cfg.steps == 20  # the run's GRPO steps, unaffected by the SFT-only max_steps


def test_sft_steps_cap_respected():
    # For SFT, max_steps DOES bound the reconstructed count (the worker's optimizer-step cap),
    # so an estimate can't exceed what actually ran.
    status = {
        **SFT_STATUS,
        "spec": {**SFT_STATUS["spec"], "train": {**SFT_STATUS["spec"]["train"], "max_steps": 8}},
    }
    cfg = runconfig_from_status(status)
    assert cfg.steps == 8  # min(ceil(512/16)=32, 8)


def test_sft_omitted_batch_uses_recipe_effective_batch():
    # An omitted SFT batch_size reconstructs against the recipe EFFECTIVE batch (32), the
    # value the worker sizes grad-accum to hit -- not a bare per-device default. So a
    # 512-example / 1-epoch run is 16 steps (ceil(512/32)), not 64 (ceil(512/8)).
    status = {
        "spec": {
            "model": "openbmb/MiniCPM5-1B",
            "algorithm": "sft",
            "train": {"epochs": 1, "max_examples": 512},  # batch_size omitted
        }
    }
    cfg = runconfig_from_status(status)
    assert cfg.steps == 16


def test_grpo_omitted_steps_default_to_recipe_num_steps():
    # A GRPO run that omits train.steps runs RECIPE.rl.num_steps (worker falls back to it),
    # not the SFT 100 last-resort -- so a default-GRPO measured run isn't priced as shorter.
    from flash.engine.recipe import RECIPE

    status = {
        "spec": {
            "model": "openbmb/MiniCPM5-1B",
            "algorithm": "grpo",
            "train": {"group_size": 4, "max_tokens": 256},  # steps omitted
        }
    }
    cfg = runconfig_from_status(status)
    assert cfg.steps == RECIPE.rl.num_steps  # 150, not the SFT-path 100


def test_multi_seed_run_scales_steps_by_seed_count():
    # A multi-seed run executes the full step count once per seed and bills the sum, so the
    # reconstructed config must price the combined work (3 seeds x 20 steps = 60).
    status = {
        **GRPO_STATUS,
        "spec": {
            **GRPO_STATUS["spec"],
            "train": {**GRPO_STATUS["spec"]["train"], "seeds": [0, 1, 2]},
        },
    }
    cfg = runconfig_from_status(status)
    assert cfg.steps == 60  # 20 steps x 3 seeds
    assert cfg.setup_repeats == 3  # each seed reprovisions and re-pays the cold start
    # A single (default) seed is unchanged.
    single = runconfig_from_status(GRPO_STATUS)
    assert single.steps == 20
    assert single.setup_repeats == 1


def test_multi_seed_run_scales_setup_in_analytical_cost():
    # The repeated per-seed setup must show up in the priced cost: a 3-seed run pays 3x
    # the cold-start setup (plus 3x the steps), not a single setup -- otherwise the
    # analytical estimate under-prices the multi-seed bill the measured dollars cover.
    from flash.cost.analytical import estimate_cost

    one_seed = estimate_cost(runconfig_from_status(GRPO_STATUS))
    three_seed = estimate_cost(
        runconfig_from_status(
            {
                **GRPO_STATUS,
                "spec": {
                    **GRPO_STATUS["spec"],
                    "train": {**GRPO_STATUS["spec"]["train"], "seeds": [0, 1, 2]},
                },
            }
        )
    )
    # Setup is charged once per seed: 3 seeds -> 3x the single-run setup.
    assert three_seed.setup_seconds == pytest.approx(one_seed.setup_seconds * 3)
    # And it carries through to wall clock (3x steps + 3x setup), so cost scales by 3x
    # exactly here (no wall cap, single per-step rate, same $/hr).
    assert three_seed.wall_clock_seconds == pytest.approx(one_seed.wall_clock_seconds * 3)
    assert three_seed.total_usd == pytest.approx(one_seed.total_usd * 3)


def test_duplicate_measured_labels_are_rejected():
    # Two runs with the same display label can't both be the ground truth for one cell;
    # reject the collision instead of silently keeping the last and mis-grading the rest.
    a = measured_from_status(GRPO_STATUS, label="same-label")
    b = measured_from_status(SFT_STATUS, label="same-label")
    with pytest.raises(ValueError, match="duplicate measured-run label"):
        measured_ground_truth_fn([a, b])


def test_measured_from_status_fields():
    m = measured_from_status(GRPO_STATUS)
    assert m.ok
    assert m.cost_usd == pytest.approx(0.0473)
    assert m.wall_seconds == pytest.approx(366.3)  # updated - started_ts
    assert m.gpu == "RTX 5090"
    assert m.provider == "vast"
    assert m.hourly_usd == pytest.approx(0.4689)


def test_measured_as_estimate_is_the_ground_truth():
    m = measured_from_status(GRPO_STATUS)
    e = measured_as_estimate(m)
    assert e.total_usd == pytest.approx(m.cost_usd)
    assert e.gpu == m.gpu
    assert any("MEASURED" in n for n in e.notes)


def test_measured_ground_truth_grades_through_the_harness():
    runs = [measured_from_status(GRPO_STATUS), measured_from_status(SFT_STATUS)]
    grid = [r.config for r in runs]
    gt = measured_ground_truth_fn(runs)
    result = run_experiment(
        grid=grid,
        versions=(1, 6),
        estimator_factory=decaying_stub_factory(ground_truth_fn=gt),
        ground_truth_fn=gt,
        max_workers=2,
    )
    # The stub matches its (measured) ground truth at the top version.
    assert result.summary(6).mape == pytest.approx(0.0)
    # Truth dollars came from the measured runs.
    truths = sorted(c.truth_usd for c in result.cells if c.version == 6)
    assert truths == pytest.approx([0.02, 0.0473])
