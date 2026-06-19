"""Cost estimator: the progressive prompt layers.

No network. The system prompt must grow monotonically across versions and stay
faithful to the live framework data it is rendered from.
"""

from __future__ import annotations

import itertools

import pytest

from flash.cost import RunConfig
from flash.cost import analytical as A
from flash.cost.prompts import (
    BASE_SYSTEM,
    NUM_VERSIONS,
    VERSION_LABELS,
    prompt_for_version,
    render_run,
)
from flash.engine.recipe import RECIPE


def test_version_constants():
    assert NUM_VERSIONS == 6
    assert set(VERSION_LABELS) == set(range(1, NUM_VERSIONS + 1))


def test_v1_is_base_only():
    assert prompt_for_version(1).strip() == BASE_SYSTEM.strip()


def test_prompt_grows_monotonically():
    lengths = [len(prompt_for_version(v)) for v in range(1, NUM_VERSIONS + 1)]
    assert lengths == sorted(lengths)
    assert all(a < b for a, b in itertools.pairwise(lengths))


def test_out_of_range_version_raises():
    for bad in (0, NUM_VERSIONS + 1, -1):
        with pytest.raises(ValueError, match="version must be"):
            prompt_for_version(bad)


def test_layers_appear_at_the_right_version():
    # Each layer's section header shows up exactly from its version onward.
    markers = {
        2: "## GPU hardware & pricing",
        3: "## How Flash picks the GPU",
        4: "## Per-step compute / throughput",
        5: "## GRPO rollout cost, MoE, and QLoRA",
        6: "## Cold-start overhead and the exact cost formula",
    }
    for v, header in markers.items():
        assert header not in prompt_for_version(v - 1)
        assert header in prompt_for_version(v)


def test_pricing_layer_renders_live_data():
    p = prompt_for_version(2)
    # A real validated class and its registry price appear verbatim.
    assert "RTX 5090" in p
    assert "$ 0.99/hr" in p


def test_timing_layer_exposes_calibration_constants():
    p = prompt_for_version(4)
    assert f"MFU_train = {A.MFU_TRAIN}" in p
    # recipe SFT tokens/step is spelled out
    assert str(RECIPE.sft.effective_batch * RECIPE.sft.max_seq_len) in p


def test_formula_layer_states_the_cap_and_formula():
    p = prompt_for_version(6)
    assert f"{A.DEFAULT_WALL_CAP_S / 3600:.0f}h" in p
    assert "cost_usd" in p
    assert "gpu_hourly_usd" in p


def test_render_run_describes_the_config():
    sft = render_run(RunConfig("Qwen/Qwen3.5-4B", "sft", 250))
    assert all(tok in sft for tok in ("Qwen/Qwen3.5-4B", "SFT", "250", "effective_batch"))
    assert "completion" not in sft

    grpo = render_run(RunConfig("Qwen/Qwen3.5-4B", "grpo", 150))
    assert all(tok in grpo for tok in ("GRPO", "completion", "group"))


def test_render_run_includes_gpu_pin_and_environment():
    r = render_run(
        RunConfig("Qwen/Qwen3.5-4B", "grpo", 100, gpu="RTX 5090", environment="acme/math")
    )
    assert "RTX 5090" in r
    assert "acme/math" in r
