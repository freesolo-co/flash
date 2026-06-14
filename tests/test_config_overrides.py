"""Tests for --set overrides and composed-config deep merge."""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

BASE = (
    'model = "Qwen/Qwen3-4B-Instruct-2507"\n'
    'algorithm = "grpo"\n'
    '[environment]\nid = "gsm8k"\n'
    "[train]\nsteps = 100\nseeds = [0]\n"
    '[gpu]\ntype = "RTX 5090"\n'
)


def _write(tmp, name, text):
    p = os.path.join(tmp, name)
    with open(p, "w") as f:
        f.write(text)
    return p


def test_set_overrides_scalar_and_list():
    from autoslm.config_schema import spec_from_file

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _write(tmp, "c.toml", BASE)
        spec = spec_from_file(
            cfg, run_id="x", overrides=["train.steps=7", "gpu.type=RTX 4090", "train.seeds=[0,1,2]"]
        )
        assert spec.train.steps == 7
        assert spec.gpu.type == "RTX 4090"
        assert spec.train.seeds == (0, 1, 2)


def test_composed_config_deep_merge():
    from autoslm.config_schema import spec_from_file

    with tempfile.TemporaryDirectory() as tmp:
        base = _write(tmp, "base.toml", BASE)
        override = _write(tmp, "prod.toml", '[train]\nsteps = 250\n[gpu]\ntype = "RTX 4090"\n')
        spec = spec_from_file(base, run_id="x", extra_configs=[override])
        assert spec.train.steps == 250
        assert spec.gpu.type == "RTX 4090"
        assert spec.environment.id == "gsm8k"  # untouched key preserved


def test_set_requires_key_value():
    from autoslm.config_schema import ConfigError, spec_from_file

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _write(tmp, "c.toml", BASE)
        with pytest.raises(ConfigError):
            spec_from_file(cfg, run_id="x", overrides=["bogus"])
