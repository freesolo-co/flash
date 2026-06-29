"""Tests for --set overrides and composed-config deep merge."""

from __future__ import annotations

import os
import tempfile

import pytest

BASE = (
    'model = "Qwen/Qwen3.5-4B"\n'
    'algorithm = "grpo"\n'
    '[environment]\nid = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
    '[train]\nsteps = 100\nhf_repo = "owner/runs"\n'
    '[gpu]\ntype = "RTX 5090"\n'
)


def _write(tmp, name, text):
    p = os.path.join(tmp, name)
    with open(p, "w") as f:
        f.write(text)
    return p


def test_set_overrides_scalar_and_list():
    from flash.schema import spec_from_file

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _write(tmp, "c.toml", BASE)
        spec = spec_from_file(
            cfg,
            run_id="x",
            overrides=["train.steps=7", "gpu.type=RTX 4090", "train.stop_sequences=[STOP1,STOP2]"],
        )
        assert spec.train.steps == 7
        assert spec.train.stop_sequences == ("STOP1", "STOP2")
        # GPU pinning is gone: a gpu.type override is parsed but IGNORED — the schema always
        # resolves the cheapest fitting VALIDATED class for the model (4B GRPO ~35 GB -> the 80 GB
        # A100 PCIe @ $1.39, the cheapest validated class that fits), not the override.
        assert spec.gpu.type == "A100 PCIe"


def test_composed_config_deep_merge():
    from flash.schema import spec_from_file

    with tempfile.TemporaryDirectory() as tmp:
        base = _write(tmp, "base.toml", BASE)
        override = _write(tmp, "prod.toml", '[train]\nsteps = 250\n[gpu]\ntype = "RTX 4090"\n')
        spec = spec_from_file(base, run_id="x", extra_configs=[override])
        assert spec.train.steps == 250  # deep-merged override of a scalar
        assert spec.environment.id == "github:freesolo-co/envs@main:gsm8k/environment.py"  # untouched key preserved
        # The merged [gpu] type is IGNORED (no pin): the schema resolves the cheapest fitting
        # VALIDATED class for the model (4B GRPO ~35 GB -> the 80 GB A100 PCIe @ $1.39), regardless
        # of the composed gpu.type.
        assert spec.gpu.type == "A100 PCIe"


def test_set_requires_key_value():
    from flash.schema import ConfigError, spec_from_file

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _write(tmp, "c.toml", BASE)
        with pytest.raises(ConfigError):
            spec_from_file(cfg, run_id="x", overrides=["bogus"])
