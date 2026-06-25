"""Tests for --set overrides and composed-config deep merge."""

from __future__ import annotations

import os
import tempfile

import pytest

BASE = (
    'model = "Qwen/Qwen3.5-4B"\n'
    'algorithm = "grpo"\n'
    '[environment]\nid = "freesolo-co/gsm8k"\n'
    '[train]\nsteps = 100\nseeds = [0]\nhf_repo = "owner/runs"\n'
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
            cfg, run_id="x", overrides=["train.steps=7", "gpu.type=RTX 4090", "train.seeds=[0,1,2]"]
        )
        assert spec.train.steps == 7
        assert spec.train.seeds == (0, 1, 2)
        # GPU pinning is gone: a gpu.type override is parsed but IGNORED — the schema always
        # resolves the cheapest fitting VALIDATED class for the model (4B GRPO ~35 GB -> the 48 GB
        # RTX A6000, validated 2026-06-22 @ $0.49, cheaper than the 80 GB A100 PCIe), not the
        # override. (A40 stays unvalidated.)
        assert spec.gpu.type == "RTX A6000"


def test_composed_config_deep_merge():
    from flash.schema import spec_from_file

    with tempfile.TemporaryDirectory() as tmp:
        base = _write(tmp, "base.toml", BASE)
        override = _write(tmp, "prod.toml", '[train]\nsteps = 250\n[gpu]\ntype = "RTX 4090"\n')
        spec = spec_from_file(base, run_id="x", extra_configs=[override])
        assert spec.train.steps == 250  # deep-merged override of a scalar
        assert spec.environment.id == "freesolo-co/gsm8k"  # untouched key preserved
        # The merged [gpu] type is IGNORED (no pin): the schema resolves the cheapest fitting
        # VALIDATED class for the model (4B GRPO ~35 GB -> the 48 GB RTX A6000, $0.49), regardless
        # of the composed gpu.type.
        assert spec.gpu.type == "RTX A6000"


def test_set_requires_key_value():
    from flash.schema import ConfigError, spec_from_file

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _write(tmp, "c.toml", BASE)
        with pytest.raises(ConfigError):
            spec_from_file(cfg, run_id="x", overrides=["bogus"])
