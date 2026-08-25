"""Tests for --set overrides and composed-config deep merge."""

from __future__ import annotations

import os
import tempfile

import pytest

BASE = (
    'model = "Qwen/Qwen3.5-9B"\n'
    'algorithm = "grpo"\n'
    '[environment]\nid = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
    "[train]\nepochs = 1\nmax_examples = 100\n"
    '[gpu]\ntype = "B200"\n'
)


def _write(tmp, name, text):
    p = os.path.join(tmp, name)
    with open(p, "w") as f:
        f.write(text)
    return p


def test_set_overrides_scalar_and_list():
    from flash.schema import spec_and_train_keys_from_file

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _write(tmp, "c.toml", BASE)
        spec = spec_and_train_keys_from_file(
            cfg,
            run_id="x",
            overrides=[
                "train.epochs=3",
                "gpu.type=H100",
                "train.stop_sequences=[STOP1,STOP2]",
            ],
        )[0]
        assert spec.train.epochs == 3
        assert spec.train.stop_sequences == ("STOP1", "STOP2")
        # gpu.type overrides hard-pin the requested active validated class.
        assert spec.gpu.type == "H100"


def test_composed_config_deep_merge():
    from flash.schema import spec_and_train_keys_from_file

    with tempfile.TemporaryDirectory() as tmp:
        base = _write(tmp, "base.toml", BASE)
        override = _write(
            tmp,
            "prod.toml",
            '[train]\nmax_examples = 250\n[gpu]\ntype = "H100"\n',
        )
        spec = spec_and_train_keys_from_file(base, run_id="x", extra_configs=[override])[0]
        assert spec.train.max_examples == 250  # deep-merged override of a scalar
        assert (
            spec.environment.id == "github:freesolo-co/envs@main:gsm8k/environment.py"
        )  # untouched key preserved
        # the merged gpu.type remains the authoritative hard pin.
        assert spec.gpu.type == "H100"


def test_authored_train_keys_follow_composed_config() -> None:
    from flash.schema import spec_and_train_keys_from_file

    with tempfile.TemporaryDirectory() as tmp:
        base = _write(tmp, "base.toml", BASE)
        # BASE is grpo, so the composed key has to be one grpo accepts: teacher_model is opd-only
        # and is now rejected by the parser, which would fail before key tracking is exercised.
        extra = _write(tmp, "extra.toml", "[train]\nentropy_quantile = 0.8\n")
        spec, keys = spec_and_train_keys_from_file(
            base,
            run_id="x",
            extra_configs=[extra],
            overrides=["train.temperature=0", "train.stop_sequences=[]"],
        )

    assert keys == frozenset(
        {
            "epochs",
            "max_examples",
            "entropy_quantile",
            "temperature",
            "stop_sequences",
        }
    )
    assert spec.train.temperature == 0.0
    assert spec.train.stop_sequences == ()
    assert spec.train.entropy_quantile == 0.8


def test_set_requires_key_value():
    from flash.schema import ConfigError, spec_and_train_keys_from_file

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _write(tmp, "c.toml", BASE)
        with pytest.raises(ConfigError):
            spec_and_train_keys_from_file(cfg, run_id="x", overrides=["bogus"])
