"""Tests for `flash train --gpus N`, the first-class way to ask for a card count.

The flag is sugar for the `gpu.count` override, so these assert the wiring (does it reach the
spec?), the precedence (does the flag beat the file and a --set of the same key?) and that it
inherits the SAME validation as [gpu] count rather than getting a second, laxer one.
"""

from __future__ import annotations

import types

import pytest

BASE = (
    'model = "Qwen/Qwen3.5-4B"\n'
    'project = "11111111-1111-4111-8111-111111111111"\n'
    'algorithm = "grpo"\n'
    '[environment]\nid = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
    "[train]\nepochs = 1\nmax_examples = 100\n"
    '[gpu]\ntype = "B200"\nprovider = "runpod"\n'
)


def _args(config, *, gpus=None, overrides=None):
    return types.SimpleNamespace(
        config=str(config),
        overrides=list(overrides or []),
        extra_configs=[],
        gpus=gpus,
        cost=False,
        dry_run=False,
        background=False,
    )


def test_the_parser_exposes_gpus_on_train():
    from flash.cli import _build_parser

    # the ask is a first-class CLI item, not just `--set gpu.count=N`, so it must parse as its own
    # flag and land as an int (argparse type=int, so "4" must not reach the spec as a string).
    args = _build_parser().parse_args(["train", "run.toml", "--gpus", "4"])
    assert args.gpus == 4


def test_gpus_defaults_to_none_so_a_config_without_it_is_untouched():
    from flash.cli import _build_parser

    # None, not 1: an absent flag must leave [gpu] count alone. defaulting to 1 would silently
    # DOWNGRADE every multi-gpu config whose count lives in the file.
    args = _build_parser().parse_args(["train", "run.toml"])
    assert args.gpus is None


def test_gpus_reaches_the_spec(tmp_path):
    from flash.cli.commands import _train_overrides
    from flash.schema import spec_from_file

    cfg = tmp_path / "run.toml"
    cfg.write_text(BASE)
    spec = spec_from_file(
        str(cfg), run_id="x", overrides=_train_overrides(_args(cfg, gpus=4)), project_required=True
    )
    assert spec.gpu.count == 4


def test_no_gpus_flag_leaves_the_configs_count_alone(tmp_path):
    from flash.cli.commands import _train_overrides
    from flash.schema import spec_from_file

    # the flag is opt-in. a config that already asks for 2 cards must still get 2.
    cfg = tmp_path / "run.toml"
    cfg.write_text(BASE + "count = 2\n")
    spec = spec_from_file(
        str(cfg), run_id="x", overrides=_train_overrides(_args(cfg)), project_required=True
    )
    assert spec.gpu.count == 2


def test_gpus_beats_the_config_file(tmp_path):
    from flash.cli.commands import _train_overrides
    from flash.schema import spec_from_file

    cfg = tmp_path / "run.toml"
    cfg.write_text(BASE + "count = 2\n")
    spec = spec_from_file(
        str(cfg), run_id="x", overrides=_train_overrides(_args(cfg, gpus=4)), project_required=True
    )
    assert spec.gpu.count == 4


def test_gpus_beats_a_set_of_the_same_key(tmp_path):
    from flash.cli.commands import _train_overrides
    from flash.schema import spec_from_file

    # both spellings target gpu.count. the dedicated flag is the more specific request, so it wins
    # regardless of argv order -- otherwise `--gpus` would be silently ignored next to a --set.
    cfg = tmp_path / "run.toml"
    cfg.write_text(BASE)
    spec = spec_from_file(
        str(cfg),
        run_id="x",
        overrides=_train_overrides(_args(cfg, gpus=4, overrides=["gpu.count=8"])),
        project_required=True,
    )
    assert spec.gpu.count == 4


def test_gpus_does_not_disturb_other_overrides(tmp_path):
    from flash.cli.commands import _train_overrides
    from flash.schema import spec_from_file

    # folding the flag into the --set list must APPEND, not replace: an unrelated --set alongside
    # --gpus has to survive.
    cfg = tmp_path / "run.toml"
    cfg.write_text(BASE)
    spec = spec_from_file(
        str(cfg),
        run_id="x",
        overrides=_train_overrides(_args(cfg, gpus=2, overrides=["train.epochs=3"])),
        project_required=True,
    )
    assert spec.gpu.count == 2
    assert spec.train.epochs == 3


@pytest.mark.parametrize("bad", [0, -1, 9, 100])
def test_gpus_is_validated_by_the_same_bound_as_the_config_key(tmp_path, bad):
    from flash.cli.commands import _train_overrides
    from flash.schema import spec_from_file

    # the flag routes through gpu.count, so JobSpec's 1..8 bound is the ONLY validator. a second
    # bound in the CLI could drift from it; assert the shared one actually fires on the flag.
    cfg = tmp_path / "run.toml"
    cfg.write_text(BASE)
    with pytest.raises(ValueError, match=r"gpu\.count must be between 1 and 8"):
        spec_from_file(
            str(cfg),
            run_id="x",
            overrides=_train_overrides(_args(cfg, gpus=bad)),
            project_required=True,
        )


def test_gpus_still_hits_the_multi_gpu_provider_gate(tmp_path):
    from flash import runner
    from flash.cli.commands import _train_overrides
    from flash.schema import spec_from_file

    # a convenient flag must not become a way AROUND the submit gate: vast/lambda ignore gpu_count,
    # so 4 ranks would land on one rented card and be billed for four.
    # H100 rather than the shared BASE's B200: vast does not list B200 at all, so that pairing
    # would be rejected by the catalog before the multi-gpu gate this test is about.
    cfg = tmp_path / "run.toml"
    cfg.write_text(BASE.replace('type = "B200"', 'type = "H100"').replace("runpod", "vast"))
    spec = spec_from_file(
        str(cfg), run_id="x", overrides=_train_overrides(_args(cfg, gpus=4)), project_required=True
    )
    with pytest.raises(ValueError, match=r"gpu\.provider"):
        runner._require_supported_gpu_count(spec)
