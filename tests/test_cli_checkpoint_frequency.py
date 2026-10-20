"""Checkpoint cadence remains the same contract across TOML and CLI overrides."""

import pytest

from flash.cli.parsing.main import _build_parser


def test_save_every_is_a_normal_ordered_override():
    args = _build_parser().parse_args(
        [
            "train",
            "run.toml",
            "--set",
            "train.save_every=20",
            "--save-every",
            "100",
            "--set",
            "train.save_at_steps=[10,50]",
        ]
    )
    assert args.overrides == [
        "train.save_every=20",
        "train.save_every=100",
        "train.save_at_steps=[10,50]",
    ]


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "no", "true"])
def test_save_every_rejects_invalid_intervals(value):
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["train", "run.toml", "--save-every", value])
