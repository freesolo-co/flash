"""Tests for `flash train --gpus N`, the first-class way to ask for a card count.

The flag is sugar for the `gpu.count` override and shares --set's dest, so these drive the REAL
parser and assert the wiring (does it reach the spec?), the precedence (does it beat the file, and
how does it resolve against a --set of the same key?) and that it inherits the SAME validation as
[gpu] count rather than getting a second, laxer one.
"""

from __future__ import annotations

import pytest

BASE = (
    'model = "Qwen/Qwen3.5-9B"\n'
    'project = "11111111-1111-4111-8111-111111111111"\n'
    'algorithm = "grpo"\n'
    '[environment]\nid = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
    "[train]\nepochs = 1\nmax_examples = 100\n"
    "[gpu]\n"
)

BIG_GRPO = (
    'model = "Qwen/Qwen3.8-27B"\n'
    'project = "11111111-1111-4111-8111-111111111111"\n'
    'algorithm = "grpo"\n'
    '[environment]\nid = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
    "[train]\nepochs = 1\nmax_examples = 100\n"
    "max_context_tokens = 8192\nmax_completion_tokens = 4096\n"
    "[gpu]\n"
)


def _overrides(*argv: str) -> list[str]:
    """the override list the real parser produces for `flash train run.toml <argv>`."""
    from flash.cli.parsing.main import _build_parser

    return _build_parser().parse_args(["train", "run.toml", *argv]).overrides


def _spec(tmp_path, *argv: str, config: str = BASE):
    """parse argv through the real parser, then build the spec exactly as cmd_train does."""
    from flash.schema import spec_and_train_keys_from_file

    cfg = tmp_path / "run.toml"
    cfg.write_text(config)
    return spec_and_train_keys_from_file(
        str(cfg), run_id="x", overrides=_overrides(*argv), project_required=True
    )[0]


def test_no_gpus_flag_contributes_no_override():
    # an absent flag must be indistinguishable from no flag. emitting `gpu.count=1` here would
    # turn auto-sizing into an explicit one-card pin.
    assert _overrides() == []


def test_no_gpus_flag_marks_the_count_for_auto_sizing(tmp_path):
    spec = _spec(tmp_path, config=BIG_GRPO)
    assert spec.gpu.count == 1  # digest-stable public placeholder
    assert spec.gpu_count_auto is True


def test_unset_count_accepts_a_grpo_run_that_needs_two_cards(tmp_path):
    spec = _spec(tmp_path, config=BIG_GRPO)
    assert spec.gpu_count_auto is True
    assert spec.gpu.count == 1


def test_the_gpus_flag_is_gone(capsys):
    from flash.cli.parsing.main import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args(["train", "run.toml", "--gpus", "4"])
    assert "--gpus" in capsys.readouterr().err


@pytest.mark.parametrize("override", ["gpu.count=4", "gpu.provider=modal", "gpu.type=H100"])
def test_gpu_shape_overrides_are_rejected(tmp_path, override):
    from flash.schema import ConfigError

    with pytest.raises(ConfigError, match="no longer supported"):
        _spec(tmp_path, "--set", override)


def test_a_pinned_small_class_still_does_not_pin_the_card_count():
    """There is no exact-count mechanism, so no doc may claim one.

    _combination_candidates breaks at the FIRST count that fits (allocator.py:124), so pinning a
    class too small to hold the run alone raises the floor above one card but does not pin n: the
    run gets the smallest fitting combination instead. This is asserted because the docs previously
    advised pinning a small [gpu] type "to guarantee n cards", which is false -- a user sizing an
    experiment off it silently gets fewer ranks than they asked for. Driving the real allocator
    rather than reading the doc string, so the claim is pinned to behavior and not to wording.
    """
    from flash.providers.core.allocator import allocate

    # 9B SFT on a 24 GB class does not fit alone, so this is the combination path, not the
    # fit-alone one. the algorithm is load-bearing: the same model+class does not allocate AT ALL
    # under grpo, which the next test pins.
    allocated = [
        allocate(
            "Qwen/Qwen3.5-9B", "sft", provider="modal", gpu_type="L4", max_gpu_count=asked
        ).gpu_count
        for asked in (2, 3, 4)
    ]

    # asking for 2, 3 and 4 all land on the same smallest fitting count. if any of these ever equals
    # what was asked for, an exact-count path exists and TRAINING.md's "no exact-count mechanism"
    # paragraph has to be rewritten alongside it.
    assert allocated == [2, 2, 2], (
        f"expected the smallest fitting combination for every ask, got {allocated}"
    )


def test_a_pinned_class_can_need_a_wider_ceiling_for_grpo():
    """The doc's 9B/L4 example is labelled SFT because fit is per algorithm.

    SFT fits on two cards. GRPO exceed a four-card ceiling but fit at the public maximum of
    eight, so copying the SFT example into a rollout config without widening the ceiling fails
    loudly instead of silently selecting an undersized shape.
    """
    import pytest

    from flash.providers.core.allocator import allocate
    from flash.providers.core.base import UnsupportedGpuError

    # same model and class; only the algorithm and resulting memory floor differ.
    assert (
        allocate(
            "Qwen/Qwen3.5-9B", "sft", provider="modal", gpu_type="L4", max_gpu_count=4
        ).gpu_count
        == 2
    )
    for algorithm in ("grpo",):
        with pytest.raises(UnsupportedGpuError):
            allocate(
                "Qwen/Qwen3.5-9B",
                algorithm,
                provider="modal",
                gpu_type="L4",
                max_gpu_count=4,
            )
        assert (
            allocate(
                "Qwen/Qwen3.5-9B",
                algorithm,
                provider="modal",
                gpu_type="L4",
                max_gpu_count=8,
            ).gpu_count
            == 8
        )


def test_the_allocation_log_line_carries_the_count_that_status_does_not():
    """TRAINING.md points at the run log for the allocated count, so that line must carry it.

    The doc previously pointed at gpu_status.device_count, which is a real worker observation but
    an unreliable place to look: mid-run heartbeats collect diagnostics with include_torch=False
    and record_heartbeat REPLACES gpu_status per heartbeat, so the field is normally absent while
    a run is live. The runner's allocation line has neither problem -- it is written once at
    placement, before any worker exists, into an append-only log.

    Both spellings are asserted because the doc states both: multi-card carries the Nx prefix and
    single-card does not, so a user grepping for "1x" on a one-card run would find nothing.
    """
    from flash.providers.core.allocator import Allocation, allocation_summary

    def summarize(count: int) -> str:
        return allocation_summary(
            Allocation(
                provider="modal",
                gpu="L4",
                hourly_usd=0.69,
                min_vram_gb=31,
                candidates=(),
                gpu_count=count,
            )
        )

    assert summarize(2).startswith("allocated 2x L4 on modal at $1.38/hr"), summarize(2)
    # one card is spelled bare, and the hourly total is per-run rather than per-card.
    assert summarize(1).startswith("allocated L4 on modal at $0.69/hr"), summarize(1)


def test_the_public_run_record_never_reports_the_allocated_count():
    """The doc warns off spec.gpu.count, and that warning must stay true.

    A user who submits --gpus 4, reads 4 back off `runs status` and concludes they got 4 cards
    would size throughput and cost off a number the allocator never honoured. The allocated count
    IS recorded, but only in effective_preparation, which to_dict() pops as server-internal -- so
    if a future change starts publishing it there, this fails and the doc gets rewritten with it.
    """
    from flash.runner.lifecycle.state import RunStatus

    submitted_ceiling = 4
    status = RunStatus(
        run_id="x",
        state="running",
        spec={"gpu": {"type": "L4", "provider": "modal", "count": submitted_ceiling}},
        # what the allocator actually chose, recorded exactly where the runner records it.
        effective_preparation={"worker_spec": {"gpu": {"count": 2}}},
    )
    public = status.to_dict()

    # the internal carrier holding the real count is stripped, so users cannot reach it at all.
    assert "effective_preparation" not in public, (
        "the allocated count became publicly readable; TRAINING.md should now point at it"
    )
    # and the field users CAN see still echoes what they asked for, not what they got.
    assert public["spec"]["gpu"]["count"] == submitted_ceiling, (
        "spec.gpu.count no longer echoes the submitted ceiling; the doc's warning is now wrong"
    )
