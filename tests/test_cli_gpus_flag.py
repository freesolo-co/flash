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
    '[gpu]\ntype = "B200"\nprovider = "runpod"\n'
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


def test_the_parser_turns_gpus_into_the_override_it_is_sugar_for():
    # the ask is a first-class CLI item, not something the user spells as --set. it parses as its
    # own flag, and the conversion to gpu.count happens at parse time so nothing downstream needs
    # to know the flag exists.
    assert _overrides("--gpus", "4") == ["gpu.count=4"]


def test_no_gpus_flag_contributes_no_override():
    # an absent flag must be indistinguishable from no flag. emitting `gpu.count=1` here would
    # turn auto-sizing into an explicit one-card pin.
    assert _overrides() == []


def test_no_gpus_flag_marks_the_count_for_auto_sizing(tmp_path):
    spec = _spec(tmp_path, config=BIG_GRPO)
    assert spec.gpu.count == 1  # digest-stable public placeholder
    assert spec.gpu_count_auto is True


def test_gpus_reaches_the_spec(tmp_path):
    assert _spec(tmp_path, "--gpus", "4").gpu.count == 4


def test_unset_count_accepts_a_grpo_run_that_needs_two_cards(tmp_path):
    spec = _spec(tmp_path, config=BIG_GRPO)
    assert spec.gpu_count_auto is True
    assert spec.gpu.count == 1


def test_explicit_one_card_pin_rejects_with_a_fitting_count(tmp_path):
    from flash.schema import ConfigError

    with pytest.raises(ConfigError) as exc:
        _spec(tmp_path, "--gpus", "1", config=BIG_GRPO)
    message = str(exc.value)
    assert "grpo needs >= 235 GB VRAM" in message
    assert "gpu.count=1 provides at most 180 GB (B200)" in message
    assert "--gpus 2" in message
    assert "max_context_tokens" in message
    assert "max_completion_tokens" in message
    assert "lora_rank" in message
    assert "batch_size" not in message


def test_no_gpus_flag_leaves_the_configs_count_alone(tmp_path):
    # the flag is opt-in. a config that already asks for 2 cards must still get 2.
    assert _spec(tmp_path, config=BASE + "count = 2\n").gpu.count == 2


def test_gpus_beats_the_config_file(tmp_path):
    assert _spec(tmp_path, "--gpus", "4", config=BASE + "count = 2\n").gpu.count == 4


def test_gpus_and_set_of_the_same_key_resolve_left_to_right(tmp_path):
    # both spellings target gpu.count and land in ONE list, so they resolve by argv order exactly
    # as two --set of the same key already do. no spelling gets a hidden priority: what the user
    # typed last is what they meant.
    assert _spec(tmp_path, "--gpus", "4", "--set", "gpu.count=8").gpu.count == 8
    assert _spec(tmp_path, "--set", "gpu.count=8", "--gpus", "4").gpu.count == 4


def test_gpus_does_not_disturb_other_overrides(tmp_path):
    # sharing the dest must APPEND, not replace: an unrelated --set alongside --gpus has to survive.
    spec = _spec(tmp_path, "--set", "train.epochs=3", "--gpus", "2")
    assert spec.gpu.count == 2
    assert spec.train.epochs == 3


def test_repeatable_flags_keep_argv_order():
    # the shared dest must preserve interleaving, not group by flag. asserting the list itself
    # (rather than only the resolved spec) pins the ordering the precedence rule depends on.
    assert _overrides("--gpus", "2", "--set", "train.epochs=3", "--gpus", "4") == [
        "gpu.count=2",
        "train.epochs=3",
        "gpu.count=4",
    ]


@pytest.mark.parametrize("bad", ["0", "-1", "9", "100"])
def test_gpus_is_validated_by_the_same_bound_as_the_config_key(tmp_path, bad):
    # the flag routes through gpu.count, so JobSpec's 1..8 bound is the ONLY validator. a second
    # bound in the CLI could drift from it; assert the shared one actually fires on the flag.
    with pytest.raises(ValueError, match=r"gpu\.count must be between 1 and 8"):
        _spec(tmp_path, "--gpus", bad)


@pytest.mark.parametrize("bad", ["abc", "2.5", ""])
def test_a_non_integer_gpus_is_refused_at_parse_time(bad, capsys):
    # `--gpus 2.5` must not reach the config as the literal string "gpu.count=2.5" and get
    # diagnosed as a TOML problem. argparse rejects it where the user typed it.
    from flash.cli.parsing.main import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args(["train", "run.toml", "--gpus", bad])
    assert "--gpus" in capsys.readouterr().err


@pytest.mark.parametrize("provider", ["runpod", "lambda", "vast"])
def test_gpus_is_accepted_on_every_provider(tmp_path, provider):
    """The flag carries no provider special-casing: 4 cards is a legal ask on all three.

    This replaces a test of the old RunPod-only submit gate. That gate existed because Lambda and
    Vast ignored the count and would have landed 4 ranks on one rented card while billing for four;
    both now rent the allocated shape (Lambda names the count in the instance type, Vast filters
    offers by num_gpus), so rejecting the ask is no longer correct for any provider.

    H100 rather than the shared BASE's B200: it is the one managed class all three stock, so the
    spec is not rejected on catalog grounds before the count is ever considered.
    """
    config = BASE.replace('type = "B200"', 'type = "H100"').replace('"runpod"', f'"{provider}"')
    assert _spec(tmp_path, "--gpus", "4", config=config).gpu.count == 4


def test_the_help_explains_auto_sizing_and_authored_ceilings():
    """The flag must distinguish an omitted auto-size from an authored hard ceiling."""
    from flash.cli.parsing.main import _build_parser

    # render the train subparser's help, which is the text `flash train --help` prints.
    sub = _build_parser()._subparsers._group_actions[0].choices["train"]  # type: ignore[union-attr]
    rendered = sub.format_help()

    # rindex, not index: "--gpus N" appears first in the usage line, whose text is just the metavar.
    # collapse argparse's wrapping first, or a phrase split across two lines reads as absent.
    start = rendered.rindex("--gpus")
    window = " ".join(rendered[start : start + 400].split()).lower()
    assert "auto-size" in window, "help must say omission auto-sizes"
    assert "pins the ceiling" in window, "help must say an authored value pins the ceiling"
    assert "1, 2, 4, 8" in window, "help must name the public maximum as a rentable shape"
    assert not window.startswith("--gpus n cards to run the job on"), (
        "that phrasing promises exact-count semantics the allocator does not provide"
    )


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
    # under grpo or opd, which the next test pins.
    allocated = [
        allocate(
            "Qwen/Qwen3.5-9B", "sft", provider="runpod", gpu_type="RTX 4090", max_gpu_count=asked
        ).gpu_count
        for asked in (2, 3, 4)
    ]

    # asking for 2, 3 and 4 all land on the same smallest fitting count. if any of these ever equals
    # what was asked for, an exact-count path exists and TRAINING.md's "no exact-count mechanism"
    # paragraph has to be rewritten alongside it.
    assert allocated == [2, 2, 2], (
        f"expected the smallest fitting combination for every ask, got {allocated}"
    )


def test_a_pinned_class_can_need_a_wider_ceiling_for_grpo_and_opd():
    """The doc's 9B/RTX 4090 example is labelled SFT because fit is per algorithm.

    SFT fits on two cards. GRPO and OPD exceed a four-card ceiling but fit at the public maximum of
    eight, so copying the SFT example into a rollout config without widening the ceiling fails
    loudly instead of silently selecting an undersized shape.
    """
    import pytest

    from flash.providers.core.allocator import allocate
    from flash.providers.core.base import UnsupportedGpuError

    # same model and class; only the algorithm and resulting memory floor differ.
    assert (
        allocate(
            "Qwen/Qwen3.5-9B", "sft", provider="runpod", gpu_type="RTX 4090", max_gpu_count=4
        ).gpu_count
        == 2
    )
    for algorithm in ("grpo", "opd"):
        with pytest.raises(UnsupportedGpuError):
            allocate(
                "Qwen/Qwen3.5-9B",
                algorithm,
                provider="runpod",
                gpu_type="RTX 4090",
                max_gpu_count=4,
            )
        assert (
            allocate(
                "Qwen/Qwen3.5-9B",
                algorithm,
                provider="runpod",
                gpu_type="RTX 4090",
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
                provider="runpod",
                gpu="RTX 4090",
                hourly_usd=0.69,
                min_vram_gb=31,
                candidates=(),
                gpu_count=count,
            )
        )

    assert summarize(2).startswith("allocated 2x RTX 4090 on runpod at $1.38/hr"), summarize(2)
    # one card is spelled bare, and the hourly total is per-run rather than per-card.
    assert summarize(1).startswith("allocated RTX 4090 on runpod at $0.69/hr"), summarize(1)


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
        spec={"gpu": {"type": "RTX 4090", "provider": "runpod", "count": submitted_ceiling}},
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


def test_a_pinned_class_below_the_whole_run_floor_is_only_rejected_on_one_card(tmp_path):
    """The offline VRAM gate compares a WHOLE-RUN floor to ONE card, so it must respect gpu.count.

    required_vram_gb is the memory the run needs in total. Comparing it to a single card's VRAM is
    correct only when the run is confined to a single card; above that the allocator shards the run
    and applies its own multi-card fit test (allocator.py:181), which guards the identical
    comparison with `and max_gpu_count <= 1`.

    Without that guard the gate rejects a pin the allocator would have accepted, and it does so
    client-side before any allocation runs -- which makes --gpus useless for its main purpose,
    since the reason to ask for more cards is precisely that one card is too small. Found on a 35B
    GRPO run: a 141 GB class at count=2 holds 282 GB against a 180 GB floor and was still refused.
    """
    from flash.schema import ConfigError

    # 35B MoE GRPO needs more than one 141 GB card holds, so the floor is above the pinned class.
    big = (
        'model = "Qwen/Qwen3.6-35B-A3B"\n'
        'project = "11111111-1111-4111-8111-111111111111"\n'
        'algorithm = "grpo"\n'
        '[environment]\nid = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
        "[train]\nepochs = 1\nmax_examples = 100\n"
        '[gpu]\ntype = "H200"\nprovider = "runpod"\n'
    )

    # one card genuinely cannot hold it, so the gate must still fire. the 35B GRPO floor now exceeds
    # every validated class (the routed experts train), so the refusal names that rather than the
    # per-pin shortfall; either way --gpus 1 must not be admitted.
    with pytest.raises(
        ConfigError,
        match=r"(but this run requires at least|no validated GPU class has)",
    ):
        _spec(tmp_path, "--gpus", "1", config=big)

    # two cards clear the floor, so the gate must stand aside and let the allocator decide.
    assert _spec(tmp_path, "--gpus", "2", config=big).gpu.count == 2
