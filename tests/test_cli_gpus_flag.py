"""Tests for `flash train --gpus N`, the first-class way to ask for a card count.

The flag is sugar for the `gpu.count` override and shares --set's dest, so these drive the REAL
parser and assert the wiring (does it reach the spec?), the precedence (does it beat the file, and
how does it resolve against a --set of the same key?) and that it inherits the SAME validation as
[gpu] count rather than getting a second, laxer one.
"""

from __future__ import annotations

import pytest

BASE = (
    'model = "Qwen/Qwen3.5-4B"\n'
    'project = "11111111-1111-4111-8111-111111111111"\n'
    'algorithm = "grpo"\n'
    '[environment]\nid = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
    "[train]\nepochs = 1\nmax_examples = 100\n"
    '[gpu]\ntype = "B200"\nprovider = "runpod"\n'
)


def _overrides(*argv: str) -> list[str]:
    """the override list the real parser produces for `flash train run.toml <argv>`."""
    from flash.cli import _build_parser

    return _build_parser().parse_args(["train", "run.toml", *argv]).overrides


def _spec(tmp_path, *argv: str, config: str = BASE):
    """parse argv through the real parser, then build the spec exactly as cmd_train does."""
    from flash.schema import spec_from_file

    cfg = tmp_path / "run.toml"
    cfg.write_text(config)
    return spec_from_file(str(cfg), run_id="x", overrides=_overrides(*argv), project_required=True)


def test_the_parser_turns_gpus_into_the_override_it_is_sugar_for():
    # the ask is a first-class CLI item, not something the user spells as --set. it parses as its
    # own flag, and the conversion to gpu.count happens at parse time so nothing downstream needs
    # to know the flag exists.
    assert _overrides("--gpus", "4") == ["gpu.count=4"]


def test_no_gpus_flag_contributes_no_override():
    # an absent flag must be indistinguishable from no flag. emitting `gpu.count=1` here would
    # silently DOWNGRADE every multi-gpu config whose count lives in the file.
    assert _overrides() == []


def test_gpus_reaches_the_spec(tmp_path):
    assert _spec(tmp_path, "--gpus", "4").gpu.count == 4


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
    from flash.cli import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args(["train", "run.toml", "--gpus", bad])
    assert "--gpus" in capsys.readouterr().err


def test_gpus_still_hits_the_multi_gpu_provider_gate(tmp_path):
    from flash import runner

    # a convenient flag must not become a way AROUND the submit gate: vast/lambda ignore gpu_count,
    # so 4 ranks would land on one rented card and be billed for four.
    # H100 rather than the shared BASE's B200: vast does not list B200 at all, so that pairing
    # would be rejected by the catalog before the multi-gpu gate this test is about.
    vast = BASE.replace('type = "B200"', 'type = "H100"').replace("runpod", "vast")
    spec = _spec(tmp_path, "--gpus", "4", config=vast)
    with pytest.raises(ValueError, match=r"gpu\.provider"):
        runner._require_supported_gpu_count(spec)


def test_the_help_does_not_promise_an_exact_card_count():
    """The flag must describe gpu.count as the CEILING the allocator treats it as.

    allocate() takes this field as max_gpu_count and keeps fit-alone candidates, so a class that
    fits the run on one card is allocated as one card and _spec_with_gpu rewrites the effective
    count to 1 -- the worker then launches one rank. Help that reads "cards to run the job on"
    states something the allocator does not honor, and a user sizing an experiment off it would be
    wrong with nothing in the output saying so. Asserted on the rendered help (not the source
    string) because that is what the user actually reads.
    """
    from flash.cli import _build_parser

    # render the train subparser's help, which is the text `flash train --help` prints.
    sub = _build_parser()._subparsers._group_actions[0].choices["train"]  # type: ignore[union-attr]
    rendered = sub.format_help()

    # rindex, not index: "--gpus N" appears first in the usage line, whose text is just the metavar.
    # collapse argparse's wrapping first, or a phrase split across two lines reads as absent.
    start = rendered.rindex("--gpus")
    window = " ".join(rendered[start : start + 400].split()).lower()
    assert "ceiling" in window, "help must say the count is a ceiling, not an exact card count"
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
    from flash.providers.allocator import allocate

    # 9B on a 24 GB class does not fit alone, so this is the combination path, not the fit-alone one.
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


def test_the_public_record_reports_the_ceiling_not_the_allocated_count():
    """TRAINING.md tells users to read the allocated count off gpu_status, not spec.gpu.count.

    Both halves of that sentence are load-bearing and both are asserted here, because the doc
    previously said to read `gpu.count` back off the run record -- which silently reports the
    ceiling. A user who submits --gpus 4, reads 4 back and concludes they got 4 cards would size
    throughput and cost off a number the allocator never honoured.

    The allocated count IS recorded, but only in effective_preparation, which to_dict() pops as
    server-internal. So the assertion is that the public record does NOT carry it -- if a future
    change starts publishing it there, this fails and the doc gets rewritten with it.
    """
    from flash.runner import RunStatus

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
