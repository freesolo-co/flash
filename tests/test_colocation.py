"""Co-location routing: does the idle a slow grader leaves get filled, and only when it pays."""

from __future__ import annotations

import pytest

from flash.cost.analytical import step_seconds_split
from flash.cost.colocation import (
    RunShape,
    evaluate_placement,
    plan_colocation,
    sharing_efficiency,
    wall_stretch,
)
from flash.cost.types import RunConfig


def _shape(label: str, gpu_s: float, reward_s: float, vram_gb: float = 0.0) -> RunShape:
    return RunShape(label=label, gpu_seconds=gpu_s, reward_seconds=reward_s, vram_gb=vram_gb)


def test_duty_cycle_is_the_gpu_share_of_a_step():
    assert _shape("r", 2.0, 8.0).duty_cycle == pytest.approx(0.2)
    assert _shape("r", 2.0, 8.0).idle_fraction == pytest.approx(0.8)


def test_a_run_with_no_step_time_claims_the_whole_card():
    # a degenerate shape must not read as "100% idle, pack it anywhere" -- that would let a run of
    # unknown cost be co-located for free. absent information, assume it wants the card.
    assert _shape("empty", 0.0, 0.0).duty_cycle == 1.0


def test_negative_inputs_are_rejected():
    with pytest.raises(ValueError, match="step seconds cannot be negative"):
        _shape("bad", -1.0, 1.0)
    with pytest.raises(ValueError, match="vram cannot be negative"):
        _shape("bad", 1.0, 1.0, vram_gb=-1.0)


def test_efficiency_decays_with_each_added_tenant():
    # a solo run pays nothing; every additional tenant multiplies the penalty again.
    assert sharing_efficiency(1) == pytest.approx(1.0)
    assert sharing_efficiency(2) == pytest.approx(0.9)
    assert sharing_efficiency(3) == pytest.approx(0.81)
    assert sharing_efficiency(4) == pytest.approx(0.729)


def test_two_compute_bound_runs_are_not_worth_pairing():
    # both need the card ~all the time: sharing halves each one's speed and gains nothing.
    a = _shape("a", 10.0, 0.0)
    b = _shape("b", 10.0, 0.0)
    assert not evaluate_placement((a, b)).worth_sharing


def test_a_latency_bound_run_pairs_with_a_compute_bound_one():
    judge = _shape("judge", 1.0, 9.0)  # 10% duty
    compute = _shape("compute", 9.0, 1.0)  # 90% duty
    placement = evaluate_placement((judge, compute))
    assert placement.worth_sharing
    # combined duty is exactly 1.0, so the only cost is the contention factor.
    assert placement.wall_stretch == pytest.approx(1 / 0.9)
    assert placement.throughput_gain == pytest.approx(1.8)


def test_two_latency_bound_runs_fit_inside_each_others_idle():
    a = _shape("a", 1.0, 9.0)
    b = _shape("b", 1.0, 9.0)
    placement = evaluate_placement((a, b))
    # combined duty 0.2 < 1.0: neither run waits on the other, so nothing is stretched beyond the
    # contention factor, and the pair does 2x the work in the same wall time.
    assert placement.wall_stretch == pytest.approx(1 / 0.9)
    assert placement.throughput_gain == pytest.approx(1.8)


def test_pairing_is_scale_free_across_wildly_different_step_lengths():
    # a 1s step and a 100s step share a card exactly as well as two equal steps, provided each
    # leaves the same proportion idle. keying on durations instead of ratios would invent a
    # conflict here. this is the assertion that fails if duty_cycle ever becomes a duration.
    fast = _shape("fast", 0.1, 0.9)
    slow = _shape("slow", 10.0, 90.0)
    matched = _shape("matched", 0.1, 0.9)
    assert evaluate_placement((fast, slow)).throughput_gain == pytest.approx(
        evaluate_placement((fast, matched)).throughput_gain
    )


def test_break_even_is_a_combined_duty_equal_to_the_tenant_count():
    # runs that each want the whole card are the definition of break-even before contention:
    # N runs / N stretch == 1.0x. the gate must reject it, since contention makes it a real loss.
    full = tuple(_shape(f"r{i}", 1.0, 0.0) for i in range(3))
    assert wall_stretch(full) * sharing_efficiency(3) == pytest.approx(3.0)
    assert evaluate_placement(full).throughput_gain < 1.0


def test_the_gate_admits_a_placement_that_clears_contention_and_rejects_one_that_does_not():
    # derived from the model's economics rather than read off its output: 2/(d/0.9) >= 1.01 is
    # d <= 1.782, so a combined duty either side of that must land either side of the gate.
    clears = evaluate_placement((_shape("a", 0.88, 0.12), _shape("b", 0.88, 0.12)))  # 1.76
    fails = evaluate_placement((_shape("c", 0.90, 0.10), _shape("d", 0.90, 0.10)))  # 1.80
    assert clears.worth_sharing
    assert not fails.worth_sharing


def test_a_single_run_is_never_a_sharing_win():
    # a group of one has nothing to interleave with, so it must never be reported as a placement.
    assert not evaluate_placement((_shape("alone", 1.0, 99.0),)).worth_sharing


def test_three_latency_bound_runs_share_one_card():
    # the ask is 100% utilization, not merely pairs: when three graders each leave ~90% idle, all
    # three belong on one card. a pair-only planner would strand the third on its own gpu.
    runs = [_shape(f"judge{i}", 1.0, 9.0) for i in range(3)]
    placements, solo = plan_colocation(runs)
    assert len(placements) == 1
    assert len(placements[0].runs) == 3
    assert solo == []


def test_a_group_never_exceeds_the_tenant_cap():
    runs = [_shape(f"judge{i}", 0.1, 99.9) for i in range(9)]
    placements, _ = plan_colocation(runs, max_tenants=4)
    assert placements, "near-idle runs must be packed"
    assert all(len(p.runs) <= 4 for p in placements)


def test_vram_capacity_blocks_a_group_that_card_time_would_allow():
    # the two are complementary in time and would pair happily, but their weights do not both fit.
    # memory does not timeshare, so this must be refused rather than stretched.
    a = _shape("a", 1.0, 9.0, vram_gb=60.0)
    b = _shape("b", 9.0, 1.0, vram_gb=60.0)
    assert evaluate_placement((a, b)).worth_sharing  # time says yes
    placements, solo = plan_colocation([a, b], vram_capacity_gb=80.0)
    assert placements == []  # memory says no
    assert {r.label for r in solo} == {"a", "b"}


def test_vram_capacity_admits_the_group_that_fits():
    a = _shape("a", 1.0, 9.0, vram_gb=30.0)
    b = _shape("b", 9.0, 1.0, vram_gb=30.0)
    placements, solo = plan_colocation([a, b], vram_capacity_gb=80.0)
    assert len(placements) == 1
    assert set(placements[0].labels) == {"a", "b"}
    assert solo == []


def test_a_run_too_large_for_the_card_is_reported_not_dropped():
    # it cannot be placed here, but silently losing it would be worse than saying so.
    big = _shape("big", 1.0, 9.0, vram_gb=200.0)
    small = _shape("small", 1.0, 9.0, vram_gb=10.0)
    placements, solo = plan_colocation([big, small], vram_capacity_gb=80.0)
    assert "big" in {r.label for r in solo}
    placed = {lbl for p in placements for lbl in p.labels} | {r.label for r in solo}
    assert placed == {"big", "small"}


def test_a_run_is_refused_when_admitting_it_would_lower_the_groups_throughput():
    # a group is not packed by tenant count. two idle runs pair happily, but admitting a third that
    # wants most of the card pushes the group past what their idle can absorb, and the trio is worth
    # LESS than the pair. the planner must decline that admission and leave the third on its own.
    hog = _shape("hog", 0.95, 0.05)  # 95% duty, seeds the group
    idle = _shape("idle", 0.30, 0.70)  # 30% duty, a profitable partner
    middling = _shape("middling", 0.60, 0.40)  # 60% duty, one tenant too many

    # the fixture has to sit on the path the planner actually walks: it seeds with the busiest run,
    # so the group it builds is (hog, idle), and `middling` is the admission under test.
    assert (
        evaluate_placement((hog, idle, middling)).throughput_gain
        < evaluate_placement((hog, idle)).throughput_gain
    ), "fixture must actually represent a losing admission"

    placements, solo = plan_colocation([hog, idle, middling], max_tenants=3)
    assert [p.labels for p in placements] == [("hog", "idle")]
    assert [r.label for r in solo] == ["middling"]


def test_an_unprofitable_seed_releases_its_partners_instead_of_stranding_them():
    # a marginal admission must not take its partners down with it. the seed draws in a run because
    # that admission improves THIS group, the group still misses the threshold, and the whole thing
    # is discarded -- so a pair that would have shared profitably with each other never gets built.
    # the runs the seed collected have only been measured against the seed's group, never against
    # every group they could have joined, so they go back in the pool rather than to solo.
    seed = _shape("s_busiest", 0.95, 0.05)
    mid = _shape("c_mid", 0.86, 0.14)
    least = _shape("a_least", 0.84, 0.16)

    # the fixture only bites if the greedy path really walks into the trap: the seed's best
    # admission has to be an improvement that still lands under the threshold.
    seeded_pair = evaluate_placement((seed, least))
    assert seeded_pair.throughput_gain > evaluate_placement((seed,)).throughput_gain
    assert not seeded_pair.worth_sharing, "fixture must represent a marginal, discarded group"
    assert evaluate_placement((mid, least)).worth_sharing, "the released pair must be profitable"

    placements, solo = plan_colocation([seed, mid, least])
    assert [p.labels for p in placements] == [("c_mid", "a_least")]
    assert [r.label for r in solo] == ["s_busiest"]
    assert len(placements) + len(solo) == 2, "three runs must fit on two cards, not three"


def test_seeding_with_the_busiest_run_pairs_it_off_instead_of_stranding_it():
    # three idle judges and one compute-bound run. seeding from the busiest run gives the compute
    # run a partner; seeding from the most idle would pair the judges together first and strand the
    # compute run on its own card. same runs, one fewer gpu.
    runs = [
        _shape("compute", 9.5, 0.5),
        _shape("j1", 0.5, 9.5),
        _shape("j2", 0.5, 9.5),
        _shape("j3", 0.5, 9.5),
    ]
    placements, solo = plan_colocation(runs, max_tenants=2)
    assert solo == [], "every run should find a partner here"
    assert len(placements) == 2
    compute_group = next(p for p in placements if "compute" in p.labels)
    assert len(compute_group.runs) == 2


def test_planning_never_places_a_run_twice():
    runs = [
        _shape("judge_a", 1.0, 9.0),
        _shape("judge_b", 1.0, 9.0),
        _shape("compute", 9.0, 1.0),
    ]
    placements, solo = plan_colocation(runs)
    placed = [lbl for p in placements for lbl in p.labels]
    assert len(placed) == len(set(placed))
    assert set(placed) | {r.label for r in solo} == {r.label for r in runs}


def test_planning_leaves_compute_bound_runs_alone():
    runs = [_shape("a", 10.0, 0.0), _shape("b", 10.0, 0.0)]
    placements, solo = plan_colocation(runs)
    assert placements == []
    assert {r.label for r in solo} == {"a", "b"}


def test_a_nonsensical_tenant_cap_is_rejected():
    # max_tenants=0 would otherwise return an empty plan that silently loses every run.
    with pytest.raises(ValueError, match="max_tenants must be at least 1"):
        plan_colocation([_shape("a", 1.0, 9.0)], max_tenants=0)


def test_duplicate_labels_are_rejected():
    # labels are what identifies a placement; two runs sharing one would silently drop a placement.
    with pytest.raises(ValueError, match="labels must be unique"):
        plan_colocation([_shape("same", 1.0, 9.0), _shape("same", 9.0, 1.0)])


def test_planning_is_deterministic_regardless_of_input_order():
    runs = [
        _shape("regex", 9.5, 0.5),
        _shape("judge", 0.5, 9.5),
        _shape("sandbox", 2.0, 8.0),
        _shape("verify", 6.0, 4.0),
    ]
    forward, forward_solo = plan_colocation(runs)
    reverse, reverse_solo = plan_colocation(list(reversed(runs)))
    assert [p.labels for p in forward] == [p.labels for p in reverse]
    assert [r.label for r in forward_solo] == [r.label for r in reverse_solo]


def test_a_measured_grader_latency_changes_the_placement_decision():
    """The end-to-end point: the same run pairs or does not depending on its MEASURED grader.

    Both configs are identical apart from the profiled per-completion latency, so any difference in
    the verdict comes from the measurement -- which is exactly the routing behavior being claimed.
    """
    base = {
        "model_id": "Qwen/Qwen3.5-4B",
        "method": "grpo",
        "steps": 100,
        "batch_size": 8,
        "group_size": 4,
        "completion_len": 512,
        "seq_len": 1024,
    }
    regex_env = RunConfig(**base, reward_seconds_per_completion=0.01)
    llm_judge = RunConfig(**base, reward_seconds_per_completion=3.0)

    regex_gpu, regex_reward = step_seconds_split(regex_env, "H100")
    judge_gpu, judge_reward = step_seconds_split(llm_judge, "H100")

    fast = RunShape("regex", regex_gpu, regex_reward)
    slow = RunShape("judge", judge_gpu, judge_reward)

    # the measurement must actually move the shape, or the rest of this proves nothing.
    assert slow.idle_fraction > fast.idle_fraction
    # a slow-grader run has idle worth selling; two fast-grader runs just contend.
    assert evaluate_placement((fast, slow)).worth_sharing
    assert not evaluate_placement((fast, RunShape("regex2", regex_gpu, regex_reward))).worth_sharing
