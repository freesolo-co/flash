"""CPU parity for backend-neutral per-turn credit and the OpenRLHF coordinate bridge (plan #11).

These prove the OpenRLHF multi-turn GRPO advantage construction is byte-identical to the TRL per-turn
trainer given the same per-turn rewards, without any GPU. The live wiring that populates per-turn
rewards from real experiences rides the multi-turn executor and stays fail-closed behind the
multi-turn GPU gate; only the pure advantage math is exercised here.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from flash.engine.worker import grpo_perturn_trainer, perturn_credit
from flash.engine.worker.openrlhf_multiturn import build_openrlhf_perturn_advantages
from flash.engine.worker.perturn_credit import build_per_turn_advantages


def _shift(spans: list[tuple[int, int]], prompt_len: int) -> list[tuple[int, int]]:
    """Push completion-local spans into the full-sequence coordinates the executor records."""
    return [(start + prompt_len, end + prompt_len) for start, end in spans]


def test_trl_trainer_reexports_the_shared_math():
    # single source of truth: the TRL trainer imports the neutral module's symbols, so per-turn
    # credit math cannot drift between the TRL and OpenRLHF backends.
    assert (
        grpo_perturn_trainer.build_per_turn_advantages is perturn_credit.build_per_turn_advantages
    )
    assert grpo_perturn_trainer.TurnCreditRow is perturn_credit.TurnCreditRow


def test_openrlhf_bridge_matches_trl_expected_tensor():
    # exact spans/rewards/expected from the published TRL per-turn test, but with the action ranges
    # pushed into full-sequence coordinates by a per-row prompt length; the bridge must recover TRL's
    # advantages byte-for-byte.
    local_spans = [
        [(0, 2), (4, 5)],
        [(0, 1)],
        [(0, 1), (3, 5)],
        [(0, 2), (4, 6)],
    ]
    rewards = [[1.0, 10.0], [5.0], [2.0, 7.0], [6.0, 11.0]]
    prompt_lens = [3, 5, 0, 7]
    action_ranges = [_shift(s, p) for s, p in zip(local_spans, prompt_lens, strict=True)]

    actual = build_openrlhf_perturn_advantages(
        action_ranges,
        rewards,
        prompt_lens,
        num_generations=2,
        completion_len=8,
        episode_advantages=torch.zeros(4),
    )

    expected = torch.tensor(
        [
            [-2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0, -2.0, -2.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(actual, expected)


# (label, local_spans, turn_rewards, num_generations, completion_len, episode_advantages, prompt_lens)
_EQUIVALENCE_CASES = [
    (
        "multi_turn_centering",
        [[(0, 2), (4, 5)], [(0, 1)], [(0, 1), (3, 5)], [(0, 2), (4, 6)]],
        [[1.0, 10.0], [5.0], [2.0, 7.0], [6.0, 11.0]],
        2,
        8,
        torch.zeros(4),
        [3, 5, 0, 7],
    ),
    (
        "variable_turns_and_none_fallback",
        [
            [(0, 1), (3, 4)],
            [(0, 2)],
            [(0, 1), (2, 3)],
            [(0, 1), (3, 5)],
            [(0, 2)],
            [(0, 1), (2, 4)],
        ],
        [[1.0, 10.0], [3.0], [5.0, 14.0], None, [9.0], [5.0, 1.0]],
        3,
        7,
        torch.tensor([0.0, 0.0, 0.0, 2.5, -1.0, 1.0], dtype=torch.float64),
        [4, 4, 4, 2, 2, 2],
    ),
    (
        "zero_variance_group_is_zero",
        [[(0, 2)], [(0, 2)]],
        [[4.0], [4.0]],
        2,
        3,
        torch.zeros(2),
        [6, 6],
    ),
    (
        "zero_width_turn_is_skipped",
        [[(0, 1), (1, 3)], [(0, 1), (1, 1)]],
        [[2.0, 8.0], [4.0, 100.0]],
        2,
        3,
        torch.zeros(2),
        [5, 9],
    ),
    (
        "single_turn_degenerate",
        [[(0, 3)], [(0, 2)]],
        [[7.0], [1.0]],
        2,
        4,
        torch.zeros(2),
        [0, 11],
    ),
    (
        "uneven_turn_counts",
        [[(0, 1), (2, 4)], [(0, 2)]],
        [[3.0, 9.0], [7.0]],
        2,
        5,
        torch.tensor([1.0, -1.0]),
        [8, 3],
    ),
]


@pytest.mark.parametrize(
    (
        "label",
        "local_spans",
        "turn_rewards",
        "num_generations",
        "completion_len",
        "episode",
        "prompt_lens",
    ),
    _EQUIVALENCE_CASES,
    ids=[case[0] for case in _EQUIVALENCE_CASES],
)
def test_openrlhf_bridge_equals_direct_perturn_math(
    label, local_spans, turn_rewards, num_generations, completion_len, episode, prompt_lens
):
    # shifting into full-sequence coordinates then delegating must be byte-identical to calling the
    # shared (TRL-shared) math on the completion-local spans directly, for every credit shape.
    action_ranges = [_shift(s, p) for s, p in zip(local_spans, prompt_lens, strict=True)]

    bridged = build_openrlhf_perturn_advantages(
        action_ranges,
        turn_rewards,
        prompt_lens,
        num_generations=num_generations,
        completion_len=completion_len,
        episode_advantages=episode,
    )
    direct = build_per_turn_advantages(
        local_spans,
        turn_rewards,
        num_generations=num_generations,
        completion_len=completion_len,
        episode_advantages=episode,
    )

    torch.testing.assert_close(bridged, direct)
    assert bridged.dtype == episode.dtype


def test_none_turn_rewards_fall_back_to_episode_advantage_under_offset():
    # a completion with no per-turn decomposition falls back to its episode advantage over its own
    # emitted span, even when its action ranges carry a nonzero prompt offset.
    action_ranges = [[(4, 6), (7, 9)], [(4, 5)]]
    episode = torch.tensor([2.0, -3.0])

    actual = build_openrlhf_perturn_advantages(
        action_ranges,
        [None, None],
        [4, 4],
        num_generations=2,
        completion_len=6,
        episode_advantages=episode,
    )

    expected = torch.tensor(
        [
            [2.0, 2.0, 2.0, 2.0, 2.0, 0.0],
            [-3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_bridge_recovers_completion_local_span_from_offset():
    # a full-sequence span [p, p+k] must map to completion-local [0, k].
    prompt_len = 12
    actual = build_openrlhf_perturn_advantages(
        [[(prompt_len, prompt_len + 2)], [(prompt_len, prompt_len + 2)]],
        [[5.0], [1.0]],
        [prompt_len, prompt_len],
        num_generations=2,
        completion_len=2,
        episode_advantages=torch.zeros(2),
    )
    # group mean 3.0 -> +2 / -2 on the first two completion tokens only.
    torch.testing.assert_close(actual, torch.tensor([[2.0, 2.0], [-2.0, -2.0]]))


def test_row_count_mismatch_raises():
    with pytest.raises(ValueError, match="row counts must match"):
        build_openrlhf_perturn_advantages(
            [[(0, 1)], [(0, 1)]],
            [[1.0]],
            [0, 0],
            num_generations=2,
            completion_len=1,
            episode_advantages=torch.zeros(2),
        )


def test_action_range_before_prompt_boundary_raises():
    with pytest.raises(ValueError, match="precedes its prompt boundary"):
        build_openrlhf_perturn_advantages(
            [[(2, 4)], [(3, 4)]],
            [[1.0], [2.0]],
            [3, 3],
            num_generations=2,
            completion_len=2,
            episode_advantages=torch.zeros(2),
        )


def test_negative_prompt_length_raises():
    with pytest.raises(ValueError, match="prompt length for row 0 must be non-negative"):
        build_openrlhf_perturn_advantages(
            [[(0, 1)], [(0, 1)]],
            [[1.0], [2.0]],
            [-1, 0],
            num_generations=2,
            completion_len=1,
            episode_advantages=torch.zeros(2),
        )


def test_shifted_span_exceeding_completion_width_raises():
    # after shifting, a span wider than the completion is rejected by the shared width guard.
    with pytest.raises(ValueError, match="exceeds completion width"):
        build_openrlhf_perturn_advantages(
            [[(3, 9)], [(3, 5)]],
            [[1.0], [2.0]],
            [3, 3],
            num_generations=2,
            completion_len=2,
            episode_advantages=torch.zeros(2),
        )
