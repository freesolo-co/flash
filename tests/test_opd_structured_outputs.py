"""CPU parity tests for the shared OPD structured-outputs primitives.

forced_from_logprobs and drop_fully_forced_groups were moved out of the TRL-only opd_vllm.py/opd.py
into flash.engine.structured_outputs so the OpenRLHF OPD rollout runs byte-identical guided-decoding
correctness logic (and so they survive the eventual removal of the TRL workers). resolve_opd_structured_plan
is the CPU-provable plan the OpenRLHF OPD path validates up front before the (GPU-gated) live rollout.

No torch/openrlhf/vllm imports here -- these helpers are pure and must stay CPU-importable.
"""

from __future__ import annotations

import pytest

from flash.engine.structured_outputs import (
    THINKING_REASONING_PARSER,
    OpdStructuredPlan,
    drop_fully_forced_groups,
    forced_from_logprobs,
    resolve_opd_structured_plan,
)


class _LP:
    """Minimal stand-in for a vLLM Logprob object (only .logprob is read)."""

    def __init__(self, logprob: float) -> None:
        self.logprob = logprob


_NEG_INF = float("-inf")


# --- forced_from_logprobs ---------------------------------------------------------------------


def test_forced_none_logprobs_returns_empty():
    # unconstrained rollouts request no logprobs -> () -> loss runs unmasked
    assert forced_from_logprobs(None, 5) == ()


def test_forced_exactly_one_finite_is_forced():
    # one finite entry (the single legal token) + one -inf pad == grammar-forced
    lps = [{1: _LP(0.0), 2: _LP(_NEG_INF)}]
    assert forced_from_logprobs(lps, 1) == (True,)


def test_forced_two_finite_is_free():
    lps = [{1: _LP(-0.1), 2: _LP(-2.3)}]
    assert forced_from_logprobs(lps, 1) == (False,)


def test_forced_zero_finite_row_is_free_not_forced():
    # an all -inf row is a wiring anomaly, never a real forced position -> treated as free
    lps = [{1: _LP(_NEG_INF), 2: _LP(_NEG_INF)}]
    assert forced_from_logprobs(lps, 1) == (False,)


def test_forced_missing_rows_tail_is_unmasked():
    # fewer rows than tokens -> mask the visible prefix, leave the tail False (unmasked)
    lps = [{1: _LP(0.0), 2: _LP(_NEG_INF)}]
    assert forced_from_logprobs(lps, 3) == (True, False, False)


def test_forced_accepts_plain_float_values():
    # getattr(lp, "logprob", lp) falls back to a bare float when the value is not an object
    lps = [{1: 0.0, 2: _NEG_INF}, {1: -0.5, 2: -1.5}]
    assert forced_from_logprobs(lps, 2) == (True, False)


def test_forced_mixed_sequence():
    lps = [
        {1: _LP(0.0), 2: _LP(_NEG_INF)},  # forced
        {1: _LP(-0.2), 2: _LP(-1.1)},  # free
        {1: _LP(0.0), 2: _LP(_NEG_INF)},  # forced
    ]
    assert forced_from_logprobs(lps, 3) == (True, False, True)


# --- drop_fully_forced_groups -----------------------------------------------------------------


def test_drop_empty_forced_is_noop_identity():
    groups = [((0, 1), 1.0), ((2,), 2.0)]
    # empty forced -> return the same object unchanged (unconstrained rollouts)
    assert drop_fully_forced_groups(groups, ()) is groups


def test_drop_fully_forced_group_removed():
    groups = [((0, 1), 1.0), ((2, 3), 2.0)]
    forced = (True, True, False, False)  # group 0 all forced, group 1 all free
    assert drop_fully_forced_groups(groups, forced) == [((2, 3), 2.0)]


def test_drop_partially_forced_group_kept():
    groups = [((0, 1), 1.0)]
    forced = (True, False)  # not ALL forced -> keep
    assert drop_fully_forced_groups(groups, forced) == [((0, 1), 1.0)]


def test_drop_index_beyond_forced_len_is_treated_free():
    # a student index outside forced's range counts as not-forced -> group kept
    groups = [((0, 5), 1.0)]
    forced = (True,)  # index 5 >= len(forced) -> not forced -> group survives
    assert drop_fully_forced_groups(groups, forced) == [((0, 5), 1.0)]


def test_drop_empty_index_group_kept():
    # a group with no student indices is never "fully forced" -> kept
    groups = [((), 1.0), ((0,), 2.0)]
    forced = (True,)
    assert drop_fully_forced_groups(groups, forced) == [((), 1.0)]


# --- resolve_opd_structured_plan --------------------------------------------------------------


@pytest.mark.parametrize("spec_json", ["", None])
def test_resolve_unconstrained_returns_none(spec_json):
    assert resolve_opd_structured_plan(spec_json, thinking=True) is None
    assert resolve_opd_structured_plan(spec_json, thinking=False) is None


def test_resolve_valid_json_with_thinking_defers_grammar():
    plan = resolve_opd_structured_plan('{"json": {"type": "object"}}', thinking=True)
    assert isinstance(plan, OpdStructuredPlan)
    assert plan.constraint == {"json": {"type": "object"}}
    assert plan.reasoning_parser == THINKING_REASONING_PARSER


def test_resolve_valid_choice_without_thinking_has_no_parser():
    plan = resolve_opd_structured_plan('{"choice": ["a", "b"]}', thinking=False)
    assert plan.constraint == {"choice": ["a", "b"]}
    assert plan.reasoning_parser is None


def test_resolve_corrupt_payload_raises():
    with pytest.raises(ValueError, match=r"corrupt train\.structured_outputs payload"):
        resolve_opd_structured_plan("not-json", thinking=False)


def test_resolve_no_constraint_key_raises():
    with pytest.raises(ValueError, match="no constraint"):
        resolve_opd_structured_plan('{"unrelated": 1}', thinking=True)
