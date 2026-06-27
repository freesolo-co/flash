"""Pure eval (metrics, split, leakage) and scoring (normalization) — all offline."""

from __future__ import annotations

import pytest

from autoenv.eval.metrics import METRICS, aggregate, score_one
from autoenv.eval.split import leakage_check, split_rows
from autoenv.score.normalize import improvement_normalized, noise_band, ratio, score


def test_metric_registry_has_expected_names():
    for name in (
        "exact_match",
        "accuracy",
        "normalized_match",
        "contains",
        "numeric_match",
        "token_f1",
    ):
        assert name in METRICS


@pytest.mark.parametrize(
    ("gold", "resp", "expected"),
    [
        ("The answer is 7.", "so the answer is 7", 1.0),
        ("The answer is 7.", "the answer is 8", 0.0),
        ("42", "blah blah 42 blah 99... final 42", 1.0),
        ("no numbers", "still none", 0.0),
    ],
)
def test_numeric_match(gold, resp, expected):
    assert score_one("numeric_match", gold, resp) == expected


def test_token_f1_partial_credit():
    f1 = score_one("token_f1", "the quick brown fox", "the brown fox")
    assert 0.0 < f1 < 1.0


def test_aggregate_is_mean():
    pairs = [("5", "5"), ("6", "7"), ("8", "8")]
    assert aggregate("numeric_match", pairs) == pytest.approx(2 / 3)
    assert aggregate("numeric_match", []) == 0.0


def test_unknown_metric_raises():
    with pytest.raises(KeyError):
        score_one("rougeXYZ", "a", "a")


def test_split_is_deterministic_and_disjoint():
    rows = [{"input": f"q{i}", "output": str(i)} for i in range(20)]
    t1, e1 = split_rows(rows, eval_fraction=0.25, seed=0)
    t2, e2 = split_rows(list(reversed(rows)), eval_fraction=0.25, seed=0)
    # Same partition regardless of input order.
    assert {r["input"] for r in e1} == {r["input"] for r in e2}
    assert {r["input"] for r in t1} == {r["input"] for r in t2}
    # Disjoint and complete.
    assert not ({r["input"] for r in t1} & {r["input"] for r in e1})
    assert len(t1) + len(e1) == 20
    assert len(e1) == 5


def test_leakage_check_flags_overlap():
    train = [{"input": "shared", "output": "1"}, {"input": "t", "output": "2"}]
    clean_eval = [{"input": "fresh", "output": "3"}]
    dirty_eval = [{"input": "shared", "output": "1"}]
    assert leakage_check(train, clean_eval).clean
    report = leakage_check(train, dirty_eval)
    assert report.leaked
    assert report.overlap_count == 1


def test_improvement_normalized():
    # agent closes half the gap from base(0.2) to paper(0.6).
    assert improvement_normalized(0.4, 0.2, 0.6) == pytest.approx(0.5)
    # matched or exceeded the paper -> clamped to 1.0.
    assert improvement_normalized(0.7, 0.2, 0.6) == 1.0
    # below base -> clamped to 0.0.
    assert improvement_normalized(0.1, 0.2, 0.6) == 0.0
    # paper == base -> undefined.
    assert improvement_normalized(0.5, 0.3, 0.3) is None


def test_improvement_normalized_lower_is_better():
    # Lower metric is better (e.g. error rate): agent 0.3 between base 0.5 and paper 0.1.
    val = improvement_normalized(0.3, 0.5, 0.1, higher_is_better=False)
    assert val == pytest.approx(0.5)


def test_ratio_and_noise_band():
    assert ratio(0.3, 0.6) == pytest.approx(0.5)
    assert ratio(0.3, 0.0) is None
    # lower-is-better: a perfect agent (~0 error) matched/exceeded the paper -> 1.0, not None.
    assert ratio(0.0, 0.1, higher_is_better=False) == 1.0
    assert ratio(0.2, 0.1, higher_is_better=False) == pytest.approx(0.5)
    band = noise_band(0.5, 100)
    assert band == pytest.approx(1.96 * 0.05)
    assert noise_band(0.5, 0) is None


def test_score_bundles_components_and_notes_within_noise():
    s = score(agent_metric=0.21, base_metric=0.20, paper_metric=0.60, eval_n=50)
    assert s.achievement is not None
    assert "noise band" in s.note  # tiny agent-vs-base gap flagged as no signal
