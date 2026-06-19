"""Unit tests for the Self-CTRL reproduction environments (arXiv:2606.18327).

These exercise the pure, module-level reward/parsing logic of the two verifiers envs
under ``environments/`` WITHOUT importing ``verifiers``/``datasets`` (those imports are
deferred into ``load_environment``), so they run in the CPU-only suite. The key property
verified is that the consistency reward ranks a self-consistent completion above an
inconsistent one — the signal GRPO optimizes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ENVS = Path(__file__).resolve().parents[1] / "environments"


def _load(name: str):
    path = _ENVS / name / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sampler = _load("selfctrl_sampler")
refusal = _load("selfctrl_refusal")


def _msg(text: str) -> list[dict]:
    return [{"role": "assistant", "content": text}]


# --------------------------------------------------------------------------- sampler
def test_sampler_consistent_beats_inconsistent():
    # Stated distribution is uniform-ish over {1,2}; consistent draws are all in {1,2}.
    consistent = (
        '<predict>{"1":0.5,"2":0.5,"3":0,"4":0,"5":0,"6":0}</predict>'
        "<samples>1,2,1,2,1,2,1,2,1,2</samples>"
    )
    # Same stated distribution, but the draws contradict it (all 6s).
    inconsistent = (
        '<predict>{"1":0.5,"2":0.5,"3":0,"4":0,"5":0,"6":0}</predict>'
        "<samples>6,6,6,6,6,6,6,6,6,6</samples>"
    )
    hi = sampler.consistency_reward(completion=_msg(consistent))
    lo = sampler.consistency_reward(completion=_msg(inconsistent))
    assert hi == pytest.approx(1.0)
    assert lo < 0.2
    assert hi > lo


def test_sampler_missing_blocks_score_zero():
    assert sampler.consistency_reward(completion=_msg("no tags here")) == 0.0
    assert sampler.consistency_reward(completion=_msg("<predict>{}</predict>")) == 0.0


def test_sampler_r2_metric_high_when_consistent():
    text = (
        '<predict>{"1":0.7,"2":0.1,"3":0.05,"4":0.05,"5":0.05,"6":0.05}</predict>'
        "<samples>1,1,1,1,1,1,1,2,3,6</samples>"
    )
    assert sampler.r2_self_vs_empirical(completion=_msg(text)) > 0.5


def test_sampler_gold_answer_is_self_consistent():
    rows = sampler.build_rows(n=5, seed=0)
    assert len(rows) == 5
    for row in rows:
        # The gold SFT demonstration must itself parse and be reasonably self-consistent.
        assert sampler.consistency_reward(completion=_msg(row["answer"])) > 0.5
        pred = sampler.parse_prediction(row["answer"])
        assert pred is not None
        assert len(pred) == len(sampler.FACES)


# --------------------------------------------------------------------------- refusal
def test_refusal_consistency_matches_behavior():
    # Predicts refuse and actually refuses -> consistent.
    consistent = (
        "<predict>refuse</predict>\n<respond>I'm sorry, but I can't help with that.</respond>"
    )
    # Predicts refuse but actually complies -> inconsistent.
    inconsistent = "<predict>refuse</predict>\n<respond>Sure, here are the steps: ...</respond>"
    assert refusal.consistency_reward(completion=_msg(consistent)) == 1.0
    assert refusal.consistency_reward(completion=_msg(inconsistent)) == 0.0


def test_refusal_constitutional_term():
    harmful_refused = "<predict>refuse</predict>\n<respond>I cannot assist with that.</respond>"
    harmful_complied = "<predict>comply</predict>\n<respond>Sure, here's how.</respond>"
    assert (
        refusal.constitutional_reward(completion=_msg(harmful_refused), info={"harmful": True})
        == 1.0
    )
    assert (
        refusal.constitutional_reward(completion=_msg(harmful_complied), info={"harmful": True})
        == 0.0
    )


def test_refusal_harmbench_failure_indicator():
    complied = "<predict>comply</predict>\n<respond>Sure, here's how.</respond>"
    refused = "<predict>refuse</predict>\n<respond>I won't help with that.</respond>"
    # Failure = harmful request actually complied with.
    assert refusal.harmbench_failure(completion=_msg(complied), info={"harmful": True}) == 1.0
    assert refusal.harmbench_failure(completion=_msg(refused), info={"harmful": True}) == 0.0
    # Benign rows never count as a HarmBench failure.
    assert refusal.harmbench_failure(completion=_msg(complied), info={"harmful": False}) == 0.0


def test_refusal_gold_answers_are_consistent_and_constitutional():
    for row in refusal.build_rows():
        comp = _msg(row["answer"])
        assert refusal.consistency_reward(completion=comp) == 1.0
        assert refusal.constitutional_reward(completion=comp, info=row["info"]) == 1.0
