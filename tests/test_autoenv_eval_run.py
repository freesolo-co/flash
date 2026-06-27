"""Offline test of the eval orchestrator (evaluate_run) with a fake Flash client."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import autoenv
from autoenv.eval.score import EvalConfig, evaluate_run
from autoenv.manifest import PaperCase

SMOKE_CASE = Path(autoenv.__file__).parent / "cases" / "arithmetic_smoke_sft.toml"


class FakeClient:
    """Records the deploy/chat/undeploy lifecycle and returns canned responses by question."""

    def __init__(self, answer_by_input: dict[str, str]):
        self._answers = answer_by_input
        self.deployed = False
        self.undeployed = False
        self.chat_calls = 0

    def deploy(self, run_id, dry_run=False, step=None):
        self.deployed = True
        return {"state": "deployed"}

    def undeploy(self, run_id):
        self.undeployed = True
        return {}

    def chat(self, run_id, messages, temperature=0.0, max_tokens=512):
        self.chat_calls += 1
        question = messages[-1]["content"]
        content = self._answers.get(question, "I don't know")
        return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _eval_rows(case: PaperCase) -> list[dict]:
    from autoenv.ingest import fetch_rows

    return fetch_rows(case.resolved_eval(), input_field="input", output_field="output")


def test_evaluate_run_scores_and_manages_deploy_lifecycle():
    case = PaperCase.load(SMOKE_CASE)
    rows = _eval_rows(case)
    # Answer every eval row correctly (echo the gold final number).
    answers = {r["input"]: r["output"] for r in rows}
    client = FakeClient(answers)

    result = evaluate_run(
        case,
        "run-xyz",
        client=client,
        eval_rows=rows,
        train_rows=[],
        config=EvalConfig(max_eval=len(rows)),
    )
    assert client.deployed
    assert client.undeployed
    assert client.chat_calls == len(rows)
    assert result.state == "scored"
    assert result.agent_metric == 1.0  # all correct
    assert result.achievement == 1.0  # matched/exceeded the paper target
    assert result.run_id == "run-xyz"


def test_evaluate_run_partial_and_diagnostics():
    case = PaperCase.load(SMOKE_CASE)
    rows = _eval_rows(case)
    # Answer half correctly, half wrong.
    answers = {}
    for i, r in enumerate(rows):
        answers[r["input"]] = r["output"] if i % 2 == 0 else "the answer is 999999"
    client = FakeClient(answers)

    result = evaluate_run(
        case,
        "run-xyz",
        client=client,
        eval_rows=rows,
        train_rows=[],
        config=EvalConfig(max_eval=len(rows)),
    )
    assert 0.0 < result.agent_metric < 1.0
    assert result.diagnostics["empty_responses"] == 0
    assert "samples" in result.diagnostics


def test_evaluate_run_flags_leakage_and_skips_score():
    case = PaperCase.load(SMOKE_CASE)
    rows = _eval_rows(case)
    client = FakeClient({})
    # Train rows overlap the eval inputs -> leakage -> invalid, no deploy.
    result = evaluate_run(
        case,
        "run-xyz",
        client=client,
        eval_rows=rows,
        train_rows=rows,
        config=EvalConfig(max_eval=len(rows)),
    )
    assert result.state == "invalid"
    assert result.diagnostics["leakage"] is True
    assert not client.deployed  # never deployed a leaked eval


def test_evaluate_run_without_base_reported_skips_normalization():
    case = PaperCase.load(SMOKE_CASE)
    case = dataclasses.replace(case, metric=dataclasses.replace(case.metric, base_reported=None))
    rows = _eval_rows(case)
    client = FakeClient({r["input"]: r["output"] for r in rows})
    result = evaluate_run(
        case,
        "r",
        client=client,
        eval_rows=rows,
        train_rows=[],
        config=EvalConfig(max_eval=len(rows)),
    )
    assert result.state == "scored"
    assert result.achievement is None
    assert result.agent_metric == 1.0
