"""CPU-only tests for the Flash-vs-Tinker benchmark harness (no GPU / network / tinker).

Covers the version-independent eval scorer, the comparison aggregation (smoothing,
active-compute parsing, the tinker record), and config validity. Run with:

    uv run python -m pytest benchmark/tests -q
"""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BENCH))

import assemble  # noqa: E402
import eval_runner  # noqa: E402
import eval_unified  # noqa: E402

CONFIGS = _BENCH / "configs"


# --------------------------------------------------------------------------- #
# eval scorer (version-independent answer extraction)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"...therefore \boxed{72}", "72"),
        (r"\boxed{1,234}", "1234"),          # thousands separators stripped
        (r"the answer is \boxed{$50}", "50"),  # currency stripped
        ("no box, the final answer is 42.", "42"),  # last-number fallback, trailing dot
        (r"\boxed{3.5}", "3.5"),
        ("", ""),                              # nothing extractable
        ("answer: 30%", "30"),                # percent stripped
        (r"first \boxed{1} then \boxed{9}", "9"),  # LAST box wins
    ],
)
def test_extract_answer(text, expected):
    assert eval_runner.extract_answer(text) == expected


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"answer": "72"}, "72"),
        ({"answer": "Janet ... #### 18"}, "18"),  # gsm8k gold format
        ({"answer": "1,000"}, "1000"),
        ({"answer": "$42"}, "42"),
    ],
)
def test_gold_answer(row, expected):
    assert eval_runner.gold_answer(row) == expected


def test_scorer_matches_gold_end_to_end():
    # A generation that ends in the right boxed answer scores as correct vs the gsm8k gold row.
    gen = r"Let me compute step by step ... so the total is \boxed{18}."
    row = {"answer": "Working ... #### 18"}
    assert eval_runner.extract_answer(gen) == eval_runner.gold_answer(row)


# --------------------------------------------------------------------------- #
# assemble: reward smoothing + best
# --------------------------------------------------------------------------- #

def test_smoothed_is_mean_of_last_k():
    rh = [0.0, 1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # last 5 -> 0.2,0.4,0.6,0.8,1.0 mean 0.6
    assert assemble._smoothed(rh, k=5) == pytest.approx(0.6)
    assert assemble._smoothed([], k=5) is None
    assert assemble._smoothed([0.5], k=5) == pytest.approx(0.5)  # fewer than k


def test_best_is_max():
    assert assemble._best([0.1, 0.9, 0.3]) == 0.9
    assert assemble._best([]) is None


# --------------------------------------------------------------------------- #
# assemble: tinker active-compute parsing (pause excluded) + collect_tinker
# --------------------------------------------------------------------------- #

def test_pause_detection_flags_only_extreme_steps(tmp_path):
    # per-step time/total is sequential ~= wall; a >300s step is a capacity pause, not compute.
    # Normal steps + one 1500s pause step (median of the set is 30) -> pause = 1500 - 30.
    rows = [{"time/total": 30.0}, {"time/total": 30.0}, {"time/total": 1500.0}]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert assemble._tinker_pause_s(str(tmp_path)) == pytest.approx(1470.0)
    # A checkpoint-ish step (217s, < 300s) must NOT be mistaken for a pause.
    (tmp_path / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in [{"time/total": 23.0}, {"time/total": 217.0}])
    )
    assert assemble._tinker_pause_s(str(tmp_path)) is None
    assert assemble._tinker_pause_s(None) is None
    assert assemble._tinker_pause_s(str(tmp_path / "missing")) is None


def test_collect_tinker_splits_pause_and_costs_active(tmp_path, monkeypatch):
    logp = tmp_path / "log"
    logp.mkdir()
    # 28 normal 60s steps + one 600s capacity-pause step. median=60 -> pause = 600-60 = 540.
    rows = [{"time/total": 60.0} for _ in range(28)] + [{"time/total": 600.0}]
    (logp / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    result = {"platform": "tinker", "status": "done", "steps": 30,
              "reward_history": [0.0, 0.2, 0.4], "wall_s": 2400.0, "log_path": str(logp)}
    (tmp_path / "tinker_gsm8k.json").write_text(json.dumps(result))
    monkeypatch.setattr(assemble, "_RESULTS", tmp_path)

    rec = assemble.collect_tinker("gsm8k", "tinker_gsm8k.json")
    assert rec["paused_s"] == pytest.approx(540.0)          # 600 - median(60)
    assert rec["train_s"] == pytest.approx(1860.0)          # active = wall(2400) - pause(540)
    assert rec["total_s"] == pytest.approx(2400.0)          # wall (incl pause)
    assert rec["final_smoothed"] == pytest.approx(0.2)      # mean of <=5 rewards
    assert rec["cost_usd"] == pytest.approx(round(1860.0 / 3600 * assemble._TINKER_PROXY_USD_PER_HR, 4))


def test_collect_tinker_no_pause_costs_full_wall(tmp_path, monkeypatch):
    logp = tmp_path / "log"
    logp.mkdir()
    rows = [{"time/total": 50.0} for _ in range(30)]  # no >300s step
    (logp / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    result = {"status": "done", "steps": 30, "reward_history": [0.3],
              "wall_s": 1500.0, "log_path": str(logp)}
    (tmp_path / "tinker_gsm8k.json").write_text(json.dumps(result))
    monkeypatch.setattr(assemble, "_RESULTS", tmp_path)
    rec = assemble.collect_tinker("gsm8k", "tinker_gsm8k.json")
    assert rec["paused_s"] is None
    assert rec["train_s"] == pytest.approx(1500.0)  # active == wall when no pause
    assert rec["cost_usd"] == pytest.approx(round(1500.0 / 3600 * assemble._TINKER_PROXY_USD_PER_HR, 4))


def test_collect_tinker_pending_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(assemble, "_RESULTS", tmp_path)
    rec = assemble.collect_tinker("gsm8k", "tinker_gsm8k.json")
    assert "pending" in rec["status"]


# --------------------------------------------------------------------------- #
# config validity
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["gsm8k_4b", "reverse_text_4b", "hendrycks_math_4b"])
def test_flash_config_has_required_fields(name):
    cfg = tomllib.loads((CONFIGS / f"{name}.toml").read_text())
    assert cfg["model"] == "Qwen/Qwen3.5-4B"
    assert cfg["algorithm"] == "grpo"
    assert cfg["environment"]["id"].startswith("primeintellect/")
    train = cfg["train"]
    assert train["steps"] == 30
    assert train["max_tokens"] == 1024  # truncation fix: matched on both stacks
    assert train["group_size"] == 4
    assert train["batch_size"] == 4
    assert "/" in train["hf_repo"]


def test_all_three_tasks_share_identical_grpo_knobs():
    """Apples-to-apples: every task uses the same model + GRPO hyper-parameters."""
    cfgs = [tomllib.loads((CONFIGS / f"{n}.toml").read_text())
            for n in ("gsm8k_4b", "reverse_text_4b", "hendrycks_math_4b")]
    knobs = [(c["model"], c["train"]["steps"], c["train"]["group_size"],
              c["train"]["batch_size"], c["train"]["max_tokens"]) for c in cfgs]
    assert len(set(knobs)) == 1, knobs


# --------------------------------------------------------------------------- #
# unified eval: Flash-trained generation via the serving (mocked HTTP)
# --------------------------------------------------------------------------- #

class _FakeResp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_eval_via_serving_scores_with_shared_scorer(monkeypatch):
    """The Flash-serving path posts chat completions and grades the SAME way as base/Tinker:
    extract the boxed/last-number answer and exact-match the gold. Two rows: one right, one wrong."""
    rows = [
        {"prompt": [{"role": "user", "content": "q1"}], "answer": "#### 18"},
        {"prompt": [{"role": "user", "content": "q2"}], "answer": "5"},
    ]
    replies = iter([
        {"choices": [{"message": {"content": r"reasoning ... \boxed{18}"}, "finish_reason": "stop"}]},
        {"choices": [{"message": {"content": r"...\boxed{99}"}, "finish_reason": "length"}]},  # wrong + truncated
    ])
    monkeypatch.setattr(eval_unified.urllib.request, "urlopen",
                        lambda *a, **k: _FakeResp(next(replies)))
    res = eval_unified.eval_via_serving(rows, "https://serve.example", "run-x", max_tokens=256)
    assert res["n"] == 2
    assert res["correct"] == 1            # only the first matches gold 18
    assert res["accuracy"] == pytest.approx(0.5)
    assert res["truncated_frac"] == pytest.approx(0.5)  # the second hit the length cap
    assert res["errors"] == 0
