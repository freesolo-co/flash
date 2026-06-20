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

def test_active_compute_sums_rollout_and_train_only(tmp_path):
    # One paused step (huge save_checkpoint) must NOT inflate active compute: only rollout+train.
    rows = [
        {"time/do_group_rollout_and_filter_constant_reward:total": 40.0, "time/train_step": 10.0},
        {"time/do_group_rollout_and_filter_constant_reward:total": 40.0, "time/train_step": 10.0,
         "time/save_checkpoint": 1500.0},  # the capacity-pause step — ignored by active compute
    ]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert assemble._tinker_active_compute_s(str(tmp_path)) == pytest.approx(100.0)
    assert assemble._tinker_active_compute_s(None) is None
    assert assemble._tinker_active_compute_s(str(tmp_path / "missing")) is None


def test_collect_tinker_uses_active_compute_for_cost_and_splits_pause(tmp_path, monkeypatch):
    logp = tmp_path / "log"
    logp.mkdir()
    (logp / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"time/do_group_rollout_and_filter_constant_reward:total": 1700.0, "time/train_step": 100.0},
    ]))
    result = {"platform": "tinker", "status": "done", "steps": 30,
              "reward_history": [0.0, 0.2, 0.4], "wall_s": 2400.0, "log_path": str(logp)}
    (tmp_path / "tinker_gsm8k.json").write_text(json.dumps(result))
    monkeypatch.setattr(assemble, "_RESULTS", tmp_path)

    rec = assemble.collect_tinker("gsm8k", "tinker_gsm8k.json")
    assert rec["train_s"] == pytest.approx(1800.0)          # active compute, not wall
    assert rec["total_s"] == pytest.approx(2400.0)          # wall (incl pause)
    assert rec["paused_s"] == pytest.approx(600.0)          # 2400 - 1800
    assert rec["final_smoothed"] == pytest.approx(0.2)      # mean of <=5 rewards
    # cost proxy is on ACTIVE compute, not wall
    assert rec["cost_usd"] == pytest.approx(1800.0 / 3600 * assemble._TINKER_PROXY_USD_PER_HR)


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
