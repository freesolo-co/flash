"""Offline unit tests for the benchmark/ harness.

Guards the review-thread fixes that are easy to silently regress:
  - Flash/Tinker step count comes from the worker's authoritative notes['steps']
    (resp. the runner's d['steps']), NOT len(reward_history), and per_step_s uses it.
  - The Flash API-key resolver is shared (assemble imports flash_runner's) and honors
    FLASH_PLANE_KEY -> FREESOLO_API_KEY -> ~/.flash.
  - The Tinker cost proxy + active-compute basis are a single shared source.
  - eval_runner normalizes gold and predicted answers identically.

These import only stdlib-backed benchmark modules (the verifiers/tinker imports in those
modules are function-local), so they run in CI without GPU/Tinker deps.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

# benchmark/ is a script dir, not a package — put it on the path so its modules import.
_BENCH = pathlib.Path(__file__).resolve().parents[1] / "benchmark"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import assemble
import eval_runner
import flash_runner
import tinker_runner

# ---------------------------------------------------------------------------
# Central steps-accounting fix
# ---------------------------------------------------------------------------

def _flash_metrics(*, steps, reward_history, wall_seconds=300.0):
    """A flash metrics.json-shaped dict (the worker records notes['steps'] separately)."""
    return {
        "wall_seconds": wall_seconds,
        "setup_seconds": 10.0,
        "allocated_gpu": "A100 PCIe",
        "cost_usd": 1.0,
        "notes": {"steps": steps, "reward_history": reward_history, "eval_history": []},
    }


def test_collect_flash_uses_notes_steps_not_reward_history_len(monkeypatch):
    # Worker ran 30 steps but only logged 15 reward points (logging cadence / resume).
    rh = [1.0] * 15
    metrics = _flash_metrics(steps=30, reward_history=rh, wall_seconds=3000.0)
    monkeypatch.setattr(assemble, "_get", lambda *a, **k: {"state": "done", "artifacts_dir": "/x"})
    monkeypatch.setattr(assemble, "_read_flash_metrics", lambda _d: metrics)

    rec = assemble.collect_flash("gsm8k", {"api_url": "http://x", "run_id": "flash-1-a"}, "k")

    assert rec["steps"] == 30, "steps must come from notes['steps'], not len(reward_history)"
    assert rec["steps"] != len(rh)
    # per_step_s must divide by the authoritative count (3000/30=100), not len(rh) (3000/15=200).
    assert rec["per_step_s"] == pytest.approx(100.0)


def test_collect_flash_falls_back_to_reward_history_when_steps_absent(monkeypatch):
    rh = [0.5, 0.6, 0.7]
    metrics = {"wall_seconds": 30.0, "notes": {"reward_history": rh}}  # no notes['steps']
    monkeypatch.setattr(assemble, "_get", lambda *a, **k: {"state": "done", "artifacts_dir": "/x"})
    monkeypatch.setattr(assemble, "_read_flash_metrics", lambda _d: metrics)

    rec = assemble.collect_flash("gsm8k", {"api_url": "http://x", "run_id": "flash-1-a"}, "k")
    assert rec["steps"] == 3  # falls back to len(reward_history)
    assert rec["per_step_s"] == pytest.approx(10.0)


def test_collect_tinker_returns_configured_steps_and_consistent_per_step(tmp_path, monkeypatch):
    # Result JSON says 30 steps but logs fewer reward points; steps must be 30 and
    # per_step_s must divide by the SAME 30 (previously the dict hard-coded len(rh)).
    result = {
        "status": "done",
        "steps": 30,
        "reward_history": [0.1] * 12,
        "wall_s": 600.0,
        "active_compute_s": 300.0,
        "log_path": "/nonexistent",
    }
    p = tmp_path / "tinker_gsm8k.json"
    p.write_text(json.dumps(result))
    monkeypatch.setattr(assemble, "_RESULTS", tmp_path)

    rec = assemble.collect_tinker("gsm8k", "results/tinker_gsm8k.json")
    assert rec["steps"] == 30
    assert rec["steps"] != len(result["reward_history"])
    # per_step_s = active(300)/steps(30) = 10, not active/len(rh)=25.
    assert rec["per_step_s"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Shared Flash key resolver (dedup + FLASH_PLANE_KEY)
# ---------------------------------------------------------------------------

def test_flash_key_helper_is_deduped():
    # assemble must use flash_runner's single canonical resolver, not its own copy.
    assert assemble._flash_key is flash_runner._flash_api_key


def test_flash_api_key_prefers_flash_plane_key(monkeypatch):
    monkeypatch.setenv("FLASH_PLANE_KEY", "plane-key")
    monkeypatch.setenv("FREESOLO_API_KEY", "freesolo-key")
    assert flash_runner._flash_api_key() == "plane-key"


def test_flash_api_key_falls_back_to_freesolo(monkeypatch):
    monkeypatch.delenv("FLASH_PLANE_KEY", raising=False)
    monkeypatch.setenv("FREESOLO_API_KEY", "freesolo-key")
    assert flash_runner._flash_api_key() == "freesolo-key"


# ---------------------------------------------------------------------------
# Shared Tinker cost proxy + active-compute basis
# ---------------------------------------------------------------------------

def test_tinker_proxy_rate_is_single_source():
    assert assemble._TINKER_PROXY_USD_PER_HR == tinker_runner.TINKER_PROXY_USD_PER_HR == 2.00


def test_active_compute_sums_rollout_and_train_step():
    records = [
        {"time/do_group_rollout_and_filter_constant_reward:total": 5.0, "time/train_step": 2.0},
        {"time/do_group_rollout_and_filter_constant_reward:total": 3.0, "time/train_step": 1.0},
    ]
    assert tinker_runner.active_compute_s(records) == pytest.approx(11.0)
    assert tinker_runner.active_compute_s([]) is None


def test_tinker_cost_uses_active_compute_times_shared_proxy(tmp_path, monkeypatch):
    # active=1800s -> 0.5h -> 0.5 * $2.00 = $1.00 (NOT wall, NOT $3.50).
    result = {
        "status": "done", "steps": 30, "reward_history": [0.1] * 30,
        "wall_s": 3600.0, "active_compute_s": 1800.0, "log_path": "/nonexistent",
    }
    p = tmp_path / "tinker_gsm8k.json"
    p.write_text(json.dumps(result))
    monkeypatch.setattr(assemble, "_RESULTS", tmp_path)

    rec = assemble.collect_tinker("gsm8k", "results/tinker_gsm8k.json")
    assert rec["cost_usd"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# eval_runner: identical normalization of gold vs prediction
# ---------------------------------------------------------------------------

def test_eval_gold_and_pred_normalize_identically():
    # Gold keeps a trailing period and a percent sign; prediction strips them. They must
    # canonicalize to the same string so exact-match doesn't spuriously fail.
    assert eval_runner.gold_answer({"answer": "#### 1,234.0"}) == eval_runner.extract_answer("1,234.0")
    assert eval_runner.gold_answer({"answer": "42%"}) == eval_runner.extract_answer("\\boxed{42%}")
    assert eval_runner.gold_answer({"answer": "7."}) == eval_runner.extract_answer("the answer is 7.")


# ---------------------------------------------------------------------------
# Committed results are internally consistent with the corrected code
# ---------------------------------------------------------------------------

def test_committed_results_steps_and_cost_consistent():
    results = pathlib.Path(__file__).resolve().parents[1] / "benchmark" / "results"
    comp = json.loads((results / "comparison.json").read_text())
    for task, fname in [
        ("gsm8k", "tinker_gsm8k.json"),
        ("reverse-text", "tinker_reverse-text.json"),
        ("hendrycks-math", "tinker_hendrycks-math.json"),
    ]:
        # Flash reports the configured 30 steps (not the 15 logged reward points).
        assert comp[task]["flash"]["steps"] == 30
        # Per-run Tinker cost matches the comparison table (same active-compute x $2/hr basis).
        run = json.loads((results / fname).read_text())
        assert run["cost_usd_estimated"] == pytest.approx(comp[task]["tinker"]["cost_usd"])
        assert run["active_compute_s"] == pytest.approx(comp[task]["tinker"]["train_s"])
