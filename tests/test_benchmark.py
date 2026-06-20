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
import bench
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


# ---------------------------------------------------------------------------
# (1) Reported active_compute_s and cost_usd come from the SAME basis
# ---------------------------------------------------------------------------

def _write_tinker_result(tmp_path, **fields):
    base = {"status": "done", "steps": 30, "reward_history": [0.1] * 30}
    base.update(fields)
    (tmp_path / "tinker_gsm8k.json").write_text(json.dumps(base))


def test_reported_active_and_cost_share_one_basis_when_stored(tmp_path, monkeypatch):
    # Runner stored a MATCHED (active, cost) pair from active=1800s. assemble must report
    # that same active AND that same cost — not a wall-derived cost next to active.
    _write_tinker_result(
        tmp_path, wall_s=3600.0, active_compute_s=1800.0,
        cost_usd_estimated=1.0, log_path="/nonexistent",
    )
    monkeypatch.setattr(assemble, "_RESULTS", tmp_path)
    rec = assemble.collect_tinker("gsm8k", "results/tinker_gsm8k.json")
    assert rec["train_s"] == pytest.approx(1800.0)          # reported active = stored active
    assert rec["cost_usd"] == pytest.approx(1.0)            # reported cost = stored matched cost
    # The reported cost is the proxy of the reported active basis (no wall leak):
    assert rec["cost_usd"] == pytest.approx(assemble._tinker_cost_from_basis(rec["train_s"]))
    # And NOT the wall-based proxy (3600s -> $2.00), which is what a mismatch would give.
    assert rec["cost_usd"] != pytest.approx(assemble._tinker_cost_from_basis(3600.0))


def test_recomputed_active_drives_recomputed_cost(tmp_path, monkeypatch):
    # active_compute_s is NULL in the JSON but a metrics.jsonl exists -> assemble recomputes
    # active from it AND derives the cost from THAT recomputed active (consistent pair),
    # NOT the stale wall-based cost_usd_estimated the runner wrote when active was missing.
    log_dir = tmp_path / "logdir"
    log_dir.mkdir()
    # rollout 1000 + train 800 = 1800s active -> $1.00 at $2/hr.
    (log_dir / "metrics.jsonl").write_text(
        json.dumps({"time/do_group_rollout_and_filter_constant_reward:total": 1000.0,
                    "time/train_step": 800.0}) + "\n"
    )
    _write_tinker_result(
        tmp_path, wall_s=3600.0, active_compute_s=None,
        cost_usd_estimated=2.0,  # stale WALL-based cost (3600s) — must be ignored
        log_path=str(log_dir),
    )
    monkeypatch.setattr(assemble, "_RESULTS", tmp_path)
    rec = assemble.collect_tinker("gsm8k", "results/tinker_gsm8k.json")
    assert rec["train_s"] == pytest.approx(1800.0)
    assert rec["cost_usd"] == pytest.approx(1.0)            # from recomputed active, not 2.0
    assert rec["cost_usd"] == pytest.approx(assemble._tinker_cost_from_basis(rec["train_s"]))


# ---------------------------------------------------------------------------
# (2) Runner and assembler pick the SAME metrics file via the shared helper
# ---------------------------------------------------------------------------

def test_both_call_sites_select_same_metrics_file(tmp_path):
    # A direct file AND a nested file exist with DIFFERENT timing. The shared selector must
    # prefer the direct one, and both call sites (runner _read_metrics_jsonl, assembler
    # _tinker_active_compute_s) must agree because they route through it.
    direct = {"time/do_group_rollout_and_filter_constant_reward:total": 10.0, "time/train_step": 5.0}
    nested_rec = {"time/do_group_rollout_and_filter_constant_reward:total": 999.0, "time/train_step": 1.0}
    (tmp_path / "metrics.jsonl").write_text(json.dumps(direct) + "\n")
    nested_dir = tmp_path / "sub" / "deeper"
    nested_dir.mkdir(parents=True)
    (nested_dir / "metrics.jsonl").write_text(json.dumps(nested_rec) + "\n")

    chosen = tinker_runner.select_metrics_file(str(tmp_path))
    assert chosen == str(tmp_path / "metrics.jsonl")  # direct preferred over nested

    runner_records = tinker_runner._read_metrics_jsonl(str(tmp_path))
    assembler_active = assemble._tinker_active_compute_s(str(tmp_path))
    # Both must reflect the DIRECT file (15.0), never the nested 1000.0.
    assert tinker_runner.active_compute_s(runner_records) == pytest.approx(15.0)
    assert assembler_active == pytest.approx(15.0)


def test_shared_selector_falls_back_to_sorted_nested(tmp_path):
    # No direct file: deterministic sorted-first nested pick, identical for both call sites.
    for sub in ("b_later", "a_first"):
        d = tmp_path / sub
        d.mkdir()
        (d / "metrics.jsonl").write_text(json.dumps({"time/train_step": 1.0}) + "\n")
    chosen = tinker_runner.select_metrics_file(str(tmp_path))
    assert chosen == str(tmp_path / "a_first" / "metrics.jsonl")  # sorted -> a_first


# ---------------------------------------------------------------------------
# (3) active_compute_s: 0.0 (not None) when rows exist but sum to zero; callers keep 0.0
# ---------------------------------------------------------------------------

def test_active_compute_zero_when_rows_sum_to_zero():
    # Rows present (timing keys present) but values are 0 -> 0.0, NOT None.
    rows = [{"time/do_group_rollout_and_filter_constant_reward:total": 0.0, "time/train_step": 0.0}]
    assert tinker_runner.active_compute_s(rows) == 0.0
    # Genuinely no timing keys anywhere -> None (no data).
    assert tinker_runner.active_compute_s([{"some/other/metric": 1.0}]) is None
    assert tinker_runner.active_compute_s([]) is None


def test_collect_tinker_keeps_zero_active_not_wall(tmp_path, monkeypatch):
    # Stored active is a legitimate 0.0 -> latency/cost basis stays 0.0, NOT wall (3600s).
    # The old `active if active else wall` truthiness bug would have substituted wall here.
    _write_tinker_result(
        tmp_path, wall_s=3600.0, active_compute_s=0.0,
        cost_usd_estimated=0.0, log_path="/nonexistent",
    )
    monkeypatch.setattr(assemble, "_RESULTS", tmp_path)
    rec = assemble.collect_tinker("gsm8k", "results/tinker_gsm8k.json")
    assert rec["train_s"] == 0.0          # kept 0.0 active, not 3600 wall
    assert rec["cost_usd"] == 0.0         # cost from 0.0 active, not wall proxy
    assert rec["per_step_s"] == 0.0
    assert rec["total_s"] == pytest.approx(3600.0)  # wall still reported as total


def test_collect_tinker_falls_back_to_wall_only_when_active_is_none(tmp_path, monkeypatch):
    # active genuinely absent (None, no log) -> basis falls back to wall; cost from wall.
    _write_tinker_result(
        tmp_path, wall_s=3600.0, active_compute_s=None,
        cost_usd_estimated=None, log_path="/nonexistent",
    )
    monkeypatch.setattr(assemble, "_RESULTS", tmp_path)
    rec = assemble.collect_tinker("gsm8k", "results/tinker_gsm8k.json")
    assert rec["train_s"] == pytest.approx(3600.0)  # wall fallback
    assert rec["cost_usd"] == pytest.approx(assemble._tinker_cost_from_basis(3600.0))


# ---------------------------------------------------------------------------
# (4) --flash-config whose [environment].id != --env-id is rejected
# ---------------------------------------------------------------------------

def _flash_cfg(tmp_path, env_id):
    p = tmp_path / "cfg.toml"
    p.write_text(
        f'model = "Qwen/Qwen3.5-4B"\nalgorithm = "grpo"\n\n[environment]\nid = "{env_id}"\n'
        '\n[train]\nsteps = 30\n'
    )
    return str(p)


def _args(**kw):
    import argparse
    base = {"flash_config": None, "env_id": None}  # env_id None = "omitted" sentinel
    base.update(kw)
    return argparse.Namespace(**base)


def test_flash_config_env_mismatch_is_rejected(tmp_path):
    cfg = _flash_cfg(tmp_path, "primeintellect/reverse-text")
    args = _args(flash_config=cfg, env_id="gsm8k")  # explicit, conflicting --env-id
    with pytest.raises(SystemExit) as ei:
        bench._resolve_env_consistency(args)
    msg = str(ei.value)
    assert "reverse-text" in msg
    assert "gsm8k" in msg


def test_flash_config_env_match_passes(tmp_path):
    # Full slug primeintellect/gsm8k matches short explicit --env-id gsm8k (last path segment).
    cfg = _flash_cfg(tmp_path, "primeintellect/gsm8k")
    args = _args(flash_config=cfg, env_id="gsm8k")
    bench._resolve_env_consistency(args)  # no raise
    assert args.env_id == "gsm8k"


def test_flash_config_derives_env_id_when_omitted(tmp_path):
    # --env-id omitted (None) + --flash-config -> derive env-id from the config.
    cfg = _flash_cfg(tmp_path, "primeintellect/reverse-text")
    args = _args(flash_config=cfg)  # env_id stays None
    bench._resolve_env_consistency(args)
    assert args.env_id == "reverse-text"


def test_no_flash_config_is_noop_but_defaults_env(tmp_path):
    # Explicit env-id, no flash-config: selection is by env-id; value preserved.
    args = _args(env_id="hendrycks-math")
    bench._resolve_env_consistency(args)
    assert args.env_id == "hendrycks-math"
    # Omitted env-id, no flash-config: falls back to the module default.
    args2 = _args()
    bench._resolve_env_consistency(args2)
    assert args2.env_id == bench._ENV_ID_DEFAULT
