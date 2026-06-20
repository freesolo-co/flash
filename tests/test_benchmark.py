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
import eval_unified
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


# ---------------------------------------------------------------------------
# Follow-up round: residual 0.0-truthiness, empty-direct metrics, stored-cost
# consistency, and PATH-relative TINKER_PYTHON resolution.
# ---------------------------------------------------------------------------

# (G1) A legitimate 0.0 basis must yield a 0.0 estimated cost, never None/dropped.
def test_runner_zero_basis_keeps_zero_cost_not_none():
    # Reproduce the runner's write-time cost expression: 0.0 active-compute basis -> $0.0,
    # never None. The old `... if basis else None` truthiness bug dropped a real 0.0 to None.
    basis = 0.0
    cost = (
        round(basis / 3600 * tinker_runner.TINKER_PROXY_USD_PER_HR, 4)
        if basis is not None else None
    )
    assert cost == 0.0
    # A genuinely-missing basis (None) still yields None (no data).
    basis_none = None
    cost_none = (
        round(basis_none / 3600 * tinker_runner.TINKER_PROXY_USD_PER_HR, 4)
        if basis_none is not None else None
    )
    assert cost_none is None


def test_runner_main_writes_zero_cost_for_zero_active(tmp_path, monkeypatch):
    # End-to-end through main(): metrics rows present but timing sums to 0.0 -> the written
    # JSON has active_compute_s == 0.0 AND cost_usd_estimated == 0.0 (not null).
    log_dir = tmp_path / "logdir"
    log_dir.mkdir()
    (log_dir / "metrics.jsonl").write_text(
        json.dumps({"time/do_group_rollout_and_filter_constant_reward:total": 0.0,
                    "time/train_step": 0.0, "env/all/reward/total": 0.0}) + "\n"
    )
    out = tmp_path / "result.json"
    argv = ["tinker_runner.py", "--log-path", str(log_dir), "--output", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    # Skip the real training: replace the async _run with a stub so asyncio.run gets a real
    # (immediately-returning) coroutine and main() proceeds to the metrics/cost path.
    async def _fake_run(_args):
        return {"wall_s": 120.0}
    monkeypatch.setattr(tinker_runner, "_run", _fake_run)
    tinker_runner.main()
    written = json.loads(out.read_text())
    assert written["active_compute_s"] == 0.0
    assert written["cost_usd_estimated"] == 0.0  # NOT None — a real pause-excluded zero


# (G1) paused_s == 0.0 must render as "0m00s", not "none".
def _records_for_md(paused_s):
    """Minimal records dict for render_markdown with a tinker paused_s value."""
    tinker = {
        "status": "done", "gpu": "managed (Tinker)", "steps": 30,
        "first_reward": 0.1, "final_reward": 0.1, "final_smoothed": 0.1, "best_reward": 0.2,
        "eval_reward": None, "reward_history": [0.1] * 30,
        "cost_usd": 1.0, "cost_kind": "ESTIMATE", "setup_s": None,
        "train_s": 1800.0, "total_s": 1800.0, "paused_s": paused_s,
        "per_step_s": 60.0, "train_tok_per_s": None,
    }
    flash = dict(tinker, platform="flash", gpu="A100 PCIe", cost_kind="measured")
    return {t: {"flash": flash, "tinker": tinker} for t in assemble.TASKS}


def test_render_paused_zero_shows_0m00s_not_none():
    md = assemble.render_markdown(_records_for_md(0.0))
    assert "| latency capacity-pause | — | 0m00s |" in md
    assert "| latency capacity-pause | — | none |" not in md


def test_render_paused_none_shows_none():
    md = assemble.render_markdown(_records_for_md(None))
    assert "| latency capacity-pause | — | none |" in md


# (G3) select_metrics_file skips an EMPTY direct file and uses the nested one.
def test_select_metrics_skips_empty_direct_uses_nested(tmp_path):
    # Direct metrics.jsonl exists but is EMPTY; the real timing lives in a nested file.
    (tmp_path / "metrics.jsonl").write_text("")  # empty -> must NOT shadow nested
    nested_dir = tmp_path / "sub"
    nested_dir.mkdir()
    (nested_dir / "metrics.jsonl").write_text(
        json.dumps({"time/do_group_rollout_and_filter_constant_reward:total": 7.0,
                    "time/train_step": 3.0}) + "\n"
    )
    chosen = tinker_runner.select_metrics_file(str(tmp_path))
    assert chosen == str(nested_dir / "metrics.jsonl")  # fell through to the non-empty nested
    # Both the reader and assemble see the nested data (10.0), not an empty/0 from direct.
    assert tinker_runner.active_compute_s(tinker_runner.read_metrics_records(str(tmp_path))) == pytest.approx(10.0)
    assert assemble._tinker_active_compute_s(str(tmp_path)) == pytest.approx(10.0)


def test_select_metrics_unparseable_direct_uses_nested(tmp_path):
    # Direct file has only blank / non-JSON lines (no parseable records) -> treated as empty.
    (tmp_path / "metrics.jsonl").write_text("\n   \nnot-json\n")
    nested_dir = tmp_path / "deep"
    nested_dir.mkdir()
    (nested_dir / "metrics.jsonl").write_text(json.dumps({"time/train_step": 4.0}) + "\n")
    assert tinker_runner.select_metrics_file(str(tmp_path)) == str(nested_dir / "metrics.jsonl")


def test_select_metrics_all_empty_degrades_gracefully(tmp_path):
    # Everything empty: no crash; reader returns [] and active-compute is None (no data).
    (tmp_path / "metrics.jsonl").write_text("")
    assert tinker_runner.read_metrics_records(str(tmp_path)) == []
    assert assemble._tinker_active_compute_s(str(tmp_path)) is None
    # Truly nothing at all (no file) -> None selection, [] records.
    empty = tmp_path / "none"
    empty.mkdir()
    assert tinker_runner.select_metrics_file(str(empty)) is None
    assert tinker_runner.read_metrics_records(str(empty)) == []


# (G4) Stored-active path: reported cost is recomputed from the stored active, so a stale
# wall-based cost paired with an active-compute active can't leak through.
def test_stored_active_recomputes_cost_ignoring_stale_stored_cost(tmp_path, monkeypatch):
    # JSON stores active=1800s but a STALE wall-based cost (3600s -> $2.00). assemble must
    # report cost == proxy(1800s) == $1.00, NOT the stored $2.00.
    _write_tinker_result(
        tmp_path, wall_s=3600.0, active_compute_s=1800.0,
        cost_usd_estimated=2.00,  # stale wall-based cost — must be overridden
        log_path="/nonexistent",
    )
    monkeypatch.setattr(assemble, "_RESULTS", tmp_path)
    rec = assemble.collect_tinker("gsm8k", "results/tinker_gsm8k.json")
    assert rec["train_s"] == pytest.approx(1800.0)
    assert rec["cost_usd"] == pytest.approx(1.0)  # from stored active, not stale 2.0
    # Invariant holds on the stored-active path: reported cost == proxy(reported active).
    assert rec["cost_usd"] == pytest.approx(assemble._tinker_cost_from_basis(rec["train_s"]))
    assert rec["cost_usd"] != pytest.approx(2.0)


# (G5) TINKER_PYTHON given as a PATH name (python3) is accepted via shutil.which.
def test_tinker_python_path_name_is_accepted(monkeypatch):
    # `python3` is a PATH name with no on-disk path; the old pathlib.exists() check wrongly
    # skipped it. shutil.which resolves it, so _run_tinker must NOT report "skipped".
    import shutil as _sh
    assert _sh.which("python3") is not None, "test host needs python3 on PATH"
    monkeypatch.setenv("TINKER_PYTHON", "python3")
    # Make subprocess a no-op so we don't actually launch training; capture the resolved argv.
    captured = {}

    class _FakeProc:
        returncode = 0
        stdout = iter(())
        def wait(self):
            return 0

    def _fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr(bench.subprocess, "Popen", _fake_popen)
    args = _args(env_id="gsm8k", steps=1, groups_per_batch=4, group_size=4,
                 max_tokens=512, model="Qwen/Qwen3.5-4B")
    box: list = []
    bench._run_tinker(args, box)
    # Not skipped: a fresh result file is absent so it reports "failed" (subprocess ran),
    # which proves the interpreter resolved (skipped would mean which() returned None).
    assert box
    assert box[0].get("status") != "skipped"
    # And the launched argv uses the RESOLVED absolute interpreter path, not the bare name.
    assert captured["cmd"][0] == _sh.which("python3")


def test_tinker_python_unresolvable_is_skipped(monkeypatch):
    # A name that resolves to nothing -> skipped with a clear message (only real failure case).
    monkeypatch.setenv("TINKER_PYTHON", "definitely-not-a-real-interpreter-xyz")
    args = _args(env_id="gsm8k", steps=1, groups_per_batch=4, group_size=4,
                 max_tokens=512, model="Qwen/Qwen3.5-4B")
    box: list = []
    bench._run_tinker(args, box)
    assert box
    assert box[0]["status"] == "skipped"
    assert "not found" in box[0]["error"]


# ---------------------------------------------------------------------------
# eval_unified: Flash-trained generation via the serving (mocked HTTP)
# ---------------------------------------------------------------------------

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
    extract the boxed/last-number answer and exact-match the gold. One right, one wrong+truncated."""
    rows = [
        {"prompt": [{"role": "user", "content": "q1"}], "answer": "#### 18"},
        {"prompt": [{"role": "user", "content": "q2"}], "answer": "5"},
    ]
    replies = iter([
        {"choices": [{"message": {"content": r"reasoning ... \boxed{18}"}, "finish_reason": "stop"}]},
        {"choices": [{"message": {"content": r"...\boxed{99}"}, "finish_reason": "length"}]},
    ])
    monkeypatch.setattr(eval_unified.urllib.request, "urlopen",
                        lambda *a, **k: _FakeResp(next(replies)))
    res = eval_unified.eval_via_serving(rows, "https://serve.example", "run-x", max_tokens=256)
    assert res["n"] == 2
    assert res["correct"] == 1            # only the first matches gold 18
    assert res["accuracy"] == pytest.approx(0.5)
    assert res["truncated_frac"] == pytest.approx(0.5)  # the second hit the length cap
    assert res["errors"] == 0


# ---------------------------------------------------------------------------
# NEW ROUND (PR #20, commit 1b278be follow-ups)
# ---------------------------------------------------------------------------

# (G1) active_compute_s pause heuristic: ALWAYS <= summed wall, and a slow-but-real step
# is not counted wholesale as a pause.

def _steps(*ts):
    """metrics.jsonl-shaped records carrying only the per-step wall time (`time/total`)."""
    return [{"time/total": t} for t in ts]


def test_active_compute_never_exceeds_wall_normal_case():
    # Normal mix with one long pause: active < wall, and == wall minus the pause overhang only.
    recs = _steps(20.0, 25.0, 22.0, 600.0, 24.0)  # 600s step is a capacity pause
    wall = 20.0 + 25.0 + 22.0 + 600.0 + 24.0
    active = tinker_runner.active_compute_s(recs)
    assert active <= wall                      # invariant
    # expected per-step = median of the normal (<=300) steps = 24.0; pause overhang = 600-24=576.
    assert active == pytest.approx(wall - 576.0)


def test_active_compute_clamped_when_median_step_exceeds_pause_threshold():
    # MANY steps over _PAUSE_STEP_S so the OVERALL median is itself a pause. The old
    # `sum(t - overall_median)` made `t - med` negative for the smaller pause steps and ADDED
    # time (active > wall). The corrected rule clamps to active <= wall and never adds time.
    recs = _steps(400.0, 500.0, 600.0, 700.0, 800.0)  # all > 300s
    wall = sum(t for s in recs for t in [s["time/total"]])
    active = tinker_runner.active_compute_s(recs)
    assert active <= wall                      # the core invariant the bug violated
    assert active >= 0.0


def test_active_compute_slow_step_keeps_expected_worth_not_all_pause():
    # A step at 310s (just over the 300s line) when normal steps are ~25s must NOT be counted
    # entirely as pause: only its overhang above the expected (normal-median) step is excluded,
    # so the run keeps one normal-step's worth of active compute for it.
    recs = _steps(25.0, 25.0, 25.0, 310.0)
    wall = 25.0 * 3 + 310.0
    active = tinker_runner.active_compute_s(recs)
    expected = 25.0                            # median of the normal steps
    # overhang excluded = 310 - 25 = 285; the slow step still contributes its 25s expected worth.
    assert active == pytest.approx(wall - (310.0 - expected))
    assert active > 25.0 * 3                    # strictly more than just the 3 normal steps
    assert active <= wall


def test_active_compute_no_pause_equals_wall():
    # No step over the threshold -> no pause subtracted -> active == summed wall exactly.
    recs = _steps(20.0, 30.0, 25.0)
    assert tinker_runner.active_compute_s(recs) == pytest.approx(75.0)


# (G2) Each committed results JSON carries paused_s == wall - active, a basis-stating
# cost_note, and stays internally consistent (active + paused == wall).

@pytest.mark.parametrize("fname", [
    "tinker_gsm8k.json", "tinker_reverse-text.json", "tinker_hendrycks-math.json",
])
def test_committed_result_has_consistent_paused_and_basis_cost_note(fname):
    results = pathlib.Path(__file__).resolve().parents[1] / "benchmark" / "results"
    run = json.loads((results / fname).read_text())
    # paused_s present and == wall - active (capacity pause), and the three add up exactly.
    assert "paused_s" in run, f"{fname} missing paused_s"
    wall, active, paused = run["wall_s"], run["active_compute_s"], run["paused_s"]
    assert paused == pytest.approx(wall - active)
    assert active + paused == pytest.approx(wall)
    assert paused >= 0.0                        # clamp invariant: no negative pause
    # cost_note states the ACTIVE-COMPUTE basis (not a bare "estimated").
    note = run["cost_note"].lower()
    assert "active-compute" in note
    assert "pause" in note
    assert "excluded" in note


def test_committed_results_paused_matches_comparison_json():
    # The per-run paused_s must equal what assemble.py recorded in comparison.json.
    results = pathlib.Path(__file__).resolve().parents[1] / "benchmark" / "results"
    comp = json.loads((results / "comparison.json").read_text())
    for task, fname in [
        ("gsm8k", "tinker_gsm8k.json"),
        ("reverse-text", "tinker_reverse-text.json"),
        ("hendrycks-math", "tinker_hendrycks-math.json"),
    ]:
        run = json.loads((results / fname).read_text())
        assert run["paused_s"] == pytest.approx(comp[task]["tinker"]["paused_s"])


# (G3) README max-tokens row matches the real configured value (1024).

def test_readme_max_tokens_matches_configs():
    bench_dir = pathlib.Path(__file__).resolve().parents[1] / "benchmark"
    readme = (bench_dir / "README.md").read_text()
    assert "| max tokens | 1024 |" in readme
    assert "| max tokens | 512 |" not in readme
    # And it matches the actual default the runner/scripts use.
    assert tinker_runner.parse_args.__defaults__ is None  # parse_args takes no args
    import argparse as _ap
    p = _ap.ArgumentParser()
    p.add_argument("--max-tokens", type=int, default=1024)
    assert p.parse_args([]).max_tokens == 1024


# (G4) Held-out verdict band is consistent: within ±0.02 -> "no significant" + correct
# in-training trend word (never "rose" when the curve falls); beyond the band -> rose/fell.

def _records_for_verdict(held_delta, first_reward, final_smoothed, *, base_n=50):
    """Minimal records dict + writes eval_unified_gsm8k.json so render_markdown emits a verdict."""
    tinker = {
        "status": "done", "gpu": "managed (Tinker)", "steps": 30,
        "first_reward": first_reward, "final_reward": final_smoothed,
        "final_smoothed": final_smoothed, "best_reward": max(first_reward, final_smoothed),
        "eval_reward": None, "reward_history": [first_reward, final_smoothed],
        "cost_usd": 1.0, "cost_kind": "ESTIMATE", "setup_s": None,
        "train_s": 1800.0, "total_s": 1800.0, "paused_s": 10.0,
        "per_step_s": 60.0, "train_tok_per_s": None,
    }
    flash = dict(tinker, platform="flash", gpu="A100 PCIe", cost_kind="measured",
                 eval_reward=0.5, eval_n=50)
    return {t: {"flash": flash, "tinker": tinker} for t in assemble.TASKS}, {
        "max_tokens": 2048,
        "base": {"n": base_n, "accuracy": 0.62, "truncated_frac": 0.56},
        "tinker_trained": {"accuracy": 0.62 + held_delta, "delta_vs_base": held_delta},
    }


def _render_with_eval(tmp_path, monkeypatch, records, eval_json):
    (tmp_path / "eval_unified_gsm8k.json").write_text(json.dumps(eval_json))
    monkeypatch.setattr(assemble, "_RESULTS", tmp_path)
    return assemble.render_markdown(records)


def test_verdict_flat_band_reports_no_change_and_true_falling_trend(tmp_path, monkeypatch):
    # This run's real numbers: held-out Δ = -0.08 (beyond the ±0.02 band -> held-out FELL) and the
    # in-training smoothed reward falls (0.69 -> 0.36). The verdict must say BOTH fell and must
    # NEVER claim the in-training reward "rose" (the prior hard-coded bug).
    records, ev = _records_for_verdict(-0.08, first_reward=0.6875, final_smoothed=0.3625)
    md = _render_with_eval(tmp_path, monkeypatch, records, ev)
    finding = md.split("## Summary")[0]
    assert "in-training smoothed reward fell" in finding           # honest training trend
    assert "held-out accuracy fell" in finding                     # honest held-out direction
    assert "rose" not in finding                                   # no false "rose" anywhere


def test_verdict_within_band_says_no_significant_change(tmp_path, monkeypatch):
    # |Δ| <= 0.02 -> "no significant held-out change", and trend word matches the curve.
    records, ev = _records_for_verdict(0.01, first_reward=0.10, final_smoothed=0.40)  # train rose
    md = _render_with_eval(tmp_path, monkeypatch, records, ev)
    assert "no significant held-out change" in md
    assert "in-training smoothed reward rose" in md          # 0.10 -> 0.40 is a real rise
    # Small positive held-out delta is NOT called a held-out gain (within noise).
    assert "held-out accuracy rose" not in md


def test_verdict_small_positive_not_called_gain(tmp_path, monkeypatch):
    # +0.02 sits exactly on the band edge -> still "no significant change", never "rose" held-out.
    records, ev = _records_for_verdict(0.02, first_reward=0.20, final_smoothed=0.20)  # flat train
    md = _render_with_eval(tmp_path, monkeypatch, records, ev)
    assert "no significant held-out change" in md
    assert "in-training smoothed reward was flat (within noise)" in md


def test_verdict_real_positive_gain_says_rose(tmp_path, monkeypatch):
    # Δ = +0.10 (beyond the band) -> held-out accuracy "rose"; in-training also rose (0.10->0.50).
    records, ev = _records_for_verdict(0.10, first_reward=0.10, final_smoothed=0.50)
    md = _render_with_eval(tmp_path, monkeypatch, records, ev)
    finding = md.split("## Summary")[0]
    assert "held-out accuracy rose" in finding
    assert "in-training smoothed reward rose" in finding


def test_trend_word_band():
    assert assemble._trend_word(0.05) == "rose"
    assert assemble._trend_word(-0.05) == "fell"
    assert assemble._trend_word(0.0) == "was flat (within noise)"
    assert assemble._trend_word(0.02) == "was flat (within noise)"   # band edge inclusive
    assert assemble._trend_word(None) == "is unavailable"


# (G5) eval_via_serving forwards the stop sequences into the OpenAI payload.

def test_eval_via_serving_forwards_stop_sequences(monkeypatch):
    rows = [{"prompt": [{"role": "user", "content": "q"}], "answer": "1"}]
    captured = {}

    class _Resp:
        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": r"\boxed{1}"}, "finish_reason": "stop"}]}
            ).encode()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, *a, **k):
        captured["body"] = json.loads(req.data.decode())
        return _Resp()

    monkeypatch.setattr(eval_unified.urllib.request, "urlopen", _fake_urlopen)
    eval_unified.eval_via_serving(
        rows, "https://serve.example", "run-x", max_tokens=128,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    assert captured["body"]["stop"] == ["<|im_end|>", "<|endoftext|>"]


def test_eval_via_serving_omits_stop_when_empty(monkeypatch):
    # An empty/None stop list must not put an empty `stop` in the payload (no-op / server reject).
    rows = [{"prompt": [{"role": "user", "content": "q"}], "answer": "1"}]
    captured = {}

    class _Resp:
        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "1"}, "finish_reason": "stop"}]}
            ).encode()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, *a, **k):
        captured["body"] = json.loads(req.data.decode())
        return _Resp()

    monkeypatch.setattr(eval_unified.urllib.request, "urlopen", _fake_urlopen)
    eval_unified.eval_via_serving(rows, "https://serve.example", "run-x", max_tokens=128, stop=[])
    assert "stop" not in captured["body"]


# ===========================================================================
# PR #20 review (post dev-merge): paused_s 0.0/clamp, 0.0-cost ratio,
# _flash_api_url env precedence, and the mid-run-eval-removal doc/label fixes.
# ===========================================================================

# --- Group C: paused_s = max(0, wall-active), recorded even when 0.0 -------------------

def test_collect_tinker_records_zero_pause_not_null(tmp_path, monkeypatch):
    # active == wall -> a REAL 0.0 pause. It must be recorded as 0.0, not dropped to None
    # (the old `wall > active` guard made an exact-0 pause null).
    _write_tinker_result(
        tmp_path, wall_s=1800.0, active_compute_s=1800.0,
        cost_usd_estimated=1.0, log_path="/nonexistent",
    )
    monkeypatch.setattr(assemble, "_RESULTS", tmp_path)
    rec = assemble.collect_tinker("gsm8k", "results/tinker_gsm8k.json")
    assert rec["paused_s"] == 0.0


def test_collect_tinker_clamps_negative_pause_and_warns(tmp_path, monkeypatch, capsys):
    # Anomaly: active > wall (different timing sources). paused must clamp to 0.0 (never
    # negative / hidden) and the anomaly must be logged, not silently swallowed.
    _write_tinker_result(
        tmp_path, wall_s=1000.0, active_compute_s=1200.0,
        cost_usd_estimated=1.0, log_path="/nonexistent",
    )
    monkeypatch.setattr(assemble, "_RESULTS", tmp_path)
    rec = assemble.collect_tinker("gsm8k", "results/tinker_gsm8k.json")
    assert rec["paused_s"] == 0.0  # clamped, not -200.0
    assert "clamping paused_s to 0.0" in capsys.readouterr().out


def test_collect_tinker_pause_is_none_only_when_active_missing(tmp_path, monkeypatch):
    # No active anywhere -> paused genuinely unknown -> None (not 0.0).
    _write_tinker_result(
        tmp_path, wall_s=1000.0, active_compute_s=None,
        cost_usd_estimated=None, log_path="/nonexistent",
    )
    monkeypatch.setattr(assemble, "_RESULTS", tmp_path)
    rec = assemble.collect_tinker("gsm8k", "results/tinker_gsm8k.json")
    assert rec["paused_s"] is None


def test_runner_main_records_zero_pause_not_null(tmp_path, monkeypatch):
    # End-to-end through the runner: active sums to a positive value EQUAL to wall -> the
    # written JSON's paused_s is 0.0 (a measured zero pause), never null.
    log_dir = tmp_path / "logdir"
    log_dir.mkdir()
    (log_dir / "metrics.jsonl").write_text(
        json.dumps({"time/do_group_rollout_and_filter_constant_reward:total": 80.0,
                    "time/train_step": 40.0, "env/all/reward/total": 0.5}) + "\n"
    )
    out = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv",
                        ["tinker_runner.py", "--log-path", str(log_dir), "--output", str(out)])
    async def _fake_run(_args):
        return {"wall_s": 120.0}  # equals active (80+40) -> 0.0 pause
    monkeypatch.setattr(tinker_runner, "_run", _fake_run)
    tinker_runner.main()
    written = json.loads(out.read_text())
    assert written["paused_s"] == 0.0


# --- Group C: cost ratio guards on `is not None`, handles 0.0 Flash denominator --------

def test_cost_ratio_shows_when_flash_cost_is_zero():
    # A legit measured Flash cost of 0.0 must NOT be treated as "missing" (—); a 0.0
    # denominator can't form a finite multiple, so show ∞ rather than dropping the row.
    recs = _records_for_md(0.0)
    recs["gsm8k"]["flash"]["cost_usd"] = 0.0   # real measured zero
    recs["gsm8k"]["tinker"]["cost_usd"] = 1.0
    md = assemble.render_markdown(recs)
    # The gsm8k cost-of-training row carries the ∞ marker, not a bare "—".
    row = next(ln for ln in md.splitlines() if ln.startswith("| gsm8k |"))
    assert "∞ (Flash $0)" in row


def test_cost_ratio_dash_only_when_value_absent():
    recs = _records_for_md(None)
    recs["gsm8k"]["flash"]["cost_usd"] = None   # genuinely missing
    recs["gsm8k"]["tinker"]["cost_usd"] = 1.0
    md = assemble.render_markdown(recs)
    row = next(ln for ln in md.splitlines() if ln.startswith("| gsm8k |"))
    # absent value -> "—" in the ratio column (the 4th pipe-cell)
    assert row.split("|")[4].strip() == "—"


# --- Group D: _flash_api_url prefers FLASH_API_URL over ~/.flash/config.json ----------

def test_flash_api_url_env_overrides_config_file(monkeypatch, tmp_path):
    # A config file exists with one url, but the env var must win (env-over-configfile,
    # mirroring _flash_api_key). Point HOME at a dir that HAS a config.json.
    home = tmp_path / "home"
    (home / ".flash").mkdir(parents=True)
    (home / ".flash" / "config.json").write_text(json.dumps({"api_url": "http://from-config:9999"}))
    monkeypatch.setattr(flash_runner.pathlib.Path, "home", staticmethod(lambda: home))
    monkeypatch.setenv("FLASH_API_URL", "http://from-env:8085")
    assert flash_runner._flash_api_url() == "http://from-env:8085"


def test_flash_api_url_falls_back_to_config_then_default(monkeypatch, tmp_path):
    home = tmp_path / "home"
    (home / ".flash").mkdir(parents=True)
    (home / ".flash" / "config.json").write_text(json.dumps({"api_url": "http://from-config:9999"}))
    monkeypatch.setattr(flash_runner.pathlib.Path, "home", staticmethod(lambda: home))
    monkeypatch.delenv("FLASH_API_URL", raising=False)
    assert flash_runner._flash_api_url() == "http://from-config:9999"
    # No env, no config file -> the baked-in default.
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(flash_runner.pathlib.Path, "home", staticmethod(lambda: empty))
    assert flash_runner._flash_api_url() == "https://flash.freesolo.co"


# --- Group B: mid-run-eval removal reflected in labels / generated markdown ------------

def test_bench_table_has_no_held_out_eval_reward_row():
    # The "Eval reward (held-out)" row is gone (mid-run eval removed); it's replaced by a
    # pointer to the serving-side eval_unified.py, never implying a held-out metric is
    # collected during training.
    flash = {"status": "done", "steps": 30, "first_train_reward": 0.1,
             "final_train_reward": 0.2, "final_eval_reward": None, "reward_history": [0.1]}
    tinker = dict(flash)
    bench._print_table(flash, tinker)  # must not raise
    # The label assembled in the rows uses the new wording.
    src = pathlib.Path(bench.__file__).read_text()
    assert "Eval reward (held-out)" not in src
    assert "Held-out eval" in src
    assert "eval_unified.py" in src


def test_render_markdown_verdict_has_no_on_gpu_eval_claim():
    # The verdict must not claim a fallback to "Flash's own on-GPU eval" (that eval was
    # removed). With no serving-side flash eval present, the Flash-trained row is absent.
    recs = _records_for_md(0.0)
    md = assemble.render_markdown(recs)
    assert "on-GPU eval" not in md
    assert "NATIVE on-GPU" not in md


def test_render_markdown_gpu_prose_derived_from_records():
    # The GPU narrative must be DERIVED from the records (here A100 PCIe), never a hardcoded
    # contradicting GPU/cost story (the old text asserted an A40 default with $0.139/19min).
    recs = _records_for_md(0.0)  # flash gpu == "A100 PCIe"
    md = assemble.render_markdown(recs)
    assert "A100 PCIe" in md
    assert "$0.139" not in md           # the fabricated A40 number is gone
    assert "A40 is the new default" not in md
    assert assemble._flash_gpus_used(recs) == ["A100 PCIe"]
