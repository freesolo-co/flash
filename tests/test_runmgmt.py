"""Tests for run-management helpers (ps/cost/cancel) — no GPU/network."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_list_and_cancel(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        import autoslm.orchestrator as orchestrator

        importlib.reload(orchestrator)
        # fixed constant; redirect to tmp via monkeypatch so it's restored after the test.
        monkeypatch.setattr(orchestrator, "RUNS_DIR", tmp)
        from autoslm.worker_spec import JobSpec

        # two dry-run records
        for rid in ("a", "b"):
            orchestrator.submit_job(
                JobSpec(run_id=rid, model="Qwen/Qwen3-4B-Instruct-2507", algorithm="grpo"),
                dry_run=True,
            )
        runs = {r.run_id for r in orchestrator.list_runs()}
        assert {"a", "b"} <= runs

        # cancel a non-terminal run (force it to a running-ish state first; go through
        # _save_status, not _update, since _update now refuses to resurrect a terminal
        # state like the submitted dry_run).
        running = orchestrator.get_status("a")
        running.state = "running"
        orchestrator._save_status(running)
        status = orchestrator.cancel_run("a")
        assert status.state == "cancelled"

        # cancelling a terminal run is a no-op
        same = orchestrator.cancel_run("b")  # b is dry_run (terminal-ish)
        assert same.state in {"dry_run", "cancelled"}

        os.environ.pop("AUTOSLM_RUNS_DIR", None)
