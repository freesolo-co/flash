from __future__ import annotations

import time

from flash.cli.ui import render


def test_running_resource_observation_keeps_sparse_progress_visibly_active(monkeypatch) -> None:
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    observed_at = time.time() - 3
    occurred_at = time.time() - 3600
    payload = {
        "run_id": "flash-sparse-progress",
        "state": "running",
        "spec": {"model": "Qwen/Qwen3.5-4B", "algorithm": "grpo"},
        "attempt": {
            "attempt_id": 2,
            "fence": 9,
            "state": "active",
            "work_deadline_at": time.time() + 600,
        },
        "progress": {
            "attempt_id": 2,
            "fence": 9,
            "sequence": 4,
            "phase": "rl_step",
            "completed_steps": 1,
            "occurred_at": occurred_at,
            "observed_at": occurred_at + 2,
        },
        "resource": {
            "attempt_id": 2,
            "fence": 9,
            "state": "running",
            "observed_at": observed_at,
            "transport": "ok",
        },
        "result": None,
    }

    output = render.run_status(payload)

    assert "attempt" in output
    assert "resource" in output
    assert "running" in output
    assert "observed 3s ago" in output
    assert "progress observed" in output
    assert "stalled" not in output.lower()
    assert "heartbeat" not in output.lower()
