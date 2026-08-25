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


def test_progress_from_a_previous_fence_is_not_rendered() -> None:
    from flash.cli.ui.lifecycle import _lifecycle_pairs

    pairs = _lifecycle_pairs(
        {
            "attempt": {"attempt_id": 2, "fence": 9},
            "progress": {
                "attempt_id": 2,
                "fence": 8,
                "phase": "rl_step",
                "completed_steps": 7,
            },
            "resource": {
                "attempt_id": 2,
                "fence": 9,
                "state": "running",
            },
        }
    )

    assert dict(pairs)["resource"] == "running"
    assert "progress" not in dict(pairs)


def test_lifecycle_rows_include_current_progress_metrics_and_result(monkeypatch) -> None:
    from flash.cli.ui import lifecycle

    monkeypatch.setattr(lifecycle.time, "time", lambda: 1000.0)
    pairs = lifecycle._lifecycle_pairs(
        {
            "attempt": {
                "attempt_id": 3,
                "fence": 11,
                "state": "result_pending",
                "work_deadline_at": 1060.0,
            },
            "progress": {
                "attempt_id": 3,
                "fence": 11,
                "phase": "sft_step",
                "completed_steps": 4,
                "occurred_at": 970.0,
                "observed_at": 980.0,
                "metrics": {"loss": 0.25},
                "checkpoint": {"step": 4},
            },
            "result": {
                "attempt_id": 3,
                "fence": 11,
                "outcome": "failed",
                "failure_class": "oom",
            },
        }
    )

    assert pairs == [
        ("attempt", "3 / fence 11 · result_pending"),
        ("work deadline", "60s left"),
        ("progress", "sft_step · 4 completed steps"),
        ("progress occurred", "30s ago"),
        ("progress observed", "20s ago"),
        ("metrics", "loss=0.25"),
        ("checkpoint", "{'step': 4}"),
        ("result", "failed · oom"),
    ]


def test_progress_age_never_creates_a_failure_or_health_inference(monkeypatch) -> None:
    from flash.cli.ui import lifecycle

    monkeypatch.setattr(lifecycle.time, "time", lambda: 10_000.0)
    pairs = lifecycle._lifecycle_pairs(
        {
            "attempt": {"attempt_id": 1, "fence": 2},
            "progress": {
                "attempt_id": 1,
                "fence": 2,
                "phase": "opd_step",
                "completed_steps": 1,
                "occurred_at": 1.0,
                "observed_at": 2.0,
            },
        }
    )

    rendered = " ".join(value for _label, value in pairs).lower()
    assert "2.8h ago" in rendered
    assert "stalled" not in rendered
    assert "failed" not in rendered
    assert "retry" not in rendered


def test_malformed_attempt_identity_does_not_bind_observations() -> None:
    from flash.cli.ui.lifecycle import _lifecycle_pairs, live_attempt

    payload = {
        "attempt": {"attempt_id": True, "fence": 1},
        "progress": {"attempt_id": 1, "fence": 1, "phase": "rl_step"},
    }

    assert live_attempt(payload) is None
    assert _lifecycle_pairs(payload) == []
