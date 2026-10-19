"""Status writes reject invalid requests before changing durable state or reporting."""

from unittest.mock import Mock

import pytest

from flash.runner.lifecycle import reporting, state, status


@pytest.fixture
def run_record(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "RUNS_DIR", str(tmp_path))
    report = Mock()
    monkeypatch.setattr(reporting, "_report_status", report)
    state._save_status(state.RunStatus(run_id="status-update", state="queued", spec={}))
    return tmp_path / "status-update.json", report


@pytest.mark.parametrize("new_state", ["runnning", "", "RUNNING", None, 1, []])
def test_invalid_state_does_not_write_or_report(run_record, new_state):
    path, report = run_record
    before = path.read_bytes()

    with pytest.raises(ValueError, match="invalid run state"):
        status._update("status-update", new_state, error="must not be written")

    assert path.read_bytes() == before
    report.assert_not_called()


@pytest.mark.parametrize(
    "field",
    [
        "billing_sate",
        "state",
        "spec",
        "created_at",
        "updated_at",
        "finished_at",
        "report_sequence",
        "last_heartbeat",
        "deployment",
        "estimated_cost_usd",
        "to_dict",
        "__dict__",
    ],
)
@pytest.mark.parametrize("initial_state", ["queued", "cancelled"])
def test_invalid_fields_are_rejected_even_when_transition_would_be_sticky(
    run_record, field, initial_state
):
    path, report = run_record
    state._save_status(state.RunStatus(run_id="status-update", state=initial_state, spec={}))
    before = path.read_bytes()

    with pytest.raises(ValueError, match="invalid status update fields"):
        status._update("status-update", "running", error="must not be written", **{field: None})

    assert path.read_bytes() == before
    report.assert_not_called()


@pytest.mark.parametrize(
    "new_state",
    ["queued", "provisioning", "running", "deployed", "done", "failed", "cancelled", "dry_run"],
)
def test_valid_states_persist_and_report(run_record, new_state):
    _, report = run_record

    assert status._update("status-update", new_state, error=None)

    saved = status.get_status("status-update")
    assert saved.state == new_state
    assert saved.report_sequence == 2
    assert (saved.finished_at is not None) == (new_state in state.TERMINAL_STATES)
    report.assert_called_once_with(saved)


def test_terminal_updates_preserve_stickiness_and_first_completion_time(run_record):
    path, report = run_record
    assert status._update("status-update", "done", cost_usd=1.0, artifacts_dir="/artifacts")
    finished_at = status.get_status("status-update").finished_at
    before = path.read_bytes()
    report.reset_mock()

    assert status._update("status-update", "running", cost_usd=2.0) is False
    assert path.read_bytes() == before
    report.assert_not_called()
    assert status._update("status-update", "done", cost_usd=2.0)
    assert status.get_status("status-update").cost_usd == 2.0
    assert status._update("status-update", "cancelled", allow_from_terminal=True)
    assert status.get_status("status-update").finished_at == finished_at


def test_terminal_override_cannot_bypass_validation(run_record):
    path, report = run_record
    assert status._update("status-update", "done")
    before = path.read_bytes()
    report.reset_mock()

    with pytest.raises(ValueError, match="invalid run state"):
        status._update("status-update", "runnning", allow_from_terminal=True)

    assert path.read_bytes() == before
    report.assert_not_called()


def test_lifecycle_milestones_remain_write_once(run_record):
    assert status._update(
        "status-update", "running", lifecycle_started_attempt=0, lifecycle_progressed_attempt=0
    )
    assert status._update(
        "status-update", "running", lifecycle_started_attempt=1, lifecycle_progressed_attempt=1
    )

    saved = status.get_status("status-update")
    assert saved.lifecycle_started_attempt == 0
    assert saved.lifecycle_progressed_attempt == 0
