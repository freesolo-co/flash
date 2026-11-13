"""An unconfirmed paid submit leaves its ID behind, so rerunning the command attaches to it."""

from __future__ import annotations

import json
import os
import socket

import pytest

from flash.client import ApiClient, ApiError, RequestTimeoutError, pending_submissions
from flash.client.submissions import MAX_CONSECUTIVE_TRANSPORT_FAILURES, submit_run

_PROJECT_ID = "11111111-1111-4111-8111-111111111111"
_BODY = {"spec": {"model": "m", "project": _PROJECT_ID}}
_CREATED = {"run_id": "flash-1-a", "state": "queued"}


class _Plane:
    """Scripted control plane: each _request call pops the next outcome."""

    def __init__(self, *outcomes, api_url="http://plane.invalid", api_key="fslo-user-test"):
        self.api_url, self.api_key, self.timeout = api_url, api_key, 60.0
        self.outcomes = list(outcomes)
        self.ids: list[str] = []

    def _request(self, method, path, body=None, **_kwargs):
        self.ids.append(path.rsplit("/", 1)[1])
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _unconfirmed(plane, body=_BODY):
    with pytest.raises(RequestTimeoutError) as error:
        submit_run(plane, body, sleep=lambda _s: None)
    return error.value


def _timeouts():
    return [RequestTimeoutError("timed out")] * (MAX_CONSECUTIVE_TRANSPORT_FAILURES + 1)


def _records() -> dict:
    path = pending_submissions._path()
    return json.loads(path.read_text())["pending"] if path.exists() else {}


def test_rerun_after_unconfirmed_submit_reuses_the_submission_id():
    first = _Plane(*_timeouts())
    error = _unconfirmed(first)
    assert "re-running the same command attaches" in str(error)
    assert "flash runs list" in str(error)
    rerun = _Plane(_CREATED)
    assert submit_run(rerun, _BODY) == _CREATED
    assert rerun.ids == [first.ids[0]]
    assert _records() == {}, "the confirmed run clears the record"


def test_process_killed_mid_request_leaves_the_id_for_the_rerun():
    first = _Plane(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        submit_run(first, _BODY)
    rerun = _Plane(_CREATED)
    submit_run(rerun, _BODY)
    assert rerun.ids == [first.ids[0]]


def test_confirmed_submit_lets_an_identical_resubmit_create_a_new_run():
    first, second = _Plane(_CREATED), _Plane(_CREATED)
    submit_run(first, _BODY)
    submit_run(second, _BODY)
    assert first.ids[0] != second.ids[0]
    assert _records() == {}


def test_different_bodies_never_share_an_id():
    _unconfirmed(first := _Plane(*_timeouts()))
    other = _Plane(_CREATED)
    submit_run(other, {"spec": {"model": "other", "project": _PROJECT_ID}})
    assert other.ids[0] != first.ids[0]


@pytest.mark.parametrize("plane", [_Plane(api_url="http://other.invalid"), _Plane(api_key="k2")])
def test_ids_are_scoped_to_the_plane_and_api_key(plane):
    _unconfirmed(first := _Plane(*_timeouts()))
    plane.outcomes = [_CREATED]
    submit_run(plane, _BODY)
    assert plane.ids[0] != first.ids[0]


@pytest.mark.parametrize("status", [400, 402, 410, 422])
def test_definitive_rejection_clears_the_record(status):
    _unconfirmed(first := _Plane(*_timeouts()))
    with pytest.raises(ApiError):
        submit_run(rejected := _Plane(ApiError(status, "no")), _BODY)
    assert rejected.ids == [first.ids[0]]
    assert _records() == {}
    submit_run(fresh := _Plane(_CREATED), _BODY)
    assert fresh.ids[0] != first.ids[0]


def test_server_error_keeps_the_record():
    with pytest.raises(ApiError):
        submit_run(first := _Plane(ApiError(500, "boom")), _BODY)
    submit_run(rerun := _Plane(_CREATED), _BODY)
    assert rerun.ids == [first.ids[0]]


@pytest.mark.parametrize("content", ["{not json", "[]", '{"pending": {"k": [{"x": 1}]}}'])
def test_corrupt_store_is_ignored(content):
    path = pending_submissions._path()
    path.parent.mkdir(parents=True)
    path.write_text(content)
    assert submit_run(plane := _Plane(_CREATED), _BODY) == _CREATED
    assert len(plane.ids[0]) >= 16


def test_unwritable_store_still_submits_without_promising_reattach(monkeypatch, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    monkeypatch.setattr(pending_submissions, "_path", lambda: blocker / "sub" / "pending.json")
    error = _unconfirmed(_Plane(*_timeouts()))
    assert "re-running the same command" not in str(error)
    assert "flash runs list" in str(error)


def test_rotated_runtime_secrets_still_reuse_the_id():
    _unconfirmed(first := _Plane(*_timeouts()), {**_BODY, "runtime_secrets": {"K": "old"}})
    rerun = _Plane(_CREATED)
    submit_run(rerun, {**_BODY, "runtime_secrets": {"K": "new"}})
    assert rerun.ids == [first.ids[0]]


def test_concurrent_identical_submit_never_borrows_a_live_process_id():
    key = pending_submissions.request_key(_Plane(), _BODY)
    held = {"submission_id": "held-by-a-running-train", "pid": os.getppid()}

    def hold(records, now):
        records[key] = [{**held, "created_at": now, "host": socket.gethostname()}]

    pending_submissions._update(hold)
    submit_run(plane := _Plane(_CREATED), _BODY)
    assert plane.ids[0] != "held-by-a-running-train"
    assert [e["submission_id"] for e in _records()[key]] == ["held-by-a-running-train"]


def test_expired_record_is_pruned_not_reused(monkeypatch):
    _unconfirmed(first := _Plane(*_timeouts()))
    monkeypatch.setattr(pending_submissions, "PENDING_TTL_SECONDS", -1.0)
    submit_run(rerun := _Plane(_CREATED), _BODY)
    assert rerun.ids[0] != first.ids[0]
    assert _records() == {}


def test_legacy_fallback_records_nothing(monkeypatch):
    client = ApiClient("http://plane.invalid", "fslo-user-test")
    plane = _Plane(ApiError(404, "Not Found", detail="Not Found"), _CREATED)
    monkeypatch.setattr(client, "_request", plane._request)
    assert client.create_run({"model": "m", "project": _PROJECT_ID}) == _CREATED
    assert _records() == {}
