"""Paid submits are idempotent: retries reuse one submission ID and never create a second run."""

from __future__ import annotations

import pytest

from flash.client import ApiClient, ApiError, RequestTimeoutError
from flash.client.http import ServiceUnreachableError
from flash.client.submissions import (
    MAX_CONSECUTIVE_TRANSPORT_FAILURES,
    new_submission_id,
    submit_run,
)

_PROJECT_ID = "11111111-1111-4111-8111-111111111111"
_CREATED = {"run_id": "flash-1-a", "state": "queued", "submission_id": "x"}


class _Plane:
    """Scripted control plane: each _request call pops the next outcome."""

    def __init__(self, *outcomes):
        self.timeout = 60.0
        self.request_options: list[dict] = []
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, str, dict | None]] = []

    def _request(self, method, path, body=None, **_kwargs):
        self.calls.append((method, path, body))
        self.request_options.append(_kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _in_progress() -> ApiError:
    return ApiError(409, "busy", detail="submission is in progress; retry the same ID")


def test_submission_ids_fit_the_plane_contract():
    ids = {new_submission_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(16 <= len(i) <= 128 and i.replace("-", "").replace("_", "").isalnum() for i in ids)


def test_timed_out_submit_reattaches_to_the_same_submission():
    plane = _Plane(
        RequestTimeoutError("timed out"),
        _in_progress(),
        _in_progress(),
        _CREATED,
    )
    waits: list[float] = []
    out = submit_run(plane, {"spec": {}}, sleep=waits.append)
    assert out == _CREATED
    assert len(plane.calls) == 4
    assert {method for method, _path, _body in plane.calls} == {"PUT"}
    assert len({path for _method, path, _body in plane.calls}) == 1, "one ID for every retry"
    assert plane.calls[0][1].startswith("/v1/run-submissions/")
    assert len(waits) == 3


@pytest.mark.parametrize("status", [502, 503, 504])
def test_gateway_errors_are_retried_with_the_same_id(status):
    plane = _Plane(ApiError(status, "gateway"), _CREATED)
    assert submit_run(plane, {}, sleep=lambda _s: None) == _CREATED
    assert plane.calls[0][1] == plane.calls[1][1]


@pytest.mark.parametrize(
    "error",
    [
        ApiError(400, "bad spec"),
        ApiError(402, "insufficient credits"),
        ApiError(409, "conflict", detail="submission ownership conflict"),
        ApiError(410, "cancelled", detail="submission was cancelled"),
        ApiError(422, "mismatch", detail="submission ID is already bound to a different request"),
    ],
)
def test_definitive_rejections_are_not_retried(error):
    plane = _Plane(error)
    with pytest.raises(ApiError) as excinfo:
        submit_run(plane, {}, sleep=lambda _s: pytest.fail("must not wait"))
    assert excinfo.value is error
    assert len(plane.calls) == 1


def test_unreachable_plane_gives_up_with_a_check_before_resubmitting_hint():
    plane = _Plane(*[ServiceUnreachableError("refused")] * (MAX_CONSECUTIVE_TRANSPORT_FAILURES + 1))
    with pytest.raises(RequestTimeoutError, match="flash runs list"):
        submit_run(plane, {}, sleep=lambda _s: None)
    assert len(plane.calls) == MAX_CONSECUTIVE_TRANSPORT_FAILURES + 1


def test_deadline_bounds_waiting_on_a_stuck_submission():
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    plane = _Plane(*[_in_progress()] * 100)
    with pytest.raises(RequestTimeoutError, match="may still create a run"):
        submit_run(plane, {}, deadline_seconds=35.0, sleep=sleep, clock=lambda: now[0])
    assert len(plane.calls) == 4  # t=0, 10, 20, 30; the next wait would pass the deadline


@pytest.mark.parametrize(("status", "detail"), [(404, "Not Found"), (405, "Method Not Allowed")])
def test_create_run_falls_back_to_post_on_a_plane_without_the_route(monkeypatch, status, detail):
    client = ApiClient("http://plane.invalid", "fslo-user-test")
    plane = _Plane(ApiError(status, detail, detail=detail), _CREATED)
    monkeypatch.setattr(client, "_request", plane._request)
    assert client.create_run({"model": "m", "project": _PROJECT_ID}) == _CREATED
    assert [method for method, _path, _body in plane.calls] == ["PUT", "POST"]
    assert plane.calls[1][1] == "/v1/runs"
    assert plane.calls[0][2] == plane.calls[1][2], "the fallback sends the identical body"


def test_unknown_404_from_the_route_is_not_mistaken_for_an_old_plane():
    plane = _Plane(ApiError(404, "unknown project", detail="unknown project"))
    with pytest.raises(ApiError):
        submit_run(plane, {}, sleep=lambda _s: None)


def test_dry_runs_stay_on_post(monkeypatch):
    client = ApiClient("http://plane.invalid", "fslo-user-test")
    plane = _Plane({"run_id": "flash-dry", "state": "dry_run"})
    monkeypatch.setattr(client, "_request", plane._request)
    client.create_run({"model": "m", "project": _PROJECT_ID}, dry_run=True)
    assert plane.calls == [
        ("POST", "/v1/runs", {"spec": {"model": "m", "project": _PROJECT_ID}, "dry_run": True})
    ]


@pytest.mark.parametrize(
    "first",
    [
        RequestTimeoutError("timed out"),
        ServiceUnreachableError("connection dropped"),
        ApiError(502, "gateway"),
        ApiError(503, "unavailable"),
        ApiError(504, "gateway timeout"),
        _in_progress(),
    ],
)
@pytest.mark.parametrize(("status", "detail"), [(404, "Not Found"), (405, "Method Not Allowed")])
def test_ambiguous_submission_never_falls_back_to_post(monkeypatch, first, status, detail):
    from flash.client import submissions

    client = ApiClient("http://plane.invalid", "fslo-user-test")
    plane = _Plane(first, ApiError(status, detail, detail=detail), _CREATED)
    monkeypatch.setattr(client, "_request", plane._request)
    monkeypatch.setattr(
        submissions,
        "submit_run",
        lambda client, body: submit_run(client, body, sleep=lambda _s: None),
    )
    with pytest.raises(RequestTimeoutError, match="may still create a run") as error:
        client.create_run({"model": "m", "project": _PROJECT_ID})
    assert [method for method, _, _ in plane.calls] == ["PUT", "PUT"]
    assert plane.calls[0][1] == plane.calls[1][1]
    assert plane.calls[0][1].rsplit("/", 1)[1] in str(error.value)


def test_requests_use_remaining_deadline_and_configured_timeout():
    now = [0.0]
    plane = _Plane(_in_progress(), _CREATED)
    plane.timeout = 25 * 60.0

    def sleep(seconds):
        now[0] += seconds

    assert submit_run(plane, {}, deadline_seconds=12, sleep=sleep, clock=lambda: now[0]) == _CREATED
    assert [(o["timeout"], o["body_deadline"]) for o in plane.request_options] == [(12, 12), (2, 2)]
    short = _Plane(_CREATED)
    short.timeout = 1
    submit_run(short, {}, deadline_seconds=12, clock=lambda: 0)
    assert short.request_options[0]["timeout"] == 1
    assert short.request_options[0]["body_deadline"] == 12


def test_no_request_starts_after_deadline_even_if_sleep_overshoots():
    now = [0.0]
    plane = _Plane(_in_progress(), _CREATED)

    def sleep(_seconds):
        now[0] = 20

    with pytest.raises(RequestTimeoutError, match="may still create a run"):
        submit_run(plane, {}, deadline_seconds=15, sleep=sleep, clock=lambda: now[0])
    assert len(plane.calls) == 1


def test_expired_deadline_does_not_send_a_request():
    plane = _Plane(_CREATED)
    with pytest.raises(RequestTimeoutError):
        submit_run(plane, {}, deadline_seconds=0)
    assert plane.calls == []
