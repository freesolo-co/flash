"""What `teardown_reconciled_remote` is allowed to treat as proof the captured worker is gone.

Reconciliation tears down a stalled attempt and then launches a replacement on the same run. That
replacement is only safe once the previous worker cannot still be running: two live workers on one
run double-bill and race each other's checkpoint writes. So the question this module pins is narrow
and load-bearing -- which teardown answers permit `_resume_after_confirmed_teardown` to run.

Teardown reports its conclusion as a `CleanupResult` value rather than by raising, and only
`DELETED` and `ABSENT` are confirmations. `PRESENT`, `RETRYABLE`, and `UNCONFIRMED` are the exact
cases where the resource may still be live, and they arrive as ordinary return values that a caller
reading "did this raise?" cannot distinguish from success.
"""

from __future__ import annotations

import pytest

from flash.providers.core.capabilities import CleanupOutcome, CleanupResult
from flash.runner.supervise import attach_reconcile

_RUN_ID = "reconcile-teardown-evidence"


def _remote(provider: str) -> dict:
    if provider == "runpod":
        return {
            "provider": "runpod",
            "endpoint_id": "endpoint-1",
            "job_id": "job-1",
            "key_fingerprint": "fp-1",
            "attempt": 0,
        }
    return {
        "provider": "vast",
        "instance_id": 101,
        "offer_id": 202,
        "machine_id": 303,
        "label": "flash-reconcile",
        "gpu": "RTX 4090",
        "hourly_usd": 0.5,
        "attempt": 0,
        "started_ts": 100.0,
    }


class _Handle:
    """The minimum a reconciled handle has to expose: which provider it belongs to."""

    def __init__(self, provider: str) -> None:
        self.provider = provider


def _arrange(monkeypatch, *, teardown, provably_gone, record=lambda *_a, **_k: True):
    """Bind the three collaborators teardown reconciliation consults.

    `teardown` is a callable so a test can make it raise as well as return, which is the exact
    distinction under test.
    """
    from flash.runner.accounting import reconciliation
    from flash.runner.supervise import lifecycle

    monkeypatch.setattr(lifecycle, "_strict_teardown_handle", teardown)
    monkeypatch.setattr(lifecycle, "_worker_provably_gone", lambda *_a, **_k: provably_gone)
    monkeypatch.setattr(reconciliation, "_record_cleanup_remote", record)


@pytest.mark.parametrize("provider", ["runpod", "vast"])
@pytest.mark.parametrize(
    "outcome",
    [CleanupOutcome.PRESENT, CleanupOutcome.RETRYABLE, CleanupOutcome.UNCONFIRMED],
)
def test_an_unconfirmed_teardown_does_not_release_recovery_on_its_own(
    monkeypatch, provider, outcome
):
    """Returning without raising is not evidence of absence.

    Teardown reports a surviving or unverified resource as a value. A caller that reads only
    "did this raise?" credits that value as success and lets a replacement attempt start while the
    captured instance or endpoint is still live.
    """
    _arrange(
        monkeypatch,
        teardown=lambda *_a, **_k: CleanupResult(outcome, **_evidence(outcome)),
        provably_gone=False,
    )

    may_continue, resource_deleted = attach_reconcile.teardown_reconciled_remote(
        _RUN_ID,
        _remote(provider),
        _Handle(provider),
        confirmed_teardown=False,
    )

    assert may_continue is False, "recovery must not resume over a possibly live worker"
    assert resource_deleted is False


def _evidence(outcome: CleanupOutcome) -> dict:
    """Each unconfirmed outcome carries the evidence its contract requires."""
    if outcome is CleanupOutcome.PRESENT:
        return {"surviving_ids": ("resource-1",)}
    if outcome is CleanupOutcome.UNCONFIRMED:
        return {"unresolved_ids": ("resource-1",)}
    return {}


@pytest.mark.parametrize("provider", ["runpod", "vast"])
@pytest.mark.parametrize("outcome", [CleanupOutcome.DELETED, CleanupOutcome.ABSENT])
def test_a_confirmed_teardown_releases_recovery(monkeypatch, provider, outcome):
    """Confirmed deletion or owner-authenticated absence is the whole point of the call.

    `_worker_provably_gone` is made to answer False so this cannot pass by accident: the release has
    to come from the teardown result itself, not from a second opinion.
    """
    _arrange(
        monkeypatch,
        teardown=lambda *_a, **_k: CleanupResult(
            outcome, confirmed_deleted_ids=("i-1",) if outcome is CleanupOutcome.DELETED else ()
        ),
        provably_gone=False,
    )

    assert attach_reconcile.teardown_reconciled_remote(
        _RUN_ID, _remote(provider), _Handle(provider), confirmed_teardown=False
    ) == (True, True)


@pytest.mark.parametrize("provider", ["runpod", "vast"])
def test_an_unconfirmed_teardown_still_releases_on_authoritative_absence(monkeypatch, provider):
    """An unconfirmed delete plus a proven-absent worker is safe to resume.

    The billable resource was not confirmed deleted, so `resource_deleted` stays false and the
    caller keeps the cleanup obligation; what absence buys is only the right to start a replacement.
    """
    recorded: list[dict] = []
    _arrange(
        monkeypatch,
        teardown=lambda *_a, **_k: CleanupResult(
            CleanupOutcome.UNCONFIRMED, unresolved_ids=("resource-1",)
        ),
        provably_gone=True,
        record=lambda _run_id, remote: recorded.append(remote) or True,
    )

    may_continue, resource_deleted = attach_reconcile.teardown_reconciled_remote(
        _RUN_ID, _remote(provider), _Handle(provider), confirmed_teardown=False
    )

    assert (may_continue, resource_deleted) == (True, False)
    if provider == "runpod":
        assert recorded == [_remote(provider)], "the undeleted endpoint stays owed a cleanup"


@pytest.mark.parametrize("provider", ["runpod", "vast"])
def test_a_raising_teardown_still_consults_the_absence_check(monkeypatch, provider):
    """Teardown may also fail by raising, and that path must keep its second opinion.

    This is the behavior that existed before teardown returned outcomes; losing it would make an
    exception permanently wedge reconciliation even for a worker that is provably gone.
    """
    _arrange(
        monkeypatch,
        teardown=_raise(RuntimeError("provider unreachable")),
        provably_gone=True,
    )

    assert attach_reconcile.teardown_reconciled_remote(
        _RUN_ID, _remote(provider), _Handle(provider), confirmed_teardown=False
    ) == (True, False)


@pytest.mark.parametrize("provider", ["runpod", "vast"])
def test_a_raising_teardown_without_absence_blocks_recovery(monkeypatch, provider):
    _arrange(
        monkeypatch,
        teardown=_raise(RuntimeError("provider unreachable")),
        provably_gone=False,
    )

    assert attach_reconcile.teardown_reconciled_remote(
        _RUN_ID, _remote(provider), _Handle(provider), confirmed_teardown=False
    ) == (False, False)


def _raise(exc: Exception):
    def _teardown(*_args, **_kwargs):
        raise exc

    return _teardown


def test_a_non_result_teardown_return_is_never_a_confirmation(monkeypatch):
    """Anything that is not a validated `CleanupResult` is unconfirmed.

    A `CleanupResult` has no boolean protocol, but arbitrary objects do -- a stale fake returning
    `True`, or any truthy stand-in, would otherwise read as confirmed deletion.
    """
    for stand_in in (True, 1, "deleted", object()):
        _arrange(
            monkeypatch,
            teardown=lambda *_a, _v=stand_in, **_k: _v,
            provably_gone=False,
        )

        assert attach_reconcile.teardown_reconciled_remote(
            _RUN_ID, _remote("vast"), _Handle("vast"), confirmed_teardown=False
        ) == (False, False)


@pytest.mark.parametrize("provider", ["runpod", "vast"])
def test_an_already_confirmed_teardown_short_circuits(monkeypatch, provider):
    """A teardown confirmed on an earlier pass is not repeated.

    The confirmation is persisted, so re-tearing down would issue a second provider delete for a
    resource already proven gone.
    """
    called: list[str] = []
    _arrange(
        monkeypatch,
        teardown=lambda *_a, **_k: called.append("teardown"),
        provably_gone=False,
    )

    assert attach_reconcile.teardown_reconciled_remote(
        _RUN_ID, _remote(provider), _Handle(provider), confirmed_teardown=True
    ) == (True, True)
    assert called == []


def test_an_unrecordable_runpod_cleanup_blocks_recovery(monkeypatch):
    """A RunPod endpoint that was not deleted must be written down before recovery continues.

    Clearing the active remote without persisting the surviving endpoint would drop the only record
    of a billing resource nothing will ever revisit.
    """
    _arrange(
        monkeypatch,
        teardown=lambda *_a, **_k: CleanupResult(
            CleanupOutcome.UNCONFIRMED, unresolved_ids=("endpoint-1",)
        ),
        provably_gone=True,
        record=lambda *_a, **_k: False,
    )

    assert attach_reconcile.teardown_reconciled_remote(
        _RUN_ID, _remote("runpod"), _Handle("runpod"), confirmed_teardown=False
    ) == (False, False)
