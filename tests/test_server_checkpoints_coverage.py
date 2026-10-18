"""Hermetic coverage for checkpoint backend response and best-effort failure handling."""

from __future__ import annotations

import io
import urllib.request
from types import SimpleNamespace

import pytest

import flash.server.domain.registry.checkpoints as checkpoints


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.exited = True

    def read(self) -> bytes:
        return self.body


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b'{"ok": true, "count": 2}', {"ok": True, "count": 2}),
        (b"", {}),
        (b"not-json", {}),
    ],
)
def test_post_checkpoints_handles_valid_empty_and_malformed_json(
    monkeypatch, raw, expected
) -> None:
    """Backend response decoding must accept valid JSON and safely normalize empty or malformed bodies."""
    response = _Response(raw)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: response,
    )

    result = checkpoints._post_checkpoints(token="secret", body={"runId": "flash-1"})

    assert result == expected
    assert response.entered is True
    assert response.exited is True


def test_register_run_checkpoints_rejects_an_empty_batch() -> None:
    """Checkpoint persistence must reject empty batches before org or network inspection."""
    status = SimpleNamespace(run_id="flash-1", spec={}, billing_context={"org_id": "org"})

    with pytest.raises(ValueError, match="no checkpoints to record"):
        checkpoints.register_run_checkpoints(internal_key="key", status=status, checkpoints=[])


def test_best_effort_logs_and_skips_a_bad_persisted_spec(monkeypatch) -> None:
    """Corrupt persisted specs must remain visible while never disrupting completion or deployment."""
    status = SimpleNamespace(run_id="flash-1", spec={"algorithm": "not-valid"})
    log = io.StringIO()
    monkeypatch.setattr(checkpoints, "internal_key", lambda: "internal-key")

    assert checkpoints.register_checkpoints_best_effort(status, log=log) == 0
    assert "bad spec" in log.getvalue()
    assert "flash-1" in log.getvalue()
