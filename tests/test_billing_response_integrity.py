"""A malformed charge acknowledgement must remain retryable, never become charged."""

import http.client
import io
import json
from contextlib import closing

import pytest

from flash.runner.lifecycle import state, status
from flash.server.billing import charges, retry
from tests.test_billing_retry import _save_run


@pytest.mark.parametrize(
    "payload",
    [None, [], "ok", {}, {"amountCents": True}, {"amountCents": -1}, {"amountCents": "123"}],
)
def test_malformed_charge_response_remains_retryable(tmp_path, monkeypatch, payload):
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "test-internal-key")
    monkeypatch.delenv("FLASH_STANDALONE", raising=False)
    monkeypatch.setattr(state, "RUNS_DIR", str(tmp_path / "runs"))
    _save_run(tmp_path)
    responses = iter([payload, {"amountCents": 123, "replay": True}])
    monkeypatch.setattr(
        charges,
        "_urlopen_no_redirect",
        lambda *args, **kwargs: closing(io.BytesIO(json.dumps(next(responses)).encode())),
    )

    assert retry.retry_completion_charges_once() == 0
    first = status.get_status("run-1")
    assert first.billing_state == "failed"
    assert "invalid" in first.billing_error
    assert retry.retry_completion_charges_once() == 1
    assert status.get_status("run-1").billing_charge == {"amountCents": 123, "replay": True}


def test_truncated_charge_response_is_a_billing_error(monkeypatch):
    class TruncatedResponse(io.BytesIO):
        def read(self, *args):
            raise http.client.IncompleteRead(b'{"amount', 20)

    monkeypatch.setattr(
        charges, "_urlopen_no_redirect", lambda *a, **kw: closing(TruncatedResponse())
    )
    with pytest.raises(charges.BillingError, match="incomplete response") as exc:
        charges._post_billing(token="test-key", path="/test-charge", body={})
    assert exc.value.status_code == 502
