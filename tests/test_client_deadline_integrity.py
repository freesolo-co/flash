"""Client retries share one deadline and verification timeouts remain expected errors."""

import urllib.error

import pytest

from flash.client import freesolo_api
from flash.client.http import ApiClient, ClientError, RequestTimeoutError


def test_cancel_confirmation_caps_each_poll_and_sleep_to_remaining_budget(monkeypatch):
    now = [0.0]
    polls = []
    sleeps = []
    client = ApiClient("https://plane.example", "test-key")
    monkeypatch.setattr("flash.client.http.time.monotonic", lambda: now[0])

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    def request(method, path, **kwargs):
        if method == "POST":
            raise RequestTimeoutError("cancel timed out")
        polls.append(kwargs)
        now[0] += min(59.5, kwargs.get("body_deadline", float("inf")))
        return {"state": "running"}

    monkeypatch.setattr("flash.client.http.time.sleep", sleep)
    monkeypatch.setattr(client, "_request", request)
    with pytest.raises(ClientError, match="cancel request timed out before confirmation"):
        client.cancel_run("run-1")

    assert polls == [
        {"timeout": 60.0, "body_deadline": 120.0},
        {"timeout": 58.5, "body_deadline": 58.5},
    ]
    assert sleeps == [2.0]
    assert now[0] == 120.0


@pytest.mark.parametrize("wrapped", [False, True])
def test_key_verification_translates_socket_timeout(monkeypatch, wrapped):
    def timeout(*_args, **_kwargs):
        error = TimeoutError("socket timed out")
        raise urllib.error.URLError(error) if wrapped else error

    monkeypatch.setattr(freesolo_api, "_urlopen_no_redirect", timeout)
    with pytest.raises(RequestTimeoutError, match="timed out"):
        freesolo_api.verify_freesolo_key("test-key", "https://auth.example")
