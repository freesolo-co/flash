"""connect() must survive transient control-plane failures while a custom run attaches."""

import pytest

from flash.client.http import (
    ApiClient,
    ApiError,
    ClientError,
    RequestTimeoutError,
    ServiceUnreachableError,
)
from flash.custom import controller


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeApi:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def _request(self, method, path):
        assert (method, path) == ("GET", "/v1/runs/run-1/custom")
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeHttp:
    timeout = None


class FakeTrainingClient:
    def __init__(self, url, *, api_key, timeout):
        self.url, self.api_key = url, api_key
        self._http = FakeHttp()


CONNECTION = {"url": "https://tunnel.example", "api_key": "session-token"}


@pytest.fixture
def clock(monkeypatch):
    monkeypatch.setattr(controller, "TrainingClient", FakeTrainingClient)
    return FakeClock()


def attach(api, clock, timeout=600):
    return controller.connect(api, "run-1", timeout=timeout, sleep=clock.sleep, clock=clock)


def test_provisioning_then_ready(clock):
    api = FakeApi([ApiError(425, "custom GPU is provisioning"), CONNECTION])
    client = attach(api, clock)
    assert (client.url, client.run_id, client._http.timeout) == (
        "https://tunnel.example",
        "run-1",
        300,
    )
    assert clock.sleeps == [2.0]


@pytest.mark.parametrize(
    "transient",
    [
        ApiError(500, "Internal Server Error"),
        ApiError(502, "Bad Gateway"),
        ApiError(503, "Service Unavailable"),
        ApiError(504, "Gateway Timeout"),
        RequestTimeoutError("request to the Flash service timed out"),
        ServiceUnreachableError("connection reset"),
    ],
)
def test_transient_control_plane_failures_are_retried(clock, transient):
    # a 500 from the tunnel lookup used to raise out of connect, and run_custom then
    # cancelled a healthy run.
    api = FakeApi([transient, transient, CONNECTION])
    client = attach(api, clock)
    assert client.run_id == "run-1"
    assert api.calls == 3
    assert clock.sleeps == [4.0, 8.0]  # capped exponential backoff, not the 2 s provisioning poll


def test_backoff_is_capped(clock):
    api = FakeApi([ApiError(500, "boom")] * 8 + [CONNECTION])
    attach(api, clock, timeout=3600)
    assert max(clock.sleeps) == controller._ATTACH_RETRY_MAX_SECONDS


@pytest.mark.parametrize("body", [b"<html>wrong service</html>", b"[]", b"null"])
def test_malformed_control_plane_response_fails_without_retry(clock, body):
    decoder = ApiClient("https://plane.example", "test-key")
    with pytest.raises(ClientError) as decoded:
        decoder._decode_response("/v1/runs/run-1/custom", body)
    api = FakeApi([decoded.value, CONNECTION])

    with pytest.raises(ClientError) as caught:
        attach(api, clock)

    assert caught.value is decoded.value
    assert api.calls == 1
    assert clock.sleeps == []


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409])
def test_definitive_client_errors_stay_fatal(clock, status):
    api = FakeApi([ApiError(status, "no")])
    with pytest.raises(ApiError) as caught:
        attach(api, clock)
    assert caught.value.status == status
    assert api.calls == 1


def test_persistent_failure_times_out_with_the_last_error(clock):
    api = FakeApi([ApiError(500, "Internal Server Error")] * 100)
    with pytest.raises(TimeoutError, match="Internal Server Error"):
        attach(api, clock, timeout=120)
    assert clock.now <= 120


def test_tunnel_not_ready_keeps_polling(clock, monkeypatch):
    attempts = []

    class SlowTunnel(FakeTrainingClient):
        def __init__(self, url, *, api_key, timeout):
            attempts.append(url)
            if len(attempts) == 1:
                raise ClientError("tunnel not serving yet")
            super().__init__(url, api_key=api_key, timeout=timeout)

    monkeypatch.setattr(controller, "TrainingClient", SlowTunnel)
    api = FakeApi([CONNECTION, CONNECTION])
    assert attach(api, clock).run_id == "run-1"
    assert clock.sleeps == [2.0]
