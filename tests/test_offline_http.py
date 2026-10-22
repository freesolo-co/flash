"""The offline guard must fail before a serving request opens a network connection."""

import asyncio

import httpcore
import httpx
import pytest

from tests import conftest as harness
from tests.serving_conformance import conftest as conformance


@pytest.fixture(autouse=True)
def no_network_even_if_guard_regresses(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("request escaped the offline guard")

    monkeypatch.setattr(httpcore.ConnectionPool, "handle_request", unexpected)
    monkeypatch.setattr(httpcore.AsyncConnectionPool, "handle_async_request", unexpected)


@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
@pytest.mark.parametrize("host", ["serve.freesolo.co", "serve-dev.freesolo.co"])
def test_unstubbed_serving_requests_fail_locally(method, host):
    with httpx.Client() as client, pytest.raises(pytest.fail.Exception, match="offline test"):
        client.request(method, f"https://{host}/adapters/r1%2Ffinal")


def test_unstubbed_async_requests_fail_locally():
    async def request():
        async with httpx.AsyncClient() as client:
            await client.get("https://serve.freesolo.co/adapters/r1%2Ffinal")

    with pytest.raises(pytest.fail.Exception, match="offline test"):
        asyncio.run(request())


def test_mock_transport_still_exercises_serving_requests():
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(404))) as client:
        assert client.get("https://serve.freesolo.co/adapters/r1%2Ffinal").status_code == 404


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:9000", "http://localhost:9000", "http://[::1]:9000"]
)
def test_loopback_and_preexisting_clients_remain_usable(monkeypatch, url):
    monkeypatch.setattr(
        httpx.HTTPTransport, "handle_request", lambda self, request: httpx.Response(200)
    )
    with httpx.Client() as client:
        harness._offline_http._get_wrapped_function()(monkeypatch, set())
        assert client.get(url).status_code == 200
        with pytest.raises(pytest.fail.Exception, match="offline test"):
            client.get("https://serve.freesolo.co/healthz")


def test_conformance_allows_only_selected_origin_and_blocks_redirects(monkeypatch):
    target = "https://conformance.example:8443"
    origins = conformance.allowed_http_origins._get_wrapped_function()(target)

    def send(self, request):
        if request.url.path == "/redirect":
            return httpx.Response(302, headers={"location": "https://serve.freesolo.co/healthz"})
        return httpx.Response(200)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", send)
    harness._offline_http._get_wrapped_function()(monkeypatch, origins)
    with httpx.Client(follow_redirects=True) as client:
        assert client.get(target + "/healthz").status_code == 200
        for url in (
            "https://conformance.example/healthz",
            "http://conformance.example:8443/healthz",
            target + "/redirect",
        ):
            with pytest.raises(pytest.fail.Exception, match="offline test"):
                client.get(url)
