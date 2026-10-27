"""Platform uncertainty must not release capacity or authorize retirement."""

import subprocess

import httpx
import modal
import pytest

from flash.serving.deepinfer.app.endpoints import validate_endpoint
from flash.serving.deepinfer.app.fleet_ops import FleetOps
from flash.serving.deepinfer.app.replicas import replica_app


@pytest.fixture
def ops(monkeypatch):
    monkeypatch.setenv("DEEPINFER_PROXY_KEY", "key")
    monkeypatch.setenv("DEEPINFER_PROXY_SECRET", "secret")
    return FleetOps()


@pytest.mark.parametrize(
    "sample",
    [
        {"active_http_requests": 0},
        {"active_http_requests": -1, "draining": True},
        {"active_http_requests": False, "draining": True},
        {"active_http_requests": 0, "draining": False},
    ],
)
def test_ambiguous_drain_acknowledgment_never_authorizes_stop(ops, monkeypatch, sample):
    response = httpx.Response(
        200, json=sample, request=httpx.Request("POST", "https://one.modal.direct")
    )
    monkeypatch.setattr(httpx, "request", lambda *args, **kwargs: response)
    assert ops.drain({"url": "https://one.modal.direct"}) is None


def test_stopped_app_retains_slot_while_any_actual_container_remains(ops, monkeypatch):
    entry = {"name": "freesolo-deepseek-replica-one", "app_id": "ap-one"}
    monkeypatch.setattr(ops, "_cli", lambda *args: '[{"app_id":"ap-one"}]')
    assert not ops.stopped(entry)
    monkeypatch.setattr(ops, "_cli", lambda *args: "[]")
    assert ops.stopped(entry)


def test_repeated_stop_handles_the_modal_cli_already_stopped_exit(ops, monkeypatch):
    def already_stopped(*args):
        raise subprocess.CalledProcessError(1, args, output="", stderr="App is already stopped.")

    monkeypatch.setattr(ops, "_cli", already_stopped)
    ops.stop({"name": "freesolo-deepseek-replica-one", "app_id": "ap-one", "phase": "stopping"})


@pytest.mark.parametrize(
    "url",
    [
        "http://one.modal.direct",
        "https://one.modal.direct.evil.test",
        "https://secret@one.modal.direct",
        "https://one.modal.direct/redirect",
        "https://one.modal.direct?target=evil",
        "https://one.modal.direct:8000",
    ],
)
def test_directory_cannot_forward_proxy_credentials_to_untrusted_origins(url):
    with pytest.raises(ValueError, match="Invalid private Modal"):
        validate_endpoint(url)


def test_replica_keeps_its_pinned_image_source_and_single_container_budget(monkeypatch):
    settings = {}
    original = modal.App.server

    def capture(app, **kwargs):
        settings.update(kwargs)
        return original(app, **kwargs)

    monkeypatch.setattr(modal.App, "server", capture)
    replica_app("fleet-config-test", modal.Image.debian_slim(), "version")
    assert settings["include_source"] is False
    assert settings["min_containers"] == settings["max_containers"] == 1
    assert settings["gpu"] == "B200:4"
    assert settings["target_concurrency"] is None
