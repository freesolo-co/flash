"""Client recovery, private streaming exports, and editable CLI scaffolding."""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
from types import SimpleNamespace

import pytest

from flash.client.http import ApiError, ClientError, RequestTimeoutError
from flash.custom import TrainingClient
from flash.custom.artifacts import download_checkpoint


class LostReplyHTTP:
    def __init__(self):
        self.value = 0
        self.envelopes = []
        self.drop = True

    def _request(self, method, path, body=None, **kwargs):
        if method == "GET":
            return {"session_id": "session", "sequence": 0, "model_version": 0}
        self.envelopes.append(body)
        if len(self.envelopes) == 1:
            self.value += 1
        if self.drop:
            self.drop = False
            raise RequestTimeoutError("lost receipt")
        return {"session_id": "session", "sequence": body["sequence"], "model_version": self.value}


def test_pending_optimizer_request_freezes_intent_and_blocks_new_work(monkeypatch):
    http = LostReplyHTTP()
    monkeypatch.setattr("flash.custom.client.ApiClient", lambda *args, **kwargs: http)
    client = TrainingClient("https://gpu.example", api_key="session-token")
    with pytest.raises(RequestTimeoutError):
        client.optim_step(learning_rate=0.001)
    pending = client.pending_request
    pending["arguments"]["learning_rate"] = 1.0
    with pytest.raises(RuntimeError, match=r"retry_pending\(\)"):
        client.optim_step(learning_rate=0.5)
    client.retry_pending()
    assert http.envelopes[0] == http.envelopes[1]
    assert http.envelopes[1]["arguments"]["learning_rate"] == 0.001
    assert http.value == 1
    assert client.pending_request is None


@pytest.mark.parametrize(
    ("status", "detail", "remains_pending"),
    [
        (400, {"code": "custom_rejected"}, False),
        (409, {"code": "custom_rejected"}, False),
        (502, "proxy failure", True),
        (400, "unrecognized proxy response", True),
    ],
)
def test_only_known_preexecution_rejections_clear_pending(
    monkeypatch, status, detail, remains_pending
):
    http = LostReplyHTTP()
    original = http._request

    def reject(method, path, body=None, **kwargs):
        if method == "GET":
            return original(method, path, body, **kwargs)
        raise ApiError(status, "rejected", detail=detail)

    http._request = reject
    monkeypatch.setattr("flash.custom.client.ApiClient", lambda *args, **kwargs: http)
    client = TrainingClient("https://gpu.example", api_key="session-token")
    with pytest.raises(ApiError):
        client.optim_step(learning_rate=0.001)
    assert (client.pending_request is not None) is remains_pending


def test_checkpoint_download_streams_and_rejects_incomplete_file(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        @property
        def headers(self):
            return {"Content-Length": "8"}

    http = SimpleNamespace(
        api_url="https://gpu.example",
        _auth_headers=lambda: {"Authorization": "Bearer test"},
        _translate_http_errors=lambda **kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        "flash.custom.artifacts._urlopen_no_redirect", lambda *args, **kwargs: Response(b"complete")
    )
    path = download_checkpoint(http, "model", tmp_path / "model.zip")
    assert path.read_bytes() == b"complete"
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        download_checkpoint(http, "model", path)
    monkeypatch.setattr(
        "flash.custom.artifacts._urlopen_no_redirect", lambda *args, **kwargs: Response(b"short")
    )
    with pytest.raises(ClientError, match="incomplete"):
        download_checkpoint(http, "model", tmp_path / "short.zip")
    assert not (tmp_path / "short.zip").exists()
    assert list(tmp_path.glob(".flash-checkpoint-*")) == []


def test_scaffold_is_editable_and_never_overwrites(tmp_path):
    from flash.cli.parsing.main import _build_parser

    destination = tmp_path / "algorithm.py"
    args = _build_parser().parse_args(["custom", "init", str(destination)])
    args.func(args)
    assert "make_dpo_loss" in destination.read_text()
    compile(destination.read_text(), str(destination), "exec")
    with pytest.raises(FileExistsError):
        args.func(args)


def test_custom_api_imports_without_any_site_packages():
    result = subprocess.run(
        [sys.executable, "-S", "-c", "from flash.custom import Datum, TrainingClient"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
