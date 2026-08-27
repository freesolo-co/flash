"""ApiClient: auth headers, error mapping, log paging (stdlib stub server, CPU-only)."""

from __future__ import annotations

import contextlib
import io
import json
import socket
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from flash.client import ApiClient, ApiError, ClientError, RequestTimeoutError
from flash.client.http import _parse_chat_target, _prepare_chat_request
from flash.client.specs import spec_payload
from flash.client.streaming import _cap_socket_timeout, _read_capped_response
from flash.schema import spec_from_dict

_PROJECT_ID = "11111111-1111-4111-8111-111111111111"


@pytest.fixture
def stub():
    seen: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, code: int, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", "application/gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            seen["auth"] = self.headers.get("Authorization")
            seen["path"] = self.path
            if self.path.startswith("/v1/envs/") and self.path.endswith("/package"):
                self._send_bytes(200, b"package-bytes")
            elif self.path == "/v1/health":
                seen["health_calls"] = seen.get("health_calls", 0) + 1
                capabilities = [] if seen.get("old_chat_server") else ["chat_step_selector_v1"]
                self._send(200, {"ok": True, "capabilities": capabilities})
            elif self.path == "/v1/runs/old-api/worker":
                self._send(404, {"detail": "Not Found"})
            elif self.path == "/v1/runs/proxy-old-api/worker":
                self.send_response(404)
                self.end_headers()
            elif self.path.startswith("/v1/runs/authfail"):
                self._send(401, {"detail": "invalid or missing API key"})
            elif self.path.startswith("/v1/runs/missing"):
                self._send(404, {"detail": "unknown run_id: missing"})
            elif self.path.startswith("/v1/runs/r1/logs"):
                self._send(200, {"run_id": "r1", "logs": "hi\n", "offset": 3, "state": "running"})
            else:
                self._send(200, {"runs": []})

        def do_POST(self):
            seen["auth"] = self.headers.get("Authorization")
            seen["path"] = self.path
            n = int(self.headers.get("Content-Length") or 0)
            seen["body"] = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/v1/envs":
                self._send(200, {"id": "freesolo-co/my-project/e"})
                return
            if self.path == "/v1/runs/json-chat/chat":
                self._send(200, {"choices": [{"message": {"content": "json reply"}}]})
                return
            if self.path == "/v1/runs" and seen["body"].get("spec", {}).get("model") == "rejected":
                self._send(
                    400,
                    {
                        "detail": {
                            "code": "packaged_dataset_unavailable",
                            "path": "dataset/train.jsonl",
                            "retryable": False,
                        }
                    },
                )
                return
            if (
                self.path in {"/v1/runs/r1/chat", "/v1/runs/run-a/chat"}
                and seen["body"].get("stream") is True
            ):
                body = "héllo".encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                for byte in body:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                return
            self._send(200, {"run_id": "r1", "state": "queued"})

        def do_DELETE(self):
            seen["auth"] = self.headers.get("Authorization")
            seen["project_id"] = self.headers.get("X-Freesolo-Project-Id")
            seen["path"] = self.path
            seen["method"] = "DELETE"
            if self.path.startswith("/v1/envs/"):
                slug = self.path[len("/v1/envs/") :]
                self._send(200, {"id": slug, "deleted": True})
                return
            self._send(200, {})

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", seen
    finally:
        server.shutdown()
        server.server_close()


def test_bearer_header_and_payload(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    out = client.create_run({"model": "m", "project": f" {_PROJECT_ID.upper()} "})
    assert out["run_id"] == "r1"
    assert seen["auth"] == "Bearer fslo-user-test"
    assert seen["body"] == {"spec": {"model": "m", "project": _PROJECT_ID}}


def test_create_run_sends_runtime_secrets_outside_spec(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    client.create_run(
        {"model": "m", "project": _PROJECT_ID},
        runtime_secrets={"WANDB_API_KEY": "wb-user"},
    )
    assert seen["body"] == {
        "spec": {"model": "m", "project": _PROJECT_ID},
        "runtime_secrets": {"WANDB_API_KEY": "wb-user"},
    }


def test_create_run_dry_run_flag_travels_in_body(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    client.create_run({"model": "m", "project": _PROJECT_ID}, dry_run=True)
    assert seen["body"] == {
        "spec": {"model": "m", "project": _PROJECT_ID},
        "dry_run": True,
    }
    # default omits dry_run, so live submissions keep the same validated spec payload.
    client.create_run({"model": "m", "project": _PROJECT_ID})
    assert seen["body"] == {"spec": {"model": "m", "project": _PROJECT_ID}}


@pytest.mark.parametrize("project", [None, "", "   ", "not-a-uuid", 7])
def test_create_run_rejects_missing_or_invalid_project_before_request(stub, project) -> None:
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    spec = {"model": "m"}
    if project is not None:
        spec["project"] = project

    with pytest.raises(ClientError, match="project"):
        client.create_run(spec)

    assert "body" not in seen


def test_spec_payload_filters_normalized_train_values_by_authored_keys() -> None:
    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "project": "11111111-1111-4111-8111-111111111111",
            "algorithm": "opd",
            "environment": {"id": "owner/project/env"},
            "train": {
                "epochs": 1,
                "max_examples": 1,
                "temperature": 0,
                "stop_sequences": [],
                "teacher_model": "GLM 5.2",
                "structured_outputs": False,
            },
        }
    )
    authored = {
        "epochs",
        "max_examples",
        "temperature",
        "stop_sequences",
        "teacher_model",
        "structured_outputs",
    }

    full = spec_payload(spec)
    sparse = spec_payload(spec, authored_train_keys=authored)

    assert set(full["train"]) > authored
    assert sparse["train"] == {
        "epochs": 1,
        "max_examples": 1,
        "temperature": 0.0,
        "stop_sequences": (),
        "teacher_model": "glm-5.2",
        "structured_outputs": "",
    }
    assert "lora_rank" not in sparse["train"]
    assert (
        spec_payload(spec, authored_train_keys=authored | {"lora_rank"})["train"]["lora_rank"] == 32
    )
    assert spec_payload(spec, authored_train_keys=set())["train"] == {}


def test_create_run_sends_schema_metadata_for_dry_run_and_live_submit(stub) -> None:
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    spec = {"project": "11111111-1111-4111-8111-111111111111", "model": "m", "train": {"epochs": 1}}
    metadata = {
        "version": "0.2.56",
        "fields": {"epochs": "0.2.0"},
        "authored_keys": ["epochs"],
    }

    client.create_run(spec, dry_run=True, client_train_schema=metadata)
    assert seen["body"] == {
        "spec": spec,
        "dry_run": True,
        "client_train_schema": metadata,
    }

    client.create_run(spec, client_train_schema=metadata)
    assert seen["body"] == {"spec": spec, "client_train_schema": metadata}


def test_api_error_carries_server_detail(stub):
    url, _ = stub
    client = ApiClient(url, "fslo-user-test")
    with pytest.raises(ApiError) as excinfo:
        client.get_run("missing")
    assert excinfo.value.status == 404
    assert "unknown run_id: missing" in str(excinfo.value)


def test_api_error_preserves_a_structured_detail_over_the_wire(stub):
    """structured server detail arrives as a dict rather than its repr."""
    url, _ = stub
    client = ApiClient(url, "fslo-user-test")

    with pytest.raises(ApiError) as excinfo:
        client.create_run({"model": "rejected", "project": _PROJECT_ID})

    exc = excinfo.value
    assert exc.status == 400
    assert exc.detail == {
        "code": "packaged_dataset_unavailable",
        "path": "dataset/train.jsonl",
        "retryable": False,
    }
    assert exc.code == "packaged_dataset_unavailable"
    assert exc.detail["retryable"] is False
    assert "packaged_dataset_unavailable" in str(exc)


def test_api_error_mentions_env_override(stub):
    url, _ = stub
    client = ApiClient(url, "fslo-user-test", key_source="FREESOLO_API_KEY")
    with pytest.raises(ApiError) as excinfo:
        client.get_run("authfail")
    assert excinfo.value.status == 401
    assert "invalid or missing API key" in str(excinfo.value)
    assert "FREESOLO_API_KEY is set and overrides" in str(excinfo.value)


def test_logs_offset_in_query(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    page = client.get_logs("r1", offset=3)
    assert page["offset"] == 3
    assert page["logs"] == "hi\n"
    assert seen["path"].endswith("/v1/runs/r1/logs?offset=3")


def test_get_worker_output_tolerates_missing_optional_route(stub):
    url, _ = stub
    client = ApiClient(url, "fslo-user-test")
    assert client.get_worker_output("old-api") == {}
    assert client.get_worker_output("proxy-old-api") == {}


def test_get_worker_output_preserves_unknown_run_404(stub):
    url, _ = stub
    client = ApiClient(url, "fslo-user-test")
    with pytest.raises(ApiError) as excinfo:
        client.get_worker_output("missing")
    assert excinfo.value.status == 404
    assert "unknown run_id: missing" in str(excinfo.value)


def test_parse_chat_target_requires_permanent_checkpoint_identity() -> None:
    assert _parse_chat_target("run-abc/final") == ("run-abc", "run-abc/final")
    assert _parse_chat_target("run-abc/step-5") == ("run-abc", "run-abc/step-5")

    for target in ("run-abc", "run-abc@step-5." + "a" * 40):
        with pytest.raises(ClientError, match="invalid checkpoint id"):
            _parse_chat_target(target)


def test_prepare_chat_request_sends_exact_checkpoint_id() -> None:
    base_run_id, body = _prepare_chat_request(
        "run-abc/step-5",
        [{"role": "user", "content": "hi"}],
        0.0,
        32,
    )

    assert base_run_id == "run-abc"
    assert body["checkpoint_id"] == "run-abc/step-5"
    assert "step" not in body
    assert "adapter_revision" not in body


def test_prepare_chat_request_defaults_tool_controls() -> None:
    base_run_id, body = _prepare_chat_request(
        "run-abc/final",
        [{"role": "user", "content": "weather"}],
        0.0,
        32,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            }
        ],
    )

    assert base_run_id == "run-abc"
    assert body["tool_choice"] == "auto"
    assert body["parallel_tool_calls"] is True


@pytest.mark.parametrize(
    "controls",
    [
        {"tool_choice": "none"},
        {"parallel_tool_calls": True},
    ],
    ids=["tool-choice", "parallel-tool-calls"],
)
def test_prepare_chat_request_rejects_tool_controls_without_tools(controls) -> None:
    with pytest.raises(ClientError, match="tool controls require tools"):
        _prepare_chat_request(
            "run-abc/final",
            [{"role": "user", "content": "weather"}],
            0.0,
            32,
            **controls,
        )


def test_chat_omits_thinking_template_controls(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    client.chat("json-chat/final", messages=[{"role": "user", "content": "hi"}])
    assert seen["body"] == {
        "checkpoint_id": "json-chat/final",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
        "max_tokens": 512,
    }


def test_chat_sends_user_supplied_system_prompt(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")

    client.chat(
        "json-chat/final",
        messages=[
            {"role": "system", "content": "stay terse"},
            {"role": "user", "content": "hi"},
        ],
    )

    assert seen["body"] == {
        "checkpoint_id": "json-chat/final",
        "messages": [
            {"role": "system", "content": "stay terse"},
            {"role": "user", "content": "hi"},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }


@pytest.mark.parametrize("checkpoint_id", ["run-a/final", "run-a/step-40"])
def test_chat_forwards_exact_checkpoint_id(stub, checkpoint_id):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")

    client.chat(checkpoint_id, [{"role": "user", "content": "hi"}])

    assert seen["path"] == "/v1/runs/run-a/chat"
    assert seen["body"]["checkpoint_id"] == checkpoint_id
    assert "step" not in seen["body"]
    assert "adapter_revision" not in seen["body"]


@pytest.mark.parametrize("target", ["run-a", "run-a/step-00", "run-a@step-1." + "a" * 40])
def test_chat_rejects_noncanonical_checkpoint_identity(stub, target):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")

    with pytest.raises(ClientError, match="invalid checkpoint id"):
        client.chat(target, [{"role": "user", "content": "hi"}])

    assert seen == {}


def test_chat_stream_sends_exact_checkpoint_and_yields_text(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    chunks = list(
        client.chat_stream(
            "r1/final",
            [{"role": "user", "content": "hi"}],
            temperature=0.2,
            max_tokens=7,
        )
    )
    assert "".join(chunks) == "héllo"
    assert seen["path"] == "/v1/runs/r1/chat"
    assert seen["auth"] == "Bearer fslo-user-test"
    assert seen["body"] == {
        "checkpoint_id": "r1/final",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2,
        "max_tokens": 7,
        "stream": True,
    }


def test_chat_stream_step_checkpoint_is_forwarded_unchanged(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")

    chunks = list(client.chat_stream("run-a/step-40", [{"role": "user", "content": "hi"}]))

    assert "".join(chunks) == "héllo"
    assert seen["path"] == "/v1/runs/run-a/chat"
    assert seen["body"]["checkpoint_id"] == "run-a/step-40"
    assert "step" not in seen["body"]
    assert "adapter_revision" not in seen["body"]


def test_chat_stream_accepts_json_fallback(stub):
    url, _ = stub
    client = ApiClient(url, "fslo-user-test")
    chunks = list(client.chat_stream("json-chat/final", [{"role": "user", "content": "hi"}]))
    assert chunks == ["json reply"]


def test_publish_env_plain_without_progress(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    out = client.publish_env(
        name="e", package_b64="QQ==", project_id="11111111-1111-4111-8111-111111111111"
    )
    assert out["id"] == "freesolo-co/my-project/e"
    assert seen["path"] == "/v1/envs"
    assert seen["body"] == {
        "name": "e",
        "package_b64": "QQ==",
        "project_id": "11111111-1111-4111-8111-111111111111",
    }


def test_publish_env_sends_project_id_when_given(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    out = client.publish_env(
        name="e", package_b64="QQ==", project_id="11111111-1111-4111-8111-111111111111"
    )
    assert out["id"] == "freesolo-co/my-project/e"
    assert seen["body"] == {
        "name": "e",
        "package_b64": "QQ==",
        "project_id": "11111111-1111-4111-8111-111111111111",
    }


def test_publish_env_rejects_blank_project_id_before_request(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    with pytest.raises(ClientError, match="project id is required"):
        client.publish_env(name="e", package_b64="QQ==", project_id="   ")
    assert "body" not in seen


def test_delete_env_sends_delete_to_slug_path(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    out = client.delete_env(
        "acme/checkout-bot/my-env", project_id="11111111-1111-4111-8111-111111111111"
    )
    assert out == {"id": "acme/checkout-bot/my-env", "deleted": True}
    assert seen["method"] == "DELETE"
    # the namespace/project/name slug (with its slashes) goes straight into the path
    assert seen["path"] == "/v1/envs/acme/checkout-bot/my-env"
    assert seen["auth"] == "Bearer fslo-user-test"
    assert seen["project_id"] == "11111111-1111-4111-8111-111111111111"


def test_delete_env_rejects_blank_project_before_request(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    with pytest.raises(ClientError, match="project id is required"):
        client.delete_env("acme/checkout-bot/my-env", project_id="   ")
    assert "method" not in seen


def test_delete_env_percent_encodes_reserved_chars(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    # A programmatic caller passing reserved characters must NOT be able to truncate the request
    # target: `?` becomes %3F (not a query string), `#` becomes %23 (not a dropped fragment), while
    # the namespace/project/name separators `/` are preserved so the server still routes the :path param.
    client.delete_env(
        "team/project/env?x=1#frag", project_id="11111111-1111-4111-8111-111111111111"
    )
    assert seen["method"] == "DELETE"
    assert seen["path"] == "/v1/envs/team/project/env%3Fx%3D1%23frag"


def test_download_env_package_uses_flash_control_plane(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")

    data = client.download_env_package("acme/checkout-bot/my-env")

    assert data == b"package-bytes"
    assert seen["path"] == "/v1/envs/acme/checkout-bot/my-env/package"
    assert seen["auth"] == "Bearer fslo-user-test"


def test_download_env_package_percent_encodes_reserved_chars(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")

    client.download_env_package("team/project/env?x=1#frag")

    assert seen["path"] == "/v1/envs/team/project/env%3Fx%3D1%23frag/package"


def test_download_env_package_caps_response_body(stub, monkeypatch):
    from flash.envs.loading import loader as adapter

    url, _seen = stub
    monkeypatch.setattr(adapter, "_MAX_ARCHIVE_BYTES", 5)
    client = ApiClient(url, "fslo-user-test")

    with pytest.raises(ClientError, match="maximum allowed size"):
        client.download_env_package("acme/checkout-bot/my-env")


def test_publish_env_streams_body_and_reports_progress(stub, monkeypatch):
    import flash.client.http as http_mod

    url, seen = stub
    client = ApiClient(url, "fslo-user-test")

    # spy on the streaming reader so we prove the streaming _request(progress=...) path (not the
    # plain one-shot path) ran — a refactor that faked progress around a one-shot send must fail.
    wrapped: list[int] = []
    real_reader = http_mod._ProgressReader

    class _SpyReader(real_reader):
        def __init__(self, data, progress):
            wrapped.append(len(data))
            super().__init__(data, progress)

    monkeypatch.setattr(http_mod, "_ProgressReader", _SpyReader)

    # a payload large enough to span several 8192-byte http.client send chunks, so the
    # callback fires repeatedly with a growing count instead of one all-at-once call.
    big = "A" * 30000
    body = {"name": "e", "package_b64": big, "project_id": "11111111-1111-4111-8111-111111111111"}
    calls: list[tuple[int, int]] = []
    out = client.publish_env(
        name="e",
        package_b64=big,
        project_id="11111111-1111-4111-8111-111111111111",
        progress=lambda sent, total: calls.append((sent, total)),
    )
    assert out["id"] == "freesolo-co/my-project/e"
    # the server reads exactly Content-Length bytes, so a correct multi-chunk stream
    # round-trips the full 30 KB body byte-for-byte across the chunk boundaries.
    assert seen["body"] == body
    expected_total = len(json.dumps(body).encode())
    assert wrapped == [expected_total]  # the streaming reader wrapped the full payload
    assert len(calls) > 1  # multiple chunks => multiple progress updates
    assert all(sent <= total for sent, total in calls)
    assert calls[0][0] < calls[-1][0]  # the byte count grew across chunks
    assert calls[-1] == (expected_total, expected_total)  # reached 100% of the real payload


def test_publish_env_progress_errors_do_not_abort_upload(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")

    def boom(sent, total):
        raise RuntimeError("render failed")

    # a raising progress widget must never abort an in-flight upload (contextlib.suppress).
    out = client.publish_env(
        name="e",
        package_b64="QQ==",
        project_id="11111111-1111-4111-8111-111111111111",
        progress=boom,
    )
    assert out["id"] == "freesolo-co/my-project/e"
    assert seen["body"] == {
        "name": "e",
        "package_b64": "QQ==",
        "project_id": "11111111-1111-4111-8111-111111111111",
    }


def test_unreachable_server_is_actionable():
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    with pytest.raises(ClientError, match="FLASH_API_URL"):
        client.health()


def test_raw_read_timeout_maps_to_client_error(monkeypatch):
    def timeout(req, timeout=None):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(urllib.request, "urlopen", timeout)

    client = ApiClient("http://flash.example", "fslo-user-test", timeout=2)
    with pytest.raises(RequestTimeoutError, match="timed out"):
        client.health()


def test_cancel_timeout_returns_authoritative_cancelled_status(monkeypatch):
    client = ApiClient("http://flash.example", "fslo-user-test")
    calls: list[tuple[str, str, float | None]] = []

    def request(method, path, body=None, timeout=None, progress=None, require=()):
        calls.append((method, path, timeout))
        if method == "POST":
            raise RequestTimeoutError("cancel timed out")
        if method == "GET" and path == "/v1/runs/r1":
            return {"run_id": "r1", "state": "cancelled", "remote": {"gpu": "B200"}}
        raise AssertionError((method, path))

    monkeypatch.setattr(client, "_request", request)

    out = client.cancel_run("r1")

    assert out["state"] == "cancelled"
    assert calls == [
        ("POST", "/v1/runs/r1/cancel", 60.0),
        ("GET", "/v1/runs/r1", None),
    ]


@pytest.mark.parametrize("run_state", ["cancelled", "done", "failed", "dry_run"])
def test_cancel_timeout_raises_when_backend_revocation_is_unconfirmed(monkeypatch, run_state):
    client = ApiClient("http://flash.example", "fslo-user-test")

    def request(method, path, body=None, timeout=None, progress=None, require=()):
        if method == "POST":
            raise RequestTimeoutError("cancel timed out")
        if method == "GET" and path == "/v1/runs/r1":
            return {
                "run_id": "r1",
                "state": run_state,
                "deployment": {
                    "state": "revocation_failed",
                    "retryable": True,
                    "error": "backend unavailable",
                },
            }
        raise AssertionError((method, path))

    monkeypatch.setattr(client, "_request", request)

    with pytest.raises(
        ClientError,
        match="backend revocation is unconfirmed: backend unavailable; retry cancellation",
    ):
        client.cancel_run("r1")


def test_cancel_timeout_keeps_polling_nonterminal_revocation_failure(monkeypatch):
    client = ApiClient("http://flash.example", "fslo-user-test")
    polls = iter(
        [
            {
                "run_id": "r1",
                "state": "running",
                "deployment": {"state": "revocation_failed", "retryable": True},
            },
            {"run_id": "r1", "state": "cancelled", "deployment": {"state": "undeployed"}},
        ]
    )

    def request(method, path, body=None, timeout=None, progress=None, require=()):
        if method == "POST":
            raise RequestTimeoutError("cancel timed out")
        if method == "GET" and path == "/v1/runs/r1":
            return next(polls)
        raise AssertionError((method, path))

    monkeypatch.setattr(client, "_request", request)
    monkeypatch.setattr("flash.client.http.time.sleep", lambda _seconds: None)

    assert client.cancel_run("r1")["state"] == "cancelled"


def test_deploy_rejects_malformed_checkpoint_ref():
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    for bad in ("flash-run", "flash-run/step-", "flash-run/checkpoints/step-4"):
        with pytest.raises(ClientError, match="invalid checkpoint id"):
            client.deploy(bad)


def test_deploy_posts_exact_checkpoint_id(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    client.deploy("flash-run/step-40")
    assert seen["path"] == "/v1/runs/flash-run/deploy"
    assert seen["body"] == {
        "dry_run": False,
        "checkpoint_id": "flash-run/step-40",
    }
    # smoke verification is mandatory server-side; the client sends no opt-out knob
    assert "verify" not in seen["body"]


def test_deploy_final_ref_posts_exact_checkpoint_id(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    client.deploy("flash-run/final")
    assert seen["path"] == "/v1/runs/flash-run/deploy"
    assert seen["body"] == {
        "dry_run": False,
        "checkpoint_id": "flash-run/final",
    }


def test_export_sends_repository_token_and_checkpoint_id(stub):
    """`flash export` posts the destination repo, token, and exact checkpoint id."""
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    client.export("r1/step-40", repository="me/adapters", hf_token="hf_secret", private=False)
    assert seen["path"] == "/v1/runs/r1/export"
    assert seen["auth"] == "Bearer fslo-user-test"
    assert seen["body"] == {
        "repository": "me/adapters",
        "hf_token": "hf_secret",
        "private": False,
        "checkpoint_id": "r1/step-40",
    }


def test_export_final_checkpoint_defaults_private(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    client.export("r1/final", repository="me/adapters", hf_token="hf_secret")
    assert seen["body"] == {
        "repository": "me/adapters",
        "hf_token": "hf_secret",
        "private": True,
        "checkpoint_id": "r1/final",
    }


def test_export_rejects_malformed_checkpoint_ref():
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    for bad in ("r1", "r1/step-", "r1/checkpoints/step-4"):
        with pytest.raises(ClientError, match="invalid checkpoint id"):
            client.export(bad, repository="me/a", hf_token="hf")


def _deployment_reader(monkeypatch, client, record, *, status=None):
    """Patch the run-scoped deployment read; record the calls it receives.

    Each call is recorded as `(method, path, timeout, body_deadline)` -- both bounds, because
    they bound different things: `timeout` restarts on every byte received, so only the wall-clock
    `body_deadline` bounds a whole read.
    """
    calls: list[tuple] = []

    def request(method, path, body=None, timeout=None, progress=None, body_deadline=None):
        calls.append((method, path, timeout, body_deadline))
        if status is not None:
            raise ApiError(status, "boom")
        return record

    monkeypatch.setattr(client, "_request", request)
    return calls


def test_deployment_for_reads_the_run_scoped_route_not_the_listing(monkeypatch):
    """`/v1/deployments` walks every run the key owns before one record is picked out.

    That makes each poll cost grow with the account's run history, so a long-lived account can
    spend its whole `--wait` budget loading unrelated runs and time out against a checkpoint that
    is already ready. The run-scoped route resolves exactly the run being polled.
    """
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    calls = _deployment_reader(
        monkeypatch,
        client,
        {"state": "queued", "checkpoint_id": "flash-1/final"},
    )

    # the id is carried onto the returned record: `models deploy --wait` prints this in place of
    # the POST body, so dropping it renders an empty run field and omits it from the json.
    assert client.deployment_for("flash-1/final") == {
        "state": "queued",
        "checkpoint_id": "flash-1/final",
        "run_id": "flash-1",
    }
    assert calls == [("GET", "/v1/runs/flash-1/deploy", None, None)]


def test_deployment_for_keeps_a_run_id_already_on_the_record(monkeypatch):
    """The record's own id wins; the requested id must not overwrite it."""
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    _deployment_reader(
        monkeypatch,
        client,
        {"state": "ready", "run_id": "flash-1", "checkpoint_id": "flash-1/final"},
    )

    assert client.deployment_for("flash-1/final") == {
        "state": "ready",
        "run_id": "flash-1",
        "checkpoint_id": "flash-1/final",
    }


def test_deployment_for_asks_about_the_base_run_not_the_checkpoint_ref(monkeypatch):
    """`RUN/step-N` is not a path segment; the route is keyed by the run alone."""
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    calls = _deployment_reader(
        monkeypatch,
        client,
        {
            "state": "ready",
            "checkpoint_id": "flash-1/step-40",
            "checkpoint_step": 40,
            "run_id": "flash-1",
        },
    )

    assert client.deployment_for("flash-1/step-40") == {
        "state": "ready",
        "checkpoint_id": "flash-1/step-40",
        "checkpoint_step": 40,
        "run_id": "flash-1",
    }
    assert calls == [("GET", "/v1/runs/flash-1/deploy", None, None)]


def test_deployment_for_requires_the_requested_checkpoint_id(monkeypatch):
    """The requested checkpoint id is the identity, not decoration.

    Matching on the run id alone let `deploy RUN/step-40 --wait` settle on whichever checkpoint
    was deployed, including an older one still marked ready or a replacement deployed mid-wait.
    """
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    _deployment_reader(
        monkeypatch,
        client,
        {
            "state": "ready",
            "checkpoint_id": "flash-1/step-20",
            "checkpoint_step": 20,
        },
    )

    assert client.deployment_for("flash-1/step-40") is None
    assert client.deployment_for("flash-1/step-20") == {
        "state": "ready",
        "checkpoint_id": "flash-1/step-20",
        "checkpoint_step": 20,
        "run_id": "flash-1",
    }
    assert client.deployment_for("flash-1/final") is None


def test_deployment_for_matches_the_final_checkpoint_id(monkeypatch):
    """The permanent final checkpoint id must match exactly."""
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    _deployment_reader(
        monkeypatch,
        client,
        {
            "state": "ready",
            "checkpoint_id": "flash-1/final",
            "checkpoint_step": None,
        },
    )

    assert client.deployment_for("flash-1/final") == {
        "state": "ready",
        "checkpoint_id": "flash-1/final",
        "checkpoint_step": None,
        "run_id": "flash-1",
    }
    assert client.deployment_for("flash-1/step-40") is None


def test_deployed_checkpoint_reports_whatever_step_is_serving(monkeypatch):
    """`deployed_checkpoint` answers what this run serves, not whether one target is live."""
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    calls = _deployment_reader(
        monkeypatch,
        client,
        {
            "state": "ready",
            "checkpoint_id": "flash-1/step-100",
            "checkpoint_step": 100,
        },
    )

    assert client.deployed_checkpoint("flash-1/step-50") == {
        "state": "ready",
        "checkpoint_id": "flash-1/step-100",
        "checkpoint_step": 100,
        "run_id": "flash-1",
    }
    # the same run-scoped route costs one read and never walks the account's history.
    assert calls == [("GET", "/v1/runs/flash-1/deploy", None, None)]
    assert client.deployment_for("flash-1/step-50") is None


@pytest.mark.parametrize("state", ["undeployed", "dry_run"])
def test_deployed_checkpoint_still_hides_a_never_deployed_run(monkeypatch, state):
    """Dropping the id check does not make an undeployed run into a deployed checkpoint."""
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    _deployment_reader(monkeypatch, client, {"state": state})

    assert client.deployed_checkpoint("flash-1/step-50") is None


def test_deployed_checkpoint_reports_an_unknown_run_as_absent(monkeypatch):
    """A 404 is "nothing deployed", the same answer the listing gave by omitting the row."""
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    _deployment_reader(monkeypatch, client, None, status=404)

    assert client.deployed_checkpoint("flash-1/final") is None


@pytest.mark.parametrize("state", ["undeployed", "dry_run"])
def test_deployment_for_reports_a_never_deployed_run_as_absent(monkeypatch, state):
    """The route answers for an undeployed run with a synthesized record instead of 404.

    The listing this replaced omitted `undeployed` and `dry_run` rows outright, so returning them
    here would make `--wait` treat "nothing is deployed" as a live checkpoint and exit successfully.
    """
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    _deployment_reader(monkeypatch, client, {"state": state, "run_id": "flash-1"})

    assert client.deployment_for("flash-1/final") is None


def test_deployment_for_reports_an_unknown_run_as_absent(monkeypatch):
    """A run this key cannot see reads the same as one that is not deployed.

    The listing said "absent" by omitting the row. Letting the 404 out instead would turn a
    vanished deployment into a failed command rather than the reported end of the wait.
    """
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    _deployment_reader(monkeypatch, client, None, status=404)

    assert client.deployment_for("flash-1/final") is None


def test_deployment_for_still_raises_a_rejected_key(monkeypatch):
    """Only 404 means absent. A rejected key must stay an error the caller can act on."""
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    _deployment_reader(monkeypatch, client, None, status=401)

    with pytest.raises(ApiError):
        client.deployment_for("flash-1/final")


def test_deployment_for_bounds_the_read(monkeypatch):
    """A caller polling against its own deadline has to be able to bound the read.

    The client default is 60s, so an unbounded read inside a short --wait overshoots the timeout
    the user asked for.
    """
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    calls = _deployment_reader(monkeypatch, client, {"state": "undeployed"})

    assert client.deployment_for("flash-1/final", timeout=3.0) is None
    assert calls == [("GET", "/v1/runs/flash-1/deploy", 3.0, None)]


def test_deployed_checkpoint_takes_a_wall_clock_deadline(monkeypatch):
    """`timeout` restarts on every byte received, so alone it does not bound a whole read.

    The advisory pre-deploy read runs before a deploy the user actually asked for, so it needs the
    bound that holds regardless of how the bytes arrive. `--wait` polling deliberately passes no
    deadline: it owns one spanning many reads and recomputes each read's share.
    """
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    calls = _deployment_reader(monkeypatch, client, {"state": "undeployed"})

    assert client.deployed_checkpoint("flash-1/final", timeout=5.0, body_deadline=5.0) is None
    assert calls == [("GET", "/v1/runs/flash-1/deploy", 5.0, 5.0)]


@contextlib.contextmanager
def _trickling_server(body: bytes, gap: float):
    """A server that dribbles one byte at a time, like a slow proxy relaying a response."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            for byte in body:
                try:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return  # the client gave up on its deadline, which is the point
                time.sleep(gap)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.wallclock
def test_a_wall_clock_deadline_bounds_a_body_that_arrives_a_byte_at_a_time():
    """The deadline is checked between reads, so it only bounds a read that returns promptly.

    `read(n)` on a buffered reader blocks until all n bytes arrive, so a peer trickling a short
    body keeps one call inside the socket timeout indefinitely and the between-reads check never
    runs: measured at 12s against a 2s deadline before this was fixed. Reading whatever has
    already arrived is what makes the deadline real, so this asserts the elapsed time rather than
    just the raised error -- the error alone was already raised while the bound was ineffective.

    `ClientError` rather than the "stalled" message specifically: each read now re-caps the socket
    to what is left of the deadline, so on a body still arriving the socket bound is by
    construction always a shade tighter than the between-reads check, and a trickling peer ends as
    `RequestTimeoutError` ("timed out") instead. Both are `ClientError`, both bound the time, and
    which one wins is a race the caller should not be asserting on -- the bound is the contract.
    """
    body = b'{"state": "ready", "checkpoint_step": 100}'
    # every byte lands well inside the socket timeout, so nothing here is a socket-level stall.
    with _trickling_server(body, gap=0.3) as url:
        client = ApiClient(url, "fslo-user-test", timeout=60)
        start = time.monotonic()
        with pytest.raises(ClientError):
            client.deployed_checkpoint("flash-1/final", timeout=2.0, body_deadline=0.5)
        elapsed = time.monotonic() - start

    # generous against CI scheduling noise, and still far under the ~12s of the unbounded read.
    assert elapsed < 5.0, f"the deadline did not bound the read: {elapsed:.2f}s"


@contextlib.contextmanager
def _slow_header_then_stalling_server(header_delay: float, prefix: bytes, promised: int):
    """Spend most of the deadline before the headers land, then stall mid-body.

    The shape matters. `_capped_timeout` already opens the socket at `min(timeout, remaining)`, so
    a socket timeout larger than the whole deadline is never actually installed and a server that
    stalls immediately cannot overrun. The exposure needs the deadline to be partly consumed
    *before* the body loop starts: the socket keeps the timeout it was opened with while the clock
    runs down, so the read that follows can block for longer than the deadline has left.
    """
    ready = threading.Event()
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]

    def serve():
        try:
            conn, _ = sock.accept()
        except OSError:
            return
        try:
            conn.recv(65536)
            time.sleep(header_delay)  # burn deadline while the socket timeout stays put
            conn.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: " + str(promised).encode() + b"\r\n\r\n"
            )
            conn.sendall(prefix)
            ready.wait(30)  # hold the connection open, sending nothing more
        except OSError:
            pass
        finally:
            with contextlib.suppress(OSError):
                conn.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        ready.set()
        with contextlib.suppress(OSError):
            sock.close()
        thread.join(5)


@pytest.mark.wallclock
def test_a_stalled_body_read_cannot_outlast_the_remaining_deadline():
    """A read begun near the deadline must not block past it for the socket's original timeout.

    The socket timeout is installed once, when the request opens. Time spent connecting and waiting
    for headers is charged to the deadline but not to that timeout, so by the time the body loop
    runs the socket is entitled to block for longer than the deadline has left -- measured at 3.51s
    against a 2.0s deadline, and `flash env list` passes 230s as both bounds. Re-capping the socket
    to the remaining budget before each read makes the two agree.

    Asserts elapsed time, not the exception type: the same `RequestTimeoutError` is raised either
    way, so only the clock distinguishes a bounded read from an unbounded one.
    """
    budget = 2.0
    with _slow_header_then_stalling_server(1.5, b"{", promised=500) as url:
        client = ApiClient(url, "fslo-user-test", timeout=60)
        start = time.monotonic()
        with pytest.raises(ClientError):
            client.deployed_checkpoint("flash-1/final", timeout=5.0, body_deadline=budget)
        elapsed = time.monotonic() - start

    # without the re-cap this measured 3.51s. generous against CI noise, still well under that.
    assert elapsed < budget + 0.75, f"the read outlasted the remaining deadline: {elapsed:.2f}s"


def test_capping_the_socket_timeout_never_raises_it():
    """Only ever lower it: raising it would grant more patience than the caller configured."""

    class _Sock:
        def __init__(self, timeout):
            self.timeout = timeout

        def gettimeout(self):
            return self.timeout

        def settimeout(self, value):
            self.timeout = value

    def _resp(sock):
        raw = SimpleNamespace(_sock=sock)
        return SimpleNamespace(fp=SimpleNamespace(raw=raw))

    already_tighter = _Sock(2.0)
    _cap_socket_timeout(_resp(already_tighter), 10.0)
    assert already_tighter.timeout == 2.0

    too_patient = _Sock(10.0)
    _cap_socket_timeout(_resp(too_patient), 2.0)
    assert too_patient.timeout == 2.0

    blocking = _Sock(None)
    _cap_socket_timeout(_resp(blocking), 3.0)
    assert blocking.timeout == 3.0


def test_a_reader_without_a_reachable_socket_still_reads():
    """The socket sits behind private attributes, so an unexpected reader must degrade, not fail."""

    class _Bare:
        def __init__(self, data):
            self._buf = io.BytesIO(data)

        def read(self, size=-1):
            return self._buf.read(size)

        def read1(self, size=-1):
            return self._buf.read(size)

    body = b'{"state": "ready"}'
    got = _read_capped_response(_Bare(body), 10_000, deadline=time.monotonic() + 5.0)
    assert got == body
    # and the cap itself must be inert on anything that does not expose a socket
    _cap_socket_timeout(object(), 1.0)


def test_a_complete_body_still_reads_whole_under_a_deadline():
    """Reading in smaller pieces must not truncate a body that arrives normally.

    The deadline path reads what has already arrived rather than a fixed count, so a body
    delivered across several chunks has to be reassembled rather than cut at the first one.
    """
    payload = {"state": "ready", "checkpoint_step": 100, "pad": "x" * 200_000}
    body = json.dumps(payload).encode()
    with _fixed_2xx_server("application/json", body) as url:
        client = ApiClient(url, "fslo-user-test", timeout=30)
        got = client.deployed_checkpoint("flash-1/final", timeout=30.0, body_deadline=30.0)

    assert got["pad"] == payload["pad"]
    assert got["checkpoint_step"] == 100


@contextlib.contextmanager
def _fixed_2xx_server(content_type: str, body: bytes):
    """A server that answers every GET 200 with one fixed body, like a proxy or another service."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_wrong_shape_2xx_names_the_endpoint_and_the_missing_key():
    """Valid JSON of the wrong shape must not escape as a bare KeyError nothing translates."""
    with _fixed_2xx_server("application/json", b'{"hello": "world"}') as url:
        client = ApiClient(url, "fslo-user-test", timeout=5)
        cases = [
            (client.list_runs, "/v1/runs", "runs"),
            (lambda: client.checkpoints("r1"), "/v1/runs/r1/checkpoints", "checkpoints"),
            (client.deployments, "/v1/deployments", "deployments"),
            (lambda: client.get_logs("r1"), "/v1/runs/r1/logs?offset=0", "logs"),
        ]
        for call, path, key in cases:
            with pytest.raises(ClientError) as caught:
                call()
            message = str(caught.value)
            assert f"{url}{path} returned an unexpected response shape" in message
            assert repr(key) in message
            assert "rather than at a proxy or another service" in message


def test_present_but_unusable_values_are_client_errors_too():
    """A null value passes a presence check and then raises a bare TypeError out of int() or
    iteration, which nothing in the CLI translates any more than it does a KeyError."""
    with (
        _fixed_2xx_server("application/json", b'{"runs": null}') as url,
        pytest.raises(ClientError) as caught,
    ):
        ApiClient(url, "fslo-user-test", timeout=5).list_runs()
    assert f"{url}/v1/runs returned an unexpected response shape" in str(caught.value)
    assert "'runs'" in str(caught.value)
    with (
        _fixed_2xx_server("application/json", b'{"logs": "x", "offset": null}') as url,
        pytest.raises(ClientError) as caught,
    ):
        ApiClient(url, "fslo-user-test", timeout=5).get_logs("r1")
    assert "'offset'" in str(caught.value)


def test_unusable_elements_inside_a_required_list_are_client_errors_too():
    """A list of the wrong elements passes a bare `list` check and then crashes one level down:
    `cmd_runs` sorts `{"runs": [null]}` straight into an AttributeError nothing translates."""
    for body, key, call in [
        (b'{"runs": [null]}', "runs", lambda c: c.list_runs()),
        (b'{"runs": [{}, "x"]}', "runs", lambda c: c.list_runs()),
        (b'{"checkpoints": [null]}', "checkpoints", lambda c: c.checkpoints("r1")),
        (b'{"deployments": [null]}', "deployments", lambda c: c.deployments()),
    ]:
        with (
            _fixed_2xx_server("application/json", body) as url,
            pytest.raises(ClientError) as caught,
        ):
            call(ApiClient(url, "fslo-user-test", timeout=5))
        message = str(caught.value)
        assert "returned an unexpected response shape" in message
        assert repr(key) in message
        assert "rather than at a proxy or another service" in message


def test_healthy_lists_of_objects_still_pass_the_element_check():
    """The element check must not reject the shape the plane actually returns, empty included."""
    with _fixed_2xx_server("application/json", b'{"runs": [{"id": "r1"}]}') as url:
        assert ApiClient(url, "fslo-user-test", timeout=5).list_runs() == [{"id": "r1"}]
    with _fixed_2xx_server("application/json", b'{"runs": []}') as url:
        assert ApiClient(url, "fslo-user-test", timeout=5).list_runs() == []
    # bool subclasses int, so a json true would otherwise satisfy the int requirement and
    # flow into offset arithmetic.
    with (
        _fixed_2xx_server("application/json", b'{"logs": "x", "offset": true}') as url,
        pytest.raises(ClientError) as caught,
    ):
        ApiClient(url, "fslo-user-test", timeout=5).get_logs("r1")
    assert "'offset'" in str(caught.value)


def test_non_object_json_is_a_client_error_even_without_requirements():
    """A proxy or wrong service answering `[]` must not surface as AttributeError downstream."""
    with (
        _fixed_2xx_server("application/json", b"[]") as url,
        pytest.raises(ClientError) as caught,
    ):
        ApiClient(url, "fslo-user-test", timeout=5).me()
    assert "returned JSON that is not an object (list)" in str(caught.value)


def test_non_json_2xx_is_a_client_error_not_a_decode_error():
    """A reverse proxy answering 200 text/html must not leak `Expecting value: line 1 column 1`."""
    with _fixed_2xx_server("text/html", b"<html>hi</html>") as url:
        client = ApiClient(url, "fslo-user-test", timeout=5)
        with pytest.raises(ClientError) as caught:
            client.me()
    message = str(caught.value)
    assert f"{url}/v1/me did not return JSON (Content-Type: text/html)" in message
    assert "rather than at a proxy or another service" in message
    assert "Expecting value" not in message


def test_empty_2xx_body_still_decodes_to_an_empty_object():
    """An empty 2xx body decodes to {}, so callers reading a dict from it still get one."""
    with _fixed_2xx_server("application/json", b"") as url:
        assert ApiClient(url, "fslo-user-test", timeout=5).me() == {}


def test_spec_payload_sends_an_omitted_gpu_count_as_omitted() -> None:
    """The server re-derives "auto-size" from the ABSENCE of gpu.count, so the placeholder must go.

    `to_dict()` keeps `count: 1` for preparation-digest stability and strips the provenance marker.
    Sending that verbatim made the server reparse an auto-sized run as an authored one-card pin and
    reject it at the pinned-count preflight before a run was ever created -- so managed submit never
    auto-sized at all. Each authored form must still arrive as a hard pin.
    """
    base = {
        "model": "Qwen/Qwen3.5-9B",
        "project": "11111111-1111-4111-8111-111111111111",
        "algorithm": "grpo",
        "environment": {"id": "owner/project/env"},
        "train": {"max_steps": 10},
    }

    omitted = spec_from_dict(base)
    assert omitted.gpu_count_auto is True
    payload = spec_payload(omitted)
    assert "count" not in payload["gpu"]
    assert spec_from_dict(payload).gpu_count_auto is True

    for authored, expected_count in (
        ({"count": 1}, 1),
        ({"count": 4}, 4),
        ({"type": "B200"}, 1),
    ):
        pinned = spec_from_dict({**base, "gpu": authored})
        reparsed = spec_from_dict(spec_payload(pinned))
        assert reparsed.gpu_count_auto is False, f"{authored} must stay a hard pin"
        assert reparsed.gpu.count == expected_count
