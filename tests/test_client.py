"""ApiClient: auth headers, error mapping, log paging (stdlib stub server, CPU-only)."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from flash.client import ApiClient, ApiError, ClientError, RequestTimeoutError
from flash.client.http import _parse_chat_target, _prepare_chat_request
from flash.client.specs import spec_payload
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
                self._send(200, {"id": "freesolo-co/e"})
                return
            if self.path == "/v1/runs/json-chat/chat":
                self._send(200, {"choices": [{"message": {"content": "json reply"}}]})
                return
            if self.path == "/v1/runs" and seen["body"].get("spec", {}).get("model") == "pending":
                self._send(
                    409,
                    {
                        "detail": {
                            "code": "workload_profile_pending",
                            "profile_run_id": "profile-sft-abc",
                            "state": "queued",
                            "owned": False,
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
            "model": "Qwen/Qwen3.5-4B",
            "project": "11111111-1111-4111-8111-111111111111",
            "algorithm": "opd",
            "environment": {"id": "owner/env"},
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
        "teacher_model": "accounts/fireworks/models/glm-5p2",
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
    """The pending payload has to arrive as a dict, not as the repr of one.

    Every caller-visible behaviour on a profile miss -- naming the run, quoting its separate
    charge, deciding whether to tell the user to poll it -- is a branch on a key of this dict. If
    the transport flattened it to a string the code lookup would return "" and the miss would
    surface as a bare 409 traceback, which is the same failure whether the server is right or not.
    """
    url, _ = stub
    client = ApiClient(url, "fslo-user-test")

    with pytest.raises(ApiError) as excinfo:
        client.create_run({"model": "pending", "project": _PROJECT_ID})

    exc = excinfo.value
    assert exc.status == 409
    assert exc.detail == {
        "code": "workload_profile_pending",
        "profile_run_id": "profile-sft-abc",
        "state": "queued",
        "owned": False,
    }
    # `code` reads through to the dict, and False survives as False rather than as the string
    # "False" -- which is truthy, and would flip the ownership branch to the wrong wording.
    assert exc.code == "workload_profile_pending"
    assert exc.detail["owned"] is False
    # the message still stringifies for callers that only print it.
    assert "workload_profile_pending" in str(exc)


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


def test_parse_chat_target_supports_bare_step_and_immutable_revision() -> None:
    revision = "run-abc@step-5." + "a" * 40

    assert _parse_chat_target("run-abc") == ("run-abc", None, None)
    assert _parse_chat_target("run-abc/step-5") == ("run-abc", None, 5)
    assert _parse_chat_target(revision) == ("run-abc", revision, None)


def test_prepare_chat_request_sends_step_without_adapter_revision() -> None:
    base_run_id, body = _prepare_chat_request(
        "run-abc/step-5",
        [{"role": "user", "content": "hi"}],
        0.0,
        32,
    )

    assert base_run_id == "run-abc"
    assert body["step"] == 5
    assert "adapter_revision" not in body


def test_chat_omits_thinking_template_controls(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    client.chat("json-chat", messages=[{"role": "user", "content": "hi"}])
    assert seen["body"] == {
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
        "max_tokens": 512,
    }


def test_chat_sends_user_supplied_system_prompt(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")

    client.chat(
        "json-chat",
        messages=[
            {"role": "system", "content": "stay terse"},
            {"role": "user", "content": "hi"},
        ],
    )

    assert seen["body"] == {
        "messages": [
            {"role": "system", "content": "stay terse"},
            {"role": "user", "content": "hi"},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }


def test_chat_full_immutable_revision_is_forwarded_unchanged(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    revision = "run-a@step-40." + "a" * 40

    client.chat(revision, [{"role": "user", "content": "hi"}])

    assert seen["path"] == "/v1/runs/run-a/chat"
    assert seen["body"]["adapter_revision"] == revision


def test_chat_checkpoint_shorthand_forwards_step(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")

    client.chat("run-a/step-40", [{"role": "user", "content": "hi"}])

    assert seen["path"] == "/v1/runs/run-a/chat"
    assert seen["body"]["step"] == 40
    assert "adapter_revision" not in seen["body"]


def test_chat_checkpoint_shorthand_rejects_older_server(stub):
    url, seen = stub
    seen["old_chat_server"] = True
    client = ApiClient(url, "fslo-user-test")

    with pytest.raises(ClientError, match="chat_step_selector_v1"):
        client.chat("run-a/step-40", [{"role": "user", "content": "hi"}])

    assert seen["path"] == "/v1/health"
    assert "body" not in seen


def test_chat_checkpoint_capability_is_probed_once_per_client(stub):
    # the capability is a property of the control plane, not of the request. `env eval` sends one
    # chat per case, so re-probing each time doubled the request count and let a single transient
    # /v1/health blip fail an arbitrary case while the chat endpoint was healthy (codex[bot]).
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")

    for _ in range(3):
        client.chat("run-a/step-40", [{"role": "user", "content": "hi"}])
    list(client.chat_stream("run-a/step-40", [{"role": "user", "content": "hi"}]))

    assert seen["health_calls"] == 1
    # a separate client has its own plane to verify, so the cache is not process-global
    ApiClient(url, "fslo-user-test").chat("run-a/step-40", [{"role": "user", "content": "hi"}])
    assert seen["health_calls"] == 2


def test_warming_the_capability_serves_later_chats_from_the_cache(stub):
    # a caller about to fan out settles the check up front, so its workers inherit the cached
    # answer instead of every one of them missing the cold cache at once and firing its own
    # /v1/health (codex[bot]). a target with no step selector needs nothing and must stay silent.
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")

    client.warm_chat_step_selector("run-a@step-40." + "a" * 40)
    assert seen.get("health_calls") is None

    client.warm_chat_step_selector("run-a/step-40")
    assert seen["health_calls"] == 1

    for _ in range(3):
        client.chat("run-a/step-40", [{"role": "user", "content": "hi"}])
    assert seen["health_calls"] == 1


def test_warming_the_capability_raises_on_a_plane_that_lacks_it(stub):
    # the warm-up must not swallow a genuine capability failure into a silently uncached state:
    # it raises exactly what the per-request check raises, so the caller fails before fanning out.
    url, seen = stub
    seen["old_chat_server"] = True
    client = ApiClient(url, "fslo-user-test")

    with pytest.raises(ClientError, match="chat_step_selector_v1"):
        client.warm_chat_step_selector("run-a/step-40")

    assert "body" not in seen


def test_chat_checkpoint_capability_failure_is_not_cached(stub):
    # only a positive result is cached: a plane that genuinely lacks the capability must keep
    # failing, not fall through to an unsupported request on the second call.
    url, seen = stub
    seen["old_chat_server"] = True
    client = ApiClient(url, "fslo-user-test")

    for _ in range(2):
        with pytest.raises(ClientError, match="chat_step_selector_v1"):
            client.chat("run-a/step-40", [{"role": "user", "content": "hi"}])

    assert seen["health_calls"] == 2
    assert "body" not in seen


@pytest.mark.parametrize("step", ["00", "01"])
def test_chat_rejects_zero_padded_immutable_revision(stub, step):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    revision = f"run-a@step-{step}." + "a" * 40

    with pytest.raises(ClientError, match="invalid run id"):
        client.chat(revision, [{"role": "user", "content": "hi"}])

    assert seen == {}


def test_chat_stream_sends_stream_request_and_yields_text(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    chunks = list(
        client.chat_stream("r1", [{"role": "user", "content": "hi"}], temperature=0.2, max_tokens=7)
    )
    assert "".join(chunks) == "héllo"
    assert seen["path"] == "/v1/runs/r1/chat"
    assert seen["auth"] == "Bearer fslo-user-test"
    assert seen["body"] == {
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2,
        "max_tokens": 7,
        "stream": True,
    }


def test_chat_stream_full_immutable_revision_is_forwarded_unchanged(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    revision = "run-a@step-40." + "a" * 40

    chunks = list(
        client.chat_stream(
            revision,
            [{"role": "user", "content": "hi"}],
            temperature=0.2,
            max_tokens=7,
        )
    )

    assert "".join(chunks) == "héllo"
    assert seen["path"] == "/v1/runs/run-a/chat"
    assert seen["body"] == {
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2,
        "max_tokens": 7,
        "stream": True,
        "adapter_revision": revision,
    }


def test_chat_stream_checkpoint_shorthand_forwards_step(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")

    chunks = list(client.chat_stream("run-a/step-40", [{"role": "user", "content": "hi"}]))

    assert "".join(chunks) == "héllo"
    assert seen["path"] == "/v1/runs/run-a/chat"
    assert seen["body"]["step"] == 40
    assert "adapter_revision" not in seen["body"]


def test_chat_stream_checkpoint_shorthand_rejects_older_server(stub):
    url, seen = stub
    seen["old_chat_server"] = True
    client = ApiClient(url, "fslo-user-test")

    with pytest.raises(ClientError, match="chat_step_selector_v1"):
        list(client.chat_stream("run-a/step-40", [{"role": "user", "content": "hi"}]))

    assert seen["path"] == "/v1/health"
    assert "body" not in seen


def test_chat_stream_accepts_json_fallback(stub):
    url, _ = stub
    client = ApiClient(url, "fslo-user-test")
    chunks = list(client.chat_stream("json-chat", [{"role": "user", "content": "hi"}]))
    assert chunks == ["json reply"]


def test_publish_env_plain_without_progress(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    out = client.publish_env(
        name="e", package_b64="QQ==", project_id="11111111-1111-4111-8111-111111111111"
    )
    assert out["id"] == "freesolo-co/e"
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
    assert out["id"] == "freesolo-co/e"
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
    out = client.delete_env("acme/my-env", project_id="11111111-1111-4111-8111-111111111111")
    assert out == {"id": "acme/my-env", "deleted": True}
    assert seen["method"] == "DELETE"
    # the namespace/name slug (with its slash) goes straight into the path
    assert seen["path"] == "/v1/envs/acme/my-env"
    assert seen["auth"] == "Bearer fslo-user-test"
    assert seen["project_id"] == "11111111-1111-4111-8111-111111111111"


def test_delete_env_rejects_blank_project_before_request(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    with pytest.raises(ClientError, match="project id is required"):
        client.delete_env("acme/my-env", project_id="   ")
    assert "method" not in seen


def test_delete_env_percent_encodes_reserved_chars(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    # A programmatic caller passing reserved characters must NOT be able to truncate the request
    # target: `?` becomes %3F (not a query string), `#` becomes %23 (not a dropped fragment), while
    # the namespace/name separator `/` is preserved so the server still routes the :path param.
    client.delete_env("team/env?x=1#frag", project_id="11111111-1111-4111-8111-111111111111")
    assert seen["method"] == "DELETE"
    assert seen["path"] == "/v1/envs/team/env%3Fx%3D1%23frag"


def test_download_env_package_uses_flash_control_plane(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")

    data = client.download_env_package("acme/my-env")

    assert data == b"package-bytes"
    assert seen["path"] == "/v1/envs/acme/my-env/package"
    assert seen["auth"] == "Bearer fslo-user-test"


def test_download_env_package_percent_encodes_reserved_chars(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")

    client.download_env_package("team/env?x=1#frag")

    assert seen["path"] == "/v1/envs/team/env%3Fx%3D1%23frag/package"


def test_download_env_package_caps_response_body(stub, monkeypatch):
    from flash.envs import loader as adapter

    url, _seen = stub
    monkeypatch.setattr(adapter, "_MAX_ARCHIVE_BYTES", 5)
    client = ApiClient(url, "fslo-user-test")

    with pytest.raises(ClientError, match="maximum allowed size"):
        client.download_env_package("acme/my-env")


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
    assert out["id"] == "freesolo-co/e"
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
    assert out["id"] == "freesolo-co/e"
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

    def request(method, path, body=None, timeout=None, progress=None):
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

    def request(method, path, body=None, timeout=None, progress=None):
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

    def request(method, path, body=None, timeout=None, progress=None):
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
    for bad in ("flash-run/step-", "flash-run/checkpoints/step-4", "flash-run/step-4/adapter"):
        with pytest.raises(ClientError, match="invalid adapter id"):
            client.deploy(bad)


def test_deploy_checkpoint_ref_posts_step(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    client.deploy("flash-run/step-40")
    assert seen["path"] == "/v1/runs/flash-run/deploy"
    assert seen["body"]["step"] == 40
    assert seen["body"]["dry_run"] is False
    # smoke verification is mandatory server-side; the client sends no opt-out knob
    assert "verify" not in seen["body"]


def test_deploy_final_ref_omits_step(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    client.deploy("flash-run")
    assert seen["path"] == "/v1/runs/flash-run/deploy"
    assert seen["body"] == {"dry_run": False}


def test_export_sends_repository_token_and_checkpoint_ref(stub):
    """`flash export` posts the destination repo, the user's HF token, and parsed checkpoint step."""
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    client.export("r1/step-40", repository="me/adapters", hf_token="hf_secret", private=False)
    assert seen["path"] == "/v1/runs/r1/export"
    assert seen["auth"] == "Bearer fslo-user-test"
    assert seen["body"] == {
        "repository": "me/adapters",
        "hf_token": "hf_secret",
        "private": False,
        "step": 40,
    }


def test_export_omits_step_when_unset_and_defaults_private(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    client.export("r1", repository="me/adapters", hf_token="hf_secret")
    assert seen["body"] == {
        "repository": "me/adapters",
        "hf_token": "hf_secret",
        "private": True,
    }
    assert "step" not in seen["body"]


def test_export_rejects_malformed_checkpoint_ref():
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    for bad in ("r1/step-", "r1/checkpoints/step-4", "r1/step-4/adapter"):
        with pytest.raises(ClientError, match="invalid adapter id"):
            client.export(bad, repository="me/a", hf_token="hf")


def _deployment_reader(monkeypatch, client, record, *, status=None):
    """Patch the run-scoped deployment read; record the calls it receives."""
    calls: list[tuple] = []

    def request(method, path, body=None, timeout=None, progress=None):
        calls.append((method, path, timeout))
        if status is not None:
            raise ApiError(status, "boom")
        return record

    monkeypatch.setattr(client, "_request", request)
    return calls


def test_deployment_for_reads_the_run_scoped_route_not_the_listing(monkeypatch):
    """`/v1/deployments` walks every run the key owns before one record is picked out.

    That makes each poll cost grow with the account's run history, so a long-lived account can
    spend its whole `--wait` budget loading unrelated runs and time out against a revision that is
    already ready. The run-scoped route resolves exactly the run being polled.
    """
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    calls = _deployment_reader(monkeypatch, client, {"state": "queued"})

    # the id is carried onto the returned record: `models deploy --wait` prints this in place of
    # the POST body, so dropping it renders an empty run field and omits it from the json.
    assert client.deployment_for("flash-1") == {"state": "queued", "run_id": "flash-1"}
    assert calls == [("GET", "/v1/runs/flash-1/deploy", None)]


def test_deployment_for_keeps_a_run_id_already_on_the_record(monkeypatch):
    """The record's own id wins; the requested id must not overwrite it."""
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    _deployment_reader(monkeypatch, client, {"state": "ready", "run_id": "flash-1"})

    assert client.deployment_for("flash-1") == {"state": "ready", "run_id": "flash-1"}


def test_deployment_for_asks_about_the_base_run_not_the_checkpoint_ref(monkeypatch):
    """`RUN/step-N` is not a path segment; the route is keyed by the run alone."""
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    calls = _deployment_reader(
        monkeypatch, client, {"state": "ready", "checkpoint_step": 40, "run_id": "flash-1"}
    )

    assert client.deployment_for("flash-1/step-40") == {
        "state": "ready",
        "checkpoint_step": 40,
        "run_id": "flash-1",
    }
    assert calls == [("GET", "/v1/runs/flash-1/deploy", None)]


def test_deployment_for_requires_the_requested_checkpoint_step(monkeypatch):
    """The requested step is part of the identity, not decoration.

    Matching on the run id alone let `deploy RUN/step-40 --wait` settle on whichever revision was
    deployed -- an older one still marked ready, or a replacement another shell deployed mid-wait
    -- and report it as the caller's own.
    """
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    _deployment_reader(monkeypatch, client, {"state": "ready", "checkpoint_step": 20})

    assert client.deployment_for("flash-1/step-40") is None
    assert client.deployment_for("flash-1/step-20") == {
        "state": "ready",
        "checkpoint_step": 20,
        "run_id": "flash-1",
    }
    # the bare run id is the FINAL adapter, which the plane reports as a null step -- a checkpoint
    # revision must not answer for it either.
    assert client.deployment_for("flash-1") is None


def test_deployment_for_matches_the_final_adapters_null_step(monkeypatch):
    """`checkpoint_step` is None for the final adapter; the bare run id must still match it."""
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    _deployment_reader(monkeypatch, client, {"state": "ready", "checkpoint_step": None})

    assert client.deployment_for("flash-1") == {
        "state": "ready",
        "checkpoint_step": None,
        "run_id": "flash-1",
    }
    assert client.deployment_for("flash-1/step-40") is None


@pytest.mark.parametrize("state", ["undeployed", "dry_run"])
def test_deployment_for_reports_a_never_deployed_run_as_absent(monkeypatch, state):
    """The route answers for an undeployed run with a synthesized record instead of 404.

    The listing this replaced omitted `undeployed` and `dry_run` rows outright, so returning them
    here would make `--wait` treat "nothing is deployed" as a live revision and, since neither is
    a busy state, exit 0 against nothing serving.
    """
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    _deployment_reader(monkeypatch, client, {"state": state, "run_id": "flash-1"})

    assert client.deployment_for("flash-1") is None


def test_deployment_for_reports_an_unknown_run_as_absent(monkeypatch):
    """A run this key cannot see reads the same as one that is not deployed.

    The listing said "absent" by omitting the row. Letting the 404 out instead would turn a
    vanished deployment into a failed command rather than the reported end of the wait.
    """
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    _deployment_reader(monkeypatch, client, None, status=404)

    assert client.deployment_for("flash-1") is None


def test_deployment_for_still_raises_a_rejected_key(monkeypatch):
    """Only 404 means absent. A rejected key must stay an error the caller can act on."""
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    _deployment_reader(monkeypatch, client, None, status=401)

    with pytest.raises(ApiError):
        client.deployment_for("flash-1")


def test_deployment_for_bounds_the_read(monkeypatch):
    """A caller polling against its own deadline has to be able to bound the read.

    The client default is 60s, so an unbounded read inside a short --wait overshoots the timeout
    the user asked for.
    """
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    calls = _deployment_reader(monkeypatch, client, {"state": "undeployed"})

    assert client.deployment_for("flash-1", timeout=3.0) is None
    assert calls == [("GET", "/v1/runs/flash-1/deploy", 3.0)]
