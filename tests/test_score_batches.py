"""Score batches run every request on Lambda only, wait out capacity, and return all at once."""

import json
import threading
import time

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request

from flash.server.domain.ops import score_batches
from flash.server.platform import db
from flash.server.platform.deps import require_key

MODEL = "Qwen/Qwen3.5-9B-Base"
INTERNAL = {"id": 7, "auth_kind": "internal"}


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))
    monkeypatch.setattr(score_batches, "PROBE_RETRY_SECONDS", 0)
    monkeypatch.setattr(score_batches, "ITEM_BACKOFF_SECONDS", 0)


def _body(count=3, **extra):
    requests = [
        {"id": f"case-{i}", "messages": [{"role": "user", "content": f"q{i}"}], "max_tokens": 8}
        for i in range(count)
    ]
    return {"model": MODEL, "requests": requests, **extra}


def _submit(body=None, key=INTERNAL, org="org-test"):
    return score_batches.submit(body or _body(), key, org, None)


def _drain(runner, timeout=10.0):
    """Run the runner until no batch is active (each batch runs on its own thread)."""
    deadline = time.monotonic() + timeout
    runner.tick()
    while time.monotonic() < deadline:
        threads = [thread for _, thread in runner.active.values()]
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        runner.tick()
        if not runner.active:
            return
    raise AssertionError("score batch did not finish")


def _result(batch_id):
    return score_batches.public(score_batches.get_row(batch_id), include_results=True)


def _ok(request):
    return {"choices": [{"message": {"content": request["messages"][0]["content"]}}]}


def test_batch_waits_for_lambda_capacity_then_returns_every_result_together():
    calls = []
    unavailable = {"left": 3}
    lock = threading.Lock()

    def send(row, key, request):
        with lock:
            calls.append(request["messages"][0]["content"])
            if unavailable["left"]:
                unavailable["left"] -= 1
                raise HTTPException(status_code=503, detail="Lambda capacity unavailable")
        assert row["target"] == MODEL
        assert row["org_id"] == "org-test"
        return _ok(request)

    batch = _submit()
    assert batch["state"] == "queued"
    assert batch["provider"] == "lambda"
    assert "results" not in score_batches.public(
        score_batches.get_row(batch["batch_id"]), include_results=True
    )
    runner = score_batches.ScoreBatchRunner(send=send)
    _drain(runner)

    done = _result(batch["batch_id"])
    assert done["state"] == "done"
    assert done["capacity_at"] is not None
    # One request at a time while Lambda had no capacity (rotating), then the rest fanned out.
    assert calls[:4] == ["q0", "q1", "q2", "q0"]
    assert sorted(calls[4:]) == ["q1", "q2"]
    assert [r["id"] for r in done["results"]] == ["case-0", "case-1", "case-2"]
    assert [r["response"]["choices"][0]["message"]["content"] for r in done["results"]] == [
        "q0",
        "q1",
        "q2",
    ]
    assert done["succeeded"] == 3
    assert done["failed"] == 0


def test_batch_fails_every_request_when_capacity_never_arrives():
    def send(row, key, request):
        raise httpx.ConnectError("no lambda worker")

    batch = _submit()
    runner = score_batches.ScoreBatchRunner(send=send, capacity_wait=lambda: 0.001)
    time.sleep(0.01)
    _drain(runner)
    done = _result(batch["batch_id"])
    assert done["state"] == "failed"
    assert "Lambda capacity was unavailable" in done["error"]
    assert all("Lambda capacity" in r["error"] for r in done["results"])


def test_a_bad_request_is_recorded_without_counting_as_a_capacity_miss():
    def send(row, key, request):
        if request["messages"][0]["content"] in {"q0", "q2"}:
            raise HTTPException(status_code=400, detail="bad schema")
        return _ok(request)

    batch = _submit()
    _drain(score_batches.ScoreBatchRunner(send=send))
    done = _result(batch["batch_id"])
    assert done["state"] == "done"
    assert done["results"][0] == {"id": "case-0", "error": "bad schema", "status_code": 400}
    assert "response" in done["results"][1]
    assert done["results"][2]["error"] == "bad schema"
    assert done["failed"] == 2


def test_retryable_failures_after_capacity_retry_then_record_the_error(monkeypatch):
    monkeypatch.setattr(score_batches, "ITEM_ATTEMPTS", 3)
    attempts = {}

    def send(row, key, request):
        content = request["messages"][0]["content"]
        attempts[content] = attempts.get(content, 0) + 1
        if content == "q1":
            raise HTTPException(status_code=502, detail="inference failure")
        if content == "q2" and attempts[content] == 1:
            raise HTTPException(status_code=503, detail="busy")
        return _ok(request)

    batch = _submit()
    _drain(score_batches.ScoreBatchRunner(send=send))
    done = _result(batch["batch_id"])
    assert attempts == {"q0": 1, "q1": 3, "q2": 2}
    assert done["results"][1]["error"] == "inference failure"
    assert "response" in done["results"][2]


def test_a_stale_batch_resumes_on_another_plane_with_only_unrecorded_requests():
    batch = _submit()
    batch_id = batch["batch_id"]
    folder = score_batches._batch_dir(batch_id)
    (folder / "results.jsonl").write_text(
        json.dumps({"index": 0, "id": "case-0", "response": {"cached": True}}) + "\n"
    )
    conn = score_batches._conn()
    with conn:
        conn.execute(
            "UPDATE score_batches SET state='running', owner='dead-plane', heartbeat_at=?,"
            " capacity_at=? WHERE batch_id=?",
            (time.time() - 3600, time.time() - 3600, batch_id),
        )
    sent = []

    def send(row, key, request):
        sent.append(request["messages"][0]["content"])
        return _ok(request)

    _drain(score_batches.ScoreBatchRunner(send=send))
    done = _result(batch_id)
    assert done["state"] == "done"
    assert sorted(sent) == ["q1", "q2"]
    assert done["results"][0]["response"] == {"cached": True}


def test_a_live_owner_keeps_its_batch():
    batch = _submit()
    conn = score_batches._conn()
    with conn:
        conn.execute(
            "UPDATE score_batches SET owner='other-plane', heartbeat_at=? WHERE batch_id=?",
            (time.time(), batch["batch_id"]),
        )
    runner = score_batches.ScoreBatchRunner(send=lambda *a: pytest.fail("must not send"))
    runner.tick()
    assert not runner.active


def test_released_batches_are_claimed_at_once_by_the_next_runner():
    gate = threading.Event()

    def slow(row, key, request):
        gate.wait(5)
        raise HTTPException(status_code=503, detail="still waiting")

    batch = _submit()
    first = score_batches.ScoreBatchRunner(send=slow)
    first.tick()
    assert batch["batch_id"] in first.active
    gate.set()
    first.release()
    assert score_batches.get_row(batch["batch_id"])["owner"] is None
    second = score_batches.ScoreBatchRunner(send=lambda row, key, request: _ok(request))
    _drain(second)
    assert _result(batch["batch_id"])["state"] == "done"


def test_deadline_marks_requests_not_yet_sent_as_unfinished():
    def send(row, key, request):
        if request["messages"][0]["content"] == "q0":
            # The probe succeeds, then the clock passes the deadline before the rest are sent.
            conn = score_batches._conn()
            with conn:
                conn.execute("UPDATE score_batches SET deadline_s=60")
            return _ok(request)
        return _ok(request)

    batch = _submit(_body(deadline_seconds=60))
    batch_id = batch["batch_id"]
    runner = score_batches.ScoreBatchRunner(send=send)
    original = score_batches._Batch._update

    def update(self, **fields):
        if "capacity_at" in fields:
            fields["capacity_at"] = time.time() - 120
        original(self, **fields)

    score_batches._Batch._update = update
    try:
        _drain(runner)
    finally:
        score_batches._Batch._update = original
    done = _result(batch_id)
    assert "response" in done["results"][0]
    assert done["results"][1]["unfinished"] is True
    assert done["results"][2]["unfinished"] is True


def test_cancel_stops_a_waiting_batch():
    entered = threading.Event()

    def send(row, key, request):
        entered.set()
        raise HTTPException(status_code=503, detail="waiting")

    batch = _submit()
    runner = score_batches.ScoreBatchRunner(send=send)
    runner.tick()
    assert entered.wait(5)
    cancelled = score_batches.cancel(batch["batch_id"], INTERNAL)
    assert cancelled["state"] == "cancelled"
    _drain(runner)
    assert _result(batch["batch_id"])["state"] == "cancelled"


def test_client_batch_id_makes_submission_idempotent():
    first = _submit(_body(batch_id="round-1"))
    again = _submit(_body(batch_id="round-1"))
    assert first["batch_id"] == again["batch_id"]
    other = _submit(_body(batch_id="round-2"))
    assert other["batch_id"] != first["batch_id"]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"model": "deepseek-ai/DeepSeek-V4", "requests": [{"messages": []}]}, "Lambda-served"),
        (
            {"model": MODEL, "checkpoint_id": "r/final", "requests": [{"messages": []}]},
            "exactly one",
        ),
        ({"model": MODEL, "requests": []}, "requests must be a list"),
        (_body(concurrency=0), "concurrency"),
        (_body(deadline_seconds=5), "deadline_seconds"),
        (
            {
                "model": MODEL,
                "requests": [{"messages": [{"role": "user", "content": "x"}], "stream": True}],
            },
            "cannot stream",
        ),
        ({"model": MODEL, "requests": [{"messages": "nope"}]}, "request 1"),
        ({"model": MODEL, "requests": [{"model": "x", "messages": []}]}, "whole batch"),
        ({**_body(), "surprise": 1}, "unsupported"),
    ],
)
def test_invalid_batches_are_rejected_before_anything_is_stored(body, message):
    with pytest.raises(HTTPException) as exc:
        _submit(body)
    assert exc.value.status_code == 400
    assert message in exc.value.detail
    assert not score_batches.batches_dir().exists() or not any(
        score_batches.batches_dir().iterdir()
    )


def test_base_model_batches_need_an_org_to_bill():
    with pytest.raises(HTTPException) as exc:
        score_batches.submit(_body(), {"id": 3, "auth_kind": "freesolo_api_key"}, None, None)
    assert "organization" in exc.value.detail
    batch = score_batches.submit(
        _body(), {"id": 3, "auth_kind": "freesolo_api_key", "org_id": "org-9"}, None, None
    )
    assert score_batches.get_row(batch["batch_id"])["org_id"] == "org-9"


def test_checkpoint_batches_validate_each_request_through_managed_chat(monkeypatch):
    from flash.server.routes import serving_chat

    seen = []

    def resolve(run_id, payload, key, org, project):
        from types import SimpleNamespace

        from flash.serve.request.openai import parse_chat_request

        seen.append((run_id, payload["checkpoint_id"]))
        if payload["messages"][0]["content"] == "q1":
            raise HTTPException(status_code=400, detail="tools need thinking off")
        request = parse_chat_request(payload, require_model=False, allow_managed_selectors=True)
        spec = SimpleNamespace(train=SimpleNamespace(stop_sequences=()))
        return request, [], spec, payload["checkpoint_id"], "org"

    monkeypatch.setattr(serving_chat, "_resolve_chat_request", resolve)
    body = {k: v for k, v in _body().items() if k != "model"}
    with pytest.raises(HTTPException) as exc:
        _submit({**body, "checkpoint_id": "flash-1-abc/final"})
    assert "request 2: tools need thinking off" in exc.value.detail
    assert seen[0] == ("flash-1-abc", "flash-1-abc/final")


def test_checkpoint_requests_forward_through_managed_chat_pinned_to_lambda(monkeypatch):
    from flash.server.routes import serving_chat

    captured = {}

    def managed_chat(run_id, payload, key, org, project, *, timeout_s, serving_provider):
        captured.update(
            run_id=run_id,
            payload=payload,
            key=key,
            timeout_s=timeout_s,
            serving_provider=serving_provider,
        )
        return {"ok": True}

    monkeypatch.setattr(serving_chat, "managed_chat", managed_chat)
    row = {
        "run_id": "flash-1-abc",
        "target": "flash-1-abc/step-4",
        "org_id": "org-test",
        "project_id": "proj",
    }
    assert score_batches._send_request(row, INTERNAL, {"messages": []}) == {"ok": True}
    assert captured["serving_provider"] == "lambda"
    assert captured["payload"]["checkpoint_id"] == "flash-1-abc/step-4"
    assert captured["timeout_s"] == score_batches.REQUEST_TIMEOUT_SECONDS


def test_base_requests_forward_to_serving_pinned_to_lambda(monkeypatch):
    import flash.serve.deployment.deploy as serving_deploy
    from flash.serve.request import transport

    captured = {}

    def request_chat_json(_client, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-secret")
    monkeypatch.setattr(transport, "request_chat_json", request_chat_json)
    monkeypatch.setattr(transport, "serving_openai_base_url", lambda: "https://serving/v1")
    row = {"run_id": None, "target": MODEL, "org_id": "org-test", "thinking": 1}
    result = score_batches._send_request(
        row, INTERNAL, {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}
    )
    assert result == {"ok": True}
    assert captured["headers"]["X-Flash-Serving-Provider"] == "lambda"
    assert captured["headers"]["X-Freesolo-Org-Id"] == "org-test"
    assert captured["run_id"] == MODEL
    assert captured["chat_template_kwargs"]["enable_thinking"] is True
    assert captured["thinking"] is True
    assert captured["max_tokens"] == 5
    assert serving_deploy._provider_header(None) == {}


def test_checkpoint_chat_sends_the_provider_pin(monkeypatch):
    import flash.serve.deployment.deploy as serving_deploy
    from flash.serve.request import transport

    captured = {}
    monkeypatch.setattr(
        transport, "request_chat_json", lambda _c, **kwargs: captured.update(kwargs) or {}
    )
    monkeypatch.setattr(transport, "serving_openai_base_url", lambda: "https://serving/v1")
    serving_deploy.chat("flash-1-abc/final", [], org_id="org", serving_provider="lambda")
    assert captured["headers"]["X-Flash-Serving-Provider"] == "lambda"
    serving_deploy.chat("flash-1-abc/final", [], org_id="org")
    assert "X-Flash-Serving-Provider" not in captured["headers"]


def _request(headers):
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "headers": raw})


def test_only_trusted_internal_callers_can_pin_serving_to_lambda():
    from flash.serving.src.http import provider_pin

    provider_pin.pin_from_request(
        _request({"X-Flash-Serving-Provider": "lambda"}), trusted_internal=True
    )
    assert provider_pin.lambda_pinned()
    provider_pin.pin_from_request(_request({}), trusted_internal=True)
    assert not provider_pin.lambda_pinned()
    with pytest.raises(HTTPException) as exc:
        provider_pin.pin_from_request(
            _request({"X-Flash-Serving-Provider": "lambda"}), trusted_internal=False
        )
    assert exc.value.status_code == 403
    with pytest.raises(HTTPException) as exc:
        provider_pin.pin_from_request(
            _request({"X-Flash-Serving-Provider": "modal"}), trusted_internal=True
        )
    assert exc.value.status_code == 400


def test_routes_submit_poll_and_cancel():
    from flash.server.routes import score_batches as routes

    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[require_key] = lambda: INTERNAL

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            created = await client.post(
                "/v1/score-batches", json=_body(), headers={"X-Freesolo-Org-Id": "org-test"}
            )
            assert created.status_code == 202, created.text
            batch_id = created.json()["batch_id"]
            polled = await client.get(f"/v1/score-batches/{batch_id}")
            assert polled.json()["state"] == "queued"
            assert "results" not in polled.json()
            cancelled = await client.delete(f"/v1/score-batches/{batch_id}")
            assert cancelled.json()["state"] == "cancelled"
            assert len(cancelled.json()["results"]) == 3
            missing = await client.get("/v1/score-batches/sb-unknown")
            assert missing.status_code == 404

    import asyncio

    asyncio.run(exercise())


def test_other_keys_cannot_read_a_batch():
    batch = _submit()
    with pytest.raises(HTTPException) as exc:
        score_batches.owned_row(batch["batch_id"], {"id": 999})
    assert exc.value.status_code == 404


def test_a_cancel_that_races_the_claim_is_not_overwritten():
    batch = _submit()
    runner = score_batches.ScoreBatchRunner(send=lambda row, key, request: _ok(request))
    row = runner._claim(time.time())
    score_batches.cancel(batch["batch_id"], INTERNAL)
    score_batches._Batch(runner, row).run()
    assert score_batches.get_row(batch["batch_id"])["state"] == "cancelled"


def test_a_batch_can_be_polled_and_cancelled_by_the_callers_own_batch_id():
    # A caller whose submit reply was lost knows only its idempotency key.
    batch = _submit(_body(batch_id="ptp-lost-reply"))
    row = score_batches.owned_row("ptp-lost-reply", INTERNAL)
    assert row["batch_id"] == batch["batch_id"]
    cancelled = score_batches.cancel("ptp-lost-reply", INTERNAL)
    assert cancelled["state"] == "cancelled"
    assert cancelled["batch_id"] == batch["batch_id"]
    with pytest.raises(HTTPException):
        score_batches.owned_row("ptp-lost-reply", {"id": 999})


def test_a_request_whose_slot_opens_after_the_deadline_is_unfinished_not_sent():
    batch = _submit()
    runner = score_batches.ScoreBatchRunner(send=lambda *a: pytest.fail("must not send"))
    work = score_batches._Batch(runner, runner._claim(time.time()))
    work.stop_at = time.time() - 1
    with pytest.raises(score_batches._DeadlineReached):
        work.send({"messages": []})
    assert score_batches.get_row(batch["batch_id"])["state"] == "queued"


def test_checkpoint_batches_reject_trained_stops_that_break_tool_calls(monkeypatch):
    from types import SimpleNamespace

    from flash.serve.request.openai import parse_chat_request
    from flash.server.routes import serving_chat

    spec = SimpleNamespace(train=SimpleNamespace(stop_sequences=("<tool_call>",)))

    def resolve(run_id, payload, key, org, project):
        request = parse_chat_request(payload, require_model=False, allow_managed_selectors=True)
        return request, [], spec, payload["checkpoint_id"], "org"

    monkeypatch.setattr(serving_chat, "_resolve_chat_request", resolve)
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
    body = {k: v for k, v in _body(2).items() if k != "model"}
    body["requests"][1]["tools"] = [tool]
    body["requests"][1]["tool_choice"] = "auto"
    with pytest.raises(HTTPException) as exc:
        _submit({**body, "checkpoint_id": "flash-1-abc/final"})
    assert exc.value.status_code == 400
    assert "request 2" in exc.value.detail


def test_base_batches_keep_the_external_keys_identity_for_serving_admission(monkeypatch):
    from flash.serve.request import transport

    external = {"id": 3, "auth_kind": "freesolo_api_key", "org_id": "org-9", "api_key_id": "k-1"}
    batch = score_batches.submit(_body(), external, None, None)
    row = score_batches.get_row(batch["batch_id"])
    assert row["api_key_id"] == "k-1"
    assert score_batches.get_row(_submit()["batch_id"])["api_key_id"] is None
    captured = {}
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-secret")
    monkeypatch.setattr(
        transport, "request_chat_json", lambda _c, **kwargs: captured.update(kwargs) or {}
    )
    monkeypatch.setattr(transport, "serving_openai_base_url", lambda: "https://serving/v1")
    score_batches._send_request(row, external, {"messages": [{"role": "user", "content": "x"}]})
    assert captured["headers"]["X-Freesolo-Acting-Api-Key-Id"] == "k-1"


def test_serving_applies_the_acting_keys_limits_to_trusted_internal_traffic():
    from flash.serving.src.accounting.usage import TrustedInternalAuthorization
    from flash.serving.src.http.context import require_attributed_traffic

    target = type("Target", (), {"is_checkpoint": False, "org_id": None})()
    traffic = require_attributed_traffic(
        TrustedInternalAuthorization(org_id="org-9", api_key_id="k-1"), target
    )
    assert traffic.api_key_id == "k-1"
    assert traffic.principal.orgId == "org-9"
    plain = require_attributed_traffic(TrustedInternalAuthorization(org_id="org-9"), target)
    assert plain.api_key_id is None


def test_null_concurrency_is_a_bad_request_not_a_server_error():
    with pytest.raises(HTTPException) as exc:
        _submit(_body(concurrency=None))
    assert exc.value.status_code == 400
    assert "concurrency" in exc.value.detail
    # An explicit null deadline still means "no deadline".
    assert _submit(_body(deadline_seconds=None))["deadline_seconds"] is None


def test_thinking_base_batches_reject_logprobs_before_anything_runs():
    body = _body(thinking=True)
    body["requests"][1]["logprobs"] = True
    with pytest.raises(HTTPException) as exc:
        _submit(body)
    assert "request 2" in exc.value.detail
    assert "thinking" in exc.value.detail
    body["thinking"] = False
    assert _submit(body)["state"] == "queued"


def test_a_request_waiting_for_a_slot_does_not_start_after_cancel():
    batch = _submit()
    runner = score_batches.ScoreBatchRunner(send=lambda *a: pytest.fail("must not send"))
    row = runner._claim(time.time())
    work = score_batches._Batch(runner, row)
    score_batches.cancel(batch["batch_id"], INTERNAL)
    with pytest.raises(score_batches._Cancelled):
        work.send({"messages": []})


def test_a_plane_that_lost_the_batch_cannot_publish_or_replace_results():
    batch = _submit()
    batch_id = batch["batch_id"]
    old = score_batches.ScoreBatchRunner(send=lambda row, key, request: _ok(request))
    old_work = score_batches._Batch(old, old._claim(time.time()))
    old.release()
    new = score_batches.ScoreBatchRunner(send=lambda row, key, request: _ok(request))
    _drain(new)
    first = _result(batch_id)["results"]
    with pytest.raises(score_batches._Cancelled):
        old_work._record(0, "case-0", response={"late": True})
    # Even a record that slipped in is ignored: the first record per request wins.
    with (score_batches._batch_dir(batch_id) / "results.jsonl").open("a") as handle:
        handle.write(json.dumps({"index": 0, "response": {"late": True}}) + "\n")
    assert _result(batch_id)["results"] == first
    assert "owner" not in first[0]


def test_a_torn_final_line_is_trimmed_before_a_resumed_batch_appends():
    batch = _submit()
    batch_id = batch["batch_id"]
    path = score_batches._batch_dir(batch_id) / "results.jsonl"
    path.write_text(
        json.dumps({"index": 0, "id": "case-0", "response": {"kept": True}}) + '\n{"index": 1, "res'
    )
    _drain(score_batches.ScoreBatchRunner(send=lambda row, key, request: _ok(request)))
    done = _result(batch_id)
    assert done["results"][0]["response"] == {"kept": True}
    assert "response" in done["results"][1]
    assert "response" in done["results"][2]
    assert all(json.loads(line) for line in path.read_text().splitlines())
