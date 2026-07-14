from __future__ import annotations

import http.client
import http.server
import io
import json
import threading
import urllib.error

import pytest

from flash.engine.recipe import FORWARD_TEACHER_MODEL_ID, FORWARD_TEACHER_URL
from flash.engine.worker.forward_teacher import (
    FORWARD_TEACHER_REALIZED_LOGPROB_ABS_TOLERANCE,
    FORWARD_TEACHER_TERMINAL,
    FORWARD_TEACHER_THINK_BOUNDARY,
    ForwardTeacherClient,
    ForwardTeacherError,
    ForwardTeacherRecordKind,
    ForwardTeacherTransientError,
)


def _content_logprobs():
    return [
        {
            "token": "invented",
            "logprob": -0.3,
            "top_logprobs": [{"token": "invented", "logprob": -0.3}],
        },
        {
            "token": " rationale",
            "logprob": -0.4,
            "top_logprobs": [{"token": " rationale", "logprob": -0.4}],
        },
        {
            "token": "</think>",
            "logprob": -0.01,
            "top_logprobs": [{"token": "</think>", "logprob": -0.01}],
        },
        {
            "token": "visible",
            "logprob": -0.1,
            "top_logprobs": [
                {"token": "visible", "logprob": -0.1},
                {"token": "shown", "logprob": -2.4},
            ],
        },
        {
            "token": " answer",
            "logprob": -0.2,
            "top_logprobs": [
                {"token": " answer", "logprob": -0.2},
                {"token": " response", "logprob": -1.9},
            ],
        },
        {
            "token": FORWARD_TEACHER_TERMINAL,
            "logprob": -0.01,
            "top_logprobs": [{"token": FORWARD_TEACHER_TERMINAL, "logprob": -0.01}],
        },
    ]


def _records_with_visible(content):
    records = _content_logprobs()
    return [
        *records[:3],
        {"token": content, "logprob": -0.1, "top_logprobs": [{"token": content, "logprob": -0.1}]},
        records[-1],
    ]


def _choice(**overrides):
    choice = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": "visible answer", "reasoning": "invented rationale"},
        "logprobs": {"content": _content_logprobs()},
    }
    choice.update(overrides)
    return choice


def _payload(**overrides):
    payload = {
        "model": FORWARD_TEACHER_MODEL_ID,
        "choices": [_choice()],
        "usage": {"prompt_tokens": 7, "completion_tokens": 6, "total_tokens": 13},
    }
    payload.update(overrides)
    return payload


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class _RawResponse(_Response):
    def read(self):
        return self.payload


@pytest.mark.parametrize("seed", [None, True, -1, 2**31, 1.5, "7"])
def test_forward_teacher_rejects_invalid_seed(seed):
    with pytest.raises(ForwardTeacherError, match="seed is invalid"):
        ForwardTeacherClient("key", seed=seed)


def test_forward_teacher_pins_request_and_returns_visible_usage():
    captured = {}

    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["auth"] = request.headers["Authorization"]
        captured["timeout"] = timeout
        return _Response(_payload())

    messages = [
        {"role": "system", "content": "follow the schema", "metadata": {"mode": "strict"}},
        {"role": "user", "content": "hello", "name": "requester"},
    ]
    result = ForwardTeacherClient(
        "secret-key", seed=123, opener=opener, clock=iter([10.0, 10.25]).__next__
    ).generate(messages)

    assert captured["url"] == FORWARD_TEACHER_URL
    assert captured["body"] == {
        "model": FORWARD_TEACHER_MODEL_ID,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.2,
        "seed": 123,
        "logprobs": True,
        "top_logprobs": 20,
    }
    assert captured["auth"] == "Bearer secret-key"
    assert captured["timeout"] == 90.0
    assert result.content == "visible answer"
    assert result.finish_reason == "stop"
    assert len(result.content_logprobs) == 6
    assert result.content_logprobs[3].token == "visible"
    assert result.content_logprobs[3].top_logprobs[1].logprob == pytest.approx(-2.4)
    assert result.reasoning == "invented rationale"
    assert (result.prompt_tokens, result.completion_tokens, result.total_tokens) == (7, 6, 13)
    assert result.latency_seconds == pytest.approx(0.25)
    assert result.attempts == 1
    assert result.ambiguous_paid_requests == 0


def test_forward_teacher_caps_request_timeout_at_run_deadline(monkeypatch):
    from flash.engine.worker import forward_teacher as forward_teacher_mod

    captured = {}
    monkeypatch.setenv("FLASH_RUN_DEADLINE_AT", "102.5")
    monkeypatch.setattr(forward_teacher_mod.time, "time", lambda: 100.0)

    def opener(_request, timeout):
        captured["timeout"] = timeout
        return _Response(_payload())

    ForwardTeacherClient(
        "key",
        seed=123,
        opener=opener,
        clock=iter([10.0, 10.25]).__next__,
    ).generate([{"role": "user", "content": "x"}])

    assert captured["timeout"] == pytest.approx(2.5)


def test_forward_teacher_retry_wait_stops_at_run_deadline(monkeypatch):
    from flash.engine.worker import forward_teacher as forward_teacher_mod

    now = [100.0]
    timeouts = []
    sleeps = []
    monkeypatch.setenv("FLASH_RUN_DEADLINE_AT", "101.0")
    monkeypatch.setattr(forward_teacher_mod.time, "time", lambda: now[0])

    def opener(_request, timeout):
        timeouts.append(timeout)
        raise urllib.error.URLError("private provider response")

    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    with pytest.raises(ForwardTeacherTransientError, match="run wall deadline") as caught:
        ForwardTeacherClient(
            "key",
            seed=123,
            opener=opener,
            sleep=sleep,
            clock=lambda: now[0],
        ).generate([{"role": "user", "content": "private prompt"}])

    assert timeouts == [1.0]
    assert sleeps == [1.0]
    assert caught.value.attempts == 1
    assert caught.value.latency_seconds == pytest.approx(1.0)
    assert caught.value.ambiguous_paid_requests == 1
    assert "private" not in str(caught.value)


@pytest.mark.parametrize("content", ["  visible answer", "visible answer  ", "visible answer\n"])
def test_forward_teacher_preserves_visible_content_boundary_whitespace(content):
    result = ForwardTeacherClient(
        "key",
        seed=123,
        opener=lambda *_a, **_k: _Response(
            _payload(
                choices=[
                    _choice(
                        message={"content": content, "reasoning": "invented rationale"},
                        logprobs={"content": _records_with_visible(content)},
                    )
                ],
                usage={"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
            )
        ),
    ).generate([{"role": "user", "content": "x"}])

    assert result.content == content


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_forward_teacher_default_transport_rejects_redirect_before_forwarding_authorization(
    monkeypatch, status
):
    from flash.engine.worker import forward_teacher as forward_teacher_mod

    requests = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append((self.path, self.headers.get("Authorization")))
            if self.path == "/source":
                self.send_response(status)
                self.send_header("Location", "/redirected")
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()

        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            self.send_response(200)
            self.end_headers()

        def log_message(self, _format, *_args):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(
        forward_teacher_mod,
        "FORWARD_TEACHER_URL",
        f"http://127.0.0.1:{server.server_port}/source",
    )
    try:
        with pytest.raises(ForwardTeacherError, match=f"HTTP {status}"):
            ForwardTeacherClient("key", seed=123).generate([{"role": "user", "content": "x"}])
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert [path for path, _authorization in requests] == ["/source"]
    assert requests[0][1] is not None


def test_forward_teacher_paces_outbound_attempt_starts_without_sleeping_first():
    now = [100.0]
    sleeps = []
    starts = []

    def clock():
        return now[0]

    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    def opener(_request, timeout):
        starts.append(now[0])
        return _Response(_payload())

    client = ForwardTeacherClient("key", seed=123, opener=opener, sleep=sleep, clock=clock)
    first = client.generate([{"role": "user", "content": "first"}])
    now[0] += 0.25
    second = client.generate([{"role": "user", "content": "second"}])

    assert starts == [100.0, 102.0]
    assert sleeps == [1.75]
    assert first.latency_seconds == pytest.approx(0.0)
    assert second.latency_seconds == pytest.approx(1.75)


@pytest.mark.parametrize(
    "messages",
    [
        [],
        ["not-a-message"],
        [{"role": "", "content": "x"}],
        [{"role": "user", "content": object()}],
        [{"role": "user", "content": "x", "metadata": object()}],
    ],
)
def test_forward_teacher_rejects_malformed_requests_without_opening_transport(messages):
    with pytest.raises(ForwardTeacherError, match="messages are invalid"):
        ForwardTeacherClient("key", seed=123, opener=lambda *_a, **_k: pytest.fail("must not open")).generate(
            messages
        )


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (_payload(model="deepseek-v4-pro"), "pinned model"),
        (_payload(choices=[]), "exactly one"),
        (_payload(choices=[{**_payload()["choices"][0], "index": 1}]), "index"),
        (_payload(choices=[{**_payload()["choices"][0], "finish_reason": "length"}]), "naturally"),
        (_payload(choices=[{**_payload()["choices"][0], "message": {"content": "  "}}]), "empty"),
        (
            _payload(usage={"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 10}),
            "arithmetic",
        ),
    ],
)
def test_forward_teacher_rejects_alias_length_empty_and_malformed_contract(payload, match):
    with pytest.raises(ForwardTeacherError, match=match) as caught:
        ForwardTeacherClient(
            "key",
            seed=123,
            opener=lambda *_a, **_k: _Response(payload),
            clock=iter([10.0, 10.25]).__next__,
        ).generate([{"role": "user", "content": "x"}])

    assert caught.value.attempts == 1
    assert caught.value.latency_seconds == pytest.approx(0.25)
    assert caught.value.ambiguous_paid_requests == 1
    assert set(vars(caught.value)) == {
        "attempts",
        "latency_seconds",
        "ambiguous_paid_requests",
    }
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def _http_error(code, retry_after=None):
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return urllib.error.HTTPError(FORWARD_TEACHER_URL, code, "failure", headers, io.BytesIO(b"sensitive"))


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_forward_teacher_never_retries_permanent_http_errors(code):
    calls = []

    def opener(*_a, **_k):
        calls.append(code)
        raise _http_error(code)

    with pytest.raises(ForwardTeacherError, match=f"HTTP {code}") as caught:
        ForwardTeacherClient(
            "key",
            seed=123,
            opener=opener,
            sleep=lambda _s: pytest.fail("must not sleep"),
            clock=iter([20.0, 20.4]).__next__,
        ).generate([{"role": "user", "content": "private prompt"}])
    assert calls == [code]
    assert caught.value.retriable is False
    assert caught.value.attempts == 1
    assert caught.value.latency_seconds == pytest.approx(0.4)
    assert caught.value.ambiguous_paid_requests == 0


@pytest.mark.parametrize("code", [408, 409, 425, 429, 500, 599])
def test_forward_teacher_retries_transient_http_errors_once(code):
    calls = []

    def opener(*_a, **_k):
        calls.append(code)
        if len(calls) == 1:
            raise _http_error(code)
        return _Response(_payload())

    result = ForwardTeacherClient("key", seed=123, opener=opener, sleep=lambda _s: None).generate(
        [{"role": "user", "content": "x"}]
    )
    assert calls == [code, code]
    assert result.attempts == 2
    assert result.ambiguous_paid_requests == 1


def test_forward_teacher_preserves_retry_ambiguity_before_permanent_http_failure():
    calls = []

    def opener(*_a, **_k):
        calls.append(1)
        if len(calls) == 1:
            raise _http_error(503)
        raise _http_error(400)

    with pytest.raises(ForwardTeacherError, match="HTTP 400") as caught:
        ForwardTeacherClient(
            "key",
            seed=123,
            opener=opener,
            sleep=lambda _delay: None,
            clock=iter([30.0, 32.0, 32.5]).__next__,
        ).generate([{"role": "user", "content": "private prompt"}])

    assert calls == [1, 1]
    assert caught.value.retriable is False
    assert caught.value.attempts == 2
    assert caught.value.latency_seconds == pytest.approx(2.5)
    assert caught.value.ambiguous_paid_requests == 1
    assert "private prompt" not in str(caught.value)


@pytest.mark.parametrize(
    ("retry_after", "expected"), [("2.5", 2.5), ("0", 1.0), ("99", 60.0), ("date", 10.0)]
)
def test_forward_teacher_retries_once_and_parses_only_delta_retry_after(retry_after, expected):
    now = [100.0]
    starts = []
    sleeps = []

    def clock():
        return now[0]

    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    def opener(*_a, **_k):
        starts.append(now[0])
        if len(starts) == 1:
            raise _http_error(429, retry_after)
        return _Response(_payload())

    result = ForwardTeacherClient("key", seed=123, opener=opener, sleep=sleep, clock=clock).generate(
        [{"role": "user", "content": "x"}]
    )
    assert len(starts) == 2
    assert starts[1] - starts[0] == pytest.approx(max(2.0, expected))
    assert sum(sleeps) == pytest.approx(max(2.0, expected))
    assert result.latency_seconds == pytest.approx(max(2.0, expected))
    assert result.attempts == 2


def test_forward_teacher_retry_after_one_still_waits_two_seconds_between_attempt_starts():
    now = [25.0]
    starts = []

    def sleep(delay):
        now[0] += delay

    def opener(*_args, **_kwargs):
        starts.append(now[0])
        if len(starts) == 1:
            raise _http_error(429, "1")
        return _Response(_payload())

    result = ForwardTeacherClient("key", seed=123, opener=opener, sleep=sleep, clock=lambda: now[0]).generate(
        [{"role": "user", "content": "x"}]
    )

    assert starts == [25.0, 27.0]
    assert result.attempts == 2
    assert result.latency_seconds == pytest.approx(2.0)


@pytest.mark.parametrize("code", [408, 409, 425, 429, 500, 599])
def test_forward_teacher_exhausted_transient_http_error_is_typed_and_bounded(code):
    calls = []

    def opener(*_a, **_k):
        calls.append(code)
        raise _http_error(code)

    with pytest.raises(ForwardTeacherTransientError, match=f"HTTP {code}") as caught:
        ForwardTeacherClient(
            "key",
            seed=123,
            opener=opener,
            sleep=lambda _s: None,
            clock=iter([30.0, 32.0, 32.75]).__next__,
        ).generate([{"role": "user", "content": "private prompt"}])

    assert caught.value.retriable is True
    assert caught.value.attempts == 2
    assert caught.value.latency_seconds == pytest.approx(2.75)
    assert caught.value.ambiguous_paid_requests == 2
    assert calls == [code, code]


@pytest.mark.parametrize("first_body", [b"not-json", b'"\xff"'])
def test_forward_teacher_retries_http_200_decode_failures_with_start_pacing(first_body):
    now = [10.0]
    starts = []
    sleeps = []

    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    def opener(*_args, **_kwargs):
        starts.append(now[0])
        if len(starts) == 1:
            return _RawResponse(first_body)
        return _Response(_payload())

    result = ForwardTeacherClient("key", seed=123, opener=opener, sleep=sleep, clock=lambda: now[0]).generate(
        [{"role": "user", "content": "x"}]
    )

    assert starts == [10.0, 20.0]
    assert sleeps == [10.0]
    assert result.attempts == 2
    assert result.latency_seconds == pytest.approx(10.0)
    assert result.ambiguous_paid_requests == 1


@pytest.mark.parametrize("body", [b"not-json", b'"\xff"'])
def test_forward_teacher_exhausted_http_200_decode_failure_is_sanitized_and_transient(body):
    calls = []

    def opener(*_args, **_kwargs):
        calls.append(1)
        return _RawResponse(body)

    with pytest.raises(ForwardTeacherTransientError, match="response decoding failure") as caught:
        ForwardTeacherClient("key", seed=123, opener=opener, sleep=lambda _delay: None).generate(
            [{"role": "user", "content": "private prompt"}]
        )

    assert calls == [1, 1]
    assert caught.value.attempts == 2
    assert caught.value.latency_seconds >= 0
    assert caught.value.ambiguous_paid_requests == 2
    assert "private prompt" not in str(caught.value)
    assert body.decode("utf-8", errors="ignore") not in str(caught.value)


def test_forward_teacher_preserves_transport_retry_ambiguity_on_success():
    calls = []

    def opener(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.URLError("sensitive transport detail")
        return _Response(_payload())

    result = ForwardTeacherClient("key", seed=123, opener=opener, sleep=lambda _delay: None).generate(
        [{"role": "user", "content": "private prompt"}]
    )

    assert calls == [1, 1]
    assert result.attempts == 2
    assert result.ambiguous_paid_requests == 1


def test_forward_teacher_incomplete_read_remains_transient_and_bounded():
    calls = []

    def opener(*_args, **_kwargs):
        calls.append(1)
        raise http.client.IncompleteRead(b"sensitive partial response", 99)

    with pytest.raises(ForwardTeacherTransientError, match="transport failure") as caught:
        ForwardTeacherClient("key", seed=123, opener=opener, sleep=lambda _delay: None).generate(
            [{"role": "user", "content": "private prompt"}]
        )

    assert calls == [1, 1]
    assert caught.value.attempts == 2
    assert caught.value.latency_seconds >= 0
    assert caught.value.ambiguous_paid_requests == 2
    assert "sensitive" not in str(caught.value)
    assert "private prompt" not in str(caught.value)


@pytest.mark.parametrize(
    "choice",
    [
        _choice(logprobs=None),
        _choice(logprobs={"content": []}),
        _choice(logprobs={"content": [{"token": "x", "logprob": 0.1, "top_logprobs": []}]}),
        _choice(
            logprobs={
                "content": [
                    {
                        "token": "x",
                        "logprob": -0.1,
                        "top_logprobs": [{"token": 7, "logprob": -0.1}],
                    }
                ]
            }
        ),
    ],
)
def test_forward_teacher_rejects_malformed_content_top_logprobs_without_text_leakage(choice):
    private = "private answer never leak"
    choice = {
        **choice,
        "message": {"content": private, "reasoning": "invented rationale"},
    }
    with pytest.raises(ForwardTeacherError, match="logprobs") as caught:
        ForwardTeacherClient(
            "key", seed=123, opener=lambda *_a, **_k: _Response(_payload(choices=[choice]))
        ).generate([{"role": "user", "content": "x"}])
    assert private not in str(caught.value)


def test_forward_teacher_accepts_and_classifies_pinned_semantic_completion_contract():
    payload = _payload()

    assert "completion_tokens_details" not in payload["usage"]
    result = ForwardTeacherClient("key", seed=123, opener=lambda *_a, **_k: _Response(payload)).generate(
        [{"role": "user", "content": "x"}]
    )

    parsed = result.parsed_completion
    assert tuple(record.kind for record in parsed.records) == (
        ForwardTeacherRecordKind.HIDDEN_REASONING,
        ForwardTeacherRecordKind.HIDDEN_REASONING,
        ForwardTeacherRecordKind.THINK_BOUNDARY,
        ForwardTeacherRecordKind.VISIBLE_CONTENT,
        ForwardTeacherRecordKind.VISIBLE_CONTENT,
        ForwardTeacherRecordKind.TERMINAL,
    )
    assert "".join(record.token for record in parsed.hidden_reasoning_records) == result.reasoning
    assert "".join(record.token for record in parsed.visible_content_records) == result.content
    assert parsed.boundary_record.token == FORWARD_TEACHER_THINK_BOUNDARY
    assert parsed.terminal_record.token == FORWARD_TEACHER_TERMINAL
    assert len(result.content_logprobs) == result.completion_tokens == 6
    diagnostic = repr(result)
    assert result.content not in diagnostic
    assert result.reasoning not in diagnostic
    assert all(record.token not in diagnostic for record in parsed.records)


def test_forward_teacher_allows_literal_think_boundaries_in_reasoning_and_content():
    tokens = (
        "explain ",
        FORWARD_TEACHER_THINK_BOUNDARY,
        " literally",
        FORWARD_TEACHER_THINK_BOUNDARY,
        "answer ",
        FORWARD_TEACHER_THINK_BOUNDARY,
        " text",
        FORWARD_TEACHER_TERMINAL,
    )
    records = [
        {
            "token": token,
            "logprob": -0.1,
            "top_logprobs": [{"token": token, "logprob": -0.1}],
        }
        for token in tokens
    ]
    payload = _payload(
        choices=[
            _choice(
                message={
                    "reasoning": "".join(tokens[:3]),
                    "content": "".join(tokens[4:7]),
                },
                logprobs={"content": records},
            )
        ],
        usage={"prompt_tokens": 7, "completion_tokens": 8, "total_tokens": 15},
    )

    result = ForwardTeacherClient(
        "key", seed=123, opener=lambda *_args, **_kwargs: _Response(payload)
    ).generate([{"role": "user", "content": "x"}])

    parsed = result.parsed_completion
    assert parsed.boundary_record.index == 3
    assert parsed.hidden_reasoning_records[1].kind is ForwardTeacherRecordKind.HIDDEN_REASONING
    assert parsed.visible_content_records[1].kind is ForwardTeacherRecordKind.VISIBLE_CONTENT


def test_forward_teacher_rejects_completion_logprob_count_mismatch():
    with pytest.raises(ForwardTeacherError, match="token count is inconsistent"):
        ForwardTeacherClient(
            "key",
            seed=123,
            opener=lambda *_a, **_k: _Response(
                _payload(choices=[_choice(logprobs={"content": _content_logprobs()[:1]})])
            ),
        ).generate([{"role": "user", "content": "x"}])


def test_forward_teacher_transport_is_bounded_and_exception_is_sanitized():
    secret = "secret-key-never-leak"
    prompt = "private prompt never leak"
    calls = []

    def opener(*_a, **_k):
        calls.append(1)
        raise urllib.error.URLError(f"transport included {secret} {prompt}")

    with pytest.raises(ForwardTeacherTransientError) as caught:
        ForwardTeacherClient(secret, seed=123, opener=opener, sleep=lambda _s: None).generate(
            [{"role": "user", "content": prompt}]
        )
    assert len(calls) == 2
    text = str(caught.value)
    assert secret not in text
    assert prompt not in text
    assert "transport failure" in text


@pytest.mark.parametrize(
    ("records", "message", "match"),
    [
        (
            [
                record
                for record in _content_logprobs()
                if record["token"] != FORWARD_TEACHER_THINK_BOUNDARY
            ],
            {"content": "visible answer", "reasoning": "invented rationale"},
            "boundary count",
        ),
        (
            [*_content_logprobs()[:3], _content_logprobs()[2], *_content_logprobs()[3:]],
            {"content": "visible answer", "reasoning": "invented rationale"},
            "boundary count",
        ),
        (
            [
                *_content_logprobs()[:-1],
                {
                    **_content_logprobs()[-1],
                    "token": "wrong terminal",
                    "top_logprobs": [{"token": "wrong terminal", "logprob": -0.01}],
                },
            ],
            {"content": "visible answerwrong terminal", "reasoning": "invented rationale"},
            "terminal count",
        ),
        (
            _content_logprobs(),
            {"content": "different visible content", "reasoning": "invented rationale"},
            "content reconstruction mismatch",
        ),
        (
            _content_logprobs(),
            {"content": "visible answer", "reasoning": "different reasoning"},
            "reasoning reconstruction mismatch",
        ),
    ],
)
def test_forward_teacher_semantic_parser_rejects_structural_deviations_without_text_leakage(
    records, message, match
):
    private = "synthetic private marker"
    payload = _payload(
        choices=[_choice(message=message, logprobs={"content": records})],
        usage={
            "prompt_tokens": 7,
            "completion_tokens": len(records),
            "total_tokens": 7 + len(records),
        },
        private_marker=private,
    )

    with pytest.raises(ForwardTeacherError, match=match) as caught:
        ForwardTeacherClient("key", seed=123, opener=lambda *_a, **_k: _Response(payload)).generate(
            [{"role": "user", "content": private}]
        )

    assert private not in str(caught.value)
    assert all(record["token"] not in str(caught.value) for record in records)


@pytest.mark.parametrize(
    ("record_index", "field", "value", "match"),
    [
        (0, "logprob", 0.1, "content logprobs"),
        (0, "logprob", float("inf"), "content logprobs"),
        (3, "candidate_logprob", 0.1, "top logprobs"),
        (3, "candidate_logprob", float("nan"), "top logprobs"),
    ],
)
def test_forward_teacher_semantic_parser_rejects_positive_and_nonfinite_logprobs(
    record_index, field, value, match
):
    records = _content_logprobs()
    if field == "logprob":
        records[record_index] = {**records[record_index], "logprob": value}
    else:
        candidates = list(records[record_index]["top_logprobs"])
        candidates[0] = {**candidates[0], "logprob": value}
        records[record_index] = {**records[record_index], "top_logprobs": candidates}

    with pytest.raises(ForwardTeacherError, match=match):
        ForwardTeacherClient(
            "key",
            seed=123,
            opener=lambda *_a, **_k: _Response(
                _payload(choices=[_choice(logprobs={"content": records})])
            ),
        ).generate([{"role": "user", "content": "x"}])


@pytest.mark.parametrize(
    ("top_logprobs", "match"),
    [
        ([{"token": "shown", "logprob": -2.4}], "top-logprob count"),
        (
            [
                {
                    "token": "visible",
                    "logprob": -0.1 - 2 * FORWARD_TEACHER_REALIZED_LOGPROB_ABS_TOLERANCE,
                },
                {"token": "shown", "logprob": -2.4},
            ],
            "does not match",
        ),
    ],
)
def test_forward_teacher_rejects_absent_or_mismatched_realized_top_logprob_without_leakage(
    top_logprobs, match
):
    records = _content_logprobs()
    records[3] = {**records[3], "top_logprobs": top_logprobs}

    with pytest.raises(ForwardTeacherError, match=match) as caught:
        ForwardTeacherClient(
            "key",
            seed=123,
            opener=lambda *_a, **_k: _Response(
                _payload(choices=[_choice(logprobs={"content": records})])
            ),
        ).generate([{"role": "user", "content": "private prompt"}])

    diagnostic = str(caught.value)
    assert "private prompt" not in diagnostic
    assert all(record["token"] not in diagnostic for record in records)


def test_forward_teacher_accepts_realized_logprob_within_json_float_tolerance():
    records = _content_logprobs()
    candidates = list(records[3]["top_logprobs"])
    candidates[0] = {
        **candidates[0],
        "logprob": records[3]["logprob"] + FORWARD_TEACHER_REALIZED_LOGPROB_ABS_TOLERANCE / 2,
    }
    records[3] = {**records[3], "top_logprobs": candidates}

    result = ForwardTeacherClient(
        "key",
        seed=123,
        opener=lambda *_a, **_k: _Response(
            _payload(choices=[_choice(logprobs={"content": records})])
        ),
    ).generate([{"role": "user", "content": "x"}])

    assert result.parsed_completion.visible_content_records[0].token == "visible"


def test_forward_teacher_rejects_duplicate_exact_teacher_tokens_and_oversized_top_k():
    duplicate_records = _content_logprobs()
    duplicate_records[3] = {
        **duplicate_records[3],
        "top_logprobs": [
            {"token": "same", "logprob": -0.7},
            {"token": "same", "logprob": -0.8},
        ],
    }
    with pytest.raises(ForwardTeacherError, match="duplicate token strings"):
        ForwardTeacherClient(
            "key",
            seed=123,
            opener=lambda *_a, **_k: _Response(
                _payload(choices=[_choice(logprobs={"content": duplicate_records})])
            ),
        ).generate([{"role": "user", "content": "x"}])

    oversized_records = _content_logprobs()
    oversized_records[3] = {
        **oversized_records[3],
        "top_logprobs": [{"token": f"candidate-{index}", "logprob": -10.0} for index in range(21)],
    }
    with pytest.raises(ForwardTeacherError, match="content logprobs"):
        ForwardTeacherClient(
            "key",
            seed=123,
            opener=lambda *_a, **_k: _Response(
                _payload(choices=[_choice(logprobs={"content": oversized_records})])
            ),
        ).generate([{"role": "user", "content": "x"}])
