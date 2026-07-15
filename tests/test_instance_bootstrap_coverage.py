"""Focused hermetic coverage for the shared instance bootstrap (Vast/Lambda worker container).

These target under-covered helpers in ``flash.providers._instance_bootstrap``: payload loading,
code-prefix validation, the Hugging Face transient-retry machinery (status/Retry-After parsing +
backoff), the HF upload/exists/fetch wrappers, ``run_mode``'s subprocess tee (success + wall-clock
timeout), the attempt-marker writer, the preload wall-cap watchdog ``_fire`` path, and ``main()``'s
preload branch. Everything is CPU-only and offline: the huggingface_hub package is stubbed via
sys.modules, subprocess.Popen is faked, and targeted real multiprocessing children exercise uploader
reaping without making network requests.
"""

from __future__ import annotations

import builtins
import json
import multiprocessing
import signal
import subprocess
import sys
import threading
import time
import types

import pytest

from flash.providers import _instance_bootstrap as b
from flash.providers._deadline import deadline_kwargs

CODE_PREFIX = "code/0123456789abcdef0123456789abcdef/flash"


def _sleeping_upload_child():
    time.sleep(60.0)


def _sigterm_ignoring_final_upload(payload, _console, _mode, _extra):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    payload["ready"].set()
    while True:
        time.sleep(1.0)


class _InspectableProcess:
    def __init__(self, process):
        self._process = process
        self.close_called = False

    def start(self):
        self._process.start()

    def join(self, timeout=None):
        self._process.join(timeout)

    def is_alive(self):
        return self._process.is_alive()

    def terminate(self):
        self._process.terminate()

    def kill(self):
        self._process.kill()

    def close(self):
        self.close_called = True

    def close_real(self):
        self._process.close()


def _install_fake_hf(monkeypatch, **attrs):
    """Replace ``huggingface_hub`` in sys.modules with a stub exposing only ``attrs``."""
    hub = types.ModuleType("huggingface_hub")
    for name, value in attrs.items():
        setattr(hub, name, value)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return hub


class _HFError(Exception):
    """An HTTPError-shaped exception (carries .response.status_code/.headers)."""


def _err(status=None, headers=None):
    """An exception shaped like an httpx/requests HTTPError with .response.status_code/.headers."""
    exc = _HFError("boom")
    if status is not None or headers is not None:
        exc.response = types.SimpleNamespace(status_code=status, headers=headers or {})
    return exc


# ---------------------------------------------------------------------------
# load_payload
# ---------------------------------------------------------------------------
def test_load_payload_reads_json_from_payload_path(tmp_path, monkeypatch):
    p = tmp_path / "payload.json"
    p.write_text(json.dumps({"phase": "sft", "seed": 7, "nested": {"a": 1}}))
    monkeypatch.setattr(b, "PAYLOAD_PATH", str(p))
    got = b.load_payload()
    assert got == {"phase": "sft", "seed": 7, "nested": {"a": 1}}


@pytest.mark.parametrize("bad", [None, True, 0, -1, float("nan"), float("inf"), float("-inf")])
def test_require_deadline_at_rejects_invalid_or_nonpositive_values(monkeypatch, bad):
    monkeypatch.setattr(b.time, "time", lambda: 100.0)
    with pytest.raises(RuntimeError, match="deadline is invalid"):
        b.require_deadline_at({"deadline_at": bad})


def test_require_deadline_at_rejects_expired_deadline(monkeypatch):
    monkeypatch.setattr(b.time, "time", lambda: 100.0)
    with pytest.raises(TimeoutError, match="exceeded before bootstrap"):
        b.require_deadline_at(
            {"deadline_at": 100.0, "run_created_at": 40.0, "run_max_wall_seconds": 60.0}
        )


def test_require_deadline_at_validates_canonical_submission_deadline(monkeypatch):
    monkeypatch.setattr(b.time, "time", lambda: 100.0)
    payload = {
        "deadline_at": 150.0,
        "run_created_at": 50.0,
        "run_max_wall_seconds": 100.0,
    }

    assert b.require_deadline_at(payload) == 150.0


@pytest.mark.parametrize(
    "payload",
    [
        {"deadline_at": 150.0, "run_created_at": 50.0},
        {"deadline_at": 150.0, "run_max_wall_seconds": 100.0},
        {"deadline_at": 150.0, "run_created_at": True, "run_max_wall_seconds": 100.0},
        {"deadline_at": 150.0, "run_created_at": 50.0, "run_max_wall_seconds": True},
        {"deadline_at": 150.0, "run_created_at": float("nan"), "run_max_wall_seconds": 100.0},
        {"deadline_at": 150.0, "run_created_at": 50.0, "run_max_wall_seconds": float("inf")},
        {"deadline_at": 151.0, "run_created_at": 50.0, "run_max_wall_seconds": 100.0},
    ],
)
def test_require_deadline_at_rejects_invalid_canonical_fields(monkeypatch, payload):
    monkeypatch.setattr(b.time, "time", lambda: 100.0)

    with pytest.raises(RuntimeError, match="deadline"):
        b.require_deadline_at(payload)


def test_require_deadline_at_rejects_unsafe_current_clock(monkeypatch):
    monkeypatch.setattr(b.time, "time", lambda: float("nan"))

    with pytest.raises(RuntimeError, match="clock is invalid"):
        b.require_deadline_at(
            {"deadline_at": 150.0, "run_created_at": 50.0, "run_max_wall_seconds": 100.0}
        )


def test_deadline_kwargs_forwards_supported_values_including_none():
    def accepts_deadline(*, deadline_at):
        return deadline_at

    assert deadline_kwargs(accepts_deadline, 123.0) == {"deadline_at": 123.0}
    assert deadline_kwargs(accepts_deadline, None) == {"deadline_at": None}


def test_deadline_kwargs_omits_keyword_for_legacy_callable():
    def legacy(value=None):
        return value

    assert deadline_kwargs(legacy, 123.0) == {}
    assert deadline_kwargs(legacy, None) == {}


def test_deadline_kwargs_forwards_to_variadic_callable():
    def variadic(**kwargs):
        return kwargs

    assert deadline_kwargs(variadic, 123.0) == {"deadline_at": 123.0}
    assert deadline_kwargs(variadic, None) == {"deadline_at": None}


def test_deadline_kwargs_fails_closed_for_uninspectable_callable():
    class _Uninspectable:
        @property
        def __signature__(self):
            raise ValueError("signature unavailable")

        def __call__(self, **kwargs):
            return kwargs

    assert deadline_kwargs(_Uninspectable(), 123.0) == {"deadline_at": 123.0}
    assert deadline_kwargs(_Uninspectable(), None) == {"deadline_at": None}


# ---------------------------------------------------------------------------
# _code_prefix validation
# ---------------------------------------------------------------------------
def test_code_prefix_rejects_missing_and_malformed():
    # Missing / blank / non-str -> "missing code_prefix".
    for bad in ({}, {"code_prefix": ""}, {"code_prefix": "   "}, {"code_prefix": 123}):
        with pytest.raises(ValueError, match="missing code_prefix"):
            b._code_prefix(bad)

    # Present but structurally invalid -> "invalid code_prefix".
    invalid = [
        "code/deadbeef/flash",  # digest too short
        "code/0123456789abcdef0123456789abcdeZ/flash",  # non-hex char (Z)
        "notcode/0123456789abcdef0123456789abcdef/flash",  # wrong root segment
        "code/0123456789abcdef0123456789abcdef/notflash",  # wrong tail segment
        "code/0123456789abcdef0123456789abcdef",  # only two segments
    ]
    for prefix in invalid:
        with pytest.raises(ValueError, match="invalid code_prefix"):
            b._code_prefix({"code_prefix": prefix})

    # The valid form round-trips (leading/trailing slashes stripped).
    assert b._code_prefix({"code_prefix": "/" + CODE_PREFIX + "/"}) == CODE_PREFIX


# ---------------------------------------------------------------------------
# _hf_status_code
# ---------------------------------------------------------------------------
def test_hf_status_code_parses_and_falls_back_to_none():
    assert b._hf_status_code(_err(status="503")) == 503  # numeric-string coerces to int
    assert b._hf_status_code(_err(status=429)) == 429
    # No response attribute at all -> int(None) -> TypeError -> None.
    assert b._hf_status_code(_err()) is None
    # Unparseable status -> ValueError -> None.
    assert b._hf_status_code(_err(status="not-a-number")) is None


# ---------------------------------------------------------------------------
# _hf_retry_after
# ---------------------------------------------------------------------------
def test_hf_retry_after_numeric_clamped_and_case_insensitive():
    # plain integer seconds
    assert b._hf_retry_after(_err(headers={"retry-after": "30"})) == 30.0
    # clamped to the 60s ceiling
    assert b._hf_retry_after(_err(headers={"retry-after": "120"})) == 60.0
    # negative clamps up to 0
    assert b._hf_retry_after(_err(headers={"retry-after": "-5"})) == 0.0
    # header key match is case-insensitive (dict .get miss -> items() fallback)
    assert b._hf_retry_after(_err(headers={"Retry-After": "45"})) == 45.0
    # no header / no response -> None
    assert b._hf_retry_after(_err(headers={})) is None
    assert b._hf_retry_after(_err()) is None


def test_hf_retry_after_http_date_variants():
    # RFC-1123 date with an explicit zone (tz-aware) far in the future -> clamped to 60.
    assert b._hf_retry_after(_err(headers={"retry-after": "Wed, 21 Oct 2099 07:28:00 GMT"})) == 60.0
    # Naive RFC date (no zone) exercises the tzinfo-is-None -> assume-UTC branch, still future -> 60.
    assert b._hf_retry_after(_err(headers={"retry-after": "Wed, 21 Oct 2099 07:28:00"})) == 60.0
    # A past date -> negative delta clamps to 0.
    assert b._hf_retry_after(_err(headers={"retry-after": "Mon, 01 Jan 2001 00:00:00 GMT"})) == 0.0
    # Neither a float nor a parseable date -> None (both inner excepts).
    assert b._hf_retry_after(_err(headers={"retry-after": "garbage-not-a-date"})) is None


# ---------------------------------------------------------------------------
# _hf_call retry machinery
# ---------------------------------------------------------------------------
def test_hf_call_retries_transient_then_raises_and_passes_nontransient_through(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(b.time, "sleep", lambda s: sleeps.append(s))

    # Persistent transient (503) with a Retry-After header: retried until the delay list is exhausted,
    # then the final exception propagates. Retry-After (5s) is honored for every backoff.
    attempts = {"n": 0}

    def always_503():
        attempts["n"] += 1
        raise _err(status=503, headers={"retry-after": "5"})

    with pytest.raises(_HFError) as ei:
        b._hf_call(always_503, "list")
    assert "boom" in str(ei.value)
    assert attempts["n"] == len(b._HF_RETRY_DELAYS_S) + 1  # initial try + one per delay
    assert sleeps == [5.0] * len(b._HF_RETRY_DELAYS_S)  # Retry-After overrides the default schedule

    # A non-transient status (400) is re-raised on the very first attempt with no sleep.
    sleeps.clear()
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise _err(status=400)

    with pytest.raises(_HFError):
        b._hf_call(bad_request, "list")
    assert calls["n"] == 1
    assert sleeps == []

    # A transient error that later succeeds returns the success value (default backoff schedule).
    sleeps.clear()
    seq = iter([_err(status=502, headers={}), "ok-result"])

    def flaky():
        nxt = next(seq)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    assert b._hf_call(flaky, "download") == "ok-result"
    assert sleeps == [b._HF_RETRY_DELAYS_S[0]]  # one default-scheduled backoff before the retry


def test_hf_call_caps_retry_sleep_at_deadline(monkeypatch):
    clock = {"now": 100.0}
    sleeps = []
    attempts = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    def always_503():
        attempts.append(True)
        raise _err(status=503, headers={"retry-after": "5"})

    monkeypatch.setattr(b.time, "time", lambda: clock["now"])
    monkeypatch.setattr(b.time, "sleep", sleep)

    with pytest.raises(TimeoutError, match="deadline"):
        b._hf_call(always_503, "list", deadline_at=102.0)

    assert attempts == [True]
    assert sleeps == [2.0]


# ---------------------------------------------------------------------------
# hf_upload / hf_file_exists / fetch_spec_from_hf
# ---------------------------------------------------------------------------
def test_hf_upload_targets_prefixed_path_and_swallows_errors(monkeypatch):
    recorded = {}

    class _Api:
        def __init__(self, token=None):
            recorded["token"] = token

        def upload_file(self, **kw):
            recorded.update(kw)

    _install_fake_hf(monkeypatch, HfApi=_Api)
    payload = {"hf_repo": "org/repo", "hf_prefix": "sft/run", "env": {"HF_TOKEN": "hf-tok"}}
    assert b.hf_upload(payload, "/tmp/x.txt", "console.txt") is None
    assert recorded["token"] == "hf-tok"
    assert recorded["path_or_fileobj"] == "/tmp/x.txt"
    assert recorded["path_in_repo"] == "sft/run/console.txt"
    assert recorded["repo_id"] == "org/repo"
    assert recorded["repo_type"] == "dataset"

    # A raising upload is swallowed (never raises) — best-effort live-log push.
    class _BoomApi:
        def __init__(self, token=None):
            pass

        def upload_file(self, **kw):
            raise RuntimeError("hf 500")

    _install_fake_hf(monkeypatch, HfApi=_BoomApi)
    assert b.hf_upload(payload, "/tmp/x.txt", "console.txt") is None


def test_hf_upload_starts_no_request_at_deadline(monkeypatch):
    calls = []

    class _Api:
        def __init__(self, token=None):
            calls.append("init")

    _install_fake_hf(monkeypatch, HfApi=_Api)
    monkeypatch.setattr(b.time, "time", lambda: 200.0)
    payload = {
        "hf_repo": "o/r",
        "hf_prefix": "p",
        "env": {},
        "deadline_at": 200.0,
        "run_created_at": 100.0,
        "run_max_wall_seconds": 100.0,
    }

    assert b.hf_upload(payload, "/tmp/x.txt", "console.txt") is None
    assert calls == []


def test_hf_file_exists_delegates_to_api(monkeypatch):
    seen = {}

    class _Api:
        def __init__(self, token=None):
            seen["token"] = token

        def file_exists(self, **kw):
            seen.update(kw)
            return kw["filename"].endswith("DONE")

    _install_fake_hf(monkeypatch, HfApi=_Api)
    payload = {"hf_repo": "o/r", "hf_prefix": "p", "env": {"HF_TOKEN": "t"}}
    assert b.hf_file_exists(payload, "DONE") is True
    assert seen["filename"] == "p/DONE"
    assert seen["repo_id"] == "o/r"
    assert seen["repo_type"] == "dataset"
    assert seen["token"] == "t"
    assert b.hf_file_exists(payload, "metrics.json") is False


def test_hf_file_exists_starts_no_request_at_deadline(monkeypatch):
    calls = []

    class _Api:
        def __init__(self, token=None):
            calls.append("init")

    _install_fake_hf(monkeypatch, HfApi=_Api)
    monkeypatch.setattr(b.time, "time", lambda: 200.0)
    payload = {
        "hf_repo": "o/r",
        "hf_prefix": "p",
        "env": {},
        "deadline_at": 200.0,
        "run_created_at": 100.0,
        "run_max_wall_seconds": 100.0,
    }

    with pytest.raises(TimeoutError, match="deadline"):
        b.hf_file_exists(payload, "DONE")
    assert calls == []


def test_fetch_spec_from_hf_returns_downloaded_file_contents(tmp_path, monkeypatch):
    local = tmp_path / "job_spec.json"
    local.write_text('{"spilled": true}')
    seen = {}

    def _dl(**kw):
        seen.update(kw)
        return str(local)

    _install_fake_hf(monkeypatch, hf_hub_download=_dl)
    created_at = b.time.time()
    payload = {
        "hf_repo": "o/r",
        "hf_prefix": "sft/run",
        "env": {"HF_TOKEN": "tok"},
        "deadline_at": created_at + 60.0,
        "run_created_at": created_at,
        "run_max_wall_seconds": 60.0,
    }
    assert b.fetch_spec_from_hf(payload) == '{"spilled": true}'
    assert seen["filename"] == "sft/run/job_spec.json"
    assert seen["repo_type"] == "dataset"
    assert seen["token"] == "tok"


def test_install_extra_pip_starts_no_process_at_deadline(monkeypatch):
    monkeypatch.setattr(b.time, "time", lambda: 200.0)
    monkeypatch.setattr(
        b.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("pip process must not start at the deadline"),
    )
    payload = {
        "extra_pip": ["private-package"],
        "env": {},
        "deadline_at": 200.0,
        "run_created_at": 100.0,
        "run_max_wall_seconds": 100.0,
    }

    with pytest.raises(TimeoutError, match="deadline"):
        b.install_extra_pip(payload)


# ---------------------------------------------------------------------------
# fetch_code: empty listing is a hard error
# ---------------------------------------------------------------------------
def test_fetch_code_raises_when_no_files_under_prefix(monkeypatch):
    class _Api:
        def __init__(self, token=None):
            pass

        def list_repo_tree(self, **kw):
            # Only directory/size-less entries -> filtered out -> no downloadable files.
            return [
                types.SimpleNamespace(path=CODE_PREFIX, size=None),
                types.SimpleNamespace(path=None, size=10),
            ]

    def _dl(**kw):  # pragma: no cover - must never be reached with an empty file set
        raise AssertionError("should not download when no files are listed")

    _install_fake_hf(monkeypatch, HfApi=_Api, hf_hub_download=_dl)
    created_at = b.time.time()
    payload = {
        "hf_repo": "org/repo",
        "code_prefix": CODE_PREFIX,
        "env": {"HF_TOKEN": "t"},
        "deadline_at": created_at + 60.0,
        "run_created_at": created_at,
        "run_max_wall_seconds": 60.0,
    }
    with pytest.raises(RuntimeError, match="no flash code files"):
        b.fetch_code(payload)


# ---------------------------------------------------------------------------
# run_mode: subprocess tee (success + wall-clock timeout)
# ---------------------------------------------------------------------------
class _StoppedUploader:
    def join(self, timeout=None):
        return None

    def is_alive(self):
        return False

    def close(self):
        return None


class _FakeProc:
    def __init__(self, lines, rc=0, timeout_once=False):
        self.stdout = iter(lines)
        self.returncode = rc
        self._timeout_once = timeout_once
        self._waits = 0
        self.killed = False

    def wait(self, timeout=None):
        self._waits += 1
        if self._timeout_once and self._waits == 1:
            raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout)
        return self.returncode

    def kill(self):
        self.killed = True


class _RaisingWaitProc:
    def __init__(self, error):
        self.args = ["worker"]
        self.stdout = iter(())
        self.returncode = None
        self.error = error

    def wait(self, timeout=None):
        raise self.error

    def kill(self):
        pytest.fail("non-timeout wait errors must not kill the worker")


class _ReapTrackedUploader:
    def __init__(self):
        self.alive = True
        self.close_called = False

    def join(self, timeout=None):
        return None

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.alive = False

    def kill(self):
        self.alive = False

    def close(self):
        assert self.alive is False
        self.close_called = True


def _disable_periodic_console_upload(monkeypatch):
    monkeypatch.setattr(
        b,
        "_start_console_uploader",
        lambda *_args: (_StoppedUploader(), threading.Event()),
    )


def _run_final_console_upload_inline(
    payload,
    console,
    mode,
    extra,
    upload_deadline_at,
    reaping_deadline_at,
):
    assert upload_deadline_at < reaping_deadline_at
    assert upload_deadline_at > b.time.time()
    b._upload_console_snapshot(payload, console, mode, extra)
    return True


def test_run_mode_success_returns_rc_and_uploads_console(monkeypatch):
    uploads: list[tuple] = []
    popen_calls = []
    _disable_periodic_console_upload(monkeypatch)
    monkeypatch.setattr(b, "_upload_console_tail_bounded", _run_final_console_upload_inline)
    monkeypatch.setattr(b, "hf_upload", lambda p, path, sub: uploads.append((path, sub)))
    proc = _FakeProc(["hello\n", "world\n"], rc=0)

    def popen(*args, **kwargs):
        popen_calls.append((args, kwargs))
        return proc

    monkeypatch.setattr(b.subprocess, "Popen", popen)

    payload = {"hf_repo": "o/r", "hf_prefix": "sft/run", "env": {}, "code_prefix": CODE_PREFIX}
    deadline = b.time.time() + 100
    rc = b.run_mode(payload, {"E": "1"}, "sft", deadline_ts=deadline)
    assert rc == 0
    assert popen_calls[0][0][0] == [sys.executable, "-m", "flash.engine.worker_entrypoint"]
    upload_deadline, _reaping_deadline = b._upload_cleanup_deadlines(deadline)
    expected_worker_deadline = b._worker_execution_deadline(upload_deadline)
    assert float(popen_calls[0][1]["env"]["FLASH_RUN_DEADLINE_AT"]) == pytest.approx(
        expected_worker_deadline
    )
    # the console tee is uploaded under console_<mode>.txt and captured the child's stdout.
    assert uploads
    assert uploads[-1][1] == "console_sft.txt"
    with open("/tmp/console_sft.txt") as f:
        body = f.read()
    assert "hello" in body
    assert "world" in body


@pytest.mark.parametrize(
    "wait_error",
    [SystemExit(17), KeyboardInterrupt("interrupted")],
    ids=["system-exit", "keyboard-interrupt"],
)
def test_run_mode_reaps_uploader_before_base_exception_propagates(monkeypatch, wait_error):
    uploader = _ReapTrackedUploader()
    stop_upload = threading.Event()
    monkeypatch.setattr(
        b,
        "_start_console_uploader",
        lambda *_args: (uploader, stop_upload),
    )
    monkeypatch.setattr(
        b.subprocess,
        "Popen",
        lambda *_args, **_kwargs: _RaisingWaitProc(wait_error),
    )

    with pytest.raises(type(wait_error)) as raised:
        b.run_mode({}, {}, "sft", deadline_ts=b.time.time() + 100)

    assert raised.value is wait_error
    assert stop_upload.is_set()
    assert uploader.is_alive() is False
    assert uploader.close_called is True


@pytest.mark.parametrize(
    "wait_error",
    [SystemExit(17), KeyboardInterrupt("interrupted"), RuntimeError("wait failed")],
    ids=["system-exit", "keyboard-interrupt", "unexpected-exception"],
)
def test_wait_error_reaps_uploader_before_propagating_to_terminal_marker(
    monkeypatch, wait_error
):
    uploader = _ReapTrackedUploader()
    stop_upload = threading.Event()
    marker_states = []
    created_at = b.time.time()
    payload = {
        "phase": "sft",
        "run_created_at": created_at,
        "run_max_wall_seconds": 100.0,
        "deadline_at": created_at + 100.0,
    }

    class _Timer:
        def cancel(self):
            return None

    monkeypatch.setattr(b, "load_payload", lambda: payload)
    monkeypatch.setattr(
        b,
        "arm_deadline_watchdog",
        lambda *_args: (_Timer(), threading.Event()),
    )
    monkeypatch.setattr(b, "install_extra_pip", lambda _payload: None)
    monkeypatch.setattr(b, "fetch_code", lambda _payload: None)
    monkeypatch.setattr(b, "build_worker_env", lambda _payload: {})
    monkeypatch.setattr(
        b.subprocess,
        "Popen",
        lambda *_args, **_kwargs: _RaisingWaitProc(wait_error),
    )
    monkeypatch.setattr(
        b,
        "_start_console_uploader",
        lambda *_args: (uploader, stop_upload),
    )
    monkeypatch.setattr(
        b,
        "_upload_console_tail_bounded",
        lambda *_args: pytest.fail("final upload must not run after a wait error"),
    )
    monkeypatch.setattr(
        b,
        "write_attempt_marker",
        lambda _payload, ok, error="", retriable=False: marker_states.append(
            (ok, error, retriable, uploader.is_alive(), uploader.close_called)
        ),
    )

    assert b.main() == 1

    assert stop_upload.is_set()
    assert uploader.is_alive() is False
    assert uploader.close_called is True
    assert len(marker_states) == 1
    ok, error, retriable, uploader_alive, uploader_closed = marker_states[0]
    assert ok is False
    assert type(wait_error).__name__ in error
    assert retriable is False
    assert uploader_alive is False
    assert uploader_closed is True


def test_run_mode_reaps_final_uploader_before_terminal_marker_reserve(monkeypatch):
    clock = {"now": 100.0}
    deadline = (
        clock["now"]
        + b._CONSOLE_UPLOAD_STOP_TIMEOUT_S
        + b._CONSOLE_UPLOAD_FINAL_TIMEOUT_S
        + b._CONSOLE_UPLOAD_REAP_RESERVE_S
        + b._TERMINAL_BOOKKEEPING_RESERVE_S
        + 1.0
    )
    _upload_deadline, reaping_deadline = b._upload_cleanup_deadlines(deadline)
    events = []
    _disable_periodic_console_upload(monkeypatch)

    class _TimedFinalUploader:
        def __init__(self):
            self.alive = False

        def start(self):
            self.alive = True
            events.append(("start", clock["now"]))

        def join(self, timeout=None):
            events.append(("join", timeout))
            clock["now"] += timeout or 0.0

        def is_alive(self):
            return self.alive

        def terminate(self):
            events.append(("terminate", clock["now"]))

        def kill(self):
            events.append(("kill", clock["now"]))
            self.alive = False

        def close(self):
            events.append(("close", clock["now"]))

    uploader = _TimedFinalUploader()

    class _Context:
        def Process(self, **kwargs):
            assert kwargs["target"] is b._upload_console_snapshot
            return uploader

    monkeypatch.setattr(b.time, "time", lambda: clock["now"])
    monkeypatch.setattr(b.multiprocessing, "get_context", lambda _method: _Context())
    monkeypatch.setattr(
        b.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakeProc(["done\n"], rc=0),
    )
    payload = {"hf_repo": "o/r", "hf_prefix": "sft/run", "env": {}, "code_prefix": CODE_PREFIX}

    assert b.run_mode(payload, {}, "sft", deadline_ts=deadline) == 0

    marker_started_at = clock["now"]
    events.append(("marker", marker_started_at, uploader.is_alive()))
    assert not uploader.is_alive()
    assert [event[1] for event in events if event[0] == "join"] == pytest.approx(
        [
            b._CONSOLE_UPLOAD_FINAL_TIMEOUT_S,
            b._CONSOLE_UPLOAD_TERMINATE_TIMEOUT_S,
            b._CONSOLE_UPLOAD_TERMINATE_TIMEOUT_S,
        ]
    )
    assert marker_started_at <= reaping_deadline
    assert deadline - marker_started_at >= b._TERMINAL_BOOKKEEPING_RESERVE_S
    assert [event[0] for event in events].index("kill") < [event[0] for event in events].index("marker")
    assert events[-1] == ("marker", marker_started_at, False)


class _DelayedOutput:
    def __init__(self):
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == 0:
            self._index += 1
            return "early\n"
        if self._index == 1:
            self._index += 1
            threading.Event().wait(1.1)
            return "final-after-delay\n"
        raise StopIteration


class _TrackedConsole:
    def __init__(self, file, final_write_attempt, late_writes):
        self._file = file
        self._final_write_attempt = final_write_attempt
        self._late_writes = late_writes
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._closed = True
        return self._file.__exit__(exc_type, exc, tb)

    def __getattr__(self, name):
        return getattr(self._file, name)

    def write(self, text):
        if "final-after-delay" in text:
            self._final_write_attempt.set()
            if self._closed:
                self._late_writes.append(text)
        return self._file.write(text)


def test_run_mode_drains_delayed_terminal_output_before_upload(monkeypatch):
    console = "/tmp/console_sft.txt"
    _disable_periodic_console_upload(monkeypatch)
    monkeypatch.setattr(b, "_upload_console_tail_bounded", _run_final_console_upload_inline)
    final_write_attempt = threading.Event()
    late_writes = []
    uploads = []
    original_open = builtins.open

    def tracked_open(path, mode="r", *args, **kwargs):
        file = original_open(path, mode, *args, **kwargs)
        if path == console and mode == "w":
            return _TrackedConsole(file, final_write_attempt, late_writes)
        return file

    def upload(_payload, path, subpath):
        with original_open(path, encoding="utf-8") as file:
            uploads.append((subpath, file.read()))

    monkeypatch.setattr(b, "open", tracked_open, raising=False)
    monkeypatch.setattr(b, "hf_upload", upload)
    proc = _FakeProc(_DelayedOutput(), rc=0)
    monkeypatch.setattr(b.subprocess, "Popen", lambda *args, **kwargs: proc)
    payload = {"hf_repo": "o/r", "hf_prefix": "sft/run", "env": {}, "code_prefix": CODE_PREFIX}

    assert b.run_mode(payload, {}, "sft", deadline_ts=b.time.time() + 20) == 0

    assert final_write_attempt.wait(2.0)
    assert late_writes == []
    assert uploads[-1][0] == "console_sft.txt"
    assert "final-after-delay" in uploads[-1][1]


def test_run_mode_reaps_periodic_uploader_before_terminal_marker_reserve(monkeypatch):
    clock = {"now": 100.0}
    deadline = (
        clock["now"]
        + b._CONSOLE_UPLOAD_STOP_TIMEOUT_S
        + b._CONSOLE_UPLOAD_FINAL_TIMEOUT_S
        + b._CONSOLE_UPLOAD_REAP_RESERVE_S
        + b._TERMINAL_BOOKKEEPING_RESERVE_S
        + 1.0
    )
    _upload_deadline, reaping_deadline = b._upload_cleanup_deadlines(deadline)
    events = []

    class _HungUploader:
        def __init__(self):
            self.alive = True

        def join(self, timeout=None):
            events.append(("join", timeout))
            clock["now"] += timeout or 0.0

        def is_alive(self):
            return self.alive

        def terminate(self):
            events.append(("terminate", clock["now"]))

        def kill(self):
            events.append(("kill", clock["now"]))
            self.alive = False

        def close(self):
            events.append(("close", clock["now"]))

    uploader = _HungUploader()
    stop_upload = threading.Event()
    monkeypatch.setattr(b.time, "time", lambda: clock["now"])
    monkeypatch.setattr(
        b,
        "_start_console_uploader",
        lambda *_args: (uploader, stop_upload),
    )
    monkeypatch.setattr(b, "_upload_console_tail_bounded", lambda *_args: True)
    monkeypatch.setattr(
        b.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakeProc(["hello\n"], rc=0),
    )
    payload = {"hf_repo": "o/r", "hf_prefix": "sft/run", "env": {}, "code_prefix": CODE_PREFIX}

    assert b.run_mode(payload, {}, "sft", deadline_ts=deadline) == 0

    marker_started_at = clock["now"]
    events.append(("marker", marker_started_at, uploader.is_alive()))
    assert stop_upload.is_set()
    assert not uploader.is_alive()
    assert [event[1] for event in events if event[0] == "join"] == pytest.approx(
        [
            b._CONSOLE_UPLOAD_STOP_TIMEOUT_S,
            b._CONSOLE_UPLOAD_TERMINATE_TIMEOUT_S,
            b._CONSOLE_UPLOAD_TERMINATE_TIMEOUT_S,
        ]
    )
    assert marker_started_at <= reaping_deadline
    assert deadline - marker_started_at >= b._TERMINAL_BOOKKEEPING_RESERVE_S
    assert [event[0] for event in events].index("kill") < [event[0] for event in events].index("marker")
    assert events[-1] == ("marker", marker_started_at, False)


@pytest.mark.parametrize("_iteration", range(3))
def test_run_mode_reserves_cleanup_before_watchdog_marker_with_real_timing(
    monkeypatch, _iteration
):
    stop_timeout = 0.03
    final_timeout = 0.03
    terminate_timeout = 0.02
    reap_reserve = 2 * terminate_timeout
    bookkeeping_reserve = 0.04
    monkeypatch.setattr(b, "_CONSOLE_UPLOAD_STOP_TIMEOUT_S", stop_timeout)
    monkeypatch.setattr(b, "_CONSOLE_UPLOAD_FINAL_TIMEOUT_S", final_timeout)
    monkeypatch.setattr(b, "_CONSOLE_UPLOAD_TERMINATE_TIMEOUT_S", terminate_timeout)
    monkeypatch.setattr(b, "_CONSOLE_UPLOAD_REAP_RESERVE_S", reap_reserve)
    monkeypatch.setattr(b, "_TERMINAL_BOOKKEEPING_RESERVE_S", bookkeeping_reserve)
    monkeypatch.setattr(b.signal, "signal", lambda *_args: None)

    events = []
    started_at = time.time()
    deadline = started_at + 0.30
    upload_deadline, reaping_deadline = b._upload_cleanup_deadlines(deadline)
    worker_cutoff = upload_deadline - stop_timeout - final_timeout

    class _TimedWorker:
        def __init__(self):
            self.args = ["worker"]
            self.stdout = iter(())
            self.returncode = None
            self.wait_timeouts = []
            self.killed_at = None

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            if self.killed_at is None:
                time.sleep(timeout or 0.0)
                raise subprocess.TimeoutExpired(self.args, timeout)
            self.returncode = -9
            return self.returncode

        def kill(self):
            self.killed_at = time.time()
            events.append(("worker-killed", self.killed_at))

    class _TimedUploader:
        def __init__(self):
            self.alive = True
            self.closed = False

        def join(self, timeout=None):
            time.sleep(timeout or 0.0)

        def is_alive(self):
            return self.alive

        def terminate(self):
            events.append(("uploader-terminated", time.time()))
            self.alive = False

        def kill(self):
            events.append(("uploader-killed", time.time()))
            self.alive = False

        def close(self):
            self.closed = True
            events.append(("uploader-closed", time.time()))

    class _RecordingStopEvent:
        def __init__(self):
            self.set_at = None

        def set(self):
            self.set_at = time.time()
            events.append(("uploader-stop", self.set_at))

    worker = _TimedWorker()
    uploader = _TimedUploader()
    stop_upload = _RecordingStopEvent()
    markers = []
    payload = {
        "hf_repo": "org/repo",
        "job_spec_json": "{}",
        "phase": "sft",
        "seed": 0,
        "flash_arm": "lambda",
        "env": {},
        "extra_pip": [],
        "hf_prefix": "sft/run",
        "code_prefix": CODE_PREFIX,
        "run_id": "run",
        "run_created_at": started_at,
        "run_max_wall_seconds": deadline - started_at,
        "deadline_at": deadline,
        "attempt": 0,
    }

    def record_marker(kind):
        markers.append((kind, time.time(), uploader.is_alive()))

    monkeypatch.setattr(b, "load_payload", lambda: payload)
    monkeypatch.setattr(b, "install_extra_pip", lambda _payload: None)
    monkeypatch.setattr(b, "fetch_code", lambda _payload: None)
    monkeypatch.setattr(b, "build_worker_env", lambda _payload: {})
    monkeypatch.setattr(b.subprocess, "Popen", lambda *_args, **_kwargs: worker)
    monkeypatch.setattr(
        b,
        "_start_console_uploader",
        lambda *_args: (uploader, stop_upload),
    )
    monkeypatch.setattr(b, "_upload_console_tail_bounded", lambda *_args: True)
    monkeypatch.setattr(
        b,
        "_publish_timeout_marker_then_exit",
        lambda *_args: record_marker("watchdog"),
    )
    monkeypatch.setattr(
        b,
        "write_attempt_marker",
        lambda *_args, **_kwargs: record_marker("terminal"),
    )

    assert b.main() == 1

    assert worker.killed_at is not None
    assert worker.killed_at <= worker_cutoff + 0.02
    assert stop_upload.set_at is not None
    assert worker.killed_at <= stop_upload.set_at
    assert uploader.is_alive() is False
    assert uploader.closed is True
    assert markers
    assert all(uploader_alive is False for _, _, uploader_alive in markers)
    terminal_markers = [marker for marker in markers if marker[0] == "terminal"]
    assert len(terminal_markers) == 1
    assert deadline - terminal_markers[0][1] >= bookkeeping_reserve - 0.01
    assert terminal_markers[0][1] <= reaping_deadline


def test_stop_upload_process_reaps_real_sleeping_child_before_marker_reserve():
    context = multiprocessing.get_context("spawn")
    process = _InspectableProcess(
        context.Process(target=_sleeping_upload_child, daemon=True)
    )
    stop_upload = context.Event()
    process.start()
    try:
        upload_deadline = time.time() + 0.25
        reaping_deadline = upload_deadline + b._CONSOLE_UPLOAD_REAP_RESERVE_S

        clean = b._stop_upload_process(
            process,
            stop_upload,
            upload_deadline,
            reaping_deadline,
        )
        finished_at = time.time()

        assert clean is False
        assert stop_upload.is_set()
        assert process.is_alive() is False
        assert process.close_called is True
        assert finished_at < reaping_deadline
    finally:
        if process.is_alive():
            process.kill()
            process.join(b._CONSOLE_UPLOAD_TERMINATE_TIMEOUT_S)
        process.close_real()


def test_final_upload_reaps_real_sigterm_ignoring_child_before_marker_reserve(monkeypatch):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    processes = []

    class _RecordingContext:
        def Process(self, **kwargs):
            process = _InspectableProcess(context.Process(**kwargs))
            processes.append(process)
            return process

    monkeypatch.setattr(b.multiprocessing, "get_context", lambda _method: _RecordingContext())
    monkeypatch.setattr(b, "_upload_console_snapshot", _sigterm_ignoring_final_upload)
    upload_deadline = time.time() + 2.0
    reaping_deadline = upload_deadline + b._CONSOLE_UPLOAD_REAP_RESERVE_S

    try:
        clean = b._upload_console_tail_bounded(
            {"ready": ready},
            "/tmp/unused-console.txt",
            "sft",
            "",
            upload_deadline,
            reaping_deadline,
        )
        finished_at = time.time()
        process = processes[0]

        assert ready.is_set()
        assert clean is False
        assert process.is_alive() is False
        assert process.close_called is True
        assert finished_at < reaping_deadline
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(b._CONSOLE_UPLOAD_TERMINATE_TIMEOUT_S)
            process.close_real()


def test_unreaped_uploader_exits_before_terminal_bookkeeping(monkeypatch):
    clock = {"now": 100.0}
    events = []

    class _HardExit(BaseException):
        pass

    class _NeverDeadUploader:
        def join(self, timeout=None):
            events.append(("join", timeout))
            clock["now"] += timeout or 0.0

        def is_alive(self):
            return True

        def terminate(self):
            events.append(("terminate", clock["now"]))

        def kill(self):
            events.append(("kill", clock["now"]))

        def close(self):
            pytest.fail("a live uploader must not be closed")

    def hard_exit(code):
        events.append(("exit", code, clock["now"]))
        raise _HardExit

    upload_deadline = clock["now"] + 0.5
    reaping_deadline = upload_deadline + b._CONSOLE_UPLOAD_REAP_RESERVE_S
    monkeypatch.setattr(b.time, "time", lambda: clock["now"])
    monkeypatch.setattr(b.os, "_exit", hard_exit)

    with pytest.raises(_HardExit):
        b._stop_upload_process(
            _NeverDeadUploader(),
            threading.Event(),
            upload_deadline,
            reaping_deadline,
        )

    assert clock["now"] == pytest.approx(reaping_deadline)
    assert events[-1] == ("exit", 124, reaping_deadline)
    assert [event[0] for event in events].count("kill") == 2


def test_run_mode_timeout_kills_child_and_raises(monkeypatch):
    _disable_periodic_console_upload(monkeypatch)
    monkeypatch.setattr(b, "_upload_console_tail_bounded", lambda *_args: True)
    monkeypatch.setattr(b, "hf_upload", lambda p, path, sub: None)
    proc = _FakeProc(["partial\n"], rc=0, timeout_once=True)
    monkeypatch.setattr(b.subprocess, "Popen", lambda *a, **k: proc)

    payload = {"hf_repo": "o/r", "hf_prefix": "sft/run", "env": {}, "code_prefix": CODE_PREFIX}
    with pytest.raises(TimeoutError, match="wall-clock cap"):
        b.run_mode(payload, {}, "grpo", deadline_ts=b.time.time() + 100)
    assert proc.killed is True  # the child was killed on the deadline


def test_run_mode_starts_no_subprocess_at_deadline(monkeypatch):
    monkeypatch.setattr(b.time, "time", lambda: 200.0)
    monkeypatch.setattr(
        b.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("worker process must not start at the deadline"),
    )
    payload = {"hf_repo": "o/r", "hf_prefix": "sft/run", "env": {}, "code_prefix": CODE_PREFIX}

    with pytest.raises(TimeoutError, match="wall-clock cap"):
        b.run_mode(payload, {}, "sft", deadline_ts=200.0)


def test_main_arms_same_absolute_deadline_before_setup_and_training(monkeypatch):
    events = []
    payload = {
        "hf_repo": "org/repo",
        "job_spec_json": "{}",
        "phase": "sft",
        "seed": 0,
        "flash_arm": "lambda",
        "env": {},
        "extra_pip": [],
        "hf_prefix": "sft/run",
        "code_prefix": CODE_PREFIX,
        "run_id": "run",
        "run_created_at": 100.0,
        "run_max_wall_seconds": 400.0,
        "deadline_at": 500.0,
        "attempt": 0,
    }

    class _Done:
        def set(self):
            events.append("deadline_done")

    class _Timer:
        def cancel(self):
            events.append("deadline_cancel")

    monkeypatch.setattr(b.time, "time", lambda: 100.0)
    monkeypatch.setattr(b, "load_payload", lambda: payload)
    monkeypatch.setattr(
        b,
        "arm_deadline_watchdog",
        lambda deadline, _payload: events.append(("watchdog", deadline)) or (_Timer(), _Done()),
    )
    monkeypatch.setattr(b, "install_extra_pip", lambda _payload: events.append("install"))
    monkeypatch.setattr(b, "fetch_code", lambda _payload: events.append("code"))
    monkeypatch.setattr(b, "build_worker_env", lambda _payload: {})
    monkeypatch.setattr(
        b,
        "run_mode",
        lambda _payload, _env, _phase, deadline: events.append(("training", deadline)) or 0,
    )
    monkeypatch.setattr(b.os.path, "exists", lambda path: path == "/tmp/metrics.json")
    monkeypatch.setattr(b, "write_attempt_marker", lambda *_args, **_kwargs: None)

    assert b.main() == 0
    assert events[:4] == [
        ("watchdog", 500.0),
        "install",
        "code",
        ("training", 500.0),
    ]
    assert events[-2:] == ["deadline_done", "deadline_cancel"]


@pytest.mark.parametrize("boundary", ["run_mode", "remote_confirmation"])
def test_main_accepts_required_completion_artifacts_at_deadline(monkeypatch, boundary):
    markers = []
    remote_checks = []
    clock = {"now": 100.0}
    payload = {
        "hf_repo": "org/repo",
        "job_spec_json": "{}",
        "phase": "sft",
        "seed": 0,
        "flash_arm": "lambda",
        "env": {},
        "extra_pip": [],
        "hf_prefix": "sft/run",
        "code_prefix": CODE_PREFIX,
        "run_id": "run",
        "run_created_at": 100.0,
        "run_max_wall_seconds": 100.0,
        "deadline_at": 200.0,
        "attempt": 0,
    }

    class _Done:
        def set(self):
            return None

    class _Timer:
        def cancel(self):
            return None

    def run_mode(*_args):
        clock["now"] = 200.0 if boundary == "run_mode" else 199.0
        return 1

    def remote_completion_confirmed(_payload):
        remote_checks.append(True)
        if boundary == "remote_confirmation":
            clock["now"] = 200.0
        return True

    monkeypatch.setattr(b.time, "time", lambda: clock["now"])
    monkeypatch.setattr(b, "load_payload", lambda: payload)
    monkeypatch.setattr(b, "arm_deadline_watchdog", lambda deadline, _payload: (_Timer(), _Done()))
    monkeypatch.setattr(b, "install_extra_pip", lambda _payload: None)
    monkeypatch.setattr(b, "fetch_code", lambda _payload: None)
    monkeypatch.setattr(b, "build_worker_env", lambda _payload: {})
    monkeypatch.setattr(b, "run_mode", run_mode)
    monkeypatch.setattr(b.os.path, "exists", lambda path: path == "/tmp/metrics.json")
    monkeypatch.setattr(b, "remote_completion_confirmed", remote_completion_confirmed)
    monkeypatch.setattr(
        b,
        "write_attempt_marker",
        lambda _payload, ok, error="", retriable=False: markers.append((ok, error, retriable)),
    )

    assert b.main() == 0
    assert remote_checks == [True]
    assert markers == [(True, "", False)]


# ---------------------------------------------------------------------------
# write_attempt_marker
# ---------------------------------------------------------------------------
def test_write_attempt_marker_truncates_error_and_uploads_arm_named(monkeypatch):
    uploads: list[tuple] = []
    monkeypatch.setattr(
        b,
        "hf_upload",
        lambda p, path, sub, **kwargs: uploads.append((path, sub, kwargs)),
    )
    monkeypatch.setattr(b.time, "time", lambda: 100.0)
    payload = {
        "hf_repo": "o/r",
        "hf_prefix": "p",
        "flash_arm": "vast",
        "attempt": 3,
        "run_id": "run-marker",
        "deadline_at": 200.0,
        "run_created_at": 100.0,
        "run_max_wall_seconds": 100.0,
        "env": {},
    }
    long_error = "E" * 3000
    b.write_attempt_marker(payload, ok=False, error=long_error, retriable=True)

    path, sub, kwargs = uploads[-1]
    assert path == "/tmp/attempt_marker.json"
    assert sub == "vast_attempt3.json"  # <arm>_attempt<N>.json
    assert kwargs == {"enforce_deadline": False}
    with open(path) as f:
        marker = json.load(f)
    assert marker["ok"] is False
    assert marker["retriable"] is True
    assert marker["attempt"] == 3
    assert marker["run_id"] == "run-marker"
    assert marker["ts"] == 100.0
    assert marker["error"] == long_error[-2000:]  # tail-truncated to 2000 chars
    assert len(marker["error"]) == 2000


def test_write_attempt_marker_preserves_success_after_deadline(monkeypatch):
    monkeypatch.setattr(b, "hf_upload", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(b.time, "time", lambda: 205.0)
    payload = {
        "hf_repo": "o/r",
        "hf_prefix": "p",
        "flash_arm": "vast",
        "attempt": 3,
        "run_id": "run-marker",
        "deadline_at": 200.0,
        "run_created_at": 100.0,
        "run_max_wall_seconds": 100.0,
        "env": {},
    }

    b.write_attempt_marker(payload, ok=True)

    with open("/tmp/attempt_marker.json") as f:
        marker = json.load(f)
    assert marker["ok"] is True
    assert marker["retriable"] is False
    assert marker["error"] == ""
    assert marker["ts"] == 205.0


def test_write_attempt_marker_rejects_noncanonical_deadline(monkeypatch):
    uploads = []
    monkeypatch.setattr(b, "hf_upload", lambda *_args: uploads.append(True))
    monkeypatch.setattr(b.time, "time", lambda: 250.0)
    payload = {
        "hf_repo": "o/r",
        "hf_prefix": "p",
        "flash_arm": "vast",
        "attempt": 3,
        "run_id": "run-marker",
        "deadline_at": 200.0,
        "run_created_at": 100.0,
        "run_max_wall_seconds": 99.0,
        "env": {},
    }

    with pytest.raises(RuntimeError, match="canonical submission deadline"):
        b.write_attempt_marker(payload, ok=False, error="safe", retriable=True)

    assert uploads == []


# ---------------------------------------------------------------------------
# deadline watchdogs: marker publication before hard exit
# ---------------------------------------------------------------------------
def test_arm_deadline_watchdog_publishes_marker_before_exit(monkeypatch):
    marks: list[tuple] = []
    exits: list[int] = []
    monkeypatch.setattr(b.time, "time", lambda: 100.0)
    monkeypatch.setattr(
        b, "write_attempt_marker", lambda p, ok, error="", **k: marks.append((ok, error))
    )
    monkeypatch.setattr(b.os, "_exit", lambda code: exits.append(code))
    payload = {
        "deadline_at": 160.0,
        "run_created_at": 100.0,
        "run_max_wall_seconds": 60.0,
        "flash_arm": "vast",
        "attempt": 0,
        "run_id": "run",
        "hf_repo": "o/r",
        "env": {},
    }

    timer, done = b.arm_deadline_watchdog(160.0, payload)
    timer.cancel()
    timer.function()

    assert not done.is_set()
    assert marks == [(False, "run wall deadline exceeded; self-terminating box")]
    assert exits == [124]


def test_deadline_watchdog_hard_exits_when_terminal_marker_writer_blocks(monkeypatch):
    writer_started = threading.Event()
    release_writer = threading.Event()
    exits = []

    def blocked_writer(*_args, **_kwargs):
        writer_started.set()
        release_writer.wait(2.0)

    monkeypatch.setattr(b, "_TERMINAL_MARKER_GRACE_S", 0.01)
    monkeypatch.setattr(b, "write_attempt_marker", blocked_writer)
    monkeypatch.setattr(b.os, "_exit", lambda code: exits.append(code))

    b._publish_timeout_marker_then_exit({}, "deadline reached")

    assert writer_started.is_set()
    assert exits == [124]
    release_writer.set()


# ---------------------------------------------------------------------------
# _arm_preload_wall_cap: the _fire watchdog path
# ---------------------------------------------------------------------------
def test_arm_preload_wall_cap_uses_remaining_absolute_deadline(monkeypatch):
    marks: list[tuple] = []
    exits: list[int] = []
    monkeypatch.setattr(b.time, "time", lambda: 100.0)
    monkeypatch.setattr(
        b, "write_attempt_marker", lambda p, ok, error="", **k: marks.append((ok, error))
    )
    monkeypatch.setattr(b.os, "_exit", lambda code: exits.append(code))

    payload = {
        "deadline_at": 160.0,
        "run_created_at": 100.0,
        "run_max_wall_seconds": 60.0,
        "flash_arm": "lambda",
        "attempt": 0,
        "hf_repo": "o/r",
        "env": {},
    }
    timer, _done = b._arm_preload_wall_cap(payload)
    timer.cancel()
    assert timer.interval == 60.0

    # manually invoke the watchdog closure: not done means marker then hard exit.
    timer.function()
    assert exits == [124]
    assert marks
    assert marks[-1][0] is False
    assert "run wall deadline" in marks[-1][1]

    # if the download already finished, the watchdog is a no-op.
    marks.clear()
    exits.clear()
    timer2, done2 = b._arm_preload_wall_cap(payload)
    timer2.cancel()
    done2.set()
    timer2.function()
    assert exits == []
    assert marks == []


# ---------------------------------------------------------------------------
# run_preload: per-model download failure is captured, not raised
# ---------------------------------------------------------------------------
def test_run_preload_records_download_failure(tmp_path, monkeypatch):
    (tmp_path / ".flash-cache-mounted").write_text("")  # real-mount sentinel

    def _snap(**k):
        if k.get("local_files_only"):
            raise FileNotFoundError("not cached")  # force the real download attempt
        raise RuntimeError("network exploded")  # the real download fails

    _install_fake_hf(monkeypatch, snapshot_download=_snap)
    cache_dir = str(tmp_path / "hf-cache" / "hub")
    r = b.run_preload(
        {
            "env": {"FLASH_WEIGHT_CACHE_DIR": cache_dir},
            "models": ["a/b"],
            "cache_mount_marker": ".flash-cache-mounted",
        }
    )
    assert r["preloaded"] == []
    assert r["already_cached"] == []
    assert r["failed"]["a/b"] == "RuntimeError: network exploded"


# ---------------------------------------------------------------------------
# main(): the preload branch
# ---------------------------------------------------------------------------
def _preload_payload(**over):
    created_at = b.time.time()
    base = {
        "mode": "preload",
        "hf_repo": "org/repo",
        "hf_prefix": "preload/run",
        "flash_arm": "lambda",
        "run_id": "preload-run",
        "seed": 0,
        "attempt": 0,
        "env": {},
        "run_created_at": created_at,
        "run_max_wall_seconds": 999.0,
        "deadline_at": created_at + 999.0,
    }
    base.update(over)
    return base


def test_main_preload_success_returns_zero_and_writes_ok_marker(monkeypatch):
    markers: list[tuple] = []
    monkeypatch.setattr(b, "load_payload", lambda path=b.PAYLOAD_PATH: _preload_payload())
    monkeypatch.setattr(
        b, "run_preload", lambda p: {"preloaded": ["a/b"], "already_cached": [], "failed": {}}
    )
    monkeypatch.setattr(b, "hf_upload", lambda p, path, sub: None)
    monkeypatch.setattr(
        b, "hf_file_exists", lambda p, sub: True
    )  # completion file confirmed first try
    monkeypatch.setattr(
        b,
        "write_attempt_marker",
        lambda p, ok, error="", retriable=False: markers.append((ok, error, retriable)),
    )
    assert b.main() == 0
    assert markers == [(True, "", False)]
    with open("/tmp/preload_result.json") as f:
        assert json.load(f)["preloaded"] == ["a/b"]


def test_main_preload_failure_arms_wall_cap_and_reports_failed_models(monkeypatch):
    markers: list[tuple] = []
    monkeypatch.setattr(b.time, "sleep", lambda s: None)  # neutralize the confirm-retry backoff
    monkeypatch.setattr(b, "load_payload", lambda path=b.PAYLOAD_PATH: _preload_payload())
    monkeypatch.setattr(
        b,
        "run_preload",
        lambda p: {"preloaded": [], "already_cached": [], "failed": {"a/b": "oom"}},
    )
    monkeypatch.setattr(b, "hf_upload", lambda p, path, sub: None)
    monkeypatch.setattr(
        b, "hf_file_exists", lambda p, sub: False
    )  # confirm loop exhausts -> else branch
    monkeypatch.setattr(
        b,
        "write_attempt_marker",
        lambda p, ok, error="", retriable=False: markers.append((ok, error, retriable)),
    )
    assert b.main() == 1
    ok, error, retriable = markers[0]
    assert ok is False
    assert error == "model preload failed"
    assert retriable is False
