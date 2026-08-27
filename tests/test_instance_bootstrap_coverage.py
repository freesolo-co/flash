"""Focused hermetic coverage for the shared instance bootstrap (Vast/Lambda worker container).

These target under-covered helpers in ``flash.providers._lifecycle.bootstrap``: payload loading,
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

from flash.providers._lifecycle.bootstrapping import bootstrap as b
from flash.providers._lifecycle.net.deadline import deadline_kwargs
from tests._helpers.source_snapshot import valid_source_snapshot

SOURCE_SNAPSHOT = valid_source_snapshot()


@pytest.mark.parametrize("arm", ["lambda", "vast"])
def test_arm_accepts_current_provider_identity(arm):
    assert b._arm({"flash_arm": arm}) == arm


@pytest.mark.parametrize(
    "payload",
    [pytest.param({}, id="missing"), {"flash_arm": None}, {"flash_arm": ""}],
)
def test_arm_rejects_missing_provider_identity(payload):
    with pytest.raises(ValueError, match="missing flash_arm"):
        b._arm(payload)


def _sleeping_upload_child():
    time.sleep(60.0)


def _sigterm_ignoring_final_upload(payload, _console, _mode, _extra, _final=False):
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
# source descriptor validation
# ---------------------------------------------------------------------------
def test_source_descriptor_rejects_missing_and_malformed():
    with pytest.raises(RuntimeError, match="descriptor"):
        b._source_descriptor({})

    for field, value in (
        ("sha256", "a" * 63),
        ("size", 0),
        ("revision", "b" * 39),
        ("archive_path", "source/not-the-digest/flash-source.zip"),
    ):
        malformed = dict(SOURCE_SNAPSHOT)
        malformed[field] = value
        with pytest.raises(RuntimeError):
            b._source_descriptor({"source_snapshot": malformed})

    assert b._source_descriptor({"source_snapshot": SOURCE_SNAPSHOT}).to_dict() == SOURCE_SNAPSHOT


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

    deadline_at = time.time() + 3600.0
    with pytest.raises(_HFError) as ei:
        b._hf_call(always_503, "list", deadline_at=deadline_at)
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
        b._hf_call(bad_request, "list", deadline_at=deadline_at)
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

    assert b._hf_call(flaky, "download", deadline_at=deadline_at) == "ok-result"
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
    created_at = time.time()
    payload = {
        "hf_repo": "org/repo",
        "hf_prefix": "sft/run",
        "env": {"HF_TOKEN": "hf-tok"},
        "deadline_at": created_at + 60.0,
        "run_created_at": created_at,
        "run_max_wall_seconds": 60.0,
    }
    # True only when the artifact landed: the error is swallowed, so a caller tracking what is
    # already stored would otherwise read a failed upload as success and skip its retry.
    assert b.hf_upload(payload, "/tmp/x.txt", "console.txt") is True
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
    assert b.hf_upload(payload, "/tmp/x.txt", "console.txt") is False


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

    assert b.hf_upload(payload, "/tmp/x.txt", "console.txt") is False
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
    created_at = time.time()
    payload = {
        "hf_repo": "o/r",
        "hf_prefix": "p",
        "env": {"HF_TOKEN": "t"},
        "deadline_at": created_at + 60.0,
        "run_created_at": created_at,
        "run_max_wall_seconds": 60.0,
    }
    assert b.hf_file_exists(payload, "DONE") is True
    assert seen["filename"] == "p/DONE"
    assert seen["repo_id"] == "o/r"
    assert seen["repo_type"] == "dataset"
    assert seen["token"] == "t"
    assert b.hf_file_exists(payload, "metrics.json") is False


def test_bootstrap_network_helpers_reject_payload_without_deadline(monkeypatch):
    calls = []

    class _Api:
        def __init__(self, token=None):
            calls.append(("init", token))

    _install_fake_hf(monkeypatch, HfApi=_Api)
    monkeypatch.setattr(
        b.bootstrap_pip.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("pip must not start without a run deadline"),
    )
    payload = {
        "hf_repo": "o/r",
        "hf_prefix": "p",
        "env": {},
        "extra_pip": ["private-package"],
    }

    assert b.hf_upload(payload, "/tmp/x.txt", "console.txt") is False
    with pytest.raises(RuntimeError, match="run wall deadline"):
        b.hf_file_exists(payload, "DONE")
    with pytest.raises(RuntimeError, match="run wall deadline"):
        b.install_extra_pip(payload)
    assert calls == []


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
# fetch_code: pinned download and failure classification
# ---------------------------------------------------------------------------
def _source_payload() -> dict:
    created_at = b.time.time()
    return {
        "hf_repo": "org/repo",
        "source_snapshot": SOURCE_SNAPSHOT,
        "attempt": 0,
        "run_id": "run-1",
        "env": {"HF_TOKEN": "t"},
        "deadline_at": created_at + 60.0,
        "run_created_at": created_at,
        "run_max_wall_seconds": 60.0,
    }


def test_fetch_code_uses_exact_revision_and_verified_file_materialization(monkeypatch):
    seen = {}
    events = []

    def download(**kwargs):
        seen.update(kwargs)
        events.append("download")
        return "/tmp/archive.zip"

    _install_fake_hf(monkeypatch, hf_hub_download=download)
    monkeypatch.setattr(
        b._source_snapshot,
        "materialize_verified_archive_file",
        lambda path, descriptor, destination: events.append(
            ("materialize", path, descriptor.sha256, destination)
        ),
    )

    b.fetch_code(_source_payload())

    assert seen["filename"] == SOURCE_SNAPSHOT["archive_path"]
    assert seen["revision"] == SOURCE_SNAPSHOT["revision"]
    assert events == [
        "download",
        ("materialize", "/tmp/archive.zip", SOURCE_SNAPSHOT["sha256"], "/runcode/run-1-attempt-0"),
    ]


def test_fetch_code_distinguishes_transport_from_integrity_failure(monkeypatch):
    monkeypatch.setattr(
        b,
        "_hf_call",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ConnectionError("network down")),
    )
    with pytest.raises(b.RetriableBootstrapError, match="pinned flash source"):
        b.fetch_code(_source_payload())

    _install_fake_hf(monkeypatch, hf_hub_download=lambda **_kwargs: "/tmp/archive.zip")
    monkeypatch.setattr(b, "_hf_call", lambda call, *_args, **_kwargs: call())
    monkeypatch.setattr(
        b._source_snapshot,
        "materialize_verified_archive_file",
        lambda *_args: (_ for _ in ()).throw(
            b._source_snapshot.SourceSnapshotError("integrity failed")
        ),
    )
    with pytest.raises(b._source_snapshot.SourceSnapshotError, match="integrity"):
        b.fetch_code(_source_payload())


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_fetch_code_http_client_failures_are_terminal(monkeypatch, status):
    monkeypatch.setattr(
        b,
        "_hf_call",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(_err(status=status)),
    )
    with pytest.raises(RuntimeError, match="pinned flash source") as raised:
        b.fetch_code(_source_payload())
    assert not isinstance(raised.value, b.RetriableBootstrapError)


@pytest.mark.parametrize("status", [429, 500, 503, 599])
def test_fetch_code_http_transient_failures_are_retriable(monkeypatch, status):
    monkeypatch.setattr(
        b,
        "_hf_call",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(_err(status=status)),
    )
    with pytest.raises(b.RetriableBootstrapError, match="pinned flash source"):
        b.fetch_code(_source_payload())


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
    # the worker is launched as its own process-group leader, so a fake needs a pid to be
    # addressable as a group. a real one is never signalled: the group teardown is stubbed out in
    # the tests that reach it.
    def __init__(self, lines, rc=0, timeout_once=False, pid=4242):
        self.stdout = iter(lines)
        self.returncode = rc
        self.pid = pid
        self._timeout_once = timeout_once
        self._waits = 0
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self._waits += 1
        if self._timeout_once and self._waits == 1:
            raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout)
        return self.returncode

    def kill(self):
        self.killed = True


class _RaisingWaitProc:
    def __init__(self, error, pid=4243):
        self.args = ["worker"]
        self.stdout = iter(())
        self.returncode = None
        self.pid = pid
        self.error = error

    def poll(self):
        return self.returncode

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


class _InterruptingCleanupUploader:
    def __init__(self, operation, error, events, clock):
        self.operation = operation
        self.error = error
        self.events = events
        self.clock = clock
        self.alive = True
        self.calls = {"join": 0, "terminate": 0, "kill": 0, "close": 0}

    def start(self):
        self.events.append(("start", self.clock["now"]))

    def join(self, timeout=None):
        self.calls["join"] += 1
        self.events.append(("join", timeout))
        if self.operation == "join" and self.calls["join"] == 1:
            raise self.error
        self.clock["now"] += timeout or 0.0

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.calls["terminate"] += 1
        self.events.append(("terminate", self.clock["now"]))
        if self.operation == "terminate" and self.calls["terminate"] == 1:
            raise self.error
        if self.operation != "kill":
            self.alive = False

    def kill(self):
        self.calls["kill"] += 1
        self.events.append(("kill", self.clock["now"]))
        if self.operation == "kill" and self.calls["kill"] == 1:
            raise self.error
        self.alive = False

    def close(self):
        self.calls["close"] += 1
        self.events.append(("close", self.clock["now"]))
        assert self.alive is False
        if self.operation == "close" and self.calls["close"] == 1:
            raise self.error


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
    b._upload_console_snapshot(payload, console, mode, extra, True)
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

    payload = {
        "hf_repo": "o/r",
        "hf_prefix": "sft/run",
        "env": {},
        "source_snapshot": SOURCE_SNAPSHOT,
        "attempt": 0,
        "run_id": "run-1",
    }
    deadline = b.time.time() + 100
    rc = b.run_mode(payload, {"E": "1"}, "sft", deadline_ts=deadline)
    assert rc == 0
    assert popen_calls[0][0][0] == [sys.executable, "-m", "flash.engine.support.worker_entrypoint"]
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


def test_run_mode_sanitizes_the_echoed_child_line_but_not_the_console_file(monkeypatch, capfd):
    """this process's stdout IS the instance's container log.

    the control plane pulls that log as the failure detail -- vast holds the box after a non-zero
    exit precisely so it can -- and only this process knows the run's secret VALUES, since the
    container starts with an empty environment and the credentials arrive in the payload. echoing
    the child's stdout raw therefore published every runtime secret the worker printed. the console
    FILE keeps the raw line: its upload path sanitizes the tail, and redacting twice would lose
    the byte offsets the tail limit is measured in.
    """
    secret = "vast-runtime-7c1de9f4b3a20685"
    _disable_periodic_console_upload(monkeypatch)
    monkeypatch.setattr(b, "_upload_console_tail_bounded", lambda *a, **k: True)
    monkeypatch.setattr(b, "hf_upload", lambda *a, **k: None)
    proc = _FakeProc([f"boto3 auth failed with {secret}\n", "worker exiting\n"], rc=1)
    monkeypatch.setattr(b.subprocess, "Popen", lambda *a, **k: proc)

    payload = {
        "hf_repo": "o/r",
        "hf_prefix": "sft/run",
        "source_snapshot": SOURCE_SNAPSHOT,
        "attempt": 0,
        "run_id": "run-1",
        # AWS_SECRET_ACCESS_KEY matches no suffix heuristic, so this covers the declared channel.
        "env": {"FLASH_SECRET_ENV_KEYS": "AWS_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY": secret},
    }
    b.run_mode(payload, {"E": "1"}, "sft", deadline_ts=b.time.time() + 100)

    echoed = capfd.readouterr().out
    assert secret not in echoed
    assert "boto3 auth failed with <redacted>" in echoed
    # redaction must not eat the surrounding diagnostics, which are the reason for the echo.
    assert "worker exiting" in echoed
    with open("/tmp/console_sft.txt") as f:
        assert f.read() == f"boto3 auth failed with {secret}\nworker exiting\n"


def test_run_mode_echoes_the_end_of_an_oversized_child_line(monkeypatch, capfd):
    """the sanitizing bound must keep the END of a line, which is where the root cause is.

    a native stack, a json blob or a progress stream puts its conclusion last, and the control
    plane's failure detail reads the PROVIDER's instance log rather than the uploaded console
    artifact -- so a prefix cut here loses the diagnosis everywhere, not just in one copy.
    """
    _disable_periodic_console_upload(monkeypatch)
    monkeypatch.setattr(b, "_upload_console_tail_bounded", lambda *a, **k: True)
    monkeypatch.setattr(b, "hf_upload", lambda *a, **k: None)
    proc = _FakeProc(["x" * 120_000 + "ROOTCAUSE: CUDA OOM\n"], rc=1)
    monkeypatch.setattr(b.subprocess, "Popen", lambda *a, **k: proc)

    payload = {
        "hf_repo": "o/r",
        "hf_prefix": "sft/run",
        "source_snapshot": SOURCE_SNAPSHOT,
        "attempt": 0,
        "run_id": "run-1",
        "env": {},
    }
    b.run_mode(payload, {"E": "1"}, "sft", deadline_ts=b.time.time() + 100)

    echoed = capfd.readouterr().out
    assert "ROOTCAUSE: CUDA OOM" in echoed
    # the bound still applies -- this is a cut, not an unbounded echo.
    assert len(echoed) <= 100_001


def test_safe_detail_keeps_the_requested_side_of_an_over_limit_string():
    """`keep` selects which side survives: a message whose subject comes first keeps its front, a
    streamed console line keeps its end. redaction runs on the whole text before either cut, so
    neither can split a credential into a fragment nothing matches."""
    assert b._safe_detail("abcdef", 3) == "abc"
    assert b._safe_detail("abcdef", 3, keep="end") == "def"
    # a secret spanning the cut point is gone either way, not halved.
    secret = "sk-live-abc123456789"
    for keep in ("start", "end"):
        out = b._safe_detail(f"aa{secret}zz", 12, secrets={"K": secret}, keep=keep)
        assert not any(secret[-n:] in out for n in range(1, len(secret)))
        assert not any(secret[:n] in out for n in range(8, len(secret)))


def test_run_mode_caps_the_worker_at_the_declared_wall_budget(monkeypatch):
    """Unspent provisioning time must not become extra WORK time.

    The absolute deadline is minted when the box is RENTED, so a job whose deadline deliberately
    carries a boot allowance on top of its wall budget -- a workload profile does exactly this --
    would hand a fast-booting box the whole remainder to work in. The plane tightens its own
    deadline at the first heartbeat, but FLASH_RUN_DEADLINE_AT is absolute and already delivered,
    so nothing downstream would ever narrow it: a 10-minute profile could work ~30 minutes on a job
    priced for its wall alone.
    """
    _disable_periodic_console_upload(monkeypatch)
    monkeypatch.setattr(b, "_upload_console_tail_bounded", lambda *a, **k: True)
    monkeypatch.setattr(b, "hf_upload", lambda *a, **k: None)
    popen_calls = []

    def popen(*args, **kwargs):
        popen_calls.append((args, kwargs))
        return _FakeProc([], rc=0)

    monkeypatch.setattr(b.subprocess, "Popen", popen)

    budget = 600.0
    # rent + work + a 20-minute provisioning allowance, with the boot having cost almost nothing.
    deadline = b.time.time() + budget + 20 * 60
    payload = {
        "hf_repo": "o/r",
        "hf_prefix": "profile/run",
        "env": {},
        "source_snapshot": SOURCE_SNAPSHOT,
        "attempt": 0,
        "run_id": "run-1",
        "run_max_wall_seconds": budget,
    }
    assert b.run_mode(payload, {}, "profile", deadline_ts=deadline) == 0

    handed = float(popen_calls[0][1]["env"]["FLASH_RUN_DEADLINE_AT"])
    assert handed - b.time.time() <= budget, (
        "the worker was handed more than the run's declared wall budget, so unspent provisioning "
        "time became extra work time on a job priced for its wall alone"
    )
    # and the cap is the ONLY thing that shortened it -- not the cleanup reserves.
    assert handed < b._worker_execution_deadline(b._upload_cleanup_deadlines(deadline)[0])


def test_run_mode_leaves_a_deadline_already_inside_the_budget_alone(monkeypatch):
    """The cap must not shorten an ordinary run whose deadline is already within its budget."""
    _disable_periodic_console_upload(monkeypatch)
    monkeypatch.setattr(b, "_upload_console_tail_bounded", lambda *a, **k: True)
    monkeypatch.setattr(b, "hf_upload", lambda *a, **k: None)
    popen_calls = []

    def popen(*args, **kwargs):
        popen_calls.append((args, kwargs))
        return _FakeProc([], rc=0)

    monkeypatch.setattr(b.subprocess, "Popen", popen)

    deadline = b.time.time() + 100
    payload = {
        "hf_repo": "o/r",
        "hf_prefix": "sft/run",
        "env": {},
        "source_snapshot": SOURCE_SNAPSHOT,
        "attempt": 0,
        "run_id": "run-1",
        "run_max_wall_seconds": 3600.0,
    }
    assert b.run_mode(payload, {}, "sft", deadline_ts=deadline) == 0

    upload_deadline, _reaping = b._upload_cleanup_deadlines(deadline)
    assert float(popen_calls[0][1]["env"]["FLASH_RUN_DEADLINE_AT"]) == pytest.approx(
        b._worker_execution_deadline(upload_deadline)
    )


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
        b.run_mode({"attempt": 0, "run_id": "run-1"}, {}, "sft", deadline_ts=b.time.time() + 100)

    assert raised.value is wait_error
    assert stop_upload.is_set()
    assert uploader.is_alive() is False
    assert uploader.close_called is True


@pytest.mark.parametrize(
    "wait_error",
    [SystemExit(17), KeyboardInterrupt("interrupted"), RuntimeError("wait failed")],
    ids=["system-exit", "keyboard-interrupt", "unexpected-exception"],
)
def test_wait_error_reaps_uploader_before_propagating_to_terminal_marker(monkeypatch, wait_error):
    uploader = _ReapTrackedUploader()
    stop_upload = threading.Event()
    marker_states = []
    created_at = b.time.time()
    payload = {
        "phase": "sft",
        "attempt": 0,
        "run_id": "run-1",
        "source_snapshot": SOURCE_SNAPSHOT,
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


@pytest.mark.parametrize("cleanup_path", ["periodic", "final"])
@pytest.mark.parametrize("operation", ["join", "terminate", "kill", "close"])
@pytest.mark.parametrize(
    ("error_type", "error_value"),
    [(KeyboardInterrupt, "interrupted"), (SystemExit, 17)],
    ids=["keyboard-interrupt", "system-exit"],
)
def test_uploader_cleanup_defers_base_exception_until_dead_before_marker(
    monkeypatch,
    cleanup_path,
    operation,
    error_type,
    error_value,
):
    clock = {"now": 100.0}
    events = []
    error = error_type(error_value)
    uploader = _InterruptingCleanupUploader(operation, error, events, clock)
    stop_upload = threading.Event()
    initial_timeout = (
        b._CONSOLE_UPLOAD_STOP_TIMEOUT_S
        if cleanup_path == "periodic"
        else b._CONSOLE_UPLOAD_FINAL_TIMEOUT_S
    )
    upload_deadline = clock["now"] + initial_timeout
    reaping_deadline = upload_deadline + b._CONSOLE_UPLOAD_REAP_RESERVE_S
    monkeypatch.setattr(b.time, "time", lambda: clock["now"])

    if cleanup_path == "final":

        class _Context:
            def Process(self, **kwargs):
                assert kwargs["target"] is b._upload_console_snapshot
                return uploader

        monkeypatch.setattr(b.multiprocessing, "get_context", lambda _method: _Context())

    def cleanup_then_publish_marker():
        try:
            if cleanup_path == "periodic":
                b._stop_upload_process(
                    uploader,
                    stop_upload,
                    upload_deadline,
                    reaping_deadline,
                )
            else:
                b._upload_console_tail_bounded(
                    {},
                    "/tmp/unused-console.txt",
                    "sft",
                    "",
                    upload_deadline,
                    reaping_deadline,
                )
        finally:
            events.append(("marker", clock["now"], uploader.is_alive()))

    with pytest.raises(error_type) as raised:
        cleanup_then_publish_marker()

    assert raised.value is error
    if cleanup_path == "periodic":
        assert stop_upload.is_set()
    assert uploader.is_alive() is False
    assert uploader.calls["close"] == 1
    event_names = [event[0] for event in events]
    assert event_names.index("close") < event_names.index("marker")
    assert events[-1] == ("marker", clock["now"], False)


@pytest.mark.parametrize(
    ("error_type", "error_value"),
    [(KeyboardInterrupt, "interrupted"), (SystemExit, 17)],
    ids=["keyboard-interrupt", "system-exit"],
)
def test_uploader_cleanup_preserves_first_base_exception_over_cleanup_noise(
    monkeypatch,
    error_type,
    error_value,
):
    clock = {"now": 100.0}
    original_error = error_type(error_value)

    class _NoisyCleanupUploader:
        def __init__(self):
            self.alive = True
            self.join_calls = 0

        def join(self, timeout=None):
            self.join_calls += 1
            if self.join_calls == 1:
                raise original_error
            clock["now"] += timeout or 0.0

        def is_alive(self):
            return self.alive

        def terminate(self):
            raise RuntimeError("terminate cleanup noise")

        def kill(self):
            self.alive = False

        def close(self):
            raise RuntimeError("close cleanup noise")

    uploader = _NoisyCleanupUploader()
    upload_deadline = clock["now"] + b._CONSOLE_UPLOAD_STOP_TIMEOUT_S
    reaping_deadline = upload_deadline + b._CONSOLE_UPLOAD_REAP_RESERVE_S
    monkeypatch.setattr(b.time, "time", lambda: clock["now"])

    with pytest.raises(error_type) as raised:
        b._stop_upload_process(
            uploader,
            threading.Event(),
            upload_deadline,
            reaping_deadline,
        )

    assert raised.value is original_error
    assert uploader.is_alive() is False


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
    payload = {
        "hf_repo": "o/r",
        "hf_prefix": "sft/run",
        "env": {},
        "source_snapshot": SOURCE_SNAPSHOT,
        "attempt": 0,
        "run_id": "run-1",
    }

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
    assert [event[0] for event in events].index("kill") < [event[0] for event in events].index(
        "marker"
    )
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
    payload = {
        "hf_repo": "o/r",
        "hf_prefix": "sft/run",
        "env": {},
        "source_snapshot": SOURCE_SNAPSHOT,
        "attempt": 0,
        "run_id": "run-1",
    }

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
    payload = {
        "hf_repo": "o/r",
        "hf_prefix": "sft/run",
        "env": {},
        "source_snapshot": SOURCE_SNAPSHOT,
        "attempt": 0,
        "run_id": "run-1",
    }

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
    assert [event[0] for event in events].index("kill") < [event[0] for event in events].index(
        "marker"
    )
    assert events[-1] == ("marker", marker_started_at, False)


@pytest.mark.wallclock
@pytest.mark.parametrize("_iteration", range(3))
def test_run_mode_reserves_cleanup_before_watchdog_marker_with_real_timing(monkeypatch, _iteration):
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
            self.pid = 4244
            self.wait_timeouts = []
            self.killed_at = None

        def poll(self):
            return self.returncode

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
        "source_snapshot": SOURCE_SNAPSHOT,
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
    # route the group teardown at the fake rather than a real killpg on its invented pid: the
    # timing of the kill is what this measures, not the signalling mechanics.
    monkeypatch.setattr(
        b._bootstrap_processes,
        "terminate_process_group",
        lambda process, **_kwargs: process.kill(),
    )
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


@pytest.mark.wallclock
def test_stop_upload_process_reaps_real_sleeping_child_before_marker_reserve():
    context = multiprocessing.get_context("spawn")
    process = _InspectableProcess(context.Process(target=_sleeping_upload_child, daemon=True))
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


@pytest.mark.wallclock
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


@pytest.mark.parametrize(
    ("error_type", "error_value"),
    [(KeyboardInterrupt, "interrupted"), (SystemExit, 17)],
    ids=["keyboard-interrupt", "system-exit"],
)
def test_unreapable_uploader_with_interrupt_exits_before_marker(
    monkeypatch,
    error_type,
    error_value,
):
    clock = {"now": 100.0}
    events = []
    error = error_type(error_value)

    class _HardExit(BaseException):
        pass

    class _InterruptedNeverDeadUploader:
        def __init__(self):
            self.join_calls = 0

        def join(self, timeout=None):
            self.join_calls += 1
            events.append(("join", timeout))
            if self.join_calls == 1:
                events.append(("interrupt", error))
                raise error
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

    def cleanup_then_publish_marker():
        b._stop_upload_process(
            _InterruptedNeverDeadUploader(),
            threading.Event(),
            upload_deadline,
            reaping_deadline,
        )
        events.append(("marker", clock["now"]))

    upload_deadline = clock["now"] + 0.5
    reaping_deadline = upload_deadline + b._CONSOLE_UPLOAD_REAP_RESERVE_S
    monkeypatch.setattr(b.time, "time", lambda: clock["now"])
    monkeypatch.setattr(b.os, "_exit", hard_exit)

    with pytest.raises(_HardExit):
        cleanup_then_publish_marker()

    assert ("interrupt", error) in events
    assert clock["now"] == pytest.approx(reaping_deadline)
    assert events[-1] == ("exit", 124, reaping_deadline)
    assert [event[0] for event in events].count("kill") == 2
    assert all(event[0] != "marker" for event in events)


def test_run_mode_timeout_tears_down_the_whole_worker_group_and_raises(monkeypatch):
    _disable_periodic_console_upload(monkeypatch)
    monkeypatch.setattr(b, "_upload_console_tail_bounded", lambda *_args: True)
    monkeypatch.setattr(b, "hf_upload", lambda p, path, sub: None)
    proc = _FakeProc(["partial\n"], rc=0, timeout_once=True)
    monkeypatch.setattr(b.subprocess, "Popen", lambda *a, **k: proc)
    # never let a fake pid reach a real killpg; record the call instead.
    torn_down = {}

    def _terminate(process, *, process_group_id, **_kwargs):
        torn_down["process"] = process
        torn_down["group"] = process_group_id

    monkeypatch.setattr(b._bootstrap_processes, "terminate_process_group", _terminate)

    payload = {
        "hf_repo": "o/r",
        "hf_prefix": "sft/run",
        "env": {},
        "source_snapshot": SOURCE_SNAPSHOT,
        "attempt": 0,
        "run_id": "run-1",
    }
    with pytest.raises(TimeoutError, match="wall-clock cap"):
        b.run_mode(payload, {}, "grpo", deadline_ts=b.time.time() + 100)
    # the whole group is torn down on the deadline, not just the leader pid: the worker's
    # torchrun/vllm children hold the gpu and outlive a bare proc.kill().
    assert torn_down["process"] is proc
    assert torn_down["group"] == proc.pid


def test_run_mode_timeout_reports_a_worker_group_that_survived_teardown(monkeypatch):
    """A stranded gpu must not be filed as an ordinary capped run."""
    _disable_periodic_console_upload(monkeypatch)
    uploaded = {}

    def _upload(_payload, _console, _mode, extra, *_args):
        uploaded["extra"] = extra
        return True

    monkeypatch.setattr(b, "_upload_console_tail_bounded", _upload)
    monkeypatch.setattr(b, "hf_upload", lambda p, path, sub: None)
    proc = _FakeProc(["partial\n"], rc=0, timeout_once=True)
    monkeypatch.setattr(b.subprocess, "Popen", lambda *a, **k: proc)

    def _terminate(_process, *, process_group_id, **_kwargs):
        raise RuntimeError(f"process group {process_group_id} survived term and kill supervision")

    monkeypatch.setattr(b._bootstrap_processes, "terminate_process_group", _terminate)

    payload = {
        "hf_repo": "o/r",
        "hf_prefix": "sft/run",
        "env": {},
        "source_snapshot": SOURCE_SNAPSHOT,
        "attempt": 0,
        "run_id": "run-1",
    }
    with pytest.raises(TimeoutError, match="survived term and kill supervision"):
        b.run_mode(payload, {}, "grpo", deadline_ts=b.time.time() + 100)
    # the console tail still uploads and carries the survivor, rather than being skipped by the
    # raise: without it there is no record of why the box is still occupied.
    assert "survived term and kill supervision" in uploaded["extra"]


def test_group_teardown_is_bounded_by_the_deadline_it_is_given():
    """A SIGTERM-ignoring group must not spend time the caller reserved for what follows teardown.

    ``terminate_process_group`` runs at the worker's cutoff, and its term-then-kill grace is 15s by
    default while the bootstrap holds only 12s before ``upload_deadline_at``. Unbounded, teardown
    overruns that window and the final console tail is skipped -- discarding the survivor line that
    is the only record the gpu is still held. Bounding it against a deadline (rather than reserving
    the worst case up front) keeps short wall budgets startable and preserves the escalation.
    """
    import flash.providers._lifecycle.bootstrapping.processes as processes

    # a group that ignores every signal, so the waits run to their full allowance.
    monkey_now = [1_000.0]
    real_time = processes.time.time
    try:
        processes.time.time = lambda: monkey_now[0]
        # only 0.4s left: the kill wait is funded first, then whatever remains goes to the term.
        term_wait, kill_wait = processes._bounded_graces(10.0, 5.0, monkey_now[0] + 0.4)
        assert kill_wait == pytest.approx(0.4), "the escalation that frees the gpu is funded first"
        assert term_wait == pytest.approx(0.0)
        assert term_wait + kill_wait <= 0.4, "the supervision must fit the window it was given"

        # a comfortable budget still gets the full default graces, unchanged.
        term_wait, kill_wait = processes._bounded_graces(10.0, 5.0, monkey_now[0] + 60.0)
        assert (term_wait, kill_wait) == (10.0, 5.0)

        # already past the deadline: no waiting at all, but the caller still sends the signals.
        assert processes._bounded_graces(10.0, 5.0, monkey_now[0] - 1.0) == (0.0, 0.0)
    finally:
        processes.time.time = real_time

    # and with no deadline the behaviour is the documented default.
    assert processes._bounded_graces(10.0, 5.0, None) == (10.0, 5.0)


def test_group_teardown_deadline_is_later_than_the_cutoff_that_triggers_teardown():
    """Teardown cannot be bounded by the instant it starts, or the kill gets no time to land.

    ``terminate_process_group`` runs when the worker deadline fires. Handing it that same instant
    leaves zero remaining, so ``_bounded_graces`` yields (0, 0): SIGKILL is sent but never waited
    on, ``_group_exists`` still sees the not-yet-reaped group, and an ordinary wall-clock timeout
    is reported as a survivor holding the gpu. The teardown cap must therefore sit strictly after
    the worker cutoff, while still leaving the final console tail -- which carries the survivor
    line -- its own slice.
    """
    import flash.providers._lifecycle.bootstrapping.processes as processes

    upload_deadline = 1_000.0
    worker_cutoff = b._worker_execution_deadline(upload_deadline)
    teardown_cap = b._group_teardown_deadline(upload_deadline)

    assert teardown_cap > worker_cutoff, (
        "teardown is bounded by the moment it begins, so the kill is never waited on and every "
        "capped run reports a surviving process group"
    )
    # the slack is real time to reap a killed group, not a rounding artefact.
    assert teardown_cap - worker_cutoff == pytest.approx(b._CONSOLE_UPLOAD_STOP_TIMEOUT_S)
    # and it stops short of the tail that reports the survivor.
    assert teardown_cap <= upload_deadline - b._CONSOLE_UPLOAD_FINAL_TIMEOUT_S

    # standing at the cutoff -- where teardown actually begins -- that slack is what the
    # supervision gets to spend. bounded by the cutoff instead, both waits would be zero.
    real_time = processes.time.time
    try:
        processes.time.time = lambda: worker_cutoff
        term_wait, kill_wait = processes._bounded_graces(10.0, 5.0, teardown_cap)
        assert kill_wait > 0, "the escalation that frees the gpu must be given time to be reaped"
        # the whole window is handed to the supervision -- none of it is lost to the split.
        assert term_wait + kill_wait == pytest.approx(teardown_cap - worker_cutoff)
        assert term_wait + kill_wait <= b._CONSOLE_UPLOAD_STOP_TIMEOUT_S
        # bounded by the cutoff instead, both waits collapse and the kill is never awaited.
        assert processes._bounded_graces(10.0, 5.0, worker_cutoff) == (0.0, 0.0)
    finally:
        processes.time.time = real_time


def test_run_mode_hands_teardown_a_deadline_it_can_actually_use(monkeypatch):
    """The cap ``run_mode`` passes to teardown must outlast the cutoff that triggered it."""
    _disable_periodic_console_upload(monkeypatch)
    monkeypatch.setattr(b, "_upload_console_tail_bounded", lambda *a, **k: True)
    monkeypatch.setattr(b, "hf_upload", lambda *a, **k: None)

    handed = {}

    def terminate(process, *, process_group_id, deadline_at=None):
        handed["deadline_at"] = deadline_at
        handed["at"] = b.time.time()

    monkeypatch.setattr(b._bootstrap_processes, "terminate_process_group", terminate)

    class _NeverExits:
        def __init__(self):
            self.args = ["worker"]
            self.stdout = iter(())
            self.returncode = None
            self.pid = 909

        def poll(self):
            return None

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(self.args, timeout)

    monkeypatch.setattr(
        b._bootstrap_processes,
        "start_process_group",
        lambda *a, **k: (_NeverExits(), 909),
    )

    deadline = b.time.time() + 100
    payload = {
        "hf_repo": "o/r",
        "hf_prefix": "sft/run",
        "env": {},
        "source_snapshot": SOURCE_SNAPSHOT,
        "attempt": 0,
        "run_id": "run-1",
    }
    # the worker never exits, so the cap fires and the run ends as an ordinary timeout. that is
    # the path under test: teardown runs, and what it was handed is what decides whether the kill
    # is waited on or the run is misreported as a stranded gpu.
    with pytest.raises(TimeoutError):
        b.run_mode(payload, {}, "sft", deadline_ts=deadline)

    upload_deadline, _reaping = b._upload_cleanup_deadlines(deadline)
    assert handed["deadline_at"] == pytest.approx(b._group_teardown_deadline(upload_deadline))
    # the cap must outlast the worker cutoff that triggered teardown. handing over the cutoff
    # itself leaves zero remaining, so SIGKILL is sent but never waited on and an ordinary timeout
    # is reported as a stranded gpu.
    assert handed["deadline_at"] > b._worker_execution_deadline(upload_deadline)
    assert handed["deadline_at"] > handed["at"], "teardown was handed an already-expired deadline"


def test_run_mode_starts_no_subprocess_at_deadline(monkeypatch):
    monkeypatch.setattr(b.time, "time", lambda: 200.0)
    monkeypatch.setattr(
        b.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("worker process must not start at the deadline"),
    )
    payload = {
        "hf_repo": "o/r",
        "hf_prefix": "sft/run",
        "env": {},
        "source_snapshot": SOURCE_SNAPSHOT,
        "attempt": 0,
        "run_id": "run-1",
    }

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
        "source_snapshot": SOURCE_SNAPSHOT,
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
        "code",
        "install",
        ("training", 500.0),
    ]
    assert events[-2:] == ["deadline_done", "deadline_cancel"]


def test_main_source_verification_failure_prevents_pip(monkeypatch):
    events = []
    payload = {
        **_source_payload(),
        "extra_pip": ["private-package"],
        "run_created_at": 100.0,
        "run_max_wall_seconds": 400.0,
        "deadline_at": 500.0,
    }

    class _Done:
        def set(self):
            return None

    class _Timer:
        def cancel(self):
            return None

    monkeypatch.setattr(b.time, "time", lambda: 100.0)
    monkeypatch.setattr(b, "load_payload", lambda: payload)
    monkeypatch.setattr(b, "arm_deadline_watchdog", lambda *_args: (_Timer(), _Done()))
    monkeypatch.setattr(
        b,
        "fetch_code",
        lambda _payload: (_ for _ in ()).throw(
            b._source_snapshot.SourceSnapshotError("source verification failed")
        ),
    )
    monkeypatch.setattr(b, "install_extra_pip", lambda _payload: events.append("pip"))
    monkeypatch.setattr(b, "write_attempt_marker", lambda *_args, **_kwargs: None)

    assert b.main() == 1
    assert events == []


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
        "source_snapshot": SOURCE_SNAPSHOT,
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
        "source_snapshot": SOURCE_SNAPSHOT,
    }

    b.write_attempt_marker(payload, ok=True)

    with open("/tmp/attempt_marker.json") as f:
        marker = json.load(f)
    assert marker["ok"] is True
    assert marker["retriable"] is False
    assert marker["error"] == ""
    assert marker["ts"] == 205.0
    assert marker["source_attestation"]["sha256"] == SOURCE_SNAPSHOT["sha256"]
    assert marker["source_attestation"]["attempt"] == 3


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
# payload-secret redaction
# ---------------------------------------------------------------------------
# The worker container starts with an empty environment: the run's HF_TOKEN, GITHUB_TOKEN and user
# runtime secrets reach the worker subprocess through payload["env"] only, so the bootstrap's own
# os.environ value-redacts none of them. Everything the bootstrap uploads has to be redacted against
# the payload env instead.
_PAYLOAD_SECRET = "wandb-local-9f3ac1d2e4b5f7a8"


def test_periodic_console_snapshot_cannot_clobber_the_terminal_one(tmp_path, monkeypatch):
    """the periodic uploader is reaped at teardown, but reaping is best-effort under deadline
    pressure: a periodic child killed mid-write must not truncate the terminal scratch file or
    overwrite the terminal artifact the control plane reads as the failure detail."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    uploads: list[tuple] = []
    monkeypatch.setattr(b, "hf_upload", lambda p, path, sub: uploads.append((path, sub)) or True)
    console = tmp_path / "console_sft.txt"
    console.write_text(f"training step 1 {_PAYLOAD_SECRET}\n")
    payload = {
        "hf_repo": "o/r",
        "hf_prefix": "sft/run",
        "attempt": 3,
        "env": {"WANDB_API_KEY": _PAYLOAD_SECRET},
    }
    cap = "\n--- bootstrap: mode 'sft' hit the wall-clock cap; killed ---\n"

    assert b._upload_console_snapshot(payload, str(console), "sft") is True
    assert b._upload_console_snapshot(payload, str(console), "sft", cap, True) is True

    live_path, live_name = uploads[0]
    terminal_path, terminal_name = uploads[1]
    # distinct scratch files and distinct destinations, so neither write can reach the other.
    assert (live_path, terminal_path) != (terminal_path, terminal_path)
    assert live_path == str(console) + "_attempt3.tail"
    assert terminal_path == str(console) + ".tail"
    # the terminal artifact keeps the canonical name the control plane reads; the live one takes the
    # attempt-scoped name it reads separately. bootstrap.py cannot import flash, so the inlined
    # format is pinned against the canonical helper here rather than trusted to stay in step.
    from flash.adapters.artifacts import attempt_scoped_artifact_name

    assert live_name == attempt_scoped_artifact_name("console", "sft", 3)
    assert terminal_name == "console_sft.txt"
    # only the terminal artifact carries the wall-clock-cap evidence, and it survives intact.
    terminal_tail = (tmp_path / "console_sft.txt.tail").read_text()
    live_tail = (tmp_path / "console_sft.txt_attempt3.tail").read_text()
    assert "hit the wall-clock cap" in terminal_tail
    assert "hit the wall-clock cap" not in live_tail
    # both are sanitized: the split must not create an unredacted path.
    assert _PAYLOAD_SECRET not in terminal_tail
    assert _PAYLOAD_SECRET not in live_tail
    assert "training step 1 <redacted>" in terminal_tail


def test_console_snapshot_redacts_a_payload_env_secret(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    uploads: list[tuple] = []
    monkeypatch.setattr(b, "hf_upload", lambda p, path, sub: uploads.append((path, sub)))
    console = tmp_path / "console_sft.txt"
    console.write_text(
        "Traceback (most recent call last):\n"
        '  File "train.py", line 7, in <module>\n'
        f"RuntimeError: wandb login rejected {_PAYLOAD_SECRET}\n"
    )
    payload = {
        "hf_repo": "o/r",
        "hf_prefix": "sft/run",
        "env": {"WANDB_API_KEY": _PAYLOAD_SECRET},
    }

    b._upload_console_snapshot(payload, str(console), "sft")

    tail = (tmp_path / "console_sft.txt_attempt0.tail").read_text()
    assert _PAYLOAD_SECRET not in tail
    assert "RuntimeError: wandb login rejected <redacted>" in tail
    # the surrounding traceback is the whole point of the upload; redaction must not eat it.
    assert "Traceback (most recent call last):" in tail
    assert '  File "train.py", line 7, in <module>' in tail
    assert uploads == [(str(console) + "_attempt0.tail", "console_sft_attempt0.txt")]


def test_attempt_marker_error_redacts_a_payload_env_secret(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(b, "hf_upload", lambda p, path, sub, **kwargs: None)
    monkeypatch.setattr(b.time, "time", lambda: 100.0)
    payload = {
        "hf_repo": "o/r",
        "hf_prefix": "p",
        "flash_arm": "vast",
        "attempt": 1,
        "run_id": "run-marker",
        "deadline_at": 200.0,
        "run_created_at": 100.0,
        "run_max_wall_seconds": 100.0,
        "env": {"WANDB_API_KEY": _PAYLOAD_SECRET},
    }

    b.write_attempt_marker(payload, ok=False, error=f"worker died holding {_PAYLOAD_SECRET}")

    with open("/tmp/attempt_marker.json") as f:
        marker = json.load(f)
    assert marker["error"] == "worker died holding <redacted>"


def test_safe_detail_redacts_declared_secrets_with_arbitrary_names(monkeypatch):
    """[environment] secrets accepts any env name; the explicit FLASH_SECRET_ENV_KEYS list is what
    lets the bootstrap redact names the suffix heuristic misses."""
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    payload = {
        "env": {
            "FLASH_SECRET_ENV_KEYS": "AWS_SECRET_ACCESS_KEY",
            "AWS_SECRET_ACCESS_KEY": "aws-declared-3f9e1c7b5a2d4680",
            "RUN_ID": "run-visible",
        }
    }

    detail = b._safe_detail(
        "boto3 auth failed with aws-declared-3f9e1c7b5a2d4680 for run-visible",
        secrets=b._payload_secrets(payload),
    )

    assert detail == "boto3 auth failed with <redacted> for run-visible"


def test_safe_detail_redacts_overlapping_secrets_longest_first():
    """replacing a shorter secret first would turn the longer one into <redacted>+suffix and leave
    the suffix behind; longest-first replacement cannot."""
    secrets = {
        "SHORT_TOKEN": "abc123456789",
        "LONG_TOKEN": "abc1234567890diagnosticsuffix",
    }

    detail = b._safe_detail(
        "rejected: abc1234567890diagnosticsuffix and abc123456789", secrets=secrets
    )

    assert detail == "rejected: <redacted> and <redacted>"


def test_safe_detail_preserves_punctuation_for_wordless_short_secret():
    assert b._safe_detail("module.py: failed at /tmp/a.py", secrets={"PIN": "."}) == (
        "module.py: failed at /tmp/a.py"
    )


def test_safe_detail_redacts_wordless_short_secret_in_keyed_syntax():
    assert b._safe_detail("token=.", secrets={"PIN": "."}) == "token=<redacted>"


def test_safe_detail_redacts_declared_wordless_values_by_exact_shape():
    secrets = {"KEYED_PIN": ";", "BEARER_PIN": "!"}

    assert b._safe_detail("token=;", secrets=secrets) == "token=<redacted>"
    assert b._safe_detail("Bearer !", secrets=secrets) == "Bearer <redacted>"


def test_safe_detail_protects_shape_before_overlapping_values():
    detail = b._safe_detail("token=;", secrets={"KEY": "token", "PIN": ";"})
    assert detail == "<redacted>"
    assert ";" not in detail

    detail = b._safe_detail("Bearer !", secrets={"KEY": "Bearer", "PIN": "!"})
    assert detail == "<redacted>"
    assert "!" not in detail


def test_safe_detail_redacts_percent_octets_without_folding_literal_case():
    for secret, encoded in ((".", "%2E"), ("-", "%2D"), ("~", "%7E"), ("/", "%2f")):
        assert b._safe_detail(f"encoded {encoded}", secrets={"PIN": secret}) == (
            "encoded <redacted>"
        )

    secrets = {"PIN": "A/B"}
    assert b._safe_detail("encoded A%2fB", secrets=secrets) == "encoded <redacted>"
    assert b._safe_detail("encoded a%2fb", secrets=secrets) == "encoded a%2fb"

    for secret, case_variant in (
        ("A%2FB", "A%2fB"),
        ("literal%2Fsecret", "literal%2fsecret"),
    ):
        assert b._safe_detail(f"literal {secret}", secrets={"PIN": secret}) == (
            "literal <redacted>"
        )
        assert b._safe_detail(f"literal {case_variant}", secrets={"PIN": secret}) == (
            f"literal {case_variant}"
        )


def test_safe_detail_redacts_cross_group_encoded_overlap():
    secrets = {"LONG_TOKEN": "a%2Fb%2B", "SHORT_TOKEN": "a/b+c&d"}

    detail = b._safe_detail("fetch failed for a%2Fb%2Bc%26d", secrets=secrets)

    assert detail == "fetch failed for <redacted>"
    for secret in ("a%2Fb%2B", "c%26d", "a/b+c&d"):
        assert secret not in detail


def test_safe_detail_redacts_each_line_of_a_multiline_secret():
    """a PEM key never reaches a redactor whole: console tails are cut and the child's stdout is
    sanitized one line at a time, so only a component line is ever seen. the whole value alone as a
    needle would match nothing and the component would print verbatim."""
    pem = (
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n"
        "KoZIhvcNAQEBBQADggEPADCCAQoCggEB\n"
        "-----END PRIVATE KEY-----"
    )
    secrets = {"DEPLOY_KEY": pem}

    assert (
        b._safe_detail("ssh failed: MIIEvQIBADANBgkqhkiG9w0BAQEFAASC", secrets=secrets)
        == "ssh failed: <redacted>"
    )
    assert b._safe_detail(pem, secrets=secrets) == "<redacted>"


def test_safe_detail_keeps_short_components_of_a_multiline_secret_readable():
    """the component floor exists so a structural fragment such as `}` in a multiline json
    credential cannot blank out innocent diagnostics everywhere it appears."""
    secrets = {"BLOB": "{\n}\nabc\nlongenoughsecretcomponent"}

    detail = b._safe_detail("parse error near } and abc", secrets=secrets)

    assert detail == "parse error near } and abc"


def test_safe_detail_redacts_a_very_short_declared_secret_at_word_boundaries():
    """a declared secret can carry any value, including a 3-char one, and it must not leak.

    the length floor used to drop such a value from the needle set entirely, so it printed
    verbatim: nothing else covers it unless the surrounding text happens to have credential shape.
    it cannot become an unconstrained global needle either -- the value `ati` would rewrite
    `authentication` -- so it is redacted only where it stands alone.
    """
    short = {"PIN": "ati"}

    # the standalone value is the leak the floor used to allow.
    assert b._safe_detail("worker rejected pin ati", secrets=short) == (
        "worker rejected pin <redacted>"
    )
    # ... while the same letters inside a word stay readable.
    assert b._safe_detail("trainer crashed after validation", secrets=short) == (
        "trainer crashed after validation"
    )
    # an unrelated long value is untouched by a short needle.
    assert b._safe_detail("trainer crashed holding sk-live-abc123456", secrets=short) == (
        "trainer crashed holding sk-live-abc123456"
    )
    assert (
        b._safe_detail(
            "trainer crashed holding sk-live-abc123456", secrets={"PIN": "sk-live-abc123456"}
        )
        == "trainer crashed holding <redacted>"
    )


def test_safe_detail_redacts_a_short_secret_in_the_shapes_diagnostics_print():
    """the boundary form has to fire where a credential actually appears in output: quoted, in a
    url, as a key=value pair, at the very start and end of the text."""
    short = {"PIN": "ati", "KEY": "a/b+c"}

    assert b._safe_detail("auth failed: 'ati'", secrets=short) == "auth failed: '<redacted>'"
    assert b._safe_detail("ati rejected", secrets=short) == "<redacted> rejected"
    assert b._safe_detail("rejected ati", secrets=short) == "rejected <redacted>"
    assert b._safe_detail("https://host/ati/repo.git", secrets=short) == (
        "https://host/<redacted>/repo.git"
    )
    # the percent-encoded form of a short value is registered too.
    assert b._safe_detail("https://host/a%2Fb%2Bc/x", secrets=short) == (
        "https://host/<redacted>/x"
    )


def test_read_console_tail_keeps_a_complete_line_at_the_boundary(tmp_path):
    """the first retained line is dropped only when the byte boundary actually SPLIT it. a boundary
    landing right after a newline starts a complete line, and discarding it would throw away a full
    line of diagnostics -- possibly the root-cause exception -- for a split that never happened."""
    console = tmp_path / "console.txt"
    console.write_bytes(b"older\nROOTCAUSE: boom\nshutdown\n")

    # 24 bytes = exactly "ROOTCAUSE: boom\nshutdown\n", so the boundary sits on the newline.
    assert b._read_console_tail(str(console), 25) == "ROOTCAUSE: boom\nshutdown\n"
    # one byte less splits the line, so it is dropped rather than half-redacted.
    assert b._read_console_tail(str(console), 24) == "shutdown\n"
    # no truncation at all keeps everything.
    assert b._read_console_tail(str(console), 64_000) == "older\nROOTCAUSE: boom\nshutdown\n"


def test_read_console_tail_drops_a_split_credential(tmp_path):
    """the guard's reason for existing: a value cut in half no longer matches full-value
    redaction, so the partial line must never reach the sanitizer."""
    console = tmp_path / "console.txt"
    console.write_bytes(b"token=abc123456789secret\nnext\n")

    tail = b._read_console_tail(str(console), 15)

    assert "secret" not in tail
    assert tail == "next\n"


def test_read_console_tail_drops_a_single_unterminated_line(tmp_path, monkeypatch):
    """A tail holding no newline at all is dropped, even though it costs the only diagnostic.

    Keeping it was tried and reverted. Every bound that would let the line through is measured
    against the credentials this process KNOWS, and the value at risk is the one it does not: a
    capability minted at runtime (a presigned url, a token a provider echoed back) is in neither
    the payload nor the environment, so it contributes no needle. A margin sized from an unrelated
    configured secret then removes a prefix sized for THAT secret and leaves a long fragment of the
    runtime one, which full-value redaction can never match. The empty tail never leaked.
    """
    from flash.providers._lifecycle.bootstrapping import secrets as bootstrap_secrets

    for name in [key for key in b.os.environ if bootstrap_secrets._secret_env_name(key)]:
        monkeypatch.delenv(name, raising=False)
    # the boundary lands 200 characters INTO a runtime credential, so it is genuinely split: the
    # part that survives the cut is what no downstream redactor could match.
    limit, prefix = 64_000, 6_000
    runtime_secret = "dyn-" + "D" * 400
    size = limit + prefix + 200
    body = b"q" * prefix + runtime_secret.encode() + b"q" * (size - prefix - len(runtime_secret))
    assert prefix < size - limit < prefix + len(runtime_secret), "boundary must split the secret"
    console = tmp_path / "console.txt"
    console.write_bytes(body)

    # an unrelated configured credential must not buy the line a pass.
    assert b._read_console_tail(str(console), limit, secrets={"K": "sk-live-abc123456789"}) == ""
    assert b._read_console_tail(str(console), limit, secrets={}) == ""

    # a line the boundary did NOT split is still kept: that is the case worth keeping.
    terminated = tmp_path / "terminated.txt"
    terminated.write_bytes(b"x" * 70_000 + b"\nROOTCAUSE: cuda oom")
    assert b._read_console_tail(str(terminated), limit, secrets={}) == "ROOTCAUSE: cuda oom"


def test_safe_detail_redacts_a_short_secret_whose_edges_are_not_word_characters():
    """The word guard is per EDGE, applied only where the needle's own edge is a word character.

    A short value with a punctuation edge already separates itself from neighbouring text.
    Demanding a non-word character beyond it asks the wrong question: "/a" inside
    "https://host/a/repo" is preceded by the "t" of "host", so an unconditional left guard fails
    and the secret prints verbatim. [environment] secrets accepts any value, so a path- or
    dash-shaped one is not exotic.
    """
    for secret, text in (("/a", "https://host/a/repo"), ("a-", "value a- here"), ("-x", "see -x")):
        out = b._safe_detail(text, 1000, secrets={"S": secret})
        assert secret not in out, f"{secret!r} leaked from {text!r}"
        assert "<redacted>" in out

    # the guard that made this necessary still holds: a value that IS a word cannot rewrite a
    # longer word that merely contains it.
    assert b._safe_detail("authentication ok", 1000, secrets={"S": "ati"}) == "authentication ok"


def test_safe_detail_keep_end_rejects_a_typo_and_honours_a_zero_limit():
    """``text[-0:]`` is ``text[0:]`` -- the WHOLE string, the exact opposite of a zero bound -- so
    the zero case is spelled out. An unknown mode raises rather than silently keeping the front,
    which would cut the side the caller asked to preserve."""
    assert b._safe_detail("abcdef", 0, keep="end") == ""
    assert b._safe_detail("abcdef", 0, keep="start") == ""
    assert b._safe_detail("abcdef", 3, keep="end") == "def"
    assert b._safe_detail("abcdef", 3, keep="start") == "abc"
    with pytest.raises(ValueError, match="keep must be"):
        b._safe_detail("abcdef", 3, keep="edn")


def test_safe_detail_redacts_the_percent_encoded_form_of_a_secret():
    """http and git errors print encoded request urls, so the encoded form leaks the secret even
    when the configured value never appears literally."""
    secrets = {"REPO_TOKEN": "abc/def+ghi="}

    detail = b._safe_detail(
        "fatal: unable to access https://x@host/abc%2Fdef%2Bghi%3D/repo.git", secrets=secrets
    )

    assert "abc%2Fdef%2Bghi%3D" not in detail
    assert "<redacted>" in detail


def test_console_snapshot_drops_the_truncated_first_line_before_redacting(tmp_path, monkeypatch):
    """when the 64k tail boundary lands inside a one-line credential, the surviving suffix no
    longer value-matches; the partial first line must go before sanitizing."""
    monkeypatch.setattr(b, "hf_upload", lambda p, path, sub: None)
    secret = "wandb-boundary-secret-0123456789abcdef"
    console = tmp_path / "console_sft.txt"
    head = f"key {secret}\n"
    tail_line = "tail line survives\n"
    # size the filler so the 64k byte boundary falls inside the secret on the first line.
    boundary_offset = len(head) // 2
    filler = "x" * (64_000 - len(head) - len(tail_line) + boundary_offset - 1) + "\n"
    console.write_text(head + filler + tail_line)
    assert len(head) + len(filler) + len(tail_line) - 64_000 == boundary_offset
    payload = {"hf_repo": "o/r", "hf_prefix": "sft/run", "env": {"WANDB_API_KEY": secret}}

    b._upload_console_snapshot(payload, str(console), "sft")

    tail = (tmp_path / "console_sft.txt_attempt0.tail").read_text()
    assert secret not in tail
    for fragment_length in range(6, len(secret)):
        assert secret[-fragment_length:] not in tail
    assert "tail line survives" in tail


def test_hf_call_retry_log_redacts_payload_secrets(monkeypatch, capsys):
    """the retried hf error message can echo the payload-only token, which os.environ does not
    know; the retry path must redact against the payload env."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(b.time, "sleep", lambda s: None)
    secret = "hf_ZZZretrypathsecret0123456789"
    payload = {"env": {"HF_TOKEN": secret}}
    attempts = []

    def call():
        attempts.append(1)
        if len(attempts) == 1:
            error = RuntimeError(f"429 rate limited for token {secret}")
            error.response = types.SimpleNamespace(status_code=429, headers={})
            raise error
        return "ok"

    assert (
        b._hf_call(
            call,
            "download spec",
            deadline_at=time.time() + 3600.0,
            secrets=b._payload_secrets(payload),
        )
        == "ok"
    )
    printed = capsys.readouterr().out
    assert secret not in printed
    assert "<redacted>" in printed


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
