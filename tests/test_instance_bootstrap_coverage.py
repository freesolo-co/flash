"""Focused hermetic coverage for the shared instance bootstrap (Vast/Lambda worker container).

These target under-covered helpers in ``flash.providers._instance_bootstrap``: payload loading,
code-prefix validation, the Hugging Face transient-retry machinery (status/Retry-After parsing +
backoff), the HF upload/exists/fetch wrappers, ``run_mode``'s subprocess tee (success + wall-clock
timeout), the attempt-marker writer, the preload wall-cap watchdog ``_fire`` path, and ``main()``'s
preload branch. Everything is CPU-only and offline: the huggingface_hub package is stubbed via
sys.modules, subprocess.Popen is faked, os._exit is monkeypatched, and time.sleep is neutralized so
no test ever waits on real backoff.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types

import pytest

from flash.providers import _instance_bootstrap as b

CODE_PREFIX = "code/0123456789abcdef0123456789abcdef/flash"


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


def test_fetch_spec_from_hf_returns_downloaded_file_contents(tmp_path, monkeypatch):
    local = tmp_path / "job_spec.json"
    local.write_text('{"spilled": true}')
    seen = {}

    def _dl(**kw):
        seen.update(kw)
        return str(local)

    _install_fake_hf(monkeypatch, hf_hub_download=_dl)
    payload = {"hf_repo": "o/r", "hf_prefix": "sft/run", "env": {"HF_TOKEN": "tok"}}
    assert b.fetch_spec_from_hf(payload) == '{"spilled": true}'
    assert seen["filename"] == "sft/run/job_spec.json"
    assert seen["repo_type"] == "dataset"
    assert seen["token"] == "tok"


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
    payload = {"hf_repo": "org/repo", "code_prefix": CODE_PREFIX, "env": {"HF_TOKEN": "t"}}
    with pytest.raises(RuntimeError, match="no flash code files"):
        b.fetch_code(payload)


# ---------------------------------------------------------------------------
# run_mode: subprocess tee (success + wall-clock timeout)
# ---------------------------------------------------------------------------
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


def test_run_mode_success_returns_rc_and_uploads_console(monkeypatch):
    uploads: list[tuple] = []
    monkeypatch.setattr(b, "hf_upload", lambda p, path, sub: uploads.append((path, sub)))
    proc = _FakeProc(["hello\n", "world\n"], rc=0)
    monkeypatch.setattr(b.subprocess, "Popen", lambda *a, **k: proc)

    payload = {"hf_repo": "o/r", "hf_prefix": "sft/run", "env": {}, "code_prefix": CODE_PREFIX}
    rc = b.run_mode(payload, {"E": "1"}, "sft", deadline_ts=b.time.time() + 100)
    assert rc == 0
    # The console tee is uploaded under console_<mode>.txt and captured the child's stdout.
    assert uploads
    assert uploads[-1][1] == "console_sft.txt"
    with open("/tmp/console_sft.txt") as f:
        body = f.read()
    assert "hello" in body
    assert "world" in body


def test_run_mode_timeout_kills_child_and_raises(monkeypatch):
    monkeypatch.setattr(b, "hf_upload", lambda p, path, sub: None)
    proc = _FakeProc(["partial\n"], rc=0, timeout_once=True)
    monkeypatch.setattr(b.subprocess, "Popen", lambda *a, **k: proc)

    payload = {"hf_repo": "o/r", "hf_prefix": "sft/run", "env": {}, "code_prefix": CODE_PREFIX}
    with pytest.raises(TimeoutError, match="wall-clock cap"):
        b.run_mode(payload, {}, "grpo", deadline_ts=b.time.time() + 100)
    assert proc.killed is True  # the child was killed on the deadline


# ---------------------------------------------------------------------------
# write_attempt_marker
# ---------------------------------------------------------------------------
def test_write_attempt_marker_truncates_error_and_uploads_arm_named(monkeypatch):
    uploads: list[tuple] = []
    monkeypatch.setattr(b, "hf_upload", lambda p, path, sub: uploads.append((path, sub)))
    payload = {"hf_repo": "o/r", "hf_prefix": "p", "flash_arm": "vast", "attempt": 3, "env": {}}
    long_error = "E" * 3000
    b.write_attempt_marker(payload, ok=False, error=long_error, retriable=True)

    path, sub = uploads[-1]
    assert path == "/tmp/attempt_marker.json"
    assert sub == "vast_attempt3.json"  # <arm>_attempt<N>.json
    with open(path) as f:
        marker = json.load(f)
    assert marker["ok"] is False
    assert marker["retriable"] is True
    assert marker["attempt"] == 3
    assert isinstance(marker["ts"], float)
    assert marker["error"] == long_error[-2000:]  # tail-truncated to 2000 chars
    assert len(marker["error"]) == 2000


# ---------------------------------------------------------------------------
# _arm_preload_wall_cap: the _fire watchdog path
# ---------------------------------------------------------------------------
def test_arm_preload_wall_cap_fire_marks_and_hard_exits(monkeypatch):
    marks: list[tuple] = []
    exits: list[int] = []
    monkeypatch.setattr(b, "write_attempt_marker", lambda p, ok, error="", **k: marks.append((ok, error)))
    monkeypatch.setattr(b.os, "_exit", lambda code: exits.append(code))

    payload = {"max_wall_s": 999, "flash_arm": "lambda", "attempt": 0, "hf_repo": "o/r", "env": {}}
    cap = b._arm_preload_wall_cap(payload)
    assert cap is not None
    timer, _done = cap
    timer.cancel()  # prevent the real timer from auto-firing during the test

    # Manually invoke the watchdog closure: not done -> writes a failure marker then os._exit(1).
    timer.function()
    assert exits == [1]
    assert marks
    assert marks[-1][0] is False
    assert "wall-clock cap" in marks[-1][1]

    # If the download already finished (done is set), _fire is a no-op: no marker, no exit.
    marks.clear()
    exits.clear()
    cap2 = b._arm_preload_wall_cap(payload)
    assert cap2 is not None
    timer2, done2 = cap2
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
    r = b.run_preload({
        "env": {"FLASH_WEIGHT_CACHE_DIR": cache_dir},
        "models": ["a/b"],
        "cache_mount_marker": ".flash-cache-mounted",
    })
    assert r["preloaded"] == []
    assert r["already_cached"] == []
    assert "network exploded" in r["failed"]["a/b"]


# ---------------------------------------------------------------------------
# main(): the preload branch
# ---------------------------------------------------------------------------
def _preload_payload(**over):
    base = {
        "mode": "preload",
        "hf_repo": "org/repo",
        "hf_prefix": "preload/run",
        "flash_arm": "lambda",
        "seed": 0,
        "attempt": 0,
        "env": {},
        "max_wall_s": 0,
    }
    base.update(over)
    return base


def test_main_preload_success_returns_zero_and_writes_ok_marker(monkeypatch):
    markers: list[tuple] = []
    monkeypatch.setattr(b, "load_payload", lambda path=b.PAYLOAD_PATH: _preload_payload())
    monkeypatch.setattr(b, "run_preload", lambda p: {"preloaded": ["a/b"], "already_cached": [], "failed": {}})
    monkeypatch.setattr(b, "hf_upload", lambda p, path, sub: None)
    monkeypatch.setattr(b, "hf_file_exists", lambda p, sub: True)  # completion file confirmed first try
    monkeypatch.setattr(
        b, "write_attempt_marker",
        lambda p, ok, error="", retriable=False: markers.append((ok, error, retriable)),
    )
    assert b.main() == 0
    assert markers == [(True, "", False)]
    with open("/tmp/preload_result.json") as f:
        assert json.load(f)["preloaded"] == ["a/b"]


def test_main_preload_failure_arms_wall_cap_and_reports_failed_models(monkeypatch):
    markers: list[tuple] = []
    monkeypatch.setattr(b.time, "sleep", lambda s: None)  # neutralize the confirm-retry backoff
    # max_wall_s>0 -> the real wall-cap watchdog is armed and then cancelled in main()'s finally.
    monkeypatch.setattr(b, "load_payload", lambda path=b.PAYLOAD_PATH: _preload_payload(max_wall_s=999))
    monkeypatch.setattr(
        b, "run_preload",
        lambda p: {"preloaded": [], "already_cached": [], "failed": {"a/b": "oom"}},
    )
    monkeypatch.setattr(b, "hf_upload", lambda p, path, sub: None)
    monkeypatch.setattr(b, "hf_file_exists", lambda p, sub: False)  # confirm loop exhausts -> else branch
    monkeypatch.setattr(
        b, "write_attempt_marker",
        lambda p, ok, error="", retriable=False: markers.append((ok, error, retriable)),
    )
    assert b.main() == 1
    ok, error, retriable = markers[0]
    assert ok is False
    assert "models failed" in error
    assert "a/b" in error
    assert retriable is False
