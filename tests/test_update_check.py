"""Unit coverage for the `flash` "a new release is available" notice.

The notice is gated on ``sys.stderr.isatty()`` (so it never shows in pipes/CI/tests), which
means we exercise the helpers directly rather than through a subprocess.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

import flash._update_check as uc
from flash import __version__


@pytest.fixture
def cache_path(tmp_path, monkeypatch):
    """Point the update-check cache at a temp file for the duration of a test."""
    path = tmp_path / "update_check.json"
    monkeypatch.setattr(uc, "CACHE_PATH", path)
    return path


# -- version comparison ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.2.12", ((0, 2, 12), 1, 0)),
        ("v1.0", ((1,), 1, 0)),  # trailing zeros stripped
        ("  1.0.0  ", ((1,), 1, 0)),
        ("0.3.0rc1", ((0, 3), 0, 0)),  # pre-release ranks below final
        ("1.0.0.dev4", ((1,), 0, 0)),
        ("0.2.18.post1", ((0, 2, 18), 1, 1)),  # post ranks above final
        ("0.2.18-1", ((0, 2, 18), 1, 1)),  # implicit post
        ("not-a-version", ((), 1, 0)),
        ("", ((), 1, 0)),
    ],
)
def test_version_key(version, expected):
    assert uc._version_key(version) == expected


@pytest.mark.parametrize(
    ("latest", "current", "expected"),
    [
        ("0.2.13", "0.2.12", True),
        ("0.3.0", "0.2.12", True),
        ("1.0.0", "0.9.9", True),
        ("0.2.12", "0.2.12", False),  # equal
        ("0.2.11", "0.2.12", False),  # older
        ("0.2", "0.2.12", False),  # 0.2 < 0.2.12 (shorter prefix is smaller)
        ("0.2.12", "0.2", True),  # 0.2.12 > 0.2
        ("1.0", "1.0.0", False),  # trailing zero: equal, not newer
        ("0.2.18.post1", "0.2.18", True),  # post-release is newer
        ("0.2.18", "0.2.18.post1", False),  # ...and the reverse is not
        ("0.3.0", "0.3.0rc1", True),  # final beats its own rc
        ("0.3.0rc1", "0.3.0", False),  # rc does not beat the final
        ("garbage", "0.2.12", False),  # unparseable latest never nags
    ],
)
def test_is_newer(latest, current, expected):
    assert uc._is_newer(latest, current) is expected


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.3.0rc1", True),
        ("1.0a1", True),
        ("1.0b2", True),
        ("1.0.dev1", True),
        ("2.0.0.dev0", True),
        ("0.2.18", False),
        ("0.2.18.post1", False),  # post-release is not a pre-release
        ("1.0", False),
    ],
)
def test_is_prerelease(version, expected):
    assert uc._is_prerelease(version) is expected


# -- cache / scheduling ----------------------------------------------------------------------


def test_check_due_when_no_cache(cache_path):
    assert uc._check_due(now=1_000_000.0) is True


def test_check_due_false_when_fresh(cache_path):
    cache_path.write_text(json.dumps({"checked_at": 1_000_000.0, "pypi_version": "0.2.12"}))
    # half a day later -> still fresh
    assert uc._check_due(now=1_000_000.0 + uc._CHECK_INTERVAL_S / 2) is False


def test_check_due_true_when_stale(cache_path):
    cache_path.write_text(json.dumps({"checked_at": 1_000_000.0, "pypi_version": "0.2.12"}))
    assert uc._check_due(now=1_000_000.0 + uc._CHECK_INTERVAL_S + 1) is True


def test_check_due_true_when_cache_corrupt(cache_path):
    cache_path.write_text("{ not json")
    assert uc._check_due(now=1_000_000.0) is True


# -- PyPI fetch ------------------------------------------------------------------------------


def _fake_urlopen(payload: dict):
    def _open(req, timeout=None):
        return io.BytesIO(json.dumps(payload).encode())

    return _open


def test_fetch_latest_version_success(monkeypatch):
    monkeypatch.setattr(uc.urllib.request, "urlopen", _fake_urlopen({"info": {"version": "9.9.9"}}))
    assert uc._fetch_latest_version() == "9.9.9"


def test_fetch_latest_version_network_error_returns_none(monkeypatch):
    def _boom(req, timeout=None):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(uc.urllib.request, "urlopen", _boom)
    assert uc._fetch_latest_version() is None


def test_fetch_latest_version_bad_json_returns_none(monkeypatch):
    def _open(req, timeout=None):
        return io.BytesIO(b"<html>not json</html>")

    monkeypatch.setattr(uc.urllib.request, "urlopen", _open)
    assert uc._fetch_latest_version() is None


def test_fetch_latest_version_missing_field_returns_none(monkeypatch):
    monkeypatch.setattr(uc.urllib.request, "urlopen", _fake_urlopen({"info": {}}))
    assert uc._fetch_latest_version() is None


@pytest.mark.parametrize("payload", [[], {"info": None}, "just a string", {"info": [1, 2]}])
def test_fetch_latest_version_tolerates_odd_json(payload, monkeypatch):
    # a wrong-shaped (but valid) JSON response must not raise out of the fetch
    monkeypatch.setattr(uc.urllib.request, "urlopen", _fake_urlopen(payload))
    assert uc._fetch_latest_version() is None


def test_refresh_cache_writes_latest(cache_path, monkeypatch):
    monkeypatch.setattr(uc, "_fetch_latest_version", lambda: "9.9.9")
    uc._refresh_cache()
    saved = json.loads(cache_path.read_text())
    assert saved["pypi_version"] == "9.9.9"
    assert isinstance(saved["checked_at"], (int, float))


def test_refresh_cache_noop_on_failure(cache_path, monkeypatch):
    # the attempt time is stamped synchronously elsewhere, so a failed lookup just leaves the cache
    cache_path.write_text(json.dumps({"checked_at": 1.0, "pypi_version": "9.9.9"}))
    monkeypatch.setattr(uc, "_fetch_latest_version", lambda: None)
    uc._refresh_cache()
    assert json.loads(cache_path.read_text()) == {"checked_at": 1.0, "pypi_version": "9.9.9"}


def test_stamp_check_time_records_timestamp(cache_path):
    uc._stamp_check_time()
    saved = json.loads(cache_path.read_text())
    assert isinstance(saved["checked_at"], (int, float))
    assert "pypi_version" not in saved


def test_stamp_check_time_preserves_known_version(cache_path):
    cache_path.write_text(json.dumps({"checked_at": 1.0, "pypi_version": "9.9.9"}))
    uc._stamp_check_time()
    saved = json.loads(cache_path.read_text())
    assert saved["pypi_version"] == "9.9.9"  # not dropped
    assert saved["checked_at"] > 1.0  # but refreshed


def test_join_timeout_covers_fetch_timeout():
    # the once-a-day refresh must be able to finish (and record its result) within the join wait,
    # otherwise the daemon worker is killed at process exit and the cache stays stale
    assert uc._JOIN_TIMEOUT_S >= uc._FETCH_TIMEOUT_S


def test_maybe_start_stamps_check_time_synchronously(cache_path, monkeypatch):
    # back-off must be recorded even if the worker thread never gets to write (here it's a no-op)
    monkeypatch.setattr(uc, "_enabled", lambda: True)
    monkeypatch.setattr(uc, "_refresh_cache", lambda: None)
    thread = uc.maybe_start_update_check()
    if thread is not None:
        thread.join(timeout=2.0)
    saved = json.loads(cache_path.read_text())
    assert isinstance(saved["checked_at"], (int, float))


def test_read_cache_coerces_non_object(cache_path):
    # a non-object cache (valid JSON, wrong shape) must not make .get() callers raise
    cache_path.write_text("[1, 2, 3]")
    assert uc._read_cache() == {}


def test_check_due_true_when_cache_not_object(cache_path):
    cache_path.write_text("[]")
    assert uc._check_due(now=1_000_000.0) is True  # no crash, treated as "due"


# -- notice building -------------------------------------------------------------------------


def test_build_notice_when_newer(cache_path, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    cache_path.write_text(json.dumps({"checked_at": 1.0, "pypi_version": "99.0.0"}))
    notice = uc._build_notice()
    assert notice is not None
    assert "freesolo-flash" in notice
    assert __version__ in notice
    assert "99.0.0" in notice
    assert "uv tool upgrade freesolo-flash" in notice
    assert "\033[31m" in notice  # red
    assert notice.endswith("\033[0m")


def test_build_notice_skips_prerelease(cache_path):
    # a newer-looking pre-release must not be advertised as a stable upgrade
    cache_path.write_text(json.dumps({"checked_at": 1.0, "pypi_version": "99.0.0rc1"}))
    assert uc._build_notice() is None


def test_build_notice_none_when_up_to_date(cache_path):
    cache_path.write_text(json.dumps({"checked_at": 1.0, "pypi_version": __version__}))
    assert uc._build_notice() is None


def test_build_notice_none_when_no_cache(cache_path):
    assert uc._build_notice() is None


def test_build_notice_respects_no_color(cache_path, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    cache_path.write_text(json.dumps({"checked_at": 1.0, "pypi_version": "99.0.0"}))
    notice = uc._build_notice()
    assert notice is not None
    assert "\033[" not in notice


# -- gating ----------------------------------------------------------------------------------


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv(uc._OPT_OUT_ENV, "1")
    monkeypatch.setattr(uc.sys.stderr, "isatty", lambda: True, raising=False)
    assert uc._enabled() is False
    assert uc.maybe_start_update_check() is None


def test_disabled_when_not_a_tty(monkeypatch):
    monkeypatch.delenv(uc._OPT_OUT_ENV, raising=False)
    monkeypatch.setattr(uc.sys.stderr, "isatty", lambda: False, raising=False)
    assert uc._enabled() is False
    assert uc.maybe_start_update_check() is None


def test_emit_update_notice_silent_when_disabled(capsys, cache_path, monkeypatch):
    monkeypatch.setenv(uc._OPT_OUT_ENV, "1")
    cache_path.write_text(json.dumps({"checked_at": 1.0, "pypi_version": "99.0.0"}))
    uc.emit_update_notice(None)
    assert capsys.readouterr().err == ""


def test_emit_update_notice_prints_to_stderr_when_enabled(capsys, cache_path, monkeypatch):
    monkeypatch.delenv(uc._OPT_OUT_ENV, raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(uc, "_enabled", lambda: True)
    cache_path.write_text(json.dumps({"checked_at": 1.0, "pypi_version": "99.0.0"}))
    uc.emit_update_notice(None)
    captured = capsys.readouterr()
    assert captured.out == ""  # never pollutes stdout
    assert "99.0.0" in captured.err
    assert "uv tool upgrade freesolo-flash" in captured.err


# -- hardening: must never crash a command, never emit untrusted escape codes ----------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0.2.12", "0.2.12"),
        ("1.0.0rc1", "1.0.0rc1"),
        ("2.0.0.post1", "2.0.0.post1"),
        ("1!2.3", "1!2.3"),  # PEP 440 epoch
        ("0.2.12\033[31mEVIL\033[0m", None),  # ANSI escape injection
        ("1.0\nrm -rf", None),  # newline
        ("1.0.0\x00", None),  # null byte
        ("", None),
        (None, None),
        (123, None),  # non-string
    ],
)
def test_clean_version(value, expected):
    assert uc._clean_version(value) == expected


def test_fetch_latest_version_rejects_escape_codes(monkeypatch):
    payload = {"info": {"version": "99.0.0\033[2J"}}
    monkeypatch.setattr(uc.urllib.request, "urlopen", _fake_urlopen(payload))
    assert uc._fetch_latest_version() is None


def test_build_notice_ignores_poisoned_cache(cache_path):
    # a newer-looking numeric prefix must not let escape codes through to the terminal
    cache_path.write_text(json.dumps({"checked_at": 1.0, "pypi_version": "99.0.0\033[31mX"}))
    assert uc._build_notice() is None


def test_enabled_false_when_isatty_raises_anything(monkeypatch):
    monkeypatch.delenv(uc._OPT_OUT_ENV, raising=False)

    def _boom():
        raise RuntimeError("custom stream")

    monkeypatch.setattr(uc.sys.stderr, "isatty", _boom, raising=False)
    assert uc._enabled() is False  # any exception -> treated as non-TTY, never propagates


def test_emit_update_notice_never_raises_on_broken_stderr(cache_path, monkeypatch):
    # broken pipe / closed stderr while writing the notice must not crash (runs in finally)
    monkeypatch.setattr(uc, "_enabled", lambda: True)
    cache_path.write_text(json.dumps({"checked_at": 1.0, "pypi_version": "99.0.0"}))

    class _BrokenStderr:
        def write(self, *_a, **_k):
            raise BrokenPipeError("downstream closed")

        def flush(self, *_a, **_k):
            raise BrokenPipeError("downstream closed")

    monkeypatch.setattr(uc.sys, "stderr", _BrokenStderr())
    uc.emit_update_notice(None)  # must return without raising
