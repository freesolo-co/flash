"""Background "a new release is available" notice for the `flash` CLI.

The client CLI is pure standard library (no extra deps), so this is too: it queries PyPI
with ``urllib`` and compares the published version against the installed ``__version__``.

Design constraints that keep it from ever getting in the way:

- **Never blocks or breaks a command.** The PyPI lookup runs in a daemon thread and every
  failure (offline, timeout, bad JSON) is swallowed. The notice is built from a cached
  result, so the common path does zero network I/O.
- **Cached once per day.** The latest version is stored in ``~/.flash/update_check.json``;
  we only hit PyPI when that cache is older than :data:`_CHECK_INTERVAL_S`.
- **stderr only, TTY only.** The notice prints to stderr (never stdout), so it can't corrupt
  JSON piped to ``jq`` or captured output, and it's suppressed entirely when stderr isn't a
  terminal (pipes, redirects, CI, tests). Color is dropped when ``NO_COLOR`` is set.
- **Opt-out.** Set ``FLASH_NO_UPDATE_CHECK=1`` to disable the check and notice completely.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

from flash import __version__
from flash._fileio import read_json_or_empty, secure_json_write
from flash._logging import get_logger
from flash.client.config import CONFIG_DIR

logger = get_logger("flash.update_check")

# The PyPI distribution name (== pyproject `name`) and the command that upgrades it.
PACKAGE_NAME = "freesolo-flash"
UPGRADE_COMMAND = f"uv tool upgrade {PACKAGE_NAME}"
_PYPI_JSON_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"

CACHE_PATH = CONFIG_DIR / "update_check.json"

# Re-check PyPI at most once a day; the notice itself is shown on every command from cache.
_CHECK_INTERVAL_S = 24 * 60 * 60
# How long the lookup itself may take, and how long we'll wait for it at the end of a command.
_FETCH_TIMEOUT_S = 2.0
_JOIN_TIMEOUT_S = 1.0

_OPT_OUT_ENV = "FLASH_NO_UPDATE_CHECK"

# A PEP 440 version only uses this charset. We reject anything else (control chars, ANSI escape
# sequences, newlines) before printing the value to a terminal, so a poisoned cache or a hostile
# response can't inject escape codes into the notice. The length bound is just a sanity cap.
_SAFE_VERSION = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9.+!_-]{0,63}\Z")


def _enabled() -> bool:
    """The whole feature is off unless stderr is a TTY and the user hasn't opted out."""
    if os.environ.get(_OPT_OUT_ENV):
        return False
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        # stderr may be detached/closed/replaced (e.g. some embedded contexts); any failure
        # here is treated as "not a TTY" so the check can never crash a command.
        return False


def _release_tuple(version: str) -> tuple[int, ...]:
    """Leading numeric release segment of a version, e.g. ``"0.2.12"`` -> ``(0, 2, 12)``.

    Anything after the release (pre-release/dev/local suffixes) is ignored. Returns ``()``
    when no leading numeric segment is found, which compares as "older than everything".
    """
    match = re.match(r"\s*v?(\d+(?:\.\d+)*)", version or "")
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def _is_newer(latest: str, current: str) -> bool:
    """True only when ``latest`` is a strictly newer release than ``current``."""
    latest_release = _release_tuple(latest)
    return bool(latest_release) and latest_release > _release_tuple(current)


def _clean_version(value: object) -> str | None:
    """Return ``value`` only if it's a safe, escape-free version string, else ``None``.

    Guards both the PyPI response and the on-disk cache: ``_release_tuple`` parses just the
    numeric prefix, so without this an injected suffix (ANSI codes, newlines) could reach the
    terminal. Non-strings (and anything outside the PEP 440 charset) are rejected.
    """
    return value if isinstance(value, str) and _SAFE_VERSION.match(value) else None


def _read_cache() -> dict:
    return read_json_or_empty(CACHE_PATH)


def _check_due(now: float) -> bool:
    """True when there's no fresh cached check (so we should hit PyPI)."""
    cache = _read_cache()
    checked_at = cache.get("checked_at")
    if not isinstance(checked_at, (int, float)):
        return True
    return (now - checked_at) >= _CHECK_INTERVAL_S


def _fetch_latest_version(timeout: float = _FETCH_TIMEOUT_S) -> str | None:
    """Return PyPI's latest stable version for the package, or ``None`` on any failure."""
    req = urllib.request.Request(
        _PYPI_JSON_URL,
        headers={
            "Accept": "application/json",
            "User-Agent": f"{PACKAGE_NAME}/{__version__}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            version = json.loads(resp.read()).get("info", {}).get("version")
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
        logger.debug("update check: PyPI lookup failed: %s", exc)
        return None
    return _clean_version(version)


def _refresh_cache() -> None:
    """Hit PyPI and persist the result; runs in a daemon thread, so never raises out."""
    try:
        latest = _fetch_latest_version()
        if not latest:
            return
        secure_json_write(CACHE_PATH, {"checked_at": time.time(), "pypi_version": latest})
    except Exception as exc:  # truly never let a background thread escape
        logger.debug("update check: refresh failed: %s", exc)


def _supports_color() -> bool:
    return not os.environ.get("NO_COLOR")


def _red(text: str) -> str:
    return f"\033[31m{text}\033[0m" if _supports_color() else text


def _build_notice() -> str | None:
    """Build the upgrade notice from the cached PyPI version, or ``None`` if up to date."""
    latest = _clean_version(_read_cache().get("pypi_version"))
    if not latest or not _is_newer(latest, __version__):
        return None
    return _red(
        f"A new release of {PACKAGE_NAME} is available: {__version__} -> {latest}\n"
        f"Update with `{UPGRADE_COMMAND}`."
    )


def maybe_start_update_check() -> threading.Thread | None:
    """Kick off a background PyPI refresh if one is due. Returns the thread (or ``None``).

    Pass the return value to :func:`emit_update_notice`. No-ops (returns ``None``) when the
    feature is disabled or the cached check is still fresh, so the common path is free.
    """
    if not _enabled() or not _check_due(time.time()):
        return None
    thread = threading.Thread(target=_refresh_cache, name="flash-update-check", daemon=True)
    try:
        thread.start()
    except RuntimeError:
        # can't spawn a thread (e.g. interpreter shutting down) — skip the check silently.
        return None
    return thread


def emit_update_notice(notifier: threading.Thread | None = None) -> None:
    """Print the upgrade notice (if any) to stderr at the end of a command.

    Briefly waits for an in-flight refresh so a freshly fetched version can be shown the same
    run; if it doesn't finish in time we just use whatever is already cached.
    """
    if not _enabled():
        return
    if notifier is not None:
        with contextlib.suppress(RuntimeError):
            notifier.join(timeout=_JOIN_TIMEOUT_S)
    # This runs from main()'s finally block, so it must never raise: a broken pipe
    # (`flash ... | head`), full disk, or closed stderr would otherwise crash the command.
    with contextlib.suppress(Exception):
        notice = _build_notice()
        if notice:
            print(notice, file=sys.stderr)
