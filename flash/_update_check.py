"""Background "a new release is available" notice for the `flash` CLI."""

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
from flash._channel import DIST_NAME
from flash._fileio import read_json_or_empty, secure_json_write
from flash._logging import get_logger
from flash.client.config import CONFIG_DIR

logger = get_logger("flash.update_check")

PACKAGE_NAME = DIST_NAME
UPGRADE_COMMAND = f"uv tool upgrade {PACKAGE_NAME}"
_PYPI_JSON_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"

CACHE_PATH = CONFIG_DIR / "update_check.json"

_CHECK_INTERVAL_S = 24 * 60 * 60
# join >= fetch timeout so the worker finishes writing before process exit kills it.
_FETCH_TIMEOUT_S = 1.5
_JOIN_TIMEOUT_S = 2.0

_OPT_OUT_ENV = "FLASH_NO_UPDATE_CHECK"

# Reject anything outside the PEP 440 charset before printing — prevents escape-code injection.
_SAFE_VERSION = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9.+!_-]{0,63}\Z")

# Coarse PEP 440 subset; stdlib-only client can't depend on `packaging`.
_VERSION_RE = re.compile(
    r"""\A\s*v?
        (?P<release>\d+(?:\.\d+)*)
        (?P<pre>[._-]?(?:alpha|beta|preview|pre|rc|a|b|c)\d*)?
        (?P<post>[._-]?(?:post|rev|r)\d*|-\d+)?
        (?P<dev>[._-]?dev\d*)?
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _enabled() -> bool:
    """Off unless stderr is a TTY and the user hasn't opted out."""
    if os.environ.get(_OPT_OUT_ENV):
        return False
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        return False


def _normalize_release(release: tuple[int, ...]) -> tuple[int, ...]:
    """Drop trailing zeros so ``1.0`` and ``1.0.0`` compare equal (keep at least one segment)."""
    parts = list(release)
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def _version_key(version: str) -> tuple[tuple[int, ...], int, int]:
    """Sort key ``(release, final_rank, post)`` where higher means newer."""
    match = _VERSION_RE.match(version or "")
    if not match:
        return ((), 1, 0)
    release = _normalize_release(tuple(int(part) for part in match.group("release").split(".")))
    is_pre = bool(match.group("pre") or match.group("dev"))
    post_digits = re.search(r"\d+", match.group("post") or "")
    post = int(post_digits.group()) if post_digits else 0
    return (release, 0 if is_pre else 1, post)


def _is_prerelease(version: str) -> bool:
    """True when the version carries a pre-release or dev marker (a/b/c/rc/alpha/beta/dev)."""
    match = _VERSION_RE.match(version or "")
    return bool(match and (match.group("pre") or match.group("dev")))


def _is_newer(latest: str, current: str) -> bool:
    """True only when ``latest`` is a strictly newer version than ``current`` (PEP 440 order)."""
    latest_key = _version_key(latest)
    return bool(latest_key[0]) and latest_key > _version_key(current)


def _clean_version(value: object) -> str | None:
    """Return ``value`` if it's a safe version string, else ``None``."""
    return value if isinstance(value, str) and _SAFE_VERSION.match(value) else None


def _read_cache() -> dict:
    # Coerce non-dict results (e.g. []) since _check_due runs before main()'s error handling.
    cache = read_json_or_empty(CACHE_PATH)
    return cache if isinstance(cache, dict) else {}


def _check_due(now: float) -> bool:
    """True when there's no fresh cached check (so we should hit PyPI)."""
    cache = _read_cache()
    checked_at = cache.get("checked_at")
    if not isinstance(checked_at, (int, float)):
        return True
    return (now - checked_at) >= _CHECK_INTERVAL_S


def _fetch_latest_version(timeout: float = _FETCH_TIMEOUT_S) -> str | None:
    """Return PyPI's latest version for the package, or ``None`` on any failure/odd response."""
    req = urllib.request.Request(
        _PYPI_JSON_URL,
        headers={
            "Accept": "application/json",
            "User-Agent": f"{PACKAGE_NAME}/{__version__}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
        logger.debug("update check: PyPI lookup failed: %s", exc)
        return None
    info = payload.get("info") if isinstance(payload, dict) else None
    version = info.get("version") if isinstance(info, dict) else None
    return _clean_version(version)


def _stamp_check_time() -> None:
    """Record "checked just now" synchronously so the daily back-off holds even if the daemon is killed."""
    with contextlib.suppress(Exception):
        cache = _read_cache()
        cache["checked_at"] = time.time()
        secure_json_write(CACHE_PATH, cache)


def _refresh_cache() -> None:
    """Fetch from PyPI and persist the version; runs in a daemon thread, never raises."""
    try:
        latest = _fetch_latest_version()
        if not latest:
            return
        cache = _read_cache()
        cache["checked_at"] = time.time()
        cache["pypi_version"] = latest
        secure_json_write(CACHE_PATH, cache)
    except Exception as exc:
        logger.debug("update check: refresh failed: %s", exc)


def _supports_color() -> bool:
    return not os.environ.get("NO_COLOR")


def _red(text: str) -> str:
    return f"\033[31m{text}\033[0m" if _supports_color() else text


def _build_notice() -> str | None:
    """Build the upgrade notice from the cached PyPI version, or ``None`` if up to date."""
    latest = _clean_version(_read_cache().get("pypi_version"))
    if not latest or _is_prerelease(latest) or not _is_newer(latest, __version__):
        return None
    return _red(
        f"A new release of {PACKAGE_NAME} is available: {__version__} -> {latest}\n"
        f"Update with `{UPGRADE_COMMAND}`."
    )


def maybe_start_update_check() -> threading.Thread | None:
    """Kick off a background PyPI refresh if one is due; returns the thread (or ``None``)."""
    if not _enabled() or not _check_due(time.time()):
        return None
    _stamp_check_time()
    thread = threading.Thread(target=_refresh_cache, name="flash-update-check", daemon=True)
    try:
        thread.start()
    except RuntimeError:
        return None
    return thread


def emit_update_notice(notifier: threading.Thread | None = None) -> None:
    """Print the upgrade notice (if any) to stderr at the end of a command."""
    if not _enabled():
        return
    if notifier is not None:
        with contextlib.suppress(RuntimeError):
            notifier.join(timeout=_JOIN_TIMEOUT_S)
    # Must never raise: broken pipe or closed stderr would crash the command.
    with contextlib.suppress(Exception):
        notice = _build_notice()
        if notice:
            print(notice, file=sys.stderr)
