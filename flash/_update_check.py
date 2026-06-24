"""Background "a new release is available" notice for the `flash` CLI.

The client CLI is pure standard library (no extra deps), so this is too: it queries PyPI
with ``urllib`` and compares the published version against the installed ``__version__``.

Design constraints that keep it from ever getting in the way:

- **Stays out of the way.** The PyPI lookup runs in a daemon thread (so it overlaps the
  command) and every failure (offline, timeout, bad JSON) is swallowed. Only the once-a-day
  refresh waits briefly for that thread; every other command builds the notice from cache with
  zero network I/O.
- **Cached once per day.** The latest version is stored in ``~/.flash/update_check.json``;
  we only hit PyPI when that cache is older than :data:`_CHECK_INTERVAL_S`. The check time is
  stamped synchronously before the lookup so the daily back-off holds even if the worker thread
  is killed at process exit before it records a result.
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
# How long the lookup may take, and how long the once-a-day refresh waits for it at the end of a
# command. Keep the join >= the fetch timeout so the worker thread finishes (and records its
# result) within the wait instead of being killed at process exit mid-write.
_FETCH_TIMEOUT_S = 1.5
_JOIN_TIMEOUT_S = 2.0

_OPT_OUT_ENV = "FLASH_NO_UPDATE_CHECK"

# A PEP 440 version only uses this charset. We reject anything else (control chars, ANSI escape
# sequences, newlines) before printing the value to a terminal, so a poisoned cache or a hostile
# response can't inject escape codes into the notice. The length bound is just a sanity cap.
_SAFE_VERSION = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9.+!_-]{0,63}\Z")

# A coarse subset of the PEP 440 grammar: the numeric release plus optional pre/post/dev markers.
# Enough to order the simple versions this package ships and to spot pre-releases; this is not a
# full PEP 440 implementation (the stdlib-only client can't depend on `packaging`).
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
    """The whole feature is off unless stderr is a TTY and the user hasn't opted out."""
    if os.environ.get(_OPT_OUT_ENV):
        return False
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        # stderr may be detached/closed/replaced (e.g. some embedded contexts); any failure
        # here is treated as "not a TTY" so the check can never crash a command.
        return False


def _normalize_release(release: tuple[int, ...]) -> tuple[int, ...]:
    """Drop trailing zeros so ``1.0`` and ``1.0.0`` compare equal (keep at least one segment)."""
    parts = list(release)
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def _version_key(version: str) -> tuple[tuple[int, ...], int, int]:
    """A coarse PEP 440 sort key ``(release, final_rank, post)`` where higher means newer.

    The release segment is normalized (``1.0 == 1.0.0``). A pre-release/dev version ranks below
    the final release of the same number (``final_rank`` 0 vs 1); a post-release ranks above it
    via ``post``. Epochs and local versions are ignored — the catalog ships only simple versions.
    Returns an empty release for unparseable input, which compares as "older than everything".
    """
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
    """Return ``value`` only if it's a safe, escape-free version string, else ``None``.

    Guards both the PyPI response and the on-disk cache: ``_version_key`` parses just the
    numeric/marker prefix, so without this an injected suffix (ANSI codes, newlines) could reach
    the terminal. Non-strings (and anything outside the PEP 440 charset) are rejected.
    """
    return value if isinstance(value, str) and _SAFE_VERSION.match(value) else None


def _read_cache() -> dict:
    # read_json_or_empty returns whatever the file parses to; a non-object (e.g. ``[]``) would
    # make the ``.get()`` callers raise, and _check_due runs before main()'s error handling — so
    # coerce anything that isn't a dict back to an empty one.
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
    # Expected shape is {"info": {"version": ...}}; tolerate anything else (a proxy error page,
    # ``[]``, ``{"info": null}``, ...) instead of letting a dereference raise into the caller.
    info = payload.get("info") if isinstance(payload, dict) else None
    version = info.get("version") if isinstance(info, dict) else None
    return _clean_version(version)


def _stamp_check_time() -> None:
    """Record "checked just now" (keeping any cached version), synchronously and best-effort.

    Done before the background lookup starts so the daily back-off holds even if the daemon worker
    is killed at process exit before it records its own result — otherwise a stale/missing cache
    would make every command re-run (and wait on) the lookup. Never raises (runs before main()'s
    error handling).
    """
    with contextlib.suppress(Exception):
        cache = _read_cache()
        cache["checked_at"] = time.time()
        secure_json_write(CACHE_PATH, cache)


def _refresh_cache() -> None:
    """Fetch from PyPI and persist the version on success; runs in a daemon thread, never raises.

    The attempt time is already stamped by :func:`_stamp_check_time`, so a failed lookup just
    returns and lets the daily back-off (set there) stand.
    """
    try:
        latest = _fetch_latest_version()
        if not latest:
            return
        cache = _read_cache()
        cache["checked_at"] = time.time()
        cache["pypi_version"] = latest
        secure_json_write(CACHE_PATH, cache)
    except Exception as exc:  # truly never let a background thread escape
        logger.debug("update check: refresh failed: %s", exc)


def _supports_color() -> bool:
    return not os.environ.get("NO_COLOR")


def _red(text: str) -> str:
    return f"\033[31m{text}\033[0m" if _supports_color() else text


def _build_notice() -> str | None:
    """Build the upgrade notice from the cached PyPI version, or ``None`` if up to date."""
    latest = _clean_version(_read_cache().get("pypi_version"))
    # Only nudge toward stable releases: never advertise a pre-release (rc/dev) as an upgrade.
    if not latest or _is_prerelease(latest) or not _is_newer(latest, __version__):
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
    # Stamp the attempt synchronously before spawning the worker, so the daily back-off holds even
    # if the daemon is killed at process exit before it writes (see _stamp_check_time).
    _stamp_check_time()
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
