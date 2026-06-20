"""Tiny shared helpers for reading Flash environment-variable toggles."""

from __future__ import annotations

import os

# Values (case-insensitive, whitespace-stripped) that mean "yes / on" for a boolean
# env var. Anything else — including "0", "false", "" or unset — means "no / off".
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def env_flag(name: str, default: bool = False) -> bool:
    """Truthy-aware read of a boolean environment variable.

    ``1/true/yes/on`` (case-insensitive) => ``True``; ``0/false/no/off/""`` => ``False``.
    When the variable is unset, returns ``default``. Unlike a bare
    ``os.environ.get(name)`` truthiness check, an explicit ``name=0`` reads as ``False``
    rather than (as the raw non-empty string ``"0"``) ``True``.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


def flash_skip_net() -> bool:
    """Whether Flash should stay offline (no provider/network calls).

    ``FLASH_SKIP_NET=1/true/yes/on`` => offline; ``0/false/no/""``/unset => online. This
    is the single source of truth for the offline toggle so ``FLASH_SKIP_NET=0`` reliably
    re-enables the network (e.g. ``make test-live``) instead of every non-empty value —
    including the string ``"0"`` — counting as offline.
    """
    return env_flag("FLASH_SKIP_NET")
