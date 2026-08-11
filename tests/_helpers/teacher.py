"""Shared helper that satisfies the managed-teacher configuration gate for opd submissions.

An opd submission is validated against the plane's managed-teacher configuration before a run
record exists, so a test that submits opd for some *other* reason (provider acceptance, resolved
backend, multi-gpu shapes) now needs that configuration present. The autouse credential scrub in
``conftest`` deliberately clears both names, so each such test opts back in through this helper.

Tests that assert gate behaviour itself set the environment directly, so a helper change cannot
mask a contract change.
"""

from __future__ import annotations

from typing import Any

# not a credential: the scrub in conftest guarantees no operator key is present, and the gate only
# checks that the name is set and non-blank.
_PLANE_API_KEY = "control-plane-only-canary"
_PLANE_PUBLIC_URL = "https://broker.example"


def configure_managed_teacher(monkeypatch, spec: Any = None) -> None:
    """Set the plane-side managed-teacher configuration an opd submit requires.

    A non-opd spec is left completely alone, so a parametrized suite can call this for every
    algorithm exactly as it calls ``satisfy_sft_profile``. Pass no spec to configure unconditionally.
    """
    if spec is not None and getattr(spec, "algorithm", "") != "opd":
        return
    monkeypatch.setenv("PARASAIL_API_KEY", _PLANE_API_KEY)
    monkeypatch.setenv("FLASH_PUBLIC_URL", _PLANE_PUBLIC_URL)
