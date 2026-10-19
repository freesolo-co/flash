"""Package logging is quiet on import and configurable via the CLI / env."""

from __future__ import annotations

import logging
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_configure_logging_sets_level():
    import flash._internal.logging as ctlog

    ctlog.configure_logging(verbosity=2)
    assert logging.getLogger("flash").level == logging.DEBUG
    ctlog.configure_logging(verbosity=1)
    assert logging.getLogger("flash").level == logging.INFO
    ctlog.configure_logging(verbosity=0)
    assert logging.getLogger("flash").level == logging.WARNING
    # configure twice -> exactly one console handler (no stacking).
    ctlog.configure_logging(verbosity=0)
    consoles = [
        h for h in logging.getLogger("flash").handlers if getattr(h, "_flash_console", False)
    ]
    assert len(consoles) == 1
