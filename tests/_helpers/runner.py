"""Shared ``fresh_runner`` helper.

Reloads ``flash.runner`` with FLASH_SKIP_NET unset and RUNS_DIR/RESULTS_DIR
redirected under a tmp dir — the reload+monkeypatch dance the audit found duplicated
across the orchestrator/jobs tests.
"""

from __future__ import annotations

import importlib
import os


def fresh_runner(tmp, monkeypatch):
    """Reload ``flash.runner`` for the network-shaped submit/poll path with mocks.

    Unsets FLASH_SKIP_NET (monkeypatch auto-restores it) and redirects the fixed
    RUNS_DIR/RESULTS_DIR module constants under ``tmp`` via monkeypatch so they're
    restored after the test (the module object is shared, so a bare assignment leaks).
    """
    monkeypatch.delenv("FLASH_SKIP_NET", raising=False)
    import flash.runner as runner

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(str(tmp), "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", os.path.join(str(tmp), "results"))
    return runner
