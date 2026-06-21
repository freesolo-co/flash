"""Shared ``fresh_runner`` helper.

Reloads ``flash.runner`` and redirects RUNS_DIR/RESULTS_DIR under a tmp dir — the
reload+monkeypatch dance the audit found duplicated across the orchestrator/jobs tests.
"""

from __future__ import annotations

import importlib
import os


def fresh_runner(tmp, monkeypatch):
    """Reload ``flash.runner`` for the network-shaped submit/poll path with mocks.

    Redirects the fixed RUNS_DIR/RESULTS_DIR module constants under ``tmp`` via
    monkeypatch so they're restored after the test (the module object is shared, so a
    bare assignment leaks).
    """
    import flash.runner as runner

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(str(tmp), "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", os.path.join(str(tmp), "results"))
    return runner
