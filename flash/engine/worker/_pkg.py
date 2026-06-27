"""Live reference to the ``flash.engine.worker`` package for the split submodules.

The submodules read run-scoped STATE and call the patchable helpers THROUGH this proxy
(``_w.<name>``) so every access resolves against the CURRENT worker module object in
``sys.modules`` at CALL time. That is what makes the module-level monkeypatch contract hold even
across the ``sys.modules.pop("flash.engine.worker") + re-import`` and ``importlib.reload(worker)``
that several worker tests use to reset module state from the environment: a plain
``from flash.engine import worker as _w`` binds ONCE to the module object that existed at submodule
import time, which goes stale after such a re-import (the submodules are NOT re-imported), so
``monkeypatch.setattr(worker, ...)`` on the fresh module would no longer reach them.
"""

from __future__ import annotations

import sys


class _LiveWorker:
    """Attribute proxy delegating every get/set to ``sys.modules['flash.engine.worker']``."""

    __slots__ = ()

    def __getattr__(self, name):
        return getattr(sys.modules["flash.engine.worker"], name)

    def __setattr__(self, name, value):
        setattr(sys.modules["flash.engine.worker"], name, value)


W = _LiveWorker()
