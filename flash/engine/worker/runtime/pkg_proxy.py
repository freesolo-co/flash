"""Live proxy to ``flash.engine.worker`` so submodule monkeypatches survive reload/re-import.

A plain ``from flash.engine import worker as _w`` binds once and goes stale after
``sys.modules.pop`` + re-import; accessing through this proxy always hits the current module.
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
