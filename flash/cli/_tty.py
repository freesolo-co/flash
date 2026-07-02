"""TTY helpers shared by CLI progress renderers."""

from __future__ import annotations

import sys


class TtyStatusLine:
    """Carriage-return status line on stderr; no-op off a TTY."""

    def __init__(self) -> None:
        self._enabled = sys.stderr.isatty()
        self._last_len = 0
        self._active = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _write(self, message: str) -> None:
        padding = " " * max(0, self._last_len - len(message))
        sys.stderr.write(f"\r{message}{padding}")
        sys.stderr.flush()
        self._last_len = len(message)
        self._active = True

    def clear(self) -> None:
        if not (self._enabled and self._active):
            return
        sys.stderr.write(f"\r{' ' * self._last_len}\r")
        sys.stderr.flush()
        self._active = False
