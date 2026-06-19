"""Package logging helpers.

Library code logs through the ``flash`` logger and never configures handlers on import (it
attaches a :class:`logging.NullHandler`), so importing Flash stays silent for downstream
applications. The CLI calls :func:`configure_logging` to attach a console handler whose
level is controlled by ``-v/--verbose``.
"""

from __future__ import annotations

import logging

_ROOT_NAME = "flash"

# Attach a NullHandler once so "No handlers could be found" warnings never appear and
# importing the library produces no output unless the app opts in.
_root = logging.getLogger(_ROOT_NAME)
if not any(isinstance(h, logging.NullHandler) for h in _root.handlers):
    _root.addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the ``flash`` namespace (e.g. ``get_logger(__name__)``)."""
    if not name or name == _ROOT_NAME:
        return logging.getLogger(_ROOT_NAME)
    if name.startswith(_ROOT_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")


def configure_logging(verbosity: int = 0, level: int | None = None) -> None:
    """Attach a console handler to the ``flash`` logger and set its level.

    ``verbosity`` maps repeated ``-v`` flags to levels (0=WARNING, 1=INFO, >=2=DEBUG).
    An explicit ``level`` overrides the verbosity mapping.
    """
    if level is None:
        level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)

    logger = logging.getLogger(_ROOT_NAME)
    logger.setLevel(level)
    # Replace any prior console handler we installed so repeated calls don't stack handlers.
    for h in [h for h in logger.handlers if getattr(h, "_flash_console", False)]:
        logger.removeHandler(h)
    handler = logging.StreamHandler()  # stderr
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    handler._flash_console = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
