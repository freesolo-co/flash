"""Package logging helpers."""

from __future__ import annotations

import logging

_ROOT_NAME = "flash"

# NullHandler: keep library import silent; app opts in via configure_logging.
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
    """Attach a console handler to the ``flash`` logger and set its level."""
    if level is None:
        level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)

    logger = logging.getLogger(_ROOT_NAME)
    logger.setLevel(level)
    for h in [h for h in logger.handlers if getattr(h, "_flash_console", False)]:
        logger.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    handler._flash_console = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
