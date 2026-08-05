"""Package logging helpers."""

from __future__ import annotations

import json
import logging
import os

_ROOT_NAME = "flash"

LOG_LEVEL_ENV = "FLASH_LOG_LEVEL"
LOG_FORMAT_ENV = "FLASH_LOG_FORMAT"

_TEXT_FORMAT = "%(levelname)s %(name)s: %(message)s"


class _JsonFormatter(logging.Formatter):
    """One JSON object per line, for deployments that ship logs to a structured sink.

    This formats the SAME record the text formatter would, through the same
    ``record.getMessage()`` path. Nothing is added to the payload that the text line does not
    already carry, so choosing a format cannot change what ends up in the log -- only how it is
    punctuated. Anything a caller must not log stays un-logged in both formats.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _resolve_level(verbosity: int, level: int | None, default_level: int) -> int:
    """Explicit level wins, then -v flags, then FLASH_LOG_LEVEL, then the caller's default.

    A caller that passed an explicit level or counted -v flags has said what it wants for this
    invocation, so the environment does not override it. ``default_level`` is the opposite: it
    is what the entrypoint would have used anyway, so an operator's FLASH_LOG_LEVEL beats it.
    That ordering is what makes the variable useful -- the server defaults to INFO and can still
    be turned down to WARNING or up to DEBUG without a code change.
    """
    if level is not None:
        return level
    if verbosity:
        return {1: logging.INFO}.get(verbosity, logging.DEBUG)
    configured = os.environ.get(LOG_LEVEL_ENV, "").strip().upper()
    if configured:
        # getLevelName maps a known name to its int; an unknown name comes back as the string
        # "Level FOO", which is not an int -- so a typo falls back rather than raising at startup.
        # A log level is not worth refusing to boot over.
        resolved = logging.getLevelName(configured)
        if isinstance(resolved, int):
            return resolved
    return default_level


def _resolve_formatter() -> logging.Formatter:
    if os.environ.get(LOG_FORMAT_ENV, "").strip().lower() == "json":
        return _JsonFormatter()
    return logging.Formatter(_TEXT_FORMAT)


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


def configure_logging(
    verbosity: int = 0, level: int | None = None, default_level: int = logging.WARNING
) -> None:
    """Attach a console handler to the ``flash`` logger and set its level.

    ``FLASH_LOG_LEVEL`` sets the level when the caller has not asked for a specific one, and
    ``FLASH_LOG_FORMAT=json`` switches the console handler to one JSON object per line.
    """
    level = _resolve_level(verbosity, level, default_level)

    logger = logging.getLogger(_ROOT_NAME)
    logger.setLevel(level)
    for h in [h for h in logger.handlers if getattr(h, "_flash_console", False)]:
        logger.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(_resolve_formatter())
    handler._flash_console = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
