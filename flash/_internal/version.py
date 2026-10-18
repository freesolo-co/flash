"""Installed Flash package version."""

from importlib.metadata import version as _dist_version

from flash._internal.channel import DIST_NAME

try:
    __version__ = _dist_version(DIST_NAME)
except Exception:
    __version__ = "0+unknown"
