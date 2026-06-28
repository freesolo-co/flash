"""Flash — managed LoRA post-training: log in with your freesolo key, train."""

from importlib.metadata import version as _dist_version

from flash._channel import DIST_NAME as _DIST_NAME

__all__ = ["__version__"]

# Read from dist metadata so version literal doesn't drift from pyproject (caused a nag loop in 0.2.20).
try:
    __version__ = _dist_version(_DIST_NAME)
except Exception:
    __version__ = "0+unknown"
