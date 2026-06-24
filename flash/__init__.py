"""Flash — managed LoRA post-training: log in with your freesolo key, train.

A focused developer experience (TOML run specs, pluggable environments,
CLI/API/MCP entry points, adapter deployment). Users authenticate with their
freesolo API key (`flash login`); the control plane runs each job on a managed
RunPod GPU behind the scenes.
"""

__all__ = ["__version__"]

# Derive the version from installed package metadata so it can never drift from
# pyproject.toml (the single source of truth the publish workflow reads). A
# hand-maintained literal here previously lagged the published version, making
# `flash --version` / `flash version` report a stale number.
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("freesolo-flash")
except PackageNotFoundError:  # source checkout without installed metadata
    __version__ = "0.0.0+unknown"

del _pkg_version, PackageNotFoundError
