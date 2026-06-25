"""Flash — managed LoRA post-training: log in with your freesolo key, train.

A focused developer experience (TOML run specs, pluggable environments,
CLI/API entry points, adapter deployment). Users authenticate with their
freesolo API key (`flash login`); the control plane runs each job on a managed
RunPod GPU behind the scenes.
"""

from importlib.metadata import version as _dist_version

__all__ = ["__version__"]

# single source of truth for the version is pyproject `[project].version`, which hatchling bakes
# into the installed distribution metadata at build time. read it back here instead of keeping a
# second hand-maintained literal: that duplicate is what desynced in 0.2.20 (the wheel said 0.2.20
# while __init__ still hard-coded 0.2.19), making flash nag to upgrade forever while uv reported
# nothing to upgrade. the distribution name ("freesolo-flash") differs from the import package.
try:
    __version__ = _dist_version("freesolo-flash")
except Exception:
    # no readable dist metadata: running from a source tree that was never installed, or an
    # unreadable/corrupt METADATA file. fall back to a clearly-fake version rather than letting
    # `import flash` (the package root, imported by every entry point) crash. this only happens off
    # the installed path; a released wheel always has real metadata. a bare-checkout run on a tty
    # may then show the update notice, which is fine for an uninstalled dev build.
    __version__ = "0+unknown"
