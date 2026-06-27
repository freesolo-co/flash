"""autoenv — automated post-training environment generator / paper-replication benchmark.

A harness that measures how well an AI agent can replicate a paper's small-LM training
result using Flash. Given a ``PaperCase`` manifest, it gates the paper for Flash
compatibility, drives an agent to build a Freesolo environment + config and train via
Flash, benchmarks the trained adapter on the paper's held-out eval split, and scores the
result against the paper's reported number.

This package imports ``flash.*`` one-directionally; ``flash`` never imports ``autoenv``
(asserted by ``tests/test_autoenv_flash_contract.py``). That keeps Flash's zero-dependency
client intact — the heavy bits (``freesolo``/``datasets``/``huggingface-hub``) live behind
the ``autoenv`` optional extra and are imported lazily, only on the paths that need them.
"""

from __future__ import annotations

from importlib.metadata import version as _dist_version

from flash._channel import DIST_NAME as _DIST_NAME

__all__ = ["__version__"]

# autoenv ships inside the `freesolo-flash` distribution; read the version from dist metadata
# (the single source of truth in pyproject) rather than a hand-maintained literal that drifts —
# the same desync guard flash/__init__.py uses. Falls back off the installed path (source tree).
try:
    __version__ = _dist_version(_DIST_NAME)
except Exception:
    __version__ = "0+unknown"
