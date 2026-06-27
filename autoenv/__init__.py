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

__all__ = ["__version__"]

__version__ = "0.0.1"
