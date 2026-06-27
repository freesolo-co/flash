"""Evaluation: the harness Flash lacks (eval is otherwise 'on the serving side').

Pure, dependency-free pieces (metric registry, deterministic split + leakage guard) live
here and are unit-tested offline. Generation (``generate.py``) needs a deployed adapter and
the network, so it is imported lazily by the run pipeline, not at package import.
"""

from __future__ import annotations

from autoenv.eval.metrics import METRICS, aggregate, score_one
from autoenv.eval.split import LeakageReport, leakage_check, split_rows

__all__ = ["METRICS", "LeakageReport", "aggregate", "leakage_check", "score_one", "split_rows"]
