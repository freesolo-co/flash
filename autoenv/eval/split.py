"""Deterministic train/eval split and the leakage guard.

If a paper ships a separate test set the harness uses it verbatim; otherwise it splits the
train rows with a fixed seed. Either way, ``leakage_check`` hashes every eval ``input`` and
asserts none appear in the rows the run actually trained on — overlap invalidates the case
rather than letting a memorised-test-set number masquerade as a replication.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

_INPUT_KEY = "input"


def _row_key(row: dict, key: str = _INPUT_KEY) -> str:
    """Stable content hash of a row's input field (whitespace-normalised)."""
    text = str(row.get(key, "")).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_rows(
    rows: list[dict], *, eval_fraction: float = 0.2, seed: int = 0, max_eval: int | None = None
) -> tuple[list[dict], list[dict]]:
    """Deterministically partition ``rows`` into (train, eval).

    The order is a fixed hash-based shuffle (seeded), so the same rows always split the same
    way regardless of input order — reproducible across machines and runs.
    """
    if not rows:
        return [], []
    ordered = sorted(
        rows,
        key=lambda r: hashlib.sha256(f"{seed}:{_row_key(r)}".encode()).hexdigest(),
    )
    n_eval = max(1, round(len(ordered) * eval_fraction)) if len(ordered) > 1 else 0
    if max_eval is not None:
        n_eval = min(n_eval, max_eval)
    eval_rows = ordered[:n_eval]
    train_rows = ordered[n_eval:]
    return train_rows, eval_rows


@dataclass
class LeakageReport:
    """Result of checking the eval split against the rows the run trained on."""

    leaked: bool
    overlap_count: int
    eval_size: int
    train_size: int
    overlap_examples: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.leaked


def leakage_check(
    train_rows: list[dict], eval_rows: list[dict], *, input_key: str = _INPUT_KEY
) -> LeakageReport:
    """Flag any eval input that also appears in the trained rows (content-hash equality)."""
    train_hashes = {_row_key(r, input_key) for r in train_rows}
    overlap = [r for r in eval_rows if _row_key(r, input_key) in train_hashes]
    examples = [str(r.get(input_key, ""))[:120] for r in overlap[:5]]
    return LeakageReport(
        leaked=bool(overlap),
        overlap_count=len(overlap),
        eval_size=len(eval_rows),
        train_size=len(train_rows),
        overlap_examples=examples,
    )
