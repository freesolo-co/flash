# pure-python opd teacher-failure classification + skip accounting, shared by the openrlhf opd
# child (embedded into its sitecustomize) and the offline parity tests. no torch / no openrlhf
# imports so it stays importable in the offline test environment.
#
# it encodes the trl teacher semantics (flash/engine/worker/opd.py: _score_one and the step-loop
# under-run gate):
#   - a PERMANENT teacher failure aborts the run immediately (bad key / model id / malformed).
#   - a TRANSIENT teacher failure is retried up to a per-sample bound, then the sample is SKIPPED
#     (no teacher signal) and the run CONTINUES rather than aborting the whole run.
#   - a step that ends with zero aligned teacher signal is a no-signal step; it does not update.
#   - the terminal under-run gate classifies a signal shortfall: a shortfall that involved NEW
#     transient teacher failures is retriable infra (transient); a shortfall with no new transient
#     failure is a deterministic shortfall the run genuinely could not align.
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

PERMANENT = "permanent"
TRANSIENT = "transient"
_VALID_CLASSIFICATIONS = frozenset({PERMANENT, TRANSIENT})

# per-sample teacher-failure decision
RETRY = "retry"
SKIP_TRANSIENT = "skip_transient"
ABORT_PERMANENT = "abort_permanent"

# terminal under-run decision
COMPLETE = "complete"
TRANSIENT_SHORTFALL = "transient_shortfall"
DETERMINISTIC_SHORTFALL = "deterministic_shortfall"


def classify_teacher_failure(classification: str, attempt: int, max_attempts: int) -> str:
    """Decide how to handle a teacher scoring failure on a single sample.

    ``attempt`` is 1-based (the attempt that just failed); ``max_attempts`` is the per-sample bound
    (>= 1). A permanent failure aborts regardless of attempts (matching TRL raising immediately). A
    transient failure retries while attempts remain, else the sample is skipped (not aborted).
    """
    if classification not in _VALID_CLASSIFICATIONS:
        # an unclassifiable failure is treated as permanent, matching the child's fail-closed default.
        return ABORT_PERMANENT
    if int(max_attempts) < 1:
        raise ValueError("max_attempts must be >= 1")
    if int(attempt) < 1:
        raise ValueError("attempt is 1-based and must be >= 1")
    if classification == PERMANENT:
        return ABORT_PERMANENT
    if int(attempt) < int(max_attempts):
        return RETRY
    return SKIP_TRANSIENT


@dataclass
class SkipAccounting:
    """Exact per-run teacher skip/coverage accounting (mirrors TRL's skip_counts + teacher_transient).

    ``restored`` seeds cumulative counters from a resumed checkpoint so only NEW transient failures on
    this attempt make its shortfall retriable (TRL keeps a ``teacher_transient_baseline``)."""

    ok: int = 0
    transient: int = 0
    no_signal: int = 0
    skip_counts: Counter = field(default_factory=Counter)
    transient_baseline: int = 0
    steps_total: int = 0
    steps_with_signal: int = 0

    @classmethod
    def restored(cls, *, transient: int = 0, skip_counts: dict | None = None) -> SkipAccounting:
        acc = cls(transient=int(transient), transient_baseline=int(transient))
        if skip_counts:
            acc.skip_counts = Counter({str(k): int(v) for k, v in skip_counts.items()})
        return acc

    def record_ok(self) -> None:
        self.ok += 1

    def record_transient(self) -> None:
        self.transient += 1
        self.skip_counts[TRANSIENT] += 1

    def record_no_signal(self) -> None:
        self.no_signal += 1
        self.skip_counts["no_signal"] += 1

    def record_step(self, *, had_signal: bool) -> None:
        self.steps_total += 1
        if had_signal:
            self.steps_with_signal += 1

    @property
    def new_transient(self) -> int:
        """Transient failures observed on THIS attempt (excludes restored/cumulative ones)."""
        return max(0, self.transient - self.transient_baseline)

    def summary(self) -> str:
        parts = [f"{reason}={count}" for reason, count in sorted(self.skip_counts.items())]
        return (
            f"ok={self.ok} transient={self.transient} no_signal={self.no_signal} "
            f"steps={self.steps_with_signal}/{self.steps_total}"
            + (f" ({', '.join(parts)})" if parts else "")
        )


def classify_under_run(
    steps_with_signal: int, steps_expected: int, new_transient_failures: int
) -> str:
    """Terminal classification of a training run's signal coverage.

    Returns COMPLETE when every expected optimizer step landed on aligned teacher signal. When some
    expected steps produced no signal, the shortfall is TRANSIENT (retriable infra) if any NEW
    transient teacher failure occurred this attempt, otherwise a DETERMINISTIC shortfall the run
    could not align. Mirrors TRL's post-loop guard turning a transient-caused shortfall into a RETRY.
    """
    steps_with_signal = int(steps_with_signal)
    steps_expected = int(steps_expected)
    if steps_expected < 0 or steps_with_signal < 0:
        raise ValueError("step counts must be nonnegative")
    if steps_with_signal >= steps_expected:
        return COMPLETE
    if int(new_transient_failures) > 0:
        return TRANSIENT_SHORTFALL
    return DETERMINISTIC_SHORTFALL
