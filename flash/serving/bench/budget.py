"""Prospective budget ledger enforcing the campaign's hard spend ceiling.

The ceiling is enforced PROSPECTIVELY: capacity is reserved before a GPU is allocated and settled
against measured seconds afterwards. Checking spend after the fact cannot stop an overrun, because
by then the money is gone.

Three conservative choices, all deliberate:

* A reservation is held at its full estimate until settled, so concurrent lanes cannot both spend the
  same headroom.
* Settling never releases more than was reserved for that entry, so an under-run frees real headroom
  while an over-run is recorded at its true cost rather than being clamped away.
* ``ceiling_usd`` has NO default. The authorized amount comes from the user, so a default here would
  be an authorization this module invented for itself.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Modal list rates per GPU-hour, snapshot date below. Recorded in the report as a snapshot: a rate
# change invalidates the spend figures, not the measurements. Each hosted tier is measured on its own
# card, so one campaign rate cannot cover the sweep.
USD_PER_GPU_HOUR: dict[str, float] = {
    "L40S": 1.9512,
    "H100": 3.9492,
    "H100!": 3.9492,
    "H200": 4.5396,
    "B200": 6.2496,
}
RATE_SNAPSHOT_DATE = "2026-08-29"

# Fraction of the ceiling above which no NEW lane may start, preserving the remainder for delayed
# charges and teardown.
SUBMISSION_STOP_FRACTION = 0.80


class BudgetExceeded(RuntimeError):
    """Raised instead of allocating when a reservation would breach the ceiling."""


class UnknownGpuRate(KeyError):
    """Raised when a tier has no recorded rate.

    Fail closed: estimating an unknown card at some neighbour's rate is how a sweep overspends
    without ever tripping the ceiling.
    """


def rate_for_gpu(gpu: str) -> float:
    """List rate per GPU-hour for a Modal tier spelling, or raise."""
    key = gpu.strip().upper()
    if key in USD_PER_GPU_HOUR:
        return USD_PER_GPU_HOUR[key]
    # Modal spells a non-preemptible pin with a trailing "!"; the rate is the same card.
    stripped = key.rstrip("!")
    if stripped in USD_PER_GPU_HOUR:
        return USD_PER_GPU_HOUR[stripped]
    raise UnknownGpuRate(f"no recorded USD/GPU-hour for tier {gpu!r}")


def usd_for_gpu_seconds(seconds: float, gpu: str) -> float:
    """Cost of ``seconds`` on ``gpu``. The tier is required, never defaulted."""
    if seconds < 0:
        raise ValueError("seconds must be non-negative")
    return seconds / 3600.0 * rate_for_gpu(gpu)


@dataclass
class LedgerEntry:
    label: str
    gpu: str
    reserved_usd: float
    settled_usd: float | None = None
    gpu_seconds: float | None = None
    note: str = ""

    @property
    def effective_usd(self) -> float:
        """Reserved until settled; actual afterwards."""
        return self.reserved_usd if self.settled_usd is None else self.settled_usd

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "gpu": self.gpu,
            "reserved_usd": round(self.reserved_usd, 4),
            "settled_usd": None if self.settled_usd is None else round(self.settled_usd, 4),
            "gpu_seconds": self.gpu_seconds,
            "effective_usd": round(self.effective_usd, 4),
            "note": self.note,
        }


@dataclass
class BudgetLedger:
    """Tracks reservations and settlements against the campaign ceiling."""

    ceiling_usd: float
    submission_stop_usd: float | None = None
    entries: list[LedgerEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Finiteness is checked BEFORE positivity, because the positivity test cannot catch either
        # non-finite spelling: `nan <= 0` is False and `inf <= 0` is False, so both would install a
        # ceiling this ledger can never enforce. `projected > nan` is False for every projection, and
        # every finite projection is below `inf`, so `reserve` would approve unbounded spend while
        # reporting a configured ceiling.
        if not math.isfinite(self.ceiling_usd):
            raise ValueError("ceiling_usd must be a finite dollar amount")
        if self.ceiling_usd <= 0:
            raise ValueError("ceiling_usd must be positive and explicitly authorized")
        if self.submission_stop_usd is None:
            self.submission_stop_usd = self.ceiling_usd * SUBMISSION_STOP_FRACTION
        elif not math.isfinite(self.submission_stop_usd):
            raise ValueError("submission_stop_usd must be a finite dollar amount")
        elif self.submission_stop_usd < 0:
            raise ValueError("submission_stop_usd must not be negative")

    @property
    def committed_usd(self) -> float:
        return sum(entry.effective_usd for entry in self.entries)

    @property
    def remaining_usd(self) -> float:
        return self.ceiling_usd - self.committed_usd

    def can_submit(self, estimated_seconds: float, gpu: str) -> bool:
        """Whether a new lane of this size on this tier may start.

        Gated on ``submission_stop_usd`` rather than the ceiling so the reserve for delayed charges
        and teardown is never consumed by a new submission.
        """
        projected = self.committed_usd + usd_for_gpu_seconds(estimated_seconds, gpu)
        # `is None` rather than truthiness: an explicit `submission_stop_usd=0` is the emergency
        # stop, and `or` would read it as unset and restore the full ceiling -- permitting spend in
        # the one configuration whose whole purpose is to permit none.
        stop = self.ceiling_usd if self.submission_stop_usd is None else self.submission_stop_usd
        return projected <= stop

    def reserve(
        self, label: str, estimated_seconds: float, gpu: str, note: str = ""
    ) -> LedgerEntry:
        """Reserve headroom for a NEW lane, or raise ``BudgetExceeded``.

        Gated on ``can_submit`` -- i.e. on ``submission_stop_usd`` -- and not on the raw ceiling.
        Checking only the ceiling here made the submission stop advisory: a lane projecting between
        the stop and the ceiling was admitted, consuming the very reserve held back for delayed
        charges and teardown, and the reserve existed only in whatever code remembered to call
        ``can_submit`` first. Every caller of ``reserve`` is starting a new lane, so the stop is the
        correct bar for all of them.
        """
        amount = usd_for_gpu_seconds(estimated_seconds, gpu)
        projected = self.committed_usd + amount
        if projected > self.ceiling_usd:
            raise BudgetExceeded(
                f"reserving {label!r} on {gpu} (${amount:.2f}) would reach ${projected:.2f}, "
                f"over the ${self.ceiling_usd:.2f} ceiling"
            )
        if not self.can_submit(estimated_seconds, gpu):
            stop = (
                self.ceiling_usd if self.submission_stop_usd is None else self.submission_stop_usd
            )
            raise BudgetExceeded(
                f"reserving {label!r} on {gpu} (${amount:.2f}) would reach ${projected:.2f}, "
                f"past the ${stop:.2f} submission stop held back for delayed charges and teardown "
                f"under the ${self.ceiling_usd:.2f} ceiling"
            )
        entry = LedgerEntry(label=label, gpu=gpu, reserved_usd=amount, note=note)
        self.entries.append(entry)
        return entry

    def settle(self, entry: LedgerEntry, actual_seconds: float, note: str = "") -> None:
        """Record a lane's measured cost, replacing its reservation.

        Settled at the entry's OWN tier, so a mixed-tier sweep cannot be settled at one rate.
        """
        entry.settled_usd = usd_for_gpu_seconds(actual_seconds, entry.gpu)
        entry.gpu_seconds = actual_seconds
        if note:
            entry.note = f"{entry.note}; {note}".strip("; ")

    def to_json(self) -> dict[str, Any]:
        return {
            "rate_snapshot_date": RATE_SNAPSHOT_DATE,
            "usd_per_gpu_hour": dict(USD_PER_GPU_HOUR),
            "ceiling_usd": self.ceiling_usd,
            "submission_stop_usd": self.submission_stop_usd,
            "committed_usd": round(self.committed_usd, 4),
            "remaining_usd": round(self.remaining_usd, 4),
            "entries": [entry.to_json() for entry in self.entries],
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")


__all__ = [
    "RATE_SNAPSHOT_DATE",
    "SUBMISSION_STOP_FRACTION",
    "USD_PER_GPU_HOUR",
    "BudgetExceeded",
    "BudgetLedger",
    "LedgerEntry",
    "UnknownGpuRate",
    "rate_for_gpu",
    "usd_for_gpu_seconds",
]
