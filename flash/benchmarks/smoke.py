"""Deterministic offline exercise of the evidence pipeline. Never GPU benchmark results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from flash.benchmarks.evidence import Recorder
from flash.benchmarks.schema import (
    Cache,
    Case,
    Complete,
    Configuration,
    Ledger,
    Money,
    Phase,
    Provenance,
    Quality,
    Request,
    Span,
)


def main(request_path: Path) -> None:
    request = json.loads(request_path.read_text())
    if not request["synthetic"]:
        raise ValueError("smoke driver only accepts explicitly synthetic requests")
    case = Case.model_validate(request["case"])
    recorder = Recorder(request_path.parent / "events.jsonl", request["origin_monotonic_s"])
    recorder.emit(
        Provenance(
            revisions=case.revisions,
            configuration_digest=request["configuration_digest"],
            resolved_configuration=Configuration.model_validate(request["configuration"]),
            evidence="synthetic",
        )
    )
    recorder.emit(Cache(observed=request["cache"], evidence="synthetic"))
    startup = 10.0 if request["cache"] == "cold" else 2.0
    phases: list[tuple[Phase, float, float]] = [
        ("allocation", 0.0, 1.0),
        ("startup", 1.0, startup),
        ("rollout", startup, startup + 4),
        ("environment_reward", startup + 2, startup + 4),
        ("training", startup + 4, startup + 6),
        ("weight_publication", startup + 6, startup + 7),
        ("checkpointing", startup + 6, startup + 8),
        ("evaluation", startup + 8, startup + 10),
    ]
    for phase, start, end in phases:
        recorder.emit(Span(phase=phase, start_s=start, end_s=end, source="synthetic"))
    end = startup + 10
    for index in range(2):
        recorder.emit(
            Request(
                start_s=end - 2,
                end_s=end,
                customer=f"customer-{index}",
                adapter=f"adapter-{index}",
                success=True,
                ttft_s=0.25,
            )
        )
    recorder.emit(
        Quality(
            at_s=end,
            metric=case.target.metric,
            value=case.target.value,
            samples=case.target.min_samples,
            checkpoint="synthetic/final",
        )
    )
    amounts: list[tuple[Ledger, float]] = [
        ("infrastructure_actual", 0.1),
        ("customer_charge", 0.25),
    ]
    for ledger, amount in amounts:
        recorder.emit(
            Money(
                start_s=0.0,
                end_s=end,
                ledger=ledger,
                amount_usd=amount,
                receipt="synthetic",
                line_item=ledger,
                resource="synthetic-gpu",
            )
        )
    recorder.emit(
        Complete(
            at_s=end,
            outcome="succeeded",
            complete_ledgers=["infrastructure_actual", "customer_charge"],
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
