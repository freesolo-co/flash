"""What cost to show for a run, and how sure we are of it.

This module is the canonical owner of run-cost selection.

This module holds no rendering of its own: it decides the NUMBER and whether that number is soft.
Turning either into styled text stays in `render`, which owns the palette.
"""

from __future__ import annotations

# the states after which `cost_usd` is the settled charge. before one of these the field is still
# 0.0, because the server only writes it on the terminal transition. kept stdlib-only rather than
# imported from flash.runner; the set is asserted against the runner's terminal states in the test
# suite so the two cannot drift.
SETTLED_COST_STATES = frozenset({"done", "failed", "cancelled", "dry_run", "deployed"})


def run_cost(obj: dict) -> tuple[float, bool]:
    """The cost to show for a run, and whether it is a pre-settlement estimate.

    ``cost_usd`` is only written when a run reaches a terminal state, so a run that is queued or
    training reports 0.0 -- which reads as "this has cost nothing", exactly when the GPU is billing.
    The submit-time quote is already on the record as ``estimated_cost_usd``; prefer it while the
    run is live so current spend is never understated as free.

    A FAILED run reporting 0.0 is the one terminal case where that zero is not a measurement
    either. ``cost_usd`` is derived from the WORKER's metrics, so a run whose attempts all died
    before producing any never had one written, and a failed run is the case most likely to have
    rented hardware repeatedly. one historical failure rented 47 instances, could not confirm
    teardown on 44 of them, and still reported $0.0000 as settled fact.

    Only ``failed`` gets that treatment. The other settled states earn their zero: ``dry_run``
    never rents anything, and a ``cancelled``/``done`` run with no charge went through the normal
    accounting path. Showing them a quote would invent a charge nobody incurred.
    """
    settled = float(obj.get("cost_usd") or 0.0)
    if settled:
        # a measured charge is the best number available, settled or not.
        return settled, str(obj.get("state") or "") not in SETTLED_COST_STATES
    quote = obj.get("estimated_cost_usd")
    if isinstance(quote, (int, float)) and (
        str(obj.get("state") or "") not in SETTLED_COST_STATES or _spent_without_measuring(obj)
    ):
        # the quote priced the work, not the failure, so it is not what a dead run cost -- but it
        # is the only non-zero evidence that money was at stake, and flagging it as unmeasured is
        # honest where printing a bare $0.0000 was not.
        return float(quote), True
    return settled, False


def _spent_without_measuring(obj: dict) -> bool:
    """A failed run that rented hardware but never produced a measured charge.

    ``realized_cost_usd`` is deliberately NOT used as the number to show: it is provider COGS
    pulled by reconciliation (see ``runner.RunStatus``), distinct from the ``cost_usd`` estimate
    the customer is charged. Presenting our internal spend as their bill would be wrong, and
    ``run_status`` already prints it on its own ``realized`` row.
    """
    return str(obj.get("state") or "") == "failed"


def cost_estimate_reason(obj: dict) -> str:
    """Why a flagged amount is soft: the run is unfinished, or it died unmeasured.

    A settled state with a flagged amount is the failed-run case -- the run is over and the quote
    stands in for a charge the worker never measured -- so "run in progress" would be false there.
    """
    if str(obj.get("state") or "") in SETTLED_COST_STATES:
        return "estimate, not measured"
    return "estimate, run in progress"
