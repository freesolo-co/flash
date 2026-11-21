"""Recheck the exact shape immediately before bidding; never spend while waiting."""

import contextlib
import time

from flash.providers.core.base import CapacityLookupError
from flash.providers.mithril import queue_board
from flash.providers.mithril.capacity import quota_available
from flash.providers.mithril.client import MithrilApiError
from flash.providers.mithril.prices import affordable_offers
from flash.providers.mithril.region_health import boot_failures


def _raise_if_cancelled(run_id):
    from flash.runner.lifecycle.status import get_status
    from flash.runner.supervise.errors import _RunCancelled

    if get_status(run_id).state == "cancelled":
        queue_board.leave(run_id, allocated=False)
        raise _RunCancelled(f"run {run_id} was cancelled")


def set_account_error(run_id, text):
    """Set (``text``) or clear (``None``) a run's account-refusal ``error``; diagnostic only.

    Only an account-refusal error is ever replaced or cleared, and never on a terminal run.
    """
    from flash.runner.lifecycle import reporting, state
    from flash.runner.lifecycle.status import get_status

    with contextlib.suppress(Exception):
        with state._status_guard(run_id):
            status = get_status(run_id)
            current = status.error
            if status.state in state.TERMINAL_STATES or current == text:
                return
            if current and not queue_board.is_account_refusal(current):
                return
            status.error = text
            state._save_status_unlocked(status)
        reporting._report_status(status)


def _account_error(run_id):
    from flash.runner.lifecycle.status import get_status

    with contextlib.suppress(Exception):
        error = get_status(run_id).error
        if queue_board.is_account_refusal(error):
            return error
    return None


def _queue_reason(lookup_failed, shaped_offers, view, account_error=None):
    """Why the run is still waiting, as ``(reason, detail)``; quota evidence wins over price.

    An account or billing refusal wins over everything: no bid can succeed until it is fixed.
    """
    if account_error:
        return queue_board.REASON_ACCOUNT, account_error
    if shaped_offers and not lookup_failed:
        # an offer below the cap exists, and every such shape's quota lookup answered "used up".
        return queue_board.REASON_QUOTA, "project Spot quota for every affordable shape is in use"
    if view is not None and view.quota_refused_at is not None:
        minutes = max(0, int((view.at - view.quota_refused_at) // 60))
        return (
            queue_board.REASON_QUOTA,
            f"Mithril refused a Flash bid for GPU quota {minutes} min ago",
        )
    if lookup_failed:
        return queue_board.REASON_LOOKUP, "Mithril prices or quotas could not be read; retrying"
    return queue_board.REASON_PRICE, (
        "no matching Spot offer in a US region is at or below the GPU price cap"
    )


def wait_for_price(gpu, run_id, *, gpu_count=1, disk_gb=0, deadline_at, log=None):
    """Wait for any acceptable class to be below its cap with free quota; never spend meanwhile.

    ``gpu`` is one class or a tuple of interchangeable classes (``pricing.interchangeable``). The
    run takes whichever comes up first; when several qualify at once, priority order then price.
    While it waits, status and log say why, where the run stands among Flash's queued Spot runs,
    and a rough ETA when recent allocations support one.
    """
    from flash.providers.mithril.pricing import priority
    from flash.runner.lifecycle.startup import record_startup_stage
    from flash.runner.lifecycle.status import _update
    from flash.runner.supervise.errors import _RunCancelled

    gpus = (gpu,) if isinstance(gpu, str) else tuple(gpu)

    while time.time() < deadline_at:
        _raise_if_cancelled(run_id)
        lookup_failed, shaped = False, []
        try:
            shaped = [
                offer
                for offer in affordable_offers(deadline_at=deadline_at)
                if offer.gpu in gpus and offer.count == gpu_count and offer.disk_gb >= disk_gb
            ]
            offers = [
                offer
                for offer in shaped
                if quota_available(offer.instance_type, deadline_at=deadline_at)
            ]
            if offers:
                # a cancel landing during the lookups must win: never hand back an offer to bid on.
                _raise_if_cancelled(run_id)
                if _update(run_id, "provisioning") is False:
                    queue_board.leave(run_id, allocated=False)
                    raise _RunCancelled(f"run {run_id} went terminal while queued")
                # First class to come up wins; ties: class priority, then the pool (region and
                # instance type) with the fewest recent boot timeouts, then price and region. The
                # class order is unchanged; within a class, a pool whose VMs keep failing to boot
                # is only bid on when no healthier pool has an offer.
                failures = {
                    (offer.region, offer.instance_type): boot_failures(
                        offer.region, offer.instance_type
                    )
                    for offer in offers
                }
                return min(
                    offers,
                    key=lambda offer: (
                        priority(offer.gpu),
                        failures[(offer.region, offer.instance_type)],
                        offer.hourly_usd,
                        offer.region,
                        offer.instance_type,
                    ),
                )
        except (CapacityLookupError, MithrilApiError):
            lookup_failed = True
        view = queue_board.join(run_id, gpus)
        reason, detail = _queue_reason(lookup_failed, shaped, view, _account_error(run_id))
        _update(run_id, "queued")
        record_startup_stage(
            run_id,
            "waiting_for_gpu",
            extra=queue_board.status_extra(view, reason),
            queue=None if view is None else view.public(reason, detail),
        )
        print(queue_board.queue_line(view, gpus, reason, detail), file=log, flush=True)
        until = min(time.time() + 60, deadline_at)
        while time.time() < until:
            _raise_if_cancelled(run_id)
            time.sleep(max(0, min(1, until - time.time())))
    queue_board.leave(run_id, allocated=False)
    raise TimeoutError("Spot queue reached run deadline")
