"""Recheck the exact shape immediately before bidding; never spend while waiting."""

import time

from flash.providers.core.base import CapacityLookupError
from flash.providers.mithril.capacity import quota_available
from flash.providers.mithril.client import MithrilApiError
from flash.providers.mithril.prices import affordable_offers


def wait_for_price(gpu, run_id, *, gpu_count=1, disk_gb=0, deadline_at, log=None):
    """Wait for any acceptable class to be below its cap with free quota; never spend meanwhile.

    ``gpu`` is one class or a tuple of interchangeable classes (``pricing.interchangeable``). The
    run takes whichever comes up first; when several qualify at once, priority order then price.
    """
    from flash.providers.mithril.pricing import priority
    from flash.runner.lifecycle.status import _update, get_status

    gpus = (gpu,) if isinstance(gpu, str) else tuple(gpu)
    from flash.runner.supervise.errors import _RunCancelled

    while time.time() < deadline_at:
        if get_status(run_id).state == "cancelled":
            raise _RunCancelled(f"run {run_id} was cancelled")
        try:
            offers = [
                offer
                for offer in affordable_offers(deadline_at=deadline_at)
                if offer.gpu in gpus and offer.count == gpu_count and offer.disk_gb >= disk_gb
            ]
            offers = [
                offer
                for offer in offers
                if quota_available(offer.instance_type, deadline_at=deadline_at)
            ]
            if offers:
                _update(run_id, "provisioning")
                # First class to come up wins; ties: class priority, price, region.
                return min(
                    offers,
                    key=lambda offer: (
                        priority(offer.gpu),
                        offer.hourly_usd,
                        offer.region,
                        offer.instance_type,
                    ),
                )
        except (CapacityLookupError, MithrilApiError):
            pass
        _update(run_id, "queued")
        print(
            f"Spot queued: waiting for {' or '.join(gpus)} below its cap with free quota",
            file=log,
            flush=True,
        )
        until = min(time.time() + 60, deadline_at)
        while time.time() < until:
            if get_status(run_id).state == "cancelled":
                raise _RunCancelled(f"run {run_id} was cancelled")
            time.sleep(max(0, min(1, until - time.time())))
    raise TimeoutError("Spot queue reached run deadline")
