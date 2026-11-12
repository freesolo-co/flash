"""Recheck the exact shape immediately before bidding; never spend while waiting."""

import time

from flash.providers.core.base import CapacityLookupError
from flash.providers.mithril.config import settings
from flash.providers.mithril.prices import affordable_offers


def wait_for_price(gpu, run_id, *, gpu_count=1, disk_gb=0, deadline_at, log=None):
    from flash.runner.lifecycle.status import _update, get_status
    from flash.runner.supervise.errors import _RunCancelled

    while time.time() < deadline_at:
        if get_status(run_id).state == "cancelled":
            raise _RunCancelled(f"run {run_id} was cancelled")
        try:
            offers = [
                offer
                for offer in affordable_offers(settings()["region"], deadline_at=deadline_at)
                if offer.gpu == gpu and offer.count == gpu_count and offer.disk_gb >= disk_gb
            ]
            if offers:
                _update(run_id, "provisioning")
                return min(offers, key=lambda offer: (offer.hourly_usd, offer.instance_type))
        except CapacityLookupError:
            pass
        _update(run_id, "queued")
        print("Spot queued: waiting for a Mithril shape below the cap", file=log, flush=True)
        until = min(time.time() + 60, deadline_at)
        while time.time() < until:
            if get_status(run_id).state == "cancelled":
                raise _RunCancelled(f"run {run_id} was cancelled")
            time.sleep(max(0, min(1, until - time.time())))
    raise TimeoutError("Spot queue reached run deadline")
