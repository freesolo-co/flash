"""Recheck admission immediately before provisioning; never spend while waiting."""

import time

from flash.providers.core.base import CapacityLookupError
from flash.providers.nebius.config import settings
from flash.providers.nebius.prices import affordable_gpus


def wait_for_price(gpu, run_id, *, deadline_at, log=None):
    from flash.runner.lifecycle.status import _update, get_status
    from flash.runner.supervise.errors import _RunCancelled

    while time.time() < deadline_at:
        if get_status(run_id).state == "cancelled":
            raise _RunCancelled(f"run {run_id} was cancelled")
        try:
            if gpu in affordable_gpus(settings()["region"]):
                _update(run_id, "provisioning")
                return
        except CapacityLookupError:
            pass
        _update(run_id, "queued")
        print("Spot queued: waiting for a fresh Nebius price below the cap", file=log, flush=True)
        until = min(time.time() + 60, deadline_at)
        while time.time() < until:
            if get_status(run_id).state == "cancelled":
                raise _RunCancelled(f"run {run_id} was cancelled")
            time.sleep(max(0, min(1, until - time.time())))
    raise TimeoutError("Spot queue reached run deadline")
