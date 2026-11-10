"""Recheck admission immediately before provisioning; never spend while waiting."""

import time

from flash.providers.core.base import CapacityLookupError
from flash.providers.nebius import api
from flash.providers.nebius.config import preemptible_vm_limit, settings
from flash.providers.nebius.prices import affordable_gpus


def _wait(run_id, seconds, deadline_at):
    from flash.runner.lifecycle.status import get_status
    from flash.runner.supervise.errors import _RunCancelled

    until = min(time.time() + seconds, deadline_at)
    while time.time() < until:
        if get_status(run_id).state == "cancelled":
            raise _RunCancelled(f"run {run_id} was cancelled")
        time.sleep(max(0, min(1, until - time.time())))


def wait_for_slot(run_id, *, deadline_at, log=None):
    """Wait, queued and without spending, until the project has a free preemptible-VM slot.

    With FLASH_NEBIUS_PREEMPTIBLE_VM_LIMIT unset this returns immediately. A slot frees when
    another job's VM is deleted. Two waiters may see the same free slot; the loser's create is
    refused for capacity and requeued (see api.create), so the race only costs a retry.
    """
    from flash.runner.lifecycle.status import _update, get_status
    from flash.runner.supervise.errors import _RunCancelled

    limit = preemptible_vm_limit()
    if limit is None:
        return
    announced = False
    while time.time() < deadline_at:
        if get_status(run_id).state == "cancelled":
            raise _RunCancelled(f"run {run_id} was cancelled")
        try:
            in_use = api.preemptible_in_use(deadline_at=deadline_at)
        except (api.NebiusApiError, TimeoutError):
            return  # count unknown: try the create; a capacity refusal is requeued safely
        if in_use < limit:
            return
        _update(run_id, "queued")
        if not announced:
            print(
                f"Spot queued: all {limit} preemptible VM slots are in use; "
                "waiting for a running job to finish",
                file=log,
                flush=True,
            )
            announced = True
        _wait(run_id, 60, deadline_at)
    raise TimeoutError("Spot slot queue reached run deadline")


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
