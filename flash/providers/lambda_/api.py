"""Lambda instance catalog, launch, ownership and confirmed teardown.

Ownership is the instance name: every training VM is named ``instance_label(run_id, attempt)``, so
the run label prefix (``run_label_prefix``) identifies a run's instances on the account. Launch tags
repeat the run and attempt for operators reading the Lambda console.
"""

from __future__ import annotations

import contextlib
import re
import time
from dataclasses import dataclass

from flash.providers._lifecycle.instances.instance import run_label_prefix
from flash.providers.core.base import (
    CapacityLookupError,
    CapacityUnavailableError,
    UnreconciledCreateError,
)
from flash.providers.lambda_.client import LAUNCH_PATH, LambdaApiError, call
from flash.providers.lambda_.config import settings

# Short: capacity moves quickly, and launch decisions re-read it anyway.
CATALOG_TTL_S = 30.0
# After an ambiguous launch failure, how long to wait before listing the attempt's instances again.
LAUNCH_RECHECK_SECONDS = 10
TERMINATE_CONFIRM_SECONDS = 180.0
# Instance states that hold (or may still hold) a billed VM. Everything else is gone.
GONE_STATES = frozenset({"terminated"})
_INSUFFICIENT_CAPACITY = "instance-operations/launch/insufficient-capacity"
_INSTANCE_ID = re.compile(r"[A-Za-z0-9_-]{8,128}")
_OWNED_NAME = re.compile(r"flash-[A-Za-z0-9._-]+-a\d{1,10}")

_catalog: dict = {"at": 0.0, "data": None}


@dataclass(frozen=True)
class Offer:
    gpu: str
    count: int
    instance_type: str
    hourly_usd: float  # whole instance, all cards
    disk_gb: float | None  # fixed storage the type ships with; None when not reported
    regions: tuple[str, ...]  # regions with capacity right now, sorted


def valid_instance_id(raw: object) -> bool:
    return isinstance(raw, str) and bool(_INSTANCE_ID.fullmatch(raw))


def instance_types(*, force=False, deadline_at=None) -> dict:
    """The live ``/instance-types`` map, cached for ``CATALOG_TTL_S``."""
    now = time.time()
    if not force and _catalog["data"] is not None and now - _catalog["at"] < CATALOG_TTL_S:
        return _catalog["data"]
    data = call("GET", "/instance-types", deadline_at=deadline_at)
    if not isinstance(data, dict):
        raise LambdaApiError("Lambda returned an invalid instance-type catalog")
    _catalog.update(at=now, data=data)
    return data


def _disk_gb(instance) -> float | None:
    specs = instance.get("specs")
    if not isinstance(specs, dict):
        return None
    for key in ("storage_gib", "storage_gb"):
        value = specs.get(key)
        if not isinstance(value, bool) and isinstance(value, (int, float)) and value > 0:
            # GiB compared against a GB floor errs strict (a GiB is larger); never over-promises.
            return float(value)
    return None


def offers(*, force=False, deadline_at=None, require_capacity=True) -> list[Offer]:
    """Rentable x86_64 instance types Flash maps to a GPU class, with their live regions.

    ``require_capacity`` drops types no region can launch right now. Raises
    ``CapacityLookupError`` when the catalog cannot be read or parsed.
    """
    from flash.providers.lambda_.pricing import classify

    try:
        catalog = instance_types(force=force, deadline_at=deadline_at)
        result = []
        for name, entry in catalog.items():
            if not isinstance(entry, dict):
                continue
            instance = entry.get("instance_type")
            if not isinstance(instance, dict):
                continue
            shape = classify(name, instance)
            if shape is None or instance.get("name", name) != name:
                continue
            cents = instance.get("price_cents_per_hour")
            if isinstance(cents, bool) or not isinstance(cents, (int, float)) or cents <= 0:
                continue
            regions = tuple(
                sorted(
                    {
                        region["name"]
                        for region in entry.get("regions_with_capacity_available") or []
                        if isinstance(region, dict)
                        and isinstance(region.get("name"), str)
                        and region["name"]
                    }
                )
            )
            if require_capacity and not regions:
                continue
            result.append(
                Offer(shape[0], shape[1], name, cents / 100.0, _disk_gb(instance), regions)
            )
        return result
    except (LambdaApiError, TimeoutError, AttributeError, TypeError, ValueError, KeyError):
        raise CapacityLookupError("Lambda instance catalog unavailable") from None


def _instance_view(row) -> dict | None:
    if not isinstance(row, dict) or not valid_instance_id(row.get("id")):
        return None
    instance_type = row.get("instance_type")
    region = row.get("region")
    return {
        "id": row["id"],
        "name": row.get("name") if isinstance(row.get("name"), str) else "",
        "status": row.get("status") if isinstance(row.get("status"), str) else "",
        "instance_type": instance_type.get("name") if isinstance(instance_type, dict) else "",
        "region": region.get("name") if isinstance(region, dict) else "",
    }


def owned(instance: dict) -> bool:
    """Whether an instance carries a Flash training attempt name (``flash-<run>-a<N>``).

    The same Lambda account runs serving VMs (``flash-<env>-<32 hex>``,
    ``flash-h100-reservation-<date>-<n>``). None of them ends in ``-a<digits>``, so none is ever
    listed, swept or terminated by training.
    """
    return bool(_OWNED_NAME.fullmatch(instance.get("name") or ""))


def attempt_of(name: str, run_id: str) -> int | None:
    """The attempt number when ``name`` is exactly ``instance_label(run_id, attempt)``, else None.

    Exact: the run's label prefix, then ``-a``, then only digits. A bare prefix match is not enough
    (a serving VM ``flash-main-a3f...`` starts with ``flash-main-a`` for a run named ``main``).
    """
    if not _OWNED_NAME.fullmatch(name or ""):
        return None
    head = run_label_prefix(run_id) + "-a"
    suffix = name[len(head) :] if name.startswith(head) else ""
    return int(suffix) if suffix.isdigit() and len(suffix) <= 10 else None


def list_instances(*, deadline_at=None) -> list[dict]:
    """Every Flash training instance on the account that may still bill."""
    rows = call("GET", "/instances", deadline_at=deadline_at)
    if not isinstance(rows, list):
        raise LambdaApiError("Lambda returned an invalid instance listing")
    result = []
    for row in rows:
        view = _instance_view(row)
        if view is None:
            raise LambdaApiError("Lambda returned an invalid instance row")
        if owned(view) and view["status"] not in GONE_STATES:
            result.append(view)
    return result


def run_instances(run_id: str, *, deadline_at=None) -> list[dict]:
    return [
        item
        for item in list_instances(deadline_at=deadline_at)
        if attempt_of(item["name"], run_id) is not None
    ]


def get_instance(instance_id: str, *, name: str | None = None, deadline_at=None) -> dict | None:
    """One instance, or None once Lambda no longer knows it (or reports it terminated).

    ``name`` binds the lookup to the attempt that launched it: an id that resolves to an instance
    under any other name is never treated as ours.
    """
    if not valid_instance_id(instance_id):
        raise ValueError("invalid Lambda instance id")
    try:
        row = call("GET", f"/instances/{instance_id}", deadline_at=deadline_at)
    except LambdaApiError as error:
        if error.status == 404:
            return None
        raise
    view = _instance_view(row)
    if view is None or view["id"] != instance_id:
        raise LambdaApiError("Lambda returned an invalid instance")
    if name is not None and view["name"] != name:
        raise LambdaApiError("Lambda instance does not belong to this attempt")
    if view["status"] in GONE_STATES:
        return None
    return view


def terminate_confirmed(instance_id: str, *, name: str | None = None) -> None:
    """Terminate one exact instance and confirm it is gone.

    Per id: Lambda's batch terminate rejects the whole request when one id is bad, which would leak
    every other VM in it. No run-deadline dependency, so a VM that outlived its run is still
    removable. Raises unless disappearance (or the ``terminated`` state) is observed.

    Refuses (before any terminate request) an instance that is not a Flash training VM, whatever
    id a caller passes: serving VMs on the same account are never terminated here.
    """
    deadline = time.time() + TERMINATE_CONFIRM_SECONDS

    def request():
        # a lost or refused terminate is settled by observing the instance afterwards
        with contextlib.suppress(LambdaApiError):
            call(
                "POST",
                "/instance-operations/terminate",
                body={"instance_ids": [instance_id]},
                deadline_at=deadline,
            )

    current = get_instance(instance_id, name=name, deadline_at=deadline)
    if current is None:
        return
    if not owned(current):
        raise LambdaApiError("Lambda instance is not a Flash training VM; refusing to terminate")
    request()
    while time.time() < deadline:
        instance = get_instance(instance_id, name=name, deadline_at=deadline)
        if instance is None:
            return
        if instance["status"] != "terminating":
            request()  # still up: the terminate was lost or refused, and it is safe to repeat
        time.sleep(min(3, max(0, deadline - time.time())))
    raise LambdaApiError("Lambda instance termination is not confirmed")


def _terminate_earlier_attempts(run_id: str, attempt: int, *, deadline_at) -> None:
    """Terminate this run's VMs from earlier attempts before launching a new one.

    An ambiguous launch requeues under a new attempt name, and Lambda can still create the old
    attempt's VM afterwards. Only one attempt of a run is ever live, so an earlier attempt's VM is
    never one Flash is still supervising.
    """
    for item in list_instances(deadline_at=deadline_at):
        earlier = attempt_of(item["name"], run_id)
        if earlier is not None and earlier < attempt:
            terminate_confirmed(item["id"], name=item["name"])


def launch(*, name, run_id, attempt, offer: Offer, region, user_data, deadline_at) -> str:
    """Launch exactly one VM for this attempt and return its id. Never retried.

    An ambiguous failure (5xx, transport error, unreadable 2xx) is reconciled by listing instances
    under this attempt's name; a definitive refusal raises ``CapacityUnavailableError`` so the run
    requeues instead of failing.
    """
    if region not in offer.regions:
        raise ValueError("Lambda region does not have capacity for this instance type")

    def reconcile():
        matches = [item for item in list_instances(deadline_at=deadline_at) if item["name"] == name]
        if len(matches) > 1:
            raise UnreconciledCreateError("Multiple Lambda instances exist for this attempt")
        if not matches:
            return None
        if matches[0]["instance_type"] not in ("", offer.instance_type):
            raise UnreconciledCreateError("Lambda attempt name belongs to another launch")
        return matches[0]["id"]

    existing = reconcile()
    if existing:
        return existing
    if int(attempt) > 0:
        try:
            _terminate_earlier_attempts(run_id, int(attempt), deadline_at=deadline_at)
        except (LambdaApiError, TimeoutError, ValueError):
            # Without confirming earlier attempts hold no VM, a new one could run beside a late
            # one. Requeue instead; the next attempt retries the cleanup first.
            raise CapacityUnavailableError(
                "Could not confirm earlier Lambda attempts are terminated; requeuing",
                cause="earlier attempt cleanup unavailable",
            ) from None
    body = {
        "region_name": region,
        "instance_type_name": offer.instance_type,
        "ssh_key_names": [settings()["ssh_key_name"]],
        "name": name,
        "quantity": 1,
        "user_data": user_data,
        "tags": [
            {"key": "flash-owner", "value": "flash-training"},
            {"key": "flash-attempt", "value": str(int(attempt))},
        ],
    }
    try:
        result = call("POST", LAUNCH_PATH, body=body, deadline_at=deadline_at, retries=0)
        ids = result.get("instance_ids") if isinstance(result, dict) else None
        if not isinstance(ids, list) or len(ids) != 1 or not valid_instance_id(ids[0]):
            raise ValueError("invalid launch identity")
        return ids[0]
    except (LambdaApiError, TimeoutError, ValueError) as error:
        status = error.status if isinstance(error, LambdaApiError) else 0
        code = error.code if isinstance(error, LambdaApiError) else ""
        if status:
            cause = f"http {status}" + (f": {code}" if code else "")
        elif isinstance(error, (LambdaApiError, TimeoutError)):
            cause = "transport error or timeout"
        else:
            cause = "invalid launch response"

        def reconciled():
            try:
                return reconcile()
            except (LambdaApiError, TimeoutError):
                raise UnreconciledCreateError(
                    "Lambda launch reconciliation unavailable; refusing duplicate allocation",
                    cause=cause,
                ) from None

        existing = reconciled()
        if existing:
            return existing
        if 400 <= status < 500:
            if status == 429 or code == _INSUFFICIENT_CAPACITY:
                raise CapacityUnavailableError(
                    f"Lambda has no capacity for {offer.instance_type} in {region} ({cause}); "
                    "requeuing",
                    cause=cause,
                ) from None
            # Any other 4xx is a definitive refusal of this request (bad key name, quota, ...):
            # nothing was created, and resubmitting the same body cannot change the answer.
            raise LambdaApiError(
                f"Lambda refused the launch ({cause})", status=status, code=code
            ) from None
        # An ambiguous outcome may still create the VM after the listing above. Look once more
        # after a bounded pause and adopt it; otherwise requeue (the next attempt terminates any
        # VM this one creates late, before it launches).
        time.sleep(max(0, min(LAUNCH_RECHECK_SECONDS, deadline_at - time.time())))
        existing = reconciled()
        if existing:
            return existing
        raise CapacityUnavailableError(
            f"Lambda launch outcome unknown ({cause}) and no instance appeared; requeuing",
            cause=cause,
        ) from None
