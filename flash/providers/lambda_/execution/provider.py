"""Lambda Cloud provider: GPU instances bootstrapped via cloud-init."""

from __future__ import annotations

from typing import Any

from flash._internal.logging import get_logger
from flash.providers._lifecycle.instances.instance import InstanceJobHandle
from flash.providers._lifecycle.instances.provider import InstanceProvider
from flash.providers.core._decoding import MalformedProviderFieldError
from flash.providers.core.base import (
    AllocationConstraints,
    Candidate,
    CapacityLookupError,
    JobHandle,
    PollResult,
    Provider,
    UnsupportedGpuError,
    rentable_gpu_counts,
)
from flash.providers.core.sharding import combined_vram_gb

logger = get_logger(__name__)


def _carries_any_entry(catalog: object) -> bool:
    """Whether the catalog is shaped like Lambda's own ``/instance-types`` map.

    ``/instance-types`` lists every type Lambda sells regardless of stock -- availability is the
    per-entry ``regions_with_capacity_available`` field, not omission -- so an empty or entry-less
    response is a broken feed rather than an empty product line. This is what makes a MISSING key
    trustworthy evidence that a shape is not sold: absence only means something once the catalog
    around it is known to be a catalog.
    """
    return isinstance(catalog, dict) and any(isinstance(entry, dict) for entry in catalog.values())


def _sku_holds_run(catalog: dict, sku: str, constraints: AllocationConstraints) -> bool:
    """Whether one catalog SKU's fixed disk can hold the run.

    Lambda ships storage WITH the instance type and takes no launch-time disk parameter, so renting
    an undersized box pays for a machine that dies mid-setup. An unreported disk is left alone -- a
    caller must not invent a refusal the catalog cannot prove.

    A malformed catalog field raises ``MalformedProviderFieldError`` so the caller can contain it to
    this one SKU instead of losing every sibling shape.
    """
    from flash.providers.lambda_.client.gpus import instance_type_disk_gb

    sku_disk_gb = instance_type_disk_gb(catalog, sku)
    return not (
        constraints.disk_gb and sku_disk_gb is not None and sku_disk_gb < constraints.disk_gb
    )


class LambdaProvider(InstanceProvider):
    """``base.Provider`` for the Lambda Cloud substrate."""

    name = "lambda"
    _gpu_identity_attr = "lambda_name"

    @property
    def _handle_cls(self) -> type[InstanceJobHandle]:
        from flash.providers.lambda_.jobs import LambdaJobHandle

        return LambdaJobHandle

    def _load_api_key(self) -> Any:
        from flash.providers.lambda_.client.auth import load_api_key

        return load_api_key()

    def _missing_credentials(self, require_hf: bool) -> list[str]:
        from flash.providers.lambda_.client.preflight import missing_credentials

        return missing_credentials(require_hf=require_hf)

    def _hourly_rate(self, gpu: str) -> float:
        from flash.providers.lambda_.client.pricing import hourly_rate

        return hourly_rate(gpu)

    def _submit_run(
        self,
        spec,
        seed: int,
        *,
        log: Any,
        on_handle: Any,
        attempt: int,
        runtime_secrets: dict[str, str] | None,
        source_snapshot: dict | None,
        deadline_at: float | None,
    ) -> PollResult:
        from flash.providers.lambda_.jobs import submit_run_lambda

        return submit_run_lambda(
            spec,
            seed,
            log=log,
            on_handle=on_handle,
            attempt=attempt,
            runtime_secrets=runtime_secrets,
            source_snapshot=source_snapshot,
            deadline_at=deadline_at,
        )

    def _poll_job(
        self,
        handle: JobHandle,
        spec,
        seed: int,
        *,
        log: Any,
        heartbeat_reader: Any,
        deadline_at: float | None,
    ) -> PollResult:
        from flash.providers.lambda_.jobs import poll_lambda_job

        return poll_lambda_job(
            handle,
            spec,
            seed,
            log=log,
            heartbeat_reader=heartbeat_reader,
            deadline_at=deadline_at,
        )

    def _teardown_reattached(self, handle: JobHandle, spec) -> None:
        from flash.providers.lambda_.client import api as lambda_api

        lambda_api.terminate_instance_confirmed(handle.instance_id)

    def _gc(self, run_id: str) -> None:
        from flash.providers.lambda_.jobs import terminate_run_instances

        terminate_run_instances(run_id)

    def _sweep_orphans(
        self,
        *,
        active_labels,
        known_labels,
    ) -> list[str]:
        from flash.providers.lambda_.jobs import sweep_orphans

        return sweep_orphans(active_labels=active_labels, known_labels=known_labels)

    def run_instances_remaining(self, run_id: str) -> list[str]:
        """Exact run-owned instance ids still observable after cleanup."""
        from flash.providers.lambda_.jobs import run_instances_remaining

        return run_instances_remaining(run_id)

    def live_candidates(
        self, need_vram_gb: int, constraints: AllocationConstraints
    ) -> list[Candidate]:
        """Lambda classes with live regional capacity fitting the VRAM requirement.

        A capacity-lookup failure raises ``CapacityLookupError``; ``allocate`` degrades to the other
        providers, failing the run retryably only if this was the sole fitting source.
        ``constraints.max_wall_seconds`` is ignored (Lambda rents open-ended), but
        ``constraints.disk_gb`` is not: Lambda sells a FIXED disk with the instance type and takes
        no launch-time disk parameter, so a SKU that cannot hold the run's floor is filtered here
        exactly as Vast filters thin-disk offers out of its search.

        Lambda sells fixed card counts per class as distinct instance types, so each allowed count
        is probed against the live catalog and only the counts with real capacity are reported. The
        rate stays per-card (``usable_instances`` divides the per-instance price).

        A malformed catalog field is contained to the SKU carrying it, exactly as a malformed Vast
        row is dropped from the offer search: one bad field must not delete every valid sibling
        shape and take the whole provider out of the allocation. A catalog carrying no entry at
        all, or one where every decode attempt fails and none succeeds, is not a bad field but a
        broken feed, and still fails the lookup retryably.

        A shape simply absent from a well-formed catalog is the opposite: a complete answer that
        Lambda does not sell it, which no retry can change. It is therefore excluded from the
        broken-feed evidence entirely and left to fall through to ``UnsupportedGpuError``.
        """
        from flash.providers.lambda_.client import api as lambda_api
        from flash.providers.lambda_.client.gpus import instance_type_for
        from flash.providers.lambda_.jobs import usable_instances

        out: list[Candidate] = []
        counts = rentable_gpu_counts(constraints.max_gpu_count)
        try:
            catalog = lambda_api.list_instance_types()
            structurally_fitting = False
            malformed = 0
            decoded = 0
            for g in self.gpu_classes():
                if constraints.gpu_type and g.name != constraints.gpu_type:
                    continue
                if g.vram_gb < need_vram_gb:
                    continue
                for count in counts:
                    # The allocator passes a reduced per-card market floor. Keep the whole-run floor
                    # too so both exact and unpinned searches can distinguish "Lambda does not sell
                    # this shape" from "the sold SKU has no region free right now".
                    if (
                        constraints.required_vram_gb
                        and combined_vram_gb(g.vram_gb, count) < constraints.required_vram_gb
                    ):
                        continue
                    try:
                        sku = instance_type_for(g.name, count, catalog)
                        if sku not in catalog:
                            # Lambda does not sell this shape. A missing key is a COMPLETE answer
                            # that needed no decoding, so it is evidence of neither health nor
                            # breakage and belongs in neither tally.
                            continue
                        if not _sku_holds_run(catalog, sku, constraints):
                            decoded += 1
                            continue
                        structurally_fitting = True
                        live = usable_instances(g.name, gpu_count=count)
                    except MalformedProviderFieldError as exc:
                        # A decode failure is evidence about FEED HEALTH wherever in the shape it
                        # happens -- resolving the multi-card name reads the catalog's own
                        # ``gpu_count`` and VRAM fields, so a corrupt sibling entry aborts here
                        # before the sku is ever in hand. Tallying it keeps the broken-feed gate
                        # able to see it; not tallying it let a corrupt N-card entry read exactly
                        # like a healthy catalog that does not sell the shape, and a retryable
                        # outage surfaced as a terminal refusal.
                        logger.warning("dropping malformed lambda catalog sku: %s", exc)
                        malformed += 1
                        continue
                    decoded += 1
                    if live:
                        out.append(
                            Candidate("lambda", g.name, live[0].price_usd_hr, g.vram_gb, count)
                        )
            if not _carries_any_entry(catalog) or (malformed and not decoded):
                raise MalformedProviderFieldError(
                    "lambda", "instance-types", "at least one well-formed sku"
                )
            if constraints.required_vram_gb and not structurally_fitting:
                requested = f" {constraints.gpu_type}" if constraints.gpu_type else ""
                disk = f" with {constraints.disk_gb:g} GB of disk" if constraints.disk_gb else ""
                raise UnsupportedGpuError(
                    f"lambda does not offer a rentable{requested} shape up to "
                    f"{constraints.max_gpu_count} cards large enough for "
                    f"{constraints.required_vram_gb} GB{disk}"
                )
        except UnsupportedGpuError:
            raise
        except Exception as exc:
            # Transient capacity-lookup blip -> signal allocate() so it degrades to the other providers but
            # can still tell "no fit" from "outage" if this was the only fitting source (see CapacityLookupError).
            raise CapacityLookupError("lambda live capacity lookup failed") from exc
        return out

    def cancel(self, handle: JobHandle) -> None:
        _terminate_handle_instance(handle)

    destroy = cancel


PROVIDER: Provider = LambdaProvider()


def _terminate_handle_instance(handle: JobHandle) -> None:
    from flash.providers.lambda_.client import api as lambda_api

    d = handle.to_dict()
    if d.get("instance_id"):
        lambda_api.terminate_instance_confirmed(str(d["instance_id"]))
