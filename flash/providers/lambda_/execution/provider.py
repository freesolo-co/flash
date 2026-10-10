"""Lambda Cloud provider: GPU instances bootstrapped via cloud-init."""

from __future__ import annotations

from typing import Any

from flash.providers._lifecycle.instances.instance import InstanceJobHandle
from flash.providers._lifecycle.instances.provider import InstanceProvider
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


class LambdaProvider(InstanceProvider):
    """``base.Provider`` for the Lambda Cloud substrate."""

    name = "lambda"
    supports_weight_cache = True
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

    def _submit_attempt(
        self,
        spec,
        *,
        log: Any,
        on_handle: Any,
        attempt: int,
        runtime_secrets: dict[str, str] | None,
        source_snapshot: dict | None,
        deadline_at: float | None,
    ) -> PollResult:
        from flash.providers.lambda_.jobs import submit_attempt_lambda

        return submit_attempt_lambda(
            spec,
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
        *,
        log: Any,
        heartbeat_reader: Any,
        deadline_at: float | None,
    ) -> PollResult:
        from flash.providers.lambda_.jobs import poll_lambda_job

        return poll_lambda_job(
            handle,
            spec,
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
        """
        from flash.providers.lambda_.client import api as lambda_api
        from flash.providers.lambda_.client.gpus import instance_type_disk_gb, instance_type_for
        from flash.providers.lambda_.jobs import usable_instances

        out: list[Candidate] = []
        counts = rentable_gpu_counts(constraints.max_gpu_count)
        try:
            catalog = lambda_api.list_instance_types()
            structurally_fitting = False
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
                    sku = instance_type_for(g.name, count, catalog)
                    if sku not in catalog:
                        continue
                    # A SKU whose fixed disk cannot hold the run is not capacity at all: renting it
                    # would pay for a box that dies mid-setup. An unreported disk is left alone.
                    sku_disk_gb = instance_type_disk_gb(catalog, sku)
                    if (
                        constraints.disk_gb
                        and sku_disk_gb is not None
                        and sku_disk_gb < constraints.disk_gb
                    ):
                        continue
                    structurally_fitting = True
                    live = usable_instances(g.name, gpu_count=count)
                    if live:
                        out.append(
                            Candidate("lambda", g.name, live[0].price_usd_hr, g.vram_gb, count)
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
