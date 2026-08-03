"""Lambda Cloud provider: GPU instances bootstrapped via cloud-init."""

from __future__ import annotations

from typing import Any

from flash.providers._instance import InstanceJobHandle
from flash.providers._instance_provider import InstanceProvider
from flash.providers.base import (
    AllocationConstraints,
    Candidate,
    CapacityLookupError,
    JobHandle,
    PollResult,
    Provider,
    UnsupportedGpuError,
    combined_vram_gb,
    rentable_gpu_counts,
)


class LambdaProvider(InstanceProvider):
    """``base.Provider`` for the Lambda Cloud substrate."""

    name = "lambda"
    _gpu_identity_attr = "lambda_name"

    @property
    def _handle_cls(self) -> type[InstanceJobHandle]:
        from flash.providers.lambdalabs.jobs import LambdaJobHandle

        return LambdaJobHandle

    def _load_api_key(self) -> Any:
        from flash.providers.lambdalabs.auth import load_api_key

        return load_api_key()

    def _missing_credentials(self, require_hf: bool) -> list[str]:
        from flash.providers.lambdalabs.preflight import missing_credentials

        return missing_credentials(require_hf=require_hf)

    def _hourly_rate(self, gpu: str) -> float:
        from flash.providers.lambdalabs.pricing import hourly_rate

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
        code_prefix: str | None,
        deadline_at: float | None,
    ) -> PollResult:
        from flash.providers.lambdalabs.jobs import submit_run_lambda

        return submit_run_lambda(
            spec,
            seed,
            log=log,
            on_handle=on_handle,
            attempt=attempt,
            runtime_secrets=runtime_secrets,
            code_prefix=code_prefix,
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
        from flash.providers.lambdalabs.jobs import poll_lambda_job

        return poll_lambda_job(
            handle,
            spec,
            seed,
            log=log,
            heartbeat_reader=heartbeat_reader,
            deadline_at=deadline_at,
        )

    def _teardown_reattached(self, handle: JobHandle, spec) -> None:
        from flash.providers.lambdalabs import api as lambda_api

        lambda_api.terminate_instance_confirmed(handle.instance_id)

    def _gc(self, run_id: str) -> None:
        from flash.providers.lambdalabs.jobs import terminate_run_instances

        terminate_run_instances(run_id)

    def _sweep_orphans(
        self,
        *,
        active_labels,
        known_labels,
    ) -> list[str]:
        from flash.providers.lambdalabs.jobs import sweep_orphans

        return sweep_orphans(active_labels=active_labels, known_labels=known_labels)

    def run_instances_remaining(self, run_id: str) -> list[str]:
        """Exact run-owned instance ids still observable after cleanup."""
        from flash.providers.lambdalabs.jobs import run_instances_remaining

        return run_instances_remaining(run_id)

    def live_candidates(
        self, need_vram_gb: int, constraints: AllocationConstraints
    ) -> list[Candidate]:
        """Lambda classes with live regional capacity fitting the VRAM requirement.

        A capacity-lookup failure raises ``CapacityLookupError``; ``allocate`` degrades to the other
        providers, failing the run retryably only if this was the sole fitting source. Lambda's
        capacity check needs neither disk nor wall, so those constraints are ignored.

        Lambda sells fixed card counts per class as distinct instance types, so each allowed count
        is probed against the live catalog and only the counts with real capacity are reported. The
        rate stays per-card (``usable_instances`` divides the per-instance price).
        """
        from flash.providers.lambdalabs import api as lambda_api
        from flash.providers.lambdalabs.gpus import instance_type_for
        from flash.providers.lambdalabs.jobs import usable_instances

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
                    # too so an exact pin can distinguish "this SKU does not exist" from "sold out".
                    if (
                        not constraints.required_vram_gb
                        or combined_vram_gb(g.vram_gb, count) >= constraints.required_vram_gb
                    ):
                        sku = instance_type_for(g.name, count, catalog)
                        structurally_fitting = structurally_fitting or sku in catalog
                    live = usable_instances(g.name, gpu_count=count)
                    if live:
                        out.append(
                            Candidate("lambda", g.name, live[0].price_usd_hr, g.vram_gb, count)
                        )
            if constraints.gpu_type and constraints.required_vram_gb and not structurally_fitting:
                raise UnsupportedGpuError(
                    f"lambda does not offer the required {constraints.max_gpu_count}-card "
                    f"{constraints.gpu_type} shape large enough for "
                    f"{constraints.required_vram_gb} GB"
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
    from flash.providers.lambdalabs import api as lambda_api

    d = handle.to_dict()
    if d.get("instance_id"):
        lambda_api.terminate_instance_confirmed(str(d["instance_id"]))
