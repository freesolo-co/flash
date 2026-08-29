"""RunPod provider for Flash."""

from __future__ import annotations

import contextlib
from typing import Any

from flash.providers._lifecycle.net.deadline import deadline_kwargs
from flash.providers.core.base import (
    AllocationConstraints,
    Candidate,
    GpuClass,
    JobHandle,
    PollResult,
    Provider,
    rentable_gpu_counts,
)


def terminate_persisted_endpoints(spec: Any, run_id: str) -> None:
    """Best-effort teardown for every GPU class named by a raw persisted spec."""
    from flash.core.spec import persisted_gpu_types
    from flash.providers.runpod.serverless.endpoints import terminate_endpoint

    for gpu_type in persisted_gpu_types(spec):
        with contextlib.suppress(Exception):
            terminate_endpoint(gpu_type, run_id)


class RunpodProvider:
    name = "runpod"
    # Optional capability (read via getattr, kept off the runtime_checkable Protocol like
    # run_instances_remaining): only RunPod offers the shared weight-cache network volume, so the
    # runner's one-shot cache-less retry fallback is gated on it. Instance providers omit it -> False.
    supports_weight_cache = True

    def is_configured(self) -> bool:
        # require a usable parsed key pool, not merely a set env var. otherwise the allocator ranks
        # RunPod classes the operator cannot provision.
        from flash.providers.runpod.client import auth

        return bool(auth.keys())

    def preflight(self, require_hf: bool = True) -> list[str]:
        from flash.providers.runpod.client.preflight import missing_credentials

        return missing_credentials(require_hf=require_hf)

    def gpu_classes(self) -> list[GpuClass]:
        from flash.providers.runpod.client.gpus import gpu_classes

        return gpu_classes()

    def hourly_rate(self, gpu: str) -> float:
        from flash.providers.runpod.client.pricing import hourly_rate

        return hourly_rate(gpu)

    def live_candidates(
        self, need_vram_gb: int, constraints: AllocationConstraints
    ) -> list[Candidate]:
        """RunPod validated classes fitting the VRAM requirement, priced by the static table.

        RunPod takes the card count as a launch parameter and bills per card, so every allowed count
        is offered at the same per-card rate; the allocator picks which one the run actually needs.
        """
        return [
            Candidate("runpod", g.name, self.hourly_rate(g.name), g.vram_gb, count)
            for g in self.gpu_classes()
            if g.vram_gb >= need_vram_gb and g.validated
            for count in rentable_gpu_counts(constraints.max_gpu_count)
        ]

    def submit_attempt(
        self,
        spec,
        *,
        log: Any = None,
        on_handle: Any = None,
        attempt: int = 0,
        runtime_secrets: dict[str, str] | None = None,
        on_last_gpu: bool = False,
        source_snapshot: dict | None = None,
        _deadline_at: float | None = None,
    ) -> PollResult:
        from flash.providers.runpod.execution.job_execution import submit_attempt

        kwargs = {
            "log": log,
            "on_handle": on_handle,
            "attempt": attempt,
            "on_last_gpu": on_last_gpu,
            "source_snapshot": source_snapshot,
            **deadline_kwargs(submit_attempt, _deadline_at),
        }
        if runtime_secrets:
            kwargs["runtime_secrets"] = runtime_secrets
        return submit_attempt(spec, **kwargs)

    def poll_attempt(
        self,
        handle: JobHandle,
        spec,
        *,
        log: Any = None,
        _deadline_at: float | None = None,
    ) -> PollResult:
        from flash.core.spec import gpu_count_of
        from flash.providers.artifacts.hf import (
            make_hf_failure_detail_reader,
            make_hf_heartbeat_reader,
        )
        from flash.providers.runpod.execution import polling as runpod_polling
        from flash.providers.runpod.execution.jobs import JobHandle as RunpodJobHandle
        from flash.providers.runpod.execution.jobs import stall_kwargs

        hf_repo = spec.train.hf_repo
        prefix = f"{spec.phase}/{spec.run_id}"
        hd = handle.to_dict()
        rh = RunpodJobHandle.from_dict(hd)
        if not rh.job_id:
            raise ValueError("endpoint-only RunPod handles cannot be polled")
        reader = (
            make_hf_heartbeat_reader(
                hf_repo,
                prefix,
                **deadline_kwargs(make_hf_heartbeat_reader, _deadline_at),
            )
            if hf_repo
            else None
        )
        failure_reader = (
            make_hf_failure_detail_reader(
                hf_repo,
                prefix,
                spec.phase,
                attempt=rh.attempt,
                **deadline_kwargs(make_hf_failure_detail_reader, _deadline_at),
            )
            if hf_repo
            else None
        )
        if log is not None:
            print(f"attaching: job={rh.job_id} endpoint={rh.endpoint_name}", file=log, flush=True)
        on_last_gpu = bool(hd.get("on_last_gpu", False))
        return runpod_polling.poll_job(
            rh,
            log=log,
            heartbeat_reader=reader,
            failure_detail_reader=failure_reader,
            current_attempt=rh.attempt,
            **deadline_kwargs(runpod_polling.poll_job, _deadline_at),
            # the card count comes from the spec rather than the handle's `allocated_gpu_count`:
            # attach polls the persisted EFFECTIVE worker spec, which submission already stamped
            # with the count allocation resolved, and the attach context pops that handle key off
            # before the handle reaches here. same number, but sourced where it is always present.
            **stall_kwargs(on_last_gpu=on_last_gpu, gpu_count=gpu_count_of(spec)),
        )

    def cancel(self, handle: JobHandle) -> None:
        from flash.providers.runpod.client import api as runpod_api
        from flash.providers.runpod.execution.jobs import JobHandle as RunpodJobHandle

        strict = RunpodJobHandle.from_dict(handle.to_dict())
        if not strict.job_id:
            raise runpod_api.RunpodApiError("runpod cancellation could not be confirmed")
        response = runpod_api.cancel_job(
            strict.endpoint_id,
            strict.job_id,
            key_fingerprint=strict.key_fingerprint,
        )
        if (
            not isinstance(response, dict)
            or response.get("id") != strict.job_id
            or response.get("status") != "CANCELLED"
        ):
            raise runpod_api.RunpodApiError("runpod cancellation could not be confirmed")

    def destroy(self, handle: JobHandle) -> None:
        from flash.providers.runpod.client import api as runpod_api
        from flash.providers.runpod.execution.jobs import JobHandle as RunpodJobHandle

        strict = RunpodJobHandle.from_dict(handle.to_dict())
        if runpod_api.delete_endpoint_for_fingerprint(strict.endpoint_id, strict.key_fingerprint):
            return

        if (
            runpod_api.endpoint_absent_for_fingerprint(strict.endpoint_id, strict.key_fingerprint)
            is not True
        ):
            raise runpod_api.RunpodApiError(
                f"runpod delete_endpoint({strict.endpoint_id}) unconfirmed; endpoint may still bill"
            )

    def gc(self, spec) -> None:
        from flash.providers.runpod.serverless.endpoints import terminate_endpoint

        # every acceptable class, not just the head: allocation ranks a multi-class pin on cost and
        # may rent a fallback, whose endpoint name is derived from that class. matching is by name,
        # so naming a class this run never rented is a no-op.
        for gpu_type in spec.gpu.acceptable_types:
            terminate_endpoint(gpu_type, spec.run_id)

    def sweep_orphans(
        self,
        active_labels: set[str] | None = None,
        known_labels: set[str] | None = None,
    ) -> list[int]:
        return []


PROVIDER: Provider = RunpodProvider()
