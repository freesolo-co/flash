"""RunPod provider for Flash."""

from __future__ import annotations

from typing import Any

from flash.providers._deadline import deadline_kwargs
from flash.providers.base import (
    AllocationConstraints,
    Candidate,
    GpuClass,
    JobHandle,
    PollResult,
    Provider,
)


class RunpodProvider:
    name = "runpod"
    # Optional capability (read via getattr, kept off the runtime_checkable Protocol like
    # run_instances_remaining): only RunPod offers the shared weight-cache network volume, so the
    # runner's one-shot cache-less retry fallback is gated on it. Instance providers omit it -> False.
    supports_weight_cache = True

    def is_configured(self) -> bool:
        # Missing key surfaces at preflight, not here.
        return True

    def preflight(self, require_hf: bool = True) -> list[str]:
        from flash.providers.runpod.preflight import missing_credentials

        return missing_credentials(require_hf=require_hf)

    def gpu_classes(self) -> list[GpuClass]:
        from flash.providers.runpod.gpus import gpu_classes

        return gpu_classes()

    def hourly_rate(self, gpu: str) -> float:
        from flash.providers.runpod.pricing import hourly_rate

        return hourly_rate(gpu)

    def live_candidates(
        self, need_vram_gb: int, constraints: AllocationConstraints
    ) -> list[Candidate]:
        """RunPod validated classes fitting the VRAM requirement, priced by the static table."""
        return [
            Candidate("runpod", g.name, self.hourly_rate(g.name), g.vram_gb)
            for g in self.gpu_classes()
            if g.vram_gb >= need_vram_gb and g.validated
        ]

    def submit_run(
        self,
        spec,
        seed: int,
        *,
        log: Any = None,
        on_handle: Any = None,
        attempt: int = 0,
        runtime_secrets: dict[str, str] | None = None,
        on_last_gpu: bool = False,
        code_prefix: str | None = None,
        _deadline_at: float | None = None,
    ) -> PollResult:
        from flash.providers.runpod.jobs import submit_run
        from flash.spec import require_matching_seed

        seed = require_matching_seed(spec, seed)
        kwargs = {
            "log": log,
            "on_handle": on_handle,
            "attempt": attempt,
            "on_last_gpu": on_last_gpu,
            "code_prefix": code_prefix,
            **deadline_kwargs(submit_run, _deadline_at),
        }
        if runtime_secrets:
            kwargs["runtime_secrets"] = runtime_secrets
        return submit_run(spec, seed, **kwargs)

    def poll(
        self,
        handle: JobHandle,
        spec,
        seed: int,
        *,
        log: Any = None,
        _deadline_at: float | None = None,
    ) -> PollResult:
        from flash.providers.runpod.jobs import JobHandle as RunpodJobHandle
        from flash.providers.runpod.jobs import (
            make_hf_failure_detail_reader,
            make_hf_heartbeat_reader,
            poll_job,
            stall_kwargs,
        )
        from flash.spec import require_matching_seed

        seed = require_matching_seed(spec, seed)
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
        return poll_job(
            rh,
            log=log,
            heartbeat_reader=reader,
            failure_detail_reader=failure_reader,
            current_attempt=rh.attempt,
            **deadline_kwargs(poll_job, _deadline_at),
            **stall_kwargs(on_last_gpu=on_last_gpu),
        )

    def cancel(self, handle: JobHandle) -> None:
        from flash.providers.runpod import api as runpod_api
        from flash.providers.runpod.jobs import JobHandle as RunpodJobHandle

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
        from flash.providers.runpod import api as runpod_api
        from flash.providers.runpod.jobs import JobHandle as RunpodJobHandle

        strict = RunpodJobHandle.from_dict(handle.to_dict())
        if not runpod_api.delete_endpoint_for_fingerprint(
            strict.endpoint_id, strict.key_fingerprint
        ):
            raise runpod_api.RunpodApiError(
                f"runpod delete_endpoint({strict.endpoint_id}) unconfirmed; endpoint may still bill"
            )

    def gc(self, spec) -> None:
        from flash.providers.runpod.train import terminate_endpoint

        terminate_endpoint(spec.gpu.type, spec.run_id)

    def sweep_orphans(
        self,
        active_labels: set[str] | None = None,
        known_labels: set[str] | None = None,
    ) -> list[int]:
        return []


PROVIDER: Provider = RunpodProvider()
