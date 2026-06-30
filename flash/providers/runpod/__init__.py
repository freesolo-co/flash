"""RunPod provider for Flash."""

from __future__ import annotations

from typing import Any

from flash.providers.base import GpuClass, JobHandle, PollResult, Provider


class RunpodProvider:

    name = "runpod"

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
    ) -> PollResult:
        from flash.providers.runpod.jobs import submit_run

        kwargs = {
            "log": log,
            "on_handle": on_handle,
            "attempt": attempt,
            "on_last_gpu": on_last_gpu,
            "code_prefix": code_prefix,
        }
        if runtime_secrets:
            kwargs["runtime_secrets"] = runtime_secrets
        return submit_run(spec, seed, **kwargs)

    def poll(self, handle: JobHandle, spec, seed: int, *, log: Any = None) -> PollResult:
        from flash.providers._poll import _attempt_int
        from flash.providers.runpod.jobs import JobHandle as RunpodJobHandle
        from flash.providers.runpod.jobs import (
            make_hf_failure_detail_reader,
            make_hf_heartbeat_reader,
            poll_job,
            stall_kwargs,
        )

        hf_repo = spec.train.hf_repo
        prefix = f"{spec.phase}/{spec.run_id}"
        hd = handle.to_dict()
        rh = RunpodJobHandle.from_dict(hd)
        reader = make_hf_heartbeat_reader(hf_repo, prefix) if hf_repo else None
        failure_reader = (
            make_hf_failure_detail_reader(hf_repo, prefix, spec.phase, attempt=rh.attempt)
            if hf_repo
            else None
        )
        if log is not None:
            print(f"attaching: job={rh.job_id} endpoint={rh.endpoint_name}", file=log, flush=True)
        on_last_gpu = bool(hd.get("on_last_gpu", False))
        current_attempt = _attempt_int(hd.get("attempt"))
        return poll_job(
            rh,
            log=log,
            heartbeat_reader=reader,
            failure_detail_reader=failure_reader,
            current_attempt=current_attempt,
            **stall_kwargs(on_last_gpu=on_last_gpu),
        )

    def cancel(self, handle: JobHandle) -> None:
        from flash.providers.runpod import api as runpod_api

        d = handle.to_dict()
        if d.get("endpoint_id") and d.get("job_id"):
            runpod_api.cancel_job(d["endpoint_id"], d["job_id"])

    def destroy(self, handle: JobHandle) -> None:
        from flash.providers.runpod import api as runpod_api

        d = handle.to_dict()
        if d.get("endpoint_id"):
            runpod_api.delete_endpoint(d["endpoint_id"])

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
