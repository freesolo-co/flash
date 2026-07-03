"""Lambda Cloud provider: single-GPU instances bootstrapped via cloud-init."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from flash.providers.base import GpuClass, JobHandle, PollResult, Provider


class LambdaProvider:
    """``base.Provider`` for the Lambda Cloud substrate."""

    name = "lambda"

    def is_configured(self) -> bool:
        from flash.providers.lambdalabs.auth import load_api_key

        return load_api_key() is not None

    def preflight(self, require_hf: bool = True) -> list[str]:
        from flash.providers.lambdalabs.preflight import missing_credentials

        return missing_credentials(require_hf=require_hf)

    def gpu_classes(self) -> list[GpuClass]:
        from flash.providers.lambdalabs.gpus import gpu_classes

        return gpu_classes()

    def hourly_rate(self, gpu: str) -> float:
        from flash.providers.lambdalabs.pricing import hourly_rate

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
        from flash.providers.lambdalabs.jobs import submit_run_lambda

        return submit_run_lambda(
            spec,
            seed,
            log=log,
            on_handle=on_handle,
            attempt=attempt,
            runtime_secrets=runtime_secrets,
            code_prefix=code_prefix,
        )

    def poll(self, handle: JobHandle, spec, seed: int, *, log: Any = None) -> PollResult:
        import contextlib

        from flash.providers._hf_artifacts import make_hf_heartbeat_reader
        from flash.providers.lambdalabs import api as lambda_api
        from flash.providers.lambdalabs.jobs import (
            PROVISION_GRACE_S,
            LambdaJobHandle,
            poll_lambda_job,
        )

        hf_repo = spec.train.hf_repo
        prefix = f"{spec.phase}/{spec.run_id}"
        reader = make_hf_heartbeat_reader(hf_repo, prefix) if hf_repo else None
        lh = LambdaJobHandle.from_dict(handle.to_dict())
        if log is not None:
            print(f"attaching: lambda instance={lh.instance_id}", file=log, flush=True)
        # Deadline is launch-relative, not reattach-relative: resetting on recovery would extend billable window unbounded.
        deadline = max(60.0, int(spec.gpu.max_wall_seconds) + PROVISION_GRACE_S)
        try:
            return poll_lambda_job(
                lh,
                spec,
                seed,
                log=log,
                heartbeat_reader=reader,
                deadline_s=deadline,
            )
        finally:
            # attach_run has no submit_run_lambda teardown; terminate here so recovered seeds stop billing.
            with contextlib.suppress(Exception):
                lambda_api.terminate_instances([lh.instance_id])

    def cancel(self, handle: JobHandle) -> None:
        _terminate_handle_instance(handle)

    destroy = cancel

    def gc(self, spec) -> None:
        from flash.providers.lambdalabs.jobs import terminate_run_instances

        terminate_run_instances(spec.run_id)

    def sweep_orphans(
        self,
        active_labels: set[str] | Callable[[], set[str]] | None = None,
        known_labels: set[str] | Callable[[], set[str]] | None = None,
    ) -> list[str]:
        """Lambda crash-recovery sweep."""
        from flash.providers.lambdalabs.jobs import sweep_orphans

        return sweep_orphans(active_labels=active_labels, known_labels=known_labels)


PROVIDER: Provider = LambdaProvider()


def _terminate_handle_instance(handle: JobHandle) -> None:
    from flash.providers.lambdalabs import api as lambda_api

    d = handle.to_dict()
    if d.get("instance_id"):
        lambda_api.terminate_instances([str(d["instance_id"])])
