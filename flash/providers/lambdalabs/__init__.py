"""Lambda Cloud provider: single-GPU instances bootstrapped via cloud-init (the instance-based
complement to RunPod's serverless Flash endpoints).

Fine-tuning runs on a Lambda Cloud GPU instance launched by Flash. The instance's cloud-init
``user_data`` runs the prebuilt, PUBLIC ``WORKER_IMAGE`` via Docker (the byte-identical training
stack RunPod bakes), which executes ``flash.engine.worker`` on the GPU; completion is detected
purely from the worker's HF artifacts (no inbound network, no serverless queue). It implements the
SAME ``base.Provider`` interface as RunPod, so the orchestrator/allocator treat the two
interchangeably.

``PROVIDER`` is the ``base.Provider`` implementation the registry hands out.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from flash.providers.base import GpuClass, JobHandle, PollResult, Provider


class LambdaProvider:
    """``base.Provider`` for the Lambda Cloud substrate."""

    name = "lambda"

    def is_configured(self) -> bool:
        from flash.providers.lambdalabs.auth import load_api_key

        # Lambda is an opt-in instance substrate: it is available only when its operator key is
        # present. Without LAMBDA_API_KEY (tests / CI / RunPod-only operators) allocation degrades
        # deterministically to RunPod's catalog — exactly the prior RunPod-only behavior.
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
    ) -> PollResult:
        # ``on_last_gpu`` is accepted for the shared Provider interface (RunPod stretches its grace on
        # the last GPU); the instance providers use a UNIFORM per-GPU wait, so it is intentionally unused.
        from flash.providers.lambdalabs.jobs import submit_run_lambda

        return submit_run_lambda(
            spec,
            seed,
            log=log,
            on_handle=on_handle,
            attempt=attempt,
            runtime_secrets=runtime_secrets,
        )

    def poll(self, handle: JobHandle, spec, seed: int, *, log: Any = None) -> PollResult:
        import contextlib

        from flash.providers.lambdalabs import api as lambda_api
        from flash.providers.lambdalabs.jobs import (
            PROVISION_GRACE_S,
            LambdaJobHandle,
            poll_lambda_job,
        )
        from flash.providers.runpod.jobs import make_hf_heartbeat_reader

        hf_repo = spec.train.hf_repo
        prefix = f"{spec.phase}/{spec.run_id}/seed{seed}"
        reader = make_hf_heartbeat_reader(hf_repo, prefix) if hf_repo else None
        lh = LambdaJobHandle.from_dict(handle.to_dict())
        if log is not None:
            print(f"attaching: lambda instance={lh.instance_id}", file=log, flush=True)
        # The wall-cap deadline counts from the instance's LAUNCH, not from this reattach — Lambda
        # has no server-side execution timeout, so resetting it on every recovery would let a
        # control-plane restart extend the billable window unbounded. The poll loop already anchors
        # its deadline check to ``handle.started_ts`` (start = launch), so we pass the FULL
        # launch-relative budget here; pre-subtracting elapsed too would double-count and tear down
        # a still-valid instance the moment a recovered run is past half its window.
        deadline = max(60.0, int(spec.gpu.max_wall_seconds) + PROVISION_GRACE_S)
        try:
            # Uniform per-GPU wait: poll_lambda_job uses its default FIRST_LIVENESS_S / SETUP_GRACE_S
            # (no last-GPU scaling), matching the submit path.
            return poll_lambda_job(
                lh,
                spec,
                seed,
                log=log,
                heartbeat_reader=reader,
                deadline_s=deadline,
            )
        finally:
            # Recovery (attach_run) has no submit_run_lambda teardown ``finally``; terminate the
            # reattached instance here so a finished/abandoned recovered seed stops billing
            # immediately instead of idling until the whole run ends.
            with contextlib.suppress(Exception):
                lambda_api.terminate_instances([lh.instance_id])

    def cancel(self, handle: JobHandle) -> None:
        # Terminating the instance both stops the job and tears down the (only) billable resource —
        # Lambda has no separate "cancel job" vs "destroy resource".
        from flash.providers.lambdalabs import api as lambda_api

        d = handle.to_dict()
        if d.get("instance_id"):
            lambda_api.terminate_instances([str(d["instance_id"])])

    def destroy(self, handle: JobHandle) -> None:
        from flash.providers.lambdalabs import api as lambda_api

        d = handle.to_dict()
        if d.get("instance_id"):
            lambda_api.terminate_instances([str(d["instance_id"])])

    def gc(self, spec) -> None:
        from flash.providers.lambdalabs.jobs import terminate_run_instances

        terminate_run_instances(spec.run_id)

    def sweep_orphans(
        self, active_labels: set[str] | Callable[[], set[str]] | None = None
    ) -> list[str]:
        """Lambda crash-recovery sweep (called via the provider object at startup).

        Lambda instance ids are opaque hex STRINGS (the ``base.Provider`` protocol widens the return
        to ``list[int | str]`` to cover both substrates); the orchestrator only logs/counts them."""
        from flash.providers.lambdalabs.jobs import sweep_orphans

        return sweep_orphans(active_labels=active_labels)


PROVIDER: Provider = LambdaProvider()
