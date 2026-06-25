"""Hyperstack (NexGen Cloud) provider: single-GPU VMs bootstrapped via cloud-init (a second
instance-based complement to RunPod, alongside Lambda).

Fine-tuning runs on a Hyperstack GPU VM launched by Flash. The VM's cloud-init ``user_data`` runs
the prebuilt, PUBLIC ``WORKER_IMAGE`` via Docker (Hyperstack's Docker-preinstalled Ubuntu/CUDA boot
image), which executes ``flash.engine.worker`` on the GPU; completion is detected from the worker's
HF artifacts. It implements the SAME ``base.Provider`` interface as RunPod/Lambda.

``PROVIDER`` is the ``base.Provider`` implementation the registry hands out.
"""

from __future__ import annotations

from typing import Any

from flash.providers.base import GpuClass, JobHandle, PollResult, Provider


class HyperstackProvider:
    """``base.Provider`` for the Hyperstack substrate."""

    name = "hyperstack"

    def is_configured(self) -> bool:
        from flash.providers.hyperstack.auth import load_api_key

        # Opt-in: available only when HYPERSTACK_API_KEY is present (else allocation degrades to the
        # other configured providers).
        return load_api_key() is not None

    def preflight(self, require_hf: bool = True) -> list[str]:
        from flash.providers.hyperstack.preflight import missing_credentials

        return missing_credentials(require_hf=require_hf)

    def gpu_classes(self) -> list[GpuClass]:
        from flash.providers.hyperstack.gpus import gpu_classes

        return gpu_classes()

    def hourly_rate(self, gpu: str) -> float:
        from flash.providers.hyperstack.pricing import hourly_rate

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
        from flash.providers.hyperstack.jobs import submit_run_hyperstack

        return submit_run_hyperstack(
            spec, seed, log=log, on_handle=on_handle, attempt=attempt,
            runtime_secrets=runtime_secrets, on_last_gpu=on_last_gpu,
        )

    def poll(self, handle: JobHandle, spec, seed: int, *, log: Any = None) -> PollResult:
        import contextlib
        import time

        from flash.providers.hyperstack import api as hs_api
        from flash.providers.hyperstack.jobs import (
            PROVISION_GRACE_S,
            HyperstackJobHandle,
            poll_hs_job,
        )
        from flash.providers.runpod.jobs import make_hf_heartbeat_reader

        hf_repo = spec.train.hf_repo
        prefix = f"{spec.phase}/{spec.run_id}/seed{seed}"
        reader = make_hf_heartbeat_reader(hf_repo, prefix) if hf_repo else None
        hh = HyperstackJobHandle.from_dict(handle.to_dict())
        if log is not None:
            print(f"attaching: hyperstack vm={hh.vm_id}", file=log, flush=True)
        # Deadline counts from LAUNCH (started_ts), not this reattach (no server-side timeout, so a
        # restart must not extend the billable window). Subtract elapsed-since-launch.
        elapsed = max(0.0, time.time() - hh.started_ts) if hh.started_ts else 0.0
        deadline = max(60.0, int(spec.gpu.max_wall_seconds) + PROVISION_GRACE_S - elapsed)
        try:
            return poll_hs_job(hh, spec, seed, log=log, heartbeat_reader=reader, deadline_s=deadline)
        finally:
            # Recovery has no submit_run_hyperstack teardown ``finally``; delete the reattached VM
            # here so a finished/abandoned recovered seed stops billing immediately.
            with contextlib.suppress(Exception):
                hs_api.delete_vm(hh.vm_id)

    def cancel(self, handle: JobHandle) -> None:
        from flash.providers.hyperstack import api as hs_api

        d = handle.to_dict()
        if d.get("vm_id"):
            hs_api.delete_vm(str(d["vm_id"]))

    def destroy(self, handle: JobHandle) -> None:
        from flash.providers.hyperstack import api as hs_api

        d = handle.to_dict()
        if d.get("vm_id"):
            hs_api.delete_vm(str(d["vm_id"]))

    def gc(self, spec) -> None:
        from flash.providers.hyperstack.jobs import terminate_run_instances

        terminate_run_instances(spec.run_id)

    def sweep_orphans(self, active_labels: set[str] | None = None) -> list[int]:
        from flash.providers.hyperstack.jobs import sweep_orphans

        return sweep_orphans(active_labels=active_labels)


PROVIDER: Provider = HyperstackProvider()
