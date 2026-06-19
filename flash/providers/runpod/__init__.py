"""RunPod Flash provider: managed, serverless GPUs (no Docker) for Flash.

Fine-tuning runs on a dedicated RunPod GPU provisioned by Flash. A decorated Python
handler (``train._train_body``) executes ``flash.engine.worker`` on the GPU; Flash
handles provisioning, dependency install, execution, and scale-to-zero teardown.
Serving exposes an OpenAI-compatible endpoint for a trained LoRA adapter.

``PROVIDER`` is the ``base.Provider`` implementation the registry hands out; the
orchestrator/allocator only talk to its interface, never these modules directly.
"""

from __future__ import annotations

from typing import Any

from flash.providers.base import GpuClass, JobHandle, PollResult, Provider


class RunpodProvider:
    """``base.Provider`` for the RunPod Flash substrate."""

    name = "runpod"

    def is_configured(self) -> bool:
        # RunPod is the ALWAYS-ON default substrate, so it is always "available" for
        # allocation (offline pricing degrades to the static snapshot, and a missing
        # RUNPOD_API_KEY surfaces at provision time via ensure_auth / the preflight —
        # never as a silent empty candidate list). This matches the historical
        # ``available_providers()`` which listed runpod unconditionally.
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
        offers: Any = None,
        exclude_machine_ids: Any = frozenset(),
    ) -> PollResult:
        # ``offers``/``exclude_machine_ids`` are Vast live-market concerns; RunPod
        # provisions a fresh serverless endpoint and never re-searches a market, so it
        # ignores both (kept in the signature for cross-provider symmetry).
        from flash.providers.runpod.jobs import submit_run

        return submit_run(spec, seed, log=log, on_handle=on_handle, attempt=attempt)

    def poll(self, handle: JobHandle, spec, seed: int, *, log: Any = None) -> PollResult:
        from flash.providers.runpod.jobs import JobHandle as RunpodJobHandle
        from flash.providers.runpod.jobs import (
            make_hf_heartbeat_reader,
            poll_job,
            stall_kwargs,
        )

        hf_repo = spec.train.hf_repo
        prefix = f"{spec.phase}/{spec.run_id}/seed{seed}"
        reader = make_hf_heartbeat_reader(hf_repo, prefix) if hf_repo else None
        rh = RunpodJobHandle.from_dict(handle.to_dict())
        if log is not None:
            print(f"attaching: job={rh.job_id} endpoint={rh.endpoint_name}", file=log, flush=True)
        # Same stall tuning as the submit path so a reattached run isn't judged differently.
        return poll_job(rh, log=log, heartbeat_reader=reader, **stall_kwargs())

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

    def sweep_orphans(self, active_labels: set[str] | None = None) -> list[int]:
        # No-op: RunPod serverless endpoints have no standing per-run billing to reap on
        # crash recovery (a failed-before-submit endpoint is GC'd by reconstructed name in
        # recover_runs). Present for ``base.Provider`` symmetry with Vast's instance sweep.
        return []


PROVIDER: Provider = RunpodProvider()
