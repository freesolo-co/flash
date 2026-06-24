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
        # allocation. Pricing is static, and a missing RUNPOD_API_KEY surfaces at provision time
        # via ensure_auth / the preflight, never as a silent empty candidate list. This matches the
        # historical ``available_providers()`` which listed runpod unconditionally.
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
    ) -> PollResult:
        from flash.providers.runpod.jobs import submit_run

        kwargs = {"log": log, "on_handle": on_handle, "attempt": attempt}
        if runtime_secrets:
            kwargs["runtime_secrets"] = runtime_secrets
        return submit_run(spec, seed, **kwargs)

    def poll(self, handle: JobHandle, spec, seed: int, *, log: Any = None) -> PollResult:
        from flash.providers.runpod.jobs import JobHandle as RunpodJobHandle
        from flash.providers.runpod.jobs import (
            make_hf_failure_detail_reader,
            make_hf_heartbeat_reader,
            poll_job,
            stall_kwargs,
        )

        hf_repo = spec.train.hf_repo
        prefix = f"{spec.phase}/{spec.run_id}/seed{seed}"
        reader = make_hf_heartbeat_reader(hf_repo, prefix) if hf_repo else None
        failure_reader = (
            make_hf_failure_detail_reader(hf_repo, prefix, spec.phase) if hf_repo else None
        )
        rh = RunpodJobHandle.from_dict(handle.to_dict())
        if log is not None:
            print(f"attaching: job={rh.job_id} endpoint={rh.endpoint_name}", file=log, flush=True)
        # Same stall tuning as the submit path so a reattached run isn't judged differently.
        return poll_job(
            rh,
            log=log,
            heartbeat_reader=reader,
            failure_detail_reader=failure_reader,
            **stall_kwargs(),
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

    def sweep_orphans(self, active_labels: set[str] | None = None) -> list:
        """Reclaim idle flash-* endpoints that no active run owns; return deleted ids.

        RunPod serverless workers scale to zero, but the *endpoints* persist after a run
        ends and a crash (or a control-plane restart that bypasses per-run GC) leaves the
        worker warm with nothing to do — so they accumulate and keep eating the account's
        worker quota. Run at server startup (crash recovery) and after runs, alongside
        Vast's instance sweep. ``_endpoint_is_reclaimable`` only deletes endpoints that
        have finished a job and are doing nothing now; ``active_labels`` (raw run ids) are
        additionally shielded by their reconstructed endpoint-name suffix. Best-effort:
        never raises. No RunPod key configured => nothing to sweep.
        """
        import os

        if not os.environ.get("RUNPOD_API_KEY"):
            return []
        from flash.providers.runpod.jobs import _sweep_idle_flash_endpoints
        from flash.providers.runpod.train import _run_suffix

        # An active run's endpoint is named flash-<gpu>-<suffix>[rN]; protect by suffix token.
        protected = frozenset(
            s for s in (_run_suffix(rid) for rid in (active_labels or set())) if s
        )
        try:
            return _sweep_idle_flash_endpoints(protected_names=protected)
        except Exception:
            return []


PROVIDER: Provider = RunpodProvider()
