"""RunPod Secure Cloud Pod provider for managed training."""

from __future__ import annotations

import time
from typing import Any

from flash.providers._lifecycle.instances.instance import InstanceJobHandle
from flash.providers._lifecycle.instances.provider import InstanceProvider
from flash.providers.core.base import (
    AllocationConstraints,
    Candidate,
    JobHandle,
    PollResult,
    Provider,
    rentable_gpu_counts,
)


class RunpodProvider(InstanceProvider):
    name = "runpod"
    _gpu_identity_attr = "runpod_gpu_type_id"
    # runpod accepts a card count directly; its static class table defines rentable shapes.
    live_capacity = False
    supports_weight_cache = True

    @property
    def _handle_cls(self) -> type[InstanceJobHandle]:
        from flash.providers.runpod.execution.identity import RunpodPodHandle

        return RunpodPodHandle

    def is_configured(self) -> bool:
        from flash.providers.runpod.client.auth import keys

        return bool(keys())

    def _load_api_key(self) -> Any:
        from flash.providers.runpod.client.auth import load_api_key

        return load_api_key()

    def _missing_credentials(self, require_hf: bool) -> list[str]:
        from flash.providers.runpod.client.preflight import missing_credentials

        return missing_credentials(require_hf=require_hf)

    def _hourly_rate(self, gpu: str) -> float:
        from flash.providers.runpod.client.pricing import hourly_rate

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
        source_snapshot: dict | None,
        deadline_at: float | None,
    ) -> PollResult:
        from flash.providers.runpod.execution.pods import submit_runpod_pod

        return submit_runpod_pod(
            spec,
            seed,
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
        seed: int,
        *,
        log: Any,
        heartbeat_reader: Any,
        deadline_at: float | None,
        on_handle: Any = None,
    ) -> PollResult:
        from flash.providers.runpod.execution.pods import poll_runpod_pod

        return poll_runpod_pod(
            handle,
            spec,
            seed,
            log=log,
            heartbeat_reader=heartbeat_reader,
            on_handle=on_handle,
            deadline_at=deadline_at,
        )

    def _teardown_reattached(self, handle: JobHandle, spec) -> None:
        from flash.providers.runpod.execution.pods import terminate_handle

        terminate_handle(handle, deadline_at=time.time() + 120.0)

    def poll(
        self,
        handle: JobHandle,
        spec,
        seed: int,
        *,
        log: Any = None,
        _deadline_at: float | None = None,
    ) -> PollResult:
        """Poll a recovered Pod and require complete Pod plus secret teardown."""
        from flash._internal.diagnostics import sanitize_diagnostic
        from flash.core.spec import require_matching_seed
        from flash.providers.artifacts.hf import heartbeat_reader_for
        from flash.runner.accounting.reconciliation import (
            _compare_and_enrich_remote,
            _compare_and_remove_cleanup_remote,
            _record_cleanup_remote,
        )

        seed = require_matching_seed(spec, seed)
        reader = heartbeat_reader_for(spec, deadline_at=_deadline_at)
        strict = self._handle_cls.from_dict(handle.to_dict())
        if strict.pending:
            raise ValueError(
                "pending RunPod Pod handles must be resolved and persisted before polling"
            )
        if log is not None:
            print(f"attaching: runpod instance={strict.instance_id}", file=log, flush=True)

        fallback_cleanup_remote: dict | None = None

        def add_recovery_note(original: BaseException, message: str, error: BaseException) -> None:
            try:
                detail = sanitize_diagnostic(error, limit=256)
            except BaseException:
                detail = type(error).__name__
            original.add_note(f"{message}: {detail}")

        def persist_enriched_handle(remote: dict) -> None:
            nonlocal fallback_cleanup_remote, strict
            enriched = self._handle_cls.from_dict(remote)
            expected = strict
            strict = enriched
            if not _compare_and_enrich_remote(
                spec.run_id,
                expected.to_dict(),
                enriched.to_dict(),
            ):
                if not _record_cleanup_remote(spec.run_id, enriched.to_dict()):
                    raise RuntimeError("enriched runpod cleanup target could not be persisted")
                fallback_cleanup_remote = enriched.to_dict()
                raise RuntimeError("enriched runpod handle ownership changed before persistence")

        try:
            result = self._poll_job(
                strict,
                spec,
                seed,
                log=log,
                heartbeat_reader=reader,
                deadline_at=_deadline_at,
                on_handle=persist_enriched_handle,
            )
        except BaseException as original:
            try:
                self._teardown_reattached(strict, spec)
            except BaseException as teardown_error:
                add_recovery_note(
                    original,
                    "runpod recovered teardown also failed",
                    teardown_error,
                )
                try:
                    cleanup_recorded = _record_cleanup_remote(spec.run_id, strict.to_dict())
                except BaseException as persistence_error:
                    add_recovery_note(
                        original,
                        "runpod recovered cleanup persistence also failed",
                        persistence_error,
                    )
                else:
                    if not cleanup_recorded:
                        original.add_note("runpod recovered cleanup persistence also failed")
            else:
                if fallback_cleanup_remote is not None:
                    try:
                        cleanup_removed = _compare_and_remove_cleanup_remote(
                            spec.run_id,
                            fallback_cleanup_remote,
                        )
                    except BaseException as persistence_error:
                        add_recovery_note(
                            original,
                            "runpod recovered cleanup record removal also failed",
                            persistence_error,
                        )
                    else:
                        if not cleanup_removed:
                            original.add_note("runpod recovered cleanup record removal also failed")
            raise
        try:
            self._teardown_reattached(strict, spec)
        except Exception:
            if not _record_cleanup_remote(spec.run_id, strict.to_dict()):
                raise RuntimeError("runpod cleanup target could not be persisted") from None
        return result

    def _gc(self, run_id: str) -> None:
        from flash.providers.runpod.execution.pods import destroy_run_pods

        destroy_run_pods(run_id)

    def _sweep_orphans(self, *, active_labels, known_labels) -> list[str]:
        from flash.providers.runpod.execution.pods import sweep_orphan_pods

        return sweep_orphan_pods(active_labels=active_labels, known_labels=known_labels)

    def live_candidates(
        self, need_vram_gb: int, constraints: AllocationConstraints
    ) -> list[Candidate]:
        return [
            Candidate("runpod", gpu.name, self.hourly_rate(gpu.name), gpu.vram_gb, count)
            for gpu in self.gpu_classes()
            if gpu.vram_gb >= need_vram_gb and gpu.validated
            for count in rentable_gpu_counts(constraints.max_gpu_count)
        ]

    def cancel(self, handle: JobHandle) -> None:
        from flash.providers.runpod.execution.pods import terminate_handle

        strict = self._handle_cls.from_dict(handle.to_dict())
        terminate_handle(strict, deadline_at=time.time() + 120.0)

    destroy = cancel

    def run_instances_remaining(self, run_id: str) -> list[str]:
        from flash.providers.runpod.execution.pods import run_pods_remaining

        return run_pods_remaining(run_id)


PROVIDER: Provider = RunpodProvider()
