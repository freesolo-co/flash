"""Lambda Cloud provider: single-GPU instances bootstrapped via cloud-init."""

from __future__ import annotations

from typing import Any

from flash.providers._instance_provider import InstanceProvider
from flash.providers.base import (
    AllocationConstraints,
    Candidate,
    CapacityLookupError,
    JobHandle,
    PollResult,
    Provider,
)


class LambdaProvider(InstanceProvider):
    """``base.Provider`` for the Lambda Cloud substrate."""

    name = "lambda"
    _gpu_identity_attr = "lambda_name"

    @property
    def _handle_cls(self) -> type[JobHandle]:
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

    def _poll_job(
        self,
        handle: JobHandle,
        spec,
        seed: int,
        *,
        log: Any,
        heartbeat_reader: Any,
        deadline_s: float,
    ) -> PollResult:
        from flash.providers.lambdalabs.jobs import poll_lambda_job

        return poll_lambda_job(
            handle,
            spec,
            seed,
            log=log,
            heartbeat_reader=heartbeat_reader,
            deadline_s=deadline_s,
        )

    def _reattach_deadline(self, spec) -> float:
        from flash.providers.lambdalabs.jobs import PROVISION_GRACE_S

        # Lambda floors AFTER adding grace (Vast floors before); the formulas intentionally differ.
        return max(60.0, int(spec.gpu.max_wall_seconds) + PROVISION_GRACE_S)

    def _teardown_reattached(self, handle: JobHandle, spec) -> None:
        from flash.providers.lambdalabs import api as lambda_api

        lambda_api.terminate_instances([handle.instance_id])

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

    def live_candidates(
        self, need_vram_gb: int, constraints: AllocationConstraints
    ) -> list[Candidate]:
        """Lambda classes with live regional capacity fitting the VRAM requirement.

        A capacity-lookup failure raises ``CapacityLookupError``; ``allocate`` degrades to the other
        providers, failing the run retryably only if this was the sole fitting source. Lambda's
        capacity check needs neither disk nor wall, so ``constraints`` is ignored.
        """
        from flash.providers.lambdalabs.jobs import usable_instances

        out: list[Candidate] = []
        try:
            for g in self.gpu_classes():
                if g.vram_gb < need_vram_gb:
                    continue
                if usable_instances(g.name):
                    out.append(Candidate("lambda", g.name, self.hourly_rate(g.name), g.vram_gb))
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
        lambda_api.terminate_instances([str(d["instance_id"])])
