"""Lambda Cloud on-demand training provider, used only when a run pins ``[gpu] lambda = true``."""

from flash.providers.core import base
from flash.providers.core.base import Candidate
from flash.providers.lambda_.config import configured
from flash.providers.lambda_.handles import LambdaJobHandle, require_scope
from flash.providers.lambda_.pricing import LAMBDA_GPUS


class LambdaProvider:
    name = "lambda"
    supports_weight_cache = False
    live_capacity = True

    def is_configured(self):
        return configured()

    def preflight(self, require_hf=True):
        from flash.artifacts.config import configured as artifacts_configured

        problems = []
        if not self.is_configured():
            problems.append("Configure FLASH_LAMBDA_API_KEY and FLASH_LAMBDA_SSH_KEY_NAME")
        if require_hf and not artifacts_configured():
            problems.append("Configure private Azure artifact storage")
        return problems

    def gpu_classes(self):
        return [base.GPU_INFO[name] for name in LAMBDA_GPUS]

    def hourly_rate(self, gpu):
        from flash.providers.lambda_.pricing import hourly_rate

        return hourly_rate(gpu)

    def live_candidates(self, need_vram_gb, constraints):
        """Shapes with live x86_64 regional capacity, at the cheapest type's per-card price.

        A catalog failure raises ``CapacityLookupError``; an empty list with a readable catalog is
        "sold out right now", which the allocator reports as retryable for a pinned Lambda run.
        """
        from flash.providers.lambda_.api import offers

        best = {}
        for offer in offers():
            info = base.GPU_INFO[offer.gpu]
            if (
                info.vram_gb < need_vram_gb
                or (constraints.gpu_type and constraints.gpu_type != offer.gpu)
                or offer.count > constraints.max_gpu_count
                or (
                    constraints.disk_gb
                    and offer.disk_gb is not None
                    and offer.disk_gb < constraints.disk_gb
                )
            ):
                continue
            key = (offer.gpu, offer.count)
            per_card = offer.hourly_usd / offer.count
            if key not in best or per_card < best[key]:
                best[key] = per_card
        return [
            Candidate(self.name, gpu, rate, base.GPU_INFO[gpu].vram_gb, count)
            for (gpu, count), rate in sorted(best.items())
        ]

    def submit_attempt(
        self,
        spec,
        *,
        log=None,
        on_handle=None,
        attempt=0,
        runtime_secrets=None,
        on_last_gpu=False,
        source_snapshot=None,
        _deadline_at=None,
    ):
        from flash.providers.lambda_.jobs import submit_attempt

        return submit_attempt(
            spec,
            log=log,
            on_handle=on_handle,
            attempt=attempt,
            runtime_secrets=runtime_secrets,
            source_snapshot=source_snapshot,
            deadline_at=_deadline_at,
        )

    def poll_attempt(self, handle, spec, *, log=None, _deadline_at=None):
        from flash.providers.lambda_.jobs import poll

        return poll(handle, spec, log=log, deadline_at=_deadline_at)

    def destroy(self, handle):
        from flash.providers.lambda_.api import terminate_confirmed

        h = LambdaJobHandle.from_dict(handle.to_dict())
        require_scope(h)
        terminate_confirmed(h.instance_id, name=h.name)

    def cancel(self, handle):
        self.destroy(handle)

    def gc(self, spec):
        self.terminate_run(spec.run_id)

    def terminate_run(self, run_id):
        from flash.providers.lambda_ import api

        for item in api.run_instances(run_id):
            api.terminate_confirmed(item["id"], name=item["name"])

    def run_instances_remaining(self, run_id):
        from flash.providers.lambda_ import api

        return [item["id"] for item in api.run_instances(run_id)]

    def sweep_orphans(self, active_labels=None, known_labels=None):
        """Terminate Flash VMs of runs this plane knows and no longer runs.

        Instances are listed first and protection sets read afterwards, so a launch racing the
        sweep is protected. A VM is attributed to a run by its label prefix; a VM of a run this
        plane never recorded (another plane on the same account) is left alone.
        """
        from flash.providers._lifecycle.instances.instance import (
            label_matches_run,
            run_label_prefix,
        )
        from flash.providers.lambda_ import api

        instances = api.list_instances()
        active = active_labels() if callable(active_labels) else active_labels
        known = known_labels() if callable(known_labels) else known_labels

        def matches(name, run_ids):
            return any(label_matches_run(name, run_label_prefix(run_id)) for run_id in run_ids)

        deleted = []
        for item in instances:
            if known is not None and not matches(item["name"], known):
                continue
            if active and matches(item["name"], active):
                continue
            api.terminate_confirmed(item["id"], name=item["name"])
            deleted.append(item["id"])
        return deleted


PROVIDER = LambdaProvider()


def terminate_persisted_resources(spec, run_id):
    """Reconcile a launch even when its public job spec can no longer be parsed."""
    PROVIDER.terminate_run(run_id)
