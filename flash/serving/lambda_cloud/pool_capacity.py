"""Replica budget for the Lambda-only hosted GPU pool (no Modal GPU fallback)."""

import logging
import math
import time

from flash.serving.app.inference_provider import lambda_models, minimum_for, policy_for
from flash.serving.app.rollout import model_engine_name
from flash.serving.src.engine.model_config import engine_overrides_for

from .config import NoCapacityError
from .reserved import enabled as reserved_enabled

logger = logging.getLogger(__name__)

# A replacement boots before the old GPU drains. Never start a second handoff
# until the first has completed and the shared pool is back within its budget.
HANDOFF_SURGE = 1


def sequence_capacity(model):
    return int(engine_overrides_for(model).get("max_num_seqs", 64))


class SharedCapacity:
    def __init__(self, fleet, pools, state, revision, *, clock=time.time, refresh=None):
        self.fleet, self.pools, self.state, self.revision = fleet, pools, state, revision
        self.clock = clock
        self.refresh = refresh

    async def serving_release(self):
        release = await self.state.get("serving_release")
        if release is not None:
            return release
        # Upgrade controllers without requiring an already-serving router to
        # redeploy. Older controllers stored only the active SHA.
        return {"revision": await self.state.get("active_revision"), "retired": []}

    async def activate(self, revision):
        release = await self.serving_release()
        if revision and revision == release["revision"]:
            return  # A new controller must still admit the existing router.
        if not revision or revision != self.revision or revision in release["retired"]:
            raise RuntimeError("Cannot activate a superseded Lambda capacity controller")
        # Do not retire the serving release if configuring its replacement fails.
        await self.reconcile(strict=True, activating=True)
        retired = release["retired"] + ([release["revision"]] if release["revision"] else [])
        # Publish activation and retirement together: an old router cannot
        # reactivate itself after the replacement has taken over.
        await self.state.put("serving_release", {"revision": revision, "retired": retired})

    async def active(self):
        release = await self.serving_release()
        return (
            bool(self.revision and release["revision"]) and self.revision not in release["retired"]
        )

    async def reconcile(self, *, strict=False, activating=False):
        if not activating and not await self.active():
            return
        errors = []
        for model in lambda_models():
            if reserved_enabled(model, self.fleet.environment):
                continue  # Served by the reserved fleet: no on-demand lease pool.
            try:
                await self.reconcile_model(model, revision=self.revision if activating else None)
            except Exception as exc:
                errors.append(exc)
                logger.warning(
                    "Capacity reconciliation failed for %s; retrying in one minute", model
                )
        if strict and errors:
            raise ExceptionGroup("Cannot configure shared serving capacity", errors)

    async def serving_ready_count(self, model, revision=None):
        revision = revision or (await self.serving_release())["revision"]
        if not revision:
            return 0
        return await self.fleet.ready_count(model, revision=revision)

    async def reconcile_model(self, model, *, revision=None):
        if reserved_enabled(model, self.fleet.environment):
            raise NoCapacityError(f"{model} uses the separately managed reserved fleet")
        revision = revision or (await self.serving_release())["revision"]
        policy = policy_for(model)
        pool = self.pools(model)
        ready = await self.serving_ready_count(model, revision)
        stats = await pool.health.get_current_stats.aio()
        inputs = stats.num_running_inputs + stats.backlog
        desired = min(
            policy.max_capacity,
            max(
                minimum_for(model, self.fleet.environment),
                int(stats.num_total_runners > 0 or inputs > 0),
                math.ceil(inputs / sequence_capacity(model)),
            ),
        )
        await pool.update_autoscaler.aio(
            min_containers=minimum_for(model, self.fleet.environment),
            max_containers=policy.max_capacity,
            buffer_containers=0,
        )
        return ready, desired

    async def before_launch(self, model, revision, allocated):
        release = await self.serving_release()
        if not release["revision"] and revision != self.revision:
            # The first shared-pool controller may replace a legacy controller
            # whose serving release never called activate(). Its existing CPU
            # pools still enforce their bounds; Fleet also checks the leases.
            return
        if revision != release["revision"] or not await self.active():
            raise NoCapacityError("Lambda pool is not active for this deployment")
        ready, desired = await self.reconcile_model(model)
        if desired == 0:
            raise NoCapacityError("Model is idle; waiting for new demand before allocating Lambda")
        # Draining old-release workers still count, so a rolling deploy may hold
        # at most one extra GPU above the model's ceiling.
        if allocated >= policy_for(model).max_capacity + HANDOFF_SURGE:
            raise NoCapacityError("Waiting for the previous GPU handoff to drain")
        # Scale up one GPU at a time: draining old workers (healthy or not) consume
        # budget, but only an unready current-release allocation is a booting one.
        old_allocated = await self.fleet.allocated_count(model, exclude_revision=revision)
        if allocated - old_allocated > ready:
            raise NoCapacityError("A Lambda replacement is already starting")


def modal_pools(environment):
    """The Modal CPU proxy pool that owns each model's Lambda GPU workers."""
    import modal

    def lookup(model):
        return modal.Cls.from_name(
            "freesolo-lora-serving", model_engine_name(model), environment_name=environment
        )(base_model=model)

    return lookup
