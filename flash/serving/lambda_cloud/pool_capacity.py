"""One replica budget shared by Lambda and its scale-to-zero Modal fallback."""

import logging
import math
import time

from flash.serving.app.inference_provider import lambda_models, minimum_for, policy_for
from flash.serving.app.rollout import model_engine_name
from flash.serving.src.engine.model_config import engine_overrides_for

from .config import NoCapacityError
from .reserved import MODEL as RESERVED_MODEL

logger = logging.getLogger(__name__)

RECONCILE_SECONDS = 60
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
            if model == RESERVED_MODEL:
                continue  # The reserved fleet is managed independently, with no Modal GPU fallback.
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
        if model == RESERVED_MODEL:
            raise NoCapacityError("9B Base uses the separately managed reserved fleet")
        revision = revision or (await self.serving_release())["revision"]
        policy = policy_for(model)
        primary, fallback = self.pools(model)
        ready = await self.serving_ready_count(model, revision)
        primary_stats = await primary.health.get_current_stats.aio()
        fallback_stats = await fallback.health.get_current_stats.aio()
        inputs = sum(s.num_running_inputs + s.backlog for s in (primary_stats, fallback_stats))
        # Remember short requests that finish between minute samples, so a
        # zero-floor model still moves to Lambda while requests keep arriving.
        last_request = await self.state.get("demand:" + model)
        recent_demand = last_request is not None and self.clock() - last_request < 120
        desired = min(
            policy.max_capacity,
            max(
                minimum_for(model, self.fleet.environment),
                int(recent_demand),
                math.ceil(inputs / sequence_capacity(model)),
            ),
        )
        # Modal uses unoccupied slots until current-release Lambda workers
        # become healthy, then reduces its ceiling and drains idle GPUs.
        allocated = await self.fleet.allocated_count(model)
        old_allocated = await self.fleet.allocated_count(model, exclude_revision=revision)
        # Draining old releases occupy steady slots. Leave one handoff slot for
        # a new Lambda worker instead of letting fallback consume all headroom.
        # At a completely old/full pool, the surge can serve one fallback until
        # an old worker drains. Never exceed the aggregate allocation budget.
        steady_slots = max(int(ready == 0), policy.max_capacity - old_allocated - ready)
        remaining = max(
            0,
            min(
                steady_slots,
                policy.max_capacity - ready,
                policy.max_capacity + HANDOFF_SURGE - allocated,
            ),
        )
        await fallback.update_autoscaler.aio(
            min_containers=0,
            max_containers=max(1, remaining),
            buffer_containers=0,
            scaledown_window=2 if ready >= desired and ready else 60,
        )
        await self.state.put(model, {"modal_slots": remaining})
        await primary.update_autoscaler.aio(
            min_containers=desired, max_containers=policy.max_capacity, buffer_containers=0
        )
        return ready, old_allocated, fallback_stats, desired

    async def before_launch(self, model, revision, allocated):
        release = await self.serving_release()
        if not release["revision"] and revision != self.revision:
            # The first shared-pool controller may replace a legacy controller
            # whose serving release never called activate(). Its existing CPU
            # pools still enforce their bounds; Fleet also checks the leases.
            return
        if revision != release["revision"] or not await self.active():
            raise NoCapacityError("Lambda pool is not active for this deployment")
        ready, old_allocated, fallback_stats, desired = await self.reconcile_model(model)
        if desired == 0:
            raise NoCapacityError("Model is idle; waiting for new demand before allocating Lambda")
        maximum = policy_for(model).max_capacity
        # Reserve room for requests that can still scale Modal up to its ceiling,
        # as well as old containers that have not finished draining yet.
        routing = await self.state.get(model)
        modal_reserved = max(routing["modal_slots"], fallback_stats.num_total_runners)
        if (
            allocated >= maximum + HANDOFF_SURGE
            or allocated + modal_reserved >= maximum + HANDOFF_SURGE
        ):
            raise NoCapacityError("Waiting for the previous GPU handoff to drain")
        # Draining old workers (healthy or not) consume budget, but only an
        # unready current-release allocation is a second booting replacement.
        if allocated - old_allocated > ready:
            raise NoCapacityError("A Lambda replacement is already starting")

    async def choose(self, model):
        await self.state.put("demand:" + model, self.clock())
        ready = await self.serving_ready_count(model)
        routing = await self.state.get(model)
        if routing is None:
            raise RuntimeError("Serving capacity has not been configured")
        if routing["modal_slots"] == 0 and ready < policy_for(model).max_capacity:
            # A healthy pool may have filled every slot at the last minute tick.
            # Recheck the allocation budget before routing to a now-unhealthy
            # primary; the handoff surge may already be occupied, so do not
            # blindly start a fallback GPU.
            if self.refresh is None:
                await self.reconcile_model(model)
            else:
                await self.refresh(model)
            routing = await self.state.get(model)
        if routing["modal_slots"] == 0:
            return "lambda"
        if not ready:
            return "modal"
        maximum = policy_for(model).max_capacity
        if ready >= maximum:
            return "lambda"  # Queue on the full preferred pool; never boot GPU N+1.
        primary, _ = self.pools(model)
        stats = await primary.health.get_current_stats.aio()
        if stats.backlog or stats.num_running_inputs >= ready * sequence_capacity(model):
            return "modal"
        return "lambda"


def modal_pools(environment):
    import modal

    def lookup(model):
        name = model_engine_name(model)
        return tuple(
            modal.Cls.from_name(
                "freesolo-lora-serving", name + suffix, environment_name=environment
            )(base_model=model)
            for suffix in ("", "_ModalGPU")
        )

    return lookup
