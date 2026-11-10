"""One replica budget shared by Lambda and its scale-to-zero Modal fallback."""

import logging
import math
import time

from flash.serving.app.inference_provider import lambda_models, policy_for
from flash.serving.app.rollout import model_engine_name
from flash.serving.src.engine.model_config import engine_overrides_for

from .config import NoCapacityError

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

    async def activate(self, revision):
        if not revision or revision != self.revision:
            raise RuntimeError("Cannot activate a superseded Lambda capacity controller")
        await self.state.put("active_revision", revision)
        await self.reconcile(strict=True)

    async def active(self):
        return bool(self.revision) and await self.state.get("active_revision") == self.revision

    async def reconcile(self, *, strict=False):
        if not await self.active():
            return
        errors = []
        for model in lambda_models():
            try:
                await self.reconcile_model(model)
            except Exception as exc:
                errors.append(exc)
                logger.warning(
                    "Capacity reconciliation failed for %s; retrying in one minute", model
                )
        if strict and errors:
            raise ExceptionGroup("Cannot configure shared serving capacity", errors)

    async def reconcile_model(self, model):
        policy = policy_for(model)
        primary, fallback = self.pools(model)
        ready = await self.fleet.ready_count(model)
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
                policy.min_capacity,
                int(recent_demand),
                math.ceil(inputs / sequence_capacity(model)),
            ),
        )
        # With no ready Lambda worker, Modal can use the entire budget. Once
        # replacements are healthy, reduce its ceiling and drain idle GPUs.
        allocated = await self.fleet.allocated_count(model)
        remaining = max(
            0, min(policy.max_capacity - ready, policy.max_capacity + HANDOFF_SURGE - allocated)
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
        return ready, primary_stats, fallback_stats, desired

    async def before_launch(self, model, revision, allocated):
        if revision != self.revision or not await self.active():
            raise NoCapacityError("Lambda pool is not active for this deployment")
        ready, _, fallback_stats, desired = await self.reconcile_model(model)
        if desired == 0:
            raise NoCapacityError("Model is idle; waiting for new demand before allocating Lambda")
        maximum = policy_for(model).max_capacity
        # Reserve room for requests that can still scale Modal up to its ceiling,
        # as well as old containers that have not finished draining yet.
        modal_reserved = max(maximum - ready, fallback_stats.num_total_runners)
        if (
            allocated >= maximum + HANDOFF_SURGE
            or allocated + modal_reserved >= maximum + HANDOFF_SURGE
        ):
            raise NoCapacityError("Waiting for the previous GPU handoff to drain")
        if allocated > ready:
            raise NoCapacityError("A Lambda replacement is already starting")

    async def choose(self, model):
        await self.state.put("demand:" + model, self.clock())
        ready = await self.fleet.ready_count(model)
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
