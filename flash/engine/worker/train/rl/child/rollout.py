"""GRPO rollout workers and manager with explicit sampling and identity policy."""

import asyncio
import os

import ray
from verl import DataProto
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    auto_await,
    get_trajectory_info,
    register,
)

from flash.engine.worker.train.core.child.options import RuntimeOptions
from flash.engine.worker.train.core.child.startup import initialize_process
from flash.engine.worker.train.rl.child.identity import (
    _identity_from_trajectory,
    _trajectory_info_from_identities,
)
from flash.engine.worker.train.rl.child.multiturn import (
    build_flash_grpo_multi_turn_agent_loop,
    post_json,
)

FlashGrpoMultiTurnAgentLoop = build_flash_grpo_multi_turn_agent_loop(
    register=register,
    agent_loop_base=AgentLoopBase,
    agent_loop_output=AgentLoopOutput,
)


class FlashGrpoWorker(AgentLoopWorker):
    def __init__(self, config, *args, **kwargs):
        initialize_process()
        super().__init__(config, *args, **kwargs)
        self.runtime = RuntimeOptions.decode(dict(config.actor_rollout_ref.model.runtime_options))

    async def generate_sequences_with_flash_identities(self, batch, identities):
        index = batch.non_tensor_batch.get("index")
        indexes = list(range(len(batch))) if index is None else index.tolist()
        trajectories = _trajectory_info_from_identities(
            batch.meta_info.get("global_steps", -1),
            indexes,
            batch.meta_info.get("validate", False),
            identities,
        )
        return await super().generate_sequences(batch, trajectory_info=trajectories)

    async def _run_agent_loop(
        self, sampling_params, trajectory, *, agent_name, trace=True, **kwargs
    ):
        identity = _identity_from_trajectory(trajectory)
        extra = kwargs.get("extra_info")
        if not isinstance(extra, dict):
            raise RuntimeError("flash rollout trajectory is missing dictionary extra_info")
        if "flash_rollout_identity" in extra or "flash_rollout_identity" in kwargs:
            raise RuntimeError("rollout identity was already attached")
        forwarded = dict(
            kwargs,
            extra_info=dict(extra, flash_rollout_identity=dict(identity)),
            flash_rollout_identity=dict(identity),
        )
        params = dict(sampling_params)
        if self.runtime.stop_sequences:
            params["stop"] = list(self.runtime.stop_sequences)
        if self.runtime.image_pad_token_id is not None:
            params["logit_bias"] = dict(params.get("logit_bias") or {})
            params["logit_bias"][self.runtime.image_pad_token_id] = -100.0
        if self.runtime.structured_outputs:
            from vllm.sampling_params import StructuredOutputsParams

            params["structured_outputs"] = StructuredOutputsParams(
                **self.runtime.structured_outputs
            )
        return await super()._run_agent_loop(
            params, trajectory, agent_name=agent_name, trace=trace, **forwarded
        )


class FlashGrpoManager(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(FlashGrpoWorker)
        super().__init__(*args, **kwargs)

    @auto_await
    async def generate_sequences(self, prompts):
        index = prompts.non_tensor_batch.get("index")
        indexes = list(range(len(prompts))) if index is None else index.tolist()
        trajectories = await get_trajectory_info(
            prompts.meta_info.get("global_steps", -1),
            indexes,
            prompts.meta_info.get("validate", False),
        )
        identities = [_identity_from_trajectory(trajectory) for trajectory in trajectories]
        url = os.environ.get("FLASH_VERL_REWARD_URL", "")
        if not url:
            raise RuntimeError(
                "flash reward bridge url is not configured for identity registration"
            )
        post_json(
            url,
            "/identity/register",
            {"identities": [dict(identity) for identity in identities]},
            error_style="reward",
        )
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        identity_chunks = []
        offset = 0
        for chunk in chunkes:
            identity_chunks.append(identities[offset : offset + len(chunk)])
            offset += len(chunk)
        if offset != len(identities):
            raise RuntimeError("flash rollout identity sidecar did not cover the manager batch")
        outputs = await asyncio.gather(
            *[
                worker.generate_sequences_with_flash_identities.remote(chunk, identity_chunk)
                for worker, chunk, identity_chunk in zip(
                    self.agent_loop_workers, chunkes, identity_chunks, strict=True
                )
            ]
        )
        output = DataProto.concat(outputs)
        metrics = [item.meta_info.pop("metrics") for item in outputs]
        timing = self._performance_metrics(metrics, output)
        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output
