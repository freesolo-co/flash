"""Flash agent policies dispatched by upstream's unmodified TransferQueue workers."""

import asyncio
import os

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ
from verl.utils import tensordict_utils as tu

from flash.engine.worker.train.core.child.observations import ObservedRolloutClient
from flash.engine.worker.train.core.child.options import RuntimeOptions
from flash.engine.worker.train.rl.child.identity import _identity_from_trajectory
from flash.engine.worker.train.rl.child.multiturn import (
    build_flash_grpo_multi_turn_agent_loop,
    post_json,
)
from flash.engine.worker.train.rl.child.reward import compute_score


class FlashGrpoManager(AgentLoopManagerTQ):
    def generate_sequences(self, prompts):
        tu.assign_non_tensor_data(prompts, "flash_validate", bool(prompts.get("validate", False)))
        return super().generate_sequences(prompts)


class FlashGrpoPolicy:
    def __init__(self, *args, server_manager, **kwargs):
        self.observations = ObservedRolloutClient(server_manager)
        super().__init__(*args, server_manager=self.observations, **kwargs)

    async def run(self, sampling_params, **kwargs):
        runtime = RuntimeOptions.decode(dict(self.config.actor_rollout_ref.model.runtime_options))
        identity = _identity_from_trajectory(
            {
                "step": int(kwargs["global_steps"]),
                "sample_index": int(kwargs["index"]),
                "rollout_n": int(kwargs["session_id"]),
                "validate": bool(kwargs.get("flash_validate", False)),
            }
        )
        extra = dict(kwargs.get("extra_info") or {})
        if "flash_rollout_identity" in extra or "flash_rollout_identity" in kwargs:
            raise RuntimeError("rollout identity was already attached")
        extra["flash_rollout_identity"] = identity
        extra.setdefault("index", identity["sample_index"])
        kwargs = dict(kwargs, extra_info=extra, flash_rollout_identity=identity)
        url = os.environ["FLASH_VERL_REWARD_URL"]
        await asyncio.to_thread(
            post_json, url, "/identity/register", {"identities": [identity]}, error_style="reward"
        )
        params = dict(sampling_params)
        if runtime.stop_sequences:
            params["stop"] = list(runtime.stop_sequences)
        if runtime.image_pad_token_id is not None:
            params["logit_bias"] = dict(params.get("logit_bias") or {})
            params["logit_bias"][runtime.image_pad_token_id] = -100.0
        if runtime.structured_outputs:
            from vllm.sampling_params import StructuredOutputsParams

            params["structured_outputs"] = StructuredOutputsParams(**runtime.structured_outputs)
        output = await super().run(params, **kwargs)
        if output.reward_score is None:
            output.reward_score = await asyncio.to_thread(
                compute_score,
                kwargs.get("data_source"),
                self.tokenizer.decode(output.response_ids, skip_special_tokens=True),
                (kwargs.get("reward_model") or {}).get("ground_truth"),
                extra,
            )
        if (
            runtime.mask_truncated_completions
            and output.extra_fields.get("finish_reason") == "length"
        ):
            output.response_mask = [0] * len(output.response_mask)
        return self.observations.annotate(output)


@register("flash_grpo_single_turn")
class FlashGrpoSingleTurnAgentLoop(FlashGrpoPolicy, SingleTurnAgentLoop):
    pass


_MultiTurn = build_flash_grpo_multi_turn_agent_loop(
    register=register,
    agent_loop_base=AgentLoopBase,
    agent_loop_output=AgentLoopOutput,
)


@register("flash_grpo_multi_turn")
class FlashGrpoMultiTurnAgentLoop(FlashGrpoPolicy, _MultiTurn):
    pass
