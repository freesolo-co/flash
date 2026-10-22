"""explicit OPD rollout worker and transfer-queue manager types."""

import copy
import json
import os
from typing import Any

import transfer_queue as tq
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ

from flash.content.structured_options import vllm_structured_kwargs
from flash.engine.worker.train.opd.child.bridge import (
    _defer_score_failure,
    _fatal_rollout_exit_code,
    _post_json,
)
from flash.engine.worker.train.opd.child.multiturn import build_flash_multi_turn_agent_loop
from flash.engine.worker.train.opd.child.policy import (
    _PERMANENT_TEACHER_EXIT,
    _TRANSIENT_TEACHER_EXIT,
    _raw_prompt_has_image_block,
    _require_structured_runtime_versions,
    _resolve_image_token_id,
    deterministic_rollout_seed,
)
from flash.engine.worker.train.opd.child.teacher import (
    FlashBridgeTeacherManager,
    compute_flash_teacher_logprobs,
)


@register("flash_single_turn")
class FlashSingleTurnAgentLoop(SingleTurnAgentLoop):
    async def run(self, sampling_params: dict[str, Any], **kwargs):
        params = dict(sampling_params)
        if _raw_prompt_has_image_block(kwargs.get("raw_prompt")):
            image_token_id = _resolve_image_token_id(self.processor, self.tokenizer)
            logit_bias = dict(params.get("logit_bias") or {})
            logit_bias[image_token_id] = -100.0
            params["logit_bias"] = logit_bias
        flash_seed = int(os.environ["FLASH_OPD_SEED"])
        global_step = int(kwargs["global_steps"])
        example_index = int(kwargs["index"])
        rollout_ordinal = int(kwargs.get("session_id", 0))
        no_signal_attempt_ordinal = int(kwargs.get("flash_no_signal_attempt", 0))
        params["seed"] = deterministic_rollout_seed(
            flash_seed,
            global_step,
            example_index,
            rollout_ordinal,
            no_signal_attempt_ordinal=no_signal_attempt_ordinal,
        )
        stops = json.loads(os.environ.get("FLASH_OPD_STOP_SEQUENCES", "[]"))
        if stops:
            params["stop"] = stops
            params["include_stop_str_in_output"] = True
        eos_ids = json.loads(os.environ.get("FLASH_OPD_EOS_TOKEN_IDS", "[]"))
        if eos_ids:
            params["stop_token_ids"] = eos_ids
        structured_spec = os.environ.get("FLASH_OPD_STRUCTURED_OUTPUTS", "")
        if structured_spec:
            _require_structured_runtime_versions()
            from vllm.sampling_params import StructuredOutputsParams

            params["structured_outputs"] = StructuredOutputsParams(
                **vllm_structured_kwargs(json.loads(structured_spec))
            )
        output = await super().run(params, **kwargs)
        output.reward_score = 0.0
        output.extra_fields.setdefault("reward_extra_info", {})
        self.distillation_enabled = True
        self.teacher_key = self.config.distillation.teacher_key
        self.teacher_server_manager = FlashBridgeTeacherManager(self.config)
        await compute_flash_teacher_logprobs(
            self, output, output.prompt_ids, output.response_ids, False, sample_kwargs=kwargs
        )
        return output


def mark_prompt_failure(uid, partition_id, global_steps, status):
    tq.kv_put(
        key=uid, partition_id=partition_id, tag={"global_steps": global_steps, "status": status}
    )


_MultiTurn = build_flash_multi_turn_agent_loop(
    register=register,
    agent_loop_base=AgentLoopBase,
    agent_loop_output=AgentLoopOutput,
    post_json=_post_json,
    score_failure_handler=_defer_score_failure,
    fatal_rollout_exit_code=_fatal_rollout_exit_code,
    mark_prompt_failure=mark_prompt_failure,
    deterministic_seed=deterministic_rollout_seed,
    permanent_teacher_exit=_PERMANENT_TEACHER_EXIT,
    transient_teacher_exit=_TRANSIENT_TEACHER_EXIT,
)


@register("flash_multi_turn")
class FlashMultiTurnAgentLoop(_MultiTurn):
    async def run(self, sampling_params, **kwargs):
        outputs = await super().run(sampling_params, **kwargs)
        for output in outputs:
            # The bridge already supplies the distillation target for every turn.
            output.reward_score = 0.0
            output.extra_fields.setdefault("reward_extra_info", {})
        return outputs


class FlashOpdManager(AgentLoopManagerTQ):
    def __init__(self, config, *args, **kwargs):
        # the packaged agent scores through the remote bridge; no native teacher pool is needed.
        config = copy.deepcopy(config)
        config.distillation.enabled = False
        super().__init__(config, *args, **kwargs)
