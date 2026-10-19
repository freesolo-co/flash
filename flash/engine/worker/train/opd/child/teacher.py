"""remote teacher manager owned by each OPD rollout worker."""

import asyncio
import json
import os

import torch
from verl.utils.config import omega_conf_to_dataclass

from flash.engine.worker.train.opd.child.bridge import (
    FlashTeacherBridgeError,
    _exit_for_score_failure,
    _post_json,
)
from flash.engine.worker.train.opd.child.policy import (
    _PERMANENT_TEACHER_EXIT,
    StructuredOutputReplay,
    _bridge_score_payload,
    canonical_structured_spec,
)


class FlashBridgeTeacherManager:
    def __init__(self, config, teacher_client=None):
        resolved = omega_conf_to_dataclass(config.distillation)
        self.bridge_url = resolved.bridge_url
        self.bridge_token = resolved.bridge_token

    async def compute_teacher_logprobs_single(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        multi_modal_data=None,
        mm_processor_kwargs=None,
        routing_key=None,
        forced: list[bool] | None = None,
    ):
        if routing_key is None:
            raise RuntimeError("flash OPD bridge requires the indexed dataset row")
        request_payload = _bridge_score_payload(
            routing_key,
            prompt_ids,
            response_ids,
            multi_modal_data,
            forced,
        )
        try:
            payload = await asyncio.to_thread(
                _post_json,
                self.bridge_url,
                self.bridge_token,
                "/score",
                request_payload,
            )
        except FlashTeacherBridgeError as error:
            _exit_for_score_failure(error)
        except Exception:
            os._exit(_PERMANENT_TEACHER_EXIT)
        teacher_ids = torch.tensor(payload["teacher_ids"], dtype=torch.int32).unsqueeze(-1)
        teacher_logprobs = torch.tensor(payload["teacher_logprobs"], dtype=torch.float32).unsqueeze(
            -1
        )
        if teacher_ids.shape != teacher_logprobs.shape:
            raise RuntimeError("flash OPD bridge returned inconsistent teacher tensors")
        if teacher_ids.shape[0] != len(request_payload["sequence_ids"]):
            raise RuntimeError("flash OPD bridge returned the wrong sequence length")
        return teacher_ids, teacher_logprobs


async def compute_flash_teacher_logprobs(
    self,
    output,
    prompt_ids: list[int],
    response_ids: list[int],
    validate: bool,
    sample_kwargs=None,
) -> None:
    if self.distillation_enabled and not validate:
        existing_ids = output.extra_fields.get("teacher_ids")
        existing_logprobs = output.extra_fields.get("teacher_logprobs")
        if existing_ids is not None or existing_logprobs is not None:
            if existing_ids is None or existing_logprobs is None:
                raise RuntimeError("flash OPD teacher metadata is incomplete")
            return
        routing_key = None
        if sample_kwargs is not None:
            routing_value = sample_kwargs.get(self.teacher_key)
            if routing_value is not None:
                routing_key = (
                    routing_value.item() if hasattr(routing_value, "item") else routing_value
                )
        forced = None
        structured_spec = os.environ.get("FLASH_OPD_STRUCTURED_OUTPUTS", "")
        if structured_spec:
            replay = getattr(self, "_flash_structured_output_replay", None)
            if replay is None:
                replay = StructuredOutputReplay(
                    self.tokenizer,
                    int(os.environ["FLASH_OPD_MODEL_VOCAB_SIZE"]),
                )
                self._flash_structured_output_replay = replay
            forced = replay.forced_mask(
                prompt_ids,
                response_ids,
                canonical_structured_spec(json.loads(structured_spec)),
                thinking=os.environ.get("FLASH_OPD_THINKING") == "1",
            )
        (
            teacher_ids,
            teacher_logprobs,
        ) = await self.teacher_server_manager.compute_teacher_logprobs_single(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            multi_modal_data=output.multi_modal_data,
            mm_processor_kwargs=output.mm_processor_kwargs,
            routing_key=routing_key,
            forced=forced,
        )
        output.extra_fields["teacher_ids"] = teacher_ids
        output.extra_fields["teacher_logprobs"] = teacher_logprobs
