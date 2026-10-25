"""Count the prompt with generation's tokenizer and reserve every possible output token."""

from typing import Any

from flash.serving.src.engine.generation import _payload
from flash.serving.src.engine.model_config import engine_overrides_for
from flash.serving.src.io.multimodal import has_image_blocks
from flash.serving.src.io.schemas import AdapterRecord


async def token_budget(owner: Any, payload_dict: dict, record_dict: dict) -> int:
    payload, _ = _payload(payload_dict)
    record = AdapterRecord.model_validate(record_dict)
    if record.base_model != owner.base_model or record.model_id != payload.model_id:
        raise ValueError("Token budget target does not match request")
    thinking = owner._thinking_default(record, payload)
    if has_image_blocks(payload.messages):
        # Image expansion is performed inside vLLM. Reserve the complete served context per
        # choice rather than undercounting image patches from the pre-expansion text tokenizer.
        return int(engine_overrides_for(owner.base_model)["max_model_len"]) * payload.n
    prompt = await owner._prompt_input(payload, thinking)
    return len(prompt["prompt_token_ids"]) + payload.max_tokens * payload.n
