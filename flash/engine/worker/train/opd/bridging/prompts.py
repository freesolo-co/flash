"""Prompt fingerprinting and token-mask helpers for the OPD bridge.

These sit between the rollout the student produced and the payload the teacher is asked to score:
normalizing prompt ids across text and multimodal processors, shifting group metadata onto the
student's token grid, and validating or trimming the forced-token mask that marks which tokens the
teacher supplied rather than the student.

Split out of `flash.engine.worker.train.entry.opd_train` to keep that module under the file-size limit.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from flash.content.multimodal import messages_with_decoded_images
from flash.content.thinking import messages_for_chat_template
from flash.engine.worker.train.opd.orchestration.gkd import _trim_trailing_stop

if TYPE_CHECKING:
    from flash.engine.worker.train.opd.orchestration.state import _BridgePrompt


def _prompt_pool_fingerprint(prompts: list[_BridgePrompt]) -> str:
    digest = hashlib.sha256()
    for prompt in prompts:
        fingerprint_fields = [prompt.student_messages, list(prompt.prompt_ids)]
        if prompt.example is not None:
            fingerprint_fields.append(prompt.example)
        if prompt.image_descriptors:
            fingerprint_fields.extend([prompt.teacher_messages, list(prompt.image_descriptors)])
        payload = json.dumps(
            fingerprint_fields,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _normalize_prompt_ids(value) -> tuple[int, ...]:
    if isinstance(value, dict):
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if value and isinstance(value[0], list | tuple):
        value = value[0]
    if not isinstance(value, list):
        raise TypeError("processor prompt input_ids must be list-like")
    return tuple(
        int(token_id.item() if hasattr(token_id, "item") else token_id) for token_id in value
    )


def _processor_expanded_prompt(
    processor,
    messages: list[dict],
    image_descriptors: tuple[str, ...],
    package_root: str | None,
    *,
    enable_thinking: bool,
) -> tuple[tuple[int, ...], str]:
    from flash.content.multimodal import decode_image_descriptors

    images = decode_image_descriptors(list(image_descriptors), package_root)
    prepared = messages_with_decoded_images(messages, images)
    raw_prompt = processor.apply_chat_template(
        messages_for_chat_template(prepared),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        preserve_thinking=False,
    )
    processor_kwargs = {
        "text": [raw_prompt],
        "videos": None,
        "return_tensors": "pt",
    }
    if images:
        processor_kwargs["images"] = images
    model_inputs = processor(**processor_kwargs)
    return _normalize_prompt_ids(model_inputs), raw_prompt


def _processor_expanded_prompt_ids(
    processor,
    messages: list[dict],
    image_descriptors: tuple[str, ...],
    package_root: str | None,
    *,
    enable_thinking: bool,
) -> tuple[int, ...]:
    prompt_ids, _raw_prompt = _processor_expanded_prompt(
        processor,
        messages,
        image_descriptors,
        package_root,
        enable_thinking=enable_thinking,
    )
    return prompt_ids


def encode_shifted_group_metadata(
    prompt_length: int,
    response_length: int,
    groups,
) -> tuple[list[int], list[float]]:
    """Encode response group metadata on verl's full-sequence one-token-shift layout."""
    if prompt_length <= 0:
        raise ValueError("flash OPD prompts must contain at least one token")
    response_group_ids = [-1] * response_length
    response_teacher_logsums = [0.0] * response_length
    for local_group_id, (student_indices, teacher_logsum) in enumerate(groups):
        for student_index in student_indices:
            if student_index < 0 or student_index >= response_length:
                raise ValueError("flash OPD alignment group index is outside the response")
            if response_group_ids[student_index] != -1:
                raise ValueError("flash OPD response token belongs to multiple alignment groups")
            response_group_ids[student_index] = local_group_id
            response_teacher_logsums[student_index] = float(teacher_logsum)
    teacher_ids = [-1] * (prompt_length - 1) + response_group_ids + [-1]
    teacher_logprobs = [0.0] * (prompt_length - 1) + response_teacher_logsums + [0.0]
    if len(teacher_ids) != prompt_length + response_length:
        raise AssertionError("flash OPD shifted teacher metadata has the wrong length")
    return teacher_ids, teacher_logprobs


def _validate_forced_mask(
    forced,
    response_length: int,
    *,
    required: bool,
) -> list[bool]:
    if not required:
        return []
    if forced is None:
        raise ValueError("structured OPD bridge payload is missing the forced mask")
    if not isinstance(forced, list) or any(type(value) is not bool for value in forced):
        raise ValueError("structured OPD bridge forced mask must contain only booleans")
    if len(forced) != response_length:
        raise ValueError(
            "structured OPD bridge forced mask length does not match the untrimmed response"
        )
    return forced


def _trim_response_and_forced(
    tokenizer,
    response_ids: list[int],
    stop_text: str,
    stop_sequences: tuple[str, ...],
    forced: list[bool],
) -> tuple[list[int], str, list[bool]]:
    kept_ids, completion_text = _trim_trailing_stop(
        tokenizer, response_ids, stop_text, stop_sequences
    )
    return kept_ids, completion_text, forced[: len(kept_ids)]
