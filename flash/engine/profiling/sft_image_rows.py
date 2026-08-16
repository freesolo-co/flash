"""explicit image-row tokenization for sft profiling and training."""

from __future__ import annotations

import io
from pathlib import Path

from flash.content.multimodal import (
    IMAGE_PAD_TOKEN,
    decode_image_descriptors,
    messages_with_decoded_images,
)
from flash.engine.profiling.image_tokens import (
    ImageGeometry,
    ImageProfileValidationState,
    descriptor_pad_tokens,
    expand_image_pad_runs,
)
from flash.engine.worker.model.chatml_mask import assistant_only_mask
from flash.engine.worker.model.packing import completion_mask_from_ids


def _serialize_multimodal_inputs(values: dict) -> bytes:
    if not values:
        return b""
    import numpy as np

    arrays = {}
    for key, value in values.items():
        if value is None:
            continue
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arrays[key] = np.asarray(value)
    if not arrays:
        return b""
    payload = io.BytesIO()
    np.savez(payload, **arrays)
    return payload.getvalue()


def _ids(value) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def estimate_sft_image_row(
    tokenizer,
    prompt_messages: list[dict],
    completion_messages: list[dict],
    descriptors: list[str],
    *,
    package_root: str | Path | None,
    geometry: ImageGeometry,
    validation_state: ImageProfileValidationState,
    max_length: int,
    thinking: bool,
) -> tuple[list[int], list[int], bytes, int, bool]:
    """count the exact processor token ids without constructing image tensors."""
    pad_token_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
    if not isinstance(pad_token_id, int) or pad_token_id < 0:
        raise ValueError(
            f"tokenizer does not define the image placeholder {IMAGE_PAD_TOKEN!r}, so an "
            "image-bearing sft dataset cannot be quoted for this model"
        )
    pad_counts = descriptor_pad_tokens(
        descriptors,
        package_root,
        geometry,
        validation_state,
    )

    def token_ids(messages: list[dict], *, add_generation_prompt: bool) -> list[int]:
        rendered = dict(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=thinking,
            )
        )["input_ids"]
        return expand_image_pad_runs(_ids(rendered), pad_token_id, pad_counts)

    full_messages = [*prompt_messages, *completion_messages]
    untruncated_ids = token_ids(full_messages, add_generation_prompt=False)
    untruncated_length = len(untruncated_ids)
    input_ids = untruncated_ids[:max_length]
    prompt_ids = token_ids(prompt_messages, add_generation_prompt=True)[:max_length]
    # the same narrowing the processor path applies. the quote has to agree with what training will
    # actually supervise: leaving the contiguous mask here would count the environment's replies as
    # supervised tokens the worker then masks away, and report a step budget for a different run.
    mask, role_aware = assistant_only_mask(
        completion_mask_from_ids(prompt_ids, input_ids),
        input_ids,
        tokenizer,
        completion_messages,
        # this path appends nothing of its own, so there is no appended EOS to preserve.
        appended_eos=False,
        template_source=tokenizer,
        source_messages=full_messages,
        template_kwargs={"enable_thinking": thinking},
    )
    return input_ids, mask, b"", untruncated_length, role_aware


def process_sft_image_row(
    processor,
    prompt_messages: list[dict],
    completion_messages: list[dict],
    descriptors: list[str],
    *,
    package_root: str | Path | None,
    max_length: int,
    thinking: bool,
) -> tuple[list[int], list[int], bytes, int, bool]:
    """tokenize one image row through the real processor and serialize its tensors."""
    images = decode_image_descriptors(descriptors, package_root)
    try:
        prepared_prompt = messages_with_decoded_images(prompt_messages, images)
        full_messages = [*prepared_prompt, *completion_messages]
        common = {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "enable_thinking": thinking,
        }
        full = dict(
            processor.apply_chat_template(
                full_messages,
                add_generation_prompt=False,
                **common,
            )
        )
        prompt = dict(
            processor.apply_chat_template(
                prepared_prompt,
                add_generation_prompt=True,
                **common,
            )
        )
        untruncated_ids = _ids(full.pop("input_ids"))
        untruncated_length = len(untruncated_ids)
        input_ids = untruncated_ids[:max_length]
        prompt_ids = _ids(prompt["input_ids"])[:max_length]
        # the span reader needs the text tokenizer's vocabulary; a processor exposes it as
        # .tokenizer. the IMAGE-PREPARED messages are what the processor actually rendered, so
        # probing the unprepared list would validate a different transcript than the one masked.
        mask, role_aware = assistant_only_mask(
            completion_mask_from_ids(prompt_ids, input_ids),
            input_ids,
            getattr(processor, "tokenizer", processor),
            completion_messages,
            # this path appends nothing of its own, so there is no appended EOS to preserve.
            appended_eos=False,
            template_source=processor,
            source_messages=full_messages,
            template_kwargs={"enable_thinking": thinking},
        )
        full.pop("attention_mask", None)
        return (
            input_ids,
            mask,
            _serialize_multimodal_inputs(full),
            untruncated_length,
            role_aware,
        )
    finally:
        for image in images:
            image.close()
