"""image-row tokenization shared by sft profiling paths."""

from __future__ import annotations

from collections.abc import Callable


def _estimated_tokenized_row(
    tokenizer,
    prompt_messages: list[dict],
    completion_messages: list[dict],
    descriptors: list[str],
    *,
    package_root,
    geometry,
    max_length: int,
    thinking: bool,
) -> tuple[list[int], list[int], int]:
    """Tokenize one image row without a processor, for the torch-free control plane.

    The plain tokenizer renders an image block to a single ``<|image_pad|>``; expanding that
    placeholder to the run the vision tower occupies reproduces the processor's exact id sequence,
    so the prompt/full boundary -- and therefore the completion mask -- is unchanged. the shared
    worker validator fully loads each bounded payload before its dimensions are used here.
    """
    from flash.content.multimodal import IMAGE_PAD_TOKEN
    from flash.engine.profiling.image_tokens import descriptor_pad_tokens, expand_image_pad_runs
    from flash.engine.worker.model.packing import completion_mask_from_ids

    pad_token_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
    if not isinstance(pad_token_id, int) or pad_token_id < 0:
        raise ValueError(
            f"tokenizer does not define the image placeholder {IMAGE_PAD_TOKEN!r}, so an "
            "image-bearing sft dataset cannot be quoted for this model"
        )
    pad_counts = descriptor_pad_tokens(descriptors, package_root, geometry)

    def ids(messages: list[dict], *, add_generation_prompt: bool) -> list[int]:
        rendered = dict(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=thinking,
            )
        )["input_ids"]
        if rendered and isinstance(rendered[0], list):
            rendered = rendered[0]
        return expand_image_pad_runs([int(item) for item in rendered], pad_token_id, pad_counts)

    untruncated_ids = ids([*prompt_messages, *completion_messages], add_generation_prompt=False)
    untruncated_length = len(untruncated_ids)
    input_ids = untruncated_ids[:max_length]
    prompt_ids = ids(prompt_messages, add_generation_prompt=True)[:max_length]
    return input_ids, completion_mask_from_ids(prompt_ids, input_ids), untruncated_length


def _tokenize_sft_image_row(
    tokenizer,
    processor,
    prompt_messages: list[dict],
    completion_messages: list[dict],
    descriptors: list[str],
    *,
    package_root,
    geometry,
    max_length: int,
    thinking: bool,
    decode_image_descriptors: Callable,
    processor_tokenized_row: Callable,
) -> tuple[list[int], list[int], bytes, int]:
    """tokenize one normalized image row through the selected profiling path."""
    if processor is None:
        # torch-free control plane: count the tokens the processor would produce rather
        # than producing them. pillow validates the pixels but no training tensors exist, so
        # the row carries no multimodal_inputs -- the worker, which has the processor, builds
        # the real ones (sft_train_runner passes a real image_dir).
        input_ids, loss_mask, untruncated_length = _estimated_tokenized_row(
            tokenizer,
            prompt_messages,
            completion_messages,
            descriptors,
            package_root=package_root,
            geometry=geometry,
            max_length=max_length,
            thinking=thinking,
        )
        return input_ids, loss_mask, b"", untruncated_length

    decoded_images = decode_image_descriptors(descriptors, package_root)
    return processor_tokenized_row(
        processor,
        prompt_messages,
        completion_messages,
        decoded_images,
        max_length=max_length,
        thinking=thinking,
    )
