"""explicit image-row tokenization for sft profiling and training."""

from __future__ import annotations

import io
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from flash.content.multimodal import (
    IMAGE_PAD_TOKEN,
    NormalizedImages,
    content_has_images,
    decode_image_descriptors,
    image_block_count,
    messages_with_decoded_images,
    text_only_prompt_messages,
)
from flash.content.thinking import messages_for_chat_template
from flash.engine.profiling.image_tokens import (
    ImageGeometry,
    ImageProfileValidationState,
    descriptor_pad_tokens,
    expand_image_pad_runs,
)
from flash.engine.worker.model.chatml_mask import (
    assistant_only_mask,
    reject_rendered_message_token,
)
from flash.engine.worker.model.packing import completion_mask_from_ids


def _serialize_multimodal_inputs(values: dict) -> bytes:
    if not values:
        return b""
    import numpy as np

    arrays: dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            continue
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arrays[key] = np.asarray(value)
    if not arrays:
        return b""
    payload = io.BytesIO()
    # token-type ids are mostly zeros; compression keeps per-row metadata small.
    np.savez_compressed(payload, **arrays)
    return payload.getvalue()


def _normalize_token_ids(value) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


# successful _registered_token_id verdicts, per live tokenizer object and token.
_REGISTERED_TOKEN_IDS: weakref.WeakKeyDictionary[Any, dict[str, int]] = weakref.WeakKeyDictionary()
_REGISTERED_TOKEN_IDS_LOCK = threading.Lock()


def _registered_token_id(tokenizer, token: str) -> int:
    """return one exact, canonical registered token id without aliases or unknown fallback.

    the check copies and scans the whole vocabulary (~250k entries for qwen), and submission
    profiling asks it once per image row: recomputing it cost ~58 ms a row, most of a 4k-row
    dataset's quote, held under the gil in the shared control plane. a successful verdict is
    therefore memoized per tokenizer object (loaded tokenizers are not given new tokens while
    profiling; even len() is ~14 ms, so the vocabulary size cannot cheaply be part of the key).
    failures are never cached and always re-raise from a fresh check.
    """
    try:
        with _REGISTERED_TOKEN_IDS_LOCK:
            cached = _REGISTERED_TOKEN_IDS.get(tokenizer, {}).get(token)
    except TypeError:  # not hashable or not weak-referenceable: check every time
        return _check_registered_token_id(tokenizer, token)
    if cached is not None:
        return cached
    token_id = _check_registered_token_id(tokenizer, token)
    with _REGISTERED_TOKEN_IDS_LOCK:
        _REGISTERED_TOKEN_IDS.setdefault(tokenizer, {})[token] = token_id
    return token_id


def _check_registered_token_id(tokenizer, token: str) -> int:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    token_id = convert(token) if callable(convert) else None
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if (
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        or token_id == unk_token_id
    ):
        raise ValueError(f"tokenizer does not define the exact token {token!r}")

    to_token = getattr(tokenizer, "convert_ids_to_tokens", None)
    canonical = to_token(token_id) if callable(to_token) else None
    if canonical is not None and canonical != token:
        raise ValueError(f"tokenizer id {token_id} resolves to {canonical!r}, not {token!r}")

    get_vocab = getattr(tokenizer, "get_vocab", None)
    vocab = get_vocab() if callable(get_vocab) else None
    if isinstance(vocab, dict):
        alias = next(
            (name for name, value in vocab.items() if value == token_id and name != token),
            None,
        )
        if alias is not None:
            raise ValueError(f"tokenizer id {token_id} is shared by {token!r} and {alias!r}")
        if vocab.get(token) == token_id:
            return token_id
    if canonical == token:
        return token_id
    raise ValueError(f"tokenizer does not define the exact token {token!r}")


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
    normalized_prompt = messages_for_chat_template(prompt_messages)
    normalized_completion = messages_for_chat_template(completion_messages)
    full_messages = [*normalized_prompt, *normalized_completion]
    rendered_probe = reject_rendered_message_token(
        tokenizer,
        full_messages,
        IMAGE_PAD_TOKEN,
        template_kwargs={"enable_thinking": thinking, "preserve_thinking": False},
    )
    try:
        pad_token_id = _registered_token_id(tokenizer, IMAGE_PAD_TOKEN)
    except ValueError as exc:
        raise ValueError(
            f"tokenizer does not define the image placeholder {IMAGE_PAD_TOKEN!r}, so an "
            "image-bearing sft dataset cannot be quoted for this model"
        ) from exc
    pad_counts = descriptor_pad_tokens(
        descriptors,
        package_root,
        geometry,
        validation_state,
    )

    def token_ids(
        messages: list[dict], *, add_generation_prompt: bool, pad_counts: list[int] = pad_counts
    ) -> list[int]:
        rendered = dict(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=thinking,
                preserve_thinking=False,
            )
        )["input_ids"]
        return expand_image_pad_runs(_normalize_token_ids(rendered), pad_token_id, pad_counts)

    untruncated_ids = token_ids(full_messages, add_generation_prompt=False)
    untruncated_length = len(untruncated_ids)
    input_ids = untruncated_ids[:max_length]
    # observation images in the completion follow the prompt's, so the prompt render holds exactly
    # the leading pad runs.
    prompt_ids = token_ids(
        normalized_prompt,
        add_generation_prompt=True,
        pad_counts=pad_counts[: image_block_count(normalized_prompt)],
    )[:max_length]
    # the same narrowing the processor path applies. the quote has to agree with what training will
    # actually supervise: leaving the contiguous mask here would count the environment's replies as
    # supervised tokens the worker then masks away, and report a step budget for a different run.
    mask, role_aware = assistant_only_mask(
        completion_mask_from_ids(prompt_ids, input_ids),
        input_ids,
        tokenizer,
        normalized_completion,
        # this path appends nothing of its own, so there is no appended EOS to preserve.
        appended_eos=False,
        template_source=tokenizer,
        source_messages=full_messages,
        template_kwargs={"enable_thinking": thinking, "preserve_thinking": False},
        rendered_probe=rendered_probe,
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
    images = cast("list[Any]", decode_image_descriptors(descriptors, package_root))
    try:
        # images bind in conversation order: the prompt's first, then any user observation images
        # between the completion's assistant turns, which are input context like a prompt image.
        prepared = messages_with_decoded_images([*prompt_messages, *completion_messages], images)
        prepared_prompt = messages_for_chat_template(prepared[: len(prompt_messages)])
        normalized_completion = messages_for_chat_template(prepared[len(prompt_messages) :])
        full_messages = [*prepared_prompt, *normalized_completion]
        rendered_probe = reject_rendered_message_token(
            processor,
            full_messages,
            IMAGE_PAD_TOKEN,
            template_kwargs={"enable_thinking": thinking, "preserve_thinking": False},
        )
        common = {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "enable_thinking": thinking,
            "preserve_thinking": False,
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
        untruncated_ids = _normalize_token_ids(full.pop("input_ids"))
        untruncated_length = len(untruncated_ids)
        input_ids = untruncated_ids[:max_length]
        prompt_ids = _normalize_token_ids(prompt["input_ids"])[:max_length]
        # the span reader needs the text tokenizer's vocabulary; a processor exposes it as
        # .tokenizer. the IMAGE-PREPARED messages are what the processor actually rendered, so
        # probing the unprepared list would validate a different transcript than the one masked.
        mask, role_aware = assistant_only_mask(
            completion_mask_from_ids(prompt_ids, input_ids),
            input_ids,
            getattr(processor, "tokenizer", processor),
            normalized_completion,
            # this path appends nothing of its own, so there is no appended EOS to preserve.
            appended_eos=False,
            template_source=processor,
            source_messages=full_messages,
            template_kwargs={"enable_thinking": thinking, "preserve_thinking": False},
            rendered_probe=rendered_probe,
        )
        full.pop("attention_mask", None)
        # pixel tensors are the bulk of an image row (~4 MB per 498x321 screenshot, repeated for
        # every row whose history shows it), so they are not stored: the training child rebuilds
        # them from the row's saved image files and checks the stored image_grid_thw.
        full.pop("pixel_values", None)
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


def normalize_observation_conversation(
    normalize: Callable[[dict, list[dict], str | Path | None], NormalizedImages],
    record: dict,
    prompt_messages: list[dict],
    completion_messages: list[dict],
    package_root: str | Path | None,
) -> tuple[list[dict], list[dict], list[str]]:
    """normalize a row whose completion carries user observation images.

    the prompt and every image-bearing completion message are normalized as ONE conversation, so the
    descriptors come out in rendered order and the per-row image limits (count, source and decoded
    bytes) bound the whole conversation, not each half. completion messages without images keep the
    text-only form the prompt-image path gives every target.
    """
    observed = [
        index
        for index, message in enumerate(completion_messages)
        if content_has_images(message.get("content"))
    ]
    normalized = normalize(
        record,
        [*prompt_messages, *(completion_messages[index] for index in observed)],
        package_root,
    )
    prompt_count = len(prompt_messages)
    by_index = dict(zip(observed, normalized.messages[prompt_count:], strict=True))
    text_only = text_only_prompt_messages(completion_messages)
    completion = [by_index.get(index, text_only[index]) for index in range(len(text_only))]
    return normalized.messages[:prompt_count], completion, list(normalized.descriptors)


@dataclass(frozen=True)
class FittedObservationRow:
    """one image-observation row tokenized at a whole-turn prefix that fits the context cap."""

    result: tuple[list[int], list[int], bytes, int, bool]
    completion_messages: list[dict]
    descriptors: list[str]
    # the WHOLE conversation's length, so truncation reporting still sees the row the cap bound.
    untruncated_length: int


def fit_observation_conversation(
    tokenize: Callable[..., tuple[list[int], list[int], bytes, int, bool]],
    prompt_messages: list[dict],
    completion_messages: list[dict],
    descriptors: list[str],
    *,
    package_root: str | Path | None,
    max_length: int,
) -> FittedObservationRow:
    """tokenize an observation-image row, trimming it to whole turns when the cap binds.

    slicing the token row would cut a screenshot's pad run (its image grid would no longer match the
    placeholders) or an observation's closing delimiter (the assistant-only mask then cannot read
    the transcript and would supervise the environment's replies). an overlong row is therefore
    trimmed to the longest completion prefix that ends on an assistant turn and fits ``max_length``;
    only that prefix's images are kept. a row whose first assistant turn alone overflows keeps the
    ordinary right truncation when that prefix shows no completion image, and is refused otherwise.
    a row without completion images is tokenized once, exactly as before.
    """
    prompt_images = image_block_count(prompt_messages)

    def run(count: int) -> FittedObservationRow:
        part = completion_messages[:count]
        used = descriptors[: prompt_images + image_block_count(part)]
        result = tokenize(prompt_messages, part, used, package_root=package_root)
        return FittedObservationRow(result, part, used, result[3])

    fitted = run(len(completion_messages))
    whole_length = fitted.untruncated_length
    if not image_block_count(completion_messages):
        # a prompt-image row keeps the ordinary right truncation: no observation image can be cut.
        return fitted
    if whole_length > max_length:
        cuts = [
            count
            for count in range(1, len(completion_messages))
            if completion_messages[count - 1].get("role") == "assistant"
        ]
        best: FittedObservationRow | None = None
        # rendered length is NOT monotonic in the prefix: a template that drops earlier assistant
        # reasoning once a later observation becomes the last query (Qwen3.6 with
        # preserve_thinking=False) can render a longer prefix shorter than a shorter one. so no
        # oversized cut rules out a longer one; scan longest first and keep the first that fits.
        for count in reversed(cuts):
            candidate = run(count)
            if candidate.untruncated_length <= max_length:
                best = candidate
                break
        if best is None and cuts:
            best = run(cuts[0])
        fitted = best or fitted
        if fitted.untruncated_length > max_length and image_block_count(fitted.completion_messages):
            raise ValueError(
                f"an sft row needs {whole_length} tokens but max_context_tokens is {max_length}, "
                "and its first whole turn would cut an observation image; raise "
                "max_context_tokens or shorten the trajectory"
            )
    if image_block_count(fitted.completion_messages) and not fitted.result[4]:
        raise ValueError(
            "an sft completion with observation images needs a ChatML chat template whose "
            "assistant turns can be located, so the observations stay out of the loss"
        )
    return FittedObservationRow(
        fitted.result, fitted.completion_messages, fitted.descriptors, whole_length
    )
