"""multi-turn OPD media identity and environment reply preparation."""

from __future__ import annotations

from dataclasses import dataclass

from flash.content.multimodal import normalize_prompt_images, text_only_prompt_messages
from flash.engine.worker.train.core.child.glue import (
    dedup_seam_terminator,
    parent_environment_glue,
    parent_image_digests,
    validate_structured_messages,
)


@dataclass(frozen=True)
class PreparedEnvironmentReply:
    messages: list[dict]
    descriptors: tuple[str, ...]
    data_uris: tuple[str, ...]
    image_digests: tuple[str, ...]
    glue_ids: tuple[int, ...]


def validate_start_media(
    prompt,
    processor,
    index: int,
    image_count: int,
    image_digests,
) -> tuple[str, ...]:
    expected_image_count = len(prompt.image_descriptors)
    if int(image_count) != expected_image_count:
        raise ValueError(
            f"multi-turn rollout reported {int(image_count)} image(s) for dataset index "
            f"{index}; the frozen prompt has {expected_image_count}"
        )
    if image_digests is None and expected_image_count == 0:
        image_digests = []
    if not isinstance(image_digests, list) or any(
        not isinstance(value, str) for value in image_digests
    ):
        raise ValueError("multi-turn rollout image digests must be a list of strings")
    expected_digests = tuple(prompt.image_digests) or tuple(
        parent_image_digests(processor, prompt.image_descriptors, prompt.package_root)
    )
    if tuple(image_digests) != expected_digests:
        raise ValueError("multi-turn rollout media does not match the frozen flash prompt")
    return expected_digests


def normalize_initial_prompt(prompt, state: dict, processor) -> tuple[list[dict], tuple[str, ...]]:
    # the INITIAL prefix, so `prompt` only: `new_rollout_state` seeds `messages` with a copy of it
    # and appends every turn, so falling back to `messages` normalizes the growing transcript
    # against the frozen prompt's media and mismatches its digests once a turn lands.
    initial_messages = state.get("prompt")
    if processor is not None or prompt.image_descriptors:
        normalized = normalize_prompt_images(
            prompt.example,
            initial_messages,
            prompt.package_root,
        )
        normalized_messages = (
            normalized.messages
            if normalized.descriptors
            else text_only_prompt_messages(normalized.messages)
        )
        initial_messages = validate_structured_messages(
            normalized_messages,
            source="environment initial prompt",
        )
        fresh_descriptors = tuple(normalized.descriptors)
    else:
        initial_messages = validate_structured_messages(
            initial_messages,
            source="environment initial prompt",
        )
        fresh_descriptors = ()
    return initial_messages, fresh_descriptors


def step_media_identity(payload: dict) -> tuple[int, list[str]]:
    """Read the media identity the child attests for this turn.

    the caller compares the result against the session's own media, which is what catches a parent
    and child that have drifted apart. defaulting a missing key to the session's values would make
    that comparison compare the session to itself and pass unconditionally, turning the one drift
    this detects -- a child that stopped reporting media at all -- into the one case it cannot see.
    the child always sends both keys, so require them.
    """
    if "image_count" not in payload or "image_digests" not in payload:
        raise ValueError("multi-turn rollout step must report its image count and digests")
    image_count = int(payload["image_count"])
    supplied_digests = payload["image_digests"]
    if not isinstance(supplied_digests, list) or any(
        not isinstance(value, str) for value in supplied_digests
    ):
        raise ValueError("multi-turn rollout image digests must be a list of strings")
    return image_count, supplied_digests


def prepare_environment_reply(
    raw_messages,
    *,
    normalize_reply,
    prompt,
    cumulative_descriptors,
    processor,
    tokenizer,
    thinking: bool,
    response_ids: list[int],
) -> PreparedEnvironmentReply:
    normalized = normalize_reply(
        raw_messages,
        prompt.package_root,
        cumulative_descriptors,
    )
    glue_ids, new_digests = parent_environment_glue(
        processor,
        tokenizer,
        normalized.messages,
        normalized.descriptors,
        prompt.package_root,
        thinking=thinking,
    )
    return PreparedEnvironmentReply(
        messages=normalized.messages,
        descriptors=normalized.descriptors,
        data_uris=normalized.data_uris,
        image_digests=tuple(new_digests),
        glue_ids=tuple(dedup_seam_terminator(response_ids, glue_ids)),
    )
