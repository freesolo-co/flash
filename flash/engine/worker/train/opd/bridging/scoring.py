"""Teacher-route selection and canonical multimodal item construction for OPD scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flash.engine.worker.train.opd.orchestration.gkd import _teacher_prompt_text

if TYPE_CHECKING:
    from flash.engine.worker.train.opd.orchestration.state import _BridgePrompt


def build_multimodal_score_items(
    prompt: _BridgePrompt,
    history_completions_and_descriptors: list[tuple[list[dict] | None, str, tuple[str, ...]]],
    *,
    thinking_prefill: str,
) -> list[tuple[list[dict], str, list[str], bool]]:
    """build ordered teacher items from each turn's immutable cumulative media snapshot."""
    from flash.content.multimodal import (
        image_descriptors_to_data_uris,
        image_teacher_prompt_messages,
    )

    image_cache: dict[tuple[str, ...], list[str]] = {}
    items = []
    for history, completion_text, descriptors in history_completions_and_descriptors:
        descriptors = tuple(descriptors)
        teacher_images = image_cache.get(descriptors)
        if teacher_images is None:
            teacher_images = image_descriptors_to_data_uris(descriptors, prompt.package_root)
            image_cache[descriptors] = teacher_images
        teacher_messages = (
            list(prompt.teacher_messages)
            if history is None
            else image_teacher_prompt_messages(history, len(descriptors))
        )
        if thinking_prefill:
            teacher_messages.append({"role": "assistant", "content": thinking_prefill})
        # only the synthetic thinking prefill continues the trailing assistant turn. an assistant
        # turn from environment history must not absorb the completion.
        items.append(
            (
                teacher_messages,
                completion_text,
                teacher_images,
                bool(thinking_prefill),
            )
        )
    return items


def score_multimodal_items(teacher, items, *, on_scored=None):
    """score one ordered image batch and report each completed teacher request."""
    # these are paid requests, so the keyword is chosen by type rather than by catching a TypeError:
    # a TypeError raised inside the request would otherwise be retried and billed twice.
    from flash.engine.worker.teacher.client import TeacherClient

    if isinstance(teacher, TeacherClient):
        return teacher.score_many_multimodal(items, on_scored=on_scored)
    scored = teacher.score_many_multimodal(items)
    if on_scored is not None:
        for _score in scored:
            on_scored()
    return scored


def score_rollout(
    prompt: _BridgePrompt,
    completion_text: str,
    *,
    teacher,
    thinking_prefill: str,
    text_teacher_batcher,
    on_scored=None,
):
    """score one completion against the teacher over the image or text route."""
    if prompt.image_descriptors:
        items = build_multimodal_score_items(
            prompt,
            [(None, completion_text, prompt.image_descriptors)],
            thinking_prefill=thinking_prefill,
        )
        return score_multimodal_items(teacher, items, on_scored=on_scored)[0]
    teacher_prompt = _teacher_prompt_text(prompt.teacher_messages, thinking_prefill)
    if text_teacher_batcher is None:
        return teacher.score(teacher_prompt, completion_text)
    return text_teacher_batcher.score(teacher_prompt, completion_text)
