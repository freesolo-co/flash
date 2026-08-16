"""Teacher-route selection for a single scored rollout.

A rollout is scored against the teacher over one of two transports: image-bearing prompts go to
the multimodal chat route as a message list with data-uri images attached, and text-only prompts
go to the plain completion route, optionally through the shared batcher that coalesces concurrent
scoring calls. Choosing between them is a small decision, but it is the only place image state
changes how the teacher is called, so it lives apart from the bridge's alignment bookkeeping.

Split out of `flash.engine.worker.train.opd.bridge` to keep that module under the file-size limit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flash.engine.worker.train.opd.gkd import _teacher_prompt_text

if TYPE_CHECKING:  # annotation-only: the prompt record lives in the orchestrator
    from flash.engine.worker.opd_train import _BridgePrompt


def score_rollout(
    prompt: _BridgePrompt,
    completion_text: str,
    *,
    teacher,
    thinking_prefill: str,
    text_teacher_batcher,
    on_scored=None,
):
    """Score one completion against the teacher, over the image or text route.

    Raises `TeacherError` through to the caller: the bridge distinguishes permanent from transient
    failures and does the no-signal accounting, which is not this module's concern.

    `on_scored` reports a completed teacher request so the parent counts as working while the verl
    child is silent. The text route reports through the batcher, so only the image route passes it.
    """
    if prompt.image_descriptors:
        from flash.content.multimodal import image_descriptors_to_data_uris

        teacher_messages = list(prompt.teacher_messages)
        if thinking_prefill:
            teacher_messages.append({"role": "assistant", "content": thinking_prefill})
        teacher_images = image_descriptors_to_data_uris(
            prompt.image_descriptors,
            prompt.package_root,
        )
        # only the synthetic thinking prefill continues the trailing assistant turn; an assistant
        # turn from the environment's own history must not absorb the completion.
        items = [
            (
                teacher_messages,
                completion_text,
                teacher_images,
                bool(thinking_prefill),
            )
        ]
        # these are PAID requests, so the keyword is chosen by type rather than by catching a
        # TypeError: a TypeError raised inside the request would otherwise be retried and billed
        # twice. a test double that lacks the keyword is reported here instead.
        from flash.engine.worker.teacher.client import TeacherClient

        if isinstance(teacher, TeacherClient):
            return teacher.score_many_multimodal(items, on_scored=on_scored)[0]
        scored = teacher.score_many_multimodal(items)[0]
        if on_scored is not None:
            on_scored()
        return scored
    teacher_prompt = _teacher_prompt_text(prompt.teacher_messages, thinking_prefill)
    if text_teacher_batcher is None:
        return teacher.score(teacher_prompt, completion_text)
    return text_teacher_batcher.score(teacher_prompt, completion_text)
