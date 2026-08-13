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
):
    """Score one completion against the teacher, over the image or text route.

    Raises `TeacherError` through to the caller: the bridge distinguishes permanent from transient
    failures and does the no-signal accounting, which is not this module's concern.
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
        return teacher.score_many_multimodal(
            [
                (
                    teacher_messages,
                    completion_text,
                    teacher_images,
                    bool(thinking_prefill),
                )
            ]
        )[0]
    teacher_prompt = _teacher_prompt_text(prompt.teacher_messages, thinking_prefill)
    if text_teacher_batcher is None:
        return teacher.score(teacher_prompt, completion_text)
    return text_teacher_batcher.score(teacher_prompt, completion_text)


def score_multiturn_batch(
    turns: list[dict],
    prompt: _BridgePrompt,
    *,
    teacher,
    thinking_prefill: str,
):
    """Score one episode's scorable turns, over the same image or text route as a single rollout.

    The turn list is scored in ONE call so the teacher's bounded pool keeps its concurrency
    ceiling; both `score_many` and `score_many_multimodal` wrap the same `map_bounded` at the same
    cap, and both preserve input order, so the caller can zip the results back onto its turns.

    Images come from the frozen initial prompt, so every turn of the episode shares one decoded
    URI list: turn 2's context still contains the image-bearing messages that opened the episode.
    An image returned by the environment mid-episode is a separate, unsupported case and never
    reaches here as an image -- it would have to be added to the prompt's descriptors first.
    """
    if prompt.image_descriptors:
        from flash.content.multimodal import (
            image_descriptors_to_data_uris,
            image_teacher_prompt_messages,
        )

        # decode ONCE per episode rather than per turn: the descriptors are identical across turns
        # and decoding is the expensive part of building the request.
        teacher_images = image_descriptors_to_data_uris(
            prompt.image_descriptors,
            prompt.package_root,
        )
        return teacher.score_many_multimodal(
            [
                (
                    _multiturn_teacher_messages(
                        # the session stores the STUDENT messages, which carry raw image blocks and
                        # no placeholders; the teacher route needs the rendered form. rendering per
                        # turn also re-asserts the count: the renderer raises unless the context
                        # holds exactly one placeholder per descriptor, so a future change that
                        # trimmed the image-bearing turns out of the history fails loudly here
                        # instead of silently pairing a URI with the wrong turn.
                        image_teacher_prompt_messages(
                            turn["context_messages"], len(prompt.image_descriptors)
                        ),
                        thinking_prefill,
                    ),
                    turn["completion_text"],
                    teacher_images,
                    bool(thinking_prefill),
                )
                for turn in turns
            ]
        )
    return teacher.score_many(
        [
            (
                _teacher_prompt_text(turn["context_messages"], thinking_prefill),
                turn["completion_text"],
            )
            for turn in turns
        ]
    )


def _multiturn_teacher_messages(context_messages, thinking_prefill: str) -> list[dict]:
    """This turn's context as chat messages, with the thinking prefill continuing it if present.

    The text route renders the same thing as a flat string via `_teacher_prompt_text`; the image
    route needs the list form so `_chat_messages` can split each message on its image placeholders.
    """
    messages = [dict(message) for message in context_messages]
    if thinking_prefill:
        messages.append({"role": "assistant", "content": thinking_prefill})
    return messages
