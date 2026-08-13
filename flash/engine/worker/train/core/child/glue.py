"""child-side multi-turn rollout primitives shared by the OPD and GRPO verl agent loops.

stdlib only. no verl import and no flash import, so the parent can copy this file into a verl
child's workdir the same way it copies the loop modules themselves.

everything here is algorithm-neutral: turning an environment reply into the exact tokens the chat
template would have produced, deciding whether an assistant turn terminated or was truncated, and
the two small accounting helpers the loops share. what is NOT here is either loop's body -- OPD
distils each turn against a teacher and returns one output per turn, GRPO scores the episode and
returns one output per episode, and those differences are the point.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any
from uuid import uuid4

_ALLOWED_MESSAGE_KEYS = frozenset({"role", "content"})
_PROBE_PREFIX = "flash-env-glue-probe"
# the media block types verl's own dataset parser looks for when it decides whether a prompt has
# anything to extract (rl_dataset.py: `item.get("type") in {"image", "video"}`). duplicated here as
# literals rather than imported from flash.content.multimodal because this module is copied into the
# verl child, where flash is not importable -- see the module docstring.
_MEDIA_BLOCK_TYPES = frozenset({"image", "video", "audio"})


def normalize_token_ids(value) -> list[int]:
    """normalize tokenizer outputs to one flat list of integer token ids."""
    if isinstance(value, dict):
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if value and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, list):
        raise TypeError("tokenizer output must be list-like")
    return [int(token_id.item() if hasattr(token_id, "item") else token_id) for token_id in value]


def content_block_text(content, *, source: str, position: int) -> str:
    """flatten one openai-style content block list to the text the transcript represents.

    an image-bearing prompt does NOT reach the child as a string. verl's RLHFDataset rewrites the
    parquet's string content into blocks -- it splits on the `<image>` placeholder and replaces that
    segment with an image block (rl_dataset.py `_build_messages`) -- so `raw_prompt` arrives as
    [{"type": "image", ...}, {"type": "text", "text": ...}] for exactly the multimodal rows flash
    writes. validating that shape as text-only would reject every image episode at turn one.

    the media blocks are dropped rather than rendered here: they are carried separately, as decoded
    pixels in `multi_modal_data`, and the prompt ids come from the chat template applied to the
    ORIGINAL block list. this text view exists for the transcript and the inter-turn glue, which are
    token-level text operations. a block whose type is neither text nor known media is an error
    rather than a silent drop -- it would mean the model saw something this transcript cannot show.
    """
    parts = []
    for block in content:
        if not isinstance(block, dict):
            raise ValueError(
                f"{source} message {position} has a content block that is not an object"
            )
        block_type = block.get("type")
        if block_type in _MEDIA_BLOCK_TYPES:
            continue
        if block_type != "text":
            raise ValueError(
                f"{source} message {position} has an unsupported content block type "
                f"{block_type!r}; a multi-turn transcript can represent text and media blocks only"
            )
        text = block.get("text")
        if not isinstance(text, str):
            raise ValueError(f"{source} message {position} has a text block without text")
        parts.append(text)
    return "".join(parts)


def validate_transcript_messages(
    messages: list[dict], *, source: str, allow_content_blocks: bool = False
) -> list[dict]:
    """require the exact role/content transcript shape the child rollout loops can represent.

    ``allow_content_blocks`` flattens openai-style block lists to their text instead of rejecting
    them. it is OFF by default and opted into only by a caller that has already extracted the media
    out of the ORIGINAL blocks, because dropping to text is lossless only once the pixels are held
    somewhere else. a caller that has not done that extraction must keep raising: silently training
    on a caption whose image was discarded is the failure this default exists to prevent.
    """
    if not isinstance(messages, list):
        raise ValueError(f"{source} messages must be a list")
    normalized = []
    for position, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{source} message {position} must be an object")
        extras = sorted(
            key
            for key, value in message.items()
            if key not in _ALLOWED_MESSAGE_KEYS and value is not None
        )
        if extras:
            raise ValueError(
                f"{source} message {position} carries unsupported transcript metadata {extras}; "
                "tool names, call ids, tool calls, and other message fields cannot be represented "
                "in a role/content multi-turn transcript"
            )
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"{source} message {position} has an invalid role")
        if allow_content_blocks and isinstance(content, list):
            # a multimodal prompt arrives as content BLOCKS, not a string; flatten to the text this
            # transcript can represent. the media rides separately as decoded pixels.
            content = content_block_text(content, source=source, position=position)
        elif not isinstance(content, str):
            raise ValueError(f"{source} message {position} content must be text for multi-turn")
        normalized.append({"role": role, "content": content})
    return normalized


def _tokenize_text(tokenizer, text: str) -> list[int]:
    return normalize_token_ids(tokenizer(text, add_special_tokens=False))


def _unique_glue_probe(messages: list[dict]) -> str:
    contents = [message["content"] for message in messages]
    while True:
        probe = f"{_PROBE_PREFIX}-{uuid4().hex}"
        if all(probe not in content for content in contents):
            return probe


def dedup_seam_terminator(response_ids: list[int], glue_ids: list[int]) -> list[int]:
    """keep the sampled terminator and drop the duplicate leading glue copy."""
    if response_ids and glue_ids and response_ids[-1] == glue_ids[0]:
        return glue_ids[1:]
    return glue_ids


class EnvGlueTokenizer:
    """derive exact inter-turn glue without re-rendering accepted transcript history."""

    def __init__(self, tokenizer, *, thinking: bool, cache_size: int = 8192) -> None:
        self.tokenizer = tokenizer
        self.thinking = bool(thinking)
        self.cache_size = int(cache_size)
        self.cache: dict[str, list[int]] = {}

    def __call__(self, env_messages: list[dict]) -> list[int]:
        messages = validate_transcript_messages(env_messages, source="environment reply")
        key = json.dumps(messages, sort_keys=True, separators=(",", ":"))
        cached = self.cache.get(key)
        if cached is not None:
            return list(cached)
        probe = _unique_glue_probe(messages)
        text = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": probe}, *messages],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=self.thinking,
        )
        first = text.find(probe)
        if first == -1 or text.find(probe, first + len(probe)) != -1:
            raise ValueError(
                "multi-turn rollout could not uniquely locate the assistant-content probe in the "
                "chat template; exact inter-turn glue cannot be recovered"
            )
        glue_ids = _tokenize_text(self.tokenizer, text[first + len(probe) :])
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = list(glue_ids)
        return glue_ids


def validate_glue_template(tokenizer, *, thinking: bool) -> None:
    """fail before rollout when assistant content cannot round-trip through the template."""
    EnvGlueTokenizer(tokenizer, thinking=thinking)(
        [{"role": "user", "content": "flash multi-turn glue validation"}]
    )


def trim_trailing_stop(
    tokenizer, response_ids, stop_text: str, stop_sequences
) -> tuple[list[int], str]:
    """Drop a trailing stop delimiter from BOTH the sampled ids and the decoded text (token-level).

    HF ``stop_strings`` halts only AFTER the delimiter text is emitted, so a run with e.g. ``[train]
    stop_sequences=["</answer>"]`` would otherwise score/distil the delimiter the user asked to stop
    at.
    """
    ids = [int(token_id) for token_id in response_ids]
    # Pick the LONGEST configured stop that is a trailing match (the earliest stop boundary in the
    # text). Overlapping delimiters like ["\n", "\n\n"] would otherwise trim only the first-listed
    # shorter suffix off a "\n\n" tail, leaving one newline for the teacher to score/distil; taking
    # the longest match removes the whole delimiter in one shot regardless of config order.
    stop = max(
        (value for value in stop_sequences if value and stop_text.endswith(value)),
        key=len,
        default="",
    )
    if not stop:
        return ids, tokenizer.decode(ids, skip_special_tokens=True)
    keep_length = len(stop_text) - len(stop)
    # Locate the kept prefix by scanning from the END on the SAME (special-tokens-included) decode
    # keep_length is measured in, so a special-token delimiter's id(s) are dropped correctly. Scanning
    # from the END is O(dropped * completion): the delimiter is short so only a handful of trailing
    # tokens drop; decoding growing prefixes from the START would be O(completion^2).
    kept = len(ids)
    while kept > 0 and len(tokenizer.decode(ids[:kept], skip_special_tokens=False)) > keep_length:
        kept -= 1
    # Return the kept ids decoded WITHOUT special tokens -- the teacher/alignment text. Decoding the
    # kept ids (not slicing stop_text at keep_length) keeps the teacher-scored text and the student
    # ids identical even when the stop starts INSIDE the final sampled token (that token decodes to
    # e.g. "B</answer>"): the whole token is excluded from `kept`, so the fused "B" is dropped rather
    # than left dangling to desync the loss alignment / token count.
    return ids[:kept], tokenizer.decode(ids[:kept], skip_special_tokens=True)


def prepare_assistant_turn(
    tokenizer,
    token_ids: list[int],
    *,
    stop_reason: str | None,
    max_tokens: int,
    eos_token_ids: frozenset[int],
    stop_sequences: tuple[str, ...],
) -> dict[str, Any]:
    """apply the shared termination, stop trimming, empty, and replacement-char gates."""
    raw_ids = [int(token_id) for token_id in token_ids]
    stop_text = tokenizer.decode(raw_ids, skip_special_tokens=False)
    ended_by_eos = bool(eos_token_ids and not eos_token_ids.isdisjoint(raw_ids))
    ended_by_stop = any(value and stop_text.endswith(value) for value in stop_sequences)
    ended_before_cap = stop_reason == "completed" and len(raw_ids) < int(max_tokens)
    terminated = ended_by_eos or ended_by_stop or ended_before_cap
    if stop_reason == "aborted":
        # an aborted rollout is truncated REGARDLESS of what signals appear inside the sampled
        # ids: the validator requires termination == "truncated" for truncated turns, so the
        # label must not leak eos/stop from partial content.
        terminated = False
        ended_by_eos = ended_by_stop = ended_before_cap = False
    termination = (
        "eos"
        if ended_by_eos
        else "stop"
        if ended_by_stop
        else "accepted_stop"
        if ended_before_cap
        else "truncated"
    )
    response_ids = raw_ids
    completion_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    # trim a trailing stop suffix whenever one ended the text — including when an EOS is ALSO
    # present (the label prefers eos, but the bridge's eos validation requires the trimmed span
    # to match; leaving the stop text in place desyncs response_ids from raw_response_ids).
    if terminated and ended_by_stop and not ended_by_eos:
        response_ids, completion_text = trim_trailing_stop(
            tokenizer, response_ids, stop_text, stop_sequences
        )
    if not terminated:
        skip_reason = "truncated_rollout"
        truncated = True
    elif not completion_text.strip():
        skip_reason = "empty_completion"
        truncated = False
    elif "�" in completion_text:
        skip_reason = "replacement_char"
        truncated = False
    else:
        skip_reason = ""
        truncated = False
    return {
        "raw_response_ids": raw_ids,
        "response_ids": response_ids,
        "completion_text": completion_text,
        "termination": termination,
        "stop_reason": stop_reason,
        "max_tokens": int(max_tokens),
        "truncated": truncated,
        "skip_reason": skip_reason,
    }


def turn_is_unusable(turn: dict[str, Any]) -> bool:
    """Whether the environment neither saw nor scored this turn.

    The bridge returns before ``record_model_turn`` for a truncated, empty, or replacement-char
    turn, so it never enters environment state and earns no reward. Both consequences follow from
    this one predicate: the turn's tokens must stay out of the loss (a zeroed ``response_mask``,
    the same way environment glue is excluded), and it must NOT take a turn span (``score_rollouts``
    rejects a span/reward count mismatch and drops the whole group to episode credit).
    """
    return bool(turn["truncated"] or turn["skip_reason"])


def sum_preemptions(current: int, value: int | None) -> int:
    if value is None:
        return current
    if current < 0:
        return int(value)
    return current + int(value)


async def run_executor_call(loop, callback):
    """finish a bridge request before propagating task cancellation."""
    task = asyncio.ensure_future(loop.run_in_executor(None, callback))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        with contextlib.suppress(Exception):
            await asyncio.shield(task)
        raise
