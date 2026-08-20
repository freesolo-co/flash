"""Response shaping for the serving front door: reasoning-block splitting (streaming and
non-streaming) and OpenAI response_format -> structured-outputs translation.

Split out of router.py, which is a single large FastAPI app builder; these helpers are pure
functions over strings and dicts, with no engine or app state, so they are unit-testable on their
own and keep the router focused on routing.

Do NOT add `from __future__ import annotations`: router.py re-exports these next to FastAPI
handlers whose closure-local body models are used as annotations, and the future import turns
those into unresolvable strings -> silent 422.
"""

import re
from typing import Any

from flash.serving.src.structured_outputs import StructuredOutputsError

_THINK_CLOSE = "</think>"

def _split_reasoning(text: str, thinking: bool) -> tuple[str | None, str]:
    """Split a completion into ``(reasoning_content, content)`` for the OpenAI surface.

    The chat template opens the reasoning block in the PROMPT, so a thinking completion begins
    inside the block and its text is ``<reasoning></think><answer>`` -- a closing tag with no
    opener. Returned verbatim, the caller sees a stray ``</think>`` glued to the answer and an
    empty ``reasoning_content``, and every downstream parser that keys off the opening tag fails.

    ``thinking`` is the mode the generation was RENDERED with, reported by the engine. It is
    required rather than inferred: a non-thinking completion that merely quotes ``</think>`` must
    not be silently torn in half.
    """
    if not thinking:
        return None, text
    head, sep, tail = text.partition(_THINK_CLOSE)
    if not sep:
        # thinking mode, but the block never closed (hit max_tokens mid-reasoning). all of it is
        # reasoning; there is no answer yet. an empty content is the honest report.
        return head, ""
    return head, tail


def _partial_tag_suffix(text: str) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of the closing tag."""
    for size in range(min(len(text), len(_THINK_CLOSE) - 1), 0, -1):
        if _THINK_CLOSE.startswith(text[-size:]):
            return size
    return 0


class _ReasoningStreamSplitter:
    """Routes streamed deltas to ``reasoning_content`` until the reasoning block closes.

    Streaming cannot reuse :func:`_split_reasoning`: the closing tag arrives token by token and
    can straddle a chunk boundary, so a trailing partial match is held back rather than emitted
    as reasoning text it might turn out not to be.
    """

    def __init__(self, thinking: bool) -> None:
        self._closed = not thinking
        self._pending = ""

    def feed(self, text: str) -> tuple[str, str]:
        """Split one delta into ``(reasoning_delta, content_delta)``."""
        if self._closed:
            return "", text
        buffer = self._pending + text
        head, sep, tail = buffer.partition(_THINK_CLOSE)
        if sep:
            self._closed = True
            self._pending = ""
            return head, tail
        hold = _partial_tag_suffix(buffer)
        split = len(buffer) - hold
        self._pending = buffer[split:]
        return buffer[:split], ""

    def flush(self) -> str:
        """Held-back text at end of stream.

        Reaching here means the block never closed (the generation stopped mid-reasoning), so the
        remainder is reasoning with no answer -- the same call :func:`_split_reasoning` makes.
        """
        pending, self._pending = self._pending, ""
        return pending


def _usage_block(prompt_tokens: int, completion_tokens: int, cached_tokens: Any) -> dict[str, Any]:
    """OpenAI-style ``usage`` object for a completion response.

    Mirrors the OpenAI schema, including ``prompt_tokens_details.cached_tokens`` (the
    prefix-cached subset of the prompt) so a client can see how many prompt tokens were served
    from cache — the same count the backend bills at a discount. ``cached_tokens`` is clamped to
    ``prompt_tokens`` and the details block is omitted entirely when there were no cached tokens.
    """
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    try:
        cached = max(0, min(int(cached_tokens or 0), prompt_tokens))
    except (TypeError, ValueError):
        cached = 0
    if cached > 0:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return usage


def _openai_structured_outputs(payload: dict[str, Any]) -> Any:
    """The structured-outputs spec carried by an OpenAI-style chat payload, or None.

    Our first-class extension (``structured_outputs``) wins; failing that,
    the OpenAI-standard ``response_format`` is accepted — but ONLY here, at the OpenAI-compatible
    ``/v1/chat/completions`` boundary — and translated to our canonical spec, so an OpenAI-SDK
    client gets structured output with no code change. The result is returned RAW so
    GenerateRequest's validator normalizes it (consistent 422s across every entry point). Checked
    with ``is not None``, not truthiness: ``{}`` / ``false`` are explicit "unconstrained" markers,
    distinct from an absent field.
    """
    if (spec := payload.get("structured_outputs")) is not None:
        return spec
    response_format = payload.get("response_format")
    if response_format is not None:
        return _response_format_to_spec(response_format)
    return None


def _response_format_to_spec(response_format: Any) -> Any:
    """Translate an OpenAI ``response_format`` object to our canonical structured-outputs spec.

    Kept at this endpoint only (the core normalizer stays strict-canonical). ``{"type": "text"}``
    -> ``{}`` (explicit "unconstrained", overriding any adapter default); ``{"type":
    "json_object"}`` -> ``{"json_object": True}``; ``{"type": "json_schema", "json_schema":
    {"schema": {...}}}`` (or the flattened ``{"type": "json_schema", "schema": {...}}``) ->
    ``{"json": schema}``. An unrecognized ``type`` is rejected here with a clean 422 (rather than
    letting the greedy normalizer silently wrap it as a schema); a non-dict is returned as-is for
    the GenerateRequest validator to normalize.
    """
    if not isinstance(response_format, dict):
        return response_format
    rf_type = response_format.get("type")
    if rf_type == "text":
        return {}
    if rf_type == "json_object":
        return {"json_object": True}
    if rf_type == "json_schema":
        wrapper = response_format.get("json_schema")
        schema = (
            wrapper.get("schema") if isinstance(wrapper, dict) else response_format.get("schema")
        )
        if schema is None:
            raise StructuredOutputsError(
                'response_format {"type": "json_schema"} requires a schema '
                "(json_schema.schema = {...} or schema = {...})"
            )
        return {"json": schema}
    # OpenAI response_format has exactly these three types. Reject anything else here (a typo'd type
    # or a bare schema) with a clean 422, rather than letting the greedy normalizer silently wrap the
    # whole object as a JSON schema; to pass a raw schema, use the structured_outputs field instead.
    raise StructuredOutputsError(
        f'unsupported response_format type {rf_type!r}; use "text", "json_object", or '
        '"json_schema" (or pass a raw schema via the structured_outputs field)'
    )


