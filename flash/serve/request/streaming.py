"""Server-sent-event decoding for streamed chat completions.

`chat_stream` and the SSE line decoder it drives are separated from `flash.serve.deploy`
to keep that module under the file-size limit.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator

import flash.serve.request.thinking as thinking_support
import flash.serve.request.transport as transport
from flash.client.http import ClientError


def _raise_for_stream_error(chunk: dict) -> None:
    choices = chunk.get("choices") or []
    failed = any(
        isinstance(choice, dict) and choice.get("finish_reason") == "error" for choice in choices
    )
    error = chunk.get("error")
    if not failed and not isinstance(error, dict):
        return
    message = error.get("message") if isinstance(error, dict) else None
    raise ClientError(str(message or "chat stream ended with an engine error"))


def _openai_stream_content(lines: Iterator[str], *, thinking: bool) -> Iterator[str]:
    # reasoning arrives on its own delta field (see thinking_support._balanced_thinking_content). re-open the block
    # around it and close it at the answer boundary, so the streamed text matches the balanced
    # string the non-streaming path returns.
    reasoning_open = False
    # whether a block was ever emitted, which is not the same as whether one is open now: a backend
    # serializing `reasoning_content: ""` on every delta must not open a second block.
    reasoning_done = False
    # buffered content after the block closed, while a retained delimiter may still be arriving.
    closing: str | None = None
    # how far into `closing` the delimiter search has already looked.
    closing_scanned = 0
    # what arrived on the reasoning field, kept because a compatibility build repeats it inline
    # ahead of the retained close, and recognising that repeat is how the block's own delimiter is
    # told from answer text that merely mentions the tag.
    reasoning_text = ""
    # legacy inline-thinking streams begin mid-block because the opener is in the prompt; hold until
    # </think> proves the phase, unless non-thinking mode or a reasoning delta proves split output.
    # preserve original delta boundaries while maintaining a joined string for linear-time search.
    held: list[str] | None = [] if thinking else None
    # `held` joined, kept in step with it. never bound to a second name while being appended to,
    # since that blocks the in-place concatenation and restores the copy this avoids.
    held_text = ""
    # how far into `held_text` the delimiter search has already looked.
    held_scanned = 0
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            break
        if not data:
            continue
        chunk = json.loads(data)
        _raise_for_stream_error(chunk)
        for choice in chunk.get("choices") or []:
            delta = (choice.get("delta") or {}) if isinstance(choice, dict) else {}
            raw_reasoning = delta.get("reasoning_content")
            # `thinking` gates this as it gates `thinking_support._balanced_thinking_content`, and for the same
            # reason: this path also backs the public chat route. tested by type, not falsiness --
            # a model that closed its reasoning immediately streams `reasoning_content: ""`, which
            # still needs a pair.
            if thinking and isinstance(raw_reasoning, str):
                if held:
                    # the backend splits after all, so whatever arrived first was answer text and
                    # not an unopened block. release it untouched, delta by delta as it arrived.
                    yield from held
                held = None
                held_text = ""
                # an empty field after the block closed cannot reopen it: it carries no text to
                # label. a non-empty one still opens a block, or its reasoning streams as answer.
                if not reasoning_open and not (reasoning_done and not raw_reasoning):
                    if closing is not None:
                        # a new block opens here, so no further content can join the buffer and it
                        # is decidable now, exactly as at end of stream.
                        settled = (
                            ""
                            if thinking_support._is_only_retained_delimiter(closing, reasoning_text)
                            else closing
                        )
                        closing = None
                        if settled:
                            yield settled
                    reasoning_open = True
                    # the text belongs to the block that is opening, not to every block so far.
                    # accumulating across blocks made the duplicate checks below compare a later
                    # block's retained close against both blocks' text, so they stopped recognising
                    # it and streamed the delimiter a second time.
                    reasoning_text = ""
                    yield thinking_support._TAG_OPEN
                if raw_reasoning:
                    reasoning_text += raw_reasoning
                    yield raw_reasoning
            content = delta.get("content") or ""
            if content:
                content = str(content)
                if reasoning_open:
                    reasoning_open = False
                    reasoning_done = True
                    held = None
                    yield thinking_support._TAG_CLOSE
                    closing = ""
                    closing_scanned = 0
                if closing is not None:
                    closing += content
                    answer, closing_scanned = thinking_support._strip_retained_close(
                        closing, reasoning_text, closing_scanned
                    )
                    if answer is None:
                        # the tag may still be completing across deltas. keep buffering.
                        continue
                    closing = None
                    if answer:
                        yield answer
                    continue
                if held is not None:
                    held.append(content)
                    held_text += content
                    # resume where the last scan stopped: rescanning the whole buffer per delta is
                    # quadratic in the completion length, and token-sized deltas make that the
                    # common case. only the last few characters can still be a partial tag.
                    close = thinking_support._find_delimiter(held_text, held_scanned)
                    if close < 0:
                        held_scanned = max(
                            0, len(held_text) - (len(thinking_support._TAG_CLOSE) - 1)
                        )
                        continue
                    joined = held_text
                    held = None
                    held_text = ""
                    inline = thinking_support._inline_reasoning_block(joined)
                    if inline is not None and inline[1] == close:
                        # already balanced: the legacy stream carried its own opener, so re-opening
                        # would nest one block inside another. the pair must be the one that
                        # MATCHED this close, or an answer-side pair further down would disable the
                        # re-open for the very shape it exists for.
                        yield joined
                        continue
                    yield (
                        f"{thinking_support._TAG_OPEN}{joined[:close]}{thinking_support._TAG_CLOSE}"
                        f"{joined[close + len(thinking_support._TAG_CLOSE) :]}"
                    )
                    continue
                yield content
    if reasoning_open:
        # generation stopped inside the block (a length cap, usually). still close it: an
        # unbalanced opener is the same defect as the unbalanced closer, mirrored.
        yield thinking_support._TAG_CLOSE
    # the buffer holds the block's own retained close and nothing else, in either the bare or the
    # opener-carrying form. only decidable at end of stream, since nothing more can arrive. any
    # other buffer was answer text after all, covering the answer that IS the delimiter.
    if closing and not thinking_support._is_terminal_reasoning_repeat(closing, reasoning_text):
        yield closing
    if held:
        # no delimiter ever arrived, so nothing marked a reasoning phase: a plain answer. release
        # it as sent rather than wrapping it, which would label a valid answer as reasoning.
        yield from held


def chat_stream(
    run_id: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 512,
    thinking: bool = False,
    stop: list[str] | None = None,
) -> Iterator[str]:
    """Yield text deltas from the freesolo OpenAI-compatible streaming endpoint.

    ``stop`` carries the run's own stop sequences, as in ``chat``.

    Not a generator function: the upstream request is sent and its status validated here, at
    call time. The caller hands the returned iterator to a ``StreamingResponse``, and a
    generator would defer the request (and ``raise_for_status``) until iteration, after the
    200 and headers had been flushed, so an upstream 4xx/5xx could only surface as a
    truncated success.
    """
    base = transport.serving_openai_base_url()
    body = {
        "model": run_id,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
        "stream": True,
    }
    if stop:
        body["stop"] = [str(value) for value in stop]
    ctx = transport._stream_http_client().stream(
        "POST",
        f"{base}/chat/completions",
        json=body,
        headers=transport._internal_key_header(),
        timeout=30 * 60.0,
    )
    resp = ctx.__enter__()
    try:
        resp.raise_for_status()
    except BaseException:
        ctx.__exit__(*sys.exc_info())
        raise
    stream = _streamed_body(ctx, resp, thinking=thinking)
    # advance to the priming yield before handing the generator out: close() on a generator
    # that was never started skips its finally block, and that block is what releases the
    # upstream connection when the caller closes without iterating.
    next(stream)
    return stream


def _streamed_body(ctx, resp, *, thinking: bool) -> Iterator[str]:
    """Decode the already-validated streaming response, closing it on every exit path.

    ``ctx`` is the entered ``client.stream`` context manager; exiting it closes ``resp``.
    the leading empty yield is a priming value consumed by ``chat_stream`` so the generator
    is running before it is handed out (see there).
    """
    try:
        yield ""
        if "application/json" in resp.headers.get("content-type", ""):
            # client.stream() leaves body unread; must call resp.read() before .json().
            resp.read()
            payload = resp.json()
            content = thinking_support._balanced_thinking_content(
                (payload.get("choices") or [{}])[0].get("message") or {}, thinking=thinking
            )
            if content:
                yield str(content)
            return
        yield from _openai_stream_content(resp.iter_lines(), thinking=thinking)
    finally:
        # runs on normal exhaustion, on a mid-stream error, and on close() from a client
        # disconnect. a mid-stream error keeps propagating, which makes the server abort the
        # chunked response, so the client sees a truncated-transfer error instead of a clean
        # eof it would mistake for a finished answer.
        ctx.__exit__(None, None, None)
