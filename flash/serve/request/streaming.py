"""Server-sent-event decoding for streamed chat completions.

`chat_stream` and the SSE line decoder it drives are separated from `flash.serve.deployment.deploy`
to keep that module under the file-size limit.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from flash._internal.openai_sse import (
    DeltaEvent,
    ErrorEvent,
    OpenAISSEError,
    iter_openai_sse_events,
    sse_data_is_terminal,
)
from flash.client.http import ClientError
from flash.serve.request.thinking import (
    _TAG_CLOSE,
    _TAG_OPEN,
    _balanced_thinking_content,
    _inline_reasoning_block,
    _is_only_retained_delimiter,
    _is_terminal_reasoning_repeat,
    _strip_retained_close,
)
from flash.serve.request.thinking import (
    _find_delimiter as _default_find_delimiter,
)
from flash.serve.request.transport import OpenAIStreamResponse


def _openai_stream_content(
    lines: Iterator[str],
    *,
    thinking: bool,
    find_delimiter: Callable[[str, int], int] = _default_find_delimiter,
) -> Iterator[str]:
    # reasoning arrives on its own delta field (see _balanced_thinking_content). re-open the block
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
    try:
        events = iter_openai_sse_events(f"{line}\n" for line in lines)
        for event in events:
            if isinstance(event, ErrorEvent):
                if reasoning_open:
                    reasoning_open = False
                    yield _TAG_CLOSE
                raise ClientError(event.message)
            if not isinstance(event, DeltaEvent):
                continue
            raw_reasoning = event.reasoning_content
            # `thinking` gates this as it gates `_balanced_thinking_content`, and for the same
            # reason: this path also backs the public chat route. tested by type, not falsiness --
            # a model that closed its reasoning immediately streams `reasoning_content: ""`, which
            # still needs a pair.
            if thinking and raw_reasoning is not None:
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
                            "" if _is_only_retained_delimiter(closing, reasoning_text) else closing
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
                    yield _TAG_OPEN
                if raw_reasoning:
                    reasoning_text += raw_reasoning
                    yield raw_reasoning
            content = event.content or ""
            if content:
                if reasoning_open:
                    reasoning_open = False
                    reasoning_done = True
                    held = None
                    yield _TAG_CLOSE
                    closing = ""
                    closing_scanned = 0
                if closing is not None:
                    closing += content
                    answer, closing_scanned = _strip_retained_close(
                        closing,
                        reasoning_text,
                        closing_scanned,
                        find_delimiter=find_delimiter,
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
                    close = find_delimiter(held_text, held_scanned)
                    if close < 0:
                        held_scanned = max(0, len(held_text) - (len(_TAG_CLOSE) - 1))
                        continue
                    joined = held_text
                    held = None
                    held_text = ""
                    inline = _inline_reasoning_block(joined)
                    if inline is not None and inline[1] == close:
                        # already balanced: the legacy stream carried its own opener, so re-opening
                        # would nest one block inside another. the pair must be the one that
                        # MATCHED this close, or an answer-side pair further down would disable the
                        # re-open for the very shape it exists for.
                        yield joined
                        continue
                    yield (
                        f"{_TAG_OPEN}{joined[:close]}{_TAG_CLOSE}"
                        f"{joined[close + len(_TAG_CLOSE) :]}"
                    )
                    continue
                yield content
    except OpenAISSEError as exc:
        if reasoning_open:
            reasoning_open = False
            yield _TAG_CLOSE
        raise ClientError(str(exc)) from exc
    if reasoning_open:
        # generation stopped inside the block (a length cap, usually). still close it: an
        # unbalanced opener is the same defect as the unbalanced closer, mirrored.
        yield _TAG_CLOSE
    # the buffer holds the block's own retained close and nothing else, in either the bare or the
    # opener-carrying form. only decidable at end of stream, since nothing more can arrive. any
    # other buffer was answer text after all, covering the answer that IS the delimiter.
    if closing and not _is_terminal_reasoning_repeat(closing, reasoning_text):
        yield closing
    if held:
        # no delimiter ever arrived, so nothing marked a reasoning phase: a plain answer. release
        # it as sent rather than wrapping it, which would label a valid answer as reasoning.
        yield from held


def _next_sse_frame_end(buffered: bytearray, *, final: bool = False) -> int | None:
    line_start = 0
    index = 0
    while index < len(buffered):
        value = buffered[index]
        if value == 10:
            line_end = index + 1
        elif value == 13:
            if index + 1 == len(buffered) and not final:
                return None
            line_end = index + (2 if buffered[index + 1 : index + 2] == b"\n" else 1)
        else:
            index += 1
            continue
        if index == line_start:
            return line_end
        line_start = line_end
        index = line_end
    return None


def _complete_sse_frames(chunks: Iterator[bytes]) -> Iterator[bytes]:
    """yield only complete sse frames while preserving every upstream byte."""

    buffered = bytearray()
    first_frame = True
    for chunk in chunks:
        buffered.extend(chunk)
        while end := _next_sse_frame_end(buffered):
            frame = bytes(buffered[:end])
            del buffered[:end]
            terminal_data = frame.removeprefix(b"\xef\xbb\xbf") if first_frame else frame
            first_frame = False
            terminal = sse_data_is_terminal(terminal_data)
            yield frame
            if terminal:
                return
    while end := _next_sse_frame_end(buffered, final=True):
        frame = bytes(buffered[:end])
        del buffered[:end]
        terminal_data = frame.removeprefix(b"\xef\xbb\xbf") if first_frame else frame
        first_frame = False
        terminal = sse_data_is_terminal(terminal_data)
        yield frame
        if terminal:
            return
    if buffered:
        raise ClientError("chat stream ended with an incomplete server-sent event frame")
    raise ClientError("chat stream ended before the terminal [DONE] event")


def _streamed_body(
    upstream: OpenAIStreamResponse,
    *,
    thinking: bool,
    find_delimiter: Callable[[str, int], int],
) -> Iterator[str]:
    """decode one validated response while retaining its lifetime through exhaustion."""

    response = upstream.response
    try:
        yield ""
        content_type = response.headers.get("content-type", "")
        media_type = content_type.partition(";")[0].strip().lower()
        if media_type == "application/json":
            # client.stream() leaves body unread; must call read before json.
            response.read()
            payload = response.json()
            content = _balanced_thinking_content(
                (payload.get("choices") or [{}])[0].get("message") or {}, thinking=thinking
            )
            if content:
                yield str(content)
            return
        yield from _openai_stream_content(
            response.iter_lines(),
            thinking=thinking,
            find_delimiter=find_delimiter,
        )
    finally:
        # runs on normal exhaustion, a midstream error, or downstream disconnect.
        upstream.close()
