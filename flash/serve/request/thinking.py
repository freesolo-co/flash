"""Reasoning-delimiter handling for served chat responses.

Thinking models emit their reasoning inside `<think>`/`</think>`, and the delimiter can be
split across streamed chunks, duplicated, or left unbalanced by a truncated generation.
Parsing that back into a clean payload is pure string work with no serving dependencies, so
it lives here rather than in `flash.serve.deployment.deploy`.

Split out to keep that module under the file-size limit. The tag constants are defined here
and consumed directly by the request paths.
"""

from __future__ import annotations

from collections.abc import Callable

_TAG_CLOSE = "</think>"
_TAG_OPEN = "<think>"


def _inline_reasoning_block(content: str) -> tuple[int, int] | None:
    """Span of a balanced ``<think>...</think>`` in ``content``, or ``None``.

    Order matters: an answer may contain the literal ``<think>`` (a ``choice`` constraint, or JSON
    quoting it), so the close must be found after the open rather than anywhere in the string.
    """
    opened = content.find(_TAG_OPEN)
    if opened < 0:
        return None
    closed = content.find(_TAG_CLOSE, opened + len(_TAG_OPEN))
    if closed < 0:
        return None
    return opened, closed


def _is_sampled_delimiter(prefix: str, reasoning: str) -> bool:
    """Whether the text before an inline ``</think>`` marks it as the block's own close.

    Two shapes prove it: nothing before it (the opener went into the prompt), or text duplicating
    ``reasoning_content`` (a compatibility build emitting the reasoning both inline and on the
    field), with or without its own opener. Any other prefix is answer text mentioning the tag.

    Compared stripped, since the model often samples the delimiter on its own line.
    """
    body = prefix.strip()
    if body.startswith(_TAG_OPEN):
        # an explicit opener means a whole block precedes the close, so it is the repeat only if
        # its body is the reasoning. an empty body is the answer's own pair, not the delimiter.
        return body[len(_TAG_OPEN) :].strip() == reasoning.strip()
    return body in ("", reasoning.strip())


def _find_delimiter(buffer: str, start: int) -> int:
    """Index of the sampled close tag at or after ``start``, or ``-1``.

    A named seam for one `str.find`, so the streamed hold path can be measured for scan cost
    without instrumenting the buffer (deltas are coerced with `str()` before being appended).
    """
    return buffer.find(_TAG_CLOSE, start)


def _retained_delimiter_end(content: str, close: int, reasoning: str) -> int | None:
    """Index just past a retained sampled ``</think>``, or ``None`` if it is the answer itself.

    A compatibility build's retained delimiter and an answer that IS the delimiter are identical on
    the wire, so no prefix test separates them. What does is whether stripping leaves anything: a
    retained delimiter is followed by the answer, an answer that is the tag by nothing.
    """
    if not _is_sampled_delimiter(content[:close], reasoning):
        return None
    end = close + len(_TAG_CLOSE)
    if content[end:].strip():
        return end
    return None


def _duplicates_reasoning(content: str, inline: tuple[int, int], reasoning: str) -> bool:
    """Whether this inline pair is ``reasoning_content`` emitted a second time inline.

    A compatibility build repeats the reasoning where a reasoning phase belongs, ahead of the
    answer, so a pair appearing after answer text is the answer quoting the tag instead. Bodies are
    compared stripped, since the repeat tends to carry the newline the model sampled after it.
    """
    opened, closed = inline
    if content[:opened].strip():
        return False
    return content[opened + len(_TAG_OPEN) : closed].strip() == reasoning.strip()


def _balanced_thinking_content(message: dict, *, thinking: bool) -> str:
    """Fold a split-out ``reasoning_content`` back into a balanced ``<think>...</think>`` block.

    Thinking templates put the opener in the prompt, so reopen split reasoning for content-only
    consumers. Only do this when the request's ``thinking`` flag is set.
    """
    content = str(message.get("content") or "")
    if not thinking:
        return content
    reasoning = message.get("reasoning_content")
    inline = _inline_reasoning_block(content)
    if not isinstance(reasoning, str):
        # tested by type, not falsiness: an explicitly empty string means the model closed its
        # block immediately, which still needs a pair, while absent means serving never split.
        close = content.find(_TAG_CLOSE)
        # the pair must be the one that matched the FIRST close, not merely present somewhere:
        # an unmatched close followed by an answer-side pair still needs re-opening.
        if inline is not None and inline[1] == close:
            return content
        # a legacy build predating the split leaves `reasoned</think>answer` inline with no field.
        # with nothing to contradict it, an unbalanced close can only be the sampled delimiter.
        if close >= 0:
            return f"{_TAG_OPEN}{content[:close]}{_TAG_CLOSE}{content[close + len(_TAG_CLOSE) :]}"
        # no close either: a plain answer. leave it rather than inventing a block around it.
        return content
    if inline is not None:
        if _duplicates_reasoning(content, inline, reasoning):
            return content
        opened, _ = inline
        close = content.find(_TAG_CLOSE)
        if close >= 0 and close < opened and _is_sampled_delimiter(content[:close], reasoning):
            # the retained close precedes the pair, so the pair belongs to the answer and this
            # close is the block's own. strip it and keep the answer whole.
            return f"{_TAG_OPEN}{reasoning}{_TAG_CLOSE}{content[close + len(_TAG_CLOSE) :]}"
        # a different pair: the answer itself contains a literal block. fold, so the reasoning
        # survives and the answer stays intact behind the delimiter.
        return f"{_TAG_OPEN}{reasoning}{_TAG_CLOSE}{content}"
    close = content.find(_TAG_CLOSE)
    if _is_terminal_reasoning_repeat(content, reasoning):
        # the repeat with nothing behind it, so there is no answer to keep. appending it would
        # hand the smoke `why</think>` as an answer and activate a deployment that answered
        # nothing; folding to the block alone lets the smoke reject it. same failure direction
        # the opener-carrying form already takes.
        return f"{_TAG_OPEN}{reasoning}{_TAG_CLOSE}"
    end = _retained_delimiter_end(content, close, reasoning) if close >= 0 else None
    if end is not None:
        return f"{_TAG_OPEN}{reasoning}{_TAG_CLOSE}{content[end:]}"
    # an empty reasoning string reaches here and still gets a pair: a thinking consumer splits the
    # answer on `</think>`, so a bare answer would read as no answer at all.
    return f"{_TAG_OPEN}{reasoning}{_TAG_CLOSE}{content}"


def _balance_thinking_payload(payload: object, *, thinking: bool) -> None:
    """Rewrite each choice's ``content`` in place so the returned payload is balanced."""
    if not thinking or not isinstance(payload, dict):
        return
    for choice in payload.get("choices") or []:
        message = choice.get("message") if isinstance(choice, dict) else None
        if isinstance(message, dict):
            message["content"] = _balanced_thinking_content(message, thinking=thinking)


def _delimiter_may_complete(text: str, reasoning: str) -> bool:
    """Whether ``text`` holds too little to rule out a sampled delimiter still arriving.

    The delimiter has two shapes on the wire (the bare tag, and the tag behind a repeat of
    ``reasoning_content``) and a backend may split either anywhere, so the question is not "is this
    the tag" but "could this still become one".
    """
    stripped = text.strip()
    if _TAG_CLOSE.startswith(stripped) or _TAG_OPEN.startswith(stripped):
        # a strict prefix of either tag, including the empty buffer.
        return True
    if stripped.startswith(_TAG_OPEN):
        # past the repeat's opener, so what remains is judged as the bare repeat is, below.
        stripped = stripped[len(_TAG_OPEN) :].strip()
    body = reasoning.strip()
    # an empty reasoning body is repeated as `<think></think>`, leaving nothing to match past the
    # opener, so what remains is tested against the close tag directly.
    if not body:
        return _TAG_CLOSE.startswith(stripped)
    if not body.startswith(stripped[: len(body)]):
        return False
    # the repeat is still arriving, or it landed whole and the tag behind it has not.
    return _TAG_CLOSE.startswith(stripped[len(body) :].strip())


def _strip_retained_close(
    text: str,
    reasoning: str,
    start: int = 0,
    *,
    find_delimiter: Callable[[str, int], int] = _find_delimiter,
) -> tuple[str | None, int]:
    """Drop a retained sampled ``</think>`` from the head of the post-reasoning content.

    Return ``None`` while a split delimiter may still be arriving. Resume from ``start`` to avoid
    rescanning the growing buffer; end-of-stream distinguishes a delayed answer from the tag itself.
    """
    close = find_delimiter(text, start)
    if close >= 0:
        end = _retained_delimiter_end(text, close, reasoning)
        if end is not None:
            return text[end:], close
        if _is_sampled_delimiter(text[:close], reasoning):
            return None, close
        # answer text that merely mentions the tag, so it stays put.
        return text, close
    # nothing before the last few characters can still become a tag.
    clean = max(0, len(text) - (len(_TAG_CLOSE) - 1))
    if _delimiter_may_complete(text, reasoning):
        return None, clean
    return text, clean


def _is_only_retained_delimiter(text: str, reasoning: str) -> bool:
    """Whether a settled buffer holds the block's own retained close and nothing else.

    Asked once no further content can join the buffer, which is the point at which the ambiguity
    `_strip_retained_close` buffers on becomes decidable.
    """
    close = text.find(_TAG_CLOSE)
    if close < 0 or not _is_sampled_delimiter(text[:close], reasoning):
        return False
    return not text[close + len(_TAG_CLOSE) :].strip()


def _is_terminal_reasoning_repeat(text: str, reasoning: str) -> bool:
    """Whether ``text`` is the reasoning repeated inline with nothing behind its close.

    Narrower than `_is_only_retained_delimiter` in requiring text before the close, with or
    without an opener: a delimiter with nothing at all ahead of it is the answer that IS the tag,
    and a checkpoint answering its grammar that way must still reach the smoke.
    """
    close = text.find(_TAG_CLOSE)
    if close < 0 or not text[:close].strip():
        return False
    return _is_only_retained_delimiter(text, reasoning)
