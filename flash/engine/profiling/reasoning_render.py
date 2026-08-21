"""How much of a transcript's authored reasoning survives the chat template into the loss.

Qwen3.5's template keeps ``<think>`` only on assistant turns after the last user message and strips
it from earlier history, so a K-turn gold transcript delivers roughly 1/K of its reasoning to the
loss. The stored messages are never wrong -- only the render is -- so a correct-looking dataset and
a green ``flash env test`` both survive it.

Survival is answered by IDENTITY: the row is rendered again with each reasoning-authoring turn's
reasoning marked, and a turn survives exactly when its marker reaches the render. Counting
``<think>`` tags cannot answer it -- the template opens an EMPTY block on every trailing assistant
turn, so a transcript stripped of all its reasoning still renders one and scores as a survivor,
the exact case this exists to catch. A marker also rides inside the reasoning itself, so a
``<think>`` an answer merely quotes carries none and can never be credited.
"""

from __future__ import annotations

import re
from typing import Any

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _message_text(content: object) -> str:
    """The text of a message's ``content`` in either the string or content-block shape."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
    return ""


def _reasoning_body_end(text: str) -> int | None:
    """Where this turn's reasoning BODY ends, or ``None`` when the turn authored no reasoning.

    The template's own rule: the text after the LAST ``<think>`` preceding the FIRST ``</think>``.
    An opener-less ``reasoned</think>answer`` counts, being the shape a completion sampled against a
    prompt-supplied opening tag carries.

    Emptiness is judged on the body: the template stamps an empty ``<think>\\n\\n</think>`` onto
    qualifying trailing turns, and counting that as authored puts a turn in the denominator that
    never had reasoning to lose.

    The end excludes trailing whitespace because the template renders reasoning through ``|trim``. A
    marker placed after that whitespace shields it from the trim, so the marked render keeps bytes
    the real one drops and tokenizes one token longer -- a cap at the block's real end then reports
    a block that fits as cut.
    """
    close = text.find(_THINK_CLOSE)
    if close < 0:
        return None
    open_at = text.rfind(_THINK_OPEN, 0, close)
    body = 0 if open_at < 0 else open_at + len(_THINK_OPEN)
    return len(text[:close].rstrip()) if text[body:close].strip() else None


def reasoned_assistant_turns(messages: list[dict[str, Any]]) -> int:
    """Assistant turns that author reasoning, counted from the SOURCE messages.

    Every accepted shape counts -- inline span, content blocks, ``reasoning_content``, opener-less --
    because a shape missed here reads as "authored no reasoning" and silences the warning for a row
    losing all of it.

    A ``reasoning_content`` STRING is authoritative even when EMPTY: the template renders that field
    and leaves ``content`` whole, so an empty field means the turn authored nothing however the
    answer is written, and falling back to inline detection would read a ``<think>`` the ANSWER
    quotes as reasoning. Absent and ``None`` differ -- both parse the inline span.
    """
    turns = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str):
            turns += 1 if reasoning.strip() else 0
        elif _reasoning_body_end(_message_text(message.get("content"))) is not None:
            turns += 1
    return turns


def reasoning_marker_prefix(text: str) -> str:
    """A marker stem guaranteed absent from ``text``, so a marker cannot match user content.

    Extended rather than assumed unique: a dataset containing the stem would otherwise let its own
    text answer "did this turn's reasoning survive?".

    Doubling then bisecting, not growing one character at a time: text holding the stem followed by
    N filler characters keeps every one-character extension a substring, so each of N scans walks
    the whole row (1.4s on a 50k row, before tokenization).
    """
    stem = "flashreasoningmark"
    if stem not in text:
        return stem
    low, high = 0, 1
    while stem + "x" * high in text:
        low, high = high, high * 2
    while low + 1 < high:
        mid = (low + high) // 2
        low, high = (mid, high) if stem + "x" * mid in text else (low, mid)
    return stem + "x" * high


def _turn_marker(prefix: str, index: int) -> str:
    """The marker naming ONE turn's reasoning.

    The trailing ``e`` terminates the index: without it ``{prefix}1`` is a substring of
    ``{prefix}10``, so a dropped turn 1 reads as surviving whenever turn 10 does.
    """
    return f"{prefix}{index}e"


def _marks_reasoning(message: dict) -> bool:
    """Whether this turn gets a marker stamped into it.

    One predicate for both stamping and listing: a turn listed but not stamped reads as a template
    drop, and a turn stamped but not listed cannot be checked at all.
    """
    return message.get("role") == "assistant" and bool(reasoned_assistant_turns([message]))


def reasoning_markers(messages: list[dict], prefix: str) -> list[str]:
    """The marker for each reasoning-authoring turn, in order.

    Separate markers because survival is asked per turn: searching for the shared prefix answers
    only "did ANY reasoning survive", so one surviving turn would cover a stripped neighbour.
    """
    return [
        _turn_marker(prefix, index)
        for index, message in enumerate(messages)
        if _marks_reasoning(message)
    ]


def _marked_inline_reasoning(content: object, marker: str) -> object:
    """``content`` with ``marker`` stamped at the end of the reasoning the template will keep."""
    if isinstance(content, str):
        at = _reasoning_body_end(content)
        return content if at is None else content[:at] + marker + content[at:]
    if not isinstance(content, list):
        return content
    # the template concatenates the text blocks before splitting, so the offset is resolved on the
    # joined text and then mapped back into whichever block contains it.
    texts = [
        block["text"]
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        else None
        for block in content
    ]
    at = _reasoning_body_end("".join(text or "" for text in texts))
    if at is None:
        return content
    marked: list = []
    consumed = 0
    placed = False
    for block, text in zip(content, texts, strict=True):
        if text is None:
            marked.append(block)
            continue
        local = at - consumed
        consumed += len(text)
        # an offset on a block boundary matches the end of one block and the start of the next, so
        # the guard keeps it stamped exactly once.
        if placed or not 0 <= local <= len(text):
            marked.append(block)
            continue
        placed = True
        marked.append({**block, "text": text[:local] + marker + text[local:]})
    return marked


def with_marked_reasoning(messages: list[dict], prefix: str) -> list[dict]:
    """The same messages with each reasoning-authoring assistant turn's reasoning marked.

    Only reasoning TEXT changes, so the template's ``last_query_index`` rule sees identical roles in
    identical positions and keeps exactly the blocks it would have anyway.

    Both shapes mark the ``|trim``-ed reasoning; marking after trailing whitespace would shield it
    from the template's trim (see ``_reasoning_body_end``).
    """
    marked: list[dict] = []
    for index, message in enumerate(messages):
        copied = dict(message)
        if _marks_reasoning(message):
            marker = _turn_marker(prefix, index)
            reasoning = copied.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                # the template reads this field in preference to an inline span, so it is the
                # reasoning and the marker belongs in it
                copied["reasoning_content"] = reasoning.rstrip() + marker
            else:
                copied["content"] = _marked_inline_reasoning(copied.get("content"), marker)
        marked.append(copied)
    return marked


def strip_reasoning_markers(text: str, prefix: str) -> str:
    """``text`` with every marker removed, so a marked render can be measured against the cap."""
    return re.sub(rf"{re.escape(prefix)}\d+e", "", text)


def marked_reasoning_end(marked_text: str, marker: str) -> int | None:
    """Where this turn's rendered reasoning ENDS in the marked render, or ``None`` if it was dropped.

    ``None`` IS the survival answer: a marker rides only inside text the template kept as this
    turn's reasoning, so its absence is exactly a stripped turn.

    The offset runs through the closing tag -- so the caller asks whether the WHOLE block fit -- and
    stops there rather than at the blank line before the answer, which costs a real token and would
    call a fully retained block truncated.
    """
    at = marked_text.find(marker)
    if at < 0:
        return None
    close = marked_text.find(_THINK_CLOSE, at)
    return at + len(marker) if close < 0 else close + len(_THINK_CLOSE)


def horizon_row_count(row_count: int, *, examples_per_update: int, updates: int) -> int:
    """How many retained rows an update horizon reaches, in retained order.

    Updates wrap at the end of an epoch, so a horizon past one full pass saturates at ``row_count``
    rather than counting a row twice. Shared with the CLI so the denominator matches: a bounded
    count over a whole-dataset denominator would understate the survival rate.
    """
    per_update = max(int(examples_per_update), 1)
    return min(int(row_count), max(int(updates) * per_update, 0))


def reasoning_warning_rows(profile: object) -> int:
    """The row denominator the reasoning counts were totalled over, from a profile OBJECT or dict.

    Both shapes render the same line: the worker holds the profile, the CLI only sees the dict that
    travelled on the quote. It is serialized rather than re-derived because a horizon-bounded count
    is indistinguishable from a whole-dataset one by inspection -- both carry ``examples_per_update``
    and ``authoritative_steps`` -- so deriving it would pair a binding horizon with unbounded counts.
    """
    value = profile["reasoning_rows"] if isinstance(profile, dict) else profile.reasoning_rows
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("workload profile reasoning_rows must be an integer")
    return value


def rendered_reasoning_loss_warning(
    *,
    authored_turns: int,
    rendered_spans: int,
    rows: int,
    truncated_spans: int = 0,
) -> str | None:
    """One user-facing line when authored reasoning does not reach the loss.

    Reasoning is lost two ways with OPPOSITE remedies -- the template strips earlier history, or
    ``max_context_tokens`` cuts a block the template KEPT off the end of the row -- so the causes are
    counted separately and each names its own fix. Silent when nothing was authored (the thinking-
    mode check owns that) or when everything survived; any drop is reported, since the count is
    exact rather than sampled and any loss is real lost supervision.
    """
    if authored_turns <= 0:
        return None
    stripped = authored_turns - rendered_spans
    if stripped <= 0 and truncated_spans <= 0:
        return None
    reaching = rendered_spans - truncated_spans
    causes = []
    if stripped > 0:
        causes.append(
            f"the chat template dropped {stripped} of {authored_turns} authored reasoning blocks "
            "-- it keeps <think> only on assistant turns after the last user message and strips it "
            "from earlier history, so a multi-turn transcript trains on a fraction of its reasoning "
            "and on a tag layout inference never produces. split each K-turn transcript into K "
            "single-turn rows, so every turn's reasoning sits in the final assistant target where "
            "the template keeps it"
        )
    if truncated_spans > 0:
        causes.append(
            f"max_context_tokens cut {truncated_spans} rendered reasoning "
            f"{'block' if truncated_spans == 1 else 'blocks'} off the end of the row -- the "
            "template kept these, so raise max_context_tokens or shorten the rows rather than "
            "restructuring the transcript"
        )
    return (
        f"{'; '.join(causes)}. across {rows} SFT rows, {reaching} of {authored_turns} authored "
        f"reasoning blocks reach the loss ({reaching / authored_turns:.0%})."
    )
