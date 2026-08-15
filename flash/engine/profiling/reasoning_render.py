"""How much of a transcript's authored reasoning survives the chat template into the loss.

Qwen3.5's template keeps ``<think>`` only on assistant turns after the last user message and strips
it from earlier history, so a K-turn gold transcript delivers roughly 1/K of its reasoning to the
loss. The stored messages are never wrong -- only the render is -- so a correct-looking dataset and
a green ``flash env test`` both survive it.

Survival is answered by IDENTITY, never by counting ``<think>`` tags in the render. The row is
rendered a second time with each reasoning-authoring turn's reasoning marked, and a turn survives
exactly when its marker reaches the render. Counting cannot answer it: the template opens an empty
``<think>`` block on every trailing assistant turn whether or not it authored anything, so a
transcript stripped of all its reasoning still renders one block and counts as a survivor -- the
exact case this measurement exists to catch. A marker rides inside the reasoning itself, so a
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

    The template takes the reasoning as the text after the LAST ``<think>`` preceding the FIRST
    ``</think>``. An opener-less ``reasoned</think>answer`` is reasoning too: the prompt supplies
    the opening tag, so a completion sampled against it carries only the close.

    Emptiness is judged on the body, because the template stamps an empty ``<think>\\n\\n</think>``
    onto qualifying trailing turns -- treating that as authored would put a turn in the denominator
    that never had reasoning to lose.

    One rule for both jobs: it decides whether a turn authored reasoning and where that turn's
    marker goes. Were the two to disagree, a turn would enter the denominator with no way to prove
    it survived, and report a drop that never happened.
    """
    close = text.find(_THINK_CLOSE)
    if close < 0:
        return None
    open_at = text.rfind(_THINK_OPEN, 0, close)
    body = 0 if open_at < 0 else open_at + len(_THINK_OPEN)
    return close if text[body:close].strip() else None


def reasoned_assistant_turns(messages: list[dict[str, Any]]) -> int:
    """Assistant turns that author reasoning, counted from the SOURCE messages.

    Every shape the template accepts counts, because a shape missed here reads as "authored no
    reasoning" and silences the warning for a row losing all of it: an inline span in string or
    content-block ``content``, a separate ``reasoning_content`` field, and the opener-less form.

    A ``reasoning_content`` STRING is authoritative even when empty: the template renders that field
    and leaves ``content`` whole as the answer, so an empty field means the turn authored nothing
    however the answer is written. Falling back to inline detection there reads a ``<think>`` the
    ANSWER quotes as reasoning. Absent and ``None`` are different -- both parse the inline span.
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

    Extending one character at a time is quadratic on the input that forces it -- text holding the
    stem followed by N filler characters keeps every candidate a substring, so each of the N scans
    walks the whole row. Doubling bounds it at a logarithmic number of scans, then bisection trims
    back to the shortest absent length so the marker stays short.
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

    Per-turn rather than one shared stamp: survival is asked per turn, and a caller holding only the
    shared prefix could not tell which turn a surviving marker belongs to.

    The trailing ``e`` terminates the index. Without it ``{prefix}1`` is a substring of
    ``{prefix}10``, so a dropped turn 1 would read as surviving whenever turn 10 does.
    """
    return f"{prefix}{index}e"


def _marks_reasoning(message: dict) -> bool:
    """Whether this turn gets a marker stamped into it.

    One predicate for both the stamping and the listing: a turn listed but not stamped reads as
    reasoning the template dropped, and a turn stamped but not listed cannot be checked at all.
    """
    return message.get("role") == "assistant" and bool(reasoned_assistant_turns([message]))


def reasoning_markers(messages: list[dict], prefix: str) -> list[str]:
    """The marker for each reasoning-authoring turn, in order.

    Callers need them separately to ask survival per turn. Searching a render for the shared prefix
    answers only "did any reasoning survive", which is the wrong question for a row whose turns are
    answered differently -- one surviving turn would cover a stripped neighbour.
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
    identical positions and keeps every turn's reasoning exactly where it would have anyway.
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
                copied["reasoning_content"] = reasoning + marker
            else:
                copied["content"] = _marked_inline_reasoning(copied.get("content"), marker)
        marked.append(copied)
    return marked


def strip_reasoning_markers(text: str, prefix: str) -> str:
    """``text`` with every marker removed, so a marked render can be measured against the cap."""
    return re.sub(rf"{re.escape(prefix)}\d+e", "", text)


def marked_reasoning_end(marked_text: str, marker: str) -> int | None:
    """Where this turn's rendered reasoning ENDS in the marked render, or ``None`` if it was dropped.

    ``None`` is the survival answer: a marker rides only inside text the template chose to keep as
    this turn's reasoning, so its absence is exactly a turn whose reasoning was stripped.

    The offset runs through the block's closing tag rather than stopping at the marker, so a caller
    comparing it against ``max_context_tokens`` asks whether the whole block reached the loss. It
    deliberately stops THERE rather than at the blank line separating reasoning from the answer:
    that separator tokenizes to a real token, so measuring to it would call a block whose every
    reasoning token was retained truncated whenever the cap lands just past the closing tag.
    """
    at = marked_text.find(marker)
    if at < 0:
        return None
    close = marked_text.find(_THINK_CLOSE, at)
    return at + len(marker) if close < 0 else close + len(_THINK_CLOSE)


def horizon_row_count(row_count: int, *, examples_per_update: int, updates: int) -> int:
    """How many retained rows an update horizon reaches, in retained order.

    Updates consume ``examples_per_update`` rows each, wrapping at the end of an epoch, so a horizon
    at or past one full pass reaches every row and the answer saturates at ``row_count`` rather than
    counting a row twice. Shared by the profile producer, which bounds the counts it serializes, and
    by the CLI, which needs the matching denominator: a bounded count over a whole-dataset
    denominator would understate the survival rate it reports.
    """
    per_update = max(int(examples_per_update), 1)
    return min(int(row_count), max(int(updates) * per_update, 0))


def reasoning_warning_rows(profile: object) -> int:
    """The row denominator the reasoning counts were totalled over, from a profile OBJECT or dict.

    Both shapes are accepted because the same line is rendered from both: the worker holds the
    profile itself, while the CLI only ever sees the dict that travelled on the quote.

    The producer serializes this alongside the counts rather than leaving it to be re-derived. A
    reader cannot tell a horizon-bounded count from a whole-dataset one by inspection -- both carry
    ``examples_per_update`` and ``authoritative_steps`` -- so deriving the denominator from those
    would pair a binding horizon with counts measured over every retained row.
    """

    def _int(key: str) -> int | None:
        value = profile.get(key) if isinstance(profile, dict) else getattr(profile, key, None)
        return None if isinstance(value, bool) or not isinstance(value, int) else value

    rows = _int("reasoning_rows")
    if rows is not None:
        return rows
    retained = _int("retained_examples")
    return 0 if retained is None else retained


def rendered_reasoning_loss_warning(
    *,
    authored_turns: int,
    rendered_spans: int,
    rows: int,
    truncated_spans: int = 0,
) -> str | None:
    """One user-facing line when authored reasoning does not reach the loss.

    Reasoning is lost two ways with OPPOSITE remedies: the template strips it from earlier history,
    or ``max_context_tokens`` cuts a block the template kept off the end of the row. Telling the
    second user to split their transcript would be wrong advice for a dataset whose structure is
    fine, so the causes are counted separately and each names its own fix.

    Silent when nothing was authored (the existing thinking-mode check owns that case) and when
    everything survived. Reports the measurement rather than a fixed threshold: any drop is real
    lost supervision, and the count is exact rather than sampled.
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
