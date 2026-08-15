"""Where the chat template puts reasoning, and how much of it survives into the loss.

Separated from ``workload_profile`` because identifying reasoning is its own problem: it depends
only on the rendered text and the template's layout, not on any part of a workload profile.
"""

from __future__ import annotations

import re
from typing import Any

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_TURN_END = "<|im_end|>"
# the template opens reasoning in exactly one place: straight after an assistant header. that
# anchor is what separates a block the template OWNS from the same characters appearing as
# ordinary text elsewhere, and it is where a block's start is taken from.
#
# a REAL header always follows the previous turn's terminator, or begins the render. requiring that
# is what keeps a header QUOTED inside a block from reading as the start of a new turn: text can
# contain the header characters, but only the template can put them after `<|im_end|>`.
_TEMPLATE_REASONING_START = re.compile(
    rf"(?:\A|{re.escape(_TURN_END)}\n)<\|im_start\|>assistant\n(?P<open>{_THINK_OPEN})\n"
)
# the start of ANY real turn, reasoning-bearing or not. the horizon has to advance on every turn
# rather than only on the next reasoning one: a turn that authored no reasoning -- a tool response,
# a bare answer -- still ends the previous turn, and a scan that ignores it runs past the block's
# own closer into whatever those later turns contain.
_TEMPLATE_TURN_START = re.compile(rf"(?:\A|{re.escape(_TURN_END)}\n)<\|im_start\|>\w+\n")
# the block's own closer: the newline-delimited form the template emits before the answer.
_TEMPLATE_REASONING_END = f"\n{_THINK_CLOSE}\n\n"


def reasoning_spans(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` of each NON-EMPTY reasoning block the TEMPLATE owns, in rendered text.

    Non-empty because Qwen3.5's template opens a ``<think>`` block on every trailing assistant turn
    whether or not that turn authored reasoning: a transcript whose reasoning was entirely stripped
    still renders one EMPTY block, and counting it would score full survival for the exact case
    this measurement exists to catch.

    Both ENDS are anchored on the template's layout, because ``<think>`` and ``</think>`` are
    ordinary characters a transcript may contain anywhere -- a user asking what the tag means, or
    reasoning that quotes the thinking layout. Each half of the anchor answers a failure the other
    cannot:

    * the START is the ``<think>`` immediately following an assistant header. Tracking openers by
      depth instead lets an unmatched ``<think>`` in an EARLIER turn hold the depth permanently
      positive, so the template's own closer never completes a block and every later survivor reads
      as stripped -- a total-loss warning for a transcript that lost nothing.
    * the END is the LAST newline-delimited closer before the turn's terminator. Taking the first
      one instead ends the span early whenever the reasoning quotes the layout it is reasoning
      about, and a cap landing between that quoted closer and the real one then scores a cut block
      as fully retained -- overstating what reaches the loss, the direction that hides the problem.

    The turn's terminator is the LAST ``<|im_end|>`` at or before the next turn's header, not the
    first one after the anchor: reasoning that mentions the token literally would otherwise end the
    scan before the template's own closer, returning no span at all and reporting an intact
    survivor as dropped. That horizon is the next header of ANY role -- a tool response or a bare
    answer ends the turn just as an assistant one does, and stopping only at the next REASONING
    turn would let a block run into text those later turns contain.

    Both the anchor and that horizon require a REAL header -- one following the previous turn's
    terminator -- so reasoning that quotes the header layout cannot open a phantom turn or pull the
    horizon in front of the closer. A quote is text; only the template writes a header after
    ``<|im_end|>``.

    What remains ambiguous is a transcript whose text contains the two TOGETHER, verbatim: a
    ``<|im_end|>`` newline and a header. Those bytes are what a turn boundary IS, so no rule reading
    the rendered text alone can tell them from one, and such a row reports no span for the turn.
    Reaching it takes a transcript that writes the control-token layout out in full rather than
    merely mentioning ``<think>`` or ``<|im_end|>``, both of which are handled above.

    Both endpoint rules read the rendered text alone, which likewise cannot separate a closer quoted
    at the END of the reasoning from one quoted at the START of the answer -- those two rows render
    identically and want opposite answers. Callers that need either distinction pair these spans
    with a marker stamped into the reasoning itself (see ``sft_workload._row_reasoning``), which is
    what keeps a mis-bounded span from being counted as a survivor.
    """
    spans: list[tuple[int, int]] = []
    for match in _TEMPLATE_REASONING_START.finditer(text):
        next_turn = _TEMPLATE_TURN_START.search(text, match.end())
        if next_turn is not None:
            # the horizon match CONSUMES this turn's terminator, so its start IS that terminator --
            # searching for one below the horizon would find only a terminator the reasoning quotes,
            # and end the scan in front of the real closer.
            limit = next_turn.start()
        else:
            # nothing follows, so the turn's own terminator is the last one in the render. reasoning
            # that quotes the token sits before it, which is why this looks backwards from the end.
            turn_end = text.rfind(_TURN_END, match.end())
            limit = len(text) if turn_end < 0 else turn_end
        close = text.rfind(_TEMPLATE_REASONING_END, match.end(), limit)
        if close < 0 or not text[match.end() : close].strip():
            continue
        spans.append((match.start("open"), close + len(_TEMPLATE_REASONING_END)))
    return spans


# why a run resolved to `exact-unpacked`, keyed by the architecture label the packing decision
# froze alongside the mode (`sft_workload._packing_mode`). that label is the only part of the


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


def reasoning_marker_prefix(text: str) -> str:
    """A marker stem guaranteed absent from ``text``, so a marker cannot match user content.

    Extended rather than assumed unique: a dataset that happens to contain the stem would otherwise
    let its own text answer "did this turn's reasoning survive?".

    Growing the stem one character per scan is quadratic on the input that forces it: text holding
    the stem followed by N filler characters keeps every extended candidate a substring, so each of
    the N scans walks the whole row. A 50k-character run measured 1.4s in this helper alone, on a
    row that is perfectly valid, and profiling runs before tokenization -- so a large packaged row
    stalls the control plane rather than failing anything.

    Doubling instead bounds it at a logarithmic number of scans. The stem is then trimmed back to
    the shortest absent length by bisection, so the marker stays as short as the one-at-a-time loop
    produced and callers see no change in what they get back.
    """
    stem = "flashreasoningmark"
    if stem not in text:
        return stem
    # smallest power-of-two run length that is absent, so the answer is bracketed in O(log n) scans
    low, high = 0, 1
    while stem + "x" * high in text:
        low, high = high, high * 2
    # then the shortest absent run in the bracket, keeping the marker minimal
    while low + 1 < high:
        mid = (low + high) // 2
        if stem + "x" * mid in text:
            low = mid
        else:
            high = mid
    return stem + "x" * high


def _reasoning_body_offset(text: str) -> tuple[int, int] | None:
    """Where this turn's reasoning BODY starts and ENDS, by the template's own rule.

    The template takes the reasoning as the text after the LAST ``<think>`` that precedes the first
    ``</think>``, over the whole concatenated message text. Stamping the first opener instead puts
    the marker outside the block the template keeps whenever an extra opener precedes the real one,
    and the marker then never reaches the render -- reporting a drop for reasoning that survived.

    An OPENER-LESS ``reasoned</think>answer`` is reasoning too. The prompt supplies the opening tag,
    so a sampled completion carries only the close, and ``flash.serve.thinking`` recognises the same
    shape. Requiring a balanced pair would score such a turn as authoring nothing, leaving it out of
    the denominator and understating -- or entirely suppressing -- the warning.

    Emptiness is judged on the BODY, after the opener is located, never on the text preceding the
    close. The template stamps an empty ``<think>\\n\\n</think>`` onto qualifying trailing assistant
    turns, so that leading text is present on a turn that authored nothing; treating it as authored
    marks a block the real render does not have, and the resulting span-count mismatch reports the
    whole row as template-dropped -- a total loss warning for a dataset that lost nothing.

    The END comes back with the start because the body is bracketed rather than merely stamped: the
    closing tag the template will honour is already located here, and a caller re-deriving it would
    have to repeat this same last-opener-before-first-close rule to find it.
    """
    close = text.find(_THINK_CLOSE)
    if close < 0:
        return None
    open_at = text.rfind(_THINK_OPEN, 0, close)
    body = 0 if open_at < 0 else open_at + len(_THINK_OPEN)
    return (body, close) if text[body:close].strip() else None


def _marked_inline_reasoning(content: object, head: str, tail: str) -> object:
    """``content`` with the reasoning the template will keep BRACKETED by ``head`` and ``tail``."""
    if isinstance(content, str):
        bounds = _reasoning_body_offset(content)
        if bounds is None:
            return content
        start, end = bounds
        return content[:start] + head + content[start:end] + tail + content[end:]
    if not isinstance(content, list):
        return content
    # the template concatenates the text blocks before splitting, so the delimiters are found on the
    # joined text and can straddle a block boundary. the offset is resolved there and then mapped
    # back into whichever block contains it.
    texts = [
        block["text"]
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        else None
        for block in content
    ]
    bounds = _reasoning_body_offset("".join(text or "" for text in texts))
    if bounds is None:
        return content
    # both ends are placed in one pass, highest offset first within a block, so inserting the head
    # cannot shift the tail's index out from under it when the two land in the SAME block.
    inserts = sorted(zip(bounds, (head, tail), strict=True), reverse=True)
    marked: list = []
    consumed = 0
    placed: set[int] = set()
    for block, text in zip(content, texts, strict=True):
        if text is None:
            marked.append(block)
            continue
        stamped = text
        for offset, marker in inserts:
            local = offset - consumed
            # an offset sitting exactly on a block boundary matches the END of one block and the
            # START of the next, so each end is placed once and only once.
            if offset not in placed and 0 <= local <= len(text):
                stamped = stamped[:local] + marker + stamped[local:]
                placed.add(offset)
        consumed += len(text)
        marked.append(block if stamped is text else {**block, "text": stamped})
    return marked


def _marks_reasoning(message: dict) -> bool:
    """Whether this turn gets a marker stamped into it.

    One predicate for both the stamping and the listing below: a turn listed as marked but not
    stamped reads as reasoning the template dropped, and a turn stamped but not listed cannot be
    checked for survival. Either way the row is mis-reported, so the two never get their own copy.
    """
    return message.get("role") == "assistant" and bool(reasoned_assistant_turns([message]))


def _turn_marker(prefix: str, index: int) -> tuple[str, str]:
    """The pair of markers BRACKETING one turn's reasoning: ``(head, tail)``.

    Per-turn rather than a single shared stamp: survival is asked per turn, and a caller holding
    only the shared prefix cannot tell WHICH turn a surviving marker belongs to -- one located
    block would answer for every other block in the row.

    Bracketing rather than a lone head stamp, because a head alone cannot see a span bounded SHORT.
    ``reasoning_spans`` ends a block at the last closer below its horizon, and reasoning that quotes
    a newline-delimited ``</think>`` before quoting a turn boundary pulls that horizon in front of
    the real closer, so the scan accepts the QUOTED closer as the end. Such a span still contains
    the head, so a head-only check reads it as resolved and the caller then measures truncation
    against an end that is too early -- scoring a block the cap cuts as fully retained, the
    direction that hides the loss. A tail sits after the last reasoning byte, so a span holding the
    head without it is exactly a span that stopped early.
    """
    return f"{prefix}{index} ", f" {prefix}e{index}"


def reasoning_markers(messages: list[dict], prefix: str) -> list[tuple[str, str]]:
    """The ``(head, tail)`` pair bracketing each reasoning-authoring turn, in order.

    Callers need the markers SEPARATELY to ask survival per turn. Searching a render for the shared
    prefix instead answers only "did any reasoning survive", which is the wrong question for a row
    whose turns can be answered differently: one turn the scan resolves would report the row
    resolvable while another turn's block sits unlocated (see ``sft_workload._row_reasoning``).

    Both ends come back because a turn is only judgeable when a span holds BOTH: a span with the
    head alone was bounded before the reasoning ended (see ``_turn_marker``), and its end offset is
    then too early to measure truncation against.
    """
    return [
        _turn_marker(prefix, index)
        for index, message in enumerate(messages)
        if _marks_reasoning(message)
    ]


def with_marked_reasoning(messages: list[dict], prefix: str) -> list[dict]:
    """The same messages with each reasoning-authoring assistant turn's reasoning stamped.

    Marking makes survival an IDENTITY question instead of a counting one, which is the only way to
    answer it correctly. Counting spans across a render with one turn's reasoning REMOVED gets two
    cases wrong, both silently:

    * a ``<think>`` tag an ANSWER merely quotes is an ordinary non-empty span, indistinguishable
      from reasoning by count. Removing a turn perturbs that quote too, so the count can fall for a
      turn whose reasoning the template actually dropped, and the loss goes unreported.
    * the two renders are different strings, so span offsets in one do not address the other, and a
      later span shifting makes an earlier one look truncated.

    A marker rides INSIDE the reasoning, so it appears in the rendered span if and only if the
    template kept that specific turn's reasoning. Quoted tags carry no marker and can never be
    credited. Only reasoning text changes, so the template's ``last_query_index`` rule sees
    identical roles in identical positions and the marked render keeps the full render's span
    sequence -- which is what lets survival be read from one render and offsets from the other.

    The reasoning is BRACKETED, head and tail, so the pair also reports where the block ENDS. A
    head alone says only that the block survived, never that the scan bounded all of it -- see
    ``_turn_marker`` for the span that stops at a closer the reasoning merely quotes.
    """
    marked: list[dict] = []
    for index, message in enumerate(messages):
        copied = dict(message)
        if _marks_reasoning(message):
            head, tail = _turn_marker(prefix, index)
            reasoning = copied.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                # the template reads this field in preference to an inline span, so it is the
                # reasoning and the markers belong in it
                copied["reasoning_content"] = head + reasoning + tail
            else:
                copied["content"] = _marked_inline_reasoning(copied.get("content"), head, tail)
        marked.append(copied)
    return marked


def reasoning_span_end_offsets(text: str) -> list[int]:
    """Character offsets just past each non-empty ``<think>`` span's CLOSING TAG.

    A truncated row keeps a rendered span only when the whole block fits inside the retained
    tokens, so the caller needs where each span ENDS to compare against the cap.

    The span itself runs to the end of the blank line that separates reasoning from the answer,
    because that separator is part of the layout identifying the block. It is NOT part of the
    reasoning: it tokenizes to a real token, so measuring to it would call a block whose every
    reasoning token was retained truncated whenever the cap lands between the closing tag and the
    answer -- a false cap-loss warning that understates how much reasoning reaches the loss.
    """
    return [
        text.rfind(_THINK_CLOSE, start, end) + len(_THINK_CLOSE)
        for start, end in reasoning_spans(text)
    ]


def reasoning_span_texts(text: str) -> list[str]:
    """Each non-empty ``<think>`` span's rendered text, in order.

    Paired positionally with ``reasoning_span_end_offsets`` of the UNMARKED render: the marked text
    says which turn owns each span, the real text says where that span ends.
    """
    return [text[start:end] for start, end in reasoning_spans(text)]


def reasoned_assistant_turns(messages: list[dict[str, Any]]) -> int:
    """Assistant turns that author reasoning, counted from the SOURCE messages.

    Every shape the chat template accepts counts, because a shape missed here reads as "authored no
    reasoning" and silences the warning for a row that is losing all of it:

    * a literal ``<think>...</think>`` span in a string ``content``;
    * the same span inside ``[{"type": "text", "text": ...}]`` content blocks, which
      ``flash.content.multimodal.text_only_prompt_messages`` flattens for rendering;
    * a separate ``reasoning_content`` field, which the template reads in preference to an inline
      span;
    * an opener-less ``reasoned</think>answer``, the shape a completion sampled against a
      prompt-supplied opening tag carries.

    Inline detection asks ``_reasoning_body_offset`` -- the same rule that places the marker -- so a
    shape counted as authored is always a shape that can be marked. Were the two to disagree, the
    turn would enter the denominator with no way to prove it survived, and report a false drop.

    A ``reasoning_content`` that is a STRING is authoritative even when it is empty: the template
    renders the field and leaves ``content`` whole as the answer, so an empty field means the turn
    authored no reasoning however the answer is written. Falling back to inline detection there
    reads a ``<think>`` the ANSWER quotes as this turn's reasoning, and since the marker then lands
    outside the empty block the template owns, the turn reports as dropped -- a loss warning for a
    row that authored nothing. Absent and ``None`` are different: the template parses the inline
    span in both, so those do fall back.
    """
    turns = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str):
            turns += 1 if reasoning.strip() else 0
            continue
        if _reasoning_body_offset(_message_text(message.get("content"))) is not None:
            turns += 1
    return turns


def count_rendered_reasoning_spans(text: str) -> int:
    """NON-EMPTY ``<think>`` spans in rendered text -- the reasoning that reaches the loss.

    Counting ``text.count("<think>")`` overstates this. Qwen3.5's template opens a ``<think>``
    block on every trailing assistant turn whether or not that turn authored reasoning, so a
    transcript whose reasoning was entirely stripped still renders one EMPTY block and would score
    as one surviving block instead of zero -- the exact case the warning exists to catch.
    """
    return len(reasoning_spans(text))


def horizon_row_count(row_count: int, *, examples_per_update: int, updates: int) -> int:
    """How many retained rows an update horizon reaches, in retained order.

    Updates consume ``examples_per_update`` rows each, wrapping at the end of an epoch, so a
    horizon at or past one full pass reaches every row and the answer saturates at ``row_count``
    rather than counting a row twice. Shared by the profile producer, which bounds the reasoning
    counts it serializes, and by the CLI, which needs the matching denominator: a bounded count
    over a whole-dataset denominator would understate the survival rate it reports.
    """
    per_update = max(int(examples_per_update), 1)
    return min(int(row_count), max(int(updates) * per_update, 0))


def reasoning_warning_rows(profile: object) -> int:
    """The row denominator the reasoning counts were totalled over, from a profile OBJECT or dict.

    Both shapes are accepted because the same line is rendered from both: the worker holds the
    profile itself, while the CLI only ever sees the dict that travelled on the quote. Serializing
    an in-memory profile just to read an integer would also compute a content digest for nothing.

    The producer serializes this alongside the counts rather than leaving it to be re-derived. A
    reader cannot tell a horizon-bounded count from a whole-dataset one by inspection -- both carry
    ``examples_per_update`` and ``authoritative_steps`` -- so deriving the denominator from those
    would pair a binding horizon with counts measured over every retained row and report more
    authored blocks than the rows it names could hold.

    Falls back to the whole retained count when the field is absent, which is what a producer
    predating it measured its counts over.
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

    Qwen3.5's template keeps reasoning only on assistant turns AFTER the last real user query and
    strips it from earlier history, so a K-turn gold transcript delivers roughly 1/K of its
    reasoning to the loss. Nothing else reports this: the rendered text is what trains, the stored
    messages were never wrong, and ``flash env test`` passes either way.

    Reasoning can also be lost a second way, with the OPPOSITE remedy: the template renders the
    block but ``max_context_tokens`` cuts it off the end of the row. Telling that user to split
    their transcript would be wrong advice for a dataset whose structure is fine, so the two causes
    are counted separately and each names its own fix. ``rendered_spans`` counts what the template
    kept; ``truncated_spans`` is how many of those the cap then removed.

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
