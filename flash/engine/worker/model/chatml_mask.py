"""Assistant-only loss masking for multi-turn SFT rows rendered through a ChatML template.

``completion_mask_from_ids`` supervises ONE contiguous span, which is right for a
prompt -> completion row and wrong for a multi-turn target: the interleaved environment, tool, and
user observations between assistant turns end up trained on, teaching the model to emit the
environment's replies. What lives here narrows that mask to the authored assistant bodies.

Kept apart from ``packing`` because the two share nothing: that module decides whether a CHECKPOINT
may take the packed path (architecture probes, GDN kernel contracts), while this one reads a
RENDERED TRANSCRIPT and decides which of its tokens are the model's own output.
"""

from __future__ import annotations

from dataclasses import dataclass

_CHATML_CONTROL_TOKENS = ("<|im_start|>", "<|im_end|>")


def _chatml_control_ids(tokenizer) -> tuple[int, int] | None:
    to_id = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(to_id):
        return None
    im_start, im_end = (to_id(token) for token in _CHATML_CONTROL_TOKENS)
    unk = getattr(tokenizer, "unk_token_id", None)
    if im_start is None or im_end is None or im_start in (unk, im_end) or im_end == unk:
        return None
    return im_start, im_end


def _appended_eos_index(
    full_ids: list[int], tokenizer, after: int, *, appended_eos: bool
) -> int | None:
    if not appended_eos or not full_ids or len(full_ids) <= after:
        return None
    eos_token = getattr(tokenizer, "eos_token", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None and isinstance(eos_token, str) and eos_token:
        to_id = getattr(tokenizer, "convert_tokens_to_ids", None)
        if callable(to_id):
            eos_id = to_id(eos_token)
    if isinstance(eos_id, int) and full_ids[-1] == eos_id:
        return len(full_ids) - 1
    decode = getattr(tokenizer, "decode", None)
    if (
        isinstance(eos_token, str)
        and eos_token
        and callable(decode)
        and decode([full_ids[-1]]) == eos_token
    ):
        return len(full_ids) - 1
    return None


def _selected_chat_template(source) -> str | None:
    """return the metadata-selected template text, if the source exposes one."""
    get_template = getattr(source, "get_chat_template", None)
    if callable(get_template):
        try:
            template = get_template()
        except (AttributeError, TypeError, ValueError):
            return None
    else:
        template = getattr(source, "chat_template", None)
        if isinstance(template, dict):
            template = template.get("default")
    return template if isinstance(template, str) else None


def _chatml_template(source) -> str | None:
    """return the selected template only when it carries both chatml controls."""
    template = _selected_chat_template(source)
    if template is None or not all(token in template for token in _CHATML_CONTROL_TOKENS):
        return None
    return template


_MessagePath = tuple[str | int, ...]


@dataclass(frozen=True)
class _RenderedMessageFields:
    paths: frozenset[_MessagePath]
    adjacent_runs: tuple[tuple[_MessagePath, ...], ...]


@dataclass(frozen=True)
class _RenderedMessageProbe:
    source_fields: tuple[_RenderedMessageFields, ...]
    source_span_indexes: tuple[frozenset[int], ...] | None
    span_roles: tuple[str, ...]


def _probe_value(value, path: _MessagePath, sentinels_for):
    """Wrap every string leaf in a paired sentinel, keeping ``type`` discriminators intact.

    ``type`` selects the template's branch for a content block, so sentinelizing it would render a
    different transcript than the one being masked. An empty string is left alone so it keeps the
    template's falsey branch; wrapping it would render as authored content.
    """
    if isinstance(value, str):
        if not value:
            return value
        start, end = sentinels_for(path)
        # reserved controls are neutralized in the probe render so a literal cannot forge a span
        # boundary; `_reject_chatml_control_bodies` is what refuses the row.
        for token in _CHATML_CONTROL_TOKENS:
            value = value.replace(token, "<|flash_reserved_control|>")
        return f"{start}{value}{end}"
    if isinstance(value, list):
        return [
            _probe_value(item, (*path, index), sentinels_for) for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            key: item if key == "type" else _probe_value(item, (*path, key), sentinels_for)
            for key, item in value.items()
        }
    return value


def _chatml_text_spans(rendered: str) -> list[tuple[int, int, str]] | None:
    """Locate each rendered message's body character range and role, or ``None`` if unparseable."""
    spans: list[tuple[int, int, str]] = []
    index = 0
    while (start := rendered.find("<|im_start|>", index)) >= 0:
        header_start = start + len("<|im_start|>")
        body_start = rendered.find("\n", header_start)
        if body_start < 0:
            return None
        body_start += 1
        body_end = rendered.find("<|im_end|>", body_start)
        if body_end < 0:
            return None
        role = rendered[header_start : body_start - 1].strip().lower()
        spans.append((body_start, body_end, role))
        index = body_end + len("<|im_end|>")
    return spans or None


@dataclass(frozen=True)
class _RenderedFieldOccurrence:
    source_index: int
    path: _MessagePath
    marker_start: int
    marker_end: int
    left: int | None
    right: int | None


def _marker_positions(rendered: str, marker: str) -> list[int]:
    positions = []
    offset = 0
    while (position := rendered.find(marker, offset)) >= 0:
        positions.append(position)
        offset = position + len(marker)
    return positions


def _containing_body_spans(
    text_spans: list[tuple[int, int, str]], start: int, end: int
) -> list[int]:
    return [
        span_index
        for span_index, (body_start, body_end, _role) in enumerate(text_spans)
        if body_start <= start and end <= body_end
    ]


def _rendered_field_occurrences(
    rendered: str,
    sentinels: list[tuple[int, _MessagePath, str, str]],
    text_spans: list[tuple[int, int, str]] | None,
) -> list[_RenderedFieldOccurrence]:
    occurrences = []
    for source_index, path, start, end in sentinels:
        starts = _marker_positions(rendered, start)
        ends = _marker_positions(rendered, end)
        if starts and ends:
            # a paired survival bounds the authored leaf exactly, so both edges are known adjacency.
            offset = 0
            while (start_at := rendered.find(start, offset)) >= 0:
                end_at = rendered.find(end, start_at + len(start))
                if end_at < 0:
                    raise ValueError("chat template field probe lost an end sentinel")
                offset = end_at + len(end)
                occurrences.append(
                    _RenderedFieldOccurrence(source_index, path, start_at, offset, start_at, offset)
                )
            continue
        if len(starts) + len(ends) != 1 or text_spans is None:
            continue
        if starts:
            marker_start = starts[0]
            marker_end = marker_start + len(start)
            left, right = marker_start, None
        else:
            marker_start = ends[0]
            marker_end = marker_start + len(end)
            left, right = None, marker_end
        if len(_containing_body_spans(text_spans, marker_start, marker_end)) == 1:
            occurrences.append(
                _RenderedFieldOccurrence(source_index, path, marker_start, marker_end, left, right)
            )
    return sorted(occurrences, key=lambda occurrence: occurrence.marker_start)


def _rendered_message_probe(
    template_source,
    template: str,
    source_messages: list[dict],
    target_messages: list[dict],
    template_kwargs: dict,
) -> _RenderedMessageProbe:
    """Probe rendered source leaves, their adjacency, and their structural message spans."""
    target_start = len(source_messages) - len(target_messages)
    if target_start < 0:
        raise ValueError("SFT target messages cannot outnumber the full source transcript")
    haystack = template + repr(source_messages)
    prefix = "flashchatmlfieldprobe"
    while prefix in haystack:
        prefix += "x"

    sentinels: list[tuple[int, _MessagePath, str, str]] = []
    probed: list[dict] = []
    for source_index, message in enumerate(source_messages):

        def sentinels_for(
            path: _MessagePath,
            source_index=source_index,
        ) -> tuple[str, str]:
            marker_index = len(sentinels)
            start = f"{prefix}s{marker_index}x"
            end = f"{prefix}e{marker_index}x"
            sentinels.append((source_index, path, start, end))
            return start, end

        probed.append(
            {
                # role selects the template's branch and is emitted as a header rather than a body,
                # so it is never sentinelized.
                key: value if key == "role" else _probe_value(value, (key,), sentinels_for)
                for key, value in message.items()
            }
        )

    rendered = template_source.apply_chat_template(
        probed,
        tokenize=False,
        add_generation_prompt=False,
        **template_kwargs,
    )
    if not isinstance(rendered, str):
        raise TypeError("chat template field probe must return rendered text")

    text_spans = _chatml_text_spans(rendered)
    occurrences = _rendered_field_occurrences(rendered, sentinels, text_spans)

    paths: list[set[_MessagePath]] = [set() for _ in source_messages]
    runs: list[list[list[_MessagePath]]] = [[] for _ in source_messages]
    previous_source = -1
    previous_right = None
    for occurrence in occurrences:
        source_index = occurrence.source_index
        paths[source_index].add(occurrence.path)
        if (
            source_index == previous_source
            and previous_right is not None
            and occurrence.left is not None
            and occurrence.left == previous_right
        ):
            runs[source_index][-1].append(occurrence.path)
        else:
            runs[source_index].append([occurrence.path])
        previous_source = source_index
        previous_right = occurrence.right
    source_fields = tuple(
        _RenderedMessageFields(
            paths=frozenset(message_paths),
            adjacent_runs=tuple(tuple(run) for run in message_runs),
        )
        for message_paths, message_runs in zip(paths, runs, strict=True)
    )

    if text_spans is None:
        return _RenderedMessageProbe(source_fields, None, ())
    span_roles = tuple(role for _, _, role in text_spans)
    source_span_indexes: list[set[int]] = [set() for _ in source_messages]
    for occurrence in occurrences:
        containing = _containing_body_spans(
            text_spans, occurrence.marker_start, occurrence.marker_end
        )
        # a leaf that straddles or escapes the body spans cannot be attributed to one message.
        if len(containing) != 1:
            return _RenderedMessageProbe(source_fields, None, span_roles)
        source_span_indexes[occurrence.source_index].add(containing[0])
    return _RenderedMessageProbe(
        source_fields, tuple(frozenset(indexes) for indexes in source_span_indexes), span_roles
    )


def _path_value(message: dict, path: _MessagePath):
    value = message
    for part in path:
        value = value[part]
    return value


def _rendered_field_runs(messages: list[dict], rendered_fields: tuple[_RenderedMessageFields, ...]):
    for message, fields in zip(messages, rendered_fields, strict=True):
        for run in fields.adjacent_runs:
            yield "".join(str(_path_value(message, path)) for path in run)


def reject_rendered_message_token(
    template_source,
    messages: list[dict],
    token: str,
    *,
    template_kwargs: dict,
) -> _RenderedMessageProbe | None:
    """reject a token from every source field the selected chat template emits."""
    template = _selected_chat_template(template_source)
    if template is None:
        return None
    rendered_probe = _rendered_message_probe(
        template_source,
        template,
        messages,
        [],
        template_kwargs,
    )
    if any(token in text for text in _rendered_field_runs(messages, rendered_probe.source_fields)):
        raise ValueError(
            f"sft message contains the reserved image marker {token!r} in a field emitted by "
            "the chat template; remove it before training"
        )
    return rendered_probe


def _reject_chatml_control_bodies(
    messages: list[dict], rendered_fields: tuple[_RenderedMessageFields, ...]
) -> None:
    """Reject controls inside rendered leaves or across directly adjacent rendered leaves."""
    for text in _rendered_field_runs(messages, rendered_fields):
        for token in _CHATML_CONTROL_TOKENS:
            if token in text:
                raise ValueError(
                    f"SFT message body contains reserved ChatML control token {token}; "
                    "remove or escape the literal token before training"
                )


def _has_authored_assistant_body(message: dict, fields: _RenderedMessageFields) -> bool:
    return any(
        isinstance((value := _path_value(message, path)), str) and bool(value.strip())
        for path in fields.paths
    )


def _source_span_association(
    source_messages: list[dict],
    direct_span_indexes: tuple[frozenset[int], ...],
    span_roles: tuple[str, ...],
) -> tuple[int, ...] | None:
    if len(direct_span_indexes) != len(source_messages) or not span_roles:
        return None
    fixed: list[int | None] = []
    for indexes in direct_span_indexes:
        if len(indexes) > 1:
            return None
        fixed.append(next(iter(indexes)) if indexes else None)

    association: list[int] = []
    for source_index, message in enumerate(source_messages):
        source_role = str(message.get("role")).strip().lower()
        expected_assistant = source_role == "assistant"
        direct = fixed[source_index]
        if direct is not None:
            candidates = [direct]
        else:
            lower = 0
            for index in range(source_index - 1, -1, -1):
                if (candidate := fixed[index]) is not None:
                    lower = candidate
                    break
            upper = len(span_roles) - 1
            for index in range(source_index + 1, len(fixed)):
                if (candidate := fixed[index]) is not None:
                    upper = candidate
                    break
            candidates = [
                span_index
                for span_index in range(lower, upper + 1)
                if (span_roles[span_index] == "assistant") == expected_assistant
            ]
        if len(candidates) != 1:
            return None
        span_index = candidates[0]
        if (span_roles[span_index] == "assistant") != expected_assistant:
            return None
        association.append(span_index)

    if any(association[index] > association[index + 1] for index in range(len(association) - 1)):
        return None
    span_sources: list[list[int]] = [[] for _ in span_roles]
    for source_index, span_index in enumerate(association):
        span_sources[span_index].append(source_index)
    if any(not sources for sources in span_sources):
        return None
    for span_index, sources in enumerate(span_sources):
        if span_roles[span_index] == "assistant" and len(sources) != 1:
            return None
    return tuple(association)


def _chatml_message_spans(
    full_ids: list[int], tokenizer
) -> list[tuple[int, int, int, int, str]] | None:
    """Split one rendered transcript into message and body token boundaries.

    Reads the ONE full render rather than re-rendering message prefixes. Re-rendering cannot be
    trusted here: Qwen3.5/3.6 pick each assistant turn's ``<think>`` layout from ``last_query_index``,
    computed over the WHOLE message list, so ``render(messages[:k])`` is not a prefix of
    ``render(messages)`` -- a shorter list re-renders earlier turns with a different tag layout and
    the derived offsets slide off the real turn boundaries.

    ``None`` means the render is not parseable ChatML, which tells the caller to leave the mask
    alone. A role header split across several tokens is joined before it is lowercased.
    """
    decode = getattr(tokenizer, "decode", None)
    control_ids = _chatml_control_ids(tokenizer)
    if control_ids is None or not callable(decode):
        return None
    im_start, im_end = control_ids
    if im_start not in full_ids or im_end not in full_ids:
        return None

    spans: list[tuple[int, int, int, int, str]] = []
    index = 0
    total = len(full_ids)
    while index < total:
        if full_ids[index] != im_start:
            index += 1
            continue
        cursor = index + 1
        role_ids: list[int] = []
        while cursor < total and full_ids[cursor] not in (im_start, im_end):
            piece = decode([full_ids[cursor]])
            if "\n" in piece:
                break
            role_ids.append(full_ids[cursor])
            cursor += 1
        role = decode(role_ids).strip().lower() if role_ids else ""
        body_start = min(cursor + 1, total)
        cursor = body_start
        while cursor < total and full_ids[cursor] != im_end:
            cursor += 1
        body_end = cursor
        message_end = min(cursor + 1, total)
        if message_end < total and decode([full_ids[message_end]]) == "\n":
            message_end += 1
        spans.append((index, body_start, body_end, message_end, role))
        index = message_end
    return spans or None


def assistant_only_mask(
    loss_mask: list[int],
    full_ids: list[int],
    tokenizer,
    target_messages: list[dict],
    *,
    appended_eos: bool,
    source_messages: list[dict],
    template_kwargs: dict,
    template_source=None,
    rendered_probe: _RenderedMessageProbe | None = None,
) -> tuple[list[int], bool]:
    """Keep only assistant body supervision when the rendered transcript is parseable ChatML.

    ``completion_mask_from_ids`` returns ONE contiguous supervised span, which is right for a
    prompt -> completion row but wrong for a multi-turn target: everything after the prompt is
    supervised, so the interleaved environment/tool/user observations between assistant turns train
    the model to emit the environment's replies.

    Strictly subtractive: it only ever turns a 1 into a 0, so the prompt boundary, the truncation
    behaviour, and the pre-opened ``<think>\\n`` handling stay intact. ChatML role headers and static
    template separators are never targets; authored assistant bodies and their original turn closers
    are. Returns ``(mask, role_aware)``; whenever the render cannot be read unambiguously the mask is
    returned unchanged with ``role_aware`` false rather than narrowing supervision on a guess.

    ``source_messages`` is the COMPLETE transcript, prompt included, because reserved ChatML control
    strings have to be rejected wherever the template renders them: their token ids are identical to
    structural delimiters afterwards, so a literal in a prompt field moves the same boundary a
    literal in a target field would. Only the target suffix decides which assistant bodies are
    authored. ``appended_eos`` is caller-supplied provenance, never an inference from the final token
    value: text packing sets it from ``_eos_terminated`` semantics, while processor tokenization
    leaves it false because that path appends nothing.
    """
    template_source = tokenizer if template_source is None else template_source
    template = _chatml_template(template_source)
    if template is None:
        return loss_mask, False
    if rendered_probe is None:
        rendered_probe = _rendered_message_probe(
            template_source,
            template,
            source_messages,
            target_messages,
            template_kwargs,
        )
    _reject_chatml_control_bodies(source_messages, rendered_probe.source_fields)
    target_start = len(source_messages) - len(target_messages)
    rendered_fields = rendered_probe.source_fields[target_start:]
    if not full_ids or not any(loss_mask):
        return loss_mask, False
    spans = _chatml_message_spans(full_ids, tokenizer)
    control_ids = _chatml_control_ids(tokenizer)
    if spans is None or control_ids is None:
        return loss_mask, False
    im_end = control_ids[1]
    # a span whose closer the cap removed has no readable end, so its body cannot be bounded.
    if any(
        body_end >= len(full_ids) or full_ids[body_end] != im_end for _, _, body_end, _, _ in spans
    ):
        return loss_mask, False

    actual_roles = tuple(span[4] for span in spans)
    full_roles = rendered_probe.span_roles
    # the token row may be a right-truncated PREFIX of the probed render, so the parsed roles have to
    # match that render's leading spans; anything else means the two disagree about the transcript.
    if (
        rendered_probe.source_span_indexes is None
        or len(actual_roles) > len(full_roles)
        or actual_roles != full_roles[: len(actual_roles)]
        or (
            source_span_indexes := _source_span_association(
                source_messages,
                rendered_probe.source_span_indexes,
                full_roles,
            )
        )
        is None
    ):
        return loss_mask, False

    authored_assistant = [False] * len(spans)
    target_span_indexes: set[int] = set()
    for source_index in range(max(0, target_start), len(source_messages)):
        target_index = source_index - target_start
        span_index = source_span_indexes[source_index]
        if span_index >= len(spans):
            continue
        message = target_messages[target_index]
        target_span_indexes.add(span_index)
        if str(message.get("role")).strip().lower() == "assistant":
            authored_assistant[span_index] = _has_authored_assistant_body(
                message, rendered_fields[target_index]
            )
    masked = [0] * len(loss_mask)
    last_target_span: tuple[str, bool] | None = None
    for span_index, (start, body_start, body_end, end, role) in enumerate(spans):
        bounded_end = min(end, len(masked))
        authored = authored_assistant[span_index]
        if span_index in target_span_indexes and any(loss_mask[start:bounded_end]):
            last_target_span = (role, authored)
        if role == "assistant" and authored:
            for position in range(min(body_start, len(masked)), min(body_end, len(masked))):
                masked[position] = loss_mask[position]
            if body_end < bounded_end and full_ids[body_end] == im_end:
                masked[body_end] = loss_mask[body_end]
    if last_target_span == ("assistant", True):
        eos_index = _appended_eos_index(
            full_ids,
            tokenizer,
            spans[-1][3],
            appended_eos=appended_eos,
        )
        if eos_index is not None:
            masked[eos_index] = loss_mask[eos_index]
    return masked, True
