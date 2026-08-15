"""Architecture probes and tokenization helpers for the packed training path.

verl does the packing itself (``pad_mode: no_padding`` collates to one concatenated
``(1, total_nnz)`` row), so what remains here is deciding whether a checkpoint may take that path.
GDN hybrids (Qwen3.5/3.6) may only pack when the interpreter running the forward can reset the
linear-attention recurrence (fla's ``cu_seqlens``) and the causal conv (``seq_idx``) at example
boundaries -- without both, state carries across examples inside a packed micro-batch.
"""

from __future__ import annotations

from dataclasses import dataclass


def _text_config(cfg):
    """Return the decoder sub-config (multimodal checkpoints nest it under ``text_config``)."""
    return getattr(cfg, "text_config", None) or cfg


def _load_text_config(model_id: str, revision: str, *, trust_remote_code: bool = True):
    """The checkpoint's decoder config. Raises when the config cannot be resolved."""
    from transformers import AutoConfig

    from flash.engine.worker.io.hf import model_revision_kwargs

    return _text_config(
        AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            **model_revision_kwargs(revision),
        )
    )


def probe_is_pure_attention(
    model_id: str, revision: str = "", *, trust_remote_code: bool = True
) -> bool:
    """Pure-attention probe that RAISES rather than swallowing a failed config read.

    Callers that freeze the answer into a digest must use this. Returning False for "the hub timed
    out" is fine for a runtime gate that only has to avoid packing, but it is a wrong ANSWER, and a
    wrong answer written into a profile is compared byte-for-byte against a later re-derivation
    that may have gotten the config just fine.
    """
    cfg = _load_text_config(model_id, revision, trust_remote_code=trust_remote_code)
    layer_types = getattr(cfg, "layer_types", None)
    if layer_types:
        return all(t == "full_attention" for t in layer_types)
    for attr in ("linear_num_key_heads", "linear_key_head_dim", "linear_conv_kernel_dim"):
        if getattr(cfg, attr, None):
            return False
    # Qwen2.5 has sliding_window but disables it via use_sliding_window=False -> still packs.
    # Mistral-style has no use_sliding_window flag -> assume window is active.
    sliding = getattr(cfg, "sliding_window", None)
    return not (sliding and getattr(cfg, "use_sliding_window", True))


def probe_is_gdn_hybrid(
    model_id: str, revision: str = "", *, trust_remote_code: bool = True
) -> bool:
    """``model_is_gdn_hybrid`` without the swallow: a failed probe RAISES.

    See ``probe_is_pure_attention`` for why a digest-freezing caller needs the distinction.
    """
    cfg = _load_text_config(model_id, revision, trust_remote_code=trust_remote_code)
    layer_types = getattr(cfg, "layer_types", None)
    if layer_types and any(t == "linear_attention" for t in layer_types):
        return True
    return any(
        getattr(cfg, a, None)
        for a in ("linear_num_key_heads", "linear_key_head_dim", "linear_conv_kernel_dim")
    )


def model_is_gdn_hybrid(model_id: str, revision: str = "") -> bool:
    """True for a GatedDeltaNet hybrid (Qwen3.5/3.6) that needs cu_seqlens + seq_idx to pack."""
    try:
        return probe_is_gdn_hybrid(model_id, revision)
    except Exception as e:
        print(f"[pack] gdn-hybrid probe failed for {model_id!r} (treating as NOT gdn): {e}")
        return False


def gdn_model_type(model_id: str | None, revision: str = "") -> str:
    """The checkpoint's ``model_type``, naming the ``transformers.models.*`` module to inspect.

    Read from the checkpoint's config, so the answer is a property of the weights rather than of
    whichever transformers is installed -- which is what lets the parent resolve it on behalf of the
    verl child (see backend_common.gdn_probe_module).
    """
    if not model_id:
        return "qwen3_5"
    try:
        from transformers import AutoConfig

        from flash.engine.worker.io.hf import model_revision_kwargs

        cfg = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
            **model_revision_kwargs(revision),
        )
        return getattr(cfg, "model_type", None) or "qwen3_5"
    except Exception:
        return "qwen3_5"


def _gdn_forward_threads_reset_kwargs(model_id: str | None, revision: str = "") -> bool:
    """Check that THIS arch's GDN forward actually accepts cu_seq_lens_q and seq_idx (varies by transformers version)."""
    try:
        import importlib
        import inspect

        model_type = gdn_model_type(model_id, revision=revision)
        mod = importlib.import_module(f"transformers.models.{model_type}.modeling_{model_type}")
        gdn_cls = next(
            (
                c
                for n, c in vars(mod).items()
                if isinstance(c, type) and n.endswith("GatedDeltaNet")
            ),
            None,
        )
        if gdn_cls is None:
            return False
        fwd = inspect.getsource(gdn_cls.forward)
        return ("cu_seq_lens_q" in fwd) and ("seq_idx" in fwd)
    except Exception:
        return False


def worker_image_packing_support(model_id: str, revision: str = "") -> tuple[str, bool]:
    """Return the packing contract supplied by the fixed worker image.

    the worker image pins transformers, fla, and causal_conv1d as one tested stack. the control plane
    deliberately does not install those gpu-only modules, so curated catalog architecture identifies
    the GDN contract without importing repository code or the worker stack. uncataloged models still
    use a data-only config probe. the worker verifies its installed modules and forward signature
    before executing a packed quote.
    """
    from flash.core.catalog import MODELS

    info = MODELS.get(model_id)
    if info is not None:
        if info.num_linear_attention_layers:
            return "gdn-hybrid", True
        if info.num_attention_layers:
            return "pure-attention", True
        return "unsupported", False
    if probe_is_pure_attention(model_id, revision=revision, trust_remote_code=False):
        return "pure-attention", True
    if probe_is_gdn_hybrid(model_id, revision=revision, trust_remote_code=False):
        return "gdn-hybrid", True
    return "unsupported", False


def gdn_packing_contract_available(model_id: str | None = None, revision: str = "") -> bool:
    """True when the installed gdn stack exposes the boundary-reset contract without opening cuda.

    device-independent by construction because the gpu worker must verify capability without opening
    cuda before it accepts a packed quote. a gate that consults ``torch.cuda`` could change the answer
    across worker lifecycle stages.

    this asks whether the installed transformers threads the reset kwargs into this arch's GDN
    forward and whether both kernels are importable. ``is_*_available()`` is deliberately not used
    because those open with ``is_torch_cuda_available()``.
    """
    try:
        import importlib

        importlib.import_module("fla")
        importlib.import_module("causal_conv1d")
        return _gdn_forward_threads_reset_kwargs(model_id, revision=revision)
    except Exception:
        return False


def packing_appends_eos(text: str, tokenizer) -> bool:
    eos = tokenizer.eos_token or ""
    return bool(eos and not text.endswith(eos))


def _eos_terminated(texts: list[str], tokenizer) -> list[str]:
    """Append EOS (once) to each text.

    EOS is appended because a packed row concatenates examples with no separator, so it is the only
    thing marking where one example ends -- without it the model learns to run past the end of an
    answer into the next one.
    """
    eos = tokenizer.eos_token or ""
    return [text + eos if packing_appends_eos(text, tokenizer) else text for text in texts]


def tokenize_for_packing(texts: list[str], tokenizer, max_length: int) -> list[list[int]]:
    """Tokenize texts for packing: EOS appended, tokenizer's default add_special_tokens."""
    enc = tokenizer(_eos_terminated(texts, tokenizer), truncation=True, max_length=max_length)
    return enc["input_ids"]


def untruncated_lengths_for_packing(texts: list[str], tokenizer) -> list[int]:
    """Token count per text with NO cap, under the same EOS rule ``tokenize_for_packing`` applies.

    Sharing ``_eos_terminated`` is what makes the two measurements comparable. Encoding the raw text
    instead would drop the appended EOS, so every row that does not already end in one would measure
    a token SHORT of its own realized length -- and "untruncated < realized" is exactly the
    contradiction that makes a truncation count untrustworthy.
    """
    enc = tokenizer(_eos_terminated(texts, tokenizer), truncation=False)
    return [len(ids) for ids in enc["input_ids"]]


def completion_mask_from_ids(prompt_ids: list[int], full_ids: list[int]) -> list[int]:
    """Return per-token mask: 0 over the shared prompt prefix, 1 over the completion.

    Uses the longest shared token prefix rather than len(prompt_ids) so the boundary is robust to
    the thinking template's pre-opened ``<think>\\n`` (which diverges from the full render by a token).
    Returns ``[]`` for empty full_ids; all-zero when truncation removed the entire completion.
    """
    n_full = len(full_ids)
    if n_full == 0:
        return []
    n = 0
    for a, b in zip(prompt_ids, full_ids, strict=False):  # different lengths by design (prefix)
        if a != b:
            break
        n += 1
    if n >= n_full:
        return [0] * n_full
    return [0] * n + [1] * (n_full - n)


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


def _active_chat_template(source) -> str | None:
    """Return the metadata-selected template without rendering user-controlled messages."""
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
    template = _active_chat_template(source)
    if template is None or not all(token in template for token in _CHATML_CONTROL_TOKENS):
        return None
    return template


def _sanitized_probe_text(text: str) -> str:
    for token in _CHATML_CONTROL_TOKENS:
        text = text.replace(token, "<|flash_reserved_control|>")
    return text


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


def _probed_text(text: str, path: _MessagePath, sentinels_for) -> str:
    if not text:
        return text
    start, end = sentinels_for(path)
    return f"{start}{_sanitized_probe_text(text)}{end}"


def _probe_content(content, path: _MessagePath, sentinels_for):
    if isinstance(content, str):
        return _probed_text(content, path, sentinels_for)
    if not isinstance(content, list):
        return content
    copied = []
    for index, block in enumerate(content):
        if not isinstance(block, dict):
            copied.append(_probe_value(block, (*path, index), sentinels_for))
            continue
        copied.append(
            {
                key: value
                if key == "type"
                else _probe_value(value, (*path, index, key), sentinels_for)
                for key, value in block.items()
            }
        )
    return copied


def _probe_value(value, path: _MessagePath, sentinels_for):
    if isinstance(value, str):
        return _probed_text(value, path, sentinels_for)
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


def _chatml_text_spans(rendered: str) -> list[tuple[int, int, int, int, str]] | None:
    spans: list[tuple[int, int, int, int, str]] = []
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
        message_end = body_end + len("<|im_end|>")
        if rendered.startswith("\n", message_end):
            message_end += 1
        spans.append(
            (
                start,
                body_start,
                body_end,
                message_end,
                rendered[header_start : body_start - 1].strip().lower(),
            )
        )
        index = message_end
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


def _rendered_field_occurrences(
    rendered: str,
    sentinels: list[tuple[int, _MessagePath, str, str]],
    text_spans: list[tuple[int, int, int, int, str]] | None,
) -> list[_RenderedFieldOccurrence]:
    occurrences = []
    for source_index, path, start, end in sentinels:
        starts = _marker_positions(rendered, start)
        ends = _marker_positions(rendered, end)
        if starts and ends:
            offset = 0
            while (start_at := rendered.find(start, offset)) >= 0:
                end_at = rendered.find(end, start_at + len(start))
                if end_at < 0:
                    raise ValueError("chat template field probe lost an end sentinel")
                after_end = end_at + len(end)
                occurrences.append(
                    _RenderedFieldOccurrence(
                        source_index, path, start_at, after_end, start_at, after_end
                    )
                )
                offset = after_end
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
        containing = [
            span_index
            for span_index, (_start, body_start, body_end, _end, _role) in enumerate(text_spans)
            if body_start <= marker_start and marker_end <= body_end
        ]
        if len(containing) == 1:
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

        copied = {}
        for key, value in message.items():
            if key == "role":
                copied[key] = value
            elif key == "content":
                copied[key] = _probe_content(value, (key,), sentinels_for)
            else:
                copied[key] = _probe_value(value, (key,), sentinels_for)
        probed.append(copied)

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

    paths = [set() for _ in source_messages]
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
    source_span_indexes = [set() for _ in source_messages]
    for occurrence in occurrences:
        containing = [
            span_index
            for span_index, (_start, body_start, body_end, _end, _role) in enumerate(text_spans)
            if body_start <= occurrence.marker_start and occurrence.marker_end <= body_end
        ]
        if len(containing) != 1:
            return _RenderedMessageProbe(
                source_fields,
                None,
                tuple(span[4] for span in text_spans),
            )
        source_span_indexes[occurrence.source_index].add(containing[0])
    return _RenderedMessageProbe(
        source_fields,
        tuple(frozenset(indexes) for indexes in source_span_indexes),
        tuple(span[4] for span in text_spans),
    )


def _path_value(message: dict, path: _MessagePath):
    value = message
    for part in path:
        value = value[part]
    return value


def _reject_chatml_control_bodies(
    messages: list[dict], rendered_fields: tuple[_RenderedMessageFields, ...]
) -> None:
    """Reject controls inside rendered leaves or across directly adjacent rendered leaves."""
    for message, fields in zip(messages, rendered_fields, strict=True):
        for run in fields.adjacent_runs:
            text = "".join(str(_path_value(message, path)) for path in run)
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
        if fixed[source_index] is not None:
            candidates = [fixed[source_index]]
        else:
            lower = next(
                (
                    fixed[index]
                    for index in range(source_index - 1, -1, -1)
                    if fixed[index] is not None
                ),
                0,
            )
            upper = next(
                (
                    fixed[index]
                    for index in range(source_index + 1, len(fixed))
                    if fixed[index] is not None
                ),
                len(span_roles) - 1,
            )
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
    span_sources = [[] for _ in span_roles]
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

    Returns ``None`` when the render is not ChatML (no ``<|im_start|>``/``<|im_end|>`` pair, or no
    message parsed), which is the signal for the caller to leave the mask untouched rather than
    guess. Roles are lowercased; a header the tokenizer splits across several tokens is joined
    before comparison so a multi-token role name still resolves.
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


@dataclass(frozen=True)
class AssistantOnlyMask:
    mask: list[int]
    role_aware: bool


def assistant_only_mask(
    loss_mask: list[int],
    full_ids: list[int],
    tokenizer,
    target_messages: list[dict],
    *,
    appended_eos: bool,
    template_source=None,
    source_messages: list[dict] | None = None,
    template_kwargs: dict | None = None,
) -> AssistantOnlyMask:
    """Keep only assistant body supervision when the rendered transcript is parseable ChatML.

    ``completion_mask_from_ids`` returns ONE contiguous supervised span, which is right for a
    prompt -> completion row but wrong for a multi-turn target: everything after the prompt is
    supervised, so the interleaved environment/tool/user observations between assistant turns train
    the model to emit the environment's replies.

    Strictly subtractive: it only ever turns a 1 into a 0, so the prompt boundary, the truncation
    behaviour, and the pre-opened ``<think>\\n`` handling stay intact. ChatML role headers and static
    template separators are never targets; authored assistant bodies and their original turn closers are.
    When the transcript does not parse as ChatML the mask is returned unchanged and ``role_aware`` is
    false rather than narrowing supervision on a guess.

    ChatML control strings are reserved in every rendered string field across the complete source
    transcript. Their token ids are identical to structural delimiters after rendering, so accepting
    a literal would make the boundary ambiguous and could either expose an observation or hide real
    assistant supervision. Only the target suffix metadata determines authored assistant bodies.
    Non-ChatML targets keep their prior behavior. ``appended_eos`` is caller-supplied provenance, not
    an inference from the final token value: text packing sets it from ``_eos_terminated`` semantics,
    while processor tokenization leaves it false because that path appends nothing.
    """
    template_source = tokenizer if template_source is None else template_source
    template = _chatml_template(template_source)
    if template is None:
        return AssistantOnlyMask(loss_mask, False)
    source_messages_supplied = source_messages is not None
    source_messages = target_messages if source_messages is None else source_messages
    rendered_probe = _rendered_message_probe(
        template_source,
        template,
        source_messages,
        target_messages,
        template_kwargs or {},
    )
    _reject_chatml_control_bodies(source_messages, rendered_probe.source_fields)
    target_start = len(source_messages) - len(target_messages)
    rendered_fields = rendered_probe.source_fields[target_start:]
    if not full_ids or not any(loss_mask):
        return AssistantOnlyMask(loss_mask, False)
    spans = _chatml_message_spans(full_ids, tokenizer)
    control_ids = _chatml_control_ids(tokenizer)
    if spans is None or control_ids is None:
        return AssistantOnlyMask(loss_mask, False)
    im_end = control_ids[1]
    if any(
        body_end >= len(full_ids) or full_ids[body_end] != im_end for _, _, body_end, _, _ in spans
    ):
        return AssistantOnlyMask(loss_mask, False)

    authored_assistant = [False] * len(spans)
    target_span_indexes: set[int] = set()
    if source_messages_supplied:
        actual_roles = tuple(span[4] for span in spans)
        full_roles = rendered_probe.span_roles
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
            return AssistantOnlyMask(loss_mask, False)
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
    else:
        target_span_indexes = {
            index
            for index, (start, _body_start, _body_end, end, _role) in enumerate(spans)
            if any(loss_mask[start : min(end, len(loss_mask))])
        }
        if len(target_span_indexes) > len(target_messages):
            return AssistantOnlyMask(loss_mask, False)
        for target_index, span_index in enumerate(sorted(target_span_indexes)):
            message = target_messages[target_index]
            role = str(message.get("role")).strip().lower()
            if spans[span_index][4] != role:
                return AssistantOnlyMask(loss_mask, False)
            if role == "assistant":
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
    return AssistantOnlyMask(masked, True)
