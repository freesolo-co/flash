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


def _eos_terminated(texts: list[str], tokenizer) -> list[str]:
    """Append EOS (once) to each text.

    EOS is appended because a packed row concatenates examples with no separator, so it is the only
    thing marking where one example ends -- without it the model learns to run past the end of an
    answer into the next one.
    """
    eos = tokenizer.eos_token or ""
    return [t if (eos and t.endswith(eos)) else t + eos for t in texts]


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
class _RenderedTargetFields:
    paths: frozenset[_MessagePath]
    adjacent_runs: tuple[tuple[_MessagePath, ...], ...]


def _probed_text(text: str, path: _MessagePath, sentinels_for) -> str:
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


def _rendered_target_body_fields(
    template_source,
    template: str,
    source_messages: list[dict],
    target_messages: list[dict],
    template_kwargs: dict,
) -> list[_RenderedTargetFields]:
    """Probe rendered source leaves and preserve their exact output adjacency."""
    target_start = len(source_messages) - len(target_messages)
    if target_start < 0:
        raise ValueError("SFT target messages cannot outnumber the full source transcript")
    haystack = template + repr(source_messages)
    prefix = "flashchatmlfieldprobe"
    while prefix in haystack:
        prefix += "x"

    sentinels: list[tuple[int, _MessagePath, str, str]] = []
    probed: list[dict] = []
    for index, message in enumerate(source_messages):
        target_index = index - target_start

        def sentinels_for(path: _MessagePath, target_index=target_index) -> tuple[str, str]:
            if target_index < 0:
                return "", ""
            marker_index = len(sentinels)
            start = f"{prefix}s{marker_index}x"
            end = f"{prefix}e{marker_index}x"
            sentinels.append((target_index, path, start, end))
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

    occurrences: list[tuple[int, int, int, _MessagePath]] = []
    for target_index, path, start, end in sentinels:
        offset = 0
        while (start_at := rendered.find(start, offset)) >= 0:
            end_at = rendered.find(end, start_at + len(start))
            if end_at < 0:
                raise ValueError("chat template field probe lost an end sentinel")
            after_end = end_at + len(end)
            occurrences.append((start_at, after_end, target_index, path))
            offset = after_end
    occurrences.sort()

    paths = [set() for _ in target_messages]
    runs: list[list[list[_MessagePath]]] = [[] for _ in target_messages]
    previous_target = -1
    previous_end = -1
    for start_at, after_end, target_index, path in occurrences:
        paths[target_index].add(path)
        if target_index == previous_target and start_at == previous_end:
            runs[target_index][-1].append(path)
        else:
            runs[target_index].append([path])
        previous_target = target_index
        previous_end = after_end
    return [
        _RenderedTargetFields(
            paths=frozenset(message_paths),
            adjacent_runs=tuple(tuple(run) for run in message_runs),
        )
        for message_paths, message_runs in zip(paths, runs, strict=True)
    ]


def _path_value(message: dict, path: _MessagePath):
    value = message
    for part in path:
        value = value[part]
    return value


def _reject_chatml_control_bodies(
    target_messages: list[dict], rendered_fields: list[_RenderedTargetFields]
) -> None:
    """Reject controls inside rendered leaves or across directly adjacent rendered leaves."""
    for message, fields in zip(target_messages, rendered_fields, strict=True):
        for run in fields.adjacent_runs:
            text = "".join(str(_path_value(message, path)) for path in run)
            for token in _CHATML_CONTROL_TOKENS:
                if token in text:
                    raise ValueError(
                        f"SFT message body contains reserved ChatML control token {token}; "
                        "remove or escape the literal token before training"
                    )


def _has_authored_assistant_body(message: dict, fields: _RenderedTargetFields) -> bool:
    return any(
        isinstance((value := _path_value(message, path)), str) and bool(value.strip())
        for path in fields.paths
    )


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
    behaviour, and the pre-opened ``<think>\\n`` handling stay intact. ChatML role headers and
    structural delimiters are never targets; assistant body tokens are. When the transcript does not
    parse as ChatML the mask is returned unchanged and ``role_aware`` is false rather than narrowing
    supervision on a guess.

    ChatML control strings are reserved in every template-rendered target body field. Their token ids
    are identical to structural delimiters after rendering, so accepting a literal would make the
    boundary ambiguous and could either expose an observation or hide real assistant supervision.
    Non-ChatML targets keep their prior behavior.
    """
    template_source = tokenizer if template_source is None else template_source
    template = _chatml_template(template_source)
    if template is None:
        return AssistantOnlyMask(loss_mask, False)
    source_messages_supplied = source_messages is not None
    source_messages = target_messages if source_messages is None else source_messages
    rendered_fields = _rendered_target_body_fields(
        template_source,
        template,
        source_messages,
        target_messages,
        template_kwargs or {},
    )
    _reject_chatml_control_bodies(target_messages, rendered_fields)
    if not full_ids or not any(loss_mask):
        return AssistantOnlyMask(loss_mask, False)
    spans = _chatml_message_spans(full_ids, tokenizer)
    if spans is None:
        return AssistantOnlyMask(loss_mask, False)

    authored_assistant = [False] * len(spans)
    target_span_indexes: set[int] = set()
    if source_messages_supplied:
        if len(spans) > len(source_messages):
            return AssistantOnlyMask(loss_mask, False)
        source_roles = [str(message.get("role")).strip().lower() for message in source_messages]
        if any(span[4] != source_roles[index] for index, span in enumerate(spans)):
            return AssistantOnlyMask(loss_mask, False)
        target_start = len(source_messages) - len(target_messages)
        for source_index in range(max(0, target_start), len(spans)):
            target_index = source_index - target_start
            message = target_messages[target_index]
            target_span_indexes.add(source_index)
            if str(message.get("role")).strip().lower() == "assistant":
                authored_assistant[source_index] = _has_authored_assistant_body(
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

    masked = list(loss_mask)
    last_target_span: tuple[int, str, bool] | None = None
    for source_index, (start, body_start, body_end, end, role) in enumerate(spans):
        bounded_end = min(end, len(masked))
        authored = authored_assistant[source_index]
        if source_index in target_span_indexes and any(loss_mask[start:bounded_end]):
            last_target_span = (bounded_end, role, authored)
        for position in range(start, bounded_end):
            masked[position] = 0
        if role == "assistant" and authored:
            for position in range(min(body_start, len(masked)), min(body_end, len(masked))):
                masked[position] = loss_mask[position]
    if last_target_span is not None and (
        last_target_span[1] != "assistant" or not last_target_span[2]
    ):
        for position in range(last_target_span[0], len(masked)):
            masked[position] = 0
    return AssistantOnlyMask(masked, True)
