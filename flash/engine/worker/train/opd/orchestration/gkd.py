"""Pure token-alignment + groupwise reverse-KL math for on-policy distillation (algorithm="opd").

Extracted from ``opd.py`` to keep that module under the size bar and make the cross-tokenizer
alignment helpers independently unit-testable. See ``opd.py`` for the orchestration and batched loss
path.
"""

from __future__ import annotations

import contextlib

from flash.content.thinking import _messages_for_content_only_serialization
from flash.engine.worker.teacher.tokenizer_align import StudentToken
from flash.engine.worker.train.core.child.glue import trim_trailing_stop


def _teacher_prompt_text(prompt_messages: list[dict], thinking_prefill: str = "") -> str:
    """Render the prompt for the teacher's scoring context (template-agnostic; ends at 'Assistant: ').

    Qwen's ``<think>\n``). The student samples its on-policy completion AFTER that prefill, so the
    teacher must condition on the SAME trailing context; otherwise every thinking-mode token is
    scored against a prompt that never opened the reasoning block, and the gkd logprobs are
    conditioned on a different prefix than the sampled tokens.
    """
    from flash.content.multimodal import message_content_text

    parts = []
    for m in _messages_for_content_only_serialization(prompt_messages):
        role = str(m.get("role", "user")).capitalize()
        # `!s` on a content-block list renders a python repr ("[{'type': 'text', ...}]") and the
        # teacher scores that literal without failing. a mixed image/text opd job carries block
        # content on its text-only rows too, so go through the one definition of message text.
        parts.append(f"{role}: {message_content_text(m.get('content', ''))}")
    parts.append("Assistant: " + thinking_prefill)
    return "\n".join(parts)


def _generation_eos_ids(model, tok) -> frozenset:
    """The FULL set of token ids that halt this model's generation — the union of the tokenizer's

    A completion ending on that secondary id is a NATURAL termination even though it !=
    ``tok.eos_token_id``, so ``_rollout_terminated`` must see the whole set; a single-id check
    misreads the rollout as truncated and skips it, which with no stop_sequences burns every sample
    and fails the run with "no trained step".
    """
    ids: set[int] = set()

    def _add(v):
        # bool is an int subclass but never a token id; exclude it from both the scalar and list forms.
        if isinstance(v, bool):
            return
        if isinstance(v, int):
            ids.add(v)
        elif isinstance(v, (list, tuple, set, frozenset)):
            ids.update(int(x) for x in v if isinstance(x, int) and not isinstance(x, bool))

    _add(getattr(tok, "eos_token_id", None))
    config = getattr(model, "config", None)
    for obj in (getattr(model, "generation_config", None), config):
        if obj is not None:
            _add(getattr(obj, "eos_token_id", None))
    if config is not None:
        text_configs = []
        get_text_config = None
        with contextlib.suppress(Exception):
            get_text_config = getattr(config, "get_text_config", None)
        if callable(get_text_config):
            with contextlib.suppress(Exception):
                text_configs.append(get_text_config(decoder=True))
        with contextlib.suppress(Exception):
            text_configs.append(getattr(config, "text_config", None))
        for text_config in text_configs:
            if text_config is not None:
                with contextlib.suppress(Exception):
                    _add(getattr(text_config, "eos_token_id", None))
    return frozenset(ids)


def generation_eos_from_cached_config(
    model_id: str, model_revision: str, tokenizer
) -> frozenset[int]:
    """``_generation_eos_ids`` for a model that is on disk but NOT loaded.

    The verl paths never hold the model in this process -- it is trained in a separate interpreter
    -- so the config and generation_config are read from the local hf cache and presented as the
    attributes ``_generation_eos_ids`` expects. ``local_files_only`` because the caller prefetched
    the model and the child runs with HF_HUB_OFFLINE.
    """
    from transformers import AutoConfig, GenerationConfig

    config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=True, revision=model_revision or None, local_files_only=True
    )
    generation_config = None
    with contextlib.suppress(OSError):
        generation_config = GenerationConfig.from_pretrained(
            model_id, revision=model_revision or None, local_files_only=True
        )
    model_like = type(
        "ModelGenerationMetadata",
        (),
        {"config": config, "generation_config": generation_config},
    )()
    return _generation_eos_ids(model_like, tokenizer)


def _rollout_terminated(completion_ids, stop_text, eos_ids, stop_sequences) -> bool:
    """True iff the rollout ended NATURALLY — the student emitted ANY of the model's EOS ids, or (when

    HF ``generate`` halts on four conditions -- EOS, a ``stop_strings`` match, the
    ``max_new_tokens`` cap, or the ``gen_cfg.max_time`` wall-clock bound -- and only the first two
    are natural completions. Fail OPEN only when NO termination signal exists at all (empty
    ``eos_ids`` AND no stop_sequences): we then can't tell a finished answer from a cut-off one, so
    we distil rather than skip every rollout.
    """
    if eos_ids and not eos_ids.isdisjoint(completion_ids):
        return True
    if stop_sequences and any(s and stop_text.endswith(s) for s in stop_sequences):
        return True
    return not eos_ids and not stop_sequences


def student_tokens_with_offsets(tok, completion_ids, completion_text: str):
    """Char spans for the ORIGINAL sampled token ids, indexed into ``completion_text`` (the exact

    Split multi-byte chars (byte-level tokenizers): a char whose bytes span two+ ids decodes to the
    Unicode replacement char ``U+FFFD`` until its FINAL byte-id arrives. Instead we GROW the window
    while the decoded prefix still ends in ``U+FFFD`` and assign the SHARED completed-char span to
    every byte-id in it, so both halves share a ``start`` and land in the same alignment group.
    """
    ids = [int(t) for t in completion_ids]
    toks: list[StudentToken] = []
    prev = 0
    n = len(completion_text)
    i = 0
    while i < len(ids):
        j = i
        # Grow the window over a split multi-byte char, decoding ONLY the window from the
        # current boundary (ids[i:j+1]) -- never the whole prefix ids[:j+1] -- so total
        # decoding is O(len), not O(len^2) (the quadratic bites once max_tokens raises
        # completions to 1000s of tokens).
        window_text = tok.decode(ids[i : j + 1], skip_special_tokens=True)
        while (
            j + 1 < len(ids)
            and window_text.endswith("\ufffd")
            and not completion_text.startswith(window_text, prev)
        ):
            j += 1
            window_text = tok.decode(ids[i : j + 1], skip_special_tokens=True)
        # Anchor to completion_text (the ground-truth full decode) via find() from prev
        # rather than prev + len(decode(window)): a SentencePiece/LLaMA tokenizer decodes
        # a word token IN ISOLATION without its leading word-boundary space
        # (decode([▁world]) == "world", not " world"), so prev + len(window) would
        # undercount the span by one char and drift EVERY following offset -- misaligning
        # teacher spans onto the wrong sampled ids. find() locates where the window text
        # actually sits (skipping any dropped leading whitespace, which is absorbed into
        # this token's start) and the start stays pinned at prev so the spans remain
        # contiguous.
        hit = completion_text.find(window_text, prev)
        end = (hit + len(window_text)) if hit != -1 else prev + len(window_text)
        end = min(n, max(prev, end))
        toks.extend(StudentToken(token_id=ids[k], start=prev, end=end) for k in range(i, j + 1))
        prev = end
        i = j + 1
    return ids, toks


# the OPD single-turn path and the shared child rollout loops trim the same delimiter the same way,
# so both read one implementation. it lives in the child glue module because that file is stdlib-only
# and gets copied into the verl child workdir, which cannot import flash.
_trim_trailing_stop = trim_trailing_stop
