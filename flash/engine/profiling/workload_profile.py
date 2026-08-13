"""Immutable workload profiles used to freeze evidence-backed cost quotes."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, fields
from typing import Any

SFT_PROFILE_KIND = "sft"
ROLLOUT_PROFILE_KINDS = ("grpo", "opd")
# 3 removes the deleted per-run worker environment map from both profile identity payloads. 4 adds
# the authored [environment] pip digest, which changes the installed worker stack and so cannot be
# absent from identity. old cached profiles use a different identity shape, so reject them and
# re-profile. 5 serializes the reasoning-loss counts so the submitting client can render that
# warning; they are digest-free, but `from_dict` requires an exact field set, so a profile cached
# under 4 cannot be rebuilt and has to re-profile. 6 adds truncated_reasoning_spans, splitting
# cap-truncated reasoning out from template-stripped reasoning so the warning names the right
# remedy; same exact-field-set reason a 5 profile cannot be rebuilt.
WORKLOAD_PROFILE_SCHEMA_VERSION = 6
# 2 lets a gdn hybrid pack when the installed stack proves it can reset example boundaries, where 1
# always answered exact-unpacked. the same config therefore resolves to a different packing_mode and
# examples_per_update, so a profile cached under 1 quotes a step count this policy would not: it has
# to be a different identity rather than a cache hit.
SFT_PACKING_POLICY_VERSION = 2
ROLLOUT_SAMPLE_POLICY_VERSION = 1

# measured generation latency ages out; the shape it was measured from does not. a provider that
# slows down, or a card whose neighbours change, invalidates the seconds without invalidating the
# token counts, so the two carry different lifetimes and only one of them is keyed.
ROLLOUT_LATENCY_MAX_AGE_S = 24 * 60 * 60

# chosen floor, not a fit: variance is dominated by prompt mix, so sample distinct prompts before
# repeats. 32 holds +/-17% at the observed spread, stays cheap enough to run per config, and admits
# 8 distinct prompts at grpo's group size 4. it will drift with models and workloads.
MIN_TRUSTWORTHY_ROLLOUTS = 32

# conservative censoring policy, not a fit. a truncated completion contributes the cap instead of
# its true length, biasing the mean DOWNWARD -- the direction that underbills. reject heavily
# clipped samples rather than report a censored mean as measurement.
MAX_TRUSTWORTHY_TRUNCATION_RATE = 0.25

# which rows the profile measured. sft is never sampled: it either measures every source row or the
# deterministic max_examples prefix training will consume, so both policies are exact.
SFT_SAMPLE_POLICY_FULL = "exact-full"
SFT_SAMPLE_POLICY_PREFIX = "exact-prefix"


def sft_sample_policy(max_examples: object) -> str:
    """The row-selection rule this workload uses, derived from the authored ``max_examples``."""
    try:
        cap = int(max_examples or 0)
    except (TypeError, ValueError):
        cap = 0
    return SFT_SAMPLE_POLICY_PREFIX if cap > 0 else SFT_SAMPLE_POLICY_FULL


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_DELIMITER = re.compile(f"{_THINK_OPEN}|{_THINK_CLOSE}")


def reasoning_spans(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` of each NON-EMPTY ``<think>`` span in rendered text, outermost only.

    Non-empty because Qwen3.5's template opens a ``<think>`` block on every trailing assistant turn
    whether or not that turn authored reasoning: a transcript whose reasoning was entirely stripped
    still renders one EMPTY block, and counting it would score full survival for the exact case
    this measurement exists to catch.

    Scanned rather than matched by regex because reasoning can CONTAIN a balanced
    ``<think>...</think>`` -- a turn reasoning about the tag format -- and the template renders that
    verbatim inside the outer block. No regular expression tracks that nesting: it either ends the
    span at the inner closer, understating where the block really ends, or lets an unguarded body
    run through a closing tag and merge two adjacent empty blocks into a phantom survivor. Depth
    counting gets both right, and the outermost span is the one the template treats as this turn's
    reasoning.
    """
    spans: list[tuple[int, int]] = []
    depth = 0
    start = 0
    for match in _THINK_DELIMITER.finditer(text):
        if match.group() == _THINK_OPEN:
            if depth == 0:
                start = match.start()
            depth += 1
        elif depth:
            depth -= 1
            if depth == 0 and text[start + len(_THINK_OPEN) : match.start()].strip():
                spans.append((start, match.end()))
    return spans


# why a run resolved to `exact-unpacked`, keyed by the architecture label the packing decision
# froze alongside the mode (`sft_workload._packing_mode`). that label is the only part of the
# decision that survives into the profile, so it is the only place the reason can come from.
_UNPACKED_REASONS = {
    "multimodal": "this run is multimodal, and image rows are never packed",
    "gdn-hybrid": (
        "this model is a gated-delta-net hybrid and the installed stack cannot reset the "
        "linear-attention recurrence at example boundaries, so packed examples would bleed state"
    ),
    "unsupported": "this model architecture has no boundary-safe packing path in flash",
    "pure-attention": "packing was disabled for this run",
}


def unpacked_batch_warning(
    *,
    packing_mode: str,
    architecture_mode: str,
    examples_per_update: int,
    configured_batch_size: object = None,
) -> str | None:
    """One user-facing line for an sft run whose packing mode pins the optimizer batch to 1.

    ``exact-unpacked`` is the boundary-safe design (see
    ``sft_workload._resolve_sft_step_horizon``), and it overrides the authored ``batch_size``.
    Returns None when packing is on, or when nothing was overridden because the authored batch
    was already 1. An omitted ``configured_batch_size`` resolves to the recipe default, which is
    the batch the run would otherwise have used and the one packing discarded.

    The horizon is deliberately not described here: ``train.max_steps`` outranks epochs over rows
    (``_resolve_sft_step_horizon``), and this helper is not given either, so any step-count claim
    would be wrong for a ``max_steps`` run.
    """
    from flash.engine.plan.recipe import RECIPE

    if packing_mode == "packed" or examples_per_update > 1:
        return None
    try:
        batch = (
            int(configured_batch_size)
            if configured_batch_size is not None
            else int(RECIPE.sft.effective_batch)
        )
    except (TypeError, ValueError):
        batch = 0
    if batch == 1:
        return None
    reason = _UNPACKED_REASONS.get(
        architecture_mode, f"packing is unavailable for architecture {architecture_mode!r}"
    )
    # an omitted batch_size resolved to the recipe default above; calling that "configured"
    # would send the reader hunting for a knob their toml never set.
    source = "configured" if configured_batch_size is not None else "default"
    authored = f"the {source} batch_size {batch}" if batch > 1 else f"the {source} batch_size"
    return (
        f"sequence packing is OFF for this SFT run ({architecture_mode}): {reason}. "
        f"every optimizer update therefore trains exactly 1 example, so {authored} no longer "
        "groups examples into an update. it is not inert: it still keys the workload estimate and "
        "sizes the gpu for an auto-sized run, so changing it can change the quote and move the card. "
        "the default learning rate is tuned for a batched update: expect noisier steps, "
        "and lower train.learning_rate if you are comparing against a packed run."
    )


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
    """
    prefix = "flashreasoningmark"
    while prefix in text:
        prefix += "x"
    return prefix


def _reasoning_body_offset(text: str) -> int | None:
    """Where this turn's reasoning BODY starts, by the template's own rule.

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
    """
    close = text.find(_THINK_CLOSE)
    if close < 0:
        return None
    open_at = text.rfind(_THINK_OPEN, 0, close)
    body = 0 if open_at < 0 else open_at + len(_THINK_OPEN)
    return body if text[body:close].strip() else None


def _marked_inline_reasoning(content: object, marker: str) -> object:
    """``content`` with ``marker`` placed at the start of the reasoning the template will keep."""
    if isinstance(content, str):
        offset = _reasoning_body_offset(content)
        return content if offset is None else content[:offset] + marker + content[offset:]
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
    offset = _reasoning_body_offset("".join(text or "" for text in texts))
    if offset is None:
        return content
    marked: list = []
    consumed = 0
    placed = False
    for block, text in zip(content, texts, strict=True):
        if text is None or placed:
            marked.append(block)
            continue
        local = offset - consumed
        consumed += len(text)
        if 0 <= local <= len(text):
            marked.append({**block, "text": text[:local] + marker + text[local:]})
            placed = True
        else:
            marked.append(block)
    return marked


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
    """
    marked: list[dict] = []
    for index, message in enumerate(messages):
        copied = dict(message)
        if copied.get("role") == "assistant" and reasoned_assistant_turns([message]):
            marker = f"{prefix}{index} "
            reasoning = copied.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                # the template reads this field in preference to an inline span, so it is the
                # reasoning and the marker belongs in it
                copied["reasoning_content"] = marker + reasoning
            else:
                copied["content"] = _marked_inline_reasoning(copied.get("content"), marker)
        marked.append(copied)
    return marked


def reasoning_span_end_offsets(text: str) -> list[int]:
    """Character offsets just past each non-empty ``<think>`` span's closing tag.

    A truncated row keeps a rendered span only when the whole block fits inside the retained
    tokens, so the caller needs where each span ENDS to compare against the cap.
    """
    return [end for _start, end in reasoning_spans(text)]


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
    """
    turns = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            turns += 1
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


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _digest_mapping(value: object) -> str:
    return _sha256(value if isinstance(value, dict) else {})


def _digest_sequence(value: object) -> str:
    """Digest an ordered sequence of strings, mirroring ``_digest_mapping`` for list-shaped fields.

    ``[environment] pip`` is part of profile identity because those packages are installed into the
    worker alongside the training stack: a different dependency set can change tokenization,
    collation, or scorer cost, so two configs differing only here must not share a measured profile.
    Order is preserved rather than sorted -- pip resolves earlier entries first, so a reordering is a
    different install.
    """
    return _sha256([str(item) for item in value] if isinstance(value, (list, tuple)) else [])


def sft_profile_input_payload(
    spec: Any,
    *,
    tokenizer_revision: str,
    producer_version: str,
) -> dict[str, object]:
    """Return the non-secret immutable inputs that determine one sft workload profile."""
    train = spec.train
    environment = spec.environment
    return {
        "schema_version": WORKLOAD_PROFILE_SCHEMA_VERSION,
        "kind": SFT_PROFILE_KIND,
        "packing_policy_version": SFT_PACKING_POLICY_VERSION,
        "producer_version": str(producer_version),
        "environment": {
            "id": str(environment.id),
            "resolved_sha": str(environment.resolved_sha or ""),
            "params_sha256": _digest_mapping(environment.params),
            "pip_sha256": _digest_sequence(environment.pip),
        },
        "model": {
            "id": str(spec.model),
            "revision": str(spec.model_revision or ""),
            "tokenizer_revision": str(tokenizer_revision),
        },
        "seed": int(spec.seed),
        "thinking": bool(spec.thinking),
        "train": {
            "epochs": train.epochs,
            "batch_size": train.batch_size,
            "max_context_tokens": train.max_context_tokens,
            "max_steps": train.max_steps,
            "max_examples": train.max_examples,
        },
    }


def sft_profile_input_digest(
    spec: Any,
    *,
    tokenizer_revision: str,
    producer_version: str,
) -> str:
    return _sha256(
        sft_profile_input_payload(
            spec,
            tokenizer_revision=tokenizer_revision,
            producer_version=producer_version,
        )
    )


def _profile_from_dict(cls, raw: object):
    """rebuild a profile and verify its serialized digest.

    shared envelope validation uses each class's own fields and ``content_digest`` so profile types
    cannot drift into accepting different identities.
    """
    if not isinstance(raw, dict):
        raise ValueError("workload profile must be an object")
    data = dict(raw)
    digest = data.pop("content_digest", None)
    if not isinstance(digest, str):
        raise ValueError("workload profile has no content digest")
    if set(data) != set(cls.__dataclass_fields__):
        raise ValueError("workload profile fields do not match the schema")
    profile = cls(**data)
    if profile.content_digest != digest:
        raise ValueError("workload profile content digest does not match")
    return profile


@dataclass(frozen=True)
class SftWorkloadProfile:
    """describe an sft estimate derived from packaged records and the static training contract.

    tokens, retention, truncation, and step horizon share one token stream. other environment prompt
    transformations can still change all four during training. ``created_at`` is provenance and stays
    outside ``_content()`` so repeated preprocessing yields the same profile identity.
    """

    input_digest: str
    producer_version: str
    tokenizer_revision: str
    environment_id: str
    environment_revision: str
    source_examples: int
    selected_examples: int
    retained_examples: int
    dropped_examples: int
    epochs: int
    max_length: int
    packing_mode: str
    architecture_mode: str
    packed_blocks: int
    real_tokens_per_epoch: int
    supervised_tokens_per_epoch: int
    padded_compute_tokens_per_epoch: int
    authoritative_real_tokens: int
    authoritative_supervised_tokens: int
    authoritative_compute_tokens: int
    realized_max_length: int
    # measured before truncation, so it reports what the rows need rather than what the cap
    # allowed. realized_max_length saturates at max_length exactly when the cap binds, which is
    # the one case where the distribution is censored and the number stops being informative.
    untruncated_max_length: int
    truncated_examples: int
    examples_per_update: int
    derived_steps: int
    authoritative_steps: int
    packing_efficiency: float
    sample_policy: str
    # reasoning authored by the retained rows' assistant turns, and how much of it the chat template
    # actually renders into the supervised span. carried on the profile rather than only printed
    # where it is measured: control-plane profiling runs inside the server, so its stderr is not the
    # submitting client's. the CLI renders the warning from these two counts, like the packing
    # override, and a user sees it before a training gpu is allocated.
    #
    # `compare=False` keeps them out of `_content()`, so they change neither the content digest nor
    # worker parity. they are a property of the dataset's rendered text rather than of the token and
    # step contract the quote freezes, and the worker legitimately measures different counts because
    # it executes environment.py while the estimate reads raw records -- folding them into parity
    # would fire the drift warning on a run whose billing contract never moved.
    authored_reasoning_turns: int = field(default=0, compare=False)
    rendered_reasoning_spans: int = field(default=0, compare=False)
    truncated_reasoning_spans: int = field(default=0, compare=False)
    schema_version: int = WORKLOAD_PROFILE_SCHEMA_VERSION
    kind: str = SFT_PROFILE_KIND
    # unix seconds stamped by the control-plane profile producer. 0.0 on worker recomputation.
    created_at: float = field(default=0.0, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != WORKLOAD_PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported workload profile schema version")
        if self.kind != SFT_PROFILE_KIND:
            raise ValueError("sft workload profile kind must be 'sft'")
        if len(self.input_digest) != 64 or any(
            c not in "0123456789abcdef" for c in self.input_digest
        ):
            raise ValueError("input_digest must be a lowercase sha256 hex digest")
        if not self.producer_version:
            raise ValueError("producer_version is required")
        if not self.tokenizer_revision:
            raise ValueError("tokenizer_revision is required")
        if not self.environment_id:
            raise ValueError("environment_id is required")
        if not self.environment_revision:
            raise ValueError("environment_revision is required")
        counts = {
            "source_examples": self.source_examples,
            "selected_examples": self.selected_examples,
            "retained_examples": self.retained_examples,
            "dropped_examples": self.dropped_examples,
            "packed_blocks": self.packed_blocks,
            "real_tokens_per_epoch": self.real_tokens_per_epoch,
            "supervised_tokens_per_epoch": self.supervised_tokens_per_epoch,
            "padded_compute_tokens_per_epoch": self.padded_compute_tokens_per_epoch,
            "authoritative_real_tokens": self.authoritative_real_tokens,
            "authoritative_supervised_tokens": self.authoritative_supervised_tokens,
            "authoritative_compute_tokens": self.authoritative_compute_tokens,
            "realized_max_length": self.realized_max_length,
            "untruncated_max_length": self.untruncated_max_length,
            "truncated_examples": self.truncated_examples,
            "examples_per_update": self.examples_per_update,
            "derived_steps": self.derived_steps,
            "authoritative_steps": self.authoritative_steps,
            "authored_reasoning_turns": self.authored_reasoning_turns,
            "rendered_reasoning_spans": self.rendered_reasoning_spans,
            "truncated_reasoning_spans": self.truncated_reasoning_spans,
        }
        for name, value in counts.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.max_length < 1:
            raise ValueError("max_length must be positive")
        if self.selected_examples > self.source_examples:
            raise ValueError("selected examples cannot exceed source examples")
        if self.retained_examples + self.dropped_examples != self.selected_examples:
            raise ValueError("retained plus dropped examples must equal selected examples")
        if self.retained_examples < 1:
            raise ValueError("at least one sft example must be retained")
        if self.packed_blocks < 1:
            raise ValueError("at least one packed block is required")
        if self.real_tokens_per_epoch < self.retained_examples:
            raise ValueError("real token count cannot be smaller than retained example count")
        if self.supervised_tokens_per_epoch > self.real_tokens_per_epoch:
            raise ValueError("supervised tokens cannot exceed real tokens")
        if self.padded_compute_tokens_per_epoch < self.real_tokens_per_epoch:
            raise ValueError("padded compute tokens cannot be smaller than real tokens")
        if self.authoritative_real_tokens < 1:
            raise ValueError("authoritative real tokens must be positive")
        if self.authoritative_supervised_tokens > self.authoritative_real_tokens:
            raise ValueError("authoritative supervised tokens cannot exceed real tokens")
        if self.authoritative_compute_tokens < self.authoritative_real_tokens:
            raise ValueError("authoritative compute tokens cannot be smaller than real tokens")
        if not 1 <= self.realized_max_length <= self.max_length:
            raise ValueError("realized max length must be within the configured maximum")
        # the untruncated measurement is unbounded above (that is the point), but it can never be
        # SMALLER than the truncated one it was measured alongside.
        if self.untruncated_max_length < self.realized_max_length:
            raise ValueError("untruncated max length cannot be smaller than realized max length")
        if self.truncated_examples > self.retained_examples:
            raise ValueError("truncated examples cannot exceed retained examples")
        # a truncated row is exactly a row longer than the cap, so the two measurements must agree
        # on whether the cap bound at all.
        if bool(self.truncated_examples) != (self.untruncated_max_length > self.max_length):
            raise ValueError("truncated example count disagrees with the untruncated max length")
        if self.examples_per_update < 1:
            raise ValueError("examples per update must be positive")
        if self.derived_steps < 1 or self.authoritative_steps < 1:
            raise ValueError("step counts must be positive")
        if self.packing_mode not in {"packed", "exact-unpacked"}:
            raise ValueError("packing_mode must be 'packed' or 'exact-unpacked'")
        if not self.architecture_mode:
            raise ValueError("architecture_mode is required")
        if not math.isfinite(self.packing_efficiency) or not 0 < self.packing_efficiency <= 1:
            raise ValueError("packing_efficiency must be finite and in (0, 1]")
        expected_efficiency = self.real_tokens_per_epoch / self.padded_compute_tokens_per_epoch
        if not math.isclose(self.packing_efficiency, expected_efficiency, rel_tol=1e-12):
            raise ValueError("packing_efficiency does not match token aggregates")
        if self.sample_policy not in {SFT_SAMPLE_POLICY_FULL, SFT_SAMPLE_POLICY_PREFIX}:
            raise ValueError("sample_policy must be 'exact-full' or 'exact-prefix'")
        if self.sample_policy == SFT_SAMPLE_POLICY_FULL and (
            self.selected_examples != self.source_examples
        ):
            raise ValueError("an exact-full profile must select every source example")
        if isinstance(self.created_at, bool) or not isinstance(self.created_at, (int, float)):
            raise ValueError("created_at must be a number")
        if not math.isfinite(self.created_at) or self.created_at < 0:
            raise ValueError("created_at must be a non-negative unix timestamp")

    def _content(self) -> dict[str, object]:
        """The measurement alone: what the content digest and worker parity are taken over."""
        return {name: getattr(self, name) for name in _measurement_field_names()}

    @property
    def content_digest(self) -> str:
        """digest of the estimate content, stable across equivalent control-plane computations."""
        return _sha256(self._content())

    def to_dict(self) -> dict[str, object]:
        return {
            **self._content(),
            "created_at": float(self.created_at),
            # serialized but digest-free, so the client can render the reasoning-loss warning from
            # the quote without the counts entering the frozen contract
            "authored_reasoning_turns": int(self.authored_reasoning_turns),
            "rendered_reasoning_spans": int(self.rendered_reasoning_spans),
            "truncated_reasoning_spans": int(self.truncated_reasoning_spans),
            "content_digest": self.content_digest,
        }

    @classmethod
    def from_dict(cls, raw: object) -> SftWorkloadProfile:
        return _profile_from_dict(cls, raw)


def _measurement_field_names(cls: type | None = None) -> tuple[str, ...]:
    """return measurement fields, excluding ``compare=False`` provenance.

    deriving digest content from the equality flag keeps parity checks and identity in sync.
    """
    return tuple(f.name for f in fields(cls or SftWorkloadProfile) if f.compare)


def rollout_profile_input_payload(
    spec: Any,
    *,
    tokenizer_revision: str,
    producer_version: str,
) -> dict[str, object]:
    """return immutable non-secret inputs for one rollout profile.

    exclude training horizon because it changes sample count, not the completion distribution. include
    every setting that changes prompts or generation behavior.
    """
    train = spec.train
    environment = spec.environment
    return {
        "schema_version": WORKLOAD_PROFILE_SCHEMA_VERSION,
        "kind": str(spec.algorithm),
        "sample_policy_version": ROLLOUT_SAMPLE_POLICY_VERSION,
        "producer_version": str(producer_version),
        "environment": {
            "id": str(environment.id),
            "resolved_sha": str(environment.resolved_sha or ""),
            "params_sha256": _digest_mapping(environment.params),
            "pip_sha256": _digest_sequence(environment.pip),
        },
        "model": {
            "id": str(spec.model),
            "revision": str(spec.model_revision or ""),
            "tokenizer_revision": str(tokenizer_revision),
        },
        "seed": int(spec.seed),
        "thinking": bool(spec.thinking),
        "generation": {
            "max_completion_tokens": train.max_completion_tokens,
            "max_context_tokens": train.max_context_tokens,
            "group_size": train.group_size,
            "teacher_model": str(getattr(train, "teacher_model", "") or ""),
            # temperature moves the length distribution this profile exists to measure, so a
            # reading taken at one setting must never be reused for a run at another. None is
            # keyed distinctly from any float: it means "whatever the backend defaults to",
            # which is not knowably the same distribution as an explicit value.
            "temperature": (None if train.temperature is None else float(train.temperature)),
        },
    }


def rollout_profile_input_digest(
    spec: Any,
    *,
    tokenizer_revision: str,
    producer_version: str,
) -> str:
    return _sha256(
        rollout_profile_input_payload(
            spec,
            tokenizer_revision=tokenizer_revision,
            producer_version=producer_version,
        )
    )


@dataclass(frozen=True)
class RolloutWorkloadProfile:
    """describe sampled generation and grading for one grpo/opd step.

    unlike exact sft census data, sampled evidence needs a trust verdict. token distributions are
    digest-keyed and do not expire; provider/card latency carries ``measured_at`` and ages out after
    ``ROLLOUT_LATENCY_MAX_AGE_S``.

    completion-token aggregates price realized work instead of the capacity cap. the profile stores no
    prompts, completions, token ids, or credentials.
    """

    input_digest: str
    producer_version: str
    tokenizer_revision: str
    environment_id: str
    environment_revision: str
    kind: str
    # what was sampled, and how much of it survived. a profile whose successes are mostly failures
    # measured the failure path, not the workload.
    sampled_prompts: int
    completed_rollouts: int
    failed_rollouts: int
    # realized generation. the distribution, not a point estimate: a mean alone cannot tell a
    # uniformly-short workload from a bimodal one whose long tail sets the step time.
    completion_tokens_mean: float
    completion_tokens_p50: int
    completion_tokens_p90: int
    completion_tokens_max: int
    prompt_tokens_mean: float
    # how many completions ended because the model stopped rather than because it ran out of room.
    # a high truncation rate means the cap IS binding and the realized distribution is censored,
    # which is the one case where billing the cap is not wrong.
    truncated_rollouts: int
    eos_rollouts: int
    # measured seconds. these age.
    generation_seconds_per_completion: float
    reward_seconds_per_completion: float
    reward_samples: int
    reward_failures: int
    reference_gpu: str
    reference_provider: str
    sample_policy: str
    sample_policy_version: int = ROLLOUT_SAMPLE_POLICY_VERSION
    schema_version: int = WORKLOAD_PROFILE_SCHEMA_VERSION
    # provenance, outside the content digest for the same reason sft's created_at is: a worker
    # re-deriving the measurement must reproduce the digest, and a timestamp never would.
    created_at: float = field(default=0.0, compare=False)
    measured_at: float = field(default=0.0, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != WORKLOAD_PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported workload profile schema version")
        if self.sample_policy_version != ROLLOUT_SAMPLE_POLICY_VERSION:
            raise ValueError("unsupported rollout sample policy version")
        if self.kind not in ROLLOUT_PROFILE_KINDS:
            raise ValueError("rollout workload profile kind must be 'grpo' or 'opd'")
        if len(self.input_digest) != 64 or any(
            c not in "0123456789abcdef" for c in self.input_digest
        ):
            raise ValueError("input_digest must be a lowercase sha256 hex digest")
        for name in ("producer_version", "tokenizer_revision", "environment_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        if not self.environment_revision:
            raise ValueError("environment_revision is required")
        if not self.sample_policy:
            raise ValueError("sample_policy is required")
        counts = {
            "sampled_prompts": self.sampled_prompts,
            "completed_rollouts": self.completed_rollouts,
            "failed_rollouts": self.failed_rollouts,
            "completion_tokens_p50": self.completion_tokens_p50,
            "completion_tokens_p90": self.completion_tokens_p90,
            "completion_tokens_max": self.completion_tokens_max,
            "truncated_rollouts": self.truncated_rollouts,
            "eos_rollouts": self.eos_rollouts,
            "reward_samples": self.reward_samples,
            "reward_failures": self.reward_failures,
        }
        for name, value in counts.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        rates = {
            "completion_tokens_mean": self.completion_tokens_mean,
            "prompt_tokens_mean": self.prompt_tokens_mean,
            "generation_seconds_per_completion": self.generation_seconds_per_completion,
            "reward_seconds_per_completion": self.reward_seconds_per_completion,
        }
        for name, value in rates.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.completion_tokens_p90 < self.completion_tokens_p50:
            raise ValueError("p90 completion tokens cannot be below p50")
        if self.completion_tokens_max < self.completion_tokens_p90:
            raise ValueError("max completion tokens cannot be below p90")
        if self.truncated_rollouts + self.eos_rollouts > self.completed_rollouts:
            raise ValueError("truncated plus eos rollouts cannot exceed completed rollouts")
        if self.reward_failures > self.reward_samples:
            raise ValueError("reward failures cannot exceed reward samples")
        for name in ("created_at", "measured_at"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative unix timestamp")

    @property
    def truncation_rate(self) -> float:
        """Share of completions that hit the cap instead of stopping on their own."""
        if self.completed_rollouts <= 0:
            return 0.0
        return self.truncated_rollouts / self.completed_rollouts

    def trustworthy(
        self, *, now: float, min_rollouts: int = MIN_TRUSTWORTHY_ROLLOUTS
    ) -> tuple[bool, str]:
        """return whether this evidence may move a quote, or the refusal reason.

        centralizing the reason keeps callers consistent. success alone is insufficient when samples
        are too thin, stale, censored, or failure-heavy.
        """
        if self.completed_rollouts < min_rollouts:
            return False, (
                f"only {self.completed_rollouts} rollout(s) completed, "
                f"below the {min_rollouts} needed to quote from"
            )
        if self.failed_rollouts >= self.completed_rollouts:
            return False, (
                f"{self.failed_rollouts} rollout(s) failed against "
                f"{self.completed_rollouts} completed; the sample describes the failure path"
            )
        if self.completion_tokens_max <= 0:
            return False, "every sampled completion was empty, so no generation was measured"
        # truncation reports the cap instead of true completion length and biases means downward.
        # tolerate a light capped tail, but reject heavily censored samples that no longer describe the
        # distribution.
        if self.truncation_rate > MAX_TRUSTWORTHY_TRUNCATION_RATE:
            return False, (
                f"{self.truncated_rollouts} of {self.completed_rollouts} sampled completions were "
                f"truncated at the cap ({self.truncation_rate:.0%}, above the "
                f"{MAX_TRUSTWORTHY_TRUNCATION_RATE:.0%} ceiling); the measured mean is censored "
                "and would underbill generation"
            )
        age = now - self.measured_at
        if self.measured_at <= 0 or age > ROLLOUT_LATENCY_MAX_AGE_S:
            return False, (
                "measured generation latency is older than "
                f"{ROLLOUT_LATENCY_MAX_AGE_S // 3600}h and must be re-measured"
            )
        return True, ""

    def _content(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in _measurement_field_names(type(self))}

    @property
    def content_digest(self) -> str:
        return _sha256(self._content())

    def to_dict(self) -> dict[str, object]:
        return {
            **self._content(),
            "created_at": float(self.created_at),
            "measured_at": float(self.measured_at),
            "content_digest": self.content_digest,
        }

    @classmethod
    def from_dict(cls, raw: object) -> RolloutWorkloadProfile:
        return _profile_from_dict(cls, raw)


class WorkloadProfileMismatch(ValueError):
    """report a profile absent, malformed, or mismatched to this exact spec.

    identity is deterministic, unlike transient hub or infrastructure failures, so callers fail fast.
    """


def require_matching_sft_profile(
    raw: object,
    *,
    input_digest: str,
    producer_version: str,
    tokenizer_revision: str,
) -> SftWorkloadProfile:
    try:
        profile = SftWorkloadProfile.from_dict(raw)
    except ValueError as exc:
        raise WorkloadProfileMismatch(str(exc)) from exc
    if profile.input_digest != input_digest:
        raise WorkloadProfileMismatch("workload profile input digest does not match")
    if profile.producer_version != producer_version:
        raise WorkloadProfileMismatch("workload profile producer version does not match")
    if profile.tokenizer_revision != tokenizer_revision:
        raise WorkloadProfileMismatch("workload profile tokenizer revision does not match")
    return profile


def require_matching_rollout_profile(
    raw: object,
    *,
    input_digest: str,
    producer_version: str,
    tokenizer_revision: str,
    now: float,
) -> RolloutWorkloadProfile:
    """require a matching rollout profile and a passing trust verdict.

    checking both together prevents quoters from accepting exact-identity profiles with no evidence.
    """
    try:
        profile = RolloutWorkloadProfile.from_dict(raw)
    except ValueError as exc:
        raise WorkloadProfileMismatch(str(exc)) from exc
    if profile.input_digest != input_digest:
        raise WorkloadProfileMismatch("workload profile input digest does not match")
    if profile.producer_version != producer_version:
        raise WorkloadProfileMismatch("workload profile producer version does not match")
    if profile.tokenizer_revision != tokenizer_revision:
        raise WorkloadProfileMismatch("workload profile tokenizer revision does not match")
    ok, reason = profile.trustworthy(now=now)
    if not ok:
        raise WorkloadProfileMismatch(f"rollout workload profile is not trustworthy: {reason}")
    return profile
