"""Immutable workload profiles used to freeze evidence-backed cost quotes."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields
from typing import Any

# re-exported: identifying reasoning in a render is its own problem and lives in a sibling module,
# but the profile's callers import these from here.
from flash.engine.profiling.reasoning_render import (  # noqa: F401
    horizon_row_count,
    marked_reasoning_end,
    reasoned_assistant_turns,
    reasoning_marker_prefix,
    reasoning_markers,
    reasoning_warning_rows,
    rendered_reasoning_loss_warning,
    strip_reasoning_markers,
    with_marked_reasoning,
)

SFT_PROFILE_KIND = "sft"
# 3 removes the deleted per-run worker environment map from the profile identity payload. 4 adds
# the authored [environment] pip digest, which changes the installed worker stack and so cannot be
# absent from identity. old cached profiles use a different identity shape, so reject them and
# re-profile. 5 serializes the reasoning-loss counts so the submitting client can render that
# warning; they are digest-free, but `from_dict` requires an exact field set, so a profile cached
# under 4 cannot be rebuilt and has to re-profile. 6 adds truncated_reasoning_spans, splitting
# cap-truncated reasoning out from template-stripped reasoning so the warning names the right
# remedy; same exact-field-set reason a 5 profile cannot be rebuilt. 7 serializes reasoning_rows,
# the denominator those counts were totalled over: under 6 the counts were whole-dataset, so a
# reader had to infer the denominator and could pair bounded counts with the wrong row count.
WORKLOAD_PROFILE_SCHEMA_VERSION = 7
# 2 lets a gdn hybrid pack when the installed stack proves it can reset example boundaries, where 1
# always answered exact-unpacked. the same config therefore resolves to a different packing_mode and
# examples_per_update, so a profile cached under 1 quotes a step count this policy would not: it has
# to be a different identity rather than a cache hit.
SFT_PACKING_POLICY_VERSION = 2
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
    # how many retained rows the three counts above were totalled over. serialized rather than
    # re-derived because a reader cannot tell a bounded count from a whole-dataset one by looking:
    # both shapes carry examples_per_update and authoritative_steps, so inferring the denominator
    # from them would pair a binding horizon with counts that predate the bounding.
    reasoning_rows: int = field(default=0, compare=False)
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
            "reasoning_rows": self.reasoning_rows,
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
        # the reasoning counts are totalled over a prefix of the retained rows, so their denominator
        # cannot name more rows than were retained.
        if self.reasoning_rows > self.retained_examples:
            raise ValueError("reasoning rows cannot exceed retained examples")
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
            "reasoning_rows": int(self.reasoning_rows),
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
