"""Exact-prefix cross-tokenizer projection and sparse conditional soft-target loss."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from flash.engine.worker.forward_teacher import ForwardTeacherSemanticRecord

REPORTED_MASS_TOLERANCE = 1e-6


class SoftTargetProjectionError(RuntimeError):
    """A sanitized structural projection failure."""


@dataclass(frozen=True)
class ProjectionDropCounts:
    zero_token: int = 0
    multi_token: int = 0
    prefix_retokenization: int = 0
    special_token: int = 0
    invalid_token_id: int = 0
    round_trip_mismatch: int = 0
    realized_multi_token: int = 0

    @property
    def total(self) -> int:
        return (
            self.zero_token
            + self.multi_token
            + self.prefix_retokenization
            + self.special_token
            + self.invalid_token_id
            + self.round_trip_mismatch
            + self.realized_multi_token
        )

    def merged(self, other: ProjectionDropCounts) -> ProjectionDropCounts:
        return ProjectionDropCounts(
            zero_token=self.zero_token + other.zero_token,
            multi_token=self.multi_token + other.multi_token,
            prefix_retokenization=self.prefix_retokenization + other.prefix_retokenization,
            special_token=self.special_token + other.special_token,
            invalid_token_id=self.invalid_token_id + other.invalid_token_id,
            round_trip_mismatch=self.round_trip_mismatch + other.round_trip_mismatch,
            realized_multi_token=self.realized_multi_token + other.realized_multi_token,
        )


@dataclass(frozen=True)
class ProjectedPosition:
    provider_record_index: int
    logits_index: int | None
    token_ids: tuple[int, ...]
    probabilities: tuple[float, ...]
    reported_top_k_mass: float
    retained_projected_mass: float
    rejected_reported_mass: float
    unreported_mass: float
    total_dropped_mass: float
    support_size: int
    collision_count: int
    conditional_entropy_nats: float
    drop_counts: ProjectionDropCounts
    provider_top_k_entropy_nats: float = 0.0

    @property
    def eligible(self) -> bool:
        return self.logits_index is not None and self.support_size > 0


@dataclass(frozen=True)
class ProjectedTarget:
    input_ids: tuple[int, ...]
    positions: tuple[ProjectedPosition, ...]
    rows: tuple[ProjectedPosition, ...]
    visible_position_count: int
    eligible_row_count: int
    drop_counts: ProjectionDropCounts


@dataclass(frozen=True)
class _CandidateProjection:
    token_id: int | None
    drop: str | None


def _encode(tokenizer, text: str) -> tuple[int, ...]:
    try:
        encoded = tokenizer(text, add_special_tokens=False)
        values = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
        if isinstance(values, (str, bytes)):
            raise TypeError
        return tuple(int(value) for value in values)
    except Exception:
        raise SoftTargetProjectionError("soft-target tokenizer encoding failed") from None


def _decode(tokenizer, token_ids: tuple[int, ...]) -> str:
    try:
        try:
            decoded = tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    except Exception:
        raise SoftTargetProjectionError("soft-target tokenizer decoding failed") from None
    if not isinstance(decoded, str):
        raise SoftTargetProjectionError("soft-target tokenizer decoding failed")
    return decoded


def _tokenizer_size(tokenizer) -> int:
    try:
        value = len(tokenizer)
    except Exception:
        value = None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SoftTargetProjectionError("soft-target tokenizer vocabulary size is unavailable")
    return value


def _special_ids(tokenizer) -> frozenset[int]:
    values = getattr(tokenizer, "all_special_ids", ()) or ()
    try:
        return frozenset(int(value) for value in values)
    except Exception:
        raise SoftTargetProjectionError("soft-target tokenizer special ids are malformed") from None


def _preserves_prefix(previous: tuple[int, ...], extended: tuple[int, ...]) -> bool:
    return len(extended) >= len(previous) and extended[: len(previous)] == previous


def _candidate_projection(
    tokenizer,
    *,
    prefix_text: str,
    prefix_ids: tuple[int, ...],
    candidate_text: str,
    tokenizer_size: int,
    special_ids: frozenset[int],
) -> _CandidateProjection:
    extended_text = prefix_text + candidate_text
    extended_ids = _encode(tokenizer, extended_text)
    if not _preserves_prefix(prefix_ids, extended_ids):
        return _CandidateProjection(None, "prefix_retokenization")
    added = extended_ids[len(prefix_ids) :]
    if not added:
        return _CandidateProjection(None, "zero_token")
    if len(added) != 1:
        return _CandidateProjection(None, "multi_token")
    token_id = added[0]
    if token_id < 0 or token_id >= tokenizer_size:
        return _CandidateProjection(None, "invalid_token_id")
    if token_id in special_ids:
        return _CandidateProjection(None, "special_token")
    try:
        decoded_full = _decode(tokenizer, extended_ids)
        round_trip_ids = _encode(tokenizer, decoded_full)
    except SoftTargetProjectionError:
        return _CandidateProjection(None, "round_trip_mismatch")
    if decoded_full != extended_text or round_trip_ids != extended_ids:
        return _CandidateProjection(None, "round_trip_mismatch")
    return _CandidateProjection(token_id, None)


def _increment_drop(counts: ProjectionDropCounts, category: str) -> ProjectionDropCounts:
    values = {
        "zero_token": counts.zero_token,
        "multi_token": counts.multi_token,
        "prefix_retokenization": counts.prefix_retokenization,
        "special_token": counts.special_token,
        "invalid_token_id": counts.invalid_token_id,
        "round_trip_mismatch": counts.round_trip_mismatch,
        "realized_multi_token": counts.realized_multi_token,
    }
    values[category] += 1
    return ProjectionDropCounts(**values)


def _stable_logaddexp(left: float, right: float) -> float:
    high = max(left, right)
    low = min(left, right)
    if high == -math.inf:
        return high
    return high + math.log1p(math.exp(low - high))


def _reported_mass(record: ForwardTeacherSemanticRecord) -> float:
    candidate_logprobs = []
    for candidate in record.top_logprobs:
        value = getattr(candidate, "logprob", None)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) > 0
        ):
            raise SoftTargetProjectionError("soft-target candidate logprob is invalid")
        candidate_logprobs.append(float(value))
    mass = math.fsum(math.exp(logprob) for logprob in candidate_logprobs)
    if not math.isfinite(mass) or mass > 1.0 + REPORTED_MASS_TOLERANCE:
        raise SoftTargetProjectionError("soft-target reported top-k mass is invalid")
    return min(1.0, max(0.0, mass))


def _provider_top_k_entropy(record: ForwardTeacherSemanticRecord) -> float:
    masses = tuple(math.exp(float(candidate.logprob)) for candidate in record.top_logprobs)
    reported_mass = math.fsum(masses)
    if reported_mass <= 0:
        return 0.0
    return -math.fsum(
        probability * math.log(probability)
        for mass in masses
        if (probability := mass / reported_mass) > 0
    )


def _empty_position(
    record: ForwardTeacherSemanticRecord,
    *,
    reported_mass: float,
    provider_entropy: float,
    drops: ProjectionDropCounts,
) -> ProjectedPosition:
    unreported_mass = max(0.0, 1.0 - reported_mass)
    return ProjectedPosition(
        provider_record_index=record.index,
        logits_index=None,
        token_ids=(),
        probabilities=(),
        reported_top_k_mass=reported_mass,
        retained_projected_mass=0.0,
        rejected_reported_mass=reported_mass,
        unreported_mass=unreported_mass,
        total_dropped_mass=1.0,
        support_size=0,
        collision_count=0,
        conditional_entropy_nats=0.0,
        drop_counts=drops,
        provider_top_k_entropy_nats=provider_entropy,
    )


def project_visible_records(
    tokenizer,
    *,
    prefix_text: str,
    visible_records: Sequence[ForwardTeacherSemanticRecord],
) -> ProjectedTarget:
    """Project visible teacher alternatives by exact contextual one-token extension.

    Reported mass may exceed one by at most ``REPORTED_MASS_TOLERANCE`` to absorb floating-point
    summation noise. Larger excess is structurally impossible and rejects the target. Retained support
    is normalized conditionally; omitted mass is never reassigned and no synthetic bucket is created.
    Input ids end immediately after the latest eligible logits position. Targets without eligible rows
    retain the original non-empty prefix as the smallest model-safe representation.
    """

    if not isinstance(prefix_text, str):
        raise SoftTargetProjectionError("soft-target prefix is malformed")
    prefix_ids = _encode(tokenizer, prefix_text)
    if not prefix_ids:
        raise SoftTargetProjectionError("soft-target prefix tokenization is empty")
    if _decode(tokenizer, prefix_ids) != prefix_text:
        raise SoftTargetProjectionError("soft-target prefix round-trip mismatch")
    tokenizer_size = _tokenizer_size(tokenizer)
    special_ids = _special_ids(tokenizer)
    base_prefix_ids = prefix_ids

    positions: list[ProjectedPosition] = []
    aggregate_drops = ProjectionDropCounts()
    current_text = prefix_text
    current_ids = prefix_ids
    for record in visible_records:
        record_kind = getattr(record, "kind", None)
        if getattr(record_kind, "value", None) != "visible_content":
            raise SoftTargetProjectionError("soft-target record kind is invalid")
        if not isinstance(getattr(record, "token", None), str):
            raise SoftTargetProjectionError("soft-target realized token is malformed")
        reported_mass = _reported_mass(record)
        provider_entropy = _provider_top_k_entropy(record)
        extended_text = current_text + record.token
        extended_ids = _encode(tokenizer, extended_text)
        if not _preserves_prefix(current_ids, extended_ids):
            raise SoftTargetProjectionError("soft-target realized prefix retokenization")
        added = extended_ids[len(current_ids) :]
        if not added:
            raise SoftTargetProjectionError("soft-target realized extension is empty")
        if any(token_id < 0 or token_id >= tokenizer_size for token_id in added):
            raise SoftTargetProjectionError("soft-target realized token id is invalid")
        if (
            _decode(tokenizer, extended_ids) != extended_text
            or _encode(tokenizer, extended_text) != extended_ids
        ):
            raise SoftTargetProjectionError("soft-target realized round-trip mismatch")

        if len(added) != 1:
            drops = ProjectionDropCounts(realized_multi_token=1)
            positions.append(
                _empty_position(
                    record,
                    reported_mass=reported_mass,
                    provider_entropy=provider_entropy,
                    drops=drops,
                )
            )
            aggregate_drops = aggregate_drops.merged(drops)
            current_text = extended_text
            current_ids = extended_ids
            continue

        log_mass_by_token: dict[int, float] = {}
        collision_count = 0
        drops = ProjectionDropCounts()
        for candidate in record.top_logprobs:
            candidate_text = getattr(candidate, "token", None)
            if not isinstance(candidate_text, str):
                raise SoftTargetProjectionError("soft-target candidate token is malformed")
            projected = _candidate_projection(
                tokenizer,
                prefix_text=current_text,
                prefix_ids=current_ids,
                candidate_text=candidate_text,
                tokenizer_size=tokenizer_size,
                special_ids=special_ids,
            )
            if projected.drop is not None:
                drops = _increment_drop(drops, projected.drop)
                continue
            token_id = projected.token_id
            if token_id is None:
                raise AssertionError("accepted soft-target candidate has no token id")
            if token_id in log_mass_by_token:
                collision_count += 1
                log_mass_by_token[token_id] = _stable_logaddexp(
                    log_mass_by_token[token_id], candidate.logprob
                )
            else:
                log_mass_by_token[token_id] = candidate.logprob

        token_ids = tuple(log_mass_by_token)
        masses = tuple(math.exp(log_mass_by_token[token_id]) for token_id in token_ids)
        raw_retained_mass = math.fsum(masses)
        retained_mass = min(reported_mass, raw_retained_mass)
        rejected_mass = max(0.0, reported_mass - retained_mass)
        unreported_mass = max(0.0, 1.0 - reported_mass)
        total_dropped_mass = rejected_mass + unreported_mass
        if not math.isfinite(raw_retained_mass) or raw_retained_mass <= 0:
            position = _empty_position(
                record,
                reported_mass=reported_mass,
                provider_entropy=provider_entropy,
                drops=drops,
            )
        else:
            probabilities = tuple(mass / raw_retained_mass for mass in masses)
            entropy = -math.fsum(
                probability * math.log(probability)
                for probability in probabilities
                if probability > 0
            )
            position = ProjectedPosition(
                provider_record_index=record.index,
                logits_index=len(current_ids) - 1,
                token_ids=token_ids,
                probabilities=probabilities,
                reported_top_k_mass=reported_mass,
                retained_projected_mass=retained_mass,
                rejected_reported_mass=rejected_mass,
                unreported_mass=unreported_mass,
                total_dropped_mass=total_dropped_mass,
                support_size=len(token_ids),
                collision_count=collision_count,
                conditional_entropy_nats=entropy,
                drop_counts=drops,
                provider_top_k_entropy_nats=provider_entropy,
            )
        positions.append(position)
        aggregate_drops = aggregate_drops.merged(drops)
        current_text = extended_text
        current_ids = extended_ids

    frozen_positions = tuple(positions)
    rows = tuple(position for position in frozen_positions if position.eligible)
    if rows:
        max_logits_index = max(row.logits_index for row in rows if row.logits_index is not None)
        model_input_ids = current_ids[: max_logits_index + 1]
    else:
        model_input_ids = base_prefix_ids
    return ProjectedTarget(
        input_ids=model_input_ids,
        positions=frozen_positions,
        rows=rows,
        visible_position_count=len(frozen_positions),
        eligible_row_count=len(rows),
        drop_counts=aggregate_drops,
    )


def _validated_probabilities(row: ProjectedPosition) -> tuple[float, ...]:
    if len(row.token_ids) != len(row.probabilities):
        raise ValueError("soft-target support cardinality is inconsistent")
    if any(
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(float(probability))
        or float(probability) < 0
        for probability in row.probabilities
    ):
        raise ValueError("soft-target probabilities are invalid")
    probabilities = tuple(float(probability) for probability in row.probabilities)
    if not math.isclose(math.fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("soft-target probabilities are not normalized")
    return probabilities


def projected_row_is_active(
    row: ProjectedPosition,
    entropy_tau: float | None,
) -> bool:
    """Route one structurally eligible row from pre-projection provider entropy."""
    if not row.eligible:
        raise ValueError("soft-target row is not structurally eligible")
    if entropy_tau is None:
        return True
    if (
        isinstance(entropy_tau, bool)
        or not isinstance(entropy_tau, (int, float))
        or not math.isfinite(float(entropy_tau))
        or float(entropy_tau) < 0
    ):
        raise ValueError("soft-target entropy tau must be finite and >= 0")
    provider_entropy = row.provider_top_k_entropy_nats
    if (
        isinstance(provider_entropy, bool)
        or not isinstance(provider_entropy, (int, float))
        or not math.isfinite(float(provider_entropy))
        or float(provider_entropy) < 0
    ):
        raise ValueError("soft-target provider entropy is invalid")
    return float(provider_entropy) > float(entropy_tau)


def sparse_projected_conditional_cross_entropy(
    logits,
    targets: Sequence[ProjectedTarget],
    *,
    entropy_tau: float | None = None,
):
    """Return a per-target then per-batch mean with optional fixed-denominator routing."""

    import torch
    if logits.ndim != 3:
        raise ValueError("soft-target logits must have shape [batch, positions, vocabulary]")
    if not torch.is_floating_point(logits):
        raise ValueError("soft-target logits must be floating point")
    if len(targets) != logits.shape[0]:
        raise ValueError("soft-target batch size must match logits")
    low_precision = logits.dtype in (torch.float16, torch.bfloat16)
    computation_dtype = torch.float32 if low_precision else logits.dtype
    target_losses = []
    for batch_index, target in enumerate(targets):
        row_losses = []
        for row in target.rows:
            if row.logits_index is None:
                raise ValueError("soft-target logits position is unavailable")
            if row.logits_index < 0 or row.logits_index >= logits.shape[1]:
                raise ValueError("soft-target logits position is out of range")
            if not projected_row_is_active(row, entropy_tau):
                row_losses.append(
                    logits[batch_index, row.logits_index].reshape(-1)[:1].sum() * 0.0
                )
                continue
            probabilities_tuple = _validated_probabilities(row)
            ids = torch.tensor(row.token_ids, dtype=torch.long, device=logits.device)
            if bool(((ids < 0) | (ids >= logits.shape[2])).any()):
                raise ValueError("soft-target token id is out of range")
            probabilities = torch.tensor(
                probabilities_tuple,
                dtype=computation_dtype,
                device=logits.device,
            )
            row_logits = logits[batch_index, row.logits_index]
            if low_precision:
                row_logits = row_logits.float()
            log_probabilities = row_logits.log_softmax(dim=-1)
            row_losses.append(-(probabilities * log_probabilities.index_select(0, ids)).sum())
        target_loss = (
            torch.stack(row_losses).mean()
            if row_losses
            else logits[batch_index].reshape(-1)[:1].sum() * 0.0
        )
        target_losses.append(target_loss)
    loss = (
        torch.stack(target_losses).mean() if target_losses else logits.reshape(-1)[:1].sum() * 0.0
    )
    if not bool(torch.isfinite(loss)):
        raise ValueError("soft-target loss is non-finite")
    return loss
