"""student-token top-k candidate distillation helpers for opd c13."""

from __future__ import annotations

import math
from dataclasses import dataclass

from flash.engine.worker.teacher import TeacherError

TOPK_CANDIDATES = 4
TEACHER_INPUT_CAP_MULTIPLIER = 2.0


@dataclass(frozen=True)
class CandidateGroup:
    surface: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class CandidateRequest:
    context: str
    surface: str
    input_tokens: int


@dataclass(frozen=True)
class CandidateScoreBatch:
    logprobs: dict[tuple[str, str], float]
    admitted_sets: tuple[bool, ...] = ()
    budget_exhausted: bool = False
    cache_hits: int = 0
    requests: int = 0


@dataclass(frozen=True)
class CandidateObjectiveInput:
    term: object | None = None
    selected_prefixes: int = 0
    scored_prefixes: int = 0
    duplicate_surfaces: int = 0
    invalid_candidates: int = 0
    budget_exhausted: bool = False
    cache_hits: int = 0
    requests: int = 0


class TeacherInputMeter:
    """hard meter that caps total input at two ordinary-run equivalents."""

    def __init__(self, multiplier: float = TEACHER_INPUT_CAP_MULTIPLIER) -> None:
        self.multiplier = float(multiplier)
        self.ordinary_tokens = 0
        self.candidate_tokens = 0
        self.budget_exhausted = False

    @property
    def total_tokens(self) -> int:
        return self.ordinary_tokens + self.candidate_tokens

    @property
    def cap_tokens(self) -> int:
        return math.floor(self.multiplier * self.ordinary_tokens)

    def record_ordinary(self, tokens: int) -> None:
        value = int(tokens)
        if value < 0:
            raise ValueError("ordinary teacher input tokens must be non-negative")
        self.ordinary_tokens += value

    def reserve_candidate_set(self, tokens: tuple[int, ...]) -> bool:
        values = tuple(int(value) for value in tokens)
        if any(value < 0 for value in values):
            raise ValueError("candidate teacher input tokens must be non-negative")
        required = sum(values)
        if self.total_tokens + required > self.cap_tokens:
            self.budget_exhausted = True
            return False
        self.candidate_tokens += required
        return True

    def reserve_candidate(self, tokens: int) -> bool:
        return self.reserve_candidate_set((tokens,))


def selected_prefix_indices(completion_length: int, cadence: int) -> tuple[int, ...]:
    if cadence <= 0:
        raise ValueError("opd top-k cadence must be positive")
    return tuple(range(int(cadence), max(0, int(completion_length)), int(cadence)))


def decode_candidate_groups(tok, prefix_ids, logits_row, eos_ids, *, k: int = TOPK_CANDIDATES):
    """select student token ids and group context-aware decoded continuation surfaces."""
    import torch

    if k != TOPK_CANDIDATES:
        raise ValueError(f"c13 requires exactly k={TOPK_CANDIDATES}")
    row = logits_row.detach().float().clone()
    for token_id in eos_ids:
        if 0 <= int(token_id) < row.numel():
            row[int(token_id)] = -torch.inf
    finite = torch.isfinite(row)
    available = int(finite.sum().item())
    if available < k:
        raise RuntimeError(f"c13 requires {k} finite non-eos student candidates, found {available}")
    candidate_ids = torch.topk(row, k=k).indices.tolist()
    prefix = [int(token_id) for token_id in prefix_ids]
    prefix_text = tok.decode(prefix, skip_special_tokens=True)
    grouped: dict[str, list[int]] = {}
    invalid = 0
    for token_id in candidate_ids:
        extended = tok.decode([*prefix, int(token_id)], skip_special_tokens=True)
        if not isinstance(extended, str) or not extended.startswith(prefix_text):
            invalid += 1
            continue
        surface = extended[len(prefix_text) :]
        if not surface or "�" in surface:
            invalid += 1
            continue
        grouped.setdefault(surface, []).append(int(token_id))
    groups = tuple(
        CandidateGroup(surface, tuple(token_ids)) for surface, token_ids in grouped.items()
    )
    duplicates = sum(len(group.token_ids) - 1 for group in groups)
    return groups, duplicates, invalid


def candidate_set_cross_entropy(logits_row, groups, teacher_logprobs):
    """cross entropy on teacher and student distributions normalized over candidate surfaces."""
    import torch

    if not groups or len(groups) != len(teacher_logprobs):
        return None
    row = logits_row.float()
    student_surface_logits = []
    for group in groups:
        ids = torch.tensor(group.token_ids, dtype=torch.long, device=row.device)
        student_surface_logits.append(torch.logsumexp(row.index_select(0, ids), dim=0))
    student_logits = torch.stack(student_surface_logits)
    teacher_logits = torch.tensor(teacher_logprobs, dtype=student_logits.dtype, device=row.device)
    if not bool(torch.isfinite(teacher_logits).all().item()):
        raise TeacherError("candidate teacher scores contain a non-finite logprob", permanent=True)
    teacher_targets = torch.softmax(teacher_logits, dim=0)
    student_logprobs = torch.log_softmax(student_logits, dim=0)
    return -(teacher_targets * student_logprobs).sum()


class CandidateTeacherScorer:
    """exact-request cache and batched teacher scoring under a hard input meter."""

    def __init__(self, teacher, meter: TeacherInputMeter, cache: dict | None = None) -> None:
        self.teacher = teacher
        self.meter = meter
        self.cache: dict[tuple[str, str], float] = cache if cache is not None else {}

    def score(self, request_sets: list[tuple[CandidateRequest, ...]]) -> CandidateScoreBatch:
        values: dict[tuple[str, str], float] = {}
        missing: dict[tuple[str, str], CandidateRequest] = {}
        cache_hit_keys: set[tuple[str, str]] = set()
        admitted_sets: list[bool] = []
        exhausted = False
        for request_set in request_sets:
            unique: dict[tuple[str, str], CandidateRequest] = {}
            for request in request_set:
                unique.setdefault((request.context, request.surface), request)
            uncached = [
                request
                for key, request in unique.items()
                if key not in self.cache and key not in missing
            ]
            if not self.meter.reserve_candidate_set(
                tuple(request.input_tokens for request in uncached)
            ):
                admitted_sets.append(False)
                exhausted = True
                continue
            admitted_sets.append(True)
            for key, request in unique.items():
                if key in self.cache:
                    values[key] = self.cache[key]
                    cache_hit_keys.add(key)
                else:
                    missing.setdefault(key, request)

        if missing:
            requests = list(missing.values())
            scored = self.teacher.score_many(
                [(request.context, request.surface) for request in requests]
            )
            if not isinstance(scored, list) or len(scored) != len(requests):
                raise TeacherError(
                    "candidate teacher batch returned the wrong number of scores",
                    permanent=True,
                )
            validated: dict[tuple[str, str], float] = {}
            for request, tokens in zip(requests, scored, strict=True):
                if not isinstance(tokens, list) or not tokens:
                    raise TeacherError(
                        "candidate teacher response contained no realized completion tokens",
                        permanent=True,
                    )
                try:
                    logprob = sum(float(token.logprob) for token in tokens)
                except (AttributeError, TypeError, ValueError) as exc:
                    raise TeacherError(
                        "candidate teacher response contained an invalid realized token",
                        permanent=True,
                    ) from exc
                if not math.isfinite(logprob) or logprob > 1e-6:
                    raise TeacherError(
                        f"candidate teacher response has invalid surface logprob {logprob!r}",
                        permanent=True,
                    )
                validated[(request.context, request.surface)] = logprob
            self.cache.update(validated)
            values.update(validated)
        return CandidateScoreBatch(
            logprobs=values,
            admitted_sets=tuple(admitted_sets),
            budget_exhausted=exhausted,
            cache_hits=len(cache_hit_keys),
            requests=len(missing),
        )
