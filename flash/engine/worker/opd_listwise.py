"""continuation-level listwise distillation helpers for opd c14."""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

C14_STUDENT_CONTINUATIONS = 2
C14_TEACHER_CANDIDATES = 2
C14_TEACHER_INPUT_MULTIPLIER = 2.0
_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}", re.IGNORECASE)
_PUBMED_ANSWER_RE = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)
_PUBMED_KEYS = ("final_answer", "answer", "label", "answer_key", "correct_answer")


@dataclass(frozen=True)
class ContinuationRecord:
    """typed continuation candidate used to build one prompt-local listwise target."""

    prompt_key: str
    candidate_index: int
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    completion_text: str
    teacher_logprob: float | None
    correctness: float | None
    boxed_format: float
    terminal_eos: float
    no_repetition: float
    no_length_termination: float
    status: str = "ok"

    @property
    def local_utility(self) -> float:
        values = [
            self.boxed_format,
            self.terminal_eos,
            self.no_repetition,
            self.no_length_termination,
        ]
        if self.correctness is not None:
            values.insert(0, self.correctness)
        return float(sum(values))


@dataclass(frozen=True)
class TeacherScoreCacheKey:
    model: str
    prompt: str
    completion: str


class ExactTeacherScoreCache:
    """thread-safe exact cache keyed by model and byte-identical request strings."""

    def __init__(self) -> None:
        self._values: dict[TeacherScoreCacheKey, object] = {}
        self._lock = threading.Lock()

    def get(self, key: TeacherScoreCacheKey) -> object | None:
        with self._lock:
            return self._values.get(key)

    def put(self, key: TeacherScoreCacheKey, value: object) -> None:
        with self._lock:
            self._values[key] = value

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)


class TeacherInputBudget:
    """hard teacher-input budget measured against an ordinary-run token baseline."""

    def __init__(
        self,
        ordinary_run_tokens: int,
        multiplier: float = C14_TEACHER_INPUT_MULTIPLIER,
    ) -> None:
        ordinary = int(ordinary_run_tokens)
        if ordinary < 0:
            raise ValueError("ordinary_run_tokens must be non-negative")
        if not math.isfinite(multiplier) or multiplier < 0:
            raise ValueError("teacher input budget multiplier must be finite and non-negative")
        self.ordinary_run_tokens = ordinary
        self.multiplier = float(multiplier)
        self.cap_tokens = math.floor(ordinary * multiplier)
        self.used_tokens = 0
        self.budget_exhausted = False
        self._lock = threading.Lock()

    def preflight(self, input_tokens: Iterable[int]) -> bool:
        """check the full requested input without assuming any cache hits."""
        requested = sum(max(0, int(tokens)) for tokens in input_tokens)
        with self._lock:
            allowed = self.used_tokens + requested <= self.cap_tokens
            if not allowed:
                self.budget_exhausted = True
            return allowed

    def charge(self, input_tokens: Iterable[int]) -> bool:
        """atomically reserve actual uncached teacher inputs before issuing a call."""
        requested = sum(max(0, int(tokens)) for tokens in input_tokens)
        with self._lock:
            if self.used_tokens + requested > self.cap_tokens:
                self.budget_exhausted = True
                return False
            self.used_tokens += requested
            return True


def robust_within_prompt_normalize(values: Sequence[float]) -> tuple[float, ...]:
    """median-center and max-absolute-scale values, returning zeros for ties."""
    if not values:
        return ()
    numbers = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError("listwise utility values must be finite")
    center = float(median(numbers))
    deviations = tuple(value - center for value in numbers)
    scale = max(abs(value) for value in deviations)
    if scale <= 1e-12:
        return tuple(0.0 for _ in numbers)
    return tuple(value / scale for value in deviations)


def detached_target_distribution(records: Sequence[ContinuationRecord], *, temperature: float = 1.0):
    """build a detached prompt-local target from normalized teacher and local utilities."""
    import torch

    if not records:
        raise ValueError("listwise target requires at least one continuation")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("listwise target temperature must be finite and positive")
    if any(record.teacher_logprob is None or record.status != "ok" for record in records):
        raise ValueError("listwise target requires successful teacher scores")
    teacher = robust_within_prompt_normalize(
        [float(record.teacher_logprob) for record in records if record.teacher_logprob is not None]
    )
    local = robust_within_prompt_normalize([record.local_utility for record in records])
    logits = torch.tensor(
        [
            (teacher_value + local_value) / temperature
            for teacher_value, local_value in zip(teacher, local, strict=True)
        ],
        dtype=torch.float32,
    )
    return torch.softmax(logits, dim=0).detach()


def listwise_sequence_score_loss(sequence_scores, target_distribution):
    """cross-entropy from a detached target to student continuation sequence scores."""
    import torch
    import torch.nn.functional as F

    if sequence_scores.ndim != 1 or target_distribution.ndim != 1:
        raise ValueError("listwise scores and target must be rank-one")
    if sequence_scores.shape != target_distribution.shape:
        raise ValueError("listwise scores and target must have identical shape")
    target = target_distribution.detach().to(
        device=sequence_scores.device, dtype=sequence_scores.dtype
    )
    if not bool(torch.isfinite(sequence_scores.detach()).all().item()):
        raise ValueError("student sequence scores must be finite")
    if not bool(torch.isfinite(target).all().item()):
        raise ValueError("listwise target must be finite")
    return -(target * F.log_softmax(sequence_scores, dim=0)).sum()


def student_sequence_score(rows, token_ids):
    """length-normalized differentiable log-probability of one sampled continuation."""
    import torch
    import torch.nn.functional as F

    if not token_ids:
        raise ValueError("student sequence score requires at least one token")
    ids = torch.tensor(tuple(int(token_id) for token_id in token_ids), device=rows.device)
    return -F.cross_entropy(rows.float(), ids, reduction="mean")


def teacher_sequence_logprob(teacher_tokens: Sequence[Any]) -> float:
    """length-normalized teacher log-probability for one continuation."""
    values = [float(token.logprob) for token in teacher_tokens]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("teacher continuation score requires finite token log-probabilities")
    return sum(values) / len(values)


def _canonical_pubmed_answer(value: object) -> str | None:
    if isinstance(value, str):
        match = _PUBMED_ANSWER_RE.search(value)
        return match.group(1).lower() if match else None
    return None


def pubmedqa_correctness(example: object, completion_text: str) -> float | None:
    """return exact yes/no/maybe correctness when a PubMedQA-style label is available."""
    reference = None
    if isinstance(example, dict):
        for key in _PUBMED_KEYS:
            if key in example and (reference := _canonical_pubmed_answer(example[key])) is not None:
                break
    else:
        for key in _PUBMED_KEYS:
            if hasattr(example, key) and (
                reference := _canonical_pubmed_answer(getattr(example, key))
            ) is not None:
                break
    if reference is None:
        return None
    boxed = _BOXED_RE.findall(completion_text)
    candidate_source = boxed[-1] if boxed else completion_text
    candidate = _canonical_pubmed_answer(candidate_source)
    return float(candidate == reference) if candidate is not None else 0.0


def has_boxed_format(text: str) -> bool:
    return bool(_BOXED_RE.search(text))


def has_repetition(text: str) -> bool:
    """detect repeated lines or repeated 3-grams without penalizing ordinary word reuse."""
    normalized_lines = [" ".join(line.lower().split()) for line in text.splitlines() if line.strip()]
    if len(normalized_lines) != len(set(normalized_lines)):
        return True
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(tokens) < 6:
        return False
    trigrams = [tuple(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
    return len(trigrams) != len(set(trigrams))


def build_continuation_record(
    *,
    prompt_key: str,
    candidate_index: int,
    prompt_ids: Sequence[int],
    completion_ids: Sequence[int],
    completion_text: str,
    teacher_tokens: Sequence[Any] | None,
    example: object = None,
    terminal_eos: bool,
    length_terminated: bool,
    status: str = "ok",
) -> ContinuationRecord:
    teacher_logprob = None
    if status == "ok" and teacher_tokens is not None:
        teacher_logprob = teacher_sequence_logprob(teacher_tokens)
    return ContinuationRecord(
        prompt_key=str(prompt_key),
        candidate_index=int(candidate_index),
        prompt_ids=tuple(int(token_id) for token_id in prompt_ids),
        completion_ids=tuple(int(token_id) for token_id in completion_ids),
        completion_text=str(completion_text),
        teacher_logprob=teacher_logprob,
        correctness=pubmedqa_correctness(example, completion_text),
        boxed_format=float(has_boxed_format(completion_text)),
        terminal_eos=float(bool(terminal_eos)),
        no_repetition=float(not has_repetition(completion_text)),
        no_length_termination=float(not length_terminated),
        status=str(status),
    )


def select_teacher_candidates(
    records: Sequence[ContinuationRecord], count: int = C14_TEACHER_CANDIDATES
) -> tuple[ContinuationRecord, ...]:
    """deterministically select usable candidates, preferring unique continuation text."""
    if count < 0:
        raise ValueError("candidate count must be non-negative")
    ordered = sorted(records, key=lambda record: (record.candidate_index, record.completion_text))
    selected: list[ContinuationRecord] = []
    seen_text: set[str] = set()
    for record in ordered:
        if record.status != "ok" or record.teacher_logprob is None:
            continue
        if record.completion_text in seen_text:
            continue
        selected.append(record)
        seen_text.add(record.completion_text)
        if len(selected) == count:
            return tuple(selected)
    for record in ordered:
        if record.status != "ok" or record.teacher_logprob is None or record in selected:
            continue
        selected.append(record)
        if len(selected) == count:
            break
    return tuple(selected)
