"""Batching and validation for text-teacher scoring requests.

The teacher is a remote model, so OPD scores rollouts in batches rather than one call per sequence.
`_TextTeacherBatcher` collects concurrent requests into one wave, and the validators here reject a
response whose token ids or logprobs do not line up with what was sent -- a mismatch would
silently train the student against the wrong distribution.

Split out of `flash.engine.worker.train.entry.opd_train` to keep that module under the file-size limit.
"""

from __future__ import annotations

import math

from flash.engine.worker.teacher.client import (
    _MAX_LOGPROB_ROUNDING_ERROR,
    TeacherClient,
    TeacherError,
    TeacherScore,
)
from flash.engine.worker.teacher.tokenizer_align import TeacherToken
from flash.engine.worker.train.entry.backend_common import BoundedThreadingHTTPServer
from flash.engine.worker.train.entry.score_batcher import ScoreBatcher
from flash.engine.worker.train.opd.orchestration import protocol
from flash.teacher.limits import OPD_TEACHER_SCORING_CONCURRENCY

# how many teacher POSTs may sit in the bridge server's accept backlog. this leaf module owns the
# value so it never imports the entry orchestrator.
_TEXT_TEACHER_REQUEST_BACKLOG = 64


def _align_granularity(groups, student_tokens) -> float:
    """mean student-tokens-per-group: the alignment-health signal coverage cannot provide.

    a degenerate alignment that collapses every student token onto a single group still scores
    coverage ~1.0, so coverage alone never flags it. this is the trl opd formula verbatim
    (``opd.py``): non-empty student spans over surviving groups, counted after fully-forced groups
    are dropped so the ratio describes the signal that actually reaches the loss.
    """
    if not groups:
        return 0.0
    n_align = sum(1 for st in student_tokens if st.end > st.start)
    return n_align / len(groups)


class _TeacherBridgeHTTPServer(BoundedThreadingHTTPServer):
    request_queue_size = _TEXT_TEACHER_REQUEST_BACKLOG


def _teacher_batch_error(error: Exception) -> Exception:
    if isinstance(error, TeacherError):
        return TeacherError(
            str(error),
            permanent=error.permanent,
            provider_status=error.provider_status,
        )
    return RuntimeError(str(error))


def _validate_text_teacher_batch(
    scored,
    items: list[tuple[str, str]],
) -> list[TeacherScore]:
    expected = len(items)
    if not isinstance(scored, list) or len(scored) != expected:
        actual = len(scored) if isinstance(scored, list) else type(scored).__name__
        raise TeacherError(
            f"teacher text batch returned {actual} result(s) for {expected} unique input(s)",
            permanent=True,
        )
    for result_index, (score, (_prompt_text, completion_text)) in enumerate(
        zip(scored, items, strict=True)
    ):
        if not isinstance(score, TeacherScore):
            raise TeacherError(
                f"teacher text batch result {result_index} is not a TeacherScore",
                permanent=True,
            )
        previous_start = -1
        previous_end = -1
        for token_index, token in enumerate(score.tokens):
            if not isinstance(token, TeacherToken):
                raise TeacherError(
                    f"teacher text batch result {result_index} contains an invalid token",
                    permanent=True,
                )
            if not isinstance(token.text, str):
                raise TeacherError(
                    f"teacher text batch result {result_index} token {token_index} has invalid text",
                    permanent=True,
                )
            if (
                isinstance(token.logprob, bool)
                or not isinstance(token.logprob, int | float)
                or not math.isfinite(token.logprob)
                or token.logprob > _MAX_LOGPROB_ROUNDING_ERROR
            ):
                raise TeacherError(
                    f"teacher text batch result {result_index} token {token_index} has invalid logprob",
                    permanent=True,
                )
            if (
                isinstance(token.start, bool)
                or isinstance(token.end, bool)
                or not isinstance(token.start, int)
                or not isinstance(token.end, int)
                or token.start < 0
                or token.end < token.start
                or token.end > len(completion_text)
                or token.start < previous_start
                or token.end < previous_end
            ):
                raise TeacherError(
                    f"teacher text batch result {result_index} token {token_index} has invalid offsets",
                    permanent=True,
                )
            previous_start = token.start
            previous_end = token.end
        if score.input_tokens <= 0 or score.output_tokens != 1:
            raise TeacherError(
                f"teacher text batch result {result_index} is missing authoritative token usage",
                permanent=True,
            )
    return scored


def _permanent_teacher_error(message: str) -> Exception:
    return TeacherError(message, permanent=True)


class _TextTeacherBatcher(ScoreBatcher):
    """Batch supplied-token teacher scoring, billing each unique item exactly once.

    Deduplication is the OPD-specific part and stays here: a generation can ask for the same
    (prompt, completion) pair more than once (identical completions off one prompt are common at low
    temperature), and the teacher charges per scored item. Sending the pair once and marking the
    repeats unbilled keeps the answer identical while paying for it once.

    ``cancel_undispatched_on_close`` is on: ``close`` cancels any batch that has not yet won the
    dispatch claim, because a teacher call bills real money and the run is already shutting down.
    """

    def __init__(
        self,
        teacher,
        *,
        max_batch_size: int = OPD_TEACHER_SCORING_CONCURRENCY,
        flush_wait_s: float | None = None,
        on_scored=None,
    ) -> None:
        super().__init__(
            self._score_items,
            max_batch_size=max_batch_size,
            flush_wait_s=(
                protocol.TEXT_TEACHER_FLUSH_WAIT_S if flush_wait_s is None else flush_wait_s
            ),
            label="text teacher batcher",
            thread_name="flash-opd-text-teacher-batcher",
            make_error=_permanent_teacher_error,
            wrap_batch_error=_teacher_batch_error,
            cancel_undispatched_on_close=True,
        )
        self.teacher = teacher
        self.on_scored = on_scored

    def _score_items(self, items: list[tuple[str, str]]) -> list[TeacherScore]:
        unique_items: list[tuple[str, str]] = []
        item_indexes: dict[tuple[str, str], int] = {}
        scatter_indexes: list[int] = []
        for item in items:
            index = item_indexes.get(item)
            if index is None:
                index = len(unique_items)
                item_indexes[item] = index
                unique_items.append(item)
            scatter_indexes.append(index)
        if self.on_scored is None:
            teacher_scores = self.teacher.score_many(unique_items)
        elif isinstance(self.teacher, TeacherClient):
            teacher_scores = self.teacher.score_many(unique_items, on_scored=self.on_scored)
        else:
            teacher_scores = self.teacher.score_many(unique_items)
            for _score in teacher_scores:
                self.on_scored()
        scored = _validate_text_teacher_batch(teacher_scores, unique_items)
        billed_indexes: set[int] = set()
        results: list[TeacherScore] = []
        for index in scatter_indexes:
            result = scored[index]
            if index in billed_indexes:
                result = result.without_billing()
            else:
                billed_indexes.add(index)
            results.append(result)
        return results

    def score(self, prompt_text: str, completion_text: str) -> TeacherScore:
        return self.submit((prompt_text, completion_text))

    def close(self, timeout_s: float | None = None) -> None:
        # resolved here rather than as a default: the shutdown budget is patched on `opd_train`
        # by the batcher tests, and a default argument would freeze it at import time.
        if timeout_s is None:
            timeout_s = protocol.TEXT_TEACHER_SHUTDOWN_WAIT_S
        super().close(timeout_s)
