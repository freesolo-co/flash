"""OPD teacher echo-scoring (single-sample path) + its result type. Runs in the worker's thread
pool: network only, no torch/model state. Extracted from ``opd`` (which re-imports these, and keeps
the batched ``_score_many`` fan-out) so they stay importable from it."""

from __future__ import annotations

from dataclasses import dataclass

from flash.engine.worker.opd_gkd import _teacher_prompt_text
from flash.engine.worker.teacher import TeacherError


@dataclass(frozen=True)
class _ScoreResult:
    """Teacher-scoring outcome from ``_score_one`` (RUN IN THE THREAD POOL). ``status`` is "ok"
    (``teacher_toks`` populated), "transient" (retryable teacher outage -> sample skipped + counted),
    or "error" (any other exception -> sample skipped, teacher uncounted). A PERMANENT ``TeacherError``
    is NOT represented here — ``_score_one`` re-raises it so the run aborts, exactly as before."""

    teacher_toks: object = None
    status: str = "ok"
    error: str = ""


def _score_one(
    teacher, gen_result, *, prompt_messages, thinking_prefill, max_attempts: int = 2
) -> _ScoreResult:
    """[THREAD POOL — network only] Build the teacher prompt and echo-score the completion over the
    API. MUST NOT touch the torch model or any shared mutable state — it reads only the completion
    string + prompt messages and calls the stateless teacher HTTP client, so it is safe to run
    concurrently for every scorable sample in a step. ``thinking_prefill`` is appended to the teacher
    prompt so it conditions on the same trailing context the student sampled after in thinking mode.
    Error semantics are the shared OPD teacher semantics: a PERMANENT ``TeacherError`` propagates (the
    run aborts), a transient one -> status "transient", any other exception -> status "error"; both
    leave the sample skipped with no teacher signal."""
    teacher_prompt = _teacher_prompt_text(prompt_messages, thinking_prefill)
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        try:
            teacher_toks = teacher.score(teacher_prompt, gen_result.completion_text)
        except TeacherError as e:
            if e.permanent:  # bad key / model id / malformed -> abort now, don't burn the whole run
                raise
            if attempt < attempts:
                print(
                    f"[opd] teacher score failed (transient, retrying sample {attempt}/{attempts}): {e}"
                )
                continue
            print(f"[opd] teacher score failed (transient, skipping sample): {e}")
            return _ScoreResult(status="transient", error=str(e))
        except Exception as e:
            if attempt < attempts:
                print(f"[opd] teacher score failed (retrying sample {attempt}/{attempts}): {e}")
                continue
            print(f"[opd] teacher score failed (skipping sample): {e}")
            return _ScoreResult(status="error", error=str(e))
        return _ScoreResult(teacher_toks=teacher_toks, status="ok")
    return _ScoreResult(status="error", error="teacher scoring attempts exhausted")
