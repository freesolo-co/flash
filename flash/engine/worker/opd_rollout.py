"""OPD on-policy rollout types + pre-scoring gates: the sample/gen result dataclasses, the vLLM
completion -> gated ``_GenResult`` conversion, and the no-loss skip mapping. Extracted from ``opd``
(which re-imports these so they stay importable from it) to keep ``run_opd`` focused on the loop."""

from __future__ import annotations

from dataclasses import dataclass

from flash.engine.worker.opd_gkd import (
    _generation_eos_ids,
    _rollout_terminated,
    _trim_trailing_stop,
)
from flash.engine.worker.opd_vllm import (
    OpdVllmOutput,
    OpdVllmRolloutEngine,
)


@dataclass(frozen=True)
class SampleResult:
    """One student sample's outcome, returned by the rollout/loss pipeline for ``run_opd`` to aggregate.

    ``loss`` is the groupwise reverse-KL loss tensor when the sample was distilled, else ``None``
    (the sample was skipped — truncated rollout, empty completion, or no teacher signal). The stats
    describe what happened so the caller can count teacher health / truncations and, on a no-loss run,
    decide whether it is a retriable infra failure."""

    loss: object = None  # torch scalar tensor when distilled, else None (module is torch-free)
    # "ok" once teacher.score returns, "transient" on a retryable teacher outage, else None (teacher
    # not reached). run_opd uses this to decide whether a no-signal run is a retriable infra failure.
    teacher_status: str | None = None
    # A rollout that didn't terminate naturally — cap hit OR max_time cut, no EOS/stop (skipped, not
    # distilled — see _rollout_terminated).
    truncated: bool = False
    coverage: float = 0.0
    gen_tokens: int = 0
    teacher_tokens: int = 0
    # Mean student-tokens-per-alignment-group; a real health signal where coverage is not.
    group_granularity: float = 0.0
    # Machine-readable reason when loss is None. Used for skipped-step diagnostics and train_meta.
    skip_reason: str = ""
    # Detached log P(eos) at the reinforced terminal position, when the EOS behaviour-cloning term
    # was applied (else None). Surfaced as mean_eos_logprob so a run's rising termination is visible.
    eos_logprob: float | None = None


@dataclass(frozen=True)
class _GenResult:
    """One student rollout after OPD's pre-scoring gates: the sampled completion plus skip/truncated
    verdicts. ``truncated`` (max_new_tokens cap hit OR max_time cut, no EOS/stop) and ``skip`` (empty or
    U+FFFD completion) mean the rollout is dropped BEFORE teacher scoring; otherwise ``completion_ids`` /
    ``completion_text`` carry the trimmed on-policy answer to score + distil. Torch-free
    (completion_ids is a CPU list) so it can be handed to the model-free scoring thread pool."""

    completion_ids: object = None
    completion_text: str = ""
    gen_tokens: int = 0
    truncated: bool = False
    skip: bool = False
    skip_reason: str = ""
    # Grammar-forced mask (parallel to completion_ids): True where guided decoding left one legal
    # token. Threaded from OpdVllmOutput.forced (sliced in lockstep with stop-trimming); () = none.
    forced: tuple = ()


def _sample_skip_reason(r: SampleResult) -> str:
    """Classify a no-loss OPD sample for skipped-step diagnostics."""

    if r.skip_reason:
        return r.skip_reason
    if r.truncated:
        return "truncated_rollout"
    if r.teacher_status == "transient":
        return "teacher_transient"
    if r.teacher_status == "error":
        return "teacher_error"
    if r.teacher_status == "ok":
        return "alignment_empty"
    return "pre_teacher_skip"


@dataclass
class _Pending:
    """A rollout awaiting concurrent teacher scoring and then loss/backward, carrying the prompt context
    both need. Mutable: ``score`` is filled in by the thread pool for scorable rollouts."""

    gen: _GenResult
    prompt_ids: object
    prompt_messages: object
    score: object = None


def _gen_from_vllm_output(out: OpdVllmOutput, tok, knobs) -> _GenResult:
    """Apply OPD's pre-scoring gates to one vLLM completion."""
    completion_ids = [int(t) for t in out.token_ids]
    forced = tuple(bool(f) for f in (getattr(out, "forced", ()) or ()))
    decode = getattr(tok, "decode", None)
    completion_text = out.text or (
        decode(completion_ids, skip_special_tokens=True) if decode else ""
    )
    stop_text = decode(completion_ids, skip_special_tokens=False) if decode else completion_text
    if not (
        out.terminated
        or _rollout_terminated(
            completion_ids,
            stop_text,
            _generation_eos_ids(None, tok),
            knobs.stop_sequences,
        )
    ):
        return _GenResult(
            truncated=True, gen_tokens=len(completion_ids), skip_reason="truncated_rollout"
        )
    # vLLM may strip stop strings unless include_stop_str_in_output is supported. Trim when the
    # delimiter is present; otherwise keep the already-stripped ids/text.
    if knobs.stop_sequences and any(
        stop_text.endswith(s) or completion_text.endswith(s) for s in knobs.stop_sequences
    ):
        completion_ids, completion_text = _trim_trailing_stop(
            tok, completion_ids, stop_text, knobs.stop_sequences
        )
    gen_tokens = len(completion_ids)
    if not completion_text.strip():
        return _GenResult(skip=True, gen_tokens=gen_tokens, skip_reason="empty_completion")
    if "\ufffd" in completion_text:
        return _GenResult(skip=True, gen_tokens=gen_tokens, skip_reason="replacement_char")
    return _GenResult(
        completion_ids=completion_ids,
        completion_text=completion_text,
        gen_tokens=gen_tokens,
        # Trimming drops a trailing stop suffix, so the kept ids are a prefix -> slice forced to match.
        forced=forced[: len(completion_ids)],
    )


def _generate_many_vllm(
    rollout: OpdVllmRolloutEngine, tok, prompt_ids_batch: list[list[int]], knobs, *, max_tokens: int
) -> list[_GenResult]:
    return [
        _gen_from_vllm_output(out, tok, knobs)
        for out in rollout.generate(prompt_ids_batch, max_tokens=max_tokens)
    ]


def _resolve_no_loss_sample(gen, score) -> SampleResult:
    """Map a generated rollout with no usable teacher score to its skipped ``SampleResult``.

    Scored samples must go through ``_resolve_samples_batched`` so OPD has exactly one loss path.
    """
    if gen.truncated:
        # Didn't terminate naturally (cap/max_time cut) — skipped, not distilled.
        return SampleResult(
            truncated=True,
            gen_tokens=gen.gen_tokens,
            skip_reason=gen.skip_reason or "truncated_rollout",
        )
    if gen.skip:  # empty completion or U+FFFD — skipped before scoring
        return SampleResult(
            gen_tokens=gen.gen_tokens,
            skip_reason=gen.skip_reason or "pre_teacher_skip",
        )
    if score is None:
        return SampleResult(
            teacher_status="error",
            gen_tokens=gen.gen_tokens,
            skip_reason="teacher_missing_score",
        )
    if score.status == "transient":  # retryable teacher outage — skipped + counted
        return SampleResult(
            teacher_status="transient",
            gen_tokens=gen.gen_tokens,
            skip_reason="teacher_transient",
        )
    if score.status != "ok":  # any other teacher exception — skipped, teacher uncounted
        return SampleResult(
            teacher_status="error",
            gen_tokens=gen.gen_tokens,
            skip_reason="teacher_error",
        )
    raise RuntimeError("opd scored samples must be resolved through _resolve_samples_batched")
