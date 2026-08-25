"""Single-turn rollout scoring.

GRPO scores each rollout by calling the environment's reward function. These helpers wrap that
call: they strip the thinking prefix the student may have emitted, coerce whatever the environment
returned into a finite float, and batch concurrent requests so one slow reward function does not
serialize the step.

Split out of `flash.engine.worker.train.entry.rl_train` to keep that module under the file-size limit.
"""

from __future__ import annotations

import math

import flash.engine.worker.model.decoding as _worker_decoding


def _single_turn_scoring_state(
    solution_str: str, *, thinking: bool, prompt_opened_thinking: bool
) -> tuple[str, dict | None]:
    graded = _worker_decoding.graded_text(
        solution_str, prompt_opened_thinking=prompt_opened_thinking
    )
    state = (
        {
            "raw": solution_str,
            "completion": graded,
            "thinking": _worker_decoding.thinking_text(
                solution_str, prompt_opened_thinking=prompt_opened_thinking
            ),
        }
        if thinking
        else None
    )
    return graded, state


def _finalize_single_turn_reward(
    reward: float,
    solution_str: str,
    *,
    tok,
    thinking: bool,
    prompt_opened_thinking: bool,
    think_penalty: float,
    raise_on_error: bool = False,
) -> float:
    if think_penalty > 0 and thinking:
        reward -= think_penalty * _worker_decoding.think_token_count(
            solution_str, tok, prompt_opened_thinking=prompt_opened_thinking
        )
    reward = float(reward)
    # verl computes the grpo baseline with plain mean/std, so one non-finite row poisons every
    # advantage in its group. match the scalar and batched paths by masking it before the bridge.
    if not math.isfinite(reward):
        if raise_on_error:
            raise ValueError(f"env scoring returned a non-finite reward: {reward}")
        print(f"[rl-verl] env scored {reward}; unscorable, scoring 0.0", flush=True)
        return 0.0
    return reward


def score_single_turn(
    env,
    solution_str: str,
    ex,
    *,
    tok,
    thinking: bool,
    prompt_opened_thinking: bool,
    think_penalty: float,
    raise_on_error: bool = False,
    breakdowns: list[dict[str, float] | None] | None = None,
) -> float:
    """score one single-turn text completion against flash's live env.

    mirrors the training path, including optional thinking penalty. training errors become 0.0;
    ``raise_on_error`` lets bridge callers propagate failed grading.

    append named breakdowns only when provided. grading failures append none so reported component
    means count them as zero without inventing metrics for scalar-only envs.
    """
    # optimistic until the probe answers: a probe that RAISES is itself a failed grading, and the
    # except branch has to be able to record it -- omitting it instead would drop this completion
    # from the denominator and bias every named metric high. narrowed the instant the probe returns.
    collect_breakdown = breakdowns is not None
    try:
        # probe INSIDE the guard: `scores_breakdown` may be a property or a proxy whose lookup
        # raises something other than AttributeError, and out here that escapes into the reward http
        # handler -- the verl child reads a bridge failure and aborts the run, when this function's
        # whole contract is to turn an env scoring fault into 0.0.
        has_breakdown = hasattr(env, "scores_breakdown")
        collect_breakdown = collect_breakdown and has_breakdown
        graded, state = _single_turn_scoring_state(
            solution_str,
            thinking=thinking,
            prompt_opened_thinking=prompt_opened_thinking,
        )
        if has_breakdown:
            breakdown = env.scores_breakdown(graded, ex, state)
            # convert total FIRST: a breakdown whose total is not a number is a failed grading, and
            # recording its named components would credit metrics to a completion that scored 0.0.
            r = float(breakdown.get("total", 0.0))
            if collect_breakdown:
                breakdowns.append(breakdown)
        else:
            r = float(env.reward(graded, ex, state))
    except Exception as exc:  # env scoring must not kill the run
        if raise_on_error:
            raise
        if collect_breakdown:
            breakdowns.append(None)
        print(
            f"[rl-verl] env scoring raised ({type(exc).__name__}: {exc}); scoring 0.0", flush=True
        )
        return 0.0
    return _finalize_single_turn_reward(
        r,
        solution_str,
        tok=tok,
        thinking=thinking,
        prompt_opened_thinking=prompt_opened_thinking,
        think_penalty=think_penalty,
        raise_on_error=raise_on_error,
    )


def score_single_turn_batch(
    env,
    requests: list[tuple[str, dict]],
    *,
    tok,
    thinking: bool,
    prompt_opened_thinking: bool,
    think_penalty: float,
) -> list[tuple[float, list[dict[str, float] | None]]]:
    """score a batch of single-turn completions while preserving scalar failure semantics."""
    if not requests:
        return []

    def score_one_serially(solution_str, ex):
        # score_single_turn finalizes internally, so what it returns is already a FINAL row -- the
        # trailing finalize loop must not touch it again.
        breakdowns: list[dict[str, float] | None] = []
        score = score_single_turn(
            env,
            solution_str,
            ex,
            tok=tok,
            thinking=thinking,
            prompt_opened_thinking=prompt_opened_thinking,
            think_penalty=think_penalty,
            breakdowns=breakdowns,
        )
        return score, breakdowns

    def score_serially():
        return [score_one_serially(solution_str, ex) for solution_str, ex in requests]

    try:
        prepared = []
        for solution_str, ex in requests:
            graded, state = _single_turn_scoring_state(
                solution_str,
                thinking=thinking,
                prompt_opened_thinking=prompt_opened_thinking,
            )
            batch_state = {"response_text": graded}
            if state is not None:
                batch_state.update(state)
            prepared.append((solution_str, ex, batch_state))

        scores_breakdown_many = getattr(env, "scores_breakdown_many", None)
        reward_many = getattr(env, "reward_many", None)
        items = [(ex, state) for _, ex, state in prepared]
        if callable(scores_breakdown_many):
            batch_values = list(scores_breakdown_many(items))
            use_breakdown = True
        elif callable(reward_many):
            batch_values = list(reward_many(items))
            use_breakdown = False
        else:
            return score_serially()
    except Exception as exc:
        # the batch call itself blew up, so NOTHING came back: no env work is credited and a full
        # serial pass is the only way to score this batch.
        print(
            f"[rl-verl] env batch scoring raised ({type(exc).__name__}: {exc}); retrying serially",
            flush=True,
        )
        return score_serially()

    # the batch scorer RETURNED, so its env calls already happened -- and for a paid or
    # side-effecting grader they are already billed. re-running every item (the old behaviour on any
    # malformed payload) charged the whole batch twice to recover a single bad row. keep every
    # well-formed row and repair only the rest serially.
    results: list[tuple[float, list[dict[str, float] | None]] | None] = [None] * len(prepared)
    repair: list[int] = []
    # a wrong-length payload cannot be trusted to be positionally aligned even where it does have
    # entries, so it condemns every row rather than just the missing tail.
    aligned = len(batch_values) == len(prepared)
    for idx, (solution_str, _, _) in enumerate(prepared):
        if not aligned:
            repair.append(idx)
            continue
        try:
            value = batch_values[idx]
            if use_breakdown:
                score, breakdowns = float(value.get("total", 0.0)), [value]
            else:
                score, breakdowns = float(value), []
        except Exception:
            # unusable row: a non-mapping breakdown, a missing/unparseable `total`, or a reward that
            # will not survive float(). recover this one item, not the batch.
            repair.append(idx)
            continue
        results[idx] = (
            _finalize_single_turn_reward(
                score,
                solution_str,
                tok=tok,
                thinking=thinking,
                prompt_opened_thinking=prompt_opened_thinking,
                think_penalty=think_penalty,
            ),
            breakdowns,
        )

    if repair:
        detail = (
            f"returned {len(repair)} unusable row(s)"
            if aligned
            else f"returned {len(batch_values)} rows for {len(prepared)} items"
        )
        print(
            f"[rl-verl] env batch scoring {detail}; re-scoring {len(repair)} of {len(prepared)} "
            f"serially",
            flush=True,
        )
        for idx in repair:
            solution_str, ex, _ = prepared[idx]
            results[idx] = score_one_serially(solution_str, ex)

    # every index is filled by one path or the other. return one row per request, in request order:
    # the caller zips these back onto its completions, so dropping a row would silently misalign the
    # whole batch rather than fail.
    return [row if row is not None else (0.0, []) for row in results]
