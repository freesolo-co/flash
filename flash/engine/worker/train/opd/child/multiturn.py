"""standalone child-side multi-turn OPD rollout support for verl 0.8.0."""

from __future__ import annotations

import contextlib
import json
import os
import time
from typing import Any
from uuid import uuid4

try:  # inside the verl child, copied in beside this file
    from flash_multiturn_glue import (
        EnvGlueTokenizer,
        dedup_seam_terminator,
        normalize_token_ids,
        prepare_assistant_turn,
        run_executor_call,
        sum_preemptions,
        validate_glue_template,
        validate_transcript_messages,
    )
except ImportError:  # in-tree (parent process, tests, lint)
    from flash.engine.worker.train.core.child.glue import (
        EnvGlueTokenizer,
        dedup_seam_terminator,
        normalize_token_ids,
        prepare_assistant_turn,
        run_executor_call,
        sum_preemptions,
        validate_glue_template,
        validate_transcript_messages,
    )

__all__ = [
    "EnvGlueTokenizer",
    "build_flash_multi_turn_agent_loop",
    "build_flash_replay_buffer",
    "dedup_seam_terminator",
    "normalize_token_ids",
    "prepare_assistant_turn",
    "run_executor_call",
    "sum_preemptions",
    "validate_glue_template",
    "validate_transcript_messages",
]


class _DeferredScoreFailure(Exception):
    def __init__(self, error):
        super().__init__(str(error))
        self.error = error


def _defer_score_failure(error) -> None:
    raise _DeferredScoreFailure(error)


def _post_multiturn_score(
    post_json,
    score_failure_handler,
    url: str,
    token: str,
    session_id: str,
):
    try:
        return post_json(
            url,
            token,
            "/multiturn/score",
            {"session_id": session_id},
        )
    except Exception as error:
        if getattr(error, "delivery_unknown", False):
            score_failure_handler(error)
        raise


def _attach_teacher_rows(outputs, score_payload) -> None:
    """attach the bridge's teacher tensors to each turn's output, rejecting any shape mismatch.

    one scored row per emitted turn. a count or length disagreement cannot be aligned to tokens,
    and training on a misaligned teacher is worse than failing the rollout, so both raise.
    """
    scored_turns = score_payload["turns"]
    if len(scored_turns) != len(outputs):
        raise RuntimeError("multi-turn bridge returned the wrong number of teacher rows")

    import torch

    for output, scored in zip(outputs, scored_turns, strict=True):
        teacher_ids = torch.tensor(scored["teacher_ids"], dtype=torch.int32).unsqueeze(-1)
        teacher_logprobs = torch.tensor(scored["teacher_logprobs"], dtype=torch.float32).unsqueeze(
            -1
        )
        expected_length = len(output.prompt_ids) + len(output.response_ids)
        if teacher_ids.shape != teacher_logprobs.shape:
            raise RuntimeError("multi-turn OPD teacher tensors are inconsistent")
        if teacher_ids.shape[0] != expected_length:
            raise RuntimeError("multi-turn OPD teacher tensors have the wrong sequence length")
        output.extra_fields["teacher_ids"] = teacher_ids
        output.extra_fields["teacher_logprobs"] = teacher_logprobs


def _opd_turn_sampling_params(
    sampling_params: dict,
    *,
    max_tokens: int,
    seed: int,
    stop_sequences,
    eos_token_ids,
) -> dict:
    """the per-turn sampling params: the caller's, plus this turn's cap, seed, and stop conditions."""
    params = dict(sampling_params)
    params["max_tokens"] = max_tokens
    params["seed"] = seed
    if stop_sequences:
        params["stop"] = list(stop_sequences)
        params["include_stop_str_in_output"] = True
    if eos_token_ids:
        params["stop_token_ids"] = sorted(eos_token_ids)
    return params


def _opd_turn_output_fields(
    prefix_ids,
    response_ids,
    response_logprobs,
    generated,
    *,
    turn_ordinal: int,
    generated_seconds: float,
    num_preempted: int,
) -> dict:
    """the fields one OPD turn contributes to its own AgentLoopOutput.

    OPD emits one output per turn rather than one per episode, so the prompt is the prefix this turn
    conditioned on and the whole response is model-generated (mask all ones).
    """
    return {
        "prompt_ids": list(prefix_ids),
        "response_ids": list(response_ids),
        "response_mask": [1] * len(response_ids),
        "response_logprobs": response_logprobs,
        "num_turns": turn_ordinal + 1,
        "metrics": {
            "generate_sequences": generated_seconds,
            "tool_calls": 0.0,
            "compute_score": 0.0,
            "num_preempted": num_preempted,
        },
        "extra_fields": dict(generated.extra_fields or {}),
    }


class _OpdEpisodeSettings:
    """the per-run rollout settings the parent hands the child through the environment.

    read once per episode rather than per turn. the verl interpreter cannot import flash, so the
    parent resolves the bridge address, seed, turn limits, halting, and thinking into strings and
    this reads them back. the capability set is validated here because an environment missing a
    capability cannot serve this loop at all.
    """

    def __init__(self):
        self.bridge_url = os.environ["FLASH_OPD_BRIDGE_URL"]
        self.bridge_token = os.environ["FLASH_OPD_BRIDGE_TOKEN"]
        self.flash_seed = int(os.environ["FLASH_OPD_SEED"])
        self.max_turns = int(os.environ["FLASH_OPD_MAX_TURNS"])
        self.max_model_len = int(os.environ["FLASH_OPD_MAX_MODEL_LEN"])
        capabilities = set(json.loads(os.environ.get("FLASH_OPD_ENV_CAPABILITIES", "[]")))
        required_capabilities = {
            "new_rollout_state",
            "record_model_turn",
            "env_reply",
            "rollout_done",
        }
        if capabilities != required_capabilities:
            raise RuntimeError("multi-turn OPD environment capability metadata is invalid")
        self.stop_sequences = tuple(
            str(value) for value in json.loads(os.environ.get("FLASH_OPD_STOP_SEQUENCES", "[]"))
        )
        self.eos_token_ids = frozenset(
            int(value) for value in json.loads(os.environ.get("FLASH_OPD_EOS_TOKEN_IDS", "[]"))
        )
        self.thinking = os.environ.get("FLASH_OPD_THINKING") == "1"


async def _opd_run(
    self,
    sampling_params: dict[str, Any],
    *,
    post_json,
    score_failure_handler,
    permanent_teacher_exit: int,
    transient_teacher_exit: int,
    exit_process,
    failure_marker,
    **kwargs,
):
    raw_prompt = validate_transcript_messages(
        [dict(message) for message in kwargs["raw_prompt"]], source="initial prompt"
    )
    prompt_ids = await self.apply_chat_template(raw_prompt)
    settings = _OpdEpisodeSettings()
    bridge_url = settings.bridge_url
    bridge_token = settings.bridge_token
    global_step = int(kwargs["global_steps"])
    example_index = int(kwargs["index"])
    rollout_ordinal = int(kwargs.get("session_id", 0))
    session_id = f"{uuid4().hex}-{global_step}-{example_index}-{rollout_ordinal}"
    outputs = []
    start_attempted = False
    failure_exit_code = None
    score_failure = None
    try:
        start_attempted = True
        start = await run_executor_call(
            self.loop,
            lambda: post_json(
                bridge_url,
                bridge_token,
                "/multiturn/start",
                {
                    "index": example_index,
                    "session_id": session_id,
                    "prompt_ids": prompt_ids,
                    "raw_prompt": raw_prompt,
                },
            ),
        )
        turn_limit = int(start["max_turns"])
        if turn_limit <= 0 or turn_limit > settings.max_turns:
            raise RuntimeError("multi-turn bridge returned an invalid per-example turn limit")
        await self._run_turns(
            sampling_params,
            outputs,
            settings=settings,
            prompt_ids=prompt_ids,
            session_id=session_id,
            turn_limit=turn_limit,
            global_step=global_step,
            example_index=example_index,
            rollout_ordinal=rollout_ordinal,
            no_signal_attempt_ordinal=int(kwargs.get("flash_no_signal_attempt", 0)),
        )
        score_payload = await run_executor_call(
            self.loop,
            lambda: _post_multiturn_score(
                post_json,
                _defer_score_failure,
                bridge_url,
                bridge_token,
                session_id,
            ),
        )
        _attach_teacher_rows(outputs, score_payload)
    except _DeferredScoreFailure as deferred:
        # score delivery failures normally call `os._exit` on the executor thread. defer only that
        # exit to this async boundary so the transfer-queue marker can complete first; the original
        # handler still owns fallback publication and the authoritative transient exit code.
        score_failure = deferred.error
        failure_exit_code = transient_teacher_exit
    except Exception as error:
        failure_exit_code = (
            transient_teacher_exit
            if getattr(error, "classification", None) == "transient"
            else permanent_teacher_exit
        )
    finally:
        if failure_exit_code is not None:
            # mark before the close below, not after it. the bridge this rollout is about to give
            # up on is the same one the close posts to, so `_post_json`'s 600s transport timeout is
            # the MODAL case on this path rather than a tail: waiting for cleanup to drain would
            # hold the trainer in its poll for ten more minutes with the diagnosis already known.
            await failure_marker.mark()
        if start_attempted:
            with contextlib.suppress(Exception):
                await run_executor_call(
                    self.loop,
                    lambda: post_json(
                        bridge_url,
                        bridge_token,
                        "/multiturn/close",
                        {"session_id": session_id},
                    ),
                )
    if failure_exit_code is not None:
        if score_failure is not None:
            score_failure_handler(score_failure)
        exit_process(failure_exit_code)
        raise AssertionError("multi-turn OPD process exit returned unexpectedly")
    return outputs


async def _opd_run_turns(
    self,
    sampling_params: dict[str, Any],
    outputs: list,
    *,
    settings,
    prompt_ids,
    session_id: str,
    turn_limit: int,
    global_step: int,
    example_index: int,
    rollout_ordinal: int,
    no_signal_attempt_ordinal: int,
    deterministic_seed,
    agent_loop_output,
    post_json,
) -> None:
    bridge_url = settings.bridge_url
    bridge_token = settings.bridge_token
    flash_seed = settings.flash_seed
    max_model_len = settings.max_model_len
    stop_sequences = settings.stop_sequences
    eos_token_ids = settings.eos_token_ids
    glue_tokenizer = EnvGlueTokenizer(self.tokenizer, thinking=settings.thinking)
    generated_seconds = 0.0
    num_preempted = -1
    prefix_ids = list(prompt_ids)
    for turn_ordinal in range(turn_limit):
        remaining = max_model_len - len(prefix_ids)
        if remaining <= 0:
            raise RuntimeError("multi-turn OPD dispatched a prompt without completion capacity")
        max_tokens = min(int(self.rollout_config.response_length), remaining)
        params = _opd_turn_sampling_params(
            sampling_params,
            max_tokens=max_tokens,
            seed=deterministic_seed(
                flash_seed,
                global_step,
                example_index,
                rollout_ordinal,
                turn_ordinal,
                no_signal_attempt_ordinal,
            ),
            stop_sequences=stop_sequences,
            eos_token_ids=eos_token_ids,
        )
        request_started = time.perf_counter()
        generated = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prefix_ids,
            sampling_params=params,
        )
        generated_seconds += time.perf_counter() - request_started
        num_preempted = sum_preemptions(num_preempted, generated.num_preempted)
        turn = prepare_assistant_turn(
            self.tokenizer,
            generated.token_ids,
            stop_reason=generated.stop_reason,
            max_tokens=max_tokens,
            eos_token_ids=eos_token_ids,
            stop_sequences=stop_sequences,
        )
        response_ids = turn["response_ids"]
        response_logprobs = generated.log_probs
        if response_logprobs is not None:
            response_logprobs = list(response_logprobs[: len(response_ids)])
        step = await run_executor_call(
            self.loop,
            lambda turn_ordinal=turn_ordinal, prefix_ids=list(prefix_ids), turn=dict(turn): (
                post_json(
                    bridge_url,
                    bridge_token,
                    "/multiturn/step",
                    {
                        "session_id": session_id,
                        "turn_ordinal": turn_ordinal,
                        "accepted_prefix": prefix_ids,
                        "raw_response_ids": turn["raw_response_ids"],
                        "response_ids": turn["response_ids"],
                        "completion_text": turn["completion_text"],
                        "termination": turn["termination"],
                        "stop_reason": turn["stop_reason"],
                        "max_tokens": turn["max_tokens"],
                        "truncated": turn["truncated"],
                        "skip_reason": turn["skip_reason"],
                    },
                )
            ),
        )
        outputs.append(
            agent_loop_output(
                **_opd_turn_output_fields(
                    prefix_ids,
                    response_ids,
                    response_logprobs,
                    generated,
                    turn_ordinal=turn_ordinal,
                    generated_seconds=generated_seconds,
                    num_preempted=num_preempted,
                )
            )
        )
        if turn["truncated"] or turn["skip_reason"] or step["terminal"]:
            break
        prefix_ids.extend(response_ids)
        env_messages = validate_transcript_messages(step["messages"], source="environment reply")
        if not env_messages:
            break
        glue_ids = dedup_seam_terminator(response_ids, glue_tokenizer(env_messages))
        # stop while at least a minimal generation window remains: gluing right up to
        # max_model_len leaves the next turn zero tokens to generate (the engine would
        # immediately truncate), so reserve a small slack for the next model turn.
        if len(prefix_ids) + len(glue_ids) + 8 > max_model_len:
            break
        prefix_ids.extend(glue_ids)


async def _mark_prompt_failed(kwargs: dict[str, Any]) -> None:
    """Record this prompt as failed in the transfer queue before the actor hard-exits.

    verl marks every prompt "running" on dispatch (`AgentLoopManagerTQ.generate_sequences`) and
    rewrites it in exactly two places: "finished" when the rollout returns, "failure" when
    `_run_prompt` catches. A hard exit reaches neither, and the trainer's `ReplayBuffer.sample`
    waits on "running" forever. Writing the tag here restores the failure edge the exit destroys.

    The trainer-local replay-buffer mirror merges transfer-queue tags but creates an empty key when
    a uid is first observed there. Its upstream `sample` then hard-subscripts `tag["global_steps"]`,
    so the failure write must carry the same step metadata as verl's success write rather than rely
    on a prior "running" observation to have populated it.

    The key is the prompt `uid`, which verl forwards into the loop's kwargs alongside the prompt
    itself. The partition is always "train": verl decides it from `trajectory["validate"]`, which
    `_run_agent_loop` consumes without forwarding, and flash's OPD overrides pin
    `trainer.val_before_train=false` and `trainer.test_freq=-1`, so an OPD run dispatches no
    validation rollout to mark. Reading a "validate" key off these kwargs would be dead code --
    the dispatcher pops it from the batch before building the per-prompt payload.

    This runs on the way to a hard exit and must not raise: the exit code is the run's diagnosis,
    and an exception here would replace it with a bare actor traceback while leaving the marker
    exactly as stuck as before.
    """
    try:
        try:
            import transfer_queue as tq
        except ImportError:
            from verl.utils.transferqueue_utils import tq

        await tq.async_kv_put(
            key=kwargs["uid"],
            partition_id="train",
            tag={"global_steps": int(kwargs["global_steps"]), "status": "failure"},
        )
    except BaseException:  # pragma: no cover - the exit code must remain the diagnosis
        pass


class _PromptFailureMarker:
    """Write the rollout's marker at most once across all multi-turn exit paths."""

    def __init__(self, kwargs, mark_prompt_failed):
        self._kwargs = kwargs
        self._mark_prompt_failed = mark_prompt_failed
        self._marked = False

    async def mark(self) -> None:
        if self._marked:
            return
        self._marked = True
        with contextlib.suppress(BaseException):
            # every caller is immediately preserving a meaningful 86/87 hard exit
            await self._mark_prompt_failed(self._kwargs)


def build_flash_replay_buffer(ReplayBuffer):
    """A replay buffer that fails the step when a rollout marked its prompt failed.

    The other half of the fix above. Clearing the marker wakes the trainer, but verl's `sample`
    returns only the "success" keys, so a dead rollout does not stop the step -- it silently
    shrinks the batch, and `_balance_batch` pads it back to a divisible size. Without this guard
    the wedge merely downgrades from a visible hang to an invisible partial-batch train, which is
    worse: a hang is at least diagnosable, while a padded short batch publishes a model trained on
    data the run never collected.

    The failure check must run BEFORE verl's `sample`, not after it. A ray actor owns every prompt
    in one chunk of the step's batch, often several because flash caps the worker pool at 8. The
    hard exit kills those sibling tasks too, leaving their tags "running" forever. Upstream
    `sample` breaks on the first "running" tag and never returns, so a post-`super` guard is
    unreachable on exactly the multi-prompt worker shape this fix has to handle.

    Raising reaches the parent as a nonzero child exit, the only channel that carries a rollout
    failure out of the verl child -- the rollout dies in a ray actor whose exit code ray never
    reports to the driver.
    """

    class FlashReplayBuffer(ReplayBuffer):
        def sample(
            self,
            partition_id: str,
            global_steps: int | None = None,
            batch_size: int | None = None,
        ):
            if global_steps is None:
                return super().sample(
                    partition_id, global_steps=global_steps, batch_size=batch_size
                )

            while True:
                time.sleep(self.poll_interval)
                with self.lock:
                    step_tags = [
                        (key, tag)
                        for key, tag in self.partitions[partition_id].items()
                        if tag.get("global_steps") == global_steps
                    ]
                    failed = sorted(key for key, tag in step_tags if tag.get("status") == "failure")
                    running = any(tag.get("status") == "running" for _, tag in step_tags)
                if failed:
                    raise RuntimeError(
                        f"flash OPD rollout failed for {len(failed)} prompt(s) at step "
                        f"{global_steps} (first: {failed[0]}); refusing to train on a partial batch"
                    )
                if not running:
                    return super().sample(
                        partition_id, global_steps=global_steps, batch_size=batch_size
                    )

    return FlashReplayBuffer


def build_flash_multi_turn_agent_loop(
    *,
    register,
    agent_loop_base,
    agent_loop_output,
    post_json,
    score_failure_handler,
    deterministic_seed,
    permanent_teacher_exit: int = 86,
    transient_teacher_exit: int = 87,
    process_exit=None,
    mark_prompt_failed=None,
):
    """build and register the child loop without importing verl in the parent interpreter."""
    exit_process = process_exit or os._exit
    mark_failed = mark_prompt_failed or _mark_prompt_failed

    class FlashMultiTurnAgentLoop(agent_loop_base):
        async def run(self, sampling_params: dict[str, Any], **kwargs):
            failure_marker = _PromptFailureMarker(kwargs, mark_failed)
            try:
                return await self._run(
                    sampling_params,
                    failure_marker=failure_marker,
                    **kwargs,
                )
            except Exception as error:
                exit_code = (
                    transient_teacher_exit
                    if getattr(error, "classification", None) == "transient"
                    else permanent_teacher_exit
                )
                # this loop runs inside a ray AgentLoopWorkerTQ ACTOR, not the driver, so the exit
                # code below never reaches flash: the driver's `generate_sequences` already returned
                # (its tasks are fire-and-forget) and ray reports nothing to the trainer. verl's
                # `_run_prompt` handler is the only writer that turns this prompt's "running" marker
                # into "failure", and `os._exit` skips it, so `ReplayBuffer.sample` -- an unbounded
                # `while True` with no deadline or actor-health check -- polls a marker nobody will
                # ever clear. mark the prompt BEFORE exiting so the trainer wakes and fails.
                await failure_marker.mark()
                exit_process(exit_code)
                raise AssertionError("multi-turn OPD process exit returned unexpectedly") from error

        async def _run(self, sampling_params: dict[str, Any], *, failure_marker, **kwargs):
            return await _opd_run(
                self,
                sampling_params,
                post_json=post_json,
                score_failure_handler=score_failure_handler,
                permanent_teacher_exit=permanent_teacher_exit,
                transient_teacher_exit=transient_teacher_exit,
                exit_process=exit_process,
                failure_marker=failure_marker,
                **kwargs,
            )

        async def _run_turns(
            self,
            sampling_params: dict[str, Any],
            outputs: list,
            *,
            settings,
            prompt_ids,
            session_id: str,
            turn_limit: int,
            global_step: int,
            example_index: int,
            rollout_ordinal: int,
            no_signal_attempt_ordinal: int,
        ) -> None:
            """generate and record turns until the env, the budget, or the turn limit ends the episode.

            appends one output per emitted turn to ``outputs``; the caller owns the session's start,
            scoring, and close.
            """
            return await _opd_run_turns(
                self,
                sampling_params,
                outputs,
                settings=settings,
                prompt_ids=prompt_ids,
                session_id=session_id,
                turn_limit=turn_limit,
                global_step=global_step,
                example_index=example_index,
                rollout_ordinal=rollout_ordinal,
                no_signal_attempt_ordinal=no_signal_attempt_ordinal,
                deterministic_seed=deterministic_seed,
                agent_loop_output=agent_loop_output,
                post_json=post_json,
            )

    FlashMultiTurnAgentLoop.__module__ = __name__
    FlashMultiTurnAgentLoop.__qualname__ = "FlashMultiTurnAgentLoop"
    globals()["FlashMultiTurnAgentLoop"] = FlashMultiTurnAgentLoop
    register("flash_multi_turn")(FlashMultiTurnAgentLoop)
    return FlashMultiTurnAgentLoop
