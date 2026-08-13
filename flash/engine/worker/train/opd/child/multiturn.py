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
    "dedup_seam_terminator",
    "normalize_token_ids",
    "prepare_assistant_turn",
    "run_executor_call",
    "sum_preemptions",
    "validate_glue_template",
    "validate_transcript_messages",
]


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
    child_failure_handler,
    **kwargs,
):
    failure_stage = "template"
    failure_error = None
    failure_exit_code = None
    start_attempted = False
    try:
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
        failure_stage = "multiturn_start"
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
        failure_stage = "generate"
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
        failure_stage = "score"
        score_payload = await run_executor_call(
            self.loop,
            lambda: _post_multiturn_score(
                post_json,
                score_failure_handler,
                bridge_url,
                bridge_token,
                session_id,
            ),
        )
        _attach_teacher_rows(outputs, score_payload)
    except Exception as error:
        failure_error = error
        failure_exit_code = (
            transient_teacher_exit
            if getattr(error, "classification", None) == "transient"
            else permanent_teacher_exit
        )
    finally:
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
    if failure_error is not None and failure_exit_code is not None:
        classification = "transient" if failure_exit_code == transient_teacher_exit else "permanent"
        child_failure_handler(classification, failure_stage, failure_error)
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


def build_flash_multi_turn_agent_loop(
    *,
    register,
    agent_loop_base,
    agent_loop_output,
    post_json,
    score_failure_handler,
    child_failure_handler,
    deterministic_seed,
    permanent_teacher_exit: int = 86,
    transient_teacher_exit: int = 87,
    process_exit=None,
):
    """build and register the child loop without importing verl in the parent interpreter."""
    exit_process = process_exit or os._exit

    class FlashMultiTurnAgentLoop(agent_loop_base):
        async def run(self, sampling_params: dict[str, Any], **kwargs):
            try:
                return await self._run(sampling_params, **kwargs)
            except Exception as error:
                classification = (
                    "transient"
                    if getattr(error, "classification", None) == "transient"
                    else "permanent"
                )
                exit_code = (
                    transient_teacher_exit
                    if classification == "transient"
                    else permanent_teacher_exit
                )
                child_failure_handler(classification, "template", error)
                exit_process(exit_code)
                raise AssertionError("multi-turn OPD process exit returned unexpectedly") from error

        async def _run(self, sampling_params: dict[str, Any], **kwargs):
            return await _opd_run(
                self,
                sampling_params,
                post_json=post_json,
                score_failure_handler=score_failure_handler,
                permanent_teacher_exit=permanent_teacher_exit,
                transient_teacher_exit=transient_teacher_exit,
                exit_process=exit_process,
                child_failure_handler=child_failure_handler,
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
