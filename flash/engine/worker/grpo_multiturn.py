"""standalone child-side multi-turn GRPO rollout support for verl 0.8.0.

runs INSIDE the verl interpreter. stdlib only -- no verl import at module scope and no flash
import, so the parent copies this file into the child workdir the same way OPD copies its own
multi-turn helper.

the shape differs from OPD's multi-turn loop in one decisive way. OPD distils each assistant turn
as its own single-turn sample and returns a LIST of ``AgentLoopOutput``s; only
``AgentLoopWorkerTQ._agent_loop_postprocess`` (main_ppo_sync.py) tolerates a list, and that class
belongs to the TransferQueue entrypoint OPD launches. GRPO launches ``verl.trainer.main_ppo``,
whose stock ``_agent_loop_postprocess`` (agent_loop.py) sets ``output.extra_fields[...]`` on a bare
object and would raise AttributeError on a list. so this loop returns exactly ONE output per
EPISODE: the whole interleaved transcript as a single sequence, with ``response_mask`` marking
model tokens 1 and environment glue 0, and one episode reward on it.

that mask is not an approximation of trl's ``env_mask`` -- verl documents ``response_mask`` as "1s
for LLM generated token, 0 for tool response token" (agent_loop.py), which is the same quantity
trl's multi-turn rollout_func builds.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any
from uuid import uuid4

try:  # inside the verl child, copied in beside this file
    from flash_multiturn_glue import (
        EnvGlueTokenizer,
        dedup_seam_terminator,
        prepare_assistant_turn,
        run_executor_call,
        sum_preemptions,
        validate_transcript_messages,
    )
except ImportError:  # in-tree (tests, lint)
    from flash.engine.worker.multiturn_glue import (
        EnvGlueTokenizer,
        dedup_seam_terminator,
        prepare_assistant_turn,
        run_executor_call,
        sum_preemptions,
        validate_transcript_messages,
    )

# reserve enough room after gluing an environment reply that the next model turn can actually
# generate. gluing right up to the limit leaves zero completion tokens, so the engine would
# immediately truncate a turn that was never really sampled. same constant and same reason as the
# trl driver's token budget (multiturn_rollout.rollout_one).
_NEXT_TURN_SLACK = 8


def post_json(url: str, path: str, payload: dict) -> dict:
    """post one json request to the parent's multi-turn bridge and return the decoded reply.

    every failure raises. unlike the single-turn reward bridge -- where a scoring error degrades to
    0.0 because one bad completion must not kill a run -- a broken turn here has already consumed
    gpu time and left the parent's episode state half-advanced, and continuing would train on a
    transcript the environment never agreed to.
    """
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        with contextlib.suppress(Exception):
            detail = json.loads(error.read().decode("utf-8"))["error"]
            raise RuntimeError(f"flash multi-turn bridge rejected {path}: {detail}") from error
        raise RuntimeError(
            f"flash multi-turn bridge returned HTTP {error.code} for {path}"
        ) from error
    except (OSError, TimeoutError, urllib.error.URLError, http.client.HTTPException) as error:
        raise RuntimeError(
            f"flash multi-turn bridge transport failed on {path}: {type(error).__name__}"
        ) from error
    try:
        return json.loads(body.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError) as error:
        raise RuntimeError(
            f"flash multi-turn bridge returned malformed json for {path}"
        ) from error


def build_flash_grpo_multi_turn_agent_loop(
    *,
    register,
    agent_loop_base,
    agent_loop_output,
    bridge_post=post_json,
):
    """build and register the child loop without importing verl in the parent interpreter."""

    class FlashGrpoMultiTurnAgentLoop(agent_loop_base):
        async def run(self, sampling_params: dict[str, Any], **kwargs):
            raw_prompt = validate_transcript_messages(
                [dict(message) for message in kwargs["raw_prompt"]], source="initial prompt"
            )
            prompt_ids = await self.apply_chat_template(raw_prompt)
            bridge_url = os.environ["FLASH_VERL_MULTITURN_URL"]
            example_index = int(kwargs["index"])
            max_turns = int(os.environ["FLASH_VERL_MAX_TURNS"])
            max_model_len = int(os.environ["FLASH_VERL_MAX_MODEL_LEN"])
            max_completion = int(os.environ["FLASH_VERL_MAX_COMPLETION"])
            stop_sequences = tuple(
                str(value) for value in json.loads(os.environ.get("FLASH_VERL_STOP_SEQUENCES", "[]"))
            )
            eos_token_ids = frozenset(
                int(value) for value in json.loads(os.environ.get("FLASH_VERL_EOS_TOKEN_IDS", "[]"))
            )
            thinking = os.environ.get("FLASH_VERL_THINKING") == "1"
            glue_tokenizer = EnvGlueTokenizer(self.tokenizer, thinking=thinking)
            session_id = uuid4().hex

            # episode accumulators. `response_ids` is the flat interleaved transcript AFTER the
            # prompt and `response_mask` marks which of those tokens the model produced;
            # `prefix_ids` is what the engine conditions on and stays equal to
            # prompt_ids + response_ids.
            response_ids: list[int] = []
            response_mask: list[int] = []
            response_logprobs: list[float] = []
            # [start, end) of each model turn within response_ids, in turn order. verl right-pads
            # response_ids on the right (agent_loop._postprocess pads with padding_side="right"),
            # so these offsets index the response-width advantage tensor unchanged.
            turn_spans: list[tuple[int, int]] = []
            prefix_ids = list(prompt_ids)
            have_logprobs = True
            generated_seconds = 0.0
            num_preempted = -1
            turn_count = 0
            # the response tensor's width. the whole EPISODE has to fit in it, not just one turn:
            # verl right-pads response_ids to this length and truncates anything longer
            # (_pad_token_ids), so a transcript that overruns it would be silently cut mid-turn.
            response_capacity = int(self.rollout_config.response_length)
            start_attempted = False
            try:
                start_attempted = True
                start = await run_executor_call(
                    self.loop,
                    lambda: bridge_post(
                        bridge_url,
                        "/multiturn/start",
                        {
                            "index": example_index,
                            "session_id": session_id,
                            "prompt_ids": prompt_ids,
                        },
                    ),
                )
                turn_limit = int(start["max_turns"])
                if turn_limit <= 0 or turn_limit > max_turns:
                    raise RuntimeError(
                        "flash multi-turn bridge returned an invalid per-example turn limit"
                    )
                for turn_ordinal in range(turn_limit):
                    # three ceilings, and the turn gets the tightest. the first two are the episode's
                    # remaining room; max_completion is what ONE turn is allowed regardless of how
                    # much episode is left, which is the cap the trl driver applies
                    # (per_turn_max_tokens) and the one the response tensor's width does not carry.
                    max_tokens = min(
                        max_completion,
                        max_model_len - len(prefix_ids),
                        response_capacity - len(response_ids),
                    )
                    if max_tokens <= 0:
                        break
                    params = dict(sampling_params)
                    params["max_tokens"] = max_tokens
                    if stop_sequences:
                        params["stop"] = list(stop_sequences)
                        params["include_stop_str_in_output"] = True
                    if eos_token_ids:
                        params["stop_token_ids"] = sorted(eos_token_ids)
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
                    turn_ids = turn["response_ids"]
                    turn_start = len(response_ids)
                    response_ids.extend(turn_ids)
                    response_mask.extend([1] * len(turn_ids))
                    # a truncated or unusable turn is NOT recorded into env state by the bridge
                    # (MultiTurnBridge.step returns before record_model_turn), so the environment
                    # never sees it and returns no reward for it. spanning it here would leave one
                    # more span than there are rewards, which score_rollouts rejects as a count
                    # mismatch -- dropping the row, and with it its whole group, to episode credit.
                    # the tokens stay in response_ids and stay trained on; they just carry no turn
                    # coordinate. this is the same identity the trl driver scores on, where the
                    # turn IS recorded and so IS spanned (multiturn_rollout.rollout_one).
                    if not (turn["truncated"] or turn["skip_reason"]):
                        turn_spans.append((turn_start, len(response_ids)))
                    if have_logprobs and generated.log_probs is not None:
                        response_logprobs.extend(list(generated.log_probs[: len(turn_ids)]))
                    else:
                        # one turn without logprobs makes the whole episode's vector unusable: it
                        # would be shorter than response_ids and silently misalign every token
                        # after the gap. drop the vector rather than emit a misaligned one.
                        have_logprobs = False
                    prefix_ids.extend(turn_ids)
                    turn_count = turn_ordinal + 1
                    step = await run_executor_call(
                        self.loop,
                        lambda turn_ordinal=turn_ordinal, turn=dict(turn): bridge_post(
                            bridge_url,
                            "/multiturn/step",
                            {
                                "session_id": session_id,
                                "turn_ordinal": turn_ordinal,
                                "completion_text": turn["completion_text"],
                                "truncated": turn["truncated"],
                                "skip_reason": turn["skip_reason"],
                            },
                        ),
                    )
                    # a truncated or unusable turn ends the episode: a model that could not finish
                    # its turn cannot meaningfully answer an environment reply, and the bridge has
                    # marked the session terminal for the same reason. matches trl's driver, which
                    # breaks out of the turn loop on the same conditions.
                    if turn["truncated"] or turn["skip_reason"] or step["terminal"]:
                        break
                    env_messages = validate_transcript_messages(
                        step["messages"], source="environment reply"
                    )
                    if not env_messages:
                        break
                    glue_ids = dedup_seam_terminator(turn_ids, glue_tokenizer(env_messages))
                    if (
                        len(prefix_ids) + len(glue_ids) + _NEXT_TURN_SLACK > max_model_len
                        or len(response_ids) + len(glue_ids) + _NEXT_TURN_SLACK > response_capacity
                    ):
                        break
                    response_ids.extend(glue_ids)
                    response_mask.extend([0] * len(glue_ids))
                    if have_logprobs:
                        # the model did not generate the glue, so it has no logprob. 0.0 is the
                        # same filler trl uses (multiturn_rollout.rollout_one); the zeroed
                        # response_mask is what keeps these positions out of the loss.
                        response_logprobs.extend([0.0] * len(glue_ids))
                    prefix_ids.extend(glue_ids)
                score_payload = await run_executor_call(
                    self.loop,
                    lambda: bridge_post(
                        bridge_url,
                        "/multiturn/score",
                        # the count the ENV was told about, not `turn_count`: an aborted turn is
                        # generated (and counted in num_turns) but never recorded into env state,
                        # so asking for a reward per generated turn would request one the env has
                        # no turn for. len(turn_spans) is the same quantity trl scores on.
                        {"session_id": session_id, "turn_count": len(turn_spans)},
                    ),
                )
                reward_score = float(score_payload["score"])
                # the bridge only sends `turns` when per-turn credit is active and the environment
                # returned a validated per-turn vector; one entry per turn the bridge was told
                # about. a count that disagrees with the spans this loop actually emitted cannot be
                # aligned to tokens, so drop to episode credit rather than guess an alignment.
                raw_turns = score_payload.get("turns")
                turn_rewards = None
                if raw_turns is not None:
                    if len(raw_turns) == len(turn_spans):
                        turn_rewards = [float(value) for value in raw_turns]
                    else:
                        print(
                            f"[rl-verl] per-turn rewards ({len(raw_turns)}) do not match emitted "
                            f"turns ({len(turn_spans)}); falling back to episode credit",
                            flush=True,
                        )
            finally:
                if start_attempted:
                    # best effort: a failing close would mask the real error from the body above,
                    # and the bridge reaps stale sessions on its own lease anyway.
                    with contextlib.suppress(Exception):
                        await run_executor_call(
                            self.loop,
                            lambda: bridge_post(
                                bridge_url,
                                "/multiturn/close",
                                {"session_id": session_id},
                            ),
                        )
            return agent_loop_output(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=response_mask,
                response_logprobs=response_logprobs if have_logprobs else None,
                num_turns=turn_count,
                # verl reads reward_score BEFORE its reward manager runs: _compute_score skips any
                # output that already carries one, _postprocess writes it into rm_scores on the
                # last response token, and both NaiveRewardManager and ray_trainer short-circuit
                # reward computation when rm_scores is present. so the episode reward set here is
                # authoritative and the single-turn custom_reward_function is never consulted for
                # these rows.
                reward_score=reward_score,
                # per-turn credit assignment. the bridge returns one reward per emitted turn when
                # the environment scores turns and the run asked for per-turn credit; it returns
                # None otherwise, and the advantage shim then falls back to episode credit for the
                # whole group. carried through extra_fields because that is the only channel the
                # agent loop transports into non_tensor_batch (agent_loop._postprocess).
                extra_fields={
                    "flash_turn_spans": list(turn_spans),
                    "flash_turn_rewards": turn_rewards,
                },
                metrics={
                    "generate_sequences": generated_seconds,
                    "tool_calls": 0.0,
                    "compute_score": 0.0,
                    "num_preempted": num_preempted,
                },
            )

    FlashGrpoMultiTurnAgentLoop.__module__ = __name__
    FlashGrpoMultiTurnAgentLoop.__qualname__ = "FlashGrpoMultiTurnAgentLoop"
    globals()["FlashGrpoMultiTurnAgentLoop"] = FlashGrpoMultiTurnAgentLoop
    register("flash_grpo_multi_turn")(FlashGrpoMultiTurnAgentLoop)
    return FlashGrpoMultiTurnAgentLoop
