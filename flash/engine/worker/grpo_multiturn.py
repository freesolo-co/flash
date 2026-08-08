"""standalone child-side multi-turn GRPO rollout support for verl 0.8.0.

stdlib only: the parent copies it into the child interpreter. GRPO must return one output per
episode
because main_ppo cannot postprocess OPD's list form; response_mask marks model tokens only.
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
        turn_is_unusable,
        validate_transcript_messages,
    )
except ImportError:  # in-tree (tests, lint)
    from flash.engine.worker.multiturn_glue import (
        EnvGlueTokenizer,
        dedup_seam_terminator,
        prepare_assistant_turn,
        run_executor_call,
        sum_preemptions,
        turn_is_unusable,
        validate_transcript_messages,
    )

# reserve enough room after gluing an environment reply that the next model turn can actually
# generate. gluing right up to the limit leaves zero completion tokens, so the engine would
# immediately truncate a turn that was never really sampled. carried over from the retired trl
# driver's token budget, which reserved the same slack for the same reason.
_NEXT_TURN_SLACK = 8


def post_json(url: str, path: str, payload: dict) -> dict:
    """post one json request to the parent's multi-turn bridge and return the decoded reply.

    Every failure raises because continuing would train on environment state that never completed.
    Use no request deadline: lock queueing is batch-size dependent; the stall watchdog in
    ``flash/providers/_poll.py`` bounds genuine wedges.
    """
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
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
        raise RuntimeError(f"flash multi-turn bridge returned malformed json for {path}") from error


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
            # an image-bearing prompt tokenizes to placeholder tokens that carry no pixels. both
            # apply_chat_template and every generate call need the decoded media alongside the ids,
            # or the engine sees placeholders it cannot expand and the rollout either dies on a
            # feature/placeholder mismatch or conditions on nothing. text-only prompts yield {}.
            multi_modal_data = await self.process_multi_modal_info(raw_prompt)
            images = multi_modal_data.get("images")
            videos = multi_modal_data.get("videos")
            audios = multi_modal_data.get("audios")
            mm_processor_kwargs = self._get_mm_processor_kwargs(audios)
            prompt_ids = await self.apply_chat_template(
                raw_prompt,
                images=images,
                videos=videos,
                audios=audios,
                mm_processor_kwargs=mm_processor_kwargs,
            )
            bridge_url = os.environ["FLASH_VERL_MULTITURN_URL"]
            example_index = int(kwargs["index"])
            max_turns = int(os.environ["FLASH_VERL_MAX_TURNS"])
            max_model_len = int(os.environ["FLASH_VERL_MAX_MODEL_LEN"])
            max_completion_tokens = int(os.environ["FLASH_VERL_MAX_COMPLETION_TOKENS"])
            stop_sequences = tuple(
                str(value)
                for value in json.loads(os.environ.get("FLASH_VERL_STOP_SEQUENCES", "[]"))
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
                    # three budgets, all binding: the configured per-turn cap, the engine's
                    # remaining context, and the response tensor's remaining width. the first is
                    # what the user asked for and the other two are what the episode can still
                    # hold, so one turn never spends the whole transcript.
                    max_tokens = min(
                        max_completion_tokens,
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
                    # the media rides along on EVERY turn, not just the first: each turn re-sends the
                    # whole prefix, whose placeholder tokens still need their pixels to expand.
                    generated = await self.server_manager.generate(
                        request_id=uuid4().hex,
                        prompt_ids=prefix_ids,
                        sampling_params=params,
                        image_data=images,
                        video_data=videos,
                        audio_data=audios,
                        mm_processor_kwargs=mm_processor_kwargs,
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
                    unusable_turn = turn_is_unusable(turn)
                    response_ids.extend(turn_ids)
                    # the bridge does not record or score an unusable turn, so keep its tokens
                    # out of the loss with response_mask=0. verl's single-turn truncation handling
                    # cannot reach
                    # this custom AgentLoopOutput.
                    response_mask.extend([0 if unusable_turn else 1] * len(turn_ids))
                    # spanning it would also leave one more span than there are rewards, which
                    # score_rollouts rejects as a count mismatch -- dropping the row, and with it its
                    # whole group, to episode credit. the span list and the mask are independent
                    # fields on the output, so zeroing above does not change this count.
                    if not unusable_turn:
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
                        # same filler the retired trl driver used; the zeroed
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
                # the training pass re-tokenizes this episode through the processor, so it needs the
                # same media the rollout conditioned on; without it the actor's forward would see
                # placeholders with no pixels behind them.
                multi_modal_data=multi_modal_data or None,
                mm_processor_kwargs=mm_processor_kwargs,
                num_turns=turn_count,
                # reward_score is authoritative: verl copies it to rm_scores and skips
                # reward-manager
                # computation, so the single-turn custom reward function is not used for these rows.
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
