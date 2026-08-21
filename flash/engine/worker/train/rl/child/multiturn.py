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

if __name__ == "flash_grpo_multiturn":
    from flash_multiturn_glue import (
        EnvGlueProcessor,
        dedup_seam_terminator,
        prepare_assistant_turn,
        prepare_episode_prompt,
        run_executor_call,
        sum_preemptions,
        turn_is_unusable,
    )
else:
    from flash.engine.worker.train.core.child.glue import (
        EnvGlueProcessor,
        dedup_seam_terminator,
        prepare_assistant_turn,
        prepare_episode_prompt,
        run_executor_call,
        sum_preemptions,
        turn_is_unusable,
    )

# reserve enough room after gluing an environment reply that the next model turn can actually
# generate. gluing right up to the limit leaves zero completion tokens, so the engine would
# immediately truncate a turn that was never really sampled. carried over from the retired trl
# driver's token budget, which reserved the same slack for the same reason.
_NEXT_TURN_SLACK = 8


def post_json(url: str, path: str, payload: dict, *, error_style: str = "multi-turn") -> dict:
    """post one json request to a parent bridge and return the decoded reply.

    Every failure raises because continuing would train on state that never completed. Use no request
    deadline: lock queueing is batch-size dependent; the stall watchdog bounds genuine wedges.
    """
    if error_style not in {"multi-turn", "reward"}:
        raise ValueError(f"unknown flash bridge error style: {error_style}")
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
        # decode inside the guard, raise outside it: suppressing the raise itself drops the bridge's
        # detail and leaves only a bare status code.
        detail = None
        with contextlib.suppress(Exception):
            detail = str(json.loads(error.read().decode("utf-8"))["error"])
        fault = "could not serve" if error.code >= 500 else "rejected"
        if error_style == "reward":
            if detail:
                raise RuntimeError(
                    f"flash reward bridge {fault} the request (HTTP {error.code}): {detail}"
                ) from error
            raise RuntimeError(
                f"flash reward bridge {fault} the request: HTTP {error.code} with no error detail"
            ) from error
        if detail:
            raise RuntimeError(
                f"flash multi-turn bridge {fault} {path} (HTTP {error.code}): {detail}"
            ) from error
        raise RuntimeError(
            f"flash multi-turn bridge {fault} {path}: HTTP {error.code} with no error detail"
        ) from error
    except (OSError, http.client.HTTPException) as error:
        if error_style == "reward":
            raise RuntimeError(f"flash reward bridge request failed: {error}") from error
        raise RuntimeError(
            f"flash multi-turn bridge transport failed on {path}: {type(error).__name__}"
        ) from error
    except Exception as error:
        if error_style == "reward":
            raise RuntimeError(
                f"flash reward bridge returned an invalid response: {error}"
            ) from error
        raise
    try:
        return json.loads(body.decode("utf-8"))
    except (TypeError, ValueError) as error:
        if error_style == "reward":
            raise RuntimeError(
                f"flash reward bridge returned an invalid response: {error}"
            ) from error
        raise RuntimeError(f"flash multi-turn bridge returned malformed json for {path}") from error


class _EpisodeTranscript:
    """the flat token transcript one episode accumulates across its turns.

    ``response_ids`` is everything after the prompt and ``response_mask`` marks which of those the
    model produced; ``prefix_ids`` is what the engine conditions on and stays equal to
    prompt_ids + response_ids. these were separate locals in the turn loop, and they are only ever
    appended to together, so they live on one object rather than being threaded pairwise.
    """

    def __init__(self, prompt_ids, *, response_capacity: int, max_model_len: int):
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        # [start, end) of each model turn within response_ids, in turn order. verl right-pads
        # response_ids on the right (agent_loop._postprocess pads with padding_side="right"),
        # so these offsets index the response-width advantage tensor unchanged.
        self.turn_spans: list[tuple[int, int]] = []
        self.prefix_ids = list(prompt_ids)
        self.have_logprobs = True
        self.generated_seconds = 0.0
        self.num_preempted = -1
        self.turn_count = 0
        self.response_capacity = response_capacity
        self.max_model_len = max_model_len

    def turn_budget(self, max_completion_tokens: int) -> int:
        """the largest completion this turn may request under all three binding budgets.

        the configured per-turn cap, the engine's remaining context, and the response tensor's
        remaining width. the first applies fresh to each model turn. response_capacity is the
        whole-episode tensor width, widened by the parent for multi-turn, and the other two terms
        bound what the accumulated episode can still hold.
        """
        return min(
            max_completion_tokens,
            self.max_model_len - len(self.prefix_ids),
            self.response_capacity - len(self.response_ids),
        )

    def append_model_turn(self, turn, log_probs) -> None:
        """record one generated turn's tokens, mask, span, and logprobs."""
        turn_ids = turn["response_ids"]
        turn_start = len(self.response_ids)
        unusable_turn = turn_is_unusable(turn)
        self.response_ids.extend(turn_ids)
        # the bridge does not record or score an unusable turn, so keep its tokens
        # out of the loss with response_mask=0. verl's single-turn truncation handling
        # cannot reach
        # this custom AgentLoopOutput.
        self.response_mask.extend([0 if unusable_turn else 1] * len(turn_ids))
        # spanning it would also leave one more span than there are rewards, which
        # score_rollouts rejects as a count mismatch -- dropping the row, and with it its
        # whole group, to episode credit. the span list and the mask are independent
        # fields on the output, so zeroing above does not change this count.
        if not unusable_turn:
            self.turn_spans.append((turn_start, len(self.response_ids)))
        if self.have_logprobs and log_probs is not None:
            self.response_logprobs.extend(list(log_probs[: len(turn_ids)]))
        else:
            # one turn without logprobs makes the whole episode's vector unusable: it
            # would be shorter than response_ids and silently misalign every token
            # after the gap. drop the vector rather than emit a misaligned one.
            self.have_logprobs = False
        self.prefix_ids.extend(turn_ids)

    def glue_fits(self, glue_ids) -> bool:
        """whether an environment reply still leaves room for the next model turn to generate."""
        return not (
            len(self.prefix_ids) + len(glue_ids) + _NEXT_TURN_SLACK > self.max_model_len
            or len(self.response_ids) + len(glue_ids) + _NEXT_TURN_SLACK > self.response_capacity
        )

    def append_environment_glue(self, glue_ids) -> None:
        """record an environment reply, masked out of the loss because the model did not write it."""
        self.response_ids.extend(glue_ids)
        self.response_mask.extend([0] * len(glue_ids))
        if self.have_logprobs:
            # the model did not generate the glue, so it has no logprob. 0.0 is the
            # same filler the retired trl driver used; the zeroed
            # response_mask is what keeps these positions out of the loss.
            self.response_logprobs.extend([0.0] * len(glue_ids))
        self.prefix_ids.extend(glue_ids)


class _EpisodeSettings:
    """the per-run rollout settings the parent hands the child through the environment.

    read once per episode rather than per turn. the verl interpreter cannot import flash, so the
    parent resolves turn limits, budgets, halting, and thinking into strings and this reads them
    back.
    """

    def __init__(self):
        self.bridge_url = os.environ["FLASH_VERL_MULTITURN_URL"]
        self.max_turns = int(os.environ["FLASH_VERL_MAX_TURNS"])
        self.max_model_len = int(os.environ["FLASH_VERL_MAX_MODEL_LEN"])
        max_completion_tokens = int(os.environ["FLASH_VERL_MAX_COMPLETION_TOKENS"])
        self.max_completion_tokens = max_completion_tokens
        self.stop_sequences = tuple(
            str(value) for value in json.loads(os.environ.get("FLASH_VERL_STOP_SEQUENCES", "[]"))
        )
        self.eos_token_ids = frozenset(
            int(value) for value in json.loads(os.environ.get("FLASH_VERL_EOS_TOKEN_IDS", "[]"))
        )
        self.thinking = os.environ.get("FLASH_VERL_THINKING") == "1"


def _turn_sampling_params(
    sampling_params: dict,
    *,
    max_tokens: int,
    stop_sequences,
    eos_token_ids,
) -> dict:
    """the per-turn sampling params: the caller's, plus this turn's cap and stop conditions."""
    params = dict(sampling_params)
    params["max_tokens"] = max_tokens
    if stop_sequences:
        params["stop"] = list(stop_sequences)
        params["include_stop_str_in_output"] = True
    if eos_token_ids:
        params["stop_token_ids"] = sorted(eos_token_ids)
    return params


def _agent_loop_output_fields(prompt, episode, *, reward_score, turn_rewards) -> dict:
    """the fields one finished episode contributes to verl's AgentLoopOutput."""
    return {
        "prompt_ids": prompt.prompt_ids,
        "response_ids": episode.response_ids,
        "response_mask": episode.response_mask,
        "response_logprobs": episode.response_logprobs if episode.have_logprobs else None,
        # the training pass re-tokenizes this episode through the processor, so it needs the
        # same media the rollout conditioned on; without it the actor's forward would see
        # placeholders with no pixels behind them.
        "multi_modal_data": prompt.media_snapshot() or None,
        "mm_processor_kwargs": prompt.mm_processor_kwargs,
        "num_turns": episode.turn_count,
        # reward_score is authoritative: verl copies it to rm_scores and skips
        # reward-manager
        # computation, so the single-turn custom reward function is not used for these rows.
        "reward_score": reward_score,
        # per-turn credit assignment. the bridge returns one reward per emitted turn when
        # the environment scores turns and the run asked for per-turn credit; it returns
        # None otherwise, and the advantage shim then falls back to episode credit for the
        # whole group. carried through extra_fields because that is the only channel the
        # agent loop transports into non_tensor_batch (agent_loop._postprocess).
        "extra_fields": {
            "flash_turn_spans": list(episode.turn_spans),
            "flash_turn_rewards": turn_rewards,
        },
        "metrics": {
            "generate_sequences": episode.generated_seconds,
            "tool_calls": 0.0,
            "compute_score": 0.0,
            "num_preempted": episode.num_preempted,
        },
    }


def _episode_turn_rewards(score_payload, turn_spans):
    """the validated per-turn reward vector, or None to fall back to episode credit.

    the bridge only sends ``turns`` when per-turn credit is active and the environment returned a
    validated per-turn vector; one entry per turn the bridge was told about. a count that disagrees
    with the spans the loop actually emitted cannot be aligned to tokens, so drop to episode credit
    rather than guess an alignment.
    """
    raw_turns = score_payload.get("turns")
    if raw_turns is None:
        return None
    if len(raw_turns) == len(turn_spans):
        return [float(value) for value in raw_turns]
    print(
        f"[rl-verl] per-turn rewards ({len(raw_turns)}) do not match emitted "
        f"turns ({len(turn_spans)}); falling back to episode credit",
        flush=True,
    )
    return None


async def _score_episode(self, bridge_post, bridge_url, session_id, episode, identity):
    score_payload = await run_executor_call(
        self.loop,
        lambda: bridge_post(
            bridge_url,
            "/multiturn/score",
            # use the turns the env was told about, not generated turns: an aborted turn is never
            # recorded into env state, so the env cannot return a reward for it.
            {
                "session_id": session_id,
                "turn_count": len(episode.turn_spans),
                "identity": dict(identity),
            },
        ),
    )
    return float(score_payload["score"]), _episode_turn_rewards(score_payload, episode.turn_spans)


def _close_reply_images(images) -> None:
    for image in images:
        with contextlib.suppress(Exception):
            image.close()


def _validate_grpo_reply_media(prompt, reply_glue, step) -> None:
    expected_digests = [*prompt.image_digests, *reply_glue.image_digests]
    try:
        returned_count = int(step["image_count"])
        returned_digests = list(step["image_digests"])
    except (KeyError, TypeError, ValueError) as error:
        _close_reply_images(reply_glue.images)
        raise RuntimeError(
            "flash multi-turn bridge returned invalid cumulative media identity"
        ) from error
    if returned_count != len(expected_digests) or returned_digests != expected_digests:
        _close_reply_images(reply_glue.images)
        raise RuntimeError(
            "flash multi-turn bridge returned cumulative media that does not match the decoded "
            "environment reply"
        )


async def _grpo_run_turns(
    self,
    sampling_params: dict[str, Any],
    *,
    prompt,
    episode,
    settings,
    session_id: str,
    glue_processor,
    bridge_post,
    turn_limit: int,
) -> None:
    max_completion_tokens = settings.max_completion_tokens
    for turn_ordinal in range(turn_limit):
        max_tokens = episode.turn_budget(max_completion_tokens)
        if max_tokens <= 0:
            break
        params = _turn_sampling_params(
            sampling_params,
            max_tokens=max_tokens,
            stop_sequences=settings.stop_sequences,
            eos_token_ids=settings.eos_token_ids,
        )
        accepted_prefix = list(episode.prefix_ids)
        accepted_image_count = prompt.image_count()
        accepted_image_digests = list(prompt.image_digests)
        request_started = time.perf_counter()
        # every generation receives the exact cumulative media authenticated with its prefix.
        generated = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=episode.prefix_ids,
            sampling_params=params,
            image_data=list(prompt.images) or None,
            video_data=list(prompt.videos) or None,
            audio_data=list(prompt.audios) or None,
            mm_processor_kwargs=prompt.mm_processor_kwargs,
        )
        episode.generated_seconds += time.perf_counter() - request_started
        episode.num_preempted = sum_preemptions(episode.num_preempted, generated.num_preempted)
        turn = prepare_assistant_turn(
            self.tokenizer,
            generated.token_ids,
            stop_reason=generated.stop_reason,
            max_tokens=max_tokens,
            eos_token_ids=settings.eos_token_ids,
            stop_sequences=settings.stop_sequences,
        )
        turn_ids = turn["response_ids"]
        episode.append_model_turn(turn, generated.log_probs)
        episode.turn_count = turn_ordinal + 1
        step = await run_executor_call(
            self.loop,
            lambda turn_ordinal=turn_ordinal, turn=dict(turn), accepted_prefix=accepted_prefix, accepted_image_count=accepted_image_count, accepted_image_digests=accepted_image_digests: (
                bridge_post(
                    settings.bridge_url,
                    "/multiturn/step",
                    {
                        "session_id": session_id,
                        "turn_ordinal": turn_ordinal,
                        "accepted_prefix": accepted_prefix,
                        "response_ids": turn["response_ids"],
                        "completion_text": turn["completion_text"],
                        "truncated": turn["truncated"],
                        "skip_reason": turn["skip_reason"],
                        "image_count": accepted_image_count,
                        "image_digests": accepted_image_digests,
                    },
                )
            ),
        )
        if turn["truncated"] or turn["skip_reason"] or step["terminal"]:
            break
        env_messages = step["messages"]
        if not env_messages:
            break
        reply_glue = await glue_processor(env_messages, step.get("image_data_uris"))
        _validate_grpo_reply_media(prompt, reply_glue, step)
        glue_ids = dedup_seam_terminator(turn_ids, reply_glue.token_ids)
        if not episode.glue_fits(glue_ids):
            _close_reply_images(reply_glue.images)
            break
        prompt.append_images(reply_glue.images, reply_glue.image_digests)
        episode.append_environment_glue(glue_ids)


async def _grpo_run(
    self,
    sampling_params: dict[str, Any],
    *,
    bridge_post,
    agent_loop_output,
    **kwargs,
):
    prompt = await prepare_episode_prompt(self, kwargs["raw_prompt"])
    prompt_ids = prompt.prompt_ids
    settings = _EpisodeSettings()
    bridge_url = settings.bridge_url
    example_index = int(kwargs["index"])
    identity = kwargs.get("flash_rollout_identity")
    if not isinstance(identity, dict):
        raise RuntimeError("flash multi-turn rollout is missing its completion identity")
    identity = dict(identity)
    glue_processor = EnvGlueProcessor(self, thinking=settings.thinking)
    session_id = uuid4().hex

    # the response tensor's width. the whole EPISODE has to fit in it, not just one turn:
    # verl right-pads response_ids to this length and truncates anything longer
    # (_pad_token_ids), so a transcript that overruns it would be silently cut mid-turn.
    episode = _EpisodeTranscript(
        prompt_ids,
        response_capacity=int(self.rollout_config.response_length),
        max_model_len=settings.max_model_len,
    )
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
                    "raw_prompt": prompt.structured_messages,
                    "prompt_ids": prompt_ids,
                    "identity": dict(identity),
                    "image_count": prompt.image_count(),
                    "image_digests": list(prompt.image_digests),
                },
            ),
        )
        turn_limit = int(start["max_turns"])
        if turn_limit <= 0 or turn_limit > settings.max_turns:
            raise RuntimeError("flash multi-turn bridge returned an invalid per-example turn limit")
        await _grpo_run_turns(
            self,
            sampling_params,
            prompt=prompt,
            episode=episode,
            settings=settings,
            session_id=session_id,
            glue_processor=glue_processor,
            bridge_post=bridge_post,
            turn_limit=turn_limit,
        )
        reward_score, turn_rewards = await _score_episode(
            self, bridge_post, bridge_url, session_id, episode, identity
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
        **_agent_loop_output_fields(
            prompt,
            episode,
            reward_score=reward_score,
            turn_rewards=turn_rewards,
        )
    )


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
            return await _grpo_run(
                self,
                sampling_params,
                bridge_post=bridge_post,
                agent_loop_output=agent_loop_output,
                **kwargs,
            )

    FlashGrpoMultiTurnAgentLoop.__module__ = __name__
    FlashGrpoMultiTurnAgentLoop.__qualname__ = "FlashGrpoMultiTurnAgentLoop"
    globals()["FlashGrpoMultiTurnAgentLoop"] = FlashGrpoMultiTurnAgentLoop
    register("flash_grpo_multi_turn")(FlashGrpoMultiTurnAgentLoop)
    return FlashGrpoMultiTurnAgentLoop
