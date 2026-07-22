"""standalone child-side multi-turn OPD rollout support for verl 0.8.0."""

from __future__ import annotations

import contextlib
import json
import os
import time
from typing import Any
from uuid import uuid4

_ALLOWED_MESSAGE_KEYS = frozenset({"role", "content"})
_PROBE = "flash-env-glue-probe"


def normalize_token_ids(value) -> list[int]:
    """normalize tokenizer outputs to one flat list of integer token ids."""
    if isinstance(value, dict):
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if value and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, list):
        raise TypeError("tokenizer output must be list-like")
    return [int(token_id.item() if hasattr(token_id, "item") else token_id) for token_id in value]


def validate_teacher_messages(messages: list[dict], *, source: str) -> list[dict]:
    """require the exact role/content transcript shape used by legacy teacher scoring."""
    if not isinstance(messages, list):
        raise ValueError(f"{source} messages must be a list")
    normalized = []
    for position, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{source} message {position} must be an object")
        extras = sorted(
            key for key, value in message.items() if key not in _ALLOWED_MESSAGE_KEYS and value is not None
        )
        if extras:
            raise ValueError(
                f"{source} message {position} carries unsupported transcript metadata {extras}; "
                "tool names, call ids, tool calls, and other message fields cannot be represented "
                "in the legacy OPD teacher transcript"
            )
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"{source} message {position} has an invalid role")
        if not isinstance(content, str):
            raise ValueError(
                f"{source} message {position} content must be text for multi-turn OPD"
            )
        normalized.append({"role": role, "content": content})
    return normalized


def _tokenize_text(tokenizer, text: str) -> list[int]:
    return normalize_token_ids(tokenizer(text, add_special_tokens=False))


def _dedup_seam_terminator(response_ids: list[int], glue_ids: list[int]) -> list[int]:
    """keep the sampled terminator and drop the duplicate leading glue copy."""
    if response_ids and glue_ids and response_ids[-1] == glue_ids[0]:
        return glue_ids[1:]
    return glue_ids


class EnvGlueTokenizer:
    """derive exact inter-turn glue without re-rendering accepted transcript history."""

    def __init__(self, tokenizer, *, thinking: bool, cache_size: int = 8192) -> None:
        self.tokenizer = tokenizer
        self.thinking = bool(thinking)
        self.cache_size = int(cache_size)
        self.cache: dict[str, list[int]] = {}

    def __call__(self, env_messages: list[dict]) -> list[int]:
        messages = validate_teacher_messages(env_messages, source="environment reply")
        key = json.dumps(messages, sort_keys=True, separators=(",", ":"))
        cached = self.cache.get(key)
        if cached is not None:
            return list(cached)
        text = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": _PROBE}, *messages],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=self.thinking,
        )
        first = text.find(_PROBE)
        if first == -1 or text.find(_PROBE, first + len(_PROBE)) != -1:
            raise ValueError(
                "multi-turn OPD could not uniquely locate the assistant-content probe in the chat "
                "template; exact inter-turn glue cannot be recovered"
            )
        glue_ids = _tokenize_text(self.tokenizer, text[first + len(_PROBE) :])
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = list(glue_ids)
        return glue_ids


def validate_glue_template(tokenizer, *, thinking: bool) -> None:
    """fail before rollout when assistant content cannot round-trip through the template."""
    EnvGlueTokenizer(tokenizer, thinking=thinking)(
        [{"role": "user", "content": "flash multi-turn glue validation"}]
    )


def _trim_trailing_stop(
    tokenizer, response_ids: list[int], stop_text: str, stop_sequences: tuple[str, ...]
) -> tuple[list[int], str]:
    stop = max(
        (value for value in stop_sequences if value and stop_text.endswith(value)),
        key=len,
        default="",
    )
    ids = list(response_ids)
    if not stop:
        return ids, tokenizer.decode(ids, skip_special_tokens=True)
    keep_length = len(stop_text) - len(stop)
    kept = len(ids)
    while kept > 0 and len(tokenizer.decode(ids[:kept], skip_special_tokens=False)) > keep_length:
        kept -= 1
    ids = ids[:kept]
    return ids, tokenizer.decode(ids, skip_special_tokens=True)


def prepare_assistant_turn(
    tokenizer,
    token_ids: list[int],
    *,
    stop_reason: str | None,
    max_tokens: int,
    eos_token_ids: frozenset[int],
    stop_sequences: tuple[str, ...],
) -> dict[str, Any]:
    """apply the legacy termination, stop trimming, empty, and replacement-char gates."""
    raw_ids = [int(token_id) for token_id in token_ids]
    stop_text = tokenizer.decode(raw_ids, skip_special_tokens=False)
    ended_by_eos = bool(eos_token_ids and not eos_token_ids.isdisjoint(raw_ids))
    ended_by_stop = any(
        value and stop_text.endswith(value) for value in stop_sequences
    )
    ended_before_cap = stop_reason == "completed" and len(raw_ids) < int(max_tokens)
    terminated = ended_by_eos or ended_by_stop or ended_before_cap
    if stop_reason == "aborted":
        terminated = False
    termination = (
        "eos"
        if ended_by_eos
        else "stop"
        if ended_by_stop
        else "accepted_stop"
        if ended_before_cap
        else "truncated"
    )
    response_ids = raw_ids
    completion_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    if terminated and ended_by_stop:
        response_ids, completion_text = _trim_trailing_stop(
            tokenizer, response_ids, stop_text, stop_sequences
        )
    if not terminated:
        return {
            "raw_response_ids": raw_ids,
            "response_ids": response_ids,
            "completion_text": completion_text,
            "termination": termination,
            "stop_reason": stop_reason,
            "max_tokens": int(max_tokens),
            "truncated": True,
            "skip_reason": "truncated_rollout",
        }
    if not completion_text.strip():
        return {
            "raw_response_ids": raw_ids,
            "response_ids": response_ids,
            "completion_text": completion_text,
            "termination": termination,
            "stop_reason": stop_reason,
            "max_tokens": int(max_tokens),
            "truncated": False,
            "skip_reason": "empty_completion",
        }
    if "�" in completion_text:
        return {
            "raw_response_ids": raw_ids,
            "response_ids": response_ids,
            "completion_text": completion_text,
            "termination": termination,
            "stop_reason": stop_reason,
            "max_tokens": int(max_tokens),
            "truncated": False,
            "skip_reason": "replacement_char",
        }
    return {
        "raw_response_ids": raw_ids,
        "response_ids": response_ids,
        "completion_text": completion_text,
        "termination": termination,
        "stop_reason": stop_reason,
        "max_tokens": int(max_tokens),
        "truncated": False,
        "skip_reason": "",
    }


def _sum_preemptions(current: int, value: int | None) -> int:
    if value is None:
        return current
    if current < 0:
        return int(value)
    return current + int(value)


def build_flash_multi_turn_agent_loop(
    *,
    register,
    agent_loop_base,
    agent_loop_output,
    post_json,
    deterministic_seed,
):
    """build and register the child loop without importing verl in the parent interpreter."""

    @register("flash_multi_turn")
    class FlashMultiTurnAgentLoop(agent_loop_base):
        async def run(self, sampling_params: dict[str, Any], **kwargs):
            raw_prompt = validate_teacher_messages(
                [dict(message) for message in kwargs["raw_prompt"]], source="initial prompt"
            )
            prompt_ids = await self.apply_chat_template(raw_prompt)
            bridge_url = os.environ["FLASH_OPD_BRIDGE_URL"]
            bridge_token = os.environ["FLASH_OPD_BRIDGE_TOKEN"]
            flash_seed = int(os.environ["FLASH_OPD_SEED"])
            global_step = int(kwargs["global_steps"])
            example_index = int(kwargs["index"])
            rollout_ordinal = int(kwargs.get("session_id", 0))
            max_turns = int(os.environ["FLASH_OPD_MAX_TURNS"])
            max_model_len = int(os.environ["FLASH_OPD_MAX_MODEL_LEN"])
            capabilities = set(
                json.loads(os.environ.get("FLASH_OPD_ENV_CAPABILITIES", "[]"))
            )
            required_capabilities = {
                "new_rollout_state",
                "record_model_turn",
                "env_reply",
                "rollout_done",
            }
            if capabilities != required_capabilities:
                raise RuntimeError("multi-turn OPD environment capability metadata is invalid")
            stop_sequences = tuple(
                str(value) for value in json.loads(os.environ.get("FLASH_OPD_STOP_SEQUENCES", "[]"))
            )
            eos_token_ids = frozenset(
                int(value) for value in json.loads(os.environ.get("FLASH_OPD_EOS_TOKEN_IDS", "[]"))
            )
            thinking = os.environ.get("FLASH_OPD_THINKING") == "1"
            glue_tokenizer = EnvGlueTokenizer(self.tokenizer, thinking=thinking)
            session_id = f"{uuid4().hex}-{global_step}-{example_index}-{rollout_ordinal}"
            outputs = []
            generated_seconds = 0.0
            num_preempted = -1
            started = False
            try:
                start = await self.loop.run_in_executor(
                    None,
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
                started = True
                turn_limit = int(start["max_turns"])
                if turn_limit <= 0 or turn_limit > max_turns:
                    raise RuntimeError("multi-turn bridge returned an invalid per-example turn limit")
                prefix_ids = list(prompt_ids)
                for turn_ordinal in range(turn_limit):
                    remaining = max_model_len - len(prefix_ids) - 8
                    if remaining <= 0:
                        break
                    max_tokens = min(int(self.rollout_config.response_length), remaining)
                    params = dict(sampling_params)
                    params["max_tokens"] = max_tokens
                    params["seed"] = deterministic_seed(
                        flash_seed,
                        global_step,
                        example_index,
                        rollout_ordinal,
                        turn_ordinal,
                    )
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
                    num_preempted = _sum_preemptions(num_preempted, generated.num_preempted)
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
                    step = await self.loop.run_in_executor(
                        None,
                        lambda turn_ordinal=turn_ordinal, prefix_ids=list(prefix_ids), turn=dict(turn): post_json(
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
                        ),
                    )
                    outputs.append(
                        agent_loop_output(
                            prompt_ids=list(prefix_ids),
                            response_ids=list(response_ids),
                            response_mask=[1] * len(response_ids),
                            response_logprobs=response_logprobs,
                            num_turns=turn_ordinal + 1,
                            metrics={
                                "generate_sequences": generated_seconds,
                                "tool_calls": 0.0,
                                "compute_score": 0.0,
                                "num_preempted": num_preempted,
                            },
                            extra_fields=dict(generated.extra_fields or {}),
                        )
                    )
                    if turn["truncated"] or turn["skip_reason"] or step["terminal"]:
                        break
                    prefix_ids.extend(response_ids)
                    env_messages = validate_teacher_messages(
                        step["messages"], source="environment reply"
                    )
                    if not env_messages:
                        break
                    glue_ids = _dedup_seam_terminator(
                        response_ids, glue_tokenizer(env_messages)
                    )
                    if len(prefix_ids) + len(glue_ids) > max_model_len - 8:
                        break
                    prefix_ids.extend(glue_ids)
                score_payload = await self.loop.run_in_executor(
                    None,
                    lambda: post_json(
                        bridge_url,
                        bridge_token,
                        "/multiturn/score",
                        {"session_id": session_id},
                    ),
                )
                scored_turns = score_payload["turns"]
                if len(scored_turns) != len(outputs):
                    raise RuntimeError("multi-turn bridge returned the wrong number of teacher rows")
                for output, scored in zip(outputs, scored_turns, strict=True):
                    output.extra_fields["flash_teacher_ids"] = scored["teacher_ids"]
                    output.extra_fields["flash_teacher_logprobs"] = scored["teacher_logprobs"]
                return outputs
            finally:
                if started:
                    with contextlib.suppress(Exception):
                        await self.loop.run_in_executor(
                            None,
                            lambda: post_json(
                                bridge_url,
                                bridge_token,
                                "/multiturn/close",
                                {"session_id": session_id},
                            ),
                        )

    FlashMultiTurnAgentLoop.__module__ = __name__
    return FlashMultiTurnAgentLoop
