"""Validation for multi-turn assistant turns reported back to the OPD bridge.

Split out of `bridge.py` to keep that module under the file-size gate. The bridge trusts nothing
the child reports about a turn: the child samples the tokens, but the parent is what scores them,
so a turn whose ids, text, and termination do not agree is rejected rather than scored. Keeping
that check here makes the contract readable on its own, without the surrounding HTTP plumbing.

Takes the tokenizer, eos ids, and stop sequences as arguments instead of reading them off the
bridge, so the rules stay testable without standing up a server.
"""

from __future__ import annotations

from typing import Any

from flash.engine.worker.train.opd.orchestration.gkd import _trim_trailing_stop

SKIP_REASONS = {
    "",
    "empty_completion",
    "replacement_char",
    "truncated_rollout",
}


def validated_multiturn_response(
    payload: dict,
    *,
    tokenizer: Any,
    eos_token_ids: set[int],
    stop_sequences: Any,
) -> tuple[list[int], list[int], str, str]:
    raw_response_ids = [int(token_id) for token_id in payload.get("raw_response_ids", [])]
    response_ids = [int(token_id) for token_id in payload.get("response_ids", [])]
    completion_text = payload.get("completion_text")
    if not isinstance(completion_text, str):
        raise ValueError("multi-turn assistant completion text must be a string")
    termination = str(payload.get("termination") or "")
    truncated = bool(payload.get("truncated"))
    skip_reason = str(payload.get("skip_reason") or "")
    if skip_reason not in SKIP_REASONS:
        raise ValueError("multi-turn assistant turn has an unknown skip reason")
    if truncated:
        if termination != "truncated" or skip_reason != "truncated_rollout":
            raise ValueError("multi-turn truncated assistant turn has inconsistent metadata")
        if response_ids != raw_response_ids:
            raise ValueError("multi-turn truncated assistant ids must preserve the sampled span")
    elif termination == "eos":
        # the LAST sampled token, not merely one somewhere in the span. a set-intersection test
        # accepts a backend or child bridge that kept emitting after its own declared terminal
        # boundary, and the parent then records and teacher-scores that trailing span -- corrupting
        # the transcript and the loss targets with tokens generation had already ended before. this
        # validator is the fail-closed boundary for child-reported termination, so it has to check
        # the boundary itself.
        if not raw_response_ids or raw_response_ids[-1] not in eos_token_ids:
            raise ValueError("multi-turn eos termination must end the sampled ids")
        if response_ids != raw_response_ids:
            raise ValueError("multi-turn eos response ids must preserve the sampled span")
    elif termination == "stop":
        stop_text = tokenizer.decode(raw_response_ids, skip_special_tokens=False)
        expected_ids, expected_text = _trim_trailing_stop(
            tokenizer, raw_response_ids, stop_text, stop_sequences
        )
        if expected_ids != response_ids or expected_text != completion_text:
            raise ValueError("multi-turn stop trimming does not match the legacy OPD contract")
    elif termination == "accepted_stop":
        max_tokens = int(payload.get("max_tokens", 0))
        if (
            payload.get("stop_reason") != "completed"
            or max_tokens <= 0
            or len(raw_response_ids) >= max_tokens
            or response_ids != raw_response_ids
        ):
            raise ValueError("multi-turn accepted-stop metadata is not verifiable")
    else:
        raise ValueError("multi-turn assistant turn did not end at a verified boundary")
    decoded = tokenizer.decode(response_ids, skip_special_tokens=True)
    if decoded != completion_text:
        raise ValueError("multi-turn assistant text does not match its accepted token span")
    return raw_response_ids, response_ids, completion_text, skip_reason
