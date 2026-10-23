"""Compact episode receipts checked before tensors enter optimization."""

from .contract import (
    PROTOCOL_VERSION,
    RolloutRequest,
    digest,
    exact_fields,
    integer,
    token_digest,
    validate_result_header,
)

EPISODE_FIELD = "flash_rollout_contract"


def validate_episode(receipt, prompt, response, logprobs, loss_mask, *, run_id, optimizer_step):
    exact_fields(
        receipt, ("protocol_version", "turns", "assistant_mask", "token_digest", "termination")
    )
    if (
        type(receipt["protocol_version"]) is not int
        or receipt["protocol_version"] != PROTOCOL_VERSION
    ):
        raise ValueError("unsupported rollout episode protocol version")
    if receipt["token_digest"] != token_digest(prompt, response, logprobs, loss_mask):
        raise ValueError("rollout training tensors do not match the accepted episode")
    if receipt["termination"] not in {
        "stop",
        "length",
        "turn_limit",
        "environment_terminal",
        "no_followup",
        "context_limit",
        "response_limit",
        "unusable_turn",
    }:
        raise ValueError("unknown rollout episode termination reason")
    assistant_mask = [0] * len(response)
    previous_end = 0
    first_request = None
    if not isinstance(receipt["turns"], list) or not receipt["turns"]:
        raise ValueError("rollout episode must contain generation evidence")
    for index, turn in enumerate(receipt["turns"]):
        exact_fields(turn, ("start", "end", "result"))
        start, end = integer(turn["start"], "turn start"), integer(turn["end"], "turn end")
        if not previous_end <= start <= end <= len(response):
            raise ValueError("rollout turn boundaries overlap or exceed tokens")
        previous_end = end
        result = turn["result"]
        exact_fields(result, ("request", "load_ack", "termination", "sampling", "token_digest"))
        request = RolloutRequest.decode(result["request"])
        validate_result_header(request, result)
        if request.prompt_digest != digest(prompt + response[:start]):
            raise ValueError("rollout turn context does not match its request")
        if (
            request.policy.run_id != run_id
            or request.optimizer_step != optimizer_step
            or request.policy.policy_version != optimizer_step - 1
            or request.turn_index != index
        ):
            raise ValueError("stale or mismatched rollout episode")
        if first_request is None:
            first_request = request
        elif (
            request.policy != first_request.policy
            or request.sample_index != first_request.sample_index
            or request.rollout_ordinal != first_request.rollout_ordinal
            or request.attempt_id != first_request.attempt_id
            or request.logprob_definition != first_request.logprob_definition
        ):
            raise ValueError("rollout episode mixes policies or attempts")
        if not result["load_ack"] or result["termination"] not in {"stop", "length"}:
            raise ValueError("rollout episode contains unacknowledged or cancelled generation")
        assistant_mask[start:end] = [1] * (end - start)
    if receipt["assistant_mask"] != assistant_mask or any(
        type(value) is not int for value in receipt["assistant_mask"]
    ):
        raise ValueError("rollout assistant mask does not match turn boundaries")
    if any(
        loss and not assistant for loss, assistant in zip(loss_mask, assistant_mask, strict=True)
    ):
        raise ValueError("rollout loss mask includes environment tokens")
    return first_request
