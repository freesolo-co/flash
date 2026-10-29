"""GRPO adapter for contract-checked generation and episode ingestion."""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

from flash.engine.worker.train.core.child.observations import ObservedRolloutClient
from flash.engine.worker.train.core.child.rollout_contract import REQUEST_FIELD, RESULT_FIELD
from flash.engine.worker.train.core.rollout.contract import (
    PROTOCOL_VERSION,
    RolloutRequest,
    digest,
    integer,
    token_digest,
    validate_result,
)
from flash.engine.worker.train.core.rollout.episode import EPISODE_FIELD, validate_episode
from flash.engine.worker.train.core.rollout.identity import model_identity, policy_identity
from flash.envs.scoring.context import RolloutContext, RolloutTurn


class ContractRolloutClient(ObservedRolloutClient):
    def __init__(self, client):
        super().__init__(client)
        self._scope = ContextVar("rollout_contract", default=None)
        self._model_identity = None

    @contextmanager
    def episode_contract(self, agent, identity):
        if identity["validate"]:
            raise ValueError("rollout V1 GRPO adapter supports training episodes only")
        if self._model_identity is None:
            self._model_identity = agent_model_identity(agent)
        policy = policy_identity(self._model_identity, identity["optimizer_step"] - 1)
        definition = agent.config.actor_rollout_ref.rollout.logprobs_mode
        token = self._scope.set(
            {
                "identity": identity,
                "policy": policy,
                "definition": definition,
                "attempt_id": uuid4().hex,
                "turns": [],
            }
        )
        try:
            with self.episode():
                yield
        finally:
            self._scope.reset(token)

    async def generate(self, *, prompt_ids, sampling_params, request_id, **kwargs):
        scope = self._scope.get()
        if scope is None:
            raise RuntimeError("rollout generation requires a contract scope")
        request = RolloutRequest(
            PROTOCOL_VERSION,
            scope["policy"],
            scope["identity"]["optimizer_step"],
            scope["identity"]["sample_index"],
            scope["identity"]["rollout_ordinal"],
            scope["attempt_id"],
            len(scope["turns"]),
            request_id,
            digest(prompt_ids),
            scope["definition"],
        )
        params = dict(sampling_params)
        if REQUEST_FIELD in params:
            raise ValueError("rollout contract was already attached")
        params[REQUEST_FIELD] = request.encode()
        output = await super().generate(
            prompt_ids=prompt_ids,
            sampling_params=params,
            request_id=request_id,
            **kwargs,
        )
        result = output.extra_fields.get(RESULT_FIELD)
        validate_result(request, result, output.token_ids, output.log_probs)
        scope["turns"].append(
            (list(prompt_ids), list(output.token_ids), list(output.log_probs), result)
        )
        return output

    def reward_context(self) -> dict:
        """Snapshot validated engine observations before any text cleanup or loss masking."""
        scope = self._scope.get()
        if scope is None:
            raise RuntimeError("reward context requires a contract scope")
        identity = scope["identity"]
        return RolloutContext(
            policy=scope["policy"],
            optimizer_step=identity["optimizer_step"],
            sample_index=identity["sample_index"],
            rollout_ordinal=identity["rollout_ordinal"],
            logprob_definition=scope["definition"],
            turns=tuple(
                RolloutTurn(
                    tuple(prompt),
                    tuple(tokens),
                    tuple(logs),
                    deepcopy(result["sampling"]),
                    result["termination"],
                )
                for prompt, tokens, logs, result in scope["turns"]
            ),
        ).encode()

    def annotate(self, output):
        scope = self._scope.get()
        if scope is None:
            raise RuntimeError("rollout annotation requires a contract scope")
        generated = scope["turns"]
        spans = output.extra_fields.pop("flash_all_turn_spans", None)
        if spans is None:
            spans = [(0, len(output.response_ids))]
        if len(spans) != len(generated):
            raise ValueError("rollout generation count does not match episode turn boundaries")
        turns = []
        assistant_mask = [0] * len(output.response_ids)
        for (start, end), (prompt, tokens, logs, result) in zip(spans, generated, strict=True):
            if (
                prompt != output.prompt_ids + output.response_ids[:start]
                or output.response_ids[start:end] != tokens[: end - start]
                or output.response_logprobs is None
                or output.response_logprobs[start:end] != logs[: end - start]
            ):
                raise ValueError("rollout episode changed generated tokens, context or logprobs")
            assistant_mask[start:end] = [1] * (end - start)
            turns.append({"start": start, "end": end, "result": result})
        receipt = {
            "protocol_version": PROTOCOL_VERSION,
            "turns": turns,
            "assistant_mask": assistant_mask,
            "termination": output.extra_fields.pop(
                "flash_episode_termination", turns[-1]["result"]["termination"] if turns else "stop"
            ),
            "token_digest": token_digest(
                output.prompt_ids,
                output.response_ids,
                output.response_logprobs,
                output.response_mask,
            ),
        }
        validate_episode(
            receipt,
            output.prompt_ids,
            output.response_ids,
            output.response_logprobs,
            output.response_mask,
            run_id=scope["policy"].run_id,
            optimizer_step=scope["identity"]["optimizer_step"],
        )
        output.extra_fields.pop(RESULT_FIELD, None)
        output.extra_fields[EPISODE_FIELD] = receipt
        return super().annotate(output)


def agent_model_identity(agent):
    model = agent.config.actor_rollout_ref.model
    return model_identity(
        SimpleNamespace(
            path=model.path,
            runtime_options=model.runtime_options,
            tokenizer=agent.tokenizer,
            processor=getattr(agent, "processor", None),
        )
    )


def expected_policy(agent, version):
    if not hasattr(agent, "_rollout_model_identity"):
        agent._rollout_model_identity = agent_model_identity(agent)
    return policy_identity(agent._rollout_model_identity, version)


def validate_queue_rollouts(
    batch,
    *,
    run_id,
    optimizer_step,
    accepted,
    expected_policy,
    logprob_definition,
):
    import transfer_queue as tq

    fields = tq.kv_batch_get(
        keys=batch.keys,
        partition_id=batch.partition_id,
        select_fields=[
            "prompts",
            "responses",
            "rollout_log_probs",
            "response_mask",
            "extra_fields",
            "index",
            "session_id",
            "global_steps",
        ],
    )
    pending = set()
    for index, extra in enumerate(fields["extra_fields"]):
        request = validate_episode(
            extra.get(EPISODE_FIELD),
            fields["prompts"][index].tolist(),
            fields["responses"][index].tolist(),
            fields["rollout_log_probs"][index].tolist(),
            fields["response_mask"][index].tolist(),
            run_id=run_id,
            optimizer_step=optimizer_step,
        )
        if request.policy != expected_policy or request.logprob_definition != logprob_definition:
            raise ValueError("rollout policy or logprob definition does not match the trainer")
        for field, expected in (
            ("index", request.sample_index),
            ("session_id", request.rollout_ordinal),
            ("global_steps", request.optimizer_step),
        ):
            if integer(fields[field][index], field) != expected:
                raise ValueError("rollout receipt identity does not match its training row")
        key = (request.sample_index, request.rollout_ordinal)
        if key in pending or key in accepted:
            raise ValueError("duplicate logical rollout at training ingestion")
        pending.add(key)
    accepted.update(pending)
