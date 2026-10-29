"""Generation-time probabilities for custom GRPO rewards (no tensors or gradients)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

from flash.engine.worker.train.core.rollout.contract import (
    LOGPROB_DEFINITIONS,
    PolicyIdentity,
    exact_fields,
    integer,
    validate_tokens,
)


@dataclass(frozen=True)
class RolloutTurn:
    """One engine generation, before text cleanup or loss masking.

    Only generated tokens have probabilities. Environment replies occur in the next
    turn's prompt, never in ``token_ids``. Special/stop and reasoning tokens are kept.
    """

    prompt_token_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    sampling: dict
    finish_reason: str

    def __post_init__(self):
        for token in self.prompt_token_ids:
            integer(token, "prompt token id")
        validate_tokens(self.token_ids, self.token_logprobs)
        if not isinstance(self.sampling, dict) or not self.sampling:
            raise ValueError("reward context requires effective sampling parameters")
        if self.finish_reason not in {"stop", "length"}:
            raise ValueError("reward context has an invalid finish reason")


@dataclass(frozen=True)
class RolloutContext:
    """Optional scorer input, identified by the policy that generated the rollout.

    ``raw_logprobs`` names the model distribution; ``processed_logprobs`` names the
    engine's distribution after sampling transformations. Values are natural logs.
    These are sampled-token scores, not full-vocabulary or target-model scores.
    """

    policy: PolicyIdentity
    optimizer_step: int
    sample_index: int
    rollout_ordinal: int
    logprob_definition: str
    turns: tuple[RolloutTurn, ...]

    def __post_init__(self):
        if not isinstance(self.policy, PolicyIdentity):
            raise ValueError("reward context requires a policy identity")
        integer(self.optimizer_step, "optimizer_step", 1)
        integer(self.sample_index, "sample_index")
        integer(self.rollout_ordinal, "rollout_ordinal")
        if self.logprob_definition not in LOGPROB_DEFINITIONS:
            raise ValueError("reward context has an invalid logprob definition")
        if self.policy.policy_version != self.optimizer_step - 1:
            raise ValueError("reward context policy version does not match optimizer step")

    def encode(self) -> dict:
        return asdict(self)

    @classmethod
    def decode(cls, value: dict) -> RolloutContext:
        exact_fields(value, cls.__dataclass_fields__)
        if not isinstance(value["turns"], (list, tuple)):
            raise ValueError("reward context turns must be a sequence")
        turns = []
        for turn in value["turns"]:
            exact_fields(turn, RolloutTurn.__dataclass_fields__)
            turns.append(
                RolloutTurn(
                    tuple(turn["prompt_token_ids"]),
                    tuple(turn["token_ids"]),
                    tuple(turn["token_logprobs"]),
                    deepcopy(turn["sampling"]),
                    turn["finish_reason"],
                )
            )
        return cls(**dict(value, policy=PolicyIdentity.decode(value["policy"]), turns=tuple(turns)))

    def validate_identity(self, identity: dict) -> None:
        if any(
            getattr(self, name) != identity.get(name)
            for name in ("optimizer_step", "sample_index", "rollout_ordinal")
        ):
            raise ValueError("reward context does not match rollout identity")
