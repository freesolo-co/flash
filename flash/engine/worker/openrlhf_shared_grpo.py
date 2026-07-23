"""GRPO run adapter for the shared OpenRLHF controller."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from flash.engine.worker.grpo_openrlhf import (
    OpenRLHFGRPOLoss,
    RewardResult,
    openrlhf_grpo_loss,
    post_reward_request,
)
from flash.engine.worker.openrlhf_shared_engine import (
    AdapterHandle,
    RolloutEnvelope,
    SharedMultiLoRARolloutEngine,
)
from flash.engine.worker.openrlhf_shared_scheduler import (
    SchedulerRunHooks,
    SharedEngineRunController,
)
from flash.engine.worker.openrlhf_shared_scoring import (
    ScoringKind,
    ScoringResult,
    bind_scoring_bridge,
)
from flash.engine.worker.openrlhf_shared_training import (
    SharedMultiLoRATrainingActor,
    TrainingRunState,
    TrainingStepResult,
)
from flash.engine.worker.rng import rollout_request_seed


@dataclass(frozen=True, slots=True)
class SharedGRPOConfig:
    """run-local GRPO values that do not affect engine compatibility."""

    seed: int
    prompts_per_step: int
    group_size: int
    max_response_length: int
    pad_token_id: int
    kl_coef: float = 0.0
    reward_timeout_s: float = 180.0

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if isinstance(self.prompts_per_step, bool) or self.prompts_per_step < 1:
            raise ValueError("prompts_per_step must be positive")
        if isinstance(self.group_size, bool) or self.group_size <= 1:
            raise ValueError("OpenRLHF DR-GRPO requires group_size greater than 1")
        if isinstance(self.max_response_length, bool) or self.max_response_length < 1:
            raise ValueError("max_response_length must be positive")
        if isinstance(self.pad_token_id, bool) or not isinstance(self.pad_token_id, int):
            raise ValueError("pad_token_id must be an integer")
        if not math.isfinite(float(self.kl_coef)) or self.kl_coef < 0:
            raise ValueError("kl_coef must be non-negative and finite")
        if not math.isfinite(float(self.reward_timeout_s)) or self.reward_timeout_s <= 0:
            raise ValueError("reward_timeout_s must be positive and finite")


@dataclass(frozen=True, slots=True)
class SharedGRPOPrompt:
    """one immutable scheduled prompt and reward label."""

    rendered: str
    label: int
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.rendered, str) or not self.rendered:
            raise ValueError("rendered prompt must not be empty")
        if isinstance(self.label, bool) or not isinstance(self.label, int):
            raise ValueError("prompt label must be an integer")
        if not self.token_ids or any(
            isinstance(token, bool) or not isinstance(token, int) for token in self.token_ids
        ):
            raise ValueError("prompt token ids must be a nonempty integer sequence")


@dataclass(frozen=True, slots=True)
class SharedGRPOSample:
    """one generated completion in deterministic prompt and sample order."""

    prompt: SharedGRPOPrompt
    action_token_ids: tuple[int, ...]
    rollout_log_probs: tuple[float, ...]
    truncated: bool
    canonical_prompt: str
    query: str
    request_id: str


@dataclass(frozen=True, slots=True)
class SharedGRPOTrainingBatch:
    """padded tensors consumed by the run-local differentiable log-prob hook."""

    sequences: Any
    attention_mask: Any
    action_mask: Any
    rollout_log_probs: Any
    prompt_lengths: tuple[int, ...]
    action_lengths: tuple[int, ...]

    def to(self, device: Any) -> SharedGRPOTrainingBatch:
        """move every training tensor to the active run adapter device."""

        return SharedGRPOTrainingBatch(
            sequences=self.sequences.to(device),
            attention_mask=self.attention_mask.to(device),
            action_mask=self.action_mask.to(device),
            rollout_log_probs=self.rollout_log_probs.to(device),
            prompt_lengths=self.prompt_lengths,
            action_lengths=self.action_lengths,
        )


@dataclass(frozen=True, slots=True)
class SharedGRPORolloutBatch:
    """rollouts bound to one immutable adapter version and training batch."""

    run_id: str
    step: int
    prompt_cursor: int
    handle: AdapterHandle
    samples: tuple[SharedGRPOSample, ...]
    training_batch: SharedGRPOTrainingBatch


@dataclass(frozen=True, slots=True)
class SharedGRPORewardBatch:
    """ordered reward results returned by one run-local reward bridge."""

    results: tuple[RewardResult, ...]


@dataclass(frozen=True, slots=True)
class SharedGRPOUpdateResult:
    """observable result of one exact GRPO update and adapter publication."""

    training_step: TrainingStepResult
    objective: OpenRLHFGRPOLoss


SamplingParamsFactory = Callable[[int], Any]
DecodeTokens = Callable[[Sequence[int]], str]
PolicyLogProbFunction = Callable[[Any, TrainingRunState, SharedGRPOTrainingBatch], Any]
ReferenceLogProbFunction = Callable[[Any, TrainingRunState, SharedGRPOTrainingBatch], Any]
RewardRequest = Callable[[str, dict[str, Any]], Mapping[str, Any]]


def _selected_rollout_log_prob(value: Any, token_id: int) -> float:
    if isinstance(value, Mapping):
        selected = value.get(token_id)
        if selected is None:
            raise ValueError("vLLM rollout log probabilities omitted the selected token")
        value = getattr(selected, "logprob", selected)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("vLLM rollout log probabilities must be finite")
    return number


def _decode_rollout(
    prompt: SharedGRPOPrompt,
    envelope: RolloutEnvelope,
    decode_tokens: DecodeTokens,
    max_response_length: int,
) -> SharedGRPOSample:
    outputs = getattr(envelope.output, "outputs", None)
    if not isinstance(outputs, Sequence) or len(outputs) != 1:
        raise ValueError("shared GRPO rollout must contain exactly one completion")
    generated = outputs[0]
    action_token_ids = tuple(int(token) for token in getattr(generated, "token_ids", ()))
    if len(action_token_ids) > max_response_length:
        raise ValueError("shared GRPO rollout exceeded max_response_length")
    raw_log_probs = getattr(generated, "logprobs", None)
    if not action_token_ids and raw_log_probs is None:
        raw_log_probs = ()
    if not isinstance(raw_log_probs, Sequence) or len(raw_log_probs) != len(action_token_ids):
        raise ValueError("shared GRPO rollout requires one log probability per action token")
    rollout_log_probs = tuple(
        _selected_rollout_log_prob(value, token_id)
        for token_id, value in zip(action_token_ids, raw_log_probs, strict=True)
    )
    canonical_prompt = decode_tokens(prompt.token_ids)
    query = decode_tokens((*prompt.token_ids, *action_token_ids))
    if not isinstance(canonical_prompt, str) or not canonical_prompt:
        raise ValueError("decoded shared GRPO prompt must not be empty")
    if not isinstance(query, str) or not query:
        raise ValueError("decoded shared GRPO query must not be empty")
    if not query.startswith(canonical_prompt):
        raise ValueError("decoded shared GRPO query does not start with its canonical prompt")
    return SharedGRPOSample(
        prompt=prompt,
        action_token_ids=action_token_ids,
        rollout_log_probs=rollout_log_probs,
        truncated=getattr(generated, "finish_reason", None) == "length",
        canonical_prompt=canonical_prompt,
        query=query,
        request_id=envelope.request_id,
    )


def _collate_training_batch(
    samples: Sequence[SharedGRPOSample],
    *,
    pad_token_id: int,
) -> SharedGRPOTrainingBatch:
    import torch

    if not samples:
        raise ValueError("cannot collate an empty shared GRPO rollout batch")
    prompt_lengths = tuple(len(sample.prompt.token_ids) for sample in samples)
    action_lengths = tuple(len(sample.action_token_ids) for sample in samples)
    max_sequence_length = max(
        prompt_length + action_length
        for prompt_length, action_length in zip(prompt_lengths, action_lengths, strict=True)
    )
    max_action_length = max(1, max(action_lengths))
    batch_size = len(samples)
    sequences = torch.full(
        (batch_size, max_sequence_length),
        int(pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(sequences)
    action_mask = torch.zeros((batch_size, max_action_length), dtype=torch.float32)
    rollout_log_probs = torch.zeros((batch_size, max_action_length), dtype=torch.float32)
    for row, sample in enumerate(samples):
        tokens = (*sample.prompt.token_ids, *sample.action_token_ids)
        sequence_length = len(tokens)
        action_length = len(sample.action_token_ids)
        sequences[row, :sequence_length] = torch.tensor(tokens, dtype=torch.long)
        attention_mask[row, :sequence_length] = 1
        rollout_log_probs[row, :action_length] = torch.tensor(
            sample.rollout_log_probs,
            dtype=torch.float32,
        )
        if not sample.truncated:
            action_mask[row, :action_length] = 1
    return SharedGRPOTrainingBatch(
        sequences=sequences,
        attention_mask=attention_mask,
        action_mask=action_mask,
        rollout_log_probs=rollout_log_probs,
        prompt_lengths=prompt_lengths,
        action_lengths=action_lengths,
    )


def _finite_reward_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"OpenRLHF reward bridge returned invalid {field}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"OpenRLHF reward bridge returned non-finite {field}")
    return number


def _single_reward_result(response: Mapping[str, Any]) -> RewardResult:
    if not isinstance(response, Mapping):
        raise TypeError("OpenRLHF reward bridge returned a non-object response")
    rewards = response.get("rewards")
    scores = response.get("scores")
    extra_logs = response.get("extra_logs")
    if not isinstance(rewards, list) or len(rewards) != 1:
        raise ValueError("OpenRLHF reward bridge returned invalid rewards")
    if not isinstance(scores, list) or len(scores) != 1:
        raise ValueError("OpenRLHF reward bridge returned invalid scores")
    if not isinstance(extra_logs, Mapping):
        raise ValueError("OpenRLHF reward bridge returned invalid extra_logs")
    reward = _finite_reward_number(rewards[0], "rewards")
    score = _finite_reward_number(scores[0], "scores")
    metrics: dict[str, float] = {}
    for name, values in extra_logs.items():
        if not isinstance(name, str) or not isinstance(values, list) or len(values) != 1:
            raise ValueError("OpenRLHF reward bridge returned invalid metric values")
        metrics[name] = _finite_reward_number(values[0], f"metric {name!r}")
    return RewardResult(reward, score, metrics)


def _validate_reward_url(url: str) -> str:
    endpoint = str(url).strip()
    parsed = urlsplit(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("shared GRPO reward bridge must use an HTTP loopback endpoint")
    if not parsed.path.startswith("/reward/"):
        raise ValueError("shared GRPO reward bridge must include its run-local reward path")
    return endpoint


class SharedGRPORunAdapter:
    """compose one GRPO run with the shared rollout, scoring, training, and scheduler layers."""

    def __init__(
        self,
        run_id: str,
        *,
        config: SharedGRPOConfig,
        training_actor: SharedMultiLoRATrainingActor,
        rollout_engine: SharedMultiLoRARolloutEngine,
        reward_url: str,
        sampling_params_factory: SamplingParamsFactory,
        decode_tokens: DecodeTokens,
        policy_log_probs: PolicyLogProbFunction,
        reference_log_probs: ReferenceLogProbFunction | None = None,
        reward_request: RewardRequest | None = None,
    ) -> None:
        normalized_run_id = str(run_id).strip()
        if not normalized_run_id:
            raise ValueError("run_id must not be empty")
        if not callable(sampling_params_factory):
            raise TypeError("sampling_params_factory must be callable")
        if not callable(decode_tokens):
            raise TypeError("decode_tokens must be callable")
        if not callable(policy_log_probs):
            raise TypeError("policy_log_probs must be callable")
        if config.kl_coef > 0 and not callable(reference_log_probs):
            raise ValueError("KL-enabled shared GRPO requires reference_log_probs")
        if reward_request is not None and not callable(reward_request):
            raise TypeError("reward_request must be callable")

        state = training_actor.run_state(normalized_run_id)
        prompts = state.dataloader
        if not isinstance(prompts, Sequence) or not prompts:
            raise ValueError("shared GRPO requires a nonempty indexable prompt schedule")
        if not all(isinstance(prompt, SharedGRPOPrompt) for prompt in prompts):
            raise TypeError("shared GRPO prompt schedule contains an invalid prompt")
        if state.handle.run_id != normalized_run_id:
            raise ValueError("training adapter handle does not match the GRPO run")

        self.run_id = normalized_run_id
        self.config = config
        self._training_actor = training_actor
        self._rollout_engine = rollout_engine
        if reward_request is None:

            def request_with_timeout(url: str, payload: dict[str, Any]) -> Mapping[str, Any]:
                return post_reward_request(
                    url,
                    payload,
                    timeout=config.reward_timeout_s,
                )

            reward_request = request_with_timeout
        self._reward_bridge = bind_scoring_bridge(
            _validate_reward_url(reward_url),
            reward_request,
        )
        self._sampling_params_factory = sampling_params_factory
        self._decode_tokens = decode_tokens
        self._policy_log_probs = policy_log_probs
        self._reference_log_probs = reference_log_probs
        self._initial_global_step = state.global_step
        self._last_update: SharedGRPOUpdateResult | None = None

    @property
    def last_update(self) -> SharedGRPOUpdateResult | None:
        """return the most recent completed objective and publication result."""

        return self._last_update

    def scheduler_hooks(self) -> SchedulerRunHooks:
        """return the PR4 callbacks for this GRPO run."""

        return SchedulerRunHooks(
            rollout=self.rollout,
            scoring_payload=self.scoring_payload,
            update_and_publish=self.update_and_publish,
            cleanup=self.cleanup,
        )

    def add_to_controller(
        self,
        controller: SharedEngineRunController,
        *,
        total_steps: int,
        weight: float = 1.0,
    ) -> None:
        """admit this run to the shared scheduler with its isolated reward bridge."""

        controller.add_run(
            self.run_id,
            hooks=self.scheduler_hooks(),
            scoring_kind=ScoringKind.REWARD,
            scoring_bridge=self.score,
            total_steps=total_steps,
            weight=weight,
        )

    async def cleanup(self, run_id: str) -> None:
        """remove this run's current adapter from the shared rollout engine."""

        self._require_identity(run_id)
        state = self._training_actor.run_state(self.run_id)
        await self._rollout_engine.remove_run(self.run_id, state.handle.version)

    async def rollout(self, run_id: str, step: int) -> SharedGRPORolloutBatch:
        """generate one ordered GRPO batch through the run's current adapter handle."""

        self._require_identity(run_id)
        state = self._training_actor.run_state(self.run_id)
        expected_global_step = self._initial_global_step + step
        if state.global_step != expected_global_step:
            raise ValueError(
                f"shared GRPO scheduler step {step} expected training step "
                f"{expected_global_step}, got {state.global_step}"
            )
        prompts = state.dataloader
        prompt_cursor = state.prompt_cursor
        selected = tuple(
            prompts[(prompt_cursor + offset) % len(prompts)]
            for offset in range(self.config.prompts_per_step)
        )
        handle = state.handle
        requests = []
        for prompt_offset, prompt in enumerate(selected):
            for sample_offset in range(self.config.group_size):
                ordinal = (prompt_cursor + prompt_offset) * self.config.group_size + sample_offset
                seed = rollout_request_seed(self.config.seed, ordinal)
                sampling_params = self._sampling_params_factory(seed)
                if getattr(sampling_params, "seed", None) != seed:
                    raise ValueError("sampling_params_factory must preserve the deterministic seed")
                request_id = (
                    f"{self.run_id}-step-{step}-prompt-{prompt_offset}-sample-{sample_offset}"
                )
                requests.append(
                    self._rollout_engine.generate(
                        handle,
                        {"prompt_token_ids": list(prompt.token_ids)},
                        sampling_params,
                        request_id=request_id,
                    )
                )
        envelopes = await asyncio.gather(*requests)
        if any(envelope.handle != handle for envelope in envelopes):
            raise ValueError("shared GRPO rollout returned a mismatched adapter handle")
        ordered_prompts = tuple(
            prompt for prompt in selected for _sample_offset in range(self.config.group_size)
        )
        samples = tuple(
            _decode_rollout(
                prompt,
                envelope,
                self._decode_tokens,
                self.config.max_response_length,
            )
            for prompt, envelope in zip(ordered_prompts, envelopes, strict=True)
        )
        return SharedGRPORolloutBatch(
            run_id=self.run_id,
            step=step,
            prompt_cursor=prompt_cursor,
            handle=handle,
            samples=samples,
            training_batch=_collate_training_batch(
                samples,
                pad_token_id=self.config.pad_token_id,
            ),
        )

    def scoring_payload(
        self,
        run_id: str,
        step: int,
        rollout: SharedGRPORolloutBatch,
    ) -> dict[str, Any]:
        """build ordered one-completion requests for the existing reward bridge."""

        self._require_batch(run_id, step, rollout)
        return {
            "samples": [
                {
                    "query": [sample.query],
                    "prompts": [sample.canonical_prompt],
                    "labels": [sample.prompt.label],
                }
                for sample in rollout.samples
            ]
        }

    def score(self, payload: dict[str, Any]) -> SharedGRPORewardBatch:
        """score one batch through only this run's fail-closed loopback bridge."""

        samples = payload.get("samples") if isinstance(payload, Mapping) else None
        expected = self.config.prompts_per_step * self.config.group_size
        if not isinstance(samples, list) or len(samples) != expected:
            raise ValueError("shared GRPO scoring payload has an invalid sample count")
        if not all(isinstance(sample, Mapping) for sample in samples):
            raise TypeError("shared GRPO scoring payload contains a non-object sample")
        results = tuple(
            _single_reward_result(self._reward_bridge(dict(sample))) for sample in samples
        )
        return SharedGRPORewardBatch(results)

    async def update_and_publish(
        self,
        run_id: str,
        step: int,
        rollout: SharedGRPORolloutBatch,
        scoring_result: ScoringResult,
    ) -> SharedGRPOUpdateResult:
        """apply exact GRPO math to one adapter and publish its next immutable version."""

        import torch

        self._require_batch(run_id, step, rollout)
        if scoring_result.identity.run_id != self.run_id or scoring_result.identity.step != step:
            raise ValueError("shared GRPO scoring identity does not match the update")
        rewards = scoring_result.value
        if not isinstance(rewards, SharedGRPORewardBatch):
            raise TypeError("shared GRPO scoring result has an invalid value")
        if len(rewards.results) != len(rollout.samples):
            raise ValueError("shared GRPO reward count does not match the rollout batch")
        state = self._training_actor.run_state(self.run_id)
        if state.handle != rollout.handle:
            raise ValueError("shared GRPO update received a stale adapter rollout")
        if state.prompt_cursor != rollout.prompt_cursor:
            raise ValueError("shared GRPO update received a stale prompt cursor")
        reward_tensor = torch.tensor(
            [result.reward for result in rewards.results],
            dtype=torch.float32,
        )
        captured: list[OpenRLHFGRPOLoss] = []

        def loss_function(model: Any, active_state: TrainingRunState) -> Any:
            device_batch = rollout.training_batch.to(active_state.adapter_parameters[0].device)
            model.eval()
            try:
                with torch.no_grad():
                    old_action_log_probs = self._policy_log_probs(
                        model,
                        active_state,
                        device_batch,
                    ).detach()
                    base_action_log_probs = (
                        self._reference_log_probs(
                            model,
                            active_state,
                            device_batch,
                        ).detach()
                        if self._reference_log_probs is not None
                        else None
                    )
            finally:
                model.train()
            action_log_probs = self._policy_log_probs(
                model,
                active_state,
                device_batch,
            )
            objective = openrlhf_grpo_loss(
                action_log_probs,
                old_action_log_probs,
                device_batch.rollout_log_probs,
                reward_tensor.to(action_log_probs.device),
                device_batch.action_mask,
                self.config.max_response_length,
                self.config.group_size,
                base_action_log_probs=base_action_log_probs,
                kl_coef=self.config.kl_coef,
            )
            captured.append(objective)
            return objective.loss

        training_step = await self._training_actor.step(self.run_id, loss_function)
        if len(captured) != 1:
            raise RuntimeError("shared GRPO loss hook did not execute exactly once")
        self._training_actor.advance_prompt_cursor(
            self.run_id,
            self.config.prompts_per_step,
        )
        objective = captured[0]
        detached_objective = OpenRLHFGRPOLoss(
            objective.loss.detach(),
            objective.policy_loss.detach(),
            objective.kl_loss.detach(),
            objective.advantages.detach(),
        )
        result = SharedGRPOUpdateResult(training_step, detached_objective)
        self._last_update = result
        return result

    def _require_identity(self, run_id: str) -> None:
        if run_id != self.run_id:
            raise ValueError("shared GRPO callback received another run's identity")

    def _require_batch(
        self,
        run_id: str,
        step: int,
        rollout: SharedGRPORolloutBatch,
    ) -> None:
        self._require_identity(run_id)
        if not isinstance(rollout, SharedGRPORolloutBatch):
            raise TypeError("shared GRPO callback received an invalid rollout batch")
        if rollout.run_id != self.run_id or rollout.step != step:
            raise ValueError("shared GRPO rollout identity does not match the callback")
