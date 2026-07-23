"""OPD run adapter for the shared OpenRLHF controller."""

from __future__ import annotations

import asyncio
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from flash.engine.worker.opd_openrlhf import (
    deterministic_rollout_seed,
    flash_groupwise_reverse_kl,
    post_teacher_request_with_retry,
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


@dataclass(frozen=True, slots=True)
class SharedOPDConfig:
    """run-local OPD values that do not affect engine compatibility."""

    seed: int
    prompts_per_step: int
    group_size: int
    max_response_length: int
    pad_token_id: int
    kl_penalty_coef: float
    no_signal_attempts: int = 3
    stop_sequences: tuple[str, ...] = ()
    eos_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if isinstance(self.prompts_per_step, bool) or self.prompts_per_step < 1:
            raise ValueError("prompts_per_step must be positive")
        if isinstance(self.group_size, bool) or self.group_size < 1:
            raise ValueError("group_size must be positive")
        if isinstance(self.max_response_length, bool) or self.max_response_length < 1:
            raise ValueError("max_response_length must be positive")
        if isinstance(self.pad_token_id, bool) or not isinstance(self.pad_token_id, int):
            raise ValueError("pad_token_id must be an integer")
        if not math.isfinite(float(self.kl_penalty_coef)) or self.kl_penalty_coef <= 0:
            raise ValueError("kl_penalty_coef must be positive and finite")
        if isinstance(self.no_signal_attempts, bool) or self.no_signal_attempts < 1:
            raise ValueError("no_signal_attempts must be positive")
        if any(not isinstance(stop, str) or not stop for stop in self.stop_sequences):
            raise ValueError("stop_sequences must contain nonempty strings")
        if any(
            isinstance(token, bool) or not isinstance(token, int) for token in self.eos_token_ids
        ):
            raise ValueError("eos_token_ids must contain integers")


@dataclass(frozen=True, slots=True)
class SharedOPDPrompt:
    """one immutable scheduled prompt with its teacher bridge index."""

    example_index: int
    rendered: str
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.example_index, bool)
            or not isinstance(self.example_index, int)
            or self.example_index < 0
        ):
            raise ValueError("example_index must be a non-negative integer")
        if not isinstance(self.rendered, str) or not self.rendered:
            raise ValueError("rendered prompt must not be empty")
        if not self.token_ids or any(
            isinstance(token, bool) or not isinstance(token, int) for token in self.token_ids
        ):
            raise ValueError("prompt token ids must be a nonempty integer sequence")


@dataclass(frozen=True, slots=True)
class SharedOPDRolloutAttempt:
    """one deterministic bounded attempt for a logical OPD sample."""

    prompt: SharedOPDPrompt
    rollout_ordinal: int
    no_signal_attempt: int
    action_token_ids: tuple[int, ...]
    terminated: bool
    request_id: str


@dataclass(frozen=True, slots=True)
class SharedOPDRolloutSample:
    """ordered attempts for one prompt and sample ordinal."""

    attempts: tuple[SharedOPDRolloutAttempt, ...]


@dataclass(frozen=True, slots=True)
class SharedOPDRolloutBatch:
    """rollouts bound to one immutable adapter version."""

    run_id: str
    step: int
    handle: AdapterHandle
    samples: tuple[SharedOPDRolloutSample, ...]


@dataclass(frozen=True, slots=True)
class SharedOPDTeacherSample:
    """one selected rollout attempt and its aligned teacher metadata."""

    sample_index: int
    attempt_index: int
    request_id: str
    action_token_ids: tuple[int, ...]
    group_ids: tuple[int, ...]
    teacher_logsums: tuple[float, ...]
    signal_mask: tuple[bool, ...]
    coverage: float


@dataclass(frozen=True, slots=True)
class SharedOPDTeacherBatch:
    """ordered teacher alignments returned by one run-local bridge."""

    samples: tuple[SharedOPDTeacherSample, ...]


@dataclass(frozen=True, slots=True)
class SharedOPDTrainingBatch:
    """padded tensors consumed by the differentiable log-prob hook."""

    sequences: Any
    attention_mask: Any
    response_mask: Any
    group_ids: Any
    teacher_logsums: Any
    prompt_lengths: tuple[int, ...]
    action_lengths: tuple[int, ...]
    coverage: Any


@dataclass(frozen=True, slots=True)
class SharedOPDUpdateResult:
    """observable result of one exact OPD update and adapter publication."""

    training_step: TrainingStepResult
    objective: Any
    aligned_samples: int
    teacher_coverage: float


SamplingParamsFactory = Callable[[int], Any]
PolicyLogProbFunction = Callable[[Any, TrainingRunState, SharedOPDTrainingBatch], Any]
TeacherRequest = Callable[[str, dict[str, Any]], Mapping[str, Any]]


def _validate_teacher_url(url: str) -> str:
    endpoint = str(url).strip()
    parsed = urlsplit(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("shared OPD teacher bridge must use an HTTP loopback endpoint")
    if not parsed.path.startswith("/teacher/"):
        raise ValueError("shared OPD teacher bridge must include its run-local teacher path")
    return endpoint


def _decode_attempt(
    prompt: SharedOPDPrompt,
    envelope: RolloutEnvelope,
    *,
    rollout_ordinal: int,
    no_signal_attempt: int,
    max_response_length: int,
) -> SharedOPDRolloutAttempt:
    outputs = getattr(envelope.output, "outputs", None)
    if not isinstance(outputs, Sequence) or len(outputs) != 1:
        raise ValueError("shared OPD rollout must contain exactly one completion")
    generated = outputs[0]
    action_token_ids = tuple(int(token) for token in getattr(generated, "token_ids", ()))
    if len(action_token_ids) > max_response_length:
        raise ValueError("shared OPD rollout exceeded max_response_length")
    return SharedOPDRolloutAttempt(
        prompt=prompt,
        rollout_ordinal=rollout_ordinal,
        no_signal_attempt=no_signal_attempt,
        action_token_ids=action_token_ids,
        terminated=getattr(generated, "finish_reason", None) != "length",
        request_id=envelope.request_id,
    )


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"OpenRLHF teacher bridge returned invalid {field}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"OpenRLHF teacher bridge returned non-finite {field}")
    return number


def _teacher_sample(
    response: Mapping[str, Any],
    *,
    sample_index: int,
    attempt_index: int,
    attempt: Mapping[str, Any],
) -> SharedOPDTeacherSample:
    if not isinstance(response, Mapping):
        raise TypeError("OpenRLHF teacher bridge returned a non-object response")
    action_token_ids = attempt.get("action_token_ids")
    prompt_length = attempt.get("prompt_length")
    request_id = attempt.get("request_id")
    if not isinstance(action_token_ids, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in action_token_ids
    ):
        raise TypeError("shared OPD scoring attempt has invalid action_token_ids")
    if isinstance(prompt_length, bool) or not isinstance(prompt_length, int) or prompt_length < 1:
        raise ValueError("shared OPD scoring attempt has invalid prompt_length")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("shared OPD scoring attempt has invalid request_id")
    action_length = prompt_length + len(action_token_ids) - 1
    raw_group_ids = response.get("teacher_group_ids")
    raw_teacher_logsums = response.get("teacher_logsums")
    raw_signal_mask = response.get("teacher_signal_mask")
    if not isinstance(raw_group_ids, list) or len(raw_group_ids) != action_length:
        raise ValueError("OpenRLHF teacher bridge returned invalid teacher_group_ids")
    if not isinstance(raw_teacher_logsums, list) or len(raw_teacher_logsums) != action_length:
        raise ValueError("OpenRLHF teacher bridge returned invalid teacher_logsums")
    if not isinstance(raw_signal_mask, list) or len(raw_signal_mask) != action_length:
        raise ValueError("OpenRLHF teacher bridge returned invalid teacher_signal_mask")
    if any(
        isinstance(group_id, bool) or not isinstance(group_id, int) for group_id in raw_group_ids
    ):
        raise TypeError("OpenRLHF teacher bridge returned non-integer group ids")
    teacher_logsums = tuple(
        _finite_number(value, "teacher_logsums") for value in raw_teacher_logsums
    )
    if any(not isinstance(selected, bool) for selected in raw_signal_mask):
        raise TypeError("OpenRLHF teacher bridge returned non-boolean signal mask")
    signal_mask = tuple(raw_signal_mask)
    signal_count = response.get("signal_count")
    if (
        isinstance(signal_count, bool)
        or not isinstance(signal_count, int)
        or signal_count != sum(signal_mask)
    ):
        raise ValueError("OpenRLHF teacher bridge returned an invalid signal_count")
    coverage = _finite_number(response.get("coverage", 0.0), "coverage")
    return SharedOPDTeacherSample(
        sample_index=sample_index,
        attempt_index=attempt_index,
        request_id=request_id,
        action_token_ids=tuple(action_token_ids),
        group_ids=tuple(raw_group_ids),
        teacher_logsums=teacher_logsums,
        signal_mask=signal_mask,
        coverage=coverage,
    )


def _collate_training_batch(
    rollout: SharedOPDRolloutBatch,
    teacher: SharedOPDTeacherBatch,
    *,
    pad_token_id: int,
) -> SharedOPDTrainingBatch:
    import torch

    if len(rollout.samples) != len(teacher.samples) or not teacher.samples:
        raise ValueError("shared OPD rollout and teacher sample counts do not match")
    selected: list[tuple[SharedOPDRolloutAttempt, SharedOPDTeacherSample]] = []
    for sample_index, (rollout_sample, teacher_sample) in enumerate(
        zip(rollout.samples, teacher.samples, strict=True)
    ):
        if teacher_sample.sample_index != sample_index:
            raise ValueError("shared OPD teacher sample order does not match the rollout")
        if not 0 <= teacher_sample.attempt_index < len(rollout_sample.attempts):
            raise ValueError("shared OPD teacher selected an unknown rollout attempt")
        attempt = rollout_sample.attempts[teacher_sample.attempt_index]
        if (
            teacher_sample.request_id != attempt.request_id
            or teacher_sample.action_token_ids != attempt.action_token_ids
        ):
            raise ValueError("shared OPD teacher result does not match its rollout attempt")
        selected.append((attempt, teacher_sample))

    prompt_lengths = tuple(len(attempt.prompt.token_ids) for attempt, _sample in selected)
    action_lengths = tuple(
        prompt_length + len(attempt.action_token_ids) - 1
        for prompt_length, (attempt, _sample) in zip(prompt_lengths, selected, strict=True)
    )
    max_sequence_length = max(
        len(attempt.prompt.token_ids) + len(attempt.action_token_ids)
        for attempt, _sample in selected
    )
    max_action_length = max_sequence_length - 1
    batch_size = len(selected)
    sequences = torch.full(
        (batch_size, max_sequence_length),
        int(pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(sequences)
    response_mask = torch.zeros((batch_size, max_action_length), dtype=torch.bool)
    group_ids = torch.full((batch_size, max_action_length), -1, dtype=torch.long)
    teacher_logsums = torch.zeros((batch_size, max_action_length), dtype=torch.float64)
    coverage = torch.zeros(batch_size, dtype=torch.float64)
    for row, (attempt, teacher_sample) in enumerate(selected):
        tokens = (*attempt.prompt.token_ids, *attempt.action_token_ids)
        sequence_length = len(tokens)
        action_length = sequence_length - 1
        if (
            len(teacher_sample.group_ids) != action_length
            or len(teacher_sample.teacher_logsums) != action_length
            or len(teacher_sample.signal_mask) != action_length
        ):
            raise ValueError("shared OPD teacher metadata length does not match its rollout")
        sequences[row, :sequence_length] = torch.tensor(tokens, dtype=torch.long)
        attention_mask[row, :sequence_length] = 1
        response_mask[row, :action_length] = torch.tensor(
            teacher_sample.signal_mask,
            dtype=torch.bool,
        )
        group_ids[row, :action_length] = torch.tensor(
            teacher_sample.group_ids,
            dtype=torch.long,
        )
        teacher_logsums[row, :action_length] = torch.tensor(
            teacher_sample.teacher_logsums,
            dtype=torch.float64,
        )
        coverage[row] = teacher_sample.coverage
    return SharedOPDTrainingBatch(
        sequences=sequences,
        attention_mask=attention_mask,
        response_mask=response_mask,
        group_ids=group_ids,
        teacher_logsums=teacher_logsums,
        prompt_lengths=prompt_lengths,
        action_lengths=action_lengths,
        coverage=coverage,
    )


class SharedOPDRunAdapter:
    """compose one OPD run with the shared rollout, scoring, training, and scheduler layers."""

    def __init__(
        self,
        run_id: str,
        *,
        config: SharedOPDConfig,
        training_actor: SharedMultiLoRATrainingActor,
        rollout_engine: SharedMultiLoRARolloutEngine,
        teacher_url: str,
        sampling_params_factory: SamplingParamsFactory,
        policy_log_probs: PolicyLogProbFunction,
        teacher_request: TeacherRequest = post_teacher_request_with_retry,
    ) -> None:
        normalized_run_id = str(run_id).strip()
        if not normalized_run_id:
            raise ValueError("run_id must not be empty")
        if not callable(sampling_params_factory):
            raise TypeError("sampling_params_factory must be callable")
        if not callable(policy_log_probs):
            raise TypeError("policy_log_probs must be callable")
        if not callable(teacher_request):
            raise TypeError("teacher_request must be callable")

        state = training_actor.run_state(normalized_run_id)
        prompts = state.dataloader
        if not isinstance(prompts, Sequence) or not prompts:
            raise ValueError("shared OPD requires a nonempty indexable prompt schedule")
        if not all(isinstance(prompt, SharedOPDPrompt) for prompt in prompts):
            raise TypeError("shared OPD prompt schedule contains an invalid prompt")
        if state.handle.run_id != normalized_run_id:
            raise ValueError("training adapter handle does not match the OPD run")

        self.run_id = normalized_run_id
        self.config = config
        self._training_actor = training_actor
        self._rollout_engine = rollout_engine
        self._teacher_bridge = bind_scoring_bridge(
            _validate_teacher_url(teacher_url),
            teacher_request,
        )
        self._sampling_params_factory = sampling_params_factory
        self._policy_log_probs = policy_log_probs
        self._initial_global_step = state.global_step
        self._last_update: SharedOPDUpdateResult | None = None

    @property
    def last_update(self) -> SharedOPDUpdateResult | None:
        """return the most recent completed objective and publication result."""

        return self._last_update

    def scheduler_hooks(self) -> SchedulerRunHooks:
        """return the PR4 callbacks for this OPD run."""

        return SchedulerRunHooks(
            rollout=self.rollout,
            scoring_payload=self.scoring_payload,
            update_and_publish=self.update_and_publish,
        )

    def add_to_controller(
        self,
        controller: SharedEngineRunController,
        *,
        total_steps: int,
        weight: float = 1.0,
    ) -> None:
        """admit this run to the shared scheduler with its isolated teacher bridge."""

        controller.add_run(
            self.run_id,
            hooks=self.scheduler_hooks(),
            scoring_kind=ScoringKind.TEACHER,
            scoring_bridge=self.score,
            total_steps=total_steps,
            weight=weight,
        )

    async def rollout(self, run_id: str, step: int) -> SharedOPDRolloutBatch:
        """generate one ordered OPD batch through the run's current adapter handle."""

        self._require_identity(run_id)
        state = self._training_actor.run_state(self.run_id)
        expected_global_step = self._initial_global_step + step
        if state.global_step != expected_global_step:
            raise ValueError(
                f"shared OPD scheduler step {step} expected training step "
                f"{expected_global_step}, got {state.global_step}"
            )
        prompts = state.dataloader
        prompt_cursor = state.prompt_cursor
        selected = tuple(
            prompts[(prompt_cursor + offset) % len(prompts)]
            for offset in range(self.config.prompts_per_step)
        )
        handle = state.handle
        ordinal_counts: Counter[int] = Counter()
        request_specs: list[tuple[SharedOPDPrompt, int, int, Any, str]] = []
        for prompt_offset, prompt in enumerate(selected):
            for sample_offset in range(self.config.group_size):
                rollout_ordinal = ordinal_counts[prompt.example_index]
                ordinal_counts[prompt.example_index] += 1
                for no_signal_attempt in range(self.config.no_signal_attempts):
                    seed = deterministic_rollout_seed(
                        self.config.seed,
                        state.global_step,
                        prompt.example_index,
                        rollout_ordinal,
                        no_signal_attempt_ordinal=no_signal_attempt,
                    )
                    sampling_params = self._sampling_params_factory(seed)
                    if getattr(sampling_params, "seed", None) != seed:
                        raise ValueError(
                            "sampling_params_factory must preserve the deterministic seed"
                        )
                    if self.config.stop_sequences:
                        sampling_params.stop = list(self.config.stop_sequences)
                        sampling_params.include_stop_str_in_output = True
                    if self.config.eos_token_ids:
                        sampling_params.stop_token_ids = list(self.config.eos_token_ids)
                    request_id = (
                        f"{self.run_id}-step-{step}-prompt-{prompt_offset}-sample-"
                        f"{sample_offset}-attempt-{no_signal_attempt}"
                    )
                    request_specs.append(
                        (
                            prompt,
                            rollout_ordinal,
                            no_signal_attempt,
                            sampling_params,
                            request_id,
                        )
                    )
        envelopes = await asyncio.gather(
            *(
                self._rollout_engine.generate(
                    handle,
                    {"prompt_token_ids": list(prompt.token_ids)},
                    sampling_params,
                    request_id=request_id,
                )
                for prompt, _ordinal, _attempt, sampling_params, request_id in request_specs
            )
        )
        if any(envelope.handle != handle for envelope in envelopes):
            raise ValueError("shared OPD rollout returned a mismatched adapter handle")
        decoded = tuple(
            _decode_attempt(
                prompt,
                envelope,
                rollout_ordinal=rollout_ordinal,
                no_signal_attempt=no_signal_attempt,
                max_response_length=self.config.max_response_length,
            )
            for (
                prompt,
                rollout_ordinal,
                no_signal_attempt,
                _sampling_params,
                _request_id,
            ), envelope in zip(request_specs, envelopes, strict=True)
        )
        attempts_per_sample = self.config.no_signal_attempts
        samples = tuple(
            SharedOPDRolloutSample(decoded[index : index + attempts_per_sample])
            for index in range(0, len(decoded), attempts_per_sample)
        )
        self._training_actor.advance_prompt_cursor(
            self.run_id,
            self.config.prompts_per_step,
        )
        return SharedOPDRolloutBatch(
            run_id=self.run_id,
            step=step,
            handle=handle,
            samples=samples,
        )

    def scoring_payload(
        self,
        run_id: str,
        step: int,
        rollout: SharedOPDRolloutBatch,
    ) -> dict[str, Any]:
        """build bounded alignment requests for the existing teacher bridge."""

        self._require_batch(run_id, step, rollout)
        return {
            "samples": [
                {
                    "attempts": [
                        {
                            "label": {
                                "global_step": self._initial_global_step + step,
                                "example_index": attempt.prompt.example_index,
                                "rollout_ordinal": attempt.rollout_ordinal,
                                "no_signal_attempt": attempt.no_signal_attempt,
                            },
                            "prompt_length": len(attempt.prompt.token_ids),
                            "sequence_ids": [
                                *attempt.prompt.token_ids,
                                *attempt.action_token_ids,
                            ],
                            "terminated": attempt.terminated,
                            "action_token_ids": list(attempt.action_token_ids),
                            "request_id": attempt.request_id,
                        }
                        for attempt in sample.attempts
                    ]
                }
                for sample in rollout.samples
            ]
        }

    def score(self, payload: dict[str, Any]) -> SharedOPDTeacherBatch:
        """align one batch through only this run's fail-closed teacher bridge."""

        samples = payload.get("samples") if isinstance(payload, Mapping) else None
        expected = self.config.prompts_per_step * self.config.group_size
        if not isinstance(samples, list) or len(samples) != expected:
            raise ValueError("shared OPD scoring payload has an invalid sample count")
        results: list[SharedOPDTeacherSample] = []
        for sample_index, sample in enumerate(samples):
            attempts = sample.get("attempts") if isinstance(sample, Mapping) else None
            if not isinstance(attempts, list) or len(attempts) != self.config.no_signal_attempts:
                raise ValueError("shared OPD scoring payload has an invalid attempt count")
            selected: SharedOPDTeacherSample | None = None
            for attempt_index, attempt in enumerate(attempts):
                if not isinstance(attempt, Mapping):
                    raise TypeError("shared OPD scoring attempt must be an object")
                bridge_payload = {
                    key: attempt[key]
                    for key in ("label", "prompt_length", "sequence_ids", "terminated")
                }
                response = self._teacher_bridge(bridge_payload)
                selected = _teacher_sample(
                    response,
                    sample_index=sample_index,
                    attempt_index=attempt_index,
                    attempt=attempt,
                )
                if any(selected.signal_mask):
                    break
            if selected is None:
                raise RuntimeError("shared OPD teacher scoring selected no rollout attempt")
            results.append(selected)
        return SharedOPDTeacherBatch(tuple(results))

    async def update_and_publish(
        self,
        run_id: str,
        step: int,
        rollout: SharedOPDRolloutBatch,
        scoring_result: ScoringResult,
    ) -> SharedOPDUpdateResult:
        """apply exact OPD math to one adapter and publish its next immutable version."""

        self._require_batch(run_id, step, rollout)
        if (
            scoring_result.identity.run_id != self.run_id
            or scoring_result.identity.step != step
            or scoring_result.kind is not ScoringKind.TEACHER
        ):
            raise ValueError("shared OPD scoring identity does not match the update")
        teacher = scoring_result.value
        if not isinstance(teacher, SharedOPDTeacherBatch):
            raise TypeError("shared OPD scoring result has an invalid value")
        state = self._training_actor.run_state(self.run_id)
        if state.handle != rollout.handle:
            raise ValueError("shared OPD update received a stale adapter rollout")
        training_batch = _collate_training_batch(
            rollout,
            teacher,
            pad_token_id=self.config.pad_token_id,
        )
        captured: list[Any] = []
        aligned_samples = int(
            (training_batch.response_mask & training_batch.group_ids.ge(0)).any(dim=-1).sum().item()
        )

        def loss_function(model: Any, active_state: TrainingRunState) -> Any:
            student_logprobs = self._policy_log_probs(model, active_state, training_batch)
            if student_logprobs.shape != training_batch.response_mask.shape:
                raise ValueError(
                    "shared OPD policy log probabilities do not match the action tensor shape"
                )
            if aligned_samples:
                objective = flash_groupwise_reverse_kl(
                    student_logprobs,
                    training_batch.teacher_logsums.to(
                        device=student_logprobs.device,
                        dtype=student_logprobs.dtype,
                    ),
                    training_batch.group_ids.to(student_logprobs.device),
                    training_batch.response_mask.to(student_logprobs.device),
                    self.config.kl_penalty_coef,
                )
            else:
                objective = student_logprobs.sum() * 0.0
            captured.append(objective)
            return objective

        training_step = await self._training_actor.step(self.run_id, loss_function)
        if len(captured) != 1:
            raise RuntimeError("shared OPD loss hook did not execute exactly once")
        result = SharedOPDUpdateResult(
            training_step=training_step,
            objective=captured[0].detach(),
            aligned_samples=aligned_samples,
            teacher_coverage=float(training_batch.coverage.mean().item()),
        )
        self._last_update = result
        return result

    def _require_identity(self, run_id: str) -> None:
        if run_id != self.run_id:
            raise ValueError("shared OPD callback received another run's identity")

    def _require_batch(
        self,
        run_id: str,
        step: int,
        rollout: SharedOPDRolloutBatch,
    ) -> None:
        self._require_identity(run_id)
        if not isinstance(rollout, SharedOPDRolloutBatch):
            raise TypeError("shared OPD callback received an invalid rollout batch")
        if rollout.run_id != self.run_id or rollout.step != step:
            raise ValueError("shared OPD rollout identity does not match the callback")
