"""OpenRLHF-backed single-turn text on-policy distillation.

The flash parent owns the remote teacher and exposes only an authenticated localhost alignment
bridge. The isolated OpenRLHF child performs deterministic student rollouts, carries aligned
teacher group tensors on each experience, and replaces PPO's actor loss with flash's exact
realized-token groupwise reverse-KL objective through ``sitecustomize``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import re
import secrets
import shutil
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from flash.catalog import MODELS
from flash.engine.recipe import RECIPE
from flash.engine.steps import (
    final_save_due,
    on_policy_steps,
    resolve_update_horizon,
    validate_save_steps,
)
from flash.engine.structured_outputs import resolve_opd_structured_plan
from flash.engine.vram import opd_rollout_concurrency
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.adapter import _download_adapter
from flash.engine.worker.grpo_openrlhf import (
    _OPENRLHF_ENTRYPOINT,
    _is_sm120,
    _resolve_cached_model_snapshot,
    build_openrlhf_child_env,
)
from flash.engine.worker.grpo_openrlhf import (
    _sitecustomize_source as _grpo_sitecustomize_source,
)
from flash.engine.worker.heartbeat import liveness_heartbeat
from flash.engine.worker.hf import _deployable_adapter_on_hf
from flash.engine.worker.opd import _resolve_opd_knobs, _thinking_prefill_text
from flash.engine.worker.opd_gkd import (
    _generation_eos_ids,
    _rollout_terminated,
    _teacher_prompt_text,
    _trim_trailing_stop,
    student_tokens_with_offsets,
)
from flash.engine.worker.openrlhf_common import (
    export_openrlhf_adapter,
    resolve_openrlhf_python,
    run_openrlhf_training,
)
from flash.engine.worker.perf import (
    gpu_diagnostics,
    optimal_attn_impl,
    setup_perf_backends,
    wait_for_gpu,
)
from flash.engine.worker.rng import backend_seed
from flash.engine.worker.teacher import TeacherError, TeacherToken
from flash.engine.worker.tokenizer_align import groupwise_alignment, groupwise_coverage
from flash.opd_retry_contract import OPD_RESUME_STATE_VERSION, validate_opd_resume_state_metadata

_OPENRLHF_OPD_MAX_BODY_BYTES = 4 * 1024 * 1024
_OPENRLHF_TARGET_MODULES = "all-linear"
_OPENRLHF_PPO_EPOCHS = 1
_OPENRLHF_NO_SIGNAL_ATTEMPTS = 3
_OPENRLHF_BRIDGE_TRANSPORT_ATTEMPTS = 3
_OPENRLHF_PERMANENT_TEACHER_EXIT = 86
_OPENRLHF_TRANSIENT_TEACHER_EXIT = 87
_SITE_CUSTOMIZE_NAME = "sitecustomize.py"


@dataclass(frozen=True)
class OpenRLHFOPDConfig:
    model_path: str
    dataset_path: str
    teacher_url: str
    output_dir: str
    checkpoint_dir: str
    model_id: str
    model_revision: str
    max_length: int
    max_completion: int
    prompts_per_step: int
    group_size: int
    scheduled_prompt_count: int
    learning_rate: float
    temperature: float
    top_p: float
    seed: int
    lora_rank: int
    lora_alpha: int
    lora_target_modules: tuple[str, ...]
    kl_penalty_coef: float
    save_every: int
    save_at_steps: tuple[int, ...]
    final_step: int
    gpu_count: int
    actor_attn_implementation: str | None
    resume: bool = False


@dataclass(frozen=True)
class _PromptRecord:
    messages: list[dict]
    prompt_ids: tuple[int, ...]
    rendered: str


class OpenRLHFTeacherBridgeError(RuntimeError):
    """Typed child-side bridge failure with the teacher retry classification."""

    def __init__(self, message: str, *, classification: str) -> None:
        super().__init__(message)
        self.classification = classification


def deterministic_rollout_seed(
    flash_seed: int,
    global_step: int,
    example_index: int,
    rollout_ordinal: int,
    *,
    no_signal_attempt_ordinal: int = 0,
) -> int:
    """Derive one stable nonnegative 63-bit seed from the complete rollout identity."""
    attempt = int(no_signal_attempt_ordinal)
    if attempt < 0:
        raise ValueError("flash OPD no-signal attempt ordinal must be nonnegative")
    payload = f"{int(flash_seed)}:{int(global_step)}:{int(example_index)}:{int(rollout_ordinal)}:0"
    if attempt:
        payload += f":retry:{attempt}"
    digest = hashlib.blake2b(payload.encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def flash_groupwise_reverse_kl(
    student_logprobs,
    teacher_logsums,
    group_ids,
    response_mask,
    kl_penalty_coef: float,
):
    """Return flash's sequence-mean realized-token groupwise reverse-KL loss."""
    import torch

    if float(kl_penalty_coef) <= 0:
        raise ValueError("flash OPD kl_penalty_coef must be positive")
    if student_logprobs.shape != teacher_logsums.shape or student_logprobs.shape != group_ids.shape:
        raise ValueError("flash OPD teacher metadata must match student response logprob shape")
    if response_mask.shape != student_logprobs.shape:
        raise ValueError("flash OPD response mask must match student response logprob shape")

    response_mask = response_mask.bool()
    selected = response_mask & group_ids.ge(0)
    signal_rows = selected.any(dim=-1)
    if not bool(signal_rows.any().item()):
        raise RuntimeError("flash OPD actor batch contains no aligned teacher signal")

    row_losses = []
    for row in range(student_logprobs.shape[0]):
        row_selected = selected[row]
        if not bool(row_selected.any().item()):
            continue
        token_losses = torch.zeros_like(student_logprobs[row])
        for group_id in torch.unique(group_ids[row][row_selected], sorted=True):
            group_mask = row_selected & group_ids[row].eq(group_id)
            group_length = group_mask.sum().to(dtype=student_logprobs.dtype)
            student_logsum = student_logprobs[row][group_mask].detach().sum()
            teacher_logsum = teacher_logsums[row][group_mask][0]
            coefficient = float(kl_penalty_coef) * (student_logsum - teacher_logsum) / group_length
            token_losses[group_mask] = coefficient * student_logprobs[row][group_mask]
        row_losses.append(token_losses[row_selected].mean())
    return torch.stack(row_losses).mean()


def _encode_action_metadata(
    prompt_length: int,
    response_length: int,
    groups,
) -> tuple[list[int], list[float], list[bool]]:
    if prompt_length <= 0:
        raise ValueError("flash OPD prompts must contain at least one token")
    action_length = prompt_length + response_length - 1
    group_ids = [-1] * action_length
    teacher_logsums = [0.0] * action_length
    signal_mask = [False] * action_length
    response_offset = prompt_length - 1
    for local_group_id, (student_indices, teacher_logsum) in enumerate(groups):
        for student_index in student_indices:
            index = response_offset + int(student_index)
            if index < response_offset or index >= action_length:
                raise ValueError("flash OPD alignment group index is outside the response")
            if signal_mask[index]:
                raise ValueError("flash OPD response token belongs to multiple alignment groups")
            group_ids[index] = local_group_id
            teacher_logsums[index] = float(teacher_logsum)
            signal_mask[index] = True
    return group_ids, teacher_logsums, signal_mask


def _prompt_pool_fingerprint(prompts: list[_PromptRecord]) -> str:
    digest = hashlib.sha256()
    for prompt in prompts:
        payload = json.dumps(
            [prompt.messages, list(prompt.prompt_ids)],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _group_granularity(student_tokens, groups) -> float:
    """Mean student-tokens-per-alignment-group; a real alignment-health signal where coverage is not.

    Coverage stays ~1.0 even for a degenerate collapsed alignment, so it cannot flag that failure
    mode. Byte-identical to TRL opd.py: n_align (student tokens carrying content) divided by the
    group count, and 0.0 when the sample produced no alignment groups.
    """
    if not groups:
        return 0.0
    n_align = sum(1 for st in student_tokens if st.end > st.start)
    return n_align / len(groups)


class _TeacherHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


class TeacherAlignmentBridge:
    """Authenticated localhost bridge that owns teacher scoring and token alignment."""

    def __init__(
        self,
        *,
        prompts: list[_PromptRecord],
        tokenizer,
        teacher,
        thinking_prefill: str,
        eos_token_ids: frozenset[int],
        stop_sequences: tuple[str, ...],
        mutation_callback=None,
        metrics_callback=None,
        checkpoint_callback=None,
        initial_state: dict[str, Any] | None = None,
        token: str | None = None,
    ) -> None:
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.teacher = teacher
        self.thinking_prefill = thinking_prefill
        self.eos_token_ids = eos_token_ids
        self.stop_sequences = stop_sequences
        self.mutation_callback = mutation_callback or (lambda: None)
        self.metrics_callback = metrics_callback or (lambda _step, _loss, _coverage: None)
        self.checkpoint_callback = checkpoint_callback or (lambda _step: None)
        self.token = token or secrets.token_urlsafe(32)
        self._path = f"/teacher/{self.token}"
        self._mutation_lock = threading.Lock()
        self._mutation_notified = False
        self._stats_lock = threading.Lock()
        self._failure: tuple[str, str] | None = None
        # per-step teacher echo memo: identical (teacher_prompt, completion_text) echo requests are
        # deterministic under the frozen temperature=0 teacher, so collapse them to one remote call
        # (parity with the TRL worker's unique_prompts dedup). scoped to one optimizer step because
        # rollouts are collected step-synchronously before the single update, which bounds the memo to
        # one step's unique completions and clears it on step advance.
        self._echo_lock = threading.Lock()
        self._echo_step: int | None = None
        self._echo_cache: dict[tuple[str, str], list[TeacherToken]] = {}
        state = initial_state or {}
        skip_counts = state.get("skip_counts", {})
        self._stats = {
            "generated_tokens": int(state.get("generated_tokens", 0)),
            "teacher_input_tokens": int(state.get("teacher_input_tokens", 0)),
            "truncated_rollouts": int(state.get("truncated_rollouts", 0)),
            "aligned_sequences": int(state.get("granularity_n", 0)),
            "empty_alignments": int(skip_counts.get("empty_alignment", 0)),
            "coverage_sum": float(state.get("granularity_sum", 0.0)),
            "teacher_ok": int(state.get("teacher_ok", 0)),
            "teacher_transient": int(state.get("teacher_transient", 0)),
            "teacher_error": int(state.get("teacher_error", 0)),
            "samples_seen": int(state.get("samples_seen", 0)),
            "no_signal_resamples": int(state.get("no_signal_resamples", 0)),
            "teacher_echo_deduped": int(state.get("teacher_echo_deduped", 0)),
            "group_granularity_sum": float(state.get("group_granularity_sum", 0.0)),
        }
        bridge = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, _format, *_args):
                return

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                if not secrets.compare_digest(self.path, bridge._path):
                    self._send(
                        401, {"error": {"classification": "permanent", "message": "unauthorized"}}
                    )
                    return
                try:
                    raw_length = self.headers.get("Content-Length")
                    if raw_length is None:
                        raise ValueError("missing content length")
                    length = int(raw_length)
                    if length <= 0 or length > _OPENRLHF_OPD_MAX_BODY_BYTES:
                        raise ValueError("invalid teacher request size")
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    result = bridge.score_payload(payload)
                except Exception as exc:
                    if isinstance(exc, _w.RetriableInfraError) or (
                        isinstance(exc, TeacherError) and not exc.permanent
                    ):
                        classification = "transient"
                    else:
                        classification = "permanent"
                    bridge._record_failure(classification, str(exc))
                    self._send(
                        503 if classification == "transient" else 422,
                        {"error": {"classification": classification, "message": str(exc)}},
                    )
                    return
                self._send(200, result)

        self._server = _TeacherHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="openrlhf-opd-teacher-bridge",
            daemon=True,
        )
        self._thread.start()
        self.url = f"http://127.0.0.1:{int(self._server.server_address[1])}{self._path}"

    def _record_failure(self, classification: str, message: str) -> None:
        with self._stats_lock:
            key = "teacher_transient" if classification == "transient" else "teacher_error"
            self._stats[key] += 1
            if self._failure is None or classification == "permanent":
                self._failure = (classification, message)

    @property
    def teacher_failure(self) -> tuple[str, str] | None:
        with self._stats_lock:
            return self._failure

    def snapshot(self) -> dict[str, int | float]:
        with self._stats_lock:
            return dict(self._stats)

    def _scored_teacher_tokens(
        self, global_step: int, teacher_prompt: str, completion_text: str
    ) -> list[TeacherToken]:
        """Echo-score ``completion_text`` against ``teacher_prompt``, deduplicating within the step.

        The remote teacher call runs outside the memo lock so distinct echo requests still overlap
        (the ThreadingHTTPServer scores concurrent rollout posts in parallel); only the dict lookup
        and store are serialized. Concurrent identical requests may both miss and both call, which is
        benign because the frozen temperature=0 teacher returns identical tokens; this matches TRL,
        whose per-batch dedup likewise does not collapse identical requests across concurrent batches.
        """
        key = (teacher_prompt, completion_text)
        with self._echo_lock:
            if global_step != self._echo_step:
                self._echo_step = global_step
                self._echo_cache = {}
            cached = self._echo_cache.get(key)
        if cached is not None:
            with self._stats_lock:
                self._stats["teacher_echo_deduped"] += 1
            return cached
        tokens = self.teacher.score(teacher_prompt, completion_text)
        with self._echo_lock:
            # only retain the result if the step has not advanced while we were scoring, so the memo
            # never carries a prior step's completions past the single-update boundary.
            if global_step == self._echo_step:
                self._echo_cache[key] = tokens
        return tokens

    def score_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("teacher request must be an object")
        if payload == {"mutation": True}:
            with self._mutation_lock:
                if not self._mutation_notified:
                    self.mutation_callback()
                    self._mutation_notified = True
            return {"ok": True}
        if set(payload) == {"metrics"}:
            metrics = payload["metrics"]
            if not isinstance(metrics, dict) or set(metrics) != {"step", "loss", "coverage"}:
                raise ValueError("OpenRLHF OPD step metrics are invalid")
            step = metrics["step"]
            loss = metrics["loss"]
            coverage = metrics["coverage"]
            if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
                raise ValueError("OpenRLHF OPD metrics step is invalid")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in (loss, coverage)
            ):
                raise ValueError("OpenRLHF OPD metric values are invalid")
            self.metrics_callback(step, float(loss), float(coverage))
            return {"ok": True}
        if set(payload) == {"checkpoint"}:
            step = payload["checkpoint"]
            if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
                raise ValueError("OpenRLHF OPD checkpoint step is invalid")
            self.checkpoint_callback(step)
            return {"ok": True}
        identity_raw = payload["label"]
        identity = json.loads(identity_raw) if isinstance(identity_raw, str) else identity_raw
        if not isinstance(identity, dict):
            raise TypeError("flash OPD rollout identity must be an object")
        if set(identity) != {
            "global_step",
            "example_index",
            "rollout_ordinal",
            "no_signal_attempt",
        }:
            raise ValueError("flash OPD rollout identity schema is invalid")
        for name in ("global_step", "rollout_ordinal", "no_signal_attempt"):
            value = identity[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"flash OPD rollout identity {name} is invalid")
        index = identity["example_index"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(self.prompts)
        ):
            raise ValueError("flash OPD bridge received an unknown example index")
        retry = identity["no_signal_attempt"]

        sequence_ids = payload["sequence_ids"]
        prompt_length = payload["prompt_length"]
        terminated = payload["terminated"]
        if not isinstance(terminated, bool):
            raise TypeError("flash OPD rollout termination state must be a boolean")
        if not isinstance(sequence_ids, list) or any(
            isinstance(token, bool) or not isinstance(token, int) for token in sequence_ids
        ):
            raise TypeError("flash OPD sequence ids must be integers")
        if isinstance(prompt_length, bool) or not isinstance(prompt_length, int):
            raise TypeError("flash OPD prompt length must be an integer")
        prompt = self.prompts[index]
        if prompt_length != len(prompt.prompt_ids) or sequence_ids[:prompt_length] != list(
            prompt.prompt_ids
        ):
            raise ValueError("OpenRLHF rollout prompt ids do not match the frozen flash prompt")

        response_ids = sequence_ids[prompt_length:]
        with self._stats_lock:
            self._stats["samples_seen"] += 1
            self._stats["generated_tokens"] += len(response_ids)
            if retry:
                self._stats["no_signal_resamples"] += 1
        empty = _encode_action_metadata(prompt_length, len(response_ids), [])
        if not response_ids:
            return self._response(*empty)
        stop_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
        if not terminated and not _rollout_terminated(
            response_ids,
            stop_text,
            self.eos_token_ids,
            self.stop_sequences,
        ):
            with self._stats_lock:
                self._stats["truncated_rollouts"] += 1
            return self._response(*empty)
        kept_ids, completion_text = _trim_trailing_stop(
            self.tokenizer,
            response_ids,
            stop_text,
            self.stop_sequences,
        )
        if not completion_text.strip() or "�" in completion_text:
            return self._response(*empty)

        teacher_prompt = _teacher_prompt_text(prompt.messages, self.thinking_prefill)
        teacher_tokens = self._scored_teacher_tokens(
            identity["global_step"], teacher_prompt, completion_text
        )
        student_ids, student_tokens = student_tokens_with_offsets(
            self.tokenizer,
            kept_ids,
            completion_text,
        )
        groups = [
            (indices, logsum)
            for indices, logsum in groupwise_alignment(student_tokens, teacher_tokens)
            if indices
        ]
        coverage = groupwise_coverage(groups, student_tokens)
        group_granularity = _group_granularity(student_tokens, groups)
        with self._stats_lock:
            self._stats["teacher_ok"] += 1
            self._stats["teacher_input_tokens"] += prompt_length + len(student_ids)
            self._stats["coverage_sum"] += coverage
            self._stats["group_granularity_sum"] += group_granularity
            if groups:
                self._stats["aligned_sequences"] += 1
            else:
                self._stats["empty_alignments"] += 1
        return self._response(
            *_encode_action_metadata(prompt_length, len(response_ids), groups), coverage=coverage
        )

    @staticmethod
    def _response(
        group_ids: list[int],
        teacher_logsums: list[float],
        signal_mask: list[bool],
        *,
        coverage: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "rewards": 0.0,
            "scores": 0.0,
            "teacher_group_ids": group_ids,
            "teacher_logsums": teacher_logsums,
            "teacher_signal_mask": signal_mask,
            "signal_count": sum(signal_mask),
            "coverage": float(coverage),
        }

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    def __enter__(self) -> TeacherAlignmentBridge:
        return self

    def __exit__(self, *_exc_info) -> None:
        self.shutdown()


def post_teacher_request(url: str, payload: dict[str, Any], *, timeout: float = 10.0) -> dict:
    """Post one child scoring request and preserve typed teacher failures."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        try:
            details = json.loads(exc.read().decode("utf-8"))["error"]
            classification = str(details["classification"])
            message = str(details["message"])
        except Exception as decode_exc:
            raise OpenRLHFTeacherBridgeError(
                f"flash OPD bridge returned unclassified HTTP {exc.code}",
                classification="permanent",
            ) from decode_exc
        if classification not in {"permanent", "transient"}:
            classification = "permanent"
        raise OpenRLHFTeacherBridgeError(message, classification=classification) from exc
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        raise OpenRLHFTeacherBridgeError(
            f"flash OPD bridge transport failed: {type(exc).__name__}",
            classification="transient",
        ) from exc
    try:
        payload = json.loads(body.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise OpenRLHFTeacherBridgeError(
            "flash OPD bridge returned malformed success JSON",
            classification="permanent",
        ) from exc
    if not isinstance(payload, dict):
        raise OpenRLHFTeacherBridgeError(
            "flash OPD bridge returned a non-object response",
            classification="permanent",
        )
    return payload


def build_openrlhf_opd_args(config: OpenRLHFOPDConfig) -> list[str]:
    """Map the resolved flash OPD job to the pinned OpenRLHF PPO entrypoint."""
    if config.group_size <= 0 or config.prompts_per_step <= 0:
        raise ValueError("OpenRLHF OPD requires positive rollout group and prompt batch sizes")
    if config.scheduled_prompt_count <= 0:
        raise ValueError("OpenRLHF OPD requires a nonempty scheduled prompt dataset")
    if config.max_completion <= 0 or config.max_length <= config.max_completion:
        raise ValueError("OpenRLHF max length must leave room for an OPD prompt")
    if config.lora_rank <= 0 or config.lora_alpha <= 0:
        raise ValueError("OpenRLHF OPD requires positive LoRA rank and alpha")
    if config.kl_penalty_coef <= 0:
        raise ValueError("OpenRLHF OPD kl_penalty_coef must be positive")
    if config.gpu_count <= 0:
        raise ValueError("OpenRLHF OPD requires at least one GPU")

    completion_batch = config.prompts_per_step * config.group_size
    if completion_batch < config.gpu_count or completion_batch % config.gpu_count:
        raise ValueError(
            "OpenRLHF OPD completion batch must be at least and divisible by the actor GPU count"
        )
    target_modules = list(config.lora_target_modules) or [_OPENRLHF_TARGET_MODULES]
    args = [
        "--actor.model_name_or_path",
        config.model_path,
        "--reward.remote_url",
        config.teacher_url,
        "--data.prompt_dataset",
        config.dataset_path,
        "--data.input_key",
        "input",
        "--data.label_key",
        "label",
        "--data.max_len",
        str(config.max_length),
        "--data.max_samples",
        str(config.scheduled_prompt_count),
        "--rollout.max_new_tokens",
        str(config.max_completion),
        "--rollout.batch_size",
        str(config.prompts_per_step),
        "--rollout.micro_batch_size",
        "1",
        "--rollout.n_samples_per_prompt",
        str(config.group_size),
        "--rollout.temperature",
        str(config.temperature),
        "--rollout.top_p",
        str(config.top_p),
        "--train.batch_size",
        str(completion_batch),
        "--train.micro_batch_size",
        "1",
        "--train.num_episodes",
        "1",
        "--train.max_epochs",
        str(_OPENRLHF_PPO_EPOCHS),
        "--train.seed",
        str(config.seed),
        "--algo.advantage.estimator",
        "reinforce",
        "--algo.advantage.gamma",
        "1.0",
        "--algo.kl.init_coef",
        "0.0",
        "--actor.adam.lr",
        str(config.learning_rate),
        "--actor.adam.betas",
        "0.9",
        "0.999",
        "--actor.adam.eps",
        "1e-8",
        "--actor.adam.weight_decay",
        "0.01",
        "--actor.lr_scheduler",
        "constant",
        "--actor.lr_warmup_ratio",
        "0.0",
        "--actor.min_lr_ratio",
        "1.0",
        "--actor.max_norm",
        "1.0",
        "--ds.lora.rank",
        str(config.lora_rank),
        "--ds.lora.alpha",
        str(config.lora_alpha),
        "--ds.lora.target_modules",
        *target_modules,
        "--ds.lora.dropout",
        "0.0",
        "--ds.zero_stage",
        "3",
        "--ds.ring_attn_size",
        "1",
        "--ds.param_dtype",
        "bf16",
        "--actor.gradient_checkpointing_enable",
        "--actor.num_nodes",
        "1",
        "--actor.num_gpus_per_node",
        str(config.gpu_count),
        "--vllm.num_engines",
        str(config.gpu_count),
        "--vllm.tensor_parallel_size",
        "1",
        "--vllm.gpu_memory_utilization",
        "0.5",
        "--vllm.sync_backend",
        "nccl",
        "--vllm.enable_prefix_caching",
        # enforce_eager is decided per gpu family in the rollout actor (see the vLLM runtime shim in
        # _opd_sitecustomize_extension): validated a100/h100 keep cuda graphs, unvalidated families on
        # vllm>=0.19 fall back to eager, blackwell uses decode-only graphs. leaving openrlhf's default
        # (False) here lets the shim be the sole authority instead of pinning eager for every gpu.
        "--vllm.enable_sleep",
        "--ds.enable_sleep",
        "--train.colocate_all",
        "--ckpt.output_dir",
        config.output_dir,
        "--ckpt.path",
        config.checkpoint_dir,
        "--ckpt.save_hf",
        "--ckpt.save_steps",
        str(-1 if config.save_at_steps else config.save_every),
        "--ckpt.max_num",
        "1",
        "--logger.logging_steps",
        "1",
        "--eval.steps",
        "-1",
        "--ckpt.best_metric_key",
        "none",
    ]
    if config.resume:
        args.append("--ckpt.load_enable")
    from flash.engine.worker.perf import grpo_use_reentrant

    if grpo_use_reentrant(config.model_id):
        args.append("--actor.gradient_checkpointing_reentrant")
    if config.actor_attn_implementation:
        args.extend(["--ds.attn_implementation", config.actor_attn_implementation])
    return args


def _opd_sitecustomize_extension() -> str:
    import inspect

    from flash.engine.worker import opd_vllm_runtime as _opd_vllm_runtime_mod

    # embed the pure vLLM-runtime decision module verbatim. the isolated child runs with
    # PYTHONPATH=plugin_dir and cannot import flash, so sharing one source of truth with the
    # parent-side unit tests means embedding the module text into the sitecustomize. strip its future
    # import (it would land mid-file after the grpo sitecustomize -> SyntaxError); the child is python
    # 3.12, so the ``X | None`` annotations evaluate natively without it.
    _runtime_src = inspect.getsource(_opd_vllm_runtime_mod).replace(
        "from __future__ import annotations\n", ""
    )
    return _runtime_src + "\n" + r"""
import asyncio
import copy
import hashlib
import json
import math
import random
import time
import urllib.error
import urllib.request

import torch

_FLASH_OPD_BRIDGE_URL = os.environ["FLASH_OPENRLHF_OPD_BRIDGE_URL"]
_FLASH_OPD_KL_COEF = float(os.environ["FLASH_OPENRLHF_OPD_KL_COEF"])
_FLASH_OPD_SEED = int(os.environ["FLASH_OPENRLHF_OPD_SEED"])
_FLASH_OPD_STOPS = json.loads(os.environ.get("FLASH_OPENRLHF_OPD_STOP_SEQUENCES", "[]"))
_FLASH_OPD_EOS_IDS = json.loads(os.environ.get("FLASH_OPENRLHF_OPD_EOS_TOKEN_IDS", "[]"))
_FLASH_OPD_MAX_ATTEMPTS = int(os.environ.get("FLASH_OPENRLHF_OPD_NO_SIGNAL_ATTEMPTS", "3"))
_FLASH_OPD_EXACT_SAVE_STEPS = frozenset(
    int(step) for step in json.loads(os.environ.get("FLASH_OPENRLHF_OPD_EXACT_SAVE_STEPS", "[]"))
)
_FLASH_OPD_FINAL_STEP = int(os.environ.get("FLASH_OPENRLHF_OPD_FINAL_STEP", "0"))
_FLASH_BRIDGE_TRANSPORT_ATTEMPTS = 3
_FLASH_PERMANENT_TEACHER_EXIT = 86
_FLASH_TRANSIENT_TEACHER_EXIT = 87

# per-run teacher skip accounting (parity with TRL _score_one + the step-loop under-run gate): a
# transient per-sample scoring failure is skipped-and-counted so one flaky teacher call cannot abort
# the whole run; the terminal gate turns only a transient-caused signal shortfall into a retry.
# steps_with_signal/steps_total are recorded per optimizer step from teacher coverage.
_flash_opd_teacher_stats = {
    "new_transient": 0,
    "no_signal": 0,
    "steps_total": 0,
    "steps_with_signal": 0,
}


# transient teacher failure on a skippable per-sample scoring request: skip the sample and let the
# run continue instead of aborting. only raised when the caller passes skippable=True.
class _FlashTeacherSkip(RuntimeError):
    pass


if _FLASH_OPD_KL_COEF <= 0:
    raise RuntimeError("FLASH_OPENRLHF_OPD_KL_COEF must be positive")
if _FLASH_OPD_MAX_ATTEMPTS <= 0:
    raise RuntimeError("FLASH_OPENRLHF_OPD_NO_SIGNAL_ATTEMPTS must be positive")


def _flash_identity(label):
    identity = json.loads(label) if isinstance(label, str) else dict(label)
    required = ("global_step", "example_index", "rollout_ordinal", "no_signal_attempt")
    if set(identity) != set(required) or any(
        isinstance(identity.get(name), bool)
        or not isinstance(identity.get(name), int)
        or identity.get(name) < 0
        for name in required
    ):
        raise RuntimeError("flash OPD rollout identity is invalid")
    return identity


def _flash_seed(identity):
    attempt = int(identity.get("no_signal_attempt", 0))
    payload = (
        f"{_FLASH_OPD_SEED}:{identity['global_step']}:{identity['example_index']}:"
        f"{identity['rollout_ordinal']}:0"
    )
    if attempt:
        payload += f":retry:{attempt}"
    digest = hashlib.blake2b(payload.encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


class _FlashTeacherBridgeError(RuntimeError):
    def __init__(self, message, *, classification, retry_transport=False):
        super().__init__(message)
        self.classification = classification
        self.retry_transport = retry_transport


def _flash_bridge_timeout(payload):
    return None if set(payload) == {"checkpoint"} else 600


def _flash_post_teacher_once(payload):
    request = urllib.request.Request(
        _FLASH_OPD_BRIDGE_URL,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_flash_bridge_timeout(payload)) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        try:
            detail = json.loads(error.read().decode("utf-8"))["error"]
            classification = str(detail["classification"])
            message = str(detail["message"])
        except Exception as decode_error:
            raise _FlashTeacherBridgeError(
                f"flash OPD bridge returned unclassified HTTP {error.code}",
                classification="permanent",
            ) from decode_error
        if classification not in {"permanent", "transient"}:
            raise _FlashTeacherBridgeError(
                f"flash OPD bridge returned unknown teacher failure classification {classification!r}",
                classification="permanent",
            ) from error
        raise _FlashTeacherBridgeError(message, classification=classification) from error
    except (OSError, TimeoutError, urllib.error.URLError) as error:
        raise _FlashTeacherBridgeError(
            f"flash OPD bridge transport failed: {type(error).__name__}",
            classification="transient",
            retry_transport=True,
        ) from error
    try:
        result = json.loads(body.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError) as error:
        raise _FlashTeacherBridgeError(
            "flash OPD bridge returned malformed success JSON",
            classification="permanent",
        ) from error
    if not isinstance(result, dict):
        raise _FlashTeacherBridgeError(
            "flash OPD bridge returned a non-object response",
            classification="permanent",
        )
    return result


def _flash_post_teacher(payload, skippable=False):
    for attempt in range(_FLASH_BRIDGE_TRANSPORT_ATTEMPTS):
        try:
            return _flash_post_teacher_once(payload)
        except _FlashTeacherBridgeError as error:
            if error.retry_transport and attempt + 1 < _FLASH_BRIDGE_TRANSPORT_ATTEMPTS:
                time.sleep(0.25 * (attempt + 1))
                continue
            if error.classification == "permanent":
                # bad key / model id / malformed: abort the run now, never burn the whole run.
                os._exit(_FLASH_PERMANENT_TEACHER_EXIT)
            if skippable:
                # transient failure on a per-sample scoring request: skip this sample and let the
                # run continue (TRL _score_one skip semantics). the terminal under-run gate turns a
                # transient-caused signal shortfall into a retry.
                raise _FlashTeacherSkip(str(error)) from error
            # transient failure on a control-plane request (metrics/checkpoint): retry the whole run.
            os._exit(_FLASH_TRANSIENT_TEACHER_EXIT)
    raise AssertionError("unreachable flash OPD bridge retry state")


from openrlhf.utils.agent import SingleTurnAgentExecutor as _FlashSingleTurnExecutor


async def _flash_opd_execute(self, prompt, label, sampling_params, max_length, hf_tokenizer, llm_engine, images=None):
    if images:
        raise RuntimeError("OpenRLHF OPD multimodal support is deferred")
    base_identity = _flash_identity(label)
    rollout_key = (base_identity["global_step"], base_identity["example_index"])
    rollout_ordinals = getattr(self, "_flash_opd_rollout_ordinals", None)
    if rollout_ordinals is None:
        rollout_ordinals = {}
        self._flash_opd_rollout_ordinals = rollout_ordinals
    base_identity["rollout_ordinal"] = rollout_ordinals.get(rollout_key, 0)
    rollout_ordinals[rollout_key] = base_identity["rollout_ordinal"] + 1
    bare_executor = _FlashSingleTurnExecutor(None)
    for attempt in range(_FLASH_OPD_MAX_ATTEMPTS):
        identity = dict(base_identity)
        identity["no_signal_attempt"] = attempt
        attempt_label = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        params = copy.deepcopy(sampling_params)
        params.seed = _flash_seed(identity)
        if _FLASH_OPD_STOPS:
            params.stop = list(_FLASH_OPD_STOPS)
            params.include_stop_str_in_output = True
        if _FLASH_OPD_EOS_IDS:
            params.stop_token_ids = list(_FLASH_OPD_EOS_IDS)
        output = await _original_execute(
            bare_executor,
            prompt,
            attempt_label,
            params,
            max_length,
            hf_tokenizer,
            llm_engine,
            images=None,
        )
        start, end = output["action_ranges"][0]
        try:
            result = await asyncio.to_thread(
                _flash_post_teacher,
                {
                    "label": attempt_label,
                    "prompt_length": start,
                    "sequence_ids": output["observation_tokens"][:end],
                    "terminated": not bool(output.get("truncated", False)),
                },
                skippable=True,
            )
        except _FlashTeacherSkip as skip:
            # transient teacher flap on this sample: count it and resample this rollout. a transient
            # attempt is treated like a no-signal attempt, never a whole-run abort.
            _flash_opd_teacher_stats["new_transient"] += 1
            print(
                f"[opd] teacher score transient failure, resampling rollout "
                f"{attempt + 1}/{_FLASH_OPD_MAX_ATTEMPTS}: {skip}",
                flush=True,
            )
            continue
        output["label"] = attempt_label
        output["reward"] = 0.0
        output["scores"] = 0.0
        output["teacher_group_ids"] = result["teacher_group_ids"]
        output["teacher_logsums"] = result["teacher_logsums"]
        output["teacher_signal_mask"] = result["teacher_signal_mask"]
        output["teacher_coverage"] = float(result.get("coverage", 0.0))
        if int(result.get("signal_count", 0)) > 0:
            return output
    # every attempt produced no aligned teacher signal (genuine no-signal, or a transient flap on the
    # final attempt): drop the rollout from the optimizer update. when the final attempt was skipped
    # before the teacher fields were set, synthesize an all-no-signal payload sized exactly like
    # _encode_action_metadata (action_length = prompt_length + response_length - 1).
    if "teacher_signal_mask" not in output:
        start, end = output["action_ranges"][0]
        action_length = start + (end - start) - 1
        output["label"] = attempt_label
        output["reward"] = 0.0
        output["scores"] = 0.0
        output["teacher_group_ids"] = [-1] * action_length
        output["teacher_logsums"] = [0.0] * action_length
        output["teacher_signal_mask"] = [False] * action_length
        output["teacher_coverage"] = 0.0
    _flash_opd_teacher_stats["no_signal"] += 1
    print(
        f"flash OPD produced no aligned teacher signal after {_FLASH_OPD_MAX_ATTEMPTS} rollout attempts; "
        "dropping the rollout from the optimizer update",
        flush=True,
    )
    return output


_FlashSingleTurnExecutor.execute = _flash_opd_execute

from openrlhf.trainer.ppo_utils.samples_generator import SamplesGenerator as _FlashSamplesGenerator


def _flash_opd_process_response(self, response, **generate_kwargs):
    experience = _original_process_response(self, response, **generate_kwargs)
    expected = experience.action_mask.shape[-1]
    fields = {
        "flash_teacher_group_ids": (response.get("teacher_group_ids"), torch.long),
        "flash_teacher_logsums": (response.get("teacher_logsums"), torch.float32),
        "flash_teacher_signal_mask": (response.get("teacher_signal_mask"), torch.bool),
    }
    for name, (value, dtype) in fields.items():
        if not isinstance(value, list) or len(value) != expected:
            raise RuntimeError(f"OpenRLHF OPD response has invalid {name}")
        experience.info[name] = torch.tensor(value, dtype=dtype).unsqueeze(0)
    signal_mask = experience.info["flash_teacher_signal_mask"]
    experience.action_mask = experience.action_mask.bool() & signal_mask
    experience.info["teacher_coverage"] = torch.tensor([float(response.get("teacher_coverage", 0.0))])
    return experience


_FlashSamplesGenerator._process_response_into_experience = _flash_opd_process_response

from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker as _FlashExperienceMaker


def _flash_make_experience_batch(self, rollout_samples):
    return self.split_rollout_samples(rollout_samples)


_FlashExperienceMaker.make_experience_batch = _flash_make_experience_batch


# --- opd rollout vLLM runtime shim ---------------------------------------------------------------
# restore fp8 kv cache, chunked prefill, dynamic max_num_seqs, the mamba scheduler-budget floor, and
# the gpu-family enforce_eager safeguard on the openrlhf rollout engine. the decision and the async
# actor __init__ wrapping both live in the embedded opd_vllm_runtime module (shared with parent-side
# unit tests); here we only supply the live-gpu plan provider and the sidecar meta writer.
_FLASH_OPD_VLLM_MAX_NUM_SEQS = int(os.environ.get("FLASH_OPENRLHF_OPD_VLLM_MAX_NUM_SEQS", "0"))
_FLASH_OPD_VLLM_MAMBA_BLOCK = int(os.environ.get("FLASH_OPENRLHF_OPD_VLLM_MAMBA_BLOCK_SIZE", "0"))
_FLASH_OPD_VLLM_SEQ_CAP = int(os.environ.get("FLASH_OPENRLHF_MAX_RESPONSE_LENGTH", "0"))
_FLASH_OPD_VLLM_META_PATH = os.environ.get("FLASH_OPENRLHF_OPD_VLLM_META_PATH", "")


def _flash_write_opd_vllm_meta(meta):
    if not _FLASH_OPD_VLLM_META_PATH:
        return
    try:
        directory = os.path.dirname(_FLASH_OPD_VLLM_META_PATH)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = _FLASH_OPD_VLLM_META_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(meta, handle)
        os.replace(tmp, _FLASH_OPD_VLLM_META_PATH)
    except OSError as exc:
        print("[opd][warn] could not write vLLM runtime meta: " + str(exc))


def _flash_opd_vllm_plan_provider():
    import vllm as _flash_vllm_module

    cc = None
    card_gb = None
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        card_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return opd_rollout_runtime_plan(
        cc=cc,
        vllm_version=getattr(_flash_vllm_module, "__version__", ""),
        seq_cap=_FLASH_OPD_VLLM_SEQ_CAP,
        max_num_seqs=_FLASH_OPD_VLLM_MAX_NUM_SEQS or None,
        card_gb=card_gb,
        mamba_block_size=_FLASH_OPD_VLLM_MAMBA_BLOCK,
    )


try:
    from openrlhf.trainer.ray import vllm_engine as _flash_vllm_engine_module

    install_rollout_runtime_shim(
        _flash_vllm_engine_module.RolloutRayActor,
        _flash_opd_vllm_plan_provider,
        meta_writer=_flash_write_opd_vllm_meta,
    )
    print("[opd] installed vLLM rollout runtime shim on RolloutRayActor")
except Exception as exc:
    print("[opd][warn] could not install vLLM rollout runtime shim: " + str(exc))


from openrlhf.trainer import ppo_trainer as _flash_ppo_trainer_module

_FlashPolicyModelActor = _ppo_actor_module.PolicyModelActor
_flash_policy_ray_metadata = getattr(_FlashPolicyModelActor, "__ray_metadata__", None)
_FlashPolicyModelActorRuntime = (
    getattr(_flash_policy_ray_metadata, "modified_class", None) or _FlashPolicyModelActor
)
_original_policy_save_checkpoint = _FlashPolicyModelActorRuntime.save_checkpoint


def _flash_capture_training_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": None,
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    with contextlib.suppress(ImportError):
        import numpy as np

        state["numpy"] = np.random.get_state()
    return state


@functools.wraps(_original_policy_save_checkpoint)
def _flash_policy_save_checkpoint(
    self,
    tag,
    client_states=None,
    metric_value=None,
    metric_key=None,
):
    result = _original_policy_save_checkpoint(
        self,
        tag,
        client_states=client_states,
        metric_value=metric_value,
        metric_key=metric_key,
    )
    if self.strategy.is_rank_0() and str(tag).startswith("global_step"):
        # deepspeed restores authoritative per-rank rng from its native actor checkpoint; this
        # child rank-zero sidecar satisfies flash's shared resume-bundle contract.
        torch.save(
            _flash_capture_training_rng_state(),
            os.path.join(self.strategy.args.ckpt.path, "_actor", str(tag), "rng_state.pth"),
        )
    return result


_FlashPolicyModelActorRuntime.save_checkpoint = _flash_policy_save_checkpoint
_FlashPPOTrainer = _flash_ppo_trainer_module.PPOTrainer
_flash_ray_metadata = getattr(_FlashPPOTrainer, "__ray_metadata__", None)
_FlashPPOTrainerRuntime = (
    getattr(_flash_ray_metadata, "modified_class", None) or _FlashPPOTrainer
)


class _FlashOrderedPromptStrategy:
    def __init__(self, strategy):
        self._strategy = strategy

    def __getattr__(self, name):
        return getattr(self._strategy, name)

    def setup_dataloader(
        self,
        replay_buffer,
        batch_size,
        pin_memory=False,
        shuffle=True,
        collate_fn=None,
        drop_last=True,
        sampler=None,
        consumed_samples=0,
        num_workers=0,
    ):
        del shuffle
        return self._strategy.setup_dataloader(
            replay_buffer,
            batch_size,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=drop_last,
            sampler=sampler,
            consumed_samples=consumed_samples,
            num_workers=num_workers,
        )


_original_prepare_datasets = _flash_ppo_trainer_module.prepare_datasets


@functools.wraps(_original_prepare_datasets)
def _flash_prepare_datasets(strategy, tokenizer):
    return _original_prepare_datasets(_FlashOrderedPromptStrategy(strategy), tokenizer)


_flash_ppo_trainer_module.prepare_datasets = _flash_prepare_datasets
_original_save_logs_and_checkpoints = _FlashPPOTrainerRuntime.save_logs_and_checkpoints


def _flash_metric_scalar(logs, *names):
    for name in names:
        if name not in logs:
            continue
        value = logs[name]
        tensor_type = getattr(torch, "Tensor", ())
        if tensor_type and isinstance(value, tensor_type):
            value = value.detach().float().mean().item()
        value = float(value)
        if not math.isfinite(value):
            raise RuntimeError(f"OpenRLHF OPD metric {name} is not finite")
        return value
    raise RuntimeError(f"OpenRLHF OPD metrics are missing {names[0]}")


@functools.wraps(_original_save_logs_and_checkpoints)
def _flash_save_logs_and_checkpoints(self, global_step, logs_dict=None, client_states=None):
    global_step = int(global_step)
    logs = logs_dict or {}
    step_coverage = _flash_metric_scalar(logs, "teacher_coverage")
    # per-optimizer-step signal accounting: a step with zero teacher coverage delivered no aligned
    # signal (every rollout was dropped). the terminal gate classifies a coverage shortfall as a
    # transient (retriable) or deterministic outcome.
    _flash_opd_teacher_stats["steps_total"] += 1
    if step_coverage > 0:
        _flash_opd_teacher_stats["steps_with_signal"] += 1
    _flash_post_teacher(
        {
            "metrics": {
                "step": global_step,
                "loss": _flash_metric_scalar(logs, "distillation_loss", "policy_loss"),
                "coverage": step_coverage,
            }
        }
    )
    original_save_steps = self.args.ckpt.save_steps
    periodic_due = (
        not _FLASH_OPD_EXACT_SAVE_STEPS
        and original_save_steps != float("inf")
        and global_step % int(original_save_steps) == 0
    )
    save_due = (
        global_step in _FLASH_OPD_EXACT_SAVE_STEPS
        or global_step == _FLASH_OPD_FINAL_STEP
        or periodic_due
    )
    self.args.ckpt.save_steps = global_step if save_due else float("inf")
    save_succeeded = False
    try:
        result = _original_save_logs_and_checkpoints(
            self,
            global_step,
            logs_dict=logs_dict,
            client_states=client_states,
        )
        save_succeeded = True
    finally:
        self.args.ckpt.save_steps = original_save_steps
    if save_due and save_succeeded:
        _flash_post_teacher({"checkpoint": global_step})
    return result


_FlashPPOTrainerRuntime.save_logs_and_checkpoints = _flash_save_logs_and_checkpoints
_original_ppo_fit = _FlashPPOTrainerRuntime.fit


@functools.wraps(_original_ppo_fit)
def _flash_ppo_fit(self, *args, **kwargs):
    global_step = kwargs.get("global_step", args[0] if args else 0)
    load_enabled = bool(getattr(getattr(self.args, "ckpt", None), "load_enable", False))
    warmstart = bool(os.environ.get("FLASH_OPENRLHF_WARMSTART_ADAPTER"))
    already_synced = bool(getattr(self, "_flash_warmstart_broadcast_done", False))
    if warmstart and not global_step and not load_enabled and not already_synced:
        self.broadcast_to_vllm()
        self._flash_warmstart_broadcast_done = True
    result = _original_ppo_fit(self, *args, **kwargs)
    # terminal under-run classification (parity with TRL's post-loop guard). rank 0 checks whether
    # every optimizer step landed on aligned teacher signal. a coverage shortfall that involved a NEW
    # transient teacher failure this attempt is retriable infra -> exit transient so the platform
    # retries the whole run (bounded by max_retries); a shortfall with no transient failure is a
    # deterministic shortfall the run genuinely could not align, and it completes on the signal it
    # delivered rather than looping.
    stats = _flash_opd_teacher_stats
    if self.strategy.is_rank_0() and stats["steps_with_signal"] < stats["steps_total"]:
        summary = (
            f"steps_with_signal={stats['steps_with_signal']}/{stats['steps_total']} "
            f"new_transient={stats['new_transient']} no_signal_drops={stats['no_signal']}"
        )
        if stats["new_transient"] > 0:
            print(f"flash OPD teacher signal shortfall is transient (retrying run): {summary}", flush=True)
            os._exit(_FLASH_TRANSIENT_TEACHER_EXIT)
        print(
            f"flash OPD teacher signal shortfall is deterministic (completing on delivered signal): {summary}",
            flush=True,
        )
    return result


_FlashPPOTrainerRuntime.fit = _flash_ppo_fit


def _flash_pad_info(value, action_mask, pad_value):
    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, list) and value and all(isinstance(item, torch.Tensor) for item in value):
        tensor = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=pad_value)
    else:
        raise RuntimeError("OpenRLHF OPD teacher metadata was not preserved through replay batching")
    tensor = tensor.to(action_mask.device)
    if tensor.shape != action_mask.shape:
        if tensor.shape[0] != action_mask.shape[0] or tensor.shape[1] > action_mask.shape[1]:
            raise RuntimeError("OpenRLHF OPD teacher metadata shape does not match action logprobs")
        tensor = torch.nn.functional.pad(tensor, (0, action_mask.shape[1] - tensor.shape[1]), value=pad_value)
    return tensor


_FLASH_OPD_LOGIT_CHUNK_SIZE = 256


def _flash_output_embeddings(actor):
    model = getattr(actor.model, "module", actor.model)
    output_embeddings = model.get_output_embeddings()
    input_embeddings = model.get_input_embeddings()
    if output_embeddings is None or output_embeddings is input_embeddings:
        raise RuntimeError("OpenRLHF OPD chunked projection requires a distinct output embedding module")
    return output_embeddings


def _flash_logit_transforms(actor):
    model = getattr(actor.model, "module", actor.model)
    config = getattr(model, "config", None)
    if config is None:
        raise RuntimeError("OpenRLHF OPD chunked projection requires an accessible model config")
    text_config = getattr(config, "text_config", None) or config
    logit_scale = float(getattr(text_config, "logit_scale", 1.0))
    final_logit_softcapping = getattr(text_config, "final_logit_softcapping", None)
    if final_logit_softcapping is not None:
        final_logit_softcapping = float(final_logit_softcapping)
    return logit_scale, final_logit_softcapping


def _flash_zero3_managed_parameters(module):
    return [parameter for parameter in module.parameters() if hasattr(parameter, "ds_id")]


def _flash_zero3_synchronized_chunk_count(selected_count, chunk_size, output_head, device):
    local_chunks = (int(selected_count) + int(chunk_size) - 1) // int(chunk_size)
    parameters = _flash_zero3_managed_parameters(output_head)
    if not parameters or not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return local_chunks
    process_group = getattr(parameters[0], "ds_process_group", None)
    if torch.distributed.get_world_size(group=process_group) <= 1:
        return local_chunks
    max_chunks = torch.tensor(local_chunks, device=device, dtype=torch.int64)
    torch.distributed.all_reduce(
        max_chunks,
        op=torch.distributed.ReduceOp.MAX,
        group=process_group,
    )
    return int(max_chunks.item())


def _flash_register_zero3_external_output_head(actor, output_head):
    parameters = _flash_zero3_managed_parameters(output_head)
    if not parameters:
        return False
    import deepspeed

    forward_module = getattr(actor.model, "module", actor.model)
    register = deepspeed.zero.register_external_parameter
    for parameter in parameters:
        register(forward_module, parameter)
    return True


def _flash_action_hidden_states(actor, sequences, attention_mask):
    if sequences.ndim != 2 or attention_mask.shape != sequences.shape:
        raise ValueError("OpenRLHF OPD chunked projection expects matching sequence and attention tensors")
    output_embeddings = _flash_output_embeddings(actor)
    captured = {}
    zero3_output_head = False
    had_instance_forward = "forward" in output_embeddings.__dict__
    original_instance_forward = output_embeddings.__dict__.get("forward")

    def _capture_hidden(hidden_states, *_args, **_kwargs):
        nonlocal zero3_output_head
        if "hidden_states" in captured:
            raise RuntimeError("OpenRLHF OPD output embedding was invoked more than once")
        captured["hidden_states"] = hidden_states
        # register after the backbone, while the enclosing actor forward is still active
        zero3_output_head = _flash_register_zero3_external_output_head(actor, output_embeddings)
        return hidden_states[..., :1] * 0

    output_embeddings.forward = _capture_hidden
    try:
        if getattr(actor, "is_vlm", False):
            position_ids = None
        else:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        actor.model(
            sequences,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
    finally:
        if had_instance_forward:
            output_embeddings.forward = original_instance_forward
        else:
            del output_embeddings.forward

    hidden_states = captured.get("hidden_states")
    if hidden_states is None or hidden_states.shape[:2] != sequences.shape:
        raise RuntimeError("OpenRLHF OPD could not capture full-sequence hidden states")
    return hidden_states, zero3_output_head


def _flash_chunked_token_logps(
    lm_head,
    hidden_states,
    token_ids,
    *,
    temperature,
    logit_scale=1.0,
    final_logit_softcapping=None,
    chunk_size=_FLASH_OPD_LOGIT_CHUNK_SIZE,
    zero3_active=None,
):
    from torch.utils.checkpoint import checkpoint as torch_checkpoint

    if hidden_states.ndim != 2 or token_ids.ndim != 1:
        raise ValueError("OpenRLHF OPD chunked projection expects [tokens, hidden] states and [tokens] ids")
    if hidden_states.shape[0] != token_ids.shape[0]:
        raise ValueError("OpenRLHF OPD chunked projection token counts differ")
    if float(temperature) <= 0:
        raise ValueError("OpenRLHF OPD rollout temperature must be positive")

    def _logps(hidden_chunk, ids_chunk):
        logits = lm_head(hidden_chunk).float()
        if float(logit_scale) != 1.0:
            logits.mul_(float(logit_scale))
        if final_logit_softcapping is not None:
            softcap = float(final_logit_softcapping)
            logits = softcap * torch.tanh(logits / softcap)
        if float(temperature) != 1.0:
            logits.div_(float(temperature))
        return -torch.nn.functional.cross_entropy(logits, ids_chunk, reduction="none")

    if zero3_active is None:
        zero3_active = bool(_flash_zero3_managed_parameters(lm_head))
    if hidden_states.shape[0] == 0 and not zero3_active:
        return hidden_states.float().sum(dim=-1)

    def checkpoint(function, *args):
        return torch_checkpoint(function, *args, use_reentrant=False)

    if zero3_active and hidden_states.requires_grad:

        def checkpoint(function, *args):
            return torch_checkpoint(function, *args, use_reentrant=True)

        try:
            import deepspeed
        except ImportError:
            pass
        else:
            deepspeed_checkpoint = getattr(
                getattr(deepspeed, "checkpointing", None), "checkpoint", None
            )
            if callable(deepspeed_checkpoint):
                checkpoint = deepspeed_checkpoint

    size = max(1, int(chunk_size))
    selected_count = int(hidden_states.shape[0])
    if zero3_active:
        synchronized_chunks = _flash_zero3_synchronized_chunk_count(
            selected_count,
            size,
            lm_head,
            hidden_states.device,
        )
        padded_count = synchronized_chunks * size
        if not hidden_states.requires_grad:
            padded_count = max(padded_count, 1)
        padding_count = padded_count - selected_count
        if padding_count:
            zero_hidden = (
                hidden_states[:1] * 0.0
                if selected_count
                else hidden_states.sum(dim=0, keepdim=True) * 0.0
            )
            hidden_states = torch.cat(
                [hidden_states, zero_hidden.expand(padding_count, -1)],
                dim=0,
            )
            token_ids = torch.cat(
                [token_ids, token_ids.new_zeros(padding_count)],
                dim=0,
            )
    else:
        padded_count = selected_count

    chunks = []
    for start in range(0, padded_count, size):
        hidden_chunk = hidden_states[start : start + size]
        ids_chunk = token_ids[start : start + size]
        if torch.is_grad_enabled() and (not zero3_active or hidden_chunk.requires_grad):
            chunks.append(
                checkpoint(
                    _logps,
                    hidden_chunk,
                    ids_chunk,
                )
            )
        else:
            chunks.append(_logps(hidden_chunk, ids_chunk))
    if not chunks:
        return hidden_states.float().sum(dim=-1)
    return torch.cat(chunks)[:selected_count]


def _flash_chunked_action_log_probs(actor, sequences, action_mask, attention_mask, *, chunk_size=None):
    if action_mask.shape != (sequences.shape[0], sequences.shape[1] - 1):
        raise ValueError("OpenRLHF OPD action mask must align with next-token predictions")
    response_mask = action_mask.bool()
    hidden_states, zero3_output_head = _flash_action_hidden_states(
        actor, sequences, attention_mask
    )
    hidden_states = hidden_states[:, :-1]
    target_ids = sequences[:, 1:]
    output_head = _flash_output_embeddings(actor)
    logit_scale, final_logit_softcapping = _flash_logit_transforms(actor)
    flat_logps = _flash_chunked_token_logps(
        output_head,
        hidden_states[response_mask],
        target_ids[response_mask],
        temperature=actor.temperature,
        logit_scale=logit_scale,
        final_logit_softcapping=final_logit_softcapping,
        chunk_size=_FLASH_OPD_LOGIT_CHUNK_SIZE if chunk_size is None else chunk_size,
        zero3_active=zero3_output_head,
    )
    return flat_logps.new_zeros(action_mask.shape).masked_scatter(response_mask, flat_logps)


def _flash_reverse_kl(student_logprobs, teacher_logsums, group_ids, response_mask):
    selected = response_mask.bool() & group_ids.ge(0)
    if not bool(selected.any().item()):
        raise RuntimeError("flash OPD actor batch contains no aligned teacher signal")
    row_losses = []
    for row in range(student_logprobs.shape[0]):
        row_selected = selected[row]
        if not bool(row_selected.any().item()):
            continue
        token_losses = torch.zeros_like(student_logprobs[row])
        for group_id in torch.unique(group_ids[row][row_selected], sorted=True):
            group_mask = row_selected & group_ids[row].eq(group_id)
            group_length = group_mask.sum().to(dtype=student_logprobs.dtype)
            student_logsum = student_logprobs[row][group_mask].detach().sum()
            teacher_logsum = teacher_logsums[row][group_mask][0]
            coefficient = _FLASH_OPD_KL_COEF * (student_logsum - teacher_logsum) / group_length
            token_losses[group_mask] = coefficient * student_logprobs[row][group_mask]
        row_losses.append(token_losses[row_selected].mean())
    return torch.stack(row_losses).mean(), selected


def _flash_is_optimizer_update(step, accumulated_gradient):
    accumulated_gradient = int(accumulated_gradient)
    if accumulated_gradient <= 0:
        raise RuntimeError("OpenRLHF OPD gradient accumulation must be positive")
    return (int(step) + 1) % accumulated_gradient == 0


def _flash_should_mark_mutation(trainer, step, accumulated_gradient, has_signal):
    window_has_signal = bool(
        getattr(trainer, "_flash_opd_accumulation_has_signal", False) or has_signal
    )
    trainer._flash_opd_accumulation_has_signal = window_has_signal
    if not _flash_is_optimizer_update(step, accumulated_gradient):
        return False
    trainer._flash_opd_accumulation_has_signal = False
    return window_has_signal


def _flash_aligned_sample_count(selected):
    return float(selected.any(dim=-1).sum().item())


def _flash_current_batch_has_signal(strategy, aligned_samples):
    global_aligned_samples = strategy.all_reduce(
        {"flash_aligned_samples": aligned_samples},
        op="sum",
    )["flash_aligned_samples"]
    return float(global_aligned_samples) > 0


def _flash_global_sequence_mean_loss(
    local_loss,
    aligned_samples,
    global_batch_size,
    world_size,
):
    if global_batch_size <= 0:
        return local_loss * 0.0
    return local_loss * (aligned_samples * int(world_size) / global_batch_size)


def _flash_loss_metrics(loss, coverage, has_signal):
    metric_weight = "sample" if has_signal else None
    return (
        {
            "policy_loss": loss,
            "distillation_loss": loss,
            "teacher_coverage": coverage,
        },
        {
            "policy_loss": metric_weight,
            "distillation_loss": metric_weight,
            "teacher_coverage": metric_weight,
        },
    )


def _flash_advance_empty_accumulation(trainer):
    engine = trainer.actor.model
    previous_boundary = getattr(engine, "_is_gradient_accumulation_boundary", None)
    engine.set_gradient_accumulation_boundary(False)
    try:
        trainer.strategy.optimizer_step(
            trainer.actor_optim,
            trainer.actor,
            trainer.actor_scheduler,
            name="actor",
        )
    finally:
        engine.set_gradient_accumulation_boundary(previous_boundary)
    optimizer = getattr(engine, "optimizer", None)
    if optimizer is not None and hasattr(optimizer, "zero_grad"):
        optimizer.zero_grad()
    else:
        engine.zero_grad()


def _flash_opd_training_step(self, experience, kl_ctl, step, loss_batch_info=None):
    del kl_ctl
    if not isinstance(loss_batch_info, dict) or "global_batch_size" not in loss_batch_info:
        raise RuntimeError("OpenRLHF OPD requires global accumulation loss normalization")
    global_batch_size = loss_batch_info["global_batch_size"]
    if isinstance(global_batch_size, torch.Tensor):
        global_batch_size = global_batch_size.item()
    global_batch_size = float(global_batch_size)
    self.actor.train()
    with _flash_attention_context():
        action_log_probs = _flash_chunked_action_log_probs(
            self.actor,
            experience.sequences,
            experience.action_mask,
            experience.attention_mask,
        )
    group_ids = _flash_pad_info(
        experience.info["flash_teacher_group_ids"], action_log_probs, -1
    ).long()
    teacher_logsums = _flash_pad_info(
        experience.info["flash_teacher_logsums"], action_log_probs, 0.0
    ).to(dtype=action_log_probs.dtype)
    signal_mask = _flash_pad_info(
        experience.info["flash_teacher_signal_mask"], action_log_probs, False
    ).bool()
    response_mask = experience.action_mask.bool() & signal_mask
    selected = response_mask & group_ids.ge(0)
    aligned_samples = _flash_aligned_sample_count(selected)
    has_current_signal = _flash_current_batch_has_signal(self.strategy, aligned_samples)
    has_window_signal = global_batch_size > 0
    if bool(selected.any().item()):
        local_loss, selected = _flash_reverse_kl(
            action_log_probs,
            teacher_logsums,
            group_ids,
            response_mask,
        )
    else:
        local_loss = action_log_probs.sum() * 0.0
    loss = _flash_global_sequence_mean_loss(
        local_loss,
        aligned_samples,
        global_batch_size,
        self.strategy.world_size,
    )
    with _flash_attention_context():
        self.strategy.backward(loss, self.actor, self.actor_optim)
    is_optimizer_update = _flash_is_optimizer_update(step, self.strategy.accumulated_gradient)
    should_mutate = _flash_should_mark_mutation(
        self,
        step,
        self.strategy.accumulated_gradient,
        has_window_signal,
    )
    if should_mutate:
        _flash_post_teacher({"mutation": True})
        self.strategy.optimizer_step(
            self.actor_optim,
            self.actor,
            self.actor_scheduler,
            name="actor",
        )
    elif is_optimizer_update:
        _flash_advance_empty_accumulation(self)
    else:
        self.strategy.optimizer_step(
            self.actor_optim,
            self.actor,
            self.actor_scheduler,
            name="actor",
        )
    grad_norm = self.strategy.get_grad_norm(self.actor)
    coverage = experience.info.get("teacher_coverage", [])
    if isinstance(coverage, list):
        coverage = torch.stack(coverage).float().mean() if coverage else loss.detach().new_zeros(())
    elif isinstance(coverage, torch.Tensor):
        coverage = coverage.float().mean()
    else:
        coverage = loss.detach().new_zeros(())
    metrics = {
        "actor_lr": self.actor_scheduler.get_last_lr()[0],
        "actor_grad_norm": grad_norm,
    }
    weights = {
        "actor_lr": None,
        "actor_grad_norm": None,
    }
    loss_metrics, loss_weights = _flash_loss_metrics(
        local_loss.detach(),
        coverage.detach(),
        has_current_signal,
    )
    metrics.update(loss_metrics)
    weights.update(loss_weights)
    self._flash_opd_ppo_has_signal = bool(
        getattr(self, "_flash_opd_ppo_has_signal", False) or has_current_signal
    )
    return {
        "metrics": metrics,
        "weights": weights,
        "num_samples": aligned_samples,
        "num_action_tokens": float(selected.sum().item()),
    }


_ActorPPOTrainer.training_step = _flash_opd_training_step
_original_actor_ppo_train = _ActorPPOTrainer.ppo_train


@functools.wraps(_original_actor_ppo_train)
def _flash_actor_ppo_train(self, *args, **kwargs):
    self._flash_opd_ppo_has_signal = False
    try:
        status = _original_actor_ppo_train(self, *args, **kwargs)
    except ZeroDivisionError:
        if self._flash_opd_ppo_has_signal:
            raise
        status = {}
    status.setdefault("policy_loss", 0.0)
    status.setdefault("distillation_loss", 0.0)
    status.setdefault("teacher_coverage", 0.0)
    return status


_ActorPPOTrainer.ppo_train = _flash_actor_ppo_train

if os.environ.get("FLASH_OPENRLHF_WARMSTART_ADAPTER"):
    _flash_actor_init_before_warmstart = _Actor.__init__

    @functools.wraps(_flash_actor_init_before_warmstart)
    def _flash_actor_init_with_warmstart(self, *args, **kwargs):
        _flash_actor_init_before_warmstart(self, *args, **kwargs)
        adapter_dir = os.environ["FLASH_OPENRLHF_WARMSTART_ADAPTER"]
        safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        bin_path = os.path.join(adapter_dir, "adapter_model.bin")
        if os.path.isfile(safetensors_path):
            from safetensors.torch import load_file

            state = load_file(safetensors_path)
        elif os.path.isfile(bin_path):
            state = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            raise RuntimeError("OpenRLHF OPD warm-start adapter has no weights")
        b_weights = [tensor for name, tensor in state.items() if "lora_B" in name]
        if not b_weights or not any(bool(torch.count_nonzero(tensor).item()) for tensor in b_weights):
            raise RuntimeError("OpenRLHF OPD warm-start adapter has no nonzero LoRA B delta")
        from peft.utils.save_and_load import set_peft_model_state_dict

        result = set_peft_model_state_dict(self.model, state, adapter_name="default")
        missing = [
            key for key in (getattr(result, "missing_keys", []) or []) if "lora_" in key
        ]
        unexpected = [
            key for key in (getattr(result, "unexpected_keys", []) or []) if "lora_" in key
        ]
        if missing or unexpected:
            raise RuntimeError(
                "OpenRLHF OPD warm-start adapter load was incomplete: "
                f"missing={missing}, unexpected={unexpected}"
            )
        loaded_b = [
            parameter
            for name, parameter in self.model.named_parameters()
            if name.endswith("lora_B.default.weight") and parameter.requires_grad
        ]
        if not loaded_b or not any(bool(torch.count_nonzero(parameter).item()) for parameter in loaded_b):
            raise RuntimeError("OpenRLHF OPD warm-start adapter did not load a nonzero LoRA B delta")

    _Actor.__init__ = _flash_actor_init_with_warmstart

print(
    "[flash-openrlhf] flash opd chunked reverse-kl hooks active; "
    "32k gpu-validation-pending on h100 qwen3.5-0.8b for two steps with teacher alignment and post-sync rollout",
    flush=True,
)
""".lstrip()


def _sitecustomize_source() -> str:
    return _grpo_sitecustomize_source() + "\n" + _opd_sitecustomize_extension()


def write_openrlhf_opd_sitecustomize(directory: str) -> str:
    plugin_dir = os.path.join(directory, "flash_openrlhf_opd_plugin")
    os.makedirs(plugin_dir, exist_ok=True)
    Path(plugin_dir, _SITE_CUSTOMIZE_NAME).write_text(_sitecustomize_source(), encoding="utf-8")
    return plugin_dir


def build_openrlhf_opd_child_env(
    *,
    plugin_dir: str,
    max_response_length: int,
    language_model_only: bool,
    bridge_url: str,
    kl_penalty_coef: float,
    seed: int,
    stop_sequences: tuple[str, ...],
    eos_token_ids: frozenset[int],
    save_at_steps: tuple[int, ...] = (),
    final_step: int = 0,
    warmstart_adapter: str | None = None,
    actor_attn_implementation: str | None = None,
    rollout_max_num_seqs: int = 0,
    mamba_block_size: int = 0,
    vllm_meta_path: str | None = None,
) -> dict[str, str]:
    """Build the isolated child env with bridge capability but no provider credential."""
    child = build_openrlhf_child_env(
        plugin_dir=plugin_dir,
        max_response_length=max_response_length,
        language_model_only=language_model_only,
        sm120_vllm_backend=_is_sm120(),
        actor_attn_implementation=actor_attn_implementation,
    )
    child.update(
        {
            "FLASH_OPENRLHF_OPD_BRIDGE_URL": bridge_url,
            "FLASH_OPENRLHF_OPD_KL_COEF": str(float(kl_penalty_coef)),
            "FLASH_OPENRLHF_OPD_SEED": str(int(seed)),
            "FLASH_OPENRLHF_OPD_STOP_SEQUENCES": json.dumps(list(stop_sequences)),
            "FLASH_OPENRLHF_OPD_EOS_TOKEN_IDS": json.dumps(sorted(eos_token_ids)),
            "FLASH_OPENRLHF_OPD_NO_SIGNAL_ATTEMPTS": str(_OPENRLHF_NO_SIGNAL_ATTEMPTS),
            "FLASH_OPENRLHF_OPD_EXACT_SAVE_STEPS": json.dumps(list(save_at_steps)),
            "FLASH_OPENRLHF_OPD_FINAL_STEP": str(int(final_step)),
            # inputs for the rollout-actor vLLM runtime shim (fp8 kv / chunked prefill / max_num_seqs /
            # mamba floor / enforce_eager are decided in-actor where the gpu is visible). seq_cap comes
            # from FLASH_OPENRLHF_MAX_RESPONSE_LENGTH, already set by build_openrlhf_child_env.
            "FLASH_OPENRLHF_OPD_VLLM_MAX_NUM_SEQS": str(int(rollout_max_num_seqs)),
            "FLASH_OPENRLHF_OPD_VLLM_MAMBA_BLOCK_SIZE": str(int(mamba_block_size)),
        }
    )
    if vllm_meta_path:
        child["FLASH_OPENRLHF_OPD_VLLM_META_PATH"] = vllm_meta_path
    if warmstart_adapter:
        child["FLASH_OPENRLHF_WARMSTART_ADAPTER"] = warmstart_adapter
    return child


def _checkpoint_state(
    *,
    step: int,
    seed: int,
    prompt_pool_fingerprint: str,
    prompts_per_step: int,
    group_size: int,
    accounting: dict[str, int | float],
    loss_curve: list[float],
    coverage_curve: list[float],
    train_wall_seconds: float,
) -> dict[str, Any]:
    state = {
        "contract_version": OPD_RESUME_STATE_VERSION,
        "seed": int(seed),
        "opt_steps": int(step),
        "step": int(step),
        "rollout_seed_ordinal": int(step) * int(prompts_per_step) * int(group_size),
        "prompt_pool_fingerprint": prompt_pool_fingerprint,
        "generated_tokens": int(accounting["generated_tokens"]),
        "teacher_input_tokens": int(accounting["teacher_input_tokens"]),
        "truncated_rollouts": int(accounting["truncated_rollouts"]),
        "granularity_n": int(accounting["aligned_sequences"]),
        "samples_seen": int(accounting["samples_seen"]),
        "teacher_ok": int(accounting["teacher_ok"]),
        "teacher_transient": int(accounting["teacher_transient"]),
        "teacher_error": int(accounting["teacher_error"]),
        "no_signal_resamples": int(accounting["no_signal_resamples"]),
        "teacher_echo_deduped": int(accounting["teacher_echo_deduped"]),
        "group_granularity_sum": float(accounting["group_granularity_sum"]),
        "no_signal_skipped_steps": 0,
        "episodes_seen": int(accounting["samples_seen"]),
        "mt_turn_records": 0,
        "granularity_sum": float(accounting["coverage_sum"]),
        "train_wall_seconds": max(0.0, float(train_wall_seconds)),
        "loss_curve": list(loss_curve[:step]),
        "coverage_curve": list(coverage_curve[:step]),
        "skip_counts": {"empty_alignment": int(accounting["empty_alignments"])},
        "opd_phase_seconds": {},
        "opd_phase_counts": {},
    }
    return validate_opd_resume_state_metadata(
        state,
        expected_seed=int(seed),
        checkpoint_step=int(step),
    )


def _initial_checkpoint_step_states(
    resume_state: dict[str, Any] | None,
) -> dict[int, dict[str, Any]]:
    if resume_state is None:
        return {}
    return {int(resume_state["opt_steps"]): dict(resume_state)}


def _find_checkpoint_state_file(root: str, marker: str) -> str:
    for directory, _subdirs, files in os.walk(root):
        for name in sorted(files):
            if marker in name.lower() and name.endswith(".pt"):
                return os.path.join(directory, name)
    raise RuntimeError(f"OpenRLHF OPD checkpoint has no {marker} state file")


class _OpenRLHFOPDCheckpointPublisher:
    """Stage and publish a complete native DeepSpeed checkpoint before training continues."""

    def __init__(
        self,
        *,
        checkpoint_dir: str,
        staging_root: str,
        python_bin: str,
        model_id: str,
        model_revision: str,
        seed: int,
        prompt_pool_fingerprint: str,
        prompts_per_step: int,
        group_size: int,
        save_every: int,
        final_step: int,
        required_steps: tuple[int, ...],
        state_for_step,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.staging_root = staging_root
        self.python_bin = python_bin
        self.model_id = model_id
        self.model_revision = model_revision
        self.seed = seed
        self.prompt_pool_fingerprint = prompt_pool_fingerprint
        self.prompts_per_step = prompts_per_step
        self.group_size = group_size
        self.save_every = int(save_every)
        self.final_step = int(final_step)
        self.required_steps = frozenset(int(step) for step in required_steps)
        self.state_for_step = state_for_step
        self._lock = threading.Lock()
        self._published_steps: set[int] = set()

    def publish(self, step: int) -> None:
        step = int(step)
        with self._lock:
            if step in self._published_steps:
                return
            self._publish(step)
            self._published_steps.add(step)

    def _wait_for_checkpoint(self, step: int) -> tuple[str, str]:
        tag = f"global_step{step}"
        actor_tag = os.path.join(self.checkpoint_dir, "_actor", tag)
        hf_export = os.path.join(self.checkpoint_dir, f"{tag}_hf")
        deadline = time.monotonic() + 600.0
        latest_path = os.path.join(self.checkpoint_dir, "_actor", "latest")
        while time.monotonic() < deadline:
            latest = ""
            with contextlib.suppress(OSError):
                latest = Path(latest_path).read_text(encoding="utf-8").strip()
            adapter_ready = os.path.isfile(
                os.path.join(hf_export, "adapter_model.safetensors")
            ) or os.path.isfile(os.path.join(hf_export, "adapter_model.bin"))
            if (
                latest == tag
                and os.path.isdir(actor_tag)
                and os.path.isfile(os.path.join(hf_export, "adapter_config.json"))
                and adapter_ready
            ):
                return actor_tag, hf_export
            time.sleep(0.1)
        raise RuntimeError(f"timed out waiting for OpenRLHF OPD checkpoint step {step}")

    def _stage(self, step: int, actor_tag: str, hf_export: str) -> str:
        tag = f"global_step{step}"
        stage = os.path.join(self.staging_root, f"checkpoint-{step}")
        shutil.rmtree(stage, ignore_errors=True)
        os.makedirs(os.path.join(stage, "_actor"), exist_ok=True)
        shutil.copytree(actor_tag, os.path.join(stage, "_actor", tag))
        Path(stage, "_actor", "latest").write_text(tag, encoding="utf-8")
        shutil.copytree(hf_export, os.path.join(stage, f"{tag}_hf"))
        adapter_export = os.path.join(stage, "_adapter_export")
        export_openrlhf_adapter(
            hf_export,
            adapter_export,
            self.model_id,
            self.model_revision,
            self.python_bin,
        )
        for name in os.listdir(adapter_export):
            shutil.copy2(os.path.join(adapter_export, name), os.path.join(stage, name))
        optimizer = _find_checkpoint_state_file(actor_tag, "optim")
        shutil.copy2(optimizer, os.path.join(stage, "optimizer.pt"))
        trainer_rng = os.path.join(actor_tag, "rng_state.pth")
        if not os.path.isfile(trainer_rng):
            raise RuntimeError("OpenRLHF OPD checkpoint has no child trainer RNG state")
        shutil.copy2(trainer_rng, os.path.join(stage, "rng_state.pth"))
        state = self.state_for_step(step)
        with open(os.path.join(stage, "opd_state.json"), "w", encoding="utf-8") as state_file:
            json.dump(state, state_file, sort_keys=True)
        return stage

    def _publish(self, step: int) -> None:
        actor_tag, hf_export = self._wait_for_checkpoint(step)
        stage = self._stage(step, actor_tag, hf_export)

        required_deployable = step in self.required_steps
        periodic_deployable = (
            not self.required_steps
            and step != self.final_step
            and self.save_every > 0
            and step % self.save_every == 0
        )

        def publish_deployable() -> None:
            if required_deployable or periodic_deployable:
                _w.publish_deployable_checkpoint(
                    os.path.join(stage, "_adapter_export"),
                    step,
                    required=required_deployable,
                    _provenance_ready=True,
                )

        uploaded = _w.upload_resume_checkpoint(
            step,
            stage,
            after_upload=publish_deployable,
        )
        resume_required = required_deployable or step == self.final_step
        if resume_required and not uploaded:
            raise _w.RetriableInfraError(
                f"required OpenRLHF OPD checkpoint step {step} was not published"
            )


def _restore_openrlhf_resume(
    *,
    prompt_pool_fingerprint: str,
    seed: int,
    update_horizon: int,
) -> tuple[str | None, dict[str, Any] | None]:
    revision = _w.OPD_RESUME_REVISION or None
    resume = _w.hf_resume_checkpoint(fail_closed=bool(revision), revision=revision)
    if not resume:
        return None, None
    match = re.fullmatch(r"checkpoint-(\d+)", os.path.basename(resume))
    if match is None:
        raise RuntimeError(f"invalid OpenRLHF OPD resume checkpoint path {resume!r}")
    step = int(match.group(1))
    with open(os.path.join(resume, "opd_state.json"), encoding="utf-8") as state_file:
        state = validate_opd_resume_state_metadata(
            json.load(state_file),
            expected_seed=int(seed),
            checkpoint_step=step,
        )
    if state["prompt_pool_fingerprint"] != prompt_pool_fingerprint:
        raise RuntimeError("OpenRLHF OPD resume prompt pool does not match the current run")
    if step > int(update_horizon):
        raise RuntimeError("OpenRLHF OPD resume checkpoint exceeds the requested update horizon")
    if not os.path.isdir(os.path.join(resume, "_actor")):
        raise RuntimeError("OpenRLHF OPD resume checkpoint has no native DeepSpeed actor state")
    return resume, state


def _reconcile_required_deployable(
    resume_dir: str | None,
    resume_state: dict[str, Any] | None,
    save_at_steps: tuple[int, ...],
) -> None:
    if not resume_dir or resume_state is None:
        return
    step = int(resume_state["opt_steps"])
    if step not in save_at_steps or _deployable_adapter_on_hf(step):
        return
    _w.publish_deployable_checkpoint(
        os.path.join(resume_dir, "_adapter_export"),
        step,
        required=True,
        _provenance_ready=True,
    )


def _write_scheduled_dataset(
    path: str,
    prompts: list[_PromptRecord],
    *,
    steps: int,
    prompts_per_step: int,
) -> int:
    if not prompts:
        raise ValueError("cannot schedule an empty OpenRLHF OPD prompt dataset")
    scheduled_count = int(steps) * int(prompts_per_step)
    if scheduled_count <= 0:
        raise ValueError("OpenRLHF OPD prompt schedule must contain at least one row")
    with open(path, "w", encoding="utf-8") as dataset_file:
        for ordinal in range(scheduled_count):
            global_step = ordinal // prompts_per_step
            example_index = ordinal % len(prompts)
            identity = {
                "global_step": global_step,
                "example_index": example_index,
                "rollout_ordinal": 0,
                "no_signal_attempt": 0,
            }
            dataset_file.write(
                json.dumps(
                    {
                        "input": prompts[example_index].rendered,
                        "label": json.dumps(identity, sort_keys=True, separators=(",", ":")),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    return scheduled_count


def _generation_eos_from_cached_config(
    model_id: str, model_revision: str, tokenizer
) -> frozenset[int]:
    from transformers import AutoConfig, GenerationConfig

    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
        revision=model_revision or None,
        local_files_only=True,
    )
    generation_config = None
    with contextlib.suppress(OSError):
        generation_config = GenerationConfig.from_pretrained(
            model_id,
            revision=model_revision or None,
            local_files_only=True,
        )
    model_like = type(
        "ModelGenerationMetadata",
        (),
        {"config": config, "generation_config": generation_config},
    )()
    return _generation_eos_ids(model_like, tokenizer)


def _warmstart_config(adapter_dir: str | None, model_id: str) -> tuple[int, int, tuple[str, ...]]:
    if not adapter_dir:
        lora = _w.make_lora(model_id)
        targets = lora.target_modules
        target_modules = (targets,) if isinstance(targets, str) else tuple(sorted(targets))
        return int(lora.r), int(lora.lora_alpha), target_modules
    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as config_file:
        config = json.load(config_file)
    rank = int(config["r"])
    alpha = int(config["lora_alpha"])
    targets = config.get("target_modules") or _OPENRLHF_TARGET_MODULES
    target_modules = (
        (targets,) if isinstance(targets, str) else tuple(str(item) for item in targets)
    )
    if rank <= 0 or alpha <= 0 or not target_modules:
        raise RuntimeError("OpenRLHF OPD warm-start adapter configuration is invalid")
    return rank, alpha, target_modules


def _openrlhf_opd_structured_gpu_verified() -> bool:
    """Whether the OpenRLHF OPD guided-decode student rollout has been GPU-verified.

    Structured/guided decoding on the OpenRLHF OPD path needs a live vLLM guided-decode engine
    (StructuredOutputsParams on the rollout sampling params, logprobs>=2 so the forced-token mask is
    observable, and the reasoning-parser EngineArg that defers the grammar past </think>) that only
    exists on a GPU. The constraint is validated up front and the forced-mask/group-drop primitives
    are shared with the TRL path and CPU-tested; this predicate flips to True only once the live
    rollout is GPU-verified in the follow-up PR. Hardcoded (no env knob) so the mode cannot route to
    a half-wired live loop.
    """
    return False


def _resolve_single_turn_inputs() -> dict[str, Any]:
    spec = _w.JOB_SPEC
    if spec is None or spec.algorithm != "opd":
        raise RuntimeError("OpenRLHF OPD requires an OPD JobSpec")
    env = _w.require_active_env()
    if getattr(env, "multi_turn", False):
        raise RuntimeError("OpenRLHF OPD multi-turn support is deferred; use the TRL backend")
    if getattr(env, "is_tool_env", False):
        raise RuntimeError("OpenRLHF OPD tool environments are unsupported; use the TRL backend")
    train_spec = spec.train
    # Validate any structured-outputs constraint up front (a corrupt payload fails loud here rather
    # than silently training unconstrained). The forced-mask/group-drop correctness primitives are
    # shared with the TRL path and CPU-tested; the live guided-decode rollout (StructuredOutputsParams
    # injection + logprobs>=2 + the reasoning-parser EngineArg) needs a GPU, so it stays fail-closed
    # behind _openrlhf_opd_structured_gpu_verified() until that path is GPU-verified.
    structured_plan = resolve_opd_structured_plan(
        train_spec.structured_outputs, thinking=bool(_w.THINKING)
    )
    if structured_plan is not None and not _openrlhf_opd_structured_gpu_verified():
        raise RuntimeError(
            "OpenRLHF OPD structured outputs are deferred pending GPU verification of the "
            "guided-decode rollout; use the TRL backend"
        )

    from flash.multimodal import record_has_images

    knobs = _resolve_opd_knobs()
    model_id = spec.model
    model_revision = spec.model_revision
    if not model_revision:
        raise ValueError("OpenRLHF OPD requires model_revision for immutable adapter provenance")
    tokenizer = _w.load_tokenizer(model_id, revision=model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train = list(env.dataset())
    if train_spec.max_examples:
        train = train[: int(train_spec.max_examples)]
    random.Random(_w.SEED).shuffle(train)
    prompt_budget = (
        knobs.max_length - knobs.max_completion
        if knobs.max_length
        else int(RECIPE.opd.max_prompt_len)
    )
    if prompt_budget <= 0:
        raise RuntimeError("OpenRLHF OPD max_context_tokens leaves no room for a prompt")

    prompts: list[_PromptRecord] = []
    scanned = 0
    with liveness_heartbeat("opd_filtering_prompts", progress=lambda: scanned):
        for example in train:
            messages = env.prompt_messages(example)
            if record_has_images(example, messages):
                raise RuntimeError(
                    "OpenRLHF OPD multimodal support is deferred; use the TRL backend"
                )
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=_w.THINKING,
            )
            prompt_ids = tuple(tokenizer(rendered, add_special_tokens=False).input_ids)
            if 0 < len(prompt_ids) <= prompt_budget:
                prompts.append(
                    _PromptRecord(messages=messages, prompt_ids=prompt_ids, rendered=rendered)
                )
            scanned += 1
    if not prompts:
        raise RuntimeError("every OpenRLHF OPD prompt exceeds the configured prompt budget")

    prompts_per_step = min(knobs.prompts_per_step, len(prompts))
    derived_steps = on_policy_steps(
        epochs=knobs.epochs,
        prompt_count=len(prompts),
        prompts_per_step=prompts_per_step,
    )
    steps = resolve_update_horizon(derived_steps, knobs.max_steps)
    validate_save_steps(knobs.save_at_steps, steps)
    return {
        "env": env,
        "tokenizer": tokenizer,
        "model_id": model_id,
        "model_revision": model_revision,
        "prompts": prompts,
        "prompt_pool_fingerprint": _prompt_pool_fingerprint(prompts),
        "prompts_per_step": prompts_per_step,
        "group_size": knobs.group_size,
        "temperature": knobs.temperature,
        "top_p": knobs.top_p,
        "kl_penalty_coef": knobs.kl_coef,
        "learning_rate": knobs.learning_rate,
        "max_completion": knobs.max_completion,
        "max_length": knobs.max_length or (prompt_budget + knobs.max_completion),
        "steps": int(steps),
        "save_every": int(knobs.save_every),
        "save_at_steps": knobs.save_at_steps,
        "gpu_count": int(spec.gpu.count),
        "seed": int(backend_seed(spec.seed)),
        "stop_sequences": tuple(str(value) for value in knobs.stop_sequences),
        "thinking_prefill": _thinking_prefill_text(tokenizer),
    }


def _resolve_opd_actor_attn(language_model_only: bool) -> str | None:
    return "eager" if language_model_only else optimal_attn_impl()


def _raise_training_failure(returncode: int, failure: tuple[str, str] | None) -> None:
    if returncode == 0:
        return
    message = failure[1] if failure is not None else "the localhost teacher bridge was unavailable"
    if returncode == _OPENRLHF_TRANSIENT_TEACHER_EXIT:
        raise _w.RetriableInfraError(f"transient teacher failure after bounded retries: {message}")
    if returncode == _OPENRLHF_PERMANENT_TEACHER_EXIT:
        raise RuntimeError(f"permanent teacher failure: {message}")
    if failure is not None:
        classification, message = failure
        if classification == "transient":
            raise _w.RetriableInfraError(
                f"transient teacher failure after bounded retries: {message}"
            )
        raise RuntimeError(f"permanent teacher failure: {message}")
    raise RuntimeError(f"openrlhf.cli.train_ppo_ray exited {returncode}; see the worker log")


def run_opd_openrlhf() -> None:
    """Run single-turn text OPD through the isolated OpenRLHF trainer."""
    from flash.engine.worker.teacher import TeacherClient

    started_at = time.time()
    _w.heartbeat("opd_start", gpu=gpu_diagnostics())
    inputs = _resolve_single_turn_inputs()
    api_key = os.environ.get("FIREWORKS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("the managed teacher api key is missing from the OPD parent worker")
    knobs = _resolve_opd_knobs()
    teacher = TeacherClient(api_key, knobs.teacher_base_url, knobs.teacher_model)

    wait_for_gpu(
        _w.JOB_SPEC.gpu.type if _w.JOB_SPEC else None,
        exact_type=_w.JOB_SPEC.gpu.exact_type if _w.JOB_SPEC else "",
    )
    setup_perf_backends()
    download_seconds = _w.prefetch_model(
        inputs["model_id"],
        revision=inputs["model_revision"],
    )
    model_path = _resolve_cached_model_snapshot(inputs["model_id"], inputs["model_revision"])
    eos_token_ids = _generation_eos_from_cached_config(
        inputs["model_id"],
        inputs["model_revision"],
        inputs["tokenizer"],
    )

    warmstart_adapter = None
    if _w.JOB_SPEC.train.init_from_adapter:
        warmstart_adapter = _download_adapter(_w.JOB_SPEC.train.init_from_adapter)
        if not warmstart_adapter:
            raise RuntimeError("OpenRLHF OPD warm-start adapter could not be materialized")
    lora_rank, lora_alpha, target_modules = _warmstart_config(
        warmstart_adapter,
        inputs["model_id"],
    )

    workdir = os.path.join("/tmp", "flash-opd-openrlhf", _w.RUN_ID, f"seed-{_w.SEED}")
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir, exist_ok=True)
    output_dir = os.path.join(workdir, "final")
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    resume_dir, resume_state = _restore_openrlhf_resume(
        prompt_pool_fingerprint=inputs["prompt_pool_fingerprint"],
        seed=int(_w.SEED),
        update_horizon=inputs["steps"],
    )
    if resume_dir:
        checkpoint_dir = resume_dir
    _reconcile_required_deployable(
        resume_dir,
        resume_state,
        inputs["save_at_steps"],
    )
    dataset_path = os.path.join(workdir, "train.jsonl")
    adapter_dir = f"/tmp/opd_seed{_w.SEED}/adapter"
    plugin_dir = write_openrlhf_opd_sitecustomize(workdir)
    scheduled_prompt_count = _write_scheduled_dataset(
        dataset_path,
        inputs["prompts"],
        steps=inputs["steps"],
        prompts_per_step=inputs["prompts_per_step"],
    )
    qwen35_language_model_only = bool(
        _w.is_vl_checkpoint(inputs["model_id"], inputs["model_revision"])
    )
    actor_attn_implementation = _resolve_opd_actor_attn(qwen35_language_model_only)
    resumed_step = int(resume_state["opt_steps"]) if resume_state else 0
    last_step = [resumed_step]
    loss_curve: list[float] = list(resume_state.get("loss_curve", [])) if resume_state else []
    coverage_curve: list[float] = (
        list(resume_state.get("coverage_curve", [])) if resume_state else []
    )
    train_wall_baseline = (
        float(resume_state.get("train_wall_seconds", 0.0)) if resume_state else 0.0
    )
    train_started_at = time.time()

    with TeacherAlignmentBridge(
        prompts=inputs["prompts"],
        tokenizer=inputs["tokenizer"],
        teacher=teacher,
        thinking_prefill=inputs["thinking_prefill"],
        eos_token_ids=eos_token_ids,
        stop_sequences=inputs["stop_sequences"],
        mutation_callback=_w.publish_opd_optimizer_start_marker,
        initial_state=resume_state,
    ) as bridge:
        config = OpenRLHFOPDConfig(
            model_path=model_path,
            dataset_path=dataset_path,
            teacher_url=bridge.url,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            model_id=inputs["model_id"],
            model_revision=inputs["model_revision"],
            max_length=inputs["max_length"],
            max_completion=inputs["max_completion"],
            prompts_per_step=inputs["prompts_per_step"],
            group_size=inputs["group_size"],
            scheduled_prompt_count=scheduled_prompt_count,
            learning_rate=inputs["learning_rate"],
            temperature=inputs["temperature"],
            top_p=inputs["top_p"],
            seed=inputs["seed"],
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=target_modules,
            kl_penalty_coef=inputs["kl_penalty_coef"],
            save_every=inputs["save_every"],
            save_at_steps=inputs["save_at_steps"],
            final_step=inputs["steps"],
            gpu_count=inputs["gpu_count"],
            actor_attn_implementation=actor_attn_implementation,
            resume=bool(resume_dir),
        )
        args = build_openrlhf_opd_args(config)
        _opd_model_info = MODELS.get(inputs["model_id"])
        _opd_mamba_block = (
            int(getattr(_opd_model_info, "mamba_block_size", 0) or 0) if _opd_model_info else 0
        )
        child_env = build_openrlhf_opd_child_env(
            plugin_dir=plugin_dir,
            max_response_length=inputs["max_completion"],
            language_model_only=qwen35_language_model_only,
            bridge_url=bridge.url,
            kl_penalty_coef=inputs["kl_penalty_coef"],
            seed=int(_w.SEED),
            stop_sequences=inputs["stop_sequences"],
            eos_token_ids=eos_token_ids,
            save_at_steps=inputs["save_at_steps"],
            final_step=inputs["steps"],
            warmstart_adapter=warmstart_adapter,
            actor_attn_implementation=actor_attn_implementation,
            rollout_max_num_seqs=opd_rollout_concurrency(
                inputs["prompts_per_step"], inputs["group_size"]
            ),
            mamba_block_size=_opd_mamba_block,
            vllm_meta_path=os.path.join(config.checkpoint_dir, "_opd_vllm_runtime.json"),
        )
        python_bin = resolve_openrlhf_python(workdir)
        _w.heartbeat("opd_train_start", gpu=gpu_diagnostics())

        step_states = _initial_checkpoint_step_states(resume_state)
        step_states_condition = threading.Condition()

        def state_for_step(step: int) -> dict[str, Any]:
            deadline = time.monotonic() + 300.0
            with step_states_condition:
                while step not in step_states:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError(
                            f"timed out waiting for OpenRLHF OPD metrics through checkpoint step {step}"
                        )
                    step_states_condition.wait(remaining)
                return dict(step_states[step])

        publisher = _OpenRLHFOPDCheckpointPublisher(
            checkpoint_dir=checkpoint_dir,
            staging_root=os.path.join(workdir, "resume-staging"),
            python_bin=python_bin,
            model_id=inputs["model_id"],
            model_revision=inputs["model_revision"],
            seed=int(_w.SEED),
            prompt_pool_fingerprint=inputs["prompt_pool_fingerprint"],
            prompts_per_step=inputs["prompts_per_step"],
            group_size=inputs["group_size"],
            save_every=inputs["save_every"],
            final_step=inputs["steps"],
            required_steps=inputs["save_at_steps"],
            state_for_step=state_for_step,
        )
        bridge.checkpoint_callback = publisher.publish

        def on_step(step: int, loss: float, coverage: float) -> None:
            step = int(step)
            if step <= last_step[0]:
                return
            if len(loss_curve) != step - 1 or len(coverage_curve) != step - 1:
                raise RuntimeError(f"OpenRLHF OPD step {step} metrics are out of order")
            loss_curve.append(float(loss))
            coverage_curve.append(float(coverage))
            last_step[0] = step
            snapshot = bridge.snapshot()
            state = _checkpoint_state(
                step=step,
                seed=int(_w.SEED),
                prompt_pool_fingerprint=inputs["prompt_pool_fingerprint"],
                prompts_per_step=inputs["prompts_per_step"],
                group_size=inputs["group_size"],
                accounting=snapshot,
                loss_curve=loss_curve,
                coverage_curve=coverage_curve,
                train_wall_seconds=train_wall_baseline + time.time() - train_started_at,
            )
            with step_states_condition:
                step_states[step] = state
                step_states_condition.notify_all()
            _w.heartbeat("opd_step", step=last_step[0])

        bridge.metrics_callback = on_step

        with liveness_heartbeat(
            "opd_openrlhf_training",
            progress=lambda: last_step[0],
            progress_step=True,
        ):
            returncode = run_openrlhf_training(
                python_bin,
                args,
                env=child_env,
                entrypoint=_OPENRLHF_ENTRYPOINT,
                cwd=workdir,
            )
        _raise_training_failure(returncode, bridge.teacher_failure)
        if last_step[0] < inputs["steps"]:
            raise RuntimeError(
                f"OpenRLHF OPD completed {last_step[0]}/{inputs['steps']} requested updates"
            )
        accounting = bridge.snapshot()

    with liveness_heartbeat(
        "opd_openrlhf_finalizing",
        progress=lambda: last_step[0],
        progress_step=True,
        keepalive=True,
    ):
        export_openrlhf_adapter(
            output_dir,
            adapter_dir,
            inputs["model_id"],
            inputs["model_revision"],
            python_bin,
        )
        inputs["tokenizer"].save_pretrained(adapter_dir)
        _w.write_base_model_provenance(
            adapter_dir,
            inputs["model_id"],
            inputs["model_revision"],
        )
        _w.hf_upload_folder(adapter_dir, "adapter", required=True)
        if final_save_due(last_step[0], inputs["save_at_steps"]):
            _w.publish_deployable_checkpoint(adapter_dir, last_step[0])

    train_wall = time.time() - started_at
    aligned = int(accounting["aligned_sequences"])
    _w.heartbeat(
        "opd_trained",
        train_wall=train_wall,
        step=last_step[0],
        gpu=gpu_diagnostics(),
    )
    # the rollout actor recorded the applied vLLM runtime (fp8 kv / chunked prefill / max_num_seqs /
    # mamba floor / enforce_eager) to a sidecar because it computed the plan in-actor where the gpu is
    # visible; fold it into train_meta so the served-checkpoint record carries the real rollout config.
    vllm_runtime_meta = None
    vllm_runtime_sidecar = os.path.join(config.checkpoint_dir, "_opd_vllm_runtime.json")
    if os.path.exists(vllm_runtime_sidecar):
        try:
            with open(vllm_runtime_sidecar, encoding="utf-8") as vllm_meta_handle:
                vllm_runtime_meta = json.load(vllm_meta_handle)
        except (OSError, ValueError):
            vllm_runtime_meta = None
    _w.write_train_meta(
        phase="opd",
        step=last_step[0],
        adapter_dir=adapter_dir,
        model_id=inputs["model_id"],
        train_wall=train_wall,
        train_tokens=0,
        generated_tokens=int(accounting["generated_tokens"]),
        notes={
            "backend": "openrlhf",
            "vllm_runtime": vllm_runtime_meta,
            "steps": last_step[0],
            "retained_prompts": len(inputs["prompts"]),
            "scheduled_prompts": scheduled_prompt_count,
            "download_seconds": download_seconds,
            "method": "gkd",
            "teacher_model": knobs.teacher_model,
            "kl_penalty_coef": inputs["kl_penalty_coef"],
            "loss_curve": loss_curve,
            "mean_coverage": float(accounting["coverage_sum"]) / aligned if aligned else 0.0,
            # real alignment-health signal (mean student-tokens-per-group); mean_coverage stays ~1.0
            # even for a degenerate collapsed alignment, so it cannot flag that failure mode. the
            # denominator is teacher_ok (samples scored), matching how group_granularity_sum accrues.
            "mean_align_granularity": (
                float(accounting["group_granularity_sum"]) / int(accounting["teacher_ok"])
                if int(accounting["teacher_ok"])
                else 0.0
            ),
            "truncated_rollouts": int(accounting["truncated_rollouts"]),
            "teacher_input_tokens": int(accounting["teacher_input_tokens"]),
            "teacher_ok": int(accounting["teacher_ok"]),
            "teacher_transient": int(accounting["teacher_transient"]),
            "teacher_error": int(accounting["teacher_error"]),
            "empty_alignments": int(accounting["empty_alignments"]),
            "no_signal_resamples": int(accounting["no_signal_resamples"]),
            "temperature": inputs["temperature"],
            "group_size": inputs["group_size"],
            "prompts_per_step": inputs["prompts_per_step"],
            "max_completion_len": inputs["max_completion"],
            "warm_started": bool(warmstart_adapter),
            "prompt_pool_fingerprint": inputs["prompt_pool_fingerprint"],
        },
    )
