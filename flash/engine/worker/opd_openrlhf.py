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

from flash.engine.recipe import RECIPE
from flash.engine.steps import on_policy_steps, resolve_update_horizon, validate_save_steps
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.adapter import _download_adapter
from flash.engine.worker.grpo_openrlhf import (
    _OPENRLHF_ENTRYPOINT,
    _resolve_cached_model_snapshot,
    build_openrlhf_child_env,
)
from flash.engine.worker.grpo_openrlhf import (
    _sitecustomize_source as _grpo_sitecustomize_source,
)
from flash.engine.worker.heartbeat import liveness_heartbeat
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
from flash.engine.worker.perf import gpu_diagnostics, setup_perf_backends, wait_for_gpu
from flash.engine.worker.rng import backend_seed
from flash.engine.worker.teacher import TeacherError
from flash.engine.worker.tokenizer_align import groupwise_alignment, groupwise_coverage
from flash.opd_retry_contract import OPD_RESUME_STATE_VERSION, validate_opd_resume_state_metadata

_OPENRLHF_OPD_MAX_BODY_BYTES = 4 * 1024 * 1024
_OPENRLHF_TARGET_MODULES = "all-linear"
_OPENRLHF_PPO_EPOCHS = 1
_OPENRLHF_NO_SIGNAL_ATTEMPTS = 3
_SITE_CUSTOMIZE_NAME = "sitecustomize.py"
_STEP_RE = re.compile(r"(?i)global[_ ]step[:=\s]+(\d+)")
_LOSS_RE = re.compile(r"(?:act_loss|policy_loss)['\"\s:=]+(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.I)


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
    gpu_count: int
    qwen35_language_model_only: bool
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
        self.checkpoint_callback = checkpoint_callback or (lambda _step: None)
        self.token = token or secrets.token_urlsafe(32)
        self._path = f"/teacher/{self.token}"
        self._score_lock = threading.Lock()
        self._mutation_lock = threading.Lock()
        self._mutation_notified = False
        self._stats_lock = threading.Lock()
        self._failure: tuple[str, str] | None = None
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
                    classification = (
                        "transient"
                        if isinstance(exc, TeacherError) and not exc.permanent
                        else "permanent"
                    )
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

    def score_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("teacher request must be an object")
        if payload == {"mutation": True}:
            with self._mutation_lock:
                if not self._mutation_notified:
                    self.mutation_callback()
                    self._mutation_notified = True
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
        if not _rollout_terminated(
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
        with self._score_lock:
            teacher_tokens = self.teacher.score(teacher_prompt, completion_text)
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
        with self._stats_lock:
            self._stats["teacher_ok"] += 1
            self._stats["teacher_input_tokens"] += prompt_length + len(student_ids)
            self._stats["coverage_sum"] += coverage
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
        "--train.full_determinism_enable",
        "--algo.advantage.estimator",
        "reinforce",
        "--algo.advantage.gamma",
        "1.0",
        "--algo.kl.init_coef",
        "0.0",
        "--actor.adam.lr",
        str(config.learning_rate),
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
        "--vllm.enforce_eager",
        "--vllm.enable_sleep",
        "--ds.enable_sleep",
        "--train.colocate_all",
        "--ckpt.output_dir",
        config.output_dir,
        "--ckpt.path",
        config.checkpoint_dir,
        "--ckpt.save_hf",
        "--ckpt.save_steps",
        str(config.save_every),
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
    if config.qwen35_language_model_only:
        args.extend(["--ds.attn_implementation", "eager"])
    return args


def _opd_sitecustomize_extension() -> str:
    return r"""
import asyncio
import copy
import hashlib
import json
import urllib.error
import urllib.request

import torch

_FLASH_OPD_BRIDGE_URL = os.environ["FLASH_OPENRLHF_OPD_BRIDGE_URL"]
_FLASH_OPD_KL_COEF = float(os.environ["FLASH_OPENRLHF_OPD_KL_COEF"])
_FLASH_OPD_SEED = int(os.environ["FLASH_OPENRLHF_OPD_SEED"])
_FLASH_OPD_STOPS = json.loads(os.environ.get("FLASH_OPENRLHF_OPD_STOP_SEQUENCES", "[]"))
_FLASH_OPD_EOS_IDS = json.loads(os.environ.get("FLASH_OPENRLHF_OPD_EOS_TOKEN_IDS", "[]"))
_FLASH_OPD_MAX_ATTEMPTS = int(os.environ.get("FLASH_OPENRLHF_OPD_NO_SIGNAL_ATTEMPTS", "3"))

if _FLASH_OPD_KL_COEF <= 0:
    raise RuntimeError("FLASH_OPENRLHF_OPD_KL_COEF must be positive")
if _FLASH_OPD_MAX_ATTEMPTS <= 0:
    raise RuntimeError("FLASH_OPENRLHF_OPD_NO_SIGNAL_ATTEMPTS must be positive")


def _flash_identity(label):
    identity = json.loads(label) if isinstance(label, str) else dict(label)
    required = ("global_step", "example_index", "rollout_ordinal")
    if any(isinstance(identity.get(name), bool) or not isinstance(identity.get(name), int) for name in required):
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


def _flash_post_teacher(payload):
    request = urllib.request.Request(
        _FLASH_OPD_BRIDGE_URL,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        try:
            detail = json.loads(error.read().decode("utf-8"))["error"]
            classification = str(detail["classification"])
            message = str(detail["message"])
        except Exception as decode_error:
            raise RuntimeError(f"flash OPD bridge returned unclassified HTTP {error.code}") from decode_error
        raise RuntimeError(f"flash OPD teacher {classification} failure: {message}") from error
    except Exception as error:
        raise RuntimeError(f"flash OPD teacher transient bridge failure: {type(error).__name__}") from error
    result = json.loads(body.decode("utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError("flash OPD bridge returned a non-object response")
    return result


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
        result = await asyncio.to_thread(
            _flash_post_teacher,
            {
                "label": attempt_label,
                "prompt_length": start,
                "sequence_ids": output["observation_tokens"][:end],
            },
        )
        if int(result.get("signal_count", 0)) > 0:
            output["label"] = attempt_label
            output["reward"] = 0.0
            output["scores"] = 0.0
            output["teacher_group_ids"] = result["teacher_group_ids"]
            output["teacher_logsums"] = result["teacher_logsums"]
            output["teacher_signal_mask"] = result["teacher_signal_mask"]
            output["teacher_coverage"] = float(result.get("coverage", 0.0))
            return output
    raise RuntimeError(
        f"flash OPD produced no aligned teacher signal after {_FLASH_OPD_MAX_ATTEMPTS} rollout attempts"
    )


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
    if not bool(signal_mask.any().item()):
        raise RuntimeError("OpenRLHF OPD accepted a response without aligned teacher signal")
    experience.info["teacher_coverage"] = torch.tensor([float(response.get("teacher_coverage", 0.0))])
    return experience


_FlashSamplesGenerator._process_response_into_experience = _flash_opd_process_response

from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker as _FlashExperienceMaker


def _flash_make_experience_batch(self, rollout_samples):
    return self.split_rollout_samples(rollout_samples)


_FlashExperienceMaker.make_experience_batch = _flash_make_experience_batch

from openrlhf.trainer.ppo_trainer import BasePPOTrainer as _FlashBasePPOTrainer

_original_save_logs_and_checkpoints = _FlashBasePPOTrainer.save_logs_and_checkpoints


@functools.wraps(_original_save_logs_and_checkpoints)
def _flash_save_logs_and_checkpoints(self, global_step, logs_dict=None, client_states=None):
    result = _original_save_logs_and_checkpoints(
        self,
        global_step,
        logs_dict=logs_dict,
        client_states=client_states,
    )
    save_steps = self.args.ckpt.save_steps
    if save_steps != float("inf") and global_step % save_steps == 0:
        _flash_post_teacher({"checkpoint": int(global_step)})
    return result


_FlashBasePPOTrainer.save_logs_and_checkpoints = _flash_save_logs_and_checkpoints


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


def _flash_opd_training_step(self, experience, kl_ctl, step, loss_batch_info=None):
    del kl_ctl, loss_batch_info
    self.actor.train()
    action_log_probs, _output = self.actor(
        experience.sequences,
        experience.action_mask,
        attention_mask=experience.attention_mask,
        return_output=True,
        ring_attn_group=self.strategy.ring_attn_group,
        packed_seq_lens=None,
        return_entropy=False,
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
    loss, selected = _flash_reverse_kl(
        action_log_probs,
        teacher_logsums,
        group_ids,
        experience.action_mask.bool() & signal_mask,
    )
    self.strategy.backward(loss, self.actor, self.actor_optim)
    _flash_post_teacher({"mutation": True})
    self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
    grad_norm = self.strategy.get_grad_norm(self.actor)
    coverage = experience.info.get("teacher_coverage", [])
    if isinstance(coverage, list):
        coverage = torch.stack(coverage).float().mean() if coverage else loss.detach().new_zeros(())
    elif isinstance(coverage, torch.Tensor):
        coverage = coverage.float().mean()
    else:
        coverage = loss.detach().new_zeros(())
    return {
        "metrics": {
            "policy_loss": loss.detach(),
            "distillation_loss": loss.detach(),
            "teacher_coverage": coverage.detach(),
            "actor_lr": self.actor_scheduler.get_last_lr()[0],
            "actor_grad_norm": grad_norm,
        },
        "weights": {
            "policy_loss": "sample",
            "distillation_loss": "sample",
            "teacher_coverage": "sample",
            "actor_lr": None,
            "actor_grad_norm": None,
        },
        "num_samples": float(action_log_probs.shape[0]),
        "num_action_tokens": float(selected.sum().item()),
    }


_ActorPPOTrainer.training_step = _flash_opd_training_step

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
        if getattr(result, "unexpected_keys", None):
            raise RuntimeError(
                f"OpenRLHF OPD warm-start adapter has unexpected keys: {result.unexpected_keys}"
            )
        loaded_b = [
            parameter
            for name, parameter in self.model.named_parameters()
            if "lora_B" in name and parameter.requires_grad
        ]
        if not loaded_b or not any(bool(torch.count_nonzero(parameter).item()) for parameter in loaded_b):
            raise RuntimeError("OpenRLHF OPD warm-start adapter did not load a nonzero LoRA B delta")

    _Actor.__init__ = _flash_actor_init_with_warmstart

print("[flash-openrlhf] flash opd reverse-kl, teacher, deterministic rollout, and lora hooks active", flush=True)
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
    warmstart_adapter: str | None = None,
) -> dict[str, str]:
    """Build the isolated child env with bridge capability but no provider credential."""
    child = build_openrlhf_child_env(
        plugin_dir=plugin_dir,
        max_response_length=max_response_length,
        language_model_only=language_model_only,
    )
    child.update(
        {
            "FLASH_OPENRLHF_OPD_BRIDGE_URL": bridge_url,
            "FLASH_OPENRLHF_OPD_KL_COEF": str(float(kl_penalty_coef)),
            "FLASH_OPENRLHF_OPD_SEED": str(int(seed)),
            "FLASH_OPENRLHF_OPD_STOP_SEQUENCES": json.dumps(list(stop_sequences)),
            "FLASH_OPENRLHF_OPD_EOS_TOKEN_IDS": json.dumps(sorted(eos_token_ids)),
            "FLASH_OPENRLHF_OPD_NO_SIGNAL_ATTEMPTS": str(_OPENRLHF_NO_SIGNAL_ATTEMPTS),
        }
    )
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
        shutil.rmtree(adapter_export)
        optimizer = _find_checkpoint_state_file(actor_tag, "optim")
        rng_state = _find_checkpoint_state_file(actor_tag, "model_states")
        shutil.copy2(optimizer, os.path.join(stage, "optimizer.pt"))
        shutil.copy2(rng_state, os.path.join(stage, "rng_state.pth"))
        state = self.state_for_step(step)
        with open(os.path.join(stage, "opd_state.json"), "w", encoding="utf-8") as state_file:
            json.dump(state, state_file, sort_keys=True)
        return stage

    def _publish(self, step: int) -> None:
        actor_tag, hf_export = self._wait_for_checkpoint(step)
        stage = self._stage(step, actor_tag, hf_export)

        def publish_required() -> None:
            if step in self.required_steps:
                _w.publish_deployable_checkpoint(
                    stage,
                    step,
                    required=True,
                    _provenance_ready=True,
                )

        uploaded = _w.upload_resume_checkpoint(
            step,
            stage,
            before_upload=publish_required,
        )
        if step in self.required_steps and not uploaded:
            raise RuntimeError(f"required OpenRLHF OPD checkpoint step {step} was not published")


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
    if train_spec.structured_outputs:
        raise RuntimeError("OpenRLHF OPD structured outputs are deferred; use the TRL backend")

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
    for example in train:
        messages = env.prompt_messages(example)
        if record_has_images(example, messages):
            raise RuntimeError("OpenRLHF OPD multimodal support is deferred; use the TRL backend")
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
        "save_every": math.gcd(knobs.save_every, steps, *knobs.save_at_steps),
        "save_at_steps": knobs.save_at_steps,
        "gpu_count": int(spec.gpu.count),
        "seed": int(backend_seed(spec.seed)),
        "stop_sequences": tuple(str(value) for value in knobs.stop_sequences),
        "thinking_prefill": _thinking_prefill_text(tokenizer),
    }


def _raise_training_failure(returncode: int, failure: tuple[str, str] | None) -> None:
    if returncode == 0:
        return
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
            gpu_count=inputs["gpu_count"],
            qwen35_language_model_only=qwen35_language_model_only,
            resume=bool(resume_dir),
        )
        args = build_openrlhf_opd_args(config)
        child_env = build_openrlhf_opd_child_env(
            plugin_dir=plugin_dir,
            max_response_length=inputs["max_completion"],
            language_model_only=qwen35_language_model_only,
            bridge_url=bridge.url,
            kl_penalty_coef=inputs["kl_penalty_coef"],
            seed=int(_w.SEED),
            stop_sequences=inputs["stop_sequences"],
            eos_token_ids=eos_token_ids,
            warmstart_adapter=warmstart_adapter,
        )
        python_bin = resolve_openrlhf_python(workdir)
        _w.heartbeat("opd_train_start", gpu=gpu_diagnostics())

        step_states: dict[int, dict[str, Any]] = {}
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
            required_steps=inputs["save_at_steps"],
            state_for_step=state_for_step,
        )
        bridge.checkpoint_callback = publisher.publish

        def on_step(step: int) -> None:
            step = int(step)
            if step <= last_step[0]:
                return
            last_step[0] = step
            snapshot = bridge.snapshot()
            aligned = int(snapshot["aligned_sequences"])
            coverage = float(snapshot["coverage_sum"]) / aligned if aligned else 0.0
            if len(coverage_curve) == step - 1:
                coverage_curve.append(coverage)
            if len(loss_curve) < step or len(coverage_curve) < step:
                raise RuntimeError(f"OpenRLHF OPD step {step} log is missing loss or coverage")
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

        def on_line(line: str) -> None:
            step_match = _STEP_RE.search(line)
            loss_match = _LOSS_RE.search(line)
            if step_match and loss_match:
                step = int(step_match.group(1))
                value = float(loss_match.group(1))
                if step == len(loss_curve) + 1:
                    loss_curve.append(value)

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
                on_step=on_step,
                on_line=on_line,
                heartbeat=lambda: _w.heartbeat(
                    "opd_openrlhf_training",
                    step=last_step[0],
                    gpu=gpu_diagnostics(),
                ),
                step_pattern=r"(?i)global[_ ]step[:=\s]+(\d+)",
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
        _w.publish_deployable_checkpoint(adapter_dir, last_step[0])

    train_wall = time.time() - started_at
    aligned = int(accounting["aligned_sequences"])
    _w.heartbeat(
        "opd_trained",
        train_wall=train_wall,
        step=last_step[0],
        gpu=gpu_diagnostics(),
    )
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
            "steps": last_step[0],
            "retained_prompts": len(inputs["prompts"]),
            "scheduled_prompts": scheduled_prompt_count,
            "download_seconds": download_seconds,
            "method": "gkd",
            "teacher_model": knobs.teacher_model,
            "kl_penalty_coef": inputs["kl_penalty_coef"],
            "loss_curve": loss_curve,
            "mean_coverage": float(accounting["coverage_sum"]) / aligned if aligned else 0.0,
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
