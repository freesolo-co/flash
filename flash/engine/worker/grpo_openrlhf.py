"""OpenRLHF-backed GRPO foundation for single-turn text environments.

The OpenRLHF runtime stays out of process. The parent worker owns Flash environment state and
serves rewards over an authenticated localhost bridge. A generated ``sitecustomize`` module keeps
reward handling fail-closed, applies Flash's combined fixed-length DR-GRPO normalization, masks
truncated completions, and synchronizes effective merged LoRA weights to the rollout engines.

This foundation intentionally leaves the TRL implementation intact and rejects unsupported modes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import secrets
import shutil
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from flash.engine.recipe import RECIPE
from flash.engine.steps import (
    final_save_due,
    on_policy_steps,
    resolve_update_horizon,
    validate_save_steps,
)
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.heartbeat import liveness_heartbeat
from flash.engine.worker.hf import _deployable_adapter_on_hf
from flash.engine.worker.openrlhf_common import (
    export_openrlhf_adapter,
    resolve_openrlhf_python,
    run_openrlhf_training,
)
from flash.engine.worker.perf import (
    gpu_diagnostics,
    grpo_use_reentrant,
    optimal_attn_impl,
    setup_perf_backends,
    wait_for_gpu,
)
from flash.engine.worker.rng import backend_seed, seed_training_rngs
from flash.engine.worker.rollout_samples import select_rollout_samples
from flash.spec import DEFAULT_CREDIT_ASSIGNMENT

_OPENRLHF_ENTRYPOINT = "openrlhf.cli.train_ppo_ray"
_OPENRLHF_REWARD_MAX_BODY_BYTES = 2 * 1024 * 1024
_FLASH_REWARD_RECORD_LOG_KEY = "__flash_reward_record_id"
_OPENRLHF_TARGET_MODULES = "all-linear"
_OPENRLHF_PPO_EPOCHS = 1
_OPENRLHF_REWARD_CLIP_BOUND = "1000000000"
_OPENRLHF_ACTION_LOGPROB_CHUNK_SIZE = 256
_SITE_CUSTOMIZE_NAME = "sitecustomize.py"
_CHILD_ENV_EXACT = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "TZ",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "TOKENIZERS_PARALLELISM",
    }
)
_CHILD_ENV_PREFIXES = (
    "CUDA_",
    "NCCL_",
    "TORCH_",
    "PYTORCH_",
    "RAY_",
    "VLLM_",
    "OMP_",
    "MKL_",
    "OPENBLAS_",
    "LC_",
)

# openrlhf's multi-turn executor exposes action spans but collapses all turn rewards into one episode
# scalar before advantage estimation. exact per-turn credit therefore needs an upstream experience and
# advantage-data-model extension; keep that mode fail-closed until the backend can carry turn rewards.


class _RewardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


@dataclass(frozen=True)
class RewardResult:
    reward: float
    scores: float
    extra_logs: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenRLHFGRPOConfig:
    model_path: str
    dataset_path: str
    reward_url: str
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
    kl_coef: float
    save_every: int
    save_at_steps: tuple[int, ...]
    gpu_count: int
    qwen35_language_model_only: bool
    fp8_kv: bool
    warmstart_adapter: str
    resume: bool
    actor_attn_implementation: str | None


def deterministic_rollout_seed(
    flash_seed: int,
    global_step: int,
    example_index: int,
    rollout_ordinal: int,
    *,
    turn_ordinal: int = 0,
    retry_ordinal: int = 0,
) -> int:
    """Derive one stable nonnegative 63-bit seed from the complete rollout identity."""
    turn = int(turn_ordinal)
    retry = int(retry_ordinal)
    if turn < 0 or retry < 0:
        raise ValueError("flash GRPO turn and retry ordinals must be nonnegative")
    payload = (
        f"{int(flash_seed)}:{int(global_step)}:{int(example_index)}:{int(rollout_ordinal)}:{turn}"
    )
    if retry:
        payload += f":retry:{retry}"
    digest = hashlib.blake2b(payload.encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def build_openrlhf_grpo_args(config: OpenRLHFGRPOConfig) -> list[str]:
    """Map the resolved Flash GRPO job to ``openrlhf.cli.train_ppo_ray`` arguments."""
    if config.group_size <= 1:
        raise ValueError("OpenRLHF DR-GRPO requires group_size greater than 1")
    if config.prompts_per_step <= 0 or config.scheduled_prompt_count <= 0:
        raise ValueError("OpenRLHF GRPO requires a nonempty scheduled prompt dataset")
    if config.max_completion <= 0 or config.max_length <= config.max_completion:
        raise ValueError("OpenRLHF max length must leave room for a prompt")
    if config.lora_rank <= 0 or config.lora_alpha <= 0:
        raise ValueError("OpenRLHF GRPO requires positive LoRA rank and alpha")
    if config.gpu_count <= 0:
        raise ValueError("OpenRLHF GRPO requires at least one GPU")

    completion_batch = config.prompts_per_step * config.group_size
    args = [
        "--actor.model_name_or_path",
        config.model_path,
        "--reward.remote_url",
        config.reward_url,
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
        "dr_grpo",
        "--algo.advantage.gamma",
        "1.0",
        "--algo.advantage.is_correction_enable",
        "--algo.advantage.is_correction_threshold",
        "0.0",
        "2.0",
        "--algo.advantage.is_correction_type",
        "tis",
        "--reward.clip_range",
        f"-{_OPENRLHF_REWARD_CLIP_BOUND}",
        _OPENRLHF_REWARD_CLIP_BOUND,
        "--actor.adam.lr",
        str(config.learning_rate),
        "--actor.adam.betas",
        "0.9",
        "0.999",
        "--actor.lr_scheduler",
        "constant",
        "--actor.lr_warmup_ratio",
        "0.0",
        "--actor.min_lr_ratio",
        "1.0",
        "--ds.lora.rank",
        str(config.lora_rank),
        "--ds.lora.alpha",
        str(config.lora_alpha),
        "--ds.lora.target_modules",
        _OPENRLHF_TARGET_MODULES,
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
    if grpo_use_reentrant(config.model_id):
        args.append("--actor.gradient_checkpointing_reentrant")
    if config.resume:
        args.append("--ckpt.load_enable")
    if config.kl_coef > 0:
        args.extend(
            [
                "--algo.kl.init_coef",
                str(config.kl_coef),
                "--algo.kl.use_loss",
                "--algo.kl.estimator",
                "k2",
                "--ref.num_nodes",
                "1",
                "--ref.num_gpus_per_node",
                str(config.gpu_count),
            ]
        )
    else:
        args.extend(["--algo.kl.init_coef", "0.0"])
    if config.actor_attn_implementation:
        args.extend(["--ds.attn_implementation", config.actor_attn_implementation])
    return args


def dr_grpo_fixed_length_normalize(
    per_token_losses: list[list[float]],
    action_masks: list[list[float]],
    max_response_length: int,
) -> float:
    """Pure numeric reference for TRL's DR-GRPO fixed-length reduction."""
    if max_response_length <= 0:
        raise ValueError("max_response_length must be positive")
    if not per_token_losses or len(per_token_losses) != len(action_masks):
        raise ValueError("loss and mask batches must be nonempty and aligned")
    numerator = 0.0
    for losses, mask in zip(per_token_losses, action_masks, strict=True):
        if len(losses) != len(mask):
            raise ValueError("loss and mask rows must be aligned")
        numerator += sum(float(loss) * float(keep) for loss, keep in zip(losses, mask, strict=True))
    return numerator / (len(per_token_losses) * int(max_response_length))


def tis_weighted_dr_grpo_normalize(
    per_token_losses: list[list[float]],
    action_masks: list[list[float]],
    old_minus_rollout_log_probs: list[list[float]],
    max_response_length: int,
    *,
    c_max: float = 2.0,
) -> float:
    """Pure numeric reference for TRL token-truncate TIS plus DR-GRPO reduction."""
    if c_max <= 0:
        raise ValueError("c_max must be positive")
    if len(per_token_losses) != len(old_minus_rollout_log_probs):
        raise ValueError("loss and importance-ratio batches must be aligned")
    weighted: list[list[float]] = []
    for losses, log_ratios in zip(per_token_losses, old_minus_rollout_log_probs, strict=True):
        if len(losses) != len(log_ratios):
            raise ValueError("loss and importance-ratio rows must be aligned")
        weighted.append(
            [
                float(loss) * min(math.exp(float(log_ratio)), float(c_max))
                for loss, log_ratio in zip(losses, log_ratios, strict=True)
            ]
        )
    return dr_grpo_fixed_length_normalize(weighted, action_masks, max_response_length)


def _completion_from_openrlhf_query(query: str, prompt: str) -> str:
    if not isinstance(query, str) or not isinstance(prompt, str):
        raise TypeError("OpenRLHF reward query and prompt must be strings")
    if not query.startswith(prompt):
        raise ValueError("OpenRLHF reward query does not start with its rendered prompt")
    return query[len(prompt) :]


def completion_from_tokenizer_query(tokenizer, query: str, prompt: str) -> str:
    """Remove the prompt after reproducing OpenRLHF's tokenize-then-decode canonicalization."""
    if not isinstance(query, str) or not isinstance(prompt, str):
        raise TypeError("OpenRLHF reward query and prompt must be strings")
    prompt_ids = tokenizer(text=prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][
        0
    ]
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    canonical_prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    return _completion_from_openrlhf_query(query, canonical_prompt)


def _named_metrics(breakdown: Any) -> dict[str, float]:
    if not isinstance(breakdown, dict):
        return {}
    metrics: dict[str, float] = {}
    for name, value in breakdown.items():
        if name == "total":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            metrics[str(name)] = number
    return metrics


def score_single_turn(
    env,
    completion: str,
    example,
    *,
    tokenizer,
    thinking: bool,
    prompt_opened_thinking: bool,
    think_penalty: float,
) -> RewardResult:
    """Score one text completion with the same environment semantics as the TRL path."""
    try:
        graded = _w.graded_text(completion, prompt_opened_thinking=prompt_opened_thinking)
        state = (
            {
                "raw": completion,
                "completion": graded,
                "thinking": _w.thinking_text(
                    completion, prompt_opened_thinking=prompt_opened_thinking
                ),
            }
            if thinking
            else None
        )
        breakdown = None
        if hasattr(env, "scores_breakdown"):
            breakdown = env.scores_breakdown(graded, example, state)
            if not isinstance(breakdown, dict):
                raise TypeError("environment scores_breakdown must return a mapping")
            reward = float(breakdown.get("total", 0.0))
        else:
            reward = float(env.reward(graded, example, state))
    except Exception as exc:
        print(
            f"[rl-openrlhf] env scoring raised ({type(exc).__name__}: {exc}); scoring 0.0",
            flush=True,
        )
        return RewardResult(0.0, 0.0, {})

    if think_penalty > 0 and thinking:
        reward -= think_penalty * _w.think_token_count(
            completion,
            tokenizer,
            prompt_opened_thinking=prompt_opened_thinking,
        )
    if not math.isfinite(reward):
        raise ValueError("environment reward must be finite")
    return RewardResult(reward, reward, _named_metrics(breakdown))


class RewardBridge:
    """Authenticated localhost bridge from OpenRLHF rollout actors to the live Flash env."""

    def __init__(
        self,
        score_by_label: Callable[[int, str, str], RewardResult],
        *,
        samples_per_step: int,
        first_step: int,
        completion_from_query: Callable[[str, str], str] = _completion_from_openrlhf_query,
        token: str | None = None,
    ) -> None:
        if int(samples_per_step) <= 0:
            raise ValueError("OpenRLHF reward samples per step must be positive")
        if int(first_step) <= 0:
            raise ValueError("OpenRLHF first reward step must be positive")
        self._score_by_label = score_by_label
        self._completion_from_query = completion_from_query
        self._token = token or secrets.token_urlsafe(32)
        self._path = f"/reward/{self._token}"
        self._record_path = f"{self._path}/record"
        self._score_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._rewards: list[float] = []
        self._samples: dict[int, list[tuple[str, str, float]]] = {}
        self._sample_record_ids: dict[int, list[int]] = {}
        self._pending_records: dict[int, tuple[int, int]] = {}
        self._next_record_id = 0
        self._samples_per_step = int(samples_per_step)
        self._sample_step = int(first_step)
        self._samples_in_step = 0
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
                if not any(
                    secrets.compare_digest(self.path, path)
                    for path in (bridge._path, bridge._record_path)
                ):
                    self._send(401, {"error": "unauthorized"})
                    return
                try:
                    raw_length = self.headers.get("Content-Length")
                    if raw_length is None:
                        raise ValueError("missing content length")
                    length = int(raw_length)
                    if length <= 0 or length > _OPENRLHF_REWARD_MAX_BODY_BYTES:
                        raise ValueError("invalid reward request size")
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    if secrets.compare_digest(self.path, bridge._record_path):
                        bridge._finalize_record(payload)
                        self._send(200, {"ok": True})
                        return
                    result = bridge._score_payload(payload)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    self._send(400, {"error": "invalid reward request"})
                    return
                except Exception as exc:
                    print(
                        f"[rl-openrlhf] reward bridge failed ({type(exc).__name__}: {exc})",
                        flush=True,
                    )
                    self._send(500, {"error": "reward scoring failed"})
                    return
                self._send(
                    200,
                    {
                        "rewards": [result.reward],
                        "scores": [result.scores],
                        "extra_logs": {name: [value] for name, value in result.extra_logs.items()},
                    },
                )

        self._server = _RewardHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="openrlhf-reward-bridge",
            daemon=True,
        )
        self._thread.start()
        port = int(self._server.server_address[1])
        self.url = f"http://127.0.0.1:{port}{self._path}"
        self.record_url = f"http://127.0.0.1:{port}{self._record_path}"

    def _score_payload(self, payload: Any) -> RewardResult:
        if not isinstance(payload, dict):
            raise TypeError("reward request must be an object")
        queries = payload["query"]
        prompts = payload["prompts"]
        labels = payload["labels"]
        if not all(isinstance(items, list) for items in (queries, prompts, labels)):
            raise TypeError("reward request fields must be lists")
        if not (len(queries) == len(prompts) == len(labels) == 1):
            raise ValueError("OpenRLHF reward bridge accepts exactly one completion per request")
        completion = self._completion_from_query(queries[0], prompts[0])
        if isinstance(labels[0], bool):
            raise TypeError("reward label must be an integer")
        label = int(labels[0])
        with self._score_lock:
            result = self._score_by_label(label, completion, prompts[0])
        if not isinstance(result, RewardResult):
            raise TypeError("reward scorer must return RewardResult")
        if not all(math.isfinite(value) for value in (result.reward, result.scores)):
            raise ValueError("reward response must be finite")
        if not all(math.isfinite(float(value)) for value in result.extra_logs.values()):
            raise ValueError("reward metrics must be finite")
        with self._stats_lock:
            record_id = self._next_record_id
            self._next_record_id += 1
            reward_index = len(self._rewards)
            self._rewards.append(float(result.reward))
            sample_step = self._sample_step
            samples = self._samples.setdefault(sample_step, [])
            sample_record_ids = self._sample_record_ids.setdefault(sample_step, [])
            samples.append((prompts[0], completion, float(result.reward)))
            sample_record_ids.append(record_id)
            del samples[:-64]
            del sample_record_ids[:-64]
            self._pending_records[record_id] = (reward_index, sample_step)
            self._samples_in_step += 1
            if self._samples_in_step == self._samples_per_step:
                self._sample_step += 1
                self._samples_in_step = 0
        return RewardResult(
            result.reward,
            result.scores,
            {**result.extra_logs, _FLASH_REWARD_RECORD_LOG_KEY: float(record_id)},
        )

    def _finalize_record(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            raise TypeError("reward record request must be an object")
        record_id = payload.get("record_id")
        neutralize = payload.get("neutralize")
        if isinstance(record_id, bool) or not isinstance(record_id, int):
            raise TypeError("reward record id must be an integer")
        if not isinstance(neutralize, bool):
            raise TypeError("reward neutralization flag must be boolean")
        with self._stats_lock:
            try:
                reward_index, sample_step = self._pending_records.pop(record_id)
            except KeyError as exc:
                raise ValueError("unknown or already finalized reward record") from exc
            if not neutralize:
                return
            self._rewards[reward_index] = 0.0
            sample_record_ids = self._sample_record_ids.get(sample_step, [])
            if record_id not in sample_record_ids:
                return
            sample_index = sample_record_ids.index(record_id)
            prompt, completion, _reward = self._samples[sample_step][sample_index]
            self._samples[sample_step][sample_index] = (prompt, completion, 0.0)

    @property
    def rewards(self) -> list[float]:
        with self._stats_lock:
            return list(self._rewards)

    def drain_sampled_completions(self, generated_at_step: int) -> list[dict]:
        generated_at_step = int(generated_at_step)
        with self._stats_lock:
            samples = self._samples.pop(generated_at_step, [])
            self._sample_record_ids.pop(generated_at_step, None)
        return select_rollout_samples(samples, generated_at_step=generated_at_step)

    @property
    def call_count(self) -> int:
        with self._stats_lock:
            return len(self._rewards)

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    def __enter__(self) -> RewardBridge:
        return self

    def __exit__(self, *_exc_info) -> None:
        self.shutdown()


def post_reward_request(url: str, payload: dict[str, Any], *, timeout: float = 5.0) -> dict:
    """Small stdlib client used by CPU bridge tests."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _sitecustomize_source() -> str:
    """Return the self-contained child hook loaded before OpenRLHF imports."""
    return r"""
import contextlib
import contextvars
import copy
import functools
import hashlib
import inspect
import json
import math
import os
import threading
import time
import urllib.request

import torch
from torch.utils.checkpoint import checkpoint

_MAX_RESPONSE_LENGTH = int(os.environ["FLASH_OPENRLHF_MAX_RESPONSE_LENGTH"])
_ACTION_LOGPROB_CHUNK_SIZE = 256
_FLASH_ROLLOUT_SEED_TEXT = os.environ.get("FLASH_OPENRLHF_ROLLOUT_SEED")
_FLASH_ROLLOUT_SEED = int(_FLASH_ROLLOUT_SEED_TEXT) if _FLASH_ROLLOUT_SEED_TEXT is not None else None
_FLASH_STOP_SEQUENCES = tuple(json.loads(os.environ.get("FLASH_OPENRLHF_STOP_SEQUENCES", "[]")))
_FLASH_EOS_TOKEN_IDS = frozenset(
    int(token_id) for token_id in json.loads(os.environ.get("FLASH_OPENRLHF_EOS_TOKEN_IDS", "[]"))
)
_FLASH_REWARD_RECORD_URL = os.environ.get("FLASH_OPENRLHF_REWARD_RECORD_URL", "")
_FLASH_REWARD_RECORD_LOG_KEY = "__flash_reward_record_id"
_ATTN_IMPLEMENTATION = os.environ.get("FLASH_OPENRLHF_ATTN_IMPLEMENTATION")
_LANGUAGE_MODEL_ONLY = os.environ.get("FLASH_OPENRLHF_LANGUAGE_MODEL_ONLY") == "1"
_SM120_VLLM_BACKEND = os.environ.get("FLASH_OPENRLHF_SM120_VLLM_BACKEND") == "1"
if _MAX_RESPONSE_LENGTH <= 0:
    raise RuntimeError("FLASH_OPENRLHF_MAX_RESPONSE_LENGTH must be positive")
_WARMSTART_ADAPTER = os.environ.get("FLASH_OPENRLHF_WARMSTART_ADAPTER", "")
_FP8_KV = os.environ.get("FLASH_OPENRLHF_FP8_KV") == "1"
_EXACT_SAVE_STEPS = frozenset(
    int(value)
    for value in os.environ.get("FLASH_OPENRLHF_SAVE_AT_STEPS", "").split(",")
    if value
)

from openrlhf import models as _models_module
from openrlhf.models import loss as _loss_module

_original_aggregate_loss = _loss_module.aggregate_loss
_fixed_dr_grpo_loss = contextvars.ContextVar("flash_openrlhf_fixed_dr_grpo_loss", default=False)


@functools.wraps(_original_aggregate_loss)
def _flash_aggregate_loss(
    loss,
    loss_mask,
    token_level_loss=True,
    dp_size=1,
    batch_num_tokens=None,
    global_batch_size=None,
):
    if _fixed_dr_grpo_loss.get() and token_level_loss:
        denominator = loss.shape[0] * _MAX_RESPONSE_LENGTH
        if denominator <= 0:
            raise RuntimeError("OpenRLHF DR-GRPO received an empty fixed-length loss batch")
        return (loss * loss_mask).sum() / denominator
    return _original_aggregate_loss(
        loss,
        loss_mask,
        token_level_loss=token_level_loss,
        dp_size=dp_size,
        batch_num_tokens=batch_num_tokens,
        global_batch_size=global_batch_size,
    )


# policy loss resolves aggregate_loss through the loss module, while kl and entropy import the
# package-level symbol into ppo_actor. patch both bindings so every per-token term uses the same
# local_batch * configured_max_response_length denominator before the scalar terms are summed.
_loss_module.aggregate_loss = _flash_aggregate_loss
_models_module.aggregate_loss = _flash_aggregate_loss

_original_policy_loss_init = _loss_module.PolicyLoss.__init__


@functools.wraps(_original_policy_loss_init)
def _flash_policy_loss_init(self, *args, **kwargs):
    kwargs["enable_vllm_is_correction"] = True
    kwargs["vllm_is_truncated_threshold"] = [0.0, 2.0]
    kwargs["vllm_is_correction_type"] = "tis"
    return _original_policy_loss_init(self, *args, **kwargs)


_loss_module.PolicyLoss.__init__ = _flash_policy_loss_init

from openrlhf.trainer.ray import ppo_actor as _ppo_actor_module

_ActorPPOTrainer = _ppo_actor_module.ActorPPOTrainer
_ppo_actor_module.aggregate_loss = _flash_aggregate_loss
_original_training_step = _ActorPPOTrainer.training_step


def _flash_attention_context():
    if _ATTN_IMPLEMENTATION != "sdpa":
        return contextlib.nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(
            [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ],
            set_priority=True,
        )
    except Exception as exc:
        print(f"[flash-openrlhf] cuDNN SDPA unavailable, using default SDPA: {exc}", flush=True)
        return contextlib.nullcontext()


@functools.wraps(_original_training_step)
def _flash_training_step(self, *args, **kwargs):
    token = _fixed_dr_grpo_loss.set(True)
    try:
        with _flash_attention_context():
            return _original_training_step(self, *args, **kwargs)
    finally:
        _fixed_dr_grpo_loss.reset(token)


_ActorPPOTrainer.training_step = _flash_training_step

_original_deepspeed_initialize = _ppo_actor_module.deepspeed.initialize


@functools.wraps(_original_deepspeed_initialize)
def _flash_deepspeed_initialize(*args, **kwargs):
    config = kwargs.get("config")
    optimizer_config = config.get("optimizer") if isinstance(config, dict) else None
    if not isinstance(optimizer_config, dict) or optimizer_config.get("type") != "AdamW":
        return _original_deepspeed_initialize(*args, **kwargs)
    if kwargs.get("optimizer") is not None:
        raise RuntimeError("flash OpenRLHF GRPO expected DeepSpeed to construct no optimizer")
    model_parameters = kwargs.get("model_parameters")
    if model_parameters is None:
        raise RuntimeError("flash OpenRLHF GRPO received no actor parameters")

    import bitsandbytes as bnb

    params = optimizer_config.get("params") or {}
    optimizer = bnb.optim.PagedAdamW8bit(
        model_parameters,
        lr=float(params["lr"]),
        betas=tuple(float(value) for value in params["betas"]),
        eps=float(params["eps"]),
        weight_decay=float(params["weight_decay"]),
    )
    config = dict(config)
    config.pop("optimizer", None)
    config["zero_allow_untested_optimizer"] = True
    kwargs["config"] = config
    kwargs["optimizer"] = optimizer
    kwargs["model_parameters"] = None
    return _original_deepspeed_initialize(*args, **kwargs)


_ppo_actor_module.deepspeed.initialize = _flash_deepspeed_initialize

from openrlhf.trainer.ppo_utils.samples_generator import SamplesGenerator as _SamplesGenerator

_original_process_response = _SamplesGenerator._process_response_into_experience


@functools.wraps(_original_process_response)
def _flash_process_response(self, response, **generate_kwargs):
    experience = _original_process_response(self, response, **generate_kwargs)
    if bool(response.get("truncated", False)):
        # retain the sample row but remove every action token from policy and kl numerators. the
        # fixed dr-grpo denominator still counts the row through loss.shape[0].
        experience.action_mask.zero_()
    return experience


_SamplesGenerator._process_response_into_experience = _flash_process_response

from openrlhf.utils.agent import SingleTurnAgentExecutor as _SingleTurnAgentExecutor

_original_execute = _SingleTurnAgentExecutor.execute


def _flash_rollout_identity(label):
    identity = json.loads(label) if isinstance(label, str) else dict(label)
    required = ("global_step", "example_index", "rollout_ordinal", "turn", "retry")
    if set(identity) != set(required) or any(
        isinstance(identity.get(name), bool)
        or not isinstance(identity.get(name), int)
        or identity.get(name) < 0
        for name in required
    ):
        raise RuntimeError("flash GRPO rollout identity is invalid")
    return identity


def _flash_rollout_seed(identity):
    payload = (
        f"{_FLASH_ROLLOUT_SEED}:{identity['global_step']}:{identity['example_index']}:"
        f"{identity['rollout_ordinal']}:{identity['turn']}"
    )
    if identity["retry"]:
        payload += f":retry:{identity['retry']}"
    digest = hashlib.blake2b(payload.encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def _flash_trim_trailing_stop(tokenizer, token_ids, raw_text, output_text, stop_reason=None):
    ids = [int(token_id) for token_id in token_ids]
    if isinstance(stop_reason, str) and stop_reason in _FLASH_STOP_SEQUENCES:
        stop = stop_reason
    else:
        candidates = [
            value for value in _FLASH_STOP_SEQUENCES if value and output_text.endswith(value)
        ]
        if not candidates:
            return ids, None
        stop = max(candidates, key=len)
    keep_length = raw_text.rfind(stop)
    if keep_length < 0:
        return ids, None
    kept = len(ids)
    while kept > 0 and len(tokenizer.decode(ids[:kept], skip_special_tokens=False)) > keep_length:
        kept -= 1
    return ids[:kept], tokenizer.decode(ids[:kept], skip_special_tokens=True)


def _flash_finalize_reward_record(output, neutralize):
    extra_logs = output.get("extra_logs")
    if not isinstance(extra_logs, dict):
        if _FLASH_REWARD_RECORD_URL:
            raise RuntimeError("OpenRLHF reward output is missing record metadata")
        return
    record_id = extra_logs.pop(_FLASH_REWARD_RECORD_LOG_KEY, None)
    if isinstance(record_id, list):
        if len(record_id) != 1:
            raise RuntimeError("OpenRLHF reward record metadata is invalid")
        record_id = record_id[0]
    if not _FLASH_REWARD_RECORD_URL:
        return
    if (
        isinstance(record_id, bool)
        or not isinstance(record_id, (int, float))
        or not float(record_id).is_integer()
    ):
        raise RuntimeError("OpenRLHF reward output is missing record metadata")
    body = json.dumps(
        {"record_id": int(record_id), "neutralize": bool(neutralize)},
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        _FLASH_REWARD_RECORD_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        if response.status != 200:
            raise RuntimeError("OpenRLHF reward record finalization failed")


def _flash_naturally_terminated(finish_reason, stop_reason, token_ids, stop_text):
    if finish_reason == "stop":
        return True
    if (
        isinstance(stop_reason, int)
        and not isinstance(stop_reason, bool)
        and stop_reason in _FLASH_EOS_TOKEN_IDS
    ):
        return True
    if _FLASH_EOS_TOKEN_IDS and not _FLASH_EOS_TOKEN_IDS.isdisjoint(token_ids):
        return True
    return any(
        value
        and (
            stop_text.endswith(value)
            or (isinstance(stop_reason, str) and stop_reason == value)
        )
        for value in _FLASH_STOP_SEQUENCES
    )


class _FlashTerminationCapture:
    def __init__(self, engine, tokenizer):
        self._engine = engine
        self._tokenizer = tokenizer
        self.finish_reason = None
        self.stop_reason = None
        self.naturally_terminated = False
        self.empty_after_stop = False

    def __getattr__(self, name):
        return getattr(self._engine, name)

    async def generate(self, *args, **kwargs):
        request_output = await self._engine.generate(*args, **kwargs)
        generation_output = request_output.outputs[0]
        token_ids = [int(token_id) for token_id in generation_output.token_ids]
        stop_text = self._tokenizer.decode(token_ids, skip_special_tokens=False)
        self.finish_reason = generation_output.finish_reason
        self.stop_reason = getattr(generation_output, "stop_reason", None)
        self.naturally_terminated = _flash_naturally_terminated(
            self.finish_reason,
            self.stop_reason,
            token_ids,
            stop_text,
        )
        kept_ids, kept_text = _flash_trim_trailing_stop(
            self._tokenizer,
            token_ids,
            stop_text,
            generation_output.text,
            self.stop_reason,
        )
        if kept_text is not None:
            generation_output.token_ids = kept_ids
            generation_output.text = kept_text
            if generation_output.logprobs is not None:
                generation_output.logprobs = generation_output.logprobs[: len(kept_ids)]
            self.empty_after_stop = not kept_ids
        return request_output


@functools.wraps(_original_execute)
async def _flash_execute(self, *args, **kwargs):
    call_args = list(args)
    call_kwargs = dict(kwargs)
    sampling_params = call_args[2] if len(call_args) > 2 else call_kwargs.get("sampling_params")
    if sampling_params is None:
        raise RuntimeError("flash GRPO rollout request is missing sampling parameters")
    sampling_params = copy.deepcopy(sampling_params)
    if _FLASH_STOP_SEQUENCES:
        sampling_params.stop = list(_FLASH_STOP_SEQUENCES)
        sampling_params.include_stop_str_in_output = True
    if _FLASH_EOS_TOKEN_IDS:
        sampling_params.stop_token_ids = sorted(_FLASH_EOS_TOKEN_IDS)
        sampling_params.all_stop_token_ids = set(
            getattr(sampling_params, "all_stop_token_ids", ())
        ) | set(_FLASH_EOS_TOKEN_IDS)
    if _FLASH_ROLLOUT_SEED is not None:
        label = call_args[1] if len(call_args) > 1 else call_kwargs.get("label")
        identity = _flash_rollout_identity(label)
        rollout_key = (
            identity["global_step"],
            identity["example_index"],
            identity["turn"],
            identity["retry"],
        )
        rollout_ordinals = getattr(self, "_flash_grpo_rollout_ordinals", None)
        if rollout_ordinals is None:
            rollout_ordinals = {}
            self._flash_grpo_rollout_ordinals = rollout_ordinals
        identity["rollout_ordinal"] = rollout_ordinals.get(rollout_key, 0)
        rollout_ordinals[rollout_key] = identity["rollout_ordinal"] + 1
        sampling_params.seed = _flash_rollout_seed(identity)
        if len(call_args) > 1:
            call_args[1] = identity["example_index"]
        else:
            call_kwargs["label"] = identity["example_index"]
    if len(call_args) > 2:
        call_args[2] = sampling_params
    else:
        call_kwargs["sampling_params"] = sampling_params
    tokenizer = call_args[4] if len(call_args) > 4 else call_kwargs.get("hf_tokenizer")
    llm_engine = call_args[5] if len(call_args) > 5 else call_kwargs.get("llm_engine")
    if tokenizer is None or llm_engine is None:
        raise RuntimeError("flash GRPO rollout request is missing tokenizer or engine")
    capture = _FlashTerminationCapture(llm_engine, tokenizer)
    if len(call_args) > 5:
        call_args[5] = capture
    else:
        call_kwargs["llm_engine"] = capture
    output = await _original_execute(self, *call_args, **call_kwargs)
    if not isinstance(output, dict):
        raise RuntimeError("OpenRLHF reward executor returned a non-object output")
    _flash_finalize_reward_record(output, capture.empty_after_stop)
    output["finish_reason"] = capture.finish_reason
    output["stop_reason"] = capture.stop_reason
    output["truncated"] = capture.empty_after_stop or (
        bool(output.get("truncated", False)) and not capture.naturally_terminated
    )
    if capture.empty_after_stop:
        output["reward"] = 0.0
        output["scores"] = 0.0
        output["extra_logs"] = {}
    for key in ("reward", "scores"):
        value = output.get(key)
        if isinstance(value, list):
            if len(value) != 1:
                raise RuntimeError(f"OpenRLHF reward executor returned invalid {key}")
            value = value[0]
            output[key] = value
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"OpenRLHF reward executor returned invalid {key}")
    extra_logs = output.get("extra_logs")
    if not isinstance(extra_logs, dict):
        raise RuntimeError("OpenRLHF reward executor returned invalid extra_logs")
    for name, value in extra_logs.items():
        if not isinstance(name, str):
            raise RuntimeError("OpenRLHF reward metric names must be strings")
        if isinstance(value, list):
            if len(value) != 1:
                raise RuntimeError(f"OpenRLHF reward metric {name!r} is invalid")
            value = value[0]
            extra_logs[name] = value
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"OpenRLHF reward metric {name!r} is invalid")
    return output


_SingleTurnAgentExecutor.execute = _flash_execute

from openrlhf.models.actor import Actor as _Actor

_original_actor_init = _Actor.__init__
_original_actor_forward = _Actor.forward


def _flash_chunked_action_log_probs(
    prediction_hidden_states,
    labels,
    action_mask,
    projection,
    temperature,
    chunk_size=_ACTION_LOGPROB_CHUNK_SIZE,
    return_entropy=False,
):
    if prediction_hidden_states.ndim != 3:
        raise RuntimeError("chunked action projection expected rank-3 hidden states")
    if labels.shape != prediction_hidden_states.shape[:2] or action_mask.shape != labels.shape:
        raise RuntimeError("chunked action projection received misaligned labels or mask")
    if int(chunk_size) <= 0:
        raise RuntimeError("chunked action projection requires a positive chunk size")
    if float(temperature) <= 0:
        raise RuntimeError("chunked action projection requires a positive temperature")

    flat_hidden = prediction_hidden_states.reshape(-1, prediction_hidden_states.shape[-1])
    flat_labels = labels.reshape(-1)
    def project_chunk(chunk_hidden, chunk_labels):
        logits = projection(chunk_hidden).to(torch.float32)
        if logits.ndim != 2 or logits.shape[0] != chunk_hidden.shape[0]:
            raise RuntimeError("chunked action projection returned invalid logits")
        entropy = None
        if return_entropy:
            entropy_logsumexp = torch.logsumexp(logits, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
            entropy = entropy_logsumexp - (probabilities * logits).sum(dim=-1)
        if float(temperature) != 1.0:
            logits = logits / float(temperature)
        logsumexp = torch.logsumexp(logits, dim=-1)
        selected = logits.gather(-1, chunk_labels[:, None]).squeeze(-1)
        log_probs = selected - logsumexp
        return (log_probs, entropy) if return_entropy else log_probs

    log_prob_chunks = []
    entropy_chunks = []
    for start in range(0, flat_hidden.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), flat_hidden.shape[0])
        chunk_hidden = flat_hidden[start:end]
        chunk_labels = flat_labels[start:end]
        if torch.is_grad_enabled() and chunk_hidden.requires_grad:
            result = checkpoint(
                project_chunk,
                chunk_hidden,
                chunk_labels,
                use_reentrant=False,
            )
        else:
            result = project_chunk(chunk_hidden, chunk_labels)
        if return_entropy:
            log_probs, entropy = result
            entropy_chunks.append(entropy)
        else:
            log_probs = result
        log_prob_chunks.append(log_probs)

    shape = labels.shape
    action_log_probs = torch.cat(log_prob_chunks).view(shape) * action_mask.to(torch.float32)
    entropy = torch.cat(entropy_chunks).view(shape) if return_entropy else None
    return action_log_probs, entropy


def _flash_actor_forward(
    self,
    sequences,
    action_mask=None,
    attention_mask=None,
    return_output=False,
    allgather_logits=False,
    return_logprobs=False,
    ring_attn_group=None,
    packed_seq_lens=None,
    return_entropy=False,
    **mm_inputs,
):
    if action_mask is None:
        return _original_actor_forward(
            self,
            sequences,
            action_mask=action_mask,
            attention_mask=attention_mask,
            return_output=return_output,
            allgather_logits=allgather_logits,
            return_logprobs=return_logprobs,
            ring_attn_group=ring_attn_group,
            packed_seq_lens=packed_seq_lens,
            return_entropy=return_entropy,
            **mm_inputs,
        )
    if self.packing_samples or ring_attn_group is not None:
        raise RuntimeError("chunked action projection requires ring_attn_size=1 without packing")
    if allgather_logits or return_logprobs:
        raise RuntimeError("chunked action projection supports action log probabilities only")
    if sequences.ndim != 2 or action_mask.ndim != 2:
        raise RuntimeError("chunked action projection expected batched sequences and action masks")
    action_width = action_mask.shape[1]
    if action_width <= 0 or action_width >= sequences.shape[1]:
        raise RuntimeError("chunked action projection received an invalid action width")

    if getattr(self, "is_vlm", False):
        position_ids = None
        if mm_inputs:
            cfg = self._vlm_config
            token_type_ids = (sequences == cfg.image_token_id).to(torch.int32)
            if getattr(cfg, "video_token_id", None) is not None:
                token_type_ids[sequences == cfg.video_token_id] = 2
            key = "mm_token_type_ids" if "image_grid_thw" in mm_inputs else "token_type_ids"
            mm_inputs[key] = token_type_ids
    else:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

    output_head = self.model.get_output_embeddings()
    if output_head is None or not callable(getattr(output_head, "forward", None)):
        raise RuntimeError("chunked action projection requires an accessible output head")
    original_output_head_forward = output_head.forward
    captured = {}

    def chunked_output_head(hidden_states, *head_args, **head_kwargs):
        if head_args or head_kwargs:
            raise RuntimeError("chunked action projection does not support output-head arguments")
        if captured:
            raise RuntimeError("chunked action projection output head was invoked more than once")
        prediction_hidden = hidden_states[:, -action_width - 1 : -1, :]
        labels = sequences[:, -action_width:]

        def projection(chunk_hidden):
            if output_head.forward is chunked_output_head:
                return original_output_head_forward(chunk_hidden)
            return output_head(chunk_hidden)

        action_log_probs, entropy = _flash_chunked_action_log_probs(
            prediction_hidden,
            labels,
            action_mask,
            projection,
            self.temperature,
            return_entropy=return_entropy,
        )
        captured["action_log_probs"] = action_log_probs
        captured["entropy"] = entropy
        return action_log_probs

    output_head.forward = chunked_output_head
    try:
        output = self.model(
            sequences,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **mm_inputs,
        )
    finally:
        output_head.forward = original_output_head_forward
    if "action_log_probs" not in captured:
        raise RuntimeError("chunked action projection output head was not invoked")
    output["logits"] = None
    if return_entropy:
        setattr(output, "entropy", captured["entropy"])
    action_log_probs = captured["action_log_probs"]
    return (action_log_probs, output) if return_output else action_log_probs


_Actor.forward = _flash_actor_forward

_loading_warm_reference = contextvars.ContextVar(
    "flash_openrlhf_loading_warm_reference", default=False
)
_peft_load_lock = threading.Lock()


def _flash_assert_adapter_loaded(model, load_result):
    missing = [
        key for key in (getattr(load_result, "missing_keys", []) or []) if "lora_" in key
    ]
    unexpected = [
        key for key in (getattr(load_result, "unexpected_keys", []) or []) if "lora_" in key
    ]
    if missing or unexpected:
        raise RuntimeError(
            f"OpenRLHF warm-start adapter load was incomplete: missing={missing}, unexpected={unexpected}"
        )
    lora_b_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith("lora_B.default.weight")
    ]
    if not lora_b_parameters:
        raise RuntimeError("OpenRLHF warm-start adapter contains no default LoRA parameters")
    if not any(parameter.detach().count_nonzero().item() for parameter in lora_b_parameters):
        raise RuntimeError("OpenRLHF warm-start adapter has an all-zero LoRA delta")


@functools.wraps(_original_actor_init)
def _flash_actor_init(self, *args, **kwargs):
    if kwargs.get("target_modules") == ["all-linear"]:
        kwargs["target_modules"] = "all-linear"
    load_warm = bool(_WARMSTART_ADAPTER) and (
        int(kwargs.get("lora_rank", 0) or 0) > 0 or _loading_warm_reference.get()
    )
    if not load_warm:
        return _original_actor_init(self, *args, **kwargs)

    kwargs["lora_rank"] = 0
    result = _original_actor_init(self, *args, **kwargs)
    from peft import PeftModel

    is_reference = _loading_warm_reference.get()
    base_model = self.model
    enable_input_require_grads = getattr(base_model, "enable_input_require_grads", None)
    if callable(enable_input_require_grads):
        enable_input_require_grads()
    load_results = []
    with _peft_load_lock:
        original_load_adapter = PeftModel.load_adapter

        @functools.wraps(original_load_adapter)
        def capture_load_result(model, *load_args, **load_kwargs):
            load_result = original_load_adapter(model, *load_args, **load_kwargs)
            load_results.append(load_result)
            return load_result

        PeftModel.load_adapter = capture_load_result
        try:
            self.model = PeftModel.from_pretrained(
                base_model,
                _WARMSTART_ADAPTER,
                adapter_name="default",
                is_trainable=not is_reference,
                key_mapping=getattr(base_model, "_checkpoint_conversion_mapping", None),
            )
        finally:
            PeftModel.load_adapter = original_load_adapter
    if len(load_results) != 1:
        raise RuntimeError(
            f"OpenRLHF warm-start adapter loaded {len(load_results)} times; expected exactly once"
        )
    _flash_assert_adapter_loaded(self.model, load_results[0])
    if is_reference:
        self.model.requires_grad_(False)
        self.model.eval()
    return result


_Actor.__init__ = _flash_actor_init

def _flash_ray_modified_class(actor_class):
    metadata = getattr(actor_class, "__ray_metadata__", None)
    return getattr(metadata, "modified_class", actor_class)


from openrlhf.trainer.ray.launcher import ReferenceModelActor as _ReferenceModelActor

_ReferenceModelActorImpl = _flash_ray_modified_class(_ReferenceModelActor)
_original_reference_init = _ReferenceModelActorImpl.init_model_from_pretrained


@functools.wraps(_original_reference_init)
def _flash_reference_init(self, *args, **kwargs):
    token = _loading_warm_reference.set(bool(_WARMSTART_ADAPTER))
    try:
        return _original_reference_init(self, *args, **kwargs)
    finally:
        _loading_warm_reference.reset(token)


_ReferenceModelActorImpl.init_model_from_pretrained = _flash_reference_init

from openrlhf.trainer.ppo_trainer import PPOTrainer as _PPOTrainer

_PPOTrainerImpl = _flash_ray_modified_class(_PPOTrainer)
_original_ppo_fit = _PPOTrainerImpl.fit
_original_save_logs_and_checkpoints = _PPOTrainerImpl.save_logs_and_checkpoints


@functools.wraps(_original_ppo_fit)
def _flash_ppo_fit(self, *args, **kwargs):
    if _WARMSTART_ADAPTER:
        # vllm starts from the catalog base; synchronize the incoming adapter before rollout one.
        self.broadcast_to_vllm()
    return _original_ppo_fit(self, *args, **kwargs)


@functools.wraps(_original_save_logs_and_checkpoints)
def _flash_save_logs_and_checkpoints(self, global_step, *args, **kwargs):
    if not _EXACT_SAVE_STEPS:
        return _original_save_logs_and_checkpoints(self, global_step, *args, **kwargs)
    original_save_steps = self.args.ckpt.save_steps
    self.args.ckpt.save_steps = int(global_step) if int(global_step) in _EXACT_SAVE_STEPS else float("inf")
    try:
        return _original_save_logs_and_checkpoints(self, global_step, *args, **kwargs)
    finally:
        self.args.ckpt.save_steps = original_save_steps


_PPOTrainerImpl.fit = _flash_ppo_fit
_PPOTrainerImpl.save_logs_and_checkpoints = _flash_save_logs_and_checkpoints

_PolicyModelActor = _ppo_actor_module.PolicyModelActor
_PolicyModelActorImpl = _flash_ray_modified_class(_PolicyModelActor)
_original_actor_save_checkpoint = _PolicyModelActorImpl.save_checkpoint


@functools.wraps(_original_actor_save_checkpoint)
def _flash_actor_save_checkpoint(self, tag, *args, **kwargs):
    result = _original_actor_save_checkpoint(self, tag, *args, **kwargs)
    tag = str(tag)
    if not tag.startswith("global_step"):
        return result
    step_text = tag.removeprefix("global_step")
    if not step_text.isdigit():
        raise RuntimeError(f"OpenRLHF emitted an invalid checkpoint tag: {tag}")
    checkpoint_dir = os.path.abspath(self.strategy.args.ckpt.path)
    ack_path = os.path.join(checkpoint_dir, f".flash-uploaded-{tag}")
    if self.strategy.is_rank_0():
        marker = {
            "step": int(step_text),
            "tag": tag,
            "checkpoint_dir": checkpoint_dir,
            "adapter_dir": os.path.join(checkpoint_dir, f"{tag}_hf"),
            "ack_path": ack_path,
        }
        print("[flash-openrlhf-checkpoint] " + json.dumps(marker, sort_keys=True), flush=True)
    deadline = time.monotonic() + 1800.0
    while not os.path.isfile(ack_path):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"timed out publishing OpenRLHF checkpoint {tag}")
        time.sleep(0.1)
    return result


_PolicyModelActorImpl.save_checkpoint = _flash_actor_save_checkpoint

_original_broadcast_to_vllm = _ActorPPOTrainer.broadcast_to_vllm
_LORA_PARAMETER_SEGMENTS = frozenset(
    {"lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "lora_magnitude_vector"}
)
_NON_LM_PARAMETER_SEGMENTS = frozenset(
    {"visual", "vision_tower", "multi_modal_projector", "mtp"}
)


def _flash_vllm_base_name(name):
    parts = name.split(".")
    if any(part in _LORA_PARAMETER_SEGMENTS for part in parts):
        raise RuntimeError(f"LoRA parameter name leaked into vLLM sync: {name}")
    return ".".join(part for part in parts if part != "base_layer")


def _flash_lora_sync_entries(model):
    get_base_model = getattr(model, "get_base_model", None)
    if not callable(get_base_model):
        return None
    base_model = get_base_model()
    lora_modules = {
        name: module
        for name, module in base_model.named_modules()
        if hasattr(module, "base_layer")
        and hasattr(module, "lora_A")
        and hasattr(module, "lora_B")
        and hasattr(module, "get_delta_weight")
    }
    if not lora_modules:
        return None

    entries = []
    seen = set()
    for raw_name, parameter in base_model.named_parameters():
        parts = raw_name.split(".")
        if any(part in _LORA_PARAMETER_SEGMENTS for part in parts):
            continue
        if _LANGUAGE_MODEL_ONLY and any(
            part in _NON_LM_PARAMETER_SEGMENTS for part in parts
        ):
            continue
        name = _flash_vllm_base_name(raw_name)
        if name in seen:
            raise RuntimeError(f"duplicate merged vLLM weight name: {name}")
        seen.add(name)

        module = None
        adapters = []
        marker = ".base_layer.weight"
        if raw_name == "base_layer.weight":
            module_name = ""
        elif raw_name.endswith(marker):
            module_name = raw_name[: -len(marker)]
        else:
            module_name = None
        if module_name is not None:
            module = lora_modules.get(module_name)
            if module is None:
                raise RuntimeError(f"could not resolve LoRA module for {raw_name}")
            adapters = [
                adapter
                for adapter in module.active_adapters
                if adapter in module.lora_A and adapter in module.lora_B
            ]
            for adapter in adapters:
                if adapter in getattr(module, "lora_variant", {}):
                    raise RuntimeError("OpenRLHF vLLM sync supports vanilla LoRA only")
                if getattr(module, "lora_bias", {}).get(adapter, False):
                    raise RuntimeError("OpenRLHF vLLM sync does not support LoRA bias")

        sources = [parameter]
        for adapter in adapters:
            sources.extend(
                [
                    module.lora_A[adapter].weight,
                    module.lora_B[adapter].weight,
                ]
            )
        entries.append((name, parameter, module, adapters, sources))
    return entries


def _flash_materialize_sync_weight(parameter, module, adapters):
    if not adapters:
        return parameter
    merged = parameter.data
    for adapter in adapters:
        delta = module.get_delta_weight(adapter).to(device=parameter.device, dtype=parameter.dtype)
        merged = merged + delta
    return merged.contiguous()


def _flash_broadcast_to_vllm(self):
    model = self.actor.model.module
    entries = _flash_lora_sync_entries(model)
    if entries is None:
        return _original_broadcast_to_vllm(self)

    # correctness-first sync: materialize base + scaling * b @ a one layer at a time and send only
    # standard hf base-model names. this adds one rank-r gemm and one full layer temporary per adapted
    # weight, but keeps the existing full-weight broadcaster and avoids unsupported peft names in vllm.
    torch = _ppo_actor_module.torch
    deepspeed = _ppo_actor_module.deepspeed
    ray = _ppo_actor_module.ray
    barrier = _ppo_actor_module.torch_dist_barrier_and_cuda_sync
    use_prefix_cache = getattr(self.strategy.args.vllm, "enable_prefix_caching", False)
    cache_reset_refs = []
    if use_prefix_cache and torch.distributed.get_rank() == 0:
        cache_reset_refs = [engine.reset_prefix_cache.remote() for engine in self.vllm_engines]

    torch.cuda.empty_cache()
    use_ray = getattr(self.strategy.args.vllm, "sync_with_ray", False)

    def gather_params_ctx(parameters):
        if self.strategy.args.ds.tensor_parallel_size > 1:
            return deepspeed.module_inject.layers.GatherReplacedLayerParams(
                parameters, model, enabled=True
            )
        return deepspeed.zero.GatheredParameters(
            parameters, enabled=self.strategy.args.ds.zero_stage == 3
        )

    def broadcast_weight(name, weight, count, num_params):
        if torch.distributed.get_rank() == 0:
            refs = [
                engine.update_weight.remote(
                    name,
                    dtype=weight.dtype,
                    shape=weight.shape,
                    empty_cache=count == num_params,
                )
                for engine in self.vllm_engines
            ]
            if use_ray:
                import ray.util.collective as collective

                collective.broadcast(weight.data, 0, group_name=self._model_update_group)
            else:
                self._model_update_group.broadcast(
                    weight.data, src=0, stream=torch.cuda.current_stream()
                )
            ray.get(refs)

    def broadcast_weight_cuda_ipc(name, weight, count, num_params):
        from torch.multiprocessing.reductions import reduce_tensor

        ipc_weight = weight.data.clone()
        ipc_handle = {_ppo_actor_module.get_physical_gpu_id(): reduce_tensor(ipc_weight)}
        ipc_handle_list = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)
        if torch.distributed.get_rank() == 0:
            ipc_handles = {}
            for handle in ipc_handle_list:
                ipc_handles.update(handle)
            refs = [
                engine.update_weight_cuda_ipc.remote(
                    name,
                    dtype=weight.dtype,
                    shape=weight.shape,
                    ipc_handles=ipc_handles,
                    empty_cache=count == num_params,
                )
                for engine in self.vllm_engines
            ]
            ray.get(refs)
        barrier()

    sync = broadcast_weight_cuda_ipc if self.use_cuda_ipc else broadcast_weight
    num_params = len(entries)
    for count, (name, parameter, module, adapters, sources) in enumerate(entries, start=1):
        with gather_params_ctx(sources):
            weight = _flash_materialize_sync_weight(parameter, module, adapters)
            sync(name, weight, count, num_params)

    if cache_reset_refs:
        ray.get(cache_reset_refs)
    torch.cuda.empty_cache()
    barrier()


_ActorPPOTrainer.broadcast_to_vllm = _flash_broadcast_to_vllm

if _LANGUAGE_MODEL_ONLY or _FP8_KV or _SM120_VLLM_BACKEND:
    import vllm

    _engine_arg_defaults = {}
    if _LANGUAGE_MODEL_ONLY:
        _engine_arg_defaults["language_model_only"] = True
    if _FP8_KV:
        _engine_arg_defaults["kv_cache_dtype"] = "fp8"
    if _SM120_VLLM_BACKEND:
        try:
            import flashinfer  # noqa: F401

            _engine_arg_defaults["attention_backend"] = "FLASHINFER"
        except Exception as exc:
            _engine_arg_defaults["attention_backend"] = "TRITON_ATTN"
            print(
                f"[flash-openrlhf] sm120 flashinfer import failed ({exc}); "
                "using attention_backend=TRITON_ATTN",
                flush=True,
            )

    def _flash_patch_engine_args(engine_args_type, type_name):
        original_init = engine_args_type.__init__
        parameters = inspect.signature(original_init).parameters
        missing = [name for name in _engine_arg_defaults if name not in parameters]
        if missing:
            raise RuntimeError(f"the pinned vLLM lacks {type_name}.{missing[0]}")

        @functools.wraps(original_init)
        def flash_engine_init(self, *args, **kwargs):
            for name, value in _engine_arg_defaults.items():
                kwargs.setdefault(name, value)
            return original_init(self, *args, **kwargs)

        engine_args_type.__init__ = flash_engine_init

    _flash_patch_engine_args(vllm.AsyncEngineArgs, "AsyncEngineArgs")
    _flash_patch_engine_args(vllm.EngineArgs, "EngineArgs")

print(
    "[flash-openrlhf] flash grpo chunked action logprobs, tis, loss, truncation, reward, "
    "warm-start, checkpoint, and lora sync hooks active; 32k gpu validation pending",
    flush=True,
)
""".lstrip()


def write_openrlhf_sitecustomize(directory: str) -> str:
    plugin_dir = os.path.join(directory, "flash_openrlhf_plugin")
    os.makedirs(plugin_dir, exist_ok=True)
    Path(plugin_dir, _SITE_CUSTOMIZE_NAME).write_text(_sitecustomize_source(), encoding="utf-8")
    return plugin_dir


def build_openrlhf_child_env(
    *,
    plugin_dir: str,
    max_response_length: int,
    language_model_only: bool,
    sm120_vllm_backend: bool,
    seed: int | None = None,
    stop_sequences: tuple[str, ...] = (),
    eos_token_ids: frozenset[int] = frozenset(),
    fp8_kv: bool = False,
    warmstart_adapter: str = "",
    save_at_steps: tuple[int, ...] = (),
    actor_attn_implementation: str | None = None,
    reward_record_url: str = "",
) -> dict[str, str]:
    """Build a minimal child environment without Flash environment or provider secrets."""
    child = {
        key: value
        for key, value in os.environ.items()
        if key in _CHILD_ENV_EXACT or key.startswith(_CHILD_ENV_PREFIXES)
    }
    child["PYTHONPATH"] = plugin_dir
    child["FLASH_OPENRLHF_MAX_RESPONSE_LENGTH"] = str(int(max_response_length))
    child["FLASH_OPENRLHF_STOP_SEQUENCES"] = json.dumps(list(stop_sequences))
    child["FLASH_OPENRLHF_EOS_TOKEN_IDS"] = json.dumps(
        sorted(int(value) for value in eos_token_ids)
    )
    if seed is not None:
        child["FLASH_OPENRLHF_ROLLOUT_SEED"] = str(int(seed))
    child["HF_HUB_OFFLINE"] = "1"
    child["TRANSFORMERS_OFFLINE"] = "1"
    child["HF_HUB_DISABLE_XET"] = "1"
    if language_model_only:
        child["FLASH_OPENRLHF_LANGUAGE_MODEL_ONLY"] = "1"
    if fp8_kv:
        child["FLASH_OPENRLHF_FP8_KV"] = "1"
    if warmstart_adapter:
        child["FLASH_OPENRLHF_WARMSTART_ADAPTER"] = warmstart_adapter
    if save_at_steps:
        child["FLASH_OPENRLHF_SAVE_AT_STEPS"] = ",".join(str(step) for step in save_at_steps)
    if sm120_vllm_backend:
        child["FLASH_OPENRLHF_SM120_VLLM_BACKEND"] = "1"
    if actor_attn_implementation:
        child["FLASH_OPENRLHF_ATTN_IMPLEMENTATION"] = actor_attn_implementation
    if reward_record_url:
        child["FLASH_OPENRLHF_REWARD_RECORD_URL"] = reward_record_url
    return child


def _write_scheduled_dataset(
    path: str,
    prompts: list[dict[str, Any]],
    *,
    steps: int,
    prompts_per_step: int,
) -> int:
    if not prompts:
        raise ValueError("cannot schedule an empty OpenRLHF prompt dataset")
    scheduled_count = int(steps) * int(prompts_per_step)
    if scheduled_count <= 0:
        raise ValueError("OpenRLHF prompt schedule must contain at least one row")
    with open(path, "w", encoding="utf-8") as dataset_file:
        for ordinal in range(scheduled_count):
            prompt = prompts[ordinal % len(prompts)]
            identity = {
                "global_step": ordinal // prompts_per_step,
                "example_index": int(prompt["example_idx"]),
                "rollout_ordinal": 0,
                "turn": 0,
                "retry": 0,
            }
            dataset_file.write(
                json.dumps(
                    {
                        "input": prompt["rendered"],
                        "label": json.dumps(identity, sort_keys=True, separators=(",", ":")),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    return scheduled_count


def _resolve_cached_model_snapshot(model_id: str, model_revision: str) -> str:
    if len(model_revision) != 40 or any(
        char not in "0123456789abcdefABCDEF" for char in model_revision
    ):
        raise ValueError("OpenRLHF GRPO requires a validated immutable 40-character model revision")
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=model_id,
        revision=model_revision,
        local_files_only=True,
    )
    if os.path.basename(os.path.normpath(snapshot)).lower() != model_revision.lower():
        raise RuntimeError("cached OpenRLHF model snapshot does not match the validated revision")
    return snapshot


def _resolve_openrlhf_warmstart(
    adapter_ref: str,
    *,
    fallback_rank: int,
    fallback_alpha: int,
) -> tuple[str, int, int]:
    if not adapter_ref:
        return "", int(fallback_rank), int(fallback_alpha)
    from flash.engine.worker.adapter import _download_adapter

    adapter_dir = _download_adapter(adapter_ref)
    if not adapter_dir:
        raise RuntimeError(
            "OpenRLHF warm-start source adapter could not be downloaded; refusing to start from base"
        )
    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as config_file:
        adapter_config = json.load(config_file)
    if str(adapter_config.get("peft_type", "")).upper() != "LORA":
        raise RuntimeError("OpenRLHF warm-start requires a LoRA adapter")
    rank = int(adapter_config.get("r") or 0)
    alpha = int(adapter_config.get("lora_alpha") or 0)
    if rank <= 0 or alpha <= 0:
        raise RuntimeError("OpenRLHF warm-start adapter has invalid rank or alpha")
    return adapter_dir, rank, alpha


def _openrlhf_resume_step(checkpoint_dir: str | None) -> int:
    if not checkpoint_dir:
        return 0
    latest_path = os.path.join(checkpoint_dir, "_actor", "latest")
    try:
        with open(latest_path, encoding="utf-8") as latest_file:
            tag = latest_file.read().strip()
    except OSError as exc:
        raise RuntimeError("OpenRLHF resume checkpoint is missing its latest tag") from exc
    if not tag.startswith("global_step") or not tag.removeprefix("global_step").isdigit():
        raise RuntimeError("OpenRLHF resume checkpoint has an invalid latest tag")
    step = int(tag.removeprefix("global_step"))
    if step <= 0 or not os.path.isdir(os.path.join(checkpoint_dir, "_actor", tag)):
        raise RuntimeError("OpenRLHF resume checkpoint is missing its DeepSpeed state")
    return step


def _verified_openrlhf_published_steps(
    save_at_steps: tuple[int, ...], resumed_step: int
) -> set[int]:
    return {
        int(step)
        for step in save_at_steps
        if int(step) <= int(resumed_step) and _deployable_adapter_on_hf(int(step))
    }


def _stage_openrlhf_resume_checkpoint(checkpoint_dir: str, tag: str, step: int) -> str:
    source = os.path.join(checkpoint_dir, "_actor", tag)
    if not os.path.isdir(source):
        raise RuntimeError(f"OpenRLHF checkpoint {tag} has no DeepSpeed actor state")
    stage = os.path.join(checkpoint_dir, ".flash-resume-upload", f"checkpoint-{step}")
    shutil.rmtree(stage, ignore_errors=True)
    actor_stage = os.path.join(stage, "_actor")
    os.makedirs(actor_stage, exist_ok=True)
    try:
        shutil.copytree(source, os.path.join(actor_stage, tag), copy_function=os.link)
    except OSError:
        shutil.rmtree(os.path.join(actor_stage, tag), ignore_errors=True)
        shutil.copytree(source, os.path.join(actor_stage, tag))
    Path(actor_stage, "latest").write_text(tag, encoding="utf-8")
    Path(stage, "resume_metadata.json").write_text(
        json.dumps(
            {
                "backend": "openrlhf",
                "checkpoint_tag": tag,
                "global_step": int(step),
                "schema_version": 1,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return stage


def _checkpoint_marker(line: str) -> dict[str, Any] | None:
    prefix = "[flash-openrlhf-checkpoint] "
    marker_start = line.find(prefix)
    if marker_start < 0:
        return None
    payload = json.loads(line[marker_start + len(prefix) :])
    if not isinstance(payload, dict):
        raise RuntimeError("OpenRLHF checkpoint marker must be an object")
    return payload


def _publish_openrlhf_checkpoint(
    marker: dict[str, Any],
    *,
    checkpoint_dir: str,
    adapter_workdir: str,
    model_id: str,
    model_revision: str,
    tokenizer,
    python_bin: str,
    required_steps: tuple[int, ...],
) -> int:
    step = int(marker.get("step", 0))
    tag = str(marker.get("tag", ""))
    if step <= 0 or tag != f"global_step{step}":
        raise RuntimeError("OpenRLHF checkpoint marker has an invalid step or tag")
    expected_root = os.path.realpath(checkpoint_dir)
    if os.path.realpath(str(marker.get("checkpoint_dir", ""))) != expected_root:
        raise RuntimeError("OpenRLHF checkpoint marker escaped its checkpoint directory")
    source_adapter = os.path.join(checkpoint_dir, f"{tag}_hf")
    if os.path.realpath(str(marker.get("adapter_dir", ""))) != os.path.realpath(source_adapter):
        raise RuntimeError("OpenRLHF checkpoint marker has an invalid adapter directory")
    ack_path = str(marker.get("ack_path", ""))
    expected_ack = os.path.join(checkpoint_dir, f".flash-uploaded-{tag}")
    if os.path.realpath(ack_path) != os.path.realpath(expected_ack):
        raise RuntimeError("OpenRLHF checkpoint marker has an invalid acknowledgement path")

    deployable_dir = os.path.join(adapter_workdir, f"step-{step}")
    export_openrlhf_adapter(
        source_adapter,
        deployable_dir,
        model_id,
        model_revision,
        python_bin,
    )
    tokenizer.save_pretrained(deployable_dir)
    _w.write_base_model_provenance(deployable_dir, model_id, model_revision)
    required = step in frozenset(int(value) for value in required_steps)
    if required and not getattr(_w, "HF_REPO", ""):
        raise RuntimeError(f"required OpenRLHF save step {step} has no artifact repository")
    staged_resume = _stage_openrlhf_resume_checkpoint(checkpoint_dir, tag, step)
    try:

        def publish_deployable() -> None:
            _w.publish_deployable_checkpoint(
                deployable_dir,
                step,
                required=required,
            )

        uploaded = _w.upload_resume_checkpoint(
            step,
            staged_resume,
            before_upload=publish_deployable,
        )
        if required and not uploaded:
            raise _w.RetriableInfraError(
                f"required OpenRLHF save step {step} full-state checkpoint was not durable"
            )
        Path(ack_path).touch()
    finally:
        shutil.rmtree(staged_resume, ignore_errors=True)
    return step


def _resolve_single_turn_inputs() -> dict[str, Any]:
    spec = _w.JOB_SPEC
    if spec is None or spec.algorithm != "grpo":
        raise RuntimeError("OpenRLHF GRPO requires a GRPO JobSpec")
    env = _w.require_active_env()
    if getattr(env, "multi_turn", False) or getattr(env, "is_tool_env", False):
        raise RuntimeError(
            "OpenRLHF GRPO foundation supports single-turn non-tool environments only"
        )

    train_spec = spec.train
    if train_spec.save_every is not None:
        raise RuntimeError(
            "OpenRLHF GRPO periodic checkpoints are not uploaded; omit train.save_every or use "
            "the TRL backend"
        )
    if train_spec.structured_outputs:
        raise RuntimeError("OpenRLHF GRPO structured outputs are deferred; use the TRL backend")
    if train_spec.credit_assignment != DEFAULT_CREDIT_ASSIGNMENT:
        raise RuntimeError(
            "OpenRLHF GRPO per-turn credit requires action-span reward advantages that the pinned "
            "OpenRLHF experience schema cannot express; use the TRL backend"
        )
    if float(RECIPE.lora.dropout) != 0.0:
        raise RuntimeError("OpenRLHF GRPO requires the managed zero LoRA dropout")

    model_id = spec.model
    model_revision = spec.model_revision
    if not model_revision:
        raise ValueError("OpenRLHF GRPO requires model_revision for immutable adapter provenance")

    rl = RECIPE.rl
    overrides = _w.grpo_overrides()
    prompts_per_step = int(
        train_spec.batch_size if train_spec.batch_size is not None else rl.prompts_per_step
    )
    group_size_value = overrides.get("group_size")
    group_size = int(group_size_value if group_size_value is not None else rl.group_size)
    if group_size <= 1:
        raise ValueError("OpenRLHF DR-GRPO requires train.group_size greater than 1")
    temperature_value = overrides.get("temperature")
    temperature = float(
        temperature_value if temperature_value is not None else rl.sampling_temperature
    )
    think_penalty_value = overrides.get("thinking_length_penalty_coef")
    think_penalty = float(think_penalty_value if think_penalty_value is not None else 0.0)
    kl_coef_value = overrides.get("kl_penalty_coef")
    kl_coef = float(kl_coef_value if kl_coef_value is not None else 0.0)
    learning_rate = float(
        train_spec.learning_rate if train_spec.learning_rate is not None else rl.learning_rate
    )
    lora_rank = int(train_spec.lora_rank if train_spec.lora_rank is not None else RECIPE.lora.rank)
    lora_alpha = int(
        train_spec.lora_alpha if train_spec.lora_alpha is not None else RECIPE.lora.alpha
    )
    warmstart_adapter, lora_rank, lora_alpha = _resolve_openrlhf_warmstart(
        train_spec.init_from_adapter,
        fallback_rank=lora_rank,
        fallback_alpha=lora_alpha,
    )
    max_completion_value = overrides.get("max_tokens")
    max_completion = int(
        max_completion_value
        if max_completion_value is not None
        else (rl.max_completion_len_thinking if _w.THINKING else rl.max_completion_len)
    )
    max_length = int(
        train_spec.max_context_tokens
        if train_spec.max_context_tokens is not None
        else max(1024, int(rl.max_prompt_len) + max_completion)
    )
    prompt_budget = max_length - max_completion
    if prompt_budget <= 0:
        raise ValueError("OpenRLHF max_context_tokens leaves no room for a prompt")

    train = list(env.dataset())
    if train_spec.max_examples:
        train = train[: int(train_spec.max_examples)]
    random.Random(spec.seed).shuffle(train)
    message_prompts = [env.prompt_messages(example) for example in train]

    from flash.multimodal import record_has_images

    if any(
        record_has_images(example, messages)
        for example, messages in zip(train, message_prompts, strict=True)
    ):
        raise RuntimeError("OpenRLHF GRPO multimodal support is deferred; use the TRL backend")

    tokenizer = _w.load_tokenizer(model_id, revision=model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts: list[dict[str, Any]] = []
    for example_idx, (example, messages) in enumerate(zip(train, message_prompts, strict=True)):
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=_w.THINKING,
        )
        token_count = len(tokenizer(rendered, add_special_tokens=False).input_ids)
        if 0 < token_count <= prompt_budget:
            prompts.append(
                {
                    "rendered": rendered,
                    "example": example,
                    "example_idx": example_idx,
                }
            )
    if not prompts:
        raise ValueError(
            f"every OpenRLHF training prompt exceeds the {prompt_budget}-token prompt budget"
        )
    prompts_per_step = _w.resolve_grpo_prompts_per_step(prompts_per_step, len(prompts))
    epochs = int(train_spec.epochs if train_spec.epochs is not None else rl.num_epochs)
    derived_steps = on_policy_steps(
        epochs=epochs,
        prompt_count=len(prompts),
        prompts_per_step=prompts_per_step,
    )
    steps = resolve_update_horizon(derived_steps, train_spec.max_steps)
    save_every = -1
    save_at_steps = tuple(int(step) for step in (train_spec.save_at_steps or ()))
    validate_save_steps(save_at_steps, steps)
    prompt_opened_thinking = bool(_w.THINKING) and _w.prompt_opens_thinking(prompts[0]["rendered"])
    from flash.engine.worker.grpo import resolve_grpo_sleep_mode

    _, _, _, fp8_kv = resolve_grpo_sleep_mode()

    return {
        "env": env,
        "tokenizer": tokenizer,
        "model_id": model_id,
        "model_revision": model_revision,
        "prompts": prompts,
        "prompts_per_step": prompts_per_step,
        "group_size": group_size,
        "temperature": temperature,
        "top_p": float(rl.sampling_top_p),
        "think_penalty": think_penalty,
        "kl_coef": kl_coef,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "max_completion": max_completion,
        "max_length": max_length,
        "prompt_opened_thinking": prompt_opened_thinking,
        "steps": int(steps),
        "save_every": save_every,
        "save_at_steps": save_at_steps,
        "gpu_count": int(spec.gpu.count),
        "seed": int(backend_seed(spec.seed)),
        "stop_sequences": tuple(str(value) for value in train_spec.stop_sequences),
        "fp8_kv": fp8_kv,
        "warmstart_adapter": warmstart_adapter,
    }


def _generation_eos_from_cached_config(
    model_id: str, model_revision: str, tokenizer
) -> frozenset[int]:
    from transformers import AutoConfig, GenerationConfig

    from flash.engine.worker.opd_gkd import _generation_eos_ids

    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
        revision=model_revision or None,
        local_files_only=True,
    )
    try:
        generation_config = GenerationConfig.from_pretrained(
            model_id,
            revision=model_revision or None,
            local_files_only=True,
        )
    except OSError:
        generation_config = None
    model_like = type(
        "ModelGenerationMetadata",
        (),
        {"config": config, "generation_config": generation_config},
    )()
    return _generation_eos_ids(model_like, tokenizer)


def _is_sm120() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 12)
    except Exception as exc:
        print(f"[flash-openrlhf] sm120 vLLM backend probe skipped: {exc}", flush=True)
        return False


def run_rl_openrlhf() -> None:
    """Run single-turn text GRPO through the isolated OpenRLHF trainer."""
    started_at = time.time()
    seed_training_rngs(_w.SEED)
    _w.heartbeat("rl_start", gpu=gpu_diagnostics())
    wait_for_gpu(
        _w.JOB_SPEC.gpu.type if _w.JOB_SPEC else None,
        exact_type=_w.JOB_SPEC.gpu.exact_type if _w.JOB_SPEC else "",
    )
    setup_perf_backends()
    sm120_vllm_backend = _is_sm120()
    # none leaves openrlhf's actor attention default unchanged.
    actor_attn_implementation = optimal_attn_impl()
    with liveness_heartbeat("rl_data_loading"):
        inputs = _resolve_single_turn_inputs()

    if inputs["model_revision"]:
        download_seconds = _w.prefetch_model(inputs["model_id"], revision=inputs["model_revision"])
    else:
        download_seconds = _w.prefetch_model(inputs["model_id"])
    model_path = _resolve_cached_model_snapshot(inputs["model_id"], inputs["model_revision"])
    eos_token_ids = _generation_eos_from_cached_config(
        inputs["model_id"],
        inputs["model_revision"],
        inputs["tokenizer"],
    )
    resume_checkpoint = _w.hf_resume_checkpoint()
    resumed_step = _openrlhf_resume_step(resume_checkpoint)

    workdir = f"/tmp/rl_openrlhf_seed{inputs['seed']}"
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir, exist_ok=True)
    output_dir = os.path.join(workdir, "final")
    checkpoint_dir = resume_checkpoint or os.path.join(workdir, "checkpoints")
    dataset_path = os.path.join(workdir, "train.jsonl")
    adapter_dir = f"/tmp/rl_seed{_w.SEED}/adapter"
    checkpoint_adapter_workdir = os.path.join(workdir, "checkpoint-adapters")
    plugin_dir = write_openrlhf_sitecustomize(workdir)
    scheduled_prompt_count = _write_scheduled_dataset(
        dataset_path,
        inputs["prompts"],
        steps=inputs["steps"],
        prompts_per_step=inputs["prompts_per_step"],
    )
    examples = {int(prompt["example_idx"]): prompt["example"] for prompt in inputs["prompts"]}

    def score_by_label(label: int, completion: str, _prompt: str) -> RewardResult:
        try:
            example = examples[label]
        except KeyError as exc:
            raise ValueError(f"unknown OpenRLHF reward label {label}") from exc
        return score_single_turn(
            inputs["env"],
            completion,
            example,
            tokenizer=inputs["tokenizer"],
            thinking=bool(_w.THINKING),
            prompt_opened_thinking=inputs["prompt_opened_thinking"],
            think_penalty=inputs["think_penalty"],
        )

    qwen35_language_model_only = bool(
        _w.is_vl_checkpoint(inputs["model_id"], inputs["model_revision"])
    )
    last_step = [resumed_step]

    def split_query(query: str, prompt: str) -> str:
        return completion_from_tokenizer_query(inputs["tokenizer"], query, prompt)

    with RewardBridge(
        score_by_label,
        samples_per_step=inputs["prompts_per_step"] * inputs["group_size"],
        first_step=resumed_step + 1,
        completion_from_query=split_query,
    ) as bridge:
        config = OpenRLHFGRPOConfig(
            model_path=model_path,
            dataset_path=dataset_path,
            reward_url=bridge.url,
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
            lora_rank=inputs["lora_rank"],
            lora_alpha=inputs["lora_alpha"],
            kl_coef=inputs["kl_coef"],
            save_every=inputs["save_every"],
            save_at_steps=inputs["save_at_steps"],
            gpu_count=inputs["gpu_count"],
            qwen35_language_model_only=qwen35_language_model_only,
            fp8_kv=inputs["fp8_kv"],
            warmstart_adapter=inputs["warmstart_adapter"],
            resume=bool(resume_checkpoint),
            actor_attn_implementation=actor_attn_implementation,
        )
        args = build_openrlhf_grpo_args(config)
        child_env = build_openrlhf_child_env(
            plugin_dir=plugin_dir,
            max_response_length=inputs["max_completion"],
            language_model_only=qwen35_language_model_only,
            sm120_vllm_backend=sm120_vllm_backend,
            seed=inputs["seed"],
            stop_sequences=inputs["stop_sequences"],
            eos_token_ids=eos_token_ids,
            fp8_kv=inputs["fp8_kv"],
            warmstart_adapter=inputs["warmstart_adapter"],
            save_at_steps=inputs["save_at_steps"],
            actor_attn_implementation=actor_attn_implementation,
            reward_record_url=bridge.record_url,
        )
        python_bin = resolve_openrlhf_python(workdir)
        _w.heartbeat("rl_train_start", gpu=gpu_diagnostics())

        published_steps = _verified_openrlhf_published_steps(inputs["save_at_steps"], resumed_step)
        dumped_sample_steps: set[int] = set()
        sent_first_sample_heartbeat = [False]

        def on_step(step: int) -> None:
            step = int(step)
            last_step[0] = max(last_step[0], step)
            if step in dumped_sample_steps:
                return
            sampled_completions = bridge.drain_sampled_completions(step)
            if not sampled_completions:
                return
            dumped_sample_steps.add(step)
            print(
                f"[rl-openrlhf] step {step} sampled_completions="
                + json.dumps(sampled_completions, separators=(",", ":")),
                flush=True,
            )
            committed = _w.heartbeat(
                "rl_step",
                step=step,
                sampled_completions=sampled_completions,
                force=not sent_first_sample_heartbeat[0],
            )
            if committed:
                sent_first_sample_heartbeat[0] = True

        def on_line(line: str) -> None:
            marker = _checkpoint_marker(line)
            if marker is None:
                return
            step = _publish_openrlhf_checkpoint(
                marker,
                checkpoint_dir=checkpoint_dir,
                adapter_workdir=checkpoint_adapter_workdir,
                model_id=inputs["model_id"],
                model_revision=inputs["model_revision"],
                tokenizer=inputs["tokenizer"],
                python_bin=python_bin,
                required_steps=inputs["save_at_steps"],
            )
            published_steps.add(step)
            last_step[0] = max(last_step[0], step)

        with liveness_heartbeat(
            "rl_openrlhf_training",
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
                    "rl_openrlhf_training",
                    step=last_step[0],
                    gpu=gpu_diagnostics(),
                ),
            )
        if returncode != 0:
            raise RuntimeError(
                f"openrlhf.cli.train_ppo_ray exited {returncode}; see the worker log"
            )
        if bridge.call_count == 0 and resumed_step < inputs["steps"]:
            raise RuntimeError("OpenRLHF GRPO completed without scoring any reward")
        # a zero exit is OpenRLHF's completion contract. parsed steps are progress telemetry only:
        # releases differ between bare/global and zero/one-based counters, while export verifies the
        # final artifact exists before publication.
        last_step[0] = max(last_step[0], inputs["steps"])
        missing_required = sorted(set(inputs["save_at_steps"]) - published_steps)
        if missing_required:
            raise RuntimeError(
                f"required OpenRLHF save steps were not durably published: {missing_required}"
            )
        reward_history = bridge.rewards

    with liveness_heartbeat(
        "rl_openrlhf_finalizing",
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
    _w.heartbeat(
        "rl_trained",
        train_wall=train_wall,
        step=last_step[0],
        gpu=gpu_diagnostics(),
    )
    generated_tokens = (
        inputs["steps"]
        * inputs["prompts_per_step"]
        * inputs["group_size"]
        * inputs["max_completion"]
    )
    _w.write_train_meta(
        phase="rl",
        adapter_dir=adapter_dir,
        model_id=inputs["model_id"],
        train_wall=train_wall,
        train_tokens=0,
        generated_tokens=generated_tokens,
        notes={
            "backend": "openrlhf",
            "steps": last_step[0],
            "retained_prompts": len(inputs["prompts"]),
            "scheduled_prompts": scheduled_prompt_count,
            "download_seconds": download_seconds,
            "reward_history": reward_history,
            "resumed": bool(resume_checkpoint),
            "init_from_adapter": getattr(
                getattr(_w.JOB_SPEC, "train", None), "init_from_adapter", ""
            ),
            "vllm_kv_cache_dtype": "fp8" if inputs["fp8_kv"] else None,
            "save_at_steps": list(inputs["save_at_steps"]),
            "group_size": inputs["group_size"],
            "prompts_per_step": inputs["prompts_per_step"],
            "max_completion_len": inputs["max_completion"],
            "chunked_action_logprobs": {
                "chunk_size": _OPENRLHF_ACTION_LOGPROB_CHUNK_SIZE,
                "gpu_validation": "pending",
            },
            "grpo_recipe": {
                "advantage_estimator": "dr_grpo",
                "loss_normalization": "fixed_max_response_length",
                "max_response_length": inputs["max_completion"],
                "kl_coef": inputs["kl_coef"],
                "temperature": inputs["temperature"],
                "top_p": inputs["top_p"],
                "seed": inputs["seed"],
                "tis": {"mode": "token_truncate", "c_max": 2.0},
            },
        },
    )
