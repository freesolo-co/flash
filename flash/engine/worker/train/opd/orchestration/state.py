"""immutable state carriers for OPD parent-runner stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flash.engine.support.verl_policy import FsdpGeneration


@dataclass(frozen=True)
class _BridgePrompt:
    student_messages: list[dict]
    teacher_messages: list[dict]
    prompt_ids: tuple[int, ...]
    image_descriptors: tuple[str, ...]
    package_root: str | None
    example: dict | None = None
    image_digests: tuple[str, ...] = ()


@dataclass(frozen=True)
class _OpdRequest:
    spec: Any
    env: Any
    multi_turn: bool
    max_turns: int
    knobs: Any
    model_id: str
    model_revision: str
    structured_outputs: Any = None
    model_vocab_size: int | None = None


@dataclass(frozen=True)
class _PromptState:
    teacher: Any
    tokenizer: Any
    thinking_prefill: str
    max_model_len: int
    prompt_budget: int
    prompts: list[_BridgePrompt]
    dropped_long: int
    processor: Any = None


@dataclass(frozen=True)
class _WorkloadState:
    prompts_per_step: int
    update_horizon: int
    prompt_pool_fingerprint: str
    workdir: str
    shim_dir: str
    local_dir: str
    export_root: str
    mutation_failure_path: str
    score_delivery_failure_path: str
    rollout_failure_path: str
    abandonment_failure_path: str
    resample_failure_path: str
    cycle_commit_failure_path: str
    train_file: str
    val_file: str
    lora_rank: int
    lora_alpha: int
    target_modules: Any
    target_parameters: tuple[str, ...] | None
    fsdp_generation: FsdpGeneration
    exclude_modules: str | None
    warmstart_adapter: str | None


@dataclass(frozen=True)
class _RuntimeState:
    python_bin: str
    model_path: str
    gpu_count: int
    save_freq: int
    loggers: list[str]
    project_name: str
    experiment_name: str
    gdn_reset_arch: str | None
    entry_path: str
    reward_path: str
    resume_step: int
    resume_state: dict[str, Any] | None
    bridge: Any


@dataclass(frozen=True)
class _ChildCallbacks:
    on_line: Any
    on_step: Any
    child_heartbeat: Any
    liveness_fields: Any
    progress: dict[str, Any]
    wandb_link: dict[str, str | None]
    child_tail: Any
    silence_watchdog: Any


@dataclass(frozen=True)
class _ChildResult:
    final_accounting: dict[str, Any]
    actor_dir: str
    final_step: int
    train_wall: float
    peak_gpu_gb: float
    train_started_at: float
    wandb_url: str | None
    wandb_id: str | None
