"""Frozen, shared Flash fine-tuning recipe; per-run TOML configs override fields."""

from __future__ import annotations

from dataclasses import dataclass, field

# Keep in sync with catalog.DEFAULT_MODEL.
HF_MODEL_ID = "Qwen/Qwen3.5-4B"


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.0


@dataclass(frozen=True)
class SFTConfig:
    max_seq_len: int = 1024
    # <think> traces need extra headroom; VRAM scales with length.
    max_seq_len_thinking: int = 2048
    learning_rate: float = 1e-4
    warmup_frac: float = 0.03
    effective_batch: int = 32
    num_epochs: int = 2


@dataclass(frozen=True)
class RLConfig:
    learning_rate: float = 1e-5
    max_prompt_len: int = 2048
    max_completion_len: int = 320
    # 320 is too short for <think> blocks; overridable via [train].max_tokens.
    max_completion_len_thinking: int = 1536
    prompts_per_step: int = 64
    group_size: int = 8
    num_steps: int = 150
    sampling_temperature: float = 1.0
    sampling_top_p: float = 1.0


@dataclass(frozen=True)
class OPDConfig:
    """On-policy distillation: student samples, a remote teacher scores its tokens, and a groupwise
    reverse-KL loss (the collinear-ai spider / Tinker cross-tokenizer method) trains the student."""

    # Fireworks-hosted teacher reached over the OpenAI-compatible API; overridable via
    # [train].teacher_model. GLM-5.2 is a strong reasoning model whose per-token logprobs
    # supervise the (much smaller) student.
    teacher_model: str = "accounts/fireworks/models/glm-5p2"
    teacher_base_url: str = "https://api.fireworks.ai/inference/v1"
    # OPD is step-driven like GRPO (on-policy sampling), not epoch-driven like SFT.
    num_steps: int = 100
    learning_rate: float = 1e-5
    max_prompt_len: int = 1024
    max_completion_len: int = 512
    # 512 is short for <think> traces; overridable via [train].max_tokens.
    max_completion_len_thinking: int = 1536
    prompts_per_step: int = 8
    # Student samples per prompt. 1 is enough for a direct KD loss (no group-relative baseline
    # as in GRPO); raise for more teacher-scored coverage per prompt at higher cost.
    group_size: int = 1
    sampling_temperature: float = 1.0
    sampling_top_p: float = 1.0
    # Reverse-KL coefficient for the groupwise reverse-KL loss; scales the per-span
    # (student_logsum - teacher_logsum) advantage. 1.0 is plain reverse KL (Thinking Machines,
    # *On-Policy Distillation*). Overridable via [train].kl_penalty_coef.
    kl_coef: float = 1.0


@dataclass(frozen=True)
class Recipe:
    """The complete shared recipe."""

    hf_model_id: str = HF_MODEL_ID
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    opd: OPDConfig = field(default_factory=OPDConfig)


RECIPE = Recipe()
