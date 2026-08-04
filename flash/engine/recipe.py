"""Frozen, shared Flash fine-tuning recipe; per-run TOML configs override fields."""

from __future__ import annotations

from dataclasses import dataclass, field

# Keep in sync with catalog.DEFAULT_MODEL.
HF_MODEL_ID = "Qwen/Qwen3.5-4B"

FIREWORKS_TEACHER_BASE_URL = "https://api.fireworks.ai/inference/v1"
PARASAIL_TEACHER_BASE_URL = "https://api.parasail.io/v1"
FIREWORKS_COMPLETION_ECHO = "fireworks_completion_echo"
KIMI_K3_CHAT_PROMPT_LOGPROBS = "kimi_k3_chat_prompt_logprobs"
TEACHER_CREDENTIAL_ENV_KEYS = frozenset({"FIREWORKS_API_KEY", "PARASAIL_API_KEY"})


@dataclass(frozen=True)
class TeacherModel:
    """One closed, platform-managed OPD teacher catalog entry."""

    alias: str
    model_id: str
    display_name: str
    provider: str
    base_url: str
    credential_env: str
    scoring_mode: str
    usd_per_1m: tuple[float, float]
    supports_images: bool = False
    billed_output_tokens_per_score: int = 0
    encoding_repo: str = ""
    encoding_revision: str = ""
    tokenizer_config_sha256: str = ""
    tokenizer_model_sha256: str = ""


# The alias used when [train] omits teacher_model — the historical fixed teacher, unchanged.
DEFAULT_TEACHER_ALIAS = "glm-5.2"

# curated allow-list. every entry has a proven supplied-token scoring contract and static price.
TEACHER_MODELS: dict[str, TeacherModel] = {
    "glm-5.2": TeacherModel(
        alias="glm-5.2",
        model_id="accounts/fireworks/models/glm-5p2",
        display_name="GLM 5.2",
        provider="fireworks",
        base_url=FIREWORKS_TEACHER_BASE_URL,
        credential_env="FIREWORKS_API_KEY",
        scoring_mode=FIREWORKS_COMPLETION_ECHO,
        usd_per_1m=(1.40, 4.40),
    ),
    "kimi-k2.6": TeacherModel(
        alias="kimi-k2.6",
        model_id="accounts/fireworks/models/kimi-k2p6",
        display_name="Kimi K2.6",
        provider="fireworks",
        base_url=FIREWORKS_TEACHER_BASE_URL,
        credential_env="FIREWORKS_API_KEY",
        scoring_mode=FIREWORKS_COMPLETION_ECHO,
        usd_per_1m=(0.95, 4.00),
        supports_images=True,
    ),
    "kimi-k3": TeacherModel(
        alias="kimi-k3",
        model_id="parasail-kimi-k3",
        display_name="Kimi K3",
        provider="parasail",
        base_url=PARASAIL_TEACHER_BASE_URL,
        credential_env="PARASAIL_API_KEY",
        scoring_mode=KIMI_K3_CHAT_PROMPT_LOGPROBS,
        usd_per_1m=(3.00, 15.00),
        billed_output_tokens_per_score=1,
        encoding_repo="moonshotai/Kimi-K3",
        encoding_revision="9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
        tokenizer_config_sha256="5d0803c94db9cd78763499e0956c95fd5a225c14a727e5a6cf5db3f96f010a6e",
        tokenizer_model_sha256="b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
    ),
}


def normalize_teacher_alias(value: str) -> str:
    """Canonicalize a user-typed teacher name to a catalog key: lowercased, trimmed, and any run of
    whitespace/underscores collapsed to a single hyphen. So ``"GLM 5.2"`` / ``"glm_5.2"`` both map to
    the ``"glm-5.2"`` key. A ``.`` in a version is preserved (``kimi-k2.6``)."""
    return "-".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())


def teacher_supports_images(value: str) -> bool:
    """Return whether the resolved managed teacher accepts image-conditioned echo scoring."""
    return resolve_teacher(value).supports_images


def resolve_teacher(value: str) -> TeacherModel:
    """Resolve a ``[train] teacher_model`` value to its catalog entry.

    Empty/None selects the default GLM 5.2 teacher. Accepts a friendly alias, a normalized spaced
    alias, or an exact provider model id. Raises ``ValueError`` listing the allowed aliases for an
    unsupported teacher. The schema, worker, and cost model all use this resolver."""
    if value is None or not str(value).strip():
        return TEACHER_MODELS[DEFAULT_TEACHER_ALIAS]
    raw = str(value).strip()
    key = normalize_teacher_alias(raw)
    if key in TEACHER_MODELS:
        return TEACHER_MODELS[key]
    # compare exact provider ids after stripping surrounding whitespace; preserve identifier case.
    for info in TEACHER_MODELS.values():
        if raw == info.model_id:
            return info
    allowed = ", ".join(TEACHER_MODELS)
    raise ValueError(
        f"teacher_model {value!r} is not a supported teacher; choose one of: {allowed} "
        f"(default {DEFAULT_TEACHER_ALIAS})"
    )


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
    # 320 is too short for <think> blocks; overridable via [train].max_completion_tokens.
    max_completion_len_thinking: int = 1536
    prompts_per_step: int = 64
    group_size: int = 8
    num_epochs: int = 1
    sampling_temperature: float = 1.0
    sampling_top_p: float = 1.0


@dataclass(frozen=True)
class OPDConfig:
    """On-policy distillation: student samples, a remote teacher scores its tokens, and a groupwise
    reverse-KL loss (the collinear-ai spider / Tinker cross-tokenizer method) trains the student."""

    # the default teacher remains glm 5.2 when [train] omits teacher_model.
    teacher_model: str = TEACHER_MODELS[DEFAULT_TEACHER_ALIAS].model_id
    teacher_base_url: str = TEACHER_MODELS[DEFAULT_TEACHER_ALIAS].base_url
    learning_rate: float = 1e-5
    max_prompt_len: int = 1024
    max_completion_len: int = 512
    # 512 is short for <think> traces; overridable via [train].max_completion_tokens.
    max_completion_len_thinking: int = 1536
    prompts_per_step: int = 8
    # Student samples per prompt. 1 is enough for a direct KD loss (no group-relative baseline
    # as in GRPO); raise for more teacher-scored coverage per prompt at higher cost.
    group_size: int = 1
    num_epochs: int = 1
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
