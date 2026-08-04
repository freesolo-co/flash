"""Frozen, shared Flash fine-tuning recipe; per-run TOML configs override fields."""

from __future__ import annotations

from dataclasses import dataclass, field

# Keep in sync with catalog.DEFAULT_MODEL.
HF_MODEL_ID = "Qwen/Qwen3.5-4B"


@dataclass(frozen=True)
class TeacherModel:
    """A managed OPD teacher: a Fireworks-hosted model the student distils toward, selected by its
    short ``alias`` via ``[train] teacher_model``. The catalog below is the ONE source of truth — the
    schema validates against it, the worker resolves ``alias -> model_id`` from it, and the cost model
    derives its teacher price table from it."""

    alias: str  # short, user-facing name typed in [train] teacher_model (the TEACHER_MODELS key)
    model_id: str  # Fireworks OpenAI-compatible model id (the ``model`` field TeacherClient sends)
    display_name: str
    # Fireworks serverless list price, $/1M tokens as (input, output). OPD echo-scores completions
    # (max_tokens=0) so only the INPUT column is billed / feeds the estimate, but both are kept so a
    # mispriced row is obvious. note: the non-glm entry is a best-effort estimate for this next-gen
    # serverless model — confirm against the live fireworks.ai pricing before relying on the
    # `flash train` cost quote.
    usd_per_1m: tuple[float, float]
    supports_images: bool = False


# The alias used when [train] omits teacher_model — the historical fixed teacher, unchanged.
DEFAULT_TEACHER_ALIAS = "glm-5.2"

# Curated OPD teacher allow-list, keyed by friendly alias (the value users put in [train]
# teacher_model). A CLOSED allow-list, not "any Fireworks model": a teacher MUST support echo-scoring
# (echo=true, logprobs=1, max_tokens=0 tiling the input char-for-char — see
# engine.worker.teacher._validate_echo) to drive the groupwise reverse-KL loss, and every entry needs
# a price row for the estimate. Adding a teacher = adding ONE row here.
TEACHER_MODELS: dict[str, TeacherModel] = {
    # glm-5.2 is the default and the verified echo-scoring baseline (see DEFAULT_TEACHER_ALIAS).
    "glm-5.2": TeacherModel(
        alias="glm-5.2",
        model_id="accounts/fireworks/models/glm-5p2",
        display_name="GLM 5.2",
        usd_per_1m=(1.40, 4.40),
    ),
    "kimi-k2.6": TeacherModel(
        alias="kimi-k2.6",
        model_id="accounts/fireworks/models/kimi-k2p6",
        display_name="Kimi K2.6",
        usd_per_1m=(0.95, 4.00),
        supports_images=True,
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

    Empty/None -> the default teacher (GLM 5.2), so an omitted knob keeps the historical behavior.
    Accepts a friendly alias (``"glm-5.2"``, ``"kimi-k2.6"``, or a spaced ``"GLM 5.2"``) OR the exact
    Fireworks ``model_id`` (so a pasted ``accounts/fireworks/models/...`` still works). Raises
    ``ValueError`` listing the allowed aliases for an unsupported teacher. This is the SINGLE resolver
    the schema (parse-time), the worker, and the cost model all call, so validation can't drift."""
    if value is None or not str(value).strip():
        return TEACHER_MODELS[DEFAULT_TEACHER_ALIAS]
    raw = str(value).strip()
    key = normalize_teacher_alias(raw)
    if key in TEACHER_MODELS:
        return TEACHER_MODELS[key]
    # Compare against the stripped value so a pasted model id with stray surrounding whitespace
    # resolves just like the alias branch (which normalize_teacher_alias already strips). Case is
    # preserved — Fireworks model ids are case-sensitive identifiers.
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

    # The DEFAULT teacher (GLM 5.2), used when [train] omits teacher_model. The teacher is now
    # selectable from the managed teacher_models allow-list via [train].teacher_model; the control
    # plane broker owns provider routing and credentials for every allow-listed model.
    teacher_model: str = TEACHER_MODELS[DEFAULT_TEACHER_ALIAS].model_id
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
