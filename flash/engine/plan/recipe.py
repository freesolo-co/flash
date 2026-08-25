"""Frozen, shared Flash fine-tuning recipe; per-run TOML configs override fields."""

from __future__ import annotations

from dataclasses import dataclass, field

from flash.core.grpo import DEFAULT_GRPO_GROUP_SIZE, DEFAULT_GRPO_PROMPTS_PER_STEP

# keep in sync with catalog.default_model.
HF_MODEL_ID = "Qwen/Qwen3.5-9B"


@dataclass(frozen=True)
class TeacherModel:
    """One managed serverless OPD teacher and its pinned local tokenizer contract."""

    alias: str
    model_id: str
    display_name: str
    usd_per_1m: tuple[float, float]
    tokenizer_repo: str
    tokenizer_revision: str
    tokenizer_kind: str
    tokenizer_files: tuple[tuple[str, str], ...]
    supports_images: bool = False


DEFAULT_TEACHER_ALIAS = "glm-5.2"

TEACHER_MODELS: dict[str, TeacherModel] = {
    "kimi-k3": TeacherModel(
        alias="kimi-k3",
        model_id="parasail-kimi-k3-fast",
        display_name="Kimi K3",
        usd_per_1m=(3.00, 15.00),
        tokenizer_repo="moonshotai/Kimi-K3",
        tokenizer_revision="9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
        tokenizer_kind="kimi_tiktoken",
        tokenizer_files=(
            (
                "tokenizer_config.json",
                "5d0803c94db9cd78763499e0956c95fd5a225c14a727e5a6cf5db3f96f010a6e",
            ),
            (
                "tiktoken.model",
                "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
            ),
        ),
    ),
    "glm-5.2": TeacherModel(
        alias="glm-5.2",
        model_id="parasail-glm-52",
        display_name="GLM 5.2",
        usd_per_1m=(1.40, 4.40),
        tokenizer_repo="nvidia/GLM-5.2-NVFP4",
        tokenizer_revision="aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa",
        tokenizer_kind="tokenizer_json",
        tokenizer_files=(
            (
                "tokenizer.json",
                "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
            ),
        ),
    ),
    "deepseek-v4-pro": TeacherModel(
        alias="deepseek-v4-pro",
        model_id="parasail-deepseek-v4-pro",
        display_name="DeepSeek V4 Pro",
        usd_per_1m=(1.74, 3.48),
        tokenizer_repo="deepseek-ai/DeepSeek-V4-Pro",
        tokenizer_revision="b5968e9190ef611bbf34a7229255be88a0e937c1",
        tokenizer_kind="tokenizer_json",
        tokenizer_files=(
            (
                "tokenizer.json",
                "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf",
            ),
        ),
    ),
    "qwen3.5-397b-a17b": TeacherModel(
        alias="qwen3.5-397b-a17b",
        model_id="parasail-qwen35-397b-a17b",
        display_name="Qwen3.5 397B A17B",
        usd_per_1m=(0.50, 3.60),
        tokenizer_repo="Qwen/Qwen3.5-397B-A17B-FP8",
        tokenizer_revision="ea5b4f81096f3901c91dea97f81324302495781d",
        tokenizer_kind="tokenizer_json",
        tokenizer_files=(
            (
                "tokenizer.json",
                "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            ),
        ),
        # qwen3.5 unified text and vision in one checkpoint, so this is a vision model despite the
        # name carrying no "-vl": its config is Qwen3_5MoeForConditionalGeneration with a
        # vision_config, its tokenizer defines <|image_pad|>, and the parasail route tokenizes an
        # image (14 -> 212 prompt tokens as the image grows) and names its color correctly.
        supports_images=True,
    ),
    "qwen3-vl-235b": TeacherModel(
        alias="qwen3-vl-235b",
        model_id="parasail-qwen3-vl-235b-a22b-instruct",
        display_name="Qwen3-VL 235B A22B",
        usd_per_1m=(0.21, 1.90),
        tokenizer_repo="Qwen/Qwen3-VL-235B-A22B-Instruct",
        tokenizer_revision="710c13861be6c466e66de3f484069440b8f31389",
        tokenizer_kind="tokenizer_json",
        tokenizer_files=(
            (
                "tokenizer.json",
                "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
            ),
        ),
        supports_images=True,
    ),
}


def normalize_teacher_alias(value: str) -> str:
    """Normalize a user-facing teacher alias without accepting upstream model identifiers."""
    return "-".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())


def resolve_teacher(value: str | None) -> TeacherModel:
    """Resolve only a friendly managed alias, defaulting an omitted value to GLM 5.2."""
    if value is None or not str(value).strip():
        return TEACHER_MODELS[DEFAULT_TEACHER_ALIAS]
    key = normalize_teacher_alias(str(value))
    if key in TEACHER_MODELS:
        return TEACHER_MODELS[key]
    allowed = ", ".join(TEACHER_MODELS)
    raise ValueError(
        f"teacher_model {value!r} is not a supported teacher; choose one of: {allowed} "
        f"(default {DEFAULT_TEACHER_ALIAS})"
    )


def teacher_supports_images(value: str) -> bool:
    """Return whether the selected managed teacher can condition on image inputs."""
    return resolve_teacher(value).supports_images


def image_capable_teacher_aliases() -> tuple[str, ...]:
    """The aliases a user may select for image-bearing opd, in catalog order.

    Derived from the catalog rather than written out at the call site: the set has already changed
    twice, and a hardcoded list in an error message goes stale silently -- it still renders, just
    naming a teacher that no longer exists or omitting one that does.
    """
    return tuple(alias for alias, teacher in TEACHER_MODELS.items() if teacher.supports_images)


def teacher_for_model_id(model_id: str) -> TeacherModel:
    """Resolve an internal provider model id after public alias validation has already succeeded."""
    for teacher in TEACHER_MODELS.values():
        if model_id == teacher.model_id:
            return teacher
    raise ValueError(f"unknown managed teacher model id {model_id!r}")


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.0


@dataclass(frozen=True)
class SFTConfig:
    max_seq_len: int = 1024
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
    max_completion_len_thinking: int = 1536
    prompts_per_step: int = DEFAULT_GRPO_PROMPTS_PER_STEP
    group_size: int = DEFAULT_GRPO_GROUP_SIZE
    num_epochs: int = 1
    sampling_temperature: float = 1.0
    sampling_top_p: float = 1.0


@dataclass(frozen=True)
class OPDConfig:
    """On-policy distillation using one managed serverless teacher."""

    teacher_model: str = DEFAULT_TEACHER_ALIAS
    learning_rate: float = 1e-5
    max_prompt_len: int = 1024
    max_completion_len: int = 512
    max_completion_len_thinking: int = 1536
    prompts_per_step: int = 8
    group_size: int = 1
    num_epochs: int = 1
    sampling_temperature: float = 1.0
    sampling_top_p: float = 1.0
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
