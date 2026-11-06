"""Custom-only fields and constraints; common sections use the normal JobSpec parser."""

from flash.custom.types import integer
from flash.schema.fields import ConfigError, _train_int

TRAIN_KEYS = {
    "max_steps",
    "max_context_tokens",
    "lora_rank",
    "lora_alpha",
    "save_every",
    "save_at_steps",
    "init_from_adapter",
}
TRAIN_DEFAULTS = {"max_steps": 1000, "max_context_tokens": 2048, "lora_rank": 16, "save_every": 10}
CUSTOM_KEYS = {"file", "resume_from", "idle_seconds", "max_wall_seconds", "controller_id"}


def parse_custom_section(value, algorithm):
    from flash.schema import parse_checkpoint_ref

    if algorithm != "custom":
        if value:
            raise ConfigError("[custom] only applies to algorithm = custom")
        return {}
    if not isinstance(value, dict) or set(value) - CUSTOM_KEYS:
        raise ConfigError(f"[custom] accepts only: {', '.join(sorted(CUSTOM_KEYS))}")
    file = value.get("file")
    if not isinstance(file, str) or not file.strip() or not file.endswith(".py"):
        raise ConfigError("[custom] file must name a local Python file, e.g. algorithm.py")
    controller = value.get("controller_id")
    if controller is not None and (
        not isinstance(controller, str)
        or len(controller) != 32
        or any(c not in "0123456789abcdef" for c in controller)
    ):
        raise ConfigError("invalid custom controller identity")
    resume = value.get("resume_from", "")
    parsed = parse_checkpoint_ref(resume)
    if not isinstance(resume, str) or (resume and (parsed is None or parsed[1] is None)):
        raise ConfigError("custom.resume_from must name an exact run/step-N checkpoint")
    try:
        wall = integer(value.get("max_wall_seconds", 3600), "custom.max_wall_seconds", 300, 86400)
        idle = integer(value.get("idle_seconds", 900), "custom.idle_seconds", 30, wall)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return {
        "file": file,
        "resume_from": resume,
        "max_wall_seconds": wall,
        "idle_seconds": idle,
        **({"controller_id": controller} if controller is not None else {}),
    }


def validate_custom_spec(spec):
    if (
        any(spec.to_dict()["environment"].values())
        or spec.wandb.project
        or spec.wandb.run_name
        or spec.thinking
    ):
        raise ConfigError(
            "custom scripts own data, logging and prompt formatting; omit environment, wandb and thinking"
        )
    try:
        integer(spec.train.max_steps, "train.max_steps", 1, 1_000_000)
        # Rank/alpha use the common parser and model's serving rank cap. Context
        # uses the shared submission preflight and GPU memory-fit checks; custom
        # next-token losses only add a two-token minimum, not a separate ceiling.
        _train_int(
            {"max_context_tokens": spec.train.max_context_tokens},
            "max_context_tokens",
            minimum=2,
        )
        integer(spec.train.save_every, "train.save_every", 1, 1_000_000)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    if spec.train.init_from_adapter and spec.custom.get("resume_from"):
        raise ConfigError(
            "train.init_from_adapter starts fresh optimizer state from any adapter; "
            "custom.resume_from continues a custom checkpoint. Set only one"
        )


def validate_custom_gpu_count(model, gpu_spec, gpu_count):
    """Validate custom GPU placement and apply its single-device default."""
    from flash.providers.core.base import custom_gpu_shape

    try:
        custom_gpu_shape(model, gpu_spec.provider, gpu_spec.acceptable_types, gpu_count)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return gpu_count or 1
