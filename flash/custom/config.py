"""Authored custom runs share JobSpec, projects, artifacts and the train command."""

from flash.core.spec import GpuSpec, JobSpec, TrainSpec, parse_seed
from flash.custom.types import integer

CUSTOM_KEYS = {"file", "resume_from", "idle_seconds", "max_wall_seconds", "controller_id"}
TRAIN_KEYS = {
    "max_steps",
    "max_context_tokens",
    "lora_rank",
    "lora_alpha",
    "save_every",
    "save_at_steps",
}


def parse_spec(raw, *, run_id=None, project=""):
    from flash.schema import ConfigError, _validate_environment_section, _validate_wandb_section

    try:
        custom = raw.get("custom", {})
        train = raw.get("train", {})
        gpu = raw.get("gpu", {})
        for name, section, allowed in (
            ("custom", custom, CUSTOM_KEYS),
            ("train", train, TRAIN_KEYS),
            ("gpu", gpu, {"type", "provider", "count"}),
        ):
            if not isinstance(section, dict) or set(section) - allowed:
                raise ValueError(f"[{name}] custom mode accepts only: {', '.join(sorted(allowed))}")
        controller_id = custom.get("controller_id")
        if controller_id is not None and (
            not isinstance(controller_id, str)
            or len(controller_id) != 32
            or any(c not in "0123456789abcdef" for c in controller_id)
        ):
            raise ValueError("invalid custom controller identity")
        file = custom.get("file")
        if not isinstance(file, str) or not file.strip() or not file.endswith(".py"):
            raise ValueError("[custom] file must name a local Python file, e.g. algorithm.py")
        environment, _, _ = _validate_environment_section(raw, "custom")
        wandb = _validate_wandb_section(raw)
        if any(environment.values()) or wandb.project or wandb.run_name or raw.get("thinking"):
            raise ValueError(
                "custom scripts own data, logging and prompt formatting; omit environment, wandb and thinking"
            )
        if raw["model"] != "Qwen/Qwen3.5-9B":
            raise ValueError("managed custom mode currently supports GPU-validated Qwen/Qwen3.5-9B")
        if gpu.get("type", "H100") != "H100" or gpu.get("provider", "modal") != "modal":
            raise ValueError("managed custom mode currently requires one Modal H100")
        integer(gpu.get("count", 1), "gpu.count", 1, 1)
        resume = custom.get("resume_from", "")
        if not isinstance(resume, str):
            raise ValueError("custom.resume_from must be a run/step-N reference")
        if resume:
            from flash.schema import parse_checkpoint_ref

            parsed = parse_checkpoint_ref(resume)
            if parsed is None or parsed[1] is None:
                raise ValueError("custom.resume_from must name an exact run/step-N checkpoint")
        wall = integer(custom.get("max_wall_seconds", 3600), "custom.max_wall_seconds", 300, 86400)
        idle = integer(custom.get("idle_seconds", 900), "custom.idle_seconds", 30, wall)
        rank = integer(train.get("lora_rank", 16), "train.lora_rank", 1, 64)
        return JobSpec(
            model=raw["model"],
            algorithm="custom",
            project=project,
            run_id=run_id or "local",
            seed=parse_seed(raw.get("seed", 42)),
            custom={
                "file": file,
                "resume_from": resume,
                "idle_seconds": idle,
                "max_wall_seconds": wall,
                **({"controller_id": controller_id} if controller_id is not None else {}),
            },
            train=TrainSpec(
                max_steps=integer(train.get("max_steps", 1000), "train.max_steps", 1, 1_000_000),
                max_context_tokens=integer(
                    train.get("max_context_tokens", 2048), "train.max_context_tokens", 2, 4096
                ),
                lora_rank=rank,
                lora_alpha=integer(train.get("lora_alpha", 2 * rank), "train.lora_alpha", 1, 256),
                save_every=integer(train.get("save_every", 10), "train.save_every", 1, 1_000_000),
                save_at_steps=train.get("save_at_steps"),
            ),
            gpu=GpuSpec(
                type="H100", provider="modal", count=1, max_retries=0, max_wall_seconds=wall
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
