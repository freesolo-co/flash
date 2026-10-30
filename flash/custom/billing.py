"""Custom sessions charge measured session time against their frozen wall-time cap."""

import json
import math
from pathlib import Path


def estimate(spec, *, allocation=None):
    from flash.cost.types import CostEstimate
    from flash.providers.core.allocator import allocate

    rented = allocation or allocate(
        spec.model,
        spec.algorithm,
        train=spec.train,
        provider=spec.gpu.provider,
        providers=spec.gpu.providers,
        gpu_type=spec.gpu.type,
        gpu_type_fallbacks=spec.gpu.type_fallbacks,
        max_gpu_count=spec.authored_gpu_count,
        model_revision=spec.model_revision,
        disk_gb=spec.gpu.disk_gb,
        max_wall_seconds=spec.custom["max_wall_seconds"],
    )
    seconds = spec.custom["max_wall_seconds"]
    return CostEstimate(
        model_id=spec.model,
        method="custom",
        steps=spec.train.max_steps,
        gpu=rented.gpu,
        provider=rented.provider,
        gpu_vram_gb=rented.min_vram_gb,
        required_vram_gb=rented.min_vram_gb,
        gpu_hourly_usd=rented.hourly_usd,
        setup_seconds=0,
        seconds_per_step=seconds / spec.train.max_steps,
        train_seconds=seconds,
        wall_clock_seconds=seconds,
        wall_capped=True,
        total_usd=seconds * rented.hourly_usd / 3600,
        notes=(
            "Upper bound for a dedicated custom session, including time waiting for the local script.",
        ),
    )


def session_charge(status, spec):
    from flash.runner.lifecycle.state import artifacts_dir

    seconds = 0.0
    try:
        metrics = json.loads((Path(artifacts_dir(spec)) / "metrics.json").read_text())
        seconds = metrics["session_seconds"]
    except (OSError, ValueError, KeyError):
        heartbeat = status.last_heartbeat or {}
        started = heartbeat.get("session_started_at")
        if isinstance(started, (int, float)) and not isinstance(started, bool):
            ended = getattr(status, "finished_at", None) or status.updated_at
            seconds = max(0.0, ended - started)
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
    ):
        raise ValueError("invalid custom session time")
    quote = status.estimated_cost_usd
    if (
        isinstance(quote, bool)
        or not isinstance(quote, (int, float))
        or not math.isfinite(quote)
        or quote < 0
    ):
        raise ValueError("custom run has no valid accepted session cap")
    return quote * min(1.0, max(0.0, seconds) / spec.custom["max_wall_seconds"])
