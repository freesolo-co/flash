"""Build the pre-flight ``plan`` for a run from a parsed ``JobSpec``.

This is the planner behind ``flash plan`` (and the advice surfaced by ``flash train --dry-run``):
fully local, no credentials/server/GPU. It reports the resolved run identity, the *effective*
training knobs (the value the worker will actually use, annotating recipe defaults), and the
:mod:`flash.lint` advice. ``build_plan`` returns a JSON-friendly dict; ``render_plan`` turns it
into a human-readable block.
"""

from __future__ import annotations

from typing import Any

from .catalog import ModelInfo
from .engine.recipe import RECIPE
from .lint import lint_spec
from .spec import JobSpec


def _effective_knobs(spec: JobSpec) -> list[dict[str, Any]]:
    """Headline training knobs with the value the worker will use, tagged ``set`` (from the
    config) or ``default`` (the shared recipe default the worker applies when the knob is unset).

    Only knobs that map directly onto ``RECIPE`` are shown; the worker's finer adjustments
    (e.g. the VRAM-driven micro-batch cap) are deliberately not reimplemented here — a
    ``default`` value reads as "the recipe default, unless the worker adapts it".
    """
    t = spec.train
    grpo = spec.algorithm == "grpo"
    rows: list[dict[str, Any]] = []

    def row(knob: str, value: Any, default: Any) -> None:
        is_set = value is not None
        rows.append(
            {"knob": knob, "value": value if is_set else default, "source": "set" if is_set else "default"}
        )

    if grpo:
        row("steps", t.steps, RECIPE.rl.num_steps)
        row("learning_rate", t.learning_rate, RECIPE.rl.learning_rate)
        row("group_size", t.group_size, RECIPE.rl.group_size)
        compl_default = (
            RECIPE.rl.max_completion_len_thinking if spec.thinking else RECIPE.rl.max_completion_len
        )
        row("max_tokens", t.max_tokens, compl_default)
        row("temperature", t.temperature, RECIPE.rl.sampling_temperature)
    else:
        row("epochs", t.epochs, RECIPE.sft.num_epochs)
        row("learning_rate", t.learning_rate, RECIPE.sft.learning_rate)
        seq_default = RECIPE.sft.max_seq_len_thinking if spec.thinking else RECIPE.sft.max_seq_len
        row("max_length", t.max_length, seq_default)

    # LoRA rank/alpha are never None (TrainSpec defaults 32/64), so they are always "set"
    # (an explicit value, or the dataclass default the worker uses verbatim).
    rows.append({"knob": "lora_rank", "value": t.lora_rank, "source": "set"})
    rows.append({"knob": "lora_alpha", "value": t.lora_alpha, "source": "set"})
    return rows


def build_plan(spec: JobSpec, info: ModelInfo) -> dict[str, Any]:
    """Assemble the full plan dict (identity + effective knobs + advice) for ``spec``."""
    return {
        "run_id": spec.run_id,
        "state": "plan",
        "model": {
            "id": spec.model,
            "display_name": info.display_name,
            "params": info.params,
            "thinking_capability": info.thinking,
        },
        "algorithm": spec.algorithm,
        "thinking": spec.thinking,
        "environment": spec.environment.id,
        "hf_repo": spec.train.hf_repo,
        "gpu": {"type": spec.gpu.type, "provider": spec.gpu.provider},
        "seeds": list(spec.train.seeds),
        "effective_knobs": _effective_knobs(spec),
        "warnings": [a.to_dict() for a in lint_spec(spec, info)],
    }


def render_plan(plan: dict[str, Any]) -> str:
    """Render a plan dict (from :func:`build_plan`) as a human-readable block."""
    model = plan["model"]
    gpu = plan["gpu"]
    lines: list[str] = []
    lines.append(f"plan for run {plan['run_id']}  (local pre-flight — nothing submitted)")
    lines.append("")
    rows = [
        ("model", f"{model['id']}  ({model['params']}, thinking={model['thinking_capability']})"),
        ("algorithm", plan["algorithm"]),
        ("thinking", str(plan["thinking"]).lower()),
        ("environment", plan["environment"]),
        ("hf_repo", plan["hf_repo"]),
        ("gpu", f"{gpu['type']}  (provider {gpu['provider']})"),
        ("seeds", str(plan["seeds"])),
    ]
    width = max(len(k) for k, _ in rows)
    for key, val in rows:
        lines.append(f"  {key.ljust(width)}  {val}")

    lines.append("")
    lines.append("effective training knobs (recipe default shown where unset):")
    kw = max(len(k["knob"]) for k in plan["effective_knobs"])
    for k in plan["effective_knobs"]:
        tag = "" if k["source"] == "set" else "  (default)"
        lines.append(f"  {k['knob'].ljust(kw)}  {k['value']}{tag}")

    warnings = plan["warnings"]
    lines.append("")
    if not warnings:
        lines.append("advice: none — config looks healthy")
    else:
        lines.append(f"advice ({len(warnings)}):")
        lines.extend(f"  ! {w['field']}: {w['message']}" for w in warnings)

    lines.append("")
    lines.append("run it with:  flash train <config>")
    return "\n".join(lines)
