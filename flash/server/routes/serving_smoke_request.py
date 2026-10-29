"""JSON-aware requests for deployment verification, separate from customer inference defaults."""

from __future__ import annotations

import json

from flash.core.spec import JobSpec


def json_smoke_messages(constraint: dict | None) -> list[dict] | None:
    """Tell the model the same JSON contract the decoder enforces."""
    if not constraint:
        return None
    if "json" in constraint:
        system = "You output JSON matching this schema:\n" + json.dumps(
            constraint["json"], sort_keys=True, separators=(",", ":")
        )
    elif constraint.get("json_object"):
        system = "You output a JSON object."
    else:
        return None
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": "Return one concise example that satisfies the output contract.",
        },
    ]


def json_smoke_sampling(spec: JobSpec, messages: list[dict] | None) -> dict:
    # Validated on the affected 9B GRPO checkpoints. Greedy constrained decoding can enter
    # whitespace/reasoning loops, so use Qwen's recommended thinking temperature and presence
    # penalty for this verification only. Do not change the adapter's inference defaults.
    # https://huggingface.co/Qwen/Qwen3.5-9B#best-practices
    if messages is not None and spec.thinking and spec.model == "Qwen/Qwen3.5-9B":
        return {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 1.5, "seed": 42}
    return {}
