"""Operator settings for hosted models. Edit the table and redeploy to apply."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPolicy:
    provider: str = "modal"
    min_capacity: int = 0
    max_capacity: int = 2
    lambda_gpus: tuple[str, ...] | list[str] = ()
    # DeepSeek retains its smaller development pool; None uses max_capacity everywhere.
    dev_max_capacity: int | None = None

    def __post_init__(self):
        if self.provider not in {"modal", "lambda", "all"}:
            raise ValueError("provider must be modal, lambda, or all")
        if type(self.min_capacity) is not int or type(self.max_capacity) is not int:
            raise ValueError("capacity must use integers")
        for maximum in (self.max_capacity, self.dev_max_capacity):
            if maximum is None:
                continue
            if (
                type(self.min_capacity) is not int
                or type(maximum) is not int
                or not 0 <= self.min_capacity <= maximum
                or maximum < (2 if self.provider == "all" else 1)
            ):
                raise ValueError(
                    "capacity requires integer 0 <= min_capacity <= max_capacity; "
                    "max_capacity >= 1 (>= 2 for all)"
                )
        if not isinstance(self.lambda_gpus, (tuple, list)) or any(
            gpu not in {"A10", "A100-40GB", "A100-80GB", "H100", "H200", "B200"}
            for gpu in self.lambda_gpus
        ):
            raise ValueError("lambda_gpus must be a list or tuple of approved GPU names")
        object.__setattr__(self, "lambda_gpus", tuple(self.lambda_gpus))


# Lambda is preferred; Modal is a scale-to-zero fallback sharing max_capacity.
# The fleet controller transfers slots as Lambda workers become ready. Handoffs
# allow one temporary extra GPU, then drain the replaced Modal replica before
# starting another. lambda_gpus is an allowlist: use the cheapest available offer.
MODEL_POLICIES = {
    "Qwen/Qwen3.5-0.8B": ModelPolicy(
        "lambda", min_capacity=1, max_capacity=2, lambda_gpus=("A10",)
    ),
    "Qwen/Qwen3.5-4B": ModelPolicy("modal", min_capacity=0, max_capacity=2),
    "Qwen/Qwen3.5-9B": ModelPolicy("modal", min_capacity=0, max_capacity=2),
    "Qwen/Qwen3.5-9B-Base": ModelPolicy(
        "lambda",
        min_capacity=2,
        max_capacity=4,
        lambda_gpus=["A100-40GB", "A100-80GB", "H100"],
    ),
    "Qwen/Qwen3.8-27B": ModelPolicy("modal", min_capacity=0, max_capacity=2),
    "Qwen/Qwen3.6-35B-A3B": ModelPolicy("modal", min_capacity=0, max_capacity=2),
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": ModelPolicy(
        "modal",
        min_capacity=0,
        max_capacity=4,
        dev_max_capacity=1,
    ),
}


def policy_for(model):
    try:
        return MODEL_POLICIES[model]
    except KeyError:
        raise ValueError(f"No serving provider/capacity policy for {model}") from None


def capacity_for(model, provider):
    policy = policy_for(model)
    if policy.provider == "modal":
        return (policy.min_capacity, policy.max_capacity) if provider == "modal" else (0, 0)
    if provider == "lambda":
        return policy.min_capacity, policy.max_capacity
    if provider == "modal":
        return 0, policy.max_capacity
    return 0, 0


def lambda_enabled():
    return any(p.provider in {"lambda", "all"} for p in MODEL_POLICIES.values())


def modal_gpus_for(model):
    from flash.serving.src.engine.model_config import base_models

    if model not in base_models():
        raise ValueError("Provider policy is only defined for hosted Qwen")
    if model == "Qwen/Qwen3.6-35B-A3B":
        return ["H200"]
    if model == "Qwen/Qwen3.5-0.8B":
        return ["A10G", "A100-80GB"]
    if model == "Qwen/Qwen3.5-4B":
        return ["A10G", "L40S", "A100-80GB"]
    if model in {"Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-9B-Base"}:
        return ["L40S", "A100-80GB"]
    return ["A100-80GB"]
