"""Deployment-time provider policy for hosted Qwen; DeepSeek is independent."""

INFERENCE_PROVIDER = "modal"  # "modal", "lambda", or "all"; redeploy after changing.


def validate_provider(provider):
    if provider not in {"modal", "lambda", "all"}:
        raise ValueError("INFERENCE_PROVIDER must be modal, lambda, or all")
    return provider


def replica_limit(provider):
    return 1 if validate_provider(provider) == "all" else 2


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


validate_provider(INFERENCE_PROVIDER)
