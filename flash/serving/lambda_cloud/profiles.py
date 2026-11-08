"""Hardware-specific FP8 settings; public model identity and admission stay unchanged."""

from flash.serving.app.inference_provider import modal_gpus_for
from flash.serving.src.engine.model_config import engine_overrides_for

from .config import placements


def engine_profile(model, gpu):
    if gpu not in placements(model) and gpu not in modal_gpus_for(model):
        raise ValueError(f"Unapproved serving GPU {gpu} for {model}")
    overrides = engine_overrides_for(model)
    small = model in {"Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B"}
    online_nine = gpu in {"A10", "A100-80GB"} and model in {
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-9B-Base",
    }
    if small or online_nine:
        gate = "ba" if model == "Qwen/Qwen3.5-4B" else "(a|b|ba)"
        overrides.update(
            serve_model_id=model,
            model_revision=overrides["tokenizer_revision"],
            quantization="fp8_per_tensor",
            quantization_config={
                "ignore": ["re:.*visual.*", "re:.*lm_head$", f"re:.*in_proj_{gate}$"]
            },
        )
    if model == "Qwen/Qwen3.6-35B-A3B":
        overrides.update(
            serve_model_id=model + "-FP8",
            model_revision="c653bba46dfa06e9658dcc0b0dc87fe4685f14a2",
            quantization="fp8",
        )
    return overrides
