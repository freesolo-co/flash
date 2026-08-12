"""Render the self-hosted Modal serving app from the catalog's validated serving config.

The generated file is the user's, not a vendored library: `flash serve setup` writes it into their
repo so they can read and edit it. Everything model-specific is substituted from the catalog rather
than invented, so an OSS deploy starts from the configuration Freesolo validated on real hardware.
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

from flash.core.catalog import ModelInfo
from flash.serve.backend.gpus import MODAL_GPUS_BY_NAME, ModalGpu, default_gpu, serving_dtype

# Matches the version the production serving app runs. vLLM's Qwen3 GDN-hybrid support and the
# multi-LoRA + fp8 path are both version-sensitive, so this is a single pinned constant rather than
# a floating range.
VLLM_VERSION = "0.23.0"

# Modal stops the container after this idle period and the GPU stops costing anything. 5 minutes
# trades a little idle spend for far fewer cold starts than Modal's 60s default.
DEFAULT_SCALEDOWN_WINDOW = 300
# Modal's own bounds for scaledown_window. 0 is NOT "stop immediately" -- it is rejected.
MIN_SCALEDOWN_WINDOW = 2
MAX_SCALEDOWN_WINDOW = 20 * 60

_TEMPLATE = "modal_app.py.tmpl"
# Modal app names allow letters, digits and dashes.
_UNSAFE_NAME = re.compile(r"[^a-z0-9-]+")


def app_name_for(model_id: str) -> str:
    """A Modal app name derived from the base model, e.g. ``flash-serve-qwen3-5-4b``."""
    slug = _UNSAFE_NAME.sub("-", model_id.split("/")[-1].lower()).strip("-")
    return f"flash-serve-{slug}".rstrip("-")


def render_app(
    info: ModelInfo,
    *,
    gpu: ModalGpu | None = None,
    scaledown_window: int = DEFAULT_SCALEDOWN_WINDOW,
    secret_name: str = "flash-serving",
    app_file: str = "flash_serving_app.py",
) -> str:
    """Render the Modal app source for ``info``.

    ``gpu`` defaults to the catalog's production-validated card. Passing a different one is
    supported -- `flash serve gpus` exists to make that an informed choice -- but the engine values
    still come from the catalog, since those are what was validated for this model.
    """
    serving = getattr(info, "serving", None)
    if serving is None:
        raise ValueError(f"{info.id} has no serving configuration in the flash catalog")
    card = gpu or default_gpu(info)
    if card is None:
        raise ValueError(f"{info.id} names serving GPU {serving.gpu!r}, which Modal does not offer")
    # Modal accepts 2s to 20min. Catch it here rather than letting `modal deploy` fail after the
    # file is already written and the user thinks setup succeeded.
    if not MIN_SCALEDOWN_WINDOW <= scaledown_window <= MAX_SCALEDOWN_WINDOW:
        raise ValueError(
            f"scaledown window {scaledown_window}s is outside Modal's supported range "
            f"({MIN_SCALEDOWN_WINDOW}-{MAX_SCALEDOWN_WINDOW} seconds)"
        )

    dtype = serving_dtype(info)
    template = resources.files("flash.serve.backend.templates").joinpath(_TEMPLATE).read_text()
    return template.format(
        app_name=app_name_for(info.id),
        base_model=info.id,
        gpu=card.name,
        # None, not "bf16": vLLM reads `quantization=None` as "load the checkpoint as it is", which
        # is what the 35B MoE needs. Passing a dtype string here would request online quantization.
        quantization='"fp8"' if dtype == "fp8" else "None",
        kv_cache_dtype="fp8",
        max_model_len=serving.max_model_len,
        max_num_seqs=serving.max_num_seqs or 8,
        max_loras=serving.max_loras,
        max_lora_rank=serving.max_lora_rank,
        gpu_memory_utilization=serving.gpu_memory_utilization or 0.90,
        vllm_version=VLLM_VERSION,
        scaledown_window=scaledown_window,
        secret_name=secret_name,
        app_file=app_file,
    )


def write_app(
    info: ModelInfo,
    destination: Path,
    *,
    gpu: ModalGpu | None = None,
    scaledown_window: int = DEFAULT_SCALEDOWN_WINDOW,
    secret_name: str = "flash-serving",
    overwrite: bool = False,
) -> Path:
    """Write the rendered app to ``destination``.

    Refuses to clobber an existing file unless ``overwrite``: the generated app is meant to be
    edited, so silently overwriting it would discard the user's changes.
    """
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists; pass --force to overwrite it")
    source = render_app(
        info,
        gpu=gpu,
        scaledown_window=scaledown_window,
        secret_name=secret_name,
        app_file=destination.name,
    )
    destination.write_text(source)
    return destination


def gpu_named(name: str) -> ModalGpu | None:
    return MODAL_GPUS_BY_NAME.get(str(name or "").strip())
