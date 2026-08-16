"""Render the self-hosted Modal serving app from the catalog's validated serving config.

The generated file is the user's, not a vendored library: `flash serve setup` writes it into their
repo so they can read and edit it. Everything model-specific is substituted from the catalog rather
than invented, so an OSS deploy starts from the configuration Freesolo validated on real hardware.
"""

from __future__ import annotations

import re
import shlex
from importlib import metadata, resources
from pathlib import Path

from flash._internal.channel import DIST_NAME
from flash.content.multimodal import _IMAGE_BLOCK_TYPES
from flash.core.catalog import ModelInfo
from flash.serve.backend.gpus import MODAL_GPUS_BY_NAME, ModalGpu, default_gpu
from flash.serve.contract import ADAPTER_REVISION_PATTERN, REFERENCE_SERVING_CAPABILITIES

# Modal stops the container after this idle period and the GPU stops costing anything. 5 minutes
# trades a little idle spend for far fewer cold starts than Modal's 60s default.
DEFAULT_SCALEDOWN_WINDOW = 300
# Modal's own bounds for scaledown_window. 0 is NOT "stop immediately" -- it is rejected.
MIN_SCALEDOWN_WINDOW = 2
MAX_SCALEDOWN_WINDOW = 20 * 60

_TEMPLATE = "modal_app.py.tmpl"
_CONFIG_MARKER = "# flash generated config"
# modal app names allow letters, digits and dashes.
_UNSAFE_NAME = re.compile(r"[^a-z0-9-]+")


def flash_requirement() -> str:
    """The pip requirement installing this exact Flash build with the serving runtime.

    Derived from installed metadata rather than hardcoded, so a dev-channel CLI generates an app
    pinned to the dev-channel distribution instead of silently installing the production one.
    `DIST_NAME` is what makes that true: the two channels ship under different distribution names,
    so reading the production name unconditionally would fail to resolve on the dev channel.
    """
    distribution = metadata.distribution(DIST_NAME)
    return f"{distribution.metadata['Name']}[serve-runtime]=={distribution.version}"


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

    template = resources.files("flash.serve.backend.templates").joinpath(_TEMPLATE).read_text()
    if template.count(_CONFIG_MARKER) != 1:
        raise RuntimeError(f"{_TEMPLATE} must contain exactly one generated config marker")
    config = {
        "APP_NAME": app_name_for(info.id),
        "BASE_MODEL": info.id,
        "GPU": card.name,
        "QUANTIZATION": serving.quantization,
        "KV_CACHE_DTYPE": "fp8",
        "MAX_MODEL_LEN": serving.max_model_len,
        "MAX_NUM_SEQS": serving.max_num_seqs or 8,
        "MAX_NUM_BATCHED_TOKENS": serving.max_num_batched_tokens or 0,
        "MAX_LORAS": serving.max_loras,
        "MAX_LORA_RANK": serving.max_lora_rank,
        "GPU_MEMORY_UTILIZATION": serving.gpu_memory_utilization or 0.90,
        "FLASH_REQUIREMENT": flash_requirement(),
        "SCALEDOWN_WINDOW_SECONDS": scaledown_window,
        "SECRET_NAME": secret_name,
        "DEPLOY_COMMAND": f"modal deploy {shlex.quote(app_file)}",
        "IMAGE_BLOCK_TYPES": tuple(sorted(_IMAGE_BLOCK_TYPES)),
        "ADAPTER_REVISION_PATTERN": ADAPTER_REVISION_PATTERN,
        "SERVING_CAPABILITIES": REFERENCE_SERVING_CAPABILITIES,
    }
    generated_config = "\n".join(f"{name} = {value!r}" for name, value in config.items())
    return template.replace(_CONFIG_MARKER, generated_config)


def write_app(
    info: ModelInfo,
    destination: Path,
    *,
    gpu: ModalGpu | None = None,
    scaledown_window: int = DEFAULT_SCALEDOWN_WINDOW,
    secret_name: str = "flash-serving",
    overwrite: bool = False,
) -> Path:
    """Write the rendered app to ``destination``, creating its parent directory.

    Refuses to clobber an existing file unless ``overwrite``: the generated app is meant to be
    edited, so silently overwriting it would discard the user's changes.

    The parent is created because `--output deploy/generated/app.py` is a destination the user
    named, not a path they claimed already exists -- and without this it fails on a bare errno 2
    that names the file rather than the missing directory. Only the parent, and only after the
    clobber check, so a bad path cannot leave directories behind on a run that then refuses.
    """
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists; pass --force to overwrite it")
    source = render_app(
        info,
        gpu=gpu,
        scaledown_window=scaledown_window,
        secret_name=secret_name,
        app_file=str(destination),
    )
    if destination.parent != Path():
        destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source)
    return destination


def gpu_named(name: str) -> ModalGpu | None:
    return MODAL_GPUS_BY_NAME.get(str(name or "").strip())
