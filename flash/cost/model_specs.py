"""Model-size facts the cost model needs: total params, MoE active params, quant.

Thin layer over ``flash.catalog`` (the curated ``ModelInfo`` table) and
``flash.engine.vram.params_b_from_str``. Compute cost scales with *active* params
(an MoE only runs a subset of its experts per token), while the on-disk download and
VRAM footprint scale with *total* params -- so the two are tracked separately.
"""

from __future__ import annotations

import re

from flash.catalog import MODELS

# Bytes per parameter for the *download* footprint (always the full bf16 checkpoint;
# 4-bit QLoRA quantizes on the worker after download). Compute FLOPs don't depend on
# the stored dtype, so this is download/VRAM-only.
_DOWNLOAD_BYTES_PER_PARAM = 2.0

# Default when a model's size can't be read (open-model policy, offline HF probe).
DEFAULT_PARAMS_B = 4.0


def _params_str(model_id: str) -> str | None:
    info = MODELS.get(model_id)
    return info.params if info is not None else None


def _params_b_from_str(s: str | None) -> float | None:
    """Leading param count (billions) from a catalog/id string, e.g.
    ``"4.7B (text-only fine-tune)"`` -> 4.7, ``"...-9b-..."`` -> 9.0,
    ``"...-8x7B-..."`` -> 56.0 (MoE total), ``"...-270m"`` -> 0.27 (million suffix).

    Like ``flash.engine.vram.params_b_from_str`` but (a) case-insensitive on the size
    suffix, so a lowercase ``b`` in a model id (``Qwen/Qwen3.5-9b-instruct``) is read
    instead of falling back to ``DEFAULT_PARAMS_B`` and underestimating; (b) it expands an
    ``NxM`` expert-count MoE id (``Mixtral-8x7B`` -> 8x7 = 56B total) instead of reading
    only one expert's size; and (c) it understands an ``M`` (million) suffix so a sub-1B
    checkpoint (``gemma-3-270m``, ``foo-750M``) isn't quoted as the 4B default.
    """
    if not s:
        return None
    # MoE expert count first: "8x7B" -> 56B total (else the plain-B branch reads just 7B).
    moe = re.search(r"([0-9]+)\s*[xX]\s*([0-9]+(?:\.[0-9]+)?)\s*[Bb]\b", s)
    if moe:
        return float(moe.group(1)) * float(moe.group(2))
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[Bb]\b", s)
    if m:
        return float(m.group(1))
    # Million suffix (anchored to a digit so a stray "m" in a word isn't matched).
    mil = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[Mm]\b", s)
    if mil:
        return float(mil.group(1)) / 1000.0
    return None


def total_params_b(model_id: str, params_str: str | None = None) -> float:
    """Total parameter count in billions (the full checkpoint).

    Reads the curated catalog string (e.g. ``"4.7B (text-only fine-tune)"`` -> 4.7).
    Falls back to ``DEFAULT_PARAMS_B`` for unlisted models with no size hint.
    """
    s = params_str if params_str is not None else _params_str(model_id)
    val = _params_b_from_str(s)
    if val is None:
        # Unlisted model with no size hint: try the id itself (e.g. "...-35B-A3B",
        # "...-9b-..." -- case-insensitive so a lowercase suffix isn't dropped).
        val = _params_b_from_str(model_id)
    return float(val) if val else DEFAULT_PARAMS_B


def active_params_b(model_id: str, params_str: str | None = None) -> float:
    """Active parameters per token in billions.

    For a Mixture-of-Experts checkpoint the active count is the figure after the ``A``
    in an id like ``Qwen3.6-35B-A3B`` (35B total, 3B active). Dense models activate
    every parameter, so active == total.
    """
    s = params_str if params_str is not None else _params_str(model_id)
    # MoE "<total>B-A<active>B" appears in the model id and/or the params string.
    for hay in (model_id, s or ""):
        m = re.search(r"[Aa](\d+(?:\.\d+)?)\s*[Bb]\b", hay)
        if m:
            return float(m.group(1))
    return total_params_b(model_id, params_str)


def model_quant(model_id: str) -> str:
    """Quantization of the curated entry (``"bf16"`` or ``"4bit-qlora"``); bf16 default."""
    info = MODELS.get(model_id)
    return (getattr(info, "quant", None) or "bf16") if info is not None else "bf16"


def download_weight_gb(model_id: str, params_str: str | None = None) -> float:
    """Approximate GB pulled from the HF hub at cold start (full bf16 checkpoint)."""
    return total_params_b(model_id, params_str) * _DOWNLOAD_BYTES_PER_PARAM
