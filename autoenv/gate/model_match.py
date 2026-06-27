"""Map a paper's base model to a Flash catalog model (or the nearest-size substitute).

Flash's config path hard-codes ``model_policy="catalog"`` (``flash/schema`` sets it
unconditionally), so a managed run can only train a curated catalog model — the open-model
``"allow"`` path is unreachable from a TOML config. The gate therefore resolves the case to a
catalog model: the case's ``flash_model`` verbatim if it is in the catalog and supports the
algorithm, else the nearest-size catalog entry (a substitution recorded in the gate report
and made fair downstream by improvement-normalized scoring).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from flash.catalog import MODELS, ModelInfo, list_models, validate_model_for_algorithm

# Parameter size embedded in a model name, e.g. "Qwen2.5-1.5B", "Llama-3.2-3B-Instruct".
_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*([bm])\b", re.IGNORECASE)


@dataclass(frozen=True)
class ModelMatch:
    """The catalog model chosen for a case, and how it was chosen."""

    model_id: str
    exact: bool  # the requested model was itself a catalog entry
    substituted: bool  # a nearest-size catalog model was chosen instead
    detail: str


def params_b_from_name(name: str) -> float | None:
    """Best-effort parameter count (in billions) parsed from a model name; None if absent."""
    match = _SIZE.search(name or "")
    if not match:
        return None
    value = float(match.group(1))
    return value / 1000.0 if match.group(2).lower() == "m" else value


def nearest_by_size(params_b: float, algorithm: str) -> ModelInfo:
    """The algorithm-capable catalog model whose ``params_b`` is closest to ``params_b``."""
    candidates = [m for m in list_models() if _supports(m, algorithm) and m.params_b > 0]
    if not candidates:
        raise ValueError(f"no catalog model supports algorithm {algorithm!r}")
    return min(candidates, key=lambda m: abs(m.params_b - params_b))


def _supports(info: ModelInfo, algorithm: str) -> bool:
    required = "grpo" if algorithm == "grpo" else "sft"
    return required in info.algos


def resolve_flash_model(flash_model: str, base_model_paper: str, algorithm: str) -> ModelMatch:
    """Resolve the catalog model to actually train for this case.

    Priority: (1) the case's ``flash_model`` if it is a catalog entry that supports the
    algorithm; (2) the nearest-size catalog substitute, sized from the case's ``flash_model``
    name, else the paper's base-model name; (3) the catalog default surfaced as an error hint
    when no size can be inferred.
    """
    if flash_model in MODELS:
        try:
            validate_model_for_algorithm(flash_model, algorithm)
        except ValueError as exc:
            raise ValueError(f"{flash_model} cannot run {algorithm}: {exc}") from exc
        return ModelMatch(
            model_id=flash_model,
            exact=True,
            substituted=False,
            detail=f"{flash_model} is a Flash catalog model supporting {algorithm}",
        )

    size = params_b_from_name(flash_model) or params_b_from_name(base_model_paper)
    if size is None:
        allowed = ", ".join(sorted(MODELS))
        raise ValueError(
            f"{flash_model!r} is not a Flash catalog model and no size could be inferred "
            f"from it or the paper base {base_model_paper!r}; pick one of: {allowed}"
        )
    chosen = nearest_by_size(size, algorithm)
    return ModelMatch(
        model_id=chosen.id,
        exact=False,
        substituted=True,
        detail=(
            f"{flash_model!r} not in catalog; substituted nearest-size {chosen.id} "
            f"({chosen.params_b:.1f}B vs requested ~{size:.1f}B)"
        ),
    )
