"""Static lookup facts for the cost model: GPU price/VRAM/compute + cheapest-fit
selection, model size/quant, and reward-grader latency. Pure tables + accessors."""

from __future__ import annotations

import re

from flash.catalog import MODELS
from flash.providers.base import GPU_INFO, POLICY_NAMES, GpuClass, canonical_gpu, providers_for

# ===== GPU facts (price / VRAM / compute / cheapest-fit selection) =====
GPU_COMPUTE_TFLOPS: dict[str, float] = {
    "RTX A4000": 77.0,
    "RTX 2000 Ada": 89.0,
    "RTX A4500": 89.0,
    "RTX 4000 Ada": 90.0,
    "RTX A5000": 89.0,
    "RTX 3090": 71.0,
    "L4": 60.0,
    "RTX Pro 4000": 95.0,
    "RTX 4090": 165.0,
    "RTX 5090": 210.0,
    "RTX A6000": 155.0,
    "A40": 150.0,
    "RTX 6000 Ada": 182.0,
    "L40S": 181.0,
    "A100 SXM 40GB": 312.0,
    "A100 PCIe": 312.0,
    "A100 SXM": 312.0,
    "H100 NVL": 835.0,
    "H100": 990.0,
    "RTX Pro 6000": 250.0,
    "RTX Pro 6000 WK": 250.0,
}
_DEFAULT_TFLOPS = 100.0


def gpu_tflops(name: str) -> float:
    """Peak bf16 tensor TFLOPS for a managed GPU class."""
    return GPU_COMPUTE_TFLOPS.get(name, _DEFAULT_TFLOPS)


def gpu_hourly_usd(name: str) -> float:
    """Static fallback (on-demand list) $/hr for a class."""
    info = GPU_INFO.get(name)
    if info is None:
        raise KeyError(f"unknown GPU class {name!r}")
    return info.hourly_usd


# Realized $/hr a class is ACTUALLY billed at -- the empirical effective rate (measured cost
# / measured wall) over the RunPod/Vast runs in cost_estimator_results/real_runs/, NOT the
# static on-demand list. Usually below list (the spot/queue discount: RTX 5090 lists $0.99
# but bills ~$0.87, A100 PCIe $1.39 -> ~$1.04) but not always (A5000 ~$0.30 vs $0.27 list;
# an H100 GRPO run billed ~$10/hr vs $3.29 list) -- so it's calibrated from observed billing,
# not list +/- a discount. A calibrated price INPUT, not an output adjustment.
#
# Deliberately a single conservative rate per class, NOT method-specific. The same nominal
# card does bill more for GRPO (it rents a pricier high-VRAM instance for the vLLM rollout --
# a 3090 bills ~$0.24/hr SFT vs ~$0.95 GRPO), but pricing GRPO at that full rate over-quotes
# the *total* and worsens per-run accuracy (the low rate was canceling a slightly-high wall
# estimate). With a right-skewed cost distribution this single-rate model is per-run accurate
# AND under-quotes the aggregate -- the conservative side we prefer. KNOWN LIMITATION: it
# under-prices GRPO on consumer cards (3090/4090); tightening that needs a joint rate+wall
# recalibration once there are enough consumer-card GRPO runs to fit both without regressing.
REALIZED_HOURLY_USD: dict[str, float] = {
    "RTX 3090": 0.239,
    "RTX 4090": 0.426,
    "RTX 5090": 0.871,
    "RTX A5000": 0.304,
    "RTX 6000 Ada": 0.601,
    "A100 PCIe": 1.035,
    "A100 SXM": 1.133,
    "H100": 10.037,
}


def realized_hourly_usd(name: str) -> float:
    """Market (spot/queue) $/hr a class is billed at; list price if not yet observed."""
    return REALIZED_HOURLY_USD.get(name, gpu_hourly_usd(name))


def gpu_vram_gb(name: str) -> int:
    info = GPU_INFO.get(name)
    if info is None:
        raise KeyError(f"unknown GPU class {name!r}")
    return info.vram_gb


def pick_gpu(
    required_vram_gb: int,
    *,
    pin: str | None = None,
    provider: str | None = None,
    allow_unvalidated: bool = False,
    pin_must_fit: bool = True,
) -> str:
    """Cheapest GPU class that fits ``required_vram_gb``.

    Mirrors ``flash.providers.allocator.allocate``'s ranking -- order by
    (``hourly_usd``, ``vram_gb``) so equal-priced classes prefer the smaller card --
    but over the static registry only (no live offer search), so it's deterministic
    and needs no provider credentials. By default only validated classes are
    considered (the allocator's default pool); ``allow_unvalidated`` widens it.

    A concrete ``pin`` is normalized to its canonical class via ``canonical_gpu`` (so the
    same aliases the allocator accepts -- e.g. ``"4090"``, ``"a5000"`` -- resolve here
    too), and an unknown GPU name raises ``UnsupportedGpuError`` rather than silently
    falling back to the cheapest class and underquoting. The pin is honored only if it
    actually fits; otherwise selection escalates to the cheapest fitting class (the
    allocator's "one spot larger" behavior). Policy sentinels (``"auto"``/``"cheapest"``,
    or an empty/``None`` value) are NOT GPU names -- they mean "let the allocator pick",
    so they fall through to cheapest-fit selection instead of raising (mirrors how
    ``resolve_gpu_policy`` only canonicalizes a non-policy value).

    ``pin_must_fit`` is True for a FORWARD estimate (a too-small pin escalates -- you
    can't run on a card that doesn't fit). It is set False only when GRADING a MEASURED
    run, where the pin is the card the run *actually ran on*: that's proof it fit, so the
    pin is honored even when the offline VRAM heuristic over-estimates the requirement and
    would otherwise drop it. Without this, a real RTX-5090 GRPO row whose heuristic VRAM
    just exceeds 32 GB is silently re-priced on a cheaper, larger card (e.g. A40) -- so the
    measured 5090 bill is compared against a *different* GPU's price, corrupting the
    calibration accuracy/bias. Only a concrete, known pin is force-honored; sentinels/None
    still auto-select (a measured row always records a concrete card, so this never invents
    one).

    When ``provider`` pins a substrate ("runpod"/"vast"), candidates are restricted to
    classes the provider can PROVISION (``providers_for`` -- RunPod has the ``GpuType``
    enum_member, Vast has the ``vast_name``), mirroring the allocator's per-provider filter
    (it walks ``provider.gpu_classes()``) -- so e.g. a Vast-only class isn't priced as a
    RunPod pick. This provisionability filter holds even with ``allow_unvalidated=True``:
    that flag only relaxes the validation-status gate (it lets unvalidated-but-provisionable
    classes through), it does NOT let a class be quoted on a substrate that can't provision
    it. Provisionability is NOT ``validated_on``: a class can be provisionable on a provider
    yet validated only elsewhere (e.g. the RTX 3090 is RunPod-provisionable but vast-
    validated), and ``allow_unvalidated=True`` brings those into the pinned provider's pool.
    """

    def _selectable(g: GpuClass) -> bool:
        # Two independent gates, mirroring ``allocator.allocate``: a per-provider
        # PROVISIONABILITY filter AND a validation-status filter. ``allow_unvalidated``
        # only relaxes the latter -- it never lets a class be priced on a substrate that
        # can't serve it. So a Vast-only class is never quoted under ``provider="runpod"``,
        # even with ``allow_unvalidated=True``. Provisionability is membership in the
        # provider's own ``gpu_classes()`` (RunPod = has a ``GpuType`` enum_member; Vast =
        # has a ``vast_name``), exposed by ``providers_for`` -- NOT ``validated_on``. The two
        # diverge: e.g. the RTX 3090 (enum_member set, validated only on vast) is RunPod-
        # provisionable-but-unvalidated, so the allocator prices it under ``provider="runpod",
        # allow_unvalidated=True`` -- and the estimator must match, not exclude it.
        if provider not in (None, "auto") and provider not in providers_for(g.name):
            return False
        if allow_unvalidated:
            return True
        if provider in (None, "auto"):
            return g.validated
        return provider in g.validated_on

    candidates = [
        g for g in GPU_INFO.values() if g.vram_gb >= required_vram_gb and _selectable(g)
    ]
    # A policy sentinel (auto/cheapest/empty/None) is not a GPU pin -- it means "auto-
    # select", so leave the candidate set as-is. Only a concrete GPU name is canonicalized
    # (and an unknown one raises UnsupportedGpuError rather than silently underquoting).
    pin_key = (pin or "").strip().lower()
    if pin_key and pin_key not in POLICY_NAMES:
        canonical = canonical_gpu(pin)  # raises UnsupportedGpuError for an unknown pin
        pinned = [g for g in candidates if g.name == canonical]
        if pinned:
            candidates = pinned
        elif not pin_must_fit and _selectable(GPU_INFO[canonical]):
            # Grading a measured run: the pin is the card the run demonstrably ran on, so the
            # offline VRAM heuristic over-estimated the requirement and dropped a card that
            # actually fit. Force the recorded class back in rather than re-pricing the bill on
            # a different GPU. ``pin_must_fit=False`` bypasses ONLY the VRAM-fit gate -- the
            # provider/validation gates in ``_selectable`` still apply, so a pin the pinned
            # provider can't provision (or that policy forbids) is NOT silently force-honored;
            # such a row falls through to cheapest-fit (or the no-candidate error) instead.
            # canonical_gpu already validated it's a real managed class.
            return GPU_INFO[canonical].name
    if not candidates:
        raise ValueError(
            f"no GPU class fits >= {required_vram_gb} GB (allow_unvalidated={allow_unvalidated})"
        )
    best = min(candidates, key=lambda g: (g.hourly_usd, g.vram_gb, g.name))
    return best.name


# ===== Model-size facts (params / quant / download) =====
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


# ===== Reward-grader latency (GRPO) =====
# Seconds to grade one completion, by reward "weight".
REWARD_TIERS: dict[str, float] = {
    "trivial": 0.01,  # exact-match / regex / numeric check (gsm8k, math, bench)
    "light": 0.15,  # parsing + multi-field scoring, classification
    "medium": 0.6,  # light tool/eval, multi-step verifier
    "heavy": 3.0,  # sandboxed code execution, LLM-as-judge, web/tool rollout
}

# Environment-slug keyword -> tier (first match wins; checked in order).
_ENV_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("code", "swe", "exec", "contest", "leetcode"), "heavy"),
    (("judge", "chat", "ticket", "support", "rubric", "dialog"), "heavy"),
    (("search", "browse", "web", "tool", "agent", "linkd"), "medium"),
    (("sentiment", "classif", "extract", "ner", "label"), "light"),
    (("math", "gsm", "arith", "hendrycks", "aime", "bench", "mmlu"), "trivial"),
)

DEFAULT_REWARD_SECONDS = 0.3  # unknown env: assume a light-to-medium grader


def reward_seconds_per_completion(environment: str | None, override: float | None = None) -> float:
    """Per-completion reward latency (s): explicit override, else inferred from the env."""
    if override is not None:
        return max(0.0, override)
    if not environment:
        return DEFAULT_REWARD_SECONDS
    slug = environment.lower()
    for keywords, tier in _ENV_KEYWORDS:
        if any(k in slug for k in keywords):
            return REWARD_TIERS[tier]
    return DEFAULT_REWARD_SECONDS
