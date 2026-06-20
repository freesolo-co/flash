"""GPU facts for the cost model: price, VRAM, compute, and cheapest-fit selection.

Price and VRAM come straight from the provider-agnostic registry in
``flash.providers.base`` (the static fallback ``hourly_usd`` -- live market rates
override it in production, but the estimator is deterministic/offline by design).
Compute throughput (peak bf16 tensor TFLOPS) is the one fact the registry doesn't
carry, so it lives here. ``pick_gpu`` reproduces the allocator's selection rule
(cheapest validated class that fits) without a live offer search, so the estimate is
reproducible without provider creds.
"""

from __future__ import annotations

from flash.providers.base import (
    GPU_INFO,
    POLICY_NAMES,
    GpuClass,
    canonical_gpu,
    providers_for,
)

# Peak bf16 dense tensor-core throughput (TFLOPS) per managed class. Vendor spec sheets
# (no 2:4 sparsity). Only relative magnitudes and the calibrated MFU in ``analytical``
# matter for the dollar figure; a missing class falls back to ``_DEFAULT_TFLOPS``.
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


# Realized $/hr a class is ACTUALLY billed at -- the empirical MEDIAN effective rate
# (measured cost / measured wall) per class over the RunPod/Vast runs in
# cost_estimator_results/real_runs/measured_runs.json, NOT the static on-demand list
# snapshot. It is usually below list (the spot/queue discount: RTX 5090 lists $0.99 but
# bills ~$0.87, A100 PCIe lists $1.39 but bills ~$1.04), but it is NOT guaranteed to be --
# a class can bill ABOVE its static list when the live market is tight or surge-priced
# (e.g. RTX A5000 bills ~$0.30 vs a $0.27 list; an H100 GRPO run billed ~$10/hr against a
# $3.29 list). That's exactly why this is calibrated from observed billing rather than
# derived from list +/- a fixed discount. A calibrated price INPUT, not an output
# adjustment -- regenerate with ``calibration.fit_constants`` and refresh here as new runs
# land (pinned by tests). A class without measured runs falls back to the list price (no
# rate invented -- an honest estimate until it's been observed).
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
