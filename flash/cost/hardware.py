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
    """Static fallback $/hr for a class (live pricing overrides this in production)."""
    info = GPU_INFO.get(name)
    if info is None:
        raise KeyError(f"unknown GPU class {name!r}")
    return info.hourly_usd


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
    if not candidates:
        raise ValueError(
            f"no GPU class fits >= {required_vram_gb} GB (allow_unvalidated={allow_unvalidated})"
        )
    best = min(candidates, key=lambda g: (g.hourly_usd, g.vram_gb, g.name))
    return best.name
