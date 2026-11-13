"""model and gpu catalog command handlers."""

from __future__ import annotations

from flash.cli.ui import render, tables
from flash.core.catalog import public_model_rows


def cmd_models(args) -> int:
    rows = public_model_rows()
    if render.styled():
        print(tables.models_table(rows))
        return 0
    for row in rows:
        print(row["id"])
    return 0


def cmd_gpus(args) -> int:
    """List validated managed GPU classes, VRAM, and estimated $/hr."""
    from flash.providers.core.base import GPU_INFO
    from flash.providers.mithril.pricing import SPOT_GPUS
    from flash.providers.modal.pricing import static_rates as modal_static_rates

    modal_rates = modal_static_rates()
    infos = sorted(
        (info for info in GPU_INFO.values() if info.modal_name), key=lambda g: g.hourly_usd
    )
    tip = (
        "Tip: GPU allocation is automatic.\n"
        "The allocator picks the cheapest validated class that fits; GPU classes are not "
        "user-selectable.\n"
        "Set spot = true for Mithril only. Spot rates are fixed estimates and charges.\n"
        "Spot classes: H100, H200 (priority order). One rate each, $2.10 and $2.60 per GPU-hour, "
        "which is also the cap: jobs start at or below it and keep running up to it. A spot run "
        "takes whichever fitting "
        "class comes up first."
    )
    if render.styled():
        rows = [(info.name, info.vram_gb, modal_rates.get(info.name)) for info in infos]
        rows += [
            (name + " (Spot)", GPU_INFO[name].vram_gb, float(shape.hourly_usd))
            for name, shape in SPOT_GPUS.items()
        ]
        print(tables.gpus_table(rows, tip))
        return 0

    def fmt_rate(v: float | None) -> str:
        return f"{v:>10.2f}" if v else f"{'-':>10}"

    print(f"{'gpu':<16}{'vram':>6}{'$/hr':>11}")
    for info in infos:
        modal_rate = modal_rates.get(info.name)
        print(f"{info.name:<16}{info.vram_gb:>5}G{fmt_rate(modal_rate):>11}")
    for name, shape in SPOT_GPUS.items():
        print(f"{name + ' (Spot)':<16}{GPU_INFO[name].vram_gb:>5}G{float(shape.hourly_usd):>11.3f}")
    print(f"\n{tip}")
    return 0
