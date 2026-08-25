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
    from flash.providers.runpod.client.pricing import static_rates as runpod_static_rates

    runpod_rates = runpod_static_rates()
    infos = sorted(
        (info for info in GPU_INFO.values() if info.enum_member), key=lambda g: g.hourly_usd
    )
    tip = (
        "Tip: GPU allocation is automatic by default.\n"
        "The allocator picks the cheapest validated class that fits. Pin a specific class by "
        'adding type = "<CLASS>" to the [gpu] section.'
    )
    if render.styled():
        rows = [(info.name, info.vram_gb, runpod_rates.get(info.name)) for info in infos]
        print(tables.gpus_table(rows, tip))
        return 0

    def fmt_rate(v: float | None) -> str:
        return f"{v:>10.2f}" if v else f"{'-':>10}"

    print(f"{'gpu':<16}{'vram':>6}{'$/hr':>11}")
    for info in infos:
        runpod_rate = runpod_rates.get(info.name)
        print(f"{info.name:<16}{info.vram_gb:>5}G{fmt_rate(runpod_rate):>11}")
    print(f"\n{tip}")
    return 0
