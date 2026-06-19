"""Render an ``ExperimentResult`` as an ASCII chart, a markdown report, and a PNG.

``matplotlib`` is imported lazily inside ``save_png`` only, so the rest of the module
(ASCII + markdown + JSON) has no plotting dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

from .experiment import ExperimentResult


def _fmt_pct(v: float | None) -> str:
    return f"{v:.1f}%" if v is not None else "n/a"


def ascii_chart(result: ExperimentResult, width: int = 40) -> str:
    """Horizontal bar chart of MAPE per version (lower = better)."""
    mapes = [s.mape for s in result.summaries if s.mape is not None]
    hi = max(mapes) if mapes else 1.0
    hi = hi or 1.0
    lines = ["MAPE by prompt version  (shorter bar = closer to ground truth)", ""]
    for s in result.summaries:
        if s.mape is None:
            bar, val = "", "n/a"
        else:
            fill = max(1, round(s.mape / hi * width)) if s.mape > 0 else 0
            bar, val = "█" * fill, f"{s.mape:.1f}%"
        lines.append(f"{s.label:<14} |{bar} {val}")
    return "\n".join(lines)


def markdown_report(
    result: ExperimentResult, *, title: str = "Flash cost-estimator prompt sweep"
) -> str:
    first, last = result.summaries[0], result.summaries[-1]
    improvement = ""
    if first.mape is not None and last.mape is not None and first.mape > 0:
        drop = (1 - last.mape / first.mape) * 100
        # A sweep need not converge monotonically: if the last version is WORSE than the
        # first, ``drop`` is negative -- report it honestly as an increase, not a reduction.
        if drop >= 0:
            change = f"a **{drop:.0f}% error reduction**"
        else:
            change = f"a **{-drop:.0f}% error increase**"
        improvement = (
            f"\nFrom **{first.label}** ({_fmt_pct(first.mape)} MAPE) to **{last.label}** "
            f"({_fmt_pct(last.mape)} MAPE) — {change} as the prompt "
            f"gains Flash framework knowledge."
        )

    out = [
        f"# {title}",
        "",
        f"Grid: **{result.grid_size} runs** (models x {{SFT, GRPO}} x step counts), "
        f"priced at **{len(result.versions)} prompt versions**. Ground truth is the "
        f"analytical model in `flash.cost.analytical`; error is MAPE (mean absolute "
        f"percentage error) of the LLM's dollar figure vs. that reference." + improvement,
        "",
        "## Accuracy by prompt version",
        "",
        "MAPE = mean absolute percentage error vs. the analytical reference, also split "
        "by method (SFT / GRPO). Mean signed error shows the bias direction "
        "(+ = over-estimate).",
        "",
        "| Version | What it adds | MAPE | SFT | GRPO | Mean signed | GPU pick acc | ok |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    out.extend(
        f"| {s.label} | {_version_blurb(s.version)} | {_fmt_pct(s.mape)} | "
        f"{_fmt_pct(s.mape_sft)} | {_fmt_pct(s.mape_grpo)} | "
        f"{_fmt_signed(s.mean_signed_pct)} | "
        f"{_fmt_pct((s.gpu_accuracy or 0) * 100 if s.gpu_accuracy is not None else None)} | "
        f"{s.n_ok}/{s.n} |"
        for s in result.summaries
    )
    out += ["", "```", ascii_chart(result), "```", ""]

    # Per-config: ground truth vs first/last version side by side.
    out += [
        f"## Per-run: ground truth vs {first.label} vs {last.label}",
        "",
        f"| Run | Truth $ | {first.label} $ | {last.label} $ | Truth GPU | {last.label} GPU |",
        "|---|---:|---:|---:|---|---|",
    ]
    by_label_first = {c.label: c for c in result.cells if c.version == first.version}
    by_label_last = {c.label: c for c in result.cells if c.version == last.version}
    for label in sorted(by_label_last):
        cl = by_label_last[label]
        cf = by_label_first.get(label)
        fv = f"{cf.est_usd:.2f}" if cf and cf.est_usd is not None else "—"
        lv = f"{cl.est_usd:.2f}" if cl.est_usd is not None else "—"
        out.append(
            f"| {label} | {cl.truth_usd:.2f} | {fv} | {lv} | {cl.truth_gpu} | {cl.est_gpu or '—'} |"
        )
    out += [
        "",
        "_Ground truth = `flash.cost.estimate_cost` (wall-clock hours x GPU $/hr, "
        "built on Flash's own pricing/VRAM/allocator primitives). The estimator is "
        "Claude (`claude-opus-4-8`) under each prompt version._",
        "",
    ]
    return "\n".join(out)


_VERSION_BLURBS = {
    1: "base task only",
    2: "+ GPU price/VRAM catalog",
    3: "+ cheapest-fit GPU selection",
    4: "+ per-step compute/throughput model",
    5: "+ GRPO rollout split, MoE, QLoRA",
    6: "+ cold-start + exact cost formula",
}


def _version_blurb(v: int) -> str:
    return _VERSION_BLURBS.get(v, "")


def _fmt_signed(v: float | None) -> str:
    return f"{v:+.1f}%" if v is not None else "n/a"


def save_png(result: ExperimentResult, path: str | Path) -> Path:
    """Write a MAPE-vs-version line chart to ``path`` (lazy matplotlib import)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [s.label for s in result.summaries]

    def _series(attr: str) -> list[float]:
        return [
            getattr(s, attr) if getattr(s, attr) is not None else float("nan")
            for s in result.summaries
        ]

    mapes = _series("mape")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(labels, mapes, marker="o", color="#c1440e", linewidth=2.5, label="all runs", zorder=3)
    ax.plot(labels, _series("mape_sft"), marker="s", color="#2a6f97", linewidth=1.5, label="SFT")
    ax.plot(labels, _series("mape_grpo"), marker="^", color="#b08900", linewidth=1.5, label="GRPO")
    for x, y in zip(labels, mapes, strict=False):
        if y == y:  # not NaN
            ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 9), ha="center")
    ax.set_ylabel("MAPE vs. analytical ground truth (%)")
    ax.set_title("Flash LLM cost estimator: accuracy vs. prompt framework knowledge")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_json(result: ExperimentResult, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(result.to_dict(), indent=2))
    return path
