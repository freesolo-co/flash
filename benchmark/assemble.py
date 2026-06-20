"""Assemble the Flash vs Tinker comparison from all 6 runs.

Reads benchmark/results/runs_manifest.json, pulls each flash run's status from its
control plane (+ local metrics.json) and each tinker run's result JSON, then writes:
  - benchmark/results/comparison.json   (machine-readable)
  - benchmark/results/comparison.md     (the table that goes in the PR)

Comparison axes (per task, flash vs tinker):
  - PERFORMANCE: mean group reward at step 1 -> final (the GRPO training signal), plus
    flash's native held-out eval reward where available.
  - LATENCY:     total wall-clock, setup/queue time, and per-step time.
  - COST:        flash = measured RunPod $ (billed); tinker = wall-clock proxy
                 (Tinker does not expose per-run cost via API — labelled as an estimate).

Usage:
    uv run python benchmark/assemble.py
    FLASH_PLANE_KEY=<key> uv run python benchmark/assemble.py   # if key not in ~/.flash
"""
from __future__ import annotations

import json
import os
import pathlib
import urllib.request

_HERE = pathlib.Path(__file__).parent
_RESULTS = _HERE / "results"
_MANIFEST = _RESULTS / "runs_manifest.json"
TASKS = ["gsm8k", "reverse-text", "hendrycks-math"]

# Representative on-demand price for the GPU class Tinker is likely serving a 4B on.
# Tinker is managed and does NOT expose per-run cost; this is an explicit wall-clock proxy
# so the two columns are at least order-of-magnitude comparable. Flash cost is MEASURED.
_TINKER_PROXY_USD_PER_HR = 2.00


def _flash_key() -> str:
    key = os.environ.get("FLASH_PLANE_KEY") or os.environ.get("FREESOLO_API_KEY", "")
    if not key:
        cfg = pathlib.Path.home() / ".flash" / "config.json"
        if cfg.exists():
            key = json.loads(cfg.read_text()).get("api_key", "")
    return key


def _get(api_url: str, path: str, key: str) -> dict:
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}{path}", headers={"Authorization": f"Bearer {key}"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _read_flash_metrics(artifacts_dir: str | None) -> dict:
    if not artifacts_dir:
        return {}
    p = pathlib.Path(artifacts_dir) / "seed0" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def collect_flash(task: str, entry: dict, key: str) -> dict:
    """Pull one flash run's status + local metrics into a normalized record."""
    try:
        status = _get(entry["api_url"], f"/v1/runs/{entry['run_id']}", key)
    except Exception as exc:  # noqa: BLE001
        return {"platform": "flash", "task": task, "status": f"unreachable: {exc}"}

    m = _read_flash_metrics(status.get("artifacts_dir"))
    notes = m.get("notes", {})
    rh = notes.get("reward_history", []) or []
    eh = notes.get("eval_history", []) or []
    remote = status.get("remote") or {}

    setup_s = m.get("setup_seconds")
    train_s = m.get("wall_seconds")
    total_s = (setup_s or 0) + (train_s or 0) if (setup_s or train_s) else None

    return {
        "platform": "flash",
        "task": task,
        "status": status.get("state"),
        "run_id": entry["run_id"],
        "gpu": m.get("allocated_gpu") or remote.get("allocated_gpu"),
        "provider": notes.get("provider") or remote.get("provider"),
        "steps": len(rh),
        "first_reward": rh[0] if rh else None,
        "final_reward": rh[-1] if rh else None,
        "eval_reward": (eh[-1].get("eval_reward") if eh else None),
        "eval_n": (eh[-1].get("eval_n") if eh else None),
        "reward_history": rh,
        "cost_usd": m.get("cost_usd") if m.get("cost_usd") is not None else status.get("cost_usd"),
        "cost_kind": "measured (RunPod billed)",
        "setup_s": setup_s,
        "train_s": train_s,
        "total_s": total_s,
        "per_step_s": (train_s / len(rh)) if (train_s and rh) else None,
        "train_tok_per_s": notes.get("train_throughput_toks_per_s") or m.get("train_throughput_toks_per_s"),
    }


def collect_tinker(task: str, rel_path: str) -> dict:
    """Read one tinker run's result JSON into a normalized record."""
    p = _RESULTS / pathlib.Path(rel_path).name
    if not p.exists():
        return {"platform": "tinker", "task": task, "status": "pending (no result file yet)"}
    d = json.loads(p.read_text())
    rh = d.get("reward_history", []) or []
    wall = d.get("wall_s")
    steps = d.get("steps") or len(rh)
    cost = round((wall or 0) / 3600 * _TINKER_PROXY_USD_PER_HR, 4) if wall else None
    return {
        "platform": "tinker",
        "task": task,
        "status": d.get("status"),
        "gpu": "managed (Tinker)",
        "steps": len(rh),
        "first_reward": rh[0] if rh else None,
        "final_reward": rh[-1] if rh else None,
        "eval_reward": None,  # tinker held-out eval not run in-loop; training reward is the signal
        "reward_history": rh,
        "cost_usd": cost,
        "cost_kind": f"ESTIMATE: wall x ${_TINKER_PROXY_USD_PER_HR:.2f}/hr proxy (Tinker cost not in API)",
        "setup_s": None,
        "train_s": wall,
        "total_s": wall,
        "per_step_s": (wall / steps) if (wall and steps) else None,
        "train_tok_per_s": None,
    }


def _fmt(v, nd=3, suffix=""):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}{suffix}"
    return f"{v}{suffix}"


def _secs(v):
    if v is None:
        return "—"
    m, s = divmod(int(v), 60)
    return f"{m}m{s:02d}s"


def _delta(f, t):
    """Format a flash-vs-tinker delta line for a numeric metric (higher flash = positive)."""
    if f is None or t is None:
        return "—"
    return f"{f - t:+.3f}"


def render_markdown(records: dict) -> str:
    """records: {task: {'flash': rec, 'tinker': rec}}"""
    lines = []
    lines.append("# Flash vs Tinker — GRPO benchmark (Qwen3.5-4B, 30 steps)\n")
    lines.append("Same base model, same verifiers environment, same GRPO hyper-parameters "
                 "(group_size=4, batch_size=4, max_tokens=512, 30 steps) on each side.\n")
    lines.append("- **Flash** trains on a rented RunPod **A100 PCIe** (4B GRPO needs ≥35 GB; the "
                 "allocator escalates from the requested RTX 5090). Cost is **measured** (RunPod billed).")
    lines.append("- **Tinker** trains on Thinking Machines' **managed** backend. Per-run cost is "
                 "**not exposed via API**, so its $ column is a wall-clock proxy (labelled).")
    lines.append("- **Performance** = mean group reward over the 30-step GRPO run (step 1 → final). "
                 "Flash also reports a native held-out eval (50 examples).\n")

    # Per-task tables
    for task in TASKS:
        f = records[task]["flash"]
        t = records[task]["tinker"]
        lines.append(f"## {task}\n")
        lines.append("| metric | Flash | Tinker | Δ (Flash − Tinker) |")
        lines.append("|---|---|---|---|")
        lines.append(f"| status | {f.get('status')} | {t.get('status')} | |")
        lines.append(f"| GPU | {f.get('gpu') or '—'} | {t.get('gpu')} | |")
        lines.append(f"| **reward** step 1 | {_fmt(f.get('first_reward'))} | {_fmt(t.get('first_reward'))} | |")
        lines.append(f"| **reward** final | {_fmt(f.get('final_reward'))} | {_fmt(t.get('final_reward'))} | {_delta(f.get('final_reward'), t.get('final_reward'))} |")
        lines.append(f"| reward gain | {_fmt((f.get('final_reward') or 0) - (f.get('first_reward') or 0))} | {_fmt((t.get('final_reward') or 0) - (t.get('first_reward') or 0))} | |")
        lines.append(f"| held-out eval | {_fmt(f.get('eval_reward'))} (n={f.get('eval_n')}) | — | |")
        lines.append(f"| **latency** total | {_secs(f.get('total_s'))} | {_secs(t.get('total_s'))} | |")
        lines.append(f"| latency setup/queue | {_secs(f.get('setup_s'))} | — | |")
        lines.append(f"| latency per-step | {_fmt(f.get('per_step_s'), 1, 's')} | {_fmt(t.get('per_step_s'), 1, 's')} | |")
        lines.append(f"| **cost** | {_fmt(f.get('cost_usd'), 4, ' USD')} | {_fmt(t.get('cost_usd'), 4, ' USD')} | |")
        lines.append(f"| cost basis | {f.get('cost_kind')} | {t.get('cost_kind')} | |")
        lines.append("")

    # Roll-up
    lines.append("## Summary\n")
    lines.append("| task | winner (reward) | flash cost | tinker cost (est) | flash wall | tinker wall |")
    lines.append("|---|---|---|---|---|---|")
    for task in TASKS:
        f = records[task]["flash"]
        t = records[task]["tinker"]
        fr, tr = f.get("final_reward"), t.get("final_reward")
        if fr is None or tr is None:
            winner = "—"
        elif abs(fr - tr) < 1e-6:
            winner = "tie"
        else:
            winner = "Flash" if fr > tr else "Tinker"
        lines.append(
            f"| {task} | {winner} | {_fmt(f.get('cost_usd'), 4, ' USD')} | "
            f"{_fmt(t.get('cost_usd'), 4, ' USD')} | {_secs(f.get('total_s'))} | {_secs(t.get('total_s'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    manifest = json.loads(_MANIFEST.read_text())
    key = _flash_key()
    records: dict = {}
    for task in TASKS:
        records[task] = {
            "flash": collect_flash(task, manifest["flash"][task], key),
            "tinker": collect_tinker(task, manifest["tinker"][task]),
        }

    (_RESULTS / "comparison.json").write_text(json.dumps(records, indent=2))
    md = render_markdown(records)
    (_RESULTS / "comparison.md").write_text(md)
    print(md)
    print(f"\nWrote {_RESULTS/'comparison.json'} and {_RESULTS/'comparison.md'}")


if __name__ == "__main__":
    main()
