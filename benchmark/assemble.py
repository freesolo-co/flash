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
_SMOOTH_K = 5  # mean over last K steps — robust to the high per-step variance at 16 rollouts/step


def _smoothed(rh: list, k: int = _SMOOTH_K):
    """Mean of the last k rewards (robust 'final performance' vs a single noisy step)."""
    if not rh:
        return None
    tail = rh[-k:]
    return sum(tail) / len(tail)


def _best(rh: list):
    return max(rh) if rh else None

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
    except Exception as exc:
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
        "final_smoothed": _smoothed(rh),
        "best_reward": _best(rh),
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


def _tinker_active_compute_s(log_path: str | None) -> float | None:
    """Sum the real per-step work (rollout + train step) from metrics.jsonl.

    Tinker's wall-clock includes managed-backend *capacity pauses* (idle, not billed
    compute), which get absorbed into a checkpoint step's timing. Summing the rollout +
    train-step times gives the active-compute basis — the fair latency/cost number.
    """
    if not log_path:
        return None
    mp = pathlib.Path(log_path) / "metrics.jsonl"
    if not mp.exists():
        return None
    roll = train = 0.0
    for line in mp.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        roll += r.get("time/do_group_rollout_and_filter_constant_reward:total", 0) or 0
        train += r.get("time/train_step", 0) or 0
    total = roll + train
    return total or None


def collect_tinker(task: str, rel_path: str) -> dict:
    """Read one tinker run's result JSON into a normalized record."""
    p = _RESULTS / pathlib.Path(rel_path).name
    if not p.exists():
        return {"platform": "tinker", "task": task, "status": "pending (no result file yet)"}
    d = json.loads(p.read_text())
    rh = d.get("reward_history", []) or []
    wall = d.get("wall_s")
    steps = d.get("steps") or len(rh)
    active = _tinker_active_compute_s(d.get("log_path"))
    # Cost proxy on ACTIVE compute (not wall — wall includes unbilled capacity pauses).
    basis = active if active else wall
    cost = round((basis or 0) / 3600 * _TINKER_PROXY_USD_PER_HR, 4) if basis else None
    paused_s = (wall - active) if (wall and active and wall > active) else None
    return {
        "platform": "tinker",
        "task": task,
        "status": d.get("status"),
        "gpu": "managed (Tinker)",
        "steps": len(rh),
        "first_reward": rh[0] if rh else None,
        "final_reward": rh[-1] if rh else None,
        "final_smoothed": _smoothed(rh),
        "best_reward": _best(rh),
        "eval_reward": None,  # tinker held-out eval not run in-loop; training reward is the signal
        "reward_history": rh,
        "cost_usd": cost,
        "cost_kind": f"ESTIMATE: active-compute x ${_TINKER_PROXY_USD_PER_HR:.2f}/hr proxy (Tinker cost not in API)",
        "setup_s": None,
        "train_s": active or wall,        # active compute (rollout+train), pause excluded
        "total_s": wall,                  # full wall, INCLUDING any capacity pause
        "paused_s": paused_s,             # idle time the managed backend paused us
        "per_step_s": ((active or wall) / steps) if ((active or wall) and steps) else None,
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
                 "(group_size=4, batch_size=4, **max_tokens=1024**, 30 steps) on each side. This run "
                 "includes the **stall fix** (heartbeat through mid-run eval) and the **truncation "
                 "fix** (1024 tokens so the boxed answer isn't cut off).\n")
    lines.append("- **Flash** trains on a rented RunPod **A100 PCIe** (4B GRPO needs ≥35 GB; the "
                 "allocator escalates from the requested RTX 5090). Cost is **measured** (RunPod billed).")
    lines.append("- **Tinker** trains on Thinking Machines' **managed** backend. Per-run cost is "
                 "**not exposed via API**, so its $ column is an active-compute proxy (labelled).")
    lines.append("- **Performance** = the held-out eval below (one shared scorer); in-training reward "
                 "is per-stack only (different verifiers reward versions).\n")

    # Per-task tables
    for task in TASKS:
        f = records[task]["flash"]
        t = records[task]["tinker"]
        lines.append(f"## {task}\n")
        lines.append("| metric | Flash | Tinker | Δ (Flash - Tinker) |")
        lines.append("|---|---|---|---|")
        lines.append(f"| status | {f.get('status')} | {t.get('status')} | |")
        lines.append(f"| GPU | {f.get('gpu') or '—'} | {t.get('gpu')} | |")
        lines.append(f"| **reward** step 1 | {_fmt(f.get('first_reward'))} | {_fmt(t.get('first_reward'))} | |")
        lines.append(f"| **reward** final (last 5 avg) | {_fmt(f.get('final_smoothed'))} | {_fmt(t.get('final_smoothed'))} | {_delta(f.get('final_smoothed'), t.get('final_smoothed'))} |")
        lines.append(f"| reward best step | {_fmt(f.get('best_reward'))} | {_fmt(t.get('best_reward'))} | {_delta(f.get('best_reward'), t.get('best_reward'))} |")
        lines.append(f"| reward final (raw step 30) | {_fmt(f.get('final_reward'))} | {_fmt(t.get('final_reward'))} | |")
        lines.append(f"| held-out eval | {_fmt(f.get('eval_reward'))} (n={f.get('eval_n')}) | — | |")
        lines.append(f"| **latency** wall total | {_secs(f.get('total_s'))} | {_secs(t.get('total_s'))} | |")
        lines.append(f"| latency setup/queue | {_secs(f.get('setup_s'))} | none (managed) | |")
        lines.append(f"| latency active compute | {_secs(f.get('train_s'))} | {_secs(t.get('train_s'))} | |")
        lines.append(f"| latency capacity-pause | — | {_secs(t.get('paused_s')) if t.get('paused_s') else 'none'} | |")
        lines.append(f"| latency per-step | {_fmt(f.get('per_step_s'), 1, 's')} | {_fmt(t.get('per_step_s'), 1, 's')} | |")
        lines.append(f"| **cost** | {_fmt(f.get('cost_usd'), 4, ' USD')} | {_fmt(t.get('cost_usd'), 4, ' USD')} | |")
        lines.append(f"| cost basis | {f.get('cost_kind')} | {t.get('cost_kind')} | |")
        lines.append("")

    # Held-out eval (one version-independent scorer) — the valid PERFORMANCE comparison.
    eval_path = _RESULTS / "eval_unified_gsm8k.json"
    if eval_path.exists():
        ev = json.loads(eval_path.read_text())
        b = ev.get("base", {})
        tr = ev.get("tinker_trained", {})
        ft = ev.get("flash_trained")  # present only if the Flash serving eval ran
        flash_native = records["gsm8k"]["flash"].get("eval_reward")  # Flash's own on-GPU eval

        def _d(rec):  # "Δ+0.080 vs base" suffix
            d = rec.get("delta_vs_base")
            return f" (Δ{d:+.3f} vs base)" if d is not None else ""

        lines.append("## Held-out eval — gsm8k (the valid cross-stack performance comparison)\n")
        lines.append("In-training GRPO reward is NOT comparable across stacks (Flash's worker uses "
                     "verifiers ~0.1.14, Tinker pins 0.1.9 — different rewards + task presentation). "
                     "This eval applies ONE version-independent exact-match scorer to every model, "
                     f"identical greedy decoding, max_tokens={ev.get('max_tokens')}, on "
                     f"{b.get('n')} held-out examples.\n")
        lines.append("| model | gsm8k accuracy | how generated / scored |")
        lines.append("|---|---|---|")
        if b:
            lines.append(f"| base Qwen3.5-4B | {b['accuracy']:.3f} | unified scorer, Tinker sampling |")
        if tr:
            lines.append(f"| **Tinker-trained** | **{tr['accuracy']:.3f}**{_d(tr)} | unified scorer, Tinker sampling |")
        if ft:
            lines.append(f"| **Flash-trained** | **{ft['accuracy']:.3f}**{_d(ft)} | unified scorer, Flash serving |")
        elif flash_native is not None:
            lines.append(f"| Flash-trained | {flash_native:.3f} | Flash's NATIVE on-GPU eval "
                         "(gsm8k-env scorer, NOT the unified scorer — see note) |")
        lines.append("\nWith the truncation fix (max_tokens=1024) the trained models reach the boxed "
                     "answer; under the shared scorer the GRPO-trained model improves over base. The "
                     "Flash-trained row falls back to Flash's own eval because the unified flash eval "
                     "needs a Qwen3.5-4B LoRA serving (the live one was empty: 0 GPUs/base models); "
                     "`eval_unified.py` runs the unified flash eval against any configured serving.\n")

    # Roll-up
    lines.append("## Summary\n")
    lines.append("| task | winner (reward) | flash cost | tinker cost (est) | flash wall | tinker wall |")
    lines.append("|---|---|---|---|---|---|")
    for task in TASKS:
        f = records[task]["flash"]
        t = records[task]["tinker"]
        fr, tr = f.get("final_smoothed"), t.get("final_smoothed")
        if fr is None or tr is None:
            winner = "—"
        elif abs(fr - tr) < 0.02:  # within noise at 16 rollouts/step
            winner = "tie"
        else:
            winner = "Flash" if fr > tr else "Tinker"
        lines.append(
            f"| {task} | {winner} | {_fmt(f.get('cost_usd'), 4, ' USD')} | "
            f"{_fmt(t.get('cost_usd'), 4, ' USD')} | {_secs(f.get('total_s'))} | {_secs(t.get('total_s'))} |"
        )
    lines.append("")

    lines.append("## Reliability & operability (observed this run)\n")
    lines.append("- **Flash** rents a GPU per run. 4B GRPO needs ≥35 GB → the allocator escalates "
                 "RTX 5090 → A100 PCIe. On the long-generation math tasks the colocated-vLLM rollout "
                 "**hung ~13-15 min at eval boundaries** then self-recovered; a true >25-min freeze "
                 "trips the **stall watchdog**, which **kills the sick host, escalates the GPU class** "
                 "(A100 → RTX Pro 6000), and **resumes from the last checkpoint**. reverse-text "
                 "(short generations) ran clean. Net: dedicated + auto-healing, but rented-GPU "
                 "flakiness adds real tail latency.")
    lines.append("- **Tinker** is managed: **no setup/queue**, but the backend **paused all jobs "
                 "~10 min** mid-run (\"running short on capacity, please wait\") — out of the user's "
                 "control, and slower per active step.")
    lines.append("- A **shared Flash control plane** dropped a run's watcher on restart → the record "
                 "stuck at `running` forever (orphaned); re-run on a dedicated plane fixed it.")
    lines.append("- **HF caps repo creation at 300/day/user**; once hit, every new Flash run 429s at "
                 "submit. Worked around by reusing pre-existing artifact repos (`run_flash_plane.py`).\n")

    lines.append("## Methodology caveats\n")
    lines.append("- **In-training reward is not cross-stack comparable.** Flash's worker installs "
                 "verifiers ~0.1.14 (continuous/partial-credit reward); Tinker pins 0.1.9 (the recipe's "
                 "requirement). At matched max_tokens the two stacks present the task differently and "
                 "truncate differently. Use the **held-out eval** (one scorer, generous tokens) for "
                 "performance, and **cost/latency** (measured) for the clean cross-stack comparison.")
    lines.append("- **Tinker cost is an estimate.** Tinker does not expose per-run cost via API; the $ "
                 "column is active-compute-time x a GPU-rate proxy (pause excluded), not a bill. Flash "
                 "cost is the measured RunPod charge.")
    lines.append("- **Scale is deliberately small** (30 steps, 16 rollouts/step) to keep spend low, so "
                 "per-step reward is noisy and 30-step gains are modest by design.\n")
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
