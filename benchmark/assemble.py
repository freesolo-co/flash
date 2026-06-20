"""Assemble the Flash vs Tinker comparison from all 6 runs.

Reads benchmark/results/runs_manifest.json, pulls each flash run's status from its
control plane (+ local metrics.json) and each tinker run's result JSON, then writes:
  - benchmark/results/comparison.json   (machine-readable)
  - benchmark/results/comparison.md     (the table that goes in the PR)

Comparison axes (per task, flash vs tinker):
  - PERFORMANCE: mean group reward at step 1 -> final (the GRPO training signal), plus
    flash's native held-out eval reward where available.
  - LATENCY:     total wall-clock, setup/queue time, and per-step time.
  - COST:        flash = measured RunPod $ (billed); tinker = active-compute proxy
                 (rollout+train time, capacity pauses excluded, x a $/hr GPU rate; Tinker
                 does not expose per-run cost via API — labelled as an estimate, not a bill).

Usage:
    uv run python benchmark/assemble.py
    FLASH_PLANE_KEY=<key> uv run python benchmark/assemble.py   # if key not in ~/.flash
"""
from __future__ import annotations

import json
import pathlib
import sys
import urllib.request

_HERE = pathlib.Path(__file__).parent
# Ensure sibling benchmark modules import whether run as a script or imported as a module.
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Single canonical Flash-key resolver (FLASH_PLANE_KEY -> FREESOLO_API_KEY -> ~/.flash).
# Imported from flash_runner so poll and assemble can never drift on key resolution.
from flash_runner import _flash_api_key as _flash_key  # noqa: E402

# Shared Tinker cost proxy + active-compute basis, imported from the producer (tinker_runner)
# so the runner, assembler, and committed results all use ONE rate and ONE basis.
from tinker_runner import TINKER_PROXY_USD_PER_HR as _TINKER_PROXY_USD_PER_HR  # noqa: E402
from tinker_runner import active_compute_s as _sum_active_compute  # noqa: E402
from tinker_runner import read_metrics_records as _read_metrics_records  # noqa: E402

_RESULTS = _HERE / "results"
_MANIFEST = _RESULTS / "runs_manifest.json"
TASKS = ["gsm8k", "reverse-text", "hendrycks-math"]
_SMOOTH_K = 5  # mean over last K steps — robust to the high per-step variance at 16 rollouts/step
# ONE threshold for every "did it move?" verdict (held-out Δ and in-training trend). At 16
# rollouts/step a swing this small is within noise: |Δ| <= band -> "no change", > +band ->
# "rose", < -band -> "fell". Keeps the held-out verdict and the training-curve wording honest
# and mutually consistent (no "rose" text emitted when the curve is actually flat or falling).
_VERDICT_BAND = 0.02


def _trend_word(delta: float | None) -> str:
    """Direction of a change under the shared ±_VERDICT_BAND noise band ('rose'/'fell'/flat)."""
    if delta is None:
        return "is unavailable"
    if delta > _VERDICT_BAND:
        return "rose"
    if delta < -_VERDICT_BAND:
        return "fell"
    return "was flat (within noise)"


def _smoothed(rh: list, k: int = _SMOOTH_K):
    """Mean of the last k rewards (robust 'final performance' vs a single noisy step)."""
    if not rh:
        return None
    tail = rh[-k:]
    return sum(tail) / len(tail)


def _best(rh: list):
    return max(rh) if rh else None


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

    # Authoritative configured/run step count from the worker (notes['steps']). The reward
    # history can log fewer points than steps (cadence) or reset on resume, so len(rh) is
    # NOT the step count — use it only as a fallback when the worker didn't record steps.
    steps = notes.get("steps")
    if steps is None:
        steps = len(rh)

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
        "steps": steps,
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
        "per_step_s": (train_s / steps) if (train_s and steps) else None,
        "train_tok_per_s": notes.get("train_throughput_toks_per_s") or m.get("train_throughput_toks_per_s"),
    }


def _tinker_active_compute_s(log_path: str | None) -> float | None:
    """Active compute (rollout + train step) from a Tinker run's metrics.jsonl.

    Tinker's wall-clock includes managed-backend *capacity pauses* (idle, not billed); the
    active-compute sum is the fair latency/cost basis. BOTH the file selection/parse
    (``read_metrics_records``) and the per-record summation (``active_compute_s``) are
    imported from tinker_runner, so this call site and the runner can never diverge on
    which file they read or how they sum it. Returns None only when there is genuinely no
    timing data (matching active_compute_s); 0.0 is a real pause-excluded zero.
    """
    if not log_path:
        return None
    return _sum_active_compute(_read_metrics_records(log_path))


def _tinker_cost_from_basis(basis: float | None) -> float | None:
    """Tinker $ proxy from an active-compute basis (seconds). Mirrors tinker_runner's
    write-time formula EXACTLY so a recomputed cost equals the stored cost_usd_estimated
    for the same basis. None basis -> None cost (no data)."""
    if basis is None:
        return None
    return round(basis / 3600 * _TINKER_PROXY_USD_PER_HR, 4)


def collect_tinker(task: str, rel_path: str) -> dict:
    """Read one tinker run's result JSON into a normalized record."""
    p = _RESULTS / pathlib.Path(rel_path).name
    if not p.exists():
        return {"platform": "tinker", "task": task, "status": "pending (no result file yet)"}
    d = json.loads(p.read_text())
    rh = d.get("reward_history", []) or []
    wall = d.get("wall_s")
    # Authoritative configured step count from the runner (d['steps']); the reward history
    # may log fewer points. Use this single value for both the reported steps and per_step_s
    # so they can't disagree (previously the dict hard-coded len(rh) while dividing by steps).
    steps = d.get("steps") or len(rh)

    # --- Active-compute AND cost reported from ONE consistent source ---------------------
    # The runner writes active_compute_s and cost_usd_estimated as a matched pair (cost is
    # derived from whatever basis active-compute used at write time). We MUST keep the
    # reported active and the reported cost on the SAME basis — never report a recomputed
    # active next to the wall-based stored cost (or vice-versa). Use `is not None` so a
    # legitimate 0.0 active-compute is honored, not treated as "missing" and replaced by wall.
    stored_active = d.get("active_compute_s")
    if stored_active is not None:
        # Report the stored active, but ALWAYS derive the reported cost from THAT active via
        # the shared proxy — never trust a stored cost_usd_estimated next to it. A legacy /
        # hand-edited result can pair an active-compute `train_s` with a stale wall-based cost;
        # recomputing here guarantees the invariant `reported cost == proxy(reported active)`
        # holds on the stored-active path too (not only the recomputed-active path below).
        active = stored_active
        cost = _tinker_cost_from_basis(active)
    else:
        # No stored active: recompute it from the run's metrics.jsonl (same selector/parser
        # the runner uses) and derive the cost from THAT recomputed active so the two agree.
        active = _tinker_active_compute_s(d.get("log_path"))
        if active is not None:
            cost = _tinker_cost_from_basis(active)
        else:
            # No active anywhere: fall back to the runner's stored (wall-based) cost if it
            # wrote one, else compute the wall-based proxy ourselves. active stays None.
            cost = d.get("cost_usd_estimated")
            if cost is None:
                cost = _tinker_cost_from_basis(wall)
    # Latency/per-step basis: active when present (pause excluded), else wall (is not None).
    basis = active if active is not None else wall
    paused_s = (wall - active) if (wall is not None and active is not None and wall > active) else None
    return {
        "platform": "tinker",
        "task": task,
        "status": d.get("status"),
        "gpu": "managed (Tinker)",
        "steps": steps,
        "first_reward": rh[0] if rh else None,
        "final_reward": rh[-1] if rh else None,
        "final_smoothed": _smoothed(rh),
        "best_reward": _best(rh),
        "eval_reward": None,  # tinker held-out eval not run in-loop; training reward is the signal
        "reward_history": rh,
        "cost_usd": cost,
        "cost_kind": f"ESTIMATE: active-compute x ${_TINKER_PROXY_USD_PER_HR:.2f}/hr proxy (Tinker cost not in API)",
        "setup_s": None,
        "train_s": basis,                 # active compute (rollout+train), pause excluded; wall iff active is None
        "total_s": wall,                  # full wall, INCLUDING any capacity pause
        "paused_s": paused_s,             # idle time the managed backend paused us
        "per_step_s": (basis / steps) if (basis is not None and steps) else None,
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


def _configured_steps(records: dict) -> int | None:
    """The configured step count, derived from the run records (not hard-coded).

    Tinker's record carries the configured count; Flash now carries notes['steps']. Both
    are matched, so report the max observed across all records (None if nothing ran).
    """
    vals = [
        r.get("steps")
        for task in records.values()
        for r in task.values()
        if isinstance(r.get("steps"), int)
    ]
    return max(vals) if vals else None


def render_markdown(records: dict) -> str:
    """records: {task: {'flash': rec, 'tinker': rec}}"""
    steps = _configured_steps(records)
    steps_str = str(steps) if steps is not None else "N"
    lines = []
    lines.append(f"# Flash vs Tinker — GRPO benchmark (Qwen3.5-4B, {steps_str} steps)\n")
    lines.append("Same base model, same verifiers environment, same GRPO hyper-parameters "
                 f"(group_size=4, batch_size=4, **max_tokens=1024**, {steps_str} steps) on each side. "
                 "This run includes the **stall fix** (heartbeat through mid-run eval) and the "
                 "**truncation fix** (1024 tokens so the boxed answer isn't cut off).\n")
    lines.append("- **Flash** trains on a rented RunPod **A100 PCIe** (4B GRPO needs ≥35 GB; the "
                 "allocator escalates from the requested RTX 5090). Cost is **measured** (RunPod billed).")
    lines.append("- **Tinker** trains on Thinking Machines' **managed** backend. Per-run cost is "
                 "**not exposed via API**, so its $ column is an **active-compute** proxy "
                 "(rollout+train time, capacity pauses excluded, x a $2.00/hr GPU rate; labelled, "
                 "not a bill).")
    lines.append(f"- **Performance** = mean group reward over the {steps_str}-step GRPO run "
                 "(step 1 → final). Flash also reports a native held-out eval (50 examples).\n")

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
        # The "final raw" reward is the LAST LOGGED reward point. Flash logs fewer points than
        # steps (cadence), so label by logged-points, not a hard-coded step number.
        f_pts = len(f.get("reward_history") or [])
        t_pts = len(t.get("reward_history") or [])
        lines.append(f"| reward final (last logged: F={f_pts} / T={t_pts} pts) | {_fmt(f.get('final_reward'))} | {_fmt(t.get('final_reward'))} | |")
        lines.append(f"| held-out eval | {_fmt(f.get('eval_reward'))} (n={f.get('eval_n')}) | — | |")
        lines.append(f"| **latency** wall total | {_secs(f.get('total_s'))} | {_secs(t.get('total_s'))} | |")
        lines.append(f"| latency setup/queue | {_secs(f.get('setup_s'))} | none (managed) | |")
        lines.append(f"| latency active compute | {_secs(f.get('train_s'))} | {_secs(t.get('train_s'))} | |")
        # paused_s is a real number when present (0.0 = a measured zero pause -> "0m00s");
        # only a genuinely-absent paused_s (None) renders "none". Guard on `is not None` so a
        # valid 0.0 isn't shown as "none" by a truthiness check.
        _paused = t.get("paused_s")
        lines.append(f"| latency capacity-pause | — | {_secs(_paused) if _paused is not None else 'none'} | |")
        lines.append(f"| latency per-step | {_fmt(f.get('per_step_s'), 1, 's')} | {_fmt(t.get('per_step_s'), 1, 's')} | |")
        lines.append(f"| **cost** | {_fmt(f.get('cost_usd'), 4, ' USD')} | {_fmt(t.get('cost_usd'), 4, ' USD')} | |")
        lines.append(f"| cost basis | {f.get('cost_kind')} | {t.get('cost_kind')} | |")
        lines.append("")

    # Held-out eval (ONE version-independent scorer) — the valid PERFORMANCE comparison.
    eval_path = _RESULTS / "eval_unified_gsm8k.json"
    if eval_path.exists():
        ev = json.loads(eval_path.read_text())
        b = ev.get("base", {})
        tr = ev.get("tinker_trained", {})
        ft = ev.get("flash_trained")  # present only if the Flash serving eval ran
        flash_native = records["gsm8k"]["flash"].get("eval_reward")  # Flash's own on-GPU eval

        def _d(rec):  # "Δ+0.080 vs base" suffix
            dd = rec.get("delta_vs_base")
            return f" (Δ{dd:+.3f} vs base)" if dd is not None else ""

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
        tr_delta = tr.get("delta_vs_base") if tr else None
        # Single ±_VERDICT_BAND rule for BOTH the held-out delta and the in-training trend, so the
        # sentence can never say "rose" when the curve is flat/falling. The in-training direction
        # is taken from THIS run's tinker record (step-1 first_reward -> final_smoothed), not
        # assumed — gsm8k's smoothed reward actually FALLS (0.69 -> 0.36), so a hard-coded "rose"
        # was wrong. held_word/train_word are emitted verbatim, keeping every clause consistent.
        tk = records["gsm8k"]["tinker"]
        first_r, final_r = tk.get("first_reward"), tk.get("final_smoothed")
        train_delta = (final_r - first_r) if (first_r is not None and final_r is not None) else None
        train_word = _trend_word(train_delta)  # "rose" / "fell" / "was flat (within noise)"
        if tr_delta is not None and abs(tr_delta) <= _VERDICT_BAND:
            verdict = (f"**Key finding:** under the shared scorer the Tinker-trained model shows **no "
                       f"significant held-out change** (Δ{tr_delta:+.3f}, within n={b.get('n')} noise); "
                       f"its in-training smoothed reward {train_word} over the run. At this deliberately "
                       f"tiny scale ({steps_str} steps, 16 rollouts/step) the in-training reward curve "
                       f"does NOT track held-out accuracy — exactly what a unified held-out eval is for; "
                       f"the training-reward curve alone would not have predicted the held-out result.")
        elif tr_delta is not None:
            moved = "rose" if tr_delta > 0 else "fell"
            verdict = (f"**Key finding:** under the shared scorer the Tinker-trained model's held-out "
                       f"accuracy {moved} (Δ{tr_delta:+.3f} vs base, beyond the ±{_VERDICT_BAND:.2f} "
                       f"noise band); its in-training smoothed reward {train_word} over the run.")
        else:
            verdict = "Under the shared scorer the GRPO-trained model's held-out accuracy moves as shown."
        lines.append(f"\n{verdict} The Flash-trained row falls back to Flash's own on-GPU eval (a "
                     "similar but not identical scorer) because a unified Flash eval needs a Qwen3.5-4B "
                     "LoRA serving — the live one was empty (0 GPUs / 0 base models); `eval_unified.py` "
                     "runs the unified Flash eval against any configured serving. Truncation note: even "
                     f"at max_tokens={ev.get('max_tokens')}, {int(b.get('truncated_frac', 0) * 100)}% of "
                     "base generations still hit the cap (Qwen3.5-4B is very verbose for this format).\n")

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
    lines.append(f"- **Scale is deliberately small** ({steps_str} steps, 16 rollouts/step) to keep "
                 f"spend low, so per-step reward is noisy and {steps_str}-step gains are modest by "
                 "design.\n")
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
