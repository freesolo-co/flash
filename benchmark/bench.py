"""Flash vs Tinker head-to-head GRPO benchmark on a verifiers environment.

Trains Qwen3.5-4B on the selected env (--env-id, default gsm8k) for --steps steps
via both platforms concurrently, then prints a comparison table covering reward,
wall-clock, and cost. Both platforms receive the same --steps and --env-id.

Usage:
    # From the flash-benchmark worktree:
    uv run python benchmark/bench.py

    # Or with overrides:
    uv run python benchmark/bench.py --steps 30 --skip-flash --skip-tinker

    # The Tinker runner needs a Python with `verifiers` installed.
    # Default: /home/azureuser/workspace/flash-fulltest/.venv/bin/python
    # Override: TINKER_PYTHON=/path/to/python uv run python benchmark/bench.py

Requirements:
  - Flash CP3 reachable (FLASH_API_URL env or ~/.flash/config.json)
  - TINKER_API_KEY in environment
  - Python with `verifiers` for the Tinker side (see TINKER_PYTHON above)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import time

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_HERE = pathlib.Path(__file__).parent
_REPO_ROOT = _HERE.parent

# Python interpreter that has `tinker` + `verifiers` installed.
# Tinker trains server-side, so the local interpreter only needs the tinker SDK +
# verifiers' rollout/scoring machinery (no torch/vllm). The system python carries both.
_DEFAULT_TINKER_PYTHON = os.environ.get("TINKER_PYTHON", "/usr/bin/python3")

# Flash config per env-id (each TOML pins the matching [environment] id + 4B GRPO recipe).
# Lets --env-id pick the right Flash config instead of always running gsm8k.
_FLASH_CONFIG_BY_ENV = {
    "gsm8k": "gsm8k_4b.toml",
    "reverse-text": "reverse_text_4b.toml",
    "hendrycks-math": "hendrycks_math_4b.toml",
}


def _tinker_result_path(env_id: str) -> pathlib.Path:
    """Per-env Tinker result path so concurrent / sequential runs don't collide."""
    return pathlib.Path(tempfile.gettempdir()) / f"tinker_bench_result_{env_id}.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_s(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _fmt_reward(r: float | None) -> str:
    if r is None:
        return "N/A"
    return f"{r:.3f}"


def _fmt_usd(v: float | None, note: str = "") -> str:
    if v is None:
        return "N/A"
    s = f"${v:.4f}"
    if note:
        s += f"  ({note})"
    return s


def _reward_curve_ascii(history: list[float], width: int = 30) -> str:
    """One-line ASCII sparkline for a reward curve."""
    if not history:
        return "(no data)"
    blocks = " ▁▂▃▄▅▆▇█"
    lo, hi = min(history), max(history)
    span = hi - lo or 1.0
    return "".join(blocks[min(8, int((v - lo) / span * 8))] for v in history[:width])


def _print_table(flash: dict, tinker: dict, env_id: str = "gsm8k", steps: int = 30) -> None:
    """Print the comparison table to stdout."""
    rows = [
        ("", "FLASH", "TINKER"),
        ("─" * 28, "─" * 22, "─" * 22),
        ("Status", flash.get("status", "?"), tinker.get("status", "?")),
        ("GPU", flash.get("gpu", "RTX 5090 (RunPod)"), "Managed (Tinker)"),
        ("Steps trained", str(flash.get("steps", "?")), str(tinker.get("steps", "?"))),
        ("─" * 28, "─" * 22, "─" * 22),
        # NOTE: the two wall-clocks measure DIFFERENT scopes and are not directly comparable:
        #  - Flash = worker `wall_seconds` (on-GPU training only; excludes client poll/setup).
        #  - Tinker = full subprocess elapsed (INCLUDES managed-backend capacity pauses).
        # assemble.py reconciles these onto an active-compute basis; this is the raw view.
        ("Wall-clock (train; see note)",
         _fmt_s(flash.get("wall_s")),
         _fmt_s(tinker.get("wall_s"))),
        ("Cost (USD)",
         _fmt_usd(flash.get("cost_usd")),
         _fmt_usd(tinker.get("cost_usd_estimated"), tinker.get("cost_note", ""))),
        ("─" * 28, "─" * 22, "─" * 22),
        ("Train reward step 1",
         _fmt_reward(flash.get("first_train_reward")),
         _fmt_reward(tinker.get("first_train_reward"))),
        ("Train reward final",
         _fmt_reward(flash.get("final_train_reward")),
         _fmt_reward(tinker.get("final_train_reward"))),
        ("Eval reward (held-out)",
         _fmt_reward(flash.get("final_eval_reward")),
         "N/A (see log)"),
        ("─" * 28, "─" * 22, "─" * 22),
        ("Reward curve (train)",
         _reward_curve_ascii(flash.get("reward_history", [])),
         _reward_curve_ascii(tinker.get("reward_history", []))),
        ("─" * 28, "─" * 22, "─" * 22),
        ("Run ID / artifact",
         flash.get("run_id", "?"),
         tinker.get("log_path", "?")),
    ]
    col_w = [30, 24, 24]
    print()
    print("=" * 80)
    print(f"  Flash vs Tinker — GRPO benchmark on {env_id} (Qwen3.5-4B, {steps} steps)")
    print("=" * 80)
    for cols in rows:
        line = "  ".join(str(c).ljust(col_w[i]) for i, c in enumerate(cols))
        print(line)
    print("=" * 80)
    print("  Wall-clock note: Flash = on-GPU train time (excl. poll/setup); "
          "Tinker = full elapsed (incl. capacity pauses). Not directly comparable —")
    print("  see results/comparison.md for the active-compute reconciliation.")
    print()


# ---------------------------------------------------------------------------
# Flash runner (inline — delegates to flash_runner.py)
# ---------------------------------------------------------------------------

def _run_flash(args: argparse.Namespace, result_box: list) -> None:
    """Thread target: run flash GRPO and store result in result_box[0].

    Honors the shared CLI flags so Flash and Tinker train the SAME task/steps:
      - --flash-config (explicit) wins; otherwise the config is picked from --env-id.
      - --steps is forwarded as `--set train.steps=<N>` (the worker's authoritative count).
    """
    sys.path.insert(0, str(_HERE))
    try:
        import flash_runner
        if args.flash_config:
            toml = args.flash_config
        else:
            cfg_name = _FLASH_CONFIG_BY_ENV.get(args.env_id)
            if cfg_name is None:
                raise ValueError(
                    f"no Flash config for env-id {args.env_id!r}; pass --flash-config "
                    f"(known: {', '.join(sorted(_FLASH_CONFIG_BY_ENV))})"
                )
            toml = str(_HERE / "configs" / cfg_name)
        # Forward the shared --steps to the Flash side so both platforms run the same count.
        overrides = [f"train.steps={args.steps}"]
        result = flash_runner.run(toml, overrides=overrides)
    except Exception as exc:
        result = {"platform": "flash", "status": "error", "error": str(exc)}
    result_box.append(result)


# ---------------------------------------------------------------------------
# Tinker runner (subprocess — needs Python with `verifiers`)
# ---------------------------------------------------------------------------

def _run_tinker(args: argparse.Namespace, result_box: list) -> None:
    """Thread target: launch tinker_runner.py as a subprocess, wait for it."""
    python = os.environ.get("TINKER_PYTHON", _DEFAULT_TINKER_PYTHON)
    runner = str(_HERE / "tinker_runner.py")
    # Per-env result path so overlapping/sequential runs don't read or clobber each other.
    result_file = _tinker_result_path(args.env_id)
    output_path = str(result_file)

    cmd = [
        python, runner,
        "--steps", str(args.steps),
        "--groups-per-batch", str(args.groups_per_batch),
        "--group-size", str(args.group_size),
        "--max-tokens", str(args.max_tokens),
        "--model", args.model,
        "--env-id", args.env_id,
        "--output", output_path,
        "--log-path", f"/tmp/tinker-bench-{args.env_id}",
    ]
    if not pathlib.Path(python).exists():
        result_box.append({
            "platform": "tinker",
            "status": "skipped",
            "error": f"TINKER_PYTHON not found: {python}  — set TINKER_PYTHON env var",
        })
        return

    # Remove any stale result from a previous run BEFORE launching, so a subprocess that
    # fails without writing can't be mistaken for a fresh success (the file is keyed per
    # env, but a leftover from an earlier invocation of the same task would still be stale).
    result_file.unlink(missing_ok=True)

    print(f"  [tinker] launching: {' '.join(cmd[:4])} ...")
    t0 = time.monotonic()
    t0_wall = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Stream output so the user sees progress
    assert proc.stdout
    for line in proc.stdout:
        print(f"  [tinker] {line}", end="")
    proc.wait()
    wall = time.monotonic() - t0

    # Accept the result file only if (a) the subprocess succeeded, and (b) the file is THIS
    # run's (written after launch). Otherwise treat as failed — never report stale metrics.
    fresh = (
        result_file.exists()
        and result_file.stat().st_mtime >= t0_wall
    )
    if proc.returncode == 0 and fresh:
        result = json.loads(result_file.read_text())
        result.setdefault("wall_s", wall)
    else:
        reason = (
            f"nonzero exit ({proc.returncode})" if proc.returncode != 0
            else "no fresh result file written"
        )
        result = {
            "platform": "tinker",
            "status": "failed",
            "error": reason,
            "returncode": proc.returncode,
            "wall_s": wall,
        }
    result_box.append(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Flash vs Tinker head-to-head GRPO benchmark (verifiers env; "
        "env + steps configurable below)."
    )
    p.add_argument(
        "--steps", type=int, default=30,
        help="GRPO steps; applied to BOTH platforms (Tinker via --steps, Flash via "
        "`--set train.steps`). Flash logs the authoritative count to notes['steps'].",
    )
    p.add_argument("--groups-per-batch", type=int, default=4)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument(
        "--env-id", default="gsm8k",
        help="verifiers env id (gsm8k | reverse-text | hendrycks-math). Selects the "
        "matching Flash config unless --flash-config is given.",
    )
    p.add_argument(
        "--flash-config", default=None,
        help="explicit Flash TOML; overrides the --env-id config selection",
    )
    p.add_argument("--skip-flash", action="store_true")
    p.add_argument("--skip-tinker", action="store_true")
    p.add_argument("--output", default=None, help="Write results JSON to this path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    flash_result: list = []
    tinker_result: list = []

    threads: list[threading.Thread] = []

    if not args.skip_flash:
        t = threading.Thread(target=_run_flash, args=(args, flash_result), daemon=True)
        threads.append(t)
    if not args.skip_tinker:
        t = threading.Thread(target=_run_tinker, args=(args, tinker_result), daemon=True)
        threads.append(t)

    if not threads:
        print("Nothing to run (--skip-flash and --skip-tinker both set).")
        return

    print()
    print(f"Launching Flash + Tinker GRPO on {args.env_id} ({args.model}, {args.steps} steps)")
    print("Polling every 60s. Both platforms run concurrently.")
    print()

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    fr = flash_result[0] if flash_result else {"platform": "flash", "status": "skipped"}
    tr = tinker_result[0] if tinker_result else {"platform": "tinker", "status": "skipped"}

    _print_table(fr, tr, env_id=args.env_id, steps=args.steps)

    if args.output:
        out = pathlib.Path(args.output)
        out.write_text(json.dumps({"flash": fr, "tinker": tr}, indent=2))
        print(f"Results written to {out}")


if __name__ == "__main__":
    main()
