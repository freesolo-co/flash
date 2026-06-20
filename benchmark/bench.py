"""Flash vs Tinker head-to-head GRPO benchmark on GSM8K.

Trains Qwen3.5-4B on GSM8K for 30 steps via both platforms concurrently,
then prints a comparison table covering reward, wall-clock, and cost.

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
_TINKER_RESULT_PATH = "/tmp/tinker_bench_result.json"


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


def _print_table(flash: dict, tinker: dict) -> None:
    """Print the comparison table to stdout."""
    rows = [
        ("", "FLASH", "TINKER"),
        ("─" * 28, "─" * 22, "─" * 22),
        ("Status", flash.get("status", "?"), tinker.get("status", "?")),
        ("GPU", flash.get("gpu", "RTX 5090 (RunPod)"), "Managed (Tinker)"),
        ("Steps trained", str(flash.get("steps", "?")), str(tinker.get("steps", "?"))),
        ("─" * 28, "─" * 22, "─" * 22),
        ("Wall-clock",
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
    print("  Flash vs Tinker — GRPO benchmark on GSM8K (Qwen3.5-4B, 30 steps)")
    print("=" * 80)
    for cols in rows:
        line = "  ".join(str(c).ljust(col_w[i]) for i, c in enumerate(cols))
        print(line)
    print("=" * 80)
    print()


# ---------------------------------------------------------------------------
# Flash runner (inline — delegates to flash_runner.py)
# ---------------------------------------------------------------------------

def _run_flash(args: argparse.Namespace, result_box: list) -> None:
    """Thread target: run flash GRPO and store result in result_box[0]."""
    sys.path.insert(0, str(_HERE))
    try:
        import flash_runner
        toml = args.flash_config or str(_HERE / "configs" / "gsm8k_4b.toml")
        # Override steps in config if requested
        result = flash_runner.run(toml)
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
    output_path = _TINKER_RESULT_PATH

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

    print(f"  [tinker] launching: {' '.join(cmd[:4])} ...")
    t0 = time.monotonic()
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

    # Read result file written by tinker_runner.py
    result_file = pathlib.Path(output_path)
    if result_file.exists():
        result = json.loads(result_file.read_text())
        result.setdefault("wall_s", wall)
    else:
        result = {
            "platform": "tinker",
            "status": "failed",
            "returncode": proc.returncode,
            "wall_s": wall,
        }
    result_box.append(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flash vs Tinker GRPO benchmark on GSM8K")
    p.add_argument("--steps", type=int, default=30, help="GRPO steps for both platforms")
    p.add_argument("--groups-per-batch", type=int, default=4)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--env-id", default="gsm8k")
    p.add_argument("--flash-config", default=None, help="Path to flash TOML (default: configs/gsm8k_4b.toml)")
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

    _print_table(fr, tr)

    if args.output:
        out = pathlib.Path(args.output)
        out.write_text(json.dumps({"flash": fr, "tinker": tr}, indent=2))
        print(f"Results written to {out}")


if __name__ == "__main__":
    main()
