"""Flash GRPO runner for the benchmark.

Submits the job to the flash CP3 via `slm train --background`, polls
/v1/runs/{id} every POLL_INTERVAL seconds, then reads the local
metrics.json once the run reaches state='done'.

Returns a BenchResult dict (see bench.py for schema).
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any


POLL_INTERVAL = 60  # seconds between CP3 polls
_CONFIG_DIR = pathlib.Path(__file__).parent / "configs"
# Path to slm CLI inside the benchmark worktree's own venv.
_SLM = str(pathlib.Path(__file__).parents[1] / ".venv" / "bin" / "slm")


# ---------------------------------------------------------------------------
# Low-level CP3 HTTP helpers (stdlib only — no flash package import needed)
# ---------------------------------------------------------------------------

def _flash_api_url() -> str:
    cfg_path = pathlib.Path.home() / ".flash" / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        return cfg.get("api_url", "https://flash.freesolo.co").rstrip("/")
    return os.environ.get("FLASH_API_URL", "https://flash.freesolo.co").rstrip("/")


def _flash_api_key() -> str:
    key = os.environ.get("FREESOLO_API_KEY", "")
    if not key:
        cfg_path = pathlib.Path.home() / ".flash" / "config.json"
        if cfg_path.exists():
            key = json.loads(cfg_path.read_text()).get("api_key", "")
    return key


def _cp3_get(path: str) -> Any:
    url = f"{_flash_api_url()}{path}"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {_flash_api_key()}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Submit + poll
# ---------------------------------------------------------------------------

def submit(toml_path: str) -> str:
    """Submit the GRPO job and return the run_id."""
    result = subprocess.run(
        [_SLM, "train", toml_path, "--background"],
        capture_output=True, text=True,
    )
    output = result.stdout + result.stderr
    # slm prints: "submitted run <run_id>"
    m = re.search(r"(flash-\d+-[0-9a-f]+)", output)
    if not m:
        raise RuntimeError(f"Could not parse run_id from slm output:\n{output}")
    run_id = m.group(1)
    print(f"  [flash] submitted {run_id}")
    return run_id


def poll_until_done(run_id: str) -> dict:
    """Poll the CP3 until the run finishes. Returns the final status dict."""
    while True:
        status = _cp3_get(f"/v1/runs/{run_id}")
        state = status.get("state", "")
        cost = status.get("cost_usd") or 0.0
        print(f"  [flash] {run_id} state={state} cost=${cost:.4f}")
        if state in ("done", "failed"):
            return status
        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------

def _read_metrics(artifacts_dir: str) -> dict:
    """Load metrics.json from the local artifacts dir."""
    p = pathlib.Path(artifacts_dir) / "seed0" / "metrics.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def run(toml_path: str | None = None) -> dict:
    """Full flash benchmark run. Returns a BenchResult dict."""
    toml_path = toml_path or str(_CONFIG_DIR / "gsm8k_4b.toml")
    t0 = time.monotonic()
    run_id = submit(toml_path)
    status = poll_until_done(run_id)
    wall = time.monotonic() - t0

    if status.get("state") != "done":
        return {
            "platform": "flash",
            "status": "failed",
            "error": status.get("error"),
            "wall_s": wall,
        }

    artifacts_dir = status.get("artifacts_dir", "")
    m = _read_metrics(artifacts_dir)
    notes = m.get("notes", {})

    reward_history: list[float] = notes.get("reward_history", [])
    eval_history: list[dict] = notes.get("eval_history", [])

    first_reward = reward_history[0] if reward_history else None
    final_reward = reward_history[-1] if reward_history else None
    final_eval = (
        eval_history[-1].get("eval_reward") if eval_history else None
    )

    return {
        "platform": "flash",
        "status": "done",
        "run_id": run_id,
        "wall_s": m.get("wall_seconds") or wall,
        "setup_s": m.get("setup_seconds"),
        "cost_usd": m.get("cost_usd") or status.get("cost_usd"),
        "gpu": m.get("allocated_gpu"),
        "steps": len(reward_history),
        "first_train_reward": first_reward,
        "final_train_reward": final_reward,
        "final_eval_reward": final_eval,
        "eval_history": eval_history,
        "reward_history": reward_history,
    }


if __name__ == "__main__":
    import sys
    toml = sys.argv[1] if len(sys.argv) > 1 else None
    result = run(toml)
    print(json.dumps(result, indent=2))
