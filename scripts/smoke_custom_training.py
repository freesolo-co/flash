"""Repeatable paid release smoke against the configured Flash service.

flash login
python scripts/smoke_custom_training.py --run-paid-test --project PROJECT_UUID --output evidence

Runs local DPO through flash train, closes its GPU, resumes on a fresh managed worker,
and checks exact restored logprobs and algorithm state before another optimizer update.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from flash.client import client_from_config
from flash.custom.controller import require_custom_support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-paid-test", action="store_true")
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.run_paid_test:
        parser.error("pass --run-paid-test to provision two managed H100 sessions")
    api = client_from_config()
    require_custom_support(api)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy(Path(__file__).parent / "fixtures/custom_algorithm.py", output / "algorithm.py")
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [root, env.get("PYTHONPATH")]))
    runs = []
    try:
        for steps in (2, 3):
            resume = f'resume_from = "{runs[0]}/step-2"' if runs else ""
            config = output / f"run-{steps}.toml"
            config.write_text(
                f'project = {json.dumps(args.project)}\nmodel = "Qwen/Qwen3.5-9B"\n'
                'algorithm = "custom"\n[custom]\nfile = "algorithm.py"\n'
                f"max_wall_seconds = 1800\nidle_seconds = 300\n{resume}\n"
                f"[train]\nmax_steps = {steps}\nmax_context_tokens = 256\nsave_every = 2\n"
                "lora_rank = 16\n"
            )
            with (output / f"run-{steps}.log").open("w") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "flash.cli", "train", str(config)],
                    cwd=output,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                try:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        match = re.match(r"Custom run: ([A-Za-z0-9._-]+)", line)
                        if match:
                            runs.append(match.group(1))
                        print(line, end="", flush=True)
                    if process.wait() != 0:
                        raise RuntimeError(f"managed DPO phase {steps} failed")
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=15)
        assert "RESTORE_EXACT" in (output / "run-3.log").read_text()
        assert "UPDATE_CHANGED_LOGPROBS" in (output / "run-3.log").read_text()
        evidence = {
            "runs": runs,
            "restored_logprobs_exact": True,
            "restored_algorithm_state_exact": True,
            "continued_update_changed_logprobs": True,
            "registered_checkpoints": [api.checkpoints(run) for run in runs],
        }
        (output / "evidence.json").write_text(json.dumps(evidence, indent=2))
    finally:
        for run in runs:
            if api.get_run(run)["state"] not in {"done", "failed", "cancelled"}:
                api.cancel_run(run)


if __name__ == "__main__":
    main()
