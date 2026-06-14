"""Self-contained bootstrap that runs ON a Vast.ai instance.

Replicates ``flash/train.py:_train_body`` semantics on the Vast substrate: install
extra pip deps, fetch the autoslm package from the HF dataset repo, then run the
substrate-neutral worker (``autoslm.engine.worker``) in two fresh processes — train,
then eval — uploading per-phase console tails on failure. There is NO return channel
from the instance: the worker's HF artifacts (DONE/metrics.json/heartbeat.json) are
the success signal, and this bootstrap's attempt-scoped ``vast_attempt<N>.json`` is
the terminal marker the control plane keys failures on.

This file is shipped verbatim inside the instance's onstart script (see
``providers/vast.py:build_onstart``), so it must stay self-contained: stdlib +
huggingface_hub (installed with the worker deps) only — never import autoslm here.
It reads its payload from ``/root/autoslm/payload.json``.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time

PAYLOAD_PATH = "/root/autoslm/payload.json"
CODE_ROOT = "/runcode"
CODE_DIR = "/runcode/code"


def load_payload(path: str = PAYLOAD_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def hf_upload(payload: dict, local_path: str, repo_subpath: str) -> None:
    """Upload one artifact under the run's HF prefix; never raises."""
    try:
        from huggingface_hub import HfApi

        HfApi(token=(payload.get("env") or {}).get("HUGGINGFACE_TOKEN")).upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"{payload['hf_prefix']}/{repo_subpath}",
            repo_id=payload["hf_repo"],
            repo_type="dataset",
        )
    except Exception as exc:
        print(f"hf upload warn ({repo_subpath}): {exc}", flush=True)


def build_worker_env(payload: dict) -> dict:
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (payload.get("env") or {}).items()})
    # Pass a large spec via a file (a Freesolo bridge spec embeds the dataset records),
    # not the environment: a multi-hundred-KB env var trips execve's "Argument list too
    # long" when the worker subprocess starts. Mirrors flash/train.py:_train_body.
    spec_json = payload["job_spec_json"]
    if len(spec_json) > 96_000:
        with open("/tmp/job_spec.json", "w") as f:
            f.write(spec_json)
        env["AUTOSLM_JOB_SPEC_PATH"] = "/tmp/job_spec.json"
        env.pop("AUTOSLM_JOB_SPEC_JSON", None)
    else:
        env["AUTOSLM_JOB_SPEC_JSON"] = spec_json
    env["PHASE"] = payload["phase"]
    env["SEED"] = str(payload["seed"])
    env["PYTHONPATH"] = CODE_DIR + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def fetch_code(payload: dict) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=payload["hf_repo"],
        repo_type="dataset",
        allow_patterns=["code/**"],
        local_dir=CODE_ROOT,
        token=(payload.get("env") or {}).get("HUGGINGFACE_TOKEN"),
    )


def run_mode(payload: dict, env: dict, mode: str, deadline_ts: float) -> int:
    """One worker process; console teed to a file and streamed to the instance log.

    On failure the console tail is uploaded as console_<mode>.txt — like _train_body,
    because subprocess consoles are the only place engine-core crashes surface. On
    deadline the process is killed and we return a sentinel nonzero rc.
    """
    console = f"/tmp/console_{mode}.txt"
    timed_out = False
    with open(console, "w") as cf:
        proc = subprocess.Popen(
            [sys.executable, "-m", "autoslm.engine.worker"],
            cwd=CODE_DIR,
            env={**env, "RUN_MODE": mode},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        def pump():
            for line in proc.stdout:
                print(line, end="", flush=True)
                cf.write(line)

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        try:
            proc.wait(timeout=max(10.0, deadline_ts - time.time()))
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()
        t.join(timeout=10)
    if proc.returncode != 0 or timed_out:
        try:
            tail_path = console + ".tail"
            with open(console) as f:
                tail = f.read()[-64_000:]
            if timed_out:
                tail += f"\n--- bootstrap: mode '{mode}' hit the wall-clock cap; killed ---\n"
            with open(tail_path, "w") as f:
                f.write(tail)
            hf_upload(payload, tail_path, f"console_{mode}.txt")
        except Exception as exc:
            print(f"console upload warn: {exc}", flush=True)
    if timed_out:
        raise TimeoutError(f"worker mode '{mode}' exceeded the wall-clock cap")
    return proc.returncode


def write_attempt_marker(payload: dict, ok: bool, error: str = "") -> None:
    """Attempt-scoped terminal marker: how the control plane distinguishes THIS
    attempt's failure from a prior attempt's leftovers under the same prefix."""
    marker = {
        "ok": bool(ok),
        "ts": time.time(),
        "attempt": int(payload.get("attempt") or 0),
        "error": error[:2000],
    }
    p = "/tmp/vast_attempt.json"
    with open(p, "w") as f:
        json.dump(marker, f)
    hf_upload(payload, p, f"vast_attempt{marker['attempt']}.json")


def main() -> int:
    # Make SIGTERM (vast stop / bash `timeout`) unwind through finally so the
    # terminal marker still gets uploaded.
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(1))
    payload = load_payload()
    ok = False
    error = ""
    try:
        extra_pip = payload.get("extra_pip") or []
        if extra_pip:
            # check=True: a deterministic dependency failure (Freesolo GRPO / Prime Hub
            # / verifiers extras) must stop NOW with an actionable error, not proceed to
            # a later import crash while the paid instance runs (matches the RunPod path).
            subprocess.run([sys.executable, "-m", "pip", "install", *extra_pip], check=True)
        fetch_code(payload)
        env = build_worker_env(payload)
        deadline = time.time() + float(payload.get("max_wall_s") or 24 * 3600)
        phase = payload["phase"]
        if phase == "eval":
            rc = run_mode(payload, env, "eval_only", deadline)
            if rc != 0:
                raise RuntimeError(f"worker mode 'eval_only' exited {rc}")
        else:
            # A warm/retried Vast instance can carry a previous attempt's handoff/metrics
            # files; stale ones would let a crashed train phase evaluate the OLD adapter or
            # report the previous run's metrics. Clear both before training (mirrors the
            # RunPod Flash handler in autoslm.flash.train).
            for stale in ("/tmp/train_meta.json", "/tmp/metrics.json"):
                with contextlib.suppress(FileNotFoundError):
                    os.remove(stale)
            # Phase 1: train. Nonzero rc tolerated — RL's colocated vLLM can segfault
            # at interpreter exit AFTER the adapter+train_meta are saved.
            run_mode(payload, env, phase, deadline)
            if os.path.exists("/tmp/metrics.json"):
                # Idempotent DONE replay: an earlier attempt already finished this run.
                # The worker saw DONE on HF, restored /tmp/metrics.json, and exited 0
                # WITHOUT recreating /tmp/train_meta.json — so don't mistake a successful
                # replay for a crashed train phase, and don't re-run eval. (Mirrors the
                # RunPod handler's metrics.json short-circuit.)
                print("train phase: metrics.json present (DONE replay); skipping eval", flush=True)
            else:
                if not os.path.exists("/tmp/train_meta.json"):
                    raise RuntimeError(
                        f"train phase '{phase}' produced no /tmp/train_meta.json (it crashed "
                        f"before saving the adapter); see error_{phase}.txt and "
                        f"console_{phase}.txt in the HF dataset repo"
                    )
                # Phase 2: eval in a FRESH process.
                rc = run_mode(payload, env, "eval_after", deadline)
                if rc != 0:
                    raise RuntimeError(f"worker mode 'eval_after' exited {rc}")
        ok = True
    except Exception as exc:
        # Record genuine failures in the attempt marker (written in `finally`). Don't catch
        # BaseException — KeyboardInterrupt/SystemExit must propagate after the marker write
        # rather than be swallowed into a `return 1`.
        error = f"{type(exc).__name__}: {exc}"
        print(f"bootstrap failed: {error}", flush=True)
    finally:
        write_attempt_marker(payload, ok, error)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
