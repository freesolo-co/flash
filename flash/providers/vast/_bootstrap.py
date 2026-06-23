"""Self-contained bootstrap that runs ON a Vast.ai instance.

Replicates ``providers/runpod/train.py:_train_body`` semantics on the Vast substrate: install
extra pip deps, fetch the flash package from the HF dataset repo, then run the
substrate-neutral worker (``flash.engine.worker``) to train, uploading the console
tail on failure. There is NO return channel
from the instance: the worker's HF artifacts (DONE/metrics.json/heartbeat.json) are
the success signal, and this bootstrap's attempt-scoped ``vast_attempt<N>.json`` is
the terminal marker the control plane keys failures on.

This file is shipped verbatim inside the instance's onstart script (see
``providers/vast/jobs.py:build_onstart``), so it must stay self-contained: stdlib +
huggingface_hub (installed with the worker deps) only — never import flash here.
It reads its payload from ``/root/flash/payload.json``.
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

PAYLOAD_PATH = "/root/flash/payload.json"
CODE_ROOT = "/runcode"
CODE_DIR = "/runcode/code"


def load_payload() -> dict:
    with open(PAYLOAD_PATH) as f:
        return json.load(f)


def hf_upload(payload: dict, local_path: str, repo_subpath: str) -> None:
    """Upload one artifact under the run's HF prefix; never raises."""
    try:
        from huggingface_hub import HfApi

        HfApi(token=(payload.get("env") or {}).get("HF_TOKEN")).upload_file(
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
    # Pass a large spec via a file, not the environment: a job spec with large inline
    # params can reach multiple hundred KB, and that big an env var trips execve's
    # "Argument list too long" when the worker subprocess starts. Mirrors
    # runpod/train.py:_train_body.
    spec_json = payload["job_spec_json"]
    if len(spec_json) > 96_000:
        with open("/tmp/job_spec.json", "w") as f:
            f.write(spec_json)
        env["FLASH_JOB_SPEC_PATH"] = "/tmp/job_spec.json"
        env.pop("FLASH_JOB_SPEC_JSON", None)
    else:
        env["FLASH_JOB_SPEC_JSON"] = spec_json
    env["PHASE"] = payload["phase"]
    env["SEED"] = str(payload["seed"])
    # Compute substrate for the RunMetrics record (engine.worker reads FLASH_ARM). The
    # payload env was built by the shared runpod env builder, which stamps "runpod"; this
    # bootstrap runs on the Vast instance, so override it to the real backend.
    env["FLASH_ARM"] = "vast"
    env["PYTHONPATH"] = CODE_DIR + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def fetch_code(payload: dict) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=payload["hf_repo"],
        repo_type="dataset",
        allow_patterns=["code/**"],
        local_dir=CODE_ROOT,
        token=(payload.get("env") or {}).get("HF_TOKEN"),
    )


def run_mode(payload: dict, env: dict, mode: str, deadline_ts: float) -> int:
    """One worker process; console teed to a file and streamed to the instance log.

    On failure the console tail is uploaded as console_<mode>.txt — like _train_body,
    because subprocess consoles are the only place engine-core crashes surface. With
    FLASH_UPLOAD_CONSOLE=1 (forwarded into env via build_worker_env) it is also uploaded on
    SUCCESS so operators can verify which optimizations engaged — matching runpod/train.py. On
    deadline the process is killed and we return a sentinel nonzero rc.
    """
    console = f"/tmp/console_{mode}.txt"
    timed_out = False
    with open(console, "w") as cf:
        proc = subprocess.Popen(
            [sys.executable, "-m", "flash.engine.worker"],
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
    # FLASH_UPLOAD_CONSOLE=1 also uploads the console on SUCCESS (not just failure/timeout) so an
    # operator can confirm which optimizations engaged — mirrors run_mode() in runpod/train.py.
    _force_console = env.get("FLASH_UPLOAD_CONSOLE", "").strip().lower() not in (
        "", "0", "false", "no", "off",
    )
    if proc.returncode != 0 or timed_out or _force_console:
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
        # Fast model downloads on Vast: RunPod's Flash runtime ships hf_transfer + sets
        # HF_HUB_ENABLE_HF_TRANSFER, but Vast hosts don't — so a cold model pull is serial and
        # slow (measured ~84s for a 2 GB model vs ~6s on RunPod, where setup is now the dominant
        # cost). Install it + enable so snapshot_download/from_pretrained saturate the NIC.
        # Best-effort: only enable the flag if the package is present (enabling it WITHOUT the
        # package makes huggingface_hub hard-error).
        try:
            import importlib.util

            if importlib.util.find_spec("hf_transfer") is None:
                subprocess.run([sys.executable, "-m", "pip", "install", "hf_transfer"], check=True)
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        except Exception as _e:
            print("hf_transfer setup skipped (slow downloads):", _e)
        # W&B logging (restored post-flash-migration): the prebuilt image predates wandb being
        # added to the stack, so install it on-demand when a W&B key is present. The worker's
        # wandb_report_to() gates report_to on the package actually importing, so this is what makes
        # W&B logging real on the current image without a rebuild.
        try:
            import importlib.util  # local: the hf_transfer block above may fail before importing it

            _penv = payload.get("env") or {}
            if (_penv.get("WANDB_API_KEY") or os.environ.get("WANDB_API_KEY")) and (
                importlib.util.find_spec("wandb") is None
            ):
                _wb = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "wandb>=0.17"], check=False
                )
                if _wb.returncode == 0 and importlib.util.find_spec("wandb") is not None:
                    print("[wandb] installed wandb on-demand for W&B logging")
                else:
                    print(
                        f"[wandb] on-demand wandb install FAILED (rc={_wb.returncode}); "
                        "W&B logging will be disabled"
                    )
        except Exception as _e:
            print("wandb setup skipped:", _e)
        # NB: the Hopper fla guard lives in engine.worker._drop_fla_on_hopper (runs in the worker
        # process after all installs, before any model import) — not here, where a later
        # install could pull fla back in. The bootstrap just fetches code and runs the worker.

        extra_pip = payload.get("extra_pip") or []
        if extra_pip:
            # check=True: a deterministic dependency failure (GRPO / Freesolo extras)
            # must stop NOW with an actionable error, not proceed to
            # a later import crash while the paid instance runs (matches the RunPod path).
            subprocess.run([sys.executable, "-m", "pip", "install", *extra_pip], check=True)
        # NB: fla is dropped on Hopper (sm90) automatically by engine.worker._drop_fla_on_hopper at
        # worker startup (fla's GDN backward is miscomputed on sm90, #640) — no bootstrap uninstall
        # or env toggle. fla only ever runs on the consumer archs where its Triton kernel is correct.
        fetch_code(payload)
        env = build_worker_env(payload)
        deadline = time.time() + float(payload.get("max_wall_s") or 24 * 3600)
        phase = payload["phase"]
        # A warm/retried Vast instance can carry a previous attempt's metrics file; a
        # stale one would let a crashed train phase report the previous run's metrics.
        # Clear before training (mirrors the RunPod Flash handler in runpod/train.py).
        for stale in ("/tmp/train_meta.json", "/tmp/metrics.json"):
            with contextlib.suppress(FileNotFoundError):
                os.remove(stale)
        # Train. Nonzero rc tolerated — RL's colocated vLLM can segfault at interpreter
        # exit AFTER the adapter + metrics.json + DONE are saved. The train phase writes
        # metrics.json + DONE itself (or restores them from an earlier attempt's DONE).
        run_mode(payload, env, phase, deadline)
        if not os.path.exists("/tmp/metrics.json"):
            raise RuntimeError(
                f"train phase '{phase}' produced no /tmp/metrics.json (it crashed before "
                f"finishing); see error_{phase}.txt and console_{phase}.txt in the HF "
                f"dataset repo"
            )
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
