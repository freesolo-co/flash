"""Self-contained bootstrap that runs INSIDE the worker container on a Lambda Cloud instance.

Replicates ``providers/runpod/train.py:_train_body`` semantics on the Lambda substrate. The
Lambda cloud-init ``user_data`` (see ``jobs.builders.build_user_data``) runs the prebuilt,
PUBLIC ``WORKER_IMAGE`` via Docker on the host, and this module is the container's command:
install the run's extra pip deps, fetch the flash package from the HF dataset repo, then run the
substrate-neutral worker (``flash.engine.worker``) to train, uploading the console tail.

There is NO return channel from the instance: the worker's HF artifacts
(DONE/metrics.json/heartbeat.json) are the success signal, and this bootstrap's attempt-scoped
``lambda_attempt<N>.json`` is the terminal marker the control plane keys failures on. The full
training stack is BAKED into the image, so — unlike the historical Vast bootstrap — there is no
base-stack install here (only the per-run ``extra_pip``).

Shipped verbatim inside the container command, so it must stay self-contained: stdlib +
huggingface_hub (baked into the image) only — never import flash here. It reads its payload from
``/root/flash/payload.json``.
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
    # Pass a large spec via a file, not the environment: a job spec with large inline params can
    # reach hundreds of KB, which trips execve's "Argument list too long". Mirrors
    # runpod/train.py:_train_body and the Vast bootstrap.
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
    # Compute substrate for the RunMetrics record (engine.worker reads FLASH_ARM). The payload env
    # was built by the shared runpod env builder, which stamps "runpod"; this bootstrap runs on the
    # Lambda instance, so override it to the real backend.
    env["FLASH_ARM"] = "lambda"
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
    """One worker process; console teed to a file and streamed to the container log.

    On failure the console tail is uploaded as console_<mode>.txt — like _train_body, because
    subprocess consoles are the only place engine-core crashes surface. With FLASH_UPLOAD_CONSOLE=1
    (default from build_worker_env) it is also uploaded periodically during the run and on SUCCESS.
    On deadline the process is killed and we raise.
    """
    console = f"/tmp/console_{mode}.txt"
    timed_out = False
    upload_enabled = env.get("FLASH_UPLOAD_CONSOLE", "").strip().lower() not in (
        "", "0", "false", "no", "off",
    )
    upload_interval = max(5.0, float(env.get("FLASH_CONSOLE_UPLOAD_INTERVAL_S") or 30.0))

    def upload_console_tail(extra: str = "") -> None:
        tail_path = console + ".tail"
        with open(console) as f:
            tail = f.read()[-64_000:]
        if extra:
            tail += extra
        with open(tail_path, "w") as f:
            f.write(tail)
        hf_upload(payload, tail_path, f"console_{mode}.txt")

    stop_upload = threading.Event()

    def upload_loop() -> None:
        while not stop_upload.wait(upload_interval):
            try:
                upload_console_tail()
            except Exception as exc:
                print(f"console upload warn: {exc}", flush=True)

    uploader = None
    with open(console, "w", buffering=1) as cf:
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
        if upload_enabled:
            uploader = threading.Thread(target=upload_loop, daemon=True)
            uploader.start()
        try:
            proc.wait(timeout=max(10.0, deadline_ts - time.time()))
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()
        t.join(timeout=10)
        if uploader is not None:
            stop_upload.set()
            uploader.join(timeout=10)
    if proc.returncode != 0 or timed_out or upload_enabled:
        try:
            extra = ""
            if timed_out:
                extra = f"\n--- bootstrap: mode '{mode}' hit the wall-clock cap; killed ---\n"
            upload_console_tail(extra)
        except Exception as exc:
            print(f"console upload warn: {exc}", flush=True)
    if timed_out:
        raise TimeoutError(f"worker mode '{mode}' exceeded the wall-clock cap")
    return proc.returncode


def write_attempt_marker(payload: dict, ok: bool, error: str = "") -> None:
    """Attempt-scoped terminal marker: how the control plane distinguishes THIS attempt's failure
    from a prior attempt's leftovers under the same prefix."""
    marker = {
        "ok": bool(ok),
        "ts": time.time(),
        "attempt": int(payload.get("attempt") or 0),
        "error": error[:2000],
    }
    p = "/tmp/lambda_attempt.json"
    with open(p, "w") as f:
        json.dump(marker, f)
    hf_upload(payload, p, f"lambda_attempt{marker['attempt']}.json")


def main() -> int:
    # Make SIGTERM (docker stop / wall-cap) unwind through finally so the terminal marker still
    # gets uploaded.
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(1))
    payload = load_payload()
    ok = False
    error = ""
    try:
        # hf_transfer is baked into the worker image; enable it so model pulls saturate the NIC.
        # Best-effort: only enable the flag if the package is importable (enabling it WITHOUT the
        # package makes huggingface_hub hard-error).
        try:
            import importlib.util

            if importlib.util.find_spec("hf_transfer") is not None:
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        except Exception as _e:
            print("hf_transfer setup skipped:", _e)
        # The base training stack is baked into WORKER_IMAGE; only the per-run extras install here
        # (the verifiers/Freesolo env wheel + the chalk kernels) — exactly the payload's extra_pip.
        extra_pip = payload.get("extra_pip") or []
        if extra_pip:
            # check=True: a deterministic dependency failure must stop NOW with an actionable
            # error, not proceed to a later import crash while the paid instance runs.
            subprocess.run([sys.executable, "-m", "pip", "install", *extra_pip], check=True)
        fetch_code(payload)
        env = build_worker_env(payload)
        deadline = time.time() + float(payload.get("max_wall_s") or 24 * 3600)
        phase = payload["phase"]
        # A reused/retried run dir can carry a previous attempt's metrics file; a stale one would
        # let a crashed train phase report the previous run's metrics. Clear before training.
        for stale in ("/tmp/train_meta.json", "/tmp/metrics.json"):
            with contextlib.suppress(FileNotFoundError):
                os.remove(stale)
        # Train. Nonzero rc tolerated — RL's colocated vLLM can segfault at interpreter exit AFTER
        # the adapter + metrics.json + DONE are saved. The train phase writes metrics.json + DONE
        # itself (or restores them from an earlier attempt's DONE).
        run_mode(payload, env, phase, deadline)
        if not os.path.exists("/tmp/metrics.json"):
            raise RuntimeError(
                f"train phase '{phase}' produced no /tmp/metrics.json (it crashed before "
                f"finishing); see error_{phase}.txt and console_{phase}.txt in the HF dataset repo"
            )
        ok = True
    except Exception as exc:
        # Record genuine failures in the attempt marker (written in `finally`). Don't catch
        # BaseException — KeyboardInterrupt/SystemExit must propagate after the marker write.
        error = f"{type(exc).__name__}: {exc}"
        print(f"bootstrap failed: {error}", flush=True)
    finally:
        write_attempt_marker(payload, ok, error)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
