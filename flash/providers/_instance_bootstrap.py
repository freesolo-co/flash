"""Self-contained bootstrap shared by the instance-based providers (Lambda, Hyperstack).

Runs INSIDE the worker container on a rented GPU instance. Both providers' cloud-init ``user_data``
runs the prebuilt, PUBLIC ``WORKER_IMAGE`` via Docker on the host, and this module is the
container's command: install the run's extra pip deps, fetch the flash package from the HF dataset
repo, then run the substrate-neutral worker (``flash.engine.worker``) to train, uploading the
console tail to HF.

There is NO return channel from the instance: the worker's HF artifacts
(DONE/metrics.json/heartbeat.json) are the success signal, and the attempt-scoped
``<arm>_attempt<N>.json`` marker (``arm`` = the substrate, e.g. ``lambda``/``hyperstack``) is the
terminal marker the control plane keys failures on. The full training stack is BAKED into the
image, so there is no base-stack install here — only the per-run ``extra_pip``.

Shipped verbatim inside the container command, so it must stay self-contained: stdlib +
huggingface_hub (baked into the image) only — never import flash here. It reads its payload from
``/root/flash/payload.json``; the substrate name travels in ``payload["flash_arm"]``.
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


class RetriableBootstrapError(RuntimeError):
    """An infra-shaped bootstrap failure that should RETRY on a fresh host, not fail the run.

    The control-plane pollers classify an attempt marker carrying ``retriable=True`` as
    ``job_preempted`` (retried within the HF infra-retry budget) rather than ``job_failed`` (fails
    fast). The bootstrap is self-contained (can't import the worker's ``RetriableInfraError``), so
    this local sentinel marks the same shape: an HF-side failure (the spilled-spec fetch, or a
    required-artifact upload that never landed) that a retry on a healthy host would clear. ``main``
    keys the marker's ``retriable`` flag off whether the raised error is an instance of this."""


def load_payload() -> dict:
    with open(PAYLOAD_PATH) as f:
        return json.load(f)


def _arm(payload: dict) -> str:
    return str(payload.get("flash_arm") or "instance")


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


def hf_file_exists(payload: dict, repo_subpath: str) -> bool:
    """True iff ``<hf_prefix>/<repo_subpath>`` exists in the run's HF dataset repo.

    Used to confirm a worker's REQUIRED completion artifacts actually reached HF before the
    bootstrap treats a non-zero worker exit as success — a local /tmp/metrics.json is NOT proof,
    since the worker writes it locally before the (required, retried) upload that can still fail
    infra-shaped. Raises on a genuine API error so the caller can be conservative."""
    from huggingface_hub import HfApi

    api = HfApi(token=(payload.get("env") or {}).get("HF_TOKEN"))
    return api.file_exists(
        repo_id=payload["hf_repo"],
        filename=f"{payload['hf_prefix']}/{repo_subpath}",
        repo_type="dataset",
    )


def remote_completion_confirmed(payload: dict) -> bool:
    """True iff the worker's required completion artifacts (DONE + metrics.json) are on HF.

    The worker uploads metrics.json then DONE, both ``required=True`` (3 retries, then raises a
    RetriableInfraError -> non-zero exit). Confirming the REMOTE artifacts — not just the local
    /tmp/metrics.json — is the only proof the run actually finished; a transient upload failure
    after the local file exists must propagate as a retriable failure, not a false ok=true."""
    try:
        return hf_file_exists(payload, "DONE") and hf_file_exists(payload, "metrics.json")
    except Exception as exc:
        # A read error here is itself infra-shaped; stay conservative (treat as unconfirmed) so a
        # non-zero worker exit propagates and retries rather than masking the failure.
        print(f"remote-completion check warn: {exc}", flush=True)
        return False


def fetch_spec_from_hf(payload: dict) -> str:
    """Download the run's spec spilled out-of-band to HF (``<hf_prefix>/job_spec.json``).

    A large inline job spec (100s of KB of env params) would blow the provider's cloud-init
    ``user_data`` size cap and get the launch rejected before any handle is persisted, so the
    control plane (``_instance.build_user_data``) keeps it OUT of ``user_data`` and uploads it to
    the run's HF dataset repo instead, leaving only a sentinel in the payload. The bootstrap fetches
    it here (the code is already fetched from the same repo, so this adds no new dependency)."""
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=payload["hf_repo"],
        repo_type="dataset",
        filename=f"{payload['hf_prefix']}/job_spec.json",
        token=(payload.get("env") or {}).get("HF_TOKEN"),
    )
    with open(local) as f:
        return f.read()


def build_worker_env(payload: dict) -> dict:
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (payload.get("env") or {}).items()})
    # The job spec may have been spilled to HF at launch (a large inline spec would overflow the
    # provider's cloud-init user_data cap); fetch it here when only the sentinel rode in the payload.
    spec_json = payload.get("job_spec_json")
    if not spec_json and payload.get("job_spec_in_hf"):
        # This fetch is the FIRST HF round-trip in the bootstrap and runs pre-worker (so the worker
        # never starts and can't stamp a retriable heartbeat). Any failure here is infra-shaped, so
        # surface it as RetriableBootstrapError — otherwise main() would mark ok=false with no
        # retriable flag and the poller would fail the run fast (job_failed) instead of retrying it
        # on a fresh host (job_preempted). A permanently missing/unreadable spec (a control-plane bug
        # on the rare spilled-spec path) just burns the bounded infra-retry budget, then fails.
        try:
            spec_json = fetch_spec_from_hf(payload)
        except Exception as e:
            raise RetriableBootstrapError(
                f"failed to fetch the spilled job spec from HF: {e}"
            ) from e
    if not spec_json:
        # Neither an inline spec NOR the spilled-to-HF sentinel rode in the payload: a malformed
        # payload (the control plane always sets exactly one). Fail loudly with the cause instead of
        # crashing on the len(None) below with an opaque TypeError that buries the real problem.
        raise RuntimeError(
            "bootstrap payload carries no job spec: both job_spec_json and the job_spec_in_hf "
            "sentinel are absent/empty — the control plane built an invalid worker payload"
        )
    # Pass a large spec via a file, not the environment: a job spec with large inline params can
    # reach hundreds of KB, which trips execve's "Argument list too long". Mirrors
    # runpod/train.py:_train_body.
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
    # rented instance, so override it to the real backend carried in the payload.
    env["FLASH_ARM"] = _arm(payload)
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

    On failure/SUCCESS (FLASH_UPLOAD_CONSOLE) the console tail is uploaded as console_<mode>.txt.
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
        # Seek to the last 64k instead of reading the whole file: on long-running jobs the
        # console grows unbounded and this runs on a periodic loop, so an O(n) read each pass
        # would balloon the bootstrap container's memory/time. Read binary + decode with
        # errors="replace" so a seek landing mid-UTF-8-sequence can't raise.
        with open(console, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 64_000))
            tail = f.read().decode("utf-8", "replace")
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
            # Honor the wall-clock deadline: wait only up to the time left (floored to a small
            # positive so the call never blocks forever on a 0/negative timeout). A prior ``max(10.0,
            # …)`` floor could overshoot the deadline by ~10s when little/no time remained — that
            # leftover 10s is paid GPU time past the run's wall cap, so we clamp to the remaining
            # budget instead.
            proc.wait(timeout=max(1.0, deadline_ts - time.time()))
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


def write_attempt_marker(payload: dict, ok: bool, error: str = "", retriable: bool = False) -> None:
    """Attempt-scoped terminal marker (``<arm>_attempt<N>.json``): how the control plane
    distinguishes THIS attempt's failure from a prior attempt's leftovers under the same prefix.

    ``retriable`` stamps the same flag the host failmark and the worker heartbeat use: the pollers
    read ``marker.get("retriable")`` and classify a flagged failure as ``job_preempted`` (retried on
    a fresh host within the HF infra budget) instead of ``job_failed`` (fails fast). Set it for
    infra-shaped bootstrap failures (HF fetch/upload) so an HF outage doesn't burn the run."""
    marker = {
        "ok": bool(ok),
        "ts": time.time(),
        "attempt": int(payload.get("attempt") or 0),
        "retriable": bool(retriable),
        "error": error[:2000],
    }
    p = "/tmp/attempt_marker.json"
    with open(p, "w") as f:
        json.dump(marker, f)
    hf_upload(payload, p, f"{_arm(payload)}_attempt{marker['attempt']}.json")


def _arm_preload_wall_cap(payload: dict) -> threading.Timer | None:
    """Arm the preload path's wall-clock cap. The training path enforces ``max_wall_s`` by
    ``run_mode`` killing the worker SUBPROCESS on its deadline, but ``run_preload`` runs
    ``snapshot_download`` IN-PROCESS — a hung Hub download (or a stalled NIC) has no subprocess to
    time out, so without this the paid Lambda/Hyperstack box can keep running long past ``timeout_s``
    (the control-plane driver's ``terminate_run_instances`` only fires if that driver process is
    still alive, and nothing on the box self-terminates). Mirror the deadline here: a daemon timer
    writes a terminal failure marker (so the warm driver stops polling and the box can be reaped) and
    HARD-exits the process — ``os._exit`` because a blocked C-level socket read in ``snapshot_download``
    can't be unwound by a Python exception/signal. Returns the Timer so the caller can cancel it on a
    clean finish (the ``finally`` marker upload then runs normally)."""
    wall_s = float(payload.get("max_wall_s") or 0)
    if wall_s <= 0:
        return None

    def _fire() -> None:
        msg = f"preload exceeded the wall-clock cap ({int(wall_s)}s); self-terminating box"
        print(f"FLASH: {msg}", flush=True)
        # Best-effort terminal marker so the driver/sweeper sees a terminal failure instead of polling
        # to its own timeout. Never block the hard exit on it.
        with contextlib.suppress(Exception):
            write_attempt_marker(payload, ok=False, error=msg)
        os._exit(1)

    timer = threading.Timer(wall_s, _fire)
    timer.daemon = True
    timer.start()
    return timer


def run_preload(payload: dict) -> dict:
    """Download-only warm: pull the requested models into the bind-mounted cache (HF_HOME) and exit.

    The instance-provider mirror of runpod/train/endpoints._train_body's ``preload`` branch. NO
    training, NO env code, NO worker subprocess — just ``snapshot_download`` straight into the cache so
    the very first real run in this region is warm. HF_HOME (from the payload env) is rooted at the
    per-region bind-mounted cache mount; we pass ``cache_dir`` EXPLICITLY (huggingface_hub freezes
    HF_HUB_CACHE at import, so setting the env var here would be too late) and FAIL if the cache isn't
    mounted (otherwise we'd warm ephemeral local disk that vanishes with the box).
    """
    env = payload.get("env") or {}
    hf_home = env.get("HF_HOME") or ""
    token = env.get("HF_TOKEN")
    # The cache bind-mount must be present; HF_HOME is <mount>/hf-cache, so its parent is the mount.
    # Checked BEFORE importing huggingface_hub so a missing mount fails fast (and stays testable).
    mount = os.path.dirname(hf_home.rstrip("/")) if hf_home else ""
    if not hf_home or not mount or not os.path.isdir(mount):
        return {"preloaded": [], "already_cached": [], "failed": {},
                "error": f"weight-cache not mounted (HF_HOME={hf_home!r}); refusing to warm ephemeral disk"}
    # Require the mount sentinel for BOTH substrates. The cloud-init preamble drops it ONLY onto a
    # verified-real mount: block-volume (Hyperstack) writes it after the device mounts; NFS (Lambda)
    # writes it after confirming ``mountpoint`` (the platform auto-mount actually took). It is therefore
    # visible here only when the REAL cache is mounted. Without it, Docker's ``-v`` bind silently
    # auto-creates an EMPTY host dir -> the mount exists (isdir passes) but the sentinel is absent, which
    # would otherwise warm EPHEMERAL disk (discarded at teardown) yet report a successful warm. The
    # marker filename flows in via the payload from _instance.CACHE_MOUNT_MARKER (ONE source of truth);
    # the literal is only a defensive fallback for an older payload that predates the field. A
    # cache-attached preload payload always carries cache_mount_marker, so absence of the field is
    # treated as "no sentinel expected" only when no cache mount was requested at all.
    if payload.get("cache_mount_marker"):
        marker = os.path.join(mount, payload["cache_mount_marker"])
        if not os.path.exists(marker):
            kind = "block volume" if payload.get("cache_block_device") else "NFS filesystem"
            return {"preloaded": [], "already_cached": [], "failed": {},
                    "error": (f"weight-cache {kind} not mounted (no sentinel at {marker}); "
                              "refusing to warm ephemeral disk")}
    from huggingface_hub import snapshot_download

    cache_dir = os.path.join(hf_home, "hub")
    # weights + tokenizer/config only (same exclusions as prefetch_model / the image bake / the RunPod
    # preload branch) so the warmed cache matches exactly what workers later fetch.
    ignore_patterns = ["*.pth", "*.gguf", "original/*", "*.onnx", "*.msgpack", "*.h5"]
    done, already, failed = [], [], {}
    for repo_id in payload.get("models") or []:
        try:
            # Idempotent: probe with local_files_only (HF's own resolution, NOT a dir-name guess) — if
            # the snapshot is already on the volume, skip the network download. Mirrors the RunPod
            # preload branch; accurate (no repo_id.replace heuristic) and avoids re-downloading.
            try:
                snapshot_download(repo_id=repo_id, token=token, cache_dir=cache_dir,
                                  ignore_patterns=ignore_patterns, local_files_only=True)
                already.append(repo_id)
                print(f"preload: {repo_id} -> {cache_dir} (cached)", flush=True)
                continue
            except Exception:
                pass
            snapshot_download(repo_id=repo_id, token=token, cache_dir=cache_dir,
                              ignore_patterns=ignore_patterns)
            done.append(repo_id)
            print(f"preload: {repo_id} -> {cache_dir} (downloaded)", flush=True)
        except Exception as exc:
            failed[repo_id] = str(exc)
            print(f"preload FAILED {repo_id}: {exc}", flush=True)
    return {"preloaded": done, "already_cached": already, "failed": failed}


def main() -> int:
    # Make SIGTERM (docker stop / wall-cap) unwind through finally so the terminal marker still
    # gets uploaded.
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(1))
    payload = load_payload()
    ok = False
    error = ""
    retriable = False
    try:
        # hf_transfer is baked into the worker image; enable it so model pulls saturate the NIC.
        try:
            import importlib.util

            if importlib.util.find_spec("hf_transfer") is not None:
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        except Exception as _e:
            print("hf_transfer setup skipped:", _e)
        # Preload (warm) mode: download-only into the mounted cache, then exit. No code fetch, no
        # extra_pip, no worker subprocess — the warm driver (warm_instances) detects completion by
        # polling the `<prefix>/preload_result.json` we upload just below (the attempt marker is still
        # written in the finally, but it is NOT the preload completion signal).
        if payload.get("mode") == "preload":
            # Enforce the wall cap on the in-process download (the training path enforces it on the
            # worker subprocess via run_mode; preload has no subprocess, so arm a watchdog here).
            wall_timer = _arm_preload_wall_cap(payload)
            try:
                result = run_preload(payload)
            finally:
                if wall_timer is not None:
                    wall_timer.cancel()
            with open("/tmp/preload_result.json", "w") as f:
                json.dump(result, f)
            hf_upload(payload, "/tmp/preload_result.json", "preload_result.json")
            ok = not result.get("error") and not result.get("failed")
            error = result.get("error") or (f"models failed: {sorted(result.get('failed') or {})}" if result.get("failed") else "")
            return 0 if ok else 1
        # The base training stack is baked into WORKER_IMAGE; only the per-run extras install here
        # (the verifiers/Freesolo env wheel + the chalk kernels) — exactly the payload's extra_pip.
        extra_pip = payload.get("extra_pip") or []
        if extra_pip:
            subprocess.run([sys.executable, "-m", "pip", "install", *extra_pip], check=True)
        fetch_code(payload)
        env = build_worker_env(payload)
        deadline = time.time() + float(payload.get("max_wall_s") or 24 * 3600)
        phase = payload["phase"]
        for stale in ("/tmp/train_meta.json", "/tmp/metrics.json"):
            with contextlib.suppress(FileNotFoundError):
                os.remove(stale)
        # Train. A non-zero rc is tolerated ONLY when the run genuinely finished: RL's colocated
        # vLLM can segfault at interpreter exit AFTER the adapter + metrics.json + DONE are saved
        # AND uploaded. The local /tmp/metrics.json is NOT sufficient proof — the worker writes it
        # locally before the required (retried) upload, so a transient RetriableInfraError uploading
        # metrics.json/DONE leaves the local file present yet the run UNFINISHED (no remote
        # artifacts). In that case the worker exits non-zero; honor it and let the run retry instead
        # of stamping a false ok=true.
        rc = run_mode(payload, env, phase, deadline)
        if not os.path.exists("/tmp/metrics.json"):
            raise RuntimeError(
                f"train phase '{phase}' produced no /tmp/metrics.json (it crashed before "
                f"finishing); see error_{phase}.txt and console_{phase}.txt in the HF dataset repo"
            )
        if rc != 0 and not remote_completion_confirmed(payload):
            # The local metrics.json exists but the REQUIRED uploads (DONE/metrics.json) never landed
            # on HF — an upload/HF-infra failure, not a code error. Surface it as retriable so the
            # poller retries on a fresh host (job_preempted) within the HF infra budget instead of
            # failing the run fast. During a full HF outage the worker's own retriable heartbeat may
            # also be missing, so the marker's retriable flag is what carries the classification.
            raise RetriableBootstrapError(
                f"train phase '{phase}' exited non-zero ({rc}) and its required completion "
                f"artifacts (DONE/metrics.json) are not on HF — the run did not finish (e.g. a "
                f"failed upload after the local metrics.json was written); see error_{phase}.txt "
                f"and console_{phase}.txt in the HF dataset repo"
            )
        ok = True
    except BaseException as exc:  # incl. SIGTERM's SystemExit / KeyboardInterrupt
        # SIGTERM (docker stop / wall cap) raises SystemExit via the handler above; catching only
        # Exception would skip it, uploading an ok=false marker with an EMPTY error and obscuring the
        # cause from reattach/debugging. BaseException records a useful error and still re-exits
        # nonzero (return 1) with the marker written in `finally`.
        error = f"{type(exc).__name__}: {exc}"
        # An infra-shaped bootstrap failure (the pre-worker spilled-spec HF fetch, or a required
        # artifact that never uploaded) is raised as RetriableBootstrapError so the marker carries
        # retriable=True and the poller retries on a fresh host instead of failing the run fast.
        retriable = isinstance(exc, RetriableBootstrapError)
        print(f"bootstrap failed: {error}", flush=True)
    finally:
        write_attempt_marker(payload, ok, error, retriable=retriable)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
