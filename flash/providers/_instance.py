"""Shared building blocks for the instance-based providers (Lambda, Hyperstack).

Both rent a single-GPU instance and bootstrap it identically: ship a cloud-init ``user_data`` that
runs the prebuilt ``WORKER_IMAGE`` via Docker on the host, detect completion from the worker's HF
artifacts, and guarantee teardown control-plane-side. The per-provider packages differ only in the
REST API (launch/list/terminate) and the capacity model; everything below — the run-derived
sweep-matchable label, the bootstrap payload, and the cloud-init script — is identical, so it lives
here (single source of truth, parameterized by the substrate ``arm`` and the run's image).

The shipped bootstrap is the sibling ``_instance_bootstrap.py``; ``arm`` (e.g. ``lambda`` /
``hyperstack``) travels in ``payload["flash_arm"]`` and decides FLASH_ARM + the ``<arm>_attempt<N>``
marker name.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

# Lambda/Hyperstack cap an instance/VM ``name`` at 64 chars. We keep the label at or under this so
# the name is NEVER silently truncated at launch — truncation would desync the stored name from the
# ``run_label_prefix`` the orphan-sweep matches on, which could fail to protect (or wrongly reap) a
# live run. The seed/attempt suffix ``-s{seed}-a{attempt}`` is bounded (<=12 chars), so the prefix
# is bounded to leave room for it.
_MAX_NAME = 60
_PREFIX_BUDGET = _MAX_NAME - 12

# Above this many chars, the serialized job spec is spilled OUT of the inline cloud-init user_data
# (uploaded to HF; the bootstrap fetches it) so a large inline spec can't overflow the provider's
# user_data size limit and get the launch rejected before a handle is persisted. Below it the spec
# rides inline (the common, tiny-spec case) so launch needs no extra HF round-trip. The cap is well
# under the bootstrap's own 96_000-char execve threshold, and the base64 + heredoc framing inflates
# user_data ~1.4x, so the spilled ceiling keeps a typical run's user_data comfortably small.
_SPEC_SPILL_THRESHOLD = 16_000


def run_label_prefix(run_id: str) -> str:
    """The prefix EVERY instance label for ``run_id`` starts with, bounded to the name budget.

    Platform run ids already start with ``flash-``; anything else (direct-API callers, tests) gets
    the prefix forced. A run id long enough to overflow the provider name cap is shortened
    DETERMINISTICALLY (a stable 8-char hash suffix) so launch AND ``sweep_orphans`` compute the
    IDENTICAL bounded prefix — and two distinct long run ids never collide onto the same name (which
    could otherwise reap the wrong live instance). Short ids (the common case) pass through
    unchanged."""
    base = run_id if run_id.startswith("flash-") else f"flash-{run_id}"
    if len(base) <= _PREFIX_BUDGET:
        return base
    h = hashlib.sha1(base.encode()).hexdigest()[:8]
    return f"{base[: _PREFIX_BUDGET - 9]}-{h}"


def instance_label(run_id: str, seed: int, attempt: int) -> str:
    """Instance name: run-derived so ``sweep_orphans`` can tell ours from anything else on the
    account, and bounded (via ``run_label_prefix``) so the provider never truncates it."""
    return f"{run_label_prefix(run_id)}-s{seed}-a{attempt}"


# The worker container path the per-region cache is bind-mounted at, and the HF cache under it. The
# host mount differs per provider (Lambda NFS /lambda/nfs/<name>; Hyperstack block /mnt/flash-weights)
# but the CONTAINER path is fixed, so HF_HOME is uniform regardless of substrate.
CACHE_CONTAINER_MOUNT = "/weight-cache"
CACHE_HF_HOME = f"{CACHE_CONTAINER_MOUNT}/hf-cache"
# Sentinel file written onto a SUCCESSFULLY-mounted block-volume cache (by the cloud-init preamble),
# so the in-container preload mount-check can tell a real mount from an empty Docker bind (a failed
# attach). Lives on the device itself -> absent when the volume isn't actually mounted.
CACHE_MOUNT_MARKER = ".flash-cache-mounted"


def _cache_block_device_setup(payload: dict) -> str:
    """Cloud-init preamble (block-volume providers, e.g. Hyperstack): wait for the attached volume's
    block device, format it ONCE if it has no filesystem (NEVER reformat a populated cache — guarded
    by ``blkid``), and mount it at the host ``cache_host_mount``. No-op for NFS providers (Lambda
    auto-mounts) and for cold runs. Best-effort: if the device never appears / mount fails, the bind
    falls back to an empty dir (a correct cold run), never a hard failure."""
    if not payload.get("cache_block_device") or not payload.get("cache_host_mount"):
        return ""
    mount = payload["cache_host_mount"]
    # The attached cache volume is provisioned at an EXACT known size, so pick the candidate disk by
    # size (±20%) AND require that neither it nor any of its partitions is mounted. That excludes the
    # boot disk (its partition is mounted at /) and any differently-sized ephemeral/local NVMe — so we
    # never mkfs the wrong device. A warm cache disk (already ext4, unmounted) still matches, and the
    # blkid guard keeps its data. If nothing matches, run cold (format nothing).
    expect_bytes = int(payload.get("cache_size_gb") or 0) * 1000 * 1000 * 1000
    marker = CACHE_MOUNT_MARKER
    return f"""
# --- weight-cache block volume: wait-for-device (size-matched, unmounted), format-if-new, mount ---
echo "FLASH: waiting for the attached cache block device (~{payload.get('cache_size_gb')}GB)..."
EXPECT_BYTES={expect_bytes}
CACHE_DEV=""
for i in $(seq 1 60); do
  for d in $(lsblk -dpbn -o NAME,TYPE,SIZE | awk -v e="$EXPECT_BYTES" \
      '$2=="disk" && e>0 {{lo=e*0.8; hi=e*1.2; if ($3+0>=lo && $3+0<=hi) print $1}}'); do
    # Skip any disk with a mounted partition (boot/data disks in use) — only a free disk is ours.
    if lsblk -pnr -o MOUNTPOINT "$d" | grep -q '[^[:space:]]'; then continue; fi
    CACHE_DEV="$d"; break
  done
  [ -n "$CACHE_DEV" ] && break
  sleep 5
done
if [ -n "$CACHE_DEV" ]; then
  echo "FLASH: cache device $CACHE_DEV"
  blkid "$CACHE_DEV" >/dev/null 2>&1 || mkfs.ext4 -q "$CACHE_DEV" || true   # format ONCE; never reformat a populated cache
  mkdir -p '{mount}'
  if mount "$CACHE_DEV" '{mount}' 2>/dev/null; then
    # Sentinel written ONTO the mounted block device (not the underlying empty dir): it is only
    # visible at the bind path inside the container when the REAL volume is mounted. The preload
    # mount-check requires it, so a failed/absent attach (Docker binding an empty host dir) can't
    # masquerade as a warm cache and silently warm ephemeral disk.
    touch '{mount}/{marker}' 2>/dev/null || true
  else
    echo "FLASH: cache mount failed; running cold"
  fi
else
  echo "FLASH: no matching cache block device appeared; running cold"
fi
"""


def build_payload(
    spec,
    seed: int,
    attempt: int,
    *,
    arm: str,
    runtime_secrets: dict | None = None,
    cache_host_mount: str | None = None,
    cache_block_device: bool = False,
    mode: str | None = None,
    models: list | None = None,
) -> dict:
    """The bootstrap's input — field-compatible with the RunPod ``_train_body`` payload, plus the
    bits the instance can't infer (HF prefix for markers, wall cap, attempt, and the substrate
    ``arm`` that the bootstrap stamps as FLASH_ARM + the marker name).

    ``cache_host_mount`` (set by the provider when it attaches a per-region weight cache) points
    HF_HOME at the bind-mounted cache (``/weight-cache/hf-cache``) instead of stripping the
    RunPod redirect; ``cache_block_device`` adds the format/mount preamble for block-volume providers.
    """
    from flash.envs.registry import worker_pip_for_env
    from flash.providers.runpod.train import (
        build_worker_env,
        chalk_extra_pip,
        strip_runpod_volume_env,
    )

    # Start from the shared env with the RunPod /runpod-volume redirect stripped (that mount is
    # RunPod-only). If THIS provider attached a cache, point HF_HOME at the instance cache mount —
    # but DON'T clobber a per-run [worker_env].HF_HOME the user set on purpose. build_worker_env
    # merges [worker_env] LAST, so a user override survives the strip above (only /runpod-volume-
    # rooted vars are stripped); on RunPod that override wins, so honor it here too for parity. We
    # only install the cache path when HF_HOME is absent (i.e. the platform redirect was stripped and
    # the user set nothing).
    env = strip_runpod_volume_env(build_worker_env(spec, seed, runtime_secrets=runtime_secrets))
    if cache_host_mount and not env.get("HF_HOME"):
        env["HF_HOME"] = CACHE_HF_HOME
    payload = {
        "hf_repo": spec.train.hf_repo,
        "job_spec_json": spec.to_json(),
        "phase": spec.phase,
        "seed": int(seed),
        "flash_arm": arm,
        "env": env,
        # The bootstrap pip-installs extra_pip for every job, so the per-run env wheel + the opt-in
        # chalk spec ride along here to reach default runs (mirrors runpod/jobs.submit_run).
        "extra_pip": (list(spec.environment.pip) or worker_pip_for_env(spec.environment.id))
        + chalk_extra_pip(spec),
        "hf_prefix": f"{spec.phase}/{spec.run_id}/seed{seed}",
        "max_wall_s": max(60, int(spec.gpu.max_wall_seconds)),
        "attempt": int(attempt),
    }
    if cache_host_mount:
        payload["cache_host_mount"] = cache_host_mount
        if cache_block_device:
            payload["cache_block_device"] = True
            # Carry the mount sentinel filename so the bootstrap's mount-check reads it from ONE source
            # of truth (this constant) instead of re-hardcoding the literal — the cloud-init preamble
            # (_cache_block_device_setup) writes the same CACHE_MOUNT_MARKER, so they can't drift.
            payload["cache_mount_marker"] = CACHE_MOUNT_MARKER
            # The block-device preamble matches the attached volume by its EXACT provisioned size, so
            # carry the runner-assigned size (falls back to the default cache size). Parse tolerantly
            # via _volume_gb so a non-int / stale spec value ("0", "", "abc", bool) can't crash the
            # instance bootstrap on this best-effort device-matching hint — it defaults instead.
            from flash.runner import WEIGHT_CACHE_VOLUME_GB
            from flash.spec import _volume_gb

            payload["cache_size_gb"] = _volume_gb(
                getattr(spec.gpu, "network_volume_gb", None), default=WEIGHT_CACHE_VOLUME_GB
            )
    # Preload (warm) mode: the bootstrap downloads ``models`` into the mounted cache and exits — no
    # code fetch, no worker. Only meaningful with a cache attached (else there's nothing to warm).
    if mode:
        payload["mode"] = mode
        payload["models"] = list(models or [])
    return payload


# Host helper: best-effort upload of the consolidated boot log to HF. Neither Lambda nor Hyperstack
# exposes an instance console/log API, so the box pushes its own boot log to HF — the only window
# into a failure BEFORE the worker container can write its own artifacts (docker/GPU not ready,
# image pull failure). Reads creds from the on-box payload.json. Never raises.
_HOSTLOG_PY = """\
import json
try:
    p = json.load(open("/opt/flash/payload.json"))
    from huggingface_hub import HfApi
    HfApi(token=(p.get("env") or {}).get("HF_TOKEN")).upload_file(
        path_or_fileobj="/opt/flash/host_boot.log",
        path_in_repo=p["hf_prefix"] + "/" + p.get("flash_arm", "instance") + "_boot.log",
        repo_id=p["hf_repo"],
        repo_type="dataset",
    )
except Exception:
    pass
"""

# Host helper: write the attempt-failure marker (<arm>_attempt<N>.json, ok=false, RETRIABLE) to HF
# when the box can't even start the worker container (docker/GPU never ready, image pull failure).
# Without it a pre-container failure leaves NO marker, so the poller would burn the whole setup
# grace (~50 min) before reporting a generic stall; this surfaces a fast, RETRYABLE failure so the
# runner re-provisions on a fresh host immediately. Reads creds from the on-box payload.json.
#
# CRITICAL: the worker OWNS this marker path. A container that starts but fast-fails on a real,
# non-retriable user/config error can exit before the host's ~5s liveness check, having ALREADY
# uploaded its own ok=false marker (the TRUE error) here. The host must NOT overwrite it with a
# RETRIABLE host marker — that would relabel a genuine user error as job_preempted and silently
# retry / hide the root cause. So this writes the host marker ONLY when no worker attempt marker
# yet exists at the path (i.e. the container never got far enough to write one). The check is
# best-effort: on a read error it stays conservative and SKIPS the write (never clobbers).
_FAILMARK_PY = """\
import json, sys
try:
    p = json.load(open("/opt/flash/payload.json"))
    arm = p.get("flash_arm", "instance"); att = int(p.get("attempt") or 0)
    reason = sys.argv[1] if len(sys.argv) > 1 else "host boot failure"
    marker_path = p["hf_prefix"] + "/" + arm + "_attempt" + str(att) + ".json"
    from huggingface_hub import HfApi
    api = HfApi(token=(p.get("env") or {}).get("HF_TOKEN"))
    try:
        worker_marker_exists = api.file_exists(repo_id=p["hf_repo"], filename=marker_path, repo_type="dataset")
    except Exception:
        worker_marker_exists = True  # conservative: on a read error, never risk clobbering
    if not worker_marker_exists:
        open("/opt/flash/fm.json", "w").write(json.dumps({"ok": False, "attempt": att, "retriable": True, "error": "host: " + reason}))
        api.upload_file(
            path_or_fileobj="/opt/flash/fm.json",
            path_in_repo=marker_path,
            repo_id=p["hf_repo"], repo_type="dataset",
        )
except Exception:
    pass
"""


def _spill_large_spec_to_hf(payload: dict) -> dict:
    """Keep a large ``job_spec_json`` OUT of the inline cloud-init user_data.

    A tiny spec already yields ~17 KB of cloud-init; a 100 KB inline param balloons user_data past
    typical provider/cloud-init user-data caps and the launch is rejected before any handle is
    persisted (an unrecoverable, billing-invisible failure). When the spec is large we upload it to
    the run's HF dataset repo at ``<hf_prefix>/job_spec.json`` and replace the inline value with a
    small ``job_spec_in_hf`` sentinel; the bootstrap fetches it from the SAME repo it already pulls
    code from. Small specs (the common case) ride inline unchanged — no extra HF round-trip.

    Returns the payload to embed (a shallow copy when spilled, else the original).
    """
    spec_json = payload.get("job_spec_json") or ""
    if len(spec_json) <= _SPEC_SPILL_THRESHOLD:
        return payload
    from huggingface_hub import HfApi

    # Wrap the bytes in BytesIO: huggingface_hub.upload_file accepts a path-like for
    # path_or_fileobj, and raw ``bytes`` is itself a valid path type, so it could be
    # misinterpreted as a (huge) filesystem path. BytesIO makes it an unambiguous file-like upload.
    HfApi(token=(payload.get("env") or {}).get("HF_TOKEN")).upload_file(
        path_or_fileobj=io.BytesIO(spec_json.encode("utf-8")),
        path_in_repo=f"{payload['hf_prefix']}/job_spec.json",
        repo_id=payload["hf_repo"],
        repo_type="dataset",
    )
    spilled = dict(payload)
    spilled["job_spec_json"] = ""
    spilled["job_spec_in_hf"] = True
    return spilled


def build_user_data(payload: dict, *, image: str) -> str:
    """Cloud-init ``user_data``: run the worker ``image`` via Docker on the host.

    cloud-init runs this once at first boot as root. Everything dynamic travels base64-encoded
    inside the script (never interpolated into shell syntax), so the job-spec JSON survives
    byte-exact. The full training stack is baked into the image, so the box only needs Docker + an
    NVIDIA GPU — both shipped by the providers' default Docker-capable images — and the container
    does the rest (fetch code from HF, run the worker, stream artifacts back to HF).

    A large job spec is spilled to HF first (see ``_spill_large_spec_to_hf``) so it never inflates
    user_data past the provider's size cap.

    Secrets-wise the script carries the same content as the worker env on RunPod (HF token, env
    secrets). The operator's provider API key is NEVER shipped (teardown is control-plane-side via
    the runner ``finally`` / poll deadline / ``sweep_orphans``).
    """
    payload = _spill_large_spec_to_hf(payload)
    payload_b64 = base64.encodebytes(json.dumps(payload).encode()).decode()
    bootstrap_src = (Path(__file__).parent / "_instance_bootstrap.py").read_text()
    # Weight cache: the provider mounts its region-scoped persistent storage on the HOST at
    # ``cache_host_mount`` (Lambda auto-mounts its NFS filesystem there; Hyperstack's preamble below
    # formats+mounts the attached block device there). Bind it into the worker container at the FIXED
    # ``/weight-cache`` so the worker's HF_HOME=/weight-cache/hf-cache (set in build_payload) persists
    # the model download across runs in this region. Absent -> no bind (cold run).
    cache_host_mount = payload.get("cache_host_mount")
    # Single-quote the host path in the docker -v (defensive; the path is a controlled constant).
    cache_bind = f"-v '{cache_host_mount}':{CACHE_CONTAINER_MOUNT} \\\n  " if cache_host_mount else ""
    cache_setup = _cache_block_device_setup(payload)
    return f"""#!/bin/bash
# Flash instance worker (generated by flash.providers._instance.build_user_data; arm={payload.get('flash_arm')})
set -x
mkdir -p /opt/flash
# Consolidate ALL boot output (this script + the container) into one host log the uploader ships
# to HF — neither substrate has a console API, so this is the only window into a pre-worker failure.
exec >>/opt/flash/host_boot.log 2>&1
cat > /opt/flash/payload.b64 <<'FLASH_PAYLOAD_EOF'
{payload_b64}FLASH_PAYLOAD_EOF
base64 -d /opt/flash/payload.b64 > /opt/flash/payload.json
cat > /opt/flash/bootstrap.py <<'FLASH_BOOTSTRAP_EOF'
{bootstrap_src}FLASH_BOOTSTRAP_EOF
cat > /opt/flash/hostlog.py <<'FLASH_HOSTLOG_EOF'
{_HOSTLOG_PY}FLASH_HOSTLOG_EOF
cat > /opt/flash/failmark.py <<'FLASH_FAILMARK_EOF'
{_FAILMARK_PY}FLASH_FAILMARK_EOF
IMAGE={image!r}
# huggingface_hub on the host for the boot-log + failure-marker uploaders (best-effort).
pip3 install -q huggingface_hub >/dev/null 2>&1 \\
  || python3 -m pip install -q --break-system-packages huggingface_hub >/dev/null 2>&1 || true
fail() {{ echo "FLASH: $1" >&2; python3 /opt/flash/failmark.py "$1" >/dev/null 2>&1 || true; exit 1; }}
# The provider's default image ships Docker + the NVIDIA Container Toolkit, but cloud-init can run
# before they finish initializing — wait for both (up to ~10 min) before launching the worker.
for i in $(seq 1 100); do
  if docker info >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then break; fi
  echo "FLASH: waiting for docker+gpu ($i)"; sleep 6
done
docker info >/dev/null 2>&1 || fail "docker never became ready"
nvidia-smi >/dev/null 2>&1 || fail "gpu never became ready"
{cache_setup}
# Pull with retries (the image is large; a transient registry blip must not fail the run). On total
# failure, write a RETRYABLE marker and exit NOW instead of leaving a billed box idling the whole
# setup grace with no DONE/marker.
PULLED=0
for i in 1 2 3 4 5; do docker pull "$IMAGE" && {{ PULLED=1; break; }}; echo "FLASH: pull retry $i"; sleep 20; done
[ "$PULLED" -eq 1 ] || fail "worker image pull failed after retries"
# Run the worker container detached so cloud-init completes promptly; completion is signaled via the
# worker's HF artifacts (DONE/metrics.json/marker), never a return channel from the box.
docker run -d --name flashrun --gpus all --shm-size=16g --network host \\
  -v /opt/flash:/root/flash {cache_bind}-w /root/flash \\
  "$IMAGE" python /root/flash/bootstrap.py || fail "docker run failed"
sleep 5
# The container must be running OR have already exited CLEANLY. The bootstrap returns 0 ONLY on
# genuine success (it confirms metrics.json and uploads its ok-marker first) — so an exit code of 0
# is itself the success signal (e.g. an already-complete retry that finished in <5s), and the host
# must NOT write any marker for it: the worker OWNS the attempt marker, and writing to that path here
# would clobber its ok-marker (HF listing can lag the worker's just-finished upload). A NON-zero
# exit reaches fail(), but its failmark uploader is itself marker-aware: a container that started
# and then fast-failed on a real user/config error has ALREADY written its own ok=false marker here,
# and the host failmark SKIPS the write when that marker exists (so a genuine user error is never
# relabeled retriable/job_preempted). Only a never-started container — no worker marker — gets the
# retriable host failmark.
if ! docker ps --filter name=flashrun --filter status=running -q | grep -q .; then
  EXIT="$(docker inspect -f '{{{{.State.ExitCode}}}}' flashrun 2>/dev/null || echo 1)"
  docker logs flashrun >>/opt/flash/host_boot.log 2>&1 || true
  [ "$EXIT" = "0" ] || fail "worker container did not start (exit ${{EXIT}})"
fi
# Mirror the container's stdout into the host boot log (detached) so an early in-container crash is
# visible on HF even if it dies before uploading its own console artifact.
( docker logs -f flashrun >>/opt/flash/host_boot.log 2>&1 || true ) &
disown || true
# Host->HF boot-log uploader: THROTTLED to 120s and STOPPED once the container exits (bounded ~30
# min). The worker itself uploads rate-limited heartbeats/console once running, so a 30s diagnostic
# loop for the whole run would risk Hugging Face's per-repo hourly commit cap and starve the
# required metrics/DONE commits.
( for i in $(seq 1 15); do
    python3 /opt/flash/hostlog.py >/dev/null 2>&1 || true
    docker ps --filter name=flashrun --filter status=running -q | grep -q . || break
    sleep 120
  done ) &
disown || true
"""
