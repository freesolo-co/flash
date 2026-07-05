"""Shared building blocks for the instance-based providers (Lambda, Vast)."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

# Bounded so the name is never truncated at launch — truncation desyncs the sweep-matched prefix.
_MAX_NAME = 60
_SUFFIX_BUDGET = 12
_PREFIX_BUDGET = _MAX_NAME - _SUFFIX_BUDGET

# Above this, the job spec is spilled to HF so a large inline spec can't overflow the user_data cap.
_SPEC_SPILL_THRESHOLD = 16_000


def run_label_prefix(run_id: str) -> str:
    """The prefix EVERY instance label for ``run_id`` starts with, bounded to the name budget."""
    base = run_id if run_id.startswith("flash-") else f"flash-{run_id}"
    if len(base) <= _PREFIX_BUDGET:
        return base
    h = hashlib.sha1(base.encode()).hexdigest()[:8]
    return f"{base[: _PREFIX_BUDGET - 9]}-{h}"


def label_matches_run(label: str, prefix: str) -> bool:
    """True iff ``label`` belongs to the run whose prefix is ``prefix`` — an EXACT match, or the prefix
    followed by the ``-s`` seed boundary. Boundary-anchored so ``flash-100`` never matches
    ``flash-1000-...`` (or vice versa). The single label-ownership test every instance provider's sweep
    and run-scoped teardown shares."""
    return label == prefix or label.startswith(prefix + "-s")


def instance_label(run_id: str, seed: int, attempt: int) -> str:
    """Instance name: run-derived so ``sweep_orphans`` can tell ours from anything else on the
    account, and bounded (via ``run_label_prefix``) so the provider never truncates it."""
    try:
        seed_i = int(seed)
    except (TypeError, ValueError):
        seed_i = 0
    try:
        attempt_i = int(attempt)
    except (TypeError, ValueError):
        attempt_i = 0
    seed_s, attempt_s = str(seed_i), str(attempt_i)
    # Bound the whole suffix to _SUFFIX_BUDGET: split the digit budget between attempt and seed.
    digit_budget = _SUFFIX_BUDGET - len("-s-a")
    attempt_s = attempt_s[: max(1, min(len(attempt_s), max(1, digit_budget - 1)))]
    seed_s = seed_s[: max(0, digit_budget - len(attempt_s))]
    return f"{run_label_prefix(run_id)}-s{seed_s}-a{attempt_s}"


@dataclass
class InstanceJobHandle:
    """Fields + (de)serialization common to every rent-a-box provider handle (Lambda, Vast).

    Persisted in ``RunStatus.remote`` so any process can reattach/cancel (cf. ``base.JobHandle``). Each
    provider subclass tags its ``provider`` and adds its own locator fields (via ``_extra_to_dict`` /
    ``_extra_from_dict``); the shared ``to_dict``/``from_dict`` carry the common set so the serialized
    shape stays byte-identical across providers. ``instance_id`` is the poll/destroy target, so unlike
    every other field it has NO safe default — a missing/uncoercible one is a corrupt handle that must
    fail with a clear, actionable error (not a bare KeyError/ValueError that crashes reattach/recovery).
    """

    instance_id: int | str
    gpu: str
    hourly_usd: float
    attempt: int
    started_ts: float

    provider: ClassVar[str] = "instance"

    @staticmethod
    def _coerce_instance_id(raw: Any) -> Any:
        """Provider-specific instance_id coercion (Vast=int, Lambda=str). Overridden per subclass."""
        return raw

    def _extra_to_dict(self) -> dict:
        """Provider-specific fields, serialized between ``instance_id`` and the shared tail."""
        return {}

    @staticmethod
    def _extra_from_dict(d: dict) -> dict:
        """Provider-specific fields, parsed from a persisted handle dict (kwargs for the constructor)."""
        return {}

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "instance_id": self.instance_id,
            **self._extra_to_dict(),
            "gpu": self.gpu,
            "hourly_usd": self.hourly_usd,
            "attempt": self.attempt,
            "started_ts": self.started_ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> InstanceJobHandle:
        try:
            instance_id = cls._coerce_instance_id(d["instance_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"corrupt {cls.provider} handle: missing/non-numeric instance_id "
                f"({d.get('instance_id')!r}) in persisted handle {d!r}"
            ) from exc
        return cls(
            instance_id=instance_id,
            gpu=str(d.get("gpu") or ""),
            hourly_usd=float(d.get("hourly_usd") or 0),
            attempt=int(d.get("attempt") or 0),
            started_ts=float(d.get("started_ts") or 0),
            **cls._extra_from_dict(d),
        )


# Fixed container path the per-region cache is bind-mounted at (host mount differs per provider).
CACHE_CONTAINER_MOUNT = "/weight-cache"
CACHE_HF_HOME = f"{CACHE_CONTAINER_MOUNT}/hf-cache"
# Sentinel on a successfully-mounted cache so the preload check can tell a real mount from an empty bind.
CACHE_MOUNT_MARKER = ".flash-cache-mounted"


def _cache_block_device_setup(payload: dict) -> str:
    """Cloud-init preamble (block-volume providers): wait for the attached volume's
    block device, format it ONCE if it has no filesystem (NEVER reformat a populated cache — guarded
    by ``blkid``), and mount it at the host ``cache_host_mount``."""
    if not payload.get("cache_block_device") or not payload.get("cache_host_mount"):
        return ""
    mount = payload["cache_host_mount"]
    # Match the cache disk by size (±20%) AND require it unmounted, so we never mkfs the wrong device.
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


def _cache_nfs_mount_check(payload: dict) -> str:
    """Cloud-init preamble (NFS providers, e.g. Lambda): the platform auto-mounts the weight-cache
    filesystem on the host at ``cache_host_mount`` — but ONLY if Lambda actually attached + readied it."""
    if not payload.get("cache_host_mount") or payload.get("cache_block_device"):
        return ""
    mount = payload["cache_host_mount"]
    marker = CACHE_MOUNT_MARKER
    return f"""
# --- weight-cache NFS mount: verify the platform actually mounted it, then drop the sentinel ---
if mountpoint -q '{mount}'; then
  echo "FLASH: weight-cache NFS mounted at {mount}"
  touch '{mount}/{marker}' 2>/dev/null || true
else
  echo "FLASH: weight-cache NFS NOT mounted at {mount} (no sentinel; preload will refuse, train runs cold)"
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
    code_prefix: str | None = None,
) -> dict:
    """The bootstrap's input — field-compatible with the RunPod ``_train_body`` payload, plus the
    bits the instance can't infer (HF prefix for markers, wall cap, attempt, and the substrate
    ``arm`` that the bootstrap stamps as FLASH_ARM + the marker name)."""
    from flash.envs.registry import worker_pip_for_env
    from flash.providers._worker import (
        build_worker_env,
        chalk_extra_pip,
        strip_runpod_volume_env,
    )
    from flash.runner import flash_code_prefix

    # Strip the RunPod-only volume redirect; point base-model prefetch at this provider's cache unless the user overrode it.
    env = strip_runpod_volume_env(build_worker_env(spec, seed, runtime_secrets=runtime_secrets))
    if cache_host_mount and not env.get("FLASH_WEIGHT_CACHE_DIR") and not env.get("HF_HOME"):
        env["FLASH_WEIGHT_CACHE_DIR"] = f"{CACHE_HF_HOME}/hub"
    payload = {
        "hf_repo": spec.train.hf_repo,
        "job_spec_json": spec.to_json(),
        "phase": spec.phase,
        "seed": int(seed),
        "flash_arm": arm,
        "env": env,
        # Per-run env wheel + opt-in chalk spec; the bootstrap pip-installs extra_pip for every job.
        "extra_pip": (list(spec.environment.pip) or worker_pip_for_env(spec.environment.id))
        + chalk_extra_pip(spec),
        "hf_prefix": f"{spec.phase}/{spec.run_id}",
        "code_prefix": code_prefix or flash_code_prefix(),
        "max_wall_s": max(60, int(spec.gpu.max_wall_seconds)),
        "attempt": int(attempt),
    }
    if cache_host_mount:
        payload["cache_host_mount"] = cache_host_mount
        # Carry the mount sentinel filename so the bootstrap's mount-check reads it from one constant.
        payload["cache_mount_marker"] = CACHE_MOUNT_MARKER
        if cache_block_device:
            payload["cache_block_device"] = True
            # Carry the provisioned volume size for the block-device size-match; parse tolerantly so a bad value defaults.
            from flash.runner import WEIGHT_CACHE_VOLUME_GB
            from flash.spec import _volume_gb

            payload["cache_size_gb"] = _volume_gb(
                getattr(spec.gpu, "network_volume_gb", None), default=WEIGHT_CACHE_VOLUME_GB
            )
    # Preload (warm) mode: the bootstrap downloads ``models`` into the cache and exits — no worker.
    if mode:
        payload["mode"] = mode
        payload["models"] = list(models or [])
    return payload


# Host helper: best-effort upload of the boot log to HF (no provider console API). Attempt-scoped path.
_HOSTLOG_PY = """\
import json
try:
    p = json.load(open("/opt/flash/payload.json"))
    arm = p.get("flash_arm", "instance"); att = int(p.get("attempt") or 0)
    from huggingface_hub import HfApi
    HfApi(token=(p.get("env") or {}).get("HF_TOKEN")).upload_file(
        path_or_fileobj="/opt/flash/host_boot.log",
        path_in_repo=p["hf_prefix"] + "/" + arm + "_attempt" + str(att) + "_boot.log",
        repo_id=p["hf_repo"],
        repo_type="dataset",
    )
except Exception:
    pass
"""

# Host helper: write a RETRIABLE attempt-failure marker to HF only when the worker wrote none (it owns the path).
_FAILMARK_PY = """\
import json, sys
try:
    p = json.load(open("/opt/flash/payload.json"))
    arm = p.get("flash_arm", "instance"); att = int(p.get("attempt") or 0)
    reason = sys.argv[1] if len(sys.argv) > 1 else "host boot failure"
    marker_path = p["hf_prefix"] + "/" + arm + "_attempt" + str(att) + ".json"
    from huggingface_hub import HfApi
    api = HfApi(token=(p.get("env") or {}).get("HF_TOKEN"))
    def worker_marker_present():
        try:
            return api.file_exists(repo_id=p["hf_repo"], filename=marker_path, repo_type="dataset")
        except Exception:
            return True  # conservative: on a read error, never risk clobbering
    if not worker_marker_present():
        open("/opt/flash/fm.json", "w").write(json.dumps({"ok": False, "attempt": att, "retriable": True, "error": "host: " + reason}))
        # Re-check right before the upload: the worker may have written its own ok=false marker in
        # the gap since the first check; honor it instead of clobbering with the retriable host one.
        if not worker_marker_present():
            api.upload_file(
                path_or_fileobj="/opt/flash/fm.json",
                path_in_repo=marker_path,
                repo_id=p["hf_repo"], repo_type="dataset",
            )
except Exception:
    pass
"""


def _spill_large_spec_to_hf(payload: dict) -> dict:
    """Keep a large ``job_spec_json`` OUT of the inline cloud-init user_data."""
    spec_json = payload.get("job_spec_json") or ""
    if len(spec_json) <= _SPEC_SPILL_THRESHOLD:
        return payload
    from huggingface_hub import HfApi

    # BytesIO (not raw bytes): upload_file treats bytes as a path-like, misreading it as a huge path.
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
    """Cloud-init ``user_data``: run the worker ``image`` via Docker on the host."""
    payload = _spill_large_spec_to_hf(payload)
    payload_b64 = base64.encodebytes(json.dumps(payload).encode()).decode()
    bootstrap_src = (Path(__file__).parent / "_instance_bootstrap.py").read_text()
    # Bind the host cache mount into the container at the fixed /weight-cache so prefetch persists; absent -> cold.
    cache_host_mount = payload.get("cache_host_mount")
    cache_bind = f"-v '{cache_host_mount}':{CACHE_CONTAINER_MOUNT} \\\n  " if cache_host_mount else ""
    cache_setup = _cache_block_device_setup(payload) + _cache_nfs_mount_check(payload)
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
# Host->HF boot-log uploader, STARTED EARLY — BEFORE the docker-readiness wait and the (large, slow)
# image pull — so a box that actually executed cloud-init leaves a liveness artifact on HF within
# ~2 min. The control plane's poller keys its fast "instance active but the worker never started"
# failover on this artifact's PRESENCE: it tells a healthy box still pulling the multi-GB image
# (boot.log present, no heartbeat yet) from a silently dead one (cloud-init never ran -> no boot.log),
# so the latter fails over in ~15 min instead of burning the full ~50 min setup grace. THROTTLED to
# 120s and bounded (~30 min) to respect HF's per-repo hourly commit cap; it self-stops once the
# worker container has STARTED and then exited (inspect succeeds + not running). The `docker inspect`
# guard is what lets it keep emitting THROUGH the pull/start window: before the container exists the
# inspect fails, so it does not mistake "not started yet" for "started then exited".
( for i in $(seq 1 15); do
    python3 /opt/flash/hostlog.py >/dev/null 2>&1 || true
    if docker inspect flashrun >/dev/null 2>&1 \\
       && ! docker ps --filter name=flashrun --filter status=running -q | grep -q .; then
      python3 /opt/flash/hostlog.py >/dev/null 2>&1 || true
      break
    fi
    sleep 120
  done ) &
disown || true
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
# visible on HF even if it dies before uploading its own console artifact (the early boot-log
# uploader above keeps shipping host_boot.log to HF and self-stops once this container exits).
( docker logs -f flashrun >>/opt/flash/host_boot.log 2>&1 || true ) &
disown || true
"""
