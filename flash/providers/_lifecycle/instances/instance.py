"""Shared building blocks for the instance-based providers (Lambda, Vast)."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from dataclasses import dataclass
from typing import Any, ClassVar

from flash.providers._lifecycle.instances.poll import _attempt_int
from flash.providers._lifecycle.net.deadline import remaining_seconds, require_deadline_at
from flash.runtime_capsule import build_capsule, sha256_bytes

# the capsule the rented-box providers ship. built once and cached: the archive is deterministic,
# so rebuilding it per launch would burn cpu to produce identical bytes.
INSTANCE_BOOTSTRAP_PROFILE = "instance-bootstrap"
_INSTANCE_CAPSULE: tuple[str, str] | None = None

# Bounded so the name is never truncated at launch — truncation desyncs the sweep-matched prefix.
_MAX_NAME = 60
_SUFFIX_BUDGET = 12
_PREFIX_BUDGET = _MAX_NAME - _SUFFIX_BUDGET

# The provider-aligned user_data ceiling (cloud-init limits run ~16KB on AWS to 64KB elsewhere) less
# a stated margin for provider-side framing. This budget is the BINDING spill check: build_user_data
# measures the COMPLETE encoded user_data against it, so nothing riding along with the spec (runtime
# secrets, env, cache fields) can push a launch over the cap.
_USER_DATA_CAP = 64_000
_USER_DATA_MARGIN = 2_000
_USER_DATA_BUDGET = _USER_DATA_CAP - _USER_DATA_MARGIN

# Fast path only: above this, the spec is spilled to HF without first rendering a payload that
# cannot fit. What is left of the ~64,000-byte cap after the fixed framing -- this module's template
# plus the base64 runtime capsule -- is the inline headroom, and base64 + json escaping inflate the
# spec ~1.35x on the way in. The capsule is compressed, so it costs far less than the raw sources it
# replaced: the framing that forced this down to 2,000 when bootstrap.py and its console, secret,
# and pip siblings each rode as their own heredoc now fits in one archive, with room to spare.
# test_build_user_data_spills_large_spec_out_of_cloud_init pins the worst inline case against the
# cap so the two cannot drift apart silently. Sized for a REAL payload, which carries ~760 bytes of
# env, deadline, and cache fields that the test's minimal one does not: at 4_000 the worst case
# cleared the test but a production launch would re-render and force-spill anyway.
_SPEC_SPILL_THRESHOLD = 3_000


def run_label_prefix(run_id: str) -> str:
    """The prefix EVERY instance label for ``run_id`` starts with, bounded to the name budget."""
    base = run_id if run_id.startswith("flash-") else f"flash-{run_id}"
    if len(base) <= _PREFIX_BUDGET:
        return base
    h = hashlib.sha1(base.encode()).hexdigest()[:8]
    return f"{base[: _PREFIX_BUDGET - 9]}-{h}"


def label_matches_run(label: str, prefix: str) -> bool:
    """True iff ``label`` belongs to the run whose prefix is ``prefix`` -- an EXACT match, or the
    prefix followed by the ``-a`` attempt boundary. Boundary-anchored so ``flash-100`` never matches
    ``flash-1000-...`` (or vice versa). The single label-ownership test every instance provider's
    sweep and run-scoped teardown shares."""
    return label == prefix or label.startswith(prefix + "-a")


def instance_label(run_id: str, attempt: int) -> str:
    """Instance name: run plus the attempt executing on this host.

    Run-derived so ``sweep_orphans`` can tell ours from anything else on the account, and bounded
    (via ``run_label_prefix``) so the provider never truncates it. The seed is not part of the name:
    a run has exactly one, so it distinguished nothing here while competing with the attempt for the
    digit budget, and a truncated attempt is one two attempts of a run can collide on.
    """
    attempt_i = _attempt_int(attempt)
    if attempt_i is None:
        raise ValueError("instance attempt identity is invalid")
    attempt_s = str(attempt_i)
    if len(attempt_s) > _SUFFIX_BUDGET - len("-a"):
        raise ValueError("instance attempt identity exceeds the provider name budget")
    return f"{run_label_prefix(run_id)}-a{attempt_s}"


@dataclass
class InstanceJobHandle:
    """Fields + (de)serialization common to every rent-a-box provider handle (Lambda, Vast).

    Persisted in `RunStatus.remote` for reattach and cancellation. Subclasses add locator fields;
    `instance_id` has no safe default because it is the poll/destroy target.
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
        if d.get("provider") != cls.provider:
            raise ValueError(f"persisted {cls.provider} provider identity is invalid")
        try:
            instance_id = cls._coerce_instance_id(d["instance_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"persisted {cls.provider} instance identity is invalid") from exc
        attempt = _attempt_int(d.get("attempt"))
        if attempt is None:
            raise ValueError(f"persisted {cls.provider} attempt identity is invalid")
        started_raw = d.get("started_ts")
        if isinstance(started_raw, bool) or not isinstance(started_raw, (int, float)):
            raise ValueError(f"persisted {cls.provider} launch timestamp is invalid")
        started_ts = float(started_raw)
        if not math.isfinite(started_ts) or started_ts <= 0:
            raise ValueError(f"persisted {cls.provider} launch timestamp is invalid")
        gpu = d.get("gpu")
        if not isinstance(gpu, str) or not gpu:
            raise ValueError(f"persisted {cls.provider} gpu identity is invalid")
        hourly_raw = d.get("hourly_usd")
        if isinstance(hourly_raw, bool) or not isinstance(hourly_raw, (int, float)):
            raise ValueError(f"persisted {cls.provider} hourly rate is invalid")
        hourly_usd = float(hourly_raw)
        if not math.isfinite(hourly_usd) or hourly_usd < 0:
            raise ValueError(f"persisted {cls.provider} hourly rate is invalid")
        return cls(
            instance_id=instance_id,
            gpu=gpu,
            hourly_usd=hourly_usd,
            attempt=attempt,
            started_ts=started_ts,
            **cls._extra_from_dict(d),
        )


# fixed container path for Lambda's per-region NFS cache bind mount.
CACHE_CONTAINER_MOUNT = "/weight-cache"
CACHE_HF_HOME = f"{CACHE_CONTAINER_MOUNT}/hf-cache"
# Sentinel on a successfully-mounted cache so the preload check can tell a real mount from an empty bind.
CACHE_MOUNT_MARKER = ".flash-cache-mounted"


def _cache_nfs_mount_check(payload: dict) -> str:
    """cloud-init preamble for Lambda: the platform auto-mounts the weight-cache filesystem on the
    host at ``cache_host_mount`` only when the cache is attached and ready."""
    if not payload.get("cache_host_mount"):
        return ""
    mount = payload["cache_host_mount"]
    return f"""
# --- weight-cache NFS mount: verify the platform actually mounted it, then drop the sentinel ---
if mountpoint -q '{mount}'; then
  echo "FLASH: weight-cache NFS mounted at {mount}"
  touch '{mount}/{CACHE_MOUNT_MARKER}' 2>/dev/null || true
else
  echo "FLASH: weight-cache NFS NOT mounted at {mount} (no sentinel; preload will refuse, train runs cold)"
fi
"""


def build_payload(
    spec,
    attempt: int,
    *,
    arm: str,
    runtime_secrets: dict | None = None,
    cache_host_mount: str | None = None,
    mode: str | None = None,
    models: list | None = None,
    source_snapshot: dict | None = None,
    deadline_at: float | None = None,
) -> dict:
    """The bootstrap's input — field-compatible with the RunPod ``_train_body`` payload, plus the
    bits the instance can't infer (HF prefix for markers, wall cap, attempt, and the substrate
    ``arm`` that the bootstrap stamps as FLASH_ARM + the marker name)."""
    from flash.envs.loading.base import worker_pip_with_extras
    from flash.providers._lifecycle.net.worker import (
        build_worker_env,
        strip_runpod_volume_env,
    )
    from flash.snapshot.archive import parse_descriptor

    # strip the runpod-only volume redirect; point base-model prefetch at this provider's cache unless the user overrode it.
    env = strip_runpod_volume_env(
        build_worker_env(
            spec,
            runtime_secrets=runtime_secrets,
        )
    )
    if cache_host_mount and not env.get("FLASH_WEIGHT_CACHE_DIR") and not env.get("HF_HOME"):
        env["FLASH_WEIGHT_CACHE_DIR"] = f"{CACHE_HF_HOME}/hub"
    absolute_deadline = require_deadline_at(deadline_at)
    attempt_id = _attempt_int(attempt)
    if attempt_id is None:
        raise ValueError("instance attempt identity is invalid")
    max_wall_seconds = float(spec.gpu.max_wall_seconds)
    payload = {
        "hf_repo": spec.train.hf_repo,
        "job_spec_json": spec.to_json(),
        "phase": spec.phase,
        "run_id": spec.run_id,
        "seed": spec.seed,
        "flash_arm": arm,
        "env": env,
        # per-run env wheel; the bootstrap pip-installs extra_pip for every job. the author's
        # [environment] pip is appended to the worker requirement, never substituted for it.
        "extra_pip": worker_pip_with_extras(spec.environment.id, spec.environment.pip),
        "hf_prefix": f"{spec.phase}/{spec.run_id}",
        "deadline_at": absolute_deadline,
        "run_created_at": absolute_deadline - max_wall_seconds,
        "run_max_wall_seconds": max_wall_seconds,
        "attempt": attempt_id,
    }
    if mode != "preload":
        payload["source_snapshot"] = parse_descriptor(source_snapshot).to_dict()
    if cache_host_mount:
        payload["cache_host_mount"] = cache_host_mount
        # Carry the mount sentinel filename so the bootstrap's mount-check reads it from one constant.
        payload["cache_mount_marker"] = CACHE_MOUNT_MARKER
    # Preload (warm) mode: the bootstrap downloads ``models`` into the cache and exits — no worker.
    if mode:
        payload["mode"] = mode
        payload["models"] = list(models or [])
    return payload


def _spill_large_spec_to_hf(payload: dict, *, force: bool = False) -> dict:
    """Keep a large ``job_spec_json`` OUT of the inline cloud-init user_data. ``force`` spills a spec
    that is under the fast-path threshold, for when the COMPLETE payload is what overflows the cap."""
    spec_json = payload.get("job_spec_json") or ""
    if not spec_json or (not force and len(spec_json) <= _SPEC_SPILL_THRESHOLD):
        return payload
    if "deadline_at" in payload and remaining_seconds(payload["deadline_at"]) <= 0:
        raise TimeoutError("run wall deadline exceeded before job spec upload")
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
    """Cloud-init ``user_data``: run the worker ``image`` via Docker on the host.

    The spill decision is BINDING on the final encoded bytes, not on the spec alone: the spec shares
    user_data with runtime secrets (a multiline PEM is a valid one), so a spec under the fast-path
    threshold plus big secrets could otherwise still overflow the provider cap. Render, measure, and
    spill the spec out whenever the total exceeds the budget.

    Spilling only moves the spec, so a non-spec payload (large runtime secrets) that is oversized on
    its own stays oversized. Re-measure after spilling and fail HERE, naming the component, instead
    of handing the provider a payload it rejects opaquely after the launch call."""
    payload = _spill_large_spec_to_hf(payload)
    user_data = _render_user_data(payload, image=image)
    if len(user_data.encode()) > _USER_DATA_BUDGET and (payload.get("job_spec_json") or ""):
        payload = _spill_large_spec_to_hf(payload, force=True)
        user_data = _render_user_data(payload, image=image)
    size = len(user_data.encode())
    if size > _USER_DATA_BUDGET:
        env_bytes = len(json.dumps(payload.get("env") or {}).encode())
        raise ValueError(
            f"instance user_data is {size} bytes after spilling the job spec, over the "
            f"{_USER_DATA_BUDGET}-byte budget ({_USER_DATA_CAP}-byte provider cap less "
            f"{_USER_DATA_MARGIN} bytes of framing); the runtime secrets and env alone are "
            f"{env_bytes} bytes. Shrink the run's [environment].secrets values."
        )
    return user_data


def _instance_capsule() -> tuple[str, str]:
    """The base64 instance-bootstrap capsule and its sha256, built once per process.

    The digest returned here is what the launch script checks the decoded archive against, so both
    values must come from the same build; returning them together is what makes that structural
    rather than a convention two call sites have to remember.
    """
    global _INSTANCE_CAPSULE
    if _INSTANCE_CAPSULE is None:
        archive, _manifest = build_capsule(INSTANCE_BOOTSTRAP_PROFILE)
        _INSTANCE_CAPSULE = (base64.encodebytes(archive).decode(), sha256_bytes(archive))
    return _INSTANCE_CAPSULE


def _render_user_data(payload: dict, *, image: str) -> str:
    """The user_data text for an already-spill-decided ``payload``."""
    payload_b64 = base64.encodebytes(json.dumps(payload).encode()).decode()
    # The runtime code ships as ONE verified capsule rather than a heredoc per module. user_data
    # carries only the base64 archive plus its expected digest; the box refuses to execute anything
    # whose bytes do not match, so a corrupted or substituted payload fails before it can run.
    capsule_b64, capsule_sha256 = _instance_capsule()
    # Bind the host cache mount into the container at the fixed /weight-cache so prefetch persists; absent -> cold.
    cache_host_mount = payload.get("cache_host_mount")
    cache_bind = (
        f"-v '{cache_host_mount}':{CACHE_CONTAINER_MOUNT} \\\n  " if cache_host_mount else ""
    )
    cache_setup = _cache_nfs_mount_check(payload)
    return f"""#!/bin/bash
# flash instance worker (generated by flash.providers._lifecycle.instances.instance.build_user_data; arm={payload.get("flash_arm")})
set -x
mkdir -p /opt/flash
# collect host and container boot output for lambda diagnostics.
exec >>/opt/flash/host_boot.log 2>&1
cat > /opt/flash/payload.b64 <<'FLASH_PAYLOAD_EOF'
{payload_b64}FLASH_PAYLOAD_EOF
base64 -d /opt/flash/payload.b64 > /opt/flash/payload.json
# The runtime capsule: a versioned zipapp whose every member is sha256'd in its manifest. The
# expected digest below comes from the control plane, NOT from inside the archive -- an archive
# cannot authenticate itself, and a manifest swapped together with its members stays self-
# consistent. Verifying before the first execution is the whole point: a capsule that does not
# match these bytes never runs.
cat > /opt/flash/capsule.b64 <<'FLASH_CAPSULE_EOF'
{capsule_b64}FLASH_CAPSULE_EOF
base64 -d /opt/flash/capsule.b64 > /opt/flash/capsule.pyz
CAPSULE_SHA256={capsule_sha256!r}
echo "$CAPSULE_SHA256  /opt/flash/capsule.pyz" | sha256sum -c - >/dev/null 2>&1 \\
  || {{ echo "FLASH: runtime capsule failed verification" >&2; exit 1; }}
IMAGE={image!r}
fail() {{ echo "FLASH: $1" >&2; python3 /opt/flash/capsule.pyz failmark "$1" >/dev/null 2>&1 || true; exit 1; }}
deadline_sleep() {{ python3 /opt/flash/capsule.pyz deadline_sleep "$1"; }}
# huggingface_hub on the host for the boot-log + failure-marker uploaders (best-effort).
deadline_sleep 0 || exit 124
# exact-pinned so host bootstrap retries cannot resolve different uploader behavior.
pip3 install -q huggingface_hub==1.28.0 >/dev/null 2>&1 \\
  || python3 -m pip install -q --break-system-packages huggingface_hub==1.28.0 >/dev/null 2>&1 || true
# upload the host log while docker and the worker image start.
( for i in $(seq 1 15); do
    python3 /opt/flash/capsule.pyz hostlog >/dev/null 2>&1 || true
    if docker inspect flashrun >/dev/null 2>&1 \\
       && ! docker ps --filter name=flashrun --filter status=running -q | grep -q .; then
      python3 /opt/flash/capsule.pyz hostlog >/dev/null 2>&1 || true
      break
    fi
    deadline_sleep 120 || break
  done ) &
disown || true
# wait for docker and the gpu before launching the worker.
for i in $(seq 1 100); do
  deadline_sleep 0 || fail "run wall deadline exceeded"
  if docker info >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then break; fi
  echo "FLASH: waiting for docker+gpu ($i)"
  deadline_sleep 6 || fail "run wall deadline exceeded"
done
docker info >/dev/null 2>&1 || fail "docker never became ready"
nvidia-smi >/dev/null 2>&1 || fail "gpu never became ready"
{cache_setup}
# retry transient image-pull failures before writing the failure marker.
PULLED=0
for i in 1 2 3 4 5; do
  deadline_sleep 0 || fail "run wall deadline exceeded"
  docker pull "$IMAGE" && {{ PULLED=1; break; }}
  echo "FLASH: pull retry $i"
  deadline_sleep 20 || fail "run wall deadline exceeded"
done
[ "$PULLED" -eq 1 ] || fail "worker image pull failed after retries"
# run detached; worker artifacts signal completion.
deadline_sleep 0 || fail "run wall deadline exceeded"
docker run -d --name flashrun --gpus all --shm-size=16g --network host \\
  -v /opt/flash:/root/flash {cache_bind}-w /root/flash \\
  "$IMAGE" python /root/flash/capsule.pyz bootstrap || fail "docker run failed"
deadline_sleep 5 || fail "run wall deadline exceeded"
# a stopped container succeeds only at exit 0; failmark preserves any worker marker.
if ! docker ps --filter name=flashrun --filter status=running -q | grep -q .; then
  EXIT="$(docker inspect -f '{{{{.State.ExitCode}}}}' flashrun 2>/dev/null || echo 1)"
  [ "$EXIT" = "0" ] || fail "worker container did not start (exit ${{EXIT}})"
fi
"""
