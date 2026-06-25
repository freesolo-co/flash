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
import json
from pathlib import Path

# Lambda/Hyperstack cap an instance/VM ``name`` at 64 chars. We keep the label at or under this so
# the name is NEVER silently truncated at launch — truncation would desync the stored name from the
# ``run_label_prefix`` the orphan-sweep matches on, which could fail to protect (or wrongly reap) a
# live run. The seed/attempt suffix ``-s{seed}-a{attempt}`` is bounded (<=12 chars), so the prefix
# is bounded to leave room for it.
_MAX_NAME = 60
_PREFIX_BUDGET = _MAX_NAME - 12


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


def build_payload(spec, seed: int, attempt: int, *, arm: str, runtime_secrets: dict | None = None) -> dict:
    """The bootstrap's input — field-compatible with the RunPod ``_train_body`` payload, plus the
    bits the instance can't infer (HF prefix for markers, wall cap, attempt, and the substrate
    ``arm`` that the bootstrap stamps as FLASH_ARM + the marker name)."""
    from flash.envs.registry import worker_pip_for_env
    from flash.providers.runpod.train import build_worker_env, chalk_extra_pip

    return {
        "hf_repo": spec.train.hf_repo,
        "job_spec_json": spec.to_json(),
        "phase": spec.phase,
        "seed": int(seed),
        "flash_arm": arm,
        "env": build_worker_env(spec, seed, runtime_secrets=runtime_secrets),
        # The bootstrap pip-installs extra_pip for every job, so the per-run env wheel + the opt-in
        # chalk spec ride along here to reach default runs (mirrors runpod/jobs.submit_run).
        "extra_pip": (list(spec.environment.pip) or worker_pip_for_env(spec.environment.id))
        + chalk_extra_pip(spec),
        "hf_prefix": f"{spec.phase}/{spec.run_id}/seed{seed}",
        "max_wall_s": max(60, int(spec.gpu.max_wall_seconds)),
        "attempt": int(attempt),
    }


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

# Host helper: on a CLEAN container exit (code 0), confirm the worker actually left a success
# artifact on HF (DONE or metrics.json for this run). A clean exit normally means the worker finished
# and uploaded its artifacts — genuine success, OR the already-complete-retry path that restores
# metrics + writes its ok-marker (which the host must NOT clobber with a failure marker). But the
# worker's HF uploads are best-effort, so an exit-0 with NO artifact (every upload failed) would
# otherwise be read as success and idle the billed box until the setup-stall grace (~50 min). So:
# artifact present -> exit 0 (real success; host writes no marker, never clobbers the worker's);
# absent -> exit 1 so the caller writes a RETRIABLE failmark and the run fails fast on a fresh host.
# If HF is unreachable from the box (can't tell), exit 0 — never destroy a possibly-good run; defer
# to the poller's own stall detection. Reads creds from the on-box payload.json.
_DONECHECK_PY = """\
import json, sys
try:
    p = json.load(open("/opt/flash/payload.json"))
    from huggingface_hub import HfApi
    files = set(HfApi(token=(p.get("env") or {}).get("HF_TOKEN")).list_repo_files(
        repo_id=p["hf_repo"], repo_type="dataset"))
    pfx = p["hf_prefix"]
    sys.exit(0 if (pfx + "/DONE" in files or pfx + "/metrics.json" in files) else 1)
except Exception:
    sys.exit(0)
"""

# Host helper: write the attempt-failure marker (<arm>_attempt<N>.json, ok=false, RETRIABLE) to HF
# when the box can't even start the worker container (docker/GPU never ready, image pull failure).
# Without it a pre-container failure leaves NO marker, so the poller would burn the whole setup
# grace (~50 min) before reporting a generic stall; this surfaces a fast, RETRYABLE failure so the
# runner re-provisions on a fresh host immediately. Reads creds from the on-box payload.json.
_FAILMARK_PY = """\
import json, sys
try:
    p = json.load(open("/opt/flash/payload.json"))
    arm = p.get("flash_arm", "instance"); att = int(p.get("attempt") or 0)
    reason = sys.argv[1] if len(sys.argv) > 1 else "host boot failure"
    open("/opt/flash/fm.json", "w").write(json.dumps({"ok": False, "attempt": att, "retriable": True, "error": "host: " + reason}))
    from huggingface_hub import HfApi
    HfApi(token=(p.get("env") or {}).get("HF_TOKEN")).upload_file(
        path_or_fileobj="/opt/flash/fm.json",
        path_in_repo=p["hf_prefix"] + "/" + arm + "_attempt" + str(att) + ".json",
        repo_id=p["hf_repo"], repo_type="dataset",
    )
except Exception:
    pass
"""


def build_user_data(payload: dict, *, image: str) -> str:
    """Cloud-init ``user_data``: run the worker ``image`` via Docker on the host.

    cloud-init runs this once at first boot as root. Everything dynamic travels base64-encoded
    inside the script (never interpolated into shell syntax), so the job-spec JSON survives
    byte-exact. The full training stack is baked into the image, so the box only needs Docker + an
    NVIDIA GPU — both shipped by the providers' default Docker-capable images — and the container
    does the rest (fetch code from HF, run the worker, stream artifacts back to HF).

    Secrets-wise the script carries the same content as the worker env on RunPod (HF token, env
    secrets). The operator's provider API key is NEVER shipped (teardown is control-plane-side via
    the runner ``finally`` / poll deadline / ``sweep_orphans``).
    """
    payload_b64 = base64.encodebytes(json.dumps(payload).encode()).decode()
    bootstrap_src = (Path(__file__).parent / "_instance_bootstrap.py").read_text()
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
cat > /opt/flash/donecheck.py <<'FLASH_DONECHECK_EOF'
{_DONECHECK_PY}FLASH_DONECHECK_EOF
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
# Pull with retries (the image is large; a transient registry blip must not fail the run). On total
# failure, write a RETRYABLE marker and exit NOW instead of leaving a billed box idling the whole
# setup grace with no DONE/marker.
PULLED=0
for i in 1 2 3 4 5; do docker pull "$IMAGE" && {{ PULLED=1; break; }}; echo "FLASH: pull retry $i"; sleep 20; done
[ "$PULLED" -eq 1 ] || fail "worker image pull failed after retries"
# Run the worker container detached so cloud-init completes promptly; completion is signaled via the
# worker's HF artifacts (DONE/metrics.json/marker), never a return channel from the box.
docker run -d --name flashrun --gpus all --shm-size=16g --network host \\
  -v /opt/flash:/root/flash -w /root/flash \\
  "$IMAGE" python /root/flash/bootstrap.py || fail "docker run failed"
sleep 5
# The container must be running OR have already exited CLEANLY: an already-complete retry restores
# the prior metrics and writes DONE in well under 5s, so a clean exit (code 0) is success, not a
# failed start. A non-zero exit (or never-started) fails fast. A clean exit is trusted ONLY if the
# worker actually left a success artifact on HF (donecheck) — exit 0 with no artifact (all uploads
# failed) fails fast too, instead of idling the box to the stall grace.
if ! docker ps --filter name=flashrun --filter status=running -q | grep -q .; then
  EXIT="$(docker inspect -f '{{{{.State.ExitCode}}}}' flashrun 2>/dev/null || echo 1)"
  docker logs flashrun >>/opt/flash/host_boot.log 2>&1 || true
  if [ "$EXIT" = "0" ]; then
    python3 /opt/flash/donecheck.py || fail "worker exited 0 but left no success artifact on HF"
  else
    fail "worker container did not start (exit ${{EXIT}})"
  fi
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
