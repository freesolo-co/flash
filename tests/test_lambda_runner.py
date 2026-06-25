"""Lambda Cloud run lifecycle: cloud-init/bootstrap, region walk, poll state machine, guaranteed
terminate, orphan sweep, capacity-aware allocation (CPU-only; lambda API + HF readers mocked).

Lambda is opt-in via LAMBDA_API_KEY (the autouse offline fixture deletes it); these tests mock the
lambda API entirely, so no key is needed — except the allocator tests, which set it to make the
provider "available" and then mock the capacity lookup.
"""

from __future__ import annotations

import base64
import io
import itertools
import json
import time

import pytest

from flash.spec import JobSpec


def _spec(gpu_type="A10", **gpu_kw) -> JobSpec:
    gpu = {"type": gpu_type, "max_wall_seconds": 3600, **gpu_kw}
    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "sft",
            "run_id": "flash-1700000000-abcd1234",
            "train": {"epochs": 1, "seeds": [0], "hf_repo": "org/repo"},
            "gpu": gpu,
        }
    )


def _inst(gpu="A10", region="us-east-1", itype="gpu_1x_a10", price=1.29):
    from flash.providers.lambdalabs.jobs.builders import LambdaInstance

    return LambdaInstance(gpu=gpu, instance_type=itype, region=region, vram_gb=24, price_usd_hr=price)


def _handle(started_ts=10_000.0, rate=1.29):
    from flash.providers.lambdalabs.jobs.builders import LambdaJobHandle

    return LambdaJobHandle(
        instance_id="i-9999",
        instance_type="gpu_1x_a10",
        region="us-east-1",
        name="flash-x-s0-a0",
        gpu="A10",
        hourly_usd=rate,
        attempt=0,
        started_ts=started_ts,
    )


# ---------------------------------------------------------------------------
# cloud-init user_data + bootstrap
# ---------------------------------------------------------------------------
def test_user_data_ships_payload_and_runs_worker_image(monkeypatch):
    from flash.providers.lambdalabs.jobs import builders

    monkeypatch.setenv("LAMBDA_API_KEY", "lk-supersecret")
    monkeypatch.setenv("HF_TOKEN", "hf-worker-token")
    payload = builders.build_payload(_spec(), seed=0, attempt=1)
    assert payload["phase"] == "sft"
    assert payload["attempt"] == 1
    assert payload["hf_prefix"] == "sft/flash-1700000000-abcd1234/seed0"
    assert payload["max_wall_s"] == 3600
    assert payload["hf_repo"] == "org/repo"
    # The worker env's HF_REPO is sourced from the run's [train] hf_repo (not an operator default).
    assert payload["env"]["HF_REPO"] == "org/repo"

    script = builders.build_user_data(payload)
    # payload travels base64-encoded inside a quoted heredoc, byte-exact
    b64 = script.split("FLASH_PAYLOAD_EOF")[1].strip()
    assert json.loads(base64.b64decode(b64)) == payload
    # the self-contained bootstrap is embedded
    assert "FLASH_BOOTSTRAP_EOF" in script
    assert "metrics.json" in script
    # runs the prebuilt WORKER_IMAGE via Docker with the GPU + the bootstrap as the command
    from flash.providers.runpod.train import WORKER_IMAGE

    assert WORKER_IMAGE in script
    assert "docker run -d" in script
    assert "--gpus all" in script
    assert "/root/flash/bootstrap.py" in script
    # waits for docker + gpu before launching (cloud-init can beat them to ready)
    assert "waiting for docker+gpu" in script
    # the operator's Lambda key NEVER ships to the box (no instance-scoped key, teardown is
    # control-plane-side). The worker HF token IS carried — inside the base64 payload's env (like
    # RunPod's worker env), never interpolated raw into the shell.
    assert "lk-supersecret" not in script
    assert payload["env"]["HF_TOKEN"] == "hf-worker-token"


def test_user_data_skips_capacity_for_baked_image_default(monkeypatch):
    """build_user_data always uses the baked WORKER_IMAGE (no per-host stack install)."""
    from flash.providers.lambdalabs.jobs import builders

    payload = builders.build_payload(_spec(), seed=0, attempt=0)
    script = builders.build_user_data(payload)
    # No base training-stack pip install in the cloud-init (the image is baked); only the worker
    # container's own per-run extra_pip runs (inside _bootstrap, not the host script).
    assert "torch==2.10.0" not in script


def test_image_per_sm_opt_in_selects_arch_tag(monkeypatch):
    """Opt-in per-SM warmed images (PR #213) reach Lambda too: with FLASH_WORKER_IMAGE_PER_SM set,
    the GPU class picks the matching -smXX tag (so the worker's baked kernel cache matches the rented
    GPU's arch). Default + FLASH_WORKER_IMAGE override semantics are unchanged."""
    from flash.providers.lambdalabs.jobs import builders
    from flash.providers.runpod.train import WORKER_IMAGE

    for key in ("FLASH_WORKER_IMAGE", "FLASH_WORKER_IMAGE_PER_SM", "FLASH_WORKER_IMAGE_TEMPLATE"):
        monkeypatch.delenv(key, raising=False)

    # default: flat base image, byte-identical to pre-PR behavior
    assert builders.lambda_image() == WORKER_IMAGE
    assert builders.lambda_image("H100") == WORKER_IMAGE

    # per-SM opt-in: the GPU class appends the arch tag, and it lands in the cloud-init
    monkeypatch.setenv("FLASH_WORKER_IMAGE_PER_SM", "1")
    assert builders.lambda_image("H100") == f"{WORKER_IMAGE}-sm90"  # H100 = sm90
    assert builders.lambda_image("A10") == f"{WORKER_IMAGE}-sm86"  # A10 = sm86
    payload = builders.build_payload(_spec(gpu_type="H100"), seed=0, attempt=0)
    script = builders.build_user_data(payload, gpu="H100")
    assert f"{WORKER_IMAGE}-sm90" in script

    # absolute override still wins, even with per-SM enabled and a GPU class given
    monkeypatch.setenv("FLASH_WORKER_IMAGE", "ghcr.io/freesolo-co/flash-worker:hotfix")
    assert builders.lambda_image("H100") == "ghcr.io/freesolo-co/flash-worker:hotfix"


def _bootstrap_env(monkeypatch, phase="sft", rc=0, metrics=True):
    from flash.providers import _instance_bootstrap as lb

    calls: list[str] = []
    markers: list[tuple[bool, str, bool]] = []
    monkeypatch.setattr(
        lb,
        "load_payload",
        lambda path=lb.PAYLOAD_PATH: {
            "hf_repo": "org/repo",
            "job_spec_json": "{}",
            "phase": phase,
            "seed": 0,
            "flash_arm": "lambda",
            "env": {},
            "extra_pip": [],
            "hf_prefix": "sft/x/seed0",
            "max_wall_s": 60,
            "attempt": 0,
        },
    )
    monkeypatch.setattr(lb, "fetch_code", lambda p: None)
    monkeypatch.setattr(lb, "run_mode", lambda p, e, m, d: (calls.append(m), rc)[1])
    monkeypatch.setattr(
        lb,
        "write_attempt_marker",
        lambda p, ok, error="", retriable=False: markers.append((ok, error, retriable)),
    )
    monkeypatch.setattr(lb.os.path, "exists", lambda p: metrics if "metrics.json" in p else False)
    return lb, calls, markers


def test_bootstrap_train_success(monkeypatch):
    lb, calls, markers = _bootstrap_env(monkeypatch)
    assert lb.main() == 0
    assert calls == ["sft"]  # one fresh worker process
    assert markers == [(True, "", False)]  # success marker, not retriable


def test_bootstrap_fails_without_metrics(monkeypatch):
    lb, _calls, markers = _bootstrap_env(monkeypatch, metrics=False)
    assert lb.main() == 1
    ok, error, retriable = markers[0]
    assert not ok
    assert "metrics.json" in error
    # A genuine no-metrics crash (the worker never produced metrics) is a REAL failure, not infra:
    # it must NOT be flagged retriable (that would loop a deterministically-broken run).
    assert retriable is False


def test_bootstrap_sets_lambda_arm():
    """The shared bootstrap stamps FLASH_ARM from payload['flash_arm'] so the metrics record
    attributes the substrate (Lambda's build_payload sets it to 'lambda')."""
    from flash.providers import _instance_bootstrap as lb

    env = lb.build_worker_env({"job_spec_json": "{}", "phase": "sft", "seed": 0, "env": {}, "flash_arm": "lambda"})
    assert env["FLASH_ARM"] == "lambda"
    # And Lambda's build_payload is what sets flash_arm='lambda'.
    from flash.providers.lambdalabs.jobs.builders import build_payload
    assert build_payload(_spec(), 0, 0)["flash_arm"] == "lambda"


# ---------------------------------------------------------------------------
# launch_and_submit: capacity (region) walk
# ---------------------------------------------------------------------------
def test_launch_walks_regions_on_capacity_rejection(monkeypatch):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    attempts = []

    def fake_launch(*, region_name, instance_type_name, ssh_key_names, name, user_data, file_system_names=None):
        attempts.append(region_name)
        if len(attempts) < 3:
            raise lambda_api.LambdaApiError("PUT /asks/1/ -> HTTP 400: insufficient-capacity")
        return "i-4242"

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)
    insts = [_inst(region=r) for r in ("us-east-1", "us-west-1", "us-west-2")]
    h = jobs.launch_and_submit(_spec(), seed=0, instances=insts, attempt=2)
    assert attempts == ["us-east-1", "us-west-1", "us-west-2"]
    assert h.instance_id == "i-4242"
    assert h.region == "us-west-2"
    assert h.gpu == "A10"
    assert h.name == "flash-1700000000-abcd1234-s0-a2"


def test_launch_refreshes_capacity_once_when_all_taken(monkeypatch):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    created = []

    def fake_launch(*, region_name, **kw):
        if region_name != "us-fresh-1":
            raise lambda_api.LambdaApiError("PUT /asks/1/ -> HTTP 400: no capacity")
        created.append(region_name)
        return "i-7"

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)
    monkeypatch.setattr(jobs, "usable_instances", lambda gpu, force=False: [_inst(region="us-fresh-1")])
    h = jobs.launch_and_submit(_spec(), seed=0, instances=[_inst(region="us-east-1")], attempt=0)
    assert created == ["us-fresh-1"]
    assert h.instance_id == "i-7"


def test_launch_raises_when_no_capacity(monkeypatch):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(
        lambda_api,
        "launch_instance",
        lambda **k: (_ for _ in ()).throw(lambda_api.LambdaApiError("PUT /asks/1/ -> HTTP 400: no capacity")),
    )
    monkeypatch.setattr(jobs, "usable_instances", lambda gpu, force=False: [])
    with pytest.raises(lambda_api.LambdaApiError, match="no capacity"):
        jobs.launch_and_submit(_spec(), seed=0, instances=[_inst()], attempt=0)
    with pytest.raises(lambda_api.LambdaApiError, match="no Lambda capacity"):
        jobs.launch_and_submit(_spec(), seed=0, instances=[], attempt=0)


def test_resolve_ssh_key_names(monkeypatch):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs.jobs import resolve_ssh_key_names

    monkeypatch.setenv("LAMBDA_SSH_KEY_NAME", "pinned-key")
    assert resolve_ssh_key_names() == ["pinned-key"]
    monkeypatch.delenv("LAMBDA_SSH_KEY_NAME", raising=False)
    monkeypatch.setattr(lambda_api, "list_ssh_keys", lambda: [{"name": "jk"}, {"name": "other"}])
    assert resolve_ssh_key_names() == ["jk"]  # first registered key
    monkeypatch.setattr(lambda_api, "list_ssh_keys", lambda: [])
    with pytest.raises(lambda_api.LambdaApiError, match="requires an SSH key"):
        resolve_ssh_key_names()


# ---------------------------------------------------------------------------
# launch_and_submit: per-region weight cache (Lambda persistent filesystem)
# ---------------------------------------------------------------------------
def _wire_launch(monkeypatch):
    """Common launch wiring: ssh key + a launch that records (region, user_data, file_system_names)."""
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    calls = []

    def fake_launch(*, region_name, instance_type_name, ssh_key_names, name, user_data, file_system_names=None):
        calls.append({"region": region_name, "user_data": user_data, "fs": file_system_names})
        return "i-cache"

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)
    return jobs, lambda_api, calls


def test_cache_ensures_filesystem_and_attaches_at_launch(monkeypatch):
    jobs, lambda_api, calls = _wire_launch(monkeypatch)
    ensured = []
    monkeypatch.setattr(lambda_api, "ensure_filesystem", lambda n, r: ensured.append((n, r)) or f"/lambda/nfs/{n}")

    spec = _spec(network_volume="flash-weights")
    jobs.launch_and_submit(spec, seed=0, instances=[_inst(region="us-east-1")], attempt=0)

    assert ensured == [("flash-weights", "us-east-1")]  # create-if-absent in THIS region
    assert calls[0]["fs"] == ["flash-weights"]  # attached at launch (Lambda can't attach later)
    # The cloud-init binds the auto-mounted NFS path into the worker at the fixed cache mount.
    assert "-v '/lambda/nfs/flash-weights':/weight-cache" in calls[0]["user_data"]  # quoted host path


def test_cache_bind_uses_returned_mount_point(monkeypatch):
    """The bind-mount targets the FS's ACTUAL mount_point, not the hard-coded /lambda/nfs/<name>.

    Regression: ensure_filesystem's returned mount_point was ignored, so a region where Lambda mounts
    the FS at a non-default host path would bind the wrong path -> silently cold / failed preload mount.
    """
    jobs, lambda_api, calls = _wire_launch(monkeypatch)
    # Lambda reports a NON-default host mount for this region's filesystem.
    monkeypatch.setattr(lambda_api, "ensure_filesystem", lambda n, r: "/mnt/lambda-fs/flash-weights")

    jobs.launch_and_submit(_spec(network_volume="flash-weights"), seed=0, instances=[_inst()], attempt=0)

    assert calls[0]["fs"] == ["flash-weights"]
    # the bind uses the REAL mount_point, and never the stale default
    assert "-v '/mnt/lambda-fs/flash-weights':/weight-cache" in calls[0]["user_data"]
    assert "/lambda/nfs/flash-weights" not in calls[0]["user_data"]


def test_cache_payload_points_hf_home_at_the_bind(monkeypatch):
    """The base64 payload's worker env redirects HF_HOME onto the bind (so the model download persists)."""
    from flash.providers.lambdalabs.jobs import build_payload

    payload = build_payload(_spec(network_volume="flash-weights"), 0, 0, cache_host_mount="/lambda/nfs/flash-weights")
    assert payload["env"]["HF_HOME"] == "/weight-cache/hf-cache"
    assert payload["cache_host_mount"] == "/lambda/nfs/flash-weights"
    assert "cache_block_device" not in payload  # NFS: no format/mount preamble


def test_cache_falls_back_to_cold_when_filesystem_unavailable(monkeypatch):
    jobs, lambda_api, calls = _wire_launch(monkeypatch)
    monkeypatch.setattr(
        lambda_api, "ensure_filesystem",
        lambda n, r: (_ for _ in ()).throw(lambda_api.LambdaApiError("filesystem quota exceeded")),
    )
    jobs.launch_and_submit(_spec(network_volume="flash-weights"), seed=0, instances=[_inst()], attempt=0)
    assert calls[0]["fs"] is None  # no filesystem attached
    assert "/weight-cache" not in calls[0]["user_data"]  # cold user_data, no bind


def test_filesystem_attach_reject_retries_same_region_cold(monkeypatch):
    """A clean reject whose error mentions the FILESYSTEM retries THIS region cache-less before
    walking — so a best-effort attach can't make a region the cold path would have served fail."""
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(lambda_api, "ensure_filesystem", lambda n, r: f"/lambda/nfs/{n}")  # FS ensured
    calls = []

    def fake_launch(*, region_name, file_system_names=None, user_data=None, **kw):
        calls.append({"region": region_name, "fs": file_system_names})
        if file_system_names:  # the CACHED launch is rejected for a filesystem-attach reason
            raise lambda_api.LambdaApiError(
                "POST /instance-operations/launch -> HTTP 400: file_system_names not attachable"
            )
        return "i-cold"  # the cold retry (no fs) succeeds in the SAME region

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)
    h = jobs.launch_and_submit(_spec(network_volume="flash-weights"),
                               seed=0, instances=[_inst(region="us-east-1")], attempt=0)
    assert h.region == "us-east-1"  # served by the SAME region, not lost to the walk
    assert [c["fs"] for c in calls] == [["flash-weights"], None]  # cached attempt, then cold retry
    assert all(c["region"] == "us-east-1" for c in calls)


def test_capacity_reject_does_not_trigger_cold_fs_retry(monkeypatch):
    """A plain CAPACITY reject (no filesystem in the error) walks normally — no extra cold retry."""
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(lambda_api, "ensure_filesystem", lambda n, r: f"/lambda/nfs/{n}")
    calls = []

    def fake_launch(*, region_name, file_system_names=None, **kw):
        calls.append({"region": region_name, "fs": file_system_names})
        if region_name == "us-east-1":
            raise lambda_api.LambdaApiError("PUT /asks/1/ -> HTTP 400: insufficient-capacity")
        return "i-2"

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)
    h = jobs.launch_and_submit(_spec(network_volume="flash-weights"), seed=0,
                               instances=[_inst(region="us-east-1"), _inst(region="us-west-2")], attempt=0)
    assert h.region == "us-west-2"  # walked to the next region
    # us-east-1 tried ONCE (with fs), then walked — no extra cold retry in us-east-1
    assert [c["region"] for c in calls] == ["us-east-1", "us-west-2"]


def test_preload_mode_skips_region_when_cache_unavailable(monkeypatch):
    """In preload mode a cache-ensure failure SKIPS the region — never a cold full-training launch.

    Regression: the cold user_data carries no mode/models, so falling back to it for a preload would
    boot a full training run (GPU billing, timeout) and warm nothing. The walk must try the next
    region, and fail if none can host the cache.
    """
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(
        lambda_api, "ensure_filesystem",
        lambda n, r: (_ for _ in ()).throw(lambda_api.LambdaApiError("no FS capacity")),
    )

    launched = []
    monkeypatch.setattr(lambda_api, "launch_instance", lambda **kw: launched.append(kw) or "i-x")

    insts = [_inst(region="us-east-1"), _inst(region="us-west-2")]
    with pytest.raises(lambda_api.LambdaApiError):
        jobs.launch_and_submit(
            _spec(network_volume="flash-weights"), seed=0, instances=insts, attempt=0,
            mode="preload", models=["a/b"],
        )
    assert launched == []  # no region ever launched a cold (training) instance


def test_preload_mode_does_not_refresh_to_a_different_region(monkeypatch):
    """In preload mode a capacity rejection must NOT refresh to a NEW region and launch there.

    Regression: warm_instances pins each preload launch to one TARGET region and reports that exact
    region as warmed. If the launch is rejected and the walk refreshed (usable_instances) to a
    different region and launched there, the caller would report the cold target region as warmed.
    """
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(lambda_api, "ensure_filesystem", lambda n, r: f"/lambda/nfs/{n}")  # cache OK
    launched = []

    def reject(**kw):
        launched.append(kw)
        raise lambda_api.LambdaApiError("PUT /asks/1/ -> HTTP 400: insufficient-capacity")  # clean reject

    monkeypatch.setattr(lambda_api, "launch_instance", reject)
    refresh_calls = []
    monkeypatch.setattr(
        jobs, "usable_instances",
        lambda gpu, force=False: refresh_calls.append(force) or [_inst(region="us-fresh-9")],
    )

    with pytest.raises(lambda_api.LambdaApiError):
        jobs.launch_and_submit(
            _spec(network_volume="flash-weights"), seed=0, instances=[_inst(region="us-east-1")],
            attempt=0, mode="preload", models=["a/b"],
        )
    assert [c["region_name"] for c in launched] == ["us-east-1"]  # only the TARGET region attempted
    assert refresh_calls == []  # the stale-stock refresh was NOT consulted in preload mode


def test_no_cache_never_touches_filesystems(monkeypatch):
    jobs, lambda_api, calls = _wire_launch(monkeypatch)

    def boom(*a, **k):
        raise AssertionError("ensure_filesystem must not be called without a requested cache")

    monkeypatch.setattr(lambda_api, "ensure_filesystem", boom)
    jobs.launch_and_submit(_spec(), seed=0, instances=[_inst()], attempt=0)  # spec has no network_volume
    assert calls[0]["fs"] is None
    assert "/weight-cache" not in calls[0]["user_data"]


def test_cache_ensured_per_region_in_the_walk(monkeypatch):
    """Lazy per-region: the FS is ensured ONLY in the region the run actually lands in (walk skips on
    capacity, ensuring then launching cold/cache per region)."""
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    ensured, attempts = [], []
    monkeypatch.setattr(lambda_api, "ensure_filesystem", lambda n, r: ensured.append(r) or f"/lambda/nfs/{n}")

    def fake_launch(*, region_name, file_system_names=None, **kw):
        attempts.append(region_name)
        if len(attempts) < 2:
            raise lambda_api.LambdaApiError("PUT /asks/1/ -> HTTP 400: insufficient-capacity")
        return "i-2"

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)
    insts = [_inst(region="us-east-1"), _inst(region="us-west-2")]
    jobs.launch_and_submit(_spec(network_volume="flash-weights"), seed=0, instances=insts, attempt=0)
    # Ensured in every region we actually attempted (east failed capacity, west succeeded) — never a
    # whole-fleet pre-create.
    assert ensured == ["us-east-1", "us-west-2"]


# ---------------------------------------------------------------------------
# poll_lambda_job state machine
# ---------------------------------------------------------------------------
def _wire_poll(monkeypatch, instances, done=None, marker=None, metrics=None, boot=None, error=None, step=10.0):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    seq = iter(instances)
    last = {"inst": None}

    def fake_get(instance_id):
        last["inst"] = next(seq, last["inst"])
        return last["inst"]

    monkeypatch.setattr(lambda_api, "get_instance", fake_get)
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=10_000, step=step)
    monkeypatch.setattr(jobs.time, "time", lambda: float(next(clock)))

    def factory(hf_repo, path, min_interval_s=45.0):
        def read(force=False):
            if path.endswith("/DONE"):
                return done() if callable(done) else done
            if "lambda_attempt" in path and path.endswith(".json"):  # not the _boot.log
                return marker() if callable(marker) else marker
            if path.endswith("metrics.json"):
                return metrics() if callable(metrics) else metrics
            if path.endswith("_boot.log"):  # attempt-scoped: lambda_attempt<N>_boot.log
                return boot() if callable(boot) else boot
            if "/error_" in path:
                return error() if callable(error) else error
            return None

        return read

    monkeypatch.setattr(jobs, "_make_hf_file_reader", factory)
    return jobs


def test_poll_success_stamps_real_cost(monkeypatch):
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}],
        done="10500.0",
        metrics=json.dumps({"train_tokens": 4096, "wall_seconds": 100, "cost_usd": 0.0}),
    )
    # started_ts precedes the mocked clock (starts 10_000) so wall is positive on the first tick.
    res = jobs.poll_lambda_job(_handle(started_ts=9_000.0), _spec(), seed=0, interval_s=0)
    assert res.ok
    assert res.metrics["train_tokens"] == 4096
    # cost comes from the instance's real $/hr x wall time, not a runpod table rate
    assert res.metrics["cost_usd"] > 0
    assert res.metrics["notes"]["provider"] == "lambda"
    assert res.metrics["notes"]["lambda_rate_usd_hr"] == 1.29
    assert res.metrics["notes"]["lambda_region"] == "us-east-1"


def test_poll_caps_recovered_cost_at_done_timestamp(monkeypatch):
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}],
        done="9100.0",
        metrics=json.dumps({"wall_seconds": 100, "cost_usd": 0.0}),
    )
    res = jobs.poll_lambda_job(_handle(started_ts=9000.0), _spec(), seed=0, interval_s=0)
    assert res.ok
    assert res.metrics["cost_usd"] == round((9100.0 - 9000.0) / 3600.0 * 1.29, 6)


def test_poll_marker_failure_is_job_failed(monkeypatch):
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}],
        marker=json.dumps({"ok": False, "attempt": 0, "error": "RuntimeError: boom"}),
    )
    res = jobs.poll_lambda_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "job_failed"  # real worker error fails fast
    assert "boom" in res.detail


def test_poll_retriable_marker_is_job_preempted(monkeypatch):
    """A worker-flagged retriable failure retries on a fresh host (job_preempted), not job_failed.
    With NO training heartbeat (only the retriable bit), it is a pre-training infra fault -> the region
    is quarantined (host_fault)."""
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}],
        marker=json.dumps({"ok": False, "attempt": 0, "error": "transient"}),
    )
    res = jobs.poll_lambda_job(
        _handle(), _spec(), seed=0, interval_s=0, heartbeat_reader=lambda force=False: {"retriable": True}
    )
    assert not res.ok
    assert res.failure == "job_preempted"
    assert res.host_fault  # retriable crash before any training heartbeat -> sick region


def test_poll_midtraining_retriable_marker_does_not_quarantine(monkeypatch):
    """A retriable failure marker can land in the SAME poll iteration the worker first reaches training
    -- the marker branch decides host_fault BEFORE surface_heartbeat() advances seen_training_hb.
    reached_training_now() force-reads the heartbeat and sees the training stage, so a mid-training
    RetriableInfraError retries (job_preempted) WITHOUT quarantining the healthy region."""
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}],
        marker=json.dumps({"ok": False, "attempt": 0, "error": "RetriableInfraError: gpu fell off the bus", "retriable": True}),
    )
    res = jobs.poll_lambda_job(
        _handle(), _spec(), seed=0, interval_s=0,
        heartbeat_reader=lambda force=False: {"stage": "sft_train", "step": 7, "ts": 10_000.0},
    )
    assert not res.ok
    assert res.failure == "job_preempted"
    assert not res.host_fault  # training already reached -> region healthy, do NOT quarantine


def test_poll_reattach_just_active_floored_by_observed_grace(monkeypatch):
    """On a reattach whose first poll already sees the box active, active_since is launch-anchored, so
    the launch-relative first_liveness deadline is already blown. The observed-grace floor stops a box
    that only JUST became active (after a long provision the control plane missed) from being failed
    over before its boot-log uploader's publication window: even with the boot.log absent past
    BOOT_LOG_ABSENT_POLLS and the deadline exceeded, no 'no worker liveness' stall fires until we've
    watched it active for FIRST_LIVENESS_OBSERVED_GRACE_S. Here the box dies (host loss) inside that
    window -> job_preempted, not the premature liveness stall (which the pre-fix code would return)."""
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}] * 4 + [{"status": "terminated"}],
        boot=None,  # uploader has not published yet
        step=0.1,  # tiny steps so the 120s observed-grace floor is NOT reached in these few polls
    )
    # Launch 1_000s ago (clock starts 10_000): the first_liveness deadline (10s) is blown, but the
    # 3_000s setup grace is NOT (else its launch-anchored stall would fire first and mask the floor).
    res = jobs.poll_lambda_job(
        _handle(started_ts=9_000.0), _spec(), seed=0, interval_s=0, first_liveness_s=10.0
    )
    assert not res.ok
    assert res.failure == "job_preempted"  # died inside the observed-grace window
    assert "no worker liveness" not in (res.detail or "")


def test_poll_dead_host_without_marker_is_preempted(monkeypatch):
    """A host that died without writing DONE/marker is a host loss -> retryable, with the boot log."""
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}, {"status": "terminated"}],
        boot="+ docker pull ...\nFLASH: gpu never became ready",
    )
    res = jobs.poll_lambda_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "job_preempted"
    assert "gpu never became ready" in res.detail  # the host boot log is the only console window


def test_poll_dead_host_with_error_file_is_job_failed(monkeypatch):
    """A worker that RAN and crashed early (left error_<phase>.txt) but died before writing the
    attempt marker is a DETERMINISTIC worker error -> fail fast (job_failed), not burn fresh GPUs
    retrying a crash that will repeat. Surfaces the traceback in the detail."""
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}, {"status": "terminating"}],
        error="Traceback (most recent call last):\nFileNotFoundError: environment archive did not contain ...",
    )
    res = jobs.poll_lambda_job(
        _handle(), _spec(), seed=0, interval_s=0, heartbeat_reader=lambda force=False: {}
    )
    assert not res.ok
    assert res.failure == "job_failed"
    assert "environment archive" in res.detail


def test_poll_dead_host_with_retriable_error_still_preempted(monkeypatch):
    """Even WITH an error_<phase>.txt, a crash the worker flagged retriable (RetriableInfraError,
    stamped in the heartbeat) retries on a fresh host (job_preempted) -- same contract as
    fail_from_marker. This is also what keeps a stale prior-attempt error file from flipping a
    genuine preemption to job_failed."""
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}, {"status": "terminating"}],
        error="Traceback ...\nRetriableInfraError: cuda device not ready",
    )
    res = jobs.poll_lambda_job(
        _handle(), _spec(), seed=0, interval_s=0, heartbeat_reader=lambda force=False: {"retriable": True}
    )
    assert not res.ok
    assert res.failure == "job_preempted"


def test_poll_loading_timeout(monkeypatch):
    jobs = _wire_poll(monkeypatch, instances=[{"status": "booting"}], step=100.0)
    monkeypatch.setattr(jobs, "LOAD_TIMEOUT_S", 300.0)
    res = jobs.poll_lambda_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "stalled"
    assert "never became active" in res.detail


def test_poll_heartbeat_stall(monkeypatch):
    jobs = _wire_poll(monkeypatch, instances=[{"status": "active"}], step=100.0)
    # A FRESH training heartbeat (ts >= launch 10_000) that then FROZE: it proves liveness (so the
    # fast first-liveness failover is satisfied) AND arms the tight training stall window, so the
    # subsequent no-progress gap past stall_after_s is the stall actually under test here.
    frozen = {"stage": "rl", "step": 3, "ts": 10_000.0}
    res = jobs.poll_lambda_job(
        _handle(),
        _spec(),
        seed=0,
        interval_s=0,
        heartbeat_reader=lambda force=False: frozen,
        stall_after_s=500.0,
    )
    assert not res.ok
    assert res.failure == "stalled"
    assert "no worker progress" in res.detail
    # a mid-TRAINING stall is NOT a region fault (the region was working) -> no quarantine
    assert not res.host_fault


def test_submit_quarantines_region_on_host_fault(monkeypatch):
    """When the poll returns a host fault (worker never reached training in this region),
    submit_run_lambda quarantines the region so the next allocation/launch avoids it."""
    import flash.providers._health as health
    from flash.providers.base import PollResult
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    health.clear()
    handle = _handle()  # region us-east-1
    monkeypatch.setattr(jobs, "usable_instances", lambda gpu, force=False: [_inst()])
    monkeypatch.setattr(jobs, "launch_and_submit", lambda *a, **k: handle)
    monkeypatch.setattr(
        jobs,
        "poll_lambda_job",
        lambda *a, **k: PollResult(False, failure="stalled", detail="no worker liveness", host_fault=True),
    )
    monkeypatch.setattr(lambda_api, "terminate_instances", lambda ids: None)

    assert not health.region_is_sick("lambda", "us-east-1")
    res = jobs.submit_run_lambda(_spec(), seed=0)
    assert res.host_fault
    assert health.region_is_sick("lambda", "us-east-1")  # region now quarantined


def test_poll_active_no_liveness_fails_over_fast(monkeypatch):
    """The observed Lambda us-east-1 sick region: the instance reaches OS 'active' but the worker
    NEVER starts — no host boot.log, no heartbeat, no marker. The first-liveness deadline fails it
    over fast as a retriable 'stalled' (escaped cross-provider by the runner) instead of burning the
    full ~50 min setup grace."""
    jobs = _wire_poll(monkeypatch, instances=[{"status": "active"}], step=100.0)
    res = jobs.poll_lambda_job(_handle(), _spec(), seed=0, interval_s=0, first_liveness_s=500.0)
    assert not res.ok
    assert res.failure == "stalled"  # infra-shaped -> retried + escaped cross-provider (PR #241)
    assert "no worker liveness" in res.detail
    assert "limit 500s" in res.detail
    assert res.host_fault  # the region never booted a worker -> submit_run quarantines it


def test_poll_active_boot_log_protects_slow_cold_start(monkeypatch):
    """A HEALTHY box still in its long cold start (multi-GB image pull) emits the host boot.log but
    no heartbeat yet — the first-liveness deadline must NOT fire (that would kill a good box). Modeled
    by an active box whose attempt-scoped boot.log is present; it later dies, so the terminal result
    is job_preempted, NOT the 'no worker liveness' stalled."""
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}, {"status": "active"}, {"status": "terminated"}],
        boot="+ docker pull ... (still pulling the worker image)",
        step=100.0,
    )
    res = jobs.poll_lambda_job(_handle(), _spec(), seed=0, interval_s=0, first_liveness_s=50.0)
    assert not res.ok
    assert res.failure == "job_preempted"  # died as a host loss, NOT killed by the liveness deadline
    assert "no worker liveness" not in (res.detail or "")
    # died before any training heartbeat -> still a host fault (region quarantined)
    assert res.host_fault


def test_poll_active_empty_boot_log_counts_as_liveness(monkeypatch):
    """An empty ("") boot.log still proves cloud-init ran — its mere EXISTENCE is liveness. A bare
    ``not boot_log_reader()`` would treat "" as absent and spuriously fail the box over; the fix uses
    ``is None``. Box later dies as a host loss -> job_preempted, NOT the 'no worker liveness' stall."""
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}, {"status": "active"}, {"status": "terminated"}],
        boot="",
        step=100.0,
    )
    res = jobs.poll_lambda_job(_handle(), _spec(), seed=0, interval_s=0, first_liveness_s=50.0)
    assert res.failure == "job_preempted"
    assert "no worker liveness" not in (res.detail or "")


def test_poll_active_boot_log_seen_once_survives_rate_limited_none(monkeypatch):
    """Regression: make_hf_text_reader returns None for BOTH a missing boot.log AND a rate-limited
    read, so a bare ``not boot_log_reader()`` re-checked each poll would spuriously stall a HEALTHY box
    on the first throttled read after the log was already seen. The boot.log is read with force=True and
    latched once observed, so a later None can't re-trigger failover. Modeled by a boot.log present on
    the first read then None (rate-limited) after; the box later dies -> job_preempted, not stalled."""
    calls = {"n": 0}

    def boot_then_rate_limited():
        calls["n"] += 1
        return "+ docker pull ..." if calls["n"] == 1 else None  # seen once, then "rate-limited"

    jobs = _wire_poll(
        monkeypatch,
        instances=[
            {"status": "active"}, {"status": "active"}, {"status": "active"}, {"status": "terminated"}
        ],
        boot=boot_then_rate_limited,
        step=100.0,
    )
    res = jobs.poll_lambda_job(_handle(), _spec(), seed=0, interval_s=0, first_liveness_s=50.0)
    assert res.failure == "job_preempted"  # NOT a spurious 'stalled' from the throttled None
    assert "no worker liveness" not in (res.detail or "")
    # Latched after the first observation: the liveness check reads the boot.log once (not once per
    # poll); the only other read is the terminal-failure-detail surfacer when the box dies.
    assert calls["n"] <= 2


def test_poll_active_transient_boot_log_error_does_not_fail_over(monkeypatch):
    """make_hf_text_reader returns None for a MISSING boot.log AND a momentary HF/Hub network error,
    so a lone forced-read None at the first-liveness deadline must NOT immediately stall — a transient
    blip clears on the next poll (the absence must persist BOOT_LOG_ABSENT_POLLS times to fail over).
    Here the first forced read errors (None), the next returns the real boot.log -> latched, no
    failover; the box later dies -> job_preempted, NOT a spurious 'stalled' from the one transient
    None."""
    calls = {"n": 0}

    def transient_then_present():
        calls["n"] += 1
        return None if calls["n"] == 1 else "+ docker pull ..."  # transient error first, then readable

    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}, {"status": "active"}, {"status": "terminated"}],
        boot=transient_then_present,
        step=100.0,
    )
    res = jobs.poll_lambda_job(_handle(), _spec(), seed=0, interval_s=0, first_liveness_s=50.0)
    assert res.failure == "job_preempted"  # the single transient None did not trip a failover
    assert "no worker liveness" not in (res.detail or "")


def test_poll_active_persistent_boot_log_absence_stalls_after_threshold(monkeypatch):
    """The genuine sick-region case: the boot.log is absent on EVERY forced read (cloud-init never
    ran). After BOOT_LOG_ABSENT_POLLS consecutive absent reads the first-liveness check declares the
    region 'stalled' (retriable, escaped cross-provider). Asserts the absence-count threshold is what
    gates the failover, not a single read."""
    from flash.providers._poll import BOOT_LOG_ABSENT_POLLS

    calls = {"n": 0}

    def always_absent():
        calls["n"] += 1  # implicit None: every forced read comes back absent

    jobs = _wire_poll(
        monkeypatch, instances=[{"status": "active"}], boot=always_absent, step=100.0
    )
    res = jobs.poll_lambda_job(_handle(), _spec(), seed=0, interval_s=0, first_liveness_s=50.0)
    assert res.failure == "stalled"
    assert "no worker liveness" in res.detail
    assert calls["n"] >= BOOT_LOG_ABSENT_POLLS  # required the absence to persist, not a lone None


def test_poll_active_fresh_heartbeat_satisfies_liveness(monkeypatch):
    """Any FRESH heartbeat (even the early 'boot' stage) proves the worker started, so the
    first-liveness deadline is satisfied and must not fire."""
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}, {"status": "active"}, {"status": "terminated"}],
        step=100.0,
    )
    res = jobs.poll_lambda_job(
        _handle(),
        _spec(),
        seed=0,
        interval_s=0,
        first_liveness_s=50.0,
        heartbeat_reader=lambda force=False: {"stage": "boot", "step": 0, "ts": 10_000.0},
    )
    assert res.failure == "job_preempted"
    assert "no worker liveness" not in (res.detail or "")


def test_poll_active_stale_heartbeat_does_not_satisfy_liveness(monkeypatch):
    """A LEFTOVER heartbeat from a PRIOR attempt (ts < this launch; the heartbeat path is not
    attempt-scoped) must NOT disarm the deadline — otherwise the retry INTO the sick region it must
    catch would be let through. With no fresh boot.log either, first-liveness still fires."""
    jobs = _wire_poll(monkeypatch, instances=[{"status": "active"}], step=100.0)
    res = jobs.poll_lambda_job(
        _handle(started_ts=10_000.0),
        _spec(),
        seed=0,
        interval_s=0,
        first_liveness_s=50.0,
        heartbeat_reader=lambda force=False: {"stage": "boot", "step": 0, "ts": 1.0},  # < launch
    )
    assert res.failure == "stalled"
    assert "no worker liveness" in res.detail


def test_poll_reattach_already_active_anchors_liveness_to_launch(monkeypatch):
    """On a reattach after a control-plane restart, the FIRST status read is already ACTIVE
    (last_status starts None, so it is not a transition). active_since must stay anchored to LAUNCH,
    so a box silent since before the restart fails over on the first tick rather than getting a fresh
    full first-liveness window. (Clock starts 10_000; launch was 5_000s earlier.)

    The box is silent for ~5_000s — already PAST the 3_000s setup grace — so the setup-grace stall
    (which needs no boot.log read) fires immediately, before the first-liveness check accumulates its
    BOOT_LOG_ABSENT_POLLS confirmations. Either way it's a retriable ``stalled`` and the reported
    elapsed (~5_020s) is measured from LAUNCH, not the reattach (~0s) — which is what proves the
    anchoring. A fresh launch (elapsed < setup grace) still fails over via the fast first-liveness
    path well before the setup grace, so the FAST-failover guarantee is unaffected."""
    jobs = _wire_poll(monkeypatch, instances=[{"status": "active"}], step=10.0)
    res = jobs.poll_lambda_job(
        _handle(started_ts=5_000.0), _spec(), seed=0, interval_s=0, first_liveness_s=500.0
    )
    assert not res.ok
    assert res.failure == "stalled"
    # Elapsed counts from LAUNCH (~5_000s ago), not the reattach (~0s) — proves active_since stayed
    # anchored. Parse the reported elapsed rather than hard-coding it (clock-call count is incidental).
    import re

    elapsed = int(re.search(r"for (\d+)s", res.detail).group(1))
    assert elapsed >= 5_000  # launch-anchored, not reattach-anchored


def test_cloud_init_emits_boot_log_before_pull_and_attempt_scoped(monkeypatch):
    """The host boot-log uploader must run BEFORE the docker image pull (so a box that ran cloud-init
    leaves an HF liveness artifact within ~2 min, well before the worker's first heartbeat), and its
    HF path must be attempt-scoped so a prior attempt's boot.log can't falsely prove liveness."""
    from flash.providers.lambdalabs.jobs import builders

    monkeypatch.setenv("LAMBDA_API_KEY", "lk")
    monkeypatch.setenv("HF_TOKEN", "hf")
    payload = builders.build_payload(_spec(), seed=0, attempt=2)
    script = builders.build_user_data(payload)
    # the uploader INVOCATION precedes the image pull
    assert "python3 /opt/flash/hostlog.py" in script
    assert "docker pull" in script
    assert script.index("python3 /opt/flash/hostlog.py") < script.index("docker pull")
    # attempt-scoped boot.log path is emitted by the embedded uploader
    assert '_attempt" + str(att) + "_boot.log' in script


def test_poll_recovery_seeds_load_clock_from_launch(monkeypatch):
    """Reattach after a control-plane restart: a still-booting box has been billing since LAUNCH
    (handle.started_ts), so LOAD_TIMEOUT_S is measured from launch, NOT from this poll's first
    tick. A box already past the load window fails over on the first reattach iteration instead of
    getting another full window. (The mocked clock starts at 10_000; launch was 5000s earlier.)"""
    import re

    jobs = _wire_poll(monkeypatch, instances=[{"status": "booting"}], step=10.0)
    res = jobs.poll_lambda_job(_handle(started_ts=5_000.0), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "stalled"
    assert "never became active" in res.detail
    m = re.search(r"for (\d+)s", res.detail)
    assert m is not None, res.detail
    # launch-relative (~5000s); the old "reset to reattach tick" code would report ~LOAD_TIMEOUT_S.
    assert int(m.group(1)) >= 2000, res.detail


def test_poll_missing_started_ts_anchors_to_now_not_epoch(monkeypatch):
    """started_ts is a non-Optional float coerced to 0.0 when MISSING (old/corrupt handle), so 0.0
    means 'unknown launch' (a real launch is a large epoch ts). EVERYTHING (the timeout clocks AND
    done_is_fresh / finish_ok's wall+cost stamping) must anchor to now, NOT the epoch — otherwise a
    booting box would be 'past' a ~57-year-old load window and stall on the first tick, and wall/cost
    would be billed from 1970. DONE then completes the run normally with a sane (tiny) wall."""
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}],
        done="10500.0",
        metrics=json.dumps({"wall_seconds": 100, "cost_usd": 0.0}),
        step=10.0,
    )
    res = jobs.poll_lambda_job(_handle(started_ts=0.0), _spec(), seed=0, interval_s=0)
    assert res.ok, res  # not instantly stalled by an epoch-anchored deadline/load clock
    # wall/cost are NOT billed from the 1970 epoch: launch_ts fell back to now (~10_000 mocked),
    # so the stamped cost is a few seconds of wall, not ~57 years x $1.29/hr (= astronomically large).
    assert res.metrics["cost_usd"] < 1.0, res.metrics["cost_usd"]


def test_heartbeat_progress_ts_unknown_launch_treats_heartbeats_as_fresh():
    """When launch is UNKNOWN (launch_ts=0.0, from a recovered handle missing started_ts), the
    clamp floor must drop to 0.0 so a normal heartbeat — timestamped before it is read, i.e. < now —
    counts as FRESH and credits its own ts. Flooring to `now` would mark every such heartbeat stale
    and stall a healthy recovered worker after SETUP_GRACE_S despite continuous heartbeats."""
    from flash.providers._poll import heartbeat_progress_ts

    hb_ts = time.time() - 30.0  # a normal recent heartbeat, slightly in the past
    ts, fresh = heartbeat_progress_ts(("rl", 4, hb_ts), launch_ts=0.0)
    assert fresh is True  # unknown launch -> not discarded
    assert abs(ts - hb_ts) < 1.0  # credits the heartbeat's own ts (not clamped up to now)

    # A real (non-zero) launch still discriminates prior-attempt leftovers (ts < launch).
    launch = time.time() - 100.0
    _, fresh_old = heartbeat_progress_ts(("rl", 1, launch - 50.0), launch_ts=launch)
    assert fresh_old is False
    _, fresh_new = heartbeat_progress_ts(("rl", 9, launch + 10.0), launch_ts=launch)
    assert fresh_new is True


def test_poll_stale_heartbeat_does_not_buy_fresh_window(monkeypatch):
    """A heartbeat that was already stale before a restart must not reset the stall clock to the
    reattach time: its OWN ts is credited as last-progress, so an active worker frozen long ago
    stalls promptly instead of getting another full stall window. (Clock starts 10_000; the
    worker's last heartbeat was at 8500, launch at 8000, stall budget 500s.)"""
    import re

    jobs = _wire_poll(monkeypatch, instances=[{"status": "active"}], step=10.0)
    hb = {"stage": "rl", "step": 7, "ts": 8500.0}
    res = jobs.poll_lambda_job(
        _handle(started_ts=8_000.0),
        _spec(),
        seed=0,
        interval_s=0,
        heartbeat_reader=lambda force=False: hb,
        stall_after_s=500.0,
    )
    assert not res.ok
    assert res.failure == "stalled"
    assert "no worker progress" in res.detail
    m = re.search(r"for (\d+)s", res.detail)
    assert m is not None, res.detail
    # measured from the heartbeat ts (~1500s+), not the reattach tick (which the old code used,
    # yielding only ~stall_after_s).
    assert int(m.group(1)) >= 1000, res.detail


def test_poll_prior_attempt_heartbeat_does_not_arm_training_stall(monkeypatch):
    """A LEFTOVER heartbeat from a PRIOR attempt (ts < this attempt's launch; retries reuse the same
    seed heartbeat path) must not be treated as current progress. Clamping its ts up to launch made
    a stale training-stage heartbeat arm the tighter training stall window and fail a healthy new
    attempt mid-setup before it overwrote the file. With the freshness gate, a pre-launch heartbeat
    neither advances last_progress nor sets seen_training_hb, so the run gets the longer SETUP grace
    measured from launch. (Clock starts 10_000; launch 9000; old heartbeat ts 8000 < launch.)"""
    import re

    # boot.log present: the box DID run cloud-init THIS attempt (so the fast first-liveness failover
    # is satisfied), isolating the behavior under test — a STALE heartbeat must not arm the tighter
    # training stall, leaving the longer SETUP grace to govern.
    jobs = _wire_poll(
        monkeypatch, instances=[{"status": "active"}], step=10.0, boot="+ cloud-init\n+ docker pull"
    )
    stale = {"stage": "rl", "step": 2, "ts": 8000.0}  # training stage, but predates this launch
    res = jobs.poll_lambda_job(
        _handle(started_ts=9_000.0),
        _spec(),
        seed=0,
        interval_s=0,
        heartbeat_reader=lambda force=False: stale,
        setup_grace_s=3000.0,
        stall_after_s=500.0,
    )
    assert not res.ok
    assert res.failure == "stalled"
    # Stalls on SETUP grace (3000s from launch), not the tighter 500s training window the stale
    # heartbeat would have armed -> the reported idle time exceeds the training budget.
    assert "setup (pre-training)" in res.detail
    m = re.search(r"for (\d+)s", res.detail)
    assert m is not None, res.detail
    assert int(m.group(1)) >= 3000, res.detail


def test_poll_client_deadline(monkeypatch):
    jobs = _wire_poll(monkeypatch, instances=[{"status": "active"}], step=100.0)
    res = jobs.poll_lambda_job(_handle(), _spec(), seed=0, interval_s=0, deadline_s=250.0)
    assert not res.ok
    assert res.failure == "stalled"
    assert "deadline" in res.detail


def test_poll_recovered_deadline_persists_done_written_during_outage(monkeypatch):
    """A control-plane outage longer than the launch-anchored deadline must NOT discard a seed the
    worker actually finished during the downtime: before returning the deadline `stalled`, the poller
    reads terminal artifacts once and persists a fresh DONE. (Clock starts 10_000; launch 5_000s ago,
    so the very first deadline check fires; DONE=10_400 is fresh vs launch.)"""
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}],
        done="10400.0",
        metrics=json.dumps({"wall_seconds": 100, "cost_usd": 0.0}),
        step=10.0,
    )
    res = jobs.poll_lambda_job(
        _handle(started_ts=5_000.0), _spec(), seed=0, interval_s=0, deadline_s=250.0
    )
    assert res.ok, res  # success persisted, NOT a stalled-retry that throws away the finished seed
    assert res.metrics["cost_usd"] > 0


def test_poll_recovered_deadline_without_artifacts_still_stalls(monkeypatch):
    """When the recovered deadline fires and there is NO terminal artifact, the poller still returns
    `stalled` (the worker did not finish during the outage)."""
    jobs = _wire_poll(monkeypatch, instances=[{"status": "active"}], step=10.0)
    res = jobs.poll_lambda_job(
        _handle(started_ts=5_000.0), _spec(), seed=0, interval_s=0, deadline_s=250.0
    )
    assert not res.ok
    assert res.failure == "stalled"
    assert "deadline" in res.detail


def test_provider_poll_passes_full_launch_relative_deadline(monkeypatch):
    """The reattach path must NOT pre-subtract elapsed-since-launch from the deadline: the poll loop
    already anchors its deadline check to handle.started_ts (= launch), so subtracting elapsed here
    too double-counts and tears down a still-valid instance once a recovered run is past half its
    window. LambdaProvider.poll must pass the FULL launch-relative budget regardless of how old
    started_ts is."""
    from flash.providers.base import JobHandle
    from flash.providers.lambdalabs import LambdaProvider
    from flash.providers.lambdalabs.jobs import PROVISION_GRACE_S

    captured = {}

    def fake_poll(handle, spec, seed, *, log=None, heartbeat_reader=None, deadline_s=None,
                  first_liveness_s=None, setup_grace_s=None):
        captured["deadline_s"] = deadline_s
        from flash.providers.base import PollResult

        return PollResult(True)

    monkeypatch.setattr("flash.providers.lambdalabs.jobs.poll_lambda_job", fake_poll)
    monkeypatch.setattr("flash.providers.lambdalabs.api.terminate_instances", lambda ids: ids)
    spec = _spec()  # max_wall_seconds=3600
    # started_ts long in the past (recovered well past half its window).
    handle = JobHandle.from_dict({"provider": "lambda", **_handle(started_ts=1.0).to_dict()})
    LambdaProvider().poll(handle, spec, seed=0)
    assert captured["deadline_s"] == max(60.0, 3600 + PROVISION_GRACE_S)


def test_provider_poll_reuses_on_last_gpu_first_liveness_scaling(monkeypatch):
    """On recovery the reattach must reproduce the SUBMIT path's last-GPU stall tuning: a handle with
    persisted on_last_gpu=True (written by the runner's on_handle) gets the 1.5x-scaled first_liveness /
    setup grace, else a control-plane restart on the LAST candidate would fail an in-flight,
    cold-starting instance early — terminal there, with no GPU left to walk to. Mirrors RunPodProvider."""
    from flash.providers.base import JobHandle
    from flash.providers.lambdalabs import LambdaProvider
    from flash.providers.lambdalabs.jobs import FIRST_LIVENESS_S, SETUP_GRACE_S

    captured = {}

    def fake_poll(handle, spec, seed, *, log=None, heartbeat_reader=None, deadline_s=None,
                  first_liveness_s=None, setup_grace_s=None):
        captured["first_liveness_s"] = first_liveness_s
        captured["setup_grace_s"] = setup_grace_s
        from flash.providers.base import PollResult

        return PollResult(True)

    monkeypatch.setattr("flash.providers.lambdalabs.jobs.poll_lambda_job", fake_poll)
    monkeypatch.setattr("flash.providers.lambdalabs.api.terminate_instances", lambda ids: ids)
    spec = _spec()
    # on_last_gpu persisted in the handle -> 1.5x scaling reproduced on recovery.
    handle = JobHandle.from_dict({**_handle().to_dict(), "provider": "lambda", "on_last_gpu": True})
    LambdaProvider().poll(handle, spec, seed=0)
    assert captured["first_liveness_s"] == FIRST_LIVENESS_S * 1.5
    assert captured["setup_grace_s"] == SETUP_GRACE_S * 1.5
    # Without it -> un-scaled default.
    captured.clear()
    handle2 = JobHandle.from_dict({**_handle().to_dict(), "provider": "lambda"})
    LambdaProvider().poll(handle2, spec, seed=0)
    assert captured["first_liveness_s"] == FIRST_LIVENESS_S
    assert captured["setup_grace_s"] == SETUP_GRACE_S


def test_poll_surfaces_worker_progress_in_log(monkeypatch):
    # DONE appears only on the 2nd poll, so the loop reaches the heartbeat-surfacing block first.
    done_seq = iter([None, "10500.0", "10500.0", "10500.0"])
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}],
        done=lambda: next(done_seq, "10500.0"),
        metrics=json.dumps({"wall_seconds": 100, "cost_usd": 0.0}),
    )
    log = io.StringIO()
    hb = {"stage": "sft", "step": 5, "ts": 2.0, "loss": 1.5}
    res = jobs.poll_lambda_job(
        _handle(), _spec(), seed=0, interval_s=0, log=log, heartbeat_reader=lambda force=False: hb
    )
    assert res.ok
    assert "stage=sft" in log.getvalue()


# ---------------------------------------------------------------------------
# the cost-safety invariant: every exit path terminates the instance
# ---------------------------------------------------------------------------
def _wire_runner(monkeypatch, poll_outcome):
    from flash.providers.base import PollResult
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    terminated = []
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids: terminated.append(list(ids)) or True
    )
    monkeypatch.setattr(jobs, "usable_instances", lambda gpu, force=False: [_inst()])
    monkeypatch.setattr(jobs, "launch_and_submit", lambda *a, **k: _handle())

    def fake_poll(*a, **k):
        if isinstance(poll_outcome, BaseException):
            raise poll_outcome
        return poll_outcome

    monkeypatch.setattr(jobs, "poll_lambda_job", fake_poll)
    return jobs, terminated, PollResult


def test_runner_terminates_on_success(monkeypatch):
    from flash.providers.base import PollResult

    jobs, terminated, _ = _wire_runner(monkeypatch, PollResult(True, metrics={"a": 1}))
    handles = []
    res = jobs.submit_run_lambda(_spec(), seed=0, on_handle=handles.append)
    assert res.ok
    assert terminated == [["i-9999"]]
    assert handles
    assert handles[0]["provider"] == "lambda"
    assert handles[0]["instance_id"] == "i-9999"


def test_runner_terminates_on_failure_and_exception(monkeypatch):
    from flash.providers.base import PollResult

    jobs, terminated, _ = _wire_runner(monkeypatch, PollResult(False, failure="stalled"))
    res = jobs.submit_run_lambda(_spec(), seed=0)
    assert not res.ok
    assert terminated == [["i-9999"]]

    jobs, terminated, _ = _wire_runner(monkeypatch, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        jobs.submit_run_lambda(_spec(), seed=0)
    assert terminated == [["i-9999"]]


def test_runner_terminates_when_handle_persist_fails(monkeypatch):
    """The launched instance is terminated even if on_handle raises — the teardown finally guards
    everything after the launch, not just the poll."""
    jobs, terminated, _ = _wire_runner(monkeypatch, None)

    def boom(_h):
        raise RuntimeError("status store unreachable")

    with pytest.raises(RuntimeError, match="status store unreachable"):
        jobs.submit_run_lambda(_spec(), seed=0, on_handle=boom)
    assert terminated == [["i-9999"]]


def test_submit_rejects_policy_word_gpu():
    """submit_run_lambda needs a concrete class; a policy word ("cheapest") — which the allocator
    resolves upstream — must fail with a clear error, not an opaque KeyError."""
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs.jobs import submit_run_lambda

    with pytest.raises(lambda_api.LambdaApiError, match="concrete gpu class"):
        submit_run_lambda(_spec(gpu_type="cheapest"), seed=0)


# ---------------------------------------------------------------------------
# labels, gc, orphan sweep
# ---------------------------------------------------------------------------
def test_instance_label_always_sweepable():
    from flash.providers.lambdalabs.jobs.builders import instance_label

    assert instance_label("flash-1700-abcd", 0, 1) == "flash-1700-abcd-s0-a1"
    assert instance_label("fail-fast", 0, 0) == "flash-fail-fast-s0-a0"  # prefix forced


def test_terminate_run_instances_matches_forced_prefix(monkeypatch):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    instances = [
        {"id": "i-1", "name": "flash-fail-fast-s0-a0"},  # forced-prefix name
        {"id": "i-2", "name": "flash-other-run-s0-a0"},  # different run -> keep
    ]
    terminated = []
    monkeypatch.setattr(lambda_api, "list_instances", lambda: instances)
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids: terminated.extend(ids) or list(ids)
    )
    assert jobs.terminate_run_instances("fail-fast") == ["i-1"]
    assert terminated == ["i-1"]


def test_handle_roundtrip():
    from flash.providers.lambdalabs.jobs.builders import LambdaJobHandle

    h = _handle()
    d = h.to_dict()
    assert d["provider"] == "lambda"
    assert LambdaJobHandle.from_dict(d) == h


def test_sweep_orphans_label_safety(monkeypatch):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    instances = [
        {"id": "i-1", "name": "flash-1700-aaaa-s0-a0"},  # orphan -> terminate
        {"id": "i-2", "name": "flash-1700-bbbb-s0-a1"},  # active run -> keep
        {"id": "i-3", "name": "someone-elses-workload"},  # not ours -> NEVER touch
        {"id": "i-4", "name": ""},  # unnamed -> NEVER touch
    ]
    terminated = []
    monkeypatch.setattr(lambda_api, "list_instances", lambda: instances)
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids: terminated.extend(ids) or list(ids)
    )
    out = jobs.sweep_orphans(active_labels={"flash-1700-bbbb"})
    assert out == ["i-1"]
    assert terminated == ["i-1"]


def test_sweep_orphans_prefix_not_shielded_by_longer_run_id(monkeypatch):
    """A live run id that is a STRING prefix of another must not shield the other's orphan."""
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    instances = [
        {"id": "i-1", "name": jobs.instance_label("flash-100", 0, 0)},  # live -> KEEP
        {"id": "i-2", "name": jobs.instance_label("flash-1000", 0, 0)},  # orphan -> terminate
    ]
    terminated = []
    monkeypatch.setattr(lambda_api, "list_instances", lambda: instances)
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids: terminated.extend(ids) or list(ids)
    )
    out = jobs.sweep_orphans(active_labels={"flash-100"})
    assert out == ["i-2"]


def test_sweep_orphans_protects_unprefixed_active_run_id(monkeypatch):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    instances = [
        {"id": "i-1", "name": jobs.instance_label("fail-fast", 0, 0)},  # live run -> KEEP
        {"id": "i-2", "name": jobs.instance_label("orphan-run", 0, 0)},  # no live run -> terminate
    ]
    terminated = []
    monkeypatch.setattr(lambda_api, "list_instances", lambda: instances)
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids: terminated.extend(ids) or list(ids)
    )
    out = jobs.sweep_orphans(active_labels={"fail-fast"})  # RAW run id (what the server tracks)
    assert out == ["i-2"]


def test_sweep_orphans_exempts_warm_preload_boxes(monkeypatch):
    """Warm/preload boxes (``flash-preload-...``) are driver-owned: launched by
    preload.warm_instances, never persisted in the run DB (so never in the active set), and
    self-terminated by the warm driver. The periodic sweep must NOT reap an IN-DEADLINE preload box by
    the bare ``flash-`` prefix — a catalog warm can outlast the ~10-min sweep and would be killed
    mid-download. A box with no embedded deadline (legacy launch) is likewise exempt.
    """
    import time

    from flash.providers._instance import instance_label
    from flash.providers._poll import preload_instance_run_id
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    # Build the name the way a launch does (instance_label bounds it to the provider name budget) so the
    # reap parser is tested against the REAL, possibly-truncated VM name, not the raw run id.
    fresh = preload_instance_run_id("lambda", "us-east-1", int(time.time()) + 1800, "abcdef")
    instances = [
        {"id": "i-1", "name": instance_label(fresh, 0, 0)},  # in-deadline warm box -> KEEP
        {"id": "i-legacy", "name": "flash-preload-lambda-us-east-1-abcdef-s0-a0"},  # no deadline -> KEEP
        {"id": "i-2", "name": "flash-1700-cccc-s0-a0"},  # genuine orphan -> terminate
    ]
    terminated = []
    monkeypatch.setattr(lambda_api, "list_instances", lambda: instances)
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids: terminated.extend(ids) or list(ids)
    )
    out = jobs.sweep_orphans(active_labels=set())  # none is a tracked active run
    assert out == ["i-2"]
    assert terminated == ["i-2"]


def test_sweep_orphans_reaps_stale_preload_box(monkeypatch):
    """A preload box still alive past its embedded wall deadline + grace has lost its driver (the only
    thing that terminates an instance provider — nothing on the box self-terminates the VM). The sweep
    must reap it to bound the billing leak rather than exempt it forever."""
    import time

    from flash.providers._instance import instance_label
    from flash.providers._poll import PRELOAD_REAP_GRACE_S, preload_instance_run_id
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    # Deadline well past now + the reap grace -> driver provably gone. Name built via instance_label so
    # the front-loaded deadline token must survive the provider name-budget truncation to be reaped.
    stale_deadline = int(time.time()) - int(PRELOAD_REAP_GRACE_S) - 600
    stale = preload_instance_run_id("lambda", "us-west-1", stale_deadline, "deadbe")
    instances = [{"id": "i-9", "name": instance_label(stale, 0, 0)}]
    terminated = []
    monkeypatch.setattr(lambda_api, "list_instances", lambda: instances)
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids: terminated.extend(ids) or list(ids)
    )
    out = jobs.sweep_orphans(active_labels=set())
    assert out == ["i-9"]
    assert terminated == ["i-9"]


# ---------------------------------------------------------------------------
# provider object dispatch + capacity-aware allocation
# ---------------------------------------------------------------------------
def test_provider_cancel_destroy_terminate(monkeypatch):
    from flash.providers import get_provider
    from flash.providers.base import JobHandle
    from flash.providers.lambdalabs import api as lambda_api

    terminated = []
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids: terminated.extend(ids) or list(ids)
    )
    h = JobHandle("lambda", {"instance_id": "i-9"})
    get_provider("lambda").cancel(h)
    get_provider("lambda").destroy(h)
    assert terminated == ["i-9", "i-9"]


def test_usable_instances_only_capacity_regions(monkeypatch):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs.jobs import usable_instances

    monkeypatch.setattr(lambda_api, "regions_with_capacity", lambda itype, force=False: ["us-east-1", "us-west-1"])
    monkeypatch.setattr("flash.providers.lambdalabs.pricing.hourly_rate", lambda g: 1.29)
    out = usable_instances("A10")
    assert {i.region for i in out} == {"us-east-1", "us-west-1"}
    assert all(i.gpu == "A10" and i.instance_type == "gpu_1x_a10" for i in out)
    # no capacity -> empty (the allocator then skips the class)
    monkeypatch.setattr(lambda_api, "regions_with_capacity", lambda itype, force=False: [])
    assert usable_instances("A10") == []


def test_allocator_capacity_aware(monkeypatch):
    """Lambda joins the ranked candidate list only for classes with LIVE capacity; a class with no
    capacity is excluded so the runner never walks to a class that would immediately fail to launch."""
    from flash.providers import allocator
    from flash.providers.lambdalabs.jobs.builders import LambdaInstance

    monkeypatch.setenv("LAMBDA_API_KEY", "lk")  # make lambda "available"

    def fake_usable(gpu):
        # A10 has capacity; A100 SXM 40GB does not (excluded from candidates).
        if gpu == "A10":
            return [LambdaInstance("A10", "gpu_1x_a10", "us-east-1", 24, 1.29)]
        return []

    monkeypatch.setattr("flash.providers.lambdalabs.jobs.usable_instances", fake_usable)
    a = allocator.allocate("Qwen/Qwen3.5-0.8B", "sft")
    lam = {c.gpu for c in a.candidates if c.provider == "lambda"}
    assert lam == {"A10"}  # only the in-capacity class
    # RunPod still wins on price (cheaper static rates), so it's the chosen provider.
    assert a.provider == "runpod"


# --- review-fix regressions ---
def test_poll_ok_marker_succeeds_with_stale_done(monkeypatch):
    """A retry that hits the worker's already-complete path leaves DONE stale but writes ok marker +
    metrics; the poller must treat that as SUCCESS, not poll until it stalls."""
    jobs = _wire_poll(
        monkeypatch, instances=[{"status": "active"}],
        done="9000.0",  # STALE (before the handle's started_ts=10000)
        marker=json.dumps({"ok": True, "attempt": 0}),
        metrics=json.dumps({"wall_seconds": 50, "cost_usd": 0.0}),
    )
    res = jobs.poll_lambda_job(_handle(), _spec(), seed=0, interval_s=0)
    assert res.ok
    assert res.metrics["notes"]["provider"] == "lambda"


def test_ambiguous_launch_reconciles_and_stops(monkeypatch):
    """An ambiguous launch failure (timeout/5xx, maybe created an instance) must NOT walk to another
    region — it reconciles by name and raises so the run retries cleanly (cost safety)."""
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    reaped = []
    monkeypatch.setattr(jobs, "terminate_run_instances", lambda rid: reaped.append(rid) or [])
    attempts = []

    def fake_launch(**k):
        attempts.append(k["region_name"])
        raise lambda_api.LambdaApiError("PUT /asks/1/ failed after 5 attempts: timed out")

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)
    insts = [_inst(region=r) for r in ("us-east-1", "us-west-1")]
    with pytest.raises(lambda_api.LambdaApiError, match="ambiguous"):
        jobs.launch_and_submit(_spec(), seed=0, instances=insts, attempt=0)
    assert attempts == ["us-east-1"]  # stopped after the first ambiguous failure (no 2nd launch)
    assert reaped == ["flash-1700000000-abcd1234"]  # reconciled by run-name


# ---------------------------------------------------------------------------
# #228 follow-up: don't mask worker failures + keep large specs out of user_data
# ---------------------------------------------------------------------------
def test_bootstrap_honors_nonzero_exit_without_remote_artifacts(monkeypatch):
    """Bug: a worker that exits non-zero AFTER writing /tmp/metrics.json locally but BEFORE its
    required DONE/metrics.json upload (e.g. a transient RetriableInfraError uploading them) must
    NOT be reported as success. The local file exists, so the old code wrote ok=true; now we only
    tolerate a non-zero exit when the REMOTE completion artifacts are confirmed on HF.

    The failure is infra-shaped (a failed required upload), so the marker carries retriable=True:
    during an HF outage the worker's own retriable heartbeat may also be missing, so the marker's
    flag is what makes the poller retry on a fresh host (job_preempted) instead of failing fast."""
    lb, _calls, markers = _bootstrap_env(monkeypatch, rc=1, metrics=True)
    monkeypatch.setattr(lb, "remote_completion_confirmed", lambda p: False)
    assert lb.main() == 1
    ok, error, retriable = markers[0]
    assert not ok
    assert "non-zero" in error  # propagated as a real (retriable) failure, not a false ok=true
    assert retriable is True  # infra/upload failure -> retried, not job_failed


def test_bootstrap_tolerates_nonzero_exit_when_remote_confirmed(monkeypatch):
    """The benign case the non-zero tolerance exists for: RL's colocated vLLM segfaults at
    interpreter exit AFTER the adapter + metrics.json + DONE are uploaded. Remote artifacts present
    -> still a success despite the non-zero rc."""
    lb, _calls, markers = _bootstrap_env(monkeypatch, rc=1, metrics=True)
    monkeypatch.setattr(lb, "remote_completion_confirmed", lambda p: True)
    assert lb.main() == 0
    assert markers[0] == (True, "", False)


def test_remote_completion_confirmed_requires_done_and_metrics(monkeypatch):
    """remote_completion_confirmed is True ONLY when BOTH DONE and metrics.json exist on HF, and
    stays conservative (False) on an HF read error so a non-zero exit propagates."""
    from flash.providers import _instance_bootstrap as lb

    payload = {"hf_repo": "o/r", "hf_prefix": "sft/x/seed0", "env": {}}
    present = {"DONE", "metrics.json"}
    monkeypatch.setattr(lb, "hf_file_exists", lambda p, sub: sub in present)
    assert lb.remote_completion_confirmed(payload) is True
    present.discard("DONE")
    assert lb.remote_completion_confirmed(payload) is False  # metrics but no DONE

    def boom(p, sub):
        raise RuntimeError("hf down")

    monkeypatch.setattr(lb, "hf_file_exists", boom)
    assert lb.remote_completion_confirmed(payload) is False  # read error -> conservative


def test_bootstrap_fetches_spilled_spec_from_hf(monkeypatch):
    """A large spec is spilled to HF at launch (out of user_data); the bootstrap reconstructs it
    from the sentinel (job_spec_in_hf) by fetching <hf_prefix>/job_spec.json."""
    from flash.providers import _instance_bootstrap as lb

    big = '{"k":"' + "v" * 200_000 + '"}'
    monkeypatch.setattr(lb, "fetch_spec_from_hf", lambda p: big)
    env = lb.build_worker_env(
        {"job_spec_json": "", "job_spec_in_hf": True, "phase": "sft", "seed": 0, "env": {}, "flash_arm": "lambda"}
    )
    # A >96k spec is passed via file, mirroring the inline-large path.
    assert env["FLASH_JOB_SPEC_PATH"] == "/tmp/job_spec.json"
    assert "FLASH_JOB_SPEC_JSON" not in env
    with open("/tmp/job_spec.json") as f:
        assert f.read() == big


def test_build_worker_env_raises_clearly_without_a_spec():
    """A malformed payload carrying NEITHER an inline job_spec_json NOR the job_spec_in_hf sentinel
    must raise a clear RuntimeError naming the cause — not crash on len(None) with an opaque
    TypeError that buries the real (control-plane payload) bug."""
    from flash.providers import _instance_bootstrap as lb

    with pytest.raises(RuntimeError, match="no job spec"):
        lb.build_worker_env(
            {"phase": "sft", "seed": 0, "env": {}, "flash_arm": "lambda"}  # no job_spec_* at all
        )


def test_build_worker_env_spilled_spec_fetch_failure_is_retriable(monkeypatch):
    """The pre-worker HF fetch of a spilled spec is infra-shaped: a transient failure must surface
    as RetriableBootstrapError (not a bare error) so main() marks the attempt retriable and the
    poller retries on a fresh host instead of failing the run fast."""
    from flash.providers import _instance_bootstrap as lb

    monkeypatch.setattr(
        lb, "fetch_spec_from_hf", lambda p: (_ for _ in ()).throw(RuntimeError("hf 503"))
    )
    with pytest.raises(lb.RetriableBootstrapError, match="spilled job spec"):
        lb.build_worker_env(
            {"job_spec_json": "", "job_spec_in_hf": True, "phase": "sft", "seed": 0,
             "env": {}, "flash_arm": "lambda"}
        )


def test_main_marks_spilled_spec_fetch_failure_retriable(monkeypatch):
    """End-to-end: a payload whose spilled-spec HF fetch fails -> main() exits non-zero AND the
    written attempt marker carries retriable=True (so the poller -> job_preempted, not job_failed)."""
    from flash.providers import _instance_bootstrap as lb

    markers: list[tuple[bool, str, bool]] = []
    monkeypatch.setattr(
        lb,
        "load_payload",
        lambda path=lb.PAYLOAD_PATH: {
            "hf_repo": "org/repo", "job_spec_json": "", "job_spec_in_hf": True,
            "phase": "sft", "seed": 0, "flash_arm": "lambda", "env": {}, "extra_pip": [],
            "hf_prefix": "sft/x/seed0", "max_wall_s": 60, "attempt": 0,
        },
    )
    monkeypatch.setattr(lb, "fetch_code", lambda p: None)
    monkeypatch.setattr(lb, "fetch_spec_from_hf", lambda p: (_ for _ in ()).throw(RuntimeError("hf 503")))
    monkeypatch.setattr(
        lb, "write_attempt_marker",
        lambda p, ok, error="", retriable=False: markers.append((ok, error, retriable)),
    )
    assert lb.main() == 1
    ok, error, retriable = markers[0]
    assert not ok
    assert "spilled job spec" in error
    assert retriable is True


def test_build_user_data_spills_large_spec_out_of_cloud_init(monkeypatch):
    """A large job_spec_json must NOT be embedded inline in user_data (it can overflow the
    provider's cloud-init size cap and reject the launch). It is uploaded to HF and replaced by a
    small sentinel; small specs ride inline unchanged."""
    import huggingface_hub

    from flash.providers import _instance as inst

    uploaded = {}

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type):
            uploaded["path"] = path_in_repo
            uploaded["repo"] = repo_id
            uploaded["fileobj"] = path_or_fileobj

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    big = '{"k":"' + "v" * 100_000 + '"}'
    payload = {
        "flash_arm": "lambda",
        "job_spec_json": big,
        "hf_repo": "o/r",
        "hf_prefix": "sft/x/seed0",
        "env": {"HF_TOKEN": "t"},
        "attempt": 0,
    }
    ud = inst.build_user_data(payload, image="img:latest")
    embedded = json.loads(base64.b64decode(ud.split("FLASH_PAYLOAD_EOF")[1].strip()))
    assert embedded["job_spec_json"] == ""
    assert embedded["job_spec_in_hf"] is True
    assert uploaded["path"] == "sft/x/seed0/job_spec.json"
    assert uploaded["repo"] == "o/r"
    # The spec is uploaded as an unambiguous file-like object (io.BytesIO), NOT raw bytes — raw
    # bytes is a valid path type and huggingface_hub could misread it as a (huge) filesystem path.
    assert isinstance(uploaded["fileobj"], io.BytesIO)
    assert uploaded["fileobj"].getvalue() == big.encode()
    # The 100KB spec content is genuinely OUT of user_data (the whole point of the spill): not even
    # a fragment of it rides inline. (A brittle exact byte-size threshold would break as the bootstrap
    # script legitimately grows; assert the semantic invariant instead.)
    assert "v" * 1000 not in ud
    # And user_data stays under a generous, provider-aligned cap: cloud-init user_data limits run
    # ~16KB (AWS) to 64KB; the base64+heredoc framing inflates it, so 64KB is the ceiling the spill
    # threshold is chosen to keep us under. A 100KB inline spec alone would already blow this.
    assert len(ud) < 64_000
    # The caller's payload is untouched (the spill works on a copy).
    assert payload["job_spec_json"] == big
    assert "job_spec_in_hf" not in payload

    # A small spec rides inline with no HF upload.
    uploaded.clear()
    small = inst.build_user_data({**payload, "job_spec_json": "{}"}, image="img:latest")
    emb2 = json.loads(base64.b64decode(small.split("FLASH_PAYLOAD_EOF")[1].strip()))
    assert emb2["job_spec_json"] == "{}"
    assert "job_spec_in_hf" not in emb2
    assert uploaded == {}


def test_failmark_skips_when_worker_marker_exists(monkeypatch):
    """Bug: a container that fast-fails on a real user/config error uploads its own ok=false marker,
    then the host's ~5s liveness check fires fail() and would CLOBBER it with a retriable host
    marker (relabeling the user error as job_preempted). The host failmark must SKIP the write when
    a worker attempt marker already exists at the path (and stay conservative on a read error)."""
    import sys
    import types

    from flash.providers import _instance as inst

    payload = {"flash_arm": "lambda", "attempt": 0, "hf_prefix": "sft/x/seed0", "hf_repo": "o/r", "env": {}}

    def run_failmark(worker_marker_exists, read_raises=False):
        """Execute the embedded host _FAILMARK_PY against a fake HfApi + payload, return uploads."""
        uploaded = []

        class FakeApi:
            def __init__(self, token=None):
                pass

            def file_exists(self, *, repo_id, filename, repo_type):
                if read_raises:
                    raise RuntimeError("hf down")
                return worker_marker_exists

            def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type):
                uploaded.append(path_in_repo)

        monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))

        def fake_open(path, *a, **k):
            return io.StringIO(json.dumps(payload) if path == "/opt/flash/payload.json" else "")

        glb = {"json": json, "sys": types.SimpleNamespace(argv=["failmark.py", "boom"]), "open": fake_open}
        exec(inst._FAILMARK_PY, glb)  # controlled test of the embedded host script
        return uploaded

    # Worker already wrote its marker -> host must NOT clobber it.
    assert run_failmark(worker_marker_exists=True) == []
    # No worker marker (never-started container) -> host failmark IS written.
    assert run_failmark(worker_marker_exists=False) == ["sft/x/seed0/lambda_attempt0.json"]
    # HF read error -> conservative: skip the write (never risk clobbering).
    assert run_failmark(worker_marker_exists=False, read_raises=True) == []
