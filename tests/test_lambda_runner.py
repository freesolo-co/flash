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
import re
import time

import pytest

from flash.core.spec import JobSpec
from tests._helpers.source_snapshot import valid_source_snapshot

SOURCE_SNAPSHOT = valid_source_snapshot()


def _capsule_member(member: str) -> str:
    """The source of one member as it is SHIPPED, read back out of the built capsule archive.

    These host helpers run as standalone programs on a rented box, so their behaviour is only worth
    asserting against the bytes that actually travel. Reading the repository file instead would keep
    passing if the profile stopped shipping the module, or shipped a different one.
    """
    from flash.providers._lifecycle.instances.instance import INSTANCE_BOOTSTRAP_PROFILE
    from flash.runtime_capsule import build_capsule, read_capsule

    archive, _manifest = build_capsule(INSTANCE_BOOTSTRAP_PROFILE)
    _shipped_manifest, contents = read_capsule(archive)
    return contents[member].decode()


def _run_capsule_member(member: str, namespace: dict) -> dict:
    """RUN a shipped host program the way the capsule runs it, and return its namespace.

    The host helpers keep their work under a ``__main__`` guard so that merely importing one is
    inert, and the capsule reaches the work with ``runpy.run_module(run_name="__main__")``. A plain
    ``exec`` does not: ``__name__`` falls through to builtins rather than raising, so the guard is
    simply false and the program body never runs. Two tests here caught that by failing; a third
    asserted that no upload happened and would have passed for the wrong reason.

    So every site that means "run this program" goes through here, and nowhere else sets the name by
    hand. Library members (bootstrap_secrets) are a different case and are exec'd as imports.
    """
    namespace["__name__"] = "__main__"
    exec(_capsule_member(member), namespace)  # controlled run of the shipped host program
    return namespace


def _spec(gpu_type="A10", **gpu_kw) -> JobSpec:
    gpu = {"type": gpu_type, "max_wall_seconds": 3600, **gpu_kw}
    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "sft",
            "run_id": "flash-1700000000-abcd1234",
            "seed": 0,
            "train": {"epochs": 1, "hf_repo": "org/repo"},
            "gpu": gpu,
        }
    )


def _deadline_at() -> float:
    return time.time() + 3600.0


def _build_payload(builders, *args, **kwargs):
    if "deadline_at" not in kwargs:
        kwargs["deadline_at"] = _deadline_at()
    kwargs.setdefault("source_snapshot", SOURCE_SNAPSHOT)
    return builders.build_payload(*args, **kwargs)


def _launch(jobs, *args, **kwargs):
    if "deadline_at" not in kwargs:
        kwargs["deadline_at"] = _deadline_at()
    kwargs.setdefault("source_snapshot", SOURCE_SNAPSHOT)
    return jobs.launch_and_submit(*args, **kwargs)


def _submit(jobs, *args, **kwargs):
    if "deadline_at" not in kwargs:
        kwargs["deadline_at"] = _deadline_at()
    kwargs.setdefault("source_snapshot", SOURCE_SNAPSHOT)
    return jobs.submit_attempt_lambda(*args, **kwargs)


def _inst(gpu="A10", region="us-east-1", itype="gpu_1x_a10", price=1.29, disk_gb=None):
    from flash.providers.lambda_.jobs.builders import LambdaInstance

    return LambdaInstance(
        gpu=gpu,
        instance_type=itype,
        region=region,
        vram_gb=24,
        price_usd_hr=price,
        disk_gb=disk_gb,
    )


def _handle(started_ts=10_000.0, rate=1.29):
    from flash.providers.lambda_.jobs.builders import LambdaJobHandle

    return LambdaJobHandle(
        instance_id="i-9999",
        instance_type="gpu_1x_a10",
        region="us-east-1",
        name="flash-x-a0",
        gpu="A10",
        hourly_usd=rate,
        attempt=0,
        started_ts=started_ts,
    )


def _terminal_marker(*, ok: bool, retriable: bool = False, error: str = "") -> str:
    return json.dumps(
        {
            "attempt": 0,
            "error": error,
            "ok": ok,
            "retriable": retriable,
            "run_id": "flash-1700000000-abcd1234",
            "ts": 10_000.0,
        }
    )


# ---------------------------------------------------------------------------
# cloud-init user_data + bootstrap
# ---------------------------------------------------------------------------
def test_user_data_ships_payload_and_runs_worker_image(monkeypatch):
    from flash.providers.lambda_.jobs import builders

    monkeypatch.setenv("LAMBDA_API_KEY", "lk-supersecret")
    monkeypatch.setenv("HF_TOKEN", "hf-worker-token")
    deadline_at = time.time() + 3600
    payload = _build_payload(builders, _spec(), attempt=1, deadline_at=deadline_at)
    assert payload["phase"] == "sft"
    assert payload["attempt"] == 1
    assert payload["hf_prefix"] == "sft/flash-1700000000-abcd1234"
    assert payload["deadline_at"] == deadline_at
    assert payload["run_id"] == "flash-1700000000-abcd1234"
    assert payload["run_max_wall_seconds"] == 3600.0
    assert payload["run_created_at"] + payload["run_max_wall_seconds"] == deadline_at
    assert payload["hf_repo"] == "org/repo"
    # The worker env's HF_REPO is sourced from the run's [train] hf_repo (not an operator default).
    assert payload["env"]["HF_REPO"] == "org/repo"
    assert payload["source_snapshot"] == SOURCE_SNAPSHOT
    assert "code_prefix" not in payload

    script = builders.build_user_data(payload)
    # payload travels base64-encoded inside a quoted heredoc, byte-exact
    b64 = script.split("FLASH_PAYLOAD_EOF")[1].strip()
    assert json.loads(base64.b64decode(b64)) == payload
    # the bootstrap travels as a VERIFIED capsule, not as raw source text: the expected digest is
    # rendered by the control plane and checked before the first execution, so a payload rewritten
    # in flight fails closed instead of running.
    from flash.providers._lifecycle.instances.instance import _instance_capsule
    from flash.runtime_capsule import sha256_bytes

    capsule_b64, capsule_sha256 = _instance_capsule()
    assert capsule_sha256 in script
    assert sha256_bytes(base64.b64decode(capsule_b64)) == capsule_sha256
    assert "sha256sum -c" in script
    # no module crosses the boundary as a bare source heredoc any more.
    assert "cat > /opt/flash/bootstrap.py" not in script
    assert not re.search(r"cat > \S+\.py", script)
    # the worker still signals completion through metrics.json. That contract now lives INSIDE the
    # shipped bootstrap rather than in the launch text (the capsule is compressed, so a substring
    # check against the script would be vacuous), and the siblings it imports must ride with it.
    shipped = _capsule_member("bootstrap.py")
    assert "metrics.json" in shipped
    for sibling in ("bootstrap_secrets.py", "bootstrap_console.py", "bootstrap_pip.py"):
        assert sibling.removesuffix(".py") in shipped, sibling
    # runs the prebuilt WORKER_IMAGE via Docker with the GPU + the capsule bootstrap as the command
    from flash.providers._lifecycle.net.worker import WORKER_IMAGE

    assert WORKER_IMAGE in script
    assert "docker run -d" in script
    assert "--gpus all" in script
    assert "/root/flash/capsule.pyz bootstrap" in script
    # waits for docker + gpu before launching (cloud-init can beat them to ready)
    assert "waiting for docker+gpu" in script
    # every host-side polling/retry delay is capped by the canonical run deadline.
    assert "deadline_sleep" in script
    assert not any(line.strip().startswith("sleep ") for line in script.splitlines())
    # container output may contain private training data and must never be copied into host artifacts.
    assert "docker logs" not in script
    # the operator's Lambda key NEVER ships to the box (no instance-scoped key, teardown is
    # control-plane-side). The worker HF token IS carried — inside the base64 payload's env (like
    # RunPod's worker env), never interpolated raw into the shell.
    assert "lk-supersecret" not in script
    assert payload["env"]["HF_TOKEN"] == "hf-worker-token"


def test_user_data_skips_capacity_for_baked_image_default(monkeypatch):
    """build_user_data always uses the baked WORKER_IMAGE (no per-host stack install)."""
    from flash.providers.lambda_.jobs import builders

    payload = _build_payload(builders, _spec(), attempt=0)
    script = builders.build_user_data(payload)
    # No base training-stack pip install in the cloud-init (the image is baked); only the worker
    # container's own per-run extra_pip runs (inside _bootstrap, not the host script).
    assert "torch==2.10.0" not in script


def test_image_per_sm_selects_arch_tag():
    """Per-SM warmed images (PR #213) reach Lambda too: the GPU class always picks the matching -smXX
    tag for baked arches (so the worker's baked kernel cache matches the rented GPU's arch)."""
    from flash.providers._lifecycle.net.worker import WORKER_IMAGE
    from flash.providers.lambda_.jobs import builders

    # no GPU class -> flat base image (no arch to key a baked tag off)
    assert builders.lambda_image() == WORKER_IMAGE

    # a baked GPU class appends the arch tag by default, and it lands in the cloud-init
    assert builders.lambda_image("H100") == f"{WORKER_IMAGE}-sm90"  # H100 = sm90
    assert builders.lambda_image("A10") == f"{WORKER_IMAGE}-sm86"  # A10 = sm86
    payload = _build_payload(builders, _spec(gpu_type="H100"), attempt=0)
    script = builders.build_user_data(payload, gpu="H100")
    assert f"{WORKER_IMAGE}-sm90" in script


def _bootstrap_env(monkeypatch, phase="sft", rc=0, metrics=True, extra_pip=()):
    from flash.providers._lifecycle.bootstrapping import bootstrap as lb

    calls: list[str] = []
    markers: list[tuple[bool, str, bool]] = []

    def payload(path=lb.PAYLOAD_PATH):
        del path
        created_at = time.time()
        return {
            "hf_repo": "org/repo",
            "job_spec_json": "{}",
            "phase": phase,
            "run_id": "x",
            "seed": 0,
            "flash_arm": "lambda",
            "env": {},
            "extra_pip": list(extra_pip),
            "hf_prefix": "sft/x",
            "source_snapshot": SOURCE_SNAPSHOT,
            "deadline_at": created_at + 60.0,
            "run_created_at": created_at,
            "run_max_wall_seconds": 60.0,
            "attempt": 0,
        }

    monkeypatch.setattr(lb, "load_payload", payload)
    monkeypatch.setattr(lb, "fetch_code", lambda p: None)
    monkeypatch.setattr(lb, "run_mode", lambda p, e, m, d: (calls.append(m), rc)[1])
    monkeypatch.setattr(
        lb,
        "write_attempt_marker",
        lambda p, ok, error="", retriable=False: markers.append((ok, error, retriable)),
    )
    monkeypatch.setattr(lb.os.path, "exists", lambda p: metrics if "metrics.json" in p else False)
    return lb, calls, markers


def test_build_worker_env_exports_attempt():
    # the worker stamps every heartbeat with os.environ["ATTEMPT"], and worker_flagged_retriable
    # accepts only matching attempt and timestamp provenance. the shared instance bootstrap (Vast + Lambda)
    # must export ATTEMPT, or a worker on a nonzero retry defaults to attempt 0 and its current heartbeat is
    # rejected as mismatched, potentially losing valid retriable evidence.
    from flash.providers._lifecycle.bootstrapping import bootstrap as lb

    payload = {
        "phase": "sft",
        "seed": 0,
        "flash_arm": "vast",
        "attempt": 2,
        "run_id": "run-1",
        "job_spec_json": "{}",
        "source_snapshot": SOURCE_SNAPSHOT,
        "env": {
            "GITHUB_TOKEN": "ghp-private-vcs",
            "GIT_ASKPASS": "/tmp/payload-askpass",
        },
    }
    env = lb.build_worker_env(payload)
    assert env["ATTEMPT"] == "2"  # exported from the payload attempt, as a str (worker reads a str)
    assert "GITHUB_TOKEN" not in env
    assert "GIT_ASKPASS" not in env
    payload.pop("attempt")
    with pytest.raises(RuntimeError, match="attempt identity is invalid"):
        lb.build_worker_env(payload)


def test_bootstrap_train_success(monkeypatch):
    lb, calls, markers = _bootstrap_env(monkeypatch)
    assert lb.main() == 0
    assert calls == ["sft"]  # one fresh worker process
    assert markers == [(True, "", False)]  # success marker, not retriable


def test_bootstrap_fails_without_metrics(monkeypatch):
    lb, _calls, markers = _bootstrap_env(monkeypatch, metrics=False)
    # A genuine crash: no local metrics AND nothing on HF (stub keeps the check offline + deterministic).
    monkeypatch.setattr(lb, "remote_completion_confirmed", lambda p: False)
    assert lb.main() == 1
    ok, error, retriable = markers[0]
    assert not ok
    assert error.startswith("RuntimeError: train phase 'sft' produced no /tmp/metrics.json")
    # A genuine no-metrics crash (the worker never produced metrics) is a REAL failure, not infra:
    # it must NOT be flagged retriable (that would loop a deterministically-broken run).
    assert retriable is False


def test_bootstrap_missing_local_metrics_but_remote_confirmed_is_retriable(monkeypatch):
    """No local /tmp/metrics.json but the run IS complete on HF (DONE+metrics uploaded) — e.g. the
    idempotency replay hit a transient HF read. This is a SUCCEEDED run; the bootstrap must consult
    remote completion BEFORE the missing-local-file RuntimeError and surface a RETRIABLE marker so a
    fresh worker re-fetches the persisted metrics, never fail a confirmed-complete run."""
    lb, _calls, markers = _bootstrap_env(monkeypatch, metrics=False)
    monkeypatch.setattr(lb, "remote_completion_confirmed", lambda p: True)
    assert lb.main() == 1
    ok, error, retriable = markers[0]
    assert not ok
    assert error.startswith("RetriableBootstrapError: train phase 'sft' is complete on HF")
    assert retriable is True  # confirmed-complete -> reschedule to re-fetch, not a fatal job_failed


def test_bootstrap_fetch_code_failure_is_retriable(monkeypatch):
    # A transient HF blip fetching the run's OWN code (the control plane uploaded it before submit) is
    # infra-shaped, exactly like the already-wrapped fetch_spec_from_hf — it must surface as a
    # retriable marker so the run walks to a fresh host, NOT a fatal job_failed (the asymmetry = bug).
    lb, calls, markers = _bootstrap_env(monkeypatch)

    def boom(_payload):
        raise lb.RetriableBootstrapError("failed to fetch the pinned flash source snapshot")

    monkeypatch.setattr(lb, "fetch_code", boom)
    assert lb.main() == 1
    assert calls == []  # crashed before launching the worker subprocess
    ok, error, retriable = markers[0]
    assert not ok
    assert error == "RetriableBootstrapError: failed to fetch the pinned flash source snapshot"
    assert retriable is True


def test_bootstrap_sets_lambda_arm():
    """The shared bootstrap stamps FLASH_ARM from payload['flash_arm'] so the metrics record
    attributes the substrate (Lambda's build_payload sets it to 'lambda')."""
    from flash.providers._lifecycle.bootstrapping import bootstrap as lb

    env = lb.build_worker_env(
        {
            "job_spec_json": "{}",
            "phase": "sft",
            "seed": 0,
            "attempt": 0,
            "env": {},
            "flash_arm": "lambda",
            "run_id": "run-1",
            "source_snapshot": SOURCE_SNAPSHOT,
        }
    )
    assert env["FLASH_ARM"] == "lambda"
    # And Lambda's build_payload is what sets flash_arm='lambda'.
    from flash.providers.lambda_.jobs.builders import build_payload

    assert (
        build_payload(_spec(), 0, 0, source_snapshot=SOURCE_SNAPSHOT, deadline_at=_deadline_at())[
            "flash_arm"
        ]
        == "lambda"
    )


class _FakePipProc:
    """Popen stand-in for the extra_pip tee: an output stream plus one exit code."""

    def __init__(self, output: str = "", returncode: int = 0):
        self.stdout = io.StringIO(output)
        self._returncode = returncode

    def wait(self) -> int:
        return self._returncode


def _pip_payload(**extra) -> dict:
    created_at = time.time()
    return {
        "env": {
            "GITHUB_TOKEN": "ghp-secret",
            "GIT_ASKPASS": "/tmp/payload-askpass",
            "PYTHONPATH": "",
        },
        "extra_pip": ["git+https://github.com/example/some-env-pkg.git@abc123"],
        "deadline_at": created_at + 3600.0,
        "run_created_at": created_at,
        "run_max_wall_seconds": 3600.0,
        **extra,
    }


def _wire_pip(monkeypatch, results):
    """Patch Popen to replay ``results`` (output, rc) in order; returns the recorded calls."""
    from flash.providers._lifecycle.bootstrapping import bootstrap as lb

    calls = []
    queue = list(results)

    def fake_popen(cmd, *, env=None, **_kwargs):
        calls.append({"cmd": cmd, "env": env})
        output, rc = queue.pop(0) if queue else ("", 0)
        return _FakePipProc(output, rc)

    monkeypatch.setattr(lb.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(lb.time, "sleep", lambda _s: None)
    return lb, calls


def test_bootstrap_private_vcs_pip_uses_temporary_askpass(monkeypatch):
    import os
    from pathlib import Path

    from flash.providers._lifecycle.bootstrapping import bootstrap as lb

    calls = []
    askpass_paths = []

    def fake_popen(cmd, *, env=None, **_kwargs):
        askpass = Path(env["GIT_ASKPASS"])
        assert askpass.exists()
        assert os.access(askpass, os.X_OK)
        assert "ghp-secret" not in askpass.read_text()
        askpass_paths.append(askpass)
        calls.append({"cmd": cmd, "env": env})
        return _FakePipProc()

    monkeypatch.setenv("GITHUB_TOKEN", "operator-secret")
    monkeypatch.setenv("GIT_ASKPASS", "/tmp/operator-askpass")
    monkeypatch.setattr(lb.bootstrap_pip.subprocess, "Popen", fake_popen)
    lb.install_extra_pip(_pip_payload())

    assert len(calls) == 1
    env = calls[0]["env"]
    assert env["GITHUB_TOKEN"] == "ghp-secret"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert askpass_paths
    assert all(not path.exists() for path in askpass_paths)


def test_bootstrap_extra_pip_ignores_askpass_cleanup_errors(monkeypatch):
    from pathlib import Path

    from flash.providers._lifecycle.bootstrapping import bootstrap as lb

    askpass_paths = []

    def fake_popen(_cmd, *, env=None, **_kwargs):
        askpass_paths.append(Path(env["GIT_ASKPASS"]))
        return _FakePipProc()

    original_remove = lb.bootstrap_pip.os.remove

    def fake_remove(path):
        if Path(path) in askpass_paths:
            raise PermissionError("locked askpass helper")
        return original_remove(path)

    monkeypatch.setattr(lb.bootstrap_pip.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(lb.bootstrap_pip.os, "remove", fake_remove)

    try:
        lb.install_extra_pip(_pip_payload())
    finally:
        for askpass in askpass_paths:
            if askpass.exists():
                original_remove(askpass)

    assert askpass_paths


def test_bootstrap_extra_pip_retries_a_transient_index_failure(monkeypatch):
    # A PyPI/network blip is the one pre-worker network step that can fail a PAID run outright
    # (the adjacent HF fetches are already retriable). It must retry in place and then succeed.
    lb, calls = _wire_pip(
        monkeypatch,
        [
            (
                "WARNING: Retrying (Retry(total=4, connect=None)) after connection broken by ...\n",
                1,
            ),
            ("Successfully installed some-env-pkg-1.0\n", 0),
        ],
    )
    lb.install_extra_pip(_pip_payload())
    assert len(calls) == 2


def test_bootstrap_extra_pip_exhausted_index_failure_is_retriable(monkeypatch):
    # Still unreachable after the bounded in-place retries: infra-shaped, so the marker must carry
    # retriable=True (job_preempted -> fresh host) rather than the fail-fast job_failed.
    transient = ("ERROR: Could not install: HTTPSConnectionPool: Read timed out.\n", 1)
    lb, calls = _wire_pip(monkeypatch, [transient] * 4)
    with pytest.raises(lb.RetriableBootstrapError, match="could not reach the package index"):
        lb.install_extra_pip(_pip_payload())
    assert len(calls) == 4  # one attempt plus the three bounded retries


def test_bootstrap_extra_pip_resolution_error_stays_terminal(monkeypatch):
    # A bad package spec reaches the index fine and simply has no candidate. Retrying it would
    # re-rent a box to fail identically, so it must stay a terminal, non-retriable RuntimeError.
    lb, calls = _wire_pip(
        monkeypatch,
        [("ERROR: No matching distribution found for definitely-not-a-package\n", 1)],
    )
    with pytest.raises(RuntimeError, match="extra_pip install failed") as exc_info:
        lb.install_extra_pip(_pip_payload())
    assert not isinstance(exc_info.value, lb.RetriableBootstrapError)
    assert len(calls) == 1  # fails fast, never walks the retry ladder


def test_bootstrap_extra_pip_build_failure_outranks_earlier_transient_text(monkeypatch):
    # a wheel build failure can only happen after pip already reached the index, so it must outrank
    # an earlier transient warning in the same captured tail -- retrying would just rent another
    # host to fail identically forever.
    lb, calls = _wire_pip(
        monkeypatch,
        [
            (
                (
                    "WARNING: Retrying (Retry(total=4, connect=None)) after connection broken "
                    "by ...\n"
                    "ERROR: Failed building wheel for some-env-pkg\n"
                    "error: subprocess-exited-with-error\n"
                ),
                1,
            ),
        ],
    )
    with pytest.raises(RuntimeError, match="extra_pip install failed") as exc_info:
        lb.install_extra_pip(_pip_payload())
    assert not isinstance(exc_info.value, lb.RetriableBootstrapError)
    assert len(calls) == 1  # fails fast, never walks the retry ladder


def test_bootstrap_extra_pip_retries_a_network_interrupted_vcs_clone(monkeypatch):
    # pip shells out to `git clone` for a VCS requirement and reports ANY child failure with the
    # generic "subprocess-exited-with-error" marker, so a connection reset mid-clone carries that
    # marker beside the network shape. Classifying the bare marker terminal would fail a paid run
    # on exactly the blip this ladder exists to absorb.
    lb, calls = _wire_pip(
        monkeypatch,
        [
            (
                (
                    "Collecting some-pkg from git+https://github.com/org/repo\n"
                    "  Running command git clone --filter=blob:none -q "
                    "https://github.com/org/repo\n"
                    "  fatal: unable to access 'https://github.com/org/repo': "
                    "Connection reset by peer\n"
                    "  error: subprocess-exited-with-error\n"
                ),
                1,
            ),
            ("Successfully installed some-pkg\n", 0),
        ],
    )
    lb.install_extra_pip(_pip_payload())
    assert len(calls) == 2  # retried the clone instead of failing the run as a user error


def test_bootstrap_extra_pip_retries_a_vcs_clone_rejected_by_an_http_blip(monkeypatch):
    # a VCS pin fails through git, whose proxy/rate-limit blips carry git's own phrasing and none
    # of the urllib shapes the rest of the pattern names. Without git's form the classifier reads
    # a 502 as a bad spec and fails a paid run on a blip the ladder exists to absorb.
    lb, calls = _wire_pip(
        monkeypatch,
        [
            (
                (
                    "  Running command git clone --filter=blob:none -q "
                    "https://github.com/org/repo\n"
                    "  fatal: unable to access 'https://github.com/org/repo/': "
                    "The requested URL returned error: 502\n"
                    "  error: subprocess-exited-with-error\n"
                ),
                1,
            ),
            ("Successfully installed some-pkg\n", 0),
        ],
    )
    lb.install_extra_pip(_pip_payload())
    assert len(calls) == 2


def test_bootstrap_extra_pip_retries_a_vcs_clone_that_cannot_resolve_the_host(monkeypatch):
    # the other half of git's own vocabulary: urllib says "temporary failure in name resolution",
    # git says "could not resolve host". A DNS blip is the same infra failure either way, and
    # matching only urllib's wording fails a paid run on a VCS pin during a resolver outage.
    lb, calls = _wire_pip(
        monkeypatch,
        [
            (
                (
                    "  Running command git clone --filter=blob:none -q "
                    "https://github.com/org/repo\n"
                    "  fatal: unable to access 'https://github.com/org/repo/': "
                    "Could not resolve host: github.com\n"
                    "  error: subprocess-exited-with-error\n"
                ),
                1,
            ),
            ("Successfully installed some-pkg\n", 0),
        ],
    )
    lb.install_extra_pip(_pip_payload())
    assert len(calls) == 2


def test_bootstrap_extra_pip_retries_an_index_outage_that_ends_in_the_no_match_footer(monkeypatch):
    """An unreachable index produces the SAME footer a typo'd package name does.

    pip that cannot reach the index sees no candidate versions, so it prints its retry warnings and
    then finishes with "could not find a version" / "no matching distribution". Those footers alone
    therefore cannot prove a deterministic bad spec, and treating them as terminal fails a paid run
    on exactly the outage this ladder exists to absorb, without ever making a second attempt.
    """
    lb, calls = _wire_pip(
        monkeypatch,
        [
            (
                (
                    "WARNING: Retrying (Retry(total=4, connect=None)) after connection broken by "
                    "NewConnectionError\n"
                    "ERROR: Could not find a version that satisfies the requirement requests "
                    "(from versions: none)\n"
                    "ERROR: No matching distribution found for requests\n"
                ),
                1,
            ),
            ("Successfully installed requests\n", 0),
        ],
    )
    lb.install_extra_pip(_pip_payload())
    assert len(calls) == 2


def test_bootstrap_extra_pip_build_failure_still_outranks_a_recovered_blip(monkeypatch):
    """The counterpart bound: loosening the footer must not loosen the precedence rule.

    A wheel that failed to build is only reachable AFTER pip downloaded real content, so it names a
    deterministic cause no matter what warning preceded it. It must keep absolute precedence over a
    transient marker pip already recovered from earlier in the same attempt, or one early
    "Retrying (Retry(" makes a permanent failure walk the whole ladder for nothing.
    """
    lb, calls = _wire_pip(
        monkeypatch,
        [
            (
                (
                    "WARNING: Retrying (Retry(total=4)) after connection broken by "
                    "NewConnectionError\n"
                    "Collecting numpy\n"
                    "ERROR: No matching distribution found for numpy\n"
                    "ERROR: Failed building wheel for numpy\n"
                ),
                1,
            )
        ],
    )
    with pytest.raises(RuntimeError, match="extra_pip install failed") as exc_info:
        lb.install_extra_pip(_pip_payload())
    assert not isinstance(exc_info.value, lb.RetriableBootstrapError)
    assert len(calls) == 1  # the build failure decides it; no ladder


def test_bootstrap_extra_pip_vcs_clone_rejected_by_a_404_still_fails_fast(monkeypatch):
    # the counterpart bound: git reports a missing repo or an unauthorized private pin in the same
    # sentence as the blip above. Only 429/5xx may retry, or a typo'd pin re-rents a box three
    # times to fail identically.
    lb, calls = _wire_pip(
        monkeypatch,
        [
            (
                (
                    "  Running command git clone --filter=blob:none -q "
                    "https://github.com/org/typo\n"
                    "  fatal: unable to access 'https://github.com/org/typo/': "
                    "The requested URL returned error: 404\n"
                    "  error: subprocess-exited-with-error\n"
                ),
                1,
            )
        ],
    )
    with pytest.raises(RuntimeError, match="extra_pip install failed") as exc_info:
        lb.install_extra_pip(_pip_payload())
    assert not isinstance(exc_info.value, lb.RetriableBootstrapError)
    assert len(calls) == 1


def test_bootstrap_extra_pip_survives_undecodable_bytes_from_a_build_child(monkeypatch):
    # a build or VCS child can emit bytes invalid under the worker's locale. text=True decodes
    # strictly, so iterating the stream raised UnicodeDecodeError before the exit status was ever
    # read, failing a paid run whose install had actually succeeded.
    from flash.providers._lifecycle.bootstrapping import bootstrap as lb

    seen_kwargs = {}

    class _StrictProc:
        """Decodes its bytes the way Popen would, honouring the errors policy it was given."""

        def __init__(self, raw, rc, errors):
            self._text = raw.decode("utf-8", errors=errors or "strict")
            self._rc = rc

        @property
        def stdout(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def __iter__(self):
            return iter(self._text.splitlines(keepends=True))

        def wait(self):
            return self._rc

    def fake_popen(cmd, *, env=None, errors=None, **_kwargs):
        seen_kwargs["errors"] = errors
        return _StrictProc(
            b"Collecting some-pkg\n\xff\xfe bad bytes\nSuccessfully installed\n", 0, errors
        )

    monkeypatch.setattr(lb.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(lb.time, "sleep", lambda _s: None)

    lb.install_extra_pip(_pip_payload())  # would raise UnicodeDecodeError under strict decoding
    assert seen_kwargs["errors"] == "replace"


def test_bootstrap_extra_pip_transient_only_text_still_retries(monkeypatch):
    # regression guard: transient text alone (no terminal-shape text anywhere in the tail) must
    # still walk the full retry ladder and end retriable, unchanged by the terminal-precedence check.
    transient = (
        "WARNING: Retrying (Retry(total=4, connect=None)) after connection broken by ...\n",
        1,
    )
    lb, calls = _wire_pip(monkeypatch, [transient] * 4)
    with pytest.raises(lb.RetriableBootstrapError, match="could not reach the package index"):
        lb.install_extra_pip(_pip_payload())
    assert len(calls) == 4  # one attempt plus the three bounded retries


def test_main_marks_exhausted_extra_pip_index_failure_retriable(monkeypatch):
    # End to end through main(): the marker the poller reads must say retriable, so the attempt is
    # classified job_preempted (walk to a fresh host) instead of job_failed (fail the paid run).
    lb, calls, markers = _bootstrap_env(monkeypatch, extra_pip=["some-env-pkg"])
    monkeypatch.setattr(lb.subprocess, "Popen", lambda *_a, **_k: _FakePipProc("read timed out", 1))
    monkeypatch.setattr(lb.time, "sleep", lambda _s: None)
    assert lb.main() == 1
    assert calls == []  # crashed before launching the worker subprocess
    ok, error, retriable = markers[0]
    assert not ok
    assert error.startswith("RetriableBootstrapError: extra_pip install could not reach")
    assert retriable is True


def test_bootstrap_extra_pip_retry_sleep_never_outlives_the_deadline(monkeypatch):
    lb, _calls = _wire_pip(monkeypatch, [("connection reset by peer\n", 1)] * 4)
    slept = []
    monkeypatch.setattr(lb.time, "sleep", slept.append)
    monkeypatch.setattr(lb.time, "time", lambda: 1_000.0)
    with pytest.raises(lb.RetriableBootstrapError):
        lb.install_extra_pip(
            _pip_payload(deadline_at=1_002.0, run_created_at=1_000.0, run_max_wall_seconds=2.0)
        )
    # the ladder is 3/9/27s but only 2s of paid wall remain, so no sleep may exceed it
    assert slept
    assert max(slept) <= 2.0


def test_bootstrap_extra_pip_survives_an_unwritable_console(monkeypatch):
    """A closed console must not end the install: pip's own exit status decides the outcome.

    The tee replaced an inherited-stdio ``subprocess.run``, which never made the install depend on
    replaying each line. If a broken log collector can raise out of the drain, the function exits
    without waiting for pip, deletes the askpass helper while that child is still authenticating,
    and reports the console failure as a terminal install error on a paid box."""
    waited = []

    class _ClosedConsoleProc(_FakePipProc):
        def wait(self) -> int:
            waited.append(True)
            return super().wait()

    from flash.providers._lifecycle.bootstrapping import bootstrap as lb

    queue = [("Collecting some-env-pkg\n", 0)]

    def fake_popen(cmd, *, env=None, **_kwargs):
        output, rc = queue.pop(0)
        return _ClosedConsoleProc(output, rc)

    monkeypatch.setattr(lb.subprocess, "Popen", fake_popen)

    def closed_stream_print(*_a, **_kw):
        raise ValueError("I/O operation on closed file")

    monkeypatch.setattr("builtins.print", closed_stream_print)
    lb.install_extra_pip(_pip_payload())  # pip exited 0, so the install SUCCEEDED
    assert waited == [True]  # and pip was reaped, not left running behind the failed console


def test_bootstrap_extra_pip_unwritable_console_still_reports_pip_status(monkeypatch):
    """Same broken console, failing pip: the error names pip's status, not the console's."""
    from flash.providers._lifecycle.bootstrapping import bootstrap as lb

    queue = [("ERROR: No matching distribution found for some-env-pkg\n", 1)]
    monkeypatch.setattr(
        lb.subprocess, "Popen", lambda cmd, *, env=None, **_k: _FakePipProc(*queue.pop(0))
    )
    monkeypatch.setattr(
        "builtins.print", lambda *_a, **_kw: (_ for _ in ()).throw(BrokenPipeError("closed"))
    )
    with pytest.raises(RuntimeError, match="extra_pip install failed: pip exited 1"):
        lb.install_extra_pip(_pip_payload())


def test_bootstrap_extra_pip_backoff_leaves_time_to_run_the_retry(monkeypatch):
    """A clamped backoff must reserve a slice for the attempt it just announced.

    Clamping only to the remaining wall sleeps the entire window, so the retry that was announced
    never issues: the next iteration fails ``require_deadline_at`` (or the watchdog kills the
    process) with the ladder's remaining rungs unused on a run that was still payable."""
    lb, _calls = _wire_pip(monkeypatch, [("connection reset by peer\n", 1)] * 4)
    slept = []
    monkeypatch.setattr(lb.time, "sleep", slept.append)
    monkeypatch.setattr(lb.time, "time", lambda: 1_000.0)
    with pytest.raises(lb.RetriableBootstrapError):
        lb.install_extra_pip(
            _pip_payload(deadline_at=1_008.0, run_created_at=1_000.0, run_max_wall_seconds=8.0)
        )
    # 8s of wall against the 3/9/27s ladder: the 9s and 27s rungs clamp, and each must leave
    # nonzero wall behind for the attempt that follows it.
    assert slept
    assert max(slept) < 8.0


def test_bootstrap_promotes_attempt_to_env_for_heartbeat_gating():
    # The instance bootstrap must stamp ATTEMPT into the worker env (RunPod does it in jobs.py) — the
    # worker reads it into every heartbeat, and the poller's stale-heartbeat rejection is dead without
    # it (a prior attempt's leftover heartbeat would disarm the new attempt's fast failover).
    from flash.providers._lifecycle.bootstrapping import bootstrap as lb
    from flash.providers.lambda_.jobs.builders import build_payload

    base = {
        "job_spec_json": "{}",
        "phase": "sft",
        "seed": 0,
        "env": {},
        "flash_arm": "lambda",
        "run_id": "run-1",
        "source_snapshot": SOURCE_SNAPSHOT,
    }
    assert lb.build_worker_env({**base, "attempt": 3})["ATTEMPT"] == "3"
    assert lb.build_worker_env({**base, "attempt": 0})["ATTEMPT"] == "0"
    with pytest.raises(RuntimeError, match="attempt identity is invalid"):
        lb.build_worker_env(base)
    # And the producer end actually carries the launched attempt into the payload bootstrap reads.
    assert (
        build_payload(
            _spec(),
            attempt=2,
            source_snapshot=SOURCE_SNAPSHOT,
            deadline_at=_deadline_at(),
        )["attempt"]
        == 2
    )


def test_bootstrap_fetch_code_uses_pinned_verified_archive(monkeypatch, tmp_path):
    import huggingface_hub

    from flash.providers._lifecycle.bootstrapping import bootstrap as lb

    monkeypatch.setattr(lb, "CODE_ROOT", str(tmp_path))
    archive_path = tmp_path / "source.zip"
    archive_path.write_bytes(b"archive")
    events = []

    def fake_download(**kwargs):
        events.append(("download", kwargs))
        return str(archive_path)

    def fake_materialize(path, descriptor, destination):
        events.append(("verify-materialize", path, descriptor.to_dict(), destination))

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(
        lb._source_snapshot,
        "materialize_verified_archive_file",
        fake_materialize,
    )
    created_at = time.time()

    lb.fetch_code(
        {
            "hf_repo": "org/repo",
            "source_snapshot": SOURCE_SNAPSHOT,
            "run_id": "run-1",
            "attempt": 2,
            "env": {"HF_TOKEN": "tok"},
            "deadline_at": created_at + 3600.0,
            "run_created_at": created_at,
            "run_max_wall_seconds": 3600.0,
        }
    )

    assert events[0] == (
        "download",
        {
            "repo_id": "org/repo",
            "repo_type": "dataset",
            "filename": SOURCE_SNAPSHOT["archive_path"],
            "revision": SOURCE_SNAPSHOT["revision"],
            "token": "tok",
        },
    )
    assert events[1] == (
        "verify-materialize",
        str(archive_path),
        SOURCE_SNAPSHOT,
        str(tmp_path / "run-1-attempt-2"),
    )


# ---------------------------------------------------------------------------
# launch_and_submit: capacity (region) walk
# ---------------------------------------------------------------------------
def test_launch_walks_regions_on_capacity_rejection(monkeypatch):
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    attempts = []

    def fake_launch(
        *, region_name, instance_type_name, ssh_key_names, name, user_data, file_system_names=None
    ):
        attempts.append(region_name)
        if len(attempts) < 3:
            raise lambda_api.LambdaApiError("PUT /asks/1/ -> HTTP 400: insufficient-capacity")
        return "i-4242"

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)
    insts = [_inst(region=r) for r in ("us-east-1", "us-west-1", "us-west-2")]
    h = _launch(jobs, _spec(), instances=insts, attempt=2)
    assert attempts == ["us-east-1", "us-west-1", "us-west-2"]
    assert h.instance_id == "i-4242"
    assert h.region == "us-west-2"
    assert h.gpu == "A10"
    assert h.name == "flash-1700000000-abcd1234-a2"


def test_launch_refreshes_capacity_once_when_all_taken(monkeypatch):
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    created = []

    def fake_launch(*, region_name, **kw):
        if region_name != "us-fresh-1":
            raise lambda_api.LambdaApiError("PUT /asks/1/ -> HTTP 400: no capacity")
        created.append(region_name)
        return "i-7"

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)
    monkeypatch.setattr(
        jobs,
        "usable_instances",
        lambda gpu, force=False, gpu_count=1: [_inst(region="us-fresh-1")],
    )
    h = _launch(jobs, _spec(), instances=[_inst(region="us-east-1")], attempt=0)
    assert created == ["us-fresh-1"]
    assert h.instance_id == "i-7"


def test_launch_refuses_primary_creation_below_minimum_deadline_allowance(monkeypatch):
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(jobs.time, "time", lambda: 100.0)
    launched = []
    monkeypatch.setattr(
        lambda_api,
        "launch_instance",
        lambda **_kwargs: launched.append(True) or "i-1",
    )

    with pytest.raises(RuntimeError, match="60-second minimum provider allowance"):
        _launch(jobs, _spec(), instances=[_inst()], attempt=0, deadline_at=159.0)

    assert launched == []


def test_create_filesystem_posts_once_without_retries(monkeypatch):
    from flash.providers.lambda_.client import api as lambda_api

    calls = []

    def request(path, **kwargs):
        calls.append((path, kwargs))
        return {"data": {"mount_point": "/lambda/nfs/cache"}}

    monkeypatch.setattr(lambda_api, "request_with_retries", request)

    deadline_at = time.time() + 120
    created = lambda_api.create_filesystem("cache", "us-east-1", deadline_at=deadline_at)

    assert created["mount_point"] == "/lambda/nfs/cache"
    assert calls == [
        (
            "/filesystems",
            {
                "method": "POST",
                "body": {"name": "cache", "region": "us-east-1"},
                "retries": 0,
                "deadline_at": deadline_at,
            },
        )
    ]


def test_filesystem_listing_caps_request_and_retry_sleep_at_deadline(monkeypatch):
    import urllib.error

    from flash.providers._lifecycle.net import deadline as _deadline
    from flash.providers._lifecycle.net import http as _http
    from flash.providers.lambda_.client import api as lambda_api

    clock = {"now": 100.0}
    calls = []
    monkeypatch.setenv("LAMBDA_API_KEY", "test-key")
    sleeps = []
    monkeypatch.setattr(_deadline.time, "time", lambda: clock["now"])
    monkeypatch.setattr(_http.random, "uniform", lambda _low, _high: 1.0)

    def request(_target, **kwargs):
        calls.append(kwargs["timeout"])
        raise urllib.error.URLError("provider detail")

    def sleep(delay):
        sleeps.append(delay)
        clock["now"] += delay

    monkeypatch.setattr(lambda_api._CLIENT, "request", request)
    monkeypatch.setattr(_http.time, "sleep", sleep)

    with pytest.raises(lambda_api.LambdaApiError, match="deadline exceeded") as caught:
        lambda_api.list_filesystems(deadline_at=101.0)

    assert "provider detail" not in str(caught.value)
    assert calls == [1.0]
    assert sleeps == [1.0]


def test_create_filesystem_rejects_deadline_below_minimum(monkeypatch):
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(lambda_api.time, "time", lambda: 100.0)
    calls = []
    monkeypatch.setattr(
        lambda_api,
        "request_with_retries",
        lambda *_args, **_kwargs: calls.append(True),
    )

    with pytest.raises(RuntimeError, match="60-second minimum provider allowance"):
        lambda_api.create_filesystem("cache", "us-east-1", deadline_at=159.0)

    assert calls == []


@pytest.mark.parametrize("matches", [0, 2])
def test_ambiguous_filesystem_create_fails_closed_without_second_post(monkeypatch, matches):
    from flash.providers.lambda_.client import api as lambda_api

    posts = []
    listings = {"count": 0}

    def create(*_args, **_kwargs):
        posts.append(True)
        raise lambda_api.LambdaApiError("ambiguous create failure")

    def listing(*, deadline_at=None):
        assert deadline_at is not None
        listings["count"] += 1
        if listings["count"] == 1:
            return []
        return [
            {
                "name": "cache",
                "mount_point": f"/lambda/nfs/cache-{index}",
                "region": {"name": "us-east-1"},
            }
            for index in range(matches)
        ]

    monkeypatch.setattr(lambda_api, "create_filesystem", create)
    monkeypatch.setattr(lambda_api, "list_filesystems", listing)

    with pytest.raises(lambda_api.LambdaApiError, match="could not be reconciled"):
        lambda_api.ensure_filesystem("cache", "us-east-1", deadline_at=time.time() + 120)

    assert posts == [True]
    assert listings["count"] == 2


def test_lambda_failure_detail_is_bounded_and_redacts_credentials(monkeypatch):
    import flash.providers.lambda_.jobs as jobs

    monkeypatch.setenv("HF_TOKEN", "hf-private-token")

    def reader(_repo, path, *_args, **_kwargs):
        if path.endswith("_boot.log"):
            return lambda force=False: "boot failed Authorization: Bearer hf-private-token"
        return lambda force=False: "worker failed token=hf-private-token"

    monkeypatch.setattr(jobs, "_make_hf_file_reader", reader)

    detail = jobs._failure_detail(
        "org/repo",
        "sft/run",
        "sft",
        {"error": "RuntimeError: worker failed"},
        1,
    )

    assert "RuntimeError: worker failed" in detail
    assert "error_sft_attempt1.txt" in detail
    assert "lambda_attempt1_boot.log" in detail
    assert "hf-private-token" not in detail
    assert "<redacted>" in detail


def test_lambda_cleanup_logs_suppress_provider_detail(monkeypatch):
    from flash.providers.lambda_.client import api as lambda_api

    warnings = []
    monkeypatch.setattr(lambda_api.logger, "warning", lambda *args: warnings.append(args))
    monkeypatch.setattr(
        lambda_api,
        "request_with_retries",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("provider body secret")),
    )

    assert lambda_api.delete_filesystem("fs-1") is False
    assert lambda_api.terminate_instances(["i-1"]) == []
    assert all("provider body secret" not in " ".join(map(str, args)) for args in warnings)


def test_ambiguous_filesystem_create_adopts_single_exact_match(monkeypatch):
    from flash.providers.lambda_.client import api as lambda_api

    posts = []
    listings = {"count": 0}

    def create(*_args, **_kwargs):
        posts.append(True)
        raise lambda_api.LambdaApiError("ambiguous create failure")

    def listing(*, deadline_at=None):
        assert deadline_at is not None
        listings["count"] += 1
        if listings["count"] == 1:
            return []
        return [
            {
                "name": "cache",
                "mount_point": "/mnt/adopted-cache",
                "region": {"name": "us-east-1"},
            },
            {
                "name": "cache",
                "mount_point": "/mnt/wrong-region",
                "region": {"name": "us-west-2"},
            },
        ]

    monkeypatch.setattr(lambda_api, "create_filesystem", create)
    monkeypatch.setattr(lambda_api, "list_filesystems", listing)

    mount = lambda_api.ensure_filesystem("cache", "us-east-1", deadline_at=time.time() + 120)

    assert mount == "/mnt/adopted-cache"
    assert posts == [True]
    assert listings["count"] == 2


# ---------------------------------------------------------------------------
# gpu.disk_gb: Lambda sells a FIXED disk per instance type (no launch-time parameter)
# ---------------------------------------------------------------------------
def test_instance_type_disk_gb_reads_catalog_storage_or_reports_unknown():
    from flash.providers.lambda_.client.gpus import instance_type_disk_gb

    catalog = {
        "gpu_1x_a10": {"instance_type": {"specs": {"gpus": 1, "storage_gib": 512}}},
        "gpu_1x_a100": {"instance_type": {"specs": {"gpus": 1}}},  # storage not reported
        "gpu_8x_h100_sxm5": {"instance_type": {}},
    }
    assert instance_type_disk_gb(catalog, "gpu_1x_a10") == 512.0
    # unknown must be None, never 0: a caller may not refuse a shape the catalog cannot measure
    assert instance_type_disk_gb(catalog, "gpu_1x_a100") is None
    assert instance_type_disk_gb(catalog, "gpu_8x_h100_sxm5") is None
    assert instance_type_disk_gb(catalog, "gpu_1x_nope") is None
    assert instance_type_disk_gb(None, "gpu_1x_a10") is None


def test_usable_instances_carries_the_sku_disk(monkeypatch):
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(
        lambda_api,
        "list_instance_types",
        lambda *a, **k: {"gpu_1x_a10": {"instance_type": {"specs": {"storage_gib": 512}}}},
    )
    monkeypatch.setattr(lambda_api, "regions_with_capacity", lambda *a, **k: ["us-east-1"])
    monkeypatch.setattr("flash.providers.lambda_.client.pricing.hourly_rate", lambda *a, **k: 1.29)
    assert jobs.usable_instances("A10")[0].disk_gb == 512.0


def test_launch_refuses_an_instance_type_below_the_run_disk_floor(monkeypatch):
    # Vast sizes the volume at create and RunPod raises containerDiskInGb; Lambda can do neither, so
    # a run whose disk floor exceeds the SKU's fixed disk must be refused BEFORE the box is rented
    # (it would otherwise be paid for and then die mid-setup).
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.core.base import UnsupportedGpuError
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    launched = []
    monkeypatch.setattr(lambda_api, "launch_instance", lambda **_k: launched.append(True) or "i-1")

    with pytest.raises(UnsupportedGpuError, match=r"gpu_1x_a10 \(512 GB\).*required 800 GB"):
        _launch(
            jobs,
            _spec(disk_gb=800),
            instances=[_inst(disk_gb=512.0)],
            attempt=0,
        )

    assert launched == []


def test_launch_accepts_a_disk_capable_or_unmeasured_instance_type(monkeypatch):
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(lambda_api, "launch_instance", lambda **_k: "i-1")

    assert _launch(jobs, _spec(disk_gb=200), instances=[_inst(disk_gb=512.0)], attempt=0)
    # an unreported SKU disk is not a proven miss, so it must not block the launch
    assert _launch(jobs, _spec(disk_gb=800), instances=[_inst()], attempt=0)


def test_launch_refuses_a_disk_undersized_refreshed_candidate(monkeypatch):
    """The disk gate runs once on the initial candidate list before the walk starts; a candidate
    that only shows up via _refresh_launch_candidates (e.g. an unmeasured SKU whose refreshed
    catalog entry proves it undersized) must still be refused before it can reach launch_instance."""
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.core.base import UnsupportedGpuError
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    launched = []

    def fake_launch(*, region_name, **_kwargs):
        launched.append(region_name)
        raise lambda_api.LambdaApiError("PUT /asks/1/ -> HTTP 400: insufficient-capacity")

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)
    monkeypatch.setattr(
        jobs,
        "usable_instances",
        lambda gpu, force=False, gpu_count=1: [_inst(region="us-fresh-1", disk_gb=100.0)],
    )

    with pytest.raises(UnsupportedGpuError, match=r"gpu_1x_a10 \(100 GB\).*required 800 GB"):
        _launch(
            jobs,
            _spec(disk_gb=800),
            # unmeasured disk passes the pre-loop gate untouched; only the refresh reveals it undersized.
            instances=[_inst(region="us-east-1", disk_gb=None)],
            attempt=0,
        )

    assert launched == [
        "us-east-1"
    ]  # the undersized refreshed candidate never reached launch_instance


def test_live_candidates_drop_skus_that_cannot_hold_the_run_disk(monkeypatch):
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.core.base import AllocationConstraints
    from flash.providers.lambda_.client import api as lambda_api
    from flash.providers.lambda_.execution.provider import PROVIDER

    monkeypatch.setattr(
        lambda_api,
        "list_instance_types",
        lambda *a, **k: {"gpu_1x_a10": {"instance_type": {"specs": {"storage_gib": 512}}}},
    )
    monkeypatch.setattr(jobs, "usable_instances", lambda *a, **k: [_inst(disk_gb=512.0)])

    fits = PROVIDER.live_candidates(24, AllocationConstraints(disk_gb=200, gpu_type="A10"))
    assert [c.gpu for c in fits] == ["A10"]
    # the allocator must never hand the runner a Lambda class it could not rent for this run
    assert PROVIDER.live_candidates(24, AllocationConstraints(disk_gb=800, gpu_type="A10")) == []


def test_launch_never_rents_an_undersized_sku_from_a_mixed_candidate_list(monkeypatch):
    """A capable SKU elsewhere in the list must not license renting an undersized one.

    The gate used to answer "does SOME candidate fit?" and return on the first capable entry, but
    the walk pops candidates in order, so an undersized shape ahead of the capable one was still
    rented -- exactly the paid box the pre-rental floor exists to prevent.
    """
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    launched: list[str] = []

    def fake_launch(*, region_name, **_kwargs):
        launched.append(region_name)
        return "i-1"

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)

    assert _launch(
        jobs,
        _spec(disk_gb=800),
        instances=[
            _inst(region="us-small-1", disk_gb=100.0),  # provably undersized, listed FIRST
            _inst(region="us-big-1", disk_gb=1024.0),
        ],
        attempt=0,
    )
    assert launched == ["us-big-1"]  # the undersized region was never rented


def test_post_launch_interrupt_does_not_layer_a_run_label_reap_on_exact_cleanup(monkeypatch):
    """An interrupt after launch terminates that exact instance; the coarse reap must stand down.

    terminate_run_instances(run_id) kills every instance sharing the run label, so firing it on top
    of an exact cleanup would destroy other concurrently-launched seeds of the same run.
    """
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(lambda_api, "launch_instance", lambda **_k: "i-1")
    exact: list[str] = []
    reaped: list[str] = []
    monkeypatch.setattr(lambda_api, "terminate_instance_confirmed", lambda i: exact.append(i))
    monkeypatch.setattr(jobs, "terminate_run_instances", lambda run_id: reaped.append(run_id) or [])

    def interrupt(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(jobs, "_lambda_job_handle", interrupt)

    with pytest.raises(KeyboardInterrupt):
        _launch(jobs, _spec(), instances=[_inst()], attempt=0)

    assert exact == ["i-1"]  # the rented box was terminated by id
    assert reaped == []  # ... so the run-wide reap must NOT also fire


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_interrupt_after_publication_returns_terminates_only_this_instance(
    monkeypatch, interrupt_type
):
    """An interrupt landing AFTER the rent helper returns must not reap the run label.

    Handing the box over spans a statement boundary: _rent_instance returns, and only then does
    the caller hold a handle any teardown path can name. An interrupt in between used to leave the
    guard armed with no id, so the outer handler reaped by run label and terminated every other
    concurrently-launched seed sharing it. The guard now holds the id from the create onward, so
    this window cleans up exactly one instance.
    """
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(lambda_api, "launch_instance", lambda **_kwargs: "i-4242")

    # fire in the gap the finding names: the helper has RETURNED (so its own guarded frame is gone
    # and nothing stamped exact cleanup) but the caller does not hold the handle yet. Wrapping the
    # helper reproduces exactly that statement boundary; raising inside it would instead be caught
    # by its own handler, which is a different, already-covered window.
    real_rent = jobs._rent_instance

    def interrupt_after_rent(*args, **kwargs):
        real_rent(*args, **kwargs)
        raise interrupt_type("interrupted after the rent helper returned")

    monkeypatch.setattr(jobs, "_rent_instance", interrupt_after_rent)

    terminated: list[str] = []
    monkeypatch.setattr(
        lambda_api, "terminate_instance_confirmed", lambda iid: terminated.append(iid)
    )
    reaped: list[str] = []
    monkeypatch.setattr(jobs, "terminate_run_instances", lambda run_id: reaped.append(run_id) or [])

    with pytest.raises(interrupt_type):
        _launch(jobs, _spec(), instances=[_inst()], attempt=0)

    assert terminated == ["i-4242"]  # the box this seed rented is cleaned up by id
    assert reaped == []  # and no concurrent seed sharing the run label is touched


def test_launch_raises_when_no_capacity(monkeypatch):
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(
        lambda_api,
        "launch_instance",
        lambda **k: (_ for _ in ()).throw(
            lambda_api.LambdaApiError("PUT /asks/1/ -> HTTP 400: no capacity")
        ),
    )
    monkeypatch.setattr(jobs, "usable_instances", lambda gpu, force=False, gpu_count=1: [])
    with pytest.raises(lambda_api.LambdaApiError, match="no capacity"):
        _launch(jobs, _spec(), instances=[_inst()], attempt=0)
    with pytest.raises(lambda_api.LambdaApiError, match="no Lambda capacity"):
        _launch(jobs, _spec(), instances=[], attempt=0)


def test_resolve_ssh_key_names(monkeypatch):
    from flash.providers.lambda_.client import api as lambda_api
    from flash.providers.lambda_.jobs import resolve_ssh_key_names

    monkeypatch.setattr(lambda_api, "list_ssh_keys", lambda: [{"name": "jk"}, {"name": "other"}])
    assert resolve_ssh_key_names() == ["jk"]  # first registered key
    monkeypatch.setattr(lambda_api, "list_ssh_keys", list)
    with pytest.raises(lambda_api.LambdaApiError, match="requires an SSH key"):
        resolve_ssh_key_names()


# ---------------------------------------------------------------------------
# launch_and_submit: per-region weight cache (Lambda persistent filesystem)
# ---------------------------------------------------------------------------
def _wire_launch(monkeypatch):
    """Common launch wiring: ssh key + a launch that records (region, user_data, file_system_names)."""
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    calls = []

    def fake_launch(
        *, region_name, instance_type_name, ssh_key_names, name, user_data, file_system_names=None
    ):
        calls.append({"region": region_name, "user_data": user_data, "fs": file_system_names})
        return "i-cache"

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)
    return jobs, lambda_api, calls


def test_cache_ensures_filesystem_and_attaches_at_launch(monkeypatch):
    jobs, lambda_api, calls = _wire_launch(monkeypatch)
    ensured = []
    monkeypatch.setattr(
        lambda_api,
        "ensure_filesystem",
        lambda n, r, deadline_at=None: ensured.append((n, r)) or f"/lambda/nfs/{n}",
    )

    spec = _spec(network_volume="flash-weights")
    _launch(jobs, spec, instances=[_inst(region="us-east-1")], attempt=0)

    assert ensured == [("flash-weights", "us-east-1")]  # create-if-absent in THIS region
    assert calls[0]["fs"] == ["flash-weights"]  # attached at launch (Lambda can't attach later)
    # The cloud-init binds the auto-mounted NFS path into the worker at the fixed cache mount.
    assert (
        "-v '/lambda/nfs/flash-weights':/weight-cache" in calls[0]["user_data"]
    )  # quoted host path


def test_cache_bind_uses_returned_mount_point(monkeypatch):
    """The bind-mount targets the FS's ACTUAL mount_point, not the hard-coded /lambda/nfs/<name>.

    Regression: ensure_filesystem's returned mount_point was ignored, so a region where Lambda mounts
    the FS at a non-default host path would bind the wrong path -> silently cold / failed preload mount.
    """
    jobs, lambda_api, calls = _wire_launch(monkeypatch)
    # Lambda reports a NON-default host mount for this region's filesystem.
    monkeypatch.setattr(
        lambda_api,
        "ensure_filesystem",
        lambda n, r, deadline_at=None: "/mnt/lambda-fs/flash-weights",
    )

    _launch(jobs, _spec(network_volume="flash-weights"), instances=[_inst()], attempt=0)

    assert calls[0]["fs"] == ["flash-weights"]
    # the bind uses the REAL mount_point, and never the stale default
    assert "-v '/mnt/lambda-fs/flash-weights':/weight-cache" in calls[0]["user_data"]
    assert "/lambda/nfs/flash-weights" not in calls[0]["user_data"]


def test_cache_payload_points_base_model_prefetch_at_the_bind(monkeypatch):
    """The base64 payload points the base-model prefetch (FLASH_WEIGHT_CACHE_DIR) at the bind so the
    model download persists — NOT a process-global HF_HOME, so env/reward downloads stay ephemeral (#252)."""
    from flash.providers.lambda_.jobs import build_payload

    payload = build_payload(
        _spec(network_volume="flash-weights"),
        0,
        0,
        cache_host_mount="/lambda/nfs/flash-weights",
        source_snapshot=SOURCE_SNAPSHOT,
        deadline_at=_deadline_at(),
    )
    assert payload["env"]["FLASH_WEIGHT_CACHE_DIR"] == "/weight-cache/hf-cache/hub"
    assert "HF_HOME" not in payload["env"]
    assert payload["cache_host_mount"] == "/lambda/nfs/flash-weights"


def test_cache_discovery_failure_never_launches_cold(monkeypatch):
    jobs, lambda_api, calls = _wire_launch(monkeypatch)
    attempted_regions = []

    def unavailable(name, region, deadline_at=None):
        attempted_regions.append(region)
        raise lambda_api.LambdaApiError("filesystem quota exceeded")

    monkeypatch.setattr(lambda_api, "ensure_filesystem", unavailable)
    instances = [_inst(region="us-east-1"), _inst(region="us-west-2")]

    with pytest.raises(lambda_api.LambdaApiError, match="all 2 Lambda region"):
        _launch(
            jobs,
            _spec(network_volume="flash-weights"),
            instances=instances,
            attempt=0,
        )

    assert attempted_regions == ["us-east-1", "us-west-2"]
    assert calls == []


def test_filesystem_attach_reject_walks_cached_regions_without_cold_create(monkeypatch):
    """a cache attach rejection stays inside the cached region walk."""
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(
        lambda_api, "ensure_filesystem", lambda n, r, deadline_at=None: f"/lambda/nfs/{n}"
    )  # FS ensured
    calls = []

    def fake_launch(*, region_name, file_system_names=None, user_data=None, **kw):
        calls.append({"region": region_name, "fs": file_system_names})
        raise lambda_api.LambdaApiError(
            "POST /instance-operations/launch -> HTTP 400: file_system_names not attachable"
        )

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)
    monkeypatch.setattr(jobs, "usable_instances", lambda *_args, **_kwargs: [])
    instances = [_inst(region="us-east-1"), _inst(region="us-west-2")]

    with pytest.raises(lambda_api.LambdaApiError, match="all 2 Lambda region"):
        _launch(
            jobs,
            _spec(network_volume="flash-weights"),
            instances=instances,
            attempt=0,
        )

    assert calls == [
        {"region": "us-east-1", "fs": ["flash-weights"]},
        {"region": "us-west-2", "fs": ["flash-weights"]},
    ]


def test_capacity_reject_does_not_trigger_cold_fs_retry(monkeypatch):
    """A plain CAPACITY reject (no filesystem in the error) walks normally — no extra cold retry."""
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(
        lambda_api, "ensure_filesystem", lambda n, r, deadline_at=None: f"/lambda/nfs/{n}"
    )
    calls = []

    def fake_launch(*, region_name, file_system_names=None, **kw):
        calls.append({"region": region_name, "fs": file_system_names})
        if region_name == "us-east-1":
            raise lambda_api.LambdaApiError("PUT /asks/1/ -> HTTP 400: insufficient-capacity")
        return "i-2"

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)
    h = _launch(
        jobs,
        _spec(network_volume="flash-weights"),
        instances=[_inst(region="us-east-1"), _inst(region="us-west-2")],
        attempt=0,
    )
    assert h.region == "us-west-2"  # walked to the next region
    # us-east-1 tried ONCE (with fs), then walked — no extra cold retry in us-east-1
    assert [c["region"] for c in calls] == ["us-east-1", "us-west-2"]


def test_preload_mode_skips_region_when_cache_unavailable(monkeypatch):
    """In preload mode a cache-ensure failure SKIPS the region — never a cold full-training launch.

    Regression: the cold user_data carries no mode/models, so falling back to it for a preload would
    boot a full training run (GPU billing, timeout) and warm nothing. The walk must try the next
    region, and fail if none can host the cache.
    """
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(
        lambda_api,
        "ensure_filesystem",
        lambda n, r, deadline_at=None: (_ for _ in ()).throw(
            lambda_api.LambdaApiError("no FS capacity")
        ),
    )

    launched = []
    monkeypatch.setattr(lambda_api, "launch_instance", lambda **kw: launched.append(kw) or "i-x")

    insts = [_inst(region="us-east-1"), _inst(region="us-west-2")]
    with pytest.raises(lambda_api.LambdaApiError):
        _launch(
            jobs,
            _spec(network_volume="flash-weights"),
            instances=insts,
            attempt=0,
            mode="preload",
            models=["a/b"],
        )
    assert launched == []  # no region ever launched a cold (training) instance


def test_preload_mode_does_not_refresh_to_a_different_region(monkeypatch):
    """In preload mode a capacity rejection must NOT refresh to a NEW region and launch there.

    Regression: warm_instances pins each preload launch to one TARGET region and reports that exact
    region as warmed. If the launch is rejected and the walk refreshed (usable_instances) to a
    different region and launched there, the caller would report the cold target region as warmed.
    """
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(
        lambda_api, "ensure_filesystem", lambda n, r, deadline_at=None: f"/lambda/nfs/{n}"
    )  # cache OK
    launched = []

    def reject(**kw):
        launched.append(kw)
        raise lambda_api.LambdaApiError(
            "PUT /asks/1/ -> HTTP 400: insufficient-capacity"
        )  # clean reject

    monkeypatch.setattr(lambda_api, "launch_instance", reject)
    refresh_calls = []
    monkeypatch.setattr(
        jobs,
        "usable_instances",
        lambda gpu, force=False, gpu_count=1: (
            refresh_calls.append(force) or [_inst(region="us-fresh-9")]
        ),
    )

    with pytest.raises(lambda_api.LambdaApiError):
        _launch(
            jobs,
            _spec(network_volume="flash-weights"),
            instances=[_inst(region="us-east-1")],
            attempt=0,
            mode="preload",
            models=["a/b"],
        )
    assert [c["region_name"] for c in launched] == ["us-east-1"]  # only the TARGET region attempted
    assert refresh_calls == []  # the stale-stock refresh was NOT consulted in preload mode


def test_no_cache_never_touches_filesystems(monkeypatch):
    jobs, lambda_api, calls = _wire_launch(monkeypatch)

    def boom(*a, **k):
        raise AssertionError("ensure_filesystem must not be called without a requested cache")

    monkeypatch.setattr(lambda_api, "ensure_filesystem", boom)
    _launch(jobs, _spec(), instances=[_inst()], attempt=0)  # spec has no network_volume
    assert calls[0]["fs"] is None
    assert "/weight-cache" not in calls[0]["user_data"]


def test_cache_ensured_per_region_in_the_walk(monkeypatch):
    """Lazy per-region: the FS is ensured ONLY in the region the run actually lands in (walk skips on
    capacity, ensuring then launching with the cache per region)."""
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    ensured, attempts = [], []
    monkeypatch.setattr(
        lambda_api,
        "ensure_filesystem",
        lambda n, r, deadline_at=None: ensured.append(r) or f"/lambda/nfs/{n}",
    )

    def fake_launch(*, region_name, file_system_names=None, **kw):
        attempts.append(region_name)
        if len(attempts) < 2:
            raise lambda_api.LambdaApiError("PUT /asks/1/ -> HTTP 400: insufficient-capacity")
        return "i-2"

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)
    insts = [_inst(region="us-east-1"), _inst(region="us-west-2")]
    _launch(jobs, _spec(network_volume="flash-weights"), instances=insts, attempt=0)
    # Ensured in every region we actually attempted (east failed capacity, west succeeded) — never a
    # whole-fleet pre-create.
    assert ensured == ["us-east-1", "us-west-2"]


# ---------------------------------------------------------------------------
# poll_lambda_job state machine
# ---------------------------------------------------------------------------
_AUTO_MARKER = object()


def _wire_poll(
    monkeypatch,
    instances,
    done=None,
    marker=_AUTO_MARKER,
    metrics=None,
    boot=None,
    error=None,
    step=10.0,
):
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    if marker is _AUTO_MARKER:
        if done is None:
            marker = None
        else:

            def auto_marker():
                done_value = done() if callable(done) else done
                if done_value is None:
                    return None
                return _terminal_marker(ok=True)

            marker = auto_marker

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
        done="10000.0",
        metrics=json.dumps({"train_tokens": 4096, "wall_seconds": 100, "cost_usd": 0.0}),
    )
    # started_ts precedes the mocked clock (starts 10_000) so wall is positive on the first tick.
    res = jobs.poll_lambda_job(_handle(started_ts=9_000.0), _spec(), interval_s=0)
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
    res = jobs.poll_lambda_job(_handle(started_ts=9000.0), _spec(), interval_s=0)
    assert res.ok
    assert res.metrics["cost_usd"] == round((9100.0 - 9000.0) / 3600.0 * 1.29, 6)


def test_poll_retries_transient_metrics_blip_after_done(monkeypatch):
    # A fresh DONE guarantees metrics.json was uploaded first, so a None read is a transient HF blip
    # (the reader swallows 429/network and returns None) — it must be retried, not turned into a
    # terminal job_failed that discards (while still billing) a successful run.
    reads = {"n": 0}

    def metrics():
        reads["n"] += 1
        return None if reads["n"] <= 2 else json.dumps({"wall_seconds": 100, "cost_usd": 0.0})

    jobs = _wire_poll(
        monkeypatch, instances=[{"status": "active"}], done="10000.0", metrics=metrics
    )
    res = jobs.poll_lambda_job(_handle(started_ts=9000.0), _spec(), interval_s=0)
    assert res.ok, res
    assert reads["n"] >= 3  # retried past the two transient None reads instead of failing


def test_poll_persistent_metrics_unreadable_is_retriable_not_job_failed(monkeypatch):
    # If metrics.json stays unreadable after retries, fail RETRIABLY (poll_error) so the run is
    # re-attempted (a re-launch hits the worker's DONE-idempotency and restores the persisted
    # metrics without re-training) — NEVER the terminal job_failed that drops a billed success.
    jobs = _wire_poll(
        monkeypatch, instances=[{"status": "active"}], done="10000.0", metrics=lambda: None
    )
    res = jobs.poll_lambda_job(_handle(started_ts=9000.0), _spec(), interval_s=0)
    assert not res.ok
    assert res.failure == "poll_error"


def test_poll_marker_failure_is_job_failed(monkeypatch):
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}],
        marker=_terminal_marker(ok=False, error="RuntimeError: worker failed"),
    )
    res = jobs.poll_lambda_job(_handle(), _spec(), interval_s=0)
    assert not res.ok
    assert res.failure == "job_failed"  # real worker error fails fast
    assert "RuntimeError: worker failed" in res.detail


def test_poll_retriable_marker_is_job_preempted(monkeypatch):
    """A worker-flagged retriable failure retries on a fresh host (job_preempted), not job_failed."""
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}],
        marker=_terminal_marker(ok=False, retriable=True, error="worker failed; detail suppressed"),
    )
    res = jobs.poll_lambda_job(
        _handle(),
        _spec(),
        interval_s=0,
        heartbeat_reader=lambda force=False: {
            "retriable": True,
            "attempt": 0,
            "ts": 10_000.0,
        },
    )
    assert not res.ok
    assert res.failure == "job_preempted"


def test_poll_dead_host_without_marker_is_preempted(monkeypatch):
    """A host that died without writing DONE/marker is retryable without exposing its boot log."""
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}, {"status": "terminated"}],
        boot="+ docker pull ...\nFLASH: gpu never became ready",
    )
    res = jobs.poll_lambda_job(_handle(), _spec(), interval_s=0)
    assert not res.ok
    assert res.failure == "job_preempted"
    assert "lambda_attempt0_boot.log" in res.detail
    assert "gpu never became ready" in res.detail


def test_poll_dead_host_with_error_file_is_job_failed(monkeypatch):
    """A worker error artifact fails fast without exposing its arbitrary content."""
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}, {"status": "terminating"}],
        error="Traceback (most recent call last):\nFileNotFoundError: environment archive did not contain ...",
    )
    res = jobs.poll_lambda_job(
        _handle(), _spec(), interval_s=0, heartbeat_reader=lambda force=False: {}
    )
    assert not res.ok
    assert res.failure == "job_failed"
    assert "error_sft_attempt0.txt" in res.detail
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
        _handle(),
        _spec(),
        interval_s=0,
        heartbeat_reader=lambda force=False: {
            "retriable": True,
            "attempt": 0,
            "ts": 10_000.0,
        },
    )
    assert not res.ok
    assert res.failure == "job_preempted"


def test_poll_loading_timeout(monkeypatch):
    jobs = _wire_poll(monkeypatch, instances=[{"status": "booting"}], step=100.0)
    monkeypatch.setattr(jobs, "LOAD_TIMEOUT_S", 300.0)
    res = jobs.poll_lambda_job(_handle(), _spec(), interval_s=0)
    assert not res.ok
    assert res.failure == "stalled"
    assert "never became active" in res.detail


def test_poll_heartbeat_stall(monkeypatch):
    jobs = _wire_poll(monkeypatch, instances=[{"status": "active"}], step=100.0)
    # A FRESH training heartbeat (ts >= launch 10_000) that then FROZE: it proves liveness (so the
    # fast first-liveness failover is satisfied) AND arms the tight training stall window, so the
    # subsequent no-progress gap past stall_after_s is the stall actually under test here.
    frozen = {"stage": "rl", "step": 3, "ts": 10_000.0, "attempt": 0}
    res = jobs.poll_lambda_job(
        _handle(),
        _spec(),
        interval_s=0,
        heartbeat_reader=lambda force=False: frozen,
        stall_after_s=500.0,
    )
    assert not res.ok
    assert res.failure == "stalled"
    assert "no worker progress" in res.detail


def test_poll_active_no_liveness_fails_over_fast(monkeypatch):
    """The observed Lambda us-east-1 sick region: the instance reaches OS 'active' but the worker
    NEVER starts — no host boot.log, no heartbeat, no marker. The first-liveness deadline fails it
    over fast as a retriable 'stalled' (escaped cross-provider by the runner) instead of burning the
    full ~50 min setup grace."""
    jobs = _wire_poll(monkeypatch, instances=[{"status": "active"}], step=100.0)
    res = jobs.poll_lambda_job(_handle(), _spec(), interval_s=0, first_liveness_s=500.0)
    assert not res.ok
    assert res.failure == "stalled"  # infra-shaped -> retried + escaped cross-provider (PR #241)
    assert "no worker liveness" in res.detail
    assert "limit 500s" in res.detail


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
    res = jobs.poll_lambda_job(_handle(), _spec(), interval_s=0, first_liveness_s=50.0)
    assert not res.ok
    assert (
        res.failure == "job_preempted"
    )  # died as a host loss, NOT killed by the liveness deadline
    assert "no worker liveness" not in (res.detail or "")


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
    res = jobs.poll_lambda_job(_handle(), _spec(), interval_s=0, first_liveness_s=50.0)
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
            {"status": "active"},
            {"status": "active"},
            {"status": "active"},
            {"status": "terminated"},
        ],
        boot=boot_then_rate_limited,
        step=100.0,
    )
    res = jobs.poll_lambda_job(_handle(), _spec(), interval_s=0, first_liveness_s=50.0)
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
        return (
            None if calls["n"] == 1 else "+ docker pull ..."
        )  # transient error first, then readable

    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}, {"status": "active"}, {"status": "terminated"}],
        boot=transient_then_present,
        step=100.0,
    )
    res = jobs.poll_lambda_job(_handle(), _spec(), interval_s=0, first_liveness_s=50.0)
    assert res.failure == "job_preempted"  # the single transient None did not trip a failover
    assert "no worker liveness" not in (res.detail or "")


def test_poll_active_persistent_boot_log_absence_stalls_after_threshold(monkeypatch):
    """The genuine sick-region case: the boot.log is absent on EVERY forced read (cloud-init never
    ran). After BOOT_LOG_ABSENT_POLLS consecutive absent reads the first-liveness check declares the
    region 'stalled' (retriable, escaped cross-provider). Asserts the absence-count threshold is what
    gates the failover, not a single read."""
    from flash.providers._lifecycle.instances.poll import BOOT_LOG_ABSENT_POLLS

    calls = {"n": 0}

    def always_absent():
        calls["n"] += 1  # implicit None: every forced read comes back absent

    jobs = _wire_poll(monkeypatch, instances=[{"status": "active"}], boot=always_absent, step=100.0)
    res = jobs.poll_lambda_job(_handle(), _spec(), interval_s=0, first_liveness_s=50.0)
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
        interval_s=0,
        first_liveness_s=50.0,
        heartbeat_reader=lambda force=False: {
            "stage": "boot",
            "step": 0,
            "ts": 10_000.0,
            "attempt": 0,
        },
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
        interval_s=0,
        first_liveness_s=50.0,
        heartbeat_reader=lambda force=False: {
            "stage": "boot",
            "step": 0,
            "ts": 1.0,
            "attempt": 0,
        },
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
    elapsed (~5_000s) is measured from LAUNCH, not the reattach (~0s) — which is what proves the
    anchoring. A fresh launch (elapsed < setup grace) still fails over via the fast first-liveness
    path well before the setup grace, so the FAST-failover guarantee is unaffected."""
    jobs = _wire_poll(monkeypatch, instances=[{"status": "active"}], step=10.0)
    res = jobs.poll_lambda_job(
        _handle(started_ts=5_000.0), _spec(), interval_s=0, first_liveness_s=500.0
    )
    assert not res.ok
    assert res.failure == "stalled"
    # Elapsed counts from LAUNCH (~5_000s ago), not the reattach (~0s) — proves active_since stayed
    # anchored. Asserted as a floor (not an exact tick) so the terminal-artifact force-read before the
    # stall return can't make this brittle on the precise fake-clock count.
    elapsed = int(res.detail.split("for ", 1)[1].split("s", 1)[0])
    assert elapsed >= 5_000, res.detail


def test_cloud_init_emits_boot_log_before_pull_and_attempt_scoped(monkeypatch):
    """The host boot-log uploader must run BEFORE the docker image pull (so a box that ran cloud-init
    leaves an HF liveness artifact within ~2 min, well before the worker's first heartbeat), and its
    HF path must be attempt-scoped so a prior attempt's boot.log can't falsely prove liveness."""
    from flash.providers.lambda_.jobs import builders

    monkeypatch.setenv("LAMBDA_API_KEY", "lk")
    monkeypatch.setenv("HF_TOKEN", "hf")
    payload = _build_payload(builders, _spec(), attempt=2)
    assert payload["source_snapshot"] == SOURCE_SNAPSHOT
    assert "code_prefix" not in payload
    script = builders.build_user_data(payload)
    # the uploader INVOCATION precedes the image pull
    uploader = "python3 /opt/flash/capsule.pyz hostlog"
    assert uploader in script
    assert "docker pull" in script
    assert script.index(uploader) < script.index("docker pull")
    # ...and it precedes the capsule verification failure path as little as it must: the digest check
    # gates EVERY capsule invocation, uploader included, so an unverified capsule uploads nothing.
    assert script.index("sha256sum -c") < script.index(uploader)
    # attempt-scoped boot.log path, asserted against the SHIPPED uploader rather than the launch
    # text (the path is built inside the member now, not interpolated into the shell).
    assert '_attempt" + str(att) + "_boot.log' in _capsule_member("hostlog.py")


def test_poll_recovery_seeds_load_clock_from_launch(monkeypatch):
    """Reattach after a control-plane restart: a still-booting box has been billing since LAUNCH
    (handle.started_ts), so LOAD_TIMEOUT_S is measured from launch, NOT from this poll's first
    tick. A box already past the load window fails over on the first reattach iteration instead of
    getting another full window. (The mocked clock starts at 10_000; launch was 5000s earlier.)"""
    import re

    jobs = _wire_poll(monkeypatch, instances=[{"status": "booting"}], step=10.0)
    res = jobs.poll_lambda_job(_handle(started_ts=5_000.0), _spec(), interval_s=0)
    assert not res.ok
    assert res.failure == "stalled"
    assert "never became active" in res.detail
    m = re.search(r"for (\d+)s", res.detail)
    assert m is not None, res.detail
    # launch-relative (~5000s); the old "reset to reattach tick" code would report ~LOAD_TIMEOUT_S.
    assert int(m.group(1)) >= 2000, res.detail


def test_poll_rejects_missing_started_timestamp(monkeypatch):
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}],
        done="10000.0",
        metrics=json.dumps({"wall_seconds": 100, "cost_usd": 0.0}),
        step=10.0,
    )
    with pytest.raises(ValueError, match="launch timestamp is invalid"):
        jobs.poll_lambda_job(_handle(started_ts=0.0), _spec(), interval_s=0)


def test_poll_stale_heartbeat_does_not_buy_fresh_window(monkeypatch):
    """A heartbeat that was already stale before a restart must not reset the stall clock to the
    reattach time: its OWN ts is credited as last-progress, so an active worker frozen long ago
    stalls promptly instead of getting another full stall window. (Clock starts 10_000; the
    worker's last heartbeat was at 8500, launch at 8000, stall budget 500s.)"""
    import re

    jobs = _wire_poll(monkeypatch, instances=[{"status": "active"}], step=10.0)
    hb = {"stage": "rl", "step": 7, "ts": 8500.0, "attempt": 0}
    res = jobs.poll_lambda_job(
        _handle(started_ts=8_000.0),
        _spec(),
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


def test_poll_gapfill_step0_keeps_setup_grace(monkeypatch):
    """The train-liveness gap-filler emits rl_step/sft_step at step=0 throughout the silent FIRST step
    (a cold rollout can run minutes before global_step ticks to 1). That FRESH, non-setup but step-0
    heartbeat proves liveness yet must NOT tighten to the training window before any step completed —
    the larger SETUP grace must still govern (RunPod has the same step>=1 guard). (Clock starts
    10_000; launch 9000; fresh gap-fill ts 9500 >= launch but step 0.)"""
    import re

    jobs = _wire_poll(
        monkeypatch, instances=[{"status": "active"}], step=10.0, boot="+ cloud-init\n+ docker pull"
    )
    gapfill = {"stage": "rl_step", "step": 0, "ts": 9500.0}  # fresh, non-setup, but step 0
    res = jobs.poll_lambda_job(
        _handle(started_ts=9_000.0),
        _spec(),
        interval_s=0,
        heartbeat_reader=lambda force=False: gapfill,
        setup_grace_s=3000.0,
        stall_after_s=500.0,
    )
    assert not res.ok
    assert res.failure == "stalled"
    # SETUP grace (3000s) governs, not the tighter 500s training window a step-0 ping would have armed.
    assert "setup (pre-training)" in res.detail
    m = re.search(r"for (\d+)s", res.detail)
    assert m is not None, res.detail
    assert int(m.group(1)) >= 3000, res.detail


def test_poll_client_deadline(monkeypatch):
    jobs = _wire_poll(monkeypatch, instances=[{"status": "active"}], step=100.0)
    res = jobs.poll_lambda_job(_handle(), _spec(), interval_s=0, deadline_at=10_250.0)
    assert not res.ok
    assert res.failure == "stalled"
    assert "deadline" in res.detail


def test_poll_recovered_deadline_accepts_terminal_artifacts(monkeypatch):
    reads = {"done": 0, "metrics": 0}

    def done():
        reads["done"] += 1
        return "9900.0"

    def metrics():
        reads["metrics"] += 1
        return json.dumps({"wall_seconds": 100, "cost_usd": 0.0})

    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}],
        done=done,
        metrics=metrics,
        step=10.0,
    )
    res = jobs.poll_lambda_job(
        _handle(started_ts=5_000.0), _spec(), interval_s=0, deadline_at=10_250.0
    )
    assert res.ok
    assert reads == {"done": 2, "metrics": 1}


def test_poll_recovered_deadline_without_artifacts_still_stalls(monkeypatch):
    """When the recovered deadline fires and there is NO terminal artifact, the poller still returns
    `stalled` (the worker did not finish during the outage)."""
    jobs = _wire_poll(monkeypatch, instances=[{"status": "active"}], step=10.0)
    res = jobs.poll_lambda_job(
        _handle(started_ts=10_000.0),
        _spec(),
        interval_s=0,
        deadline_at=10_250.0,
        first_liveness_s=10_000.0,
        setup_grace_s=10_000.0,
        stall_after_s=10_000.0,
    )
    assert not res.ok
    assert res.failure == "stalled"
    assert "deadline" in res.detail


def test_provider_initial_and_reattached_poll_use_same_absolute_deadline(monkeypatch):
    """Initial and reattached polling consume the same persisted terminal cutoff."""
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.core.base import JobHandle, PollResult
    from flash.providers.lambda_.execution.provider import LambdaProvider

    deadline_at = 12_345.0
    captured = []

    def fake_poll(
        handle,
        spec,
        *,
        log=None,
        heartbeat_reader=None,
        deadline_at=None,
        first_liveness_s=None,
        setup_grace_s=None,
    ):
        captured.append(deadline_at)
        return PollResult(True)

    monkeypatch.setattr(jobs, "usable_instances", lambda _gpu, **_k: [_inst()])
    monkeypatch.setattr(jobs, "launch_and_submit", lambda *_a, **_k: _handle(started_ts=1.0))
    monkeypatch.setattr(jobs, "heartbeat_reader_for", lambda _spec: None)
    monkeypatch.setattr(jobs, "poll_lambda_job", fake_poll)
    monkeypatch.setattr(
        "flash.providers.lambda_.client.api.terminate_instance_confirmed", lambda instance_id: None
    )
    spec = _spec()
    provider = LambdaProvider()
    assert provider.submit_attempt(spec, _deadline_at=deadline_at).ok
    handle = JobHandle.from_dict({"provider": "lambda", **_handle(started_ts=1.0).to_dict()})
    assert provider.poll_attempt(handle, spec, _deadline_at=deadline_at).ok

    assert captured == [deadline_at, deadline_at]


def test_provider_poll_uses_uniform_wait_ignoring_on_last_gpu(monkeypatch):
    """The instance recovery poll uses a UNIFORM per-GPU wait: a persisted on_last_gpu does NOT scale
    first_liveness / setup grace — the poll relies on its unscaled defaults, matching the submit path.
    (on_last_gpu stays a Provider-interface param for RunPod; the instance providers ignore it.)"""
    from flash.providers.core.base import JobHandle
    from flash.providers.lambda_.execution.provider import LambdaProvider

    captured = {}

    def fake_poll(
        handle,
        spec,
        *,
        log=None,
        heartbeat_reader=None,
        deadline_at=None,
        first_liveness_s=None,
        setup_grace_s=None,
    ):
        captured["first_liveness_s"] = first_liveness_s
        captured["setup_grace_s"] = setup_grace_s
        from flash.providers.core.base import PollResult

        return PollResult(True)

    monkeypatch.setattr("flash.providers.lambda_.jobs.poll_lambda_job", fake_poll)
    monkeypatch.setattr(
        "flash.providers.lambda_.client.api.terminate_instance_confirmed", lambda instance_id: None
    )
    spec = _spec()
    # on_last_gpu=True must NOT override the timing -> the poll's unscaled defaults apply.
    handle = JobHandle.from_dict({**_handle().to_dict(), "provider": "lambda", "on_last_gpu": True})
    LambdaProvider().poll_attempt(handle, spec)
    assert captured["first_liveness_s"] is None  # not overridden -> poll uses its uniform default
    assert captured["setup_grace_s"] is None
    # on_last_gpu absent/False -> identical uniform wait.
    captured.clear()
    handle2 = JobHandle.from_dict({**_handle().to_dict(), "provider": "lambda"})
    LambdaProvider().poll_attempt(handle2, spec)
    assert captured["first_liveness_s"] is None
    assert captured["setup_grace_s"] is None


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
        _handle(), _spec(), interval_s=0, log=log, heartbeat_reader=lambda force=False: hb
    )
    assert res.ok
    assert "stage=sft" in log.getvalue()


# ---------------------------------------------------------------------------
# the cost-safety invariant: every exit path terminates the instance
# ---------------------------------------------------------------------------
def _wire_runner(monkeypatch, poll_outcome):
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.core.base import PollResult
    from flash.providers.lambda_.client import api as lambda_api

    terminated = []
    monkeypatch.setattr(
        lambda_api,
        "terminate_instance_confirmed",
        lambda instance_id: terminated.append([instance_id]),
    )
    monkeypatch.setattr(jobs, "usable_instances", lambda gpu, force=False, gpu_count=1: [_inst()])
    monkeypatch.setattr(jobs, "launch_and_submit", lambda *a, **k: _handle())

    def fake_poll(*a, **k):
        if isinstance(poll_outcome, BaseException):
            raise poll_outcome
        return poll_outcome

    monkeypatch.setattr(jobs, "poll_lambda_job", fake_poll)
    return jobs, terminated, PollResult


def test_runner_terminates_on_success(monkeypatch):
    from flash.providers.core.base import PollResult

    jobs, terminated, _ = _wire_runner(monkeypatch, PollResult(True, metrics={"a": 1}))
    handles = []
    res = _submit(jobs, _spec(), on_handle=handles.append)
    assert res.ok
    assert terminated == [["i-9999"]]
    assert handles
    assert handles[0]["provider"] == "lambda"
    assert handles[0]["instance_id"] == "i-9999"


def test_runner_preserves_success_when_teardown_is_unconfirmed(monkeypatch, caplog):
    from flash.providers.core.base import PollResult
    from flash.providers.lambda_.client import api as lambda_api

    jobs, _, _ = _wire_runner(monkeypatch, PollResult(True, metrics={"a": 1}))
    cleanup_runs = []

    def unconfirmed(_instance_id):
        raise lambda_api.LambdaApiError("instance remains")

    monkeypatch.setattr(lambda_api, "terminate_instance_confirmed", unconfirmed)
    monkeypatch.setattr(
        jobs,
        "terminate_run_instances",
        lambda run_id: cleanup_runs.append(run_id) or [],
    )
    handles = []
    caplog.set_level("ERROR")

    res = _submit(jobs, _spec(), on_handle=handles.append)

    assert res.ok
    assert res.metrics == {"a": 1}
    assert handles[0]["instance_id"] == "i-9999"
    assert cleanup_runs == [_spec().run_id]
    assert "persisted handle remains available" in caplog.text


@pytest.mark.parametrize("control_exc", [KeyboardInterrupt, SystemExit])
def test_runner_propagates_process_control_from_teardown(monkeypatch, control_exc):
    from flash.providers.core.base import PollResult
    from flash.providers.lambda_.client import api as lambda_api

    jobs, _, _ = _wire_runner(monkeypatch, PollResult(True, metrics={"a": 1}))
    monkeypatch.setattr(
        lambda_api,
        "terminate_instance_confirmed",
        lambda _instance_id: (_ for _ in ()).throw(control_exc()),
    )

    with pytest.raises(control_exc):
        _submit(jobs, _spec())


def test_runner_terminates_on_failure_and_exception(monkeypatch):
    from flash.providers.core.base import PollResult

    jobs, terminated, _ = _wire_runner(monkeypatch, PollResult(False, failure="stalled"))
    res = _submit(jobs, _spec())
    assert not res.ok
    assert terminated == [["i-9999"]]

    jobs, terminated, _ = _wire_runner(monkeypatch, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        _submit(jobs, _spec())
    assert terminated == [["i-9999"]]


def test_runner_terminates_when_handle_persist_fails(monkeypatch):
    """The launched instance is terminated even if on_handle raises — the teardown finally guards
    everything after the launch, not just the poll."""
    jobs, terminated, _ = _wire_runner(monkeypatch, None)

    def boom(_h):
        raise RuntimeError("status store unreachable")

    with pytest.raises(RuntimeError, match="status store unreachable"):
        _submit(jobs, _spec(), on_handle=boom)
    assert terminated == [["i-9999"]]


def test_submit_rejects_policy_word_gpu():
    """submit_attempt_lambda needs a concrete class; a policy word ("cheapest") — which the allocator
    resolves upstream — must fail with a clear error, not an opaque KeyError."""
    from flash.providers.lambda_.client import api as lambda_api
    from flash.providers.lambda_.jobs import submit_attempt_lambda

    spec = _spec()
    object.__setattr__(spec.gpu, "type", "cheapest")
    with pytest.raises(lambda_api.LambdaApiError, match="concrete gpu class"):
        submit_attempt_lambda(spec)


# ---------------------------------------------------------------------------
# labels, gc, orphan sweep
# ---------------------------------------------------------------------------
def test_instance_label_always_sweepable():
    from flash.providers.lambda_.jobs.builders import instance_label

    assert instance_label("flash-1700-abcd", 1) == "flash-1700-abcd-a1"
    assert instance_label("fail-fast", 0) == "flash-fail-fast-a0"  # prefix forced


def test_instance_label_bounds_the_attempt():
    """The attempt suffix is the only caller-supplied text appended after the (already-bounded) run
    prefix. It must never be truncated to fit the 60-char provider cap: a clipped ordinal is one two
    attempts of the same run can collide on, and it desyncs the name from the sweep-matched prefix.
    The seed is not in the name at all -- a run has exactly one, so it distinguished nothing while
    competing with the attempt for the digit budget."""
    from flash.providers._lifecycle.instances.instance import (
        _MAX_NAME,
        _SUFFIX_BUDGET,
        run_label_prefix,
    )
    from flash.providers.lambda_.jobs.builders import instance_label

    def suffix_of(rid, label):
        return label[len(run_label_prefix(rid)) :]

    rid = "flash-1700000000-abcd1234"
    assert instance_label(rid, 0) == f"{rid}-a0"
    assert "-s" not in suffix_of(rid, instance_label(rid, 0)), "the seed is not part of the name"

    # a long run id still fits: the prefix is bounded independently of the suffix.
    long_label = instance_label("flash-" + "x" * 80, 7)
    assert len(long_label) <= _MAX_NAME
    assert long_label.endswith("-a7")

    # the largest ordinal that fits is kept exactly, never clipped.
    widest = int("9" * (_SUFFIX_BUDGET - len("-a")))
    label = instance_label(rid, widest)
    assert label == f"{rid}-a{widest}"
    assert len(label) <= _MAX_NAME
    assert len(suffix_of(rid, label)) <= _SUFFIX_BUDGET

    # one digit past the budget raises rather than silently truncating to a colliding name.
    with pytest.raises(ValueError, match="exceeds the provider name budget"):
        instance_label(rid, widest * 10)
    with pytest.raises(ValueError, match="attempt identity is invalid"):
        instance_label(rid, "bad")


def test_terminate_run_instances_matches_forced_prefix(monkeypatch):
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    instances = [
        {"id": "i-1", "name": "flash-fail-fast-a0"},  # forced-prefix name
        {"id": "i-2", "name": "flash-other-run-a0"},  # different run -> keep
    ]
    terminated = []
    monkeypatch.setattr(lambda_api, "list_instances", lambda: instances)
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids: terminated.extend(ids) or list(ids)
    )
    assert jobs.terminate_run_instances("fail-fast") == ["i-1"]
    assert terminated == ["i-1"]


def test_run_instances_remaining_uses_exact_labels_and_exact_lookup(monkeypatch):
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import (
        api as lambda_api,
    )
    from flash.providers.lambda_.execution.provider import LambdaProvider

    run_id = "flash-100"
    rows = [
        {"id": "i-live", "name": jobs.instance_label(run_id, 0)},
        {"id": "i-gone", "name": jobs.instance_label(run_id, 0)},
        {"id": "i-other", "name": jobs.instance_label("flash-1000", 0)},
    ]
    lookups = []
    monkeypatch.setattr(lambda_api, "list_instances", lambda *, strict: rows)

    def lookup(instance_id, *, strict):
        lookups.append((instance_id, strict))
        return None if instance_id == "i-gone" else {"id": instance_id}

    monkeypatch.setattr(lambda_api, "get_instance", lookup)

    assert LambdaProvider().run_instances_remaining(run_id) == ["i-live"]
    assert lookups == [("i-live", True), ("i-gone", True)]


def test_run_instances_remaining_fails_closed_on_enumeration_lookup_or_identity(monkeypatch):
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import (
        api as lambda_api,
    )
    from flash.providers.lambda_.execution.provider import LambdaProvider

    provider = LambdaProvider()

    def listing_failure(*, strict):
        raise lambda_api.LambdaApiError("listing unavailable")

    monkeypatch.setattr(lambda_api, "list_instances", listing_failure)
    with pytest.raises(lambda_api.LambdaApiError, match="listing unavailable"):
        provider.run_instances_remaining("run1")

    monkeypatch.setattr(
        lambda_api,
        "list_instances",
        lambda *, strict: [{"id": "i-1", "name": jobs.instance_label("run1", 0)}],
    )

    def lookup_failure(instance_id, *, strict):
        raise lambda_api.LambdaApiError("lookup unavailable")

    monkeypatch.setattr(lambda_api, "get_instance", lookup_failure)
    with pytest.raises(lambda_api.LambdaApiError, match="lookup unavailable"):
        provider.run_instances_remaining("run1")

    monkeypatch.setattr(
        lambda_api,
        "list_instances",
        lambda *, strict: [{"id": None, "name": jobs.instance_label("run1", 0)}],
    )
    with pytest.raises(lambda_api.LambdaApiError, match="no usable id"):
        provider.run_instances_remaining("run1")


def test_handle_roundtrip():
    from flash.providers.lambda_.jobs.builders import LambdaJobHandle

    h = _handle()
    d = h.to_dict()
    assert d["provider"] == "lambda"
    assert LambdaJobHandle.from_dict(d) == h


def test_sweep_orphans_label_safety(monkeypatch):
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    instances = [
        {"id": "i-1", "name": "flash-1700-aaaa-a0"},  # orphan -> terminate
        {"id": "i-2", "name": "flash-1700-bbbb-a1"},  # active run -> keep
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
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    instances = [
        {"id": "i-1", "name": jobs.instance_label("flash-100", 0)},  # live -> KEEP
        {"id": "i-2", "name": jobs.instance_label("flash-1000", 0)},  # orphan -> terminate
    ]
    terminated = []
    monkeypatch.setattr(lambda_api, "list_instances", lambda: instances)
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids: terminated.extend(ids) or list(ids)
    )
    out = jobs.sweep_orphans(active_labels={"flash-100"})
    assert out == ["i-2"]


def test_sweep_orphans_protects_unprefixed_active_run_id(monkeypatch):
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    instances = [
        {"id": "i-1", "name": jobs.instance_label("fail-fast", 0)},  # live run -> KEEP
        {"id": "i-2", "name": jobs.instance_label("orphan-run", 0)},  # no live run -> terminate
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

    import flash.providers.lambda_.jobs as jobs
    from flash.providers._lifecycle.instances.instance import instance_label
    from flash.providers._lifecycle.instances.poll import preload_instance_run_id
    from flash.providers.lambda_.client import api as lambda_api

    # Build the name the way a launch does (instance_label bounds it to the provider name budget) so the
    # reap parser is tested against the REAL, possibly-truncated VM name, not the raw run id.
    fresh = preload_instance_run_id("lambda", "us-east-1", int(time.time()) + 1800, "abcdef")
    instances = [
        {"id": "i-1", "name": instance_label(fresh, 0)},  # in-deadline warm box -> KEEP
        {
            "id": "i-legacy",
            "name": "flash-preload-lambda-us-east-1-abcdef-a0",
        },  # no deadline -> KEEP
        {"id": "i-2", "name": "flash-1700-cccc-a0"},  # genuine orphan -> terminate
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

    import flash.providers.lambda_.jobs as jobs
    from flash.providers._lifecycle.instances.instance import instance_label
    from flash.providers._lifecycle.instances.poll import (
        PRELOAD_REAP_GRACE_S,
        preload_instance_run_id,
    )
    from flash.providers.lambda_.client import api as lambda_api

    # Deadline well past now + the reap grace -> driver provably gone. Name built via instance_label so
    # the front-loaded deadline token must survive the provider name-budget truncation to be reaped.
    stale_deadline = int(time.time()) - int(PRELOAD_REAP_GRACE_S) - 600
    stale = preload_instance_run_id("lambda", "us-west-1", stale_deadline, "deadbe")
    instances = [{"id": "i-9", "name": instance_label(stale, 0)}]
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
def test_provider_cancel_destroy_require_authoritative_teardown(monkeypatch):
    from flash.providers.core.base import JobHandle
    from flash.providers.core.registry import get_provider
    from flash.providers.lambda_.client import api as lambda_api

    terminated = []
    monkeypatch.setattr(
        lambda_api,
        "terminate_instance_confirmed",
        lambda instance_id: terminated.append(instance_id),
    )
    h = JobHandle("lambda", {"instance_id": "i-9"})
    get_provider("lambda").cancel(h)
    get_provider("lambda").destroy(h)
    assert terminated == ["i-9", "i-9"]

    def unconfirmed(_instance_id):
        raise lambda_api.LambdaApiError("termination unconfirmed")

    monkeypatch.setattr(lambda_api, "terminate_instance_confirmed", unconfirmed)
    with pytest.raises(lambda_api.LambdaApiError, match="unconfirmed"):
        get_provider("lambda").destroy(h)


def test_usable_instances_only_capacity_regions(monkeypatch):
    from flash.providers.lambda_.client import api as lambda_api
    from flash.providers.lambda_.jobs import usable_instances

    monkeypatch.setattr(
        lambda_api, "regions_with_capacity", lambda itype, force=False: ["us-east-1", "us-west-1"]
    )
    monkeypatch.setattr(
        "flash.providers.lambda_.client.pricing.hourly_rate", lambda g, *, gpu_count=1, **_k: 1.29
    )
    out = usable_instances("A10")
    assert {i.region for i in out} == {"us-east-1", "us-west-1"}
    assert all(i.gpu == "A10" and i.instance_type == "gpu_1x_a10" for i in out)
    # no capacity -> empty (the allocator then skips the class)
    monkeypatch.setattr(lambda_api, "regions_with_capacity", lambda itype, force=False: [])
    assert usable_instances("A10") == []


def test_allocator_capacity_aware(monkeypatch):
    """Lambda joins the ranked candidate list only for classes with LIVE capacity; a class with no
    capacity is excluded so the runner never walks to a class that would immediately fail to launch."""
    from flash.providers.core import allocator
    from flash.providers.lambda_.client import api as lambda_api
    from flash.providers.lambda_.jobs.builders import LambdaInstance

    monkeypatch.setenv("LAMBDA_API_KEY", "lk")  # make lambda available
    monkeypatch.setattr(lambda_api, "list_instance_types", lambda *a, **k: {"gpu_1x_h100_pcie": {}})

    def fake_usable(gpu, force=False, *, gpu_count=1, **_k):
        # h100 has capacity; b200 does not and is excluded from candidates.
        # a signature that rejected gpu_count would raise inside the provider, and the allocator
        # swallows that as a capacity blip, so the class would vanish for the wrong reason.
        if gpu == "H100":
            return [
                LambdaInstance(
                    "H100", "gpu_1x_h100_pcie", "us-east-1", 80, 3.29, gpu_count=gpu_count
                )
            ]
        return []

    monkeypatch.setattr("flash.providers.lambda_.jobs.usable_instances", fake_usable)
    a = allocator.allocate("Qwen/Qwen3.5-9B", "sft")
    lam = {c.gpu for c in a.candidates if c.provider == "lambda"}
    assert lam == {"H100"}  # only the fitting in-capacity class
    # RunPod still wins on price (cheaper static rates), so it's the chosen provider.
    assert a.provider == "runpod"


# --- review-fix regressions ---
def test_poll_ok_marker_succeeds_with_stale_done(monkeypatch):
    """A retry that hits the worker's already-complete path leaves DONE stale but writes ok marker +
    metrics; the poller must treat that as SUCCESS, not poll until it stalls."""
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}],
        done="9000.0",  # STALE (before the handle's started_ts=10000)
        marker=_terminal_marker(ok=True),
        metrics=json.dumps({"wall_seconds": 50, "cost_usd": 0.0}),
    )
    res = jobs.poll_lambda_job(_handle(), _spec(), interval_s=0)
    assert res.ok
    assert res.metrics["notes"]["provider"] == "lambda"


def test_ambiguous_launch_reconciles_and_stops(monkeypatch):
    """An ambiguous launch failure (timeout/5xx, maybe created an instance) must NOT walk to another
    region — it reconciles by name and raises so the run retries cleanly (cost safety)."""
    import io

    import flash.providers.lambda_.jobs as jobs
    from flash.providers.core.base import UnreconciledCreateError
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(
        lambda_api,
        "list_instances",
        lambda: (_ for _ in ()).throw(lambda_api.LambdaApiError("listing unavailable")),
    )
    attempts = []

    def fake_launch(**k):
        attempts.append(k["region_name"])
        raise lambda_api.LambdaApiError(
            "PUT /asks/1/ failed after 5 attempts: provider body secret"
        )

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)
    insts = [_inst(region=r) for r in ("us-east-1", "us-west-1")]
    log = io.StringIO()
    with pytest.raises(UnreconciledCreateError, match="refusing another create") as exc_info:
        _launch(jobs, _spec(), instances=insts, attempt=0, log=log)
    assert attempts == ["us-east-1"]  # stopped after the first ambiguous failure (no 2nd launch)
    assert "provider body secret" not in str(exc_info.value)
    assert "provider body secret" not in log.getvalue()


# ---------------------------------------------------------------------------
# post-launch success window: the box is rented but no handle exists yet (mirrors Vast)
# ---------------------------------------------------------------------------
def test_launch_success_log_failure_does_not_leak_handle(monkeypatch):
    # once launch_instance rents the box, a raising success log before the handle return must not
    # leak it: the handle is what every teardown path (finally, cancel, gc) names.
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(lambda_api, "launch_instance", lambda **_k: "i-4242")

    def raising_say(_log):
        def _say(_msg):
            raise OSError("log stream closed")

        return _say

    monkeypatch.setattr(jobs, "make_say", raising_say)
    h = _launch(jobs, _spec(), instances=[_inst()], attempt=0)
    assert h.instance_id == "i-4242"


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("terminate_confirmed", [True, False])
def test_post_launch_baseexception_cleans_and_never_walks_regions(
    monkeypatch, interrupt_type, terminate_confirmed
):
    # submit_attempt_lambda's finally only exists once launch_and_submit RETURNS a handle, so an
    # interrupt between a successful launch and that return would strand a paid box. Vast closes
    # this window with an exact destroy plus a run-label fallback; Lambda must do the same.
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    spec = _spec()
    launched = []
    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])

    def fake_launch(*, region_name, **_kwargs):
        launched.append(region_name)
        return "i-4242"

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)

    def time_after_launch():
        # _lambda_job_handle stamps started_ts, so this fires inside the unpublished window
        if launched:
            raise interrupt_type("stop")
        return 100.0

    monkeypatch.setattr(jobs.time, "time", time_after_launch)
    terminated = []

    def terminate_confirmed_instance(instance_id):
        terminated.append(instance_id)
        if not terminate_confirmed:
            raise lambda_api.LambdaApiError(f"lambda terminate({instance_id}) was not confirmed")

    monkeypatch.setattr(lambda_api, "terminate_instance_confirmed", terminate_confirmed_instance)
    reaped = []
    monkeypatch.setattr(jobs, "terminate_run_instances", lambda run_id: reaped.append(run_id) or [])

    with pytest.raises(interrupt_type):
        _launch(
            jobs,
            spec,
            instances=[_inst(region="us-east-1"), _inst(region="us-west-1")],
            attempt=0,
        )

    assert launched == ["us-east-1"]  # never walks on to rent a second box
    assert terminated == ["i-4242"]
    # an unconfirmed exact terminate escalates to the run-label reap; a confirmed one needs none
    assert reaped == ([] if terminate_confirmed else [spec.run_id])


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_post_launch_preserves_original_baseexception_when_cleanup_raises(
    monkeypatch, interrupt_type
):
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    class ExactCleanupFailure(BaseException):
        pass

    class LabelCleanupFailure(BaseException):
        pass

    spec = _spec()
    launched = []
    original = interrupt_type("original interruption")
    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(
        lambda_api, "launch_instance", lambda **_k: launched.append(True) or "i-4242"
    )

    def time_after_launch():
        if launched:
            raise original
        return 100.0

    monkeypatch.setattr(jobs.time, "time", time_after_launch)
    terminated = []
    reaped = []

    def terminate_exact(instance_id):
        terminated.append(instance_id)
        raise ExactCleanupFailure("exact cleanup failed")

    def terminate_label(run_id):
        reaped.append(run_id)
        raise LabelCleanupFailure("label cleanup failed")

    monkeypatch.setattr(lambda_api, "terminate_instance_confirmed", terminate_exact)
    monkeypatch.setattr(jobs, "terminate_run_instances", terminate_label)

    with pytest.raises(interrupt_type) as exc_info:
        _launch(jobs, spec, instances=[_inst()], attempt=0)

    # a cleanup that itself dies must never replace the interruption the caller has to see
    assert exc_info.value is original
    assert terminated == ["i-4242"]
    assert reaped == [spec.run_id]


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_launch_success_say_baseexception_does_not_trigger_run_wide_reap(
    monkeypatch, interrupt_type
):
    """say() raising a BaseException on the successful-launch route must be handled ONLY by
    _rent_instance's own exact cleanup; the outer coarse label reap must not also fire, since
    _rent_instance already owns cleanup for this instance and a run-wide reap on top of it would
    hit every other concurrently-launched seed of the same multi-seed run."""
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(lambda_api, "launch_instance", lambda **_k: "i-4242")

    def raising_say(_log):
        def _say(_msg):
            raise interrupt_type("log stream closed")

        return _say

    monkeypatch.setattr(jobs, "make_say", raising_say)
    terminated = []
    monkeypatch.setattr(
        lambda_api, "terminate_instance_confirmed", lambda iid: terminated.append(iid)
    )
    reaped = []
    monkeypatch.setattr(jobs, "terminate_run_instances", lambda run_id: reaped.append(run_id) or [])

    with pytest.raises(interrupt_type):
        _launch(jobs, _spec(), instances=[_inst()], attempt=0)

    assert terminated == ["i-4242"]  # the helper's own exact cleanup ran
    assert reaped == []  # the outer coarse label reap must not also fire


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_interrupt_while_building_the_success_message_terminates_only_this_instance(
    monkeypatch, interrupt_type
):
    """The id exists the moment launch_instance returns, so the guard must hold it from there.

    Between the create returning and the guard taking ownership, the success message is
    interpolated. An interrupt in that gap used to reach the outer handler with an armed but
    id-less guard, which reaps by run label and terminates every other concurrent seed of the run
    over a box this seed can name exactly."""
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(lambda_api, "launch_instance", lambda **_k: "i-4242")

    # interrupt DURING the message interpolation: __format__ runs while the guard is being set
    class _ExplodingPrice(float):
        def __format__(self, spec):
            raise interrupt_type("interrupted while formatting the launch message")

    terminated = []
    monkeypatch.setattr(
        lambda_api, "terminate_instance_confirmed", lambda iid: terminated.append(iid)
    )
    reaped = []
    monkeypatch.setattr(jobs, "terminate_run_instances", lambda run_id: reaped.append(run_id) or [])

    with pytest.raises(interrupt_type):
        _launch(
            jobs,
            _spec(),
            instances=[_inst(price=_ExplodingPrice(1.25))],
            attempt=0,
        )

    assert terminated == ["i-4242"]  # exact cleanup, by the id the guard already held
    assert reaped == []  # never the run-wide label sweep


def test_launch_rejected_by_the_apis_own_allowance_check_does_not_reap_the_run(monkeypatch):
    """launch_instance re-checks the create allowance before it issues the POST.

    Near the 60s threshold the caller's check can pass and the API's repeat can fail, raising
    before any request leaves the process. The guard is already armed at that point, so without
    the pre-request test the outer handler sweeps this run's label - terminating every concurrent
    seed - for a create that never happened."""
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])

    def allowance_exhausted(**_kwargs):
        raise RuntimeError(
            "run wall deadline has less than the 60-second minimum provider allowance remaining"
        )

    monkeypatch.setattr(lambda_api, "launch_instance", allowance_exhausted)
    terminated = []
    monkeypatch.setattr(
        lambda_api, "terminate_instance_confirmed", lambda iid: terminated.append(iid)
    )
    reaped = []
    monkeypatch.setattr(jobs, "terminate_run_instances", lambda run_id: reaped.append(run_id) or [])

    with pytest.raises(RuntimeError, match="provider allowance remaining"):
        _launch(jobs, _spec(), instances=[_inst()], attempt=0)

    assert reaped == []  # nothing was rented, so no seed of this run may be terminated
    assert terminated == []


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
    assert error.startswith("RetriableBootstrapError: train phase 'sft' exited non-zero")
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
    from flash.providers._lifecycle.bootstrapping import bootstrap as lb

    payload = {"hf_repo": "o/r", "hf_prefix": "sft/x", "env": {}}
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
    from flash.providers._lifecycle.bootstrapping import bootstrap as lb

    big = '{"k":"' + "v" * 200_000 + '"}'
    monkeypatch.setattr(lb, "fetch_spec_from_hf", lambda p: big)
    env = lb.build_worker_env(
        {
            "job_spec_json": "",
            "job_spec_in_hf": True,
            "phase": "sft",
            "seed": 0,
            "attempt": 0,
            "env": {},
            "flash_arm": "lambda",
            "run_id": "run-1",
            "source_snapshot": SOURCE_SNAPSHOT,
        }
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
    from flash.providers._lifecycle.bootstrapping import bootstrap as lb

    with pytest.raises(RuntimeError, match="no job spec"):
        lb.build_worker_env(
            {"phase": "sft", "seed": 0, "env": {}, "flash_arm": "lambda"}  # no job_spec_* at all
        )


def test_build_worker_env_spilled_spec_fetch_failure_is_retriable(monkeypatch):
    """The pre-worker HF fetch of a spilled spec is infra-shaped: a transient failure must surface
    as RetriableBootstrapError (not a bare error) so main() marks the attempt retriable and the
    poller retries on a fresh host instead of failing the run fast."""
    from flash.providers._lifecycle.bootstrapping import bootstrap as lb

    monkeypatch.setattr(
        lb, "fetch_spec_from_hf", lambda p: (_ for _ in ()).throw(RuntimeError("hf 503"))
    )
    with pytest.raises(lb.RetriableBootstrapError, match="spilled job spec"):
        lb.build_worker_env(
            {
                "job_spec_json": "",
                "job_spec_in_hf": True,
                "phase": "sft",
                "seed": 0,
                "env": {},
                "flash_arm": "lambda",
            }
        )


def test_main_marks_spilled_spec_fetch_failure_retriable(monkeypatch):
    """End-to-end: a payload whose spilled-spec HF fetch fails -> main() exits non-zero AND the
    written attempt marker carries retriable=True (so the poller -> job_preempted, not job_failed)."""
    from flash.providers._lifecycle.bootstrapping import bootstrap as lb

    markers: list[tuple[bool, str, bool]] = []
    created_at = time.time()
    monkeypatch.setattr(
        lb,
        "load_payload",
        lambda path=lb.PAYLOAD_PATH: {
            "hf_repo": "org/repo",
            "job_spec_json": "",
            "job_spec_in_hf": True,
            "phase": "sft",
            "seed": 0,
            "flash_arm": "lambda",
            "env": {},
            "extra_pip": [],
            "hf_prefix": "sft/x",
            "source_snapshot": SOURCE_SNAPSHOT,
            "deadline_at": created_at + 60.0,
            "run_created_at": created_at,
            "run_max_wall_seconds": 60.0,
            "attempt": 0,
        },
    )
    monkeypatch.setattr(lb, "fetch_code", lambda p: None)
    monkeypatch.setattr(
        lb, "fetch_spec_from_hf", lambda p: (_ for _ in ()).throw(RuntimeError("hf 503"))
    )
    monkeypatch.setattr(
        lb,
        "write_attempt_marker",
        lambda p, ok, error="", retriable=False: markers.append((ok, error, retriable)),
    )
    assert lb.main() == 1
    ok, error, retriable = markers[0]
    assert not ok
    assert error == "RetriableBootstrapError: failed to fetch the spilled job spec from HF"
    assert retriable is True


def test_shipped_bootstrap_secrets_is_byte_identical_to_the_repository_module():
    """The redactors that ship must BE the redactors under test, byte for byte.

    This file previously travelled as docstring-stripped source text, and the stripper was its own
    failure class: it rewrote a module nothing imports before launch, so a mangled body or an
    altered redactor first surfaced as a leak or a crash on a rented box. The capsule ships the file
    unmodified, which retires that class -- but only while it stays unmodified, so assert it rather
    than assume it. Byte identity is also what lets every other test in this suite exercise the
    importable module and still describe what the box runs.
    """
    from pathlib import Path

    from flash.providers._lifecycle.bootstrapping import secrets as bootstrap_secrets

    shipped = _capsule_member("bootstrap_secrets.py")
    assert shipped == Path(bootstrap_secrets.__file__).read_text()

    # and it is genuinely the redaction module, not an empty or wrong member that would satisfy a
    # comparison against the wrong file.
    namespace: dict = {}
    exec(compile(shipped, "<shipped>", "exec"), namespace)
    for name in ("_safe_detail", "_read_console_tail", "_payload_secrets"):
        assert name in namespace, f"the shipped member is missing {name}"
    for text, secrets in (
        ("worker rejected pin ati", {"PIN": "ati"}),
        ("trainer crashed after validation", {"PIN": "ati"}),
        ("https://host/a/repo", {"S": "/a"}),
        ("boto3 failed with sk-live-abc123456789", {"K": "sk-live-abc123456789"}),
    ):
        assert namespace["_safe_detail"](text, 1000, secrets) == bootstrap_secrets._safe_detail(
            text, 1000, secrets=secrets
        )


def test_build_user_data_spills_large_spec_out_of_cloud_init(monkeypatch):
    """A large job_spec_json must NOT be embedded inline in user_data (it can overflow the
    provider's cloud-init size cap and reject the launch). It is uploaded to HF and replaced by a
    small sentinel; small specs ride inline unchanged."""
    import huggingface_hub

    from flash.providers._lifecycle.instances import instance as inst

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
        "hf_prefix": "sft/x",
        "env": {"HF_TOKEN": "t"},
        "attempt": 0,
    }
    ud = inst.build_user_data(payload, image="img:latest")
    embedded = json.loads(base64.b64decode(ud.split("FLASH_PAYLOAD_EOF")[1].strip()))
    assert embedded["job_spec_json"] == ""
    assert embedded["job_spec_in_hf"] is True
    assert uploaded["path"] == "sft/x/job_spec.json"
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

    # The threshold is only as good as the framing it was chosen against, and that framing grows
    # every time the fixed template or the embedded runtime capsule does. Pin the WORST inline case
    # - a spec of exactly the threshold size - against the cap, so a future capsule that grows past
    # the remaining budget fails here instead of at a provider's launch call. The margin is asserted
    # too: the real payload also carries env, deadline, and cache fields this minimal one does not.
    uploaded.clear()
    representative = inst.build_payload(
        _spec(),
        attempt=7,
        arm="lambda",
        cache_host_mount="/mnt/cache",
        source_snapshot=SOURCE_SNAPSHOT,
        deadline_at=_deadline_at(),
    )
    spec_document = json.loads(representative["job_spec_json"])
    spec_document["_threshold_padding"] = ""
    compact = json.dumps(spec_document, separators=(",", ":"))
    padding = inst._SPEC_SPILL_THRESHOLD - len(compact)
    assert padding >= 0
    spec_document["_threshold_padding"] = "x" * padding
    representative["job_spec_json"] = json.dumps(spec_document, separators=(",", ":"))
    assert len(representative["job_spec_json"]) == inst._SPEC_SPILL_THRESHOLD

    worst = inst.build_user_data(representative, image="img:latest")
    embedded_worst = json.loads(base64.b64decode(worst.split("FLASH_PAYLOAD_EOF")[1].strip()))
    assert uploaded == {}
    assert embedded_worst["job_spec_json"] == representative["job_spec_json"]
    assert "job_spec_in_hf" not in embedded_worst
    assert len(worst.encode()) < 62_000

    # The spec does not ride alone: runtime secrets (a multiline PEM is a valid one) share the same
    # user_data. A spec UNDER the threshold plus a big secret must still spill, because the binding
    # check is the complete encoded payload rather than the spec component.
    #
    # The secret is sized from MEASURED headroom rather than a hardcoded constant: the fixed framing
    # is what decides how big a secret it takes to overflow, and a hardcoded size silently stops
    # reaching the force-spill branch whenever that framing shrinks (the capsule freed ~22KB doing
    # exactly that, and a 4,000-byte PEM no longer overflowed anything). Measure the inline floor,
    # then overshoot it by 1,000 bytes; base64 inflates a secret ~4/3 on the way in.
    uploaded.clear()
    inline_floor = len(
        inst.build_user_data(
            {**payload, "job_spec_json": "x" * (inst._SPEC_SPILL_THRESHOLD - 1)},
            image="img:latest",
        ).encode()
    )
    assert uploaded == {}, "the probe render must stay under the budget, or it is not a floor"
    secret_len = (inst._USER_DATA_BUDGET - inline_floor + 1_000) * 3 // 4
    pem = "-----BEGIN PRIVATE KEY-----\n" + "k" * secret_len + "\n-----END PRIVATE KEY-----"
    heavy_payload = {
        **payload,
        "job_spec_json": "x" * (inst._SPEC_SPILL_THRESHOLD - 1),
        "env": {"HF_TOKEN": "t", "DEPLOY_KEY": pem},
    }
    # unspilled, this payload genuinely overflows -- otherwise the assertions below would pass
    # without the force-spill branch ever running.
    assert (
        len(inst._render_user_data(dict(heavy_payload), image="img:latest").encode())
        > inst._USER_DATA_BUDGET
    )
    heavy = inst.build_user_data(heavy_payload, image="img:latest")
    assert uploaded["path"] == "sft/x/job_spec.json"
    emb3 = json.loads(base64.b64decode(heavy.split("FLASH_PAYLOAD_EOF")[1].strip()))
    assert emb3["job_spec_in_hf"] is True
    assert emb3["job_spec_json"] == ""
    assert len(heavy) < 64_000 - 2_000
    # this is the tightest real launch, and it clears by a margin measured in bytes. the three
    # bootstrap modules are embedded as SOURCE, and _strip_docstrings drops docstrings but keeps
    # comments -- so a comment added to any of them spends launch budget exactly like code does.
    # report the slack rather than only the pass/fail, so the next author who lands here sees how
    # much room is actually left instead of an opaque ValueError.
    slack = (64_000 - 2_000) - len(heavy.encode())
    assert slack >= 0, (
        f"the heavy-secret launch is {-slack} bytes over budget. the bootstrap modules ship as "
        "source text (comments included); shrink them rather than raising the cap."
    )


def test_build_user_data_rejects_a_payload_that_stays_oversized_after_spilling(monkeypatch):
    """spilling only moves the SPEC out. when the non-spec payload (large runtime secrets) is
    oversized on its own, spilling frees nothing and the launch would ship user_data the provider
    rejects opaquely, after the launch call. fail pre-flight instead, naming the component."""
    import huggingface_hub

    from flash.providers._lifecycle.instances import instance as inst

    class FakeApi:
        def __init__(self, token=None):
            pass

        def upload_file(self, **kwargs):
            pass

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    # a tiny spec plus a ~40KB runtime secret: base64 + json escaping put the rendering over the
    # budget, and no amount of spec spilling brings it back under.
    payload = {
        "flash_arm": "lambda",
        "job_spec_json": "{}",
        "hf_repo": "o/r",
        "hf_prefix": "sft/x",
        "env": {"HF_TOKEN": "t", "DEPLOY_KEY": "k" * 40_000},
        "attempt": 0,
    }

    with pytest.raises(ValueError, match="after spilling the job spec") as excinfo:
        inst.build_user_data(payload, image="img:latest")
    message = str(excinfo.value)
    assert "runtime secrets" in message
    assert str(inst._USER_DATA_CAP) in message
    # the error names the oversized component's size, not just the total.
    assert "40" in message


def test_build_user_data_starts_no_spec_upload_at_deadline(monkeypatch):
    import huggingface_hub

    from flash.providers._lifecycle.instances import instance as inst

    calls = []

    class FakeApi:
        def __init__(self, token=None):
            calls.append("init")

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(time, "time", lambda: 200.0)
    payload = {
        "flash_arm": "lambda",
        "job_spec_json": '{"k":"' + "v" * 100_000 + '"}',
        "hf_repo": "o/r",
        "hf_prefix": "sft/x",
        "env": {"HF_TOKEN": "t"},
        "attempt": 0,
        "deadline_at": 200.0,
    }

    with pytest.raises(TimeoutError, match="deadline"):
        inst.build_user_data(payload, image="img:latest")
    assert calls == []


def test_host_artifact_helpers_start_no_hf_request_at_deadline(monkeypatch):
    """At/after the deadline both helpers must abandon the upload BEFORE constructing HfApi.

    This asserts an absence, so it needs a positive control: a helper that never ran also touches
    HF zero times. The same clock is therefore replayed one second BEFORE the deadline, where each
    helper must reach HfApi -- so the empty result at the deadline is the guard, not a no-op.
    """
    import math
    import sys
    import types

    def run_at(now: float) -> list:
        calls = []

        class FakeApi:
            def __init__(self, token=None):
                calls.append("init")

            def file_exists(self, **kwargs):
                return True  # worker marker present: stop failmark before any upload

        monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))
        monkeypatch.setattr(time, "time", lambda: now)
        payload = {
            "flash_arm": "lambda",
            "attempt": 0,
            "run_id": "x",
            "hf_prefix": "sft/x",
            "hf_repo": "o/r",
            "env": {},
            "deadline_at": 200.0,
            "run_created_at": 100.0,
            "run_max_wall_seconds": 100.0,
        }

        def fake_open(path, *args, **kwargs):
            return io.StringIO(json.dumps(payload) if path == "/opt/flash/payload.json" else "")

        for member in ("hostlog.py", "failmark.py"):
            _run_capsule_member(
                member, {"json": json, "math": math, "time": time, "open": fake_open}
            )
        return calls

    assert run_at(200.0) == []
    assert run_at(199.0) == ["init", "init"]


def test_failmark_uses_truthful_detection_timestamp(monkeypatch):
    import math
    import sys
    import types

    uploaded = []
    written = {}

    class FakeApi:
        def __init__(self, token=None):
            pass

        def file_exists(self, **kwargs):
            return False

        def upload_file(self, **kwargs):
            uploaded.append(kwargs)

    class _Capture(io.StringIO):
        def write(self, value):
            written["marker"] = value
            return super().write(value)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))
    monkeypatch.setattr(time, "time", lambda: 150.0)
    payload = {
        "flash_arm": "lambda",
        "attempt": 0,
        "run_id": "x",
        "hf_prefix": "sft/x",
        "hf_repo": "o/r",
        "env": {},
        "deadline_at": 200.0,
        "run_created_at": 100.0,
        "run_max_wall_seconds": 100.0,
    }

    def fake_open(path, *args, **kwargs):
        if path == "/opt/flash/payload.json":
            return io.StringIO(json.dumps(payload))
        return _Capture()

    _run_capsule_member(
        "failmark.py", {"json": json, "math": math, "time": time, "open": fake_open}
    )

    assert len(uploaded) == 1
    assert json.loads(written["marker"])["ts"] == 150.0


def test_failmark_skips_when_worker_marker_exists(monkeypatch):
    """Bug: a container that fast-fails on a real user/config error uploads its own ok=false marker,
    then the host's ~5s liveness check fires fail() and would CLOBBER it with a retriable host
    marker (relabeling the user error as job_preempted). The host failmark must SKIP the write when
    a worker attempt marker already exists at the path (and stay conservative on a read error).

    Also covers the TOCTOU race: the worker can write its marker in the window BETWEEN the initial
    existence check and the host upload, so the failmark RE-checks immediately before uploading and
    still skips if the worker's marker has appeared."""
    import sys
    import types

    created_at = time.time()
    payload = {
        "flash_arm": "lambda",
        "attempt": 0,
        "run_id": "x",
        "hf_prefix": "sft/x",
        "hf_repo": "o/r",
        "env": {},
        "deadline_at": created_at + 60.0,
        "run_created_at": created_at,
        "run_max_wall_seconds": 60.0,
    }

    def run_failmark(exists_seq=(False,), read_raises=False):
        """Execute the shipped host failmark member against a fake HfApi + payload, return uploads.

        ``exists_seq`` is the sequence of file_exists() return values across the (up to two) checks
        the script makes; the last value repeats once exhausted."""
        uploaded = []
        seq = list(exists_seq)
        calls = {"n": 0}

        class FakeApi:
            def __init__(self, token=None):
                pass

            def file_exists(self, *, repo_id, filename, repo_type):
                if read_raises:
                    raise RuntimeError("hf down")
                i = min(calls["n"], len(seq) - 1)
                calls["n"] += 1
                return seq[i]

            def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type):
                uploaded.append(path_in_repo)

        monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))

        def fake_open(path, *a, **k):
            return io.StringIO(json.dumps(payload) if path == "/opt/flash/payload.json" else "")

        _run_capsule_member(
            "failmark.py",
            {
                "json": json,
                "sys": types.SimpleNamespace(argv=["failmark.py", "boom"]),
                "open": fake_open,
            },
        )
        return uploaded

    # Worker already wrote its marker -> host must NOT clobber it.
    assert run_failmark(exists_seq=(True,)) == []
    # No worker marker at all (never-started container) -> host failmark IS written.
    assert run_failmark(exists_seq=(False,)) == ["sft/x/lambda_attempt0.json"]
    # HF read error -> conservative: skip the write (never risk clobbering).
    assert run_failmark(exists_seq=(False,), read_raises=True) == []
    # RACE: absent on the first check, present on the re-check (worker uploaded in the gap) -> SKIP.
    assert run_failmark(exists_seq=(False, True)) == []
    # A mismatched canonical deadline is untrusted identity and must not produce a terminal marker.
    payload["run_max_wall_seconds"] = 59.0
    assert run_failmark(exists_seq=(False,)) == []


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_ambiguous_reject_keeps_the_guard_armed_when_the_announcement_raises(
    monkeypatch, interrupt_type
):
    """An AMBIGUOUS rejection may have rented a box, so the guard must survive a raising say.

    Standing down before the announcement loses the only handle on an instance that is rented but
    not yet named: _abort_ambiguous_launch never runs, and an unarmed guard reaches the outer
    handler with nothing to clean, leaving the box billing until a later orphan sweep."""
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    # a 500 is ambiguous: the create may have been accepted before the response was lost.
    monkeypatch.setattr(
        lambda_api,
        "launch_instance",
        lambda **_k: (_ for _ in ()).throw(lambda_api.LambdaApiError("POST -> HTTP 500")),
    )

    def raising_say(_log):
        def _say(_msg):
            raise interrupt_type("log stream closed")

        return _say

    monkeypatch.setattr(jobs, "make_say", raising_say)
    reaped = []
    monkeypatch.setattr(jobs, "terminate_run_instances", lambda run_id: reaped.append(run_id) or [])

    with pytest.raises(interrupt_type):
        _launch(jobs, _spec(), instances=[_inst()], attempt=0)

    # the guard stayed armed through the raising announcement, so the outer handler still sweeps
    # the run label - the only thing that can find a box rented but never named.
    assert reaped == ["flash-1700000000-abcd1234"]


def test_bootstrap_extra_pip_retries_when_the_console_closes_between_attempts(monkeypatch):
    """A console that closes between attempts must not consume the retry it only announces.

    The per-line tee is already best-effort, but the retry announcement runs after it, on the
    transient path where the next attempt is the whole point. An unguarded print there ends the
    install with a terminal console error instead of retrying a network failure that would have
    succeeded."""
    lb, calls = _wire_pip(monkeypatch, [("connection reset by peer\n", 1), ("", 0)])

    def closed_stream_print(*_a, **_kw):
        raise ValueError("I/O operation on closed file")

    monkeypatch.setattr("builtins.print", closed_stream_print)
    lb.install_extra_pip(_pip_payload())  # second attempt exits 0, so the install SUCCEEDS
    assert len(calls) == 2  # the retry issued despite the dead console


def test_recovered_catalog_restores_disk_metadata_after_a_failed_first_fetch(monkeypatch):
    """A transient catalog blip must not downgrade a known disk to UNMEASURED.

    ``regions_with_capacity`` fetches the same catalog, so when the first call exhausts its
    retries and the capacity call succeeds, the storage the SKU reports is available again.
    Leaving disk_gb=None there is not merely lossy: the floor treats unknown as permissive, so
    the walk would rent a shape whose fixed disk is provably below the run's requirement."""
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    catalog = {
        "gpu_1x_a10": {
            "instance_type": {"specs": {"storage_gib": 200}},
            "regions_with_capacity_available": [{"name": "us-east-1"}],
        }
    }
    calls = []

    def flaky_catalog(force=False, **_kwargs):
        calls.append(force)
        if len(calls) == 1:
            raise lambda_api.LambdaApiError("GET /instance-types -> HTTP 503")
        return catalog

    monkeypatch.setattr(lambda_api, "list_instance_types", flaky_catalog)
    monkeypatch.setattr(lambda_api, "regions_with_capacity", lambda *_a, **_k: ["us-east-1"])
    monkeypatch.setattr("flash.providers.lambda_.client.pricing.hourly_rate", lambda *a, **k: 1.29)

    instances = jobs.usable_instances("A10")

    assert [i.disk_gb for i in instances] == [200.0]  # recovered, not left unmeasured
