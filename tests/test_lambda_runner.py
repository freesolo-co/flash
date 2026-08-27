"""Lambda Cloud run lifecycle: cloud-init/bootstrap, region walk, poll state machine, guaranteed
terminate, orphan sweep, capacity-aware allocation (CPU-only; lambda API + HF readers mocked).

Lambda is opt-in via LAMBDA_API_KEY (the autouse offline fixture deletes it); these tests mock the
lambda API entirely, so no key is needed — except the allocator tests, which set it to make the
provider "available" and then mock the capacity lookup.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from types import SimpleNamespace
from unittest.mock import patch

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
    kwargs.setdefault("fence", 1)
    attempt_id = kwargs.get("attempt", args[2] if len(args) > 2 else 0)
    deadline_at = kwargs["deadline_at"]
    attempt = _instance_attempt(
        provider="lambda",
        grant=deadline_at - 120.0,
        work=deadline_at - 60.0,
        result=deadline_at,
        attempt_id=attempt_id,
        fence=kwargs["fence"],
    )
    with patch(
        "flash.runner.lifecycle.status.get_status",
        return_value=SimpleNamespace(attempt=attempt.to_dict()),
    ):
        return builders.build_payload(*args, **kwargs)


def _launch(jobs, *args, **kwargs):
    if "deadline_at" not in kwargs:
        kwargs["deadline_at"] = _deadline_at()
    kwargs.setdefault("source_snapshot", SOURCE_SNAPSHOT)
    kwargs.setdefault("fence", 1)
    deadline_at = kwargs["deadline_at"]
    attempt = _instance_attempt(
        provider="lambda",
        grant=deadline_at - 120.0,
        work=deadline_at - 60.0,
        result=deadline_at,
        attempt_id=kwargs.get("attempt", 0),
        fence=kwargs["fence"],
    )
    with patch(
        "flash.runner.lifecycle.status.get_status",
        return_value=SimpleNamespace(attempt=attempt.to_dict()),
    ):
        return jobs.launch_and_submit(*args, **kwargs)


def _submit(jobs, *args, **kwargs):
    if "deadline_at" not in kwargs:
        kwargs["deadline_at"] = _deadline_at()
    kwargs.setdefault("source_snapshot", SOURCE_SNAPSHOT)
    kwargs.setdefault("fence", 1)
    deadline_at = kwargs["deadline_at"]
    attempt = _instance_attempt(
        provider="lambda",
        grant=deadline_at - 120.0,
        work=deadline_at - 60.0,
        result=deadline_at,
        attempt_id=kwargs.get("attempt", 0),
        fence=kwargs["fence"],
    )
    with patch(
        "flash.runner.lifecycle.status.get_status",
        return_value=SimpleNamespace(attempt=attempt.to_dict()),
    ):
        return jobs.submit_run_lambda(*args, **kwargs)


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
        name="flash-x-s0-a0",
        gpu="A10",
        hourly_usd=rate,
        attempt=0,
        fence=1,
        started_ts=started_ts,
    )


# ---------------------------------------------------------------------------
# cloud-init user_data + bootstrap
# ---------------------------------------------------------------------------
def test_user_data_ships_payload_and_runs_worker_image(monkeypatch):
    from flash.providers.lambda_.jobs import builders

    monkeypatch.setenv("LAMBDA_API_KEY", "lk-supersecret")
    monkeypatch.setenv("HF_TOKEN", "hf-worker-token")
    deadline_at = time.time() + 3600
    payload = _build_payload(builders, _spec(), seed=0, attempt=1, deadline_at=deadline_at)
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
    # the worker owns fenced result publication. the shipped bootstrap only launches it and treats
    # the immutable result manifest as terminal authority.
    shipped = _capsule_member("bootstrap.py")
    assert "fenced result manifest" in shipped
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

    payload = _build_payload(builders, _spec(), seed=0, attempt=0)
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
    payload = _build_payload(builders, _spec(gpu_type="H100"), seed=0, attempt=0)
    script = builders.build_user_data(payload, gpu="H100")
    assert f"{WORKER_IMAGE}-sm90" in script


def _bootstrap_env(monkeypatch, phase="sft", rc=0, extra_pip=()):
    from flash.providers._lifecycle.bootstrapping import bootstrap as lb

    calls: list[str] = []

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
            "fence": 1,
        }

    monkeypatch.setattr(lb, "load_payload", payload)
    monkeypatch.setattr(lb, "fetch_code", lambda p: None)
    monkeypatch.setattr(
        lb,
        "run_mode",
        lambda p, e, m, d, **_kwargs: (calls.append(m), rc)[1],
    )
    return lb, calls


def test_build_worker_env_exports_fenced_identity():
    # every worker artifact is bound to the reserved attempt and fence. the shared instance bootstrap
    # must export both identities before worker execution.
    from flash.providers._lifecycle.bootstrapping import bootstrap as lb

    payload = {
        "phase": "sft",
        "seed": 0,
        "flash_arm": "vast",
        "attempt": 2,
        "fence": 3,
        "run_id": "run-1",
        "job_spec_json": "{}",
        "source_snapshot": SOURCE_SNAPSHOT,
        "env": {
            "GITHUB_TOKEN": "ghp-private-vcs",
            "GIT_ASKPASS": "/tmp/payload-askpass",
        },
    }
    env = lb.build_worker_env(payload)
    assert env["ATTEMPT"] == "2"
    assert env["FENCE"] == "3"
    assert "GITHUB_TOKEN" not in env
    assert "GIT_ASKPASS" not in env
    payload.pop("attempt")
    with pytest.raises(RuntimeError, match="attempt identity is invalid"):
        lb.build_worker_env(payload)


def test_bootstrap_train_success(monkeypatch):
    lb, calls = _bootstrap_env(monkeypatch)
    assert lb.main() == 0
    assert calls == ["sft"]


def test_bootstrap_fetch_code_failure_prevents_worker_launch(monkeypatch):
    lb, calls = _bootstrap_env(monkeypatch)

    def boom(_payload):
        raise lb.RetriableBootstrapError("failed to fetch the pinned flash source snapshot")

    monkeypatch.setattr(lb, "fetch_code", boom)
    assert lb.main() == 1
    assert calls == []


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
            "fence": 1,
            "env": {},
            "flash_arm": "lambda",
            "run_id": "run-1",
            "source_snapshot": SOURCE_SNAPSHOT,
        }
    )
    assert env["FLASH_ARM"] == "lambda"
    # And Lambda's build_payload is what sets flash_arm='lambda'.
    from flash.providers.lambda_.jobs import builders

    assert _build_payload(builders, _spec(), 0, 0)["flash_arm"] == "lambda"


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


def test_main_stops_before_worker_when_extra_pip_index_is_unreachable(monkeypatch):
    lb, calls = _bootstrap_env(monkeypatch, extra_pip=["some-env-pkg"])
    monkeypatch.setattr(lb.subprocess, "Popen", lambda *_a, **_k: _FakePipProc("read timed out", 1))
    monkeypatch.setattr(lb.time, "sleep", lambda _s: None)
    assert lb.main() == 1
    assert calls == []


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


def test_bootstrap_promotes_attempt_and_fence_to_worker_env():
    # the instance bootstrap must stamp the fenced attempt identity into the worker environment so
    # immutable progress and result records are written under the current authority boundary.
    from flash.providers._lifecycle.bootstrapping import bootstrap as lb
    from flash.providers.lambda_.jobs import builders

    base = {
        "job_spec_json": "{}",
        "phase": "sft",
        "seed": 0,
        "env": {},
        "flash_arm": "lambda",
        "run_id": "run-1",
        "source_snapshot": SOURCE_SNAPSHOT,
    }
    assert lb.build_worker_env({**base, "attempt": 3, "fence": 7})["ATTEMPT"] == "3"
    assert lb.build_worker_env({**base, "attempt": 3, "fence": 7})["FENCE"] == "7"
    assert lb.build_worker_env({**base, "attempt": 0, "fence": 1})["ATTEMPT"] == "0"
    with pytest.raises(RuntimeError, match="attempt identity is invalid"):
        lb.build_worker_env({**base, "fence": 1})
    # And the producer end actually carries the launched attempt into the payload bootstrap reads.
    assert (
        _build_payload(
            builders,
            _spec(),
            seed=0,
            attempt=2,
            fence=4,
            source_snapshot=SOURCE_SNAPSHOT,
            deadline_at=_deadline_at(),
        )["attempt"]
        == 2
    )
    assert (
        _build_payload(
            builders,
            _spec(),
            seed=0,
            attempt=2,
            fence=4,
            source_snapshot=SOURCE_SNAPSHOT,
            deadline_at=_deadline_at(),
        )["fence"]
        == 4
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
    h = _launch(jobs, _spec(), seed=0, instances=insts, attempt=2)
    assert attempts == ["us-east-1", "us-west-1", "us-west-2"]
    assert h.instance_id == "i-4242"
    assert h.region == "us-west-2"
    assert h.gpu == "A10"
    assert h.name == "flash-1700000000-abcd1234-s0-a2"


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
    h = _launch(jobs, _spec(), seed=0, instances=[_inst(region="us-east-1")], attempt=0)
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
        _launch(jobs, _spec(), seed=0, instances=[_inst()], attempt=0, deadline_at=159.0)

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
            seed=0,
            instances=[_inst(disk_gb=512.0)],
            attempt=0,
        )

    assert launched == []


def test_launch_accepts_a_disk_capable_or_unmeasured_instance_type(monkeypatch):
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(lambda_api, "launch_instance", lambda **_k: "i-1")

    assert _launch(jobs, _spec(disk_gb=200), seed=0, instances=[_inst(disk_gb=512.0)], attempt=0)
    # an unreported SKU disk is not a proven miss, so it must not block the launch
    assert _launch(jobs, _spec(disk_gb=800), seed=0, instances=[_inst()], attempt=0)


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
            seed=0,
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
        seed=0,
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
        _launch(jobs, _spec(), seed=0, instances=[_inst()], attempt=0)

    assert exact == ["i-1"]  # the rented box was terminated by id
    assert reaped == []  # ... so the run-wide reap must NOT also fire


def test_interrupt_while_the_cacheless_launch_request_is_in_flight_reaps_by_label(monkeypatch):
    """The cache-less retry's request can bill a box whose id never came back.

    The guard used to disarm on every exit from that helper, so an interrupt mid-request left the
    instance owned by nobody: no exact id to terminate, and the coarse reap already stood down.
    """
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(
        lambda_api, "ensure_filesystem", lambda n, r, deadline_at=None: f"/lambda/nfs/{n}"
    )
    reaped: list[str] = []
    monkeypatch.setattr(jobs, "terminate_run_instances", lambda run_id: reaped.append(run_id) or [])

    calls: list[str] = []

    def fake_launch(*, file_system_names=None, **_kwargs):
        calls.append("cold" if file_system_names is None else "cached")
        if file_system_names is None:
            raise KeyboardInterrupt  # interrupt with the cache-less create request in flight
        raise lambda_api.LambdaApiError(
            "POST /instance-operations/launch -> HTTP 400: file_system_names not attachable"
        )

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)

    with pytest.raises(KeyboardInterrupt):
        _launch(
            jobs,
            _spec(network_volume="flash-weights"),
            seed=0,
            instances=[_inst()],
            attempt=0,
        )

    assert calls == ["cached", "cold"]
    assert reaped == ["flash-1700000000-abcd1234"]  # only the label can name that box


def test_cacheless_ambiguous_reject_keeps_the_guard_armed_through_reconciliation(monkeypatch):
    """The cache-less leg's own AMBIGUOUS reject must not stand the guard down before it reconciles.

    The main walk splits clean from ambiguous before disarming; this helper is a second, separate
    handler and needs the same split. Disarming on every rejection here loses the only handle on a
    box the provider may have billed but never named: if anything between the disarm and
    _abort_ambiguous_launch raises, the outer handler finds an unarmed guard and the instance bills
    until a later orphan sweep.
    """
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(
        lambda_api, "ensure_filesystem", lambda n, r, deadline_at=None: f"/lambda/nfs/{n}"
    )
    calls: list[str] = []

    def fake_launch(*, file_system_names=None, **_kwargs):
        calls.append("cold" if file_system_names is None else "cached")
        if file_system_names is None:
            # a 500 is ambiguous: the create may have been accepted before the response was lost.
            raise lambda_api.LambdaApiError("POST -> HTTP 500")
        raise lambda_api.LambdaApiError(
            "POST /instance-operations/launch -> HTTP 400: file_system_names not attachable"
        )

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)
    # reconciliation observes nothing, so it raises UnreconciledCreateError rather than cleaning
    # up: the guard must still be armed when that reaches the outer handler.
    monkeypatch.setattr(lambda_api, "list_instances", list)
    reaped: list[str] = []
    monkeypatch.setattr(jobs, "terminate_run_instances", lambda run_id: reaped.append(run_id) or [])

    with pytest.raises(jobs.UnreconciledCreateError):
        _launch(
            jobs,
            _spec(network_volume="flash-weights"),
            seed=0,
            instances=[_inst()],
            attempt=0,
        )

    assert calls == ["cached", "cold"]
    # the cache-less create is ambiguous, so the guard stayed armed with no id and the outer
    # handler swept the label -- the only thing that can find a box rented but never named.
    assert reaped == ["flash-1700000000-abcd1234"]


def test_cacheless_retry_that_never_reaches_its_request_does_not_reap_the_run_label(monkeypatch):
    """A deadline miss before the cache-less create rented nothing, so the label must not be reaped.

    terminate_run_instances(run_id) kills every concurrently-launched seed sharing the run id. Only
    a window where this seed may hold an unnamed box justifies that, and the preflight is not one:
    require_create_allowance raises before any create is issued.
    """
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(
        lambda_api, "ensure_filesystem", lambda n, r, deadline_at=None: f"/lambda/nfs/{n}"
    )
    reaped: list[str] = []
    monkeypatch.setattr(jobs, "terminate_run_instances", lambda run_id: reaped.append(run_id) or [])

    calls: list[str] = []

    def fake_launch(*, file_system_names=None, **_kwargs):
        calls.append("cold" if file_system_names is None else "cached")
        if file_system_names is None:
            raise AssertionError("the cache-less request must not be issued past the deadline")
        raise lambda_api.LambdaApiError(
            "POST /instance-operations/launch -> HTTP 400: file_system_names not attachable"
        )

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)

    # the allowance check fails only on the retry, so the cached attempt still runs and rejects.
    seen = []

    def fake_allowance(_deadline_at):
        seen.append(1)
        if len(seen) > 1:
            raise TimeoutError("no create allowance left")

    monkeypatch.setattr(jobs, "require_create_allowance", fake_allowance)

    with pytest.raises(TimeoutError):
        _launch(
            jobs,
            _spec(network_volume="flash-weights"),
            seed=0,
            instances=[_inst()],
            attempt=0,
        )

    assert calls == ["cached"]  # the cold request was never issued
    assert reaped == []  # so no concurrent seed of this run was terminated


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
        _launch(jobs, _spec(), seed=0, instances=[_inst()], attempt=0)

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
        _launch(jobs, _spec(), seed=0, instances=[_inst()], attempt=0)
    with pytest.raises(lambda_api.LambdaApiError, match="no Lambda capacity"):
        _launch(jobs, _spec(), seed=0, instances=[], attempt=0)


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
    _launch(jobs, spec, seed=0, instances=[_inst(region="us-east-1")], attempt=0)

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

    _launch(jobs, _spec(network_volume="flash-weights"), seed=0, instances=[_inst()], attempt=0)

    assert calls[0]["fs"] == ["flash-weights"]
    # the bind uses the REAL mount_point, and never the stale default
    assert "-v '/mnt/lambda-fs/flash-weights':/weight-cache" in calls[0]["user_data"]
    assert "/lambda/nfs/flash-weights" not in calls[0]["user_data"]


def test_cache_payload_points_base_model_prefetch_at_the_bind(monkeypatch):
    """The base64 payload points the base-model prefetch (FLASH_WEIGHT_CACHE_DIR) at the bind so the
    model download persists — NOT a process-global HF_HOME, so env/reward downloads stay ephemeral (#252)."""
    from flash.providers.lambda_.jobs import builders

    payload = _build_payload(
        builders,
        _spec(network_volume="flash-weights"),
        0,
        0,
        cache_host_mount="/lambda/nfs/flash-weights",
    )
    assert payload["env"]["FLASH_WEIGHT_CACHE_DIR"] == "/weight-cache/hf-cache/hub"
    assert "HF_HOME" not in payload["env"]
    assert payload["cache_host_mount"] == "/lambda/nfs/flash-weights"


def test_cache_falls_back_to_cold_when_filesystem_unavailable(monkeypatch):
    jobs, lambda_api, calls = _wire_launch(monkeypatch)
    monkeypatch.setattr(
        lambda_api,
        "ensure_filesystem",
        lambda n, r, deadline_at=None: (_ for _ in ()).throw(
            lambda_api.LambdaApiError("filesystem quota exceeded")
        ),
    )
    _launch(jobs, _spec(network_volume="flash-weights"), seed=0, instances=[_inst()], attempt=0)
    assert calls[0]["fs"] is None  # no filesystem attached
    assert "/weight-cache" not in calls[0]["user_data"]  # cold user_data, no bind


def test_filesystem_attach_reject_retries_same_region_cold(monkeypatch):
    """A clean reject whose error mentions the FILESYSTEM retries THIS region cache-less before
    walking — so a best-effort attach can't make a region the cold path would have served fail."""
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(
        lambda_api, "ensure_filesystem", lambda n, r, deadline_at=None: f"/lambda/nfs/{n}"
    )  # FS ensured
    calls = []

    def fake_launch(*, region_name, file_system_names=None, user_data=None, **kw):
        calls.append({"region": region_name, "fs": file_system_names})
        if file_system_names:  # the CACHED launch is rejected for a filesystem-attach reason
            raise lambda_api.LambdaApiError(
                "POST /instance-operations/launch -> HTTP 400: file_system_names not attachable"
            )
        return "i-cold"  # the cold retry (no fs) succeeds in the SAME region

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)
    h = _launch(
        jobs,
        _spec(network_volume="flash-weights"),
        seed=0,
        instances=[_inst(region="us-east-1")],
        attempt=0,
    )
    assert h.region == "us-east-1"  # served by the SAME region, not lost to the walk
    assert [c["fs"] for c in calls] == [["flash-weights"], None]  # cached attempt, then cold retry
    assert all(c["region"] == "us-east-1" for c in calls)


def test_filesystem_reject_rechecks_deadline_before_cacheless_creation(monkeypatch):
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(
        lambda_api,
        "ensure_filesystem",
        lambda name, region, deadline_at=None: f"/lambda/nfs/{name}",
    )
    now = {"value": 100.0}
    monkeypatch.setattr(jobs.time, "time", lambda: now["value"])
    calls = []

    def fake_launch(*, file_system_names=None, **_kwargs):
        calls.append(file_system_names)
        now["value"] = 141.0
        raise lambda_api.LambdaApiError(
            "POST /instance-operations/launch -> HTTP 400: file_system_names not attachable"
        )

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)

    with pytest.raises(RuntimeError, match="60-second minimum provider allowance"):
        _launch(
            jobs,
            _spec(network_volume="flash-weights"),
            seed=0,
            instances=[_inst(region="us-east-1")],
            attempt=0,
            deadline_at=200.0,
        )

    assert calls == [["flash-weights"]]


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
        seed=0,
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
            seed=0,
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
            seed=0,
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
    _launch(jobs, _spec(), seed=0, instances=[_inst()], attempt=0)  # spec has no network_volume
    assert calls[0]["fs"] is None
    assert "/weight-cache" not in calls[0]["user_data"]


def test_cache_ensured_per_region_in_the_walk(monkeypatch):
    """Lazy per-region: the FS is ensured ONLY in the region the run actually lands in (walk skips on
    capacity, ensuring then launching cold/cache per region)."""
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
    _launch(jobs, _spec(network_volume="flash-weights"), seed=0, instances=insts, attempt=0)
    # Ensured in every region we actually attempted (east failed capacity, west succeeded) — never a
    # whole-fleet pre-create.
    assert ensured == ["us-east-1", "us-west-2"]


# ---------------------------------------------------------------------------
# lambda resource and fenced-result polling
# ---------------------------------------------------------------------------
def _instance_attempt(*, provider, grant=5.0, work=200.0, result=220.0, attempt_id=0, fence=1):
    from flash.runner.lifecycle.protocol import AttemptRecord

    return AttemptRecord.from_dict(
        {
            "attempt_id": attempt_id,
            "fence": fence,
            "state": "active",
            "reserved_at": 1.0,
            "grant_deadline_at": grant,
            "work_deadline_at": work,
            "result_deadline_at": result,
            "run_deadline_at": work,
            "provider": provider,
            "provider_contract": None,
            "resource": None,
            "allocation": None,
            "progress_receipt": None,
            "result_receipt": None,
            "cleanup": {},
            "schema_version": 1,
        }
    )


def _instance_clock(start=0.0, step=10.0):
    value = start - step

    def now():
        nonlocal value
        value += step
        return value

    return now


def _wire_lambda_poll(monkeypatch, *, attempt=None, results=(), instances=()):
    import flash.providers.lambda_.jobs as jobs
    from flash.providers._lifecycle.instances import poll_instance
    from flash.providers.lambda_.client import api as lambda_api
    from flash.runner.lifecycle import status as status_ops

    result_iter = iter(results)
    instance_iter = iter(instances)
    last = {"value": None}

    def get_instance(*_args, **_kwargs):
        last["value"] = next(instance_iter, last["value"])
        return last["value"]

    monkeypatch.setattr(lambda_api, "get_instance", get_instance)
    monkeypatch.setattr(status_ops, "get_status", lambda _run_id: SimpleNamespace(remote={}))
    monkeypatch.setattr(
        status_ops,
        "source_snapshot_from_status",
        lambda _status, required=True: dict(SOURCE_SNAPSHOT),
    )
    monkeypatch.setattr(
        poll_instance,
        "_current_attempt",
        lambda _adapter: attempt or _instance_attempt(provider="lambda"),
    )
    monkeypatch.setattr(poll_instance, "_record_resource", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(poll_instance, "_observe_result", lambda _adapter: next(result_iter, None))
    monkeypatch.setattr(poll_instance.time, "sleep", lambda _seconds: None)
    return jobs, poll_instance


def test_poll_lambda_returns_current_fenced_result_before_status(monkeypatch):
    from flash.providers.core.base import PollResult

    jobs, _poll = _wire_lambda_poll(
        monkeypatch,
        results=[PollResult(True, metrics={"wall_seconds": 12.0})],
    )

    result = jobs.poll_lambda_job(_handle(), _spec(), seed=0, interval_s=0)

    assert result.ok
    assert result.metrics == {"wall_seconds": 12.0}


def test_poll_lambda_retries_result_download_and_costs_manifest_finished_at(monkeypatch):
    from flash.providers._lifecycle.instances import poll_instance as poll_module
    from flash.providers.artifacts.attempts import AttemptArtifacts
    from flash.runner.lifecycle.protocol import ResultManifest
    from flash.snapshot.archive import source_attestation

    real_observe = poll_module._observe_result
    jobs, poll_instance = _wire_lambda_poll(
        monkeypatch,
        attempt=_instance_attempt(provider="lambda", work=100.0, result=120.0),
        instances=[{"status": "active"}],
    )
    manifest = ResultManifest(
        run_id=_spec().run_id,
        phase_namespace="sft",
        attempt_id=0,
        fence=1,
        outcome="succeeded",
        failure_class=None,
        started_at=9_000.0,
        finished_at=9_100.0,
        training_entered=True,
        completed_steps=1,
        metrics={"wall_seconds": 80.0},
        checkpoint={},
        artifacts={"adapter": "published"},
        source_attestation=source_attestation(
            SOURCE_SNAPSHOT,
            run_id=_spec().run_id,
            attempt=0,
            fence=1,
        ),
        diagnostics={},
    )
    reads = iter(
        [
            OSError("temporary result download failure"),
            AttemptArtifacts("revision", 9_101.0, None, manifest.to_dict()),
        ]
    )

    def read_artifacts(*_args, **_kwargs):
        observed = next(reads)
        if isinstance(observed, Exception):
            raise observed
        return observed

    monkeypatch.setattr(poll_instance, "_observe_result", real_observe)
    monkeypatch.setattr(poll_instance, "read_attempt_artifacts", read_artifacts)
    monkeypatch.setattr(poll_instance, "persist_attempt_artifacts", lambda *_args: None)
    monkeypatch.setattr(poll_instance.time, "time", _instance_clock(step=1.0))

    result = jobs.poll_lambda_job(
        _handle(started_ts=9_000.0),
        _spec(),
        seed=0,
        interval_s=0,
    )

    assert result.ok
    assert result.metrics["cost_usd"] == round(100.0 / 3600.0 * 1.29, 6)
    assert result.metrics["notes"]["lambda_region"] == "us-east-1"


def test_poll_lambda_dead_instance_waits_for_result_deadline(monkeypatch):
    jobs, poll_instance = _wire_lambda_poll(
        monkeypatch,
        attempt=_instance_attempt(provider="lambda", work=20.0, result=30.0),
        results=[None, None],
        instances=[{"status": "terminated"}],
    )
    monkeypatch.setattr(poll_instance.time, "time", _instance_clock())

    result = jobs.poll_lambda_job(_handle(), _spec(), seed=0, interval_s=0)

    assert result.failure == "job_preempted"
    assert "terminated" in result.detail
    assert "result manifest" in result.detail


def test_poll_lambda_dead_instance_before_grant_waits_for_result_deadline(monkeypatch):
    jobs, poll_instance = _wire_lambda_poll(
        monkeypatch,
        attempt=_instance_attempt(provider="lambda", grant=25.0, work=40.0, result=50.0),
        results=[None, None, None],
        instances=[{"status": "terminated"}],
    )
    monkeypatch.setattr(poll_instance.time, "time", _instance_clock())

    result = jobs.poll_lambda_job(_handle(), _spec(), seed=0, interval_s=0)

    assert result.failure == "job_preempted"
    assert "terminated" in result.detail
    assert "result manifest" in result.detail
    assert "grant deadline" not in result.detail


def test_poll_lambda_active_without_progress_uses_attempt_deadline(monkeypatch):
    jobs, poll_instance = _wire_lambda_poll(
        monkeypatch,
        attempt=_instance_attempt(provider="lambda", work=25.0, result=45.0),
        results=[None, None, None],
        instances=[{"status": "active"}],
    )
    monkeypatch.setattr(poll_instance.time, "time", _instance_clock())

    result = jobs.poll_lambda_job(
        _handle(),
        _spec(),
        seed=0,
        interval_s=0,
        deadline_at=10_000.0,
    )

    assert result.failure == "job_preempted"
    assert "work deadline expired" in result.detail


def test_poll_lambda_preserves_provisioning_deadline(monkeypatch):
    jobs, poll_instance = _wire_lambda_poll(
        monkeypatch,
        attempt=_instance_attempt(provider="lambda", grant=15.0, work=100.0, result=120.0),
        results=[None, None],
        instances=[{"status": "booting"}],
    )
    monkeypatch.setattr(poll_instance.time, "time", _instance_clock())

    result = jobs.poll_lambda_job(_handle(), _spec(), seed=0, interval_s=0)

    assert result.failure == "job_preempted"
    assert "grant deadline" in result.detail


def test_poll_lambda_bounds_provider_status_failures(monkeypatch):
    from flash.providers._lifecycle.instances import poll as poll_helpers
    from flash.providers.lambda_.client import api as lambda_api

    jobs, poll_instance = _wire_lambda_poll(
        monkeypatch,
        attempt=_instance_attempt(provider="lambda", work=1_000.0, result=1_020.0),
        results=[None] * 10,
    )
    monkeypatch.setattr(poll_instance.time, "time", _instance_clock(step=1.0))
    monkeypatch.setattr(poll_helpers.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        lambda_api,
        "get_instance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(lambda_api.LambdaApiError("offline")),
    )

    result = jobs.poll_lambda_job(_handle(), _spec(), seed=0, interval_s=0)

    assert result.failure == "poll_error"


def test_lambda_poll_adapter_carries_attempt_fence_and_cost_stamp(monkeypatch):
    import flash.providers.lambda_.jobs as jobs
    from flash.runner.lifecycle import status as status_ops

    captured = {}
    monkeypatch.setattr(status_ops, "get_status", lambda _run_id: SimpleNamespace())
    monkeypatch.setattr(
        status_ops,
        "source_snapshot_from_status",
        lambda _status, required=True: dict(SOURCE_SNAPSHOT),
    )

    def capture(adapter, **_kwargs):
        captured["adapter"] = adapter
        from flash.providers.core.base import PollResult

        return PollResult(True, metrics={})

    monkeypatch.setattr(jobs, "poll_instance_job", capture)

    result = jobs.poll_lambda_job(_handle(started_ts=9_000.0), _spec(), seed=0)
    metrics = {"wall_seconds": 100.0}
    captured["adapter"].stamp_cost_and_notes(metrics, end_ts=9_100.0, launch_ts=9_000.0)

    assert result.ok
    assert captured["adapter"].current_attempt == 0
    assert captured["adapter"].fence == 1
    assert metrics["cost_usd"] == round(100.0 / 3600.0 * 1.29, 6)
    assert metrics["notes"]["provider"] == "lambda"


def test_isolated_instance_miss_is_not_projected_terminal(monkeypatch):
    """an unconfirmed miss must not surface publicly as a dead resource.

    the poll loop deliberately requires ``missing_dead_threshold`` consecutive misses before it
    treats an instance as gone, so a single transient ``get_instance`` miss leaves the loop still
    treating the run as live. projecting ``terminal`` on that first miss makes ``flash runs status``
    and log-follow report a dead resource for a run the same loop may observe running next poll.
    """
    from flash.providers._lifecycle.instances import poll_instance
    from flash.runner.lifecycle import status as status_ops

    recorded = []
    monkeypatch.setattr(status_ops, "get_status", lambda _run_id: SimpleNamespace(remote={}))
    monkeypatch.setattr(
        status_ops, "record_resource", lambda _run_id, payload, **_k: recorded.append(payload)
    )
    adapter = SimpleNamespace(
        run_id="run-1",
        current_attempt=0,
        fence=1,
        provider="lambda",
        instance_id="i-1",
        running_status="active",
        dead_states=frozenset({"terminated"}),
        resource_identity=("lambda", 0, 1, "i-1"),
    )

    poll_instance._record_resource(adapter, "missing", confirmed_missing=False)
    poll_instance._record_resource(adapter, "missing", confirmed_missing=True)
    # the transport path replays the last status, so it must state its verdict too.
    poll_instance._record_resource(
        adapter, "missing", transport="unavailable", confirmed_missing=False
    )

    assert recorded[0]["provider_state"] == "missing"
    # unconfirmed stays provisioning; only the threshold-confirmed miss is terminal.
    assert recorded[0]["state"] == "provisioning"
    assert recorded[1]["state"] == "terminal"
    assert recorded[2]["state"] == "provisioning"
    assert recorded[2]["transport"] == "unavailable"
    # the verdict is required, never defaulted: a call site that forgets it must not silently
    # re-answer "is this resource gone?" -- that is how the transport path regressed once already.
    with pytest.raises(TypeError):
        poll_instance._record_resource(adapter, "missing")


def test_instance_observation_is_fenced_by_the_captured_handle(monkeypatch):
    """a cleared remote must not disable the resource fence.

    ``record_resource`` skips its compare-and-set when ``resource_identity`` is None. deriving that
    identity from the live status produced None once confirmed teardown cleared ``status.remote``,
    so an in-flight poll could publish a later ``running`` projection onto a run whose resource is
    already gone. the identity must come from the handle the poller was started for, as RunPod does.
    """
    from flash.providers._lifecycle.instances import poll_instance
    from flash.runner.lifecycle import status as status_ops

    passed = []
    monkeypatch.setattr(
        status_ops,
        "record_resource",
        lambda _run_id, _payload, **kwargs: passed.append(kwargs.get("resource_identity")),
    )
    # the run has already been torn down: status.remote is cleared.
    monkeypatch.setattr(status_ops, "get_status", lambda _run_id: SimpleNamespace(remote=None))
    captured = ("lambda", 0, 1, "i-1", "gpu_1x_a100", "us-west-1", "flash-run")
    adapter = SimpleNamespace(
        run_id="run-1",
        current_attempt=0,
        fence=1,
        provider="lambda",
        instance_id="i-1",
        running_status="active",
        dead_states=frozenset({"terminated"}),
        resource_identity=captured,
    )

    poll_instance._record_resource(adapter, "active", confirmed_missing=False)

    assert passed == [captured], "the projection must fence on the captured handle, not the status"


def test_transport_failure_sleeps_only_the_tracker_backoff(monkeypatch):
    """one transport failure must produce one wait, not two.

    ``PollErrorTracker.record`` already sleeps its escalating bounded backoff. a second sleep in
    the caller doubles the wait before the ``poll_error`` verdict, so a run whose provider status
    API is failing holds a paid resource roughly seven extra intervals before the poller gives up.
    """
    from flash.providers._lifecycle.instances import poll_instance

    jobs, _poll = _wire_lambda_poll(
        monkeypatch,
        attempt=_instance_attempt(provider="lambda", work=200.0, result=220.0),
        results=[None] * 40,
    )

    from flash.providers.lambda_.client import api as lambda_api

    def boom(*_args, **_kwargs):
        raise lambda_api.LambdaApiError("provider status unavailable")

    monkeypatch.setattr(lambda_api, "get_instance", boom)
    # hold the clock well inside every deadline so the error tracker, not a deadline, ends the loop.
    monkeypatch.setattr(poll_instance.time, "time", lambda: 50.0)
    # patch after the wiring helper, which stubs sleep on this same shared module object.
    sleeps = []
    monkeypatch.setattr(poll_instance.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = jobs.poll_lambda_job(_handle(), _spec(), seed=0, interval_s=10)

    assert result.failure == "poll_error"
    # exactly the tracker's escalating backoff for failures 1..7; the 8th returns without sleeping.
    # a flat extra interval per failure here would mean the caller slept a second time.
    assert sleeps == [10, 20, 30, 40, 50, 60, 60], sleeps


def test_provider_initial_and_reattached_poll_keep_same_deadline(monkeypatch):
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.core.base import JobHandle, PollResult
    from flash.providers.lambda_.execution.provider import LambdaProvider

    captured = []

    def fake_poll(_handle, _spec, _seed, *, log=None, deadline_at=None):
        captured.append(deadline_at)
        return PollResult(True, metrics={})

    monkeypatch.setattr(jobs, "usable_instances", lambda _gpu, **_kwargs: [_inst()])
    monkeypatch.setattr(
        jobs, "launch_and_submit", lambda *_args, **_kwargs: _handle(started_ts=1.0)
    )
    monkeypatch.setattr(jobs, "poll_lambda_job", fake_poll)
    monkeypatch.setattr(
        "flash.providers.lambda_.client.api.terminate_instance_confirmed",
        lambda _instance_id: None,
    )
    provider = LambdaProvider()
    spec = _spec()

    assert provider.submit_run(
        spec,
        seed=0,
        source_snapshot=SOURCE_SNAPSHOT,
        _deadline_at=12_345.0,
    ).ok
    handle = JobHandle.from_dict(_handle(started_ts=1.0).to_dict())
    assert provider.poll(handle, spec, seed=0, _deadline_at=12_345.0).ok

    assert captured == [12_345.0, 12_345.0]


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
    res = _submit(jobs, _spec(), seed=0, on_handle=handles.append)
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

    res = _submit(jobs, _spec(), seed=0, on_handle=handles.append)

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
        _submit(jobs, _spec(), seed=0)


def test_runner_terminates_on_failure_and_exception(monkeypatch):
    from flash.providers.core.base import PollResult

    jobs, terminated, _ = _wire_runner(monkeypatch, PollResult(False, failure="job_preempted"))
    res = _submit(jobs, _spec(), seed=0)
    assert not res.ok
    assert terminated == [["i-9999"]]

    jobs, terminated, _ = _wire_runner(monkeypatch, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        _submit(jobs, _spec(), seed=0)
    assert terminated == [["i-9999"]]


def test_runner_terminates_when_handle_persist_fails(monkeypatch):
    """The launched instance is terminated even if on_handle raises — the teardown finally guards
    everything after the launch, not just the poll."""
    jobs, terminated, _ = _wire_runner(monkeypatch, None)

    def boom(_h):
        raise RuntimeError("status store unreachable")

    with pytest.raises(RuntimeError, match="status store unreachable"):
        _submit(jobs, _spec(), seed=0, on_handle=boom)
    assert terminated == [["i-9999"]]


def test_submit_rejects_policy_word_gpu():
    """submit_run_lambda needs a concrete class; a policy word ("cheapest") — which the allocator
    resolves upstream — must fail with a clear error, not an opaque KeyError."""
    from flash.providers.lambda_.client import api as lambda_api
    from flash.providers.lambda_.jobs import submit_run_lambda

    spec = _spec()
    object.__setattr__(spec.gpu, "type", "cheapest")
    with pytest.raises(lambda_api.LambdaApiError, match="concrete gpu class"):
        submit_run_lambda(spec, seed=0)


# ---------------------------------------------------------------------------
# labels, gc, orphan sweep
# ---------------------------------------------------------------------------
def test_instance_label_always_sweepable():
    from flash.providers.lambda_.jobs.builders import instance_label

    assert instance_label("flash-1700-abcd", 0, 1) == "flash-1700-abcd-s0-a1"
    assert instance_label("fail-fast", 0, 0) == "flash-fail-fast-s0-a0"  # prefix forced


def test_instance_label_bounds_seed_and_attempt():
    """The seed/attempt suffix is the only caller-supplied text appended after the (already-bounded)
    run prefix: an absurd seed OR attempt (or a non-int) must NOT push the name past the 60-char
    provider cap, which would get the name silently truncated and desync it from the sweep-matched
    prefix. BOTH numeric fields are bounded so the WHOLE suffix stays <= _SUFFIX_BUDGET."""
    from flash.providers._lifecycle.instances.instance import (
        _MAX_NAME,
        _SUFFIX_BUDGET,
        run_label_prefix,
    )
    from flash.providers.lambda_.jobs.builders import instance_label

    def suffix_of(rid, label):
        return label[len(run_label_prefix(rid)) :]

    # Normal small ids: unchanged.
    assert instance_label("flash-1700000000-abcd1234", 0, 0) == "flash-1700000000-abcd1234-s0-a0"
    # Absurdly large seed: name stays within the cap (and keeps the -a boundary).
    rid = "flash-1700000000-abcd1234"
    huge = instance_label(rid, 123456789012345, 0)
    assert len(huge) <= _MAX_NAME
    assert len(suffix_of(rid, huge)) <= _SUFFIX_BUDGET
    assert "-a0" in huge
    # Absurdly large ATTEMPT (corrupt) alone: still bounded (the earlier fix only trimmed seed).
    huge_att = instance_label(rid, 0, 999999999999)
    assert len(huge_att) <= _MAX_NAME
    assert len(suffix_of(rid, huge_att)) <= _SUFFIX_BUDGET
    assert huge_att.startswith(rid + "-s")  # framing + prefix intact for sweep matching
    # BOTH seed and attempt huge together: whole suffix bounded.
    both_huge = instance_label(rid, 123456789, 987654321)
    assert len(both_huge) <= _MAX_NAME
    assert len(suffix_of(rid, both_huge)) <= _SUFFIX_BUDGET
    # A long run id AND both fields huge together still fit.
    both = instance_label("flash-" + "x" * 80, 99999999999, 7777777)
    assert len(both) <= _MAX_NAME
    # Seed formatting remains defensive, but attempt identity is strict.
    assert instance_label(rid, "weird", 0).startswith(rid + "-s0-a0")
    with pytest.raises(ValueError, match="attempt identity is invalid"):
        instance_label(rid, 0, "bad")


def test_terminate_run_instances_matches_forced_prefix(monkeypatch):
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

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


def test_run_instances_remaining_uses_exact_labels_and_exact_lookup(monkeypatch):
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import (
        api as lambda_api,
    )
    from flash.providers.lambda_.execution.provider import LambdaProvider

    run_id = "flash-100"
    rows = [
        {"id": "i-live", "name": jobs.instance_label(run_id, 0, 0)},
        {"id": "i-gone", "name": jobs.instance_label(run_id, 1, 0)},
        {"id": "i-other", "name": jobs.instance_label("flash-1000", 0, 0)},
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
        lambda *, strict: [{"id": "i-1", "name": jobs.instance_label("run1", 0, 0)}],
    )

    def lookup_failure(instance_id, *, strict):
        raise lambda_api.LambdaApiError("lookup unavailable")

    monkeypatch.setattr(lambda_api, "get_instance", lookup_failure)
    with pytest.raises(lambda_api.LambdaApiError, match="lookup unavailable"):
        provider.run_instances_remaining("run1")

    monkeypatch.setattr(
        lambda_api,
        "list_instances",
        lambda *, strict: [{"id": None, "name": jobs.instance_label("run1", 0, 0)}],
    )
    with pytest.raises(lambda_api.LambdaApiError, match="no usable id"):
        provider.run_instances_remaining("run1")


def test_handle_roundtrip():
    from flash.providers.lambda_.jobs.builders import LambdaJobHandle

    h = _handle()
    d = h.to_dict()
    assert d["provider"] == "lambda"
    assert LambdaJobHandle.from_dict(d) == h


@pytest.mark.parametrize("started_ts", [0, -1, float("inf")])
def test_handle_rejects_invalid_launch_timestamp(started_ts):
    from flash.providers.lambda_.jobs.builders import LambdaJobHandle

    with pytest.raises(ValueError, match="launch timestamp is invalid"):
        LambdaJobHandle.from_dict({**_handle().to_dict(), "started_ts": started_ts})


def test_sweep_orphans_label_safety(monkeypatch):
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

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
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

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
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

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

    import flash.providers.lambda_.jobs as jobs
    from flash.providers._lifecycle.instances.instance import instance_label
    from flash.providers._lifecycle.instances.poll import preload_instance_run_id
    from flash.providers.lambda_.client import api as lambda_api

    # Build the name the way a launch does (instance_label bounds it to the provider name budget) so the
    # reap parser is tested against the REAL, possibly-truncated VM name, not the raw run id.
    fresh = preload_instance_run_id("lambda", "us-east-1", int(time.time()) + 1800, "abcdef")
    instances = [
        {"id": "i-1", "name": instance_label(fresh, 0, 0)},  # in-deadline warm box -> KEEP
        {
            "id": "i-legacy",
            "name": "flash-preload-lambda-us-east-1-abcdef-s0-a0",
        },  # no deadline -> KEEP
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
        _launch(jobs, _spec(), seed=0, instances=insts, attempt=0, log=log)
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
    h = _launch(jobs, _spec(), seed=0, instances=[_inst()], attempt=0)
    assert h.instance_id == "i-4242"


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("terminate_confirmed", [True, False])
def test_post_launch_baseexception_cleans_and_never_walks_regions(
    monkeypatch, interrupt_type, terminate_confirmed
):
    # submit_run_lambda's finally only exists once launch_and_submit RETURNS a handle, so an
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
            seed=0,
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
        _launch(jobs, spec, seed=0, instances=[_inst()], attempt=0)

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
        _launch(jobs, _spec(), seed=0, instances=[_inst()], attempt=0)

    assert terminated == ["i-4242"]  # the helper's own exact cleanup ran
    assert reaped == []  # the outer coarse label reap must not also fire


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_cacheless_retry_success_say_baseexception_does_not_trigger_run_wide_reap(
    monkeypatch, interrupt_type
):
    """The same BaseException-during-say property as the primary success route, but through the
    cache-less retry (_retry_launch_without_cache): once it is entered it owns the exact cleanup
    for whatever box it rents internally, so the outer run-wide reap must not also fire."""
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])

    def fake_launch(*, file_system_names=None, **_kwargs):
        if file_system_names:  # the cached attempt is rejected for a filesystem-attach reason
            raise lambda_api.LambdaApiError(
                "POST /instance-operations/launch -> HTTP 400: file_system_names not attachable"
            )
        return "i-cold"  # the cache-less retry succeeds

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)
    monkeypatch.setattr(
        lambda_api, "ensure_filesystem", lambda n, r, deadline_at=None: f"/lambda/nfs/{n}"
    )

    def raising_say(_log):
        def _say(msg):
            if "cold, cache-less" in msg:  # only the retry's own success message raises
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
        _launch(
            jobs,
            _spec(network_volume="flash-weights"),
            seed=0,
            instances=[_inst(region="us-east-1")],
            attempt=0,
        )

    assert terminated == ["i-cold"]  # the cache-less retry's own exact cleanup ran
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
            seed=0,
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
        _launch(jobs, _spec(), seed=0, instances=[_inst()], attempt=0)

    assert reaped == []  # nothing was rented, so no seed of this run may be terminated
    assert terminated == []


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_cacheless_clean_reject_say_baseexception_does_not_trigger_run_wide_reap(
    monkeypatch, interrupt_type
):
    """A CLEAN cold rejection rents nothing, so a raising diagnostic on that path must not reap.

    The cache-less retry arms the coarse guard around its own launch request. When that request is
    cleanly rejected the guard has to stand down BEFORE the rejection is logged: the log stream can
    be closed, and an armed guard on that path sweeps the run label and terminates every other
    concurrent seed sharing it over a request that rented nothing."""
    import flash.providers.lambda_.jobs as jobs
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])

    def fake_launch(*, file_system_names=None, **_kwargs):
        if file_system_names:  # the cached attempt is rejected for a filesystem-attach reason
            raise lambda_api.LambdaApiError(
                "POST /instance-operations/launch -> HTTP 400: file_system_names not attachable"
            )
        # the cache-less retry is CLEANLY rejected too: HTTP 4xx, nothing rented
        raise lambda_api.LambdaApiError(
            "POST /instance-operations/launch -> HTTP 400: no capacity in region"
        )

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)
    monkeypatch.setattr(
        lambda_api, "ensure_filesystem", lambda n, r, deadline_at=None: f"/lambda/nfs/{n}"
    )

    def raising_say(_log):
        def _say(msg):
            if "also rejected cold" in msg:  # only the cold-rejection diagnostic raises
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
        _launch(
            jobs,
            _spec(network_volume="flash-weights"),
            seed=0,
            instances=[_inst(region="us-east-1")],
            attempt=0,
        )

    assert reaped == []  # nothing was rented, so no run-label sweep may fire
    assert terminated == []


# ---------------------------------------------------------------------------
# #228 follow-up: don't mask worker failures + keep large specs out of user_data
# ---------------------------------------------------------------------------
def test_bootstrap_propagates_nonzero_worker_exit(monkeypatch):
    lb, calls = _bootstrap_env(monkeypatch, rc=1)
    assert lb.main() == 1
    assert calls == ["sft"]


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
            "fence": 1,
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


def test_main_stops_when_spilled_spec_fetch_fails(monkeypatch):
    from flash.providers._lifecycle.bootstrapping import bootstrap as lb

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
            "fence": 1,
        },
    )
    monkeypatch.setattr(lb, "fetch_code", lambda p: None)
    monkeypatch.setattr(
        lb, "fetch_spec_from_hf", lambda p: (_ for _ in ()).throw(RuntimeError("hf 503"))
    )
    assert lb.main() == 1


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
    deadline_at = _deadline_at()
    attempt = _instance_attempt(
        provider="lambda",
        grant=deadline_at - 120.0,
        work=deadline_at - 60.0,
        result=deadline_at,
        attempt_id=7,
        fence=1,
    )
    with patch(
        "flash.runner.lifecycle.status.get_status",
        return_value=SimpleNamespace(attempt=attempt.to_dict()),
    ):
        representative = inst.build_payload(
            _spec(),
            seed=0,
            attempt=7,
            fence=1,
            arm="lambda",
            cache_host_mount="/mnt/cache",
            source_snapshot=SOURCE_SNAPSHOT,
            deadline_at=deadline_at,
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


def test_host_log_helper_starts_no_hf_request_at_deadline(monkeypatch):
    """At or after the deadline the host log helper exits before constructing HfApi.

    The same clock is replayed one second before the deadline as a positive control.
    """
    import math
    import sys
    import types

    def run_at(now: float) -> list:
        calls = []

        class FakeApi:
            def __init__(self, token=None):
                calls.append("init")

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

        _run_capsule_member(
            "hostlog.py", {"json": json, "math": math, "time": time, "open": fake_open}
        )
        return calls

    assert run_at(200.0) == []
    assert run_at(199.0) == ["init"]


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
        _launch(jobs, _spec(), seed=0, instances=[_inst()], attempt=0)

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
