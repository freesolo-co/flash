"""Test the Vast container lifecycle, polling, teardown, and orphan sweep.

The Vast API and HF readers are mocked; no API key is needed.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import time
import urllib.error
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from flash.core.spec import JobSpec
from tests._helpers.source_snapshot import valid_source_snapshot

SOURCE_SNAPSHOT = valid_source_snapshot()


def _spec(gpu_type="RTX 4090", **gpu_kw) -> JobSpec:
    gpu = {"type": gpu_type, "max_wall_seconds": 3600, **gpu_kw}
    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "sft",
            # authoritative seed 0 matches the literal seed threaded into every provider call below.
            "seed": 0,
            "run_id": "flash-1700000000-abcd1234",
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
        provider="vast",
        grant=deadline_at - 120.0,
        work=deadline_at - 60.0,
        result=deadline_at,
        attempt_id=attempt_id,
        fence=kwargs["fence"],
    )
    with patch("flash.runner.lifecycle.status.get_status", return_value=SimpleNamespace(attempt=attempt.to_dict())):
        return builders.build_payload(*args, **kwargs)


def _deploy(vast, *args, **kwargs):
    if "deadline_at" not in kwargs:
        kwargs["deadline_at"] = _deadline_at()
    kwargs.setdefault("source_snapshot", SOURCE_SNAPSHOT)
    kwargs.setdefault("fence", 1)
    deadline_at = kwargs["deadline_at"]
    attempt = _instance_attempt(
        provider="vast",
        grant=deadline_at - 120.0,
        work=deadline_at - 60.0,
        result=deadline_at,
        attempt_id=kwargs.get("attempt", 0),
        fence=kwargs["fence"],
    )
    with patch("flash.runner.lifecycle.status.get_status", return_value=SimpleNamespace(attempt=attempt.to_dict())):
        return vast.deploy_and_submit(*args, **kwargs)


def _submit(vast, *args, **kwargs):
    if "deadline_at" not in kwargs:
        kwargs["deadline_at"] = _deadline_at()
    kwargs.setdefault("source_snapshot", SOURCE_SNAPSHOT)
    kwargs.setdefault("fence", 1)
    deadline_at = kwargs["deadline_at"]
    attempt = _instance_attempt(
        provider="vast",
        grant=deadline_at - 120.0,
        work=deadline_at - 60.0,
        result=deadline_at,
        attempt_id=kwargs.get("attempt", 0),
        fence=kwargs["fence"],
    )
    with patch("flash.runner.lifecycle.status.get_status", return_value=SimpleNamespace(attempt=attempt.to_dict())):
        return vast.submit_run_vast(*args, **kwargs)


def _offer(**kw):
    from tests._helpers.vast import make_vast_offer

    return make_vast_offer(**kw)


def _handle(started_ts=10_000.0, rate=0.47, attempt=0):
    from flash.providers.vast.jobs.builders import VastJobHandle

    return VastJobHandle(
        instance_id=9999,
        offer_id=1,
        machine_id=10,
        label=f"flash-x-s0-a{attempt}",
        gpu="RTX 4090",
        hourly_usd=rate,
        attempt=attempt,
        fence=1,
        started_ts=started_ts,
    )


# ---------------------------------------------------------------------------
# container onstart + shared bootstrap
# ---------------------------------------------------------------------------
def test_onstart_ships_payload_and_runs_shared_bootstrap(monkeypatch):
    from flash.providers.vast.jobs import builders

    monkeypatch.setenv("VAST_API_KEY", "vk-supersecret")
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
    assert payload["flash_arm"] == "vast"
    # The worker env's HF_REPO is sourced from the run's [train] hf_repo (not an operator default).
    assert payload["env"]["HF_REPO"] == "org/repo"

    script = builders.build_onstart(payload)
    # payload travels base64-encoded inside a quoted heredoc, byte-exact
    b64 = script.split("FLASH_PAYLOAD_EOF")[1].strip()
    assert json.loads(base64.b64decode(b64)) == payload
    # the SHARED instance bootstrap travels as a VERIFIED capsule and is run as the container
    # command. Vast and Lambda ship the same profile, so both get the same members.
    from flash.providers._lifecycle.instances.instance import _instance_capsule
    from flash.runtime_capsule import read_capsule, sha256_bytes

    capsule_b64, capsule_sha256 = _instance_capsule()
    archive = base64.b64decode(capsule_b64)
    assert capsule_sha256 in script
    assert sha256_bytes(archive) == capsule_sha256
    assert "sha256sum -c" in script
    assert "/root/flash/capsule.pyz bootstrap" in script
    # it is genuinely the shared module (a distinctive line only that file has), asserted against
    # the SHIPPED member rather than the launch text: the capsule is compressed, so the marker no
    # longer appears verbatim in the script and a substring check there would be vacuous.
    from pathlib import Path

    import flash.providers._lifecycle.bootstrapping.bootstrap as ib

    _manifest, contents = read_capsule(archive)
    shared_src = Path(ib.__file__).read_text()
    assert "RetriableBootstrapError" in shared_src  # sanity: distinctive marker exists
    assert contents["bootstrap.py"].decode() == shared_src  # ...and it was shipped, byte for byte
    # every sibling the bootstrap imports as a bare module rides along; a missing one is a
    # ModuleNotFoundError on a box that is already rented and billing.
    for sibling in ("bootstrap_secrets.py", "bootstrap_console.py", "bootstrap_pip.py"):
        assert sibling in contents, sibling
    # the operator's Vast key NEVER ships to the box; the worker HF token rides inside the base64
    # payload's env (like RunPod), never interpolated raw into the shell.
    assert "vk-supersecret" not in script
    assert payload["env"]["HF_TOKEN"] == "hf-worker-token"
    # self-destroy backstop uses the instance-scoped CONTAINER_API_KEY, not the operator key
    assert "CONTAINER_API_KEY" in script
    assert "console.vast.ai/api/v0/instances/" in script
    # no base training-stack install (the worker image is baked); only the bootstrap's per-run extra_pip
    assert "torch==" not in script


def test_onstart_heredoc_terminators_on_own_line_and_python_fallback(monkeypatch):
    """every heredoc terminator must start on its own line (embedded content without a trailing
    newline would otherwise swallow the rest of the script), and the python-interpreter resolution
    must fall back past python3 to python with a clear diagnostic."""
    from flash.providers.vast.jobs import builders

    monkeypatch.setenv("VAST_API_KEY", "vk")
    monkeypatch.setenv("HF_TOKEN", "hf")
    script = builders.build_onstart(_build_payload(builders, _spec(), seed=0, attempt=1))
    # Derived from the script's own OPENING terminators rather than a hardcoded list, so a heredoc
    # added later is covered here instead of truncating a launch script in production.
    opened = set(re.findall(r"<<'(FLASH_\w+)'", script))
    assert opened, "expected the onstart to embed at least one heredoc"
    for term in sorted(opened):
        assert f"\n{term}\n" in script, f"{term} terminator must be on its own line"


def test_capsule_ships_every_bare_sibling_the_bootstrap_imports(monkeypatch):
    """The capsule must carry EVERY sibling module bootstrap.py imports when run as a bare script.

    The bare-script imports are unconditional (``__package__`` is empty off-package, which is how
    the capsule runs it), so a sibling the profile forgets is not a degraded install: the bootstrap
    dies with ModuleNotFoundError before any work starts, on a box already rented and billing, on
    every run.

    Derived from the bootstrap's own ``else:`` branch rather than a hardcoded list, so adding a
    fourth imported module fails here instead of in production.
    """
    import ast
    from pathlib import Path

    from flash.providers._lifecycle.instances.instance import (
        INSTANCE_BOOTSTRAP_PROFILE,
        _instance_capsule,
    )
    from flash.providers.vast.jobs import builders
    from flash.runtime_capsule import get_profile, read_capsule

    lifecycle = Path(builders.__file__).parent.parent.parent / "_lifecycle"
    # the bare-script branch imports the capsule's DESTINATION names (bootstrap_secrets, ...),
    # which the profile maps from bootstrapping/{secrets,console,pip}.py. Resolve a bare name
    # against those destinations rather than against a sibling file, which no longer exists.
    shipped = {dest for _src, dest in get_profile(INSTANCE_BOOTSTRAP_PROFILE).sources}

    def _is_shipped_sibling(name: str) -> bool:
        return f"{name}.py" in shipped

    tree = ast.parse((lifecycle / "bootstrapping" / "bootstrap.py").read_text())
    required: set[str] = set()
    for node in ast.walk(tree):
        # the bare-script branch of `if __package__:` -- plain `import x` and `from x import ...`
        # whose module is a file sitting next to bootstrap.py.
        if isinstance(node, ast.Import):
            required |= {a.name for a in node.names if _is_shipped_sibling(a.name)}
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
            and _is_shipped_sibling(node.module)
        ):
            required.add(node.module)
    assert required, "expected bootstrap.py to import at least one bare sibling"

    capsule_b64, _sha = _instance_capsule()
    _manifest, contents = read_capsule(base64.b64decode(capsule_b64))
    for module in sorted(required):
        assert f"{module}.py" in contents, (
            f"bootstrap.py imports {module} as a bare sibling but the capsule never ships it"
        )
    # the entrypoint the launch scripts invoke is the bootstrap itself, so the imports above are the
    # ones that actually run.
    assert get_profile(INSTANCE_BOOTSTRAP_PROFILE).entrypoint == "bootstrap.py"

    monkeypatch.setenv("VAST_API_KEY", "vk")
    monkeypatch.setenv("HF_TOKEN", "hf")
    script = builders.build_onstart(_build_payload(builders, _spec(), seed=0, attempt=1))
    # PYBIN never silently empty: python fallback + a diagnostic when nothing resolves.
    assert "command -v python3 || command -v python" in script
    assert "no python interpreter" in script
    # an empty PYBIN must EXIT (after a log-retrieval hold), not fall through to the
    # doomed `"$PYBIN"` bootstrap + self-destroy invocations.
    assert 'if [ -z "$PYBIN" ]; then' in script
    assert "exit 1" in script


@pytest.mark.parametrize("corrupt_capsule", [False, True])
def test_onstart_self_destroys_even_when_the_capsule_fails_verification(
    monkeypatch, tmp_path, corrupt_capsule
):
    """A capsule that fails its digest check must still reach the self-destroy backstop.

    This is the failure mode the verification itself introduces: refusing to execute an unverified
    capsule is correct, but a refusal that exits BEFORE the self-destroy leaves a rented GPU billing
    until the control plane notices. The script is EXECUTED here (with bash) rather than pattern
    matched, because the property is control flow -- `set -e`, an early `exit`, or a `&&` chain
    anywhere above the backstop would break it while every substring assertion still passed.
    """
    from flash.providers.vast.jobs import builders

    monkeypatch.setenv("VAST_API_KEY", "vk")
    monkeypatch.setenv("HF_TOKEN", "hf")
    script = builders.build_onstart(_build_payload(builders, _spec(), seed=0, attempt=1))

    # redirect the box's absolute paths into a sandbox, and replace the 10-minute log-retrieval hold
    # (which runs on the failure path) with a marker so the test does not sleep.
    root = tmp_path / "root" / "flash"
    script = script.replace("/root/flash", str(root)).replace("sleep 600", "echo FLASH_TEST_HELD")
    if corrupt_capsule:
        # a single flipped base64 character: the archive decodes to different bytes, so the digest
        # check is what must catch it.
        marker = "cat > " + str(root) + "/capsule.b64 <<'FLASH_CAPSULE_EOF'\n"
        head, _, tail = script.partition(marker)
        assert head, "the capsule heredoc marker moved"
        assert tail, "the capsule heredoc marker moved"
        first = "B" if tail[0] != "B" else "C"
        script = head + marker + first + tail[1:]

    # the bootstrap would try to rent-time install and fetch from HF, which is not what this test is
    # about: swap the capsule INVOCATION for a marker write. The verification above it is untouched,
    # so the corrupt case still never reaches this line.
    ran = root / "capsule_ran"
    script = script.replace(
        f'"$PYBIN" {root}/capsule.pyz bootstrap',
        f"touch {ran}",
    )
    # the self-destroy is the property under test, so it must run -- but against a local file rather
    # than the real vast API. It writes a marker instead of issuing the DELETE.
    destroyed = tmp_path / "destroyed"
    script = script.replace(
        "urllib.request.urlopen(req, timeout=30)",
        f"open({str(destroyed)!r}, 'w').write(iid)",
    )
    script_path = tmp_path / "onstart.sh"
    script_path.write_text(script)

    proc = subprocess.run(
        ["bash", str(script_path)],
        capture_output=True,
        text=True,
        timeout=180,
        env={
            **os.environ,
            "CONTAINER_ID": "42",
            "CONTAINER_API_KEY": "instance-scoped",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )

    # the self-destroy ran in BOTH cases -- that is the whole point. A billing box must be released
    # whether the capsule verified or not.
    assert destroyed.exists(), proc.stderr[-3000:]
    assert destroyed.read_text() == "42", proc.stderr[-3000:]
    if corrupt_capsule:
        # ...and the corrupted capsule was refused BEFORE it executed, with a non-zero exit.
        assert "runtime capsule failed verification" in proc.stderr, proc.stderr[-3000:]
        assert proc.returncode != 0
        assert not ran.exists(), "an unverified capsule was executed anyway"
    else:
        assert ran.exists(), "the verified capsule never ran, so the control case proves nothing"
        assert proc.returncode == 0


def test_onstart_spills_large_spec_to_hf(monkeypatch):
    """a large inline job spec is spilled to HF (parity with Lambda's build_user_data)
    so it never inflates the base64 onstart past Vast's exec-arg / onstart length limit and fails the
    rent before a handle is persisted. A small spec rides inline unchanged."""
    import huggingface_hub

    from flash.providers.vast.jobs import builders

    uploaded = {}

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type):
            uploaded.update(path=path_in_repo, repo=repo_id, type=repo_type)

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    big = "x" * 20_000  # > _SPEC_SPILL_THRESHOLD
    payload = {
        "job_spec_json": big,
        "hf_prefix": "sft/run/seed0",
        "hf_repo": "org/repo",
        "env": {"HF_TOKEN": "t"},
        "flash_arm": "vast",
    }
    script = builders.build_onstart(payload)
    assert big not in script  # the giant spec is NOT embedded inline...
    assert (
        uploaded["path"] == "sft/run/seed0/job_spec.json"
    )  # ...it was spilled to the dataset repo
    assert uploaded["type"] == "dataset"
    b64 = script.split("FLASH_PAYLOAD_EOF")[1].strip()
    embedded = json.loads(base64.b64decode(b64))
    assert embedded["job_spec_in_hf"] is True
    assert embedded["job_spec_json"] == ""


def test_build_payload_sets_vast_arm():
    """build_payload stamps flash_arm='vast' so the metrics record attributes the substrate, and the
    shared bootstrap turns it into FLASH_ARM."""
    from flash.providers._lifecycle.bootstrapping import bootstrap as ib
    from flash.providers.vast.jobs import builders

    payload = _build_payload(builders, _spec(), 0, 0, deadline_at=_deadline_at())
    assert payload["flash_arm"] == "vast"
    assert payload["source_snapshot"] == SOURCE_SNAPSHOT
    assert "code_prefix" not in payload
    env = ib.build_worker_env(
        {
            "job_spec_json": "{}",
            "phase": "sft",
            "seed": 0,
            "attempt": 0,
            "fence": 1,
            "run_id": "run-1",
            "env": {},
            "source_snapshot": SOURCE_SNAPSHOT,
            "flash_arm": "vast",
        }
    )
    assert env["FLASH_ARM"] == "vast"


# ---------------------------------------------------------------------------
# deploy_and_submit: offer (market) walk
# ---------------------------------------------------------------------------
def test_deploy_walks_taken_offers(monkeypatch):
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    rented = []

    def fake_create(offer_id, **kw):
        if offer_id < 3:
            raise vast_api.VastCreateRejected(f"offer {offer_id} taken")
        rented.append(offer_id)
        return 4242

    monkeypatch.setattr(vast_api, "create_instance", fake_create)
    offers = [_offer(offer_id=i, machine_id=i, dph_total=0.20 + i * 0.01) for i in (1, 2, 3)]
    h = _deploy(vast, _spec(), seed=0, offers=offers, attempt=2)
    assert rented == [3]
    assert h.instance_id == 4242
    assert h.offer_id == 3
    assert h.label == "flash-1700000000-abcd1234-s0-a2"


def test_deploy_walks_documented_non_2xx_rejection(monkeypatch):
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    cause = urllib.error.HTTPError(
        "https://console.vast.ai/api/v0/asks/1/",
        404,
        "not found",
        None,
        io.BytesIO(b'{"success": false, "error": "offer unavailable"}'),
    )
    first = vast_api.VastApiError("vast request failed: HTTP 404: offer unavailable")
    first.__cause__ = cause
    responses = iter([first, {"success": True, "new_contract": 4242}])
    requested = []

    def request(path, **kwargs):
        requested.append(path)
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(vast_api, "request_with_retries", request)
    offers = [_offer(offer_id=1, machine_id=1), _offer(offer_id=2, machine_id=2)]

    handle = _deploy(vast, _spec(), seed=0, offers=offers, attempt=0)

    assert handle.instance_id == 4242
    assert handle.offer_id == 2
    assert requested == ["/v0/asks/1/", "/v0/asks/2/"]


@pytest.mark.parametrize("destroy_confirmed", [True, False])
def test_deploy_stops_on_contradictory_create_response(monkeypatch, destroy_confirmed):
    from flash.providers.core.base import UnreconciledCreateError
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    created = []

    def create(offer_id, **kwargs):
        created.append(offer_id)
        err = vast_api.VastAmbiguousCreate("contradictory rejection with contract evidence 777")
        err.contract_id = 777
        raise err

    monkeypatch.setattr(vast_api, "create_instance", create)
    monkeypatch.setattr(vast_api, "list_instances", list)
    destroyed_exact = []
    monkeypatch.setattr(
        vast_api,
        "destroy_instance",
        lambda iid: destroyed_exact.append(iid) or destroy_confirmed,
    )
    destroyed_for = []
    monkeypatch.setattr(
        vast, "destroy_run_instances", lambda run_id: destroyed_for.append(run_id) or []
    )
    offers = [_offer(offer_id=1, machine_id=1), _offer(offer_id=2, machine_id=2)]

    with pytest.raises(UnreconciledCreateError, match="aborting the offer walk"):
        _deploy(vast, _spec(), seed=0, offers=offers, attempt=0)

    assert created == [1]
    # the contradictory response's known contract id is destroyed directly, even when the
    # eventually-consistent listing shows nothing under the label yet
    assert destroyed_exact == [777]
    expected_fallback = [] if destroy_confirmed else [_spec().run_id]
    assert destroyed_for == expected_fallback


def test_deploy_refuses_primary_creation_below_minimum_deadline_allowance(monkeypatch):
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setattr(vast.time, "time", lambda: 100.0)
    created = []
    monkeypatch.setattr(
        vast_api,
        "create_instance",
        lambda *_args, **_kwargs: created.append(True) or 4242,
    )

    with pytest.raises(RuntimeError, match="60-second minimum provider allowance"):
        _deploy(vast, _spec(), seed=0, offers=[_offer(offer_id=1)], attempt=0, deadline_at=159.0)

    assert created == []


def test_deploy_success_log_failure_does_not_leak_handle(monkeypatch):
    # once create_instance rents the box, a raising success log before handle return must not leak it
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setattr(vast_api, "create_instance", lambda offer_id, **kw: 4242)

    def raising_say(_log):
        def _say(_msg):
            raise OSError("log stream closed")

        return _say

    monkeypatch.setattr(vast, "make_say", raising_say)
    h = _deploy(vast, _spec(), seed=0, offers=[_offer(offer_id=1)], attempt=0)
    assert h.instance_id == 4242


@pytest.mark.parametrize("argument_builder", ["vast_image", "_effective_disk_gb"])
def test_deploy_argument_failure_precedes_create_and_cleanup(monkeypatch, argument_builder):
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    original = RuntimeError(f"{argument_builder} failed")

    def fail_argument(*args, **kwargs):
        raise original

    create_requests = []
    run_reaps = []
    monkeypatch.setattr(vast, argument_builder, fail_argument)
    monkeypatch.setattr(
        vast_api,
        "create_instance",
        lambda *args, **kwargs: create_requests.append((args, kwargs)),
    )
    monkeypatch.setattr(
        vast, "destroy_run_instances", lambda run_id: run_reaps.append(run_id) or []
    )

    with pytest.raises(RuntimeError) as exc_info:
        _deploy(vast, _spec(), seed=0, offers=[_offer(offer_id=1)], attempt=0)

    assert exc_info.value is original
    assert create_requests == []
    assert run_reaps == []


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("destroy_confirmed", [True, False])
def test_post_create_baseexception_cleans_and_never_walks_offers(
    monkeypatch, interrupt_type, destroy_confirmed
):
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    spec = _spec()
    offers = [_offer(offer_id=1, machine_id=1), _offer(offer_id=2, machine_id=2)]
    created = []

    def create(offer_id, **kwargs):
        created.append(offer_id)
        return 4242

    monkeypatch.setattr(vast_api, "create_instance", create)
    monkeypatch.setattr(vast, "usable_offers", lambda *args, **kwargs: offers)

    def time_after_create():
        if created:
            raise interrupt_type("stop")
        return 100.0

    monkeypatch.setattr(vast.time, "time", time_after_create)
    destroyed_ids = []
    monkeypatch.setattr(
        vast_api,
        "destroy_instance",
        lambda instance_id: destroyed_ids.append(instance_id) or destroy_confirmed,
    )
    reconciled_runs = []
    monkeypatch.setattr(
        vast, "destroy_run_instances", lambda run_id: reconciled_runs.append(run_id) or []
    )

    with pytest.raises(interrupt_type):
        _submit(vast, spec, seed=0, deadline_at=20_000.0)

    assert created == [1]
    assert destroyed_ids == [4242]
    expected_fallback = [] if destroy_confirmed else [spec.run_id]
    assert reconciled_runs == expected_fallback


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_post_create_preserves_original_baseexception_when_cleanup_raises(
    monkeypatch, interrupt_type
):
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    class ExactCleanupFailure(BaseException):
        pass

    class LabelCleanupFailure(BaseException):
        pass

    spec = _spec()
    offers = [_offer(offer_id=1, machine_id=1), _offer(offer_id=2, machine_id=2)]
    created = []

    def create(offer_id, **kwargs):
        created.append(offer_id)
        return 4242

    original = interrupt_type("original interruption")
    exact_cleanup = ExactCleanupFailure("exact cleanup failed")
    label_cleanup = LabelCleanupFailure("label cleanup failed")
    destroyed_ids = []
    reconciled_runs = []

    def destroy_exact(instance_id):
        destroyed_ids.append(instance_id)
        raise exact_cleanup

    def destroy_label(run_id):
        reconciled_runs.append(run_id)
        raise label_cleanup

    monkeypatch.setattr(vast_api, "create_instance", create)
    monkeypatch.setattr(vast, "usable_offers", lambda *args, **kwargs: offers)

    def time_after_create():
        if created:
            raise original
        return 100.0

    monkeypatch.setattr(vast.time, "time", time_after_create)
    monkeypatch.setattr(vast_api, "destroy_instance", destroy_exact)
    monkeypatch.setattr(vast, "destroy_run_instances", destroy_label)

    with pytest.raises(interrupt_type) as exc_info:
        _submit(vast, spec, seed=0, deadline_at=20_000.0)

    assert exc_info.value is original
    assert created == [1]
    assert destroyed_ids == [4242]
    assert reconciled_runs == [spec.run_id]


def test_deploy_refreshes_once_when_all_taken(monkeypatch):
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    def fake_create(offer_id, **kw):
        if offer_id != 99:
            raise vast_api.VastCreateRejected("taken")
        return 7

    monkeypatch.setattr(vast_api, "create_instance", fake_create)
    # the refresh re-search returns a fresh in-pool offer (same class) the walk then rents
    monkeypatch.setattr(
        vast, "usable_offers", lambda *a, **k: [_offer(offer_id=99, machine_id=99, gpu="RTX 4090")]
    )
    h = _deploy(vast, _spec(), seed=0, offers=[_offer(offer_id=1)], attempt=0)
    assert h.instance_id == 7
    assert h.offer_id == 99


def test_deploy_refresh_widens_the_page_it_filters(monkeypatch):
    """The refresh must not ask for a page its own exclusion can fill.

    ``search_offers`` caps rows SERVER-side on a price-sorted prefix and the burned machines are
    dropped CLIENT-side afterwards, so the exclusion this refresh exists to apply is exactly what
    makes the default page too small. A run that has lost enough boxes to fill the cheapest page
    gets an empty result while dearer usable capacity sits just past it, and then retries keep
    re-selecting the same small set instead of reaching that capacity.

    Modelled with the real ordering (cap the rows, then exclude) rather than asserting the constant:
    a mock that excluded first would find the offer at any limit and could not fail.
    """
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    burned = set(range(300))

    def fake_create(offer_id, **kw):
        if offer_id != 999:
            raise vast_api.VastCreateRejected("taken")
        return 7

    def paged_search(min_vram_gb, disk_gb, exclude_machine_ids=frozenset(), limit=256, **kw):
        # the usable box sits at row 400, past the default cap but inside a widened one
        rows = [_offer(offer_id=i, machine_id=i, gpu="RTX 4090") for i in range(300)] + [
            _offer(offer_id=999, machine_id=999, gpu="RTX 4090")
        ]
        return [o for o in rows[:limit] if o.machine_id not in exclude_machine_ids]

    monkeypatch.setattr(vast_api, "create_instance", fake_create)
    monkeypatch.setattr(vast, "usable_offers", paged_search)
    monkeypatch.setattr(vast, "dead_machine_ids", lambda _run_id: burned)

    handle = _deploy(vast, _spec(), seed=0, offers=[_offer(offer_id=1)], attempt=0)

    assert handle.offer_id == 999  # reached past the page the exclusion had filled
    assert handle.instance_id == 7


def test_deploy_refresh_uses_transient_concrete_gpu_type(monkeypatch):
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    seen: dict[str, str] = {}

    def fake_create(offer_id, **kw):
        if offer_id != 99:
            raise vast_api.VastCreateRejected("taken")
        return 7

    def capture(min_vram_gb, disk_gb, *a, gpu_type="", **k):
        seen["gpu_type"] = gpu_type
        return [_offer(offer_id=99, machine_id=99, gpu="RTX 4090")]

    monkeypatch.setattr(vast_api, "create_instance", fake_create)
    monkeypatch.setattr(vast, "usable_offers", capture)

    _deploy(vast, _spec(), seed=0, offers=[_offer(offer_id=1)], attempt=0)
    assert seen["gpu_type"] == "RTX 4090"


def test_deploy_rechecks_deadline_before_refreshed_offer_creation(monkeypatch):
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    now = {"value": 100.0}
    monkeypatch.setattr(vast.time, "time", lambda: now["value"])
    created = []

    def fake_create(offer_id, **_kwargs):
        created.append(offer_id)
        now["value"] = 141.0
        raise vast_api.VastCreateRejected("offer taken")

    monkeypatch.setattr(vast_api, "create_instance", fake_create)
    monkeypatch.setattr(
        vast, "usable_offers", lambda *a, **k: [_offer(offer_id=99, machine_id=99, gpu="RTX 4090")]
    )

    with pytest.raises(RuntimeError, match="60-second minimum provider allowance"):
        _deploy(vast, _spec(), seed=0, offers=[_offer(offer_id=1)], attempt=0, deadline_at=200.0)

    assert created == [1]


def test_deploy_adopts_instance_after_ambiguous_create(monkeypatch):
    # a 5xx/timeout on the NON-IDEMPOTENT create may have made a billed contract. The walk must
    # reconcile by our unique label and ADOPT it, not rent the next offer (double-billing).
    import io
    import urllib.error

    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    rented = []

    def fake_create(offer_id, **kw):
        rented.append(offer_id)
        e = vast_api.VastApiError("create failed: 503")
        e.__cause__ = urllib.error.HTTPError("u", 503, "boom", None, io.BytesIO(b""))
        raise e

    monkeypatch.setattr(vast_api, "create_instance", fake_create)
    # the contract DID materialize under our exact attempt label -> list_instances surfaces it, with
    # the box's real launch epoch in start_date
    label = "flash-1700000000-abcd1234-s0-a2"
    monkeypatch.setattr(
        vast_api,
        "list_instances",
        lambda: [{"id": 555, "label": label, "start_date": 1699999000.0}],
    )
    offers = [_offer(offer_id=1, machine_id=1), _offer(offer_id=2, machine_id=2)]
    h = _deploy(vast, _spec(), seed=0, offers=offers, attempt=2)
    assert h.instance_id == 555  # adopted the existing contract, not a fresh rent
    assert h.offer_id == 1
    assert h.started_ts == 1699999000.0  # real launch time, not now
    assert rented == [1]  # did NOT walk on to offer 2 (no duplicate create)


def test_deploy_aborts_walk_when_ambiguous_create_left_nothing(monkeypatch):
    # an ambiguous failure with NO instance visible under our label must ABORT the walk (the
    # contract may exist but not be visible yet) rather than rent another offer and double-bill. the
    # abort must raise the TERMINAL UnreconciledCreateError (not a plain VastApiError that the
    # orchestrator retries as poll_error) — a phantom contract that surfaces AFTER the point-in-time
    # destroy_run_instances sweep would otherwise bill under a retry's new instance.
    import io
    import urllib.error

    from flash.providers.core.base import UnreconciledCreateError
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    rented = []

    def fake_create(offer_id, **kw):
        rented.append(offer_id)
        e = vast_api.VastApiError("create failed: provider body secret")
        e.__cause__ = urllib.error.HTTPError("u", 503, "boom", None, io.BytesIO(b""))
        raise e

    monkeypatch.setattr(vast_api, "create_instance", fake_create)
    monkeypatch.setattr(vast_api, "list_instances", list)  # nothing under our label
    # the abort must proactively destroy this run's instances (kill any phantom)
    destroyed_for = []
    monkeypatch.setattr(vast, "destroy_run_instances", lambda rid: destroyed_for.append(rid) or [])
    offers = [_offer(offer_id=1, machine_id=1), _offer(offer_id=2, machine_id=2)]
    log = io.StringIO()
    with pytest.raises(UnreconciledCreateError, match="aborting the offer walk") as exc_info:
        _deploy(vast, _spec(), seed=0, offers=offers, attempt=2, log=log)
    assert rented == [1]  # aborted after the FIRST offer — never rented offer 2
    assert destroyed_for  # destroy_run_instances was called to reap any phantom contract
    assert "provider body secret" not in str(exc_info.value)
    assert "provider body secret" not in log.getvalue()


def test_deploy_aborts_when_adopted_row_has_unparseable_id(monkeypatch):
    # Codex: in the ambiguous-create reconcile a matching Vast row can carry our EXACT label but a
    # truthy-but-NON-NUMERIC id (unexpected API shape). A bare int(adopted["id"]) would raise ValueError
    # BEFORE the terminal UnreconciledCreateError, aborting with the WRONG (orchestrator-retried) error
    # -> double-provision. An unparseable id must be treated as a phantom we couldn't cleanly adopt ->
    # FALL THROUGH to the fail-closed abort (destroy by label + UnreconciledCreateError), NOT rent on.
    import io
    import urllib.error

    from flash.providers.core.base import UnreconciledCreateError
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    rented = []

    def fake_create(offer_id, **kw):
        rented.append(offer_id)
        e = vast_api.VastApiError("create failed: 503")
        e.__cause__ = urllib.error.HTTPError("u", 503, "boom", None, io.BytesIO(b""))
        raise e

    monkeypatch.setattr(vast_api, "create_instance", fake_create)
    # a row under our EXACT attempt label, but its id is non-numeric -> cannot be adopted as a handle
    label = "flash-1700000000-abcd1234-s0-a2"
    monkeypatch.setattr(vast_api, "list_instances", lambda: [{"id": "not-an-int", "label": label}])
    destroyed_for = []
    monkeypatch.setattr(vast, "destroy_run_instances", lambda rid: destroyed_for.append(rid) or [])
    offers = [_offer(offer_id=1, machine_id=1), _offer(offer_id=2, machine_id=2)]
    with pytest.raises(UnreconciledCreateError, match="aborting the offer walk"):
        _deploy(vast, _spec(), seed=0, offers=offers, attempt=2)
    assert rented == [1]  # never walked on to offer 2 (no duplicate create)
    assert destroyed_for  # fail-closed: destroy-by-label was attempted before the terminal raise


def test_deploy_adopts_only_exact_label_among_decoys(monkeypatch):
    # adoption must key on the exact run/seed/attempt label: decoys from the same run with a
    # different seed/attempt, and a similar-prefix run, must never be adopted.
    import io
    import urllib.error

    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    rented = []

    def fake_create(offer_id, **kw):
        rented.append(offer_id)
        e = vast_api.VastApiError("create failed: 503")
        e.__cause__ = urllib.error.HTTPError("u", 503, "boom", None, io.BytesIO(b""))
        raise e

    monkeypatch.setattr(vast_api, "create_instance", fake_create)
    exact = "flash-1700000000-abcd1234-s0-a2"
    monkeypatch.setattr(
        vast_api,
        "list_instances",
        lambda: [
            {"id": 111, "label": "flash-1700000000-abcd1234-s1-a2"},  # same run, other seed
            {"id": 222, "label": "flash-1700000000-abcd1234-s0-a1"},  # same run, other attempt
            {"id": 333, "label": "flash-1700000000-abcd12345-s0-a2"},  # similar-prefix run
            {"id": 555, "label": exact, "start_date": 1699999000.0},
        ],
    )
    offers = [_offer(offer_id=1, machine_id=1), _offer(offer_id=2, machine_id=2)]
    h = _deploy(vast, _spec(), seed=0, offers=offers, attempt=2)
    assert h.instance_id == 555  # only the exact label is adopted
    assert rented == [1]  # no duplicate create


def test_deploy_decoys_without_exact_match_abort_with_no_second_create(monkeypatch):
    import io
    import urllib.error

    from flash.providers.core.base import UnreconciledCreateError
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    rented = []

    def fake_create(offer_id, **kw):
        rented.append(offer_id)
        e = vast_api.VastApiError("create failed: 503")
        e.__cause__ = urllib.error.HTTPError("u", 503, "boom", None, io.BytesIO(b""))
        raise e

    monkeypatch.setattr(vast_api, "create_instance", fake_create)
    monkeypatch.setattr(
        vast_api,
        "list_instances",
        lambda: [
            {"id": 111, "label": "flash-1700000000-abcd1234-s1-a2"},
            {"id": 333, "label": "flash-1700000000-abcd12345-s0-a2"},
        ],
    )
    destroyed_for = []
    monkeypatch.setattr(vast, "destroy_run_instances", lambda rid: destroyed_for.append(rid) or [])
    offers = [_offer(offer_id=1, machine_id=1), _offer(offer_id=2, machine_id=2)]
    with pytest.raises(UnreconciledCreateError, match="aborting the offer walk"):
        _deploy(vast, _spec(), seed=0, offers=offers, attempt=2)
    assert rented == [1]  # decoys must not satisfy the reconcile: no second create
    assert destroyed_for


def test_vast_image_selects_the_per_sm_tag():
    # Vast routes through worker_image_for_gpu like RunPod/Lambda, so it gets the arch-matched
    # baked image rather than always returning the flat default.
    from flash.providers.vast.jobs.builders import vast_image

    assert vast_image("RTX 4090") == "ghcr.io/freesolo-co/flash-worker:cu128-sm89"


def test_deploy_raises_when_pool_exhausted(monkeypatch):
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setattr(
        vast_api,
        "create_instance",
        lambda *a, **k: (_ for _ in ()).throw(vast_api.VastCreateRejected("taken")),
    )
    monkeypatch.setattr(vast, "usable_offers", lambda *a, **k: [])
    with pytest.raises(vast_api.VastApiError, match="rejected the job"):
        _deploy(vast, _spec(), seed=0, offers=[_offer()], attempt=0)
    with pytest.raises(vast_api.VastApiError, match="no usable vast offers"):
        _deploy(vast, _spec(), seed=0, offers=[], attempt=0)


# ---------------------------------------------------------------------------
# vast resource and fenced-result polling
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


def _wire_vast_poll(monkeypatch, *, attempt=None, results=(), instances=()):
    from flash.providers._lifecycle.instances import poll_instance
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api
    from flash.runner.lifecycle import status as status_ops

    result_iter = iter(results)
    instance_iter = iter(instances)
    last = {"value": None}

    def get_instance(*_args, **_kwargs):
        last["value"] = next(instance_iter, last["value"])
        return last["value"]

    monkeypatch.setattr(vast_api, "get_instance", get_instance)
    monkeypatch.setattr(status_ops, "get_status", lambda _run_id: SimpleNamespace(remote={}))
    monkeypatch.setattr(
        status_ops,
        "source_snapshot_from_status",
        lambda _status, required=True: dict(SOURCE_SNAPSHOT),
    )
    monkeypatch.setattr(
        poll_instance,
        "_current_attempt",
        lambda _adapter: attempt or _instance_attempt(provider="vast"),
    )
    monkeypatch.setattr(poll_instance, "_record_resource", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(poll_instance, "_observe_result", lambda _adapter: next(result_iter, None))
    monkeypatch.setattr(poll_instance.time, "sleep", lambda _seconds: None)
    return vast, poll_instance


def test_poll_vast_returns_current_fenced_result_before_status(monkeypatch):
    from flash.providers.core.base import PollResult

    vast, _poll = _wire_vast_poll(
        monkeypatch,
        results=[PollResult(True, metrics={"wall_seconds": 12.0})],
    )

    result = vast.poll_vast_job(_handle(), _spec(), seed=0, interval_s=0)

    assert result.ok
    assert result.metrics == {"wall_seconds": 12.0}


def test_poll_vast_dead_instance_waits_for_result_deadline(monkeypatch):
    vast, poll_instance = _wire_vast_poll(
        monkeypatch,
        attempt=_instance_attempt(provider="vast", work=20.0, result=30.0),
        results=[None, None],
        instances=[{"actual_status": "exited"}],
    )
    monkeypatch.setattr(poll_instance.time, "time", _instance_clock())

    result = vast.poll_vast_job(_handle(), _spec(), seed=0, interval_s=0)

    assert result.failure == "job_preempted"
    assert "exited" in result.detail
    assert "result manifest" in result.detail


def test_poll_vast_dead_instance_before_grant_waits_for_result_deadline(monkeypatch):
    vast, poll_instance = _wire_vast_poll(
        monkeypatch,
        attempt=_instance_attempt(provider="vast", grant=25.0, work=40.0, result=50.0),
        results=[None, None, None],
        instances=[{"actual_status": "exited"}],
    )
    monkeypatch.setattr(poll_instance.time, "time", _instance_clock())

    result = vast.poll_vast_job(_handle(), _spec(), seed=0, interval_s=0)

    assert result.failure == "job_preempted"
    assert "exited" in result.detail
    assert "result manifest" in result.detail
    assert "grant deadline" not in result.detail


def test_poll_vast_active_without_progress_uses_attempt_deadline(monkeypatch):
    vast, poll_instance = _wire_vast_poll(
        monkeypatch,
        attempt=_instance_attempt(provider="vast", work=25.0, result=45.0),
        results=[None, None, None],
        instances=[{"actual_status": "running"}],
    )
    monkeypatch.setattr(poll_instance.time, "time", _instance_clock())

    result = vast.poll_vast_job(
        _handle(),
        _spec(),
        seed=0,
        interval_s=0,
        deadline_at=10_000.0,
    )

    assert result.failure == "job_preempted"
    assert "work deadline expired" in result.detail


def test_poll_vast_preserves_provisioning_deadline(monkeypatch):
    vast, poll_instance = _wire_vast_poll(
        monkeypatch,
        attempt=_instance_attempt(provider="vast", grant=15.0, work=100.0, result=120.0),
        results=[None, None],
        instances=[{"actual_status": "loading"}],
    )
    monkeypatch.setattr(poll_instance.time, "time", _instance_clock())

    result = vast.poll_vast_job(_handle(), _spec(), seed=0, interval_s=0)

    assert result.failure == "job_preempted"
    assert "grant deadline" in result.detail


def test_poll_vast_bounds_provider_status_failures(monkeypatch):
    from flash.providers._lifecycle.instances import poll as poll_helpers
    from flash.providers.vast.client import api as vast_api

    vast, poll_instance = _wire_vast_poll(
        monkeypatch,
        attempt=_instance_attempt(provider="vast", work=1_000.0, result=1_020.0),
        results=[None] * 10,
    )
    monkeypatch.setattr(poll_instance.time, "time", _instance_clock(step=1.0))
    monkeypatch.setattr(poll_helpers.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        vast_api,
        "get_instance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(vast_api.VastApiError("offline")),
    )

    result = vast.poll_vast_job(_handle(), _spec(), seed=0, interval_s=0)

    assert result.failure == "poll_error"


def test_vast_resource_loss_retires_the_machine_after_result_grace(monkeypatch):
    from flash.providers.vast import jobs as vast

    vast.forget_dead_machines(_spec().run_id)
    try:
        vast, poll_instance = _wire_vast_poll(
            monkeypatch,
            attempt=_instance_attempt(provider="vast", work=20.0, result=30.0),
            results=[None, None],
            instances=[{"actual_status": "offline"}],
        )
        monkeypatch.setattr(poll_instance.time, "time", _instance_clock())

        result = vast.poll_vast_job(_handle(), _spec(), seed=0, interval_s=0)

        assert result.failure == "job_preempted"
        assert vast.dead_machine_ids(_spec().run_id) == frozenset({10})
    finally:
        vast.forget_dead_machines(_spec().run_id)


def test_vast_poll_adapter_carries_attempt_fence_and_cost_stamp(monkeypatch):
    from flash.providers.vast import jobs as vast
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

    monkeypatch.setattr(vast, "poll_instance_job", capture)

    result = vast.poll_vast_job(_handle(started_ts=9_000.0), _spec(), seed=0)
    metrics = {"wall_seconds": 100.0}
    captured["adapter"].stamp_cost_and_notes(metrics, end_ts=9_100.0, launch_ts=9_000.0)

    assert result.ok
    assert captured["adapter"].current_attempt == 0
    assert captured["adapter"].fence == 1
    assert metrics["cost_usd"] == round(100.0 / 3600.0 * 0.47, 6)
    assert metrics["notes"]["provider"] == "vast"


def test_provider_initial_and_reattached_poll_keep_same_deadline(monkeypatch):
    from flash.providers.core.base import JobHandle, PollResult
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.execution.provider import VastProvider

    captured = []

    def fake_poll(_handle, _spec, _seed, *, log=None, deadline_at=None):
        captured.append(deadline_at)
        return PollResult(True, metrics={})

    monkeypatch.setattr(vast, "usable_offers", lambda *_args, **_kwargs: [_offer()])
    monkeypatch.setattr(vast, "deploy_and_submit", lambda *_args, **_kwargs: _handle(started_ts=1.0))
    monkeypatch.setattr(vast, "poll_vast_job", fake_poll)
    monkeypatch.setattr(vast, "_best_effort_destroy", lambda *_args, **_kwargs: True)
    provider = VastProvider()
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
# submit_run_vast: guaranteed teardown
# ---------------------------------------------------------------------------
def _wire_submit(monkeypatch, poll_result=None, poll_raises=None):
    from flash.providers.core.base import PollResult
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    destroyed = []
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: destroyed.append(iid) or True)
    monkeypatch.setattr(
        vast,
        "deploy_and_submit",
        lambda spec, seed, offers, attempt=0, fence=1, log=None, runtime_secrets=None, source_snapshot=None, deadline_at=None: (
            _handle()
        ),
    )
    monkeypatch.setattr(vast, "usable_offers", lambda *a, **k: [_offer()])

    def fake_poll(handle, spec, seed, **kw):
        if poll_raises:
            raise poll_raises
        return poll_result or PollResult(True, metrics={})

    monkeypatch.setattr(vast, "poll_vast_job", fake_poll)
    return vast, destroyed


def test_runner_destroys_on_success(monkeypatch):
    vast, destroyed = _wire_submit(monkeypatch)
    res = _submit(vast, _spec(), seed=0)
    assert res.ok
    assert destroyed == [9999]  # the rented instance is torn down


# ---------------------------------------------------------------------------
# submit_run_vast: a machine that took the rental and never booted is retired
# ---------------------------------------------------------------------------
def _offered_machines(monkeypatch, vast, market=None) -> list[frozenset[int]]:
    """Record which machines actually reached ``deploy_and_submit`` on each attempt.

    Asserts the OUTCOME rather than how it is reached: the search itself is no longer told what to
    exclude, because the message distinguishing "this run burned the pool" from "the pool is dry"
    needs the unfiltered market to compare against.
    """
    seen: list[frozenset[int]] = []
    rows = list(market) if market is not None else [_offer()]

    monkeypatch.setattr(vast, "usable_offers", lambda *_a, **_k: list(rows))
    monkeypatch.setattr(
        vast,
        "deploy_and_submit",
        lambda spec, seed, offers, attempt=0, fence=1, log=None, runtime_secrets=None, source_snapshot=None, deadline_at=None: (
            seen.append(frozenset(o.machine_id for o in offers)) or _handle()
        ),
    )
    return seen


def test_a_runs_own_failure_does_not_retire_a_healthy_machine(monkeypatch):
    """Only host-shaped failures retire a box.

    ``job_failed`` is the run's own doing and would recur on any machine, and ``poll_error`` covers
    transient HF read gaps that say nothing about the host. Blacklisting on either would shrink the
    usable pool on every attempt and recreate the starvation from the other direction.
    """
    from flash.providers.core.base import PollResult
    from flash.providers.vast import jobs as vast

    spec = _spec()
    for failure in ("job_failed", "poll_error", "oom"):
        vast.forget_dead_machines(spec.run_id)
        _wire_submit(monkeypatch, poll_result=PollResult(False, failure=failure, detail="x"))
        _submit(vast, spec, seed=0)
        assert vast.dead_machine_ids(spec.run_id) == frozenset(), failure
    vast.forget_dead_machines(spec.run_id)


def test_a_pool_exhausted_by_this_runs_own_dead_machines_says_so(monkeypatch):
    """The empty-pool error must distinguish "Vast is out" from "this run killed them all".

    The two have different operator fixes -- wait, versus move to another class or provider -- and
    the generic message only ever suggested the first.
    """
    from flash.providers.core.base import RunExhaustedProviderPoolError
    from flash.providers.vast import jobs as vast

    spec = _spec()
    vast.forget_dead_machines(spec.run_id)
    try:
        vast._note_dead_machine(spec.run_id, 10)
        # the class still HAS an offer; this run has simply already lost that host.
        monkeypatch.setattr(vast, "usable_offers", lambda *a, **k: [_offer(machine_id=10)])

        # its own type, not VastApiError: supervision withholds provider exception text from the
        # run record, so only an authored error survives to the operator.
        with pytest.raises(RunExhaustedProviderPoolError, match="already rented and lost"):
            _submit(vast, spec, seed=0)
    finally:
        vast.forget_dead_machines(spec.run_id)


def test_a_dry_market_is_not_blamed_on_this_runs_blacklist(monkeypatch):
    """An empty class must not be reported as self-inflicted just because a blacklist is non-empty.

    The blacklist is keyed by run, not by GPU class, so after an escalation it still holds hosts
    from the class the run has already moved off. Blaming a genuinely dry market on those points the
    operator at the wrong fix -- "switch class or provider" when the real answer is "wait" -- and
    the count it quotes would name machines that were never in this class's pool to begin with.
    """
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    spec = _spec()
    vast.forget_dead_machines(spec.run_id)
    try:
        # a dead host from an earlier class, and no offers at all for the one being searched now.
        vast._note_dead_machine(spec.run_id, 777)
        monkeypatch.setattr(vast, "usable_offers", lambda *a, **k: [])

        with pytest.raises(vast_api.VastApiError) as caught:
            _submit(vast, spec, seed=0)
        assert "already rented and lost" not in str(caught.value)
    finally:
        vast.forget_dead_machines(spec.run_id)


def test_a_fully_blacklisted_first_page_widens_the_search_before_giving_up(monkeypatch):
    """The row cap is a price-sorted prefix, so an all-dead page is not an exhausted class.

    ``search_offers`` applies its limit server-side and the machine exclusion runs client-side, so a
    run that burned the cheap boxes sees an empty list while dearer usable capacity sits just past
    the cap. Concluding exhaustion from that page fails the run with capacity still available.
    """
    from flash.providers.vast import jobs as vast

    spec = _spec()
    vast.forget_dead_machines(spec.run_id)
    try:
        _wire_submit(monkeypatch)
        vast._note_dead_machine(spec.run_id, 10)
        limits: list[int] = []

        def _paged(*_a, limit=256, **_k):
            limits.append(int(limit))
            # the cheap page is entirely this run's own dead host; the wider page reaches a live one.
            if int(limit) <= 256:
                return [_offer(machine_id=10)]
            return [_offer(machine_id=10), _offer(offer_id=2, machine_id=11)]

        seen: list[frozenset[int]] = []
        monkeypatch.setattr(vast, "usable_offers", _paged)
        monkeypatch.setattr(
            vast,
            "deploy_and_submit",
            lambda spec, seed, offers, attempt=0, fence=1, log=None, runtime_secrets=None, source_snapshot=None, deadline_at=None: (
                seen.append(frozenset(o.machine_id for o in offers)) or _handle()
            ),
        )

        _submit(vast, spec, seed=0)

        assert limits[-1] > limits[0], f"never widened past the default page: {limits}"
        assert seen == [frozenset({11})], f"expected the host past the cap, rented {seen}"
    finally:
        vast.forget_dead_machines(spec.run_id)


def test_gc_frees_the_runs_dead_machine_set(monkeypatch):
    """The blacklist is process-local state, so the run's reap has to release it."""
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.execution.provider import PROVIDER

    spec = _spec()
    vast._note_dead_machine(spec.run_id, 10)
    monkeypatch.setattr(vast, "destroy_run_instances", lambda _run_id: [])

    PROVIDER.gc(spec)

    assert vast.dead_machine_ids(spec.run_id) == frozenset()


def test_runner_destroys_on_exception(monkeypatch):
    vast, destroyed = _wire_submit(monkeypatch, poll_raises=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        _submit(vast, _spec(), seed=0)
    assert destroyed == [9999]  # destroyed even on interrupt


def test_runner_destroys_when_handle_persist_fails(monkeypatch):
    """on_handle (persisting the handle) raising must still tear down the already-billing instance."""
    vast, destroyed = _wire_submit(monkeypatch)

    def boom(_d):
        raise RuntimeError("db down")

    with pytest.raises(RuntimeError, match="db down"):
        _submit(vast, _spec(), seed=0, on_handle=boom)
    assert destroyed == [9999]


def test_submit_teardown_warns_on_unconfirmed_destroy_without_raising(monkeypatch, caplog):
    """The PRIMARY teardown (submit_run_vast ``finally``) must NOT silently ignore a
    success:false from destroy_instance — a raise there would mask the poll result, so instead it WARNS
    so operators see a possible leak immediately (not only at the next sweep). The run still returns."""
    import logging

    from flash.providers.core.base import PollResult
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: False)  # unconfirmed teardown
    monkeypatch.setattr(
        vast,
        "deploy_and_submit",
        lambda spec, seed, offers, attempt=0, fence=1, log=None, runtime_secrets=None, source_snapshot=None, deadline_at=None: (
            _handle()
        ),
    )
    monkeypatch.setattr(vast, "usable_offers", lambda *a, **k: [_offer()])
    monkeypatch.setattr(vast, "poll_vast_job", lambda *a, **k: PollResult(True, metrics={}))

    with caplog.at_level(logging.WARNING):
        res = _submit(vast, _spec(), seed=0)  # the finally must not raise on False
    assert res.ok
    assert any("teardown unconfirmed" in r.message for r in caplog.records), (
        "an unconfirmed teardown in the primary path must emit an operator-visible warning"
    )


def test_submit_unconfirmed_teardown_escalates_to_run_scoped_reap(monkeypatch):
    """Codex: on a SUCCESSFUL seed whose single-instance teardown is UNCONFIRMED (success:false /
    breakdown), the success still propagates and _run_seed_loop clears `remote` + launches the next seed
    — while the run stays `running` the active-run sweep SHIELDS this label, so the box could survive
    across every remaining seed with no handle. The finally must escalate to a run-scoped reap by label
    (destroy_run_instances, NOT active-shielded) so this seed's box is cleared before the next launches."""
    from flash.providers.core.base import PollResult
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setattr(
        vast_api, "destroy_instance", lambda iid: False
    )  # unconfirmed single destroy
    monkeypatch.setattr(
        vast,
        "deploy_and_submit",
        lambda spec, seed, offers, attempt=0, fence=1, log=None, runtime_secrets=None, source_snapshot=None, deadline_at=None: (
            _handle()
        ),
    )
    monkeypatch.setattr(vast, "usable_offers", lambda *a, **k: [_offer()])
    monkeypatch.setattr(vast, "poll_vast_job", lambda *a, **k: PollResult(True, metrics={}))
    reaped = []
    monkeypatch.setattr(vast, "destroy_run_instances", lambda rid: reaped.append(rid) or [])

    res = _submit(vast, _spec(), seed=0)
    assert res.ok  # the successful seed still returns
    assert reaped == [_spec().run_id], (
        "an unconfirmed teardown must escalate to a run-scoped label reap"
    )


def test_submit_confirmed_teardown_skips_run_scoped_reap(monkeypatch):
    """The escalation fires ONLY on an unconfirmed teardown: a confirmed single-instance destroy needs no
    extra run-scoped reap (avoids a redundant list+destroy on every normal seed completion)."""
    from flash.providers.core.base import PollResult
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: True)  # confirmed
    monkeypatch.setattr(
        vast,
        "deploy_and_submit",
        lambda spec, seed, offers, attempt=0, fence=1, log=None, runtime_secrets=None, source_snapshot=None, deadline_at=None: (
            _handle()
        ),
    )
    monkeypatch.setattr(vast, "usable_offers", lambda *a, **k: [_offer()])
    monkeypatch.setattr(vast, "poll_vast_job", lambda *a, **k: PollResult(True, metrics={}))
    reaped = []
    monkeypatch.setattr(vast, "destroy_run_instances", lambda rid: reaped.append(rid) or [])

    assert _submit(vast, _spec(), seed=0).ok
    assert reaped == [], "a confirmed teardown must NOT trigger the run-scoped reap"


def test_submit_no_reap_when_rejection_log_raises_baseexception(monkeypatch):
    # bugbot: a keyboardinterrupt raised by the rejection log escapes the exception-only
    # suppress; the reap flag must already be cleared for the definitive rejection so the
    # escaping interrupt does not trigger a run-label reap that could hit other seeds' boxes.
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    original = KeyboardInterrupt("interrupt during rejection log")

    def raising_say(_log):
        def _say(_msg):
            raise original

        return _say

    monkeypatch.setattr(
        vast_api,
        "create_instance",
        lambda *a, **k: (_ for _ in ()).throw(vast_api.VastCreateRejected("taken")),
    )
    monkeypatch.setattr(vast, "make_say", raising_say)
    monkeypatch.setattr(vast, "usable_offers", lambda *a, **k: [_offer()])
    reaped = []
    monkeypatch.setattr(vast, "destroy_run_instances", lambda rid: reaped.append(rid) or [])

    with pytest.raises(KeyboardInterrupt) as exc_info:
        _submit(vast, _spec(), seed=0)
    assert exc_info.value is original
    assert reaped == []  # definitive rejection rented nothing: no run-label reap


def test_submit_no_reap_when_failure_precedes_any_create(monkeypatch):
    # a failure before any create request (empty offer pool) must not run-label reap:
    # a concurrent worker for the same run could own a live instance under that label.
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setattr(vast, "usable_offers", lambda *a, **k: [])
    reaped = []
    monkeypatch.setattr(vast, "destroy_run_instances", lambda rid: reaped.append(rid) or [])

    with pytest.raises(vast_api.VastApiError, match="no usable vast offers"):
        _submit(vast, _spec(), seed=0)
    assert reaped == []


def test_submit_teardown_cleanup_baseexception_preserves_original_and_still_reaps(monkeypatch):
    # a cleanup-time interrupt during the finally's delete must not replace the in-flight
    # exception and must not skip the run-label fallback.
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    original = KeyboardInterrupt("original interruption")
    cleanup = SystemExit("cleanup interrupted")
    reaped = []

    def destroy_exact(iid):
        raise cleanup

    monkeypatch.setattr(vast_api, "destroy_instance", destroy_exact)
    monkeypatch.setattr(
        vast,
        "deploy_and_submit",
        lambda spec, seed, offers, attempt=0, fence=1, log=None, runtime_secrets=None, source_snapshot=None: (
            _handle()
        ),
    )
    monkeypatch.setattr(vast, "usable_offers", lambda *a, **k: [_offer()])

    def fake_poll(handle, spec, seed, **kw):
        raise original

    monkeypatch.setattr(vast, "poll_vast_job", fake_poll)
    monkeypatch.setattr(vast, "destroy_run_instances", lambda rid: reaped.append(rid) or [])

    with pytest.raises(KeyboardInterrupt) as exc_info:
        _submit(vast, _spec(), seed=0)
    assert exc_info.value is original
    assert reaped == [_spec().run_id]


def test_submit_teardown_cleanup_baseexception_reraised_when_no_original(monkeypatch):
    # with no in-flight exception, a cleanup-time interrupt must still surface after the
    # run-label fallback ran (silent swallowing would hide an operator ctrl-c).
    from flash.providers.core.base import PollResult
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    cleanup = KeyboardInterrupt("cleanup interrupted")
    reaped = []

    def destroy_exact(iid):
        raise cleanup

    monkeypatch.setattr(vast_api, "destroy_instance", destroy_exact)
    monkeypatch.setattr(
        vast,
        "deploy_and_submit",
        lambda spec, seed, offers, attempt=0, fence=1, log=None, runtime_secrets=None, source_snapshot=None: (
            _handle()
        ),
    )
    monkeypatch.setattr(vast, "usable_offers", lambda *a, **k: [_offer()])
    monkeypatch.setattr(vast, "poll_vast_job", lambda *a, **k: PollResult(True, metrics={}))
    monkeypatch.setattr(vast, "destroy_run_instances", lambda rid: reaped.append(rid) or [])

    with pytest.raises(KeyboardInterrupt) as exc_info:
        _submit(vast, _spec(), seed=0)
    assert exc_info.value is cleanup
    assert reaped == [_spec().run_id]


def test_best_effort_destroy_returns_confirmation(monkeypatch):
    """The helper returns the destroy_instance bool and only warns on False (no warn on a clean True)."""
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: True)
    assert vast._best_effort_destroy(123, context="t") is True
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: False)
    assert vast._best_effort_destroy(123, context="t") is False


def test_best_effort_destroy_passes_raw_id_and_never_int_raises(monkeypatch):
    """The helper must NOT int()-convert the id itself — destroy_instance does that inside
    its own try/except (-> False on a bad id, "never raises"), so converting in the wrapper would
    re-introduce a ValueError in the very finally/suppress paths this helper exists to keep quiet.
    Assert the id reaches destroy_instance UNCONVERTED and a non-numeric id returns False, no raise."""
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    seen = []
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: seen.append(iid) or False)
    assert vast._best_effort_destroy("not-a-number", context="t") is False  # must not raise
    assert seen == ["not-a-number"]  # passed through raw — no int() in the wrapper


def test_submit_run_vast_rejects_policy_word_gpu(monkeypatch):
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    spec = _spec()
    object.__setattr__(
        spec.gpu, "type", "cheapest"
    )  # a policy word that never reached the allocator
    with pytest.raises(vast_api.VastApiError, match="concrete gpu class"):
        _submit(vast, spec, seed=0)


def test_submit_uses_transient_concrete_gpu_type_for_exact_search(monkeypatch):
    seen: dict[str, str] = {}

    def capture(min_vram_gb, disk_gb, *a, gpu_type="", **k):
        seen["gpu_type"] = gpu_type
        return [_offer(gpu="H100")]

    vast, _ = _wire_submit(monkeypatch)
    monkeypatch.setattr(vast, "usable_offers", capture)

    _submit(vast, _spec(gpu_type="H100"), seed=0)
    assert seen["gpu_type"] == "H100"


def test_provider_destroy_raises_on_unconfirmed_teardown(monkeypatch):
    """``destroy_instance`` returning False (success:false / breakdown) means the box is
    STILL billing. ``VastProvider.destroy`` must SURFACE that (raise) instead of returning normally —
    else the best-effort callers log "terminated" and clear the handle while it keeps billing."""
    from flash.providers.core.base import JobHandle
    from flash.providers.vast.client import api as vast_api
    from flash.providers.vast.execution.provider import PROVIDER

    handle = JobHandle.from_dict(_handle().to_dict())
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: False)  # unconfirmed
    with pytest.raises(vast_api.VastApiError, match="unconfirmed"):
        PROVIDER.destroy(handle)
    # confirmed teardown returns normally (no raise)
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: True)
    PROVIDER.destroy(handle)
    with pytest.raises(ValueError, match="persisted vast instance identity is invalid"):
        PROVIDER.destroy(JobHandle.from_dict({"provider": "vast"}))


def test_provider_poll_recovery_unconfirmed_teardown_escalates_to_run_scoped_reap(monkeypatch):
    """Cursor: VastProvider.poll's recovery-teardown ``finally`` must escalate an UNCONFIRMED single-
    instance destroy to a run-scoped reap (mirroring the submit_run_vast finally). Otherwise a successful
    multi-seed ATTACH that clears ``remote`` and resumes the next seed leaves the box shielded by the
    active-run label and billing, with no persisted handle."""
    from flash.providers.core.base import JobHandle, PollResult
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api
    from flash.providers.vast.execution.provider import PROVIDER

    monkeypatch.setattr(
        vast_api, "destroy_instance", lambda iid: False
    )  # unconfirmed single destroy
    monkeypatch.setattr(vast, "poll_vast_job", lambda *a, **k: PollResult(True, metrics={}))
    reaped = []
    monkeypatch.setattr(vast, "destroy_run_instances", lambda rid: reaped.append(rid) or [])

    handle = JobHandle.from_dict(_handle().to_dict())
    res = PROVIDER.poll(handle, _spec(), seed=0)
    assert res.ok  # the recovered seed still returns its success
    assert reaped == [_spec().run_id], (
        "unconfirmed recovery teardown must escalate to a run-scoped reap"
    )


def test_provider_poll_recovery_confirmed_teardown_skips_run_scoped_reap(monkeypatch):
    """A confirmed recovery teardown needs no extra run-scoped reap (no redundant list+destroy)."""
    from flash.providers.core.base import JobHandle, PollResult
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api
    from flash.providers.vast.execution.provider import PROVIDER

    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: True)  # confirmed
    monkeypatch.setattr(vast, "poll_vast_job", lambda *a, **k: PollResult(True, metrics={}))
    reaped = []
    monkeypatch.setattr(vast, "destroy_run_instances", lambda rid: reaped.append(rid) or [])

    PROVIDER.poll(JobHandle.from_dict(_handle().to_dict()), _spec(), seed=0)
    assert reaped == [], "a confirmed recovery teardown must NOT trigger the run-scoped reap"


def test_provider_destroy_rejects_incomplete_handle_before_api_call(monkeypatch):
    from flash.providers.core.base import JobHandle
    from flash.providers.vast.client import api as vast_api
    from flash.providers.vast.execution.provider import PROVIDER

    seen = []
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: seen.append(iid) or False)
    handle = JobHandle.from_dict({"provider": "vast", "instance_id": "not-a-number"})
    with pytest.raises(ValueError, match="persisted vast instance identity is invalid"):
        PROVIDER.destroy(handle)
    assert seen == []


# ---------------------------------------------------------------------------
# labels, handle, sweep, gc
# ---------------------------------------------------------------------------
def test_instance_label_and_handle_roundtrip():
    from flash.providers.vast.jobs import instance_label, run_label_prefix
    from flash.providers.vast.jobs.builders import VastJobHandle

    label = instance_label("flash-run9", seed=0, attempt=2)
    assert label.startswith(run_label_prefix("flash-run9"))
    assert label.endswith("-s0-a2")
    h = _handle()
    back = VastJobHandle.from_dict(h.to_dict())
    assert back.to_dict()["provider"] == "vast"
    assert back.instance_id == h.instance_id
    assert back.offer_id == h.offer_id


def test_handle_from_dict_corrupt_instance_id_raises_clear_error():
    """A corrupt/partial PERSISTED handle (reattach/recovery) must fail with a CLEAR,
    actionable error naming the bad instance_id — not a bare KeyError/ValueError that crashes recovery
    with an opaque cause. instance_id has no safe default (it's the poll/destroy target)."""
    from flash.providers.vast.jobs.builders import VastJobHandle

    for bad in ({}, {"instance_id": None}, {"instance_id": "not-a-number"}):
        with pytest.raises(ValueError, match="persisted vast"):
            VastJobHandle.from_dict(bad)


def test_destroy_run_instances_matches_forced_prefix(monkeypatch):
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    instances = [
        {"id": 1, "label": "flash-run1-s0-a0"},  # ours
        {
            "id": 2,
            "label": "flash-run10-s0-a0",
        },  # a DIFFERENT run (prefix boundary) — must NOT match
        {"id": 3, "label": "someone-else"},  # not ours
    ]
    monkeypatch.setattr(vast_api, "list_instances", lambda: instances)
    destroyed = []
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: destroyed.append(iid) or True)
    out = vast.destroy_run_instances("run1")  # raw id; run_label_prefix forces the flash- prefix
    assert out == [1]
    assert destroyed == [1]


def test_run_instances_remaining_confirms_clear_and_raises_on_listing_failure(monkeypatch):
    # Codex: the handle-less recovery resubmit gates on this. [] == CONFIRMED no instance for the run
    # remains; a survivor (e.g. after an unconfirmed DELETE) is reported by id; it matches on the SAME
    # label boundary as destroy_run_instances (run1 must not match run10). A listing failure RAISES so
    # the caller can't mistake "couldn't list" for "clear".
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    instances = [
        {"id": 9, "label": "flash-run1-s0-a0"},  # ours -> remaining
        {"id": 10, "label": "flash-run10-s0-a0"},  # different run (boundary) -> NOT ours
        {"id": 11, "label": "someone-else"},  # not ours
    ]
    monkeypatch.setattr(vast_api, "list_instances", lambda strict=False: instances)
    assert vast.run_instances_remaining("run1") == [9]

    monkeypatch.setattr(vast_api, "list_instances", lambda strict=False: [])
    assert vast.run_instances_remaining("run1") == []  # confirmed clear

    def boom(strict=False):
        raise vast_api.VastApiError("list failed")

    monkeypatch.setattr(vast_api, "list_instances", boom)
    with pytest.raises(vast_api.VastApiError):
        vast.run_instances_remaining("run1")  # cannot confirm clear -> RAISE (caller defers)


def test_run_instances_remaining_raises_on_label_match_with_unparseable_id(monkeypatch):
    # Codex: a strict page can carry a row with THIS run's label but a missing/non-numeric id. Silently
    # skipping it (as the best-effort destroy_run_instances does) would let run_instances_remaining
    # report a FALSE clear and resubmit a handle-less run over a visible box it can't destroy. A
    # label-matching row with an unparseable id must be treated as NOT clear -> raise.
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    # a matching label, but the id is non-numeric -> can't enumerate/destroy -> not clear
    monkeypatch.setattr(
        vast_api,
        "list_instances",
        lambda strict=False: [{"id": "not-an-int", "label": "flash-run1-s0-a0"}],
    )
    with pytest.raises(vast_api.VastApiError, match="unparseable id"):
        vast.run_instances_remaining("run1")
    # an unparseable id on a NON-matching label is irrelevant -> still a confirmed clear
    monkeypatch.setattr(
        vast_api, "list_instances", lambda strict=False: [{"id": None, "label": "someone-else"}]
    )
    assert vast.run_instances_remaining("run1") == []


def test_cleanup_loops_skip_non_intable_id_without_raising(monkeypatch):
    """destroy_run_instances and sweep_orphans are documented "never raises", but
    a bare int(iid) on a non-intable id (unexpected Vast API shape) would raise mid-loop and abort the
    cleanup, leaving the remaining reapable boxes billing. A bad id must be SKIPPED, the GOOD ones still
    destroyed."""
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    instances = [
        {"id": None, "label": "flash-run1-s0-a0"},  # missing id -> skip
        {"id": "not-an-int", "label": "flash-run1-s1-a0"},  # non-intable -> skip, must NOT raise
        {"id": 7, "label": "flash-run1-s2-a0"},  # good -> destroyed
    ]
    monkeypatch.setattr(vast_api, "list_instances", lambda: instances)
    destroyed = []
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: destroyed.append(iid) or True)

    assert vast.destroy_run_instances("run1") == [7]  # bad ids skipped, good one reaped, no raise
    assert destroyed == [7]

    # sweep_orphans walks the same list (no active/known protection here) and must behave the same.
    destroyed.clear()
    assert vast.sweep_orphans(active_labels=set()) == [7]
    assert destroyed == [7]
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    instances = [
        {"id": 1, "label": "flash-runA-s0-a0"},  # active -> protected
        {"id": 2, "label": "flash-runB-s0-a0"},  # orphan -> reaped
        {"id": 3, "label": "flash-runA10-s0-a0"},  # NOT runA (boundary) -> orphan, reaped
        {"id": 4, "label": "not-ours"},  # untouched
    ]
    monkeypatch.setattr(vast_api, "list_instances", lambda: instances)
    destroyed = []
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: destroyed.append(iid) or True)
    out = vast.sweep_orphans(active_labels={"runA"})  # raw active id; prefix forced internally
    assert sorted(out) == [2, 3]
    assert 1 not in destroyed  # the active run's box survived
    assert 4 not in destroyed  # non-flash box untouched


def test_sweep_orphans_known_labels_multiplane_guard(monkeypatch):
    """With known_labels set, an instance is reaped only if its run id is one THIS plane knows — a box
    from ANOTHER control plane (run id absent from known) is left alone (multi-plane safety)."""
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    instances = [
        {"id": 1, "label": "flash-mine-s0-a0"},  # known + not active -> reaped
        {"id": 2, "label": "flash-other-s0-a0"},  # unknown to this plane -> left alone
    ]
    monkeypatch.setattr(vast_api, "list_instances", lambda: instances)
    destroyed = []
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: destroyed.append(iid) or True)
    out = vast.sweep_orphans(active_labels=set(), known_labels={"mine"})
    assert out == [1]
    assert 2 not in destroyed


def test_sweep_orphans_callable_sets_resolved_after_listing(monkeypatch):
    """active_labels/known_labels may be CALLABLES resolved AFTER the instance list (closes the launch
    race). A callable that raises SKIPS the sweep (never falls through to reaping live boxes)."""
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setattr(vast_api, "list_instances", lambda: [{"id": 1, "label": "flash-x-s0-a0"}])
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: True)
    # protected by a callable-resolved active set
    assert vast.sweep_orphans(active_labels=lambda: {"x"}) == []

    # a raising callable -> sweep skipped (returns [], reaps nothing)
    def boom():
        raise RuntimeError("db down")

    assert vast.sweep_orphans(active_labels=boom) == []
