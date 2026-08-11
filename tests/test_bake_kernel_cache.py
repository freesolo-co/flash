"""unit tests for docker/bake_kernel_cache.py's pod-create retry.

the bake rents a GPU of one arch, and RunPod picks the host at create time: a scarce class (sm86 /
sm120) is regularly rejected with "This machine does not have the resources to deploy your pod". an
unretried create therefore fails the whole arch ~30s in and ships a stale kernel cache against a
fresh worker image. these tests pin the split: a capacity rejection must be retried, and a real
error (auth, quota, bad image) must still fail on the first try instead of burning the attempt
budget.

docker/ is not a package, so import the module by path (same style as test_kernel_fingerprint.py).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BAKE_SCRIPT = ROOT / "docker" / "bake_kernel_cache.py"


def _load_bake():
    spec = importlib.util.spec_from_file_location("bake_kernel_cache", BAKE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


bake = _load_bake()

CAPACITY_MSG = (
    "This machine does not have the resources to deploy your pod. Please try a different machine"
)


class _FakeRunpod:
    """create_pod that raises the queued errors, then returns a pod."""

    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = []

    def create_pod(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return {"id": "pod-1"}


@pytest.fixture
def no_sleep(monkeypatch):
    """backoff sleeps are real seconds; record them instead of waiting."""
    slept = []
    monkeypatch.setattr(bake.time, "sleep", slept.append)
    return slept


@pytest.mark.parametrize(
    "msg",
    [
        CAPACITY_MSG,
        "There are no longer any instances available with the requested specifications.",
        "There are no longer any instances available with enough disk space.",
        "no instances available",
    ],
)
def test_capacity_shapes_are_retryable(msg):
    assert bake._is_capacity_error(RuntimeError(msg))


@pytest.mark.parametrize(
    "msg",
    [
        "Unauthorized",
        "invalid api key",
        "Your account has exceeded its spend limit",
        "manifest unknown: image not found",
    ],
)
def test_real_errors_are_not_capacity(msg):
    assert not bake._is_capacity_error(RuntimeError(msg))


def test_retries_capacity_rejection_then_succeeds(no_sleep):
    rp = _FakeRunpod([RuntimeError(CAPACITY_MSG), RuntimeError(CAPACITY_MSG)])
    pod = bake._create_pod_with_retry(rp, gpu_type_id="NVIDIA RTX A6000", name="bake")
    assert pod == {"id": "pod-1"}
    assert len(rp.calls) == 3
    # every attempt re-asks for the same class; placement is server-side, so the retry itself is
    # what moves the pod to another host.
    assert {c["gpu_type_id"] for c in rp.calls} == {"NVIDIA RTX A6000"}
    assert len(no_sleep) == 2


def test_gives_up_after_the_attempt_budget(no_sleep):
    rp = _FakeRunpod([RuntimeError(CAPACITY_MSG)] * bake.CREATE_ATTEMPTS)
    with pytest.raises(RuntimeError, match="capacity after"):
        bake._create_pod_with_retry(rp, gpu_type_id="NVIDIA B200")
    assert len(rp.calls) == bake.CREATE_ATTEMPTS
    # no trailing sleep after the last rejection.
    assert len(no_sleep) == bake.CREATE_ATTEMPTS - 1


def test_non_capacity_error_fails_fast(no_sleep):
    rp = _FakeRunpod([RuntimeError("Unauthorized")])
    with pytest.raises(RuntimeError, match="Unauthorized"):
        bake._create_pod_with_retry(rp, gpu_type_id="NVIDIA H200")
    assert len(rp.calls) == 1
    assert no_sleep == []


def test_backoff_grows_and_is_bounded(no_sleep):
    rp = _FakeRunpod([RuntimeError(CAPACITY_MSG)] * 3)
    bake._create_pod_with_retry(rp, gpu_type_id="NVIDIA L40S", backoff_s=(1, 2))
    # jitter adds up to 25%, and the last entry repeats for every further attempt.
    assert 1 <= no_sleep[0] <= 1.25
    assert 2 <= no_sleep[1] <= 2.5
    assert 2 <= no_sleep[2] <= 2.5


def test_bake_goes_through_the_retrying_helper():
    """the warm step runs this script, and its create must not bypass the retry."""
    wf = (ROOT / ".github" / "workflows" / "bake-kernel-cache.yml").read_text()
    assert "uv run python docker/bake_kernel_cache.py" in wf
    body = BAKE_SCRIPT.read_text().split("def main()")[1]
    assert "_create_pod_with_retry(" in body
    assert "runpod.create_pod(" not in body
