"""unit tests for docker/bake_kernel_cache.py's pod-create gpu walk.

the bake rents a gpu of one arch, and runpod picks the host at create time. a scarce type is regularly
rejected for capacity even when another same-sm type is free. these tests pin the split: a capacity
rejection must walk to the next type, and a real error must still fail immediately.

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


def test_walks_to_next_gpu_type_on_capacity_rejection(no_sleep):
    rp = _FakeRunpod([RuntimeError(CAPACITY_MSG), RuntimeError(CAPACITY_MSG)])
    pod, selected = bake._create_pod_with_gpu_walk(
        rp,
        gpu_type_ids=("NVIDIA RTX A6000", "NVIDIA A40", "NVIDIA RTX A5000"),
        name="bake",
    )
    assert pod == {"id": "pod-1"}
    assert selected == "NVIDIA RTX A5000"
    assert len(rp.calls) == 3
    assert [c["gpu_type_id"] for c in rp.calls] == [
        "NVIDIA RTX A6000",
        "NVIDIA A40",
        "NVIDIA RTX A5000",
    ]
    assert no_sleep == []


def test_retries_the_full_walk_after_backoff(no_sleep):
    rp = _FakeRunpod([RuntimeError(CAPACITY_MSG)] * 3)
    pod, selected = bake._create_pod_with_gpu_walk(
        rp,
        gpu_type_ids=("NVIDIA H200", "NVIDIA H100 80GB HBM3"),
        rounds=2,
        backoff_s=(1,),
    )
    assert pod == {"id": "pod-1"}
    assert selected == "NVIDIA H100 80GB HBM3"
    assert [c["gpu_type_id"] for c in rp.calls] == [
        "NVIDIA H200",
        "NVIDIA H100 80GB HBM3",
        "NVIDIA H200",
        "NVIDIA H100 80GB HBM3",
    ]
    assert 1 <= no_sleep[0] <= 1.25


def test_gives_up_after_the_round_budget(no_sleep):
    gpu_type_ids = ("NVIDIA RTX A6000", "NVIDIA A40")
    rp = _FakeRunpod([RuntimeError(CAPACITY_MSG)] * (bake.CREATE_ROUNDS * len(gpu_type_ids)))
    with pytest.raises(RuntimeError, match="no capacity across gpu walk"):
        bake._create_pod_with_gpu_walk(rp, gpu_type_ids=gpu_type_ids)
    assert len(rp.calls) == bake.CREATE_ROUNDS * len(gpu_type_ids)
    assert len(no_sleep) == bake.CREATE_ROUNDS - 1


def test_non_capacity_error_fails_fast(no_sleep):
    rp = _FakeRunpod([RuntimeError("Unauthorized")])
    with pytest.raises(RuntimeError, match="Unauthorized"):
        bake._create_pod_with_gpu_walk(rp, gpu_type_ids=("NVIDIA H200", "NVIDIA H100 80GB HBM3"))
    assert len(rp.calls) == 1
    assert no_sleep == []


def test_backoff_grows_and_is_bounded(no_sleep):
    rp = _FakeRunpod([RuntimeError(CAPACITY_MSG)] * 3)
    bake._create_pod_with_gpu_walk(rp, gpu_type_ids=("NVIDIA L40S",), rounds=4, backoff_s=(1, 2))
    # jitter adds up to 25%, and the last entry repeats for every further round.
    assert 1 <= no_sleep[0] <= 1.25
    assert 2 <= no_sleep[1] <= 2.5
    assert 2 <= no_sleep[2] <= 2.5


def test_default_gpu_walk_covers_every_baked_arch():
    from flash.providers._lifecycle.net.worker import BAKED_PER_SM_ARCHES

    assert set(bake.GPU_WALK_BY_SM) == BAKED_PER_SM_ARCHES
    assert all(len(types) == len(set(types)) for types in bake.GPU_WALK_BY_SM.values())
    assert all(types for types in bake.GPU_WALK_BY_SM.values())
    all_types = [gpu_type for types in bake.GPU_WALK_BY_SM.values() for gpu_type in types]
    assert len(all_types) == len(set(all_types))


def test_bake_goes_through_the_gpu_walk_helper():
    """the warm step runs this script, and its create must not bypass the gpu walk."""
    wf = (ROOT / ".github" / "workflows" / "bake-kernel-cache.yml").read_text()
    assert "uv run python docker/bake_kernel_cache.py" in wf
    assert "--gpu-type-id" not in wf
    body = BAKE_SCRIPT.read_text().split("def main()")[1]
    assert "_create_pod_with_gpu_walk(" in body
    assert "runpod.create_pod(" not in body
