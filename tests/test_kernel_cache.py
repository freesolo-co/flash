import json
import os
import sys
import types

import pytest

import flash.engine.worker.entry.sft as sft_entry
import flash.engine.worker.entry.worker as worker_entry
import flash.engine.worker.io.heartbeat as heartbeat_io
import flash.engine.worker.io.hf as hf_io
import flash.engine.worker.perf as worker_perf
from flash.engine.worker.runtime import kernel_warmup, state


class _FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def get_device_capability(index):
        return (8, 9)

    @staticmethod
    def get_device_name(index):
        return "fake gpu"


class _FakeTorch:
    __version__ = "test"
    cuda = _FakeCuda()
    version = types.SimpleNamespace(cuda="12.8")
    compiler = types.SimpleNamespace(load_cache_artifacts=lambda blob: None)


def _assert_isolated_backend_paths():
    paths = {
        var: os.path.realpath(os.environ[var]) for var in kernel_warmup.KERNEL_CACHE_ENV_SUBDIRS
    }
    assert all(not path.startswith("/opt/flash/kernelcache") for path in paths.values())
    for left, left_subdir in kernel_warmup.KERNEL_CACHE_ENV_SUBDIRS.items():
        for right, right_subdir in kernel_warmup.KERNEL_CACHE_ENV_SUBDIRS.items():
            if left_subdir != right_subdir:
                assert paths[left] != paths[right]


def test_kernel_warmup_out_overrides_existing_cache_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_CACHE_DIR", "/old/triton")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/old/inductor")

    kernel_warmup._point_backends_at(str(tmp_path))

    assert str(tmp_path / "triton") == os.environ["TRITON_CACHE_DIR"]
    assert str(tmp_path / "inductor") == os.environ["TORCHINDUCTOR_CACHE_DIR"]


def test_kernel_warmup_writes_arch_metadata(tmp_path):
    assert kernel_warmup.save_cache_metadata(
        _FakeTorch,
        str(tmp_path),
        requested_arch="8.9",
        warmed=3,
    )

    meta = json.loads((tmp_path / kernel_warmup.MEGA_CACHE_META_FILENAME).read_text())
    assert meta["sm"] == "sm89"
    assert meta["requested_arch"] == "8.9"
    assert meta["warmed_groups"] == 3


@pytest.mark.parametrize("cache_loaded", [False, True])
def test_worker_uploads_immutable_baked_cache_result_before_training(monkeypatch, cache_loaded):
    class WorkerExited(BaseException):
        pass

    artifact_name = "kernel_cache_sft_attempt7.json"
    expected_payload = (
        json.dumps(
            {"attempt": 7, "format_version": 1, "mega_cache_loaded": cache_loaded},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    events = []
    remote = {}

    def heartbeat(stage, **fields):
        remote["heartbeat.json"] = {"stage": stage, **fields}
        events.append(("heartbeat", stage, fields))

    def upload(local_path, repo_subpath, required=False):
        with open(local_path, encoding="utf-8") as file:
            payload = file.read()
        remote[repo_subpath] = payload
        events.append(("upload", repo_subpath, required, payload))
        return True

    def load_mega_cache():
        events.append(("load_mega_cache",))
        return cache_loaded

    def run_sft():
        events.append(("run_sft",))
        heartbeat_io.heartbeat("sft_start", step=0)

    monkeypatch.setattr(state, "RUN_MODE", "sft")
    monkeypatch.setattr(state, "ATTEMPT", 7)
    monkeypatch.setattr(state, "JOB_SPEC", None)
    monkeypatch.setattr(state, "HF_REPO", "")
    monkeypatch.setattr(state, "_remaining_worker_wall_seconds", lambda: None)
    monkeypatch.setattr(state, "_cleanup_active_env_package", lambda: None)
    monkeypatch.setattr(hf_io, "_disable_xet_upload_staging", lambda: None)
    monkeypatch.setattr(hf_io, "hf_upload_file", upload)
    monkeypatch.setattr(worker_entry, "_preflight_gpu_occupancy_for_spec", lambda: None)
    monkeypatch.setattr(worker_perf, "_force_fla_triton_gdn_on_sm100", lambda: None)
    monkeypatch.setattr(worker_perf, "_ensure_fla_fastpath_on_hopper", lambda: None)
    monkeypatch.setattr(worker_perf, "_restrict_fla_gdn_autotune_on_blackwell", lambda: None)
    monkeypatch.setattr(worker_perf, "gpu_diagnostics", lambda **_kwargs: {})
    monkeypatch.setattr(heartbeat_io, "heartbeat", heartbeat)
    monkeypatch.setattr(kernel_warmup, "load_mega_cache", load_mega_cache)
    monkeypatch.setattr(sft_entry, "run_sft", run_sft)
    monkeypatch.setattr(
        worker_entry.os,
        "_exit",
        lambda _code: (_ for _ in ()).throw(WorkerExited()),
    )

    with pytest.raises(WorkerExited):
        worker_entry._run_worker_mode()

    assert events == [
        ("heartbeat", "boot", {"gpu": {}}),
        ("load_mega_cache",),
        ("upload", artifact_name, True, expected_payload),
        ("run_sft",),
        ("heartbeat", "sft_start", {"step": 0}),
    ]
    assert remote[artifact_name] == expected_payload
    assert remote["heartbeat.json"] == {"stage": "sft_start", "step": 0}


def test_worker_skips_baked_cache_when_arch_mismatches(monkeypatch, tmp_path):
    cache = tmp_path / kernel_warmup.MEGA_CACHE_FILENAME
    meta = tmp_path / kernel_warmup.MEGA_CACHE_META_FILENAME
    cache.write_bytes(b"cache")
    meta.write_text(json.dumps({"sm": "sm90"}))
    loaded = {"called": False}

    fake_torch = types.SimpleNamespace(
        cuda=_FakeCuda(),
        compiler=types.SimpleNamespace(
            load_cache_artifacts=lambda blob: loaded.__setitem__("called", True)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(kernel_warmup, "DEFAULT_CACHE_DIR", str(tmp_path))

    assert kernel_warmup.load_mega_cache() is False
    assert loaded["called"] is False
    _assert_isolated_backend_paths()


def test_worker_skips_baked_cache_when_arch_undetermined(monkeypatch, tmp_path):
    cache = tmp_path / kernel_warmup.MEGA_CACHE_FILENAME
    meta = tmp_path / kernel_warmup.MEGA_CACHE_META_FILENAME
    cache.write_bytes(b"cache")
    # metadata arch would match a real sm89 worker, but this worker can't determine its own arch
    # (no CUDA) -> we must NOT load a blob we cannot verify; fall back to JIT.
    meta.write_text(json.dumps({"sm": "sm89"}))
    loaded = {"called": False}

    class _NoCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def get_device_capability(index):
            raise RuntimeError("no cuda")

        @staticmethod
        def get_device_name(index):
            return "none"

    fake_torch = types.SimpleNamespace(
        cuda=_NoCuda(),
        compiler=types.SimpleNamespace(
            load_cache_artifacts=lambda blob: loaded.__setitem__("called", True)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(kernel_warmup, "DEFAULT_CACHE_DIR", str(tmp_path))

    assert kernel_warmup.load_mega_cache() is False
    assert loaded["called"] is False
    _assert_isolated_backend_paths()
