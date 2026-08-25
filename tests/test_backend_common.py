"""CPU unit tests for the shared verl subprocess harness (backend_common)."""

from __future__ import annotations

import ast
import builtins
import contextlib
import ctypes
import errno
import http.client
import inspect
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from safetensors.numpy import save

import flash.engine.worker.train.entry.rl_train_runner as rl_train_runner
from flash.engine.worker.perf.lifecycle import RetriableInfraError
from flash.engine.worker.train.core.child import runtime as child_runtime
from flash.engine.worker.train.entry import backend_common as vc
from flash.engine.worker.train.entry import rl_train

# the stamp reads the run's modality off `exclude_modules`: the language-prefix regex for a
# text-only run, None for a multimodal one. these artifacts are text-only.
_TEXT_ONLY_EXCLUDE = r"^(?!model\.language_model(?:\.|$)).*$"

# several tests below drive real subprocesses that import flash from a checkout, not from the venv.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize(
    ("rollout_batch", "expected"),
    [(1, 1), (8, 8), (16, 8), (128, 8), (4, 4), (12, 6), (9, 3), (10, 5), (11, 1), (91, 7)],
)
def test_agent_loop_workers_always_divides_the_rollout_batch(rollout_batch, expected):
    assert vc.agent_loop_workers(rollout_batch) == expected


def test_agent_loop_workers_invariant_holds_for_every_batch():
    # verl asserts the split is exact, so a non-divisor aborts the run before step 1.
    for batch in range(1, 600):
        workers = vc.agent_loop_workers(batch)
        assert batch % workers == 0
        assert 1 <= workers <= min(8, batch)


def test_agent_loop_workers_rejects_a_nonpositive_batch():
    with pytest.raises(ValueError, match="rollout_batch must be positive"):
        vc.agent_loop_workers(0)


def test_ray_num_cpus_prefers_the_cgroup_quota_over_the_host_core_count():
    # the exact failure that killed both real-gpu arms: a 1x4090 pod on a 48-core host. the quota is
    # the container's truth, so a large affinity mask must NOT win over it.
    #
    # os.sched_getaffinity is linux-only, so every patch of it in this file needs create=True to
    # run on a mac. only the worker (linux) ever reaches the real syscall.
    with (
        mock.patch.object(vc, "_cgroup_cpu_quota", return_value=12),
        mock.patch.object(os, "sched_getaffinity", return_value=set(range(48)), create=True),
    ):
        assert vc.ray_num_cpus() == 12
    # a quota BELOW verl's placement-group demand is the one case the quota must not win outright:
    # honouring it exactly would schedule nothing and hang. see the floor test below.
    with (
        mock.patch.object(vc, "_cgroup_cpu_quota", return_value=4),
        mock.patch.object(os, "sched_getaffinity", return_value=set(range(48)), create=True),
    ):
        assert vc.ray_num_cpus() == vc.verl_cpu_demand(1)


def test_ray_num_cpus_falls_back_to_affinity_when_no_quota_is_set():
    with (
        mock.patch.object(vc, "_cgroup_cpu_quota", return_value=None),
        mock.patch.object(os, "sched_getaffinity", return_value=set(range(6)), create=True),
    ):
        assert vc.ray_num_cpus() == 6


def test_ray_num_cpus_caps_an_unconstrained_host():
    # no quota and a full 48-core affinity mask means flash cannot tell the pod from the host, which
    # is exactly the case that forked 48 idle workers. the cap is what makes that survivable.
    with (
        mock.patch.object(vc, "_cgroup_cpu_quota", return_value=None),
        mock.patch.object(os, "sched_getaffinity", return_value=set(range(48)), create=True),
    ):
        assert vc.ray_num_cpus() == 16
        # the cap tightens the pool but cannot push it under verl's placement-group demand, which
        # would hang instead of crash. a cap of 2 therefore yields the 1-gpu floor, not 2.
        assert vc.ray_num_cpus(cap=2) == vc.verl_cpu_demand(1)


def test_ray_num_cpus_never_returns_zero():
    # ray.init(num_cpus=0) schedules nothing and hangs; a bogus quota must not produce it.
    for quota in (0, -1):
        with mock.patch.object(vc, "_cgroup_cpu_quota", return_value=quota):
            assert vc.ray_num_cpus() >= 1
    with (
        mock.patch.object(vc, "_cgroup_cpu_quota", return_value=None),
        mock.patch.object(os, "sched_getaffinity", side_effect=OSError("boom"), create=True),
        mock.patch.object(os, "cpu_count", return_value=None),
    ):
        assert vc.ray_num_cpus() >= 1


def test_cgroup_cpu_quota_reads_v2_and_reports_none_when_uncapped(tmp_path):
    def _read(text):
        path = tmp_path / "cpu.max"
        path.write_text(text)
        real_open = builtins.open

        def fake(name, *a, **k):
            if str(name) == "/sys/fs/cgroup/cpu.max":
                return real_open(path, *a, **k)
            raise FileNotFoundError(name)

        with mock.patch.object(builtins, "open", fake):
            return vc._cgroup_cpu_quota()

    assert _read("400000 100000") == 4
    assert _read("50000 100000") == 1  # a sub-core share still has to schedule something
    assert _read("max 100000") is None


def test_an_uncapped_v2_controller_does_not_fall_through_to_v1(tmp_path):
    # the test above stubs the v1 paths as MISSING, so it cannot see the fallthrough. on a hybrid
    # layout those files exist and carry a parent-scoped or stale limit; reading them after v2 has
    # already answered "uncapped" sizes ray from a container this job is not running in.
    (tmp_path / "cpu.max").write_text("max 100000")
    (tmp_path / "cpu.cfs_quota_us").write_text("200000")
    (tmp_path / "cpu.cfs_period_us").write_text("100000")
    real_open = builtins.open
    routed = {
        "/sys/fs/cgroup/cpu.max": tmp_path / "cpu.max",
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": tmp_path / "cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us": tmp_path / "cpu.cfs_period_us",
    }

    def fake(name, *a, **k):
        if str(name) in routed:
            return real_open(routed[str(name)], *a, **k)
        raise FileNotFoundError(name)

    with mock.patch.object(builtins, "open", fake):
        assert vc._cgroup_cpu_quota() is None


def test_ray_cpu_floor_clears_what_verl_actually_reserves():
    # prevent the opposite hang: verl waits forever for a placement group with too few CPUs.
    # a 1-GPU OPD pod needs five CPUs: three worker bundles, TaskRunner, and one storage unit.
    verl_peak_cpu_demand = 3 + 1 + 1
    assert vc.verl_cpu_demand(1) == verl_peak_cpu_demand
    with (
        mock.patch.object(vc, "_cgroup_cpu_quota", return_value=None),
        mock.patch.object(os, "sched_getaffinity", return_value=set(range(48)), create=True),
    ):
        assert vc.ray_num_cpus() >= verl_peak_cpu_demand


def test_a_small_cgroup_quota_still_clears_verls_placement_group():
    # the floor above only proved the CAP path: it mocks the quota as None, so a pod whose quota is
    # itself below verl's demand walked straight past it. a 2-cpu quota is a normal rented-pod
    # value, and returning 2 hangs forever in ray.get(pg.ready()) -- verl's own resource check
    # validates GPUs only, so nothing reports the shortfall.
    for quota in (1, 2, 4):
        with mock.patch.object(vc, "_cgroup_cpu_quota", return_value=quota):
            assert vc.ray_num_cpus() >= vc.verl_cpu_demand(1)


def test_the_cpu_pool_scales_with_gpu_count():
    # verl requests one bundle PER GPU under STRICT_PACK, so demand grows with the job. flash allows
    # up to 8 gpus (spec.py _MAX_GPU_COUNT), which needs 26 cpus -- above the 16 cap. capping there
    # would hang every large job on a rented, billing gpu.
    assert vc.verl_cpu_demand(8) == 8 * 3 + 2
    with (
        mock.patch.object(vc, "_cgroup_cpu_quota", return_value=None),
        mock.patch.object(os, "sched_getaffinity", return_value=set(range(48)), create=True),
    ):
        for gpus in range(1, 9):
            assert vc.ray_num_cpus(gpus) >= vc.verl_cpu_demand(gpus), gpus


def _trainer_source(module: str) -> str:
    """Every file a trainer's verl config is rendered from, concatenated.

    opd renders its hydra overrides in `train/opd/overrides.py` rather than in `opd_train.py`, so a
    guard that reads only `worker/<module>.py` would stop seeing the opd call sites and pass
    vacuously. These are read by PATH rather than imported because the assertions are on source
    text, so the set of files has to be maintained here when a trainer is split further.
    """
    parts = [f"flash/engine/worker/train/entry/{module}.py"]
    if module == "opd_train":
        parts.append("flash/engine/worker/train/opd/orchestration/overrides.py")
    if module == "rl_train":
        parts.append("flash/engine/worker/train/rl/launch/verl_config.py")
    return "\n".join(pathlib.Path(_REPO_ROOT, p).read_text() for p in parts)


def test_both_trainers_size_the_cpu_pool_from_their_own_gpu_count():
    # a floor that the call sites never pass a gpu count to is a floor that only ever protects
    # 1-gpu jobs, which is the shape of the bug this fixes. pin the wiring, not just the helper.
    for module, expected in (
        ("rl_train", "ray_num_cpus(cfg['n_gpus'])"),
        ("opd_train", "ray_num_cpus(config['n_gpus_per_node'])"),
    ):
        assert expected in _trainer_source(module), module


def test_every_ray_backed_trainer_constrains_rays_cpu_pool():
    # asserted across BOTH ray entrypoints rather than in one file: ray autodetects the host's cores
    # and eagerly forks one idle worker per core, which killed grpo (fatal raylet fork failure) and
    # opd (host-ram oom) on real gpus. sft is excluded on purpose -- it runs torchrun, not ray.
    for module in ("rl_train", "opd_train"):
        assert "ray_kwargs.ray_init.num_cpus={ray_num_cpus(" in _trainer_source(module), module


def test_latest_global_step_dir_picks_highest(tmp_path):
    for step in (1, 5, 20, 3):
        os.makedirs(tmp_path / f"global_step_{step}" / "actor" / "huggingface", exist_ok=True)
    # a non-checkpoint dir must be ignored, not crash the scan.
    os.makedirs(tmp_path / "not_a_step", exist_ok=True)
    actor, step = vc.latest_global_step_dir(str(tmp_path))
    assert step == 20
    assert actor == os.path.join(str(tmp_path), "global_step_20", "actor")


def test_latest_global_step_dir_resolves_the_sft_layout(tmp_path):
    # verl's sft trainer saves straight into global_step_N with no actor/ level. hardcoding the rl
    # convention here pointed the model merger at a path that does not exist, and AutoConfig
    # reinterpreted it as a hub repo id -- so the run died on a bogus HFValidationError.
    for step in (1, 2):
        os.makedirs(tmp_path / f"global_step_{step}" / "huggingface", exist_ok=True)
    actor, step = vc.latest_global_step_dir(str(tmp_path))
    assert step == 2
    assert actor == os.path.join(str(tmp_path), "global_step_2")


def test_resolve_checkpoint_actor_dir_prefers_the_nested_rl_layout(tmp_path):
    # rl checkpoints carry huggingface/ at both levels (the merger wants the actor one).
    os.makedirs(tmp_path / "actor" / "huggingface")
    os.makedirs(tmp_path / "huggingface")
    assert vc.resolve_checkpoint_actor_dir(str(tmp_path)) == os.path.join(str(tmp_path), "actor")


def test_resolve_checkpoint_actor_dir_falls_back_when_no_marker_exists(tmp_path):
    # an interrupted save has no huggingface/ anywhere; name the rl path the caller likely wanted.
    os.makedirs(tmp_path / "actor")
    assert vc.resolve_checkpoint_actor_dir(str(tmp_path)) == os.path.join(str(tmp_path), "actor")
    empty = tmp_path / "empty"
    os.makedirs(empty)
    assert vc.resolve_checkpoint_actor_dir(str(empty)) == str(empty)


def test_latest_global_step_dir_raises_when_empty(tmp_path):
    with pytest.raises(RuntimeError, match="no global_step_N checkpoint"):
        vc.latest_global_step_dir(str(tmp_path))


def test_resolve_verl_loggers_console_when_no_api_key(monkeypatch):
    # no WANDB_API_KEY -> console only, whatever the verl interpreter happens to hold.
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert vc.resolve_verl_loggers({"wandb": True}) == ["console"]


def test_resolve_verl_loggers_enables_wandb_only_when_verl_env_has_it(monkeypatch):
    # api key set AND wandb importable in the verl interpreter -> wandb logger enabled.
    monkeypatch.setenv("WANDB_API_KEY", "k")
    assert vc.resolve_verl_loggers({"wandb": True}) == ["console", "wandb"]


def test_resolve_verl_loggers_falls_back_to_console_when_verl_env_lacks_wandb(monkeypatch):
    # api key set but wandb missing in the verl interpreter -> console only (never aborts verl).
    monkeypatch.setenv("WANDB_API_KEY", "k")
    assert vc.resolve_verl_loggers({"wandb": False}) == ["console"]


def test_resolve_verl_loggers_treats_an_unanswerable_probe_as_no_wandb(monkeypatch):
    # a child that died answers nothing. asking verl for a logger it may not have dies at logger
    # init on a paid gpu, so an unanswered wandb question must degrade to console, never assume it.
    monkeypatch.setenv("WANDB_API_KEY", "k")
    assert vc.resolve_verl_loggers(dict(vc._CAPABILITIES_UNAVAILABLE)) == ["console"]


def test_stamp_adapter_dir_provenance_sets_base_and_revision(tmp_path):
    cfg = tmp_path / "adapter_config.json"
    cfg.write_text(
        json.dumps({"base_model_name_or_path": None, "r": 1, "target_modules": ["q_proj"]})
    )
    prefix = "base_model.model.layers.0.self_attn.q_proj"
    (tmp_path / "adapter_model.safetensors").write_bytes(
        save(
            {
                f"{prefix}.lora_A.weight": np.ones((1, 2), dtype=np.float16),
                f"{prefix}.lora_B.weight": np.ones((2, 1), dtype=np.float16),
            }
        )
    )
    vc.stamp_adapter_dir_provenance(
        str(tmp_path), "org/model", "deadbeef", exclude_modules=_TEXT_ONLY_EXCLUDE
    )
    out = json.loads(cfg.read_text())
    assert out["base_model_name_or_path"] == "org/model"
    assert out["revision"] == "deadbeef"
    # untouched fields survive the stamp.
    assert out["r"] == 1


def test_stamp_adapter_dir_provenance_rejects_base_mismatch(tmp_path):
    cfg = tmp_path / "adapter_config.json"
    cfg.write_text(json.dumps({"base_model_name_or_path": "org/other"}))
    with pytest.raises(RuntimeError, match="does not match validated target"):
        vc.stamp_adapter_dir_provenance(
            str(tmp_path), "org/model", exclude_modules=_TEXT_ONLY_EXCLUDE
        )


def test_resolve_verl_python_prefers_preset(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASH_VERL_PYTHON", "/opt/verl/bin/python")
    assert vc.resolve_verl_python(str(tmp_path)) == "/opt/verl/bin/python"


@pytest.mark.parametrize("blank", ["", "   "])
def test_resolve_verl_python_treats_an_empty_preset_as_unset(monkeypatch, tmp_path, blank):
    # a missing or blank image preset takes the isolated pinned-fork provisioning path.
    calls = []
    monkeypatch.setenv("FLASH_VERL_PYTHON", blank)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))

    python_bin = vc.resolve_verl_python(str(tmp_path))

    assert python_bin.endswith("/verl-venv/bin/python")
    assert any(vc.VERL_REQUIREMENT_URL in arg for arg in calls[1])


def _fake_verl_venv(tmp_path, *, stamp: str | None):
    """materialize a verl-venv as an earlier attempt would have left it on the pod."""
    venv = tmp_path / "verl-venv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    (venv / "bin" / "python").write_text("")
    if stamp is not None:
        (venv / "flash-verl-requirement").write_text(stamp)
    return venv


def _record_run(calls, *, keep_check: bool = False):
    """stand in for subprocess.run, creating the venv dir `uv venv` would have created."""

    def fake_run(command, check, env=None, capture_output=False):
        calls.append((command, check) if keep_check else command)
        if command[:2] == ["uv", "venv"]:
            # the venv path is `uv venv`'s trailing positional, wherever the flags before it end.
            os.makedirs(os.path.join(command[-1], "bin"), exist_ok=True)

    return fake_run


def test_resolve_verl_python_installs_pinned_gpu_dependencies(monkeypatch, tmp_path):
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))

    python_bin = vc.resolve_verl_python(str(tmp_path))

    assert python_bin.endswith("/verl-venv/bin/python")
    # the interpreter is NAMED, not inherited: FLASH_ATTN_SPEC is a cp312-only wheel and flash
    # supports 3.11, so a bare `uv venv` on a 3.11 host builds an interpreter that required install
    # cannot enter -- killing the run during provisioning.
    assert calls[0][:4] == ["uv", "venv", "--python", "3.12"]
    assert "cp312" in vc.FLASH_ATTN_SPEC
    install = calls[1]
    assert vc.VERL_REQUIREMENT == (
        "verl @ git+https://github.com/freesolo-co/verl@f71a02ddb32a9c6a6915f7519bda6dede92e9dd0"
    )
    assert any(vc.VERL_REQUIREMENT_URL in arg for arg in install)
    # liger-kernel is deliberately NOT installed. a matched qwen3.5-9b a/b differing only in
    # model.use_liger measured train/grad_norm 0.0 with liger on versus 7.02 off, at a loss
    # identical to four decimal places: liger silently severed the gradient to the lora params
    # under verl's fsdp2 + peft + gradient-checkpointing composition, so sft trained nothing while
    # looking healthy. with the package absent, use_liger=true raises ImportError at verl's lazy
    # import (transformer_impl.py) instead of silently producing a zero-delta adapter.
    assert "liger-kernel" not in install
    assert "bitsandbytes==0.50.1" in install
    assert "qwen-vl-utils==0.0.14" in install
    assert "torchvision==0.25.0" in install
    assert "xgrammar==0.1.25" in install
    assert "tqdm==4.70.0" in install
    assert "pyarrow==25.0.1" in install
    # venv, the resolve above, the prebuilt flash_attn wheel on its own --no-build-isolation line,
    # then causal_conv1d (also its own line: it source-builds against the venv's torch), the import
    # probe that proves the conv extension actually loaded, and last the libcudart stub repair that
    # keeps vLLM importable in this venv. nothing else: another INSTALL would be unbudgeted work on
    # a paid pod (the repair and the probe are local `python -c` calls, not installs).
    assert len(calls) == 6
    assert calls[2][:3] == ["uv", "pip", "install"]
    assert vc.CAUSAL_CONV1D_REQUIREMENT in calls[3]
    assert calls[4][-1] == "import causal_conv1d"
    assert calls[5][1] == "-c"
    assert "libcudart_stub" in calls[5][2]
    assert [c for c in calls if c[:3] == ["uv", "pip", "install"]] == calls[1:4], (
        "the probe must be a check, not an install"
    )
    # the stamp is written only after a successful install, so a crashed install is never reused.
    stamp = tmp_path / "verl-venv" / "flash-verl-requirement"
    assert stamp.read_text() == vc.VERL_VENV_STAMP


def _flaky_wheel_install(calls, sleeps, *, failures: int):
    """subprocess.run stand-in whose flash-attn install fails the first `failures` attempts."""
    attempts = {"n": 0}

    def fake_run(command, check, env=None, capture_output=False):
        calls.append(command)
        if command[:2] == ["uv", "venv"]:
            os.makedirs(os.path.join(command[-1], "bin"), exist_ok=True)
            return
        # key on the SPEC, not on --no-build-isolation: causal_conv1d passes that flag too, and this
        # helper models a flaky flash-attn download specifically.
        if vc.FLASH_ATTN_INSTALL_SPEC not in command:
            return
        attempts["n"] += 1
        if attempts["n"] <= failures:
            # what uv actually exits with on `error sending request ... operation timed out`.
            raise subprocess.CalledProcessError(1, command)

    return fake_run, attempts


def test_a_transient_wheel_download_failure_is_retried_rather_than_fatal(monkeypatch, tmp_path):
    """uv exits 1 on a download timeout, and one attempt is not evidence the wheel is unreachable.

    uv already retries within one invocation; a fresh invocation is the only retry that can outlast
    a transient failure spanning that process.
    """
    calls, sleeps = [], []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    fake_run, attempts = _flaky_wheel_install(
        calls, sleeps, failures=vc.FLASH_ATTN_INSTALL_ATTEMPTS - 1
    )
    monkeypatch.setattr(vc.subprocess, "run", fake_run)
    monkeypatch.setattr(vc.time, "sleep", sleeps.append)

    python_bin = vc.resolve_verl_python(str(tmp_path))

    assert python_bin.endswith("/verl-venv/bin/python")
    # the whole budget is usable: the last attempt succeeding must still provision the run.
    assert attempts["n"] == vc.FLASH_ATTN_INSTALL_ATTEMPTS
    # and it BACKS OFF between attempts rather than hammering github at the speed of process spawn.
    assert sleeps == [vc.FLASH_ATTN_INSTALL_BACKOFF_S * i for i in (1, 2)]
    # the stamp is the proof the venv is complete -- a retried install still earns it.
    stamp = tmp_path / "verl-venv" / "flash-verl-requirement"
    assert stamp.read_text() == vc.VERL_VENV_STAMP


def test_an_exhausted_wheel_install_hands_the_arm_back_instead_of_burning_it(monkeypatch, tmp_path):
    """The failure must reach the plane as RETRIABLE, which is a claim about _worker_failure_flags.

    Assert the heartbeat flag, not only the exception type, because the poller routes on that flag.
    """
    from flash.engine.worker.entry.worker import _worker_failure_flags
    from flash.engine.worker.perf.lifecycle import RetriableInfraError

    calls, sleeps = [], []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    fake_run, attempts = _flaky_wheel_install(calls, sleeps, failures=99)
    monkeypatch.setattr(vc.subprocess, "run", fake_run)
    monkeypatch.setattr(vc.time, "sleep", sleeps.append)

    with pytest.raises(RetriableInfraError) as caught:
        vc.resolve_verl_python(str(tmp_path))

    # bounded: a genuinely broken wheel spec still terminates. the plane's own INFRA_RETRY_FLOOR
    # bounds the reprovisioning, so a false-positive retry costs attempts, not an unbounded loop.
    assert attempts["n"] == vc.FLASH_ATTN_INSTALL_ATTEMPTS
    assert _worker_failure_flags(caught.value) == {"retriable": True, "oom": False}
    # and the diagnostic must name the url, since that is what distinguishes a network fault from a
    # bad pin when the console is the only artifact a failed arm leaves behind.
    assert vc.FLASH_ATTN_SPEC in str(caught.value)
    # a crashed install must never be reused: no stamp, so the next attempt rebuilds from clean.
    assert not (tmp_path / "verl-venv" / "flash-verl-requirement").exists()


def test_the_fallback_install_overrides_the_three_ceilings_it_violates(monkeypatch, tmp_path):
    """The pin set is deliberately unsatisfiable against declared metadata, so it needs --override.

    A command-line pin remains a constraint; only `--override` ignores verl's incompatible vLLM
    declaration. See Dockerfile.worker:253-258.
    """
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))

    vc.resolve_verl_python(str(tmp_path))

    install = calls[1]
    assert "--override" in install
    override_file = install[install.index("--override") + 1]
    written = pathlib.Path(override_file).read_text()
    # all three are required: dropping any one leaves the set unsatisfiable.
    assert "vllm==0.19.1" in written  # verl[vllm] declares <=0.12.0
    assert "numpy==2.2.6" in written  # verl + transferqueue declare numpy<2.0.0
    assert "xgrammar==0.1.25" in written  # vllm declares xgrammar>=0.1.32


def test_provisioned_venv_can_import_the_entrypoints_flash_launches(monkeypatch, tmp_path):
    # grpo used to stay in flash's own interpreter, so this fallback only ever had to satisfy sft and
    # opd. now every backend routes through it, and Dockerfile.worker records what a verl-only install
    # yields: `main_ppo` dies on ModuleNotFoundError cachetools, `main_ppo_sync` on uvicorn. verl
    # declares none of vllm/cachetools/uvicorn/fastapi yet imports them at module level on the launch
    # path, so an install of VERL_REQUIREMENT alone provisions an interpreter that cannot launch.
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))

    vc.resolve_verl_python(str(tmp_path))

    install = calls[1]
    # the [vllm] extra, as Dockerfile.worker's VERL_SPEC asks for it, on the same commit.
    assert f"verl[vllm] @ {vc.VERL_REQUIREMENT_URL}" in install
    # the extra's own ceiling is vllm<=0.12.0, which registers neither Qwen3.5 arch. pin past it.
    assert "vllm==0.19.1" in install
    for requirement in ("cachetools==7.1.7", "uvicorn==0.52.4", "fastapi==0.141.1"):
        assert requirement in install, (
            f"verl must install its exact direct dependency {requirement}"
        )
    # opd's entrypoint calls tq.init()/tq.close() and verl's setup.py omits TransferQueue.
    assert "TransferQueue==0.1.7" in install


def test_provisioned_venv_gets_flash_attn_for_the_remove_padding_path(monkeypatch, tmp_path):
    # all backends require remove-padding, whose CUDA path imports `flash_attn.bert_padding`
    # unguarded (verl/utils/attention_utils.py:30, torch_functional.py:627). provision the wheel in
    # the child interpreter or the first paid training batch fails.
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))

    vc.resolve_verl_python(str(tmp_path))

    flat = [arg for command in calls for arg in command]
    assert vc.FLASH_ATTN_INSTALL_SPEC in flat, (
        "verl's remove-padding path imports flash_attn unguarded; the provisioned venv must hold it"
    )
    # the wheel is prebuilt, so it must skip build isolation exactly as the image install does --
    # a source build here would compile for minutes on a paid pod.
    install = next(c for c in calls if vc.FLASH_ATTN_INSTALL_SPEC in c)
    assert "--no-build-isolation" in install


def test_flash_attn_spec_stays_in_lockstep_with_the_worker_image():
    # the fallback venv and /opt/verl-venv must resolve the same wheel: a run that lands on the
    # no-image path otherwise trains against a different flash_attn than every baked-image run.
    dockerfile = pathlib.Path(__file__).resolve().parents[1] / "Dockerfile.worker"
    text = dockerfile.read_text()
    assert f"ARG FLASH_ATTN_SPEC={vc.FLASH_ATTN_SPEC}" in text, (
        "Dockerfile.worker's FLASH_ATTN_SPEC default drifted from backend_common.FLASH_ATTN_SPEC"
    )
    # and so must the checksum: the wheel comes from an individual's github release repo, where the
    # asset behind a fixed url can be replaced. two fetch sites verifying different digests would
    # mean the no-image path and the image no longer agree on which bytes are the pinned wheel.
    assert f"ARG FLASH_ATTN_SHA256={vc.FLASH_ATTN_SHA256}" in text, (
        "Dockerfile.worker's FLASH_ATTN_SHA256 default drifted from backend_common.FLASH_ATTN_SHA256"
    )
    # the install must actually ask for the digest: a pinned constant nothing hands to uv leaves
    # the fetch as unverified as it was without one.
    assert f"{vc.FLASH_ATTN_SPEC}#sha256={vc.FLASH_ATTN_SHA256}" == vc.FLASH_ATTN_INSTALL_SPEC
    # and the verl-venv layer must build that fragment CONDITIONALLY. `#sha256=` is url syntax, so
    # gluing it onto the documented non-url override (FLASH_ATTN_SPEC=flash-attn==X) hands uv a
    # requirement it cannot parse and fails this REQUIRED layer. pin both halves: the install takes
    # a shell-resolved spec, and the fragment is added only inside the http* case.
    install_line = (
        'uv pip install --python /opt/verl-venv/bin/python --no-build-isolation "${spec}"'
    )
    assert install_line in text, (
        "the verl-venv flash-attn install must install a shell-resolved spec, not the raw ARG"
    )
    block = text[text.rindex("\nRUN ", 0, text.index(install_line)) : text.index(install_line)]
    assert "http*)" in block, (
        "the verl-venv install must append the sha256 fragment only for an http(s) wheel url"
    )
    assert 'spec="${spec}#sha256=${FLASH_ATTN_SHA256}"' in block, (
        "the verl-venv install must still hash-verify a wheel url"
    )
    assert '"${FLASH_ATTN_SPEC}#sha256=' not in text, (
        "appending the fragment straight onto FLASH_ATTN_SPEC breaks the non-url override"
    )


def test_the_capability_probe_imports_nothing_from_flash():
    # the verl child cannot import Flash; doing so made the GDN probe fail closed permanently.
    # now one child answers every capability, so assert its parsed import graph contains no `flash`
    # import while allowing names such as flashinfer.
    probe = vc._CAPABILITY_PROBE % {"gdn_module": "transformers.models.qwen3_5.modeling_qwen3_5"}
    roots = set()
    for node in ast.walk(ast.parse(probe)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".")[0])
    assert "flash" not in roots, (
        f"the verl child cannot import flash; probe would always fail closed. imports: {sorted(roots)}"
    )
    assert roots <= {
        "causal_conv1d",
        "flashinfer",
        "importlib",
        "inspect",
        "json",
        "torch",
        # stdlib, imported only inside the gdn handler to report WHY the answer was negative.
        "traceback",
        "transformers",
        "verl",
        "wandb",
    }, f"probe reaches for a package the verl child is not guaranteed to have: {sorted(roots)}"


def test_no_optional_package_is_imported_at_the_probes_top_level():
    """THE structural risk batching introduces: one process now answers every question.

    Only stdlib imports may be top-level; optional imports must remain inside each question's
    try/except or one missing package fail-closes every capability.
    """
    probe = vc._CAPABILITY_PROBE % {"gdn_module": "transformers.models.qwen3_5.modeling_qwen3_5"}
    tree = ast.parse(probe)
    top_level = set()
    for node in tree.body:  # deliberately NOT ast.walk: only module-level statements
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level.add((node.module or "").split(".")[0])
    assert top_level <= {"json"}, (
        "an optional package imported at the probe's top level kills the whole child and "
        f"fail-closes EVERY capability, not just its own: {sorted(top_level)}"
    )


def _run_gdn_probe_branch(
    monkeypatch, tmp_path, *, forward_src, gdn_present=True, module_source=None
):
    """Execute the probe's gdn branch against a stub transformers module, returning its stdout.

    Run the real rendered probe with package checks true so only source inspection can fail.
    `module_source` supports class-shape cases such as decorated forwards.
    """
    import importlib.util
    import io as _io
    import sys as _sys
    import types as _types

    # the probe imports causal_conv1d before the source check to fail a broken ABI early; this
    # interpreter has no gpu wheel, so stub it. the case under test is the check AFTER it.
    monkeypatch.setitem(_sys.modules, "causal_conv1d", _types.ModuleType("causal_conv1d"))

    module_name = "flash_test_gdn_probe_mod"
    stub = _types.ModuleType(module_name)

    if gdn_present:
        # the probe reads the forward with inspect.getsource, which needs a real file on disk --
        # an exec'd function raises OSError and would land in the traceback branch instead of the
        # clean-negative one under test. so write the stub module out and import it for real.
        source = module_source or (
            f"class Qwen3GatedDeltaNet:\n    def forward(self):\n        {forward_src}\n"
        )
        path = tmp_path / f"{module_name}.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location(module_name, path)
        stub = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stub)

    monkeypatch.setitem(_sys.modules, module_name, stub)

    import transformers.utils.import_utils as import_utils

    monkeypatch.setattr(import_utils, "is_flash_linear_attention_available", lambda: True)
    monkeypatch.setattr(import_utils, "is_causal_conv1d_available", lambda: True)

    probe = vc._CAPABILITY_PROBE % {"gdn_module": module_name}
    # keep only the gdn block: the rest of the probe imports verl/flashinfer/wandb, which this
    # process has no reason to have, and their absence would drown the line under test.
    start = probe.index(f"if {module_name!r}:")
    captured = _io.StringIO()
    # record what the probe ANSWERS, not just what it printed: a positive case prints nothing, so
    # stdout alone cannot distinguish "answered yes" from "never reached the check".
    answers: dict = {}
    namespace = {"emit": lambda key, value: answers.__setitem__(key, value)}
    with contextlib.redirect_stdout(captured):
        exec(probe[start:], namespace)
    _run_gdn_probe_branch.last_answers = answers
    return captured.getvalue()


def test_a_gdn_source_check_failure_does_not_tell_operators_to_install_present_packages(
    monkeypatch, tmp_path
):
    # the diagnostic feeds require_gdn_boundary_resets, whose raise says "installing fla +
    # causal_conv1d in the verl interpreter is the fix" and points AT this line. when the source
    # check is what failed, both packages are already installed -- so printing "fla=True
    # causal_conv1d=True" sends an operator to reinstall something that is not broken, on a paid
    # gpu, with no other clue about which of the three checks fell over.
    out = _run_gdn_probe_branch(monkeypatch, tmp_path, forward_src="return None")

    assert "gdn boundary resets unavailable" in out
    assert "fla=True causal_conv1d=True" not in out, (
        "a source-check failure reported both packages as present, which reads as "
        "'install these' for packages that ARE installed"
    )
    assert "installing packages will not fix it" in out
    assert "cu_seq_lens_q" in out
    assert "seq_idx" in out


def test_a_decorated_gdn_forward_is_read_through_to_the_real_body(monkeypatch, tmp_path):
    """A wrapped forward must not read as "this transformers build cannot reset gdn state".

    `force_accelerate_hooks` hides the real kwargs-reading body from `inspect.getsource` and sets no
    `__wrapped__`, so the probe must read through the closure. This fixture preserves that shape.
    """
    module_source = (
        "def force_accelerate_hooks(child_module_name):\n"
        "    def decorator(func):\n"
        "        def wrapped(self, *args, **kwargs):\n"
        "            hooked = getattr(self, child_module_name, None)\n"
        "            return func(self, *args, **kwargs)\n"
        "        return wrapped\n"
        "    return decorator\n"
        "\n"
        "class Qwen3GatedDeltaNet:\n"
        "    @force_accelerate_hooks('conv1d')\n"
        "    def forward(self, hidden_states, cache_params=None, **kwargs):\n"
        "        return (kwargs.get('cu_seq_lens_q'), kwargs.get('seq_idx'))\n"
    )
    # the only POSITIVE case here, so it is the only one that reaches the probe's post-check cuda
    # smoke. this interpreter has no torch; stub it reporting no gpu, which is the branch a cpu box
    # takes anyway. without this the assertion below would pass for the wrong reason -- an
    # ImportError landing in the traceback branch rather than the source check answering yes.
    import importlib.machinery as _machinery
    import sys as _sys
    import types as _types

    torch_stub = _types.ModuleType("torch")
    torch_stub.__spec__ = _machinery.ModuleSpec("torch", loader=None)
    torch_stub.cuda = _types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(_sys.modules, "torch", torch_stub)

    out = _run_gdn_probe_branch(monkeypatch, tmp_path, forward_src="", module_source=module_source)

    assert "gdn boundary resets unavailable" not in out, (
        "a decorated forward that DOES thread cu_seq_lens_q and seq_idx was reported as incapable; "
        "grpo and opd then die on every gdn model with advice that cannot fix it"
    )
    # silence is not the answer -- a positive case prints nothing, so assert the affirmative the
    # caller actually gates on. otherwise this passes just as well when the branch never ran.
    assert _run_gdn_probe_branch.last_answers.get("gdn_boundary_resets") is True


def test_a_gdn_probe_names_the_single_missing_forward_kwarg(monkeypatch, tmp_path):
    # a forward that takes one of the two is a different repair from one that takes neither, so
    # the line must not collapse both into the same text.
    out = _run_gdn_probe_branch(monkeypatch, tmp_path, forward_src="return cu_seq_lens_q")

    assert "missing seq_idx" in out, out
    assert "takes neither" not in out


def test_a_gdn_probe_reports_a_missing_class_rather_than_a_missing_kwarg(monkeypatch, tmp_path):
    # no *GatedDeltaNet at all is a wrong-module/wrong-transformers problem, not a kwarg one.
    out = _run_gdn_probe_branch(monkeypatch, tmp_path, forward_src="", gdn_present=False)

    assert "no *GatedDeltaNet class" in out, out


def test_a_missing_package_still_names_which_package(monkeypatch):
    # the case the old line DID handle correctly must keep working: here "install it" is right.
    import transformers.utils.import_utils as import_utils

    monkeypatch.setattr(import_utils, "is_flash_linear_attention_available", lambda: False)
    monkeypatch.setattr(import_utils, "is_causal_conv1d_available", lambda: True)

    import io as _io

    probe = vc._CAPABILITY_PROBE % {"gdn_module": "flash_test_absent_mod"}
    start = probe.index(f"if {'flash_test_absent_mod'!r}:")
    captured = _io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(probe[start:], {"emit": lambda *_a, **_k: None})
    out = captured.getvalue()

    assert "fla=False causal_conv1d=True" in out, out
    assert "install the missing package" in out


def test_the_capability_probe_is_valid_python():
    # the probe is a template string, so a syntax error in it would surface as a silent
    # "every capability unavailable" on a paid gpu rather than at import time here. check BOTH
    # renderings: the gdn branch is interpolated in, and the skip case must parse too.
    probe = vc._CAPABILITY_PROBE % {"gdn_module": "transformers.models.qwen3_6.modeling_qwen3_6"}
    ast.parse(probe)
    assert "transformers.models.qwen3_6.modeling_qwen3_6" in probe
    ast.parse(vc._CAPABILITY_PROBE % {"gdn_module": ""})


def test_every_capability_question_fails_independently():
    # the batching's core risk: five questions in one process means one raising exception could
    # take the other four with it, silently turning a healthy run's rollout field / capability /
    # flashinfer answers into fail-closed defaults. each question owns its own try/except, so a
    # child with NO verl, NO cuda, NO flashinfer and NO wandb must still answer every key.
    caps = vc.probe_verl_capabilities(
        sys.executable, "transformers.models.qwen3_5.modeling_qwen3_5"
    )
    assert set(caps) == set(vc._CAPABILITIES_UNAVAILABLE)
    # this interpreter has none of them, so every answer is the fail-closed one -- but each was
    # REACHED and answered rather than skipped by an earlier raise.
    assert caps["flashinfer"] is False
    assert caps["wandb"] is False


def test_the_probe_skips_the_gdn_question_for_a_non_hybrid():
    # the gdn smoke runs a live CUDA kernel and imports transformers; nothing consumes its answer
    # for a pure-attention model, so an empty module must skip it outright rather than pay for it.
    probe = vc._CAPABILITY_PROBE % {"gdn_module": ""}
    assert "causal_conv1d_fn(" in probe  # the branch is present...
    caps = vc.probe_verl_capabilities(sys.executable, "")
    assert caps["gdn_boundary_resets"] is None  # ...but was never entered, so it stays unasked


def test_the_gate_hands_the_shim_the_arch_it_verified(monkeypatch):
    # derive model_type from the child module so the gate and shim cannot disagree. patch the module
    # object because test cleanup may replace the parent package and break dotted monkeypatch
    # lookup.
    import flash.engine.worker.model.packing as _packing

    monkeypatch.setattr(_packing, "gdn_model_type", lambda *a, **k: "qwen3_5_moe")
    gdn_module = vc.gdn_probe_module("Qwen/Qwen3.6-35B-A3B")
    assert gdn_module == "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
    arch = vc.gdn_reset_arch_from_caps({"gdn_boundary_resets": True}, gdn_module)
    assert arch == "qwen3_5_moe"
    assert arch == "qwen3_5_moe"


def test_the_verified_arch_round_trips_for_the_dense_module_too(monkeypatch):
    # the MoE/dense split is the bug this pairing exists to prevent, so prove the recovery is exact
    # for both -- a prefix-stripping slip that returned "qwen3_5" for the MoE module would patch an
    # unrelated forward while the gate reported resets active.
    import flash.engine.worker.model.packing as _packing

    monkeypatch.setattr(_packing, "gdn_model_type", lambda *a, **k: "qwen3_5")
    gdn_module = vc.gdn_probe_module("Qwen/Qwen3.5-9B")
    assert vc.gdn_reset_arch_from_caps({"gdn_boundary_resets": True}, gdn_module) == "qwen3_5"


@pytest.mark.parametrize("answer", [False, None])
def test_the_gdn_boundary_gate_treats_an_unconvincing_probe_as_unavailable(answer):
    # fail-closed is the whole safety property: anything short of an explicit affirmative must mean
    # "train padded", never "assume the kernels are there". `False` is the child answering "kernels
    # absent" or its own question raising; `None` is the child never answering at all.
    module = "transformers.models.qwen3_5.modeling_qwen3_5"
    assert vc.gdn_reset_arch_from_caps({"gdn_boundary_resets": answer}, module) is None


@pytest.mark.parametrize(
    ("returncode", "stdout"),
    [
        (1, ""),  # child missing verl/transformers: the whole probe raises
        (0, "0"),  # child ran but printed no capability blob
        (0, ""),  # child produced nothing readable
        (0, "FLASH_VERL_CAPS=not-json"),  # blob present but corrupt
    ],
)
def test_an_unconvincing_child_leaves_every_capability_fail_closed(monkeypatch, returncode, stdout):
    # the batched probe multiplies the blast radius of a dead child, so its failure must land on
    # the SAME fail-closed answer each separate subprocess used to produce -- never a partial dict
    # a caller would read as an affirmative.
    monkeypatch.setattr(
        vc.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=returncode, stdout=stdout, stderr=""),
    )
    caps = vc.probe_verl_capabilities("/nonexistent/python", "m.o.d")
    assert caps == vc._CAPABILITIES_UNAVAILABLE
    assert vc.gdn_reset_arch_from_caps(caps, "m.o.d") is None
    assert vc.verl_declares_rollout_field(caps, "mask_truncated_completions") is False
    assert vc.verl_device_capability(caps) is None


def test_a_slow_gdn_smoke_cannot_retract_the_answers_already_flushed(monkeypatch):
    # the regression the batching introduced: five subprocesses each had their own timeout, so a
    # wedged gdn smoke could not touch the cheap answers. one shared child re-couples them unless
    # answers are flushed incrementally AND the parent keeps what a killed child already sent.
    # losing rollout_fields here is not a soft degrade -- grpo raises on it and the paid run dies.
    early = "".join(
        "FLASH_VERL_CAPS=" + json.dumps({k: v}) + "\n"
        for k, v in (
            ("rollout_fields", ["mask_truncated_completions"]),
            ("capability", [9, 0]),
            ("flashinfer", True),
            ("wandb", True),
        )
    )

    def _timeout(*a, **k):
        # the gdn question never answered: the child was killed inside the live cuda kernel.
        raise vc.subprocess.TimeoutExpired(cmd="python", timeout=1, output=early, stderr="")

    monkeypatch.setattr(vc.subprocess, "run", _timeout)
    caps = vc.probe_verl_capabilities("/verl/bin/python", "m.o.d")

    assert vc.verl_declares_rollout_field(caps, "mask_truncated_completions") is True
    assert vc.verl_device_capability(caps) == (9, 0)
    assert caps["flashinfer"] is True
    assert caps["wandb"] is True
    # only the unreached question fails closed, and it fails to the safe answer: train padded.
    assert caps["gdn_boundary_resets"] is None
    assert vc.gdn_reset_arch_from_caps(caps, "m.o.d") is None


def test_a_timed_out_child_that_answered_nothing_still_fails_every_capability_closed(monkeypatch):
    # keeping partial answers must not weaken the total-failure path: a child killed before it
    # flushed anything is indistinguishable from a dead one and must fail closed everywhere.
    def _timeout(*a, **k):
        raise vc.subprocess.TimeoutExpired(cmd="python", timeout=1, output="", stderr="")

    monkeypatch.setattr(vc.subprocess, "run", _timeout)
    assert vc.probe_verl_capabilities("/verl/bin/python", "m.o.d") == vc._CAPABILITIES_UNAVAILABLE


def test_the_probe_budget_covers_the_union_of_the_probes_it_replaced():
    # the batched child does serially what five children did in parallel processes, so its budget is
    # the SUM of theirs. the gdn smoke used to get a full 600s with no torch/verl import ahead of
    # it; a 600s shared bound would hand it strictly less wall-clock than before on a cold cache.
    assert vc._CAPABILITY_PROBE_TIMEOUT_S == 300 + 600 + 120 + 120 + 120


def test_the_gdn_answer_is_flushed_after_the_cheap_ones():
    # ordering is the contract that makes incremental flushing worth anything: if the expensive
    # question ran first, a timeout inside it would still take every cheap answer with it.
    probe = vc._CAPABILITY_PROBE % {"gdn_module": "transformers.models.qwen3_5.modeling_qwen3_5"}
    gdn_at = probe.index('emit("gdn_boundary_resets"')
    for key in ("rollout_fields", "capability", "flashinfer", "wandb"):
        assert probe.index(f'emit("{key}"') < gdn_at, (
            f"{key} must be flushed before the gdn smoke, which is the question that can hang"
        )


def test_a_child_that_omits_a_key_leaves_it_fail_closed(monkeypatch):
    # a child from a different flash build can answer a subset. the missing key must keep its
    # fail-closed default rather than vanishing from the dict and turning every caller's
    # `caps.get(...)` into an implicit None that reads differently per call site.
    monkeypatch.setattr(
        vc.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout='FLASH_VERL_CAPS={"flashinfer": true}\n', stderr=""
        ),
    )
    caps = vc.probe_verl_capabilities("/verl/bin/python", "")
    assert set(caps) == set(vc._CAPABILITIES_UNAVAILABLE)
    assert caps["flashinfer"] is True
    assert caps["capability"] is None
    assert caps["wandb"] is False


def test_an_unavailable_gdn_gate_says_why():
    """The gdn probe must print WHY it answered no, in the child, where the cause exists.

    The parent sees only a boolean, so missing packages and broken ABIs are otherwise identical.
    Assert both negative-path diagnostics in the rendered source; syntax is tested separately.
    """
    probe = vc._CAPABILITY_PROBE
    # the clean negative: which of the checks failed is the difference between "install fla" and
    # "rebuild causal_conv1d for this arch".
    assert probe.count("gdn boundary resets unavailable in the child") == 2, (
        "both gdn negative paths must say why: the clean negative (checks returned False) and the "
        "exception path. a bare emit(False) on either leaves the parent's raise unexplainable."
    )
    _, _, tail = probe.partition("    except Exception as e:")
    assert tail, "the gdn handler no longer binds its exception, so it cannot report the cause"
    assert "traceback" in tail.split('emit("gdn_boundary_resets", False)')[0], (
        "the gdn exception handler discards the traceback. that is exactly the bare `except: "
        "emit(False)` that made a kernel-less child indistinguishable from a broken probe."
    )


def test_child_diagnostics_survive_the_answered_early_return(monkeypatch, capsys):
    """A child `[verl] ...` line must reach the parent log EVEN WHEN the probe answered.

    The GDN question runs last, so an early return after another answer discarded its diagnostic.
    The fixture must both answer `flashinfer` and emit the line to exercise that exact path.
    """
    monkeypatch.setattr(
        vc.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0,
            stdout=(
                "[verl] gdn boundary resets unavailable in the child: fla=True causal_conv1d=False\n"
                'FLASH_VERL_CAPS={"flashinfer": true}\n'
            ),
            stderr="",
        ),
    )
    caps = vc.probe_verl_capabilities(
        "/verl/bin/python", "transformers.models.qwen3_5.modeling_qwen3_5"
    )
    # the early return still happens -- this is not asking the probe to stop short-circuiting.
    assert caps["flashinfer"] is True
    out = capsys.readouterr().out
    assert "gdn boundary resets unavailable" in out, (
        "the child's diagnostic was swallowed by the answered-early-return. the gate's raise points "
        "operators at that line, so losing it makes a kernel-less child indistinguishable from a "
        "broken probe -- exactly the silence this change set out to remove."
    )
    # and the CAUSE has to survive, not just the verdict.
    assert "causal_conv1d=False" in out


def test_the_gdn_gate_has_exactly_three_outcomes():
    """None for non-gdn, an arch for a resettable hybrid, a raise for everything else.

    Callers equate a returned arch with GDN, so a hybrid must not return None and skip the
    boundary shim. Missing, timed-out, or partial answers fail closed by raising.
    """
    module = "transformers.models.qwen3_5.modeling_qwen3_5"

    # non-gdn: the parent passes "" and never asked the child, so there is nothing to check.
    assert vc.require_gdn_boundary_resets({"gdn_boundary_resets": None}, "") is None

    # gdn + child cleared: the arch, recovered from the literal the child was handed.
    assert vc.require_gdn_boundary_resets({"gdn_boundary_resets": True}, module) == "qwen3_5"

    # gdn + any non-affirmative answer: raise. never None.
    for caps in ({"gdn_boundary_resets": False}, {"gdn_boundary_resets": None}, {}):
        with pytest.raises(RuntimeError, match="without child-side boundary resets"):
            vc.require_gdn_boundary_resets(caps, module)


def test_fla_stays_in_lockstep_with_the_worker_image():
    # the fallback venv and /opt/verl-venv must hold the SAME fla commit. gdn boundary resets are a
    # correctness property, not a speed one: a child on a different fla could thread cu_seqlens
    # differently, and the two paths would train the same config differently.
    text = (pathlib.Path(__file__).resolve().parents[1] / "Dockerfile.worker").read_text()
    sha = vc.FLA_REQUIREMENT.rsplit("@", 1)[1]
    assert text.count(sha) >= 2, (
        "Dockerfile.worker must pin the same fla sha as backend_common.FLA_REQUIREMENT in BOTH the "
        "main interpreter and the verl venv"
    )
    assert text.count(vc.CAUSAL_CONV1D_REQUIREMENT) >= 2, (
        "causal_conv1d must be installed into the verl venv as well as the main interpreter; the "
        "two have disjoint paths and the model trains in the verl one"
    )


def test_the_verl_venv_gets_the_gdn_kernels(monkeypatch, tmp_path):
    # the model trains in THIS interpreter. without fla here, transformers binds
    # chunk_gated_delta_rule to a pure-torch fallback that accepts cu_seqlens and discards it, so the
    # boundary shim is inert and packed gdn training is contaminated while looking patched.
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))

    vc.resolve_verl_python(str(tmp_path))

    flat = [arg for command in calls for arg in command]
    assert vc.FLA_REQUIREMENT in flat, (
        "the verl child must hold fla or boundary resets are discarded"
    )


def test_causal_conv1d_install_is_best_effort_and_leaves_no_env_residue(monkeypatch):
    # best-effort on purpose: failing the provisioning would turn a compiler hiccup into a dead paid
    # run, and sft is fine without the kernel (one example per update, so nothing to contaminate).
    # the OUTCOME is not optional though -- see the stamp tests below, since grpo/opd raise without
    # it. and the build flag must not leak into the environment verl inherits from this process.
    seen = {}

    def fake_run(command, check, env=None, capture_output=False):
        # the install is the call carrying the requirement; the later call is the import probe.
        if vc.CAUSAL_CONV1D_REQUIREMENT in command:
            seen["command"], seen["check"], seen["env"] = command, check, env
        return _Completed(0)

    monkeypatch.delenv("CAUSAL_CONV1D_FORCE_BUILD", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", fake_run)

    vc._install_causal_conv1d("/venv/bin/python")

    assert vc.CAUSAL_CONV1D_REQUIREMENT in seen["command"]
    assert seen["check"] is False, "a failed conv build must not kill provisioning"
    # compiles against the venv's torch, so build isolation would resolve a different one.
    assert "--no-build-isolation" in seen["command"]
    assert seen["env"]["CAUSAL_CONV1D_FORCE_BUILD"] == "TRUE"
    assert "CAUSAL_CONV1D_FORCE_BUILD" not in os.environ, (
        "build flag leaked into verl's environment"
    )


class _Completed:
    def __init__(self, returncode):
        self.returncode = returncode


def _record_run_with_conv_exit(calls, conv_exit, *, import_exit=0, cudart_exit=0):
    """``_record_run``, but the causal-conv1d install exits with ``conv_exit``.

    ``import_exit`` drives the separate ``import causal_conv1d`` probe, because a compiled cuda
    extension can install cleanly (exit 0) and still fail to import on an ABI mismatch.
    ``cudart_exit`` drives the child libcudart stub repair, which exits nonzero when it leaves the
    stub shadowing libcudart.
    """
    from flash.engine.worker.verl.capabilities import _CHILD_CUDART_FIX

    inner = _record_run(calls)

    def fake_run(command, check, env=None, capture_output=False):
        inner(command, check, env, capture_output)
        if vc.CAUSAL_CONV1D_REQUIREMENT in command:
            return _Completed(conv_exit)
        if command[-1] == "import causal_conv1d":
            return _Completed(import_exit)
        if command[-1] == _CHILD_CUDART_FIX:
            return _Completed(cudart_exit)
        return _Completed(0)

    return fake_run


def test_a_missed_conv_build_leaves_the_venv_unstamped_so_the_next_attempt_rebuilds(
    monkeypatch, tmp_path
):
    """A best-effort install may fail; it may not be recorded as a complete provisioning.

    Stamping after a missed GDN kernel build makes the broken venv permanently reusable.
    """
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run_with_conv_exit(calls, 1))

    vc.resolve_verl_python(str(tmp_path))

    stamp = tmp_path / "verl-venv" / "flash-verl-requirement"
    assert not stamp.exists(), "a venv missing the conv kernel was stamped as fully provisioned"
    # the build was still attempted, and still best-effort: provisioning did not raise
    assert any(vc.CAUSAL_CONV1D_REQUIREMENT in command for command in calls)


def test_a_successful_conv_build_still_stamps_the_venv(monkeypatch, tmp_path):
    """The guard must key on the build outcome, not stop stamping altogether."""
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run_with_conv_exit(calls, 0))

    vc.resolve_verl_python(str(tmp_path))

    stamp = tmp_path / "verl-venv" / "flash-verl-requirement"
    assert stamp.read_text() == vc.VERL_VENV_STAMP


def test_an_unrepaired_child_cudart_stub_leaves_the_venv_unstamped(monkeypatch, tmp_path):
    """A stub still shadowing libcudart must not be recorded as a complete provisioning.

    The repair runs only on the rebuild path, so stamping here freezes the failure in: every later
    attempt on this pod reuses a venv whose stamp asserts it is provisioned while its child still
    aborts on vLLM import, and the repair is never attempted again.
    """
    from flash.engine.worker.verl.capabilities import _CHILD_CUDART_FIX

    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run_with_conv_exit(calls, 0, cudart_exit=1))

    vc.resolve_verl_python(str(tmp_path))

    stamp = tmp_path / "verl-venv" / "flash-verl-requirement"
    assert not stamp.exists(), (
        "a venv whose child libcudart stub was left shadowing was stamped as fully provisioned, so "
        "every later attempt on this pod reuses an interpreter where vLLM aborts its import"
    )
    # still fails open within this launch: provisioning returned an interpreter rather than raising.
    assert any(command[-1] == _CHILD_CUDART_FIX for command in calls)


def test_a_repaired_child_cudart_stub_still_stamps_the_venv(monkeypatch, tmp_path):
    """The guard keys on the repair outcome; it must not stop stamping altogether."""
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run_with_conv_exit(calls, 0, cudart_exit=0))

    vc.resolve_verl_python(str(tmp_path))

    stamp = tmp_path / "verl-venv" / "flash-verl-requirement"
    assert stamp.read_text() == vc.VERL_VENV_STAMP


def test_a_conv_build_that_installs_but_cannot_import_leaves_the_venv_unstamped(
    monkeypatch, tmp_path
):
    """A zero pip exit is not the kernel landing.

    `causal_conv1d` can install with an ABI mismatch; gate on the real import rather than
    transformers' `find_spec` check.
    """
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run_with_conv_exit(calls, 0, import_exit=1))

    vc.resolve_verl_python(str(tmp_path))

    stamp = tmp_path / "verl-venv" / "flash-verl-requirement"
    assert not stamp.exists(), (
        "a venv whose conv kernel installed but cannot import was stamped as fully provisioned, so "
        "every later attempt on this pod reuses it and grpo/opd stay unrunnable"
    )
    # and the broken build must not be left importable-looking for the next find_spec probe.
    assert any(c[:3] == ["uv", "pip", "uninstall"] for c in calls), (
        "a conv build that cannot import was left installed"
    )


def test_the_venv_stamp_covers_the_conv_kernel_so_an_older_venv_is_rebuilt():
    """The stricter half of the fla argument.

    The stamp must name the conv requirement so reused pre-fix venvs rebuild instead of repeatedly
    failing `require_gdn_boundary_resets`.
    """
    assert vc.CAUSAL_CONV1D_REQUIREMENT in vc.VERL_VENV_STAMP


def test_the_venv_stamp_covers_build_time_repairs_so_a_pre_fix_venv_is_rebuilt():
    """The same argument for a repair rather than a package.

    The libcudart repair runs only on the rebuild path, so a venv stamped by a release that predates
    it matches the stamp, is reused unrepaired, and keeps tilelang's stub -- vLLM then aborts its
    import in the child AFTER the gpu is rented. The stamp has to change when the repair set does,
    or the fix never reaches the venvs that need it most (retries and upgraded workers that preserve
    the workdir).
    """
    assert vc.VERL_VENV_BUILD_REPAIRS in vc.VERL_VENV_STAMP
    # and it must be the LAST field: appending keeps the earlier fields' offsets stable, so a stamp
    # written by this release is still diffable against one from before the repairs existed.
    assert vc.VERL_VENV_STAMP.endswith(vc.VERL_VENV_BUILD_REPAIRS)


def test_the_fallback_pins_transformers_like_the_image_does(monkeypatch, tmp_path):
    """The fallback venv must carry the same transformers ceiling as /opt/verl-venv.

    This path provisions the interpreter that TRAINS when no image supplies one, and transformers
    owns the gdn modelling code the boundary-reset shim patches. An unbounded resolve there puts
    training on a transformers line nothing validated. verl and vllm both depend on transformers
    with no upper bound, so the pin has to be in the override file too: a direct pin alone loses to
    their transitive declarations.

    The pin is NOT what makes the cuda-gated probes answer True -- those are identical across
    5.12.1 and 5.14.1 and depend on a gpu being present, not on the transformers version.
    """
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))

    vc.resolve_verl_python(str(tmp_path))

    install = calls[1]
    override_file = install[install.index("--override") + 1]
    assert vc.TRANSFORMERS_INSTALL_REQUIREMENT in pathlib.Path(override_file).read_text()
    assert vc.TRANSFORMERS_INSTALL_REQUIREMENT in install


def test_transformers_pin_stays_in_lockstep_with_the_worker_image():
    # same argument as VERL_SPEC lockstep: the image bakes its own pin, this path resolves its own.
    # if they drift, the interpreter that TRAINS runs a different transformers depending on whether
    # an image supplied it -- and transformers owns the gdn modelling code the shim patches.
    root = pathlib.Path(__file__).resolve().parents[1]
    dockerfile = root / "Dockerfile.worker"
    text = dockerfile.read_text()
    install_quoted = f'"{vc.TRANSFORMERS_INSTALL_REQUIREMENT}"'
    assert text.count(install_quoted) == 3, (
        "Dockerfile.worker's main, override, and venv transformers pins drifted from "
        "backend_common.TRANSFORMERS_INSTALL_REQUIREMENT"
    )
    # the exact override prevents transitive declarations from re-widening the deployed version.
    assert f'"{vc.TRANSFORMERS_INSTALL_REQUIREMENT}" > /tmp/verl-overrides.txt' in text

    # pyproject declares the same range for the gpu/server/dev extras, and a lower ceiling there
    # puts anyone installing the package (local worker, self-hosted plane) on a transformers line
    # the image never runs.
    pyproject = (root / "pyproject.toml").read_text()
    declared = set(re.findall(r'"(transformers>=[^"]+)"', pyproject))
    assert declared == {vc.TRANSFORMERS_REQUIREMENT}, (
        f"pyproject.toml transformers ranges {sorted(declared)} drifted from "
        f"backend_common.TRANSFORMERS_REQUIREMENT ({vc.TRANSFORMERS_REQUIREMENT!r}); every extra "
        "must declare the range the worker image is built and tested against"
    )
    # and every extra that names transformers at all must name it (four today: gpu, server, dev,
    # serve-runtime).
    assert pyproject.count(f'"{vc.TRANSFORMERS_REQUIREMENT}"') == 4, (
        "expected the transformers range in exactly the gpu, server, dev and serve-runtime extras"
    )


def test_the_venv_stamp_covers_the_transformers_pin_so_a_prepin_venv_is_rebuilt():
    # a venv provisioned before this pin may hold a different resolved transformers. the stamp must
    # cover the exact install so a retry rebuilds it instead of reusing drift.
    assert vc.TRANSFORMERS_INSTALL_REQUIREMENT in vc.VERL_VENV_STAMP


def test_the_venv_stamp_covers_fla_so_a_prefla_venv_is_rebuilt(monkeypatch, tmp_path):
    # a workdir provisioned by a release that predates the fla install holds no fla. if the stamp
    # ignored fla, a retry would reuse that venv, skip provisioning, and train gdn models on the
    # discarding fallback forever -- the exact bug this change exists to close.
    assert vc.FLA_REQUIREMENT in vc.VERL_VENV_STAMP


def test_verl_spec_stays_in_lockstep_with_the_worker_image():
    # the default image exports FLASH_VERL_PYTHON, so its baked interpreter bypasses the fallback
    # resolver entirely. both paths must pin the same commit or live workers miss dependency fixes.
    dockerfile = pathlib.Path(__file__).resolve().parents[1] / "Dockerfile.worker"
    expected = f"ARG VERL_SPEC=verl[vllm]@{vc.VERL_REQUIREMENT_URL}"
    assert expected in dockerfile.read_text(), (
        "Dockerfile.worker's VERL_SPEC drifted from backend_common.VERL_REQUIREMENT_URL"
    )


def test_the_venv_stamp_records_the_pin_not_the_install_extras(monkeypatch, tmp_path):
    # the stamp gates rebuilds. if it recorded the extra-bearing spec while the pin stayed bare,
    # every later call would see a mismatch and rebuild the venv from scratch on a paid pod.
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))

    vc.resolve_verl_python(str(tmp_path))
    written = (tmp_path / "verl-venv" / "flash-verl-requirement").read_text()
    assert written == vc.VERL_VENV_STAMP
    assert "[vllm]" not in written
    built = len(calls)

    (tmp_path / "verl-venv" / "bin" / "python").write_text("")
    vc.resolve_verl_python(str(tmp_path))
    assert len(calls) == built, "a venv built from the current pin must not be rebuilt"


def test_the_stamp_identifies_flash_attn_so_a_prefix_venv_is_not_reused(monkeypatch, tmp_path):
    """A workdir provisioned before flash-attn was installed must REBUILD, not be reused.

    The reuse stamp must include separately installed flash-attn or retries can skip its
    installation.
    """
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))
    # exactly what the pre-fix release wrote: the bare verl pin, nothing about the wheel.
    _fake_verl_venv(tmp_path, stamp=vc.VERL_REQUIREMENT)

    vc.resolve_verl_python(str(tmp_path))

    assert calls, "a venv stamped before flash-attn was installed must be rebuilt"
    assert any(vc.FLASH_ATTN_INSTALL_SPEC in call for call in calls)


def test_resolve_verl_python_reuses_a_venv_built_from_the_current_pin(monkeypatch, tmp_path):
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))
    _fake_verl_venv(tmp_path, stamp=vc.VERL_VENV_STAMP)

    vc.resolve_verl_python(str(tmp_path))

    # reinstalling verl's torch/vllm on every retry would cost many minutes of paid gpu time.
    assert calls == []


@pytest.mark.parametrize(
    "stale", ["verl @ git+https://github.com/freesolo-co/verl@" + "0" * 40, None]
)
def test_resolve_verl_python_rebuilds_a_venv_that_is_not_the_current_pin(
    monkeypatch, tmp_path, stale
):
    # a retry reuses the pod workdir. a venv from an earlier pin (or a partial install, stamp=None)
    # must be rebuilt rather than silently training on the wrong verl.
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))
    venv = _fake_verl_venv(tmp_path, stamp=stale)
    (venv / "marker").write_text("from the previous attempt")

    vc.resolve_verl_python(str(tmp_path))

    # three installs, then the conv import probe and the libcudart stub repair (both run the venv's
    # python, not uv).
    assert [c[:2] for c in calls] == [
        ["uv", "venv"],
        ["uv", "pip"],
        ["uv", "pip"],
        ["uv", "pip"],
        [str(venv / "bin" / "python"), "-c"],
        [str(venv / "bin" / "python"), "-c"],
    ]
    assert not (venv / "marker").exists()


def test_resolve_verl_python_clears_a_venv_whose_creation_was_interrupted(monkeypatch, tmp_path):
    # a pod killed during `uv venv` leaves a pyvenv.cfg but no bin/python. uv then refuses to write
    # into that directory ("a virtual environment already exists"), so leaving it in place would wedge
    # every later retry. it must be removed before uv runs.
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))
    venv = tmp_path / "verl-venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("half-written")

    vc.resolve_verl_python(str(tmp_path))

    assert not (venv / "pyvenv.cfg").exists()
    assert [c[:2] for c in calls] == [
        ["uv", "venv"],
        ["uv", "pip"],
        ["uv", "pip"],
        ["uv", "pip"],
        [str(venv / "bin" / "python"), "-c"],
        [str(venv / "bin" / "python"), "-c"],
    ]


def test_verl_pin_is_an_immutable_commit_on_the_freesolo_fork():
    # a branch or tag would let the runtime move under a pinned flash release.
    _, _, ref = vc.VERL_REQUIREMENT.partition("git+")
    url, _, commit = ref.rpartition("@")
    assert url == "https://github.com/freesolo-co/verl"
    assert len(commit) == 40
    assert all(c in "0123456789abcdef" for c in commit)


def test_verl_pin_matches_the_version_opd_requires_exactly():
    # the pin MUST stay on the verl 0.8.0 base. opd_plugin patches 0.8.0 internals and imports
    # verl.trainer.main_ppo_sync, which verl deleted after 0.8.0, so a pin built on a newer base
    # installs a verl that fails opd's exact-version gate and cannot import its own entrypoint.
    from flash.engine.worker.train.opd.child import plugin as plugin

    assert plugin._STRUCTURED_RUNTIME_EXACT_VERSIONS["verl"] == "0.8.0"
    # bind the pin to its verified base: f71a02dd remains verl 0.8.0 with main_ppo_sync.py plus the
    # truncation, position-id, fused-label, and qwen3.5 shift-label fixes, the opd dead-compute
    # skip, and the whole-adapter lora bucket sync plus fused-linear grad preservation on top.
    _, _, ref = vc.VERL_REQUIREMENT.partition("git+")
    _, _, commit = ref.rpartition("@")
    assert commit == "f71a02ddb32a9c6a6915f7519bda6dede92e9dd0"


def test_resolve_verl_python_installs_wandb_best_effort_when_requested(monkeypatch, tmp_path):
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls, keep_check=True))

    vc.resolve_verl_python(str(tmp_path), install_wandb=True)

    assert any(vc.VERL_REQUIREMENT_URL in arg for arg in calls[1][0])
    # wandb is the LAST call: it follows the kernel installs, and unlike the required ones before it
    # it is best-effort (check=False) so a wandb outage cannot fail a training run.
    assert calls[-1] == (
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(tmp_path / "verl-venv/bin/python"),
            "wandb==0.28.2",
        ],
        False,
    )


def _wandb_installs(calls, tmp_path):
    """The wandb install commands in a recorded run list, whatever else ran around them."""
    py = str(tmp_path / "verl-venv/bin/python")
    return [
        c for c, _check in calls if c == ["uv", "pip", "install", "--python", py, "wandb==0.28.2"]
    ]


def test_resolve_verl_python_installs_wandb_into_a_venv_it_is_reusing(monkeypatch, tmp_path):
    """A reused venv still gets wandb, because the stamp cannot say whether it has it.

    Earlier no-key runs can stamp a venv without wandb; install it on reuse or later requested W&B
    runs silently fall back to console.
    """
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls, keep_check=True))
    # exactly what an earlier run without WANDB_API_KEY leaves on the pod.
    _fake_verl_venv(tmp_path, stamp=vc.VERL_VENV_STAMP)

    vc.resolve_verl_python(str(tmp_path), install_wandb=True)

    assert _wandb_installs(calls, tmp_path), (
        "a reused venv was never offered wandb, so this run logs to console only"
    )
    # and nothing else ran: reinstalling verl's torch/vllm on every retry would cost many minutes of
    # paid gpu time, which is the reason the reuse path exists at all.
    assert [c for c, _ in calls] == _wandb_installs(calls, tmp_path)


def test_resolve_verl_python_retries_wandb_after_an_install_that_failed(monkeypatch, tmp_path):
    # the install is best-effort (check=False), so a transient failure leaves a stamped venv with no
    # wandb. that must not be permanent for the life of the pod -- the next run has to try again.
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls, keep_check=True))

    vc.resolve_verl_python(str(tmp_path), install_wandb=True)
    assert len(_wandb_installs(calls, tmp_path)) == 1
    # the fake `uv venv` makes bin/ but not the interpreter inside it, and resolve checks for the
    # file. materialize it, or the second call rebuilds and the reuse path is never reached -- the
    # assertion below would then hold against the very gating this test exists to reject.
    (tmp_path / "verl-venv/bin/python").write_text("")
    assert (tmp_path / "verl-venv/flash-verl-requirement").read_text() == vc.VERL_VENV_STAMP
    reused_from = len(calls)

    vc.resolve_verl_python(str(tmp_path), install_wandb=True)
    assert len(_wandb_installs(calls, tmp_path)) == 2, "a failed install was never retried"
    # and the retry cost exactly the wandb install: the second call reused the venv rather than
    # reprovisioning it, which is what makes an unconditional retry affordable.
    assert [c for c, _ in calls[reused_from:]] == _wandb_installs(calls, tmp_path)[1:]


def test_resolve_verl_python_does_not_install_wandb_when_it_was_not_asked(monkeypatch, tmp_path):
    # unconditional means unconditional on the REBUILD, not on the caller's request. a run with no
    # WANDB_API_KEY must not pay an install it cannot use.
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls, keep_check=True))
    _fake_verl_venv(tmp_path, stamp=vc.VERL_VENV_STAMP)

    vc.resolve_verl_python(str(tmp_path))

    assert calls == []


def test_resolve_verl_python_never_installs_wandb_into_a_preset_interpreter(monkeypatch, tmp_path):
    # flash does not own a preset interpreter and must not mutate it -- that is why the preset is
    # returned as-is. moving the wandb install later in the function must not reach past that return.
    calls = []
    monkeypatch.setenv("FLASH_VERL_PYTHON", "/opt/verl/bin/python")
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls, keep_check=True))

    assert vc.resolve_verl_python(str(tmp_path), install_wandb=True) == "/opt/verl/bin/python"
    assert calls == []


def _probe_interpreter(tmp_path, name, body):
    """write a stub interpreter that answers the capability probe like a real python would."""
    stub = tmp_path / name
    stub.write_text("#!/bin/sh\n" + body + "\n")
    stub.chmod(0o755)
    return str(stub)


def _caps_blob(**answers):
    """a shell body printing the capability lines a child with `answers` would emit.

    one line per question, like the real probe: it flushes each answer as it is known so a kill
    partway through cannot retract the ones already sent.
    """
    payload = dict(vc._CAPABILITIES_UNAVAILABLE)
    payload.update(answers)
    lines = "\n".join("FLASH_VERL_CAPS=" + json.dumps({k: v}) for k, v in payload.items())
    return "cat <<'EOF'\n" + lines + "\nEOF"


def test_verl_declares_rollout_field_true_when_field_declared(tmp_path):
    fork = _probe_interpreter(
        tmp_path, "fork-python", _caps_blob(rollout_fields=["mask_truncated_completions"])
    )
    caps = vc.probe_verl_capabilities(fork, "")
    assert vc.verl_declares_rollout_field(caps, "mask_truncated_completions") is True


def test_verl_declares_rollout_field_false_when_field_absent(tmp_path):
    # stock verl 0.8.0: RolloutConfig has no such field, so it is absent from the declared list.
    stock = _probe_interpreter(tmp_path, "stock-python", _caps_blob(rollout_fields=["n"]))
    caps = vc.probe_verl_capabilities(stock, "")
    assert vc.verl_declares_rollout_field(caps, "mask_truncated_completions") is False


def test_verl_declares_rollout_field_false_when_verl_missing(tmp_path):
    # import error inside the probe: an unanswered question must read as unsupported, so the caller
    # raises with the cause instead of composing a key that aborts in dataclass conversion.
    broken = _probe_interpreter(tmp_path, "broken-python", "exit 1")
    caps = vc.probe_verl_capabilities(broken, "")
    assert vc.verl_declares_rollout_field(caps, "mask_truncated_completions") is False


def test_verl_declares_rollout_field_false_when_interpreter_missing(tmp_path):
    # a bogus FLASH_VERL_PYTHON must not raise OSError out of the capability check.
    missing = str(tmp_path / "does-not-exist")
    caps = vc.probe_verl_capabilities(missing, "")
    assert vc.verl_declares_rollout_field(caps, "mask_truncated_completions") is False


def test_resolve_verl_python_returns_preset_unmodified(monkeypatch, tmp_path):
    # flash does not own a preset interpreter and must never mutate it; capability is checked
    # separately by the batched capability probe.
    calls = []
    monkeypatch.setenv("FLASH_VERL_PYTHON", "/opt/verl/bin/python")
    monkeypatch.setattr(vc.subprocess, "run", lambda *a, **k: calls.append(a))

    assert vc.resolve_verl_python(str(tmp_path)) == "/opt/verl/bin/python"
    assert calls == []


def test_run_verl_training_streams_steps_and_returns_code():
    seen: list[int] = []
    lines: list[str] = []
    beats: list[int] = []
    code = vc.run_verl_training(
        ["bash", "-c", "echo 'foo step: 1 bar'; echo 'step: 2'; echo done"],
        env=dict(os.environ),
        on_step=seen.append,
        on_line=lines.append,
        heartbeat=lambda: beats.append(1),
        heartbeat_interval_s=0.0,
    )
    assert code == 0
    assert seen == [1, 2]
    assert lines[-1] == "done\n"
    # heartbeat_interval_s=0 => fires on every scanned line (3 lines here).
    assert len(beats) >= 1


def test_run_verl_training_propagates_nonzero_exit():
    code = vc.run_verl_training(
        ["bash", "-c", "echo 'step: 1'; exit 7"],
        env=dict(os.environ),
        on_step=lambda _s: None,
    )
    assert code == 7


def test_run_verl_training_terminates_child_when_callback_fails():
    def fail(_line):
        raise RuntimeError("checkpoint upload failed")

    with pytest.raises(RuntimeError, match="checkpoint upload failed"):
        vc.run_verl_training(
            ["bash", "-c", "echo ready; sleep 30"],
            env=dict(os.environ),
            on_line=fail,
        )


def test_run_verl_training_kills_the_grandchild_not_just_the_direct_child(tmp_path):
    # verl spawns vllm's EngineCore as a grandchild. killing only the direct child reparents it to
    # init with its cuda context intact, stranding the gpu for every later run. the child here traps
    # SIGTERM and exits without touching its own child, so only a process-group signal reaches the
    # grandchild -- exactly the shape of the real leak.
    marker = tmp_path / "grandchild.pid"
    # "ready" is withheld until the grandchild has recorded its pid, so teardown -- which on_line
    # triggers -- can never race ahead of the marker write.
    script = (
        "trap 'exit 0' TERM; "
        f"bash -c 'echo $$ > {marker}; sleep 30' & "
        f"for _ in $(seq 1 500); do [ -s {marker} ] && break; sleep 0.02; done; "
        "echo ready; "
        "sleep 30"
    )

    def fail(_line):
        raise RuntimeError("stream died")

    with pytest.raises(RuntimeError, match="stream died"):
        vc.run_verl_training(["bash", "-c", script], env=dict(os.environ), on_line=fail)

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.05)
    assert marker.exists(), "grandchild never recorded its pid"
    grandchild = int(marker.read_text().strip())

    while time.monotonic() < deadline:
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    pytest.fail(f"grandchild {grandchild} survived teardown and is still holding the gpu")


def test_child_output_tail_retains_bounded_recent_lines():
    tail = vc.ChildOutputTail(limit=3)
    for i in range(6):
        tail.record(f"line{i}\n")
    # bounded: the oldest are evicted, the most recent survive in order.
    assert tail.tail() == ["line3", "line4", "line5"]
    assert tail.tail(limit=2) == ["line4", "line5"]
    # a limit wider than the buffer is not an error and does not pad.
    assert tail.tail(limit=99) == ["line3", "line4", "line5"]


def test_child_output_tail_drops_blank_lines_and_caps_line_width():
    tail = vc.ChildOutputTail()
    tail.record("\n")
    tail.record("   \n")  # whitespace-only survives rstrip("\n") and is intentionally kept
    tail.record("x" * 5000 + "\n")
    kept = tail.tail()
    assert "" not in kept, "a bare newline must not occupy a slot in the window"
    widest = max(len(line) for line in kept)
    assert widest <= vc._CHILD_TAIL_LINE_CHARS, (
        f"retained a {widest}-char line; verl prints multi-KB config blocks and the tail "
        "has to ride inside an uploaded heartbeat payload"
    )


def test_run_verl_training_records_child_output_into_the_tail():
    """the child's words must reach the tail, because the parent's stdout reaches no log stream."""
    tail = vc.ChildOutputTail()
    code = vc.run_verl_training(
        ["bash", "-c", "echo 'ray placement group pending'; echo 'step: 1'; echo 'wedged here'"],
        env=dict(os.environ),
        tail=tail,
    )
    assert code == 0
    captured = tail.tail()
    # this is the diagnostic that was unretrievable in production: the child's last line before it
    # stopped producing output (ISSUES VERL-061).
    assert captured[-1] == "wedged here"
    assert "ray placement group pending" in captured


def test_run_verl_training_without_a_tail_is_unchanged():
    """tail is opt-in; omitting it must not alter streaming or the exit code."""
    lines: list[str] = []
    code = vc.run_verl_training(
        ["bash", "-c", "echo 'step: 4'; exit 3"],
        env=dict(os.environ),
        on_line=lines.append,
    )
    assert code == 3
    assert lines == ["step: 4\n"]


@pytest.mark.parametrize(
    "signature",
    [
        "cudaErrorDevicesUnavailable",
        "CUDA-capable device(s) is/are busy or unavailable",
    ],
)
def test_run_verl_training_retries_nested_cuda_device_unavailable_after_tail_eviction(
    signature,
):
    tail = vc.ChildOutputTail(limit=3)
    command = [
        "bash",
        "-c",
        "printf '%s\\n' \"$TEST_SIGNATURE\"; for i in $(seq 1 70); do echo filler-$i; done; exit 1",
    ]
    env = {**os.environ, "TEST_SIGNATURE": signature}

    with pytest.raises(RetriableInfraError) as exc_info:
        vc.run_verl_training(command, env=env, tail=tail)

    assert signature in str(exc_info.value)
    assert signature not in tail.tail()


def test_child_output_tail_latches_signature_before_line_truncation():
    signature = "cudaErrorDevicesUnavailable"
    tail = vc.ChildOutputTail()
    tail.record("x" * vc._CHILD_TAIL_LINE_CHARS + signature + "\n")

    assert signature not in tail.tail()[0]
    assert tail.retriable_infra_signature == signature


def test_run_verl_training_does_not_reclassify_a_zero_exit():
    code = vc.run_verl_training(
        ["bash", "-c", "echo 'cudaErrorDevicesUnavailable'; exit 0"],
        env=dict(os.environ),
    )

    assert code == 0


def test_run_verl_training_leaves_an_unclassified_nonzero_exit_terminal():
    code = vc.run_verl_training(
        ["bash", "-c", "printf '%s\\n' 'unrelated child failure' >&2; exit 9"],
        env=dict(os.environ),
    )

    assert code == 9


def test_run_verl_training_classifies_a_child_cuda_oom_rather_than_returning_it():
    """A child's torch OOM must raise classified, not return a bare status.

    Training OOM signatures differ from vLLM startup OOMs. Raise a classified RuntimeError so the
    lifecycle escalates GPU size; it is not a `RetriableInfraError`.
    """
    with pytest.raises(RuntimeError) as exc_info:
        vc.run_verl_training(
            ["bash", "-c", "printf '%s\\n' 'CUDA out of memory' >&2; exit 9"],
            env=dict(os.environ),
        )

    assert not isinstance(exc_info.value, RetriableInfraError)
    assert "cuda out of memory" in str(exc_info.value)


@pytest.mark.parametrize("oom_first", [False, True])
def test_run_verl_training_preserves_oom_over_device_unavailable_after_eviction(
    monkeypatch, oom_first
):
    from flash.engine.worker.entry.worker import _worker_failure_flags
    from flash.engine.worker.perf import lifecycle

    monkeypatch.setattr(lifecycle, "cuda_oom_count", lambda: 0)
    device_line = "cudaErrorDevicesUnavailable"
    oom_line = "x" * vc._CHILD_TAIL_LINE_CHARS + "No available memory for the cache blocks"
    first, second = (oom_line, device_line) if oom_first else (device_line, oom_line)
    command = [
        "bash",
        "-c",
        (
            'printf \'%s\\n\' "$FIRST" "$SECOND"; '
            "for i in $(seq 1 70); do echo filler-$i; done; "
            "exit 1"
        ),
    ]
    tail = vc.ChildOutputTail(limit=3)

    with pytest.raises(RuntimeError) as exc_info:
        vc.run_verl_training(
            command,
            env={**os.environ, "FIRST": first, "SECOND": second},
            tail=tail,
        )

    assert not isinstance(exc_info.value, RetriableInfraError)
    assert _worker_failure_flags(exc_info.value) == {"retriable": False, "oom": True}
    assert tail.tail() == ["filler-68", "filler-69", "filler-70"]


def test_stall_tail_fields_reports_only_before_the_first_step():
    tail = vc.ChildOutputTail()
    tail.record("ray: placement group pending\n")

    # pre-first-step: this is the blind window, so the child's words must be carried out.
    fields = vc.stall_tail_fields(0, tail)
    assert fields == {"child_tail": ["ray: placement group pending"]}

    # once training progresses the step/loss stream is the diagnostic; the tail would be pure
    # payload bloat on every tick.
    assert vc.stall_tail_fields(1, tail) == {}
    assert vc.stall_tail_fields(500, tail) == {}


def test_stall_tail_fields_is_empty_when_the_child_has_said_nothing():
    # an empty key would claim the child spoke and said nothing, which is a different fact from
    # "the child has produced no output at all".
    assert vc.stall_tail_fields(0, vc.ChildOutputTail()) == {}


def test_stall_tail_fields_reports_how_long_the_child_has_been_silent():
    # the whole point: a child still loading shards and a child wedged forever both present a fully
    # populated tail whose newest line looks plausible. only whether the tail CHANGED separates them,
    # and without this the comparison has to be reconstructed by hand from consecutive heartbeats.
    tail = vc.ChildOutputTail()
    staleness = vc.ChildTailStaleness()
    tail.record("Started a local Ray instance\n")

    first = vc.stall_tail_fields(0, tail, staleness=staleness)
    assert first["child_tail_silent_ticks"] == 0

    # two more ticks with the child saying nothing new: same tail, rising silence.
    assert vc.stall_tail_fields(0, tail, staleness=staleness)["child_tail_silent_ticks"] == 1
    assert vc.stall_tail_fields(0, tail, staleness=staleness)["child_tail_silent_ticks"] == 2

    # the child speaks again, so it is slow rather than stuck and the counter resets.
    tail.record("loading checkpoint shards 1/4\n")
    assert vc.stall_tail_fields(0, tail, staleness=staleness)["child_tail_silent_ticks"] == 0


def test_child_tail_silence_survives_the_retention_limit():
    # staleness is counted from lines WRITTEN, not from the retained window: once the ring buffer is
    # full its contents can keep changing while its length does not, and a length-based comparison
    # would then report a talking child as silent.
    tail = vc.ChildOutputTail(limit=3)
    staleness = vc.ChildTailStaleness()
    for i in range(3):
        tail.record(f"line{i}\n")
    assert vc.stall_tail_fields(0, tail, staleness=staleness)["child_tail_silent_ticks"] == 0
    tail.record("line3\n")  # evicts line0; the deque stays length 3
    assert len(tail.tail()) == 3
    assert vc.stall_tail_fields(0, tail, staleness=staleness)["child_tail_silent_ticks"] == 0


def test_child_tail_silence_is_measured_from_the_childs_first_line():
    # a child silent for the first ticks then talking must not be credited with the silence that
    # preceded its first line -- otherwise a slow starter reports as long-wedged the moment it speaks.
    tail = vc.ChildOutputTail()
    staleness = vc.ChildTailStaleness()
    for _ in range(4):
        assert vc.stall_tail_fields(0, tail, staleness=staleness) == {}
    tail.record("first words\n")
    assert vc.stall_tail_fields(0, tail, staleness=staleness)["child_tail_silent_ticks"] == 0


def test_stall_tail_fields_omits_silence_when_no_tracker_is_supplied():
    # the field must not appear as a fabricated 0 for callers that do not track staleness: absent and
    # "zero ticks silent" are different claims.
    tail = vc.ChildOutputTail()
    tail.record("only line\n")
    assert vc.stall_tail_fields(0, tail) == {"child_tail": ["only line"]}


def test_stall_tail_fields_narrows_to_the_most_recent_lines():
    tail = vc.ChildOutputTail()
    for i in range(vc.STALL_TAIL_LINES + 25):
        tail.record(f"line{i}\n")
    carried = vc.stall_tail_fields(0, tail)["child_tail"]
    assert len(carried) == vc.STALL_TAIL_LINES
    # the most recent lines are the ones that matter: they are what the child said last.
    assert carried[-1] == f"line{vc.STALL_TAIL_LINES + 24}"


def test_child_tail_redacts_credentials_before_they_reach_a_heartbeat(monkeypatch):
    """the retained tail rides to heartbeat.json, the streamed run log and persisted status.

    the worker is the only side that knows the run's secret values, so redaction has to happen
    here; the plane-side formatter only neutralizes control characters.
    """
    secret = "hf_ZZZchildtailsecretvalue0123456789"
    monkeypatch.setenv("HF_TOKEN", secret)
    tail = vc.ChildOutputTail()
    tail.record(f"requests.HTTPError: 401 Unauthorized for hf.co (token {secret})\n")

    carried = vc.stall_tail_fields(0, tail)["child_tail"]

    assert secret not in carried[0]
    assert "<redacted>" in carried[0]
    # the diagnostic itself is what a setup stall is read from; only the credential goes.
    assert carried[0] == "requests.HTTPError: 401 Unauthorized for hf.co (token <redacted>)"


def test_child_tail_redaction_precedes_the_per_line_cap(monkeypatch):
    """sanitizing after truncation would split a credential across the cut and leak the prefix."""
    secret = "hf_ZZZstraddlesthelinecap0123456789"
    monkeypatch.setenv("HF_TOKEN", secret)
    tail = vc.ChildOutputTail()
    # the token starts inside the retained window and ends past it.
    tail.record("x" * (vc._CHILD_TAIL_LINE_CHARS - 10) + secret + "\n")

    kept = tail.tail()[0]

    assert secret[:10] not in kept
    assert len(kept) <= vc._CHILD_TAIL_LINE_CHARS


def test_child_tail_redacts_declared_secrets_with_arbitrary_names(monkeypatch):
    """[environment] secrets accepts any env name; FLASH_SECRET_ENV_KEYS is what lets the worker
    redact values whose names the suffix heuristic misses."""
    from flash._internal.diagnostics import SECRET_ENV_KEYS_ENV

    secret = "aws-declared-childtail-0123456789abcdef"
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret)
    monkeypatch.setenv(SECRET_ENV_KEYS_ENV, "AWS_SECRET_ACCESS_KEY")
    tail = vc.ChildOutputTail()
    tail.record(f"botocore.exceptions.ClientError: SignatureDoesNotMatch using {secret}\n")

    kept = tail.tail()[0]

    assert secret not in kept
    assert kept == "botocore.exceptions.ClientError: SignatureDoesNotMatch using <redacted>"


def test_sanitize_diagnostic_redacts_a_very_short_declared_secret_at_word_boundaries(monkeypatch):
    """a declared secret can carry any value, including a 3-char one, and it must not leak.

    the length floor used to drop such a value from the needle set entirely, so it printed
    verbatim. it cannot become an unconstrained global needle either -- the value `ati` would
    rewrite `authentication` -- so it is redacted only where it stands alone.
    """
    from flash._internal.diagnostics import SECRET_ENV_KEYS_ENV, sanitize_diagnostic

    monkeypatch.setenv(SECRET_ENV_KEYS_ENV, "PIN")
    monkeypatch.setenv("PIN", "ati")
    # the standalone value is the leak the floor used to allow.
    assert sanitize_diagnostic("worker rejected pin ati") == "worker rejected pin <redacted>"
    # ... while the same letters inside a word stay readable.
    assert sanitize_diagnostic("trainer crashed after validation") == (
        "trainer crashed after validation"
    )

    monkeypatch.setenv("PIN", "sk-live-abc123456")
    assert sanitize_diagnostic("trainer crashed holding sk-live-abc123456") == (
        "trainer crashed holding <redacted>"
    )


def test_sanitize_diagnostic_preserves_punctuation_for_wordless_short_secret(monkeypatch):
    from flash._internal.diagnostics import SECRET_ENV_KEYS_ENV, sanitize_diagnostic

    monkeypatch.setenv(SECRET_ENV_KEYS_ENV, "PIN")
    monkeypatch.setenv("PIN", ".")

    assert sanitize_diagnostic("module.py: failed at /tmp/a.py") == "module.py: failed at /tmp/a.py"


def test_sanitize_diagnostic_redacts_wordless_short_secret_in_keyed_syntax(monkeypatch):
    from flash._internal.diagnostics import SECRET_ENV_KEYS_ENV, sanitize_diagnostic

    monkeypatch.setenv(SECRET_ENV_KEYS_ENV, "PIN")
    monkeypatch.setenv("PIN", ".")

    assert sanitize_diagnostic("token=.") == "token=<redacted>"


def test_sanitize_diagnostic_redacts_declared_wordless_values_by_exact_shape(monkeypatch):
    from flash._internal.diagnostics import SECRET_ENV_KEYS_ENV, sanitize_diagnostic

    monkeypatch.setenv(SECRET_ENV_KEYS_ENV, "KEYED_PIN,BEARER_PIN")
    monkeypatch.setenv("KEYED_PIN", ";")
    monkeypatch.setenv("BEARER_PIN", "!")

    assert sanitize_diagnostic("token=;") == "token=<redacted>"
    assert sanitize_diagnostic("Bearer !") == "Bearer <redacted>"


def test_sanitize_diagnostic_protects_shape_before_overlapping_values(monkeypatch):
    from flash._internal.diagnostics import SECRET_ENV_KEYS_ENV, sanitize_diagnostic

    monkeypatch.setenv(SECRET_ENV_KEYS_ENV, "KEY,PIN")
    monkeypatch.setenv("KEY", "token")
    monkeypatch.setenv("PIN", ";")
    detail = sanitize_diagnostic("token=;")
    assert detail == "<redacted>"
    assert ";" not in detail

    monkeypatch.setenv("KEY", "Bearer")
    monkeypatch.setenv("PIN", "!")
    detail = sanitize_diagnostic("Bearer !")
    assert detail == "<redacted>"
    assert "!" not in detail


def test_sanitize_diagnostic_redacts_percent_octets_without_folding_literal_case(monkeypatch):
    from flash._internal.diagnostics import SECRET_ENV_KEYS_ENV, sanitize_diagnostic

    monkeypatch.setenv(SECRET_ENV_KEYS_ENV, "PIN")
    for secret, encoded in ((".", "%2E"), ("-", "%2D"), ("~", "%7E"), ("/", "%2f")):
        monkeypatch.setenv("PIN", secret)
        assert sanitize_diagnostic(f"encoded {encoded}") == "encoded <redacted>"

    monkeypatch.setenv("PIN", "A/B")
    assert sanitize_diagnostic("encoded A%2fB") == "encoded <redacted>"
    assert sanitize_diagnostic("encoded a%2fb") == "encoded a%2fb"

    for secret, case_variant in (
        ("A%2FB", "A%2fB"),
        ("literal%2Fsecret", "literal%2fsecret"),
    ):
        monkeypatch.setenv("PIN", secret)
        assert sanitize_diagnostic(f"literal {secret}") == "literal <redacted>"
        assert sanitize_diagnostic(f"literal {case_variant}") == f"literal {case_variant}"


def test_sanitize_diagnostic_redacts_cross_group_encoded_overlap(monkeypatch):
    from flash._internal.diagnostics import SECRET_ENV_KEYS_ENV, sanitize_diagnostic

    monkeypatch.setenv(SECRET_ENV_KEYS_ENV, "LONG_TOKEN,SHORT_TOKEN")
    monkeypatch.setenv("LONG_TOKEN", "a%2Fb%2B")
    monkeypatch.setenv("SHORT_TOKEN", "a/b+c&d")

    detail = sanitize_diagnostic("fetch failed for a%2Fb%2Bc%26d")

    assert detail == "fetch failed for <redacted>"
    for secret in ("a%2Fb%2B", "c%26d", "a/b+c&d"):
        assert secret not in detail


# ---------------------- child teardown escalates to SIGKILL ----------------------
# these real-process tests require fork, /proc group membership, and libc subreaper support.
# guard on those capabilities rather than platform names.
def _has_libc() -> bool:
    """whether `libc.so.6` loads here, for the subreaper probe the adoption tests need."""
    try:
        ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:  # pragma: no cover - not reachable on linux
        return False
    return True


_needs_process_teardown = pytest.mark.skipif(
    not hasattr(os, "fork") or not os.path.isdir("/proc") or not _has_libc(),
    reason="teardown tests drive real process groups: needs os.fork, /proc and libc",
)


# PR_SET_CHILD_SUBREAPER / PR_GET_CHILD_SUBREAPER, from linux/prctl.h.
_PR_SET_CHILD_SUBREAPER = 36
_PR_GET_CHILD_SUBREAPER = 37


def _child_subreaper_setting() -> int:
    """This process's current subreaper flag, skipping the test where prctl cannot answer."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    current = ctypes.c_int(0)
    if libc.prctl(_PR_GET_CHILD_SUBREAPER, ctypes.byref(current), 0, 0, 0) != 0:
        pytest.skip("PR_GET_CHILD_SUBREAPER unavailable")  # pragma: no cover - linux 3.4+
    return current.value


@contextlib.contextmanager
def _child_subreaper_enabled():
    """Adopt orphaned grandchildren for the duration, then restore the previous setting.

    Dockerfile.worker gets adoption from pid 1; pytest must opt in without changing subprocess
    semantics or accumulating zombies for later tests.
    """
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    previous = _child_subreaper_setting()
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        pytest.skip("PR_SET_CHILD_SUBREAPER unavailable")  # pragma: no cover - linux 3.4+
    try:
        yield
    finally:
        libc.prctl(_PR_SET_CHILD_SUBREAPER, previous, 0, 0, 0)


@pytest.fixture
def subreaper():
    """`_child_subreaper_enabled` as a fixture, for the tests that need adopted grandchildren."""
    with _child_subreaper_enabled():
        yield


@pytest.fixture
def quick_teardown_grace(monkeypatch):
    """shorten the escalation grace so a test that deliberately ignores SIGTERM stays fast.

    These tests assert SIGKILL and complete reaping, neither of which depends on the production
    30-second grace.
    """
    monkeypatch.setattr(vc, "_TEARDOWN_GRACE_S", 0.5)
    return 0.5


@_needs_process_teardown
def test_kill_process_group_escalates_to_sigkill_when_sigterm_is_ignored(quick_teardown_grace):
    # the whole point of this helper is the escalation, so the child has to actually ignore SIGTERM.
    # a child that dies on the term would pass just as well against a bare killpg, which is the bug.
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import signal, sys, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "print('ready', flush=True)\n"
                "time.sleep(300)\n"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        # wait until the handler is installed, otherwise the term could land before SIG_IGN and the
        # child would die to the term -- passing without ever exercising the escalation.
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        started = time.monotonic()
        vc.kill_process_group(child)
        assert child.poll() is not None, "child ignoring SIGTERM survived kill_process_group"
        # negative-signal returncode identifies which signal actually reaped it.
        assert child.returncode == -signal.SIGKILL
        assert time.monotonic() - started < 60
    finally:
        if child.poll() is None:  # pragma: no cover - only on an unexpected failure
            child.kill()
            child.wait(timeout=10)
        if child.stdout is not None:
            child.stdout.close()


@_needs_process_teardown
def test_kill_process_group_reaps_a_grandchild_that_outlives_the_leader(quick_teardown_grace):
    # the real shape of this failure: the verl trainer dies on the term but its vllm EngineCore
    # grandchild ignores it. escalating off proc.wait() alone returns the instant the leader is
    # reaped, leaving the grandchild holding the cuda context -- so drive the escalation off the
    # group, and prove it by having the leader exit cleanly on SIGTERM.
    leader_src = (
        "import subprocess, sys, time\n"
        "g = subprocess.Popen([sys.executable, '-c',\n"
        '    "import signal,time\\n"\n'
        '    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"\n'
        "    \"print('gready', flush=True)\\n\"\n"
        '    "time.sleep(300)\\n"], stdout=subprocess.PIPE, text=True)\n'
        "assert g.stdout.readline().strip() == 'gready'\n"
        "print(g.pid, flush=True)\n"
        "time.sleep(300)\n"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", leader_src],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    grandchild_pid = None
    try:
        assert leader.stdout is not None
        grandchild_pid = int(leader.stdout.readline().strip())
        started = time.monotonic()
        vc.kill_process_group(leader)
        assert leader.poll() is not None
        # the leader dying is not the property under test -- the grandchild being gone is. poll
        # rather than assert instantly: sigkill delivery and reaping are asynchronous, so a bare
        # check here races the kernel and flakes.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                os.kill(grandchild_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:  # pragma: no cover - only when the escalation genuinely fails
            pytest.fail("grandchild ignoring SIGTERM survived kill_process_group")
        assert time.monotonic() - started < 60
    finally:
        if grandchild_pid is not None:  # pragma: no cover - only on an unexpected failure
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(grandchild_pid, signal.SIGKILL)
        if leader.poll() is None:  # pragma: no cover - only on an unexpected failure
            leader.kill()
            leader.wait(timeout=10)
        if leader.stdout is not None:
            leader.stdout.close()


def _classified_exit_command(marker: pathlib.Path) -> list[str]:
    grandchild = (
        "import signal,time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('ready', flush=True)\n"
        "time.sleep(300)\n"
    )
    leader = (
        "import pathlib,subprocess,sys\n"
        f"g = subprocess.Popen([sys.executable, '-c', {grandchild!r}], "
        "stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)\n"
        "assert g.stdout.readline().strip() == 'ready'\n"
        "g.stdout.close()\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(g.pid))\n"
        "print('cudaErrorDevicesUnavailable', flush=True)\n"
        "raise SystemExit(1)\n"
    )
    return [sys.executable, "-c", leader]


def _unclassified_exit_command(marker: pathlib.Path) -> list[str]:
    """Same shape as `_classified_exit_command`, but the leader says nothing recognizable.

    The classifier returns rather than raises, proving teardown cannot depend only on its exception.
    """
    grandchild = (
        "import signal,time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('ready', flush=True)\n"
        "time.sleep(300)\n"
    )
    leader = (
        "import pathlib,subprocess,sys\n"
        f"g = subprocess.Popen([sys.executable, '-c', {grandchild!r}], "
        "stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)\n"
        "assert g.stdout.readline().strip() == 'ready'\n"
        "g.stdout.close()\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(g.pid))\n"
        "print('unrelated trainer failure', flush=True)\n"
        "raise SystemExit(9)\n"
    )
    return [sys.executable, "-c", leader]


def _orphaned_pipe_command(marker: pathlib.Path, *, leader_status: int) -> list[str]:
    """The reviewers' lead case: the leader EXITS while a grandchild holds the merged pipe open.

    The read loop sees neither data nor EOF, so only independent child-exit monitoring can end it.
    """
    grandchild = (
        "import pathlib,signal,sys,time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        # inherits the leader's stdout and holds it. never writes, never closes: the parent's loop
        # has nothing to read and no EOF to end on.
        f"pathlib.Path({str(marker)!r}).write_text(str(__import__('os').getpid()))\n"
        "sys.stderr.close()\n"
        "time.sleep(300)\n"
    )
    leader = (
        "import pathlib,subprocess,sys,time\n"
        f"g = subprocess.Popen([sys.executable, '-c', {grandchild!r}])\n"
        f"m = pathlib.Path({str(marker)!r})\n"
        "for _ in range(500):\n"
        "    if m.exists() and m.read_text().strip():\n"
        "        break\n"
        "    time.sleep(0.02)\n"
        f"raise SystemExit({leader_status})\n"
    )
    return [sys.executable, "-c", leader]


def _outlives_its_stdout_command() -> list[str]:
    """A leader that closes its own stdout and then refuses to die.

    stdout EOF is not process exit; the bounded wait must escalate to SIGKILL.
    """
    leader = (
        "import os,signal,sys,time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('step: 1', flush=True)\n"
        # BOTH descriptors. the parent merges the child's stderr into stdout, so fd 1 and fd 2 are
        # the same pipe: closing only fd 1 leaves the write end open and the parent never sees EOF,
        # which makes this the wrong scenario (a live child holding its own pipe) rather than the
        # one the test names.
        "os.close(1)\n"
        "os.close(2)\n"
        "time.sleep(300)\n"
    )
    return [sys.executable, "-c", leader]


def _assert_process_reaped(pid: int) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if not os.path.exists(f"/proc/{pid}"):
            return
        time.sleep(0.05)
    pytest.fail(f"classified exit left grandchild {pid} alive or unreaped")


@_needs_process_teardown
def test_classified_run_verl_exit_drains_group_after_leader_is_reaped(
    tmp_path, quick_teardown_grace
):
    marker = tmp_path / "classified-grandchild.pid"
    grandchild_pid = None
    try:
        with pytest.raises(RetriableInfraError, match="cudaErrorDevicesUnavailable"):
            vc.run_verl_training(_classified_exit_command(marker), env=dict(os.environ))

        assert marker.exists(), "leader exited before recording its surviving grandchild"
        grandchild_pid = int(marker.read_text())
        _assert_process_reaped(grandchild_pid)
    finally:
        if grandchild_pid is not None:  # pragma: no cover - only on an unexpected failure
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(grandchild_pid, signal.SIGKILL)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(grandchild_pid, 0)


@_needs_process_teardown
def test_unclassified_run_verl_exit_drains_the_group_it_leaves_behind(
    tmp_path, quick_teardown_grace
):
    """The failing path that raises nothing must still tear the group down.

    An unrecognized nonzero exit returns from classification, so teardown must not live only in the
    classifier exception path.
    """
    marker = tmp_path / "unclassified-grandchild.pid"
    grandchild_pid = None
    try:
        code = vc.run_verl_training(_unclassified_exit_command(marker), env=dict(os.environ))

        # returned, not raised: the status stays terminal and the caller still sees 9.
        assert code == 9
        assert marker.exists(), "leader exited before recording its surviving grandchild"
        grandchild_pid = int(marker.read_text())
        _assert_process_reaped(grandchild_pid)
    finally:
        if grandchild_pid is not None:  # pragma: no cover - only on an unexpected failure
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(grandchild_pid, signal.SIGKILL)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(grandchild_pid, 0)


@_needs_process_teardown
def test_run_verl_training_bounds_its_wait_on_a_child_that_outlives_its_stdout(
    quick_teardown_grace,
):
    """stdout EOF is not the child's exit, so the wait after the read loop has to be bounded.

    A child can close stdout and keep running; an unbounded wait would hold a paid GPU forever.
    """
    started = time.monotonic()
    code = vc.run_verl_training(_outlives_its_stdout_command(), env=dict(os.environ))
    elapsed = time.monotonic() - started

    # the child sleeps 300s. any bound at all beats that; assert on a budget derived from the
    # patched grace so this stays meaningful if the grace is retuned, rather than on a bare literal.
    budget = quick_teardown_grace * 20 + 10
    assert elapsed < budget, (
        f"waited {elapsed:.1f}s on a child that outlived its stdout; an unbounded wait parks the "
        "attempt on a paid gpu until the job's own wall-clock limit"
    )
    # the group was torn down, so the child died on a signal and its status says so. it must NOT
    # read as success: the trainer never finished, and a zero here would publish whatever partial
    # artifacts exist as a completed run.
    assert code != 0, "a child killed at teardown was reported as a successful run"
    assert code < 0, f"expected a signal-terminated status, got exit code {code}"


@_needs_process_teardown
def test_run_verl_training_ends_when_a_grandchild_holds_the_pipe_after_the_child_exits(
    tmp_path, monkeypatch, quick_teardown_grace
):
    """The reviewers' lead case: the read loop never ends, so no teardown below it is reached.

    The direct child exits while an EngineCore grandchild keeps the pipe open. Bound the test and
    require the child-exit watchdog to stop the paid-attempt stall.
    """
    monkeypatch.setattr(vc, "_ORPHANED_PIPE_GRACE_S", 1.0)
    marker = tmp_path / "pipe-holder.pid"
    grandchild_pid = None
    started = time.monotonic()
    try:
        # exit 0: the trainer "succeeded", so nothing about its own status hints at the survivor.
        # returning 0 here would publish a partial run as a completed one.
        with pytest.raises(RuntimeError, match="held its output pipe open"):
            vc.run_verl_training(
                _orphaned_pipe_command(marker, leader_status=0), env=dict(os.environ)
            )
        elapsed = time.monotonic() - started

        assert elapsed < 60, (
            f"blocked {elapsed:.1f}s reading a pipe whose writer had already exited; in production "
            "this runs until the provider's wall-clock limit on a paid gpu"
        )
        assert marker.exists(), "grandchild never recorded its pid"
        grandchild_pid = int(marker.read_text())
        _assert_process_reaped(grandchild_pid)
    finally:
        if grandchild_pid is not None:  # pragma: no cover - only on an unexpected failure
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(grandchild_pid, signal.SIGKILL)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(grandchild_pid, 0)


@_needs_process_teardown
def test_child_exit_watchdog_does_not_kill_a_reader_still_draining_a_backlog(monkeypatch):
    """The child's exit alone must not arm the teardown -- the reader's progress also counts.

    A full pipe may still be draining through slow checkpoint callbacks; killing that reader would
    turn a successful run into a failure.
    """
    # a grace this short fires between consecutive callbacks, so only progress-awareness saves it.
    monkeypatch.setattr(vc, "_ORPHANED_PIPE_GRACE_S", 0.2)
    # the child writes its whole backlog and exits immediately, so every line below is read AFTER
    # the direct child is already gone -- exactly the state that arms the watchdog.
    script = "import sys\nfor i in range(8): print(f'step: {i}', flush=True)\nsys.exit(0)"
    seen = []

    def slow_step(step: int) -> None:
        # stands in for a checkpoint upload on the step boundary.
        seen.append(step)
        time.sleep(0.15)

    code = vc.run_verl_training(
        [sys.executable, "-c", script], env=dict(os.environ), on_step=slow_step
    )

    assert code == 0, "a run whose reader was still draining a backlog was reported as failed"
    assert seen == list(range(8)), (
        f"only {len(seen)} of 8 steps were processed; the reader was torn down while it was still "
        "making progress through the child's remaining output"
    )


@_needs_process_teardown
def test_child_exit_watchdog_does_not_kill_a_reader_inside_one_long_callback(monkeypatch):
    """Progress is counted around the callback, not before it.

    One checkpoint callback can run for minutes. Counting only entry makes it look stuck and tears
    down a successful upload; completion must advance progress too.
    """
    monkeypatch.setattr(vc, "_ORPHANED_PIPE_GRACE_S", 0.2)
    # the child exits immediately, so the callback below runs with the watchdog already armed.
    script = "import sys\nprint('step: 1', flush=True)\nsys.exit(0)"
    finished = []

    def one_slow_step(step: int) -> None:
        # a single callback several graces long: the case the backlog test cannot reach.
        time.sleep(1.2)
        finished.append(step)

    code = vc.run_verl_training(
        [sys.executable, "-c", script], env=dict(os.environ), on_step=one_slow_step
    )

    assert finished == [1], "the reader was killed inside a single long callback"
    assert code == 0, (
        "a successful run was reported as failed because one callback outlasted the grace; the "
        "reader was working the whole time"
    )


@_needs_process_teardown
def test_child_exit_watchdog_leaves_a_healthy_quiet_child_alone(quick_teardown_grace):
    """The watchdog arms on the child's EXIT, never on silence.

    Quiet shard loading or generation is valid; silence belongs to `ChildTailStaleness`.
    """
    # a grace of 0 would fire the instant the child exits, so any teardown observed here is
    # attributable to quiet alone.
    script = "import time,sys; time.sleep(2); print('step: 1', flush=True); sys.exit(0)"
    code = vc.run_verl_training(
        [sys.executable, "-c", script],
        env=dict(os.environ),
        heartbeat_interval_s=0.1,
    )

    assert code == 0, "a healthy child that was merely quiet was torn down as if it had leaked"


@_needs_process_teardown
def test_a_group_whose_only_member_is_a_zombie_is_not_read_as_alive():
    # pid 1 in Dockerfile.worker does not reap orphaned EngineCore zombies. `killpg(..., 0)` still
    # sees them, so teardown must distinguish zombies; fork because Popen would reap automatically.
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child never returns to pytest
        os.setsid()
        time.sleep(300)
        os._exit(0)
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and os.getpgid(pid) != pid:
            time.sleep(0.02)
        pgid = os.getpgid(pid)
        assert vc._process_group_alive(pgid), "a running group must read as alive"

        os.kill(pid, signal.SIGKILL)
        # deliberately unreaped: waitpid here would clear the zombie and destroy the case.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not vc._process_is_zombie(pid):
            time.sleep(0.02)
        assert vc._process_is_zombie(pid), "child should be an unreaped zombie"
        assert os.killpg(pgid, 0) is None, "the zombie group is still addressable"

        assert not vc._process_group_alive(pgid), (
            "a zombie-only group must not read as alive, or teardown waits out the full deadline"
        )
    finally:
        with contextlib.suppress(ProcessLookupError, ChildProcessError):
            os.kill(pid, signal.SIGKILL)
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pid, 0)


@_needs_process_teardown
def test_teardown_returns_promptly_when_the_survivor_is_only_a_zombie():
    # the cost of the bug, measured. with a zombie-only group reading as alive, kill_process_group
    # spins out the escalation deadline AND then the drain deadline -- two full grace periods --
    # before handing the worker back, though the cuda context went with the exit. the stub stands in
    # for the direct child: subprocess.Popen would reap, and reaping is what the worker image's
    # missing init never does.
    class _ExitedChild:
        def __init__(self, pid):
            self.pid = pid

        def wait(self, timeout=None):
            return 0

    pid = os.fork()
    if pid == 0:  # pragma: no cover - child never returns to pytest
        os.setsid()
        os._exit(0)
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not vc._process_is_zombie(pid):
            time.sleep(0.02)
        assert vc._process_is_zombie(pid), "child should be an unreaped zombie"

        started = time.monotonic()
        vc.kill_process_group(_ExitedChild(pid))
        elapsed = time.monotonic() - started

        assert elapsed < vc._TEARDOWN_GRACE_S, (
            f"teardown took {elapsed:.1f}s waiting on a zombie that no signal can clear"
        )
    finally:
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pid, 0)


@_needs_process_teardown
def test_an_unreadable_process_status_does_not_read_as_exited(monkeypatch):
    # teardown can run with the worker out of file descriptors, where opening /proc/<pid>/stat
    # raises EMFILE for every live member at once. reading that as a zombie made the whole group
    # report drained, so SIGKILL was never sent and an EngineCore kept its cuda context precisely on
    # a resource-failure path. only disappearance proves an exit.
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True
    )
    try:
        pgid = os.getpgid(child.pid)
        assert not vc._process_is_zombie(child.pid), "a running pid is not a zombie"

        real_open = builtins.open

        def out_of_descriptors(path, *args, **kwargs):
            if str(path).startswith("/proc/") and str(path).endswith("/stat"):
                raise OSError(errno.EMFILE, "Too many open files")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", out_of_descriptors)
        assert not vc._process_is_zombie(child.pid), "EMFILE is not evidence that the pid exited"
        assert vc._process_group_alive(pgid), (
            "a group whose members are merely unreadable must still be signalled"
        )
    finally:
        child.kill()
        child.wait(timeout=10)


def test_a_missing_process_still_reads_as_exited():
    # the other half of the same split: narrowing to disappearance is only correct if disappearance
    # is still caught, or a group that really has drained would spin out both deadlines.
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=10)  # reaped by Popen, so the pid is fully gone rather than a zombie
    assert vc._process_is_zombie(child.pid), "a reaped pid must not read as alive"


@_needs_process_teardown
def test_an_empty_group_snapshot_is_rechecked_against_the_kernel(monkeypatch):
    # /proc can be listed just before a fork publishes a new child while the member that WAS there
    # exits inside the same window, so one snapshot can show nobody in a group that still holds the
    # gpu. teardown then returned before SIGKILL and the newly forked child -- which inherited the
    # group and never received the earlier signal -- kept its cuda context.
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True
    )
    try:
        pgid = os.getpgid(child.pid)
        monkeypatch.setattr(vc, "_process_group_members", lambda _pgid: [])
        assert vc._process_group_alive(pgid), (
            "an empty walk of a group the kernel still knows is a missed member, not a drain"
        )
    finally:
        child.kill()
        child.wait(timeout=10)


@_needs_process_teardown
def test_a_zombie_only_snapshot_taken_before_a_fork_is_not_a_drain(subreaper):
    """The same race the empty snapshot has, one member later: /proc is listed with the leader
    present, and the leader forks and exits before its status is inspected. The walk is then
    nonempty and zombie-only while the child that inherited the group is alive and unlisted, so a
    verdict taken from it skips SIGKILL and leaves the gpu held."""
    leader = os.fork()
    if leader == 0:  # pragma: no cover - runs only in the forked child
        os.setpgid(0, 0)
        if os.fork() == 0:
            time.sleep(300)
            os._exit(0)
        os._exit(0)  # the leader exits at once; its child keeps the group
    child = None
    try:
        os.setpgid(leader, leader)
        deadline = time.monotonic() + 10
        while not vc._process_is_zombie(leader):
            assert time.monotonic() < deadline, "the leader never exited"
            time.sleep(0.01)
        child = next(p for p in vc._process_group_members(leader) or () if p != leader)

        # the stale walk: taken before the fork was published, so it sees only the leader.
        scans = [[leader]]
        real = vc._process_group_members
        vc._process_group_members = lambda pgid: scans.pop(0) if scans else real(pgid)
        try:
            assert vc._process_group_alive(leader), (
                "a zombie-only walk taken mid-fork was read as a drain, so SIGKILL is skipped"
            )
        finally:
            vc._process_group_members = real
    finally:
        if child is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(child, signal.SIGKILL)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(child, 0)
        with contextlib.suppress(ChildProcessError):
            os.waitpid(leader, 0)


@_needs_process_teardown
def test_a_group_that_really_is_zombie_only_still_drains(subreaper):
    # and the recheck must not make the drain unreachable. a zombie holds the group id, so signal 0
    # keeps answering for one -- the escalation would burn both deadlines on every teardown, which
    # is the defect `_process_group_alive` exists to prevent.
    pid = os.fork()
    if pid == 0:  # pragma: no cover - runs only in the forked child
        os.setpgid(0, 0)
        os._exit(0)
    try:
        os.setpgid(pid, pid)
        deadline = time.monotonic() + 10
        while not vc._process_is_zombie(pid):
            assert time.monotonic() < deadline, "the member never exited"
            time.sleep(0.01)
        assert vc._process_group_addressable(pid), "a zombie must still hold the group id"
        assert not vc._process_group_alive(pid), (
            "a settled zombie-only group is drained; reporting it alive burns both deadlines"
        )
    finally:
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pid, 0)


@_needs_process_teardown
def test_the_subreaper_setting_does_not_outlive_the_test_that_enabled_it():
    """Both halves share one test so ordering cannot make a later restore check pass.

    Exercise initial subreaper settings 0 and 1 so the test proves restoration, not merely the
    process default.
    """
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    entered = _child_subreaper_setting()
    try:
        for previous in (0, 1):
            libc.prctl(_PR_SET_CHILD_SUBREAPER, previous, 0, 0, 0)
            with _child_subreaper_enabled():
                assert _child_subreaper_setting() == 1, (
                    "adoption was never enabled, so the teardown tests above prove nothing"
                )
            assert _child_subreaper_setting() == previous, (
                "PR_SET_CHILD_SUBREAPER leaked out, so later tests adopt orphans they never reap"
            )
    finally:
        libc.prctl(_PR_SET_CHILD_SUBREAPER, entered, 0, 0, 0)


@_needs_process_teardown
def test_a_production_entry_point_does_not_leave_this_process_adopting_orphans():
    """The entry points claim adoption unconditionally, and that claim is PROCESS-global.

    An autouse fixture must restore the flag after every test or later tests adopt unreaped orphans
    and become order-dependent.
    """
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    entered = _child_subreaper_setting()
    try:
        libc.prctl(_PR_SET_CHILD_SUBREAPER, 0, 0, 0, 0)
        assert _child_subreaper_setting() == 0

        code = vc.run_verl_training(["bash", "-c", "echo 'step: 1'"], env=dict(os.environ))
        assert code == 0
        # inside the test the claim is expected: the entry point needs it to reap its own orphans.
        assert _child_subreaper_setting() == 1, "the entry point never claimed adoption at all"
    finally:
        libc.prctl(_PR_SET_CHILD_SUBREAPER, entered, 0, 0, 0)


@_needs_process_teardown
def test_the_conftest_fixture_restores_the_flag_the_entry_point_set():
    """The half the test above cannot see: the flag is restored AFTER a test, not during it.

    Run a subprocess so pytest teardown occurs. Place the probe under `tests/` so the conftest
    fixture under test is actually loaded.
    """
    probe = (
        "import ctypes\n"
        "libc = ctypes.CDLL('libc.so.6', use_errno=True)\n"
        "def flag():\n"
        "    cur = ctypes.c_int(0)\n"
        "    libc.prctl(37, ctypes.byref(cur), 0, 0, 0)\n"
        "    return cur.value\n"
        "import os\n"
        "from flash.engine.worker.train.entry import backend_common as vc\n"
        "def test_claims_adoption():\n"
        "    vc.run_verl_training(['bash', '-c', \"echo 'step: 1'\"], env=dict(os.environ))\n"
        "    assert flag() == 1\n"
        "def test_zz_sees_it_restored():\n"
        "    assert flag() == 0, 'PR_SET_CHILD_SUBREAPER leaked into a later test'\n"
        "    assert vc._ADOPTS_ORPHANS is False, 'the module still believes a claim it lost'\n"
    )
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory(dir=tests_dir) as scratch:
        path = os.path.join(scratch, "test_subreaper_leak_probe.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(probe)
        done = subprocess.run(
            [sys.executable, "-m", "pytest", path, "-p", "no:randomly", "-q", "-o", "addopts="],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            timeout=300,
        )
    assert done.returncode == 0, f"{done.stdout[-3000:]}\n{done.stderr[-2000:]}"


@_needs_process_teardown
def test_an_empty_snapshot_of_a_gone_group_is_still_drained():
    # and the recheck must not make teardown unfailable: when the group is genuinely gone, the walk
    # being empty has to stay an answer or every teardown burns both deadlines.
    child = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    pgid = os.getpgid(child.pid)
    child.wait(timeout=10)
    assert vc._process_group_members(pgid) == []
    assert not vc._process_group_alive(pgid), (
        "a group with no members and no kernel entry is drained"
    )


@_needs_process_teardown
def test_teardown_reaps_an_adopted_grandchild_rather_than_leaving_a_zombie(
    quick_teardown_grace, subreaper
):
    # pid 1 has no init in Dockerfile.worker, so an EngineCore orphaned when the trainer exits is
    # reparented onto the worker. SIGKILL turns it into a zombie that no signal can clear -- only a
    # wait can -- so leaving it costs one permanent process-table entry per failed or cancelled run.
    # the `subreaper` fixture reproduces that adoption inside this test process.
    leader_src = (
        "import subprocess, sys, time\n"
        "g = subprocess.Popen([sys.executable, '-c',\n"
        '    "import signal,time\\n"\n'
        '    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"\n'
        "    \"print('gready', flush=True)\\n\"\n"
        '    "time.sleep(300)\\n"], stdout=subprocess.PIPE, text=True)\n'
        "assert g.stdout.readline().strip() == 'gready'\n"
        "print(g.pid, flush=True)\n"
        "time.sleep(300)\n"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", leader_src],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    grandchild_pid = None
    try:
        assert leader.stdout is not None
        grandchild_pid = int(leader.stdout.readline().strip())
        vc.kill_process_group(leader)
        # the escalation already proves it was killed; what is under test is that nothing is left in
        # the process table afterwards.
        assert not os.path.exists(f"/proc/{grandchild_pid}"), (
            "an adopted grandchild was killed but never reaped, leaking a pid per run"
        )
        # the direct child is skipped so Popen keeps the status it is owed.
        assert leader.returncode is not None
    finally:
        if grandchild_pid is not None:  # pragma: no cover - only on an unexpected failure
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(grandchild_pid, signal.SIGKILL)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(grandchild_pid, 0)
        if leader.poll() is None:  # pragma: no cover - only on an unexpected failure
            leader.kill()
            leader.wait(timeout=10)
        if leader.stdout is not None:
            leader.stdout.close()


@_needs_process_teardown
def test_a_straggler_that_dies_after_the_deadline_is_reaped_by_the_next_teardown(
    monkeypatch, subreaper
):
    # SIGKILL cannot be refused, but it also cannot be DELIVERED to a process in uninterruptible
    # sleep. such a member can outlast the drain deadline and turn into a zombie afterwards -- past
    # the last wait its own teardown performs -- so with no record no future wait is ever scheduled
    # and the entry is permanent on a pid-1 worker.
    monkeypatch.setattr(vc, "_UNREAPED_STRAGGLERS", set())

    pid = os.fork()
    if pid == 0:  # pragma: no cover - runs only in the forked child
        os.setpgid(0, 0)
        time.sleep(300)
        os._exit(0)
    try:
        os.setpgid(pid, pid)
        # the state at the drain deadline: the member is ours and STILL RUNNING, so the nonblocking
        # wait cannot take a status. skip a pid that is not in this group so nothing is excluded.
        vc._reap_group_zombies(pid, skip=-1)
        assert pid in vc._UNREAPED_STRAGGLERS, (
            "a live adopted member was forgotten at the deadline, so no later wait can reap it"
        )

        # teardown has returned. the member dies only now, exactly as one leaving uninterruptible
        # sleep with a pending SIGKILL would.
        os.kill(pid, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while os.path.exists(f"/proc/{pid}") and not vc._process_is_zombie(pid):
            assert time.monotonic() < deadline, "the straggler never exited"
            time.sleep(0.01)

        # the NEXT teardown is what must collect it, so drive a real one rather than calling the
        # sweep directly -- otherwise this passes even with the sweep never wired into the path.
        later = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.exit(0)"], start_new_session=True
        )
        later.wait(timeout=10)
        vc.kill_process_group(later)

        assert not os.path.exists(f"/proc/{pid}"), (
            "a straggler that died after its deadline was never reaped, leaking a pid per run"
        )
        assert pid not in vc._UNREAPED_STRAGGLERS
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pid, 0)


@_needs_process_teardown
def test_a_straggler_that_was_never_ours_is_not_tracked_forever(monkeypatch):
    # the other direction: `waitpid` raising ChildProcessError means the pid was never ours, which
    # is the ordinary case anywhere but pid 1. remembering those would grow the set without bound
    # on every teardown, and no wait here could ever clear them.
    monkeypatch.setattr(vc, "_UNREAPED_STRAGGLERS", set())
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True
    )
    try:
        # a real, live process in its own group -- but reached through a wait that reports it is
        # not this process's child to reap.
        def _not_ours(pid, options):
            raise ChildProcessError

        monkeypatch.setattr(vc.os, "waitpid", _not_ours)
        vc._reap_group_zombies(os.getpgid(child.pid), skip=-1)
        assert not vc._UNREAPED_STRAGGLERS, (
            "a process this worker cannot reap was tracked anyway, so the set grows unboundedly"
        )
    finally:
        monkeypatch.undo()
        child.kill()
        child.wait(timeout=10)


@_needs_process_teardown
def test_a_job_that_succeeds_still_drains_the_stragglers_an_earlier_one_left(monkeypatch):
    # `kill_process_group` is the only OTHER caller of the sweep, and both of its call sites sit in
    # `except BaseException` blocks. a worker whose later jobs all SUCCEED therefore never schedules
    # the future wait a straggler needs, holding that zombie for the process's whole life -- as pid
    # 1 there is nothing else to reap it. the drain has to happen at every job boundary.
    monkeypatch.setattr(vc, "_UNREAPED_STRAGGLERS", set())

    pid = os.fork()
    if pid == 0:  # pragma: no cover - runs only in the forked child
        os._exit(0)
    try:
        # the state an earlier teardown leaves behind: exited, owed to this process, and recorded
        # because it was still running when that teardown's deadline passed.
        deadline = time.monotonic() + 10
        while not vc._process_is_zombie(pid):
            assert time.monotonic() < deadline, "the straggler never exited"
            time.sleep(0.01)
        vc._UNREAPED_STRAGGLERS.add(pid)

        # a job that runs to completion and exits 0 -- no callback failure, no nonzero code, nothing
        # that routes through the error teardown. drive the real entry point rather than the sweep,
        # or this passes just as well with the sweep never wired into the success path.
        code = vc.run_verl_training(["bash", "-c", "echo 'step: 1'"], env=dict(os.environ))

        assert code == 0
        assert not os.path.exists(f"/proc/{pid}"), (
            "a successful job left an earlier run's zombie in the table: one leaked pid per run"
        )
        assert pid not in vc._UNREAPED_STRAGGLERS
    finally:
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pid, 0)


# the worker as the runpod handler really runs it: a short-lived process that records a straggler
# and then EXITS, with no second job to sweep it. `_train_body.run_mode` in
# `flash/providers/runpod/serverless/endpoints.py`
# spawns one of these per phase and waits for it, so anything the phase leaves behind reparents to
# the persistent handler -- which waits only on the worker -- and stays a zombie for the container's
# life. driven as a subprocess because the leak is about process EXIT, which pytest cannot perform.
_SHORT_LIVED_WORKER = r"""
import os, sys, time
sys.path.insert(0, {repo!r})
from flash.engine.worker.train.entry import backend_common as vc

# a straggler as teardown leaves one: exited, owed to this process, recorded because it was still
# running when the drain deadline passed.
pid = os.fork()
if pid == 0:
    os._exit(0)
deadline = time.monotonic() + 10
while not vc._process_is_zombie(pid):
    assert time.monotonic() < deadline, "the straggler never exited"
    time.sleep(0.01)
vc._UNREAPED_STRAGGLERS.add(pid)
sys.stdout.write(str(pid))
sys.stdout.flush()
# and now the phase ends. no later teardown, no next job: this process is the only reaper the pid
# will ever have.
"""


@_needs_process_teardown
def test_a_worker_that_exits_after_one_phase_still_collects_its_straggler():
    """The deferred reap has to happen before this process exits, because nothing follows it.

    A per-phase RunPod worker loses remembered pids at exit, leaving stragglers as permanent
    zombies.
    """
    done = subprocess.run(
        [sys.executable, "-c", _SHORT_LIVED_WORKER.format(repo=_REPO_ROOT)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, f"worker failed: {done.stderr[-2000:]}"
    straggler = int(done.stdout.strip())
    assert not os.path.exists(f"/proc/{straggler}"), (
        f"straggler {straggler} outlived the worker that owed its status, so the runpod handler "
        "inherits a zombie it never waits on: one leaked pid per phase"
    )


def test_the_exit_drain_is_registered_wherever_stragglers_are_recorded():
    """At import, not from one entry point: the pids are recorded in whichever process ran.

    `run_verl_training` and `kill_process_group` both record stragglers, so hanging the last drain
    off either one leaves the other's pids uncollected when that process exits.
    """
    source = " ".join(inspect.getsource(vc).split())
    assert "atexit.register(_drain_stragglers_before_exit)" in source, (
        "no exit drain is registered, so a straggler recorded by a short-lived worker is never "
        "collected by anything"
    )


def test_the_exit_drain_gives_up_rather_than_holding_the_interpreter_open():
    """A straggler that never exits must cost a pid, not the worker's shutdown."""
    original = set(vc._UNREAPED_STRAGGLERS)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    try:
        vc._UNREAPED_STRAGGLERS.add(child.pid)  # ours, alive, and not going to exit
        started = time.monotonic()
        with mock.patch.object(vc, "_EXIT_DRAIN_S", 0.3):
            vc._drain_stragglers_before_exit()
        assert time.monotonic() - started < 10, (
            "the exit drain blocked on a process that is running"
        )
    finally:
        vc._UNREAPED_STRAGGLERS.clear()
        vc._UNREAPED_STRAGGLERS.update(original)
        child.kill()
        child.wait(timeout=10)


def test_grpo_teardown_uses_the_shared_escalating_kill():
    # the grpo path used to hand-roll killpg(pid, 15) and swallow the wait timeout, so a vllm
    # EngineCore that ignored the term kept its cuda context and stranded the gpu for later jobs.
    # pin the call site: a bare killpg here would reintroduce exactly that.
    source = inspect.getsource(rl_train_runner)
    assert "kill_process_group(self._proc, process_group_id=self._process_group_id)" in source
    assert "os.killpg" not in source, "grpo teardown must not hand-roll a non-escalating killpg"


def test_every_test_touching_a_linux_only_api_carries_the_platform_guard():
    """The guard was applied test-by-test and kept being missed on the next one added.

    Inspect this file and require the capability marker beside every real fork/proc use; source-only
    assertions are exempt.
    """
    linux_only = ("os.fork", "os.getpgid", "os.killpg", "os.waitpid", "libc.so.6", "/proc/")
    exempt = {"test_every_test_touching_a_linux_only_api_carries_the_platform_guard"}
    source = inspect.getsource(sys.modules[__name__])
    lines = source.splitlines()

    unguarded = []
    for node in ast.parse(source).body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        if node.name in exempt:
            continue
        body = "\n".join(lines[node.lineno - 1 : node.end_lineno])
        if "inspect.getsource" in body:
            continue
        needs = [api for api in linux_only if api in body]
        needs += [f for f in ("subreaper",) if f in {a.arg for a in node.args.args}]
        guarded = any(
            isinstance(d, ast.Name) and d.id == "_needs_process_teardown"
            for d in node.decorator_list
        )
        if needs and not guarded:
            unguarded.append(f"{node.name} uses {sorted(set(needs))}")

    assert not unguarded, "these tests need @_needs_process_teardown:\n  " + "\n  ".join(unguarded)


# the container topology, as a program: a stand-in for `/rp_handler.py` (pid 1, adopts orphans, only
# ever waits on the worker) spawning the flash worker, which spawns the trainer, which spawns an
# EngineCore and exits. run as a subprocess so the WORKER is not a subreaper unless the code under
# test makes it one -- the pytest process cannot host this, because the fixture that gives these
# tests adopted grandchildren manufactures exactly the condition production lacks.
_TOPOLOGY_PROBE = r"""
import ctypes, os, signal, sys, time
sys.path.insert(0, {repo!r})
from flash.engine.worker.train.entry.backend_common import _reap, adopt_orphaned_descendants

CLAIM = {claim!r}


def state_of(pid):
    try:
        with open("/proc/%d/stat" % pid) as f:
            data = f.read()
        return data[data.rindex(")") + 2]
    except (FileNotFoundError, ProcessLookupError):
        return None


# the handler: pid 1 adopts orphans, so ask the kernel for the same. it waits only on the worker.
ctypes.CDLL("libc.so.6", use_errno=True).prctl(36, 1, 0, 0, 0)
r_e, w_e = os.pipe()
r_res, w_res = os.pipe()

worker = os.fork()
if worker == 0:
    os.close(r_res)
    try:
        if CLAIM:
            adopt_orphaned_descendants()
        trainer = os.fork()
        if trainer == 0:
            os.close(r_e)
            os.setsid()                      # start_new_session=True
            engine = os.fork()
            if engine == 0:
                os.close(w_e)
                signal.pause()
                os._exit(0)
            os.write(w_e, str(engine).encode())
            os._exit(0)                      # the trainer exits, orphaning the EngineCore
        os.close(w_e)
        engine = int(os.read(r_e, 32).decode())
        os.waitpid(trainer, 0)               # Popen reaps its own direct child
        time.sleep(0.5)                      # let the kernel finish reparenting
        os.kill(engine, signal.SIGKILL)
        time.sleep(0.5)
        claimed = _reap(engine)
        time.sleep(0.3)
        os.write(w_res, ("%s|%s|%s" % (claimed, state_of(engine) or "gone", engine)).encode())
    finally:
        os._exit(0)

os.close(w_e)
os.close(w_res)
result = os.read(r_res, 64).decode()
os.waitpid(worker, 0)
# the negative control leaves the EngineCore a zombie ON PURPOSE -- that is the result it reports.
# but this handler is its subreaper (line 1173), so the entry is OURS, and exiting here would
# reparent it to whatever runs pytest and leak one process-table slot per run of that test. the
# state above was already recorded, so collecting it now costs the test nothing.
#
# reported rather than done silently: whether the leak is VISIBLE depends on the pid 1 the suite
# happens to run under -- systemd reaps orphans, a container's `python rp_handler.py` does not -- so
# a test that only asks whether the pid disappeared cannot fail on the machine most likely to run
# it. what this handler collected is the same either way.
collected = []
while True:
    try:
        reaped, _ = os.waitpid(-1, 0)
    except ChildProcessError:
        break
    collected.append(reaped)
print("%s|%s" % (result, ",".join(str(p) for p in collected)), end="")
"""


def _run_topology_probe(*, claim: bool) -> tuple[str, str, int, list[int]]:
    """(what `_reap` claimed, the orphan's /proc state, its pid, pids the handler reaped on exit)."""
    done = subprocess.run(
        [sys.executable, "-c", _TOPOLOGY_PROBE.format(repo=_REPO_ROOT, claim=claim)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, f"probe failed: {done.stderr[-2000:]}"
    claimed, state, engine, collected = done.stdout.strip().split("|")
    return claimed, state, int(engine), [int(p) for p in collected.split(",") if p]


@_needs_process_teardown
def test_an_orphan_is_actually_reaped_in_the_container_process_topology():
    """The reaping only works if this process ADOPTS the orphan, and it is not pid 1.

    Production reparents EngineCore to `/rp_handler.py`, beyond the worker's `waitpid` reach. Run in
    a subprocess because the test subreaper fixture would otherwise hide this topology.
    """
    claimed, state, _engine, _collected = _run_topology_probe(claim=True)
    assert state == "gone", (
        f"the orphan is still {state!r} after _reap, so the worker never adopted it and every "
        "wait in this module is a no-op in the container"
    )
    assert claimed == "True"


@_needs_process_teardown
def test_without_the_claim_the_orphan_reparents_past_this_process():
    """The negative control, and the reason the claim exists: identical topology, no claim.

    If the orphan disappears here, something else reaped it and the positive test proves nothing.
    """
    claimed, state, engine, collected = _run_topology_probe(claim=False)
    assert state == "Z", (
        "the orphan was collected without the claim, so the test above proves nothing"
    )
    assert claimed == "True", (
        "`_reap` reported the zombie as handled, which is what makes this silent"
    )
    # the zombie this control deliberately produces is the probe's to clear before it exits. it is a
    # subreaper, so the entry is its own; exiting would hand it to whatever runs pytest and leak one
    # process-table slot per run. asserted on what the probe COLLECTED rather than on the pid
    # disappearing: under systemd the orphan is reaped either way, so the disappearance is not
    # evidence and a test built on it could never fail here.
    assert engine in collected, (
        f"the probe exited without reaping orphan {engine}, leaking a process-table entry to "
        "whatever inherits it -- invisible under an init that reaps, permanent under one that does not"
    )


@_needs_process_teardown
def test_the_claim_is_made_before_each_verl_process_is_spawned():
    """Order matters: the kernel only reparents to a process that was ALREADY a subreaper.

    Claiming after the child exists leaves any grandchild it has already orphaned parented
    elsewhere, so the fix would work only for the second job onward on a reused worker.
    """
    for fn in (vc._run_streaming_verl_subprocess, rl_train_runner._execute_rl_child):
        src = " ".join(inspect.getsource(fn).split())
        assert "adopt_orphaned_descendants()" in src, f"{fn.__name__} never claims its orphans"
        assert src.index("adopt_orphaned_descendants()") < src.index("subprocess.Popen("), (
            f"{fn.__name__} claims orphans after spawning, too late for the kernel to reparent"
        )


def test_the_claim_survives_a_kernel_that_refuses_it():
    """A kernel without PR_SET_CHILD_SUBREAPER (linux < 3.4) must not fail the run.

    A leaked zombie costs a pid; raising here would cost the job.
    """
    calls = []

    class _RefusingLibc:
        def prctl(self, *args):
            calls.append(args)
            return -1

    original = vc._ADOPTS_ORPHANS
    try:
        vc._ADOPTS_ORPHANS = False
        with mock.patch.object(vc.ctypes, "CDLL", return_value=_RefusingLibc()):
            assert vc.adopt_orphaned_descendants() is False
        assert calls, "the prctl was never attempted"
        assert vc._ADOPTS_ORPHANS is False, "a refused claim must not be remembered as granted"
    finally:
        vc._ADOPTS_ORPHANS = original


def test_a_missing_libc_does_not_fail_the_run():
    original = vc._ADOPTS_ORPHANS
    try:
        vc._ADOPTS_ORPHANS = False
        with mock.patch.object(vc.ctypes, "CDLL", side_effect=OSError("no libc")):
            assert vc.adopt_orphaned_descendants() is False
    finally:
        vc._ADOPTS_ORPHANS = original


def test_the_grpo_success_path_drains_stragglers_too():
    """`kill_process_group` runs on exceptions alone in the grpo loop, so without a drain on the
    ordinary exit a worker whose later jobs all SUCCEED keeps a straggler zombie for life.
    """
    entry = " ".join(inspect.getsource(rl_train.run_rl_train).split())
    teardown = " ".join(inspect.getsource(rl_train._shutdown_rl_runtime).split())
    finally_block = entry[entry.rindex("finally:") :]
    assert "_shutdown_rl_runtime(" in finally_block
    assert "reap_stragglers()" in teardown, (
        "the grpo teardown collects stragglers only when the run FAILS"
    )


def _flashinfer_caps(*, flashinfer_ok=True):
    """the batched probe's answer for a child whose `import flashinfer` did or did not work."""
    return dict(vc._CAPABILITIES_UNAVAILABLE, flashinfer=flashinfer_ok)


def _cc_probe(monkeypatch, cc):
    """stub the batched probe; `cc` is what torch reports, or None for no cuda."""

    def fake_run(cmd, **kwargs):
        payload = dict(vc._CAPABILITIES_UNAVAILABLE, capability=cc)
        return SimpleNamespace(
            returncode=0, stdout="FLASH_VERL_CAPS=" + json.dumps(payload) + "\n", stderr=""
        )

    monkeypatch.setattr(vc.subprocess, "run", fake_run)


def test_the_capability_probe_reads_what_torch_reports(monkeypatch):
    _cc_probe(monkeypatch, [8, 9])
    caps = vc.probe_verl_capabilities("/verl/bin/python", "")
    assert vc.verl_device_capability(caps) == (8, 9)


def test_the_capability_probe_reports_no_cuda_as_none(monkeypatch):
    # no visible card: the child answers null, which must surface as None rather than as a crash
    # on the launch path or as a card the caller then reasons about.
    _cc_probe(monkeypatch, None)
    caps = vc.probe_verl_capabilities("/verl/bin/python", "")
    assert vc.verl_device_capability(caps) is None


def test_the_capability_probe_reports_failure_as_none(monkeypatch):
    def boom(*a, **k):
        raise OSError("no interpreter")

    monkeypatch.setattr(vc.subprocess, "run", boom)
    caps = vc.probe_verl_capabilities("/verl/bin/python", "")
    assert vc.verl_device_capability(caps) is None


def test_a_malformed_capability_answer_reads_as_none(monkeypatch):
    # the child's answer is json now, so a wrong-shaped value reaches the parser rather than
    # literal_eval. it must degrade to "leave the default alone", never raise on the launch path.
    for bogus in ("8.9", [8], {}, "junk"):
        assert vc.verl_device_capability({"capability": bogus}) is None


def test_the_capability_probe_runs_against_the_verl_interpreter(monkeypatch):
    # verl owns the rollout engine and pins its own torch/vllm, so flash's own torch would answer
    # for the wrong environment -- and on a heterogeneous host, potentially the wrong card.
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        payload = dict(vc._CAPABILITIES_UNAVAILABLE, capability=[8, 9])
        return SimpleNamespace(
            returncode=0, stdout="FLASH_VERL_CAPS=" + json.dumps(payload) + "\n", stderr=""
        )

    monkeypatch.setattr(vc.subprocess, "run", fake_run)
    vc.probe_verl_capabilities("/verl/bin/python", "")
    assert calls
    assert all(cmd[0] == "/verl/bin/python" for cmd in calls)


def test_every_capability_question_costs_exactly_one_child(monkeypatch):
    # THE point of batching: each question used to spawn its own interpreter and pay a full
    # torch/verl import to answer an attribute lookup. every consumer below is a function OF the
    # one blob, so the whole set must cost exactly one subprocess -- and none of them may go back
    # to the interpreter for something it was already handed.
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        payload = dict(
            vc._CAPABILITIES_UNAVAILABLE,
            capability=[12, 0],
            flashinfer=True,
            wandb=True,
            gdn_boundary_resets=True,
            rollout_fields=["mask_truncated_completions"],
        )
        return SimpleNamespace(
            returncode=0, stdout="FLASH_VERL_CAPS=" + json.dumps(payload) + "\n", stderr=""
        )

    monkeypatch.setattr(vc.subprocess, "run", fake_run)
    monkeypatch.setenv("WANDB_API_KEY", "k")
    module = "transformers.models.qwen3_5.modeling_qwen3_5"
    caps = vc.probe_verl_capabilities("/verl/bin/python", module)
    assert len(calls) == 1
    cc = vc.verl_device_capability(caps)
    vc.resolve_rollout_enforce_eager(cc)
    vc.resolve_blackwell_attention_backends(caps, cc)
    vc.resolve_verl_loggers(caps)
    vc.verl_declares_rollout_field(caps, "mask_truncated_completions")
    vc.gdn_reset_arch_from_caps(caps, module)
    assert len(calls) == 1


def test_sm86_forces_the_rollout_eager():
    # the ONLY measured graph-capture failure. under graphs an A10G repeated a prompt to the token
    # cap without ever emitting EOS, where its own eager arm answered in 19 tokens; reproduced on a
    # second prompt with sm90 and sm86-eager both normal on the identical input. degeneration, not
    # the wording drift capture is entitled to.
    assert vc.resolve_rollout_enforce_eager((8, 6)) is True


@pytest.mark.parametrize("cc", [(8, 0), (8, 9), (9, 0), (10, 0), (12, 0)])
def test_arches_without_a_measured_capture_defect_keep_graphs(cc):
    # retain graphs on sm89 because matched measurements found no termination or quality regression
    # and lower host-RAM capture cost than sm90. sm89 is the default 0.8B/2B route, where eager also
    # disables torch.compile. a100/h100 and blackwell already require graphs.
    assert vc.resolve_rollout_enforce_eager(cc) is False


@pytest.mark.parametrize("cc", [(7, 5), (11, 0)])
def test_an_unmeasured_arch_is_not_forced_eager_by_another_arch_defect(cc):
    # sm86's defect is sm86's. an arch nobody has run must not inherit eager from it -- that is how
    # sm89 ended up excluded on an allowlist it was never measured against. an unmeasured card keeps
    # verl's own default, exactly as an unanswerable probe does below.
    assert vc.resolve_rollout_enforce_eager(cc) is False


def test_an_unknown_capability_leaves_verl_graph_capture_alone():
    # a probe that could not answer must not guess eager onto an unknown card.
    assert vc.resolve_rollout_enforce_eager(None) is False


@pytest.mark.parametrize("major", [10, 12])
def test_blackwell_pins_flashinfer_and_sdpa_vit(major):
    assert vc.resolve_blackwell_attention_backends(_flashinfer_caps(), (major, 0)) == (
        "FLASHINFER",
        "TORCH_SDPA",
    )


def test_blackwell_falls_back_to_triton_when_flashinfer_is_abi_broken():
    # flashinfer can install yet fail to import against this torch. an unconditional FLASHINFER would
    # ship fine and only die at engine init on a paid gpu, so degrade to a registered PTX-independent
    # decoder backend. the ViT pin is unaffected -- it is a separate selection.
    assert vc.resolve_blackwell_attention_backends(
        _flashinfer_caps(flashinfer_ok=False), (12, 0)
    ) == (
        "TRITON_ATTN",
        "TORCH_SDPA",
    )


@pytest.mark.parametrize("cc", [(8, 0), (8, 9), (9, 0), None])
def test_non_blackwell_leaves_both_backends_to_vllm(monkeypatch, cc):
    # vllm's capability-ordered defaults are correct off blackwell (flash-attn is the right decoder
    # choice on ampere/hopper), so pinning anything there would override a working selection. an
    # unknown capability is treated the same way: leave the defaults in place. the answer must not
    # depend on what flashinfer did in the child -- off blackwell nothing consumes it.
    monkeypatch.setattr(
        vc.subprocess, "run", lambda *a, **k: pytest.fail("must not spawn a child here")
    )
    for ok in (True, False):
        assert vc.resolve_blackwell_attention_backends(_flashinfer_caps(flashinfer_ok=ok), cc) == (
            None,
            None,
        )


def test_the_flashinfer_question_is_asked_of_the_verl_interpreter():
    # verl owns the rollout engine and pins its own vllm stack, so a flash-side `import flashinfer`
    # would answer for the wrong environment. the question now rides the batched child probe, so
    # assert it is IN that child's script rather than counting a subprocess of its own.
    probe = vc._CAPABILITY_PROBE % {"gdn_module": ""}
    assert "import flashinfer" in probe
    assert '"flashinfer"' in probe


def _ray_session(root, name, *, files, mtime=None):
    """build a ray session dir the way ray lays one out: session_<ts>_<pid>/logs/<file>."""
    logs = root / name / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for filename, body in files.items():
        (logs / filename).write_text(body)
    if mtime is not None:
        os.utime(root / name, (mtime, mtime))
    return root / name


def test_ray_failure_logs_are_collected_so_a_dead_raylet_leaves_evidence(tmp_path):
    # a raylet death prints only the driver's downstream symptom ("Failed to register worker to
    # Raylet: ... End of file"). the reason lives in ray's own session logs, which die with the pod.
    # without this, every raylet failure costs a paid gpu run per guess (VERL-115).
    root = tmp_path / "ray"
    _ray_session(
        root,
        "session_2026-07-31_19-19-30_1_1",
        files={
            "raylet.err": "worker_pool.cc:600: Some workers have not registered within the timeout",
            "gcs_server.out": "gcs alive",
            "dashboard_agent.log": "agent failed to start",
        },
    )

    body = vc.collect_ray_failure_logs(root=str(root))

    assert "have not registered within the timeout" in body
    assert "agent failed to start" in body
    # each section is labelled: six candidate files land in ONE artifact, so an unlabelled
    # concatenation would leave a reader unable to tell which daemon produced which lines.
    for name in ("raylet.err", "gcs_server.out", "dashboard_agent.log"):
        assert f"===== {name}" in body
    # absent files are simply absent -- not an empty section implying the daemon logged nothing.
    assert "raylet.out" not in body


def test_ray_log_collection_picks_the_newest_session_not_a_stale_one(tmp_path):
    # a reused worker accumulates session dirs. collecting an older one would report a PREVIOUS
    # run's raylet as this run's cause -- worse than collecting nothing, because it reads as evidence.
    root = tmp_path / "ray"
    _ray_session(root, "session_old", files={"raylet.err": "STALE"}, mtime=1_000_000)
    _ray_session(root, "session_new", files={"raylet.err": "CURRENT"}, mtime=2_000_000)

    body = vc.collect_ray_failure_logs(root=str(root))

    assert "CURRENT" in body
    assert "STALE" not in body


def test_ray_log_collection_keeps_the_tail_because_the_crash_is_at_the_end(tmp_path):
    # raylet.out on a 128-core box is large and the reason it died is the LAST thing in it. taking
    # the head would upload megabytes of startup chatter and drop the only line that matters.
    root = tmp_path / "ray"
    body = ("filler\n" * 20000) + "FATAL: the actual cause\n"
    _ray_session(root, "session_1", files={"raylet.out": body})

    collected = vc.collect_ray_failure_logs(root=str(root), tail_bytes=2048)

    assert "FATAL: the actual cause" in collected
    # the per-file bound still holds after the header is added.
    assert len(collected) <= 2048 + 100


def test_ray_logs_are_redacted_because_they_are_third_party_text(tmp_path):
    # ray logs a raylet's argv and flash passes credentials to the worker through the environment,
    # so this content has never been through flash's redactor -- unlike the traceback beside it,
    # which main() sanitizes before upload. an unsanitized path here would publish a live token to
    # the run's HF artifacts.
    root = tmp_path / "ray"
    _ray_session(
        root,
        "session_1",
        files={
            "raylet.err": "started with --token=hunter2seekrit and Authorization: Bearer abc.def"
        },
    )

    body = vc.collect_ray_failure_logs(root=str(root))

    assert "hunter2seekrit" not in body
    assert "abc.def" not in body
    assert "<redacted>" in body
    # redaction must not eat the diagnostic context around the secret.
    assert "started with" in body


def test_ray_log_collection_survives_a_tail_that_splits_a_codepoint(tmp_path):
    # seeking to a byte offset lands wherever it lands. a strict decode would raise on a split
    # multibyte character and lose the whole file for a cosmetic reason, on the one path whose
    # entire purpose is preserving evidence.
    root = tmp_path / "ray"
    logs = root / "session_1" / "logs"
    logs.mkdir(parents=True)
    # the seek offset must land INSIDE the multibyte character, or this test passes under a strict
    # decode and proves nothing. 47 bytes total with the 3-byte codepoint at [30:33), so a 16-byte
    # tail seeks to 31 -- one byte into it.
    raw = b"x" * 30 + "中".encode() + b"\nFATAL: cause\n"
    (logs / "raylet.err").write_bytes(raw)
    assert 30 < len(raw) - 16 < 33, "tail offset must split the codepoint or this test cannot fail"

    body = vc.collect_ray_failure_logs(root=str(root), tail_bytes=16)

    assert "FATAL: cause" in body


def test_ray_log_collection_cannot_mask_the_real_error_it_runs_after(tmp_path):
    # this runs on an ALREADY-failing path. if a missing/unreadable ray dir could raise here, the
    # collector would replace the run's real traceback with its own and make things strictly worse.
    assert vc.collect_ray_failure_logs(root=str(tmp_path / "nonexistent")) == ""
    assert vc.latest_ray_session_dir(str(tmp_path / "nonexistent")) is None


def test_a_session_predating_this_run_is_rejected_not_reported_as_its_evidence(tmp_path):
    """A run that dies BEFORE ray starts must report nothing, not the previous attempt's session.

    `/tmp/ray` survives pod reuse, so upload only sessions created by the current attempt.
    """
    root = tmp_path / "ray"
    _ray_session(root, "session_prev", files={"raylet.err": "PREVIOUS RUN"}, mtime=1_000_000)
    started = 1_500_000.0

    assert vc.latest_ray_session_dir(str(root), started_after=started) is None
    assert vc.collect_ray_failure_logs(root=str(root), started_after=started) == ""
    # and the same call without a start time still sees it, so the filter is what rejected it.
    assert "PREVIOUS RUN" in vc.collect_ray_failure_logs(root=str(root))


def test_a_session_live_during_this_run_is_kept_even_if_it_started_earlier(tmp_path):
    # the guard must not throw away real evidence: ray keeps writing its session dir for as long as
    # it runs, so mtime (not the name's timestamp) is what says whether it was live during this run.
    root = tmp_path / "ray"
    _ray_session(root, "session_old_name", files={"raylet.err": "CURRENT CAUSE"}, mtime=2_000_000)

    body = vc.collect_ray_failure_logs(root=str(root), started_after=1_500_000.0)

    assert "CURRENT CAUSE" in body


def test_a_secret_split_by_the_tail_boundary_is_not_uploaded_in_part(tmp_path):
    """Truncating first can leave the REST of a live credential unredacted.

    A tail cut inside one line removes the matching prefix and value head; drop the partial first
    line before redaction.
    """
    root = tmp_path / "ray"
    secret = "ghp_" + "z" * 60
    body = f"--token={secret}\nraylet died: worker registration timeout\n"
    _ray_session(root, "session_1", files={"raylet.err": body})
    # land the boundary INSIDE the token, or this test passes without the fix and proves nothing.
    tail = len("z" * 40) + len("\nraylet died: worker registration timeout\n")
    assert body[-tail] in "z", "tail must begin inside the secret or this test cannot fail"

    collected = vc.collect_ray_failure_logs(root=str(root), tail_bytes=tail)

    assert "z" * 40 not in collected, "a partial credential reached the artifact"
    assert "raylet died: worker registration timeout" in collected


def test_a_multiline_secret_cut_by_the_boundary_is_redacted_past_the_first_line(
    monkeypatch, tmp_path
):
    """Dropping the partial first line does NOT contain a multiline secret.

    Later whole lines of PEM or JSON credentials can remain, so redaction must include safe
    individual secret lines.
    """
    root = tmp_path / "ray"
    body_lines = ["MIIEowIBAAKCAQEAx" + "Q" * 40, "kJ9vTinRUME7Fw3n" + "R" * 40]
    secret = "-----BEGIN PRIVATE KEY-----\n" + "\n".join(body_lines) + "\n-----END PRIVATE KEY-----"
    monkeypatch.setenv("CUSTOMER_API_KEY", secret)

    log = f"--key={secret}\nraylet died: worker registration timeout\n"
    _ray_session(root, "session_1", files={"raylet.err": log})
    # begin the tail inside the FIRST line of the secret, leaving its later lines intact.
    tail = len(log) - log.index("BEGIN") - 2
    assert "-----BEGIN PRIVATE KEY-----" not in log[-tail:], "boundary must split the first line"
    assert body_lines[0] in log[-tail:], (
        "later secret lines must survive the cut, or nothing is proven"
    )

    collected = vc.collect_ray_failure_logs(root=str(root), tail_bytes=tail)

    for line in body_lines:
        assert line not in collected, "a live private key line reached the artifact"
    assert "raylet died: worker registration timeout" in collected


def test_redacting_secret_lines_does_not_strip_ordinary_log_punctuation(monkeypatch, tmp_path):
    """The line-wise needles must not be short enough to gut the diagnostic.

    Exclude tiny components such as `}` that occur throughout innocent logs.
    """
    root = tmp_path / "ray"
    monkeypatch.setenv("CUSTOMER_API_KEY", '{\n  "private_key": "' + "k" * 40 + '"\n}')
    _ray_session(root, "session_1", files={"raylet.err": "raylet died: cfg {\n}\nexit code 1\n"})

    collected = vc.collect_ray_failure_logs(root=str(root), tail_bytes=4096)

    assert "raylet died: cfg {" in collected, "an ordinary brace line was redacted as a secret"
    assert "exit code 1" in collected


def test_a_read_is_capped_even_when_ray_is_still_writing(monkeypatch, tmp_path):
    # ray writes while the worker is failing. sizing the file and THEN reading to EOF consumes
    # everything appended in between -- unbounded, on a dying pod with a bounded upload deadline.
    root = tmp_path / "ray"
    logs = root / "session_1" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    target = logs / "raylet.err"
    target.write_text("x" * 4096)
    real_open = open

    read_sizes: list[int] = []

    class _GrowingFile:
        """a handle that appends 1 MiB the moment the file is OPENED, like a concurrent writer.

        Growth at open exposes size-before-open readers; later growth would not reproduce the race.
        """

        def __init__(self, path):
            with real_open(target, "ab") as sink:
                sink.write(b"y" * (1024 * 1024))
            self._handle = real_open(path, "rb")

        def read(self, *args):
            payload = self._handle.read(*args)
            read_sizes.append(len(payload))
            return payload

        def __getattr__(self, name):
            return getattr(self._handle, name)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._handle.close()
            return False

    # patch the name the module under test resolves, NOT builtins: a global open() replacement
    # leaks into every later test in the session through pytest's own file handling.
    monkeypatch.setattr(
        vc,
        "open",
        lambda path, *a, **k: (
            _GrowingFile(path) if str(path) == str(target) else real_open(path, *a, **k)
        ),
        raising=False,
    )

    vc.collect_ray_failure_logs(root=str(root), tail_bytes=1024)

    # assert on what was READ, not on the returned string: sanitize_diagnostic(limit=tail_bytes)
    # clamps the result either way, so the returned length hides an unbounded read entirely. the
    # cost this fix exists to avoid is the allocation on a dying pod, which only the read shows.
    assert read_sizes, "the log was never read"
    assert max(read_sizes) <= 1024, f"read exceeded the declared bound: {max(read_sizes)} bytes"


def test_a_session_without_a_logs_dir_yields_nothing_rather_than_raising(tmp_path):
    # ray can die before it creates the logs subdir. this runs on an ALREADY-failing path, so a
    # raise here would replace the run's real error with this collector's own.
    (tmp_path / "ray" / "session_1").mkdir(parents=True)
    assert vc.collect_ray_failure_logs(root=str(tmp_path / "ray")) == ""


# ---------------------------------------------------------------------------------------------
# VERL-139: the reward/teacher bridges must not spawn one os thread per request.
# ---------------------------------------------------------------------------------------------
def _drive_bridge(server_cls, *, callers: int, hold: threading.Event, await_arrivals: int):
    """fire `callers` concurrent requests at a bridge whose handler blocks until `hold` is set.

    Holding handlers makes peak concurrency observable. `await_arrivals` cannot exceed the bounded
    pool size, so callers pass the peak they expect to observe.
    """
    arrived = threading.Semaphore(0)

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # keep pytest output clean
            return

        def do_POST(self):
            arrived.release()
            hold.wait(timeout=30)
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = server_cls(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    # count only threads the SERVER owns. threading.active_count() also counts the caller threads
    # this helper starts, which would put the bounded arm over the bound and measure nothing.
    def _server_threads() -> int:
        return sum(1 for t in threading.enumerate() if t not in known)

    known = set(threading.enumerate())
    errors: list[BaseException] = []

    def _call():
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            conn.request("POST", "/step", body=b"{}", headers={"Content-Type": "application/json"})
            conn.getresponse().read()
            conn.close()
        except BaseException as exc:  # recorded, not raised: the caller asserts on it
            errors.append(exc)

    threads = [threading.Thread(target=_call, daemon=True) for _ in range(callers)]
    known.update(threads)  # callers are not the server's threads
    for t in threads:
        t.start()
    # wait until the expected number of requests is actually inside a handler, so the peak reflects
    # real concurrency rather than however fast the callers happened to be scheduled.
    for _ in range(await_arrivals):
        assert arrived.acquire(timeout=30), "a request never reached the handler"
    # let any further threads the server wants to spawn actually appear before sampling, otherwise
    # an unbounded server could be measured before it has grown.
    time.sleep(1.0)
    peak = _server_threads()
    hold.set()
    for t in threads:
        t.join(timeout=30)
    server.shutdown()
    server.server_close()
    return peak, errors


def test_the_bridge_serves_concurrent_requests_from_a_bounded_pool():
    """VERL-139: 64 simultaneous callers must not create 64 handler threads.

    `ThreadingHTTPServer` exhausted container threads and caused `RemoteDisconnected`; drive the
    shipped server class so a copied pool cannot make the test self-fulfilling.
    """
    callers = 64
    assert callers > vc._BRIDGE_WORKER_THREADS * 2, (
        "callers must overshoot the pool to prove a bound"
    )
    bound = vc._BRIDGE_WORKER_THREADS + 4  # + serve_forever and a little slack

    # control: the server this replaced. it must FAIL the same assertion, otherwise the assertion
    # proves nothing and would pass on the shipped defect. give it the same listen backlog so the
    # only difference under test is the thread policy.
    class _UnboundedControl(ThreadingHTTPServer):
        request_queue_size = vc.BoundedThreadingHTTPServer.request_queue_size

    hold = threading.Event()
    try:
        unbounded_peak, unbounded_errors = _drive_bridge(
            _UnboundedControl, callers=callers, hold=hold, await_arrivals=callers
        )
    finally:
        hold.set()
    assert not unbounded_errors, f"control arm failed to run: {unbounded_errors[:3]}"
    assert unbounded_peak > bound, (
        f"control: ThreadingHTTPServer used only {unbounded_peak} threads for {callers} held "
        "requests, so this test cannot distinguish the fix from the defect"
    )

    hold = threading.Event()
    try:
        peak, errors = _drive_bridge(
            vc.BoundedThreadingHTTPServer,
            callers=callers,
            hold=hold,
            # only pool-size handlers can be resident at once -- that ceiling is the fix.
            await_arrivals=vc._BRIDGE_WORKER_THREADS,
        )
    finally:
        hold.set()

    assert not errors, f"requests failed: {errors[:3]}"
    assert peak <= bound, (
        f"bridge grew to {peak} threads for {callers} concurrent requests "
        f"(control used {unbounded_peak}): it is still spawning a thread per request"
    )


def test_both_verl_bridges_use_the_bounded_server():
    """the fix is only real if the reward AND teacher bridges inherit it.

    each bridge is defined in its own module, so a fix applied to one leaves the other able to
    exhaust the thread table on exactly the same rollout shape.
    """
    import flash.engine.worker.train.opd.bridging.batching as opd_batching
    import flash.engine.worker.train.rl.rollout.multi_turn as rl_multi_turn

    # the teacher bridge is a module-level class, so check the type itself.
    assert issubclass(opd_batching._TeacherBridgeHTTPServer, vc.BoundedThreadingHTTPServer), (
        "the opd teacher bridge does not use BoundedThreadingHTTPServer"
    )
    # the reward bridge is defined inside the function that starts it, so it is only reachable
    # through the source. parse rather than substring-match: a comment mentioning the old name
    # must not pass, and a real subclass must not be missed.
    for mod in (rl_multi_turn, opd_batching):
        tree = ast.parse(pathlib.Path(inspect.getfile(mod)).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
                assert name != "ThreadingHTTPServer", (
                    f"{mod.__name__}.{node.name} still subclasses the unbounded "
                    "ThreadingHTTPServer; it will spawn a thread per request"
                )


def _exit_delay_with_pool(pool_expr: str) -> float:
    """Seconds a fresh interpreter takes to exit while ``pool_expr``'s worker is still running.

    Measure in a subprocess because the stdlib joins pool workers only during interpreter shutdown.
    """
    program = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(pathlib.Path(vc.__file__).parents[3])!r})
        from flash.engine.worker.train.entry.backend_common import _DaemonBridgeThreadPool  # noqa: F401
        from concurrent.futures import ThreadPoolExecutor  # noqa: F401

        pool = {pool_expr}
        pool.submit(lambda: time.sleep({_EXIT_PROBE_HANDLER_S}))
        time.sleep(0.3)  # let the handler actually start; a queued future would just be cancelled
        pool.shutdown(wait=False, cancel_futures=True)
        sys.exit(0)
    """)
    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=60
    )
    elapsed = time.monotonic() - start
    assert proc.returncode == 0, f"probe failed: {proc.stderr[-800:]}"
    return elapsed


_EXIT_PROBE_HANDLER_S = 10.0


def test_a_hung_bridge_handler_cannot_hold_the_worker_process_open():
    """A wedged reward/teacher callback must not delay worker exit.

    `ThreadPoolExecutor` workers are joined at interpreter exit even after `shutdown(wait=False)`; a
    hung handler can retain a paid GPU. The stock-executor control must fail the bound.
    """
    control = _exit_delay_with_pool("ThreadPoolExecutor(max_workers=2)")
    assert control > _EXIT_PROBE_HANDLER_S * 0.5, (
        f"control exited in {control:.1f}s, so a stock pool did NOT block on its running handler "
        "here and this test cannot tell the fix from the defect"
    )

    delay = _exit_delay_with_pool("_DaemonBridgeThreadPool(max_workers=2)")
    assert delay < _EXIT_PROBE_HANDLER_S * 0.5, (
        f"exit took {delay:.1f}s with a {_EXIT_PROBE_HANDLER_S}s handler running "
        f"(control: {control:.1f}s): pool workers still hold the interpreter open"
    )


def test_the_bridge_server_uses_the_daemon_pool():
    """The pool class is only a fix if the server actually instantiates it.

    The exit-delay test above measures the pool directly, so without this a correct pool wired to
    nothing would still pass.
    """
    source = inspect.getsource(vc.BoundedThreadingHTTPServer.__init__)
    assert "_DaemonBridgeThreadPool(" in source, (
        "the bridge server still builds a stock ThreadPoolExecutor"
    )


def _exec_tf32_fragment_against(fake_torch):
    """run the rendered tf32 fragment with ``fake_torch`` standing in for torch."""
    with mock.patch.dict(sys.modules, {"torch": fake_torch}):
        exec(compile(vc.render_tf32_shim(), "sitecustomize.py", "exec"), {})


def _tf32_off_torch():
    """a torch stub at torch's real defaults: matmul tf32 OFF, precision 'highest'."""
    precision = []

    return SimpleNamespace(
        set_float32_matmul_precision=precision.append,
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cudnn=SimpleNamespace(allow_tf32=False),
        ),
        _precision_calls=precision,
    )


def test_the_tf32_fragment_actually_enables_tf32():
    """THE regression: torch defaults cuda.matmul.allow_tf32 to False, so an fp32 matmul runs at
    full fp32 rate unless something opts in. assert on the flags AFTER executing the fragment --
    a substring match on the rendered source would pass on a fragment that never runs."""
    fake = _tf32_off_torch()
    _exec_tf32_fragment_against(fake)
    assert fake.backends.cuda.matmul.allow_tf32 is True
    assert fake.backends.cudnn.allow_tf32 is True
    assert fake._precision_calls == ["high"]


def test_the_tf32_fragment_never_aborts_a_paid_run():
    """tf32 is a throughput optimization, so a torch that cannot take these flags must leave
    training running rather than kill the child at sitecustomize time."""

    class _Exploding:
        def __getattr__(self, name):
            raise RuntimeError("torch backends unavailable")

    _exec_tf32_fragment_against(_Exploding())  # must not raise

    # and an absent torch is the same story: sitecustomize runs before verl imports anything.
    with mock.patch.dict(sys.modules, {"torch": None}):
        exec(compile(vc.render_tf32_shim(), "sitecustomize.py", "exec"), {})


@pytest.mark.parametrize("backend", ["grpo", "opd", "sft"])
def test_every_verl_backend_enables_tf32_in_the_child(backend):
    source = vc.render_sitecustomize_bootstrap()

    class Stop(BaseException):
        pass

    class StopOnCudnn:
        allow_tf32 = False

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)
            raise Stop

    fake = _tf32_off_torch()
    fake.backends.cudnn = StopOnCudnn()
    with mock.patch.dict(sys.modules, {"torch": fake}), contextlib.suppress(Stop):
        exec(compile(source, "sitecustomize.py", "exec"), {})

    assert fake._precision_calls == ["high"], backend
    assert fake.backends.cuda.matmul.allow_tf32 is True
    assert fake.backends.cudnn.allow_tf32 is True


# --------------------- tilelang libcudart stub, in the verl CHILD interpreter ---------------------


def _fake_child_tilelang(tmp_path):
    """materialize a tilelang package holding the stub, as the verl venv ships it."""
    pkg = tmp_path / "tilelang"
    (pkg / "lib").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    stub = pkg / "lib" / "libcudart_stub.so"
    stub.write_bytes(b"STUB")
    return pkg, stub


def _run_cudart_fragment(source, tmp_path, *, real: str | None, extra: str = ""):
    """exec ``source`` in a CHILD interpreter with a fake tilelang on the path.

    a subprocess, not this process: the fragment mutates ``sys.path`` and the tilelang package on
    disk, and it exists precisely to run at sitecustomize time in a fresh interpreter. ``ctypes.CDLL``
    is replaced with a probe that FAILS the test if the stub is ever dlopened -- mapping the stub into
    the process is the crash being prevented, so an implementation that "verifies" the stub that way
    must not pass.
    """
    driver = f"""
import ctypes, ctypes.util, glob, os, sys

sys.path.insert(0, {str(tmp_path)!r})
_real = {real!r}
_realglob = glob.glob


def _glob(pat, *a, **k):
    # hermetic: NEVER let a libcudart pattern reach the real glob. this host may have
    # /usr/local/cuda* or a system libcudart, and the fake CDLL below reports cudaDeviceReset for
    # every non-stub candidate -- so delegating would let a CUDA-enabled runner discover a real
    # library and repoint the stub in the test that asserts it is left alone.
    if "libcudart.so" in pat:
        return [_real] if _real else []
    return _realglob(pat, *a, **k)


glob.glob = _glob
ctypes.util.find_library = lambda name: None


class _FakeCudart:
    def __getattr__(self, name):
        if name == "cudaDeviceReset":
            return lambda: 0
        raise AttributeError(name)


def _cdll(path, *a, **k):
    if "libcudart_stub" in str(path):
        raise SystemExit("FRAGMENT_DLOPENED_THE_STUB")
    return _FakeCudart()


ctypes.CDLL = _cdll
{source}
{extra}
"""
    return subprocess.run(
        [sys.executable, "-c", driver], capture_output=True, text=True, timeout=120
    )


def test_the_child_fragment_repoints_the_tilelang_libcudart_stub(tmp_path):
    """THE regression (5 of 6 GRPO models): tilelang's stub lacks ``cudaDeviceReset``, vLLM's
    CuMemAllocator binds it, and engine init dies with an undefined-symbol AttributeError.

    A parent-side repoint could not reach this: the trainer runs in a separate interpreter
    (``/opt/verl-venv``) that has its OWN tilelang and no flash at all. Assert
    on the filesystem AFTER executing the fragment -- a substring match on the rendered source would
    pass on a fragment that never runs.
    """
    _pkg, stub = _fake_child_tilelang(tmp_path)
    real = tmp_path / "libcudart.so.12"
    real.write_bytes(b"REAL-CUDART")

    result = _run_cudart_fragment(
        vc.render_tilelang_cudart_shim(),
        tmp_path,
        real=str(real),
        extra="print('FRAGMENT_DONE', flush=True)",
    )

    assert "FRAGMENT_DLOPENED_THE_STUB" not in (result.stdout + result.stderr), (
        "the fragment dlopened the stub to test it, which maps it into the process and is the "
        "exact crash it exists to prevent"
    )
    assert "FRAGMENT_DONE" in result.stdout, f"fragment aborted the child: {result.stderr[-2000:]}"
    assert stub.is_symlink(), "the stub is still tilelang's own file; vLLM will bind it and abort"
    assert os.path.realpath(stub) == os.path.realpath(str(real))
    assert (tmp_path / "tilelang" / "lib" / "libcudart_stub.so.orig").read_bytes() == b"STUB", (
        "the original stub must be preserved verbatim so the change is reversible"
    )
    # the marker naming the resolved runtime is the only evidence in a run log that the repoint
    # happened and WHICH libcudart it picked. without it a silently-skipped swap and a successful one
    # read identically, and the next undefined-symbol crash has nothing to bisect against.
    assert f"{vc.FLASH_CUDART_STUB_MARKER} -> {real}" in result.stdout, (
        f"the fragment did not report the repoint it performed: {result.stdout[-1000:]}"
    )


def test_the_child_fragment_leaves_the_stub_alone_without_a_real_libcudart(tmp_path):
    """No discoverable real runtime -> leave tilelang's stub untouched rather than break tilelang."""
    _pkg, stub = _fake_child_tilelang(tmp_path)

    result = _run_cudart_fragment(
        vc.render_tilelang_cudart_shim(),
        tmp_path,
        real=None,
        extra="print('FRAGMENT_DONE', flush=True)",
    )

    assert "FRAGMENT_DONE" in result.stdout, f"fragment aborted the child: {result.stderr[-2000:]}"
    assert not stub.is_symlink()
    assert stub.read_bytes() == b"STUB"
    assert not (tmp_path / "tilelang" / "lib" / "libcudart_stub.so.orig").exists()


def test_the_child_fragment_repoints_a_dangling_stub_symlink(tmp_path):
    """A DANGLING stub symlink is NOT "already repointed" -- it leaves tilelang with a broken
    libcudart_stub.so, so the fragment must re-point it at a real runtime.

    This is the case the ``islink and exists`` pair exists for: ``islink`` alone would read a dangling
    link as done and skip, and ``exists`` alone follows the link away and reads it as absent. Distinct
    from the stale *temp swap* link covered below -- that one is a leftover at
    ``.flash-<pid>-<rand>``, while this is the stub path itself pointing at a target that is gone.
    """
    pkg, stub = _fake_child_tilelang(tmp_path)
    stub.unlink()
    stub.symlink_to(tmp_path / "gone-libcudart.so.12")  # dangling: target does not exist
    assert stub.is_symlink()
    assert not stub.exists()

    real = tmp_path / "libcudart.so.12"
    real.write_bytes(b"REAL-CUDART")

    result = _run_cudart_fragment(
        vc.render_tilelang_cudart_shim(),
        tmp_path,
        real=str(real),
        extra="print('FRAGMENT_DONE', flush=True)",
    )

    assert "FRAGMENT_DONE" in result.stdout, f"fragment aborted the child: {result.stderr[-2000:]}"
    assert stub.is_symlink()
    assert stub.exists(), "the stub symlink still dangles; tilelang's libcudart is broken"
    assert os.path.realpath(stub) == os.path.realpath(str(real))
    # nothing to preserve: the original stub was already gone before this ran, so a .orig would be a
    # hard link to a dangling symlink rather than tilelang's real file.
    assert not (pkg / "lib" / "libcudart_stub.so.orig").exists()


def test_the_child_fragment_never_aborts_a_paid_run(tmp_path):
    """An unrepointed stub only kills runs that build a SLEEPING vLLM engine, so a fragment that
    cannot do its job must leave the child starting rather than abort it at sitecustomize time.

    tilelang absent is the ordinary case for a non-GDN venv, and it must not even probe. Note this
    half never enters the swap, so on its own it says nothing about the error path -- the outer
    ``except`` could be deleted outright and this would still pass. That is what the read-only
    sibling test below covers.
    """
    result = _run_cudart_fragment(
        vc.render_tilelang_cudart_shim(),
        tmp_path,  # no tilelang package materialized
        real=None,
        extra="print('FRAGMENT_DONE', flush=True)",
    )
    assert "FRAGMENT_DONE" in result.stdout, f"fragment aborted the child: {result.stderr[-2000:]}"
    assert result.returncode == 0
    assert "repoint failed" not in result.stdout, (
        "tilelang absent is the ordinary case; it must not report a failure"
    )


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root bypasses dir perms")
def test_a_failed_repoint_still_lets_the_child_start(tmp_path):
    """The half of never-abort that actually exercises the handler: tilelang is present and a real
    runtime was found, but the directory holding the stub refuses writes, so ``link()``/``symlink()``
    raise from inside the branch that already decided to swap.

    Only the outer ``except`` keeps that PermissionError from propagating out of sitecustomize and
    killing the interpreter before training starts. It must also SAY it failed: a silent swallow and
    a successful repoint read identically in a run log, leaving the next undefined-symbol crash with
    nothing to bisect against.
    """
    pkg, stub = _fake_child_tilelang(tmp_path)
    real = tmp_path / "libcudart.so.12"
    real.write_bytes(b"REAL-CUDART")
    lib = pkg / "lib"
    lib.chmod(0o555)
    try:
        result = _run_cudart_fragment(
            vc.render_tilelang_cudart_shim(),
            tmp_path,
            real=str(real),
            extra="print('FRAGMENT_DONE', flush=True)",
        )
    finally:
        lib.chmod(0o755)  # let tmp_path cleanup remove it

    assert result.returncode == 0, (
        f"a failed repoint aborted the child instead of importing on: {result.stderr[-2000:]}"
    )
    assert "FRAGMENT_DONE" in result.stdout, (
        f"the exception escaped sitecustomize and killed the run: {result.stderr[-2000:]}"
    )
    assert "repoint failed" in result.stdout, (
        "a swallowed failure must still say so; a silent one is unbisectable from a log"
    )
    assert not stub.is_symlink(), "the stub must be left intact when the swap could not complete"
    assert stub.read_bytes() == b"STUB", "the failed swap corrupted tilelang's own stub"


def test_the_child_fragment_is_idempotent_across_ray_workers(tmp_path):
    """Ray starts several workers against ONE venv, so the fragment runs repeatedly on the same
    files. A second pass must keep the symlink and must not overwrite the saved original with the
    symlink it just made.
    """
    _pkg, stub = _fake_child_tilelang(tmp_path)
    real = tmp_path / "libcudart.so.12"
    real.write_bytes(b"REAL-CUDART")
    source = vc.render_tilelang_cudart_shim()

    result = _run_cudart_fragment(
        source + source,  # two passes in one interpreter
        tmp_path,
        real=str(real),
        extra="print('FRAGMENT_DONE', flush=True)",
    )

    assert "FRAGMENT_DONE" in result.stdout, f"fragment aborted the child: {result.stderr[-2000:]}"
    assert stub.is_symlink()
    assert os.path.realpath(stub) == os.path.realpath(str(real))
    assert (tmp_path / "tilelang" / "lib" / "libcudart_stub.so.orig").read_bytes() == b"STUB"


def test_concurrent_ray_workers_never_lose_the_stub_or_the_original(tmp_path):
    """Ray starts several child interpreters against ONE venv, genuinely at the same time.

    The sequential idempotency test above cannot see the interleaving: with a check-then-act backup,
    two workers both observe ``.orig`` absent, one moves the stub and the other gets
    FileNotFoundError and imports on with the path missing -- or the second overwrites the preserved
    original with the symlink the first just made. Start the interpreters together and assert BOTH
    invariants survive: the stub always resolves, and ``.orig`` is still the real stub bytes.
    """
    _pkg, stub = _fake_child_tilelang(tmp_path)
    real = tmp_path / "libcudart.so.12"
    real.write_bytes(b"REAL-CUDART")
    source = vc.render_tilelang_cudart_shim()

    workers = 8
    barrier = threading.Barrier(workers)
    results: list[subprocess.CompletedProcess] = []
    lock = threading.Lock()

    def _worker():
        barrier.wait()  # release them into the critical section together
        out = _run_cudart_fragment(
            source, tmp_path, real=str(real), extra="print('FRAGMENT_DONE', flush=True)"
        )
        with lock:
            results.append(out)

    threads = [threading.Thread(target=_worker) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=180)

    assert len(results) == workers, "a worker never finished"
    for out in results:
        assert "FRAGMENT_DONE" in out.stdout, (
            f"a concurrent worker aborted instead of importing on: {out.stderr[-1500:]}"
        )

    # the stub must resolve for EVERY worker -- a moved-away stub is the crash, one step later.
    assert stub.is_symlink()
    assert os.path.realpath(stub) == os.path.realpath(str(real))
    backup = tmp_path / "tilelang" / "lib" / "libcudart_stub.so.orig"
    assert backup.exists(), "the preserved original went missing under concurrent workers"
    assert not backup.is_symlink(), (
        "the preserved original was overwritten by a racing worker's symlink"
    )
    assert backup.read_bytes() == b"STUB"
    # no temp links left behind by a worker that died mid-swap.
    leftovers = list((tmp_path / "tilelang" / "lib").glob("libcudart_stub.so.flash-*"))
    assert not leftovers, f"temporary swap links leaked: {leftovers}"


def test_a_stale_swap_link_does_not_block_the_repoint(tmp_path):
    """A worker killed between symlink() and replace() leaves its temp link on disk.

    With a pid-only temp name, the next worker to reuse that pid would hit FileExistsError,
    skip the swap, and leave the stub in place -- a silently unrepointed stub, which is the
    original crash one step later. Plant a stale link for THIS interpreter's pid and assert the
    repoint still lands.
    """
    _pkg, stub = _fake_child_tilelang(tmp_path)
    real = tmp_path / "libcudart.so.12"
    real.write_bytes(b"REAL-CUDART")

    # the child's pid is not knowable from here, so it plants the stale link on ITSELF: a dangling
    # symlink at exactly the pid-only name the old scheme would have chosen, created before the
    # fragment runs. that is precisely the post-crash state a reused pid inherits.
    plant = (
        "import os as _p_os\n"
        f"_p_stale = {str(tmp_path / 'tilelang' / 'lib' / 'libcudart_stub.so')!r}"
        ' + ".flash-" + str(_p_os.getpid())\n'
        '_p_os.symlink("/nonexistent/stale", _p_stale)\n'
    )
    result = _run_cudart_fragment(
        plant + vc.render_tilelang_cudart_shim(),
        tmp_path,
        real=str(real),
        extra="print('FRAGMENT_DONE', flush=True)",
    )

    assert "FRAGMENT_DONE" in result.stdout, f"fragment aborted the child: {result.stderr[-2000:]}"
    assert stub.is_symlink(), (
        "a stale temp swap link blocked the repoint; the stub is still tilelang's own file"
    )
    assert os.path.realpath(stub) == os.path.realpath(str(real))


@pytest.mark.parametrize("backend", ["grpo", "opd", "sft"])
def test_every_verl_backend_repoints_the_stub_in_its_child(backend, tmp_path):
    source = vc.render_sitecustomize_bootstrap()
    _pkg, stub = _fake_child_tilelang(tmp_path)
    real = tmp_path / "libcudart.so.12"
    real.write_bytes(b"REAL-CUDART")

    result = _run_cudart_fragment(source, tmp_path, real=str(real))

    assert "FRAGMENT_DLOPENED_THE_STUB" not in (result.stdout + result.stderr)
    assert stub.is_symlink(), backend
    assert os.path.realpath(stub) == os.path.realpath(str(real))


def test_the_child_probe_is_the_parents_own_source_not_a_copy():
    """The child fragment must SHIP ``perf._find_real_libcudart``, never restate it.

    The bug this whole fragment fixes was parent/child skew: the parent knew how to find libcudart
    and the child did not. A hand-copied probe recreates that skew on a delay -- the next cuda
    release moves a wheel path, someone fixes the parent, CI stays green, and the child silently
    keeps probing the old layout. So assert the rendered fragment carries the parent's real body,
    and that editing the parent moves the child with it.
    """
    from flash.engine.worker.perf import _find_real_libcudart

    rendered = vc.render_tilelang_cudart_shim()
    parent_body = textwrap.dedent(inspect.getsource(_find_real_libcudart))

    # every non-trivial line of the parent's probe must appear in what the child executes.
    missing = [
        line.strip()
        for line in parent_body.splitlines()
        if len(line.strip()) > 12 and line.strip() not in rendered
    ]
    assert not missing, (
        "the child fragment no longer ships perf._find_real_libcudart's own source, so the probe "
        f"can drift from the parent's. lines absent from the fragment: {missing[:5]!r}"
    )

    # and the fragment must CALL it under its canonical name rather than a local re-implementation.
    assert "_find_real_libcudart()" in rendered, (
        "the fragment does not call _find_real_libcudart(); a second probe implementation has crept "
        "back into the child"
    )


def test_rendering_the_probe_ignores_a_monkeypatched_parent(monkeypatch):
    """Rendering must ship the REAL probe body even while ``perf._find_real_libcudart`` is patched.

    ``perf``'s own docstring says tests monkeypatch that name and its callers resolve it through the
    patched module globals. This caller is the exception: it needs the function's SOURCE, so if it
    ever resolved through ``perf.<attr>`` a test double's body would be rendered into the child and
    the shipped shim would be whatever the last test stubbed. Pin the immunity.
    """
    import flash.engine.worker.perf as perf

    before = vc.render_tilelang_cudart_shim()
    monkeypatch.setattr(perf, "_find_real_libcudart", lambda: "/fake/libcudart.so")
    after = vc.render_tilelang_cudart_shim()

    assert before == after, (
        "the rendered fragment changed while perf._find_real_libcudart was monkeypatched; the "
        "renderer is resolving through the patched module global and can ship a test double"
    )
    assert "cudaDeviceReset" in after, (
        "the rendered probe lost its cudaDeviceReset check under a patched parent"
    )
    # asserted as a fragment, not the full path: this file's platform-guard meta-test scans function
    # bodies for the proc-filesystem prefix and would demand @_needs_process_teardown. this test only
    # inspects rendered TEXT and opens nothing, so it needs no process-teardown capability.
    assert "self/maps" in after, (
        "the rendered probe lost its soname resolution under a patched parent"
    )


def test_grpo_does_not_enable_tf32_in_the_parent():
    """the grpo parent holds a cuda context (wait_for_gpu touches the device) but runs no matmuls --
    verl does, out of process. a setup_perf_backends() call here sets flags on the wrong process and
    reads as 'tf32 enabled' in the logs while the trainer runs full fp32."""
    tree = ast.parse(pathlib.Path(inspect.getfile(rl_train)).read_text())
    called = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "setup_perf_backends" not in called, (
        "rl_train calls setup_perf_backends in the parent; the trainer child inherits none of it"
    )


def test_agent_loop_workers_warns_when_a_prime_batch_serializes_the_rollout(capsys):
    """A prime rollout batch silently costs up to 8x rollout throughput, so it must be announced.

    ``agent_loop_workers`` returns the largest divisor of the batch that is <= cap because verl
    asserts the chunk split is exact. For a prime batch the only such divisor is 1, so the pool
    collapses to a single worker and the rollout runs fully serialized. Nothing raises and nothing
    else reports it, so the only symptom is a slower run -- which is indistinguishable from a slow
    environment unless the worker says so.
    """
    assert vc.agent_loop_workers(13) == 1
    warned = capsys.readouterr().out
    assert "serialized" in warned
    assert "13" in warned

    # a composite batch keeps its parallelism and stays quiet: a warning on every run would train
    # the operator to ignore it.
    assert vc.agent_loop_workers(16) == 8
    assert capsys.readouterr().out == ""

    # batch 1 is genuinely one unit of work, not a degraded split, so it is not a warning either.
    assert vc.agent_loop_workers(1) == 1
    assert capsys.readouterr().out == ""


def _fake_torch(*, available=True, capability=(9, 0), raises=False):
    class _Cuda:
        @staticmethod
        def is_available():
            if raises:
                raise RuntimeError("driver died")
            return available

        @staticmethod
        def get_device_capability(_index=0):
            if raises:
                raise RuntimeError("driver died")
            return capability

    return SimpleNamespace(cuda=_Cuda())


def test_fused_ce_backend_picks_triton_only_where_it_measured_faster():
    """The triton fused-CE backend wins on sm90+ and LOSES below it, so the choice is per card.

    Measured fwd+bwd at 4096x2560 into a 248320 vocab, paired and alternating: triton is 2.25x
    faster on H100 (sm90) but 3.28x SLOWER on A100 (sm80) and 1.43x slower on A40 (sm86). A blanket
    flip to triton -- which reads as the obvious "use the fused kernel" change -- would therefore
    slow down every run on the two oldest fleet classes. Pin the boundary so a later edit cannot
    quietly widen it.
    """
    for capability in ((9, 0), (10, 0), (12, 0)):
        assert vc.fused_ce_backend({"capability": list(capability)}) == "triton", capability

    # sm80 (A100) and sm86 (A10/A40) are real fleet classes, not hypotheticals, and sm89 is the
    # RTX 4090 -- the cheapest class the allocator reaches first.
    for capability in ((8, 0), (8, 6), (8, 9)):
        assert vc.fused_ce_backend({"capability": list(capability)}) == "torch", capability


def test_fused_ce_backend_falls_back_to_the_current_behaviour_when_it_cannot_probe():
    """No device, or a device that will not answer, must yield `torch` -- today's behaviour.

    The fallback matters more than it looks: `torch` is what all three trainers pinned before this
    gate existed, so an unreadable capability degrades to the previously shipping configuration
    rather than to an untested one. The probe reports every one of these as a missing or unusable
    `capability` entry (`probe_verl_capabilities` emits None when cuda is absent or the import
    raises), so they all funnel through `verl_device_capability` returning None.
    """
    for caps in ({}, {"capability": None}, {"capability": []}, {"capability": ["x", "y"]}):
        assert vc.fused_ce_backend(caps) == "torch", caps


def test_fused_ce_backend_does_not_open_a_cuda_context_in_the_parent():
    """The gate must read the child's probe, never `torch.cuda` in this long-lived process.

    Initializing cuda here to answer one question retains a context for the process lifetime on the
    devices `torchrun` is about to own -- unbudgeted VRAM against a reserve sized without it. It is
    also the wrong interpreter: verl pins its own torch, and the question is about verl's kernels.

    A stub torch whose `get_device_capability` raises stands in for "somebody reintroduced the live
    probe": if the gate touches it at all, this fails instead of silently regressing.
    """
    exploding = _fake_torch(raises=True)

    with mock.patch.dict(sys.modules, {"torch": exploding}):
        assert vc.fused_ce_backend({"capability": [9, 0]}) == "triton"
        assert vc.fused_ce_backend({"capability": [8, 0]}) == "torch"


def test_every_trainer_asks_for_the_backend_rather_than_hardcoding_one():
    """All three algos must route through the gate; a leftover literal would silently opt out.

    Asserted on the source because the alternative -- building each trainer's full hydra override
    list -- needs a model download and a GPU. The literal `impl_backend=torch` is exactly what this
    change removes, so its absence is the property worth pinning.

    Each trainer resolves the backend once from `caps` and threads the ANSWER through its config
    dict, so the override builders read `cfg[...]` and only the call sites name the helper.
    """
    import flash.engine.worker.train.entry.opd_train as opd_train
    import flash.engine.worker.train.entry.rl_train as rl_train
    import flash.engine.worker.train.entry.sft_train as sft_train

    for module in (sft_train, rl_train, opd_train):
        source = pathlib.Path(module.__file__).read_text()
        code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
        assert "impl_backend=torch" not in code, module.__name__
        assert "fused_ce_backend(caps)" in code, module.__name__
        # the caps-taking form is the whole point of the fix; a bare call would be the parent-cuda
        # probe coming back.
        assert "fused_ce_backend()" not in code, module.__name__


def test_every_trainer_probes_capabilities_for_every_model_not_just_gdn():
    """`caps` must be populated unconditionally, or the backend gate silently degrades to torch.

    Only `gdn_module` is architecture-conditional -- it is `""` for a non-hybrid, which the probe
    accepts as "skip the gdn question" (`probe_verl_capabilities(python_bin, gdn_module="")`). The
    PROBE ITSELF answers questions that have nothing to do with gdn: the device capability this
    backend gate reads, the logger set, the vllm field list.

    Binding `caps` under an `if gdn_hybrid:` would leave it empty for every non-gdn checkpoint, so
    `verl_device_capability` would return None and the gate would yield `torch` on a card where
    triton measured 2.25x faster. That failure is invisible today -- every catalog model is a gdn
    hybrid -- which is exactly why it needs a test rather than a reader noticing it later.
    """
    import flash.engine.worker.train.entry.opd_train as opd_train
    import flash.engine.worker.train.entry.rl_train as rl_train
    import flash.engine.worker.train.entry.sft_train as sft_train

    for module in (sft_train, rl_train, opd_train):
        lines = pathlib.Path(module.__file__).read_text().splitlines()
        probes = [
            (n, line)
            for n, line in enumerate(lines, 1)
            if "caps = probe_verl_capabilities(" in line
        ]
        assert len(probes) == 1, f"{module.__name__}: expected exactly one probe, got {probes}"

        n, line = probes[0]
        indent = len(line) - len(line.lstrip())
        # walk back to the nearest enclosing statement that is less indented; it must not be a
        # conditional on the architecture. checking the text of the probe line alone would pass
        # even when the whole block is skipped for a non-hybrid.
        for prev in reversed(lines[: n - 1]):
            if not prev.strip() or prev.lstrip().startswith("#"):
                continue
            prev_indent = len(prev) - len(prev.lstrip())
            if prev_indent < indent:
                assert "gdn" not in prev.lower(), (
                    f"{module.__name__}:{n} probes capabilities under `{prev.strip()}` -- "
                    "non-gdn runs would get empty caps and lose the triton backend"
                )
                break


# ---------------------- fail-closed sitecustomize fragments (child_io) ----------------------


def test_verify_applied_shim_markers_raises_only_on_missing_names(tmp_path):
    marker_file = tmp_path / "applied_shims.txt"
    marker_file.write_text("entropy-quantile\nkl-ref-adapter\n")
    # complete (and over-complete: ray actors re-record) sets pass.
    vc.verify_applied_shim_markers(str(marker_file), ["entropy-quantile"])
    vc.verify_applied_shim_markers(str(marker_file), ["entropy-quantile", "kl-ref-adapter"])
    with pytest.raises(RuntimeError, match=r"never proved.*\['stop-sequences'\]"):
        vc.verify_applied_shim_markers(str(marker_file), ["entropy-quantile", "stop-sequences"])
    # an absent file is every marker missing, not a pass.
    with pytest.raises(RuntimeError, match="never proved"):
        vc.verify_applied_shim_markers(str(tmp_path / "missing.txt"), ["entropy-quantile"])


def _lora_rollout_server_module(loaded, *, lora_as_adapter=True):
    """a stub shaped like the pinned verl module the guard patches.

    at freesolo-co/verl@32d6200d, ``vLLMHttpServer.generate`` is an async method whose first
    positional argument is ``prompt_ids`` and which carries several keyword-only extras, and
    ``VLLM_LORA_INT_ID`` is a module-level constant the guard reads back rather than hardcoding.
    """
    import types as _types

    module = _types.ModuleType("verl.workers.rollout.vllm_rollout.vllm_async_server")
    module.VLLM_LORA_INT_ID = 123

    class _Engine:
        def __init__(self, ids):
            self._ids = set(ids)
            self.calls = 0

        async def list_loras(self):
            self.calls += 1
            return set(self._ids)

        def generate(self, *args, **kwargs):
            return ("generated", kwargs.get("lora_request"))

    # name matched to verl's own class: the shim looks it up by this exact attribute.
    class vLLMHttpServer:  # noqa: N801
        # set to force the request verl hands the engine, independently of what list_loras says.
        lora_request_override = "unset"

        def __init__(self):
            self.lora_as_adapter = lora_as_adapter
            self.engine = _Engine(loaded)

        async def generate(
            self,
            prompt_ids,
            sampling_params,
            request_id,
            image_data=None,
            priority=0,
            **kwargs,
        ):
            # verl's own body at vllm_async_server.py:524-538: it decides lora_request from its
            # own lookup and hands it to the engine. reproducing that here is what makes these
            # tests able to fail -- a stub that never consults list_loras cannot catch a guard
            # that checks a different value than the one generation actually uses.
            lora_request = None
            if self.lora_as_adapter and module.VLLM_LORA_INT_ID in await self.engine.list_loras():
                lora_request = f"lora:{module.VLLM_LORA_INT_ID}"
            if self.lora_request_override != "unset":
                lora_request = self.lora_request_override
            self.engine.generate(
                prompt=list(prompt_ids),
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
                priority=priority,
            )
            return ("generated", list(prompt_ids), request_id, priority)

    module.vLLMHttpServer = vLLMHttpServer
    return module


def _install_lora_rollout_module(module, monkeypatch):
    import sys as _sys

    for name in (
        "verl",
        "verl.workers",
        "verl.workers.rollout",
        "verl.workers.rollout.vllm_rollout",
    ):
        stub = type(_sys)(name)
        stub.__path__ = []
        monkeypatch.setitem(_sys.modules, name, stub)
    monkeypatch.setitem(_sys.modules, module.__name__, module)


def _apply_lora_rollout_guard(module, monkeypatch):
    """exec the fragment against an already-imported module, the way a ray actor re-import hits it."""
    _install_lora_rollout_module(module, monkeypatch)
    child_runtime._patch_lora_rollout(module)


def test_the_lora_rollout_guard_refuses_to_generate_from_the_base_model(monkeypatch):
    """the defect this guard exists for: verl leaves ``lora_request`` None when the adapter is not
    in the engine's loaded set and generates from the base model anyway -- no raise, no counter.
    an opd run then distils a policy it never rolled out, and the loss curve looks fine."""
    import asyncio

    module = _lora_rollout_server_module(set())
    _apply_lora_rollout_guard(module, monkeypatch)

    with pytest.raises(RuntimeError, match="refusing to roll out from the base model"):
        asyncio.run(module.vLLMHttpServer().generate([1, 2, 3], {}, "req-1"))


def test_the_lora_rollout_guard_names_the_adapter_in_the_failure(monkeypatch):
    """the message has to be actionable from the child log alone: which adapter was expected."""
    import asyncio

    module = _lora_rollout_server_module({7, 9})
    _apply_lora_rollout_guard(module, monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(module.vLLMHttpServer().generate([1], {}, "req-1"))
    assert "123" in str(excinfo.value)


def test_the_lora_rollout_guard_fails_a_base_model_call_its_own_lookup_would_clear(monkeypatch):
    """the guard must gate on the request verl actually built, not on a second lookup of its own.

    here ``list_loras`` reports the adapter loaded, but generation still reaches the engine with
    ``lora_request=None`` -- verl's own lookup disagreed, or the request was built before the
    adapter landed. a guard that re-queries ``list_loras`` and delegates sees a healthy engine and
    waves this through, which is exactly the base-model rollout it was written to stop. only a
    guard reading the outgoing request catches it.
    """
    import asyncio

    module = _lora_rollout_server_module({123})
    _apply_lora_rollout_guard(module, monkeypatch)
    server = module.vLLMHttpServer()
    # verl's decision, not the engine's inventory, is what reaches generation.
    server.lora_request_override = None

    with pytest.raises(RuntimeError, match="refusing to roll out from the base model"):
        asyncio.run(server.generate([1], {}, "req-no-lora-request"))


def test_the_lora_rollout_guard_adds_no_engine_round_trip_per_rollout(monkeypatch):
    """``list_loras`` is an awaited zeromq utility rpc, not a local read. checking it a second time
    would double the adapter-list control-plane traffic on every grpo/opd rollout request."""
    import asyncio

    module = _lora_rollout_server_module({123})
    _apply_lora_rollout_guard(module, monkeypatch)
    server = module.vLLMHttpServer()

    asyncio.run(server.generate([1], {}, "req-a"))
    asyncio.run(server.generate([2], {}, "req-b"))

    # exactly verl's own one lookup per request, with nothing added by the guard.
    assert server.engine.calls == 2


def test_the_lora_rollout_guard_passes_the_call_through_once_the_adapter_is_loaded(monkeypatch):
    """the guard must preserve verl's healthy-path arguments and return value."""
    import asyncio

    module = _lora_rollout_server_module({123})
    _apply_lora_rollout_guard(module, monkeypatch)

    result = asyncio.run(
        module.vLLMHttpServer().generate([1, 2], {}, "req-9", image_data=None, priority=3)
    )

    assert result == ("generated", [1, 2], "req-9", 3)


def test_the_lora_rollout_guard_stays_out_of_a_merged_lora_rollout(monkeypatch):
    """``lora_as_adapter`` is false when the adapter is merged into the base weights, and such a
    rollout legitimately carries no LoRARequest. mirroring verl's own condition keeps the guard
    from failing a run that is behaving correctly."""
    import asyncio

    module = _lora_rollout_server_module(set(), lora_as_adapter=False)
    _apply_lora_rollout_guard(module, monkeypatch)

    assert asyncio.run(module.vLLMHttpServer().generate([1], {}, "req-2"))[0] == "generated"


def test_the_lora_rollout_guard_does_not_wrap_itself_twice(monkeypatch):
    """every ray actor imports the same sitecustomize; stacking wrappers would add a
    ``list_loras`` round trip per layer to every request."""
    import asyncio

    module = _lora_rollout_server_module({123})
    _apply_lora_rollout_guard(module, monkeypatch)
    first = module.vLLMHttpServer.generate
    _apply_lora_rollout_guard(module, monkeypatch)

    assert module.vLLMHttpServer.generate is first
    assert asyncio.run(module.vLLMHttpServer().generate([1], {}, "req-3"))[0] == "generated"


@pytest.mark.parametrize(
    ("rollout_batch", "expected"),
    [(1, 16), (4, 16), (16, 16), (32, 32), (64, 64), (512, 512)],
)
def test_rollout_max_num_seqs_tracks_the_rollout_batch_above_the_floor(rollout_batch, expected):
    from flash.engine.worker.verl.capabilities import rollout_max_num_seqs

    assert rollout_max_num_seqs(rollout_batch) == expected


def test_rollout_max_num_seqs_never_returns_verls_default_for_a_small_batch():
    # the regression this exists for: verl leaves rollout.max_num_seqs at 1024, and vllm sizes
    # cuda-graph capture (min(max_num_seqs * 2, 512) over a 51-rung ladder, doubled when lora
    # specialization is on) and per-slot gdn/mamba recurrent state from it, both before the first
    # rollout token. a 32-sequence grpo step paid a 1024-sequence reservation and OOMed during
    # capture with ~80% of the card free.
    from flash.engine.worker.verl.capabilities import rollout_max_num_seqs

    for batch in (8, 32, 64, 128):
        assert rollout_max_num_seqs(batch) < 1024


def test_rollout_max_num_seqs_is_never_below_the_batch_it_must_serve():
    # provisioning FEWER slots than the step submits would queue sequences instead of running them,
    # turning one rollout into several scheduler passes.
    from flash.engine.worker.verl.capabilities import rollout_max_num_seqs

    for batch in range(1, 600):
        assert rollout_max_num_seqs(batch) >= batch


def test_rollout_layered_summon_is_on_for_fused_expert_targets():
    # the regression this exists for: verl's collect_lora_params default branch opens
    # FSDP.summon_full_params over the WHOLE model, so every layer's fused expert stack is
    # unsharded at once. on Qwen3.6-35B-A3B one layer's mlp.experts.gate_up_proj is
    # [128, 2048, 2048] bf16 = exactly 1024 MiB, which is the contiguous allocation that OOMed at
    # step 0 on a 180 GB B200 across every card count and context tested.
    from flash.engine.worker.verl.capabilities import rollout_layered_summon_overrides

    assert rollout_layered_summon_overrides(
        ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
    ) == ["actor_rollout_ref.rollout.layered_summon=true"]


@pytest.mark.parametrize("empty", [None, [], (), ""])
def test_rollout_layered_summon_stays_off_for_dense_models(empty):
    # the layered walk summons one submodule at a time and empty_cache()s between, which is slower.
    # a dense model's per-tensor gathers are small, so it keeps verl's default.
    from flash.engine.worker.verl.capabilities import rollout_layered_summon_overrides

    assert rollout_layered_summon_overrides(empty) == []


def test_rollout_layered_summon_uses_a_bare_override_not_an_append():
    # layered_summon is declared in verl's rollout.yaml (rollout.yaml:98) AND the generated trainer
    # config, so hydra already has the key. a '+' prefix -- required for dataclass-only fields like
    # enable_sleep_mode -- would be an append to an existing key and fail the config merge.
    from flash.engine.worker.verl.capabilities import rollout_layered_summon_overrides

    (override,) = rollout_layered_summon_overrides(["mlp.experts.gate_up_proj"])
    assert not override.startswith("+")
    assert override.startswith("actor_rollout_ref.rollout.layered_summon=")


@pytest.mark.parametrize(
    ("fsdp_generation", "expected"),
    [(1, "fsdp"), (2, "fsdp2")],
)
def test_actor_fsdp_strategy_renders_the_resolved_generation(fsdp_generation, expected):
    from flash.engine.worker.verl.capabilities import actor_fsdp_strategy_overrides

    assert actor_fsdp_strategy_overrides(fsdp_generation) == [
        f"actor_rollout_ref.actor.strategy={expected}"
    ]


@pytest.mark.parametrize(
    ("algorithm", "target_parameters", "expected"),
    [
        ("sft", None, 2),
        ("sft", ["mlp.experts.gate_up_proj"], 2),
        ("grpo", None, 2),
        ("grpo", ["mlp.experts.gate_up_proj"], 2),
        ("opd", None, 1),
        ("opd", ["mlp.experts.gate_up_proj"], 2),
    ],
)
def test_private_fsdp_policy_matrix(algorithm, target_parameters, expected):
    from flash.engine.support.verl_policy import _resolve_fsdp_generation

    assert _resolve_fsdp_generation(algorithm, target_parameters) == expected


def test_actor_fsdp_strategy_uses_a_bare_override_not_an_append():
    # `strategy` is declared in verl's actor/dp_actor.yaml:26, so hydra already has the key. a '+'
    # prefix would be an append to an existing key and fail the config merge.
    from flash.engine.worker.verl.capabilities import actor_fsdp_strategy_overrides

    (override,) = actor_fsdp_strategy_overrides(2)
    assert not override.startswith("+")
    assert override.startswith("actor_rollout_ref.actor.strategy=")


def test_rollout_max_num_seqs_rejects_a_nonpositive_batch():
    from flash.engine.worker.verl.capabilities import rollout_max_num_seqs

    with pytest.raises(ValueError, match="rollout_batch must be positive"):
        rollout_max_num_seqs(0)
