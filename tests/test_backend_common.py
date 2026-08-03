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
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest import mock

import pytest

from flash.engine.worker import backend_common as vc
from flash.engine.worker import rl_train

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
    with (
        mock.patch.object(vc, "_cgroup_cpu_quota", return_value=12),
        mock.patch.object(os, "sched_getaffinity", return_value=set(range(48))),
    ):
        assert vc.ray_num_cpus() == 12
    # a quota BELOW verl's placement-group demand is the one case the quota must not win outright:
    # honouring it exactly would schedule nothing and hang. see the floor test below.
    with (
        mock.patch.object(vc, "_cgroup_cpu_quota", return_value=4),
        mock.patch.object(os, "sched_getaffinity", return_value=set(range(48))),
    ):
        assert vc.ray_num_cpus() == vc.verl_cpu_demand(1)


def test_ray_num_cpus_falls_back_to_affinity_when_no_quota_is_set():
    with (
        mock.patch.object(vc, "_cgroup_cpu_quota", return_value=None),
        mock.patch.object(os, "sched_getaffinity", return_value=set(range(6))),
    ):
        assert vc.ray_num_cpus() == 6


def test_ray_num_cpus_caps_an_unconstrained_host():
    # no quota and a full 48-core affinity mask means flash cannot tell the pod from the host, which
    # is exactly the case that forked 48 idle workers. the cap is what makes that survivable.
    with (
        mock.patch.object(vc, "_cgroup_cpu_quota", return_value=None),
        mock.patch.object(os, "sched_getaffinity", return_value=set(range(48))),
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
        mock.patch.object(os, "sched_getaffinity", side_effect=OSError("boom")),
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
    # the cap must not create the OPPOSITE failure. verl blocks in ray.get(pg.ready()) with no
    # timeout, so a cluster with fewer cpus than the placement group wants hangs forever before the
    # gpu is touched -- the same trap the SimpleStorage override upstream of this guards against.
    # demand on a 1-gpu pod, read off the verl source rather than guessed: 3 for the worker bundle
    # (RayClassWithInitArgs max_colocate_count=3, seen as CPU_group_0: [30000] in the raylet dump),
    # 1 for TaskRunner (main_ppo.py ray.remote(num_cpus=1)), 1 for opd's single storage unit.
    verl_peak_cpu_demand = 3 + 1 + 1
    assert vc.verl_cpu_demand(1) == verl_peak_cpu_demand
    with (
        mock.patch.object(vc, "_cgroup_cpu_quota", return_value=None),
        mock.patch.object(os, "sched_getaffinity", return_value=set(range(48))),
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
        mock.patch.object(os, "sched_getaffinity", return_value=set(range(48))),
    ):
        for gpus in range(1, 9):
            assert vc.ray_num_cpus(gpus) >= vc.verl_cpu_demand(gpus), gpus


def test_both_trainers_size_the_cpu_pool_from_their_own_gpu_count():
    # a floor that the call sites never pass a gpu count to is a floor that only ever protects
    # 1-gpu jobs, which is the shape of the bug this fixes. pin the wiring, not just the helper.
    for module, expected in (
        ("rl_train", "ray_num_cpus(cfg['n_gpus'])"),
        ("opd_train", "ray_num_cpus(config['n_gpus_per_node'])"),
    ):
        src = pathlib.Path(_REPO_ROOT, "flash", "engine", "worker", f"{module}.py").read_text()
        assert expected in src, module


def test_every_ray_backed_trainer_constrains_rays_cpu_pool():
    # asserted across BOTH ray entrypoints rather than in one file: ray autodetects the host's cores
    # and eagerly forks one idle worker per core, which killed grpo (fatal raylet fork failure) and
    # opd (host-ram oom) on real gpus. sft is excluded on purpose -- it runs torchrun, not ray.
    for module in ("rl_train", "opd_train"):
        src = pathlib.Path(_REPO_ROOT, "flash", "engine", "worker", f"{module}.py").read_text()
        assert "ray_kwargs.ray_init.num_cpus={ray_num_cpus(" in src, module


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
    # no WANDB_API_KEY -> console only, and no wandb probe of the verl interpreter.
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(
        vc.subprocess,
        "run",
        lambda *a, **k: pytest.fail("must not probe verl env without an api key"),
    )
    assert vc.resolve_verl_loggers("/verl/bin/python") == ["console"]


def test_resolve_verl_loggers_enables_wandb_only_when_verl_env_has_it(monkeypatch):
    # api key set AND wandb importable in the verl interpreter -> wandb logger enabled.
    monkeypatch.setenv("WANDB_API_KEY", "k")
    monkeypatch.setattr(vc.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    assert vc.resolve_verl_loggers("/verl/bin/python") == ["console", "wandb"]


def test_resolve_verl_loggers_falls_back_to_console_when_verl_env_lacks_wandb(monkeypatch):
    # api key set but wandb missing in the verl interpreter -> console only (never aborts verl).
    monkeypatch.setenv("WANDB_API_KEY", "k")
    monkeypatch.setattr(vc.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    assert vc.resolve_verl_loggers("/verl/bin/python") == ["console"]


def test_stamp_adapter_dir_provenance_sets_base_and_revision(tmp_path):
    cfg = tmp_path / "adapter_config.json"
    cfg.write_text(json.dumps({"base_model_name_or_path": None, "r": 16}))
    vc.stamp_adapter_dir_provenance(str(tmp_path), "org/model", "deadbeef")
    out = json.loads(cfg.read_text())
    assert out["base_model_name_or_path"] == "org/model"
    assert out["revision"] == "deadbeef"
    # untouched fields survive the stamp.
    assert out["r"] == 16


def test_stamp_adapter_dir_provenance_rejects_base_mismatch(tmp_path):
    cfg = tmp_path / "adapter_config.json"
    cfg.write_text(json.dumps({"base_model_name_or_path": "org/other"}))
    with pytest.raises(RuntimeError, match="does not match validated target"):
        vc.stamp_adapter_dir_provenance(str(tmp_path), "org/model")


def test_resolve_verl_python_prefers_preset(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASH_VERL_PYTHON", "/opt/verl/bin/python")
    assert vc.resolve_verl_python(str(tmp_path)) == "/opt/verl/bin/python"


@pytest.mark.parametrize("blank", ["", "   "])
def test_resolve_verl_python_treats_an_empty_preset_as_unset(monkeypatch, tmp_path, blank):
    # a worker IMAGE can export FLASH_VERL_PYTHON itself, and [worker_env] can only SET a key, never
    # delete one -- so omitting it from a spec leaves the image's interpreter in place. an empty
    # value is the only way a run can say "ignore the image's verl and provision the pinned fork",
    # and the error at rl_train.py's mask_truncated_completions gate names exactly this remedy.
    calls = []
    monkeypatch.setenv("FLASH_VERL_PYTHON", blank)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))

    python_bin = vc.resolve_verl_python(str(tmp_path))

    assert python_bin.endswith("/verl-venv/bin/python")
    assert any(vc.VERL_REQUIREMENT_URL in arg for arg in calls[1])


def test_worker_env_remedies_are_copy_pasteable_toml():
    # a [worker_env] snippet in an error gets pasted into a config verbatim, so it has to survive the
    # real parser: '[worker_env] KEY = "..."' reads fine in prose but is invalid TOML, because a
    # table header must end its line. a blocked run would just hit a second, more confusing error.
    import re
    import tomllib

    from flash.engine.worker import rl_train

    # only assignment forms -- '[worker_env] can set a key but never delete one' is prose, not a
    # snippet, and carries no '=' to paste.
    pattern = re.compile(r"\[worker_env\][^\n]*?[A-Z_]+\s*=\s*(\"[^\"]*\"|'[^']*')")
    snippets = [m.group(0) for m in pattern.finditer(inspect.getsource(rl_train))]
    assert snippets, "expected rl_train to advertise at least one [worker_env] remedy"

    for snippet in snippets:
        # a valid snippet is the header, a newline, then the assignment -- exactly what we tell users.
        header, _, assignment = snippet.partition("]")
        parsed = tomllib.loads(f"{header}]\n{assignment.split('as ')[-1].strip()}")
        assert parsed == {"worker_env": {"FLASH_VERL_PYTHON": ""}}
        # and the prose must not run the header into the assignment on one line.
        assert not re.match(r"\[worker_env\]\s+[A-Z_]+\s*=", snippet), snippet


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

    def fake_run(command, check):
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
    # cannot enter -- killing the run during provisioning (codex[bot]).
    assert calls[0][:4] == ["uv", "venv", "--python", "3.12"]
    assert "cp312" in vc.FLASH_ATTN_SPEC
    install = calls[1]
    assert vc.VERL_REQUIREMENT == (
        "verl @ git+https://github.com/freesolo-co/verl@723336cf5d2aa084a9308db1f774df2f84ad9f51"
    )
    assert any(vc.VERL_REQUIREMENT_URL in arg for arg in install)
    assert "liger-kernel" in install
    assert "bitsandbytes>=0.49" in install
    assert "qwen-vl-utils" in install
    assert "torchvision" in install
    assert "xgrammar==0.1.25" in install
    assert "tqdm" in install
    assert "pyarrow" in install
    # venv, the resolve above, then the prebuilt flash_attn wheel on its own --no-build-isolation
    # line. nothing else: a fourth call would be an unbudgeted install on a paid pod.
    assert len(calls) == 3
    assert calls[2][:3] == ["uv", "pip", "install"]
    # the stamp is written only after a successful install, so a crashed install is never reused.
    stamp = tmp_path / "verl-venv" / "flash-verl-requirement"
    assert stamp.read_text() == vc.VERL_VENV_STAMP


def _flaky_wheel_install(calls, sleeps, *, failures: int):
    """subprocess.run stand-in whose flash-attn install fails the first `failures` attempts."""
    attempts = {"n": 0}

    def fake_run(command, check):
        calls.append(command)
        if command[:2] == ["uv", "venv"]:
            os.makedirs(os.path.join(command[-1], "bin"), exist_ok=True)
            return
        if "--no-build-isolation" not in command:
            return
        attempts["n"] += 1
        if attempts["n"] <= failures:
            # what uv actually exits with on `error sending request ... operation timed out`.
            raise subprocess.CalledProcessError(1, command)

    return fake_run, attempts


def test_a_transient_wheel_download_failure_is_retried_rather_than_fatal(monkeypatch, tmp_path):
    """uv exits 1 on a download timeout, and one attempt is not evidence the wheel is unreachable.

    Measured: an arm died on `error sending request ... operation timed out` for FLASH_ATTN_SPEC,
    and the identical url served 200 moments later. uv retries the DOWNLOAD 3x internally, so an
    exit here means the fault outlived one uv lifetime -- a fresh invocation is the only retry that
    can outlast it.
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

    Asserting the exception type alone would pass while the arm still died: what actually burns the
    arm is `retriable: false` in the heartbeat, because _worker_failure_flags treats anything that
    is not RetriableInfraError/GitHubRateLimitError as permanent and the poller then routes it to
    job_failed rather than job_preempted. So assert the flag the plane reads, not the type we raise.
    """
    from flash.engine.worker import _worker_failure_flags
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

    A bare `vllm==0.19.1` on the command line is a CONSTRAINT, not an override: the resolver must
    still satisfy verl[vllm]'s declared `vllm<=0.12.0` alongside it, so the pair is unsatisfiable
    and the install fails outright rather than picking the pin. Only `--override` makes uv ignore
    the declaration. Dockerfile.worker:253-258 records the same three violations and the same fix
    (codex[bot]).
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
    for module in ("cachetools", "uvicorn", "fastapi"):
        assert module in install, f"verl imports {module} at module level but never declares it"
    # opd's entrypoint calls tq.init()/tq.close() and verl's setup.py omits TransferQueue.
    assert "TransferQueue==0.1.7" in install


def test_provisioned_venv_gets_flash_attn_for_the_remove_padding_path(monkeypatch, tmp_path):
    # all three backends hard-enable remove-padding (sft_train.py:164, rl_train.py:339,
    # opd_train.py:1505) and verl's cuda remove-padding path imports flash_attn.bert_padding
    # UNGUARDED (verl/utils/attention_utils.py:30, torch_functional.py:627). there is no sdpa
    # fallback on that path, so a venv without the wheel dies at the first training batch on a paid
    # gpu. Dockerfile.worker:288-297 already treats it as REQUIRED in /opt/verl-venv; this fallback
    # provisions the same interpreter and must carry the same wheel.
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))

    vc.resolve_verl_python(str(tmp_path))

    flat = [arg for command in calls for arg in command]
    assert vc.FLASH_ATTN_SPEC in flat, (
        "verl's remove-padding path imports flash_attn unguarded; the provisioned venv must hold it"
    )
    # the wheel is prebuilt, so it must skip build isolation exactly as the image install does --
    # a source build here would compile for minutes on a paid pod.
    install = next(c for c in calls if vc.FLASH_ATTN_SPEC in c)
    assert "--no-build-isolation" in install


def test_flash_attn_spec_stays_in_lockstep_with_the_worker_image():
    # the fallback venv and /opt/verl-venv must resolve the same wheel: a run that lands on the
    # no-image path otherwise trains against a different flash_attn than every baked-image run.
    dockerfile = pathlib.Path(__file__).resolve().parents[1] / "Dockerfile.worker"
    assert f"ARG FLASH_ATTN_SPEC={vc.FLASH_ATTN_SPEC}" in dockerfile.read_text(), (
        "Dockerfile.worker's FLASH_ATTN_SPEC default drifted from backend_common.FLASH_ATTN_SPEC"
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

    The stamp is the whole reuse gate, and flash-attn is installed separately from verl. A stamp
    naming only the verl pin is byte-identical before and after this fix, so a retry landing on a
    pod whose venv predates it would skip the install entirely and hand back an interpreter with no
    flash-attn -- the exact failure this path exists to prevent (codex[bot]).
    """
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))
    # exactly what the pre-fix release wrote: the bare verl pin, nothing about the wheel.
    _fake_verl_venv(tmp_path, stamp=vc.VERL_REQUIREMENT)

    vc.resolve_verl_python(str(tmp_path))

    assert calls, "a venv stamped before flash-attn was installed must be rebuilt"
    assert any(vc.FLASH_ATTN_SPEC in call for call in calls)


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

    assert [c[:2] for c in calls] == [["uv", "venv"], ["uv", "pip"], ["uv", "pip"]]
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
    assert [c[:2] for c in calls] == [["uv", "venv"], ["uv", "pip"], ["uv", "pip"]]


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
    from flash.engine.worker import opd_plugin as plugin

    assert plugin._STRUCTURED_RUNTIME_EXACT_VERSIONS["verl"] == "0.8.0"
    # asserting the constant alone would let a newer-base commit land silently, so bind the pinned
    # commit itself to that version. this is the sha of the truncation-mask and 3d position-id
    # commits cherry-picked onto the v0.8.0 tag, plus the agent-loop position-id pad and fused-label
    # shape fixes. moving the pin must be deliberate, with the base re-verified. at 723336cf,
    # verl/version/version still reads 0.8.0 and verl/trainer/main_ppo_sync.py is still present.
    _, _, ref = vc.VERL_REQUIREMENT.partition("git+")
    _, _, commit = ref.rpartition("@")
    assert commit == "723336cf5d2aa084a9308db1f774df2f84ad9f51"


def test_resolve_verl_python_installs_wandb_best_effort_when_requested(monkeypatch, tmp_path):
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls, keep_check=True))

    vc.resolve_verl_python(str(tmp_path), install_wandb=True)

    assert any(vc.VERL_REQUIREMENT_URL in arg for arg in calls[1][0])
    # wandb is the LAST call: it follows the flash_attn wheel install, and unlike every install
    # before it, it is best-effort (check=False) so a wandb outage cannot fail a training run.
    assert calls[3] == (
        ["uv", "pip", "install", "--python", str(tmp_path / "verl-venv/bin/python"), "wandb"],
        False,
    )


def _probe_interpreter(tmp_path, name, body):
    """write a stub interpreter that answers the RolloutConfig probe like a real python would."""
    stub = tmp_path / name
    stub.write_text("#!/bin/sh\n" + body + "\n")
    stub.chmod(0o755)
    return str(stub)


def test_verl_supports_rollout_field_true_when_field_declared(tmp_path):
    fork = _probe_interpreter(tmp_path, "fork-python", "echo 1")
    assert vc.verl_supports_rollout_field(fork, "mask_truncated_completions") is True


def test_verl_supports_rollout_field_false_when_field_absent(tmp_path):
    # stock verl 0.8.0: RolloutConfig has no such field, so the probe prints 0.
    stock = _probe_interpreter(tmp_path, "stock-python", "echo 0")
    assert vc.verl_supports_rollout_field(stock, "mask_truncated_completions") is False


def test_verl_supports_rollout_field_false_when_verl_missing(tmp_path):
    # import error inside the probe: nonzero exit must read as unsupported, not crash the caller.
    broken = _probe_interpreter(tmp_path, "broken-python", "exit 1")
    assert vc.verl_supports_rollout_field(broken, "mask_truncated_completions") is False


def test_verl_supports_rollout_field_false_when_interpreter_missing(tmp_path):
    # a bogus FLASH_VERL_PYTHON must not raise OSError out of the capability check.
    missing = str(tmp_path / "does-not-exist")
    assert vc.verl_supports_rollout_field(missing, "mask_truncated_completions") is False


def test_resolve_verl_python_returns_preset_unmodified(monkeypatch, tmp_path):
    # flash does not own a preset interpreter and must never mutate it; capability is checked
    # separately by verl_supports_rollout_field.
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


# ---------------------- child teardown escalates to SIGKILL ----------------------
# these tests drive real processes through the real escalation, which needs three kernel features
# this repo only ever runs teardown on: fork, /proc for group membership, and libc for the subreaper
# probe. the rest of the suite is os-independent, so guard rather than fail on a platform that
# cannot answer (codex[bot]). checked at runtime like the /proc/self/maps test in test_worker_stack,
# instead of on the platform name, so the condition is the capability actually required.
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

    pid 1 in `Dockerfile.worker` adopts them for free; a shared pytest process does not, so these
    tests ask for the same behaviour explicitly. Leaving it set would change subprocess semantics for
    every LATER test in the process -- they would adopt orphans they never reap, accumulating
    zombies and making failures order-dependent (codex[bot]).
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

    every child below refuses the term on purpose, so each one waits out the full production grace
    before escalating -- 30s of a 30.6s file for three tests. the constant is read from the module
    at call time, so patching it needs no production seam (codex[bot]).

    it is only ever shortened. what these tests assert is that SIGKILL is what performs the
    termination and that nothing is left unreaped, and neither depends on how long the grace was.
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
            "import signal, sys, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "print('ready', flush=True)\n"
            "time.sleep(300)\n",
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


@_needs_process_teardown
def test_a_group_whose_only_member_is_a_zombie_is_not_read_as_alive():
    # Dockerfile.worker runs python directly as pid 1 with no init, so an orphaned EngineCore is
    # never reaped and sits as a zombie indefinitely. killpg(pgid, 0) succeeds for that group, so
    # driving the escalation off addressability alone burned the whole drain deadline on every
    # teardown even though the cuda context was already released -- and sigkill cannot clear a
    # zombie, only reaping can (codex[bot]). fork rather than Popen: subprocess reaps for you, which
    # is the very behaviour the worker image lacks.
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
    # a resource-failure path (codex[bot]). only disappearance proves an exit.
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
    # group and never received the earlier signal -- kept its cuda context (codex[bot]).
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
    verdict taken from it skips SIGKILL and leaves the gpu held (codex[bot])."""
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
    """Both halves in one test rather than across two: a later test asserting the flag is back to 0
    also passes when it happens to run FIRST, so it would prove nothing under random ordering.

    The restore is exercised from a 0 setting AND from a 1 setting. Only the first is what the suite
    actually runs, but a `finally` that never fires is indistinguishable from a correct one there --
    both end at 1 while the block is open and 0 is where the process already was. Entering with the
    flag set is the case that separates them.
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

    Any test reaching `run_verl_training` flips this shared pytest process from 0 to 1 for the rest
    of the session, after which every later test adopts orphaned grandchildren it never waits on --
    accumulating zombies and making results order-dependent. The `subreaper` fixture restores only
    the tests that ask for it, which is not where this flag is now set (codex[bot]).

    The autouse conftest fixture is what restores it. This asserts the guarantee the fixture makes,
    from a 0 setting -- what the suite actually runs -- and would fail on a plain `finally`-less
    entry point exactly as the leak does.
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

    Run as a subprocess so a real pytest teardown happens. Asserting from a later test in this
    process would pass under random ordering whenever it happened to run first.

    The probe is written UNDER ``tests/`` rather than to a scratch tempdir, because a conftest
    governs its own directory and below: from anywhere else `tests/conftest.py` is never loaded,
    the autouse fixture under test never runs, and the probe reports the leak whether or not the
    fixture works. Placed in a subdirectory of its own so it is never collected by the outer run.
    """
    probe = (
        "import ctypes\n"
        "libc = ctypes.CDLL('libc.so.6', use_errno=True)\n"
        "def flag():\n"
        "    cur = ctypes.c_int(0)\n"
        "    libc.prctl(37, ctypes.byref(cur), 0, 0, 0)\n"
        "    return cur.value\n"
        "import os\n"
        "from flash.engine.worker import backend_common as vc\n"
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
    # wait can -- so leaving it costs one permanent process-table entry per failed or cancelled run
    # (codex[bot]). the `subreaper` fixture reproduces that adoption inside this test process.
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
    # and the entry is permanent on a pid-1 worker (codex[bot]).
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
    # the future wait a straggler needs, holding that zombie for the process's whole life -- as pid 1
    # there is nothing else to reap it. the drain has to happen at every job boundary (codex[bot]).
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
# and then EXITS, with no second job to sweep it. `flash/providers/runpod/train/endpoints.py:538-552`
# spawns one of these per phase and waits for it, so anything the phase leaves behind reparents to
# the persistent handler -- which waits only on the worker -- and stays a zombie for the container's
# life. driven as a subprocess because the leak is about process EXIT, which pytest cannot perform.
_SHORT_LIVED_WORKER = r"""
import os, sys, time
sys.path.insert(0, {repo!r})
from flash.engine.worker import backend_common as vc

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

    Remembering a pid arranges a future wait, and on a long-lived process the next teardown is that
    wait. The runpod worker is not long-lived -- one process per phase -- so the set dies with the
    phase and the straggler reparents to the persistent handler as a permanent zombie (codex[bot]).
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
    source = inspect.getsource(rl_train)
    assert "kill_process_group(proc)" in source
    assert "os.killpg" not in source, "grpo teardown must not hand-roll a non-escalating killpg"


def test_every_test_touching_a_linux_only_api_carries_the_platform_guard():
    """The guard was applied test-by-test and kept being missed on the next one added.

    A skipif on some of them still fails the run on a platform without `fork` or `/proc`, so this
    reads THIS file and requires the marker wherever such an api appears -- the whole block, not a
    remembered list (codex[bot]). Source inspection is exempt: asserting that a string is absent
    from `rl_train` runs anywhere.
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
from flash.engine.worker.backend_common import _reap, adopt_orphaned_descendants

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
# reparent it to whatever runs pytest and leak one process-table slot per run of that test
# (codex[bot]). the state above was already recorded, so collecting it now costs the test nothing.
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

    `Dockerfile.worker` runs `/rp_handler.py` as pid 1, and that handler spawns the flash worker as
    a subprocess (`flash/providers/runpod/train/endpoints.py:539`). So an EngineCore orphaned when
    the trainer exits reparents past the worker to the HANDLER, `waitpid` raises `ChildProcessError`
    here, and `_reap` records the zombie as handled while it keeps its pid for the worker's whole
    life -- the handler waits only on the worker, so nothing else collects it either (codex[bot]).

    Driven in a subprocess because the pytest process cannot show this: the `subreaper` fixture that
    the teardown tests use manufactures the adoption production does not have, so the defect is
    invisible from inside it.
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

    Asserting the defect is REACHABLE keeps the test above honest. If this ever reports the orphan
    gone, the probe is being reaped by something other than the code under test and the pass above
    means nothing.
    """
    claimed, state, engine, collected = _run_topology_probe(claim=False)
    assert state == "Z", (
        "the orphan was collected without the claim, so the test above proves nothing"
    )
    assert claimed == "True", (
        "`_reap` reported the zombie as handled, which is what makes this silent"
    )
    # the zombie this control deliberately produces is the probe's to clear before it exits. it is
    # a subreaper, so the entry is its own; exiting would hand it to whatever runs pytest and leak
    # one process-table slot per run (codex[bot]). asserted on what the probe COLLECTED rather than
    # on the pid disappearing: under systemd the orphan is reaped either way, so the disappearance
    # is not evidence and a test built on it could never fail here.
    assert engine in collected, (
        f"the probe exited without reaping orphan {engine}, leaking a process-table entry to "
        "whatever inherits it -- invisible under an init that reaps, permanent under one that does not"
    )


@_needs_process_teardown
def test_the_claim_is_made_before_the_trainer_is_spawned():
    """Order matters: the kernel only reparents to a process that was ALREADY a subreaper.

    Claiming after the child exists leaves any grandchild it has already orphaned parented
    elsewhere, so the fix would work only for the second job onward on a reused worker.
    """
    for fn in (vc.run_verl_training, rl_train.run_rl_train):
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
    ordinary exit a worker whose later jobs all SUCCEED keeps a straggler zombie for life (cursor).
    """
    src = " ".join(inspect.getsource(rl_train.run_rl_train).split())
    finally_block = src[src.rindex("finally:") :]
    assert "reap_stragglers()" in finally_block, (
        "the grpo teardown collects stragglers only when the run FAILS"
    )


def _flashinfer_probe(monkeypatch, *, flashinfer_ok=True):
    """stub the one remaining subprocess probe: `import flashinfer`."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0 if flashinfer_ok else 1)

    monkeypatch.setattr(vc.subprocess, "run", fake_run)
    return calls


def _cc_probe(monkeypatch, cc):
    """stub the capability probe; `cc` is what torch reports, or () for no cuda."""

    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=0, stdout=f"{cc}\n", stderr="")

    monkeypatch.setattr(vc.subprocess, "run", fake_run)


def test_the_capability_probe_reads_what_torch_reports(monkeypatch):
    _cc_probe(monkeypatch, (8, 9))
    assert vc.resolve_verl_device_capability("/verl/bin/python") == (8, 9)


def test_the_capability_probe_reports_no_cuda_as_none(monkeypatch):
    # torch prints () with no visible card; literal_eval yields an empty tuple, and indexing it
    # raises -- which must surface as None, not as a crash on the launch path.
    _cc_probe(monkeypatch, ())
    assert vc.resolve_verl_device_capability("/verl/bin/python") is None


def test_the_capability_probe_reports_failure_as_none(monkeypatch):
    def boom(*a, **k):
        raise OSError("no interpreter")

    monkeypatch.setattr(vc.subprocess, "run", boom)
    assert vc.resolve_verl_device_capability("/verl/bin/python") is None


def test_the_capability_probe_runs_against_the_verl_interpreter(monkeypatch):
    # verl owns the rollout engine and pins its own torch/vllm, so flash's own torch would answer
    # for the wrong environment -- and on a heterogeneous host, potentially the wrong card.
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="(8, 9)\n", stderr="")

    monkeypatch.setattr(vc.subprocess, "run", fake_run)
    vc.resolve_verl_device_capability("/verl/bin/python")
    assert calls
    assert all(cmd[0] == "/verl/bin/python" for cmd in calls)


def test_the_capability_is_probed_once_for_both_rollout_decisions(monkeypatch):
    # both decisions below are functions OF the capability, not probes of it. asking the verl
    # interpreter twice would spawn a second torch import on every launch to re-learn a constant.
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="(12, 0)\n", stderr="")

    monkeypatch.setattr(vc.subprocess, "run", fake_run)
    cc = vc.resolve_verl_device_capability("/verl/bin/python")
    assert len(calls) == 1
    # neither decision may go back to the interpreter for the capability it was handed.
    vc.resolve_rollout_enforce_eager(cc)
    assert len(calls) == 1


@pytest.mark.parametrize("cc", [(8, 6), (8, 9), (7, 5), (11, 0)])
def test_unvalidated_arches_force_the_rollout_eager(cc):
    # vllm 0.19.1's graph capture dies in aot_compile (sm86) / triton slot-mapping on the arches the
    # retired trl driver never validated. verl defaults enforce_eager False AND asks for
    # FULL_AND_PIECEWISE, so without this an sm89 rtx 4090 -- the catalog's recommended_gpu for the
    # small models, i.e. the DEFAULT grpo route -- captures more graphs than the config known to fail.
    assert vc.resolve_rollout_enforce_eager(cc) is True


@pytest.mark.parametrize("cc", [(8, 0), (9, 0), (10, 0), (12, 0)])
def test_validated_arches_keep_verl_graph_capture(cc):
    # a100/h100 were validated with graphs, and blackwell (incl. the b200 rollout work) depends on
    # them. forcing eager here would be a silent throughput regression, not a safety net.
    assert vc.resolve_rollout_enforce_eager(cc) is False


def test_an_unknown_capability_leaves_verl_graph_capture_alone():
    # a probe that could not answer must not guess eager onto an unknown card.
    assert vc.resolve_rollout_enforce_eager(None) is False


@pytest.mark.parametrize("major", [10, 12])
def test_blackwell_pins_flashinfer_and_sdpa_vit(monkeypatch, major):
    _flashinfer_probe(monkeypatch)
    assert vc.resolve_blackwell_attention_backends("/verl/bin/python", (major, 0)) == (
        "FLASHINFER",
        "TORCH_SDPA",
    )


def test_blackwell_falls_back_to_triton_when_flashinfer_is_abi_broken(monkeypatch):
    # flashinfer can install yet fail to import against this torch. an unconditional FLASHINFER would
    # ship fine and only die at engine init on a paid gpu, so degrade to a registered PTX-independent
    # decoder backend. the ViT pin is unaffected -- it is a separate selection.
    _flashinfer_probe(monkeypatch, flashinfer_ok=False)
    assert vc.resolve_blackwell_attention_backends("/verl/bin/python", (12, 0)) == (
        "TRITON_ATTN",
        "TORCH_SDPA",
    )


@pytest.mark.parametrize("cc", [(8, 0), (8, 9), (9, 0), None])
def test_non_blackwell_leaves_both_backends_to_vllm(monkeypatch, cc):
    # vllm's capability-ordered defaults are correct off blackwell (flash-attn is the right decoder
    # choice on ampere/hopper), so pinning anything there would override a working selection. an
    # unknown capability is treated the same way: leave the defaults in place.
    calls = _flashinfer_probe(monkeypatch)
    assert vc.resolve_blackwell_attention_backends("/verl/bin/python", cc) == (None, None)
    # and off blackwell the flashinfer probe must not run at all -- nothing consumes its answer.
    assert calls == []


def test_the_flashinfer_probe_runs_against_the_verl_interpreter(monkeypatch):
    # verl owns the rollout engine and pins its own vllm stack, so a flash-side `import flashinfer`
    # would answer for the wrong environment.
    calls = _flashinfer_probe(monkeypatch)
    vc.resolve_blackwell_attention_backends("/verl/bin/python", (12, 0))
    assert len(calls) == 1
    assert calls[0][0] == "/verl/bin/python"


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

    /tmp survives a retry on a reused pod, so a failure during dependency provisioning or model
    download still finds an earlier run's session under /tmp/ray. Uploading it under this attempt's
    artifact prefix is worse than uploading nothing: it reads as a raylet death that never happened
    and sends the next diagnosis after a cause belonging to a different run (codex[bot], cursor).
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

    sanitize_diagnostic matches a secret by its ``key=`` prefix or by full value. A tail boundary
    landing inside a token strips the prefix and the value's head, so no pattern matches and the
    remainder uploads in clear. Dropping the partial first line makes that impossible, because a
    secret cannot span a newline (codex[bot]).
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

    ``environment.secrets`` values are arbitrary strings, so a PEM key or JSON credential is legal.
    When the tail begins inside such a value's first line, that line is dropped -- but every
    LATER line of the same secret is whole, lands after the cut, and no longer matches the
    whole-value replace, so it uploads verbatim. Redaction must therefore know a secret's
    individual lines (codex[bot]).
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

    A JSON credential ends in lines like ``}`` that every innocent log also contains. Registering
    those as needles would redact unrelated text and destroy the evidence this collector exists to
    produce, so short components are excluded.
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

        Growing at open (not at seek) is what makes this able to fail: a size-then-read
        implementation calls os.path.getsize BEFORE opening, so a fixture that grows any later
        would hand it a stale-but-consistent size and the unbounded read would never show.
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

    returns ``(peak_threads_over_baseline, errors)`` measured while the handlers are held open.
    holding them is what makes the measurement meaningful: it forces requests to be resident
    simultaneously, which is the rollout shape that exhausted the thread table.

    ``await_arrivals`` is how many requests must be inside a handler before the peak is sampled.
    it cannot be ``callers`` for a bounded server -- only pool-size handlers can ever be resident
    at once, so waiting for all 64 would hang. that ceiling IS the fix, so the caller passes what
    it expects to be able to observe and the difference between the arms is the evidence.
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

    the shipped bridges used ThreadingHTTPServer, which starts a thread per request and never
    bounds them. on a real rollout (agent.num_workers x rollout.n callers, next to ray/vllm/libgomp
    pools) that hit the container's thread limit: socketserver.process_request raised
    "can't start new thread", the connection died mid-request, and verl saw RemoteDisconnected.
    driving the real server class is the point -- an assertion against a hand-rolled copy of the
    pool would pass no matter what the worker shipped.
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
    from flash.engine.worker import opd_train

    # the teacher bridge is a module-level class, so check the type itself.
    assert issubclass(opd_train._TeacherBridgeHTTPServer, vc.BoundedThreadingHTTPServer), (
        "the opd teacher bridge does not use BoundedThreadingHTTPServer"
    )
    # the reward bridge is defined inside the function that starts it, so it is only reachable
    # through the source. parse rather than substring-match: a comment mentioning the old name
    # must not pass, and a real subclass must not be missed.
    for mod in (rl_train, opd_train):
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
