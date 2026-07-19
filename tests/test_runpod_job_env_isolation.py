from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import types
import zipfile
from pathlib import Path

import pytest

_WORKER_BOOTSTRAP = (
    "import os,runpy,site;"
    "site.addsitedir(os.environ['_FLASH_JOB_SITE_PACKAGES']);"
    "runpy.run_module('flash.engine.worker_entrypoint',run_name='__main__')"
)


def _deadline_fields() -> dict[str, float]:
    now = time.time()
    return {
        "run_created_at": now,
        "run_max_wall_seconds": 3600.0,
        "deadline_at": now + 3600.0,
    }


def _input_data(
    *,
    code_prefix: str,
    extra_pip: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> dict:
    worker_env = {"ATTEMPT": "7", "HF_TOKEN": "tok", "PYTHONPATH": "base-pythonpath"}
    worker_env.update(env or {})
    return {
        "phase": "sft",
        "seed": 0,
        "hf_repo": "owner/runs",
        "job_spec_json": json.dumps({"algorithm": "sft", "run_id": "flash-test-run"}),
        "env": worker_env,
        "extra_pip": list(extra_pip or []),
        "code_prefix": code_prefix,
        **_deadline_fields(),
    }


def _remove_worker_artifacts() -> None:
    for path in (
        "/tmp/console_sft.txt",
        "/tmp/console_sft.txt.tail",
        "/tmp/job_spec.json",
        "/tmp/metrics.json",
        "/tmp/train_meta.json",
    ):
        Path(path).unlink(missing_ok=True)


def _build_wheel(tmp_path: Path) -> tuple[Path, str]:
    suffix = hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12]
    module_name = f"flash_env_probe_{suffix}"
    version = "1.0.0"
    dist_info = f"{module_name}-{version}.dist-info"
    wheel_path = tmp_path / f"{module_name}-{version}-py3-none-any.whl"
    files = {
        f"{module_name}/__init__.py": 'VALUE = "isolated"\n',
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\n"
            f"Name: {module_name}\n"
            f"Version: {version}\n"
        ),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: flash-test\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ),
    }
    record_path = f"{dist_info}/RECORD"
    files[record_path] = "".join(f"{path},,\n" for path in (*files, record_path))
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        for path, content in files.items():
            wheel.writestr(path, content)
    return wheel_path, module_name


def _build_pth_wheel(tmp_path: Path) -> tuple[Path, str]:
    suffix = hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12]
    module_name = f"flash_pth_probe_{suffix}"
    dist_name = f"{module_name}_loader"
    version = "1.0.0"
    source_root = tmp_path / f"{module_name}-source"
    source_root.mkdir()
    (source_root / f"{module_name}.py").write_text('VALUE = "pth-loaded"\n', encoding="utf-8")

    dist_info = f"{dist_name}-{version}.dist-info"
    wheel_path = tmp_path / f"{dist_name}-{version}-py3-none-any.whl"
    files = {
        f"{dist_name}.pth": f"{source_root}\n",
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\n"
            f"Name: {dist_name}\n"
            f"Version: {version}\n"
        ),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: flash-test\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ),
    }
    record_path = f"{dist_info}/RECORD"
    files[record_path] = "".join(f"{path},,\n" for path in (*files, record_path))
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        for path, content in files.items():
            wheel.writestr(path, content)
    return wheel_path, module_name


def _run_successful_job(
    monkeypatch,
    tmp_path: Path,
    *,
    extra_pip: list[str],
    env: dict[str, str] | None = None,
    before_worker=None,
):
    import huggingface_hub

    from flash.providers.runpod.train import endpoints

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    code_prefix = "code/0123456789abcdef0123456789abcdef/flash"
    run_calls: list[dict] = []
    launch_calls: list[dict] = []
    real_run = subprocess.run
    real_popen = subprocess.Popen

    def recording_run(cmd, **kwargs):
        run_calls.append({"cmd": list(cmd), **kwargs})
        current_popen = subprocess.Popen
        subprocess.Popen = real_popen
        try:
            return real_run(cmd, **kwargs)
        finally:
            subprocess.Popen = current_popen

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def list_repo_tree(self, **kwargs):
            return [types.SimpleNamespace(path=f"{code_prefix}/__init__.py", size=1)]

        def upload_file(self, **kwargs):
            return None

    def fake_download(*, filename, local_dir, **kwargs):
        return os.path.join(local_dir, filename)

    class FakeProc:
        def __init__(self, cmd, **kwargs):
            launch = {"cmd": list(cmd), **kwargs}
            launch_calls.append(launch)
            if before_worker is not None:
                before_worker(launch)
            self.stdout = iter(["worker completed\n"])
            self.returncode = 0

        def wait(self):
            Path("/tmp/metrics.json").write_text('{"score": 1.0}', encoding="utf-8")
            return 0

    monkeypatch.setattr(subprocess, "run", recording_run)
    monkeypatch.setattr(subprocess, "Popen", FakeProc)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    _remove_worker_artifacts()
    try:
        result = endpoints._train_body(
            _input_data(code_prefix=code_prefix, extra_pip=extra_pip, env=env)
        )
    finally:
        _remove_worker_artifacts()

    return result, run_calls, launch_calls, code_prefix


def test_extra_pip_is_installed_only_in_job_venv_and_base_worker_sees_it(
    monkeypatch, tmp_path
):
    wheel_path, module_name = _build_wheel(tmp_path)
    base_package = Path(sysconfig.get_path("purelib")) / module_name
    assert not base_package.exists()

    venv_root = tmp_path / "flash-runpod-job-venvs"
    stale_venv = venv_root / "previous-job"
    stale_venv.mkdir(parents=True)
    (stale_venv / "stale-package.txt").write_text("old", encoding="utf-8")
    observed: dict[str, Path] = {}

    def before_worker(launch):
        pythonpath = launch["env"]["PYTHONPATH"].split(os.pathsep)
        job_site_packages = Path(pythonpath[0])
        observed["job_site_packages"] = job_site_packages
        observed["job_venv"] = job_site_packages.parents[2]
        assert (job_site_packages / module_name / "__init__.py").is_file()
        assert not base_package.exists()
        assert not stale_venv.exists()

    result, run_calls, launch_calls, code_prefix = _run_successful_job(
        monkeypatch,
        tmp_path,
        extra_pip=[str(wheel_path)],
        before_worker=before_worker,
    )

    assert result == {"score": 1.0}
    venv_call = run_calls[0]
    assert venv_call["cmd"][:4] == [
        sys.executable,
        "-m",
        "venv",
        "--system-site-packages",
    ]
    assert any(call["cmd"][1:4] == ["-m", "pip", "install"] for call in run_calls)

    launch = launch_calls[0]
    assert launch["cmd"] == [sys.executable, "-c", _WORKER_BOOTSTRAP]
    assert launch["cwd"] == f"/runcode/{os.path.dirname(code_prefix)}"
    assert launch["env"]["RUN_MODE"] == "sft"
    assert launch["env"]["HF_TOKEN"] == "tok"
    assert launch["env"]["_FLASH_JOB_SITE_PACKAGES"] == str(observed["job_site_packages"])
    pythonpath = launch["env"]["PYTHONPATH"].split(os.pathsep)
    assert pythonpath == [
        str(observed["job_site_packages"]),
        f"/runcode/{os.path.dirname(code_prefix)}",
        "base-pythonpath",
    ]
    assert not observed["job_venv"].exists()
    assert not base_package.exists()


def test_worker_bootstrap_processes_job_pth_and_preserves_import_precedence(
    monkeypatch, tmp_path
):
    wheel_path, module_name = _build_wheel(tmp_path)
    pth_wheel_path, pth_module_name = _build_pth_wheel(tmp_path)
    base_package = Path(sysconfig.get_path("purelib")) / module_name
    assert not base_package.exists()

    code_dir = tmp_path / "downloaded-code"
    engine_dir = code_dir / "flash" / "engine"
    engine_dir.mkdir(parents=True)
    flash_init = code_dir / "flash" / "__init__.py"
    flash_init.write_text('ORIGIN = "code-dir"\n', encoding="utf-8")
    (engine_dir / "__init__.py").write_text("", encoding="utf-8")

    base_dir = tmp_path / "base-site"
    base_module = base_dir / module_name
    base_module.mkdir(parents=True)
    (base_module / "__init__.py").write_text('VALUE = "base"\n', encoding="utf-8")

    output_path = tmp_path / "worker-imports.json"
    entrypoint_source = (
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "import flash\n"
        f"import {module_name} as normal_package\n"
        f"import {pth_module_name} as pth_package\n"
        "Path(os.environ['PROBE_OUTPUT']).write_text(json.dumps({\n"
        "    'flash_file': flash.__file__,\n"
        "    'normal_value': normal_package.VALUE,\n"
        "    'pth_value': pth_package.VALUE,\n"
        "}), encoding='utf-8')\n"
    )
    (engine_dir / "worker_entrypoint.py").write_text(entrypoint_source, encoding="utf-8")

    real_popen = subprocess.Popen
    observed: dict[str, object] = {}

    def run_process(cmd, *, cwd, env):
        proc = real_popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate()
        return proc.returncode, stdout, stderr

    def before_worker(launch):
        assert launch["cmd"] == [sys.executable, "-c", _WORKER_BOOTSTRAP]
        job_site_packages = Path(launch["env"]["_FLASH_JOB_SITE_PACKAGES"])
        assert (job_site_packages / module_name / "__init__.py").is_file()
        assert any(job_site_packages.glob("*.pth"))

        worker_env = dict(launch["env"])
        worker_env["PYTHONPATH"] = os.pathsep.join(
            (str(job_site_packages), str(code_dir), str(base_dir))
        )
        worker_env["PROBE_OUTPUT"] = str(output_path)

        plain_cmd = [sys.executable, "-c", f"import {pth_module_name}"]
        plain_returncode, _, plain_stderr = run_process(
            plain_cmd,
            cwd=str(code_dir),
            env=worker_env,
        )
        assert plain_returncode != 0
        assert "ModuleNotFoundError" in plain_stderr
        observed["plain_stderr"] = plain_stderr

        returncode, stdout, stderr = run_process(
            launch["cmd"],
            cwd=str(code_dir),
            env=worker_env,
        )
        assert returncode == 0, f"stdout={stdout!r} stderr={stderr!r}"
        observed.update(json.loads(output_path.read_text(encoding="utf-8")))

    try:
        result, _, _, _ = _run_successful_job(
            monkeypatch,
            tmp_path,
            extra_pip=[str(wheel_path), str(pth_wheel_path)],
            before_worker=before_worker,
        )
    except subprocess.CalledProcessError as exc:
        if list(exc.cmd)[1:4] == ["-m", "venv", "--system-site-packages"]:
            pytest.skip(f"venv creation unavailable: {exc}")
        raise

    assert result == {"score": 1.0}
    assert observed["pth_value"] == "pth-loaded"
    assert observed["normal_value"] == "isolated"
    assert Path(str(observed["flash_file"])).resolve() == flash_init.resolve()
    assert "ModuleNotFoundError" in str(observed["plain_stderr"])
    assert not base_package.exists()


def test_job_pip_env_cannot_redirect_install_outside_venv(monkeypatch, tmp_path):
    wheel_path, module_name = _build_wheel(tmp_path)
    redirected_target = tmp_path / "redirected-target"
    observed: dict[str, Path] = {}

    def before_worker(launch):
        job_site_packages = Path(launch["env"]["PYTHONPATH"].split(os.pathsep)[0])
        observed["job_site_packages"] = job_site_packages
        assert (job_site_packages / module_name).is_dir()
        assert not (redirected_target / module_name).exists()

    _, run_calls, _, _ = _run_successful_job(
        monkeypatch,
        tmp_path,
        extra_pip=[str(wheel_path)],
        env={
            "PIP_TARGET": str(redirected_target),
            "PIP_PREFIX": str(tmp_path / "redirected-prefix"),
            "PIP_ROOT": str(tmp_path / "redirected-root"),
            "PIP_USER": "1",
            "PIP_SRC": str(tmp_path / "redirected-src"),
            "pip_src": str(tmp_path / "redirected-src-lowercase"),
            "PYTHONUSERBASE": str(tmp_path / "redirected-userbase"),
            "PIP_CONFIG_FILE": str(tmp_path / "redirected-pip.conf"),
        },
        before_worker=before_worker,
    )

    pip_call = next(call for call in run_calls if call["cmd"][1:4] == ["-m", "pip", "install"])
    pip_env = pip_call["env"]
    assert pip_env["PIP_USER"] == "0"
    for key in (
        "PIP_TARGET",
        "PIP_PREFIX",
        "PIP_ROOT",
        "PIP_SRC",
        "pip_src",
        "PYTHONUSERBASE",
        "PIP_CONFIG_FILE",
    ):
        assert key not in pip_env
    assert not redirected_target.exists()
    assert not observed["job_site_packages"].parents[2].exists()


def test_job_pip_rejects_location_changing_arguments(monkeypatch, tmp_path):
    from flash.providers.runpod.train import endpoints

    redirected_target = tmp_path / "redirected-target"
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    with pytest.raises(ValueError, match="extra_pip location option is not allowed"):
        endpoints._train_body(
            _input_data(
                code_prefix="../code/flash",
                extra_pip=[f"--target={redirected_target}"],
            )
        )

    assert not redirected_target.exists()
    venv_root = tmp_path / "flash-runpod-job-venvs"
    assert not list(venv_root.iterdir())


@pytest.mark.parametrize(
    "argument",
    [
        "--target",
        "--target=/tmp/x",
        "--tar=/tmp/x",
        "--targ=/tmp/x",
        "-t",
        "-t=/tmp/x",
        "-t/tmp/x",
        "-qt/tmp/x",
        "--prefix=/tmp/x",
        "--prefi=/tmp/x",
        "--root=/tmp/x",
        "--user",
        "--user=1",
        "--src",
        "--src=/tmp/x",
        "--sr",
        "--sr=/tmp/x",
    ],
)
def test_all_pip_location_option_forms_are_rejected(monkeypatch, tmp_path, argument):
    from flash.providers.runpod.train import endpoints

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    with pytest.raises(ValueError, match="extra_pip location option is not allowed"):
        endpoints._train_body(_input_data(code_prefix="../code/flash", extra_pip=[argument]))


def test_empty_extra_pip_runs_base_worker_with_clean_pythonpath(monkeypatch, tmp_path):
    result, run_calls, launch_calls, code_prefix = _run_successful_job(
        monkeypatch,
        tmp_path,
        extra_pip=[],
    )

    assert result == {"score": 1.0}
    assert not any(call["cmd"][1:4] == ["-m", "pip", "install"] for call in run_calls)
    launch = launch_calls[0]
    assert launch["cmd"] == [sys.executable, "-c", _WORKER_BOOTSTRAP]
    pythonpath = launch["env"]["PYTHONPATH"].split(os.pathsep)
    assert Path(pythonpath[0]).name == "site-packages"
    assert pythonpath[1:] == [f"/runcode/{os.path.dirname(code_prefix)}", "base-pythonpath"]
    assert not Path(pythonpath[0]).parents[2].exists()


def test_deadline_exit_does_not_remove_venv_before_hard_exit(monkeypatch, tmp_path):
    from flash.providers.runpod.train import endpoints

    class DeadlineExit(Exception):
        pass

    timers = []
    rmtree_calls: list[tuple[Path, bool]] = []
    exit_called = False
    original_rmtree = shutil.rmtree

    class FakeTimer:
        def __init__(self, interval, callback):
            self.interval = interval
            self.callback = callback
            self.daemon = False
            self.cancelled = False
            timers.append(self)

        def start(self):
            return None

        def cancel(self):
            self.cancelled = True

    def fake_exit(code):
        nonlocal exit_called
        assert code == 124
        assert not rmtree_calls
        exit_called = True
        raise DeadlineExit

    def tracked_rmtree(path, ignore_errors=False):
        rmtree_calls.append((Path(path), exit_called))
        return original_rmtree(path, ignore_errors=ignore_errors)

    def fake_run(cmd, **kwargs):
        assert cmd[1:4] == ["-m", "venv", "--system-site-packages"]
        timers[0].callback()

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(threading, "Timer", FakeTimer)
    monkeypatch.setattr(os, "_exit", fake_exit)
    monkeypatch.setattr(shutil, "rmtree", tracked_rmtree)
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(DeadlineExit):
        endpoints._train_body(_input_data(code_prefix="../code/flash"))

    assert exit_called is True
    assert len(rmtree_calls) == 1
    assert rmtree_calls[0][1] is True
    assert timers[0].cancelled is True
