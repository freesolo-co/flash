"""Run the bounded precision comparison and always release its sandbox."""

import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

import modal
from modal.exception import SandboxFilesystemNotFoundError

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RESULTS = REPO / "results/deepinfer-precision"
RESULTS.mkdir(parents=True, exist_ok=True)
IMAGE = "im-DG4Ugn1qPde6Y5UYusMSGR"
DURATION = 7200
BASE_COMMIT = "ee68f7a97881b30836cd3e5bf41cd775a6a503f7"


def execute(sb, args, timeout):
    process = sb.exec(*args, timeout=timeout)

    def drain(stream):
        for line in stream:
            print(line, end="", flush=True)

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        readers = [pool.submit(drain, process.stdout), pool.submit(drain, process.stderr)]
        process.wait()
        for reader in readers:
            reader.result()
    return process.returncode


def main():
    app = modal.App.lookup(
        "deepinfer-precision-ab-20261024", environment_name="main", create_if_missing=True
    )
    assert not list(modal.Sandbox.list(app_id=app.app_id)), "Comparison already active"
    metadata = {
        "app_id": app.app_id,
        "image": IMAGE,
        "started": time.time(),
        "timeout_seconds": DURATION,
        "gpu": "B200:4",
        "cpu": 16,
        "host_memory_mib": 524288,
        "base_commit": BASE_COMMIT,
        "order": ["fp4-first", "fp8-first", "fp8-second", "fp4-second"],
        "similarity": {
            "throughput_relative": 0.05,
            "p95_latency_relative": 0.10,
            "quality": "no task accuracy regression",
        },
    }
    (RESULTS / "allocation.json").write_text(json.dumps(metadata, indent=2))
    sb = None
    code = 1
    try:
        with modal.enable_output():
            sb = modal.Sandbox.create(
                "sleep",
                str(DURATION),
                app=app,
                image=modal.Image.from_id(IMAGE),
                gpu="B200:4",
                cpu=(16, 16),
                memory=(524288, 524288),
                timeout=DURATION,
                volumes={
                    "/weights": modal.Volume.from_name(
                        "deepinfer-b200-1017-benchmark-weights", environment_name="main"
                    ).read_only()
                },
                env={
                    "PYTHONPATH": "/root:/opt/DeepInfer",
                    "HF_HUB_OFFLINE": "1",
                    "PYTHONUNBUFFERED": "1",
                    "PATH": "/opt/deepinfer-venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                    "VIRTUAL_ENV": "/opt/deepinfer-venv",
                    "DEEPINFER_ACCELERATOR": "nvidia",
                },
            )
        metadata["sandbox_id"] = sb.object_id
        (RESULTS / "allocation.json").write_text(json.dumps(metadata, indent=2))
        print("ALLOCATED", sb.object_id, flush=True)
        data = subprocess.check_output(
            [
                "git",
                "archive",
                "--format=tar.gz",
                BASE_COMMIT,
                "flash",
                "scripts/prepare_deepinfer_image.py",
            ],
            cwd=REPO,
        )
        sb.filesystem.write_bytes(data, "/tmp/source.tar.gz")
        assert (
            execute(
                sb,
                [
                    "python",
                    "-c",
                    "import shutil;shutil.rmtree('/root/flash/serving/deepinfer',ignore_errors=True)",
                ],
                60,
            )
            == 0
        )
        assert execute(sb, ["tar", "-xzf", "/tmp/source.tar.gz", "-C", "/root"], 60) == 0
        assert execute(sb, ["mkdir", "-p", "/results"], 30) == 0
        for name in ("stage.py", "server.py", "bench.py", "restore_cache.py", "runner.py"):
            sb.filesystem.write_text((HERE / name).read_text(), "/tmp/" + name)
        assert execute(sb, ["python", "-u", "/tmp/stage.py"], 600) == 0
        assert execute(sb, ["python", "-u", "/tmp/restore_cache.py"], 600) == 0
        remaining = int(DURATION - (time.time() - metadata["started"]) - 120)
        sb.exec("python", "-u", "/tmp/runner.py", timeout=remaining)
        deadline = time.monotonic() + remaining
        while time.monotonic() < deadline:
            try:
                code = int(sb.filesystem.read_text("/results/exit-code"))
                break
            except SandboxFilesystemNotFoundError:
                if sb.poll() is not None:
                    raise RuntimeError(
                        "Benchmark sandbox exited before saving completion"
                    ) from None
                time.sleep(15)
        else:
            raise TimeoutError("Precision comparison exceeded its bounded lifetime")
        print("BENCH_EXIT", code, flush=True)
    finally:
        if sb is not None:
            try:
                if (
                    execute(sb, ["tar", "-czf", "/tmp/evidence.tar.gz", "-C", "/results", "."], 120)
                    == 0
                ):
                    (RESULTS / "evidence.tar.gz").write_bytes(
                        sb.filesystem.read_bytes("/tmp/evidence.tar.gz")
                    )
                    print("EVIDENCE_SAVED", flush=True)
            finally:
                sb.terminate(wait=True)
                metadata.update(stopped=time.time(), sandbox_exit=sb.poll(), benchmark_exit=code)
        stopped = subprocess.run(
            [str(REPO / ".venv/bin/modal"), "app", "stop", app.app_id, "--env", "main", "--yes"],
            capture_output=True,
            text=True,
        )
        metadata["app_stop_exit"] = stopped.returncode
        (RESULTS / "shutdown.json").write_text(json.dumps(metadata, indent=2))
        print("CLEANUP", json.dumps(metadata), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
