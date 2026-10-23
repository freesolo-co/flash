"""Run the shared-training tensor and API qualification on one owned Modal GPU.

This is a bounded fixture qualification, not a throughput or hosted-service test.
The sandbox is terminated in finally, including on local test failure.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory


def run(root, output, model_id=None, revision=None):
    import modal

    from flash.providers._lifecycle.net.worker import worker_image_for_gpu

    gpu = "H100" if model_id else "L4"
    image = modal.Image.from_registry(worker_image_for_gpu(gpu))
    image = image.run_commands(
        "uv pip install --python /opt/verl-venv/bin/python "
        "torch==2.10.0 peft==0.19.0 pytest==9.1.1 httpx==0.28.1 fastapi==0.136.0 uvicorn==0.52.0"
    )
    image = image.add_local_dir(
        root / "flash", "/app/flash", ignore=["**/__pycache__/**"], copy=True
    )
    for name in (
        "test_shared_training.py",
        "test_shared_training_tensors.py",
        "test_shared_training_http.py",
    ):
        image = image.add_local_file(root / "tests" / name, f"/app/tests/{name}", copy=True)
    image = image.add_local_file(
        root / "scripts/shared_model_qualification.py", "/app/model_qualification.py", copy=True
    )
    name = "flash-shared-qualification-" + uuid.uuid4().hex[:12]
    app = modal.App.lookup(name, create_if_missing=True)
    started = time.time()
    sandbox = None
    report = {"name": name, "gpu": gpu, "fixture_only": not bool(model_id)}
    try:
        code = (
            "import subprocess, sys, importlib.metadata as m; "
            "print('VERSIONS', {p:m.version(p) for p in ('torch','peft','transformers','fastapi')}, flush=True); "
            "subprocess.run([sys.executable,'-m','pytest','-q','-ra','/app/tests'],check=True)"
        )
        if model_id:
            code += (
                "; import runpy; runpy.run_path('/app/model_qualification.py',run_name='__main__')"
            )
        environment = {
            "PYTHONPATH": "/app",
            "FLASH_SHARED_TEST_DEVICE": "cuda",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
        if model_id:
            environment.update(FLASH_SHARED_MODEL_ID=model_id, FLASH_SHARED_MODEL_REVISION=revision)
        sandbox = modal.Sandbox.create(
            "/opt/verl-venv/bin/python",
            "-c",
            code,
            app=app,
            image=image,
            gpu=gpu,
            cpu=2,
            memory=65536 if model_id else 16384,
            timeout=1200 if model_id else 600,
            workdir="/app",
            env=environment,
        )
        report["sandbox_id"] = sandbox.object_id
        stdout = sandbox.stdout.read()
        stderr = sandbox.stderr.read()
        sandbox.wait()
        report.update(exit_code=sandbox.returncode, stdout=stdout, stderr=stderr)
        print(stdout, flush=True)
        if sandbox.returncode:
            print(stderr, flush=True)
            raise RuntimeError("shared CUDA qualification failed")
    finally:
        if sandbox is not None:
            sandbox.terminate()
            report["terminated"] = True
        report["wall_seconds"] = time.time() - started
        output.write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-paid-test", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    args = parser.parse_args()
    if not args.run_paid_test:
        parser.error("--run-paid-test is required")
    if bool(args.model_id) != bool(args.revision):
        parser.error("--model-id and --revision must be supplied together")
    root = Path(__file__).resolve().parents[1]
    # snapshot before Modal reads files asynchronously; edits cannot alter a running qualification.
    with TemporaryDirectory(prefix="flash-shared-qualification-") as directory:
        snapshot = Path(directory)
        shutil.copytree(
            root / "flash", snapshot / "flash", ignore=shutil.ignore_patterns("__pycache__")
        )
        (snapshot / "tests").mkdir()
        (snapshot / "scripts").mkdir()
        shutil.copy2(
            root / "scripts/shared_model_qualification.py",
            snapshot / "scripts/shared_model_qualification.py",
        )
        for source in (root / "tests").glob("test_shared_training*.py"):
            shutil.copy2(source, snapshot / "tests" / source.name)
        run(snapshot, args.output, args.model_id, args.revision)


if __name__ == "__main__":
    main()
