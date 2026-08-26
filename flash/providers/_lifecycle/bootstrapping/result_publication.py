"""bounded fenced result publication for the shared instance bootstrap."""

from __future__ import annotations

import math
import subprocess
import sys
import time

if __package__:
    from flash.providers._lifecycle.bootstrapping import processes as bootstrap_processes
else:
    import bootstrap_processes  # type: ignore[no-redef]


def _finite_positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is invalid")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise RuntimeError(f"{label} is invalid")
    return number


def _publish(
    payload: dict,
    env: dict,
    *,
    code_dir: str,
    started_at: float,
    publisher: str,
    label: str,
) -> None:
    result_deadline = _finite_positive(
        payload.get("result_deadline_at"), "result visibility deadline"
    )
    remaining = result_deadline - _finite_positive(time.time(), "current clock")
    if remaining <= 0:
        raise TimeoutError(f"result visibility deadline expired before {label} publication")
    result_env = {**env, "FLASH_RUN_DEADLINE_AT": str(result_deadline)}
    command = (
        f"from flash.engine.worker.io.result import {publisher}; "
        f"{publisher}(started_at={started_at!r})"
    )
    process, process_group_id = bootstrap_processes.start_process_group(
        [sys.executable, "-c", command],
        cwd=code_dir,
        env=result_env,
    )
    try:
        process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        bootstrap_processes.terminate_process_group(
            process,
            process_group_id=process_group_id,
        )
        raise TimeoutError(f"{label} result publication exceeded its visibility window") from None
    if process.returncode != 0:
        raise RuntimeError(f"{label} result publication failed")


def publish_deadline_result(
    payload: dict,
    env: dict,
    *,
    code_dir: str,
    started_at: float,
) -> None:
    _publish(
        payload,
        env,
        code_dir=code_dir,
        started_at=started_at,
        publisher="publish_deadline_result",
        label="deadline",
    )


def publish_cancelled_result(
    payload: dict,
    env: dict,
    *,
    code_dir: str,
    started_at: float,
) -> None:
    _publish(
        payload,
        env,
        code_dir=code_dir,
        started_at=started_at,
        publisher="publish_cancelled_result",
        label="cancellation",
    )


def publish_bootstrap_failure_result(
    payload: dict,
    env: dict,
    *,
    code_dir: str,
    started_at: float,
    error: str,
    failure_class: str = "worker",
) -> None:
    failure_env = {
        **env,
        "FLASH_BOOTSTRAP_ERROR": error,
        "FLASH_BOOTSTRAP_FAILURE_CLASS": failure_class,
    }
    _publish(
        payload,
        failure_env,
        code_dir=code_dir,
        started_at=started_at,
        publisher="publish_bootstrap_failure_result",
        label="bootstrap failure",
    )
