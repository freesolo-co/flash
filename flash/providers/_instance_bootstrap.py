"""Bootstrap shared by instance-based providers (e.g. Lambda). Runs inside the worker container.

Stdlib + huggingface_hub only — never import flash here. Reads payload from ``/root/flash/payload.json``.
"""

from __future__ import annotations

import contextlib
import json
import math
import multiprocessing
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

PAYLOAD_PATH = "/root/flash/payload.json"
CODE_ROOT = "/runcode"
_CONSOLE_UPLOAD_INTERVAL_S = 3600.0
_CONSOLE_UPLOAD_STOP_TIMEOUT_S = 2.0
_CONSOLE_UPLOAD_FINAL_TIMEOUT_S = 10.0
_CONSOLE_UPLOAD_TERMINATE_TIMEOUT_S = 1.0
_CONSOLE_UPLOAD_REAP_RESERVE_S = 2 * _CONSOLE_UPLOAD_TERMINATE_TIMEOUT_S
_HF_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
_HF_RETRY_DELAYS_S = (1.0, 3.0, 8.0, 20.0, 60.0)
_HF_RETRY_AFTER_MAX_S = 60.0
_TERMINAL_MARKER_GRACE_S = 0.25
_TERMINAL_BOOKKEEPING_RESERVE_S = _TERMINAL_MARKER_GRACE_S
_MAX_ATTEMPT_ID = (1 << 63) - 1
_SECRET_RE = re.compile(
    r"(?i)(authorization|api[-_ ]?key|access[-_ ]?token|token|secret|password)"
    r"(\s*[:=]\s*)(?:bearer\s+)?([^\s,;]+)"
)


def _safe_detail(value: object, limit: int = 1000) -> str:
    text = f"{type(value).__name__}: {value}" if isinstance(value, BaseException) else str(value)
    for key, secret in os.environ.items():
        upper = key.upper()
        if secret and (
            upper in {"AUTHORIZATION", "HF_TOKEN"}
            or upper.endswith(("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD"))
        ):
            text = text.replace(secret, "<redacted>")
    text = re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", text)
    text = _SECRET_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}<redacted>", text)
    return text[:limit]


class RetriableBootstrapError(RuntimeError):
    """Infra-shaped failure → marker carries retriable=True → poller retries (job_preempted) instead of job_failed."""


def _finite_positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is invalid")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise RuntimeError(f"{label} is invalid")
    return number


def _canonical_deadline_at(payload: dict) -> float:
    """Return the identity-verified absolute run deadline without checking expiry."""
    deadline = _finite_positive_number(payload.get("deadline_at"), "run wall deadline")
    created_at = _finite_positive_number(
        payload.get("run_created_at"), "run wall deadline creation time"
    )
    max_wall_seconds = _finite_positive_number(
        payload.get("run_max_wall_seconds"), "run wall deadline maximum seconds"
    )
    canonical = created_at + max_wall_seconds
    if not math.isclose(deadline, canonical, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError("run wall deadline does not match canonical submission deadline")
    return deadline


def require_deadline_at(payload: dict) -> float:
    """Return the independently verified unexpired run deadline or fail closed."""
    deadline = _canonical_deadline_at(payload)
    now = _finite_positive_number(time.time(), "current clock")
    if deadline <= now:
        raise TimeoutError("run wall deadline exceeded before bootstrap")
    return deadline


def _publish_timeout_marker_then_exit(payload: dict, message: str) -> None:
    print(f"FLASH: {message}", flush=True)
    marker_done = threading.Event()

    def _publish() -> None:
        try:
            with contextlib.suppress(Exception):
                write_attempt_marker(payload, ok=False, error=message)
        finally:
            marker_done.set()

    try:
        marker_thread = threading.Thread(target=_publish, daemon=True)
        marker_thread.start()
        marker_done.wait(_TERMINAL_MARKER_GRACE_S)
    finally:
        os._exit(124)


def arm_deadline_watchdog(
    deadline_at: float,
    payload: dict,
) -> tuple[threading.Timer, threading.Event]:
    """Hard-stop setup or training that remains alive at the absolute cutoff."""
    done = threading.Event()

    def _fire() -> None:
        if done.is_set():
            return
        _publish_timeout_marker_then_exit(
            payload,
            "run wall deadline exceeded; self-terminating box",
        )

    timer = threading.Timer(max(0.0, deadline_at - time.time()), _fire)
    timer.daemon = True
    timer.start()
    return timer, done


def load_payload() -> dict:
    with open(PAYLOAD_PATH) as f:
        return json.load(f)


def _arm(payload: dict) -> str:
    return str(payload.get("flash_arm") or "instance")


def _code_prefix(payload: dict) -> str:
    raw = payload.get("code_prefix")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("missing code_prefix")
    prefix = raw.strip().strip("/")
    parts = prefix.split("/")
    digest = parts[1] if len(parts) == 3 else ""
    if (
        len(parts) != 3
        or parts[0] != "code"
        or parts[2] != "flash"
        or len(digest) != 32
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError(f"invalid code_prefix: {prefix!r}")
    return prefix


def _code_dir(payload: dict) -> str:
    raw = payload.get("code_prefix")
    if not isinstance(raw, str) or not raw.strip():
        return os.path.join(CODE_ROOT, "code")
    return os.path.join(CODE_ROOT, os.path.dirname(_code_prefix(payload)))


def _hf_status_code(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


def _hf_retry_after(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    value = headers.get("retry-after") if hasattr(headers, "get") else None
    if not value and hasattr(headers, "items"):
        for key, candidate in headers.items():
            if str(key).lower() == "retry-after":
                value = candidate
                break
    if not value:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(value))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            seconds = (retry_at - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError):
            return None
    return min(_HF_RETRY_AFTER_MAX_S, max(0.0, seconds))


def _hf_call(call, label: str, *, deadline_at: float | None = None):
    for attempt in range(len(_HF_RETRY_DELAYS_S) + 1):
        if deadline_at is not None:
            remaining = deadline_at - _finite_positive_number(time.time(), "current clock")
            if remaining <= 0:
                raise TimeoutError(f"{label} exceeded the run wall deadline")
        try:
            return call()
        except Exception as exc:
            if _hf_status_code(exc) not in _HF_TRANSIENT_STATUS_CODES or attempt >= len(
                _HF_RETRY_DELAYS_S
            ):
                raise
            retry_after = _hf_retry_after(exc)
            delay = retry_after if retry_after is not None else _HF_RETRY_DELAYS_S[attempt]
            if deadline_at is not None:
                remaining = deadline_at - _finite_positive_number(time.time(), "current clock")
                if remaining <= 0:
                    raise TimeoutError(f"{label} exceeded the run wall deadline") from None
                delay = min(delay, remaining)
            print(
                f"{label} transient Hugging Face error; retrying in {delay:.0f}s: "
                f"{_safe_detail(exc, 500)}",
                flush=True,
            )
            if delay > 0:
                time.sleep(delay)
    raise AssertionError("unreachable")


def hf_upload(
    payload: dict,
    local_path: str,
    repo_subpath: str,
    *,
    enforce_deadline: bool = True,
) -> None:
    """Upload one artifact under the run's HF prefix; never raises."""
    try:
        from huggingface_hub import HfApi

        if enforce_deadline and "deadline_at" in payload:
            require_deadline_at(payload)
        HfApi(token=(payload.get("env") or {}).get("HF_TOKEN")).upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"{payload['hf_prefix']}/{repo_subpath}",
            repo_id=payload["hf_repo"],
            repo_type="dataset",
        )
    except Exception as exc:
        print(f"hf upload warn ({repo_subpath}): {_safe_detail(exc)}", flush=True)


def _upload_console_snapshot(payload: dict, console: str, mode: str, extra: str = "") -> None:
    """Upload one console snapshot from an isolated process."""
    tail_path = console + ".tail"
    with open(console, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 64_000))
        tail = f.read().decode("utf-8", "replace")
    if extra:
        tail += extra
    with open(tail_path, "w", encoding="utf-8", errors="replace") as f:
        f.write(_safe_detail(tail, 64_000))
    hf_upload(payload, tail_path, f"console_{mode}.txt")


def _console_upload_loop(
    payload: dict,
    console: str,
    mode: str,
    interval_s: float,
    stop_upload,
) -> None:
    while not stop_upload.wait(interval_s):
        try:
            _upload_console_snapshot(payload, console, mode)
        except Exception as exc:
            print(f"console upload warn: {_safe_detail(exc)}", flush=True)


def _upload_cleanup_deadlines(deadline_at: float) -> tuple[float, float]:
    reaping_deadline_at = deadline_at - _TERMINAL_BOOKKEEPING_RESERVE_S
    upload_deadline_at = reaping_deadline_at - _CONSOLE_UPLOAD_REAP_RESERVE_S
    return upload_deadline_at, reaping_deadline_at


def _worker_execution_deadline(upload_deadline_at: float) -> float:
    return (
        upload_deadline_at
        - _CONSOLE_UPLOAD_STOP_TIMEOUT_S
        - _CONSOLE_UPLOAD_FINAL_TIMEOUT_S
    )


def _join_upload_process_before(process, deadline_at: float, max_wait_s: float) -> None:
    remaining = max(
        0.0,
        deadline_at - _finite_positive_number(time.time(), "current clock"),
    )
    process.join(min(max_wait_s, remaining))


def _defer_upload_cleanup_error(deferred_error, operation, *args, **kwargs):
    try:
        return deferred_error, operation(*args, **kwargs), True
    except BaseException as exc:
        return exc if deferred_error is None else deferred_error, None, False


def _cleanup_upload_process(
    process,
    initial_deadline_at: float,
    initial_wait_s: float,
    reaping_deadline_at: float,
    warning: str,
    stop_upload=None,
) -> bool:
    """reap an uploader before propagating the first cleanup interruption."""
    deferred_error = None
    if stop_upload is not None:
        deferred_error, _, _ = _defer_upload_cleanup_error(
            deferred_error,
            stop_upload.set,
        )
    deferred_error, _, _ = _defer_upload_cleanup_error(
        deferred_error,
        _join_upload_process_before,
        process,
        initial_deadline_at,
        initial_wait_s,
    )
    deferred_error, alive, checked = _defer_upload_cleanup_error(
        deferred_error,
        process.is_alive,
    )
    clean = checked and alive is False
    dead = clean

    if not dead:
        deferred_error, _, _ = _defer_upload_cleanup_error(
            deferred_error,
            process.terminate,
        )
        deferred_error, _, _ = _defer_upload_cleanup_error(
            deferred_error,
            _join_upload_process_before,
            process,
            reaping_deadline_at,
            _CONSOLE_UPLOAD_TERMINATE_TIMEOUT_S,
        )
        deferred_error, alive, checked = _defer_upload_cleanup_error(
            deferred_error,
            process.is_alive,
        )
        dead = checked and alive is False

    if not dead:
        deferred_error, _, _ = _defer_upload_cleanup_error(
            deferred_error,
            process.kill,
        )
        deferred_error, _, _ = _defer_upload_cleanup_error(
            deferred_error,
            _join_upload_process_before,
            process,
            reaping_deadline_at,
            _CONSOLE_UPLOAD_TERMINATE_TIMEOUT_S,
        )
        deferred_error, alive, checked = _defer_upload_cleanup_error(
            deferred_error,
            process.is_alive,
        )
        dead = checked and alive is False

    if not dead:
        deferred_error, _, _ = _defer_upload_cleanup_error(
            deferred_error,
            process.kill,
        )
        deferred_error, _, _ = _defer_upload_cleanup_error(
            deferred_error,
            _join_upload_process_before,
            process,
            reaping_deadline_at,
            _CONSOLE_UPLOAD_TERMINATE_TIMEOUT_S,
        )
        deferred_error, alive, checked = _defer_upload_cleanup_error(
            deferred_error,
            process.is_alive,
        )
        dead = checked and alive is False

    if not dead:
        _defer_upload_cleanup_error(deferred_error, print, warning, flush=True)
        os._exit(124)
        raise AssertionError("unreachable")

    deferred_error, _, _ = _defer_upload_cleanup_error(
        deferred_error,
        process.close,
    )
    if deferred_error is not None:
        raise deferred_error
    return clean


def _stop_upload_process(
    process,
    stop_upload,
    upload_deadline_at: float,
    reaping_deadline_at: float,
) -> bool:
    """stop and reap an uploader before terminal-marker bookkeeping."""
    return _cleanup_upload_process(
        process,
        upload_deadline_at,
        _CONSOLE_UPLOAD_STOP_TIMEOUT_S,
        reaping_deadline_at,
        "console uploader remained alive after kill; exiting before terminal bookkeeping",
        stop_upload,
    )


def _start_console_uploader(payload: dict, console: str, mode: str):
    context = multiprocessing.get_context("spawn")
    stop_upload = context.Event()
    process = context.Process(
        target=_console_upload_loop,
        args=(payload, console, mode, _CONSOLE_UPLOAD_INTERVAL_S, stop_upload),
        daemon=True,
    )
    process.start()
    return process, stop_upload


def _upload_console_tail_bounded(
    payload: dict,
    console: str,
    mode: str,
    extra: str,
    upload_deadline_at: float,
    reaping_deadline_at: float,
) -> bool:
    """upload the final console snapshot before terminal-marker bookkeeping."""
    if upload_deadline_at <= _finite_positive_number(time.time(), "current clock"):
        return False
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_upload_console_snapshot,
        args=(payload, console, mode, extra),
        daemon=True,
    )
    process.start()
    return _cleanup_upload_process(
        process,
        upload_deadline_at,
        _CONSOLE_UPLOAD_FINAL_TIMEOUT_S,
        reaping_deadline_at,
        "final console uploader remained alive after kill; exiting before terminal bookkeeping",
    )


def hf_file_exists(payload: dict, repo_subpath: str) -> bool:
    """True iff ``<hf_prefix>/<repo_subpath>`` exists in the run's HF dataset repo. Raises on API error."""
    from huggingface_hub import HfApi

    if "deadline_at" in payload:
        require_deadline_at(payload)
    api = HfApi(token=(payload.get("env") or {}).get("HF_TOKEN"))
    return api.file_exists(
        repo_id=payload["hf_repo"],
        filename=f"{payload['hf_prefix']}/{repo_subpath}",
        repo_type="dataset",
    )


def remote_completion_confirmed(payload: dict) -> bool:
    """True iff DONE + metrics.json are on HF. Local /tmp/metrics.json is not sufficient proof."""
    try:
        return hf_file_exists(payload, "DONE") and hf_file_exists(payload, "metrics.json")
    except Exception as exc:
        # read errors are infra-shaped and leave completion unconfirmed.
        print(f"remote-completion check warn: {_safe_detail(exc)}", flush=True)
        return False


def fetch_spec_from_hf(payload: dict) -> str:
    """Fetch the job spec spilled to HF to avoid blowing the cloud-init user_data size cap."""
    from huggingface_hub import hf_hub_download

    deadline_at = require_deadline_at(payload)
    local = _hf_call(
        lambda: hf_hub_download(
            repo_id=payload["hf_repo"],
            repo_type="dataset",
            filename=f"{payload['hf_prefix']}/job_spec.json",
            token=(payload.get("env") or {}).get("HF_TOKEN"),
        ),
        "download spilled job spec",
        deadline_at=deadline_at,
    )
    with open(local) as f:
        return f.read()


def build_worker_env(payload: dict) -> dict:
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (payload.get("env") or {}).items()})
    spec_json = payload.get("job_spec_json")
    if not spec_json and payload.get("job_spec_in_hf"):
        # Pre-worker fetch; failure is infra-shaped → raise RetriableBootstrapError so poller retries.
        try:
            spec_json = fetch_spec_from_hf(payload)
        except Exception:
            raise RetriableBootstrapError("failed to fetch the spilled job spec from HF") from None
    if not spec_json:
        raise RuntimeError(
            "bootstrap payload carries no job spec: both job_spec_json and the job_spec_in_hf "
            "sentinel are absent/empty — the control plane built an invalid worker payload"
        )
    # Large specs go via a file to avoid execve "Argument list too long".
    if len(spec_json) > 96_000:
        with open("/tmp/job_spec.json", "w") as f:
            f.write(spec_json)
        env["FLASH_JOB_SPEC_PATH"] = "/tmp/job_spec.json"
        env.pop("FLASH_JOB_SPEC_JSON", None)
    else:
        env["FLASH_JOB_SPEC_JSON"] = spec_json
    env["PHASE"] = payload["phase"]
    env["SEED"] = str(payload["seed"])
    # drives the poller's stale-heartbeat rejection across retries.
    attempt = payload.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise RuntimeError("bootstrap attempt identity is invalid")
    env["ATTEMPT"] = str(attempt)
    # Override runpod-stamped FLASH_ARM to the real backend from the payload.
    env["FLASH_ARM"] = _arm(payload)
    code_dir = _code_dir(payload)
    env["PYTHONPATH"] = code_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _extra_pip_env(payload: dict) -> tuple[dict[str, str], str | None]:
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (payload.get("env") or {}).items()})
    env["GIT_TERMINAL_PROMPT"] = "0"
    askpass = None
    if env.get("GITHUB_TOKEN"):
        fd, askpass = tempfile.mkstemp(prefix="flash-github-askpass-", suffix=".sh")
        with os.fdopen(fd, "w") as f:
            f.write(
                "#!/bin/sh\n"
                'case "$1" in\n'
                '*Username*) printf "%s\\n" "x-access-token" ;;\n'
                '*) printf "%s\\n" "$GITHUB_TOKEN" ;;\n'
                "esac\n"
            )
        os.chmod(askpass, 0o700)
        env["GIT_ASKPASS"] = askpass
    return env, askpass


def install_extra_pip(payload: dict) -> None:
    extra_pip = payload.get("extra_pip") or []
    if not extra_pip:
        return
    env, askpass = _extra_pip_env(payload)
    try:
        if "deadline_at" in payload:
            require_deadline_at(payload)
        subprocess.run([sys.executable, "-m", "pip", "install", *extra_pip], check=True, env=env)
    finally:
        if askpass:
            with contextlib.suppress(OSError):
                os.remove(askpass)


def fetch_code(payload: dict) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    deadline_at = require_deadline_at(payload)
    prefix = _code_prefix(payload)
    token = (payload.get("env") or {}).get("HF_TOKEN")
    api = HfApi(token=token)
    files = [
        entry.path
        for entry in _hf_call(
            lambda: list(
                api.list_repo_tree(
                    repo_id=payload["hf_repo"],
                    repo_type="dataset",
                    path_in_repo=prefix,
                    recursive=True,
                    token=token,
                )
            ),
            f"list flash code under {payload['hf_repo']}:{prefix}",
            deadline_at=deadline_at,
        )
        if getattr(entry, "path", None) and getattr(entry, "size", None) is not None
    ]
    if not files:
        raise RuntimeError(f"no flash code files found under {payload['hf_repo']}:{prefix}")
    for filename in files:
        _hf_call(
            lambda filename=filename: hf_hub_download(
                repo_id=payload["hf_repo"],
                repo_type="dataset",
                filename=filename,
                local_dir=CODE_ROOT,
                token=token,
            ),
            f"download flash code file {payload['hf_repo']}:{filename}",
            deadline_at=deadline_at,
        )


def run_mode(payload: dict, env: dict, mode: str, deadline_ts: float) -> int:
    """Run one worker subprocess; tee console to a file and upload periodically for live logs."""
    console = f"/tmp/console_{mode}.txt"
    upload_deadline_at, reaping_deadline_at = _upload_cleanup_deadlines(deadline_ts)
    worker_deadline_at = _worker_execution_deadline(upload_deadline_at)
    timed_out = False

    with open(console, "w", buffering=1) as cf:
        code_dir = _code_dir(payload)
        if worker_deadline_at - _finite_positive_number(time.time(), "current clock") <= 0:
            raise TimeoutError(f"worker mode '{mode}' exceeded the wall-clock cap")
        proc = subprocess.Popen(
            [sys.executable, "-m", "flash.engine.worker_entrypoint"],
            cwd=code_dir,
            env={
                **env,
                "RUN_MODE": mode,
                "FLASH_RUN_DEADLINE_AT": str(worker_deadline_at),
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        pump_done = threading.Event()
        pump_write_lock = threading.Lock()
        pump_writes_enabled = True

        def pump():
            try:
                for line in proc.stdout:
                    with pump_write_lock:
                        if not pump_writes_enabled:
                            return
                        print(line, end="", flush=True)
                        cf.write(line)
            except BaseException as exc:
                print(f"console pump warn: {_safe_detail(exc)}", flush=True)
            finally:
                pump_done.set()

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        uploader, stop_upload = _start_console_uploader(payload, console, mode)
        try:
            try:
                remaining = worker_deadline_at - _finite_positive_number(
                    time.time(), "current clock"
                )
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(proc.args, timeout=0.0)
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                remaining = max(
                    0.0,
                    worker_deadline_at - _finite_positive_number(time.time(), "current clock"),
                )
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=remaining)

            drain_timeout = max(
                0.0,
                worker_deadline_at - _finite_positive_number(time.time(), "current clock"),
            )
            pump_finished = pump_done.wait(drain_timeout)
            if pump_finished:
                t.join()
            else:
                timed_out = True
                with pump_write_lock:
                    pump_writes_enabled = False
        finally:
            uploader_clean = _stop_upload_process(
                uploader,
                stop_upload,
                upload_deadline_at,
                reaping_deadline_at,
            )
            if not uploader_clean:
                print("console uploader exceeded its shutdown allowance; terminated", flush=True)
    try:
        extra = ""
        if timed_out:
            extra = f"\n--- bootstrap: mode '{mode}' hit the wall-clock cap; killed ---\n"
        if upload_deadline_at <= _finite_positive_number(time.time(), "current clock"):
            print("final console upload skipped; uploader reaping reserve reached", flush=True)
        elif not _upload_console_tail_bounded(
            payload,
            console,
            mode,
            extra,
            upload_deadline_at,
            reaping_deadline_at,
        ):
            print("final console upload exceeded its allowance", flush=True)
    except Exception as exc:
        print(f"console upload warn: {_safe_detail(exc)}", flush=True)
    if timed_out:
        raise TimeoutError(f"worker mode '{mode}' exceeded the wall-clock cap")
    return proc.returncode


def write_attempt_marker(payload: dict, ok: bool, error: str = "", retriable: bool = False) -> None:
    """Upload one identity-bound terminal marker using the strict poller schema."""
    attempt = payload.get("attempt")
    run_id = payload.get("run_id")
    if (
        isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or not 0 <= attempt <= _MAX_ATTEMPT_ID
    ):
        raise RuntimeError("attempt marker identity is invalid")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("attempt marker identity is invalid")
    if type(ok) is not bool or type(retriable) is not bool:
        raise RuntimeError("attempt marker state is invalid")
    if not isinstance(error, str):
        raise RuntimeError("attempt marker error is invalid")
    _canonical_deadline_at(payload)
    now = _finite_positive_number(time.time(), "current clock")
    marker = {
        "attempt": attempt,
        "error": _safe_detail(error, 2000),
        "ok": ok,
        "retriable": retriable,
        "run_id": run_id,
        "ts": now,
    }
    p = "/tmp/attempt_marker.json"
    with open(p, "w") as f:
        json.dump(marker, f)
    hf_upload(
        payload,
        p,
        f"{_arm(payload)}_attempt{attempt}.json",
        enforce_deadline=False,
    )


def _arm_preload_wall_cap(payload: dict) -> tuple[threading.Timer, threading.Event]:
    """Arm the absolute run-deadline watchdog around in-process snapshot downloads."""
    deadline_at = require_deadline_at(payload)
    remaining = deadline_at - time.time()
    # done is set by the caller on clean finish; _fire checks it first to avoid a racing false alarm.
    done = threading.Event()

    def _fire() -> None:
        if done.is_set():
            return
        _publish_timeout_marker_then_exit(
            payload,
            "preload exceeded the run wall deadline; self-terminating box",
        )

    timer = threading.Timer(max(0.0, remaining), _fire)
    timer.daemon = True
    timer.start()
    return timer, done


def run_preload(payload: dict) -> dict:
    """Download models into the bind-mounted weight cache. Fails if cache isn't mounted to avoid warming ephemeral disk."""
    env = payload.get("env") or {}
    cache_dir = env.get("FLASH_WEIGHT_CACHE_DIR") or ""
    token = env.get("HF_TOKEN")
    mount = os.path.dirname(os.path.dirname(cache_dir.rstrip("/"))) if cache_dir else ""
    if not cache_dir or not mount or not os.path.isdir(mount):
        return {
            "preloaded": [],
            "already_cached": [],
            "failed": {},
            "error": f"weight-cache not mounted (FLASH_WEIGHT_CACHE_DIR={cache_dir!r}); refusing to warm ephemeral disk",
        }
    # Sentinel written by cloud-init only onto a real mount; absent sentinel means Docker silently
    # auto-created an empty host dir (isdir passes) — we'd warm ephemeral disk. Must check.
    if payload.get("cache_mount_marker"):
        marker = os.path.join(mount, payload["cache_mount_marker"])
        if not os.path.exists(marker):
            return {
                "preloaded": [],
                "already_cached": [],
                "failed": {},
                "error": (
                    f"weight-cache NFS filesystem not mounted (no sentinel at {marker}); "
                    "refusing to warm ephemeral disk"
                ),
            }
    from huggingface_hub import snapshot_download

    ignore_patterns = ["*.pth", "*.gguf", "original/*", "*.onnx", "*.msgpack", "*.h5"]
    done, already, failed = [], [], {}
    for repo_id in payload.get("models") or []:
        try:
            # Probe with local_files_only first (HF's own resolution, not a dir-name guess).
            try:
                snapshot_download(
                    repo_id=repo_id,
                    token=token,
                    cache_dir=cache_dir,
                    ignore_patterns=ignore_patterns,
                    local_files_only=True,
                )
                already.append(repo_id)
                print(f"preload: {repo_id} -> {cache_dir} (cached)", flush=True)
                continue
            except Exception:
                pass
            snapshot_download(
                repo_id=repo_id, token=token, cache_dir=cache_dir, ignore_patterns=ignore_patterns
            )
            done.append(repo_id)
            print(f"preload: {repo_id} -> {cache_dir} (downloaded)", flush=True)
        except Exception as exc:
            failed[repo_id] = _safe_detail(exc)
            print(f"preload FAILED {repo_id}: {_safe_detail(exc)}", flush=True)
    return {"preloaded": done, "already_cached": already, "failed": failed}


def main() -> int:
    # SIGTERM → sys.exit so the finally block still uploads the terminal marker.
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(1))
    payload = load_payload()
    ok = False
    error = ""
    retriable = False
    deadline_watchdog = None
    try:
        try:
            import importlib.util

            if importlib.util.find_spec("hf_transfer") is not None:
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        except Exception as exc:
            print("hf_transfer setup skipped:", exc, flush=True)
        deadline = require_deadline_at(payload)
        if payload.get("mode") == "preload":
            deadline_watchdog = _arm_preload_wall_cap(payload)
            result = run_preload(payload)
            with open("/tmp/preload_result.json", "w") as f:
                json.dump(result, f)
            # preload_result.json is the completion signal the warm driver polls.
            confirmed = False
            for attempt in range(3):
                remaining = deadline - _finite_positive_number(time.time(), "current clock")
                if remaining <= 0:
                    break
                hf_upload(payload, "/tmp/preload_result.json", "preload_result.json")
                try:
                    confirmed = hf_file_exists(payload, "preload_result.json")
                except Exception as exc:
                    print(
                        f"preload_result.json upload confirm warn: {_safe_detail(exc)}",
                        flush=True,
                    )
                if confirmed:
                    break
                if attempt < 2:
                    remaining = deadline - _finite_positive_number(time.time(), "current clock")
                    if remaining <= 0:
                        break
                    time.sleep(min(2.0 * (attempt + 1), remaining))
            if not confirmed:
                print(
                    "preload_result.json upload FAILED before the run deadline "
                    "(driver falls back to the attempt marker)",
                    flush=True,
                )
            ok = not result.get("error") and not result.get("failed")
            error = "model preload failed" if not ok else ""
            return 0 if ok else 1
        deadline_watchdog = arm_deadline_watchdog(deadline, payload)
        install_extra_pip(payload)
        # Pre-worker HF fetch of the run's own code (control plane uploaded it before submit), same
        # infra-shaped class as fetch_spec_from_hf above: a transient HF blip must retry, not fail.
        try:
            fetch_code(payload)
        except Exception:
            raise RetriableBootstrapError("failed to fetch run code from HF") from None
        env = build_worker_env(payload)
        phase = payload["phase"]
        for stale in ("/tmp/train_meta.json", "/tmp/metrics.json"):
            with contextlib.suppress(FileNotFoundError):
                os.remove(stale)
        rc = run_mode(payload, env, phase, deadline)
        # run_mode enforces the absolute subprocess deadline. once it returns, required completion
        # artifacts take precedence over bootstrap bookkeeping that happens to cross the boundary.
        if not os.path.exists("/tmp/metrics.json"):
            # Missing local metrics but the run is confirmed complete on HF (DONE+metrics uploaded) —
            # e.g. the idempotency replay hit a transient HF read. The run SUCCEEDED; retry so a fresh
            # worker re-fetches the metrics, never fail a confirmed-complete run as a crash.
            if remote_completion_confirmed(payload):
                raise RetriableBootstrapError(
                    f"train phase '{phase}' is complete on HF but its local metrics.json is missing "
                    f"(transient HF read); retrying to re-fetch the persisted metrics"
                )
            raise RuntimeError(
                f"train phase '{phase}' produced no /tmp/metrics.json (it crashed before "
                f"finishing); see error_{phase}_attempt*.txt and console_{phase}.txt in the HF "
                f"dataset repo"
            )
        # Non-zero rc is tolerated only when completion artifacts landed on HF: RL's vLLM can
        # segfault at interpreter exit AFTER DONE/metrics.json are already uploaded.
        if rc != 0 and not remote_completion_confirmed(payload):
            raise RetriableBootstrapError(
                f"train phase '{phase}' exited non-zero ({rc}) and its required completion "
                f"artifacts (DONE/metrics.json) are not on HF — the run did not finish (e.g. a "
                f"failed upload after the local metrics.json was written); see "
                f"error_{phase}_attempt*.txt and console_{phase}.txt in the HF dataset repo"
            )
        ok = True
    except BaseException as exc:  # incl. SIGTERM's SystemExit / KeyboardInterrupt
        retriable = isinstance(exc, RetriableBootstrapError)
        detail = _safe_detail(exc, 1800)
        error = f"run wall deadline exceeded: {detail}" if isinstance(exc, TimeoutError) else detail
        print(f"bootstrap failed: {error}", flush=True)
    finally:
        write_attempt_marker(payload, ok, error, retriable=retriable)
        if deadline_watchdog is not None:
            deadline_timer, deadline_done = deadline_watchdog
            deadline_done.set()
            deadline_timer.cancel()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
