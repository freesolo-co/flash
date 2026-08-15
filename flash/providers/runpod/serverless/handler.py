"""The source-shipped RunPod serverless worker handler."""

from __future__ import annotations

_CONSOLE_UPLOAD_INTERVAL_S = 3600.0
_CONSOLE_UPLOAD_FIRST_SNAPSHOT_S = 600.0
_CONSOLE_UPLOAD_POLL_S = 120.0


def _train_body(input_data: dict) -> dict:
    """Run on the RunPod GPU worker: fetch the exact code snapshot, train, and return metrics."""
    import collections
    import contextlib
    import importlib.util
    import json
    import math
    import os
    import re
    import subprocess
    import sys
    import tempfile
    import threading
    import time
    from datetime import UTC, datetime
    from email.utils import parsedate_to_datetime

    from huggingface_hub import snapshot_download

    def _needles(secrets=None):
        """Return credential needles from the process environment and supplied mapping."""
        import urllib.parse

        mapping = {**os.environ, **(secrets or {})}
        declared = {
            name.strip().upper()
            for name in str(mapping.get("FLASH_SECRET_ENV_KEYS") or "").split(",")
            if name.strip()
        }
        plain, bounded = set(), set()
        for key, secret in mapping.items():
            upper = str(key).upper()
            if not secret or not (
                upper in {"AUTHORIZATION", "HF_TOKEN"}
                or upper in declared
                or upper.endswith(("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD"))
            ):
                continue
            value_str = str(secret)
            target = plain if len(value_str) >= 8 else bounded
            target.update({value_str, urllib.parse.quote(value_str, safe="")})
            if "\n" in value_str:
                for raw in value_str.splitlines():
                    if len(line := raw.strip()) >= 8:
                        plain.update({line, urllib.parse.quote(line, safe="")})
        return plain, bounded

    def _safe_detail(value, secrets=None, limit=1000):
        text = (
            f"{type(value).__name__}: {value}" if isinstance(value, BaseException) else str(value)
        )
        plain, bounded = _needles(secrets)
        for needle in sorted(plain, key=len, reverse=True):
            text = text.replace(needle, "<redacted>")
        for needle in sorted(bounded, key=len, reverse=True):
            left = r"(?<!\w)" if needle[:1].isalnum() or needle[:1] == "_" else ""
            right = r"(?!\w)" if needle[-1:].isalnum() or needle[-1:] == "_" else ""
            text = re.sub(f"{left}{re.escape(needle)}{right}", "<redacted>", text)
        text = re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", text)
        text = re.sub(
            r"(?i)(authorization|api[-_ ]?key|access[-_ ]?token|token|secret|password)"
            r"(\s*[:=]\s*)(?:bearer\s+)?([^\s,;]+)",
            lambda match: f"{match.group(1)}{match.group(2)}<redacted>",
            text,
        )
        return text[:limit]

    def _preload() -> dict:
        overrides = {key: str(value) for key, value in (input_data.get("env") or {}).items()}
        os.environ.update(overrides)
        token = overrides.get("HF_TOKEN")
        hf_home = overrides.get("HF_HOME")
        if not hf_home or not hf_home.startswith("/runpod-volume"):
            return {
                "preloaded": [],
                "already_cached": [],
                "failed": {},
                "error": f"preload requires HF_HOME rooted at /runpod-volume (got HF_HOME={hf_home!r})",
                "hf_home": hf_home,
            }
        if not os.path.isdir("/runpod-volume"):
            return {
                "preloaded": [],
                "already_cached": [],
                "failed": {},
                "error": f"weight-cache volume not mounted at /runpod-volume (HF_HOME={hf_home})",
                "hf_home": hf_home,
            }
        cache_dir = os.path.join(hf_home, "hub")
        ignore_patterns = ["*.pth", "*.gguf", "original/*", "*.onnx", "*.msgpack", "*.h5"]
        done, already, failed = [], [], {}
        for repo_id in input_data.get("models") or []:
            try:
                try:
                    snapshot_download(
                        repo_id=repo_id,
                        token=token,
                        cache_dir=cache_dir,
                        ignore_patterns=ignore_patterns,
                        local_files_only=True,
                    )
                    already.append(repo_id)
                    continue
                except Exception:
                    pass
                snapshot_download(
                    repo_id=repo_id,
                    token=token,
                    cache_dir=cache_dir,
                    ignore_patterns=ignore_patterns,
                )
                done.append(repo_id)
            except Exception as exc:
                failed[repo_id] = _safe_detail(exc, overrides, 300)
        return {
            "preloaded": done,
            "already_cached": already,
            "failed": failed,
            "hf_home": os.environ.get("HF_HOME"),
        }

    def _require_deadline_allowance() -> float:
        now = time.time()
        if not math.isfinite(now) or now <= 0:
            raise RuntimeError("run wall deadline clock is invalid")
        remaining_seconds = deadline_at - now
        if remaining_seconds <= 0:
            raise TimeoutError("run wall deadline exceeded")
        return remaining_seconds

    def _deadline_exit() -> None:
        print("run wall deadline exceeded; terminating worker", flush=True)
        os._exit(124)

    def _extra_pip_env(overrides: dict[str, str]) -> tuple[dict[str, str], str | None]:
        env = dict(os.environ)
        env.update(overrides)
        env["GIT_TERMINAL_PROMPT"] = "0"
        askpass = None
        if env.get("GITHUB_TOKEN"):
            fd, askpass = tempfile.mkstemp(prefix="flash-github-askpass-", suffix=".sh")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(
                    "#!/bin/sh\n"
                    'case "$1" in\n'
                    '*Username*) printf "%s\\n" "x-access-token" ;;\n'
                    '*) printf "%s\\n" "$GITHUB_TOKEN" ;;\n'
                    "esac\n"
                )
            os.chmod(askpass, 0o700)
            env["GIT_ASKPASS"] = askpass
        return env, askpass

    def _install_extra_pip(overrides: dict[str, str]) -> None:
        extra_pip = input_data.get("extra_pip") or []
        if not extra_pip:
            return
        transient = re.compile(
            r"(?i)connection (?:broken|reset|aborted|refused|timed out)|read timed out"
            r"|temporary failure in name resolution|failed to establish a new connection"
            r"|network is unreachable|remote end closed connection|incompleteread|proxyerror"
            r"|newconnectionerror|maxretryerror|ssleoferror|service unavailable|bad gateway"
            r"|gateway time-?out|too many requests|retrying \(retry\("
            r"|\b(?:429|5\d\d) (?:client|server) error"
            r"|returned error: (?:429|5\d\d)|could not resolve (?:host|proxy)"
        )
        terminal = re.compile(
            r"(?i)failed building wheel|metadata-generation-failed|could not build wheels"
            r"|no matching distribution|could not find a version|resolutionimpossible"
            r"|invalid requirement"
        )
        no_candidate = re.compile(r"(?i)no matching distribution|could not find a version")

        def _is_terminal(output: str) -> bool:
            if not terminal.search(output):
                return not transient.search(output)
            if not transient.search(output):
                return True
            return bool(terminal.search(no_candidate.sub("", output)))

        retry_delays = (3.0, 9.0, 27.0)
        extra_env, askpass = _extra_pip_env(overrides)
        args = [sys.executable, "-m", "pip", "install", *extra_pip]
        try:
            for attempt in range(len(retry_delays) + 1):
                _require_deadline_allowance()
                tail = collections.deque(maxlen=400)
                process = subprocess.Popen(
                    args,
                    env=extra_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                )
                try:
                    with process.stdout:
                        for line in process.stdout:
                            tail.append(line)
                            with contextlib.suppress(OSError, ValueError):
                                print(line, end="", flush=True)
                    returncode = process.wait()
                except BaseException:
                    process.kill()
                    process.wait()
                    raise
                if returncode == 0:
                    break
                output = "".join(tail)
                if _is_terminal(output):
                    raise RuntimeError(f"extra_pip install failed: pip exited {returncode}")
                if attempt >= len(retry_delays):
                    raise RuntimeError(
                        "extra_pip install could not reach the package index after "
                        f"{attempt + 1} attempts (pip exited {returncode})"
                    )
                delay = max(
                    0.0,
                    min(retry_delays[attempt], _require_deadline_allowance() - 1.0),
                )
                with contextlib.suppress(OSError, ValueError):
                    print(
                        f"extra_pip install hit a transient index error; retrying in {delay:.0f}s",
                        flush=True,
                    )
                if delay > 0:
                    time.sleep(delay)
        finally:
            if askpass:
                with contextlib.suppress(OSError):
                    os.remove(askpass)

    def _code_prefix() -> str:
        raw = input_data.get("code_prefix")
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
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(f"invalid code_prefix: {prefix!r}")
        return prefix

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
        return min(60.0, max(0.0, seconds))

    def _hf_call(call, label: str, overrides: dict[str, str]):
        retry_delays = (1.0, 3.0, 8.0, 20.0, 60.0)
        transient_status_codes = {429, 500, 502, 503, 504}
        for attempt in range(len(retry_delays) + 1):
            _require_deadline_allowance()
            try:
                return call()
            except Exception as exc:
                if _hf_status_code(exc) not in transient_status_codes or attempt >= len(
                    retry_delays
                ):
                    raise
                retry_after = _hf_retry_after(exc)
                delay = retry_after if retry_after is not None else retry_delays[attempt]
                try:
                    delay = min(delay, _require_deadline_allowance())
                except TimeoutError:
                    raise exc from None
                print(
                    f"{label} transient Hugging Face error; retrying in {delay:.0f}s: "
                    f"{_safe_detail(exc, overrides, 500)}",
                    flush=True,
                )
                time.sleep(delay)
                try:
                    _require_deadline_allowance()
                except TimeoutError:
                    raise exc from None
        raise AssertionError("unreachable")

    def _download_code_prefix(
        repo_id: str,
        prefix: str,
        token: str | None,
        overrides: dict[str, str],
    ) -> None:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi(token=token)
        files = [
            entry.path
            for entry in _hf_call(
                lambda: list(
                    api.list_repo_tree(
                        repo_id=repo_id,
                        repo_type="dataset",
                        path_in_repo=prefix,
                        recursive=True,
                        token=token,
                    )
                ),
                f"list flash code under {repo_id}:{prefix}",
                overrides,
            )
            if getattr(entry, "path", None) and getattr(entry, "size", None) is not None
        ]
        if not files:
            raise RuntimeError(f"no flash code files found under {repo_id}:{prefix}")
        for filename in files:
            _hf_call(
                lambda filename=filename: hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=filename,
                    local_dir="/runcode",
                    token=token,
                ),
                f"download flash code file {repo_id}:{filename}",
                overrides,
            )

    def _load_exact_module(code_dir: str, relative_path: str, name: str):
        path = os.path.join(code_dir, relative_path)
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load downloaded module: {relative_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _prepare_code_and_env(overrides: dict[str, str]):
        prefix = _code_prefix()
        _download_code_prefix(input_data["hf_repo"], prefix, overrides.get("HF_TOKEN"), overrides)
        code_dir = os.path.join("/runcode", os.path.dirname(prefix) or ".")
        console_module = _load_exact_module(
            code_dir,
            "flash/providers/_lifecycle/bootstrap_console.py",
            "_flash_downloaded_bootstrap_console",
        )
        artifact_module = _load_exact_module(
            code_dir,
            "flash/adapters/artifacts.py",
            "_flash_downloaded_artifacts",
        )
        env = dict(os.environ)
        env.update(overrides)
        if not os.path.isdir("/runpod-volume"):
            for key in [
                key for key, value in env.items() if str(value).startswith("/runpod-volume")
            ]:
                env.pop(key, None)
        spec_path = "/tmp/job_spec.json"
        with open(spec_path, "w") as handle:
            handle.write(input_data["job_spec_json"])
        env["FLASH_JOB_SPEC_PATH"] = spec_path
        env.pop("FLASH_JOB_SPEC_JSON", None)
        env["PHASE"] = input_data["phase"]
        env["SEED"] = str(input_data["seed"])
        env["PYTHONPATH"] = code_dir + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        return code_dir, env, console_module, artifact_module.attempt_scoped_artifact_name

    def _read_console_tail(console: str) -> str:
        with open(console, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            start = max(0, handle.tell() - 64_000)
            handle.seek(max(0, start - 1))
            raw = handle.read()
        if start == 0:
            return raw.decode("utf-8", "replace")
        tail = raw[1:].decode("utf-8", "replace")
        if raw[:1] == b"\n":
            return tail
        cut = tail.find("\n")
        return tail[cut + 1 :] if cut >= 0 else ""

    if input_data.get("mode") == "preload":
        return _preload()

    raw_deadline = input_data.get("deadline_at")
    if isinstance(raw_deadline, bool) or not isinstance(raw_deadline, (int, float)):
        raise RuntimeError("run wall deadline is invalid")
    deadline_at = float(raw_deadline)
    if not math.isfinite(deadline_at) or deadline_at <= 0:
        raise RuntimeError("run wall deadline is invalid")
    try:
        remaining = _require_deadline_allowance()
    except TimeoutError:
        raise RuntimeError("run wall deadline exceeded before bootstrap") from None
    deadline_timer = threading.Timer(remaining, _deadline_exit)
    deadline_timer.daemon = True
    deadline_timer.start()

    try:
        overrides = {key: str(value) for key, value in (input_data.get("env") or {}).items()}
        overrides["FLASH_RUN_DEADLINE_AT"] = str(deadline_at)
        _install_extra_pip(overrides)
        code_dir, env, console_module, attempt_scoped_artifact_name = _prepare_code_and_env(
            overrides
        )
        console_teardown = threading.Event()

        def _commit_console_snapshot(mode: str, console: str, tail_path: str, final: bool) -> bool:
            try:
                from huggingface_hub import HfApi

                _require_deadline_allowance()
                spec = json.loads(input_data["job_spec_json"])
                phase_ns = "rl" if spec.get("algorithm") == "grpo" else spec["algorithm"]
                prefix = f"{phase_ns}/{spec['run_id']}"
                tail = _read_console_tail(console)
                with open(tail_path, "w", encoding="utf-8", errors="replace") as handle:
                    handle.write(_safe_detail(tail, env, 64_000))
                _require_deadline_allowance()
                if not final and console_teardown.is_set():
                    print(f"console upload dropped for {mode}; terminal snapshot has begun")
                    return False
                attempt = int(env.get("ATTEMPT") or 0)
                artifact = (
                    f"console_{mode}.txt"
                    if final
                    else attempt_scoped_artifact_name("console", mode, attempt)
                )
                HfApi(token=env.get("HF_TOKEN")).upload_file(
                    path_or_fileobj=tail_path,
                    path_in_repo=f"{prefix}/{artifact}",
                    repo_id=input_data["hf_repo"],
                    repo_type="dataset",
                )
                return True
            except Exception as exc:
                print("console upload warn:", _safe_detail(exc, env))
                return False

        def _upload_console(mode: str, final: bool = False) -> bool:
            console = f"/tmp/console_{mode}.txt"
            if not os.path.exists(console):
                return False
            if final:
                console_teardown.set()
            elif console_teardown.is_set():
                return False
            suffix = ".final.tail" if final else ".live.tail"
            return _commit_console_snapshot(mode, console, console + suffix, final)

        def run_mode(mode: str, check: bool) -> int:
            """Run the worker subprocess, stream its console, and upload live and terminal tails."""
            console = f"/tmp/console_{mode}.txt"
            stop_upload = threading.Event()

            def _upload_live() -> bool:
                return _upload_console(mode)

            with open(console, "w", buffering=1) as console_file:
                _require_deadline_allowance()
                process = subprocess.Popen(
                    [sys.executable, "-m", "flash.engine.worker_entrypoint"],
                    cwd=code_dir,
                    env={**env, "RUN_MODE": mode},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                )
                uploader = threading.Thread(
                    target=console_module._run_console_upload_loop,
                    args=(console, 3600.0, stop_upload),
                    kwargs={"upload": _upload_live},
                    daemon=True,
                )
                uploader.start()
                try:
                    for line in process.stdout:
                        print(_safe_detail(line, env, 100_000), end="")
                        console_file.write(line)
                    process.wait()
                finally:
                    stop_upload.set()
                    uploader.join(timeout=10)
            _upload_console(mode, final=True)
            if process.returncode != 0 and check:
                raise RuntimeError(
                    f"worker mode '{mode}' exited {process.returncode}; see console_{mode}.txt "
                    f"and error_{mode}_attempt*.txt in the HF dataset repo"
                )
            return process.returncode

        for stale in ("/tmp/train_meta.json", "/tmp/metrics.json"):
            with contextlib.suppress(FileNotFoundError):
                os.remove(stale)
        run_mode(input_data["phase"], check=False)
        if not os.path.exists("/tmp/metrics.json"):
            phase = input_data["phase"]
            _upload_console(phase, final=True)
            raise RuntimeError(
                f"train phase '{phase}' produced no /tmp/metrics.json (it crashed before "
                f"finishing); see error_{phase}_attempt*.txt and console_{phase}.txt in the HF "
                "dataset repo for the full traceback"
            )
        with open("/tmp/metrics.json") as handle:
            return json.load(handle)
    finally:
        deadline_timer.cancel()
