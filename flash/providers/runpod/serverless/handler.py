"""The RunPod serverless worker handler.

Split from ``endpoints`` so the endpoint lifecycle stays scannable: this module is 100% the one
function RunPod executes, and that function is exempt from the function-size gate because only its
SOURCE ships (``build_function_input`` -> ``get_function_source``). Keeping it alone here means the
exemption covers a file whose entire purpose is that transport boundary, and the lifecycle code no
longer shares a size budget with it.

Every name ``_train_body`` uses must be imported INSIDE its body: module-level imports here are out
of scope on the worker. ``test_train_body_imports_every_name_it_uses`` pins that from the other side.
The console-upload constants below are read by the tests that pin the shipped literals against
drift; the handler itself inlines the numbers for the same reason.
"""

from __future__ import annotations

_CONSOLE_UPLOAD_INTERVAL_S = 3600.0
# an interval-only uploader can never fire for a wedged run: teardown comes at 1200s (training stall)
# or 3000s (setup grace), both under the hour. one early snapshot, then the hourly cadence -- one
# extra commit per RUN, not per hour, so the repo's rate budget is unchanged.
_CONSOLE_UPLOAD_FIRST_SNAPSHOT_S = 600.0
# how often the uploader LOOKS at the console, not how often it commits: reading is free against the
# commit budget, and it is what notices a wedge before the next hourly boundary.
_CONSOLE_UPLOAD_POLL_S = 120.0


def _train_body(input_data: dict) -> dict:
    """Runs ON the RunPod GPU worker: fetch code, train, return metrics.

    All imports must be inside the function body — this handler is serialized standalone.
    """
    import collections
    import contextlib
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
        """The (plain, bounded) credential needle sets for os.environ plus ``secrets``.

        a value at or above the floor is a plain needle, replaced as a substring; a SHORTER one is
        bounded, matched only where it is not adjacent to a word character. short values used to be
        dropped outright, which leaked them verbatim -- [environment] secrets accepts any name and
        any value. plain replacement is not the alternative: a 3-char needle corrupts every
        diagnostic that merely contains those letters (the value "ati" rewrites "authentication").

        a multiline secret (a PEM key) never appears whole in any single call: the child's stdout is
        sanitized one line at a time, so only a component line is ever seen. component lines keep
        the floor as a hard skip -- a short one is punctuation such as "}", not a credential.
        Mirrors flash.providers._lifecycle.bootstrap_secrets._needles.
        """
        import urllib.parse

        mapping = {**os.environ, **(secrets or {})}
        # declared runtime secrets can carry any name, so the control plane lists them in
        # FLASH_SECRET_ENV_KEYS; the name-shape rule stays as the fail-closed fallback.
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
        # longest-first so one secret containing another cannot leave a suffix of the longer
        # one behind; encoded forms cover the percent-encoded urls http and git errors print.
        for needle in sorted(plain, key=len, reverse=True):
            text = text.replace(needle, "<redacted>")
        for needle in sorted(bounded, key=len, reverse=True):
            # the word guard is applied per EDGE, and only where the needle's own edge is a word
            # character. a value with a punctuation edge already separates itself from neighbouring
            # text, and demanding a non-word character beyond it asks the wrong question: "/a"
            # inside "https://host/a/repo" is preceded by the "t" of "host", so an unconditional
            # left guard fails and the secret prints verbatim. "ati" keeps both guards and so still
            # cannot rewrite "authentication".
            # Mirrors flash.providers._lifecycle.bootstrap_secrets._bounded_pattern.
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

    if input_data.get("mode") == "preload":
        overrides = {k: str(v) for k, v in (input_data.get("env") or {}).items()}
        os.environ.update(overrides)
        tok = overrides.get("HF_TOKEN")
        # CRITICAL: huggingface_hub froze HF_HUB_CACHE at import time, so pass cache_dir
        # explicitly; os.environ update above is ignored by snapshot_download.
        hf_home = overrides.get("HF_HOME")
        # Refuse a preload not rooted at /runpod-volume — HF_HOME elsewhere means nothing
        # gets persisted to the volume (phantom warm).
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
        # Inlined (handler is baked standalone); keep in sync with worker prefetch exclusions.
        ignore_patterns = ["*.pth", "*.gguf", "original/*", "*.onnx", "*.msgpack", "*.h5"]
        done, already, failed = [], [], {}
        for repo_id in input_data.get("models") or []:
            try:
                try:
                    snapshot_download(
                        repo_id=repo_id,
                        token=tok,
                        cache_dir=cache_dir,
                        ignore_patterns=ignore_patterns,
                        local_files_only=True,
                    )
                    already.append(repo_id)
                    continue
                except Exception:
                    pass
                snapshot_download(
                    repo_id=repo_id, token=tok, cache_dir=cache_dir, ignore_patterns=ignore_patterns
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

    raw_deadline = input_data.get("deadline_at")
    if isinstance(raw_deadline, bool) or not isinstance(raw_deadline, (int, float)):
        raise RuntimeError("run wall deadline is invalid")
    deadline_at = float(raw_deadline)
    if not math.isfinite(deadline_at) or deadline_at <= 0:
        raise RuntimeError("run wall deadline is invalid")

    def _require_deadline_allowance() -> float:
        now = time.time()
        if not math.isfinite(now) or now <= 0:
            raise RuntimeError("run wall deadline clock is invalid")
        remaining_seconds = deadline_at - now
        if remaining_seconds <= 0:
            raise TimeoutError("run wall deadline exceeded")
        return remaining_seconds

    try:
        remaining = _require_deadline_allowance()
    except TimeoutError:
        raise RuntimeError("run wall deadline exceeded before bootstrap") from None

    def _deadline_exit() -> None:
        print("run wall deadline exceeded; terminating worker", flush=True)
        os._exit(124)

    deadline_timer = threading.Timer(remaining, _deadline_exit)
    deadline_timer.daemon = True
    deadline_timer.start()

    try:
        overrides = {k: str(v) for k, v in (input_data.get("env") or {}).items()}
        overrides["FLASH_RUN_DEADLINE_AT"] = str(deadline_at)

        def _extra_pip_env() -> tuple[dict[str, str], str | None]:
            env = dict(os.environ)
            env.update(overrides)
            env["GIT_TERMINAL_PROMPT"] = "0"
            askpass = None
            if env.get("GITHUB_TOKEN"):
                fd, askpass = tempfile.mkstemp(prefix="flash-github-askpass-", suffix=".sh")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
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

        extra_pip = input_data.get("extra_pip") or []
        if extra_pip:
            # Network/index-shaped pip failures. A resolution failure ("no matching distribution",
            # an unsatisfiable pin) reaches the index fine and carries NONE of these, so a bad
            # package spec still fails fast; only a PyPI blip retries (as the instance bootstrap).
            pip_transient_re = re.compile(
                r"(?i)connection (?:broken|reset|aborted|refused|timed out)|read timed out"
                r"|temporary failure in name resolution|failed to establish a new connection"
                r"|network is unreachable|remote end closed connection|incompleteread|proxyerror"
                r"|newconnectionerror|maxretryerror|ssleoferror|service unavailable|bad gateway"
                r"|gateway time-?out|too many requests|retrying \(retry\("
                r"|\b(?:429|5\d\d) (?:client|server) error"
                # a VCS pin fails through git, not urllib, so its blips carry git's own phrasing
                # and none of the shapes above: git says "could not resolve host" where urllib
                # says "temporary failure in name resolution", and reports an http status as
                # "returned error: NNN". On the status form, only 429/5xx: a 404 or 403 is a bad
                # pin or a missing token and must still fail fast rather than burn three backoffs.
                r"|returned error: (?:429|5\d\d)|could not resolve (?:host|proxy)"
            )
            # Build/resolution failures, which name the cause and outrank a transient warning pip
            # already recovered from in the same tail; without that precedence one early "Retrying
            # (Retry(" makes a deterministic failure look retriable and this ladder repeats it for
            # nothing. Kept identical to the instance bootstrap's _PIP_TERMINAL_RE: the two
            # classifiers must agree on what is retriable, including excluding the bare
            # subprocess-exited-with-error marker a network-interrupted VCS `git clone` also prints.
            pip_terminal_re = re.compile(
                r"(?i)failed building wheel|metadata-generation-failed|could not build wheels"
                r"|no matching distribution|could not find a version|resolutionimpossible"
                r"|invalid requirement"
            )
            # The subset pip can print having downloaded NOTHING: an unreachable index yields no
            # candidate versions, so it finishes with exactly the footer a typo'd name produces.
            # When that footer is the only terminal evidence and the tail also carries a transient
            # marker, the network explains it and the run retries. Mirrors the bootstrap's
            # _PIP_NO_CANDIDATE_RE / _is_terminal.
            pip_no_candidate_re = re.compile(
                r"(?i)no matching distribution|could not find a version"
            )

            def _pip_is_terminal(output: str) -> bool:
                if not pip_terminal_re.search(output):
                    return not pip_transient_re.search(output)
                if not pip_transient_re.search(output):
                    return True
                # a build or resolver failure surviving the footer strip proves pip held real
                # content, so it stays deterministic; nothing left means the outage explains it.
                return bool(pip_terminal_re.search(pip_no_candidate_re.sub("", output)))

            pip_retry_delays = (3.0, 9.0, 27.0)
            # held back from a deadline-clamped backoff so the retry it precedes has wall to run in
            _PIP_RETRY_RESERVE_S = 1.0
            extra_env, askpass = _extra_pip_env()
            args = [sys.executable, "-m", "pip", "install", *extra_pip]
            try:
                for pip_attempt in range(len(pip_retry_delays) + 1):
                    _require_deadline_allowance()
                    tail = collections.deque(maxlen=400)
                    # errors="replace": a build or VCS child can emit bytes invalid under the
                    # container's locale, and strict decoding raises mid-stream, failing a paid
                    # run whose install actually succeeded.
                    pip_proc = subprocess.Popen(
                        args,
                        env=extra_env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        errors="replace",
                    )
                    try:
                        with pip_proc.stdout:  # tee so a long install streams into the console
                            for line in pip_proc.stdout:
                                tail.append(line)
                                # best-effort: a closed console must not end the drain, or pip is
                                # left running while the askpass helper below is deleted and the
                                # console error is reported in place of pip's own exit status.
                                with contextlib.suppress(OSError, ValueError):
                                    print(line, end="", flush=True)
                        rc = pip_proc.wait()
                    except BaseException:  # never orphan a running pip on a paid box
                        pip_proc.kill()
                        pip_proc.wait()
                        raise
                    if rc == 0:
                        break
                    pip_output = "".join(tail)
                    if _pip_is_terminal(pip_output):
                        raise RuntimeError(f"extra_pip install failed: pip exited {rc}")
                    if pip_attempt >= len(pip_retry_delays):
                        raise RuntimeError(
                            f"extra_pip install could not reach the package index after "
                            f"{pip_attempt + 1} attempts (pip exited {rc})"
                        )
                    # reserve a slice for the attempt this backoff precedes: clamping to the
                    # remaining wall alone sleeps the whole window, so the retry just announced
                    # never issues and the next pass only fails the deadline precheck.
                    delay = max(
                        0.0,
                        min(
                            pip_retry_delays[pip_attempt],
                            _require_deadline_allowance() - _PIP_RETRY_RESERVE_S,
                        ),
                    )
                    # best-effort like the tee above: a console that closed between attempts must
                    # not end the install with a terminal console error, losing the retry this
                    # line only announces.
                    with contextlib.suppress(OSError, ValueError):
                        print(
                            f"extra_pip install hit a transient index error; "
                            f"retrying in {delay:.0f}s",
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
                or any(c not in "0123456789abcdef" for c in digest)
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

        def _hf_call(call, label: str):
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

        def _download_code_prefix(repo_id: str, prefix: str, token: str | None) -> None:
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
                )

        code_prefix = _code_prefix()
        _download_code_prefix(input_data["hf_repo"], code_prefix, overrides.get("HF_TOKEN"))
        code_dir = os.path.join("/runcode", os.path.dirname(code_prefix) or ".")

        env = dict(os.environ)
        env.update(overrides)
        # inlined: handler is baked standalone (flash not importable); mirrors the worker cache cleanup.
        if not os.path.isdir("/runpod-volume"):
            for _k in [k for k, v in env.items() if str(v).startswith("/runpod-volume")]:
                env.pop(_k, None)
        # Pass spec via file to avoid ~128 KiB per-env-string exec limit.
        spec_path = "/tmp/job_spec.json"
        with open(spec_path, "w") as sf:
            sf.write(input_data["job_spec_json"])
        env["FLASH_JOB_SPEC_PATH"] = spec_path
        env.pop("FLASH_JOB_SPEC_JSON", None)
        env["PHASE"] = input_data["phase"]
        env["SEED"] = str(input_data["seed"])
        env["PYTHONPATH"] = code_dir + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )

        console_upload_lock = threading.Lock()
        # set once a terminal snapshot begins. run_mode's stop_upload is per-call and scoped inside
        # it, so these uploaders cannot see it; this is the flag they check.
        console_teardown = threading.Event()

        def _upload_console(mode: str, final: bool = False) -> bool:
            """Upload the captured console tail for ``mode`` to ``{phase_ns}/{run_id}/
            console_<mode>.txt`` in the run repo. Idempotent and best-effort, so it is safe from both
            the subprocess-failure and missing-metrics crash paths: a worker killed without a Python
            exception (OOM/SIGKILL, segfault, silent early exit) writes NO ``error_<mode>.txt``, so
            the console is then the only root-cause record — and a crash that exits 0 would otherwise
            skip the upload, leaving the failure opaque.

            The terminal snapshot must be the LAST writer: every caller commits to the same repo
            path, so an older periodic one landing after it restores a pre-failure console over the
            bytes explaining the failure. So ``final`` never yields INDEFINITELY (upload_file takes no
            timeout, so a wedged holder would suppress it forever) nor past the run wall deadline,
            periodic callers drop their commit once ``console_teardown`` is set, and a terminal upload
            forced through without the lock re-commits once it frees. Each rule has a
            test_serverless_*console* case. Returns whether the tail landed: errors are swallowed, so
            a caller tracking what is stored cannot read a normal return as success and skip its
            retry."""
            console = f"/tmp/console_{mode}.txt"
            if not os.path.exists(console):
                return False
            if final:
                console_teardown.set()
            # per-caller scratch: sharing one .tail splices an unsynchronized final snapshot
            tail = console + (".final.tail" if final else ".tail")

            def _wait_s() -> float:
                """Lock wait, clamped to what is left of the run wall deadline. The watchdog
                hard-exits AT the deadline, so a terminal upload still blocked on the lock is killed
                mid-acquire and the commit that could have landed in the window that remained never
                runs; the reserve keeps room for it. Each retry re-reads the clock, so the tries
                shrink rather than each being clamped and still summing past it. Zero polls without
                waiting: acquire rejects a negative, and a blown deadline must still try."""
                try:
                    return min(120.0, max(0.0, _require_deadline_allowance() - 30.0))
                except (TimeoutError, RuntimeError):
                    return 0.0

            held = console_upload_lock.acquire(timeout=_wait_s())
            try:
                if not final and (console_teardown.is_set() or not held):
                    print(f"console upload skipped for {mode}; the terminal snapshot supersedes it")
                    return False
                ok = _upload_console_locked(mode, console, tail, final)
            finally:
                if held:
                    console_upload_lock.release()
            # `not held`: the commit above raced a holder still inside upload_file, whose older bytes
            # can land after it. acquiring is proof that finished, so one more commit wins. the wait
            # is BOUNDED, never a retry loop: a permanently wedged holder never frees and this must
            # still return. split into tries so a holder needing longer than one timeout is still
            # caught -- an HF upload recovering after ~240s otherwise lands last and restores the
            # pre-failure console. past the bound it is treated as wedged and the raw commit stands.
            for _ in range(3) if final and not held else ():
                if console_upload_lock.acquire(timeout=_wait_s()):
                    try:
                        ok = _upload_console_locked(mode, console, tail, True) or ok
                    finally:
                        console_upload_lock.release()
                    break
            return ok

        def _upload_console_locked(mode: str, console: str, tail_path: str, final: bool) -> bool:
            try:
                from huggingface_hub import HfApi

                _require_deadline_allowance()
                spec = json.loads(input_data["job_spec_json"])
                phase_ns = "rl" if spec.get("algorithm") == "grpo" else spec["algorithm"]
                prefix = f"{phase_ns}/{spec['run_id']}"
                # Keep the newest bytes only; the uploaded tail's end is never truncated.
                tail_bytes = 64_000
                with open(console, "rb") as f:
                    f.seek(0, os.SEEK_END)
                    start = max(0, f.tell() - tail_bytes)
                    # over-read one byte so a boundary landing exactly after a newline is
                    # recognized as starting a COMPLETE line rather than assumed partial.
                    f.seek(max(0, start - 1))
                    raw = f.read()
                if start == 0:
                    tail = raw.decode("utf-8", "replace")
                else:
                    tail = raw[1:].decode("utf-8", "replace")
                    # a truncated first line is dropped before sanitizing: a boundary landing inside
                    # a one-line credential leaves a fragment full-value redaction no longer matches.
                    # duplicated, not imported: only this function's SOURCE ships to the worker.
                    # bootstrap_secrets._read_console_tail is canonical.
                    if raw[:1] != b"\n":
                        cut = tail.find("\n")
                        tail = tail[cut + 1 :] if cut >= 0 else ""
                with open(tail_path, "w", encoding="utf-8", errors="replace") as f:
                    f.write(_safe_detail(tail, env, 64_000))
                _require_deadline_allowance()
                # re-checked after staging: teardown can begin mid-flight, and committing then would
                # overwrite the terminal console with older bytes.
                if not final and console_teardown.is_set():
                    print(f"console upload dropped for {mode}; superseded by the terminal snapshot")
                    return False
                HfApi(token=env.get("HF_TOKEN")).upload_file(
                    path_or_fileobj=tail_path,
                    path_in_repo=f"{prefix}/console_{mode}.txt",
                    repo_id=input_data["hf_repo"],
                    repo_type="dataset",
                )
                return True
            except Exception as e:
                print("console upload warn:", _safe_detail(e, env))
                return False

        def run_mode(mode: str, check: bool) -> int:
            """Run worker subprocess, tee console to file, upload tail periodically and on exit."""
            console = f"/tmp/console_{mode}.txt"
            stop_upload = threading.Event()

            def _upload_loop() -> None:
                # literals, not the module-level constants: only this function's SOURCE ships to the
                # worker, so a name reference is a NameError before training. Mirrors
                # bootstrap._console_upload_loop, whose docstring has the rules; pinned against
                # drift by test_first_console_snapshot_precedes_the_stall_teardown.
                due_s, since, quiet_polls = 600.0, 0.0, 0
                uploaded_size, size, quiet_spent, armed, committed = -1, -1, 0, False, False
                while not stop_upload.wait(120.0):
                    since += 120.0
                    try:
                        # mirrors _console_progress: heartbeats not bytes, plus its rules.
                        at = max(size, 0)
                        with open(console, "rb") as hf:
                            hf.seek(at)
                            buf = hf.read()
                    except OSError:
                        buf, at = b"", -1
                    cut = buf.rfind(b"\n") + 1  # whole lines only; the offset stays a line start
                    buf, size = buf[:cut], at + cut
                    pat = rb'(?m)^HEARTBEAT (?!.*"liveness":).*$'
                    beats = re.findall(pat, buf)
                    staged = sum(b'"pending":' not in b for b in beats)
                    committed = committed or bool(staged)
                    # a wedge is progress that STOPPED: re-arm on it, spend only on a stall that
                    # BOUGHT an upload. 2 credits per RUN. `pending` counts until the first commit.
                    progress = staged if committed else len(beats)
                    armed = armed or bool(progress)
                    quiet_polls = 0 if progress else quiet_polls + 1
                    due = since >= due_s
                    wedged = armed and quiet_polls >= 4 and quiet_spent < 2 and not due
                    if size == uploaded_size or not (due or wedged):
                        continue
                    ok = _upload_console(mode)  # swallows its own errors; False if it did not land
                    uploaded_size = size if ok else uploaded_size
                    quiet_spent += 1 if wedged and ok else 0
                    armed = armed and not (wedged and ok)
                    # only a LANDED upload advances the deadline: resetting on a swallowed failure
                    # puts the retry an interval out, past the stall teardown.
                    if ok:
                        since, due_s = 0.0, 3600.0

            with open(console, "w", buffering=1) as cf:  # line-buffered so uploader sees each line
                _require_deadline_allowance()
                proc = subprocess.Popen(
                    [sys.executable, "-m", "flash.engine.worker_entrypoint"],
                    cwd=code_dir,
                    env={**env, "RUN_MODE": mode},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                )
                uploader = threading.Thread(target=_upload_loop, daemon=True)
                uploader.start()
                try:
                    for line in proc.stdout:
                        # the handler's own stdout is captured by runpod and surfaced in provider
                        # status, where only this process knows the run's worker-env secret values,
                        # so each echoed child line is sanitized here at the source. the console
                        # file keeps the raw line; its upload path sanitizes the selected tail.
                        print(_safe_detail(line, env, 100_000), end="")
                        cf.write(line)
                    proc.wait()
                finally:
                    stop_upload.set()
                    uploader.join(timeout=10)
            _upload_console(mode, final=True)
            if proc.returncode != 0 and check:
                raise RuntimeError(
                    f"worker mode '{mode}' exited {proc.returncode}; see console_{mode}.txt "
                    f"and error_{mode}_attempt*.txt in the HF dataset repo"
                )
            return proc.returncode

        # Clear stale metrics from a previous seed so a crash can't report wrong numbers.
        for stale in ("/tmp/train_meta.json", "/tmp/metrics.json"):
            with contextlib.suppress(FileNotFoundError):
                os.remove(stale)
        # check=False: RL's colocated vLLM can segfault at interpreter exit after saving — not a failure.
        run_mode(input_data["phase"], check=False)
        if not os.path.exists("/tmp/metrics.json"):
            phase = input_data["phase"]
            _upload_console(phase, final=True)
            raise RuntimeError(
                f"train phase '{phase}' produced no /tmp/metrics.json (it crashed before "
                f"finishing); see error_{phase}_attempt*.txt and console_{phase}.txt in the HF "
                f"dataset repo for the full traceback"
            )
        with open("/tmp/metrics.json") as f:
            return json.load(f)
    finally:
        deadline_timer.cancel()
