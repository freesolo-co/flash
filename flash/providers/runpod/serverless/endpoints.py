"""RunPod Flash endpoint lifecycle: provision/cache/teardown + the worker handler."""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading

from flash._internal.diagnostics import sanitize_diagnostic
from flash.providers._lifecycle.net.worker import logger
from flash.providers.core.base import canonical_gpu
from flash.providers.runpod.serverless import naming
from flash.providers.runpod.serverless.naming import endpoint_name

# runpod_flash asyncio singleton is bound to one event loop; serialize all deploy/undeploy.
FLASH_SDK_LOCK = threading.Lock()

_CONSOLE_UPLOAD_INTERVAL_S = 3600.0


def _reset_flash_resource_manager(rm_module) -> None:
    """Drop runpod_flash's in-memory ResourceManager state after switching state files."""
    manager = getattr(rm_module, "ResourceManager", None)
    if manager is None:
        return

    instances = getattr(manager, "_instances", None)
    instance = instances.get(manager) if isinstance(instances, dict) else None
    for target in (manager, instance):
        if target is None:
            continue
        for attr in ("_resources", "_resource_configs", "_deployment_locks"):
            state = getattr(target, attr, None)
            if isinstance(state, dict):
                state.clear()
            elif state is not None:
                with contextlib.suppress(Exception):
                    setattr(target, attr, {})
    with contextlib.suppress(Exception):
        manager._resources_initialized = False


def _train_body(input_data: dict) -> dict:
    """Runs ON the RunPod GPU worker: fetch code, train, return metrics.

    All imports must be inside the function body — this handler is serialized standalone.
    """
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

    try:
        import archive as _source_snapshot
    except ModuleNotFoundError:
        from flash.snapshot import archive as _source_snapshot

    class _TransientSourceFetchError(RuntimeError):
        flash_retriable = True

    def _percent_pattern(needle):
        """Regex matching only percent-escape hex digits case-insensitively."""
        escape_re = re.compile(r"%([0-9A-Fa-f]{2})")
        parts = []
        offset = 0
        for match in escape_re.finditer(needle):
            parts.append(re.escape(needle[offset : match.start()]))
            parts.append("%")
            parts.extend(
                f"[{char.lower()}{char.upper()}]" if char.isalpha() else char
                for char in match.group(1)
            )
            offset = match.end()
        parts.append(re.escape(needle[offset:]))
        return "".join(parts)

    def _needles(secrets=None):
        """Typed value matchers, shape-only needles, and raw values for all known secrets.

        each value matcher carries ``(needle, bounded, encoded)`` metadata. a value at or above the
        floor is plain and replaced as a substring; a shorter one is bounded, matched only where it is
        not adjacent to a word character. short values used to be dropped outright, which leaked them
        verbatim. plain replacement is not the alternative: a 3-char needle corrupts every diagnostic
        that merely contains those letters (the value "ati" rewrites "authentication"). a short raw
        candidate with no alphanumeric or underscore character is shape-only because it is
        indistinguishable from ordinary punctuation. explicit percent-octet forms remain bounded.

        a multiline secret never appears whole in any single call: the child's stdout is sanitized one
        line at a time, so only a component line is ever seen. component lines keep the floor as a hard
        skip: a short one is punctuation such as "}", not a credential. mirrors
        flash.providers._lifecycle.bootstrapping.secrets._needles.
        """
        import urllib.parse

        mapping = {**os.environ, **(secrets or {})}
        # declared runtime secrets can carry any name, so the control plane lists them in
        # flash_secret_env_keys; the name-shape rule stays as the fail-closed fallback.
        declared = {
            name.strip().upper()
            for name in str(mapping.get("FLASH_SECRET_ENV_KEYS") or "").split(",")
            if name.strip()
        }
        matchers, shaped, raw_values = set(), set(), set()
        for key, secret in mapping.items():
            upper = str(key).upper()
            if not secret or not (
                upper in {"AUTHORIZATION", "HF_TOKEN"}
                or upper in declared
                or upper.endswith(("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD"))
            ):
                continue
            value_str = str(secret)
            raw_values.add(value_str)
            candidates = {(value_str, False)}
            encoded = urllib.parse.quote(value_str, safe="")
            if encoded != value_str:
                candidates.add((encoded, True))
            if len(value_str) < 8 and not any(char.isalnum() or char == "_" for char in value_str):
                candidates.add(("".join(f"%{byte:02X}" for byte in value_str.encode()), True))
            if len(value_str) >= 8:
                matchers.update(
                    (candidate, False, is_encoded) for candidate, is_encoded in candidates
                )
            else:
                for candidate, is_encoded in candidates:
                    if any(char.isalnum() or char == "_" for char in candidate):
                        matchers.add((candidate, True, is_encoded))
                    else:
                        shaped.add(candidate)
            if "\n" in value_str:
                for raw in value_str.splitlines():
                    if len(line := raw.strip()) >= 8:
                        matchers.add((line, False, False))
                        encoded_line = urllib.parse.quote(line, safe="")
                        if encoded_line != line:
                            matchers.add((encoded_line, False, True))
        return matchers, shaped, raw_values

    def _safe_detail(value, secrets=None, limit=1000):
        text = (
            f"{type(value).__name__}: {value}" if isinstance(value, BaseException) else str(value)
        )
        matchers, shaped, raw_values = _needles(secrets)
        # protect exact punctuation credentials before a separate value can erase their syntax.
        for needle in sorted(shaped, key=len, reverse=True):
            escaped = re.escape(needle)
            text = re.sub(
                rf"(?i)(authorization|api[-_ ]?key|access[-_ ]?token|token|secret|password)(\s*[:=]\s*)(?:bearer\s+)?{escaped}(?=[\s,;]|$)",
                lambda match: (
                    "<redacted>"
                    if match.group(1) in raw_values
                    else f"{match.group(1)}{match.group(2)}<redacted>"
                ),
                text,
            )
            text = re.sub(
                rf"(?i)\b(bearer)\s+{escaped}(?=[\s,;]|$)",
                lambda match: "<redacted>" if match.group(1) in raw_values else "Bearer <redacted>",
                text,
            )
        # longest-first across both matcher types so a shorter plain value cannot consume the prefix
        # of a longer bounded encoded value; encoded forms cover urls http and git errors print.
        for needle, is_bounded, is_encoded in sorted(
            matchers, key=lambda item: len(item[0]), reverse=True
        ):
            if is_encoded:
                pattern = _percent_pattern(needle)
                if is_bounded:
                    left = r"(?<!\w)" if needle[:1].isalnum() or needle[:1] == "_" else ""
                    right = r"(?!\w)" if needle[-1:].isalnum() or needle[-1:] == "_" else ""
                    pattern = f"{left}{pattern}{right}"
                text = re.sub(pattern, "<redacted>", text)
            elif is_bounded:
                # the word guard is applied per edge, and only where the needle's own edge is a word
                # character. a value with a punctuation edge already separates itself from
                # neighbouring text, and demanding a non-word character beyond it asks the wrong
                # question: "/a" inside "https://host/a/repo" is preceded by the "t" of "host", so
                # an unconditional left guard fails and the secret prints verbatim. "ati" keeps both
                # guards and so still cannot rewrite "authentication".
                left = r"(?<!\w)" if needle[:1].isalnum() or needle[:1] == "_" else ""
                right = r"(?!\w)" if needle[-1:].isalnum() or needle[-1:] == "_" else ""
                text = re.sub(f"{left}{re.escape(needle)}{right}", "<redacted>", text)
            else:
                text = text.replace(needle, "<redacted>")
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
            env.pop("GIT_ASKPASS", None)
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

        def _install_extra_pip() -> None:
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
                # already recovered from in the same tail; without that precedence one early
                # "Retrying (Retry(" makes a deterministic failure look retriable and this ladder
                # repeats it for nothing. Kept identical to the instance bootstrap's _PIP_TERMINAL_RE:
                # the two classifiers must agree on what is retriable, including excluding the bare
                # subprocess-exited-with-error marker that a network-interrupted VCS `git clone` also
                # prints.
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

        def _source_descriptor():
            return _source_snapshot.parse_descriptor(input_data.get("source_snapshot"))

        def _hf_status_code(exc: BaseException) -> int | None:
            return _source_snapshot.response_status_code(exc)

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
            for attempt in range(len(retry_delays) + 1):
                _require_deadline_allowance()
                try:
                    return call()
                except Exception as exc:
                    if not _source_snapshot.is_transient_fetch_error(exc) or attempt >= len(
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

        def _load_exact_module(code_dir: str, relative_path: str, name: str):
            path = os.path.join(code_dir, relative_path)
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"could not load downloaded module: {relative_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        def _download_source_snapshot(repo_id: str, descriptor, token: str | None) -> str:
            from huggingface_hub import hf_hub_download

            try:
                archive_path = _hf_call(
                    lambda: hf_hub_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        filename=descriptor.archive_path,
                        revision=descriptor.revision,
                        token=token,
                    ),
                    "download pinned flash source snapshot",
                )
            except Exception as exc:
                error_type = (
                    _TransientSourceFetchError
                    if _source_snapshot.is_transient_fetch_error(exc)
                    else RuntimeError
                )
                raise error_type("failed to fetch the pinned flash source snapshot") from None
            destination = str(
                _source_snapshot.attempt_materialization_path(
                    "/runcode",
                    input_data.get("run_id"),
                    input_data.get("attempt"),
                )
            )
            _source_snapshot.materialize_verified_archive_file(
                archive_path,
                descriptor,
                destination,
            )
            return destination

        descriptor = _source_descriptor()
        code_dir = _download_source_snapshot(
            input_data["hf_repo"], descriptor, overrides.get("HF_TOKEN")
        )
        _install_extra_pip()
        console_module = _load_exact_module(
            code_dir,
            "flash/providers/_lifecycle/bootstrapping/console.py",
            "_flash_downloaded_bootstrap_console",
        )
        artifact_module = _load_exact_module(
            code_dir,
            "flash/adapters/artifacts.py",
            "_flash_downloaded_artifacts",
        )

        env = dict(os.environ)
        env.update(overrides)
        env.pop("GITHUB_TOKEN", None)
        env.pop("GIT_ASKPASS", None)
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
        env["ATTEMPT"] = str(input_data["attempt"])
        env["PYTHONPATH"] = code_dir + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )

        console_teardown = threading.Event()

        def _upload_console(mode: str, final: bool = False) -> bool:
            """Upload the captured console tail for ``mode`` to the run repo.

            Idempotent and best-effort, so it is safe to call from both the subprocess-failure path
            and the missing-metrics crash path: a worker killed without a Python exception
            (OOM/SIGKILL, segfault, or a silent early exit) writes NO ``error_<mode>.txt``, so the
            captured console is then the only root-cause record -- and a crash that exits 0 would
            otherwise skip the upload entirely, leaving the failure opaque.

            ``final`` writes the canonical ``console_<mode>.txt`` and closes the live path; live
            snapshots are attempt-scoped so a retry cannot overwrite the previous attempt's tail.
            """
            console = f"/tmp/console_{mode}.txt"
            if not os.path.exists(console):
                return False
            if final:
                console_teardown.set()
            elif console_teardown.is_set():
                return False
            try:
                from huggingface_hub import HfApi

                _require_deadline_allowance()
                spec = json.loads(input_data["job_spec_json"])
                phase_ns = "rl" if spec.get("algorithm") == "grpo" else spec["algorithm"]
                prefix = f"{phase_ns}/{spec['run_id']}"
                # keep the newest bytes only; the uploaded tail's end is never truncated.
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
                    # the byte boundary can land inside a one-line credential, and a partial
                    # value no longer matches full-value redaction, so a truncated first line is
                    # dropped before sanitizing. a line the boundary did not split is kept: it
                    # may hold the root-cause exception.
                    # a tail that is ONE unterminated line is dropped whole. that loses the only
                    # diagnostic on a crash whose evidence is a single huge line, but any bound
                    # that would let it through is measured against the credentials this process
                    # KNOWS, and the value at risk is the one it does not: a capability minted at
                    # runtime contributes no needle, so a margin sized from an unrelated configured
                    # secret leaves a long fragment of it behind. an empty tail never leaked.
                    # mirrors bootstrap_secrets._read_console_tail.
                    if raw[:1] != b"\n":
                        cut = tail.find("\n")
                        tail = tail[cut + 1 :] if cut >= 0 else ""
                suffix = ".final.tail" if final else ".live.tail"
                tail_path = console + suffix
                with open(tail_path, "w", encoding="utf-8", errors="replace") as f:
                    f.write(_safe_detail(tail, env, 64_000))
                _require_deadline_allowance()
                if not final and console_teardown.is_set():
                    return False
                attempt = int(env.get("ATTEMPT") or 0)
                artifact = (
                    f"console_{mode}.txt"
                    if final
                    else artifact_module.attempt_scoped_artifact_name("console", mode, attempt)
                )
                HfApi(token=env.get("HF_TOKEN")).upload_file(
                    path_or_fileobj=tail_path,
                    path_in_repo=f"{prefix}/{artifact}",
                    repo_id=input_data["hf_repo"],
                    repo_type="dataset",
                )
                return True
            except Exception as e:
                print("console upload warn:", _safe_detail(e, env))
                return False

        def run_mode(mode: str, check: bool) -> int:
            """run the worker, stream its console, and upload live and terminal tails."""
            console = f"/tmp/console_{mode}.txt"
            stop_upload = threading.Event()

            def _upload_live() -> bool:
                return _upload_console(mode)

            with open(console, "w", buffering=1) as cf:  # line-buffered so uploader sees each line
                _require_deadline_allowance()
                proc = subprocess.Popen(
                    [sys.executable, "-m", "flash.engine.support.worker_entrypoint"],
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
            raise RuntimeError(
                f"train phase '{phase}' produced no /tmp/metrics.json (it crashed before "
                f"finishing); see error_{phase}_attempt*.txt and console_{phase}.txt in the HF "
                f"dataset repo for the full traceback"
            )
        with open("/tmp/metrics.json") as f:
            metrics = json.load(f)
        if not isinstance(metrics, dict):
            raise RuntimeError("train metrics are invalid")
        metrics[_source_snapshot.TERMINAL_ATTESTATION_KEY] = _source_snapshot.source_attestation(
            descriptor,
            run_id=input_data["run_id"],
            attempt=input_data["attempt"],
        )
        return metrics
    finally:
        deadline_timer.cancel()


def isolate_flash_state(scope: str | None = None) -> None:
    """Point the Flash SDK's resource registry at a per-process dir under <data dir>/flash-state/."""
    try:
        import runpod_flash.core.resources.resource_manager as rm

        from flash._internal.paths import data_dir

        scope = scope or f"pid{os.getpid()}"
        state_dir = data_dir() / "flash-state" / scope
        state_dir.mkdir(parents=True, exist_ok=True)
        previous_state_file = getattr(rm, "RESOURCE_STATE_FILE", None)
        rm.FLASH_STATE_DIR = state_dir
        rm.RESOURCE_STATE_FILE = state_dir / "resources.pkl"
        if hasattr(rm, "RUNPOD_FLASH_DIR"):
            rm.RUNPOD_FLASH_DIR = state_dir
        if previous_state_file != rm.RESOURCE_STATE_FILE:
            _reset_flash_resource_manager(rm)
    except Exception as exc:
        logger.warning("flash state isolation skipped: %s", exc)


def _patch_runpod_backoff() -> None:
    """Cap the backoff exponent before the power to prevent OverflowError on long runs (~80 min+)."""
    try:
        import math
        import random

        from runpod_flash.core.utils import backoff as _bo

        if getattr(_bo, "_flash_backoff_patched", False):
            return

        def _safe_get_backoff_delay(
            attempt,
            base=0.1,
            max_seconds=10.0,
            jitter=0.2,
            strategy=_bo.BackoffStrategy.EXPONENTIAL,
        ):
            a = min(int(attempt), 30)
            if strategy == _bo.BackoffStrategy.EXPONENTIAL:
                delay = base * (2**a)
            elif strategy == _bo.BackoffStrategy.LINEAR:
                delay = base + (attempt * base)
            elif strategy == _bo.BackoffStrategy.LOGARITHMIC:
                delay = base * math.log2(attempt + 2)
            else:
                raise ValueError(f"Unsupported backoff strategy: {strategy}")
            delay = min(delay, max_seconds)
            return delay * random.uniform(1 - jitter, 1 + jitter)

        _bo.get_backoff_delay = _safe_get_backoff_delay
        _bo._flash_backoff_patched = True
        # serverless.py imported the symbol directly; patch its ref too.
        try:
            from runpod_flash.core.resources import serverless as _sl

            _sl.get_backoff_delay = _safe_get_backoff_delay
        except Exception:
            pass
    except Exception as exc:
        logger.warning("runpod backoff patch skipped: %s", exc)


def min_cuda_for(friendly_gpu: str) -> str:
    """Minimum host CUDA driver version for this GPU class (Blackwell requires >=13.0)."""
    from flash.providers.core.base import min_cuda_modern

    return min_cuda_modern(friendly_gpu)


def terminate_endpoint(friendly_gpu: str, run_id: str | None = None) -> list[dict]:
    """Delete the remote Flash endpoint(s) for a run via the RunPod API. Best-effort, never raises."""
    friendly = canonical_gpu(friendly_gpu)
    target = endpoint_name(friendly, naming.run_suffix(run_id))
    # Serialize isolation + lookup + undeploy: isolate_flash_state swaps process-wide globals,
    # and a concurrent call could swap the registry scope between our lookup and undeploy.
    with FLASH_SDK_LOCK:
        try:
            from flash.providers.runpod.client.auth import ensure_auth

            ensure_auth()
            isolate_flash_state(naming.run_suffix(run_id))
            from runpod_flash.core.resources.resource_manager import ResourceManager
        except Exception as exc:
            detail = sanitize_diagnostic(exc, limit=1000)
            return [{"success": False, "name": target, "message": f"flash unavailable: {detail}"}]

        try:
            rm = ResourceManager()
            resources = rm.list_all_resources()
            uids = naming.select_endpoint_resources(resources, target)
        except Exception as exc:
            detail = sanitize_diagnostic(exc, limit=1000)
            return [
                {"success": False, "name": target, "message": f"resource lookup failed: {detail}"}
            ]

        async def _undeploy_all() -> list:
            out = []
            for uid in uids:
                res = resources.get(uid)
                name = getattr(res, "name", None)
                try:
                    out.append(
                        await rm.undeploy_resource(uid, resource_name=name, force_remove=True)
                    )
                except Exception as exc:
                    out.append(
                        {
                            "success": False,
                            "name": name,
                            "message": sanitize_diagnostic(exc, limit=1000),
                        }
                    )
            return out

        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                results = asyncio.run(_undeploy_all())
            else:
                # Running event loop (FastAPI lifespan etc) — run in a daemon thread.
                _out: list = []
                _err: list = []

                def _run_undeploy() -> None:
                    try:
                        _out.append(asyncio.run(_undeploy_all()))
                    except Exception as _e:
                        _err.append(_e)

                _t = threading.Thread(target=_run_undeploy, daemon=True)
                _t.start()
                _t.join(timeout=30)
                if _err:
                    raise _err[0]
                if not _out:
                    raise TimeoutError("undeploy timed out after 30s")
                results = _out[0]
        except Exception as exc:
            results = [
                {
                    "success": False,
                    "name": target,
                    "message": sanitize_diagnostic(exc, limit=1000),
                }
            ]

    # registry-less cleanup must inspect every configured account and every attempt of this run.
    try:
        from flash.providers.runpod.client import api as runpod_api

        by_fingerprint, failed_fingerprints = runpod_api.list_endpoints_by_key()
        for fingerprint, endpoints in by_fingerprint.items():
            for endpoint in endpoints:
                if not naming.endpoint_name_matches_run(endpoint.get("name", ""), target):
                    continue
                endpoint_id = endpoint.get("id")
                if not endpoint_id or not runpod_api.delete_endpoint_for_fingerprint(
                    endpoint_id, fingerprint
                ):
                    results.append(
                        {
                            "success": False,
                            "name": endpoint.get("name") or target,
                            "message": "REST endpoint deletion was unconfirmed",
                        }
                    )
                    continue
                results.append(
                    {
                        "success": True,
                        "name": endpoint.get("name") or target,
                        "message": "deleted via REST API",
                    }
                )
        if failed_fingerprints:
            results.append(
                {
                    "success": False,
                    "name": target,
                    "message": (
                        f"could not enumerate {len(failed_fingerprints)} configured RunPod account(s)"
                    ),
                }
            )
    except Exception as exc:
        logger.warning("REST endpoint cleanup failed for %s: %s", target, exc)

    return results
