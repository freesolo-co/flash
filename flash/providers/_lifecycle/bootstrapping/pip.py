"""Extra-pip install for the box: run it, classify a failure, retry an index blip.

Shipped next to bootstrap.py and imported as a bare sibling module on the box, so this stays
stdlib-only and must never import flash. It is a leaf: the deadline check and the retriable error
class arrive as arguments rather than by importing bootstrap back, which would be a cycle.

The RunPod serverless handler carries its own copy of this logic because ``_train_body`` is
serialized standalone to the worker and cannot import a sibling. The two classifiers must agree on
what is retriable -- change both or neither.
"""

from __future__ import annotations

import collections
import contextlib
import os
import re
import subprocess
import sys
import tempfile
import time

_PIP_RETRY_DELAYS_S = (3.0, 9.0, 27.0)
_PIP_OUTPUT_TAIL_LINES = 400
_PIP_RETRY_RESERVE_S = 1.0  # kept back from a clamped backoff so the next attempt can run
# Network-shaped pip failures (retriable) vs deterministic build/resolution failures (terminal, outranking a transient
# warning pip already recovered from in the same tail). Bare "subprocess-exited-with-error" is in NEITHER: pip prints it
# for any child, including a VCS pin's `git clone`, so calling it terminal fails a paid run on a mid-clone reset.
_PIP_TRANSIENT_RE = re.compile(
    r"(?i)connection (?:broken|reset|aborted|refused|timed out)|read timed out|proxyerror"
    r"|temporary failure in name resolution|failed to establish a new connection|incompleteread"
    r"|network is unreachable|remote end closed connection|newconnectionerror|maxretryerror"
    r"|ssleoferror|service unavailable|bad gateway|gateway time-?out|too many requests"
    r"|retrying \(retry\(|\b(?:429|5\d\d) (?:client|server) error"
    # a VCS pin fails through git, not urllib, so its blips carry git's own phrasing and none of
    # the shapes above: git says "could not resolve host" where urllib says "temporary failure in
    # name resolution", and reports an http status as "returned error: NNN". On the status form,
    # only 429/5xx: a 404 or 403 is a bad pin or a missing token and must still fail fast rather
    # than burn three backoffs on a paid box.
    r"|returned error: (?:429|5\d\d)|could not resolve (?:host|proxy)"
)
_PIP_TERMINAL_RE = re.compile(
    r"(?i)failed building wheel|could not build wheels|metadata-generation-failed"
    r"|no matching distribution|could not find a version|resolutionimpossible|invalid requirement"
)
# terminal markers pip can print having downloaded NOTHING; see _is_terminal.
_PIP_NO_CANDIDATE_RE = re.compile(r"(?i)no matching distribution|could not find a version")


def _is_terminal(output: str) -> bool:
    """Whether this pip failure is deterministic, so retrying it would only re-fail identically.

    Three cases, in order:

    No terminal marker at all -- retry only if the tail looks network-shaped.

    A terminal marker and nothing transient -- deterministic.

    Both. An unreachable index yields no candidate versions, so pip finishes with exactly the
    footer a typo'd package name produces ("could not find a version" / "no matching
    distribution"): that text alone cannot tell a permanent bad spec from a total outage. Drop
    those footers and re-test. If another terminal marker still stands, pip held real content and
    the failure is deterministic regardless of the earlier warning -- the remaining alternatives
    are unreachable without content in hand (a wheel that failed to build, metadata that failed to
    generate, a genuine resolver conflict) or are decided before any request at all (an
    unparseable requirement). If nothing is left, the network fully explains the footer, so retry.
    """
    if not _PIP_TERMINAL_RE.search(output):
        return not _PIP_TRANSIENT_RE.search(output)
    if not _PIP_TRANSIENT_RE.search(output):
        return True
    return bool(_PIP_TERMINAL_RE.search(_PIP_NO_CANDIDATE_RE.sub("", output)))


def _extra_pip_env(payload: dict) -> tuple[dict[str, str], str | None]:
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (payload.get("env") or {}).items()})
    env.pop("GIT_ASKPASS", None)
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


def _drain_pip(proc, tail) -> int:
    """Tee pip's output into ``tail`` and the console; return its exit status.

    The console write is best-effort: a closed log stream must not end the drain, or pip is left
    running while the caller deletes its askpass helper and the console error is reported in place
    of pip's own status. What does escape kills the child rather than orphaning it on a paid box.
    """
    try:
        with proc.stdout:
            for line in proc.stdout:
                tail.append(line)
                with contextlib.suppress(OSError, ValueError):
                    print(line, end="", flush=True)
        return proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        raise


def install(payload: dict, *, require_deadline_at, retriable_error) -> None:
    """Install the run's extra requirements; retry an index blip, fail fast on a bad package spec.

    Same precedent as the pre-worker HF fetches: an index blip is infra, not user error, so it
    retries in place and surfaces as ``retriable_error`` rather than failing a paid run.
    """
    extra_pip = payload.get("extra_pip") or []
    if not extra_pip:
        return
    env, askpass = _extra_pip_env(payload)
    args = [sys.executable, "-m", "pip", "install", *extra_pip]
    try:
        for attempt in range(len(_PIP_RETRY_DELAYS_S) + 1):
            deadline_at = require_deadline_at(payload)
            tail = collections.deque(maxlen=_PIP_OUTPUT_TAIL_LINES)
            # errors="replace": a build or VCS child can emit bytes invalid under the worker's
            # locale, and strict decoding raises mid-stream, failing a paid run whose install
            # actually succeeded. undecodable bytes are diagnostics, never the exit status.
            proc = subprocess.Popen(
                args,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
            rc = _drain_pip(proc, tail)
            if rc == 0:
                return
            output = "".join(tail)  # a build failure outranks it; below that, network shapes retry
            if _is_terminal(output):
                raise RuntimeError(f"extra_pip install failed: pip exited {rc}")
            if attempt >= len(_PIP_RETRY_DELAYS_S):
                raise retriable_error(
                    f"extra_pip install could not reach the package index after {attempt + 1} "
                    f"attempts (pip exited {rc})"
                )
            delay = _PIP_RETRY_DELAYS_S[attempt]
            # clamping to the remaining wall alone sleeps the whole window, so the retry just
            # announced never issues: the next pass only fails the deadline precheck.
            remaining = deadline_at - time.time() - _PIP_RETRY_RESERVE_S
            delay = max(0.0, min(delay, remaining))
            # best-effort for the same reason the tee is: a console that closed between attempts
            # would otherwise raise HERE and end the install with a terminal, non-retriable console
            # error, losing the retry this line only announces.
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
