"""Server-side managed env publishing (`flash.server.envs`): an uploaded package is published to
FreeSolo's managed environment account, namespaced per identity and PRIVATE.
"""

from __future__ import annotations

import base64
import io
import json
import tarfile
from pathlib import Path

import pytest

from flash.server import envs

_MINIMAL = {
    "pyproject.toml": "[project]\nname = 'e'\n",
    "e/__init__.py": "def load_environment(**k):\n    return None\n",
}


def _pkg_b64(files: dict[str, str]) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode()


def test_namespace_is_stable_and_hub_safe():
    # A real, user-specific email is used as-is: case-insensitive + sanitized, and stable across
    # calls so re-publishes hit the same env.
    assert envs.namespace_for({"email": "Dev@Clado.ai"}) == "dev-clado-ai"
    assert envs.namespace_for({"email": "dev@clado.ai"}) == "dev-clado-ai"
    # No email at all: fall back to the per-key id, then key_prefix, then a constant.
    assert envs.namespace_for({"id": 7}) == "key-7"
    assert envs.namespace_for({"key_prefix": "fslo-abc"}) == "fslo-abc"
    assert envs.namespace_for({}) == "user"


def test_namespace_distinct_for_placeholder_emails():
    # REGRESSION: the server DB stores the SAME placeholder email for every external key
    # (db.ensure_external_key -> "freesolo-user"), so namespacing on email alone would collapse all
    # external users into one namespace and let them collide on env names. The per-key `id` must
    # disambiguate them even though the (placeholder) email and key_prefix are identical.
    k1 = {"id": 11, "email": "freesolo-user", "key_prefix": "freesolo"}
    k2 = {"id": 12, "email": "freesolo-user", "key_prefix": "freesolo"}
    ns1, ns2 = envs.namespace_for(k1), envs.namespace_for(k2)
    assert ns1 != ns2, "external keys with the placeholder email must NOT share a namespace"
    assert "freesolo-user" not in (ns1, ns2)  # the placeholder is never the namespace
    # Stable: the same key always maps to the same namespace (re-publish -> same env).
    assert envs.namespace_for(dict(k1)) == ns1


def test_publish_namespaces_and_returns_slug(monkeypatch):
    seen: dict = {}

    def fake_push(env_dir, *, name, is_new):
        seen.update(env_dir=env_dir, name=name, is_new=is_new)
        assert (Path(env_dir) / "pyproject.toml").is_file()  # the package was extracted
        return "freesolo-co/dev-clado-ai--myenv"

    monkeypatch.setattr(envs, "_prime_push", fake_push)
    slug = envs.publish_package(
        package_b64=_pkg_b64(_MINIMAL),
        name="myenv",
        is_new=True,
        key={"email": "dev@clado.ai"},
    )
    assert slug == "freesolo-co/dev-clado-ai--myenv"
    # Namespaced per identity (joined with the non-colliding "--" boundary) so two users can't
    # collide on the same env name.
    assert seen["name"] == "dev-clado-ai--myenv"
    assert seen["is_new"] is True


def test_publish_rejects_bad_input(monkeypatch):
    monkeypatch.setattr(envs, "_prime_push", lambda *a, **k: "x/y")
    with pytest.raises(envs.EnvPublishError, match="base64"):
        envs.publish_package(package_b64="not base64!!!", name="e", is_new=True, key={})
    with pytest.raises(envs.EnvPublishError, match="empty"):
        envs.publish_package(package_b64="", name="e", is_new=True, key={})
    with pytest.raises(envs.EnvPublishError, match="name"):
        envs.publish_package(package_b64=_pkg_b64(_MINIMAL), name="", is_new=True, key={})
    with pytest.raises(envs.EnvPublishError, match="pyproject"):
        envs.publish_package(
            package_b64=_pkg_b64({"e/__init__.py": "x = 1\n"}), name="e", is_new=True, key={}
        )


def test_safe_extract_rejects_traversal(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"pwn"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises(envs.EnvPublishError, match="unsafe path"):
        envs._safe_extract(buf.getvalue(), tmp_path)
    assert not (tmp_path.parent / "escape.txt").exists()


def test_safe_extract_extracts_regular_files_and_dirs(tmp_path):
    # The happy path: a normal package (a dir + a regular file) extracts.
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        d = tarfile.TarInfo("pkg")
        d.type = tarfile.DIRTYPE
        d.mode = 0o755
        tar.addfile(d)
        data = b"[project]\nname='e'\n"
        f = tarfile.TarInfo("pkg/pyproject.toml")
        f.size = len(data)
        f.mode = 0o644
        tar.addfile(f, io.BytesIO(data))
    envs._safe_extract(buf.getvalue(), tmp_path)
    assert (tmp_path / "pkg").is_dir()
    assert (tmp_path / "pkg" / "pyproject.toml").read_text() == "[project]\nname='e'\n"


def test_safe_extract_rejects_special_members(tmp_path):
    # A malicious tarball can carry non-regular members (device nodes, FIFOs). Even though they're
    # within the destination and not links, `extractall` would try to materialize them — reject them.
    for typeflag, label in ((tarfile.FIFOTYPE, "fifo"), (tarfile.CHRTYPE, "char-device")):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo(f"evil-{label}")
            info.type = typeflag
            if typeflag == tarfile.CHRTYPE:
                info.devmajor, info.devminor = 1, 3  # e.g. /dev/null
            tar.addfile(info)
        with pytest.raises(envs.EnvPublishError, match="special file"):
            envs._safe_extract(buf.getvalue(), tmp_path)
        assert not (tmp_path / f"evil-{label}").exists()


def test_slug_from_full_metadata(tmp_path):
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / ".env-metadata.json").write_text('{"owner": "freesolo-co", "name": "ns-e"}')
    assert envs._slug_from(tmp_path, "", pushed_name="ns-e") == "freesolo-co/ns-e"


def test_slug_from_owner_only_metadata_uses_pushed_name(tmp_path):
    # prime recorded the owner but no usable name: reconstruct from the name we pushed under, so a
    # successful push isn't reported as a failure (regression: clean exit was raising on no slug).
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / ".env-metadata.json").write_text('{"owner": "freesolo-co"}')
    assert envs._slug_from(tmp_path, "", pushed_name="ns-e") == "freesolo-co/ns-e"


def test_slug_from_stdout_phrasings(tmp_path):
    # No metadata file at all: parse the owner/name from prime's success line, across phrasings.
    assert envs._slug_from(tmp_path, "Successfully pushed freesolo-co/ns-e v2") == "freesolo-co/ns-e"
    assert envs._slug_from(tmp_path, "Pushed freesolo-co/ns-e") == "freesolo-co/ns-e"
    assert envs._slug_from(tmp_path, "Published freesolo-co/ns-e to the Hub") == "freesolo-co/ns-e"
    # Only the index URL names the owner -> pair with the pushed name.
    out = "uploaded to https://hub.primeintellect.ai/freesolo-co/simple/"
    assert envs._slug_from(tmp_path, out, pushed_name="ns-e") == "freesolo-co/ns-e"
    # Truly nothing -> None (the caller turns that into a 502, not a 400).
    assert envs._slug_from(tmp_path, "done", pushed_name=None) is None


def test_prime_push_success_without_slug_is_502(monkeypatch, tmp_path):
    # Clean exit but no discoverable owner/name: the env IS published, so this is a server-side
    # problem (502), NOT a 400 blaming the user's package.
    monkeypatch.setenv("PRIME_API_KEY", "pit-x")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/prime")

    class _Proc:
        returncode, stdout, stderr = 0, "all good, no slug here", ""

    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Proc())
    with pytest.raises(envs.EnvPublishError) as ei:
        envs._prime_push(tmp_path, name="ns-env", is_new=True)
    assert ei.value.status == 502
    assert "published" in str(ei.value)


def test_prime_push_503_when_control_plane_unconfigured(monkeypatch, tmp_path):
    # No PRIME_API_KEY -> 503 (this is the control plane's misconfiguration, not the user's input).
    monkeypatch.delenv("PRIME_API_KEY", raising=False)
    with pytest.raises(envs.EnvPublishError) as ei:
        envs._prime_push(tmp_path, name="ns-env", is_new=True)
    assert ei.value.status == 503
    # Key present but no `prime` CLI -> still 503.
    monkeypatch.setenv("PRIME_API_KEY", "pit-x")
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(envs.EnvPublishError) as ei2:
        envs._prime_push(tmp_path, name="ns-env", is_new=True)
    assert ei2.value.status == 503


def test_prime_push_publishes_private_and_climbs_conflicts(monkeypatch, tmp_path):
    monkeypatch.setenv("PRIME_API_KEY", "pit-x")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/prime")
    calls: list[list[str]] = []
    state = {"left": 1}

    class _Proc:
        def __init__(self, rc, out="", err=""):
            self.returncode, self.stdout, self.stderr = rc, out, err

    def run(cmd, capture_output=True, text=True, env=None, timeout=None):
        calls.append(list(cmd))
        if state["left"] > 0:
            state["left"] -= 1
            return _Proc(1, err="version already exists")
        (tmp_path / ".prime").mkdir(exist_ok=True)
        (tmp_path / ".prime" / ".env-metadata.json").write_text(
            '{"owner": "freesolo-co", "name": "ns-env"}'
        )
        return _Proc(0, out="Successfully pushed freesolo-co/ns-env\n")

    monkeypatch.setattr("subprocess.run", run)
    slug = envs._prime_push(tmp_path, name="ns-env", is_new=True)
    assert slug == "freesolo-co/ns-env"
    assert len(calls) == 2  # climbed past the version conflict
    assert "--auto-bump" in calls[1]
    assert calls[0][calls[0].index("--visibility") + 1] == "PRIVATE"
    assert calls[0][calls[0].index("--name") + 1] == "ns-env"


def test_prime_push_times_out_is_504_and_bounded(monkeypatch, tmp_path):
    """A hung `prime env push` (network stall / hub outage / CLI deadlock) must NOT block the request
    handler forever: each attempt is bounded by a timeout, and a TimeoutExpired surfaces as a 504
    without burning the remaining retries (a hang isn't a version conflict --auto-bump can fix)."""
    monkeypatch.setenv("PRIME_API_KEY", "pit-x")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/prime")
    # Make the timeout tiny so the test is instant and we can assert the bound was passed through.
    monkeypatch.setattr(envs, "_PUSH_TIMEOUT_S", 7)
    calls: list[dict] = []

    def run(cmd, capture_output=True, text=True, env=None, timeout=None):
        calls.append({"cmd": list(cmd), "timeout": timeout})
        raise envs.subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr("subprocess.run", run)
    with pytest.raises(envs.EnvPublishError) as ei:
        envs._prime_push(tmp_path, name="ns-env", is_new=True)
    assert ei.value.status == 504
    assert "timed out" in str(ei.value).lower()
    # Bounded: a hang is not retried (it's not a resolvable version conflict), and the per-attempt
    # timeout was actually handed to subprocess.run.
    assert len(calls) == 1
    assert calls[0]["timeout"] == 7


def test_prime_push_passes_timeout_each_attempt(monkeypatch, tmp_path):
    # Every attempt (including --auto-bump retries) must carry its own timeout, so total wall time
    # stays bounded by _PUSH_TIMEOUT_S * _PUSH_MAX_ATTEMPTS rather than being unbounded.
    monkeypatch.setenv("PRIME_API_KEY", "pit-x")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/prime")
    monkeypatch.setattr(envs, "_PUSH_TIMEOUT_S", 11)
    timeouts: list = []
    state = {"left": 2}

    class _Proc:
        def __init__(self, rc, out="", err=""):
            self.returncode, self.stdout, self.stderr = rc, out, err

    def run(cmd, capture_output=True, text=True, env=None, timeout=None):
        timeouts.append(timeout)
        if state["left"] > 0:
            state["left"] -= 1
            return _Proc(1, err="version already exists")
        return _Proc(0, out="Successfully pushed freesolo-co/ns-env\n")

    monkeypatch.setattr("subprocess.run", run)
    envs._prime_push(tmp_path, name="ns-env", is_new=True)
    assert timeouts == [11, 11, 11]  # every attempt got the bound


def test_assert_safe_build_backend_allows_standard_and_missing(tmp_path):
    # The standard, well-known backends are allowed (flash's own client emits hatchling.build).
    for backend in ("hatchling.build", "setuptools.build_meta", "flit_core.buildapi"):
        (tmp_path / "pyproject.toml").write_text(
            f"[build-system]\nrequires = []\nbuild-backend = '{backend}'\n[project]\nname='e'\n"
        )
        envs._assert_safe_build_backend(tmp_path)  # no raise
    # No [build-system] at all -> PEP 518 default (setuptools), allowed.
    (tmp_path / "pyproject.toml").write_text("[project]\nname='e'\n")
    envs._assert_safe_build_backend(tmp_path)
    # [build-system] present but no build-backend key -> also defaults to setuptools, allowed.
    (tmp_path / "pyproject.toml").write_text("[build-system]\nrequires = ['setuptools']\n")
    envs._assert_safe_build_backend(tmp_path)


def test_assert_safe_build_backend_rejects_custom_backend(tmp_path):
    # SECURITY (RCE): `prime env push` builds a wheel ON THIS HOST, importing the project's declared
    # build-backend. A custom backend would execute attacker code on the control plane, so reject it.
    (tmp_path / "pyproject.toml").write_text(
        "[build-system]\nrequires = []\nbuild-backend = 'evil_backend'\n[project]\nname='e'\n"
    )
    with pytest.raises(envs.EnvPublishError, match="unsupported build-backend"):
        envs._assert_safe_build_backend(tmp_path)


def test_assert_safe_build_backend_rejects_backend_path(tmp_path):
    # SECURITY (RCE): backend-path makes the build frontend load the backend FROM the upload's own
    # files (an in-tree backend), so even a standard-looking name resolves to attacker code. Refuse it
    # outright regardless of the build-backend value.
    (tmp_path / "pyproject.toml").write_text(
        "[build-system]\nrequires = []\nbuild-backend = 'hatchling.build'\n"
        "backend-path = ['.']\n[project]\nname='e'\n"
    )
    with pytest.raises(envs.EnvPublishError, match="backend-path"):
        envs._assert_safe_build_backend(tmp_path)


def test_assert_safe_build_backend_rejects_malformed_toml(tmp_path):
    # A malformed pyproject.toml is a client error (400), not a 500 from an uncaught parse error.
    (tmp_path / "pyproject.toml").write_text("this is not = valid = toml = [[[")
    with pytest.raises(envs.EnvPublishError, match="not valid TOML") as ei:
        envs._assert_safe_build_backend(tmp_path)
    assert ei.value.status == 400


def test_publish_rejects_malicious_build_backend_end_to_end(monkeypatch):
    # End-to-end: a package whose pyproject names a custom build-backend (an RCE vector via the
    # control-plane wheel build) is rejected by publish_package BEFORE `prime env push` is ever run.
    def boom(*a, **k):
        raise AssertionError("prime env push must NOT run for a package with an unsafe build backend")

    monkeypatch.setattr(envs, "_prime_push", boom)
    pkg = {
        "pyproject.toml": (
            "[build-system]\nrequires = []\nbuild-backend = 'evil_backend'\n"
            "backend-path = ['.']\n[project]\nname='e'\n"
        ),
        "evil_backend.py": "import os\nos.system('touch /tmp/flash_pwned_sentinel')\n",
    }
    with pytest.raises(envs.EnvPublishError, match=r"backend-path|unsupported build-backend"):
        envs.publish_package(
            package_b64=_pkg_b64(pkg), name="e", is_new=True, key={"email": "dev@clado.ai"}
        )


def test_prime_push_fails_fast_on_permanent_error(monkeypatch, tmp_path):
    """A clearly-permanent failure (auth/not-found/build/test) must NOT burn all retries:
    `--auto-bump` can't fix it, so surface the real error after one attempt."""
    monkeypatch.setenv("PRIME_API_KEY", "pit-x")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/prime")
    calls: list[list[str]] = []

    class _Proc:
        def __init__(self, rc, out="", err=""):
            self.returncode, self.stdout, self.stderr = rc, out, err

    def run(cmd, capture_output=True, text=True, env=None, timeout=None):
        calls.append(list(cmd))
        return _Proc(1, err="HTTP 401: Unauthorized — invalid API key")

    monkeypatch.setattr("subprocess.run", run)
    with pytest.raises(envs.EnvPublishError) as exc:
        envs._prime_push(tmp_path, name="ns-env", is_new=True)
    assert exc.value.status == 502
    assert "retrying will not help" in str(exc.value).lower()
    assert len(calls) == 1  # failed fast — did NOT climb through all _PUSH_MAX_ATTEMPTS


def test_is_permanent_push_failure_ignores_codes_in_hash_or_version():
    # REGRESSION (Cursor Bugbot, High): bare "401"/"403"/"404" substring matching mis-flagged a
    # resolvable version/content-hash CONFLICT as permanent (the digits appear inside a content hash
    # or version), making _prime_push fail-fast instead of climbing with --auto-bump. These must NOT
    # be treated as permanent failures.
    assert not envs._is_permanent_push_failure(
        "Upload failed: Wheel version with content hash a401f3b403c404d already exists"
    )
    assert not envs._is_permanent_push_failure("HTTP 400: version 1.404.0 already exists")
    assert not envs._is_permanent_push_failure("error: version 2.401.3 already exists on the hub")
    # A REAL HTTP status (with the http/status lead-in) is still permanent...
    assert envs._is_permanent_push_failure("HTTP 403: Forbidden")
    assert envs._is_permanent_push_failure("request failed with status 401")
    # ...and the textual markers are unaffected.
    assert envs._is_permanent_push_failure("error: unauthorized")
    assert envs._is_permanent_push_failure("environment not found")
    assert envs._is_permanent_push_failure("build failed: missing dependency")


def test_prime_push_retries_conflict_with_code_digits_in_hash(monkeypatch, tmp_path):
    # End-to-end of the regression: a conflict rejection whose content hash happens to contain
    # "404"/"401" must be RETRIED with --auto-bump (climbed past), not fail-fasted as permanent.
    monkeypatch.setenv("PRIME_API_KEY", "pit-x")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/prime")
    calls: list[list[str]] = []
    state = {"left": 1}

    class _Proc:
        def __init__(self, rc, out="", err=""):
            self.returncode, self.stdout, self.stderr = rc, out, err

    def run(cmd, capture_output=True, text=True, env=None, timeout=None):
        calls.append(list(cmd))
        if state["left"] > 0:
            state["left"] -= 1
            # digits 404/401 embedded in a content hash — must NOT be read as a permanent failure.
            return _Proc(1, err="HTTP 400: content hash f404a401 already exists")
        return _Proc(0, out="Successfully pushed freesolo-co/ns-env\n")

    monkeypatch.setattr("subprocess.run", run)
    slug = envs._prime_push(tmp_path, name="ns-env", is_new=True)
    assert slug == "freesolo-co/ns-env"
    assert len(calls) == 2  # it climbed past the conflict instead of failing fast
    assert "--auto-bump" in calls[1]


def test_publish_is_idempotent_on_republish(monkeypatch):
    # A re-publish passes the name part of the already-namespaced slug (e.g. "dev-clado-ai--myenv");
    # it must NOT be prefixed again into "dev-clado-ai--dev-clado-ai--myenv" (a brand-new env).
    seen: dict = {}
    monkeypatch.setattr(
        envs, "_prime_push", lambda env_dir, *, name, is_new: seen.update(name=name) or "x/y"
    )
    envs.publish_package(
        package_b64=_pkg_b64(_MINIMAL),
        name="dev-clado-ai--myenv",
        is_new=False,
        key={"email": "dev@clado.ai"},
    )
    assert seen["name"] == "dev-clado-ai--myenv"  # not double-prefixed


# --- resource caps (tar-bomb / unbounded-upload DoS defence) --------------------------------------


def _tar_with_members(count: int) -> bytes:
    """A .tar.gz holding ``count`` tiny regular files (to exercise the member-count cap)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for i in range(count):
            info = tarfile.TarInfo(f"f{i}.txt")
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
    return buf.getvalue()


def test_publish_rejects_non_string_inputs(monkeypatch):
    # The payload is arbitrary client JSON: a non-string name/package must be a clean 400, not a 500
    # from `.lower()` / `len()` on a non-string.
    monkeypatch.setattr(envs, "_prime_push", lambda *a, **k: "x/y")
    with pytest.raises(envs.EnvPublishError, match="name must be a string") as ei:
        envs.publish_package(package_b64=_pkg_b64(_MINIMAL), name=1, is_new=True, key={})
    assert ei.value.status == 400
    with pytest.raises(envs.EnvPublishError, match="package must be a base64 string") as ei2:
        envs.publish_package(package_b64=123, name="e", is_new=True, key={})
    assert ei2.value.status == 400


def test_safe_extract_rejects_filesystem_collision(tmp_path):
    # A malformed archive that names "a" as a file and then "a/b" (a file/dir collision) fails at the
    # filesystem layer with OSError during extract — that's bad client input, so it must surface as a
    # 400 EnvPublishError, not bubble out as a 500.
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"x"
        a = tarfile.TarInfo("a")
        a.size = len(data)
        tar.addfile(a, io.BytesIO(data))  # "a" is a regular file
        b = tarfile.TarInfo("a/b")  # ...now treat "a" as a directory -> collision on extract
        b.size = len(data)
        tar.addfile(b, io.BytesIO(data))
    with pytest.raises(envs.EnvPublishError) as ei:
        envs._safe_extract(buf.getvalue(), tmp_path)
    assert ei.value.status == 400
    assert "could not be extracted" in str(ei.value)


def test_safe_extract_rejects_non_tar_input(tmp_path):
    # A non-tar / non-gzip / truncated upload must surface as a client error (400 EnvPublishError),
    # not an uncaught tarfile.ReadError bubbling out as a 500.
    with pytest.raises(envs.EnvPublishError, match=r"valid \.tar\.gz") as ei:
        envs._safe_extract(b"this is not a tar.gz at all", tmp_path)
    assert ei.value.status == 400
    # And the whole publish path turns it into a 400 too (not a 500).
    with pytest.raises(envs.EnvPublishError, match=r"valid \.tar\.gz"):
        envs.publish_package(
            package_b64=base64.b64encode(b"garbage").decode(), name="e", is_new=True, key={}
        )


def test_safe_extract_rejects_too_many_members(tmp_path, monkeypatch):
    # A tiny archive with a huge member count is an inode/CPU bomb. We stream members lazily and
    # abort the moment the count cap trips — so extraction is BOUNDED by the cap (we don't pre-scan
    # the whole archive, and we never extract more than the limit before bailing).
    monkeypatch.setattr(envs, "_MAX_MEMBERS", 5)
    with pytest.raises(envs.EnvPublishError, match="too many members"):
        envs._safe_extract(_tar_with_members(1000), tmp_path)
    # Bounded: at most _MAX_MEMBERS were written before the abort (NOT all 1000) — the cap stopped us
    # early rather than after a full scan.
    assert len(list(tmp_path.iterdir())) <= 5
    # At the limit it still extracts cleanly.
    envs._safe_extract(_tar_with_members(5), tmp_path)
    assert len(list(tmp_path.iterdir())) == 5


def test_safe_extract_rejects_oversized_uncompressed(tmp_path, monkeypatch):
    # The cumulative-uncompressed cap: members whose summed sizes exceed the cap are rejected before
    # extraction. We exercise this with a tiny cap and small members (no multi-hundred-MB allocation
    # that would slow/OOM CI) — the logic is identical regardless of the absolute sizes.
    monkeypatch.setattr(envs, "_MAX_UNCOMPRESSED_BYTES", 100)
    body = b"x" * 4096  # 4 KB, comfortably over the 100-byte cap
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("big.bin")
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    with pytest.raises(envs.EnvPublishError, match="too large uncompressed"):
        envs._safe_extract(buf.getvalue(), tmp_path)
    assert not (tmp_path / "big.bin").exists()  # rejected before its payload was written


def test_publish_rejects_oversized_upload_before_decode(monkeypatch):
    # An oversized upload is rejected by length BEFORE we allocate the decoded bytes. We assert that
    # by making b64decode blow up if it's ever reached — the size guard must fire first (413).
    monkeypatch.setattr(envs, "_MAX_UPLOAD_BYTES", 64)
    monkeypatch.setattr(envs, "_prime_push", lambda *a, **k: "x/y")

    def _boom(*a, **k):
        raise AssertionError("oversized upload must be rejected before base64 decode")

    monkeypatch.setattr(envs.base64, "b64decode", _boom)
    with pytest.raises(envs.EnvPublishError, match="too large") as ei:
        envs.publish_package(package_b64="A" * 100_000, name="e", is_new=True, key={})
    assert ei.value.status == 413


def test_publish_ignores_uploaded_metadata_owner(monkeypatch, tmp_path):
    # SECURITY: the uploaded tarball ships a spoofed .prime/.env-metadata.json claiming a different
    # owner/name. The published slug must come from prime's OWN metadata (written under the managed
    # account with the server-derived --name), NOT the client's. _prime_push deletes the preexisting
    # .prime/ before pushing, so the spoof can never influence the slug.
    monkeypatch.setenv("PRIME_API_KEY", "pit-x")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/prime")
    spoof = {
        "pyproject.toml": "[project]\nname='e'\n",
        ".prime/.env-metadata.json": '{"owner": "victim", "name": "secret-env"}',
    }

    captured: dict = {}

    def run(cmd, capture_output=True, text=True, env=None, timeout=None):
        env_dir = Path(cmd[cmd.index("--path") + 1])
        captured["had_spoof_metadata"] = (env_dir / ".prime" / ".env-metadata.json").is_file()
        name = cmd[cmd.index("--name") + 1]
        # prime writes its OWN metadata under the managed account on a real push.
        (env_dir / ".prime").mkdir(exist_ok=True)
        (env_dir / ".prime" / ".env-metadata.json").write_text(
            json.dumps({"owner": "freesolo-co", "name": name})
        )

        class _Proc:
            returncode, stdout, stderr = 0, f"Successfully pushed freesolo-co/{name}\n", ""

        return _Proc()

    monkeypatch.setattr("subprocess.run", run)
    slug = envs.publish_package(
        package_b64=_pkg_b64(spoof), name="myenv", is_new=True, key={"email": "dev@clado.ai"}
    )
    # The spoofed .prime/ was deleted before push, and the slug is server-derived.
    assert captured["had_spoof_metadata"] is False
    assert slug == "freesolo-co/dev-clado-ai--myenv"
    assert "victim" not in slug
    assert "secret-env" not in slug


# --- client-side packaging: _tar_b64 must PRUNE excluded dirs, not just filter them ----------------


def _members_of(b64: str) -> set[str]:
    raw = base64.b64decode(b64)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        return {m.name for m in tar.getmembers() if m.isreg()}


def _build_env_tree(root: Path) -> None:
    """A realistic env tree: real source + sibling files, plus excluded tool/cache dirs (one of them,
    .venv, holding a file nested several levels deep to represent a huge tree we must NOT walk into)."""
    (root / "pyproject.toml").write_text("[project]\nname='e'\n")
    (root / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    (root / "utils.py").write_text("X = 1\n")
    (root / "datasets").mkdir()
    (root / "datasets" / "train.jsonl").write_text('{"q": 1}\n')
    # Excluded dirs (must be omitted AND not descended into):
    deep = root / ".venv" / "lib" / "python3.11" / "site-packages" / "big_pkg"
    deep.mkdir(parents=True)
    (deep / "huge.py").write_text("# pretend this is a massive dependency tree\n")
    (root / ".git").mkdir()
    (root / ".git" / "objects").mkdir()
    (root / ".git" / "objects" / "deadbeef").write_text("gitobj\n")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "environment.cpython-311.pyc").write_text("bytecode\n")
    (root / ".mypy_cache").mkdir()
    (root / ".mypy_cache" / "cache.json").write_text("{}\n")


def test_tar_b64_excludes_tool_dirs_and_keeps_source(tmp_path):
    from flash.cli.main.envpush import _tar_b64

    _build_env_tree(tmp_path)
    members = _members_of(_tar_b64(tmp_path))
    # Exactly the non-excluded source files, nothing from an excluded dir.
    assert members == {"pyproject.toml", "environment.py", "utils.py", "datasets/train.jsonl"}
    # In particular the deep sentinel under .venv is gone, and no excluded dir leaks in.
    assert not any(
        part in m.split("/")
        for m in members
        for part in (".venv", ".git", "__pycache__", ".mypy_cache", ".prime", ".pytest_cache")
    )


def test_tar_b64_does_not_descend_into_excluded_dirs(tmp_path, monkeypatch):
    # The CORE of the fix (vs the old rglob): excluded dirs must never be TRAVERSED. We spy on os.walk
    # and assert no walked root is inside an excluded dir — so we never stat the (potentially huge)
    # files beneath .venv/.git/etc. A plain rglob('*') would recurse into them and stat every entry.
    import os as _os

    from flash.cli.main.envpush import _tar_b64

    _build_env_tree(tmp_path)
    excluded = {".venv", ".git", "__pycache__", ".mypy_cache", ".prime", ".pytest_cache"}
    walked_roots: list[str] = []
    real_walk = _os.walk

    def spy_walk(top, *a, **k):
        for root, dirs, files in real_walk(top, *a, **k):
            walked_roots.append(root)
            yield root, dirs, files

    monkeypatch.setattr(_os, "walk", spy_walk)
    _tar_b64(tmp_path)
    # No walked root may contain an excluded directory in its path: we pruned before descending.
    for root in walked_roots:
        rel_parts = Path(root).relative_to(tmp_path).parts
        assert not (excluded & set(rel_parts)), f"descended into excluded dir: {root}"
    # Sanity: we DID walk the real source tree (top level + datasets/).
    rels = {str(Path(r).relative_to(tmp_path)) for r in walked_roots}
    assert "." in rels
    assert "datasets" in rels


def test_republish_by_other_namespace_does_not_hijack(monkeypatch):
    # SECURITY (prefix hijack): victim B's namespace is "dev-clado-ai" and their env is
    # "dev-clado-ai--myenv". Attacker A's namespace is just "dev" (a raw-string prefix of B's). A
    # re-publishes passing B's full name; the server must NOT pass it through as already-namespaced
    # (which would version B's env) — it must re-namespace under A's OWN "dev--" boundary.
    seen: dict = {}
    monkeypatch.setattr(
        envs, "_prime_push", lambda env_dir, *, name, is_new: seen.update(name=name) or "x/y"
    )
    envs.publish_package(
        package_b64=_pkg_b64(_MINIMAL),
        name="dev-clado-ai--myenv",  # victim B's slug name part
        is_new=False,
        key={"key_prefix": "dev"},  # attacker A: namespace_for -> "dev" (a raw prefix of B's ns)
    )
    # A lands under their OWN namespace, not B's — no hijack.
    assert seen["name"] == "dev--dev-clado-ai--myenv"
    assert seen["name"] != "dev-clado-ai--myenv"
