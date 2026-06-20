"""Server-side managed env publishing (`flash.server.envs`): an uploaded package is published to
FreeSolo's Prime account, namespaced per identity and PRIVATE — so users never need a Prime key.
"""

from __future__ import annotations

import base64
import io
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
    # Stable: the same key always maps to the same namespace (re-publish -> same Hub env).
    assert envs.namespace_for(dict(k1)) == ns1


def test_publish_namespaces_and_returns_slug(monkeypatch):
    seen: dict = {}

    def fake_push(env_dir, *, name, is_new):
        seen.update(env_dir=env_dir, name=name, is_new=is_new)
        assert (Path(env_dir) / "pyproject.toml").is_file()  # the package was extracted
        return "freesolo-co/dev-clado-ai-myenv"

    monkeypatch.setattr(envs, "_prime_push", fake_push)
    slug = envs.publish_package(
        package_b64=_pkg_b64(_MINIMAL),
        name="myenv",
        is_new=True,
        key={"email": "dev@clado.ai"},
    )
    assert slug == "freesolo-co/dev-clado-ai-myenv"
    # Namespaced per identity so two users can't collide on the same env name.
    assert seen["name"] == "dev-clado-ai-myenv"
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

    def run(cmd, capture_output=True, text=True, env=None):
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
