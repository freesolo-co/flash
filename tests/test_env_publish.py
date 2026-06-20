"""Server-side managed env publishing (`flash.server.envs`): an uploaded package is published to
FreeSolo's Prime account, namespaced per identity and PRIVATE — so users never need a Prime key.
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
    # Stable: the same key always maps to the same namespace (re-publish -> same Hub env).
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

    def run(cmd, capture_output=True, text=True, env=None):
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
