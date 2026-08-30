"""`flash env push` packages local Freesolo envs and uploads them through the server."""

from __future__ import annotations

import argparse
import base64
import io
import os
import subprocess
import sys
import tarfile

import pytest

import flash.cli.parsing.main as cli
from flash.cli.commands.env.ops.push import _human_bytes, _UploadProgress
from flash.envs.package.direct_tokens import (
    _CHUNK_SIZE,
    _OVERLAP,
    package_contains_direct_token,
)


def _fake_client(capture: dict, *, slug: str = "acme/environment"):
    """A stand-in ApiClient that records the publish_env call and returns an env id."""

    class _C:
        def publish_env(self, *, name, package_b64, project_id, progress=None):
            capture.update(
                name=name, package_b64=package_b64, project_id=project_id, progress=progress
            )
            return {"id": slug}

    return lambda: _C()


def _member_bytes(package_b64: str) -> dict[str, bytes]:
    raw = base64.b64decode(package_b64)
    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                out[member.name] = tar.extractfile(member).read()
    return out


def _members(package_b64: str) -> dict[str, str]:
    return {name: content.decode() for name, content in _member_bytes(package_b64).items()}


def _member_names(package_b64: str) -> list[str]:
    raw = base64.b64decode(package_b64)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        return [member.name for member in tar.getmembers()]


def _args(
    path, *, name: str = "my-env", project: str | None = "11111111-1111-4111-8111-111111111111"
):
    return argparse.Namespace(path=str(path), name=name, project=project)


def _token_body(prefix: str, length: int) -> str:
    seed = (
        "aB3_dE5-fG7hJ9kL2mN4pQ6rS8tUvW0xY1zC"
        if prefix == "fslo_"
        else "aB3dE5fG7hJ9kL2mN4pQ6rS8tUvW0xY1zC"
    )
    return (seed * 2)[:length]


def _issued_token(prefix: str) -> str:
    body_length = {"fslo_": 45, "hf_": 34, "pit_": 64}[prefix]
    body = _token_body(prefix, body_length)
    if prefix == "fslo_":
        assert {"_", "-"} <= set(body)
    return prefix + body


def _deny_archive_and_upload(monkeypatch, calls: list[str]) -> None:
    def deny_archive(_pkg):
        calls.append("archive")
        raise AssertionError("archive must not be created")

    monkeypatch.setattr("flash.cli.commands.env.ops.push._tar_b64", deny_archive)
    monkeypatch.setattr(
        "flash.client.client_from_config",
        lambda: calls.append("upload") or pytest.fail("upload must not start"),
    )


def _deny_push_side_effects(monkeypatch, calls: list[str]) -> None:
    class _Client:
        def publish_env(self, **_kwargs):
            calls.append("upload")
            return {"id": "acme/project/environment"}

    def deny_temporary_package(*_args, **_kwargs):
        calls.append("temporary-package")
        pytest.fail("temporary package must not be created")

    monkeypatch.setattr(
        "flash.envs.package.direct_tokens.package_contains_direct_token",
        lambda _pkg: calls.append("direct-token-scan") or False,
    )
    monkeypatch.setattr(
        "flash.cli.commands.env.ops.push.tempfile.TemporaryDirectory", deny_temporary_package
    )
    monkeypatch.setattr(
        "flash.cli.commands.env.ops.push._tar_b64",
        lambda _pkg: calls.append("archive") or "package",
    )
    monkeypatch.setattr(
        "flash.client.client_from_config",
        lambda: calls.append("client") or _Client(),
    )


@pytest.mark.parametrize(
    "source",
    [
        "def broken(:\n    pass\n",
        "return 1\n",
        "yield 1\n",
        "break\n",
        "continue\n",
        "nonlocal value\n",
        "await work()\n",
        "def duplicate(value, value):\n    pass\n",
        "from __future__ import unknown_feature\n",
    ],
    ids=[
        "parser-error",
        "return-outside-function",
        "yield-outside-function",
        "break-outside-loop",
        "continue-outside-loop",
        "nonlocal-at-module-level",
        "await-outside-function",
        "duplicate-argument",
        "unknown-future-feature",
    ],
)
def test_push_rejects_invalid_original_entrypoint_before_any_side_effect(
    monkeypatch, tmp_path, capsys, source
):
    env_file = tmp_path / "environment.py"
    env_file.write_text(source)
    calls: list[str] = []
    _deny_push_side_effects(monkeypatch, calls)

    assert cli.cmd_env_push(_args(env_file)) == 1

    error = capsys.readouterr().err
    assert "environment entrypoint has invalid Python syntax" in error
    assert "line " in error
    assert calls == []


def test_push_rejects_invalid_published_entrypoint_before_any_side_effect(
    monkeypatch, tmp_path, capsys
):
    from flash.cli.commands.env.ops import push as envpush

    env_file = tmp_path / "environment.py"
    env_file.write_text("VALUE = 1\n")
    monkeypatch.setattr(envpush, "_ENV_SYSPATH_BOOTSTRAP", "def broken(:\n")
    calls: list[str] = []
    _deny_push_side_effects(monkeypatch, calls)

    assert cli.cmd_env_push(_args(env_file)) == 1

    error = capsys.readouterr().err
    assert "environment entrypoint has invalid Python syntax" in error
    assert "line " in error
    assert calls == []


def test_push_syntax_error_does_not_leak_source_text(monkeypatch, tmp_path, capsys):
    secret = "super_secret_identifier"
    env_file = tmp_path / "environment.py"
    env_file.write_text(f"f'{{value!{secret}}}'\n")
    calls: list[str] = []
    _deny_push_side_effects(monkeypatch, calls)

    assert cli.cmd_env_push(_args(env_file)) == 1

    error = capsys.readouterr().err
    assert "environment entrypoint has invalid Python syntax" in error
    assert secret not in error
    assert calls == []


def test_push_rejects_invalid_bytes_for_encoding_cookie_before_side_effects(
    monkeypatch, tmp_path, capsys
):
    secret = "source_secret_marker"
    env_file = tmp_path / "environment.py"
    env_file.write_bytes(f"# coding: ascii\n{secret} = '".encode() + b"\xe9'\n")
    calls: list[str] = []
    _deny_push_side_effects(monkeypatch, calls)

    assert cli.cmd_env_push(_args(env_file)) == 1

    error = capsys.readouterr().err
    assert "environment entrypoint has invalid Python syntax" in error
    assert secret not in error
    assert calls == []


def test_push_rejects_bom_cookie_mismatch_before_side_effects(monkeypatch, tmp_path, capsys):
    env_file = tmp_path / "environment.py"
    env_file.write_bytes(b"\xef\xbb\xbf# coding: latin-1\nVALUE = 1\n")
    calls: list[str] = []
    _deny_push_side_effects(monkeypatch, calls)

    assert cli.cmd_env_push(_args(env_file)) == 1

    assert "environment entrypoint has invalid Python syntax" in capsys.readouterr().err
    assert calls == []


def test_push_preserves_valid_latin1_entrypoint_bytes_in_archive(monkeypatch, tmp_path):
    from flash.cli.commands.env.ops import push as envpush

    source = (
        b"#!/usr/bin/env python3\n"
        b"# coding: latin-1\n"
        b'"""caf\xe9 environment"""\n'
        b"from __future__ import annotations\n"
        b"from __future__ import generator_stop\n"
        b"LABEL = 'ol\xe9'"
    )
    env_file = tmp_path / "environment.py"
    env_file.write_bytes(source)
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file)) == 0

    archived = _member_bytes(cap["package_b64"])["environment.py"]
    expected = source.replace(
        b"LABEL = 'ol\xe9'",
        envpush._ENV_SYSPATH_BOOTSTRAP.encode("latin-1") + b"LABEL = 'ol\xe9'",
    )
    assert archived == expected
    assert not archived.endswith(b"\n")
    compile(archived, "environment.py", "exec")


@pytest.mark.parametrize(
    ("source", "expected_prefix"),
    [
        (b"#!/usr/bin/env python3\nVALUE = 1\n", "#!/usr/bin/env python3\n"),
        (b"# coding: latin-1\nVALUE = 1\n", "# coding: latin-1\n"),
        (b"\n# coding: latin-1\nVALUE = 1\n", "\n# coding: latin-1\n"),
        (
            b"#!/usr/bin/env python3\n# coding: latin-1\nVALUE = 1\n",
            "#!/usr/bin/env python3\n# coding: latin-1\n",
        ),
        (b"\xef\xbb\xbfVALUE = 1\n", ""),
    ],
    ids=[
        "shebang-only",
        "encoding-cookie-only",
        "second-line-encoding-cookie",
        "shebang-and-encoding-cookie",
        "utf8-bom",
    ],
)
def test_syspath_bootstrap_preserves_python_lexical_preamble(source, expected_prefix):
    import tokenize

    from flash.cli.commands.env.ops import push as envpush

    published = envpush._prepare_env_entrypoint_source(source)

    before_encoding = tokenize.detect_encoding(io.BytesIO(source).readline)[0]
    after_encoding = tokenize.detect_encoding(io.BytesIO(published).readline)[0]
    assert after_encoding == before_encoding
    decoded = published.decode(after_encoding)
    assert decoded == expected_prefix + envpush._ENV_SYSPATH_BOOTSTRAP + "VALUE = 1\n"


@pytest.mark.parametrize(
    ("source", "expected_prefix"),
    [
        ('"""environment docs"""\nVALUE = 1\n', '"""environment docs"""\n'),
        (
            "from __future__ import annotations\nVALUE: Missing = 1\n",
            "from __future__ import annotations\n",
        ),
        (
            (
                '"""environment docs"""\n'
                "from __future__ import annotations\n"
                "from __future__ import generator_stop\n"
                "VALUE: Missing = 1\n"
            ),
            (
                '"""environment docs"""\n'
                "from __future__ import annotations\n"
                "from __future__ import generator_stop\n"
            ),
        ),
        ("from __future__ import annotations", "from __future__ import annotations\n"),
    ],
    ids=["docstring", "future-import", "docstring-and-futures", "future-without-final-newline"],
)
def test_syspath_bootstrap_follows_docstring_and_future_imports(source, expected_prefix):
    import ast

    from flash.cli.commands.env.ops import push as envpush

    published = envpush._prepare_env_entrypoint_source(source.encode()).decode()

    bootstrap = envpush._ENV_SYSPATH_BOOTSTRAP
    if not source.endswith("\n"):
        bootstrap = bootstrap.removesuffix("\n")
    assert published.startswith(expected_prefix + bootstrap)
    assert published.endswith("\n") == source.endswith("\n")
    tree = ast.parse(published)
    if source.startswith('"""'):
        assert ast.get_docstring(tree, clean=False) == "environment docs"
    assert any(
        isinstance(node, ast.ImportFrom) and node.module == "__future__" for node in tree.body
    ) == ("from __future__" in source)


@pytest.mark.parametrize(
    ("prefix", "relative", "content"),
    [
        ("fslo_", "scripts/launch.sh", lambda token: f"#!/bin/sh\nexport API_KEY={token}\n"),
        ("hf_", "helper.py", lambda token: f"TOKEN = {token!r}\n"),
        ("pit_", "environment.py", lambda token: f"TOKEN = {token!r}\n"),
    ],
    ids=["freesolo-shell", "hugging-face-python", "prime-entrypoint"],
)
def test_push_rejects_direct_tokens_before_archive_or_upload(
    monkeypatch, tmp_path, capsys, prefix, relative, content
):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    target = env_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    token = _issued_token(prefix)
    target.write_text(content(token))
    calls: list[str] = []
    _deny_archive_and_upload(monkeypatch, calls)

    assert cli.cmd_env_push(_args(env_dir)) == 1

    error = capsys.readouterr().err
    assert "direct access token" in error
    assert token not in error
    assert relative not in error
    assert calls == []


@pytest.mark.parametrize(
    ("prefix", "body_length"),
    [("fslo_", 45), ("hf_", 34), ("pit_", 64)],
    ids=["freesolo", "hugging-face", "prime"],
)
@pytest.mark.parametrize("delta", [-1, 1], ids=["minus-one", "plus-one"])
def test_direct_token_neighbor_body_lengths_are_clean(tmp_path, prefix, body_length, delta):
    package = tmp_path / "package"
    package.mkdir()
    value = prefix + _token_body(prefix, body_length + delta)
    (package / "data.bin").write_bytes(b" " + value.encode() + b" ")

    assert package_contains_direct_token(package) is False


@pytest.mark.parametrize(
    "token_start",
    [_CHUNK_SIZE - 3, _CHUNK_SIZE - _OVERLAP - 3],
    ids=["read-boundary", "overlap-cutoff"],
)
def test_push_rejects_direct_token_across_chunk_boundary(
    monkeypatch, tmp_path, capsys, token_start
):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    token = _issued_token("hf_").encode()
    (env_dir / "opaque.bin").write_bytes(b" " * token_start + token + b"\x00" * 256)
    calls: list[str] = []
    _deny_archive_and_upload(monkeypatch, calls)

    assert cli.cmd_env_push(_args(env_dir)) == 1

    error = capsys.readouterr().err
    assert "direct access token" in error
    assert token.decode() not in error
    assert "opaque.bin" not in error
    assert calls == []


def test_push_allows_clean_binary_placeholders_and_embedded_direct_token_shapes(
    monkeypatch, tmp_path
):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    embedded = _issued_token("pit_")
    placeholders = [
        "fslo_" + "your_api_token_here".ljust(45, "_"),
        "hf_" + "x" * 34,
        "pit_" + "0" * 64,
    ]
    (env_dir / "environment.py").write_text(
        "def load_environment(**k):\n    return None\n" + "\n".join(placeholders)
    )
    (env_dir / "notes.txt").write_text(
        "ordinary text without credentials\n"
        "hf_resume_checkpoint\n"
        "fslo_retry_after_close\n"
        "pit_environment_identifier\n"
    )
    (env_dir / "opaque.bin").write_bytes(b"\x00\xffclean\x80binary\x00")
    (env_dir / "embedded-left.bin").write_bytes(("left" + embedded + " ").encode())
    (env_dir / "overlong.bin").write_bytes((" pit_" + "aB3" * 22 + " ").encode())
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_dir)) == 0
    assert cap["package_b64"]


def test_push_allows_direct_token_shape_with_invalid_left_at_retained_boundary(
    monkeypatch, tmp_path
):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    token = _issued_token("pit_").encode()
    prefix_padding = b" " * (_CHUNK_SIZE - _OVERLAP - 1)
    payload = prefix_padding + b"z" + token + b"\x00" + b"next chunk"
    (env_dir / "retained-boundary.bin").write_bytes(payload)
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_dir)) == 0
    assert cap["package_b64"]


def test_direct_token_scan_error_drops_oserror_details(tmp_path):
    from flash.envs.package.direct_tokens import DirectTokenScanError, package_contains_direct_token

    missing = tmp_path / "sensitive-package-path"
    with pytest.raises(DirectTokenScanError) as excinfo:
        package_contains_direct_token(missing)

    assert str(excinfo.value) == "package scan failed"
    assert excinfo.value.__context__ is None
    assert str(missing) not in str(excinfo.value)


def test_push_fails_closed_when_direct_token_scan_sees_unexpected_member(
    monkeypatch, tmp_path, capsys
):
    from flash.cli.commands.env.ops import push as envpush

    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    outside = tmp_path / "outside"
    outside.write_text("clean\n")
    real_copy = envpush._copy_env_sidecars

    def copy_with_link(env_root, dest, *, entrypoint, include_full_tree):
        real_copy(
            env_root,
            dest,
            entrypoint=entrypoint,
            include_full_tree=include_full_tree,
        )
        (dest / "unexpected-link").symlink_to(outside)

    monkeypatch.setattr(envpush, "_copy_env_sidecars", copy_with_link)
    calls: list[str] = []
    _deny_archive_and_upload(monkeypatch, calls)

    assert cli.cmd_env_push(_args(env_dir)) == 1

    error = capsys.readouterr().err
    assert "could not be scanned safely" in error
    assert "unexpected-link" not in error
    assert str(outside) not in error
    assert calls == []


def test_push_single_py_module_is_packaged(monkeypatch, tmp_path, capsys):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    cap: dict = {}
    monkeypatch.setattr(
        "flash.client.client_from_config", _fake_client(cap, slug="freesolo-co/u-environment")
    )

    rc = cli.cmd_env_push(_args(env_file, name="math-env"))
    assert rc == 0
    files = _members(cap["package_b64"])
    assert "environment.py" in files
    assert "pyproject.toml" not in files
    assert not any(name.startswith("freesolo/") for name in files)
    assert cap["name"] == "math-env"
    assert "published freesolo-co/u-environment" in capsys.readouterr().out


def test_push_single_py_module_carries_its_evaluations_sidecar(monkeypatch, tmp_path):
    # `env eval` loads evaluations.py next to an exact .py target, so a single-file push that
    # dropped it would publish an environment whose suite passed locally and is simply gone
    # once pushed -- a green local check for a package that cannot reproduce it.
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "evaluations.py").write_text("def load_evaluations(environment=None): return []\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file, name="math-env")) == 0
    files = _members(cap["package_b64"])
    assert "environment.py" in files
    assert "evaluations.py" in files


def test_push_single_py_module_carries_direct_evaluation_helper_imports(monkeypatch, tmp_path):
    # evaluations.py can import sibling helpers locally, so dropping those direct imports from an
    # exact-file push publishes a suite that passed env test but cannot load from the package.
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "evaluations.py").write_text(
        "import eval_constants\n"
        "from eval_scorer import score\n"
        "def load_evaluations(environment=None): return []\n"
    )
    (tmp_path / "eval_constants.py").write_text("EXPECTED = '4'\n")
    (tmp_path / "eval_scorer.py").write_text("def score(value): return value == '4'\n")
    (tmp_path / "unrelated.py").write_text("VALUE = 'do not publish'\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file, name="math-env")) == 0
    files = _members(cap["package_b64"])
    assert "eval_constants.py" in files
    assert "eval_scorer.py" in files
    assert "unrelated.py" not in files


def test_push_single_py_module_carries_lazily_imported_evaluation_helpers(monkeypatch, tmp_path):
    """A helper imported inside cases()/score() ships too.

    The loader deliberately keeps the package dir on sys.path so a sidecar can import its helpers
    lazily, and `env test` exercises that path successfully. Scanning only top-level nodes omitted
    those helpers from the package, so the suite passed the offline gate and then failed to import
    the first time it graded a case against the pushed environment.
    """
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "evaluations.py").write_text(
        "def load_evaluations(environment=None):\n"
        "    import eval_lazy\n"
        "    try:\n"
        "        from eval_guarded import score\n"
        "    except ImportError:\n"
        "        score = None\n"
        "    return []\n"
    )
    (tmp_path / "eval_lazy.py").write_text("EXPECTED = '4'\n")
    (tmp_path / "eval_guarded.py").write_text("def score(value): return True\n")
    (tmp_path / "unrelated.py").write_text("VALUE = 'do not publish'\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file, name="math-env")) == 0
    files = _members(cap["package_b64"])
    assert "eval_lazy.py" in files
    assert "eval_guarded.py" in files
    # widening to nested imports must not turn the bounded single-file mode into a whole-dir push.
    assert "unrelated.py" not in files


@pytest.mark.parametrize("namespace", [False, True], ids=["regular-package", "namespace-package"])
def test_push_single_py_module_carries_an_imported_sibling_package(
    monkeypatch, tmp_path, namespace: bool
):
    """`from graders.rules import score` ships the graders/ package, not just graders.py.

    Local loading succeeds because the environment directory stays on sys.path, so every offline
    check passes. Matching only the `<name>.py` spelling published an environment whose sidecar
    raises ModuleNotFoundError the first time it grades a case.

    `__init__.py` is not what makes it a package: under PEP 420 a bare directory imports the same
    way, so requiring the marker file dropped exactly the helper the sidecar needs while every
    local check still passed.
    """
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "evaluations.py").write_text(
        "from graders.rules import score\n\ndef load_evaluations(environment=None): return []\n"
    )
    graders = tmp_path / "graders"
    (graders / "nested").mkdir(parents=True)
    if not namespace:
        (graders / "__init__.py").write_text("")
        (graders / "nested" / "__init__.py").write_text("")
    (graders / "rules.py").write_text("def score(value): return True\n")
    (graders / "nested" / "deep.py").write_text("THRESHOLD = 0.5\n")
    unused = tmp_path / "unused_pkg"
    unused.mkdir()
    (unused / "__init__.py").write_text("VALUE = 'do not publish'\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file, name="math-env")) == 0
    files = _members(cap["package_b64"])
    assert "graders/rules.py" in files
    # the whole package tree ships: a subpackage the imported module itself uses is part of it.
    assert "graders/nested/deep.py" in files
    if not namespace:
        assert "graders/__init__.py" in files
    # but an unimported sibling package still must not turn this into a whole-dir push.
    assert "unused_pkg/__init__.py" not in files


def test_push_ships_the_package_when_a_module_of_the_same_name_shadows_it(monkeypatch, tmp_path):
    """With both graders.py and graders/, `import graders` is the package -- so ship the package.

    Python's path finder checks directories before same-named modules. Yielding the .py first and
    skipping the package published the file the sidecar never imports and dropped the one it does,
    so the pushed environment raised ModuleNotFoundError on `graders.rules`.
    """
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "evaluations.py").write_text(
        "from graders.rules import score\n\ndef load_evaluations(environment=None): return []\n"
    )
    (tmp_path / "graders.py").write_text("SHADOWED = True\n")
    graders = tmp_path / "graders"
    graders.mkdir()
    (graders / "__init__.py").write_text("")
    (graders / "rules.py").write_text("def score(value): return True\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file, name="math-env")) == 0
    files = _members(cap["package_b64"])
    assert "graders/rules.py" in files
    assert "graders/__init__.py" in files


def test_push_ships_a_namespace_package_helper(monkeypatch, tmp_path):
    """A PEP 420 `graders/` with no __init__.py is still importable, so it still has to ship.

    Requiring the marker file sent the helper to the `graders.py` fallback, which does not exist
    either, so the archive carried evaluations.py alone and the published environment raised
    ModuleNotFoundError on its first case.
    """
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "evaluations.py").write_text(
        "from graders.rules import score\n\ndef load_evaluations(environment=None): return []\n"
    )
    graders = tmp_path / "graders"
    graders.mkdir()
    (graders / "rules.py").write_text("def score(value): return True\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file, name="math-env")) == 0
    files = _members(cap["package_b64"])
    assert "graders/rules.py" in files


def test_push_prefers_the_module_over_a_namespace_directory(monkeypatch, tmp_path):
    """With graders.py and a bare graders/, `import graders` binds the MODULE.

    Only a regular package (one holding __init__.py) outranks a same-named module; a namespace
    directory is merely a fallback portion. Shipping the directory here would publish the file
    the sidecar never imports.
    """
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "evaluations.py").write_text(
        "import graders\n\ndef load_evaluations(environment=None): return []\n"
    )
    (tmp_path / "graders.py").write_text("WINS = True\n")
    graders = tmp_path / "graders"
    graders.mkdir()
    (graders / "rules.py").write_text("def score(value): return True\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file, name="math-env")) == 0
    files = _members(cap["package_b64"])
    assert "graders.py" in files


def test_push_ships_helpers_a_helper_imports(monkeypatch, tmp_path):
    """A helper's own siblings ship too, or the published sidecar fails on its first case.

    An exact-file push that packaged `scorer.py` but not the `thresholds` it imports passed
    every local check -- the source directory is importable -- and raised ModuleNotFoundError
    once published.
    """
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "evaluations.py").write_text(
        "import scorer\n\ndef load_evaluations(environment=None): return []\n"
    )
    (tmp_path / "scorer.py").write_text("from thresholds import CUTOFF\n")
    (tmp_path / "thresholds.py").write_text("CUTOFF = 0.5\n")
    # reached only through thresholds.py, so it proves the walk keeps following, not just one hop
    (tmp_path / "units.py").write_text("SCALE = 2\n")
    (tmp_path / "thresholds.py").write_text("import units\n\nCUTOFF = 0.5\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file, name="math-env")) == 0
    files = _members(cap["package_b64"])
    assert "scorer.py" in files
    assert "thresholds.py" in files
    assert "units.py" in files


def test_push_ships_helpers_named_by_a_literal_dynamic_import(monkeypatch, tmp_path):
    """`import_module("judge")` names a sibling as surely as an import statement does.

    Scanning only Import/ImportFrom omitted the helper from the package, and the suite passed
    every local check -- the sidecar scope makes the directory importable -- then failed on its
    first published case.
    """
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "evaluations.py").write_text(
        "import importlib\n\n"
        "def load_evaluations(environment=None):\n"
        "    importlib.import_module('judge')\n"
        "    __import__('rubric')\n"
        "    return []\n"
    )
    # reached only through judge.py, so a dynamically named helper is followed like any other
    (tmp_path / "judge.py").write_text("import weights\n")
    (tmp_path / "weights.py").write_text("W = 1\n")
    (tmp_path / "rubric.py").write_text("RULES = ()\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file, name="math-env")) == 0
    files = _members(cap["package_b64"])
    assert "judge.py" in files
    assert "rubric.py" in files
    assert "weights.py" in files


def test_push_ships_helpers_named_through_an_alias_of_import_module(monkeypatch, tmp_path):
    """`from importlib import import_module as load` imports exactly as the canonical name does.

    Matching the call's identifier against `import_module` alone skipped the aliased call, so
    `judge.py` was left out of the archive for a suite that passes locally -- its directory is
    importable there -- and raises ModuleNotFoundError on its first published case.
    """
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "evaluations.py").write_text(
        "from importlib import import_module as load\n\n"
        "def load_evaluations(environment=None):\n"
        "    load('judge')\n"
        "    return []\n"
    )
    # reached only through judge.py, so the alias is followed as far as any other import
    (tmp_path / "judge.py").write_text("import weights\n")
    (tmp_path / "weights.py").write_text("W = 1\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file, name="math-env")) == 0
    files = _members(cap["package_b64"])
    assert "judge.py" in files
    assert "weights.py" in files


def test_push_does_not_read_an_unrelated_load_call_as_a_dynamic_import(monkeypatch, tmp_path):
    """The alias is a per-file binding, not a reserved word.

    A sidecar that never imports `import_module` may still call something named `load`, and
    packaging whatever string it was handed would ship files on a guess. Only a name this module
    actually bound to `import_module` counts.
    """
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "evaluations.py").write_text(
        "import json\n\n"
        "def load_evaluations(environment=None):\n"
        "    json.load('judge')\n"
        "    return []\n"
    )
    (tmp_path / "judge.py").write_text("SHOULD_NOT_SHIP = 1\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file, name="math-env")) == 0
    assert "judge.py" not in _members(cap["package_b64"])


def test_push_ships_helpers_named_by_the_keyword_form_of_a_dynamic_import(monkeypatch, tmp_path):
    """`import_module(name="judge")` imports identically to the positional form.

    Reading only `node.args[0]` skipped the keyword spelling, so the helper never entered the
    archive and the suite raised ModuleNotFoundError on its first published case -- the same
    failure the positional scan exists to prevent.
    """
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "evaluations.py").write_text(
        "import importlib\n\n"
        "def load_evaluations(environment=None):\n"
        "    importlib.import_module(name='judge')\n"
        "    __import__(name='rubric')\n"
        "    return []\n"
    )
    (tmp_path / "judge.py").write_text("import weights\n")
    (tmp_path / "weights.py").write_text("W = 1\n")
    (tmp_path / "rubric.py").write_text("RULES = ()\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file, name="math-env")) == 0
    files = _members(cap["package_b64"])
    assert "judge.py" in files
    assert "rubric.py" in files
    # followed transitively, exactly as a positionally named helper is
    assert "weights.py" in files


def test_push_ships_helpers_named_through_an_assignment_bound_dynamic_import(monkeypatch, tmp_path):
    """`load = importlib.import_module` binds the importer without any `import ... as`.

    The alias walk read import statements, so the from-import spelling was covered and this one --
    which appears in no import statement at all -- was not. Same end state either way:
    the helper stays out of the archive, the suite passes locally because its directory is on
    sys.path, and the published environment raises ModuleNotFoundError on its first case.

    The chain is two hops deep and declared out of order, so a single pass over the assignments in
    source order resolves neither `pick` nor the helper it names.
    """
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "evaluations.py").write_text(
        "import importlib\n\n"
        # `pick` is bound from `load` BEFORE `load` itself is bound, so resolving this needs a
        # fixpoint rather than one ordered sweep.
        "pick = load\n"
        "load = importlib.import_module\n"
        "grab = __import__\n\n"
        "def load_evaluations(environment=None):\n"
        "    pick('judge')\n"
        "    grab('rubric')\n"
        "    return []\n"
    )
    (tmp_path / "judge.py").write_text("import weights\n")
    (tmp_path / "weights.py").write_text("W = 1\n")
    (tmp_path / "rubric.py").write_text("RULES = ()\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file, name="math-env")) == 0
    files = _members(cap["package_b64"])
    assert "judge.py" in files
    assert "rubric.py" in files
    # and followed transitively, exactly as a directly named helper is
    assert "weights.py" in files


def test_alias_chains_resolve_without_rescanning_every_binding() -> None:
    """A long reverse-ordered chain must cost one look per binding, not one sweep per link.

    Re-scanning every assignment per pass resolves exactly one new name per pass when the chain is
    declared in reverse -- the ordering this walk explicitly supports -- so the cost is quadratic:
    a generated 5,000-link sidecar spent ~3.4s here, and `env push` walks again while copying, so
    it paid that twice before any archive limit applied.

    Counts reads of the bound name on the real function rather than timing it. A wall-clock bound
    is flaky on a shared runner and would not say why it got slow, whereas the read count separates
    the two shapes by two orders of magnitude at n=300: ~n for a work queue against ~n**2/2 for
    repeated sweeps.
    """
    import ast

    from flash.cli.commands.env.ops import imports as envimports

    n = 300

    reads = 0

    class CountingName(ast.Name):
        """An `ast.Name` that records each time the walk reads the identifier it binds.

        `id` shadows the builtin, but that is the attribute name `ast.Name` itself uses and the
        walk reads, so counting reads means defining it under exactly that name.
        """

        def __getattribute__(self, attr):
            if attr == "id":
                nonlocal reads
                reads += 1
            return object.__getattribute__(self, attr)

    # reverse dependency order: each link names the next and only the last reaches the importer,
    # so a sweep-based walk resolves one hop per pass and re-reads every other binding meanwhile.
    src = "\n".join(
        ["import importlib"]
        + [f"a{i} = a{i + 1}" for i in range(n - 1)]
        + [f"a{n - 1} = importlib.import_module"]
    )
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
            node.value.__class__ = CountingName

    reads = 0
    names = envimports._dynamic_import_callees(tree)

    # every link resolved, in either declaration order
    assert {f"a{i}" for i in range(n)} <= set(names)
    # and each binding's source was read a bounded number of times rather than once per pass. the
    # quadratic form reads ~n**2/2 = ~45,000 for n=300; a small constant per binding is fine.
    assert reads <= 4 * n, (
        f"read bound sources {reads} times for {n} bindings -- "
        "the walk is rescanning rather than resolving each binding once"
    )


def test_push_does_not_mistake_an_unrelated_callable_for_a_dynamic_import(monkeypatch, tmp_path):
    """Binding some other function to a plausible name must not make its argument a module.

    The guard against over-matching: `load` is an ordinary name, and a sidecar that binds it to
    something unrelated calls it with strings that are not module names. Shipping a file per such
    call would be wrong in the quiet direction -- the archive grows and nothing points at why.
    """
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "evaluations.py").write_text(
        "import os.path\n\n"
        "load = os.path.join\n\n"
        "def load_evaluations(environment=None):\n"
        "    load('judge')\n"
        "    return []\n"
    )
    # present on disk, so the assertion is about the walker's judgement rather than the file
    # simply being absent: a match would package it.
    (tmp_path / "judge.py").write_text("SHOULD_NOT_SHIP = 1\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file, name="math-env")) == 0
    assert "judge.py" not in _members(cap["package_b64"])


def test_push_keeps_a_noncanonical_entrypoint_importable_by_its_local_name(monkeypatch, tmp_path):
    """`from custom import SCORER` must keep resolving after custom.py is published as environment.py.

    Packaging renames the entrypoint, so a sidecar importing it by its local name resolved
    locally and raised ModuleNotFoundError once published. The alias rebinds sys.modules so both
    names give one module object rather than two copies of its state.
    """
    env_dir = tmp_path / "single-file-env"
    env_dir.mkdir()
    (env_dir / "custom.py").write_text(
        "SCORER = 'gold'\n\ndef load_environment(**k):\n    return None\n"
    )
    (env_dir / "evaluations.py").write_text(
        "from custom import SCORER\n\ndef load_evaluations(environment=None): return []\n"
    )
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    # the exact .py file, which is the supported way to publish an entrypoint not named
    # environment.py. that push still renames, so the alias is still what keeps `import custom`
    # resolving to the same module object the runner loaded.
    assert cli.cmd_env_push(_args(env_dir / "custom.py", name="single-file-env")) == 0
    files = _members(cap["package_b64"])
    assert "custom.py" in files
    # the published tree must import cleanly, with both names bound to ONE module object
    (tmp_path / "published").mkdir()
    for name, body in files.items():
        target = tmp_path / "published" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import custom, environment; "
                "assert custom is environment, (custom, environment); "
                "print(custom.SCORER)"
            ),
        ],
        cwd=tmp_path / "published",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "gold"


def test_push_dir_without_the_canonical_entrypoint_is_refused(monkeypatch, tmp_path, capsys):
    # a directory package names its entrypoint environment.py. inferring it from "the sole
    # top-level module" made the answer depend on which OTHER files were present, so adding a
    # second module turned a working push into a rejection. the directory is refused outright
    # now, and the message names both fixes rather than leaving the user to guess.
    env_dir = tmp_path / "no-entrypoint-env"
    env_dir.mkdir()
    (env_dir / "custom.py").write_text("def load_environment(**k):\n    return None\n")
    (env_dir / "evaluations.py").write_text("def load_evaluations(environment=None): return []\n")
    monkeypatch.setattr("flash.client.client_from_config", _fake_client({}))

    assert cli.cmd_env_push(_args(env_dir, name="no-entrypoint-env")) == 1
    # the message must name the canonical entrypoint AND the single-file escape hatch, or the
    # user is told only that the push failed.
    out = capsys.readouterr()
    message = out.err + out.out
    assert "environment.py" in message
    assert ".py file" in message


def test_push_dir_with_pyproject_uses_explicit_name(monkeypatch, tmp_path):
    env_dir = tmp_path / "my-env"
    env_dir.mkdir()
    (env_dir / "pyproject.toml").write_text('[project]\nname = "my-env"\nversion = "0.1.0"\n')
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    (env_dir / "source").mkdir()
    (env_dir / "source" / "index.ts").write_text("console.log('agent workspace')\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    rc = cli.cmd_env_push(_args(env_dir, name="explicit-env"))
    assert rc == 0
    files = _members(cap["package_b64"])
    assert "environment.py" in files
    assert "pyproject.toml" not in files
    assert not any(name.startswith("source/") for name in files)
    assert cap["name"] == "explicit-env"


def test_push_preserves_explicit_namespace(monkeypatch, tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    cap: dict = {}
    monkeypatch.setattr(
        "flash.client.client_from_config", _fake_client(cap, slug="benchmark/project/math")
    )

    assert cli.cmd_env_push(_args(env_file, name="benchmark/project/Math Env")) == 0
    assert cap["name"] == "benchmark/project/math-env"


def test_push_dir_prefers_environment_py_and_ships_helpers(monkeypatch, tmp_path):
    env_dir = tmp_path / "math"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text(
        "import helper\n\ndef load_environment(**k):\n    return helper.build()\n"
    )
    (env_dir / "helper.py").write_text("def build():\n    return None\n")
    (env_dir / "evaluations.py").write_text("EVALUATIONS = []\n")
    (env_dir / "dataset").mkdir()
    (env_dir / "dataset" / "train.jsonl").write_text('{"input":"2+2","output":"4"}\n')
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_dir, name="math")) == 0
    files = _members(cap["package_b64"])
    assert "environment.py" in files
    assert "helper.py" in files
    assert "evaluations.py" in files
    assert "dataset/train.jsonl" in files
    assert cap["name"] == "math"


def test_push_single_py_ships_only_entrypoint_and_selected_sidecars(monkeypatch, tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "train.jsonl").write_text('{"x": 1}\n')
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "eval.jsonl").write_text('{"x": 2}\n')
    (tmp_path / "runtime.TOML").write_text('model = "qwen"\n')
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    (tmp_path / "credentials.toml").write_text('token = "secret"\n')
    (tmp_path / "not-a-file.toml").mkdir()
    (tmp_path / "not-a-file.toml" / "nested.toml").write_text("[nested]\n")
    configs = tmp_path / "configs"
    (configs / "nested").mkdir(parents=True)
    (configs / "base.toml").write_text("batch_size = 8\n")
    (configs / "nested" / "tuning.TOML").write_text("learning_rate = 0.001\n")
    (configs / "ignored.yaml").write_text("mode: prod\n")
    (tmp_path / "helper.py").write_text("VALUE = 1\n")
    (tmp_path / "config.yaml").write_text("mode: prod\n")
    (tmp_path / "unrelated").mkdir()
    (tmp_path / "unrelated" / "data.json").write_text("{}\n")
    (tmp_path / "unrelated" / "other.toml").write_text("publish = false\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file)) == 0
    files = set(_members(cap["package_b64"]))
    assert files == {
        "README.md",
        "configs/base.toml",
        "configs/nested/tuning.TOML",
        "dataset/train.jsonl",
        "datasets/eval.jsonl",
        "environment.py",
        "runtime.TOML",
    }


def test_push_directory_recursively_ships_full_tree(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("from exploit_env import verifier\n")
    package = env_dir / "exploit_env"
    package.mkdir()
    (package / "__init__.py").write_text("from .verifier import verify\n")
    (package / "verifier.py").write_text("def verify():\n    return True\n")
    (env_dir / "dataset").mkdir()
    (env_dir / "dataset" / "train.jsonl").write_text('{"x": 1}\n')
    (env_dir / "datasets").mkdir()
    (env_dir / "datasets" / "eval.parquet").write_bytes(b"parquet bytes")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_dir)) == 0
    files = _members(cap["package_b64"])
    assert "exploit_env/__init__.py" in files
    assert "exploit_env/verifier.py" in files
    assert "dataset/train.jsonl" in files
    assert "datasets/eval.parquet" in files


def test_push_directory_ships_full_environment_tree(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    (env_dir / "state.sqlite").write_text("sqlite bytes")
    (env_dir / "db").mkdir()
    (env_dir / "db" / "state.sqlite").write_text("sqlite bytes")
    (env_dir / "configs").mkdir()
    (env_dir / "configs" / "env.toml").write_text("[env]\n")
    (env_dir / "rl.toml").write_text('algorithm = "grpo"\n')
    (env_dir / "assets").mkdir()
    (env_dir / "assets" / "labels.json").write_text("{}\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_dir)) == 0
    files = _members(cap["package_b64"])
    assert "state.sqlite" in files
    assert "db/state.sqlite" in files
    assert "configs/env.toml" in files
    assert "rl.toml" in files
    assert "assets/labels.json" in files


def test_push_ships_training_md_and_keeps_user_readme(monkeypatch, tmp_path):
    # The TRAINING.md `flash env setup` scaffolds — and a user-authored README — must travel into
    # the pushed package so they land in the hub and round-trip back through `flash env pull`.
    env_dir = tmp_path / "math"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    (env_dir / "TRAINING.md").write_text("# TRAINING.md\n\nplaybook body\n")
    (env_dir / "README.md").write_text("# math\n\nmy own readme\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_dir, name="math")) == 0
    files = _members(cap["package_b64"])
    assert "playbook body" in files["TRAINING.md"]
    # the user's README survives — it is NOT clobbered by the synthesized stub
    assert "my own readme" in files["README.md"]


def test_push_synthesizes_readme_stub_when_absent(monkeypatch, tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file, name="math-env")) == 0
    files = _members(cap["package_b64"])
    assert "# math-env" in files["README.md"]


def test_push_single_py_does_not_ship_unimported_sibling_modules(monkeypatch, tmp_path):
    """An exact-file push carries a closure, not the directory it happens to sit in.

    A sibling nothing imports stays local: shipping every neighbour would turn `env push
    environment.py` into a whole-tree push and carry unrelated scratch files into the archive.
    """
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "helper.py").write_text("VALUE = 1\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file)) == 0
    assert "helper.py" not in _members(cap["package_b64"])


def test_push_single_py_module_carries_its_own_helper_imports(monkeypatch, tmp_path):
    """The entrypoint's own imports ship, exactly as its evaluation sidecar's already do.

    The closure walk was seeded only from evaluations.py, so an environment whose entrypoint
    imported siblings published without them: the push exits 0 and prints an id, and the
    ModuleNotFoundError surfaces only after a GPU is rented and the worker imports the module.
    """
    env_file = tmp_path / "environment.py"
    env_file.write_text(
        "import config\nfrom utils import load_jsonl\ndef load_environment(**k):\n    return None\n"
    )
    (tmp_path / "config.py").write_text("MODEL = 'qwen'\n")
    (tmp_path / "utils.py").write_text("import data\n\ndef load_jsonl(p): return []\n")
    # reached only through utils.py, so it proves the entrypoint walk keeps following
    (tmp_path / "data.py").write_text("ROWS = []\n")
    (tmp_path / "unrelated.py").write_text("VALUE = 'do not publish'\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file, name="math-env")) == 0
    files = _members(cap["package_b64"])
    assert "config.py" in files
    assert "utils.py" in files
    assert "data.py" in files
    assert "unrelated.py" not in files


def test_push_single_py_module_carries_helpers_named_by_a_relative_import(monkeypatch, tmp_path):
    """`from . import config` names a sibling too, under the package-relative spelling.

    An entrypoint authored to work both as a package member and as a loose module writes the
    relative form first and falls back to the absolute one. Reading only absolute imports left
    the helper unpackaged whenever the relative spelling came first.
    """
    env_file = tmp_path / "environment.py"
    env_file.write_text(
        "try:\n"
        "    from . import config\n"
        "    from .utils import load_jsonl\n"
        "except ImportError:\n"
        "    import config\n"
        "    from utils import load_jsonl\n"
        "def load_environment(**k):\n"
        "    return None\n"
    )
    (tmp_path / "config.py").write_text("MODEL = 'qwen'\n")
    (tmp_path / "utils.py").write_text("def load_jsonl(p): return []\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file, name="math-env")) == 0
    files = _members(cap["package_b64"])
    assert "config.py" in files
    assert "utils.py" in files


def test_push_relative_import_inside_a_package_does_not_name_a_root_module(monkeypatch, tmp_path):
    """`from . import config` in graders/__init__.py names graders/config.py, not ./config.py.

    Every name the closure collects is resolved against the env root, so reading a relative import
    from a file nested inside a packaged subdirectory published an unrelated top-level module while
    the real sibling was already shipped by the package walk.
    """
    env_file = tmp_path / "environment.py"
    env_file.write_text("import graders\n\ndef load_environment(**k):\n    return None\n")
    graders = tmp_path / "graders"
    graders.mkdir()
    (graders / "__init__.py").write_text("from . import config\n")
    (graders / "config.py").write_text("THRESHOLD = 0.5\n")
    # a root-level module of the same name that the entrypoint never imports. resolving the
    # package's relative import against the env root shipped THIS file instead.
    (tmp_path / "config.py").write_text("SECRET = 'do not publish'\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file, name="math-env")) == 0
    files = _members(cap["package_b64"])
    assert "graders/config.py" in files
    assert "config.py" not in files


def test_push_entrypoint_importing_evaluations_charges_the_sidecar_once(monkeypatch, tmp_path):
    """The eval sidecar the entrypoint imports is published once, not twice.

    The entrypoint closure runs first and shares `yielded` with the sidecar block below it. With no
    membership guard the sidecar was yielded a second time, and `_check_env_push_limits` charged
    those bytes and that member twice -- rejecting a tree that is actually under the limit.
    """
    from flash.cli.commands.env.ops import push as envpush

    env_file = tmp_path / "environment.py"
    env_file.write_text("import evaluations\n\ndef load_environment(**k):\n    return None\n")
    (tmp_path / "evaluations.py").write_text(
        "import scorers\n\ndef load_evaluations(environment=None): return []\n"
    )
    (tmp_path / "scorers.py").write_text("def score(v): return 1.0\n")
    # the walk runs once to charge `_check_env_push_limits` and again to copy, so yields are
    # counted per walk: the defect was a path yielded twice WITHIN one walk.
    walks: list[list[str]] = []
    real = envpush._iter_env_sidecar_files

    def _tracking(env_root, *, entrypoint, include_full_tree):
        walk: list[str] = []
        walks.append(walk)
        for src, rel in real(env_root, entrypoint=entrypoint, include_full_tree=include_full_tree):
            walk.append(rel.as_posix())
            yield src, rel

    monkeypatch.setattr(envpush, "_iter_env_sidecar_files", _tracking)
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file, name="math-env")) == 0
    assert walks, "the sidecar walk never ran"
    assert [w.count("evaluations.py") for w in walks] == [1] * len(walks)
    # the sidecar's own imports are still followed even though its yield was skipped.
    files = _members(cap["package_b64"])
    assert "scorers.py" in files


def test_push_alternate_py_keeps_packaged_entrypoint(monkeypatch, tmp_path):
    env_file = tmp_path / "custom_env.py"
    env_file.write_text("def load_environment(**k):\n    return 'custom'\n")
    (tmp_path / "environment.py").write_text("def load_environment(**k):\n    return 'sibling'\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file)) == 0
    files = _members(cap["package_b64"])
    assert "return 'custom'" in files["environment.py"]
    assert "return 'sibling'" not in files["environment.py"]
    # the entrypoint's SOURCE is published once, under the canonical name. custom_env.py exists
    # only as an import alias so a sidecar's `import custom_env` still resolves, and it must not
    # carry a second copy of the module body.
    assert "return 'custom'" not in files["custom_env.py"]
    assert "sys.modules[__name__] = _environment" in files["custom_env.py"]


def test_push_requires_explicit_name(tmp_path, capsys):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    assert cli.cmd_env_push(argparse.Namespace(path=str(env_file))) == 1
    assert "--name" in capsys.readouterr().err


def test_push_names_the_missing_project_segment_for_a_legacy_id(tmp_path, capsys):
    """The old two-segment id must not be reported as a missing flag.

    `namespace/name` is the form every pre-existing script passes, and it is now rejected
    because names are unique per project. Answering it with "env name required: pass --name"
    sends the user hunting for a flag they demonstrably did pass, and says nothing about the
    segment that is actually missing.
    """
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")

    assert cli.cmd_env_push(argparse.Namespace(path=str(env_file), name="acme/math")) == 1

    err = capsys.readouterr().err
    assert "acme/math" in err
    assert "<namespace>/<project>/<name>" in err
    # the misleading answer is specifically what this guards against.
    assert "env name required" not in err


def test_push_blames_the_name_segment_not_the_project_when_the_shape_is_right(tmp_path, capsys):
    """A three-segment id that fails normalization failed on its NAME, not its shape.

    Telling the user to add a project segment they already passed sends them to fix the one
    part of the id that is correct.
    """
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")

    assert cli.cmd_env_push(argparse.Namespace(path=str(env_file), name="acme/proj/---")) == 1

    err = capsys.readouterr().err
    assert "no usable characters" in err
    assert "<namespace>/<project>/<name>" not in err


def test_push_still_reports_a_genuinely_absent_name(tmp_path, capsys):
    """No `--name` at all keeps the flag-shaped message, which is the right one there."""
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")

    assert cli.cmd_env_push(argparse.Namespace(path=str(env_file), name="")) == 1

    assert "env name required" in capsys.readouterr().err


def test_push_sibling_config_does_not_override_explicit_name(monkeypatch, tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "rl.toml").write_text(
        'model = "m"\nalgorithm = "grpo"\n[environment]\nid = "user/old-name"\n'
    )
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file, name="new-name")) == 0
    names = set(_members(cap["package_b64"]))
    assert cap["name"] == "new-name"
    assert "environment.py" in names
    assert "rl.toml" in names


def test_push_needs_no_local_github_credentials(monkeypatch, tmp_path):
    # The client does not need GitHub credentials; publishing is server-side.
    monkeypatch.setattr("shutil.which", lambda name: None)
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file)) == 0
    assert cap["package_b64"]


def test_push_forwards_project_id(monkeypatch, tmp_path):
    """`--project <id>` groups the published env under that project."""
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert (
        cli.cmd_env_push(_args(env_file, project="  11111111-1111-4111-8111-111111111111  ")) == 0
    )
    # a pasted id is stripped before it travels, so the resolver never sees the padding.
    assert cap["project_id"] == "11111111-1111-4111-8111-111111111111"


@pytest.mark.parametrize("project", [None, "", "   "])
def test_push_requires_project_before_packaging(monkeypatch, tmp_path, capsys, project):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file, project=project)) == 1
    assert "project id is required" in capsys.readouterr().err
    assert cap == {}


def test_push_reports_server_error(monkeypatch, tmp_path, capsys):
    from flash.client import ClientError

    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")

    class _C:
        def publish_env(self, **k):
            raise ClientError("not logged in — run `flash login`")

    monkeypatch.setattr("flash.client.client_from_config", lambda: _C())
    assert cli.cmd_env_push(_args(env_file)) == 1
    assert "not logged in" in capsys.readouterr().err


def test_push_nonexistent_path(tmp_path, capsys):
    rc = cli.cmd_env_push(_args(tmp_path / "nope.py"))
    assert rc == 1
    assert "no such path" in capsys.readouterr().err


def test_push_rejects_symlink_entrypoints(tmp_path, capsys):
    outside = tmp_path / "outside.py"
    outside.write_text("SECRET = True\n")
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").symlink_to(outside)

    assert cli.cmd_env_push(_args(env_dir)) == 1
    assert "symlinks are not allowed" in capsys.readouterr().err


def test_push_excludes_secrets_metadata_caches_and_symlinks_at_all_depths(monkeypatch, tmp_path):
    env_dir = tmp_path / "my-env"
    env_dir.mkdir()
    (env_dir / "pyproject.toml").write_text('[project]\nname = "my-env"\nversion = "0.1.0"\n')
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    (env_dir / ".env").write_text("TOKEN=secret\n")
    (env_dir / "credentials.json").write_text('{"token": "secret"}\n')
    (env_dir / "client.P12").write_bytes(b"secret")
    (env_dir / ".prime").mkdir()
    (env_dir / ".prime" / ".env-metadata.json").write_text('{"owner": "someone-else"}')
    nested = env_dir / "helpers" / "nested"
    nested.mkdir(parents=True)
    (nested / "safe.py").write_text("VALUE = 1\n")
    (nested / "pyproject.toml").write_text("[project]\n")
    (nested / "Credentials.YAML").write_text("token: secret\n")
    (nested / "source").mkdir()
    (nested / "source" / "index.py").write_text("SECRET = True\n")
    (nested / "secrets").mkdir()
    (nested / "secrets" / "api.key").write_text("secret\n")
    (nested / "__pycache__").mkdir()
    (nested / "__pycache__" / "safe.pyc").write_bytes(b"junk")
    (nested / ".venv").mkdir()
    (nested / ".venv" / "activate.py").write_text("secret\n")
    (nested / ".git").mkdir()
    (nested / ".git" / "config").write_text("secret\n")
    outside_file = tmp_path / "outside.py"
    outside_file.write_text("SECRET = True\n")
    outside_dir = tmp_path / "outside-data"
    outside_dir.mkdir()
    (outside_dir / "secret.json").write_text('{"secret": true}\n')
    (nested / "linked.py").symlink_to(outside_file)
    (nested / "linked-data").symlink_to(outside_dir, target_is_directory=True)
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_dir, name="clean-env")) == 0
    names = set(_members(cap["package_b64"]))
    assert "environment.py" in names
    assert "helpers/nested/safe.py" in names
    assert "helpers/nested/source/index.py" in names
    assert "helpers/nested/pyproject.toml" in names
    assert ".env" not in names
    assert "credentials.json" not in names
    assert "client.P12" not in names
    assert "helpers/nested/Credentials.YAML" not in names
    assert not any(name.startswith(".prime") for name in names)
    assert not any("__pycache__" in name for name in names)
    assert not any("/.venv/" in f"/{name}/" for name in names)
    assert not any("/.git/" in f"/{name}/" for name in names)
    assert "helpers/nested/secrets/api.key" not in names
    assert "helpers/nested/linked.py" not in names
    assert "helpers/nested/linked-data/secret.json" not in names


def test_push_excludes_root_only_paths_but_keeps_nested_package_content(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text(
        "from my_pkg.source.loader import load\nfrom my_pkg.config.nested import CONFIG\n"
    )
    (env_dir / "pyproject.toml").write_text("[project]\n")
    (env_dir / "source").mkdir()
    (env_dir / "source" / "root.py").write_text("ROOT = True\n")
    (env_dir / "my_pkg" / "source").mkdir(parents=True)
    (env_dir / "my_pkg" / "source" / "loader.py").write_text("def load(): pass\n")
    (env_dir / "my_pkg" / "config").mkdir()
    (env_dir / "my_pkg" / "config" / "nested.py").write_text("CONFIG = {}\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_dir)) == 0
    names = set(_members(cap["package_b64"]))
    assert "my_pkg/source/loader.py" in names
    assert "my_pkg/config/nested.py" in names
    assert "pyproject.toml" not in names
    assert "source/root.py" not in names


def test_push_rejects_oversized_directory_before_packaging(monkeypatch, tmp_path, capsys):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    source = "def load_environment(**k):\n    return None\n"
    (env_dir / "environment.py").write_text(source)
    (env_dir / "checkpoint.bin").write_bytes(b"x" * 65)
    monkeypatch.setattr("flash.cli.commands.env.ops.push._ENV_PUSH_MAX_TOTAL_BYTES", 64)
    monkeypatch.setattr(
        "flash.client.client_from_config",
        lambda: (_ for _ in ()).throw(AssertionError("upload must not start")),
    )

    assert cli.cmd_env_push(_args(env_dir)) == 1
    error = capsys.readouterr().err
    assert "environment package totals" in error
    assert "(limit 64 B)" in error
    assert "remove large artifacts or use a smaller dataset" in error


def test_push_does_not_emit_empty_directories(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    (env_dir / "empty").mkdir()
    ignored_only = env_dir / "ignored-only"
    ignored_only.mkdir()
    (ignored_only / "credentials.json").write_text('{"token": "secret"}\n')
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_dir)) == 0
    names = _member_names(cap["package_b64"])
    assert "empty" not in names
    assert not any(name == "ignored-only" or name.startswith("ignored-only/") for name in names)


class _FakeTTY(io.StringIO):
    """A captured stderr that claims to be interactive, so the progress bar renders."""

    def isatty(self) -> bool:
        return True


def test_push_directory_ships_binary_images_across_full_tree(monkeypatch, tmp_path):
    # a directory push ships binary assets from the full tree, not just dataset/, bytes intact
    env_dir = tmp_path / "my-env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    (env_dir / "dataset").mkdir()
    (env_dir / "dataset" / "red.png").write_bytes(b"png-bytes")
    (env_dir / "assets").mkdir()
    (env_dir / "assets" / "blue.png").write_bytes(b"other-png-bytes")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_dir, name="image-env")) == 0
    raw = base64.b64decode(cap["package_b64"])
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        payloads = {
            member.name: tar.extractfile(member).read()
            for member in tar.getmembers()
            if member.isfile()
        }
    assert payloads["dataset/red.png"] == b"png-bytes"
    assert payloads["assets/blue.png"] == b"other-png-bytes"


def test_push_off_tty_passes_no_progress_callback(monkeypatch, tmp_path):
    # Under pytest stderr is not a TTY, so the upload stays on the plain single-shot path.
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file, name="starter")) == 0
    assert cap["progress"] is None


def test_push_renders_and_forwards_upload_progress(monkeypatch, tmp_path):
    fake_err = _FakeTTY()
    monkeypatch.setattr(sys, "stderr", fake_err)

    seen: dict = {}

    class _C:
        def publish_env(self, *, name, package_b64, project_id, progress=None):
            # On a TTY the CLI hands us a real callback; drive it from 0 to 100%.
            assert progress is not None
            progress(0, 10)
            progress(10, 10)
            seen["progressed"] = True
            return {"id": "acme/starter"}

    monkeypatch.setattr("flash.client.client_from_config", lambda: _C())

    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    assert cli.cmd_env_push(_args(env_file, name="starter")) == 0
    assert seen.get("progressed")
    rendered = fake_err.getvalue()
    assert "packaging environment" in rendered
    assert "uploading starter" in rendered
    assert "100%" in rendered


def test_upload_progress_renders_bar_and_clears(monkeypatch):
    fake_err = _FakeTTY()
    monkeypatch.setattr(sys, "stderr", fake_err)

    bar = _UploadProgress("starter")
    assert bar.enabled
    assert bar.callback == bar.update  # a bound method; enabled -> not None
    bar.update(0, 100)
    bar.update(50, 100)
    bar.update(100, 100)
    out = fake_err.getvalue()
    assert "uploading starter" in out
    assert " 50%" in out
    assert "100%" in out
    # clear() must blank exactly the last line's width, else stale bar text shows on a real
    # terminal. Assert the precise wipe sequence (\r, last_len spaces, \r), not just a tail \r.
    last_len = bar._last_len
    bar.clear()
    assert fake_err.getvalue() == out + "\r" + " " * last_len + "\r"


def test_upload_progress_is_noop_off_tty(monkeypatch):
    fake_err = io.StringIO()  # StringIO.isatty() is False
    monkeypatch.setattr(sys, "stderr", fake_err)

    bar = _UploadProgress("starter")
    assert not bar.enabled
    assert bar.callback is None
    bar.status("packaging environment")
    bar.update(1, 2)
    bar.clear()
    assert fake_err.getvalue() == ""


def test_human_bytes_scales_units():
    assert _human_bytes(0) == "0 B"
    assert _human_bytes(512) == "512 B"
    assert _human_bytes(1536) == "1.5 KB"
    assert _human_bytes(5 * 1024 * 1024) == "5.0 MB"
    assert _human_bytes(3 * 1024 * 1024 * 1024) == "3.0 GB"


def test_push_directory_keeps_secret_named_code_but_drops_actual_secrets(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    # legit code whose names merely start with secret-ish prefixes must be kept
    (env_dir / "credentials_helper.py").write_text("VALUE = 1\n")
    (env_dir / "id_rsa_utils.py").write_text("VALUE = 2\n")
    # real secrets must be dropped
    (env_dir / "credentials.json").write_text('{"token": "secret"}\n')
    (env_dir / "id_rsa").write_text("PRIVATE KEY\n")
    (env_dir / "id_rsa.pub").write_text("PUBLIC KEY\n")
    (env_dir / "config.env").write_text("TOKEN=secret\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))
    assert cli.cmd_env_push(_args(env_dir)) == 0
    names = set(_members(cap["package_b64"]))
    assert "credentials_helper.py" in names
    assert "id_rsa_utils.py" in names
    assert "credentials.json" not in names
    assert "id_rsa" not in names
    assert "id_rsa.pub" not in names
    assert "config.env" not in names


def test_push_excludes_nested_secrets_directory(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    (env_dir / "secrets").mkdir()
    (env_dir / "secrets" / "service_account.json").write_text('{"token": "secret"}\n')
    (env_dir / "secrets" / "token.txt").write_text("secret\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))
    assert cli.cmd_env_push(_args(env_dir)) == 0
    names = set(_members(cap["package_b64"]))
    assert not any(name.startswith("secrets/") or name == "secrets" for name in names)


def test_push_excludes_nested_virtualenv_by_marker(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    # a virtualenv named `env` (no leading dot, not in the name ignore list) is detected by its marker
    venv = env_dir / "env"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (venv / "lib").mkdir()
    (venv / "lib" / "site.py").write_text("SECRET = True\n")
    # a legit data directory without the marker must still ship
    data = env_dir / "env_data"
    data.mkdir()
    (data / "rows.jsonl").write_text('{"x": 1}\n')
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))
    assert cli.cmd_env_push(_args(env_dir)) == 0
    names = set(_members(cap["package_b64"]))
    assert not any(name == "env" or name.startswith("env/") for name in names)
    assert "env_data/rows.jsonl" in names


def test_push_single_py_ships_sibling_readme_and_training_docs(monkeypatch, tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "README.md").write_text("# my real readme\n\nuser authored\n")
    (tmp_path / "TRAINING.md").write_text("# training guide\n")
    (tmp_path / "helper.py").write_text("VALUE = 1\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))
    assert cli.cmd_env_push(_args(env_file)) == 0
    files = _members(cap["package_b64"])
    # the user's real readme is shipped, not replaced by the stub
    assert "user authored" in files["README.md"]
    assert "TRAINING.md" in files
    # sibling helper modules are still not shipped for single-file pushes
    assert "helper.py" not in files


def test_push_rejects_when_member_count_exceeds_limit_including_dirs(monkeypatch, tmp_path, capsys):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    # 3 files across 3 directories -> members well above a tiny cap even though files alone are few
    for i in range(3):
        d = env_dir / f"pkg{i}"
        d.mkdir()
        (d / "mod.py").write_text("X = 1\n")
    monkeypatch.setattr("flash.cli.commands.env.ops.push._ENV_PUSH_MAX_FILES", 4)
    monkeypatch.setattr(
        "flash.client.client_from_config",
        lambda: (_ for _ in ()).throw(AssertionError("upload must not start")),
    )
    assert cli.cmd_env_push(_args(env_dir)) == 1
    error = capsys.readouterr().err
    assert "files and directories" in error
    assert "(limit 4)" in error


def test_push_counts_an_imported_dataset_package_once(monkeypatch, tmp_path):
    """A helper package reachable BOTH ways must not be charged twice against the limit.

    The single-file walk yields imported helper packages, then falls through to the `dataset/`
    walk. A helper package that IS `dataset/` is reached by both, and the second pass did not
    consult the first's `yielded` set -- so the limit check counted those files and bytes twice and
    rejected a tree that actually fits. The archive was always correct (the copy just overwrites),
    which is why only the limit saw it.
    """
    # a SINGLE-FILE push: only that path builds the import closure whose `yielded` set the dataset
    # walk has to honour. pushing the directory takes the full-tree branch and never reaches it.
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "evaluations.py").write_text("import dataset\n\n\ndef cases():\n    return []\n")
    pkg = tmp_path / "dataset"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("ROWS = []\n")
    (pkg / "rows.jsonl").write_text('{"a": 1}\n')

    # environment.py + evaluations.py + the two dataset files + the synthesized readme = 5 members,
    # plus the `dataset` directory = 6. counting the package twice charges 8 and trips this cap.
    monkeypatch.setattr("flash.cli.commands.env.ops.push._ENV_PUSH_MAX_FILES", 6)
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file)) == 0
    names = set(_members(cap["package_b64"]))
    assert "dataset/__init__.py" in names
    assert "dataset/rows.jsonl" in names


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root bypasses dir perms")
def test_push_fails_fast_on_unreadable_directory(monkeypatch, tmp_path, capsys):
    import os as _os

    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    locked = env_dir / "locked"
    locked.mkdir()
    (locked / "data.jsonl").write_text('{"x": 1}\n')
    _os.chmod(locked, 0o000)
    monkeypatch.setattr(
        "flash.client.client_from_config",
        lambda: (_ for _ in ()).throw(AssertionError("upload must not start")),
    )
    try:
        assert cli.cmd_env_push(_args(env_dir)) == 1
        assert "cannot publish" in capsys.readouterr().err
    finally:
        _os.chmod(locked, 0o755)


def test_push_drops_ssh_keys_and_credential_data_but_keeps_python_modules(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    # python modules whose names look secret-ish must be kept: code is never name-filtered
    (env_dir / "credentials.py").write_text("TOKEN = None\n")
    (env_dir / "id_ed25519_loader.py").write_text("VALUE = 1\n")
    # private keys of every common type, including nested and extensionless, must be dropped
    keys = env_dir / "keys"
    keys.mkdir()
    (keys / "id_ed25519").write_text("PRIVATE\n")
    (keys / "id_ecdsa").write_text("PRIVATE\n")
    (keys / "id_dsa").write_text("PRIVATE\n")
    (keys / "id_ed25519.pub").write_text("PUBLIC\n")
    # credential data files must still be dropped
    (env_dir / "credentials.json").write_text('{"token": "x"}\n')
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))
    assert cli.cmd_env_push(_args(env_dir)) == 0
    names = set(_members(cap["package_b64"]))
    assert "credentials.py" in names
    assert "id_ed25519_loader.py" in names
    assert "credentials.json" not in names
    assert "keys/id_ed25519" not in names
    assert "keys/id_ecdsa" not in names
    assert "keys/id_dsa" not in names
    assert "keys/id_ed25519.pub" not in names


def test_push_drops_underscore_secret_files_but_keeps_secretish_packages(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    # backup/underscore secret file variants must be dropped, with or without an extension
    (env_dir / "credentials_backup").write_text("SECRET\n")
    (env_dir / "credentials_prod.json").write_text('{"t": 1}\n')
    (env_dir / "id_rsa_backup").write_text("PRIVATE\n")
    # a legitimate package directory whose name merely starts with a secret-ish prefix must ship
    pkg = env_dir / "credentials_store"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from .load import load\n")
    (pkg / "load.py").write_text("def load():\n    return None\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))
    assert cli.cmd_env_push(_args(env_dir)) == 0
    names = set(_members(cap["package_b64"]))
    assert "credentials_backup" not in names
    assert "credentials_prod.json" not in names
    assert "id_rsa_backup" not in names
    assert "credentials_store/__init__.py" in names
    assert "credentials_store/load.py" in names


def test_push_single_py_ships_its_evaluations_sidecar(monkeypatch, tmp_path):
    # `env eval TARGET./environment.py` loads the sibling evaluations.py, so a single-file push that
    # dropped it published a package whose suite passed locally and was simply gone once uploaded --
    # while a directory push of the same files kept it.
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'held-out'\n"
        "    def cases(self): return [EvalCase(id='c', input='2+2', expected='4')]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )
    (tmp_path / "helper.py").write_text("VALUE = 1\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file)) == 0

    files = _members(cap["package_b64"])
    assert "held-out" in files["evaluations.py"]
    # the sidecar is a known filename, not a blanket "ship every sibling module" change.
    assert "helper.py" not in files


def test_push_directory_carries_an_evaluations_sidecar(monkeypatch, tmp_path):
    # the sidecar rides along with the canonical entrypoint rather than being mistaken for a rival
    # one. (a directory whose only module is `custom.py` is no longer a supported layout: see
    # test_push_dir_without_the_canonical_entrypoint_is_refused.)
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'held-out'\n"
        "    def cases(self): return []\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_dir)) == 0

    files = _members(cap["package_b64"])
    assert "load_environment" in files["environment.py"]
    assert "held-out" in files["evaluations.py"]


def test_push_directory_still_rejects_two_real_candidate_modules(monkeypatch, tmp_path):
    # the control for the scope of that exclusion: only the known sidecar filename is skipped,
    # so a genuinely ambiguous directory is still refused rather than silently picking one.
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "custom.py").write_text("def load_environment(**k):\n    return None\n")
    (env_dir / "other.py").write_text("def load_environment(**k):\n    return None\n")
    monkeypatch.setattr("flash.client.client_from_config", _fake_client({}))

    assert cli.cmd_env_push(_args(env_dir)) == 1


def test_push_non_utf8_entrypoint_still_ships_the_helpers_it_imports(monkeypatch, tmp_path):
    # the publish path decodes the entrypoint with `tokenize.detect_encoding`, so a file carrying a
    # non-utf-8 encoding cookie is publishable. the import walk read the same file as utf-8 and
    # swallowed the decode error as "no imports", so the sibling it imports never entered the
    # archive: `env push` exits 0 and prints an id, and the missing helper only surfaces when a
    # worker imports the module -- after a gpu has been rented.
    env_file = tmp_path / "environment.py"
    env_file.write_bytes(
        b"# -*- coding: latin-1 -*-\n"
        b"# caf\xe9\n"
        b"import helper\n"
        b"def load_environment(**k):\n"
        b"    return helper.VALUE\n"
    )
    (tmp_path / "helper.py").write_text("VALUE = 1\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file)) == 0

    assert "helper.py" in _member_names(cap["package_b64"])
