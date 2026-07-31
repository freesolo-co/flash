"""`flash env push` packages local Freesolo envs and uploads them through the server."""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import tarfile

import pytest

import flash.cli as cli
from flash.cli.envpush import _human_bytes, _UploadProgress


def _fake_client(capture: dict, *, slug: str = "acme/environment"):
    """A stand-in ApiClient that records the publish_env call and returns an env id."""

    class _C:
        def publish_env(self, *, name, package_b64, project_id, progress=None):
            capture.update(
                name=name, package_b64=package_b64, project_id=project_id, progress=progress
            )
            return {"id": slug}

    return lambda: _C()


def _members(package_b64: str) -> dict[str, str]:
    raw = base64.b64decode(package_b64)
    out: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile():
                out[m.name] = tar.extractfile(m).read().decode()
    return out


def _member_names(package_b64: str) -> list[str]:
    raw = base64.b64decode(package_b64)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        return [member.name for member in tar.getmembers()]


def _args(
    path, *, name: str = "my-env", project: str | None = "11111111-1111-4111-8111-111111111111"
):
    return argparse.Namespace(path=str(path), name=name, project=project)


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


def test_push_dir_infers_entrypoint_ignoring_the_evaluations_sidecar(monkeypatch, tmp_path):
    # a legacy package whose sole module is custom.py resolved fine before evaluations.py
    # existed. counting the sidecar as a candidate entrypoint makes adding one turn that
    # directory into "multiple top-level .py modules" and reject it before either file loads.
    env_dir = tmp_path / "legacy-env"
    env_dir.mkdir()
    (env_dir / "custom.py").write_text("def load_environment(**k):\n    return None\n")
    (env_dir / "evaluations.py").write_text("def load_evaluations(environment=None): return []\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_dir, name="legacy-env")) == 0
    files = _members(cap["package_b64"])
    # packaging canonicalizes the inferred entrypoint to environment.py, so the assertion is
    # that custom.py was chosen and published at all -- before the fix this raised instead.
    assert "return None" in files["environment.py"]
    assert "evaluations.py" in files


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
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap, slug="benchmark/math"))

    assert cli.cmd_env_push(_args(env_file, name="benchmark/Math Env")) == 0
    assert cap["name"] == "benchmark/math-env"


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


def test_push_single_py_ships_only_entrypoint_and_sibling_datasets(monkeypatch, tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "train.jsonl").write_text('{"x": 1}\n')
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "eval.jsonl").write_text('{"x": 2}\n')
    (tmp_path / "helper.py").write_text("VALUE = 1\n")
    (tmp_path / "config.yaml").write_text("mode: prod\n")
    (tmp_path / "unrelated").mkdir()
    (tmp_path / "unrelated" / "data.json").write_text("{}\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file)) == 0
    files = set(_members(cap["package_b64"]))
    assert files == {
        "README.md",
        "dataset/train.jsonl",
        "datasets/eval.jsonl",
        "environment.py",
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


def test_push_single_py_does_not_ship_sibling_helper_modules(monkeypatch, tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "helper.py").write_text("VALUE = 1\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file)) == 0
    assert "helper.py" not in _members(cap["package_b64"])


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
    assert "custom_env.py" not in files


def test_push_requires_explicit_name(tmp_path, capsys):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    assert cli.cmd_env_push(argparse.Namespace(path=str(env_file))) == 1
    assert "--name" in capsys.readouterr().err


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
    monkeypatch.setattr("flash.cli.envpush._ENV_PUSH_MAX_TOTAL_BYTES", 64)
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
    monkeypatch.setattr("flash.cli.envpush._ENV_PUSH_MAX_FILES", 4)
    monkeypatch.setattr(
        "flash.client.client_from_config",
        lambda: (_ for _ in ()).throw(AssertionError("upload must not start")),
    )
    assert cli.cmd_env_push(_args(env_dir)) == 1
    error = capsys.readouterr().err
    assert "files and directories" in error
    assert "(limit 4)" in error


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
    # `env eval TARGET ./environment.py` loads the sibling evaluations.py, so a single-file push
    # that dropped it published a package whose suite passed locally and was simply gone once
    # uploaded -- while a directory push of the same files kept it (codex[bot]).
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite, EvalCase\n"
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


def test_push_directory_infers_its_entrypoint_past_an_evaluations_sidecar(monkeypatch, tmp_path):
    # a directory whose only module is `custom.py` is a supported layout. counting evaluations.py
    # as a second top-level module made adding one reject the directory outright, so the very
    # sidecar that enables evaluation disabled the push and the eval alike (codex[bot]).
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "custom.py").write_text("def load_environment(**k):\n    return None\n")
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'held-out'\n"
        "    def cases(self): return []\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_dir)) == 0

    files = _members(cap["package_b64"])
    # custom.py was resolved as the entrypoint and published under the canonical name...
    assert "load_environment" in files["environment.py"]
    # ...and the sidecar rode along rather than being mistaken for a rival entrypoint.
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
