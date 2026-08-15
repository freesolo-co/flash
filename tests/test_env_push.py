"""`flash env push` packages local Freesolo envs and uploads them through the server."""

from __future__ import annotations

import argparse
import base64
import io
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

import flash.cli as cli
from flash.cli.commands.env.push import _human_bytes, _UploadProgress


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

    from flash.cli.commands.env import imports as envimports

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
    env_dir = tmp_path / "legacy-env"
    env_dir.mkdir()
    (env_dir / "custom.py").write_text(
        "SCORER = 'gold'\n\ndef load_environment(**k):\n    return None\n"
    )
    (env_dir / "evaluations.py").write_text(
        "from custom import SCORER\n\ndef load_evaluations(environment=None): return []\n"
    )
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_dir, name="legacy-env")) == 0
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
    from flash.cli.commands.env import push as envpush

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
    monkeypatch.setattr("flash.cli.commands.env.push._ENV_PUSH_MAX_TOTAL_BYTES", 64)
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
    monkeypatch.setattr("flash.cli.commands.env.push._ENV_PUSH_MAX_FILES", 4)
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
    monkeypatch.setattr("flash.cli.commands.env.push._ENV_PUSH_MAX_FILES", 6)
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
    # a directory whose only module is `custom.py` is a supported layout. counting evaluations.py as
    # a second top-level module made adding one reject the directory outright, so the very sidecar
    # that enables evaluation disabled the push and the eval alike.
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


# a body with a digit and a capital, so it reads as issued rather than hand-written. Assembled
# from parts so the literal is not itself a credential-shaped string in this repository.
_FAKE_KEY_BODY = "a1B2c3D4" * 6


def test_push_refuses_shell_env_file_holding_a_live_api_key(monkeypatch, tmp_path, capsys):
    """A sourceable shell env file is named like tooling, so only a content scan can catch it.

    `_ENV_PUSH_SECRET_PATTERNS` drops files NAMED like secret stores; `env.sh` matches none of them
    (`*.env` matches a file ending in `.env`). Publishing it committed a working FREESOLO_API_KEY
    into an org-shared hub repo, permanently in its git history.
    """
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    (env_dir / "env.sh").write_text(f'export FREESOLO_API_KEY="fslo_{_FAKE_KEY_BODY}"\n')
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_dir)) == 1
    err = capsys.readouterr().err
    assert "env.sh" in err
    assert "Freesolo API key" in err
    # the refusal must not echo the credential it just found: this text reaches terminals and logs.
    assert _FAKE_KEY_BODY not in err
    # refused before the upload, not merely filtered out of it
    assert not cap


@pytest.mark.parametrize(
    ("filename", "contents", "kind"),
    [
        ("setenv.sh", 'export HF_TOKEN="hf_{body}"', "Hugging Face"),
        ("secrets.sh", 'export GITHUB_TOKEN="ghp_{body}"', "GitHub"),
        ("bootstrap.sh", 'export PRIME_KEY="pit_{body}"', "Prime Intellect"),
        ("notes.md", "my key is sk-ant-{body} do not share", "Anthropic"),
        # a .pem is already dropped by NAME, so the uncovered case is a private key pasted into a
        # file whose name says nothing -- which is how a deploy key reaches a bootstrap script.
        # the body line is full width: openssl wraps at 64 characters, and requiring a body is what
        # keeps documentation that merely mentions the header publishable.
        (
            "bootstrap.py",
            'KEY = """-----BEGIN RSA PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSi"""\n',
            "private key",
        ),
    ],
)
def test_push_refuses_any_published_file_carrying_a_credential(
    monkeypatch, tmp_path, capsys, filename, contents, kind
):
    # the whole sourceable-secrets-file convention is uncovered by name, and a credential pasted
    # into a note or a stray .pem is not named like a secret store at all.
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    (env_dir / filename).write_text(contents.format(body=_FAKE_KEY_BODY))
    monkeypatch.setattr("flash.client.client_from_config", _fake_client({}))

    assert cli.cmd_env_push(_args(env_dir)) == 1
    assert kind in capsys.readouterr().err


def test_push_refuses_credential_hardcoded_in_python_source(monkeypatch, tmp_path, capsys):
    """Python source is exempt from the filename filter entirely, so only content can judge it.

    `_ignore_env_push_path` skips the pattern check for `.py`/`.pyi` so helper modules travel
    instead of breaking the worker with ModuleNotFoundError. A key pasted into a helper therefore
    had nothing standing between it and the hub.
    """
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text(
        "import helper\n\ndef load_environment(**k):\n    return None\n"
    )
    (env_dir / "helper.py").write_text(f'KEY = "fslo_{_FAKE_KEY_BODY}"\n')
    monkeypatch.setattr("flash.client.client_from_config", _fake_client({}))

    assert cli.cmd_env_push(_args(env_dir)) == 1
    assert "helper.py" in capsys.readouterr().err


def test_push_refuses_credential_in_the_entrypoint_itself(monkeypatch, tmp_path, capsys):
    # the entrypoint is published but never yielded by the sidecar walk, so scanning only the
    # sidecars would leave the likeliest place for a hardcoded key unchecked.
    env_file = tmp_path / "environment.py"
    env_file.write_text(
        f'KEY = "fslo_{_FAKE_KEY_BODY}"\n\ndef load_environment(**k):\n    return None\n'
    )
    monkeypatch.setattr("flash.client.client_from_config", _fake_client({}))

    assert cli.cmd_env_push(_args(env_file)) == 1
    assert "environment.py" in capsys.readouterr().err


def test_push_allows_ordinary_environments_and_datasets(monkeypatch, tmp_path):
    """The scan must not refuse a real environment; a false refusal blocks every publish.

    Each case below is drawn from something that actually occurs in published environments: an
    `AKIA...` access key id is a PUBLIC identifier that AWS puts in signed URLs in the clear, so it
    appears verbatim in web-scraped training data, and `hf_hub_download` is ordinary code.
    """
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text(
        "from huggingface_hub import hf_hub_download\n\ndef load_environment(**k):\n    return None\n"
    )
    dataset = env_dir / "dataset"
    dataset.mkdir()
    (dataset / "train.jsonl").write_text(
        '{"prompt": "describe this image", '
        '"image": "http://s3.amazonaws.com/x.jpg?AWSAccessKeyId=AKIAJ6IHWSU3BX3X7X3Q&Expires=1"}\n'
    )
    # a hand-written placeholder is snake_case English, carrying neither digit nor capital
    (env_dir / "conftest_helper.py").write_text('STUB = "fslo_retry_after_close"\n')
    # a lowercase-hex body is a content hash, not a key: this is an ordinary CDN asset URL, and
    # the bare `sk-` alternation matched it until it was narrowed to mixed-case bodies.
    (env_dir / "README.md").write_text(
        "assets live at https://cdn.example.com/a/sk-0123456789abcdef0123456789abcdef.js\n"
    )
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_dir)) == 0
    assert "dataset/train.jsonl" in _members(cap["package_b64"])


def test_credential_scan_reads_across_chunk_boundaries_and_into_binaries(tmp_path):
    import sqlite3

    from flash.env_secrets import (
        _MAX_BODY,
        _SCAN_CHUNK_BYTES,
        _SCAN_OVERLAP_BYTES,
        credential_in_file,
    )

    # the overlap must exceed the longest match any pattern can produce, which is what makes the
    # boundary case a guarantee rather than a likelihood. bodies are bounded for exactly this.
    assert _MAX_BODY + len("github_pat_") < _SCAN_OVERLAP_BYTES

    # a key split across the chunk boundary is the case a naive per-chunk scan misses. the
    # longest body the patterns admit is the worst case, so it is the one worth pinning.
    straddle = tmp_path / "straddle.txt"
    longest = ("a1B2c3D4" * 40)[:_MAX_BODY]
    straddle.write_text("x" * (_SCAN_CHUNK_BYTES - 100) + f"fslo_{longest}" + "y" * 20)
    assert credential_in_file(straddle) == "a Freesolo API key"

    # binary members are scanned too. skipping them was a hole, not a saving: a key in a sqlite
    # state file is as published as one in a shell script, and sqlite writes NUL bytes into its
    # first 16 bytes, so a "skip on an early NUL" rule hands that file through untouched.
    db = tmp_path / "state.sqlite"
    con = sqlite3.connect(db)
    con.execute("create table t (v text)")
    con.execute("insert into t values (?)", (f"fslo_{_FAKE_KEY_BODY}",))
    con.commit()
    con.close()
    assert b"\x00" in db.read_bytes()[:16]
    assert credential_in_file(db) == "a Freesolo API key"

    # an undecodable byte is not a credential character, so a mixed-encoding file stays scannable
    mixed = tmp_path / "mixed.txt"
    mixed.write_bytes(b"\xff\xfe caf\xe9 " + f"fslo_{_FAKE_KEY_BODY}".encode())
    assert credential_in_file(mixed) == "a Freesolo API key"


def test_push_scans_generated_members_and_resists_post_scan_mutation(monkeypatch, tmp_path):
    """The staged package is what gets scanned, so generated members are covered too.

    Scanning the SOURCE tree left three holes, all of which published a credential with exit 0:
    the synthesized README embeds `--name` verbatim, the generated entrypoint alias embeds the
    entrypoint filename, and between reading a source file and copying it any local process could
    rewrite it -- the archive then carried bytes nothing had ever scanned.
    """
    from flash.cli.commands.env import push as envpush

    # 1. a credential passed as the env NAME lands in the synthesized README
    env_dir = tmp_path / "named"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    monkeypatch.setattr("flash.client.client_from_config", _fake_client({}))
    assert cli.cmd_env_push(_args(env_dir, name=f"sk-or-v1-{_FAKE_KEY_BODY}")) == 1

    # 2. a file rewritten AFTER its scan must not reach the archive unscanned
    mutated = tmp_path / "mutated"
    mutated.mkdir()
    (mutated / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    (mutated / "env.sh").write_text("export SAFE=1\n")
    real_copy = envpush._copy_env_sidecars

    def _rewrite_then_copy(env_root, dest, *, entrypoint, include_full_tree):
        (env_root / "env.sh").write_text(f'export KEY="fslo_{_FAKE_KEY_BODY}"\n')
        return real_copy(env_root, dest, entrypoint=entrypoint, include_full_tree=include_full_tree)

    monkeypatch.setattr(envpush, "_copy_env_sidecars", _rewrite_then_copy)
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))
    assert cli.cmd_env_push(_args(mutated)) == 1
    assert not cap, "a post-scan mutation reached the upload"


def test_push_refuses_every_issued_openai_key_family(tmp_path):
    from flash.env_secrets import _credential_kind

    # sk-svcacct- and sk-admin- carry project- and org-wide authority. Neither is reachable
    # through the bare `sk-` branch: the subtype's own hyphen ends its alphanumeric run early.
    for prefix in ("sk-proj-", "sk-svcacct-", "sk-admin-"):
        body = "Ab3xK9zQ_mN2pR7t-VwXyZ0123456789abcd"
        assert _credential_kind(f"{prefix}{body}".encode()) == "an OpenAI API key", prefix

    # a legacy key carries its `T3BlbkFJ` watermark around body index 20. Requiring 31 more
    # characters *after* the first capital put that squarely in the miss zone.
    legacy = "sk-" + "a" * 20 + "T3BlbkFJ" + "b" * 20
    assert _credential_kind(legacy.encode()) == "an OpenAI API key"

    # ...while a lowercase-hex body of the same length stays an ordinary content hash.
    assert _credential_kind(b"https://cdn.example/a/sk-0123456789abcdef0123456789abcdef.js") is None


def test_push_refuses_a_credential_packed_inside_an_archive(monkeypatch, tmp_path, capsys):
    """A compressed member does not contain its credential literally, so it must be expanded.

    Scanning the container's own bytes cannot see a deflated `env.sh` -- the key is not in the file
    in any form a regex can match. Archives are ordinary in an environment (a bundled dataset
    shard), and `flash env push` publishes them, so the packed key shipped with exit 0.
    """
    import bz2
    import gzip
    import lzma
    import zipfile

    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    secret = f'export FREESOLO_API_KEY="fslo_{_FAKE_KEY_BODY}"\n'.encode()

    # detected by MAGIC, not extension: a `.bin` is expanded and a mislabelled `.gz` is not missed
    with zipfile.ZipFile(env_dir / "bundle.bin", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("nested/env.sh", secret)
    for name, packed in (
        ("shard.jsonl.gz", gzip.compress(secret)),
        ("shard.jsonl.bz2", bz2.compress(secret)),
        ("shard.jsonl.xz", lzma.compress(secret)),
    ):
        (env_dir / name).write_bytes(packed)
        assert secret not in (env_dir / name).read_bytes(), name

    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))
    assert cli.cmd_env_push(_args(env_dir)) == 1
    assert not cap, "a packed credential reached the upload"
    assert "bundle.bin" in capsys.readouterr().err


def test_credential_scan_survives_a_container_it_cannot_open(tmp_path):
    """An archive that will not open falls back to its literal bytes rather than erroring.

    A truncated or unsupported container is ordinary in a dataset directory, and refusing to
    publish one -- or crashing on it -- would be a worse bug than the hole being closed.
    """
    from flash.env_secrets import credential_in_file

    truncated = tmp_path / "broken.zip"
    truncated.write_bytes(b"PK\x03\x04" + b"\x00" * 8)
    assert credential_in_file(truncated) is None

    # the literal scan of the container's own bytes still applies when expansion yields nothing
    stored = tmp_path / "stored.gz"
    stored.write_bytes(b"\x1f\x8b garbage " + f"fslo_{_FAKE_KEY_BODY}".encode())
    assert credential_in_file(stored) == "a Freesolo API key"


def test_credential_scan_reads_wide_encodings(tmp_path):
    """UTF-16/32 text interleaves NULs, so a key in it matches none of the byte patterns.

    A PowerShell `env.ps1` is UTF-16 by default on Windows, which is exactly the sourceable secrets
    file this whole check exists for -- in the one encoding that walked straight past it.
    """
    from flash.env_secrets import credential_in_file

    for encoding in ("utf-16", "utf-16-be", "utf-32", "utf-32-be"):
        wide = tmp_path / f"env-{encoding}.ps1"
        wide.write_bytes(f'$env:FREESOLO_API_KEY = "fslo_{_FAKE_KEY_BODY}"\n'.encode(encoding))
        assert credential_in_file(wide) == "a Freesolo API key", encoding

    # narrowing must not invent a credential out of unrelated NUL-padded bytes
    padded = tmp_path / "padded.bin"
    padded.write_bytes(b"\x00".join(b"ordinary binary content" for _ in range(50)))
    assert credential_in_file(padded) is None


def test_narrowing_does_not_invent_a_credential_out_of_machine_code(tmp_path):
    """Every second byte of a compiled binary is not text, and must not be read as though it were.

    Narrowing the whole file unconditionally was the bug: an ELF holds no credential, but taking
    every second byte of x86 instruction bytes spells plausible tokens often enough that 5 of 500
    ordinary system binaries were refused -- `/usr/bin/bash` narrowed to a string matching the
    Freesolo key pattern. Refusing a legitimate publish is the failure here, so the fixture is the
    real thing rather than a hand-made one: a binary on this machine, chosen for having no
    credential in its literal bytes.

    Guarded by the wide encodings above, which must keep working; a gate that fixed this by simply
    not narrowing would pass this test and lose the `env.ps1` case the narrowing exists for.
    """
    from flash.env_secrets import _decoded_kind, credential_in_file

    # Lifted verbatim from `/usr/bin/bash`, where this was first seen: three symbol names and their
    # NUL terminators. Taking every second byte spells `fslo_eietossvdrdrcsP3`, which matches the
    # Freesolo pattern and passes the entropy test. Pinned as bytes rather than read from the
    # machine's own `/usr/bin`, which differs between here and CI and would make the test prove
    # something different in each place -- or nothing, if the binaries it happened to pick were
    # clean.
    symbols = b"fos\x00loop_redirections\x00saved-redirects\x00PS3\x00"
    assert _decoded_kind(symbols[0::2]) == "a Freesolo API key", "fixture no longer narrows"

    # padded to an even offset so the fixture keeps the parity that narrows
    binary = tmp_path / "libexample.so"
    binary.write_bytes(b"\x7fELF\x02\x01\x01\x00" + symbols * 4)
    assert credential_in_file(binary) is None

    # The NULs here are string terminators, scattered through a symbol table rather than the
    # regular padding column of wide text -- which is exactly what separates this from the
    # `env.ps1` case above and what the gate keys on.
    wide = tmp_path / "env.ps1"
    wide.write_bytes(f'$env:FREESOLO_API_KEY = "fslo_{_FAKE_KEY_BODY}"\n'.encode("utf-16"))
    assert credential_in_file(wide) == "a Freesolo API key"


def test_push_refuses_a_credential_used_as_a_filename(monkeypatch, tmp_path, capsys):
    """A file NAMED after a key publishes it in the repository's file tree, contents irrelevant.

    Scanning only contents let an empty `fslo_<key>.json` through with exit 0, and the published
    tree then shows that name forever. The refusal masks the body so the message cannot re-leak the
    key it is refusing.
    """
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    (env_dir / f"cache-fslo_{_FAKE_KEY_BODY}.json").write_text("{}\n")

    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))
    assert cli.cmd_env_push(_args(env_dir)) == 1
    assert not cap, "a credential-named file reached the upload"

    err = capsys.readouterr().err
    assert "cache-fslo_***.json" in err
    assert _FAKE_KEY_BODY not in err, "the refusal echoed the credential it was refusing"


def test_push_refuses_a_credential_used_as_a_directory_name(monkeypatch, tmp_path):
    from flash.env_secrets import credential_in_name

    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    keyed = env_dir / f"runs-fslo_{_FAKE_KEY_BODY}"
    keyed.mkdir()
    (keyed / "notes.md").write_text("nothing secret here\n")

    monkeypatch.setattr("flash.client.client_from_config", _fake_client({}))
    assert cli.cmd_env_push(_args(env_dir)) == 1
    assert credential_in_name("a/b/hf_AbCdEf0123456789012345/c.txt") == "a Hugging Face token"
    assert credential_in_name("src/hf_hub_download_helper.py") is None


def test_a_container_encoded_into_a_member_name_is_expanded():
    """A name gets the same container inspection a file's contents get.

    The name scan passed no deadline, and that is exactly what switches `_decoded_container` off --
    so an encoded container in a name was only ever matched in its still-compressed form. A
    66-character filename holding `base64(gzip(key))` published clean, while decoding and inflating
    the published path recovers the whole key: the name is in the repo tree forever, so this leaks
    through the listing even with an empty file under it.
    """
    import base64
    import gzip

    from flash.env_secrets import credential_in_name

    encoded = (
        base64.urlsafe_b64encode(gzip.compress(f"fslo_{_FAKE_KEY_BODY}".encode()))
        .decode()
        .rstrip("=")
    )
    # the name really does carry the key, recoverable by ordinary tools
    assert f"fslo_{_FAKE_KEY_BODY}".encode() in gzip.decompress(
        base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    )
    assert credential_in_name(encoded) == "a Freesolo API key"

    # ordinary names must not become refusals or false matches -- base64-SHAPED names are common
    for ordinary in (
        "model-00001-of-00002.safetensors",
        "data/shard_0001.tar.gz",
        "aGVsbG8gd29ybGQK",
        "checkpoint-1000/optimizer.pt",
    ):
        assert credential_in_name(ordinary) is None, ordinary


def test_a_pem_header_without_a_key_body_is_prose_not_a_credential(tmp_path):
    """Documentation that mentions a PEM header must still publish.

    Refusing on the header alone blocked a legitimate publish over writing about credentials, which
    is the kind of false refusal that gets a check disabled.
    """
    from flash.env_secrets import _credential_kind

    prose = b"If you see -----BEGIN RSA PRIVATE KEY----- in a log, redact it before sharing."
    assert _credential_kind(prose) is None

    # a real block always carries its body, in either the bare or the encrypted-header form
    body = b"-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAvR0Y2fJ8kLmNpQrStUvWxYz0123456789ab\n"
    assert _credential_kind(body) == "a private key block"
    encrypted = b"-----BEGIN RSA PRIVATE KEY-----\nProc-Type: 4,ENCRYPTED\nDEK-Info: AES-128-CBC\n"
    assert _credential_kind(encrypted) == "a private key block"


def test_slack_app_level_tokens_are_matched(tmp_path):
    """`xapp-` is a separate prefix, not another letter in the `xox?` set."""
    from flash.env_secrets import _credential_kind

    for prefix in ("xoxb-", "xoxp-", "xoxa-", "xoxr-", "xoxs-", "xapp-"):
        token = f"{prefix}1-A012BC3DEF-1234567890123-abcdefABCDEF0123456789"
        assert _credential_kind(token.encode()) == "a Slack token", prefix


def _flip_zip_flags(
    source: bytes, *, offset_local: int, offset_central: int, value: bytes
) -> bytes:
    """`source` with a field overwritten in both the local and central directory headers."""
    raw = bytearray(source)
    for index in range(len(raw) - 4):
        if raw[index : index + 4] == b"PK\x03\x04":
            start = index + offset_local
            raw[start : start + len(value)] = value
        elif raw[index : index + 4] == b"PK\x01\x02":
            start = index + offset_central
            raw[start : start + len(value)] = value
    return bytes(raw)


def test_credential_scan_survives_archives_the_stdlib_refuses_to_read(tmp_path):
    """A container that cannot be expanded must fall back, not abort the publish.

    The standard library reports these through no common base class: an encrypted member raises
    RuntimeError, an unimplemented compression method raises NotImplementedError, and a corrupt
    deflate stream raises zlib.error. None is an OSError, so each one crashed `flash env push` with
    a traceback on an ordinary corrupt dataset shard.
    """
    import gzip
    import struct
    import zipfile

    from flash.env_secrets import _Unscannable, credential_in_file

    plain = tmp_path / "plain.zip"
    with zipfile.ZipFile(plain, "w") as archive:
        archive.writestr("member.txt", "ordinary content " * 20)
    source = plain.read_bytes()

    # bit 0 of the general-purpose flag marks a member encrypted -> RuntimeError.
    # Refused rather than approved: its bytes cannot be read, and unreadable is not clean. What
    # this test pins is that the stdlib exception does not escape as a traceback.
    encrypted = tmp_path / "encrypted.zip"
    encrypted.write_bytes(_flip_zip_flags(source, offset_local=6, offset_central=8, value=b"\x01"))
    with pytest.raises(_Unscannable, match="encrypted archive member"):
        credential_in_file(encrypted)

    # method 99 is AES, which zipfile does not implement -> NotImplementedError. Refused for the
    # same reason as the encrypted member: the payload cannot be decoded, so it cannot be cleared.
    # This asserted None until a credential in a Deflate64 member was shown to publish intact.
    unsupported = tmp_path / "unsupported.zip"
    unsupported.write_bytes(
        _flip_zip_flags(source, offset_local=8, offset_central=10, value=struct.pack("<H", 99))
    )
    with pytest.raises(_Unscannable, match="compressed in a way this check cannot read"):
        credential_in_file(unsupported)

    # a corrupt deflate stream under a valid gzip header -> zlib.error
    corrupt = tmp_path / "corrupt.gz"
    good = gzip.compress(b"x" * 5000)
    corrupt.write_bytes(good[:12] + bytes(b ^ 0xFF for b in good[12:60]) + good[60:])
    assert credential_in_file(corrupt) is None


def test_a_credential_cannot_hide_behind_a_wall_of_padding(tmp_path):
    """The whole expanded stream is scanned, so a key placed late is still found.

    Capping the scan at a byte count looked safe because the package limit is 256 MB -- but that
    limit bounds COMPRESSED size. Padding compresses about 1000:1, so a file well under the limit
    expands past any such cap, and a key after the cutoff published with exit 0.
    """
    import gzip

    from flash.env_secrets import credential_in_file

    padded = tmp_path / "shard.jsonl.gz"
    with gzip.open(padded, "wb", compresslevel=9) as handle:
        block = b"\0" * (1 << 20)
        for _ in range(300):
            handle.write(block)
        handle.write(f"fslo_{_FAKE_KEY_BODY}".encode())

    # the published artifact is small; only its expansion is large
    assert padded.stat().st_size < 8 << 20
    assert credential_in_file(padded) == "a Freesolo API key"


def test_push_refuses_an_archive_too_expensive_to_scan(monkeypatch, tmp_path, capsys):
    """An archive that cannot be finished is refused, not waved through.

    Unverifiable is not the same as clean: returning None on a timeout would hand the publisher the
    exact bypass the budget exists to prevent.
    """
    import gzip

    from flash import env_secrets as secrets
    from flash.env_secrets import _Unscannable, credential_in_file

    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    slow = env_dir / "slow.gz"
    slow.write_bytes(gzip.compress(b"\0" * (4 << 20)))

    # a budget already spent: the first chunk of expansion is over the deadline
    monkeypatch.setattr(secrets, "_MAX_DECOMPRESS_SECONDS", -1.0)
    with pytest.raises(_Unscannable):
        credential_in_file(slow)

    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))
    assert cli.cmd_env_push(_args(env_dir)) == 1
    assert not cap, "an unscannable archive reached the upload"
    assert "too long to decompress" in capsys.readouterr().err


def test_an_unreadable_zip_member_does_not_hide_the_members_behind_it(tmp_path):
    """Each member is guarded separately, so one opaque entry cannot mask the rest.

    Guarding the whole loop meant a single encrypted member at the top of an archive abandoned the
    scan of everything after it, and a real key further down published with exit 0 -- the same
    silent pass the expansion budget refuses, arriving through error handling instead.
    """
    import zipfile

    from flash.env_secrets import credential_in_file

    def _build(name: str, *, encrypt_first: bool):
        path = tmp_path / name
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("aaa_first.txt", "harmless content " * 20)
            archive.writestr("zzz_second.sh", f'export KEY="fslo_{_FAKE_KEY_BODY}"\n')
        if not encrypt_first:
            return path
        # mark ONLY the first member encrypted, in both its headers
        raw = bytearray(path.read_bytes())
        for offset, signature in ((6, b"PK\x03\x04"), (8, b"PK\x01\x02")):
            index = raw.find(signature)
            raw[index + offset] |= 0x01
        path.write_bytes(bytes(raw))
        return path

    # the control: without the bad member the key is found, so the archive itself is scannable
    assert credential_in_file(_build("clean.zip", encrypt_first=False)) == "a Freesolo API key"
    assert credential_in_file(_build("guarded.zip", encrypt_first=True)) == "a Freesolo API key"


def test_every_member_unreadable_is_not_an_error(tmp_path):
    """An archive nothing can be read from is refused, not approved.

    An encrypted member is guarded so it cannot mask the members behind it, but when NO member
    could be read the archive has been inspected in name only. Returning None there approved a
    package whose only copy of a credential sat inside it.
    """
    import zipfile

    from flash.env_secrets import _Unscannable, credential_in_file

    path = tmp_path / "opaque.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("only.txt", "x" * 200)
    raw = bytearray(path.read_bytes())
    for index in range(len(raw) - 4):
        if raw[index : index + 4] == b"PK\x03\x04":
            raw[index + 6] |= 0x01
        elif raw[index : index + 4] == b"PK\x01\x02":
            raw[index + 8] |= 0x01
    path.write_bytes(bytes(raw))

    with pytest.raises(_Unscannable, match="encrypted archive member"):
        credential_in_file(path)


def test_a_gzip_with_trailing_bytes_is_not_published_as_unreadable(tmp_path):
    """A complete member plus one stray byte hid the key it had already inflated.

    Python's reader finishes the member, looks for another where the trailer ends, finds the stray
    byte and raises `BadGzipFile` -- WITHOUT yielding any of the plaintext. That is an `OSError`,
    so the handler tuple swallowed it, the dispatch loop concluded "not this format", every
    remaining handler declined, and the file published on its literal bytes alone.

    The key is genuinely recoverable, not merely present: `gzip -dc` on this exact file prints it
    to stdout and exits 1. Any non-null trailing byte reaches it -- a newline, a stray `PK\\x03\\x04`
    -- so an accidental concatenation leaks as readily as a crafted file. Trailing NULs are the one
    shape that already worked, being skipped as padding rather than read as a member.

    The two shapes that must NOT change are pinned alongside it: ordinary text still publishes
    (`gzip.open` is the fallback opener, so it raises this same exception on its first header read
    and only the magic separates the cases), and damage INSIDE a member still falls through to the
    literal scan rather than failing a publish over a corrupt dataset shard.
    """
    import gzip

    from flash.env_secrets import _Unscannable, credential_in_file

    body = f'export KEY="fslo_{_FAKE_KEY_BODY}"\n'.encode()
    member = gzip.compress(body)

    for name, trailer in (("byte.gz", b"x"), ("nl.gz", b"\n"), ("zip.gz", b"PK\x03\x04")):
        path = tmp_path / name
        path.write_bytes(member + trailer)
        with pytest.raises(_Unscannable, match="cannot finish reading"):
            credential_in_file(path)

    clean = tmp_path / "clean.gz"
    clean.write_bytes(member)
    assert credential_in_file(clean) == "a Freesolo API key"

    padded = tmp_path / "padded.gz"
    padded.write_bytes(member + b"\x00" * 64)
    assert credential_in_file(padded) == "a Freesolo API key"

    # never a gzip: the same exception, and it must still fall through to the other formats
    text = tmp_path / "plain.txt"
    text.write_bytes(b"# ordinary configuration\nDEBUG=1\n")
    assert credential_in_file(text) is None

    # damage inside the member raises zlib.error, which stays a fall-through, not a refusal
    broken = bytearray(gzip.compress(body * 40))
    broken[len(broken) // 2] ^= 0xFF
    shard = tmp_path / "shard.gz"
    shard.write_bytes(bytes(broken))
    assert credential_in_file(shard) is None


def test_a_completed_archive_scan_discards_an_earlier_heuristics_refusal(tmp_path):
    """A handler that enumerated the whole file has answered the question a guess deferred.

    A tar whose first member is named `x data.txt` starts with bytes the zlib FDICT header rule
    accepts, so that handler defers a dictionary-stream refusal on content that is not a zlib
    stream at all. The tar walk then lists and reads every member successfully -- and the deferred
    refusal was raised anyway, because a handler returning None looks identical whether it
    enumerated the format or declined bytes that were never its own.
    """
    import io
    import tarfile
    import zipfile

    from flash.env_secrets import _Unscannable, credential_in_file

    def _tar(name: str, body: bytes) -> bytes:
        buffered = io.BytesIO()
        with tarfile.open(fileobj=buffered, mode="w") as archive:
            info = tarfile.TarInfo(name)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
        return buffered.getvalue()

    harmless = tmp_path / "dataset.tar"
    harmless.write_bytes(_tar("x data.txt", b"harmless\n"))
    assert credential_in_file(harmless) is None

    # the same archive really holding a key is still reported, so this did not silence the scan
    keyed = tmp_path / "keyed.tar"
    keyed.write_bytes(_tar("x data.txt", b'KEY="fslo_%s"\n' % _FAKE_KEY_BODY.encode()))
    assert credential_in_file(keyed) == "a Freesolo API key"

    # and a refusal about content the completed walk ACTUALLY reached must survive: an encrypted
    # member is unverifiable however successfully the archive around it enumerated
    opaque = tmp_path / "opaque.zip"
    with zipfile.ZipFile(opaque, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("only.txt", "x" * 200)
    raw = bytearray(opaque.read_bytes())
    for index in range(len(raw) - 4):
        if raw[index : index + 4] == b"PK\x03\x04":
            raw[index + 6] |= 0x01
        elif raw[index : index + 4] == b"PK\x01\x02":
            raw[index + 8] |= 0x01
    opaque.write_bytes(bytes(raw))
    with pytest.raises(_Unscannable, match="encrypted archive member"):
        credential_in_file(opaque)

    wrapped = tmp_path / "wrapped.tar"
    wrapped.write_bytes(_tar("inner.zip", bytes(raw)))
    with pytest.raises(_Unscannable, match="encrypted archive member"):
        credential_in_file(wrapped)


def test_a_plain_footer_after_a_zlib_record_is_not_a_refusal(tmp_path):
    """A remainder that does not open like a record is data, not an unreadable stream.

    Refusing on ANY trailing bytes rejected `zlib.compress(b"harmless") + b"footer"` -- one record
    that decoded perfectly, followed by bytes plainly not another one -- and with it the framed and
    cache formats that write exactly that shape. The refusal still has to fire for a remainder that
    really does begin a record and cannot be read, which is the case it exists for.
    """
    import zlib

    from flash.env_secrets import _Unscannable, credential_in_file

    key = f"fslo_{_FAKE_KEY_BODY}".encode()

    footed = tmp_path / "cache.bin"
    footed.write_bytes(zlib.compress(b"harmless") + b"footer")
    assert credential_in_file(footed) is None

    # the trailing bytes are still SCANNED, so a credential sitting in the footer is found
    keyed_footer = tmp_path / "keyed.bin"
    keyed_footer.write_bytes(zlib.compress(b"harmless") + b"footer " + key)
    assert credential_in_file(keyed_footer) == "a Freesolo API key"

    # a real second record still gets inflated, which is what the loop is for
    chained = tmp_path / "chained.z"
    chained.write_bytes(zlib.compress(b"hi") + zlib.compress(key))
    assert credential_in_file(chained) == "a Freesolo API key"

    # and a remainder that DOES open like a record but cannot be read is still refused: unreadable
    # is not clean, and narrowing the refusal must not reopen that hole
    damaged = bytearray(zlib.compress(key * 40))
    damaged[len(damaged) // 2] ^= 0xFF
    broken = tmp_path / "broken.z"
    broken.write_bytes(zlib.compress(b"hi") + bytes(damaged))
    with pytest.raises(_Unscannable, match="trailing compressed data"):
        credential_in_file(broken)


def test_a_corrupt_xz_does_not_crash_the_publish(tmp_path):
    """`lzma.LZMAError` inherits straight from Exception, so it needs naming explicitly.

    Shallow corruption is a trap here: truncating near the end of the stream raises EOFError, which
    was already caught, and the bug looks absent. Only damage deep enough that the decompressor
    rejects the data -- rather than running out of it -- produces LZMAError.
    """
    import lzma

    from flash.env_secrets import _UNREADABLE_ARCHIVE, credential_in_file

    assert issubclass(lzma.LZMAError, _UNREADABLE_ARCHIVE)

    corrupt = tmp_path / "shard.xz"
    good = lzma.compress(b"y" * 200_000)
    middle = len(good) // 2
    corrupt.write_bytes(
        good[:middle] + bytes(b ^ 0xFF for b in good[middle : middle + 200]) + good[middle + 200 :]
    )
    assert credential_in_file(corrupt) is None

    # a valid xz member is still expanded and scanned
    valid = tmp_path / "valid.xz"
    valid.write_bytes(lzma.compress(f'export KEY="fslo_{_FAKE_KEY_BODY}"\n'.encode()))
    assert credential_in_file(valid) == "a Freesolo API key"


def test_the_expansion_budget_is_not_swallowed_by_the_per_member_handler(monkeypatch, tmp_path):
    """A timeout must still refuse, not be mistaken for an unreadable member and skipped."""
    import zipfile

    from flash import env_secrets as secrets
    from flash.env_secrets import (
        _UNREADABLE_ARCHIVE,
        _Unscannable,
        credential_in_file,
    )

    assert not issubclass(_Unscannable, _UNREADABLE_ARCHIVE)

    path = tmp_path / "slow.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("big.txt", "\0" * (4 << 20))

    monkeypatch.setattr(secrets, "_MAX_DECOMPRESS_SECONDS", -1.0)
    with pytest.raises(_Unscannable):
        credential_in_file(path)


def test_a_credential_nested_two_containers_deep_is_still_found(tmp_path):
    """Expansion recurses: stopping at one level treated an inner container as final content.

    A zip holding a gzipped shard is an ordinary way to ship a dataset, and the inner member's
    bytes contain the credential nowhere a regex can see, so a key one layer further in published
    with exit 0.
    """
    import gzip
    import io
    import zipfile

    from flash.env_secrets import credential_in_file

    secret = f'export KEY="fslo_{_FAKE_KEY_BODY}"\n'.encode()

    two_deep = tmp_path / "dataset.zip"
    with zipfile.ZipFile(two_deep, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("shard.jsonl.gz", gzip.compress(secret))
    assert credential_in_file(two_deep) == "a Freesolo API key"

    middle = io.BytesIO()
    with zipfile.ZipFile(middle, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("inner.gz", gzip.compress(secret))
    three_deep = tmp_path / "bundle.zip"
    with zipfile.ZipFile(three_deep, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("middle.zip", middle.getvalue())
    assert credential_in_file(three_deep) == "a Freesolo API key"


def test_a_zip_behind_an_executable_stub_is_still_expanded(tmp_path):
    """A self-extracting archive leads with `MZ`, not `PK`, so leading magic cannot find it.

    `zipfile.is_zipfile` scans for the end-of-central-directory record and recognises it correctly;
    testing the first six bytes did not, so the container was never expanded.
    """
    import io
    import zipfile

    from flash.env_secrets import credential_in_file

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("env.sh", f'export KEY="fslo_{_FAKE_KEY_BODY}"\n')

    sfx = tmp_path / "installer.exe"
    sfx.write_bytes(b"MZ\x90\x00" + b"\x00" * 2000 + payload.getvalue())

    # the premise: it does not look like a zip, but it is one
    assert not sfx.read_bytes().startswith(b"PK")
    assert zipfile.is_zipfile(sfx)
    assert credential_in_file(sfx) == "a Freesolo API key"


def test_a_base64_encoded_credential_is_decoded_and_found(tmp_path):
    """A Kubernetes Secret stores every value base64-encoded, sharing no substring with the key.

    Measured before adopting this: 630,011 base64-shaped runs across 8,769 real hub files decode to
    zero credential matches, so decoding candidates costs no legitimate publish.
    """
    import base64

    from flash.env_secrets import _credential_kind, credential_in_file

    encoded = base64.b64encode(f"fslo_{_FAKE_KEY_BODY}".encode()).decode()
    manifest = tmp_path / "secret.yaml"
    manifest.write_text(f"apiVersion: v1\nkind: Secret\ndata:\n  api-key: {encoded}\n")
    assert credential_in_file(manifest) == "a Freesolo API key"

    # an encoded whole file, which is how a sourceable env file travels in a manifest
    whole = base64.b64encode(f'export FREESOLO_API_KEY="fslo_{_FAKE_KEY_BODY}"\n'.encode())
    assert _credential_kind(b"data:\n  env.sh: " + whole) == "a Freesolo API key"

    # ordinary prose and short tokens must not be decoded into false positives
    assert _credential_kind(b"the quick brown fox jumps over the lazy dog repeatedly") is None


@pytest.mark.parametrize(
    "placeholder",
    [
        "fslo_YOUR_API_KEY_HERE",
        "fslo_REPLACE_ME_WITH_YOUR_KEY",
        "fslo_CHANGEME_BEFORE_RUNNING",
        "fslo_XXXXXXXXXXXXXXXX",
        "fslo_your_api_key_here",
        "fslo_retry_after_close",
    ],
)
def test_scaffolding_placeholders_do_not_refuse_the_publish(placeholder):
    """A false refusal tells the author to rotate a key that never existed.

    Testing for "a digit or a capital" caught the lowercase convention and missed the uppercase and
    repeated-character ones, so a scaffolded environment could not be published at all. That is the
    failure mode that gets a check switched off.
    """
    from flash.env_secrets import _credential_kind

    assert _credential_kind(placeholder.encode()) is None


@pytest.mark.parametrize(
    ("credential", "kind"),
    [
        ("fslo_aB3xK9zQmN2pR7tVwXyZ0123", "a Freesolo API key"),
        ("hf_AbCdEf0123456789012345", "a Hugging Face token"),
        ("sk-ant-Ab3xK9zQ_mN2pR7t-VwXyZ0123456789", "an Anthropic API key"),
        ("ghp_AbCdEf0123456789012345", "a GitHub token"),
    ],
)
def test_issued_keys_are_still_caught_after_narrowing_the_entropy_test(credential, kind):
    """The placeholder allowance must not open a hole: issued bodies are mixed-case with digits."""
    from flash.env_secrets import _credential_kind

    assert _credential_kind(credential.encode()) == kind


def test_a_scan_limit_refuses_rather_than_reporting_clean(tmp_path, monkeypatch):
    """Every bound raises, so "expensive to scan" can never be spent as "no credential found".

    Both caps returned None when they bit, which is the verdict a clean file gets. That made the
    cheapest bypass of the whole check "make it expensive": pad a nested container past the buffer
    cap, or bury the key one layer past the depth cap, and the publish went through with exit 0.
    """
    import gzip
    import io

    from flash import env_secrets as secrets
    from flash.env_secrets import _MAX_CONTAINER_DEPTH, _Unscannable, credential_in_file

    def _gz(payload: bytes) -> bytes:
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as handle:
            handle.write(payload)
        return buf.getvalue()

    # Past the buffer cap, which is lowered so the fixture stays small. The padding is random so it
    # does not compress below the cap, and the credential is followed by compressible bytes so
    # deflate does not emit it as a STORED block -- were it stored, the key would survive literally
    # in the outer file and be found without expanding anything, which tests nothing.
    monkeypatch.setattr(secrets, "_MAX_NESTED_BUFFER_BYTES", 1 << 20)
    padded = os.urandom(4 << 20) + f"fslo_{_FAKE_KEY_BODY}".encode() + b"\n" + b"x" * (1 << 20)
    oversized = _gz(padded)
    assert f"fslo_{_FAKE_KEY_BODY}".encode() not in oversized, "the key survived literally"
    too_big = tmp_path / "padded.gz"
    too_big.write_bytes(_gz(oversized))
    with pytest.raises(_Unscannable, match="too large to inspect"):
        credential_in_file(too_big)

    # Past the depth cap.
    buried = f"fslo_{_FAKE_KEY_BODY}".encode()
    for _ in range(_MAX_CONTAINER_DEPTH + 2):
        buried = _gz(buried)
    too_deep = tmp_path / "buried.gz"
    too_deep.write_bytes(buried)
    with pytest.raises(_Unscannable, match="too deeply to inspect"):
        credential_in_file(too_deep)

    # the control: the same nesting inside the cap still resolves to a found credential, so the
    # refusals above come from the bounds and not from expansion being broken outright.
    shallow = f"fslo_{_FAKE_KEY_BODY}".encode()
    for _ in range(_MAX_CONTAINER_DEPTH - 1):
        shallow = _gz(shallow)
    within = tmp_path / "shallow.gz"
    within.write_bytes(shallow)
    assert credential_in_file(within) == "a Freesolo API key"


def test_a_credential_straddling_a_base64_window_is_still_decoded(tmp_path):
    """A long base64 run is decoded in OVERLAPPING windows, so no cut can hide a key.

    Bounding the run itself was worse than bounding nothing: a base64 blob longer than the bound
    was skipped entirely, so encoding a big config file was enough to publish the key inside it.
    Slicing into adjacent windows instead moved the hole rather than closing it -- a credential
    across the cut decodes into neither piece.
    """
    import base64

    from flash.env_base64 import _BASE64_WINDOW
    from flash.env_secrets import _credential_kind, credential_in_file

    secret = f"fslo_{_FAKE_KEY_BODY}".encode()

    # a run far past any single window, with the key deliberately placed on the seam
    prefix_bytes = (_BASE64_WINDOW // 4) * 3
    blob = base64.b64encode(b"A" * prefix_bytes + secret + b"B" * prefix_bytes)
    assert len(blob) > _BASE64_WINDOW, "the run must exceed one window to test the overlap"
    manifest = tmp_path / "secret.yaml"
    manifest.write_bytes(b"data:\n  bundle: " + blob + b"\n")
    assert credential_in_file(manifest) == "a Freesolo API key"

    # base64 inside a wide encoding: the two supported forms must compose, not merely pass alone
    wide = base64.b64encode(secret).decode().encode("utf-16-le")
    assert _credential_kind(wide) == "a Freesolo API key"


def test_an_assigned_key_with_no_prefix_is_caught_by_its_variable_name(tmp_path):
    """Some keys carry no issuer prefix, so only the assignment identifies them.

    Every other pattern here keys off an issued prefix. These cannot, which is why they are matched
    through the variable they are assigned to rather than by shape -- 40 hex characters on their own
    are just as likely to be a commit sha, and refusing those would block ordinary publishes.
    """
    from flash.env_secrets import _credential_kind, credential_in_file

    # a REAL legacy key's 40 hex characters, not a repeated pair: `3f` x 20 has two distinct
    # characters and is correctly read as a masked value rather than a key, so using it here would
    # have asserted the wrong behaviour.
    key = b"d5c7bfe532fe1fe056b940909986e48aee4f5112"
    script = tmp_path / "run.sh"
    script.write_bytes(b"#!/bin/sh\nexport WANDB_API_KEY=" + key + b"\n")
    assert credential_in_file(script) == "a Weights & Biases API key"

    for form in (b'WANDB_API_KEY: "' + key + b'"', b"wandb_api_key = '" + key + b"'"):
        assert _credential_kind(form) == "a Weights & Biases API key", form

    # an all-hex body reads as all-lowercase-alpha when it happens to avoid digits, which is the
    # hand-written-placeholder shape. Hex of key length is admitted explicitly so this still counts.
    assert _credential_kind(b"WANDB_API_KEY=" + b"abcdef" * 6 + b"abcd") == (
        "a Weights & Biases API key"
    )

    # W&B issued 40-hex keys historically and now issues much longer ones, and a new key does not
    # revoke an existing legacy one -- so both are live and both must be caught. Pinning the body
    # to 40 hex caught only the legacy form and published every currently-issued key.
    current = b"abcdefgh1234" * 7 + b"ab"
    assert len(current) == 86
    assert _credential_kind(b"WANDB_API_KEY=" + current) == "a Weights & Biases API key"

    # AWS secret access keys have no prefix at all, so the variable name is the only context. The
    # public `AKIA...` access key ID is deliberately NOT matched: it appears in signed URLs in the
    # clear and turns up verbatim in scraped datasets.
    aws = b"wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    assert _credential_kind(b"AWS_SECRET_ACCESS_KEY=" + aws) == "an AWS secret access key"

    # the near misses that must stay publishable
    for benign in (
        b"commit: " + key,
        b"WANDB_API_KEY=${WANDB_API_KEY}\n",
        b"WANDB_API_KEY=$(pass show wandb)\n",
        b"sha256: " + key + b"\n",
        b"AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n",
        b"AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}\n",
        b"digest: " + aws,
        b"WANDB_API_KEY=" + b"3f" * 20,  # a masked value, not a key
    ):
        assert _credential_kind(benign) is None, benign


@pytest.mark.parametrize(
    "placeholder",
    [
        b"WANDB_API_KEY=your_wandb_api_key_here_replace_before_push",
        b"WANDB_API_KEY=REPLACE_ME_WITH_YOUR_WANDB_KEY_XXXXXXXX",
        b"WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        b"AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key_here__",
    ],
)
def test_an_assigned_placeholder_does_not_refuse_the_publish(placeholder):
    """The assignment-anchored patterns take the placeholder test too.

    Exempting them was safe only while the W&B body was pinned to 40 hex characters. Widening it to
    catch currently-issued keys made `WANDB_API_KEY=your_wandb_api_key_here...` match, so a
    scaffolded environment could not be published at all -- a false refusal, which is the failure
    mode that gets a check switched off, and worse here than the hole it came from.
    """
    from flash.env_secrets import _credential_kind

    assert _credential_kind(placeholder) is None


def test_a_compressed_member_inside_a_plain_tar_is_expanded(tmp_path):
    """`tar > shard.gz` hides a credential exactly as `zip > shard.gz` does.

    A plain tar was skipped because its own member bytes appear literally, which is true of an
    UNCOMPRESSED member and false of a compressed one. Shipping a dataset as a tar of gzipped
    shards is ordinary, and the key inside published with exit 0.
    """
    import gzip
    import io
    import tarfile

    from flash.env_secrets import credential_in_file

    secret = f'export KEY="fslo_{_FAKE_KEY_BODY}"\n'.encode()

    def _tar(name: str, members: dict[str, bytes]):
        path = tmp_path / name
        with tarfile.open(path, "w") as archive:  # the tar itself is NOT compressed
            for member, body in members.items():
                info = tarfile.TarInfo(member)
                info.size = len(body)
                archive.addfile(info, io.BytesIO(body))
        return path

    packed = _tar("dataset.tar", {"shard.jsonl.gz": gzip.compress(secret)})
    # the premise: the credential survives nowhere in the tar's own bytes
    assert f"fslo_{_FAKE_KEY_BODY}".encode() not in packed.read_bytes()
    assert credential_in_file(packed) == "a Freesolo API key"

    # a member NAME that is the key leaks through the archive listing even with empty contents
    assert credential_in_file(_tar("named.tar", {f"fslo_{_FAKE_KEY_BODY}.json": b""}))

    # the control: an ordinary tar of source files stays publishable
    assert (
        credential_in_file(_tar("clean.tar", {"env.py": b"def load_environment(**k):\n    x\n"}))
        is None
    )


def test_a_nested_self_extracting_zip_is_recognised_by_structure(tmp_path):
    """Nested members were tested on leading magic, so an `MZ` stub hid a zip one layer in.

    Top-level files got `is_zipfile`, which finds the end-of-central-directory record behind any
    preamble. Nested ones did not, so the same bytes were covered when published directly and
    invisible when wrapped in a gzip.
    """
    import gzip
    import io
    import zipfile

    from flash.env_secrets import credential_in_file

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("env.sh", f'export KEY="fslo_{_FAKE_KEY_BODY}"\n')
    stub = b"MZ\x90\x00" + b"\x00" * 2000 + payload.getvalue()

    # the premise: it does not lead with zip magic, but it is a zip
    assert not stub.startswith(b"PK")
    assert zipfile.is_zipfile(io.BytesIO(stub))

    wrapped = tmp_path / "installer.gz"
    wrapped.write_bytes(gzip.compress(stub))
    assert credential_in_file(wrapped) == "a Freesolo API key"


def test_an_archive_of_many_empty_members_is_refused(tmp_path, monkeypatch):
    """`ZipFile` reads the whole central directory up front, before any per-member budget applies.

    A nested archive of millions of empty entries materialises a `ZipInfo` each, and empty members
    never enter the read loop that checks the deadline -- so neither existing bound could stop it,
    while the package extractor counted the whole archive as one ordinary file.
    """
    import zipfile

    from flash import env_secrets as secrets
    from flash.env_secrets import _Unscannable, credential_in_file

    monkeypatch.setattr(secrets, "_MAX_ARCHIVE_MEMBERS", 500)
    crowded = tmp_path / "many.zip"
    with zipfile.ZipFile(crowded, "w", zipfile.ZIP_STORED) as archive:
        for index in range(600):
            archive.writestr(f"{index}", b"")
    with pytest.raises(_Unscannable, match="too many members"):
        credential_in_file(crowded)

    # the control: an archive inside the bound is still scanned to a verdict
    modest = tmp_path / "few.zip"
    with zipfile.ZipFile(modest, "w", zipfile.ZIP_DEFLATED) as archive:
        for index in range(10):
            archive.writestr(f"{index}.txt", "harmless")
        archive.writestr("env.sh", f'export KEY="fslo_{_FAKE_KEY_BODY}"\n')
    assert credential_in_file(modest) == "a Freesolo API key"


def test_one_expansion_budget_covers_the_whole_package(tmp_path, monkeypatch):
    """A per-file budget multiplied by the member limit into hours of permitted expansion.

    A package may hold 5,000 members, so splitting compression bombs across them bought
    5,000 x 60s from an authenticated publish while every individual file stayed inside the
    apparent one-minute limit.
    """
    import gzip
    import time

    from flash import env_secrets as secrets
    from flash.env_secrets import reject_credential_bearing_package

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    bomb = gzip.compress(b"\0" * (200 << 20))
    for index in range(6):
        (package / f"bomb{index}.gz").write_bytes(bomb)

    monkeypatch.setattr(secrets, "_MAX_DECOMPRESS_SECONDS", 1.0)
    started = time.monotonic()
    with pytest.raises(ValueError, match="too long to decompress"):
        reject_credential_bearing_package(package, display={})
    elapsed = time.monotonic() - started

    # the whole package shares the budget, so this cannot approach 6 x 1s
    assert elapsed < 3.0, f"budget appears to reset per file ({elapsed:.1f}s)"


def test_a_refusal_never_echoes_a_credential_in_a_filename(tmp_path):
    """Masking covered only prefixed tokens, so the newer credential forms printed in full.

    The refusal names the member so the author can find it, and when the credential IS that name,
    printing it re-leaks the key into a terminal and whatever collects its output -- the one thing
    the rest of this module is careful never to do.
    """
    import base64

    from flash.env_secrets import _redacted, credential_in_name

    body = "0123456789abcdef" * 2 + "01234567"
    assigned = f"cache/wandb_api_key={body}.json"
    assert credential_in_name(assigned) == "a Weights & Biases API key"
    assert body not in _redacted(assigned)
    assert "wandb_api_key" in _redacted(assigned), "the author still needs to find the file"

    # a base64 name has no plaintext body to mask, so the name is withheld and the directory given
    encoded = base64.b64encode(f"fslo_{_FAKE_KEY_BODY}".encode()).decode().replace("=", "")
    nested = f"cache/{encoded}.bin"
    assert credential_in_name(nested) == "a Freesolo API key"
    assert encoded[:20] not in _redacted(nested)
    assert "cache/" in _redacted(nested)

    # an ordinary path is returned untouched
    assert _redacted("src/environment.py") == "src/environment.py"


def test_a_tar_nested_inside_another_container_is_expanded(tmp_path):
    """Tar was recognised at top level only, so one layer in it became final content again.

    `tar.gz` holding a tar of gzipped shards is an ordinary dataset layout, and the innermost key
    was unreachable: the outer gzip expanded, and what came out was treated as bytes rather than as
    the archive it is.
    """
    import gzip
    import io
    import tarfile
    import zipfile

    from flash.env_secrets import credential_in_file

    secret = f'export KEY="fslo_{_FAKE_KEY_BODY}"\n'.encode()
    inner = io.BytesIO()
    with tarfile.open(fileobj=inner, mode="w") as archive:
        payload = gzip.compress(secret)
        info = tarfile.TarInfo("shard.jsonl.gz")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    plain = inner.getvalue()

    wrapped = tmp_path / "dataset.tar.gz"
    wrapped.write_bytes(gzip.compress(plain))
    assert credential_in_file(wrapped) == "a Freesolo API key"

    zipped = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zipped, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("inner.tar", plain)
    assert credential_in_file(zipped) == "a Freesolo API key"


def test_a_corrupt_tar_does_not_crash_the_publish(tmp_path):
    """`TarError` inherits straight from `Exception`, so no other entry in the tuple covers it.

    A truncated tar raised `ReadError: unexpected end of data` out of the publish as a traceback.
    A half-written shard in a dataset directory is ordinary, so crashing on it would be a worse bug
    than the hole being closed.

    It refuses rather than returning None, which is the same distinction the unreadable zip member
    draws: the member's declared bytes are not in the file, so nothing read them, and a clean
    verdict would be one nobody checked. Not crashing is what this test is about, and a refusal is
    a decision rather than a traceback.
    """
    import io
    import tarfile

    from flash.env_secrets import _Unscannable, credential_in_file

    whole = io.BytesIO()
    with tarfile.open(fileobj=whole, mode="w") as archive:
        body = b"x" * 4096
        info = tarfile.TarInfo("a.txt")
        info.size = len(body)
        archive.addfile(info, io.BytesIO(body))

    truncated = tmp_path / "half.tar"
    truncated.write_bytes(whole.getvalue()[:900])
    with pytest.raises(_Unscannable, match="cannot read"):
        credential_in_file(truncated)


def test_a_truncated_tar_member_is_refused_rather_than_skipped(tmp_path):
    """A member declaring more bytes than the file holds was scanned by nobody and passed.

    The read fails AND the iterator fails again walking to the next header, and that second failure
    escapes any per-member guard into the dispatch loop -- which reads it as "not a tar" and
    published the file on its literal bytes. A zlib-compressed key inside such a member returned
    clean, while the same member on its own reported the key: the outer raw scan does not look for
    a compressed record at an arbitrary offset, so nothing else would have caught it.
    """
    import io
    import tarfile
    import zlib

    from flash.env_secrets import _Unscannable, credential_in_file

    inner = zlib.compress(b'export FREESOLO_API_KEY="fslo_%s"\n' % _FAKE_KEY_BODY.encode())

    # the control: the same member, honestly sized, is read and its credential found
    honest = io.BytesIO()
    with tarfile.open(fileobj=honest, mode="w") as archive:
        info = tarfile.TarInfo("payload.z")
        info.size = len(inner)
        archive.addfile(info, io.BytesIO(inner))
    intact = tmp_path / "intact.tar"
    intact.write_bytes(honest.getvalue())
    assert credential_in_file(intact) == "a Freesolo API key"

    lying = io.BytesIO()
    with tarfile.open(fileobj=lying, mode="w") as archive:
        info = tarfile.TarInfo("payload.z")
        info.size = 1 << 20
        archive.addfile(info, io.BytesIO(inner + b"\0" * ((1 << 20) - len(inner))))
    cut = tmp_path / "lying.tar"
    cut.write_bytes(lying.getvalue()[: 512 + len(inner)])
    with pytest.raises(_Unscannable, match="cannot read"):
        credential_in_file(cut)

    # a file that is not a tar raises the SAME exception from the same place, and must still be
    # scanned on its own bytes rather than refused -- that is most of any real package
    for name, data in (
        ("notes.txt", b"# ordinary\nDEBUG=1\n"),
        ("blob.bin", bytes(range(256)) * 8),
    ):
        ordinary = tmp_path / name
        ordinary.write_bytes(data)
        assert credential_in_file(ordinary) is None, name


def test_an_oversized_tar_refuses_rather_than_passing(tmp_path, monkeypatch):
    """The buffer-cap escape recognised only compressed magic, so a big tar took the pass branch.

    A tar past the cap is exactly as unverifiable as a gzip past it -- its members can be
    compressed, and those hold the credential nowhere the literal scan can see.

    What must never happen is a silent None. Refusing was the original fix; DETECTING is strictly
    better and is what happens now that the container dispatch tries every applicable format
    instead of committing to the first one that claims the bytes -- the tar is enumerated, the
    gzipped member expanded, and the key found rather than merely suspected. Both outcomes are
    accepted here because both are safe; a None is the bug.

    The padding is SEEDED rather than `os.urandom`. It only has to be incompressible enough to
    carry the tar past the cap, and 4 MiB of unpredictable bytes contains a plausible OpenPGP
    packet header often enough to matter: measured 4 failures in 40 runs, where the scan refuses
    with an unwalkable-packet message instead. That refusal is the scanner being correctly
    fail-closed about bytes it cannot account for, so widening the assertion to accept it would
    dissolve the very outcome this test exists to pin. A fixed sequence compresses no better
    (4.19 MB gzipped, still four times the cap) and removes the dice roll from CI.
    """
    import gzip
    import io
    import random
    import tarfile

    from flash import env_secrets as secrets
    from flash.env_secrets import _Unscannable, credential_in_file

    monkeypatch.setattr(secrets, "_MAX_NESTED_BUFFER_BYTES", 1 << 20)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as archive:
        for name, body in (
            ("pad.bin", random.Random(0).randbytes(4 << 20)),
            ("shard.gz", gzip.compress(f'export KEY="fslo_{_FAKE_KEY_BODY}"\n'.encode())),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))

    oversized = tmp_path / "big.tar.gz"
    oversized.write_bytes(gzip.compress(buf.getvalue()))
    outcome: str | None
    try:
        outcome = credential_in_file(oversized)
    except _Unscannable as refusal:
        outcome = str(refusal)
    assert outcome in ("a Freesolo API key", "contains an archive too large to inspect")


def test_a_credential_in_line_wrapped_base64_is_decoded(tmp_path):
    """MIME wraps base64 every 76 characters, and the run pattern stops at the newline.

    A wrapped blob therefore arrived as a series of per-line runs, and a credential crossing a break
    decoded into neither -- 20 of 60 possible key offsets, i.e. every key unlucky enough to straddle
    a line.
    """
    import base64

    from flash.env_secrets import _credential_kind

    secret = f"fslo_{_FAKE_KEY_BODY}".encode()
    # sweep the key across the 76-column boundary rather than testing one lucky placement
    for pad in range(30, 56):
        wrapped = base64.encodebytes(b"P" * pad + secret + b"Q" * 40)
        assert wrapped.count(b"\n") > 1, "the fixture must actually wrap"
        assert _credential_kind(wrapped) == "a Freesolo API key", pad

    # unrelated adjacent values must NOT be welded into one run: joining arbitrary line pairs
    # would decode bytes that neither line contained.
    pair = (
        b"KEY="
        + base64.b64encode(b"hello there friend")
        + b"\nOTHER="
        + base64.b64encode(b"goodbye now everyone")
    )
    assert _credential_kind(pair) is None


def _wrapped(raw: bytes, width: int, eol: bytes) -> bytes:
    """`raw` base64-encoded and broken every `width` characters with `eol`."""
    import base64

    body = base64.b64encode(raw)
    return eol.join(body[i : i + width] for i in range(0, len(body), width))


def test_a_credential_crossing_a_crlf_or_single_line_break_is_decoded():
    """The two shapes the first wrapping pass could not join.

    It required two or more FULL-width lines, so the commonest blob of all -- one just over the
    width, i.e. one full line plus a short tail -- never joined, and it matched only `\\n`, so a
    Windows checkout or a YAML export wrapping with CRLF never joined either. Both leave a key
    that straddles the break decoding into neither side: 20 of 70 offsets missed on CRLF, 3 on
    the two-line blob.

    Sweeping the key across the boundary is what exposes this. A single placement is not enough --
    a 53-byte key fits inside one 76-column line, so a fixed fixture reports "caught" from the
    per-line run while the join is doing nothing.
    """
    from flash.env_secrets import _credential_kind

    secret = f"fslo_{_FAKE_KEY_BODY}".encode()
    for width in (76, 64):
        for eol in (b"\n", b"\r\n"):
            for pad in range(70):
                # a tail of filler forces several lines; no tail is the one-full-line-plus-tail
                # shape, which is the case that never joined at all
                for tail in (b"", b"Q" * 40):
                    blob = _wrapped(b"P" * pad + secret + tail, width, eol)
                    if eol not in blob:
                        continue  # too short to wrap, so nothing to join
                    assert _credential_kind(blob) == "a Freesolo API key", (width, eol, pad, tail)

    # control: adjacent full-width lines of unrelated data must still not be welded together
    import base64

    unrelated = (
        base64.b64encode(b"first value here, long enough to fill a line" * 2)[:76]
        + b"\r\n"
        + base64.b64encode(b"second unrelated value entirely" * 2)[:76]
        + b"\r\n"
    )
    assert _credential_kind(unrelated) is None


def test_a_binary_der_private_key_is_detected(tmp_path):
    """A DER key is the same key a PEM block base64-wraps, with no text marker to match.

    `openssl genpkey -outform DER` writes one, and the PEM pattern cannot see it, so a private key
    published intact as long as it was not armoured.
    """
    import subprocess

    from flash.env_secrets import credential_in_file

    def _generate(*args: str, out: str):
        path = tmp_path / out
        subprocess.run(
            ["openssl", "genpkey", *args, "-out", str(path)],
            check=True,
            capture_output=True,
        )
        return path

    rsa = _generate(
        "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:2048", "-outform", "DER", out="rsa.der"
    )
    assert credential_in_file(rsa) == "a private key"
    ed = _generate("-algorithm", "ED25519", "-outform", "DER", out="ed.der")
    assert credential_in_file(ed) == "a private key"

    # the armoured form still reports as the PEM block it is
    pem = _generate("-algorithm", "ED25519", out="key.pem")
    assert credential_in_file(pem) == "a private key block"

    # a public key in DER carries the same algorithm OID but no version INTEGER, so it must pass
    public = tmp_path / "pub.der"
    subprocess.run(
        ["openssl", "pkey", "-in", str(pem), "-pubout", "-outform", "DER", "-out", str(public)],
        check=True,
        capture_output=True,
    )
    assert credential_in_file(public) is None


def test_every_rfc_8410_curve_is_detected_not_just_the_25519_pair(tmp_path):
    """The four RFC 8410 OIDs are `1.3.101.{110,111,112,113}`, final byte 0x6e-0x71.

    Naming only 0x6e and 0x70 covered X25519 and Ed25519 and left Ed448 and X448 undetected, so a
    real private key on either of those curves published intact. They are one contiguous range.
    """
    import subprocess

    from flash.env_secrets import credential_in_file

    for algorithm in ("ED25519", "X25519", "ED448", "X448"):
        path = tmp_path / f"{algorithm}.der"
        subprocess.run(
            ["openssl", "genpkey", "-algorithm", algorithm, "-outform", "DER", "-out", str(path)],
            check=True,
            capture_output=True,
        )
        assert credential_in_file(path) == "a private key", algorithm

        # the public half of the same key carries the same OID and must still publish
        public = tmp_path / f"{algorithm}.pub.der"
        subprocess.run(
            [
                "openssl",
                "pkey",
                "-in",
                str(path),
                "-inform",
                "DER",
                "-pubout",
                "-outform",
                "DER",
                "-out",
                str(public),
            ],
            check=True,
            capture_output=True,
        )
        assert credential_in_file(public) is None, algorithm


def test_an_openpgp_secret_key_is_detected_armoured_and_binary(tmp_path):
    """`gpg --export-secret-keys` writes a private key in two forms and neither was caught.

    Armoured, the header is `-----BEGIN PGP PRIVATE KEY BLOCK-----`, and `[A-Z ]*PRIVATE KEY-----`
    cannot match the trailing ` BLOCK`. Binary, there is no text header at all and the key material
    is neither base64 nor DER, so every other check passed it through.
    """
    import shutil
    import subprocess

    if shutil.which("gpg") is None:
        pytest.skip("gpg is not installed")

    from flash.env_secrets import credential_in_file

    home = tmp_path / "gnupg"
    home.mkdir(mode=0o700)
    env = {"GNUPGHOME": str(home), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    subprocess.run(
        [
            "gpg",
            "--batch",
            "--passphrase",
            "",
            "--quick-gen-key",
            "test@example.com",
            "default",
            "default",
            "never",
        ],
        check=True,
        capture_output=True,
        env=env,
    )

    def _export(*flags: str) -> bytes:
        return subprocess.run(
            [
                "gpg",
                "--batch",
                "--pinentry-mode",
                "loopback",
                "--passphrase",
                "",
                *flags,
                "test@example.com",
            ],
            check=True,
            capture_output=True,
            env=env,
        ).stdout

    armoured = tmp_path / "secret.asc"
    armoured.write_bytes(_export("--export-secret-keys", "--armor"))
    assert credential_in_file(armoured) == "a private key block"

    binary = tmp_path / "secret.gpg"
    binary.write_bytes(_export("--export-secret-keys"))
    assert credential_in_file(binary) == "a private key"

    # the PUBLIC keyring is meant to be shared: packet tags 6 and 14, not 5 and 7
    for flags, name in ((("--export", "--armor"), "public.asc"), (("--export",), "public.gpg")):
        public = tmp_path / name
        public.write_bytes(_export(*flags))
        assert credential_in_file(public) is None, name


def test_the_openpgp_packet_header_does_not_fire_on_ordinary_binaries():
    """Anchored at offset 0, because searching for these bytes anywhere would refuse real files.

    The header is only a couple of constrained bytes; matched anywhere it would hit roughly once
    per megabyte of arbitrary data, which is a false refusal on every model shard in a package.
    """
    import random

    from flash.env_openpgp import _is_openpgp_secret_key

    # measured 1 in 4,400 on tag plus version alone, and 1 in 108,000 once the algorithm byte is
    # required too. 40,000 draws would fail essentially always at the former rate and pass at this
    # one, which is what makes this a regression test rather than a coincidence.
    #
    # SEEDED, not `os.urandom`: at 1 in 108,000 an unseeded run of this size fails a few times in
    # ten by luck alone, and a security check that goes red at random gets switched off.
    draws = random.Random(0)
    assert not any(_is_openpgp_secret_key(draws.randbytes(16)) for _ in range(40_000))
    # a real ELF, a zip and a PNG all start with bytes that must not be read as a packet tag
    for head in (
        b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        b"PK\x03\x04\x14\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00",
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
    ):
        assert not _is_openpgp_secret_key(head), head


def test_a_hex_body_under_an_issuer_prefix_is_a_placeholder_not_a_key():
    """The all-hex carve-out belongs to the assignment-anchored patterns only.

    It exists because a legacy W&B key IS 40 hex characters, and `abcdef...` reads as
    all-lowercase-alpha, i.e. as a placeholder. Applying it to every pattern refused
    `hf_deadbeefdeadbeefdeadbeefdeadbeef` -- the canonical hex placeholder -- because the hex test
    ran before the rule that would have cleared it. An issued `hf_`/`fslo_` body is base62, so one
    confined to `[a-f]` with no digit is not a shape they take.
    """
    from flash.env_secrets import _credential_kind

    for placeholder in (
        b"hf_deadbeefdeadbeefdeadbeefdeadbeef",
        b"fslo_deadbeefdeadbeefdeadbeefdeadbeef",
    ):
        assert _credential_kind(placeholder) is None, placeholder

    # real keys under the same prefixes are still refused
    assert _credential_kind(f"hf_{_FAKE_KEY_BODY}".encode()) == "a Hugging Face token"
    assert _credential_kind(f"fslo_{_FAKE_KEY_BODY}".encode()) == "a Freesolo API key"

    # and the W&B key the carve-out exists for is still refused, its placeholder still allowed
    real = b"WANDB_API_KEY=d5c7bfe532fe1fe056b940909986e48aee4f5112"
    assert _credential_kind(real) == "a Weights & Biases API key"
    assert _credential_kind(b"WANDB_API_KEY=your_wandb_api_key_here_replace_before_push") is None


def test_an_armoured_key_is_detected_through_its_armor_headers():
    """RFC 4880 armor puts `Version:`/`Comment:` between the BEGIN line and the body.

    Requiring base64 immediately after the header caught only a headerless export, so the form most
    implementations actually emit -- and any hand-annotated backup -- went undetected.

    The header keys are named exactly rather than accepting `[A-Za-z-]+:`, because a general rule
    would also skip `Warning:` and `Note:`, which is prose about a key rather than a key.
    """
    from flash.env_secrets import _credential_kind

    body = "A" * 64
    for headers in (
        b"",
        b"Version: GnuPG v2.4.4\n",
        b"Comment: exported for backup\n",
        b"Version: GnuPG v2.4.4\nComment: exported for backup\n",
    ):
        armoured = b"-----BEGIN PGP PRIVATE KEY BLOCK-----\n" + headers + b"\n" + body.encode()
        assert _credential_kind(armoured) == "a private key block", headers

    # prose about a key is still not a key, which is what naming the headers exactly preserves
    prose = b"-----BEGIN RSA PRIVATE KEY-----\nWarning: never commit one of these\n"
    assert _credential_kind(prose) is None


def test_every_openpgp_packet_length_encoding_is_parsed():
    """New-format packets encode the length in one, two or five octets, not always one.

    Assuming one put the version byte at the wrong offset for any packet of 192 bytes or more --
    every RSA secret key, and anything Sequoia, RNP or `--use-new-packet-format` writes -- so those
    returned false and published intact.
    """
    from flash.env_openpgp import _is_openpgp_secret_key

    def _packet(tag: int, body_length: int) -> bytes:
        if body_length < 192:
            length = bytes([body_length])
        elif body_length < 8384:
            offset = body_length - 192
            length = bytes([(offset >> 8) + 192, offset & 0xFF])
        else:
            length = b"\xff" + body_length.to_bytes(4, "big")
        # version 4, a four-byte creation time, then algorithm 1 (RSA)
        return bytes([tag]) + length + b"\x04" + b"\x6a\x7e\x24\x0a" + b"\x01" + b"\x00" * 16

    for body_length in (10, 191, 192, 500, 8383, 8384, 20_000, 100_000):
        for tag in (0xC5, 0xC7):  # secret key and secret subkey, new format
            assert _is_openpgp_secret_key(_packet(tag, body_length)), (tag, body_length)

    # old format: the low two bits select a 1, 2 or 4 byte length, for both secret tags
    lengths = {0: b"\x64", 1: b"\x01\xf4", 2: b"\x00\x01\x00\x00"}
    tail = b"\x04\x6a\x7e\x24\x0a\x01" + b"\x00" * 16
    for tag in (5, 7):
        for length_type, length in lengths.items():
            first = 0x80 | (tag << 2) | length_type
            assert _is_openpgp_secret_key(bytes([first]) + length + tail), (tag, length_type)

    # the PUBLIC halves, tags 6 and 14, must still publish
    for tag in (6, 14):
        for length_type, length in lengths.items():
            first = 0x80 | (tag << 2) | length_type
            assert not _is_openpgp_secret_key(bytes([first]) + length + tail), (tag, length_type)


def test_a_v7_tar_is_enumerated_despite_having_no_magic(tmp_path):
    """V7 is the original pre-POSIX tar, still written by `tar --format=v7`, and has NO magic.

    Offset 257 is zero padding, so testing the magic alone left it unrecognised: a v7 tar holding a
    gzipped credential was never enumerated and published intact. The header checksum identifies it
    instead, which is a structural property an ordinary file does not satisfy by chance.
    """
    import gzip
    import subprocess

    from flash.env_secrets import credential_in_file

    source = tmp_path / "src"
    source.mkdir()
    (source / "shard.gz").write_bytes(
        gzip.compress(f'export KEY="fslo_{_FAKE_KEY_BODY}"\n'.encode())
    )
    archive = tmp_path / "v7.tar"
    subprocess.run(
        ["tar", "--format=v7", "-cf", str(archive), "-C", str(source), "shard.gz"],
        check=True,
        capture_output=True,
    )
    assert archive.read_bytes()[257:265] == b"\x00" * 8, "the fixture must have no ustar magic"
    assert credential_in_file(archive) == "a Freesolo API key"

    # an ordinary binary must not be mistaken for a magic-less tar
    plain = tmp_path / "weights.bin"
    plain.write_bytes(b"\x41" * 4096)
    assert credential_in_file(plain) is None


def test_stray_zip_bytes_in_a_tar_do_not_abandon_the_scan(tmp_path):
    """`is_zipfile` searches the last 64 KiB for `PK\\x05\\x06`, so four stray bytes claimed a tar.

    The tar was then opened as a zip, failed, and the failure was read as "nothing here" -- so
    adding four bytes to any member took a gzipped credential inside it from refused to published.
    Every applicable format is now tried rather than only the first one that claims the bytes.
    """
    import gzip
    import io
    import tarfile
    import zipfile

    from flash.env_secrets import credential_in_file

    def _tar(name: str, filler: bytes):
        path = tmp_path / name
        with tarfile.open(path, "w") as archive:
            for member, body in (
                ("shard.gz", gzip.compress(f'export KEY="fslo_{_FAKE_KEY_BODY}"\n'.encode())),
                ("readme.txt", filler),
            ):
                info = tarfile.TarInfo(member)
                info.size = len(body)
                archive.addfile(info, io.BytesIO(body))
        return path

    poisoned = _tar("poisoned.tar", b"PK\x05\x06 filler")
    assert zipfile.is_zipfile(poisoned), "the fixture must actually fool is_zipfile"
    assert credential_in_file(poisoned) == "a Freesolo API key"

    control = _tar("control.tar", b"harmless filler")
    assert not zipfile.is_zipfile(control)
    assert credential_in_file(control) == "a Freesolo API key"


def test_an_absurd_member_count_is_refused_before_the_directory_is_parsed(tmp_path):
    """`ZipFile.__init__` materializes every `ZipInfo`, so a bound checked after it pays the cost.

    Measured at 1.8 seconds and 239 MB of resident memory for 400,000 empty entries in a 35 MB
    file, all of it spent before the per-member loop ran once. The count is read from the
    end-of-central-directory record first, which brought the same archive to 4 MB.
    """
    import zipfile

    from flash import env_secrets as secrets
    from flash.env_secrets import _Unscannable, _zip_member_count, credential_in_file

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(secrets, "_MAX_ARCHIVE_MEMBERS", 100)
    try:
        crowded = tmp_path / "crowded.zip"
        with zipfile.ZipFile(crowded, "w", zipfile.ZIP_STORED) as archive:
            for index in range(500):
                archive.writestr(f"{index}", b"")
        assert _zip_member_count(crowded) == 500
        with pytest.raises(_Unscannable, match="too many members"):
            credential_in_file(crowded)
    finally:
        monkeypatch.undo()

    # an ordinary archive is unaffected, and its credential is still found
    modest = tmp_path / "modest.zip"
    with zipfile.ZipFile(modest, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("env.sh", f'export KEY="fslo_{_FAKE_KEY_BODY}"\n')
    assert credential_in_file(modest) == "a Freesolo API key"


def test_a_stray_zip_signature_does_not_refuse_an_oversized_member():
    """The bare four-byte signature occurs by chance about once per 4 GB of arbitrary data.

    Tested over the last 64 KiB of every member too large to buffer, that refused a model shard as
    an unverifiable archive with a message about an archive that was never there. A real record
    states its own comment length at offset 20, and requiring that to agree drops the chance hits.
    """
    import struct

    from flash.env_secrets import _has_zip_end_record

    genuine = b"PK\x05\x06" + b"\x00" * 16 + struct.pack("<H", 5) + b"hello"
    assert _has_zip_end_record(genuine)
    # a truthful record with no comment at all is the common case
    assert _has_zip_end_record(b"PK\x05\x06" + b"\x00" * 16 + struct.pack("<H", 0))
    # the signature alone, with a length that does not describe what follows, is chance
    assert not _has_zip_end_record(b"PK\x05\x06" + b"\x00" * 16 + struct.pack("<H", 99) + b"hi")
    assert not _has_zip_end_record(b"\x41" * 4096 + b"PK\x05\x06" + b"\x00" * 12)


def test_a_refusal_from_the_wrong_format_does_not_end_the_scan(tmp_path):
    """A handler's limits are applied to bytes that may not be its format at all.

    `_zip_member_count` reads the end-of-central-directory record, so a tar carrying a fake one
    with the zip64 sentinel and no zip64 record made the ZIP handler refuse for "too many members".
    `_Unscannable` is deliberately outside `_UNREADABLE_ARCHIVE`, so that refusal escaped before
    the tar handler ran and the real credential inside was never reported.
    """
    import gzip
    import io
    import struct
    import tarfile

    from flash.env_secrets import credential_in_file

    # an end record claiming the zip64 sentinel count, with no zip64 record behind it
    sentinel = b"PK\x05\x06" + b"\x00" * 6 + struct.pack("<H", 0xFFFF) + b"\x00" * 8
    sentinel += struct.pack("<H", 0)

    poisoned = tmp_path / "poisoned.tar"
    with tarfile.open(poisoned, "w") as archive:
        for member, body in (
            ("shard.gz", gzip.compress(f'export KEY="fslo_{_FAKE_KEY_BODY}"\n'.encode())),
            ("pad.bin", sentinel),
        ):
            info = tarfile.TarInfo(member)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))

    assert credential_in_file(poisoned) == "a Freesolo API key"


def test_a_genuinely_oversized_archive_still_refuses(tmp_path):
    """Deferring a refusal must not discard it: no handler completing means nothing was verified.

    The deferred refusal is re-raised when every handler failed, so a real archive over the member
    bound is still fail-closed rather than quietly passing.
    """
    import zipfile

    from flash import env_secrets as secrets
    from flash.env_secrets import _Unscannable, credential_in_file

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(secrets, "_MAX_ARCHIVE_MEMBERS", 100)
    try:
        crowded = tmp_path / "crowded.zip"
        with zipfile.ZipFile(crowded, "w", zipfile.ZIP_STORED) as archive:
            for index in range(500):
                archive.writestr(f"{index}", b"")
        with pytest.raises(_Unscannable, match="too many members"):
            credential_in_file(crowded)
    finally:
        monkeypatch.undo()


def test_an_indented_wrapped_base64_block_is_joined():
    """A wrapped blob is routinely indented: a YAML block scalar, a value under a `data:` key.

    Requiring the next line to start in column 0 missed 20 of 60 key offsets on a two-space-indented
    block, at both widths -- and indented is the commonest way a blob is embedded in a config file.
    """
    import base64
    import os

    from flash.env_secrets import _credential_kind

    secret = f"fslo_{_FAKE_KEY_BODY}".encode()
    for width in (76, 64):
        for indent in (b"", b"  ", b"    ", b"\t"):
            for pad in range(60):
                body = base64.b64encode(b"P" * pad + secret)
                blob = b"\n".join(
                    indent + body[index : index + width] for index in range(0, len(body), width)
                )
                if b"\n" not in blob:
                    continue
                assert _credential_kind(blob) == "a Freesolo API key", (width, indent, pad)

    # the indent must not weld two unrelated indented values into one decodable run
    unrelated = (
        base64.b64encode(os.urandom(57)) + b"\n  " + base64.b64encode(os.urandom(57)) + b"\n"
    )
    assert _credential_kind(unrelated) is None


def test_a_name_holding_a_lone_surrogate_does_not_crash_the_check():
    """`surrogateescape` is a DECODE-only handler, so encoding a lone surrogate raised.

    Raised out of a security check, that turned the publish route's 400 into an uncaught 500, and
    crashed the scan of an archive whose member name held one rather than reporting its contents.
    """
    from flash.env_secrets import _redacted, credential_in_name

    assert credential_in_name("bad\ud800name") is None
    # a real credential in a name that ALSO holds a surrogate is still found
    key = f"fslo_{_FAKE_KEY_BODY}"
    assert credential_in_name(f"bad\ud800name-{key}") == "a Freesolo API key"
    # and reporting it does not crash either, since the refusal runs the redactor on that name
    assert key not in _redacted(f"bad\ud800name-{key}")


def test_an_encrypted_pkcs8_key_in_der_is_detected(tmp_path):
    """`openssl pkcs8 -topk8 -passout` in DER hides the key inside an OCTET STRING.

    None of the plaintext structures appear anywhere in the file, so it published intact -- while
    the armoured form of the same key was caught by its `BEGIN ENCRYPTED PRIVATE KEY` header, which
    made DER the way past. A passphrase is not much protection for a key in a public repository.
    """
    import subprocess

    from flash.env_secrets import credential_in_file

    plain = tmp_path / "plain.pem"
    subprocess.run(
        [
            "openssl",
            "genpkey",
            "-algorithm",
            "RSA",
            "-pkeyopt",
            "rsa_keygen_bits:2048",
            "-out",
            str(plain),
        ],
        check=True,
        capture_output=True,
    )
    for label, extra in (("pbes2", []), ("pkcs12pbe", ["-v1", "PBE-SHA1-3DES"])):
        encrypted = tmp_path / f"{label}.der"
        subprocess.run(
            [
                "openssl",
                "pkcs8",
                "-topk8",
                "-in",
                str(plain),
                "-outform",
                "DER",
                "-out",
                str(encrypted),
                "-passout",
                "pass:hunter2",
                *extra,
            ],
            check=True,
            capture_output=True,
        )
        assert credential_in_file(encrypted) == "a private key", label

    # the public half is not a private key and must still publish
    public = tmp_path / "pub.der"
    subprocess.run(
        ["openssl", "pkey", "-in", str(plain), "-pubout", "-outform", "DER", "-out", str(public)],
        check=True,
        capture_output=True,
    )
    assert credential_in_file(public) is None


def test_a_container_that_cannot_be_expanded_refuses_rather_than_passing(tmp_path):
    """A `.zst` shard holds its credential nowhere a pattern can see, exactly like a gzip.

    The stdlib has no zstd decompressor, so treating it as final content was a silent bypass.
    Refusing says what is true -- not verified -- without adding a dependency to inspect a format
    no environment in the hub currently uses.
    """
    import shutil
    import subprocess

    if shutil.which("zstd") is None:
        pytest.skip("zstd is not installed")

    from flash.env_secrets import _Unscannable, credential_in_file

    plain = tmp_path / "data.jsonl"
    plain.write_text(f'{{"key": "fslo_{_FAKE_KEY_BODY}"}}\n')
    compressed = tmp_path / "data.jsonl.zst"
    subprocess.run(
        ["zstd", "-q", "-f", str(plain), "-o", str(compressed)], check=True, capture_output=True
    )
    with pytest.raises(_Unscannable, match="cannot expand"):
        credential_in_file(compressed)


def test_a_pem_label_line_must_be_a_real_encrypted_key_header(tmp_path):
    """`Warning:` is prose; `Proc-Type:` and `DEK-Info:` are RFC 1421 encrypted-key headers.

    Admitting any capitalized word plus a colon reopened the prose false positive that requiring a
    body was meant to close.
    """
    from flash.env_secrets import _credential_kind

    for prose in (
        b"See -----BEGIN RSA PRIVATE KEY-----\nWarning: never commit keys to the repo",
        b"-----BEGIN RSA PRIVATE KEY-----\nNote: redact this before sharing a log",
    ):
        assert _credential_kind(prose) is None, prose

    for real in (
        b"-----BEGIN RSA PRIVATE KEY-----\nProc-Type: 4,ENCRYPTED\nDEK-Info: AES-128-CBC\n",
        b"-----BEGIN RSA PRIVATE KEY-----\nDEK-Info: AES-128-CBC\n",
    ):
        assert _credential_kind(real) == "a private key block", real


def test_an_encrypted_zip_member_is_refused_rather_than_skipped(tmp_path):
    """A password-encrypted member's bytes cannot be read, and unreadable is not clean.

    `archive.open()` raises `RuntimeError` for one, which `_UNREADABLE_ARCHIVE` caught as an
    opaque member and skipped -- so a package whose only copy of a credential sat inside an
    encrypted member was approved.
    """
    import zipfile

    from flash.env_secrets import _Unscannable, credential_in_file

    readable = tmp_path / "readable.zip"
    with zipfile.ZipFile(readable, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("key", f'export KEY="fslo_{_FAKE_KEY_BODY}"\n' * 20)
    # deflated, so the credential is not sitting in the raw bytes for the outer scan to find
    assert f"fslo_{_FAKE_KEY_BODY}".encode() not in readable.read_bytes()
    assert credential_in_file(readable) == "a Freesolo API key"

    # the same archive with bit 0 of the general-purpose flags set on the member
    raw = bytearray(readable.read_bytes())
    raw[raw.find(b"PK\x03\x04") + 6] |= 0x01
    raw[raw.find(b"PK\x01\x02") + 8] |= 0x01
    encrypted = tmp_path / "encrypted.zip"
    encrypted.write_bytes(bytes(raw))
    with pytest.raises(_Unscannable, match="encrypted archive member"):
        credential_in_file(encrypted)


def test_a_patched_member_count_cannot_shrink_a_real_central_directory(tmp_path):
    """`ZipFile` walks the central directory by SIZE, not by the end record's count field.

    Patching both count fields of a real 500-entry archive down to 1 left every entry materialized
    while the bound saw a single member, restoring the memory and CPU exposure the bound exists to
    prevent. The directory is walked instead, which reads the same bytes without allocating a
    `ZipInfo` per entry.
    """
    import zipfile

    from flash import env_secrets as secrets
    from flash.env_secrets import _Unscannable, _zip_member_count, credential_in_file

    crowded = tmp_path / "crowded.zip"
    with zipfile.ZipFile(crowded, "w", zipfile.ZIP_STORED) as archive:
        for index in range(500):
            archive.writestr(f"{index}", b"")
    raw = bytearray(crowded.read_bytes())
    end = raw.rfind(b"PK\x05\x06")
    raw[end + 8 : end + 10] = (1).to_bytes(2, "little")
    raw[end + 10 : end + 12] = (1).to_bytes(2, "little")
    patched = tmp_path / "patched.zip"
    patched.write_bytes(bytes(raw))

    # the archive really does still hold all 500, whatever its end record claims
    with zipfile.ZipFile(patched) as archive:
        assert len(archive.infolist()) == 500
    assert _zip_member_count(patched) == 500

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(secrets, "_MAX_ARCHIVE_MEMBERS", 100)
    try:
        with pytest.raises(_Unscannable, match="too many members"):
            credential_in_file(patched)
    finally:
        monkeypatch.undo()


def test_a_v7_tar_with_a_non_ascii_name_is_still_expanded(tmp_path):
    """The signed-checksum branch built a `bytes` from values that go negative above 127.

    Any V7 header holding a non-ASCII byte -- a UTF-8 filename is enough -- raised `ValueError`
    there. Inside a zip that surfaced as an unreadable member and was skipped, so a credential
    compressed within the tar was approved.
    """
    import gzip
    import subprocess
    import zipfile

    from flash.env_secrets import credential_in_file

    source = tmp_path / "src"
    source.mkdir()
    (source / "é.gz").write_bytes(gzip.compress(f'export KEY="fslo_{_FAKE_KEY_BODY}"\n'.encode()))
    archive = tmp_path / "v7.tar"
    subprocess.run(
        ["tar", "--format=v7", "-cf", str(archive), "-C", str(source), "é.gz"],
        check=True,
        capture_output=True,
    )
    assert archive.read_bytes()[257:265] == b"\x00" * 8, "the fixture must have no ustar magic"
    assert any(byte > 127 for byte in archive.read_bytes()[:512]), "header must be non-ASCII"
    assert credential_in_file(archive) == "a Freesolo API key"

    # and nested inside a zip, where the exception was swallowed as an unreadable member
    nested = tmp_path / "nested.zip"
    with zipfile.ZipFile(nested, "w", zipfile.ZIP_DEFLATED) as outer:
        outer.write(archive, "inner.tar")
    assert credential_in_file(nested) == "a Freesolo API key"


def test_a_url_safe_base64_credential_is_decoded():
    """RFC 4648 section 5 swaps `+/` for `-_`, which is what a JWT or a token-in-a-URL uses.

    Accepting only the standard alphabet split such a value at its first `-`, and the surviving
    fragments decoded to neither the whole token nor anything matching.
    """
    import base64

    from flash.env_secrets import _credential_kind

    # a leading 0xf8 byte is what forces the differing character into the encoding
    raw = b"\xf8fslo_" + _FAKE_KEY_BODY.encode()
    standard = base64.b64encode(raw)
    url_safe = base64.urlsafe_b64encode(raw)
    assert standard != url_safe, "the fixture must actually differ between the two alphabets"

    assert _credential_kind(standard) == "a Freesolo API key"
    assert _credential_kind(url_safe) == "a Freesolo API key"


def test_every_pkcs1_modulus_length_form_is_recognised(tmp_path):
    """DER states a length in three forms, and only the 2-byte one was matched.

    `openssl rsa -outform DER -traditional` writes `02 82` for 2048-bit and larger keys, `02 81 81`
    for a 1024-bit key and a short-form `02 41` for a 512-bit one, so every key below 2048 bits
    published intact despite being a private key.
    """
    import subprocess

    from flash.env_secrets import credential_in_file

    for bits in (512, 1024, 2048, 4096):
        pem, der = tmp_path / f"{bits}.pem", tmp_path / f"{bits}.der"
        subprocess.run(
            ["openssl", "genrsa", "-out", str(pem), str(bits)], check=True, capture_output=True
        )
        subprocess.run(
            [
                "openssl",
                "rsa",
                "-in",
                str(pem),
                "-outform",
                "DER",
                "-traditional",
                "-out",
                str(der),
            ],
            check=True,
            capture_output=True,
        )
        assert credential_in_file(der) == "a private key", bits

    # an ordinary binary that happens to carry the version prefix is not a key
    plain = tmp_path / "weights.bin"
    plain.write_bytes(b"\x30\x82\x00\x10\x02\x01\x00\x02\x00" + b"\x41" * 4096)
    assert credential_in_file(plain) is None


def test_an_openpgp_packet_shorter_than_its_fields_is_not_a_key(tmp_path):
    """The declared body length must reach the version, timestamp and algorithm actually read.

    Ignoring it read those fields from BEYOND the packet, so `c5 01 04 00 00 00 00 01` -- an
    ordinary binary declaring a one-byte body -- was refused as a private key.
    """
    from flash.env_openpgp import _is_openpgp_secret_key
    from flash.env_secrets import credential_in_file

    assert not _is_openpgp_secret_key(bytes.fromhex("c501040000000001"))
    short = tmp_path / "short.bin"
    short.write_bytes(bytes.fromhex("c501040000000001") + b"\x00" * 200)
    assert credential_in_file(short) is None

    # a packet whose declared length DOES cover its fields is still a key
    real = bytes([0xC5, 20]) + b"\x04" + b"\x6a\x7e\x24\x0a" + b"\x01" + b"\x00" * 16
    assert _is_openpgp_secret_key(real)


def test_a_prepended_stub_does_not_hide_a_zips_real_member_count(tmp_path):
    """A self-extracting zip shifts every recorded offset, including the directory's.

    Reading the stored offset literally landed the member-count walk inside the stub, which is not
    a directory record, so the walk gave up and the (attacker-controlled) count field in the end
    record was trusted instead. `ZipFile` compensates for the prepended bytes and still
    materializes every entry, so a 102-byte stub restored the exhaustion the bound exists to
    prevent: 500 entries reported as one.
    """
    import struct
    import zipfile

    from flash.env_formats import _ZIP_END_RECORD, _zip_member_count

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for index in range(500):
            archive.writestr(f"m{index}.txt", f"payload {index}")
    raw = bytearray(buf.getvalue())
    end = raw.rfind(_ZIP_END_RECORD)
    raw[end + 8 : end + 10] = struct.pack("<H", 1)
    raw[end + 10 : end + 12] = struct.pack("<H", 1)
    forged = bytes(raw)

    # the honest archive counts correctly, with and without a stub
    assert _zip_member_count(bytes(buf.getvalue()), 100) > 100
    stub = b"MZ" + b"\x90" * 100
    assert _zip_member_count(stub + bytes(buf.getvalue()), 100) > 100
    # and the forged count is not believed just because a stub moved the directory
    assert _zip_member_count(stub + forged, 100) > 100
    assert len(zipfile.ZipFile(io.BytesIO(stub + forged)).infolist()) == 500


def test_a_seven_zip_archive_is_refused_rather_than_published(tmp_path):
    """7-Zip is as opaque to the stdlib as zstd, so its member bytes are unverifiable."""
    from flash.env_secrets import _Unscannable, credential_in_file

    archive = tmp_path / "shard.7z"
    archive.write_bytes(b"7z\xbc\xaf\x27\x1c" + b"\x00\x04" + bytes(range(256)) * 4)
    with pytest.raises(_Unscannable, match="7-zip"):
        credential_in_file(archive)


def test_text_beginning_with_rar_is_not_treated_as_an_archive(tmp_path):
    """Four printable characters are prose. A real signature carries its version bytes."""
    from flash.env_secrets import _Unscannable, credential_in_file

    readme = tmp_path / "README.md"
    readme.write_bytes(b"Rar! archives are not supported here; use tar instead.\n" * 4)
    assert credential_in_file(readme) is None

    real = tmp_path / "shard.rar"
    real.write_bytes(b"Rar!\x1a\x07\x00" + b"\x00" * 512)
    with pytest.raises(_Unscannable, match="rar"):
        credential_in_file(real)


def test_a_skippable_frame_does_not_hide_the_compressed_frame_behind_it(tmp_path):
    """zstd and LZ4 both allow a metadata envelope before the real frame.

    A head-only format check saw the skippable magic, matched neither the expandable nor the
    unexpandable list, and passed the compressed frame behind it through as ordinary content --
    where its credential is visible to nothing.
    """
    import struct

    from flash.env_secrets import _Unscannable, credential_in_file

    opaque = bytes((index * 7 + 13) % 251 for index in range(4096))
    for magic, label in ((b"\x28\xb5\x2f\xfd", "zstd"), (b"\x04\x22\x4d\x18", "lz4")):
        skippable = b"\x50\x2a\x4d\x18" + struct.pack("<I", 16) + b"\x00" * 16
        shard = tmp_path / f"shard-{label}.bin"
        shard.write_bytes(skippable + magic + opaque)
        with pytest.raises(_Unscannable, match=label):
            credential_in_file(shard)


def test_a_zip_member_using_unsupported_compression_is_refused(tmp_path):
    """`archive.open` raises NotImplementedError for Deflate64, which read as a clean member."""
    import struct
    import zipfile

    from flash.env_secrets import _Unscannable, credential_in_file

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("secret.txt", f"fslo_{_FAKE_KEY_BODY}")
    raw = bytearray(buf.getvalue())
    for signature, offset in ((b"PK\x03\x04", 8), (b"PK\x01\x02", 10)):
        at = raw.find(signature)
        while at >= 0:
            raw[at + offset : at + offset + 2] = struct.pack("<H", 9)
            at = raw.find(signature, at + 1)

    shard = tmp_path / "shard.zip"
    shard.write_bytes(bytes(raw))
    with pytest.raises(_Unscannable, match="compressed in a way this check cannot read"):
        credential_in_file(shard)


def test_a_dsa_private_key_in_der_is_recognised(tmp_path):
    """DSA's AlgorithmIdentifier is a SEQUENCE, so the 1-byte-length branches did not cover it."""
    from flash.env_secrets import credential_in_file

    # PKCS#8 PrivateKeyInfo: version 0, then AlgorithmIdentifier { OID 1.2.840.10040.4.1, params }
    body = (
        b"\x30\x82\x01\x5a\x02\x01\x00\x30\x82\x01\x33"
        b"\x06\x07\x2a\x86\x48\xce\x38\x04\x01" + b"\x00" * 300
    )
    key = tmp_path / "dsa.der"
    key.write_bytes(body)
    assert credential_in_file(key) == "a private key"


def test_a_private_key_stored_as_a_jwk_is_recognised(tmp_path):
    """A JWK carries neither a PEM header nor DER structure, so every key branch passed it."""
    import base64
    import json

    from flash.env_secrets import credential_in_file

    def value(seed):
        raw = bytes((index * seed + 11) % 251 for index in range(32))
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    private = {"kty": "EC", "crv": "P-256", "x": value(3), "y": value(5), "d": value(7)}
    key = tmp_path / "key.jwk"
    key.write_text(json.dumps(private, indent=2))
    assert credential_in_file(key) == "a private key"

    # the public half of the same key is meant to be shared, and must not be refused
    public = tmp_path / "pub.jwk"
    public.write_text(json.dumps({k: v for k, v in private.items() if k != "d"}, indent=2))
    assert credential_in_file(public) is None

    # nor is ordinary JSON that happens to carry a short `d`
    ordinary = tmp_path / "row.json"
    ordinary.write_text(json.dumps({"kty": "EC", "d": "short"}))
    assert credential_in_file(ordinary) is None


def test_a_directory_the_scan_cannot_enter_is_refused(tmp_path):
    """`os.walk` swallows descent errors, so an unreadable directory hid its contents.

    The same tree with the directory readable is refused, so passing was purely a function of what
    could be opened -- a credential inside a mode-000 directory published intact.
    """
    from flash.env_secrets import reject_credential_bearing_package

    package = tmp_path / "pkg"
    hidden = package / "sub"
    hidden.mkdir(parents=True)
    (hidden / "key.txt").write_text(f"fslo_{_FAKE_KEY_BODY}")

    # readable: refused by content, which is the behaviour the unreadable case must not escape
    with pytest.raises(ValueError, match="Freesolo API key"):
        reject_credential_bearing_package(package, display={})

    hidden.chmod(0o000)
    try:
        with pytest.raises(ValueError, match="could not be read"):
            reject_credential_bearing_package(package, display={})
    finally:
        hidden.chmod(0o755)


def test_a_frame_prelude_too_long_to_read_past_refuses(tmp_path):
    """A skippable frame bigger than the lookahead must not read as "not compressed at all".

    zstd and LZ4 both allow a metadata envelope before the real frame, and the format check reads a
    bounded prefix. A frame declaring more payload than that prefix holds sliced the lookahead to
    empty, which matched no magic, so the compressed frame behind it was treated as ordinary
    content and a credential inside it published. The size is attacker-chosen, so the bound cannot
    be raised out of the problem; running out has to refuse instead.
    """
    from flash.env_secrets import _Unscannable, credential_in_file

    zstd = bytes([0x28, 0xB5, 0x2F, 0xFD])
    for payload in (8, 70 << 10):
        frame = (
            bytes([0x50, 0x2A, 0x4D, 0x18]) + (payload).to_bytes(4, "little") + b"\x00" * payload
        )
        stream = tmp_path / f"prelude-{payload}.zst"
        stream.write_bytes(frame + zstd + b"\x11" * 256)
        with pytest.raises(_Unscannable):
            credential_in_file(stream)

    # a file that merely opens with the skippable magic and is not compressed still publishes
    ordinary = tmp_path / "not-compressed.bin"
    ordinary.write_bytes(
        bytes([0x50, 0x2A, 0x4D, 0x18]) + (4).to_bytes(4, "little") + b"abcd" + b"plain\n" * 20
    )
    assert credential_in_file(ordinary) is None


def test_a_private_jwk_is_found_however_far_its_members_sit_apart(tmp_path):
    """JWK members may appear in any order with arbitrary extension members between them.

    Requiring `kty` and the private member within a window of each other was wrong twice over: 5
    KiB of metadata between them published a real RSA key, and the span that bridged the gap
    backtracked over every position of a near-match, which cost 4.2 seconds per MiB of
    `"kty":"RSA",` repeated -- roughly 18 minutes for a permitted 256 MiB package.
    """
    from flash.env_secrets import credential_in_file

    scalar = "a1B2c3D4" * 8
    for name, body in (
        ("compact.jwk", f'{{"kty":"RSA","d":"{scalar}"}}'),
        ("padded.jwk", f'{{"kty":"RSA","note":"{"x" * 5000}","d":"{scalar}"}}'),
        ("reordered.jwk", f'{{"d":"{scalar}","note":"{"y" * 5000}","kty":"EC"}}'),
    ):
        key = tmp_path / name
        key.write_text(body)
        assert credential_in_file(key) == "a private key", name

    # a PUBLIC jwk carries no private member and must still publish, however large
    public = tmp_path / "public.jwk"
    public.write_text(f'{{"kty":"RSA","n":"{"n" * 5000}","e":"AQAB"}}')
    assert credential_in_file(public) is None


def test_a_dh_private_key_in_der_is_detected(tmp_path):
    """`dhKeyAgreement` is a PKCS#8 algorithm like any other, and its key is just as private.

    Enumerating RSA, the RFC 8410 curves, EC and DSA left `1.2.840.113549.1.3.1` uncovered, so a
    key `openssl pkey -check` accepts published intact.
    """
    from flash.env_secrets import credential_in_file

    # PrivateKeyInfo: version 0, then the dhKeyAgreement AlgorithmIdentifier
    der = tmp_path / "dh.der"
    der.write_bytes(
        b"\x30\x82\x01\x21\x02\x01\x00\x30\x81\x95\x06\x09"
        b"\x2a\x86\x48\x86\xf7\x0d\x01\x03\x01" + b"\x00" * 64
    )
    assert credential_in_file(der) == "a private key"


def test_a_filename_that_is_itself_a_private_key_is_never_echoed(tmp_path):
    """The refusal must not print the key it is refusing.

    A compact private JWK fits in a 129-character filename. Masking covered only the token and
    assignment patterns, so the key structures were echoed verbatim -- the refusal printed a
    complete Ed25519 private scalar to the terminal and anything collecting its output.
    """
    from flash.env_secrets import _redacted, credential_in_name

    scalar = "ntpBr8-RhhOkeezY5aeBh2wrN4xaQ-CIq0s6j_A26FQ"
    name = f'keys/{{"crv":"Ed25519","d":"{scalar}","kty":"OKP"}}'
    assert credential_in_name(name) == "a private key"
    redacted = _redacted(name)
    assert scalar not in redacted
    assert "keys/" in redacted  # the directory survives so the author can find it

    # ordinary names are untouched, and a token keeps its issuer prefix for the same reason
    assert _redacted("data/train.jsonl") == "data/train.jsonl"
    assert _redacted(f"cache-fslo_{_FAKE_KEY_BODY}.json") == "cache-fslo_***.json"


def test_a_zip64_member_count_cannot_be_forged_downward(tmp_path):
    """A zip64 archive states its total in 64 bits, so forging it needs no `0xffff` sentinel.

    The directory walk that defeats a forged classic count was computed and then discarded on the
    zip64 branch -- exactly the archives large enough for the bound to matter. A 70,000-entry
    archive patched to claim one member reported one while `ZipFile` still materialized all of
    them, restoring the memory cost the pre-check exists to avoid.
    """
    import struct
    import zipfile

    from flash.env_formats import _zip_member_count

    archive = tmp_path / "many.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED, allowZip64=True) as writing:
        for index in range(70_000):
            writing.writestr(f"f{index}", b"")
    raw = bytearray(archive.read_bytes())
    record = raw.rfind(b"PK\x06\x06")
    raw[record + 24 : record + 32] = struct.pack("<Q", 1)
    raw[record + 32 : record + 40] = struct.pack("<Q", 1)
    archive.write_bytes(bytes(raw))

    assert _zip_member_count(archive, 100) > 100

    # an ordinary zip64 archive under the limit still reports its real, small count
    small = tmp_path / "few.zip"
    with zipfile.ZipFile(small, "w", zipfile.ZIP_STORED, allowZip64=True) as writing:
        writing.writestr("a.txt", b"hello")
        writing.writestr("b.txt", b"world")
    assert _zip_member_count(small, 100) == 2


def test_the_package_deadline_bounds_the_raw_file_scan(tmp_path):
    """The size limit bounds bytes; only the deadline bounds time.

    Matching cost is not uniform per byte, so a large file of adversarial near-matches held a
    worker far longer than its size suggested while the budget was applied only to decompression.
    """
    import contextlib
    import time

    from flash.env_secrets import _Unscannable, credential_in_file

    adversarial = tmp_path / "slow.json"
    adversarial.write_bytes(b'"kty":"RSA",' * ((4 << 20) // 12))
    started = time.monotonic()
    # refusing is the fail-closed outcome; returning None inside the budget is also fine
    with contextlib.suppress(_Unscannable):
        credential_in_file(adversarial, deadline=time.monotonic() + 0.5)
    assert time.monotonic() - started < 30

    # a real key is still found, and an ordinary file still publishes
    key = tmp_path / "key.jwk"
    key.write_text('{"kty":"RSA","d":"' + "a1B2c3D4" * 8 + '"}')
    assert credential_in_file(key) == "a private key"
    ordinary = tmp_path / "rows.jsonl"
    ordinary.write_bytes(b'{"text":"ordinary training row"}\n' * 20_000)
    assert credential_in_file(ordinary) is None


def test_a_jwk_whose_markers_span_a_scan_chunk_is_still_found(tmp_path):
    """The two JWK markers are order-independent and window-free -- within one buffer.

    A chunked scan re-imposed a window between them at the chunk boundary. A JWK whose `kty` fell
    in the first chunk and whose `d` fell in the second matched neither half, so a real RSA key
    over the chunk size published clean. The window between them is unbounded in principle: JWK
    members may sit any amount of extension metadata apart.
    """
    import json

    from flash.env_secrets import _SCAN_CHUNK_BYTES, credential_in_file

    spread = tmp_path / "big.jwk"
    spread.write_text(
        json.dumps({"kty": "RSA", "ext": "x" * (_SCAN_CHUNK_BYTES + 4096), "d": "a1B2c3D4" * 8})
    )
    assert credential_in_file(spread) == "a private key"

    # the `kty` is what makes it a key: the same private member without one is ordinary JSON
    ordinary = tmp_path / "rows.json"
    ordinary.write_text(json.dumps({"note": "x" * (_SCAN_CHUNK_BYTES + 4096), "d": "a1B2c3D4" * 8}))
    assert credential_in_file(ordinary) is None


def test_a_symmetric_jwk_holds_its_secret_in_k_not_d(tmp_path):
    """An `oct` JWK has no `d` at all -- its whole secret is `k`.

    `oct` was named among the accepted key types while the private-member pattern listed only the
    asymmetric members, so the one key type where the secret IS the file passed as clean. That is
    what an HMAC signing key or an `A256GCM` content key exports as.
    """
    from flash.env_secrets import credential_in_file

    symmetric = tmp_path / "hmac.jwk"
    symmetric.write_text('{"kty":"oct","k":"BwcHBwcHBwcHBwcHBwcHBwcHBwcHBwcHBwcHBwcHBwc"}')
    assert credential_in_file(symmetric) == "a private key"

    public = tmp_path / "public.jwk"
    public.write_text(
        '{"kty":"RSA","n":"0vx7agoebGcQSuuPiLJXZptN9nndrQmbXEps2aiAFbWhM64","e":"AQAB"}'
    )
    assert credential_in_file(public) is None


def test_a_short_credential_is_found_in_its_wide_encoding_too(tmp_path):
    """The wide-run floor must sit at or below the shortest credential any pattern admits.

    At 24 it sat above three of them, so a `pit_` (from 20 characters), `fslo_` (21) or `hf_` (23)
    key was detected as ASCII and missed in its UTF-16 form -- the encoding the narrowing exists to
    cover. A run has to hold the whole credential for the narrowed text to match.
    """
    from flash.env_secrets import credential_in_file

    for name, text in (
        ("pit", "pit_R08qzjI6GKFSufrd"),
        ("fslo", "fslo_WJKY40uvSwMFLZDe"),
        ("hf", "hf_S2qHx6kwXoIIXGvOoNZY"),
    ):
        wide = tmp_path / f"{name}.ps1"
        wide.write_bytes(f"$env:KEY = '{text}'".encode("utf-16-le"))
        assert credential_in_file(wide) is not None, f"{name} missed in UTF-16"

    # the padding column is what identifies wide text: an ELF still narrows into nothing
    binary = tmp_path / "binary.so"
    binary.write_bytes(b"\x7fELF\x02\x01\x01\x00" + b"fos\x00loop_redirections\x00PS3\x00" * 40)
    assert credential_in_file(binary) is None


def test_a_credential_wrapped_in_the_url_safe_alphabet_is_joined(tmp_path):
    """Line-joining accepted only `+/` while the run pattern accepted `-_` as well.

    A url-safe base64 blob was therefore never recognised as wrapped, its lines were left unjoined,
    and a credential straddling a break decoded into neither side -- the exact bypass joining
    exists to close, reachable just by encoding with the other alphabet.
    """
    import base64

    from flash.env_secrets import credential_in_file

    raw = b"\xfb\xef\xff" * 10 + b"fslo_AbCdEf0123456789AbCdEf" + b"\xfe\xff\xef" * 3
    encoded = base64.urlsafe_b64encode(raw)
    assert b"-" in encoded or b"_" in encoded, "fixture no longer exercises the url-safe alphabet"
    wrapped = tmp_path / "wrapped.txt"
    wrapped.write_bytes(b"\n".join(encoded[at : at + 64] for at in range(0, len(encoded), 64)))
    assert credential_in_file(wrapped) == "a Freesolo API key"


def test_a_yaml_block_scalar_value_is_read_as_an_assignment(tmp_path):
    """YAML may put an assigned value on the FOLLOWING lines rather than after the colon.

    `KEY: |` (literal) and `KEY: >-` (folded) are the commonest multi-line forms, and a `\\s*`
    between the colon and the body does not cover the indicator characters -- so a key written that
    way in a `secrets.yaml` or a Helm values file matched nothing and published.
    """
    from flash.env_secrets import credential_in_file

    # AWS's own published example value, so the fixture is unmistakably not a live key
    body = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    for name, text in (
        ("literal", f"AWS_SECRET_ACCESS_KEY: |\n  {body}\n"),
        ("folded", f"AWS_SECRET_ACCESS_KEY: >-\n  {body}\n"),
        ("indented", f"AWS_SECRET_ACCESS_KEY: |2\n  {body}\n"),
    ):
        written = tmp_path / f"{name}.yaml"
        written.write_text(text)
        assert credential_in_file(written) == "an AWS secret access key", name

    # prose under the same key is still not a credential
    prose = tmp_path / "readme.yaml"
    prose.write_text("AWS_SECRET_ACCESS_KEY: read it from the vault at deploy time\n")
    assert credential_in_file(prose) is None


def test_a_raw_zlib_stream_is_expanded_rather_than_scanned_as_bytes(tmp_path):
    """A bare zlib stream holds its credential nowhere a pattern can see, exactly like a gzip.

    The container list was magic-based and zlib has no fixed magic -- its two header bytes are
    validated by a divisibility rule -- so the compressed bytes were scanned as if they were
    content and the stream published intact. `zlib.compress` writes this, as do `.zz` shards and
    many application caches.
    """
    import zlib

    from flash.env_secrets import credential_in_file

    compressed = tmp_path / "shard.zz"
    compressed.write_bytes(zlib.compress(b"FREESOLO_API_KEY=fslo_AbCdEf0123456789AbCdEf\n"))
    assert credential_in_file(compressed) == "a Freesolo API key"

    clean = tmp_path / "clean.zz"
    clean.write_bytes(zlib.compress(b"ordinary training rows, nothing issued here\n" * 50))
    assert credential_in_file(clean) is None


def test_a_self_extracting_rar_or_7z_is_refused_behind_its_stub(tmp_path):
    """RAR and 7-Zip both ship self-extracting archives: a stub, then the signature, then the body.

    A head-anchored magic test saw the executable stub, matched nothing, and scanned the opaque
    compressed bytes as if they were content -- the same bypass the zip stub handling closes,
    reachable with `rar a -sfx` or 7-Zip's `-sfx` switch.
    """
    import contextlib
    import os

    from flash.env_secrets import _Unscannable, credential_in_file

    for name, signature in (("rar", b"Rar!\x1a\x07\x01\x00"), ("7z", b"7z\xbc\xaf\x27\x1c")):
        packed = tmp_path / f"{name}.exe"
        packed.write_bytes(b"MZ" + b"\x00" * 4094 + signature + os.urandom(64))
        with pytest.raises(_Unscannable):
            credential_in_file(packed)

    # the signature bytes are constrained enough that ordinary content does not trip them
    ordinary = tmp_path / "notes.md"
    ordinary.write_text("Rar! archives and 7z archives are both unsupported by this scan.\n")
    with contextlib.suppress(_Unscannable):
        assert credential_in_file(ordinary) is None


def test_a_pbes1_encrypted_private_key_is_detected(tmp_path):
    """`1.2.840.113549.1.5` is the whole password-based-encryption arc, not just PBES2's `13`.

    Naming only `05 0d` let every PBES1 variant through: `openssl pkcs8 -topk8 -v1 PBE-SHA1-DES`
    (and the MD5-DES and RC2-64 forms) writes `05 03`, `05 0a` or `05 0b` and passed as clean. A
    passphrase is not much protection for a key in a public repository.
    """
    import subprocess

    from flash.env_secrets import credential_in_file

    plain = tmp_path / "rsa.pem"
    generated = subprocess.run(
        ["openssl", "genrsa", "-out", str(plain), "2048"], capture_output=True
    )
    if generated.returncode != 0:
        pytest.skip("openssl is unavailable")
    found = 0
    for algorithm in ("PBE-MD5-DES", "PBE-SHA1-DES", "PBE-SHA1-RC2-64"):
        encrypted = tmp_path / f"{algorithm}.der"
        # `-provider legacy` because OpenSSL 3 moved these algorithms out of the default provider
        options = f"-topk8 -v1 {algorithm} -outform DER -passout pass:x"
        providers = "-provider legacy -provider default"
        written = subprocess.run(
            [
                "openssl",
                "pkcs8",
                "-in",
                str(plain),
                "-out",
                str(encrypted),
                *options.split(),
                *providers.split(),
            ],
            capture_output=True,
        )
        if written.returncode != 0:
            continue  # this build cannot write the legacy algorithm; the others still prove it
        found += 1
        assert credential_in_file(encrypted) == "a private key", algorithm
    if not found:
        pytest.skip("this openssl build cannot write PBES1")


def test_a_putty_private_key_file_is_detected(tmp_path):
    """PuTTY's own key format is neither PEM nor DER, and `.ppk` is not a filtered name.

    No `-----BEGIN` header and no ASN.1, so every structure the scan knows passed a complete
    unencrypted private key as clean.
    """
    import base64
    import os

    from flash.env_secrets import credential_in_file

    key = tmp_path / "id.ppk"
    key.write_bytes(
        b"PuTTY-User-Key-File-3: ssh-ed25519\nEncryption: none\nComment: deploy\n"
        b"Public-Lines: 2\nAAAAC3NzaC1lZDI1NTE5AAAAIN\nabcdef\n"
        b"Private-Lines: 1\n" + base64.b64encode(os.urandom(48)) + b"\nPrivate-MAC: 00\n"
    )
    assert credential_in_file(key) == "a private key"

    # the header alone is prose about a key, exactly as for the PEM pattern
    prose = tmp_path / "guide.md"
    prose.write_text("A PuTTY-User-Key-File-3: header means a private key. Never commit one.\n")
    assert credential_in_file(prose) is None


def test_the_name_that_gets_written_is_scanned_not_only_the_one_supplied(tmp_path):
    """Normalization folds separators, so it can turn a passing name into a credential.

    `fslo_abcd1!efgh!ijkl!mnop` carries no key body across the `!`s and passed the raw scan, then
    normalized to `fslo_abcd1-efgh-ijkl-mnop` -- which IS a Freesolo key, and which is the string
    committed into the hub path and the commit message. Scanning only the supplied name checked a
    string the publish never writes.
    """
    from flash.env_secrets import credential_in_name
    from flash.server.domain.envs import _sanitize_name

    folded = "fslo_abcd1!efgh!ijkl!mnop"
    assert credential_in_name(folded) is None, "fixture no longer exercises the folding"
    assert credential_in_name(_sanitize_name(folded)) == "a Freesolo API key"

    ordinary = "my-training-environment"
    assert credential_in_name(ordinary) is None
    assert credential_in_name(_sanitize_name(ordinary)) is None


def test_a_member_count_cannot_be_forged_behind_a_stub_on_a_zip64_archive(tmp_path):
    """A self-extracting zip64 defeats both offset candidates at once.

    The stub makes every recorded offset short, and the zip64 record and locator between the
    directory and the classic end record make the computed shift overshoot by their combined 76
    bytes. Neither candidate landed on the directory, so the walk gave up and the forged count was
    trusted: a 70,000-entry archive reported one member while `ZipFile` still materialized all of
    them, restoring the memory cost the pre-check exists to avoid.
    """
    import struct
    import zipfile

    from flash.env_formats import _zip_member_count

    body = tmp_path / "body.zip"
    with zipfile.ZipFile(body, "w", zipfile.ZIP_STORED, allowZip64=True) as writing:
        for index in range(70_000):
            writing.writestr(f"f{index}", b"")
    raw = bytearray(body.read_bytes())
    classic = raw.rfind(b"PK\x05\x06")
    struct.pack_into("<HH", raw, classic + 8, 1, 1)
    zip64 = raw.rfind(b"PK\x06\x06")
    struct.pack_into("<QQ", raw, zip64 + 24, 1, 1)
    packed = tmp_path / "sfx.exe"
    packed.write_bytes(b"MZ" + b"\x00" * 100 + bytes(raw))

    assert _zip_member_count(packed, 100) > 100

    # an ordinary zip64 archive under the limit still reports its real, small count
    small = tmp_path / "few.zip"
    with zipfile.ZipFile(small, "w", zipfile.ZIP_STORED, allowZip64=True) as writing:
        writing.writestr("a.txt", b"hello")
        writing.writestr("b.txt", b"world")
    assert _zip_member_count(small, 100) == 2


def test_a_jwk_is_found_with_its_markers_in_either_order_across_chunks(tmp_path):
    """JSON permits any member order, so the private member may precede the `kty`.

    Tracking only whether a `kty` had gone past was one-directional: a JWK written
    `{"d": ..., <over a chunk of metadata>, "kty": "RSA"}` had its private member leave the window
    before the `kty` arrived, and a real RSA key published. Both halves are remembered now.
    """
    import json

    from flash.env_secrets import _SCAN_CHUNK_BYTES, credential_in_file

    filler = "x" * (_SCAN_CHUNK_BYTES + 4096)
    body = "a1B2c3D4" * 8
    reversed_order = tmp_path / "reversed.jwk"
    reversed_order.write_text(json.dumps({"d": body, "extension": filler, "kty": "RSA"}))
    assert credential_in_file(reversed_order) == "a private key"

    forward = tmp_path / "forward.jwk"
    forward.write_text(json.dumps({"kty": "RSA", "extension": filler, "d": body}))
    assert credential_in_file(forward) == "a private key"

    # neither half alone is a key, however far the file extends
    lone = tmp_path / "lone.json"
    lone.write_text(json.dumps({"note": filler, "d": body}))
    assert credential_in_file(lone) is None


def test_a_putty_key_is_found_however_large_its_public_section(tmp_path):
    """An RSA-4096 public blob base64-encodes to ~716 characters.

    A fixed proximity cap between the header and `Private-Lines` therefore excluded exactly the
    larger keys: the private payload is SSH mpints rather than DER, so nothing downstream caught
    it either and the complete `.ppk` published.
    """
    import base64
    import os

    from flash.env_secrets import credential_in_file

    public = base64.b64encode(os.urandom(535)).decode()
    lines = "\n".join(public[at : at + 64] for at in range(0, len(public), 64))
    key = tmp_path / "rsa4096.ppk"
    key.write_text(
        "PuTTY-User-Key-File-3: ssh-rsa\nEncryption: none\nComment: rsa-key\n"
        f"Public-Lines: {len(public) // 64 + 1}\n{lines}\n"
        f"Private-Lines: 1\n{base64.b64encode(os.urandom(48)).decode()}\nPrivate-MAC: 00\n"
    )
    assert len(public) > 512, "fixture no longer exceeds the removed cap"
    assert credential_in_file(key) == "a private key"

    # both markers are still required: either alone is not a key
    header_only = tmp_path / "header.md"
    header_only.write_text("PuTTY-User-Key-File-3: is the header.\n" + "prose\n" * 200)
    assert credential_in_file(header_only) is None
    body_only = tmp_path / "body.txt"
    body_only.write_text(f"Private-Lines: 1\n{base64.b64encode(os.urandom(48)).decode()}\n")
    assert credential_in_file(body_only) is None


def test_a_multi_prime_rsa_private_key_is_detected(tmp_path):
    """RFC 8017 defines `two-prime(0)` and `multi(1)`, and only version 0 was accepted.

    A real three-prime key from `openssl genrsa -primes 3` begins `02 01 01`, passes
    `openssl rsa -check`, and published its private factors intact.
    """
    import subprocess

    from flash.env_secrets import credential_in_file

    plain = tmp_path / "multi.pem"
    generated = subprocess.run(
        ["openssl", "genrsa", "-primes", "3", "-out", str(plain), "1024"], capture_output=True
    )
    if generated.returncode != 0:
        pytest.skip("this openssl build cannot generate multi-prime keys")
    der = tmp_path / "multi.der"
    subprocess.run(
        ["openssl", "rsa", "-in", str(plain), "-outform", "DER", "-traditional", "-out", str(der)],
        capture_output=True,
    )
    assert der.read_bytes()[4:7] == b"\x02\x01\x01", "fixture is not the multi-prime version"
    assert credential_in_file(der) == "a private key"


def test_unpadded_base64_is_padded_rather_than_truncated(tmp_path):
    """Trimming a run to a whole quartet discarded up to two decoded bytes off the END.

    That is enough to take a token below its pattern's minimum length, so unpadded base64url of a
    20-character `pit_` key or a 23-character `hf_` token decoded to something matching nothing.
    Unpadded output is what `rstrip("=")`, a JWT segment and most token-in-a-URL encodings emit.
    """
    import base64

    from flash.env_secrets import credential_in_file

    for name, token in (("pit", "pit_AbCdEf0123456789"), ("hf", "hf_AbCdEf0123456789AbCd")):
        encoded = base64.urlsafe_b64encode(token.encode()).decode()
        assert encoded.endswith("="), "fixture no longer exercises omitted padding"
        written = tmp_path / f"{name}.txt"
        written.write_text(encoded.rstrip("="))
        assert credential_in_file(written) is not None, name


def test_a_zlib_stream_needing_a_preset_dictionary_is_refused(tmp_path):
    """FDICT means the stream was compressed against a dictionary the file does not carry.

    Decompression then raises, and treating that as "not zlib after all" let the opaque bytes fall
    through to the literal scan and publish. Refusing is the honest answer: from here the content
    cannot be inspected at all.
    """
    import zlib

    from flash.env_secrets import _Unscannable, credential_in_file

    dictionary = b"FREESOLO_API_KEY="
    compressor = zlib.compressobj(zdict=dictionary)
    blob = compressor.compress(b"FREESOLO_API_KEY=fslo_AbCdEf0123456789AbCdEf\n")
    blob += compressor.flush()
    assert blob[1] & 0x20, "fixture does not set FDICT"
    stream = tmp_path / "dict.zz"
    stream.write_bytes(blob)
    with pytest.raises(_Unscannable):
        credential_in_file(stream)

    # an ordinary dictionary-free stream is still expanded and scanned rather than refused
    plain = tmp_path / "plain.zz"
    plain.write_bytes(zlib.compress(b"ordinary rows, nothing issued\n" * 50))
    assert credential_in_file(plain) is None


def test_a_triple_quoted_assignment_is_read_as_an_assignment(tmp_path):
    """A single optional quote consumed only one of the three in `\"\"\"` or `'''`.

    That left a quote sitting where the body had to begin, so an ordinary Python or TOML multiline
    assignment matched nothing while its single-quoted equivalent was caught.
    """
    from flash.env_secrets import credential_in_file

    body = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    for name, text in (
        ("double", f'AWS_SECRET_ACCESS_KEY = """{body}"""'),
        ("single", f"AWS_SECRET_ACCESS_KEY = '''{body}'''"),
        ("plain", f'AWS_SECRET_ACCESS_KEY = "{body}"'),
    ):
        written = tmp_path / f"{name}.py"
        written.write_text(text)
        assert credential_in_file(written) == "an AWS secret access key", name

    wandb = tmp_path / "conf.py"
    wandb.write_text('WANDB_API_KEY = """' + "a1b2c3d4e5" * 4 + '"""')
    assert credential_in_file(wandb) == "a Weights & Biases API key"


def test_a_yaml_block_header_is_read_in_either_indicator_order(tmp_path):
    """YAML 1.2 lets a block header carry its indentation and chomping indicators in either order.

    `|2-` and `|-2` are the same scalar, but the header pattern admitted only sign-then-digit, so
    the `|` was left unconsumed, the body failed to match, and the key published. A writer that
    emits an explicit indentation indicator -- ruamel, several Helm chart generators -- naturally
    puts the digit first.
    """
    from flash.env_secrets import credential_in_file

    body = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    for header in ("|", "|-", "|+", ">-", "|2-", ">2+", "|-2", "|2"):
        written = tmp_path / "secrets.yaml"
        written.write_text(f"AWS_SECRET_ACCESS_KEY: {header}\n  {body}\n")
        assert credential_in_file(written) == "an AWS secret access key", header


def test_a_jwk_member_name_is_matched_through_its_json_escape(tmp_path):
    """JSON says `"\\u0064"` and `"d"` are the same string, and every parser agrees.

    A JWK whose private member is spelled with escapes loads and exports identically, so it is the
    same key -- but a literal-byte pattern saw no `d` at all and published it. Both halves are
    escapable, so the `kty` that identifies the format is covered too.
    """
    import base64
    import json
    import os

    from flash.env_secrets import credential_in_file

    secret = base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip("=")
    # `k` is `k`, `d` is `d`. The hex digits are case-insensitive per RFC 8259, so the
    # capital form is the same name and must be caught too.
    escape_d, escape_kty, escape_dp = "\\u0064", "\\u006bty", "\\u0064p"
    escape_kty_upper = "\\u006Bty"
    for name, text in (
        ("plain", '{"kty":"OKP","crv":"Ed25519","x":"abc","d":"' + secret + '"}'),
        (
            "escaped-d",
            '{"kty":"OKP","crv":"Ed25519","x":"abc","' + escape_d + '":"' + secret + '"}',
        ),
        ("escaped-kty", '{"' + escape_kty + '":"OKP","x":"abc","d":"' + secret + '"}'),
        (
            "escaped-both",
            '{"' + escape_kty_upper + '":"RSA","' + escape_dp + '":"' + secret + '"}',
        ),
    ):
        written = tmp_path / f"{name}.json"
        written.write_text(text)
        assert json.loads(text), name  # the fixture must be legal JSON, or it proves nothing
        assert credential_in_file(written) == "a private key", name

    # a PUBLIC jwk written the same way is still publishable
    public = tmp_path / "public.json"
    public.write_text('{"' + escape_kty + '":"OKP","crv":"Ed25519","x":"' + secret + '"}')
    assert credential_in_file(public) is None


def test_an_sfx_archive_is_refused_however_long_its_stub(tmp_path):
    """The signature of a self-extracting archive sits past any amount of stub.

    Bounding the search to the first 64 KiB only moved the bypass: every real SFX module is larger
    than that -- 7-Zip's smallest is about 150 KiB -- so the signature landed past the window and
    the opaque compressed body behind it was scanned as ordinary content and published.
    """
    import os
    import random

    from flash.env_secrets import _Unscannable, credential_in_file

    for stub_kb in (1, 63, 64, 65, 256):
        packed = tmp_path / f"sfx{stub_kb}.exe"
        packed.write_bytes(os.urandom(stub_kb << 10) + b"7z\xbc\xaf\x27\x1c" + os.urandom(4096))
        with pytest.raises(_Unscannable):
            credential_in_file(packed)

    # an ordinary large binary with no signature anywhere is still publishable
    plain = tmp_path / "weights.bin"
    # seeded for the same reason as the raw-deflate control: a random block can legitimately
    # satisfy a container header, and a check that reddens by luck gets switched off
    plain.write_bytes(random.Random(1).randbytes(256 << 10))
    assert credential_in_file(plain) is None


def test_an_lz4_legacy_frame_is_refused_like_the_modern_one(tmp_path):
    """The LZ4 legacy frame carries its own magic rather than a variant of the modern one.

    `lz4 -l` writes it, and its body is opaque exactly like the modern frame -- so naming only
    `04 22 4d 18` meant a legacy frame was scanned as raw bytes and published intact.
    """
    import os

    from flash.env_secrets import _Unscannable, credential_in_file

    for name, magic in (("legacy", b"\x02\x21\x4c\x18"), ("modern", b"\x04\x22\x4d\x18")):
        packed = tmp_path / f"{name}.lz4"
        packed.write_bytes(magic + os.urandom(4096))
        with pytest.raises(_Unscannable):
            credential_in_file(packed)


def test_an_openpgp_secret_key_is_found_behind_a_marker_packet(tmp_path):
    """A marker packet is a legal no-op RFC 9580 requires implementations to skip.

    GnuPG parses `<marker><secret key>` as the secret key it is, but an offset-zero-only test saw
    the marker, matched nothing, and the remaining binary key material matched no textual or DER
    pattern -- so five prepended bytes published a private key intact.
    """
    import os

    from flash.env_secrets import credential_in_file

    def packet(body_bytes: int) -> bytes:
        body = bytes([4]) + b"\x66\x00\x00\x00" + bytes([1]) + os.urandom(body_bytes)
        if len(body) < 192:
            return bytes([0xC5, len(body)]) + body
        over = len(body) - 192
        return bytes([0xC5, 192 + (over >> 8), over & 0xFF]) + body

    marker = b"\xca\x03PGP"
    for body_bytes in (100, 180, 400):
        for name, data in (
            ("bare", packet(body_bytes)),
            ("marker", marker + packet(body_bytes)),
            ("two-markers", marker * 2 + packet(body_bytes)),
        ):
            written = tmp_path / f"{name}{body_bytes}.gpg"
            written.write_bytes(data)
            assert credential_in_file(written) == "a private key", f"{name}/{body_bytes}"

    # a marker in front of ordinary bytes is not a key
    innocent = tmp_path / "notes.bin"
    innocent.write_bytes(marker + b"just some text about PGP\n" * 20)
    assert credential_in_file(innocent) is None


def test_ordinary_text_is_not_refused_as_a_dictionary_compressed_stream(tmp_path):
    """The zlib header rule is about eleven bits of signal, and `x ` satisfies all of it.

    Turning that heuristic into a terminal refusal meant an ordinary `x = 1` sidecar could not be
    published at all. Decompression cannot make the call -- `zlib.error` is identical for a real
    dictionary-compressed stream and for the text -- so the discriminator is whether the bytes read
    as text at all.
    """
    import zlib

    from flash.env_secrets import _Unscannable, credential_in_file

    for name, text in (("a", "x = 1\n"), ("b", "x  = 1\n"), ("c", "x = 1\nimport os\n" * 40)):
        written = tmp_path / f"{name}.py"
        written.write_text(text)
        assert credential_in_file(written) is None, name

    # a real dictionary-compressed stream is still refused: its content cannot be inspected
    compressor = zlib.compressobj(zdict=b"AWS_SECRET_ACCESS_KEY=")
    blob = compressor.compress(b"AWS_SECRET_ACCESS_KEY=" + b"A" * 40 + b"\n") + compressor.flush()
    assert blob[1] & 0x20, "fixture must actually set FDICT"
    packed = tmp_path / "state.zz"
    packed.write_bytes(blob)
    with pytest.raises(_Unscannable):
        credential_in_file(packed)


def test_the_base64_floor_admits_the_shortest_token_the_patterns_match(tmp_path):
    """The encoded floor is derived from the shortest credential, not chosen.

    `xoxb-` plus its 10-character body is 15 bytes, which encodes to 20 characters -- below a fixed
    24 -- so a minimum-length Slack token in a Kubernetes Secret was never decoded even though the
    same token in plaintext was caught.
    """
    import base64

    from flash.env_base64 import _MIN_BASE64_RUN
    from flash.env_patterns import SHORTEST_TOKEN_BYTES
    from flash.env_secrets import credential_in_file

    token = b"xoxb-AbCdEf0123"
    assert len(token) == SHORTEST_TOKEN_BYTES
    encoded = base64.b64encode(token).decode()
    assert len(encoded) == _MIN_BASE64_RUN

    written = tmp_path / "secret.yaml"
    written.write_text(f"data:\n  token: {encoded}\n")
    assert credential_in_file(written) == "a Slack token"

    # the floor tracks the patterns rather than being written twice
    assert _MIN_BASE64_RUN == -(-SHORTEST_TOKEN_BYTES * 4 // 3)


def test_a_yaml_block_header_may_carry_a_comment(tmp_path):
    """YAML permits `KEY: | # generated`, which is how templating tools annotate injected values.

    The header pattern stopped at the `#`, so the indented body never matched and the key
    published.
    """
    from flash.env_secrets import credential_in_file

    body = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    for header in ("|", "| # injected value", ">- # generated", "|2- # note"):
        written = tmp_path / "values.yaml"
        written.write_text(f"AWS_SECRET_ACCESS_KEY: {header}\n  {body}\n")
        assert credential_in_file(written) == "an AWS secret access key", header


def test_a_jwk_private_value_is_matched_through_its_json_escapes(tmp_path):
    """A base64url scalar is all ASCII, so any character in it may legally be written `\\u00XX`.

    A Node-exported JWK whose `d` begins `"\\u0078..."` is the same key to `JSON.parse` and
    `createPrivateKey`, but a run of plain base64 characters matched nothing and the whole private
    key published.
    """
    import base64
    import json
    import os

    from flash.env_secrets import credential_in_file

    raw = base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip("=")
    value = "x" + raw[1:]
    escape_x = "\\u0078"
    for name, text in (
        ("plain", '{"kty":"OKP","crv":"Ed25519","d":"' + value + '"}'),
        ("escaped", '{"kty":"OKP","crv":"Ed25519","d":"' + escape_x + value[1:] + '"}'),
    ):
        written = tmp_path / f"{name}.json"
        written.write_text(text)
        assert json.loads(text)["d"] == value, name  # both spellings ARE the same key
        assert credential_in_file(written) == "a private key", name


def test_a_pem_body_is_found_at_any_wrap_width(tmp_path):
    """PEM does not fix a wrap width -- RFC 7468 recommends 64 but permits any.

    Requiring 32 contiguous base64 characters meant a key rewrapped at 16 columns matched neither
    this pattern nor the 64/76-column joining, while `openssl pkey -check` still accepts it.
    """
    from flash.env_secrets import credential_in_file

    header = "-----BEGIN PRIVATE KEY-----"
    footer = "-----END PRIVATE KEY-----"
    body = "MC4CAQAwBQYDK2VwBCIEIH" + "AbCdEf0123456789" * 3
    for width in (8, 16, 32, 64):
        wrapped = "\n".join(body[i : i + width] for i in range(0, len(body), width))
        written = tmp_path / f"key{width}.pem"
        written.write_text(f"{header}\n{wrapped}\n{footer}\n")
        assert credential_in_file(written) == "a private key block", width

    # prose that merely NAMES a header is still publishable
    prose = tmp_path / "README.md"
    prose.write_text("If you see -----BEGIN PRIVATE KEY----- in a log, redact it.\n")
    assert credential_in_file(prose) is None


def test_every_concatenated_zlib_record_is_scanned(tmp_path):
    """`decompressobj` stops at the end of one zlib record and hands the rest back as unused data.

    Scanning only the first plaintext meant `zlib.compress(benign) + zlib.compress(secret)`
    published clean, with the credential entirely inside the discarded remainder. Concatenated
    records are what a per-record cache or an appended log writes.
    """
    import zlib

    from flash.env_secrets import credential_in_file

    secret = zlib.compress(b"FREESOLO_API_KEY=fslo_A1b2C3d4E5f6G7h8\n")
    benign = zlib.compress(b"just some ordinary configuration text\n" * 20)
    for name, data in (
        ("first", secret + benign),
        ("second", benign + secret),
        ("third", benign + benign + secret),
    ):
        written = tmp_path / f"{name}.zz"
        written.write_bytes(data)
        assert credential_in_file(written) == "a Freesolo API key", name

    # a stream of only benign records is still publishable
    clean = tmp_path / "clean.zz"
    clean.write_bytes(benign + benign)
    assert credential_in_file(clean) is None


def test_a_record_chain_that_exhausts_the_budget_is_refused(tmp_path):
    """The expansion budget is shared across concatenated records, and running out of it says
    nothing about the records that were never read.

    Returning None there let a chain of two large benign records hide a credential in a third:
    the budget hit zero, the loop reported clean, and the key published. Exhausting a limit is
    unverifiable, and unverifiable is not clean. A chain whose end is a stream this cannot decode
    is the same case -- only a FIRST record that fails means "not zlib after all".
    """
    import zlib

    from flash.env_secrets import _MAX_NESTED_BUFFER_BYTES, _Unscannable, credential_in_file

    secret = zlib.compress(b"FREESOLO_API_KEY=fslo_A1b2C3d4E5f6G7h8\n")
    # exactly the budget, so this record inflates whole -- an OVER-budget record is a different
    # case, caught earlier by `unconsumed_tail` with the same message, and would not prove this
    # branch runs at all.
    exact = zlib.compress(b"x" * _MAX_NESTED_BUFFER_BYTES)

    starved = tmp_path / "starved.zz"
    starved.write_bytes(exact + secret)
    with pytest.raises(_Unscannable, match="too large to inspect"):
        credential_in_file(starved)

    # records inflate, then the tail is a compressed stream this cannot read
    undecodable = tmp_path / "undecodable.zz"
    undecodable.write_bytes(zlib.compress(b"ordinary text\n") + b"\x78\x9c" + b"\xff" * 64)
    with pytest.raises(_Unscannable, match="trailing compressed data"):
        credential_in_file(undecodable)

    # a file that was never zlib is still ordinary content, not a broken chain
    plain = tmp_path / "plain.txt"
    plain.write_bytes(b"just some ordinary configuration text\n")
    assert credential_in_file(plain) is None


def test_a_decoy_directory_header_cannot_defeat_the_member_count(tmp_path):
    """Selecting a directory candidate on its first four bytes made a stub decoy decisive.

    The decoy's walk failed on its second record, the failure was reported as "cannot be walked",
    and the caller fell back to the count in the end record -- so a decoy plus a count patched to 1
    made a real 500-entry archive report one member while `ZipFile` still materialized all 500.
    """
    import io
    import os
    import struct
    import zipfile

    from flash.env_formats import _zip_member_count

    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        for index in range(500):
            archive.writestr(f"m{index}.txt", "x" * 10)
    body = inner.getvalue()
    end = body.rfind(b"PK\x05\x06")
    start = struct.unpack("<I", body[end + 16 : end + 20])[0]

    packed = bytearray(os.urandom(start + 4096) + body)
    decoy = bytearray(b"PK\x01\x02" + os.urandom(42))
    struct.pack_into("<HHH", decoy, 28, 8, 0, 0)
    decoy += b"FILENAME" + os.urandom(64)
    packed[start : start + len(decoy)] = decoy
    forged = packed.rfind(b"PK\x05\x06")
    struct.pack_into("<H", packed, forged + 8, 1)
    struct.pack_into("<H", packed, forged + 10, 1)

    data = bytes(packed)
    assert _zip_member_count(data) == len(zipfile.ZipFile(io.BytesIO(data)).infolist()) == 500


def test_a_member_whose_bytes_live_in_another_volume_is_refused(tmp_path):
    """A split archive's final volume holds the directory while the bytes sit in an earlier one.

    Opening such a member raises, and the bare `continue` treated it as clean -- so both published
    parts returned None while joining the volumes recovered the key.
    """
    import io
    import struct
    import zipfile

    from flash.env_secrets import _Unscannable, credential_in_file

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_LZMA) as archive:
        archive.writestr("secret.txt", "FREESOLO_API_KEY=fslo_A1b2C3d4E5f6G7h8\n")
    raw = buffer.getvalue()
    final = bytearray(raw[raw.find(b"PK\x01\x02") :])
    struct.pack_into("<I", final, final.find(b"PK\x05\x06") + 16, 0)

    written = tmp_path / "a.zip"
    written.write_bytes(bytes(final))
    # the directory still names the member, so this is not an empty archive
    assert zipfile.ZipFile(io.BytesIO(bytes(final))).namelist() == ["secret.txt"]
    with pytest.raises(_Unscannable):
        credential_in_file(written)


def test_a_java_keystore_holding_a_private_key_is_detected(tmp_path):
    """`keytool -genkeypair -storetype JKS` writes a format no other check here understands.

    Not PEM, not DER, not a container -- so a store holding a complete private key returned None,
    and `.jks` is not in the filename exclusions either. A store of only trusted certificates
    carries no secret and must still publish.
    """
    import os
    import struct

    from flash.env_secrets import credential_in_file

    def keystore(*tags: int, magic: bytes = b"\xfe\xed\xfe\xed") -> bytes:
        """A store whose entries carry `tags`, laid out as the format actually declares them."""
        body = b""
        for tag in tags:
            alias = b"a%d" % tag
            body += struct.pack(">I", tag) + struct.pack(">H", len(alias)) + alias
            body += struct.pack(">Q", 1700000000000)
            if tag == 1:  # key bytes, then a one-certificate chain
                body += struct.pack(">I", 1200) + os.urandom(1200)
                body += struct.pack(">I", 1)
                body += struct.pack(">H", 5) + b"X.509" + struct.pack(">I", 64) + os.urandom(64)
            else:  # a trusted certificate: type then der
                body += struct.pack(">H", 5) + b"X.509" + struct.pack(">I", 64) + os.urandom(64)
        return magic + struct.pack(">II", 2, len(tags)) + body + os.urandom(20)

    private = tmp_path / "keystore.jks"
    private.write_bytes(keystore(1))
    assert credential_in_file(private) == "a key store"

    trusted = tmp_path / "truststore.jks"
    trusted.write_bytes(keystore(2))
    assert credential_in_file(trusted) is None

    # A trusted certificate BEFORE the private key, which is what `-importcert` then `-genkeypair`
    # writes. Reading only the first tag published this store's key intact.
    ordered = tmp_path / "ordered.jks"
    ordered.write_bytes(keystore(2, 1))
    assert credential_in_file(ordered) == "a key store"

    # several certificates then the key, so the walk has to stay on entry boundaries to find it
    deep = tmp_path / "deep.jks"
    deep.write_bytes(keystore(2, 2, 2, 1))
    assert credential_in_file(deep) == "a key store"

    # JCEKS is the same layout under a different magic, and was a separate bypass
    jceks = tmp_path / "keystore.jceks"
    jceks.write_bytes(keystore(2, 1, magic=b"\xce\xce\xce\xce"))
    assert credential_in_file(jceks) == "a key store"

    certs_only = tmp_path / "certs.jceks"
    certs_only.write_bytes(keystore(2, 2, magic=b"\xce\xce\xce\xce"))
    assert credential_in_file(certs_only) is None

    # Tag 3 is the JCEKS `SecretKeyEntry` that `keytool -genseckey` writes. Its payload is a
    # symmetric key, which is as much a credential as an asymmetric one -- treating an unknown tag
    # as "not this format" published a real AES store intact.
    secret = tmp_path / "secret.jceks"
    secret.write_bytes(keystore(3, magic=b"\xce\xce\xce\xce"))
    assert credential_in_file(secret) == "a key store"

    # A key past the entry bound is UNREAD, not absent -- the same fail-open the OpenPGP marker
    # bound had. The walk is bounded so a forged count cannot be a scan cost, and exhausting it
    # refuses rather than reporting the store clean.
    from flash.env_formats import _MAX_JKS_ENTRIES
    from flash.env_secrets import _Unscannable

    within = tmp_path / "within.jks"
    within.write_bytes(keystore(*([2] * (_MAX_JKS_ENTRIES - 1)), 1))
    assert credential_in_file(within) == "a key store"

    beyond = tmp_path / "beyond.jks"
    beyond.write_bytes(keystore(*([2] * _MAX_JKS_ENTRIES), 1))
    with pytest.raises(_Unscannable, match="cannot finish walking"):
        credential_in_file(beyond)


def test_the_truststore_every_jdk_ships_still_publishes(tmp_path):
    """The entry bound must sit above real stores, not above a guessed handful.

    `/etc/ssl/certs/java/cacerts` holds 146 trusted certificates and no private key at all. A bound
    chosen as "a real store holds a handful" refused the most ordinary keystore in existence, which
    is a false alarm on a file that carries no secret whatsoever.
    """
    import os
    import struct

    from flash.env_secrets import credential_in_file

    body = b""
    for index in range(200):
        alias = b"cert%d" % index
        body += struct.pack(">I", 2) + struct.pack(">H", len(alias)) + alias
        body += struct.pack(">Q", 1700000000000)
        body += struct.pack(">H", 5) + b"X.509" + struct.pack(">I", 900) + os.urandom(900)
    store = tmp_path / "cacerts"
    store.write_bytes(b"\xfe\xed\xfe\xed" + struct.pack(">II", 2, 200) + body + os.urandom(20))
    assert credential_in_file(store) is None


def test_a_key_behind_a_certificate_larger_than_a_chunk_is_detected(tmp_path):
    """The keystore walk is head-anchored, so a store larger than one chunk was half-read.

    A trusted certificate whose body crosses the chunk boundary made the bounded walk run off the
    end, which reported "not a keystore" -- indistinguishable from bytes that never were one. From
    the second chunk on nothing re-enters the parser, so the private key stored BEHIND that
    certificate published intact. One oversized certificate is all it takes.
    """
    import os
    import struct

    from flash.env_secrets import _SCAN_CHUNK_BYTES, credential_in_file

    def store(cert_bytes: int) -> bytes:
        body = struct.pack(">I", 2) + struct.pack(">H", 4) + b"cert"
        body += struct.pack(">Q", 1700000000000)
        body += struct.pack(">H", 5) + b"X.509"
        body += struct.pack(">I", cert_bytes) + os.urandom(cert_bytes)
        body += struct.pack(">I", 1) + struct.pack(">H", 3) + b"key"
        body += struct.pack(">Q", 1700000000000)
        body += struct.pack(">I", 1200) + os.urandom(1200)
        body += struct.pack(">I", 1)
        body += struct.pack(">H", 5) + b"X.509" + struct.pack(">I", 64) + os.urandom(64)
        return b"\xfe\xed\xfe\xed" + struct.pack(">II", 2, 2) + body + os.urandom(20)

    spanning = tmp_path / "spanning.jks"
    spanning.write_bytes(store(_SCAN_CHUNK_BYTES + 100))
    assert credential_in_file(spanning) == "a key store"

    # the same store with a certificate that fits, which was already detected: the control that
    # proves the assertion above is about the chunk boundary rather than about the layout
    fitting = tmp_path / "fitting.jks"
    fitting.write_bytes(store(64))
    assert credential_in_file(fitting) == "a key store"


def test_an_encrypted_openpgp_message_larger_than_the_head_is_refused(tmp_path):
    """The session packet had to be reachable, and a fixed head could not reach it.

    `gpg --encrypt` writes a public-key session packet carrying the encrypted session key inline,
    which runs to a few hundred bytes for an ordinary RSA key. Reading a fixed 64-byte head meant
    the encrypted-data packet BEHIND it was never seen, the file read as "not encrypted", and the
    ciphertext published. The whole chunk is read now, and a packet longer than that refuses rather
    than reporting the bytes clean.
    """
    import struct

    from flash.env_secrets import _Unscannable, credential_in_file

    def session(body_bytes: int, *, follow: bytes) -> bytes:
        """A v3 PKESK packet of `body_bytes`, then `follow`, laid out as real gpg writes one.

        Old-format tag 1 with a two-byte length -- `0x85` is what `gpg --encrypt` actually emits,
        verified against its output rather than assembled from the spec.
        """
        body = b"\x03" + b"\x00" * 8 + b"\x01" + b"\x00" * (body_bytes - 10)
        return b"\x85" + struct.pack(">H", body_bytes) + body + follow

    # A real 268-byte session packet: the data packet behind it sits far past any fixed head.
    encrypted = tmp_path / "secret.gpg"
    encrypted.write_bytes(session(268, follow=b"\xd2" + b"\x40" + b"\x00" * 64))
    with pytest.raises(_Unscannable, match="encrypted OpenPGP"):
        credential_in_file(encrypted)

    # A session packet declaring a body longer than the bytes that follow it cannot be walked to
    # the packet behind it, and undecided is not clean.
    truncated = tmp_path / "truncated.gpg"
    truncated.write_bytes(session(268, follow=b"")[: 3 + 200])  # body cut short of its declared 268
    with pytest.raises(_Unscannable, match="encrypted OpenPGP"):
        credential_in_file(truncated)

    # The control: a session packet followed by something that is NOT an encrypted-data packet is
    # not an encrypted message, and must still publish.
    lone = tmp_path / "lone.bin"
    lone.write_bytes(session(268, follow=b"hello there, ordinary bytes"))
    assert credential_in_file(lone) is None


def test_an_armored_openpgp_message_is_refused(tmp_path):
    """The armored form is the one an author actually commits, and it is not a key block.

    `gpg --armor --symmetric` writes `-----BEGIN PGP MESSAGE-----`, which the private-key armor
    pattern never matched, and whose base64 body is ciphertext -- so decoding it finds nothing
    either. A Freesolo key inside published clean.
    """
    from flash.env_secrets import _Unscannable, credential_in_file

    armored = tmp_path / "secret.asc"
    armored.write_text(
        "-----BEGIN PGP MESSAGE-----\n\n"
        "jA0ECQMKfBUuPuHPCr3/0nUBOUEMgb7cvEQYOuU79Qk6ecIpdWiDm1BQOI8rD2Mm\n"
        "vjOpOLTwrKclkwFi9fZHNA/ehv0mSbBXQnJhTfBCVQ==\n"
        "=abcd\n-----END PGP MESSAGE-----\n"
    )
    with pytest.raises(_Unscannable, match="encrypted OpenPGP"):
        credential_in_file(armored)

    # A clear-signed message carries its payload in the CLEAR, so the ordinary scan reads it and
    # refusing it would block a signed README.
    signed = tmp_path / "README.asc"
    signed.write_text(
        "-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA256\n\n"
        "this environment does nothing secret\n"
        "-----BEGIN PGP SIGNATURE-----\n\nabcd\n-----END PGP SIGNATURE-----\n"
    )
    assert credential_in_file(signed) is None


def test_a_credential_in_a_self_extracting_shell_archive_is_found(tmp_path):
    """`makeself` puts a script first and the compressed payload after it.

    Every container test asks what the file BEGINS with, and it begins with `#!/bin/sh` -- so the
    gzip behind the stub was scanned as opaque bytes and the key inside published. The payload is
    an ordinary gzip once its offset is known, so it is EXPANDED rather than refused.
    """
    import gzip

    from flash.env_secrets import credential_in_file

    stub = b'#!/bin/sh\n# self-extracting archive\ntail -c +NNN "$0" | gzip -dc\nexit 0\n'
    payload = gzip.compress(b'export FREESOLO_API_KEY="fslo_%s"\n' % _FAKE_KEY_BODY.encode())
    sfx = tmp_path / "install.run"
    sfx.write_bytes(stub + payload)
    assert credential_in_file(sfx) == "a Freesolo API key"

    # bzip2 and xz payloads are as ordinary as gzip for this shape
    import bz2
    import lzma

    for suffix, compress in (("bz2", bz2.compress), ("xz", lzma.compress)):
        archive = tmp_path / f"install.{suffix}.run"
        archive.write_bytes(
            stub + compress(b'export FREESOLO_API_KEY="fslo_%s"\n' % _FAKE_KEY_BODY.encode())
        )
        assert credential_in_file(archive) == "a Freesolo API key"

    # The control: a shell script with no payload at all must still publish. Three bytes of magic
    # occur by chance in any large binary, so the offset is proven by actually inflating it.
    plain = tmp_path / "setup.sh"
    plain.write_bytes(stub + b"\x1f\x8b\x08 not really a gzip stream at all\n" * 4)
    assert credential_in_file(plain) is None


def test_a_forged_directory_size_does_not_allocate_the_package(tmp_path):
    """The member-count preflight exists to avoid `ZipFile`'s large allocation.

    Reading the end record's 32-bit directory size into one `bytes` object re-created that cost in
    the preflight itself: the field is attacker-controlled, so a forged value covering the file
    allocated roughly the whole package before discovering the candidate was not a directory.
    """
    import os
    import struct
    import tracemalloc
    import zipfile

    from flash.env_formats import _zip_member_count

    archive = tmp_path / "forged.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as package:
        package.writestr("big.bin", os.urandom(6 << 20))
    raw = bytearray(archive.read_bytes())
    at = raw.rfind(b"PK\x05\x06")
    raw[at + 12 : at + 16] = struct.pack("<I", at)  # directory size covers the whole file
    raw[at + 16 : at + 20] = struct.pack("<I", 0)  # starting at offset 0
    archive.write_bytes(bytes(raw))

    tracemalloc.start()
    count = _zip_member_count(archive, 100_000)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert count == 1
    # bounded by the read window, not by the forged size: the pre-fix form peaked at the file size
    assert peak < 3 << 20, f"preflight allocated {peak} bytes for a {at}-byte claim"


def test_a_real_zip_still_counts_its_members(tmp_path):
    """The bounded walk must still read an ordinary directory, including one behind a stub."""
    import zipfile

    from flash.env_formats import _zip_member_count

    ordinary = tmp_path / "ordinary.zip"
    with zipfile.ZipFile(ordinary, "w") as package:
        for index in range(250):
            package.writestr(f"member{index}.txt", b"x")
    assert _zip_member_count(ordinary, 100_000) == 250

    sfx = tmp_path / "sfx.zip"
    sfx.write_bytes(b"MZ" + b"\x00" * 4094 + ordinary.read_bytes())
    assert _zip_member_count(sfx, 100_000) == 250


def test_decoy_magics_do_not_hide_an_appended_payload(tmp_path):
    """The overlay candidate cap was itself fail-open.

    Only a candidate that actually inflates ends the search, so an attacker chooses how many failing
    magics sit in front of the real payload. Returning "no overlay" once the cap was hit meant
    padding a stub with enough decoys published the credential in the stream behind them.
    """
    import gzip
    import random

    from flash.env_formats import _MAX_OVERLAY_CANDIDATES
    from flash.env_secrets import credential_in_file

    stub = b'#!/bin/sh\ntail -c +NNN "$0" | gzip -dc\nexit 0\n'
    payload = gzip.compress(b'export FREESOLO_API_KEY="fslo_%s"\n' % _FAKE_KEY_BODY.encode())

    # Seeded, because "cannot inflate" is a property of the noise rather than of the three fixed
    # bytes in front of it: about one draw in forty puts a decoy that DOES inflate in the spanning
    # file, which ends the search at that decoy and finds the key instead of exhausting the bound.
    # Both answers block the publish, so the escape was in the assertion and not in the scan -- but
    # a test that flips between two passing shapes cannot say which one it proved.
    noise = random.Random(0)

    def decoys(count: int) -> bytes:
        """`count` gzip magics that cannot inflate -- three fixed bytes then noise."""
        return b"".join(b"\x1f\x8b\x08" + noise.randbytes(29) for _ in range(count))

    # Comfortably past the old cap of 64, which returned None and published. Every candidate in a
    # window is probed before the bound is consulted, so a decoy count the file chooses cannot push
    # the real payload out of reach -- the key is FOUND rather than merely refused.
    many = tmp_path / "many.run"
    many.write_bytes(stub + decoys(65) + payload)
    assert credential_in_file(many) == "a Freesolo API key"

    beyond = tmp_path / "beyond.run"
    beyond.write_bytes(stub + decoys(_MAX_OVERLAY_CANDIDATES + 50) + payload)
    assert credential_in_file(beyond) == "a Freesolo API key"

    few = tmp_path / "few.run"
    few.write_bytes(stub + decoys(3) + payload)
    assert credential_in_file(few) == "a Freesolo API key"

    # The bound still exists, and when it genuinely bites it REFUSES. It is consulted per window,
    # so exhausting it takes decoys spanning more than one -- and the file behind them is then
    # unverified rather than clean.
    from flash.env_formats import _STREAM_WINDOW_BYTES
    from flash.env_secrets import _Unscannable

    per_window = decoys(_MAX_OVERLAY_CANDIDATES + 100)
    pad = b"\x00" * max(0, _STREAM_WINDOW_BYTES - len(per_window))
    spanning = tmp_path / "spanning.run"
    spanning.write_bytes(stub + per_window + pad + per_window + pad + payload)
    with pytest.raises(_Unscannable, match="candidates"):
        credential_in_file(spanning)


def test_a_marker_packet_does_not_hide_an_encrypted_message(tmp_path):
    """Marker normalization was applied to one OpenPGP predicate and not the other.

    A marker packet is a legal no-op that GnuPG skips, so `ca 03 50 47 50` in front of a real
    encrypted message still decrypts -- while the encrypted-message test saw tag 10, reported "not
    encrypted", and published the ciphertext.
    """
    import struct

    from flash.env_secrets import _Unscannable, credential_in_file

    body = b"\x03" + b"\x00" * 8 + b"\x01" + b"\x00" * 258
    message = b"\x85" + struct.pack(">H", 268) + body + b"\xd2\x40" + b"\x00" * 64

    marked = tmp_path / "marked.gpg"
    marked.write_bytes(b"\xca\x03PGP" + message)
    with pytest.raises(_Unscannable, match="encrypted OpenPGP"):
        credential_in_file(marked)

    # the same message without the marker, which was already refused: the control proving the
    # assertion above is about the marker rather than about the message
    plain = tmp_path / "plain.gpg"
    plain.write_bytes(message)
    with pytest.raises(_Unscannable, match="encrypted OpenPGP"):
        credential_in_file(plain)


def test_an_uncorroborated_member_count_does_not_refuse_a_clean_file(tmp_path):
    """The member bound refused files that contain no archive at all.

    The count is read from a record found by SEARCHING the tail, so any file can carry one: a tar
    member of ordinary text plus `PK\x05\x06` and a zip64 record claimed 100,001 members. The
    resulting refusal made a clean tar unpublishable, and `is_zipfile` cannot tell the difference --
    it accepts those bytes too. Only a count the directory walk corroborates may refuse.
    """
    import io
    import struct
    import tarfile
    import zipfile

    from flash.env_formats import _zip_member_count
    from flash.env_secrets import credential_in_file

    forged = (
        b"PK\x06\x06"
        + struct.pack("<QHHIIQQQQ", 44, 45, 45, 0, 0, 100_001, 100_001, 0, 0)
        + b"PK\x06\x07"
        + struct.pack("<IQI", 0, 0, 1)
        + b"PK\x05\x06"
        + struct.pack("<HHHHIIH", 0, 0, 0xFFFF, 0xFFFF, 0, 0, 0)
    )
    body = b"harmless text\n" * 10 + forged
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as archive:
        info = tarfile.TarInfo("data.bin")
        info.size = len(body)
        archive.addfile(info, io.BytesIO(body))
    clean = tmp_path / "clean.tar"
    clean.write_bytes(buf.getvalue())
    assert credential_in_file(clean) is None

    # The bound must still fire on a REAL oversized archive, which has a real directory to walk --
    # and a real archive whose count field is forged DOWN still reports the walked truth.
    real = tmp_path / "real.zip"
    with zipfile.ZipFile(real, "w") as package:
        for index in range(300):
            package.writestr(f"m{index}.txt", b"x")
    assert _zip_member_count(real, 100) == 300
    patched = bytearray(real.read_bytes())
    at = patched.rfind(b"PK\x05\x06")
    patched[at + 8 : at + 12] = struct.pack("<HH", 1, 1)
    shrunk = tmp_path / "shrunk.zip"
    shrunk.write_bytes(bytes(patched))
    assert _zip_member_count(shrunk, 100_000) == 300


def test_the_wide_text_floor_admits_the_shortest_token(tmp_path):
    """Lowering the base64 floor for Slack left the independent wide-text floor at 20.

    The two floors answer the same question about the same patterns. `xoxb-` plus its 10-character
    body is 15 bytes, so its UTF-16 form carries 15 NUL columns -- under a hardcoded 20, and the
    token that was detected as ASCII was missed in the encoding this narrowing exists to cover.
    """
    from flash.env_buffers import _WIDE_RUN
    from flash.env_patterns import SHORTEST_TOKEN_BYTES
    from flash.env_secrets import _credential_kind

    token = b"xoxb-AbCdEf0123"
    assert len(token) == SHORTEST_TOKEN_BYTES
    assert _credential_kind(token) == "a Slack token"
    assert _credential_kind(token.decode().encode("utf-16-le")) == "a Slack token"
    assert _credential_kind(token.decode().encode("utf-16-be")) == "a Slack token"

    # derived, not written: the floor moves with the shortest token the patterns admit
    assert _WIDE_RUN.pattern == rb"\x01{%d,}" % SHORTEST_TOKEN_BYTES

    # the NUL-column gate still keeps ordinary machine code from narrowing into a token
    import pathlib

    binary = pathlib.Path("/usr/bin/python3")
    if binary.exists():
        from flash.env_secrets import credential_in_file

        assert credential_in_file(binary) is None


def test_a_secret_key_behind_the_marker_bound_is_refused(tmp_path):
    """Bounding the marker walk turned the bound itself into the bypass.

    A marker packet is a legal no-op, so a key behind more of them than the walk allows left the
    secret-key test looking at a marker header, which is not a key -- and the file published. The
    bound has to fail closed: still sitting on a marker means what follows is unread, not absent.
    """
    from flash.env_openpgp import _MAX_OPENPGP_MARKERS
    from flash.env_secrets import _Unscannable, credential_in_file

    marker = b"\xca\x03PGP"
    # a secret-key packet laid out as `gpg --export-secret-keys` writes one: tag 5 old-format with
    # length-type 1, so a TWO-byte length, then version 4, a 4-byte timestamp and the algorithm.
    key = b"\x95\x03\x98\x04" + b"\x6a\x7e\x7e\x1e" + b"\x01" + b"\x00" * 16

    within = tmp_path / "within.pgp"
    within.write_bytes(marker * (_MAX_OPENPGP_MARKERS - 1) + key)
    assert credential_in_file(within) == "a private key"

    beyond = tmp_path / "beyond.pgp"
    beyond.write_bytes(marker * (_MAX_OPENPGP_MARKERS + 1) + key)
    with pytest.raises(_Unscannable, match="cannot walk to the end"):
        credential_in_file(beyond)

    # markers in front of nothing in particular are still not a credential
    bare = tmp_path / "bare.pgp"
    bare.write_bytes(marker + b"ordinary file contents\n")
    assert credential_in_file(bare) is None


def test_an_encrypted_openpgp_message_is_refused(tmp_path):
    """`gpg --symmetric` around a credential is opaque, and opaque is not clean.

    The ciphertext carries no secret-key tag and matches no textual or DER check, so it read as
    ordinary bytes and published -- while an encrypted ZIP member, the same situation, is refused.
    """
    from flash.env_secrets import _Unscannable, credential_in_file

    # symmetric-key ESK exactly as `gpg --symmetric` writes it: tag 3 old-format, a 13-byte body,
    # version 4, cipher 9 (AES-256), S2K type 3. The body length puts the next packet at offset 15.
    esk = b"\x8c\x0d\x04\x09\x03" + b"\x00" * 10
    assert len(esk) == 15, "the data packet must land exactly where the length says"
    encrypted = tmp_path / "secrets.gpg"
    encrypted.write_bytes(esk + b"\xd2" + b"\x40" + b"\x00" * 64)  # then a tag-18 data packet
    with pytest.raises(_Unscannable, match="encrypted OpenPGP"):
        credential_in_file(encrypted)

    # a session-key packet with no encrypted data behind it is not this structure
    lone = tmp_path / "lone.bin"
    lone.write_bytes(esk + b"ordinary trailing content\n")
    assert credential_in_file(lone) is None

    # and an ordinary binary must not be refused: the fields are what make this specific
    plain = tmp_path / "plain.bin"
    plain.write_bytes(b"\x8c\x0d\xff\xff\xff" + b"\x00" * 64)
    assert credential_in_file(plain) is None


def test_a_sec1_key_is_found_on_every_supported_curve(tmp_path):
    """Hardcoding the P-256/384/521 scalar sizes missed every smaller curve.

    `openssl ecparam -list_curves` spans 20 to 114 bytes, so real `prime192v1` (24) and
    `secp224r1` (28) keys published intact while the otherwise-identical P-256 form was caught.
    """
    import subprocess

    from flash.env_secrets import credential_in_file

    # `secp112r1` is the smallest curve OpenSSL carries, at a 14-byte scalar. A first pass at this
    # floored the range at 20 and left twelve curve families below it publishing intact.
    for curve in ("secp112r1", "secp128r1", "prime192v1", "secp224r1", "prime256v1", "secp384r1"):
        der = tmp_path / f"{curve}.der"
        result = subprocess.run(
            ["openssl", "ecparam", "-name", curve, "-genkey", "-noout", "-outform", "DER"],
            capture_output=True,
        )
        if result.returncode:  # a curve this build does not carry
            continue
        der.write_bytes(result.stdout)
        assert credential_in_file(der) == "a private key", curve


def test_an_aws_secret_is_found_under_its_json_field_name(tmp_path):
    """`SecretAccessKey` is the name the SDKs and `sts assume-role` write, not the env-var form.

    Anchoring only on `AWS_SECRET_ACCESS_KEY` meant a saved session credential -- which is exactly
    what lands beside an environment config -- published intact.
    """
    from flash.env_secrets import _credential_kind

    body = b"wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    assert _credential_kind(b'{"SecretAccessKey":"' + body + b'"}') == "an AWS secret access key"
    assert _credential_kind(b"AWS_SECRET_ACCESS_KEY=" + body) == "an AWS secret access key"
    assert _credential_kind(b'"secretAccessKey": "' + body + b'"') == "an AWS secret access key"

    # a longer identifier merely ENDING in the field name is not an assignment of one
    assert _credential_kind(b'"NotSecretAccessKey":"' + body + b'"') is None
    # and a placeholder body is still not a credential
    assert _credential_kind(b'"SecretAccessKey":"${AWS_SECRET_ACCESS_KEY}"') is None


def test_a_short_archive_magic_is_decisive_only_at_the_start(tmp_path):
    """A 4-byte magic is not distinctive enough to refuse a file over, at an arbitrary offset.

    Searching every signature across the whole stream refused an ordinary model shard that happened
    to contain four bytes; the 6-to-8-byte signatures of the formats that actually ship
    self-extracting archives stay searched, because those are the ones a stub can hide.
    """
    import os

    from flash.env_secrets import _Unscannable, credential_in_file

    for name, magic in (
        ("zstd", b"\x28\xb5\x2f\xfd"),
        ("lz4", b"\x04\x22\x4d\x18"),
        ("lz4legacy", b"\x02\x21\x4c\x18"),
        ("7z", b"7z\xbc\xaf\x27\x1c"),
        ("rar", b"Rar!\x1a\x07\x01\x00"),
    ):
        at_start = tmp_path / f"{name}.bin"
        at_start.write_bytes(magic + os.urandom(4096))
        with pytest.raises(_Unscannable):
            credential_in_file(at_start)

    # embedded in a shard: the short magics are not decisive, the long ones still are
    for name, magic, refuses in (
        ("zstd", b"\x28\xb5\x2f\xfd", False),
        ("lz4", b"\x04\x22\x4d\x18", False),
        ("7z", b"7z\xbc\xaf\x27\x1c", True),
        ("rar", b"Rar!\x1a\x07\x01\x00", True),
    ):
        shard = tmp_path / f"shard_{name}.bin"
        shard.write_bytes(os.urandom(8192) + magic + os.urandom(8192))
        if refuses:
            with pytest.raises(_Unscannable):
                credential_in_file(shard)
        else:
            assert credential_in_file(shard) is None, name


def test_a_message_encrypted_to_two_recipients_is_refused(tmp_path):
    """Every session-key packet is walked, not just the first.

    `gpg --encrypt -r a -r b` writes one PKESK per recipient before the encrypted data packet, so
    deciding the message from the first packet alone found another PKESK where a data packet was
    expected and reported "not encrypted". GnuPG decrypts the file happily, and a credential inside
    published intact -- while the same message to ONE recipient was refused.
    """
    from flash.env_secrets import _Unscannable, credential_in_file

    if not shutil.which("gpg"):
        pytest.skip("gpg is not installed")
    home = tmp_path / "gnupg"
    home.mkdir(mode=0o700)
    for uid in ("alpha@example.test", "beta@example.test"):
        subprocess.run(
            [
                "gpg",
                "--homedir",
                str(home),
                "--batch",
                "--pinentry-mode",
                "loopback",
                "--passphrase",
                "",
                "--quick-generate-key",
                uid,
                "rsa2048",
                "encr",
                "never",
            ],
            check=True,
            capture_output=True,
        )
    plain = tmp_path / "plain.txt"
    plain.write_text(f"FREESOLO_API_KEY=fslo_{_FAKE_KEY_BODY}\n")

    def encrypt(*recipients):
        command = ["gpg", "--homedir", str(home), "--batch", "--yes", "--trust-model", "always"]
        for recipient in recipients:
            command += ["-r", recipient]
        return subprocess.run(
            [*command, "-o", "-", "--encrypt", str(plain)], check=True, capture_output=True
        ).stdout

    # the control: one recipient, which was already refused
    single = tmp_path / "one.gpg"
    single.write_bytes(encrypt("alpha@example.test"))
    with pytest.raises(_Unscannable):
        credential_in_file(single)

    both = tmp_path / "two.gpg"
    both.write_bytes(encrypt("alpha@example.test", "beta@example.test"))
    with pytest.raises(_Unscannable):
        credential_in_file(both)


def test_an_appended_bzip2_whose_first_block_exceeds_the_probe_is_expanded(tmp_path):
    """A bzip2 payload behind a stub is found even when the probe reads no output.

    bzip2 emits nothing until it has a whole block of up to 900 KiB, so a stream compressing more
    than the 64 KiB probe returns empty from a perfectly valid decode. Treating that as "not a
    stream" dismissed the candidate, and a credential behind a self-extracting stub published --
    while the identical stream standing alone was detected.
    """
    import bz2
    import os

    from flash.env_secrets import credential_in_file

    payload = bz2.compress(os.urandom(200_000) + f"fslo_{_FAKE_KEY_BODY}".encode())
    # the probe really does yield nothing, or this fixture would not exercise the bug
    assert not bz2.BZ2Decompressor().decompress(payload[:65536], 4096)

    standalone = tmp_path / "payload.bz2"
    standalone.write_bytes(payload)
    assert credential_in_file(standalone) == "a Freesolo API key"

    appended = tmp_path / "installer.run"
    appended.write_bytes(b"#!/bin/sh\nexit 0\n" + payload)
    assert credential_in_file(appended) == "a Freesolo API key"


def test_an_appended_gzip_whose_name_field_exceeds_the_probe_is_expanded(tmp_path):
    """A gzip name is NUL-terminated and unbounded, so a legal one outruns the 64 KiB probe.

    The deflate bits then sit past everything the overlay search reads, the candidate inflates to
    nothing, and it was dismissed as "not a stream" -- so the credential behind the stub published
    while `gzip.decompress` recovered it from the very same bytes. Only the extra field was treated
    as a reason a header might be unfinished, and the name and comment fields have exactly the same
    shape.
    """
    import gzip
    import io

    from flash.env_secrets import credential_in_file

    body = f'export FREESOLO_API_KEY="fslo_{_FAKE_KEY_BODY}"\n'.encode()
    buffered = io.BytesIO()
    with gzip.GzipFile(fileobj=buffered, mode="wb", filename="") as writer:
        writer.write(body)
    raw = bytearray(buffered.getvalue())
    # set FNAME and splice in a name longer than the probe, which is legal and ordinary for a
    # stream produced from a long path
    named = (
        bytes(raw[:3])
        + bytes([raw[3] | 0x08])
        + bytes(raw[4:10])
        + b"n" * (80 << 10)
        + b"\x00"
        + bytes(raw[10:])
    )
    # the stream is real: an ordinary reader recovers the whole credential from it
    assert body in gzip.decompress(named)

    standalone = tmp_path / "payload.gz"
    standalone.write_bytes(named)
    assert credential_in_file(standalone) == "a Freesolo API key"

    appended = tmp_path / "installer.run"
    appended.write_bytes(b"#!/bin/sh\nexit 0\n" + named)
    assert credential_in_file(appended) == "a Freesolo API key"

    # widening the header test must not make ordinary content look like an unfinished stream
    ordinary = tmp_path / "notes.txt"
    ordinary.write_bytes(b"# ordinary configuration\nDEBUG=1\n" * 4000)
    assert credential_in_file(ordinary) is None


def test_a_json_escaped_credential_name_is_still_matched(tmp_path):
    """`"SecretAccess\\u004bey"` names the same field, so the same secret is caught.

    JSON says the two spellings are one string and every parser agrees, so an AWS credential
    document written with a single escaped character loaded identically while the literal-byte name
    matched nothing. The escape carries the character's CODE POINT, so a case-insensitive literal
    cannot supply it -- `\\u004b` is `K`, and folding `k` does not reach it.
    """
    import base64
    import os

    from flash.env_secrets import credential_in_file

    body = base64.b64encode(os.urandom(30))[:40].decode()
    for name, text in (
        ("plain.json", f'{{"SecretAccessKey": "{body}"}}'),
        ("escaped.json", f'{{"SecretAccess\\u004bey": "{body}"}}'),
        ("escaped_lower.json", f'{{"secretaccess\\u006bey": "{body}"}}'),
        ("escaped_env.json", f'{{"AWS_SECRET_ACCESS_\\u004bEY": "{body}"}}'),
    ):
        document = tmp_path / name
        document.write_text(text)
        assert credential_in_file(document) == "an AWS secret access key", name

    # the same treatment for the other assignment-anchored name
    wandb = tmp_path / "wandb.json"
    wandb.write_text(f'{{"WANDB_API_\\u004bEY": "{os.urandom(20).hex()}"}}')
    assert credential_in_file(wandb) == "a Weights & Biases API key"

    # and prose about the field is still publishable
    prose = tmp_path / "README.md"
    prose.write_text("the SecretAccessKey field is documented above\n")
    assert credential_in_file(prose) is None


def test_a_headerless_deflate_stream_is_expanded(tmp_path):
    """Raw DEFLATE has no magic, so the decode itself has to be the recognition.

    `zlib.compressobj(wbits=-15)` writes RFC 1951 with no header at all, so the magic list never
    matched it and the zlib header rule had no header to read. The compressed bytes were scanned as
    content and the credential inside published, while the zlib-wrapped form of the same payload
    was expanded.
    """
    import random
    import zlib

    from flash.env_secrets import credential_in_file

    secret = f"FREESOLO_API_KEY=fslo_{_FAKE_KEY_BODY}\n".encode()
    compressor = zlib.compressobj(9, zlib.DEFLATED, -15)
    raw = compressor.compress(secret) + compressor.flush()

    stream = tmp_path / "payload.deflate"
    stream.write_bytes(raw)
    assert credential_in_file(stream) == "a Freesolo API key"

    # completeness is what keeps this off ordinary binaries: a truncated stream is not one, and
    # neither is arbitrary data
    truncated = tmp_path / "partial.bin"
    truncated.write_bytes(raw[:-4])
    assert credential_in_file(truncated) is None
    # SEEDED, not `os.urandom`: measured 13 refusals in 20,000 random 4 KiB blocks, which is a 1.3%
    # chance of a red run per draw of twenty. Those refusals are correct -- a block that satisfies
    # the zlib header and its FDICT bit really cannot be inspected -- so this control is asserting
    # that raw deflate does not fire on arbitrary data, not that nothing else ever does.
    draws = random.Random(0)
    for index in range(20):
        noise = tmp_path / f"noise{index}.bin"
        noise.write_bytes(draws.randbytes(4096))
        assert credential_in_file(noise) is None


def test_a_base64_value_holding_a_container_is_expanded(tmp_path):
    """A Kubernetes Secret stores a gzipped credential base64, and both layers have to come off.

    The decode succeeded and the decoded bytes were then pattern-matched while still COMPRESSED, so
    `b64encode(gzip.compress(secret))` published clean -- even though base64 of the plaintext and
    the bare gzip were each detected on their own.
    """
    import base64
    import gzip

    from flash.env_secrets import credential_in_file

    secret = f"FREESOLO_API_KEY=fslo_{_FAKE_KEY_BODY}\n".encode()
    encoded = base64.b64encode(gzip.compress(secret))

    value = tmp_path / "encoded.txt"
    value.write_bytes(encoded)
    assert credential_in_file(value) == "a Freesolo API key"

    manifest = tmp_path / "secret.yaml"
    manifest.write_bytes(b"apiVersion: v1\nkind: Secret\ndata:\n  env: " + encoded + b"\n")
    assert credential_in_file(manifest) == "a Freesolo API key"

    # an ordinary compressed document encoded the same way still publishes
    innocent = tmp_path / "docs.txt"
    innocent.write_bytes(base64.b64encode(gzip.compress(b"just some documentation " * 40)))
    assert credential_in_file(innocent) is None


def test_a_netrc_machine_password_is_refused(tmp_path):
    """A `.netrc` password names no service, so nothing else here could anchor on it.

    `wandb login` and `huggingface-cli login` persist their token this way, and the value is a bare
    40-hex string: no issuer prefix for a token pattern, no assignment for the named-credential
    patterns. The CLI's filename filter does not cover `.netrc` either, so the file reached the hub
    through both the CLI and a direct upload with a live key in it.
    """
    import base64
    import os

    from flash.env_secrets import credential_in_file

    token = os.urandom(20).hex()
    for name, text in (
        ("multiline", f"machine api.wandb.ai\n  login user\n  password {token}\n"),
        ("oneline", f"machine api.wandb.ai login user password {token}\n"),
        ("default", f"default login user password {token}\n"),
        ("quoted", f'machine api.wandb.ai\npassword "{token}"\n'),
        (
            "b64",
            "machine huggingface.co\npassword " + base64.b64encode(os.urandom(30)).decode() + "\n",
        ),
    ):
        entry = tmp_path / f"{name}.netrc"
        entry.write_text(text)
        assert credential_in_file(entry) == "a machine password in a netrc file", name

    # writing ABOUT a netrc, or a placeholder in one, is not a credential
    for name, text in (
        ("prose", "set the password field on the machine you use\n"),
        ("placeholder", "machine example.com\nlogin me\npassword changeme\n"),
        ("commented", "# machine api.example.com\n# password <your-token-here>\n"),
        ("yaml", "machine: builder-01\npassword: null\n"),
        ("sha", "the sha is " + "a1b2c3d4" * 5 + " on machine two\n"),
    ):
        innocent = tmp_path / f"{name}.txt"
        innocent.write_text(text)
        assert credential_in_file(innocent) is None, name


def test_the_server_refuses_a_netrc_an_older_client_uploaded(tmp_path):
    """The CLI is not the trust boundary, so the server has to catch what it never filtered.

    A raw `POST /v1/envs` or an older client skips the CLI's exclusions entirely, and the server is
    what writes to the shared hub, whose history is permanent.
    """
    import os

    from flash.env_secrets import reject_credential_bearing_package

    package = tmp_path / "package"
    package.mkdir()
    (package / ".netrc").write_text(
        f"machine api.wandb.ai\n  login user\n  password {os.urandom(20).hex()}\n"
    )
    with pytest.raises(ValueError, match="netrc"):
        reject_credential_bearing_package(package, display={})


def test_a_credential_in_a_pdf_stream_is_found(tmp_path):
    """A PDF keeps its content in a zlib stream that does not start at byte zero.

    The head-anchored zlib check never saw it and the appended-payload search covers only gzip,
    bzip2 and xz, so a key inside a document published -- while the same zlib record standing alone
    was expanded. Located by the PDF's own grammar rather than by searching for zlib headers, which
    trip about once per 2 KiB of arbitrary data.
    """
    import random
    import zlib

    from flash.env_secrets import credential_in_file

    record = zlib.compress(f"FREESOLO_API_KEY=fslo_{_FAKE_KEY_BODY}\n".encode())
    document = tmp_path / "report.pdf"
    document.write_bytes(
        b"%PDF-1.4\n1 0 obj\n<< /Length "
        + str(len(record)).encode()
        + b" /Filter /FlateDecode >>\nstream\n"
        + record
        + b"\nendstream\nendobj\ntrailer\n%%EOF\n"
    )
    assert credential_in_file(document) == "a Freesolo API key"

    # a PDF whose streams hold ordinary content still publishes, and so does a non-PDF that
    # happens to carry the same bytes -- the signature is what gates the search
    innocent = tmp_path / "clean.pdf"
    innocent.write_bytes(
        b"%PDF-1.4\n1 0 obj\n<< /Filter /FlateDecode >>\nstream\n"
        + zlib.compress(b"ordinary page content " * 50)
        + b"\nendstream\n%%EOF\n"
    )
    assert credential_in_file(innocent) is None
    shard = tmp_path / "shard.bin"
    # seeded: see the raw-deflate control
    shard.write_bytes(random.Random(2).randbytes(1 << 20))
    assert credential_in_file(shard) is None


def test_a_credential_split_across_adjacent_literals_is_found(tmp_path):
    """Adjacent string literals concatenate, so the file holds no contiguous copy of the key.

    Python source is exempt from the filename filter by design -- helper modules have to ship or
    the worker fails to import -- so a key split this way had nothing between it and the hub, while
    the same key written as one literal was caught.
    """
    from flash.env_secrets import credential_in_file

    key = f"fslo_{_FAKE_KEY_BODY}"
    for name, source in (
        ("adjacent.py", f'FREESOLO_API_KEY = "{key[:18]}" "{key[18:]}"\n'),
        ("wrapped.py", f'FREESOLO_API_KEY = (\n    "{key[:18]}"\n    "{key[18:]}"\n)\n'),
        ("three.py", f'KEY = "{key[:12]}" "{key[12:24]}" "{key[24:]}"\n'),
    ):
        helper = tmp_path / name
        helper.write_text(source)
        assert credential_in_file(helper) == "a Freesolo API key", name

    # a comma between two literals is two values, not one string, so unrelated entries are never
    # welded into a credential nobody wrote
    for name, source in (
        ("list.json", '["alpha_value_here", "beta_value_here"]\n'),
        ("dict.json", '{"a": "xxxx", "b": "yyyy"}\n'),
        ("prose.txt", 'he said "hello" "world" and left\n'),
    ):
        innocent = tmp_path / name
        innocent.write_text(source)
        assert credential_in_file(innocent) is None, name


def test_a_pdf_with_more_streams_than_the_limit_is_refused(tmp_path):
    """The stream walk is bounded, and stopping at the bound reported the rest as clean.

    `islice` truncated the search silently, so a document holding one more `/FlateDecode` stream
    than the limit published the credential in it while the same document under the limit was
    caught. Every other bound here refuses; this one returned a verdict it had not earned.
    """
    import zlib

    from flash.env_secrets import _Unscannable, credential_in_file

    def document(streams):
        out = b"%PDF-1.4\n"
        for index in range(streams):
            body = (
                f"FREESOLO_API_KEY=fslo_{_FAKE_KEY_BODY}\n".encode()
                if index == streams - 1
                else b"ordinary page content %d " % index * 3
            )
            out += b"%d 0 obj\n<< /Filter /FlateDecode >>\nstream\n" % index
            out += zlib.compress(body) + b"\nendstream\nendobj\n"
        return out + b"trailer\n%%EOF\n"

    under = tmp_path / "small.pdf"
    under.write_bytes(document(20))
    assert credential_in_file(under) == "a Freesolo API key"

    over = tmp_path / "huge.pdf"
    over.write_bytes(document(4100))
    with pytest.raises(_Unscannable):
        credential_in_file(over)


def test_a_base64_container_longer_than_one_window_is_expanded(tmp_path):
    """A container has to be seen whole, and windowing cut it in half.

    Only the first window decoded to anything carrying a gzip header; every later window started
    mid-stream, so the expansion saw a prefix and never reached the tail. A 13 KB base64 of a
    gzip published clean while the same gzip standing alone was expanded.
    """
    import base64
    import gzip
    import random

    from flash.env_secrets import credential_in_file

    # The filler has to be BOTH bulky when compressed and genuinely compressed. `os.urandom` is
    # not: DEFLATE stores an incompressible tail verbatim, so the key survives literally in the
    # gzip bytes and is found with no expansion at all -- the test would pass without the fix.
    # Fully repetitive filler is not either: it compresses so far that the run never reaches one
    # window. Semi-random words sit between the two.
    draws = random.Random(7)
    words = [
        "".join(draws.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(8)) for _ in range(4000)
    ]
    filler = "\n".join(draws.choice(words) for _ in range(2000)).encode()
    packed = gzip.compress(filler + b"\n" + f"FREESOLO_API_KEY=fslo_{_FAKE_KEY_BODY}\n".encode())
    assert b"fslo_" not in packed, "filler must not leave the key stored literally"
    encoded = base64.b64encode(packed)
    assert len(encoded) > 8192, "the run must exceed one decode window"

    bare = tmp_path / "shard.gz"
    bare.write_bytes(packed)
    assert credential_in_file(bare) == "a Freesolo API key"

    value = tmp_path / "encoded.txt"
    value.write_bytes(encoded)
    assert credential_in_file(value) == "a Freesolo API key"


def test_a_line_continuation_inside_a_credential_is_rejoined(tmp_path):
    """The shell removes a backslash-newline before the value is ever assigned.

    So `KEY="fslo_AbCd\\<newline>Ef01"` exports the whole credential while no contiguous run of
    bytes in the file holds it. The literal-pair join added earlier covered only adjacent QUOTED
    literals, which is not how a long line gets wrapped in a shell file.
    """
    from flash.env_secrets import credential_in_file

    whole = tmp_path / "env.sh"
    whole.write_bytes(f'FREESOLO_API_KEY="fslo_{_FAKE_KEY_BODY}"\n'.encode())
    assert credential_in_file(whole) == "a Freesolo API key"

    split = tmp_path / "wrapped.sh"
    split.write_bytes(
        b'FREESOLO_API_KEY="fslo_'
        + _FAKE_KEY_BODY[:10].encode()
        + b"\\\n"
        + _FAKE_KEY_BODY[10:].encode()
        + b'"\n'
    )
    assert credential_in_file(split) == "a Freesolo API key"

    # an ESCAPED backslash at end of line is not a continuation, so two unrelated lines must not
    # be welded into a run that decodes to a credential nobody wrote
    for name, source in (
        ("paths.sh", 'ROOT="C:\\\\"\nNAME="ordinary"\n'),
        ("makefile", "all:\n\tgcc -o out \\\n\t\tmain.c\n"),
    ):
        innocent = tmp_path / name
        innocent.write_text(source)
        assert credential_in_file(innocent) is None, name


def test_an_aws_secret_with_escaped_slashes_is_matched(tmp_path):
    """`\\/` is a legal JSON escape, and an AWS secret is base64 so it carries `/` routinely.

    Encoders that escape it -- PHP's `json_encode` by default, several SDK log formatters -- broke
    the run of 40 into two shorter runs, so the SAME key published clean purely because of how the
    document happened to be serialized. The earlier fix covered escapes in the field NAME only.
    """
    import os

    from flash.env_secrets import credential_in_file

    body = base64.b64encode(os.urandom(30)).decode()[:40].replace("+", "/")
    if "/" not in body:
        body = body[:20] + "/" + body[21:]

    plain = tmp_path / "creds.json"
    plain.write_text(f'{{"SecretAccessKey": "{body}"}}')
    assert credential_in_file(plain) == "an AWS secret access key"

    escaped = tmp_path / "escaped.json"
    slashed = body.replace("/", "\\/")
    escaped.write_text(f'{{"SecretAccessKey": "{slashed}"}}')
    assert credential_in_file(escaped) == "an AWS secret access key"


def test_a_json_unicode_escape_inside_a_credential_is_resolved(tmp_path):
    """`\\u0045` IS `E` to every JSON parser, so the escape spells the key without spelling it.

    `json.loads` returns the identical credential either way -- the reader of the file gets the
    working key -- but the raw bytes the patterns read carry a six-character escape where one
    character belongs, so the run breaks and the file published clean. One escaped character
    anywhere in the body is enough, and encoders that emit `\\u` for ASCII (Python's
    `ensure_ascii`, several log formatters) can produce it without anyone choosing to.
    """
    import json

    from flash.env_secrets import credential_in_file

    def escaped(text: str) -> str:
        return "".join(f"\\u00{ord(letter):02x}" for letter in text)

    for name, key in (
        ("one.json", f"fslo_{_FAKE_KEY_BODY[:4]}{escaped(_FAKE_KEY_BODY[4])}{_FAKE_KEY_BODY[5:]}"),
        ("all.json", f"fslo_{escaped(_FAKE_KEY_BODY)}"),
        ("prefix.json", f"{escaped('f')}slo_{_FAKE_KEY_BODY}"),
    ):
        published = tmp_path / name
        published.write_text(f'{{"FREESOLO_API_KEY": "{key}"}}')
        # the escape is not a near-miss spelling: the parser hands the caller the real key
        assert json.loads(published.read_text())["FREESOLO_API_KEY"] == f"fslo_{_FAKE_KEY_BODY}"
        assert credential_in_file(published) == "a Freesolo API key", name

    # resolving escapes must not invent a credential in a document that holds none, and an escaped
    # non-ASCII character stays untouched rather than being folded into some encoding
    for name, source in (
        ("prose.json", '{"note": "caf\\u00e9 \\u0041BC", "count": 1}'),
        ("newline.json", '{"text": "line\\u000aline"}'),
    ):
        innocent = tmp_path / name
        innocent.write_text(source)
        assert credential_in_file(innocent) is None, name


def test_an_openssl_salted_envelope_is_refused(tmp_path):
    """`openssl enc` ciphertext is opaque, so approving it approves bytes nobody inspected.

    An encrypted credential file is an ordinary thing to keep beside an environment, and the hub
    copy is readable by everyone the environment is -- the passphrase travels beside the file about
    as often as not. A real AES-256-CBC envelope around a Freesolo key scanned clean while
    decryption recovered the whole key, which is the same unverifiable-content condition that
    already refuses an encrypted zip member and an OpenPGP message.
    """
    import subprocess

    from flash.env_secrets import _Unscannable, credential_in_file

    plain = tmp_path / "key.txt"
    plain.write_bytes(f"FSLO_API_KEY=fslo_{_FAKE_KEY_BODY}\n".encode())
    envelope = tmp_path / "key.enc"
    made = subprocess.run(
        [
            "openssl",
            "enc",
            "-aes-256-cbc",
            "-pbkdf2",
            "-salt",
            "-in",
            str(plain),
            "-out",
            str(envelope),
            "-pass",
            "pass:hunter2",
        ],
        capture_output=True,
    )
    if made.returncode:
        pytest.skip("openssl unavailable")

    blob = envelope.read_bytes()
    assert blob.startswith(b"Salted__")
    # the ciphertext really does hide the key from every pattern, which is why refusing is the
    # only honest answer rather than a conservative one
    assert b"fslo_" not in blob

    with pytest.raises(_Unscannable, match="cannot read"):
        credential_in_file(envelope)

    # `Salted__` is eight PRINTABLE characters, so it is recognised only at byte zero. A file that
    # merely mentions the format stays publishable, the same distinction the bare `Rar!` prefix
    # needed.
    prose = tmp_path / "README.md"
    prose.write_text("The OpenSSL envelope header is Salted__ and we do not commit those here.\n")
    assert credential_in_file(prose) is None


def test_a_pdf_filter_chain_is_decoded_before_inflating(tmp_path):
    """`/Filter [/ASCII85Decode /FlateDecode]` applies in order, so the stream is ASCII85 first.

    Handing those bytes to zlib fails, and the stream was skipped as clean while the credential
    inside decoded perfectly well one filter further in. `pdftk` and several writers emit this
    chain, so it is an ordinary document rather than a crafted one.
    """
    import base64 as b64
    import zlib

    from flash.env_secrets import _Unscannable, credential_in_file

    key = f"fslo_{_FAKE_KEY_BODY}".encode()
    inner = zlib.compress(b"BT (FSLO_API_KEY=" + key + b") Tj ET\n" * 4)
    stream = b64.a85encode(inner) + b"~>"
    chained = tmp_path / "chained.pdf"
    chained.write_bytes(
        b"%PDF-1.7\n1 0 obj\n<< /Filter [/ASCII85Decode /FlateDecode] /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream\nendobj\ntrailer\n%%EOF\n"
    )
    assert credential_in_file(chained) == "a Freesolo API key"

    # a chain this cannot fully undo is REFUSED rather than skipped: bytes behind an unreadable
    # filter are as unverified as an archive that would not expand
    unknown = tmp_path / "unknown.pdf"
    unknown.write_bytes(
        b"%PDF-1.7\n1 0 obj\n<< /Filter [/JBIG2Decode /FlateDecode] /Length 10 >>\nstream\n"
        + b"\x00" * 10
        + b"\nendstream\nendobj\ntrailer\n%%EOF\n"
    )
    with pytest.raises(_Unscannable, match="filter"):
        credential_in_file(unknown)

    # the ordinary single-filter document still works
    plain_stream = zlib.compress(b"BT (FSLO_API_KEY=" + key + b") Tj ET\n")
    plain = tmp_path / "plain.pdf"
    plain.write_bytes(
        b"%PDF-1.7\n1 0 obj\n<< /Filter /FlateDecode /Length "
        + str(len(plain_stream)).encode()
        + b" >>\nstream\n"
        + plain_stream
        + b"\nendstream\nendobj\ntrailer\n%%EOF\n"
    )
    assert credential_in_file(plain) == "a Freesolo API key"


def test_a_gzip_with_a_maximum_extra_field_is_probed_past_its_header(tmp_path):
    """A legal 65,535-byte FEXTRA is longer than the 64 KiB probe, so no payload was in view.

    The probe inflated nothing and the candidate was dismissed as "not a stream", even though the
    stream is valid and gunzip reads it. Only the extra field can do this: it declares a LENGTH,
    and it is the one part of a gzip header that can exceed the probe on its own.
    """
    import gzip
    import random
    import struct
    import zlib

    from flash.env_secrets import credential_in_file

    key = f"fslo_{_FAKE_KEY_BODY}".encode()
    body = b"export FSLO_API_KEY=" + key + b"\n"
    raw = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    extra = b"\x00" * 65535
    stream = (
        b"\x1f\x8b\x08\x04"
        + b"\x00" * 6
        + struct.pack("<H", len(extra))
        + extra
        + raw.compress(body)
        + raw.flush()
        + struct.pack("<II", zlib.crc32(body), len(body))
    )
    # the fixture has to be a stream the stdlib really reads, or this proves nothing
    assert gzip.decompress(stream) == body

    behind_a_stub = tmp_path / "installer.sh"
    behind_a_stub.write_bytes(b"#!/bin/sh\necho hello\nexit 0\n" + stream)
    assert credential_in_file(behind_a_stub) == "a Freesolo API key"

    # and the decoys stay decoys: three magic bytes plus noise are not a stream, so an ordinary
    # binary carrying chance magics is still publishable
    draws = random.Random(11)
    decoys = b"".join(b"\x1f\x8b\x08" + draws.randbytes(29) for _ in range(400))
    noise = tmp_path / "shard.bin"
    noise.write_bytes(decoys)
    assert credential_in_file(noise) is None


def test_a_base64_container_cut_by_a_chunk_boundary_is_refused(tmp_path):
    """A container does not survive being cut: only the first piece carries its header.

    The scan reads a file in bounded chunks, so a long encoded blob arrives in pieces and the
    windowed pass sees a prefix that stops before the tail. Measured: the credential in a gzip
    whose base64 crossed the 1 MiB chunk boundary published clean, while the same blob one byte
    shorter was caught. Unexpanded is not clean.
    """
    import gzip
    import json
    import random

    from flash.env_secrets import _SCAN_CHUNK_BYTES, _Unscannable, credential_in_file

    draws = random.Random(13)
    words = [
        bytes(draws.choice(b"abcdefghijklmnopqrstuvwxyz") for _ in range(8)) for _ in range(400)
    ]
    filler = bytearray()
    while len(filler) < (6 << 20):
        filler += draws.choice(words) + b" "
    key = f"fslo_{_FAKE_KEY_BODY}".encode()
    blob = gzip.compress(bytes(filler) + b"\nFSLO_API_KEY=" + key + b"\n", 1)
    encoded = base64.b64encode(blob)
    # semi-random filler on purpose: incompressible padding would leave the key stored VERBATIM in
    # the gzip bytes, and fully repetitive padding never reaches a chunk boundary at all
    assert b"fslo_" not in blob
    assert len(encoded) > _SCAN_CHUNK_BYTES

    cut = tmp_path / "secret.b64"
    cut.write_bytes(b"DATA=" + encoded + b"\n")
    with pytest.raises(_Unscannable, match="too long to expand"):
        credential_in_file(cut)

    # a long run that is NOT a container loses nothing by being cut, so it still publishes: the
    # windowed pass reads a literal credential wherever it lands
    padded = tmp_path / "rows.json"
    padded.write_text(json.dumps({"note": "x" * (_SCAN_CHUNK_BYTES + 4096), "tail": "ok"}))
    assert credential_in_file(padded) is None


def test_raw_deflate_probing_does_not_read_the_whole_file(tmp_path):
    """Every file reaching the handler is probed, so reading it whole doubled a publish's memory.

    An extracted member may be as large as the uncompressed limit allows while the request body and
    the decoded tar are both still live. Feeding the decompressor in bounded blocks keeps the probe
    to one block plus whatever actually inflated.
    """
    import random
    import time
    import zlib

    from flash.env_secrets import _blocks_of, _credential_in_raw_deflate

    key = f"fslo_{_FAKE_KEY_BODY}".encode()
    raw = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    body = b"export FSLO_API_KEY=" + key + b"\n"
    sidecar = tmp_path / "payload.deflate"
    sidecar.write_bytes(raw.compress(body) + raw.flush())
    later = time.monotonic() + 120
    assert _credential_in_raw_deflate(sidecar, deadline=later, depth=1) == "a Freesolo API key"

    # the blocks really are bounded rather than one read of the whole file
    big = tmp_path / "shard.bin"
    big.write_bytes(random.Random(17).randbytes(3 << 20))
    sizes = [len(block) for block in _blocks_of(big)]
    assert len(sizes) > 1
    assert max(sizes) <= (1 << 20)
    # and a non-stream is still rejected rather than mistaken for deflate
    assert _credential_in_raw_deflate(big, deadline=later, depth=1) is None


def test_a_zlib_stream_at_any_level_is_recognised_behind_base64(tmp_path):
    """A cut base64 run holding a zlib stream is refused whatever level wrote it.

    `_CONTAINER_MAGIC` listed `x\x9c`, which is only zlib's DEFAULT level. `zlib.compress(data, 9)`
    writes `x\xda`, so a level-9 stream whose base64 crossed the 1 MiB scan chunk was not seen as a
    container, the truncation refusal never fired, and the credential published clean.
    """
    import zlib

    from flash.env_secrets import _SCAN_CHUNK_BYTES, _Unscannable, credential_in_file

    key = f"fslo_{_FAKE_KEY_BODY}".encode()
    for level in (1, 6, 9):
        raw = zlib.compress(random.Random(level).randbytes(1 << 20) + key, level)
        encoded = base64.b64encode(raw)
        assert len(encoded) > _SCAN_CHUNK_BYTES, (level, len(encoded))
        cut = tmp_path / f"cut{level}.txt"
        cut.write_bytes(b"DATA=" + encoded + b"\n")
        with pytest.raises(_Unscannable):
            credential_in_file(cut, deadline=time.monotonic() + 120)


def test_an_unpadded_base64_container_is_padded_rather_than_truncated(tmp_path):
    """A whole run is padded up, not cut back, so the container's last bytes survive.

    Trimming to the previous multiple of four discards up to three encoded characters, which for a
    zip are bytes of the end-of-central-directory record: the archive then would not open, whole-run
    inspection declined, and the credential inside published.
    """
    import zipfile

    from flash.env_secrets import credential_in_file

    key = f"fslo_{_FAKE_KEY_BODY}".encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("a.txt", random.Random(3).randbytes(20000).hex())
        archive.writestr("env.sh", b"export FSLO_API_KEY=" + key + b"\n")
        archive.writestr("b.txt", random.Random(4).randbytes(20000).hex())
    encoded = base64.b64encode(buf.getvalue()).rstrip(b"=")
    assert len(encoded) % 4, "fixture must be unpadded to exercise the padding"
    unpadded = tmp_path / "unpadded.txt"
    unpadded.write_bytes(b"DATA=" + encoded + b"\n")
    assert credential_in_file(unpadded, deadline=time.monotonic() + 120) == "a Freesolo API key"


def test_a_base64_container_at_the_depth_cap_is_refused(tmp_path):
    """The nesting cap refuses a decoded container rather than reporting it clean.

    Returning None at the cap switched the second look off, so four nested zips around
    `base64(gzip(secret))` published while the same gzip as an ordinary fifth container refused --
    making "encode the last layer" the way past the limit.
    """
    import gzip
    import zipfile

    from flash.env_secrets import _Unscannable, credential_in_file

    key = f"fslo_{_FAKE_KEY_BODY}".encode()
    inner = io.BytesIO()
    with gzip.GzipFile(fileobj=inner, mode="wb") as handle:
        handle.write(b"export FSLO_API_KEY=" + key + b"\n")

    def nest(payload: bytes, name: str) -> bytes:
        for layer in range(4):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(name, payload)
            payload, name = buf.getvalue(), f"layer{layer}.zip"
        return payload

    encoded = tmp_path / "encoded.zip"
    encoded.write_bytes(nest(base64.b64encode(inner.getvalue()), "inner.b64"))
    with pytest.raises(_Unscannable):
        credential_in_file(encoded, deadline=time.monotonic() + 120)

    # an ordinary file nested just as deeply still publishes: the cap refuses CONTAINERS, not depth
    plain = tmp_path / "plain.zip"
    plain.write_bytes(nest(b"just some text\n", "inner.txt"))
    assert credential_in_file(plain, deadline=time.monotonic() + 120) is None


def test_a_refusal_from_a_whole_run_base64_container_propagates(tmp_path):
    """An exact whole-run decode is not speculative, so its refusal is not swallowed.

    Swallowing every refusal from the second look kept an ELF's chance base64 runs publishable, but
    also turned a real encrypted zip behind base64 into a clean result while the same archive
    scanned directly was refused.
    """
    import struct
    import zipfile

    from flash.env_secrets import _Unscannable, credential_in_file

    key = f"fslo_{_FAKE_KEY_BODY}".encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("env.sh", b"export FSLO_API_KEY=" + key + b"\n")
    raw = bytearray(buf.getvalue())
    for signature, offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        at = raw.find(signature)
        flags = struct.unpack_from("<H", raw, at + offset)[0]
        struct.pack_into("<H", raw, at + offset, flags | 0x1)

    direct = tmp_path / "direct.zip"
    direct.write_bytes(bytes(raw))
    with pytest.raises(_Unscannable):
        credential_in_file(direct, deadline=time.monotonic() + 120)

    encoded = tmp_path / "encoded.txt"
    encoded.write_bytes(b"DATA=" + base64.b64encode(bytes(raw)) + b"\n")
    with pytest.raises(_Unscannable):
        credential_in_file(encoded, deadline=time.monotonic() + 120)


def test_a_private_key_armor_header_spanning_a_chunk_is_refused(tmp_path):
    """An armor still in its headers at the end of a chunk is undecided, not clean.

    The PEM pattern needs the BEGIN line and the start of the body in ONE buffer, and RFC 4880 puts
    no length limit on an armor header -- so a 1.5 MB `Comment:` pushed the body into the next scan
    chunk, the halves appeared in no single window, and a real armored secret key published.
    """
    from flash.env_secrets import _SCAN_CHUNK_BYTES, _Unscannable, credential_in_file

    body = base64.encodebytes(random.Random(11).randbytes(2048)).decode()
    # every eighth character is a dot, so no 32-character base64 run hides inside the header itself
    line = "Comment: " + ".".join("abc" for _ in range(30)) + "\n"
    header = line * 12000
    assert len(header) > _SCAN_CHUNK_BYTES, len(header)
    opening = "-----BEGIN PGP PRIVATE KEY BLOCK-----\n"
    closing = "-----END PGP PRIVATE KEY BLOCK-----\n"

    spanning = tmp_path / "spanning.asc"
    spanning.write_text(opening + header + "\n" + body + closing)
    with pytest.raises(_Unscannable):
        credential_in_file(spanning, deadline=time.monotonic() + 120)

    # the ordinary armored key is still reported as the key it is, not refused
    short = tmp_path / "short.asc"
    short.write_text(opening + line + "\n" + body + closing)
    assert credential_in_file(short, deadline=time.monotonic() + 120) == "a private key block"


def test_a_secret_key_packet_after_a_public_one_is_found(tmp_path):
    """An OpenPGP keyring is a SEQUENCE, so the secret packet need not lead it.

    `gpg --export` followed by `gpg --export-secret-keys` -- what a "back up my keys" one-liner
    writes -- leads with a public key block, so testing only the first packet passed the secret
    material behind it, and it matched no textual or DER pattern.
    """
    from flash.env_secrets import credential_in_file

    # laid out as gpg writes them: a public key packet (tag 6), a user id (tag 13), a signature
    # (tag 2), then the secret key packet (tag 5) that the walk has to reach.
    def packet(tag: int, body: bytes) -> bytes:
        return bytes([0x80 | (tag << 2) | 0x01]) + len(body).to_bytes(2, "big") + body

    key_body = b"\x04" + b"\x6a\x7e\x7e\x1e" + b"\x01" + b"\x00" * 16
    public = packet(6, key_body)
    user_id = packet(13, b"r16 test <r16@example.invalid>")
    signature = packet(2, b"\x00" * 40)
    secret = packet(5, key_body)

    keyring = tmp_path / "keyring.gpg"
    keyring.write_bytes(public + user_id + signature + secret)
    assert credential_in_file(keyring, deadline=time.monotonic() + 120) == "a private key"

    # a public-only keyring is exactly what is meant to be shared, and still publishes
    public_only = tmp_path / "public.gpg"
    public_only.write_bytes(public + user_id + signature)
    assert credential_in_file(public_only, deadline=time.monotonic() + 120) is None


def test_paired_marker_bodies_go_through_the_entropy_test(tmp_path):
    """Both halves of a two-marker credential filter placeholders, in either scan path.

    The netrc comment says placeholders are filtered, but nothing applied the entropy test to its
    captured body, so prose containing `Machine` and a masked `password XXXX...` was refused. The
    same gap paired a public JWK in one dataset row with an unrelated long `"d"` field in another.
    """
    from flash.env_secrets import _SCAN_CHUNK_BYTES, credential_in_file

    later = time.monotonic() + 120
    prose = tmp_path / "README.md"
    prose.write_bytes(
        b"Machine learning jobs use the cluster.\n" + b"password " + b"X" * 40 + b"\n"
    )
    assert credential_in_file(prose, deadline=later) is None

    rows = tmp_path / "rows.jsonl"
    rows.write_bytes(
        b'{"kty":"RSA","n":"public-only","e":"AQAB"}\n{"d":"documentation-document"}\n'
    )
    assert credential_in_file(rows, deadline=later) is None

    # both must still be found for real values, including past a chunk boundary where the
    # streaming path -- not `_match` -- is what pairs the halves.
    real_netrc = tmp_path / "netrc"
    real_netrc.write_bytes(
        b"machine api.example.com login bob password " + f"{_FAKE_KEY_BODY}".encode() + b"\n"
    )
    assert credential_in_file(real_netrc, deadline=later) == "a machine password in a netrc file"

    spanning = tmp_path / "spanning.json"
    scalar = base64.urlsafe_b64encode(random.Random(5).randbytes(32)).rstrip(b"=")
    spanning.write_bytes(
        b'{"kty":"RSA",' + b'"pad":"' + b"x" * _SCAN_CHUNK_BYTES + b'",' + b'"d":"' + scalar + b'"}'
    )
    assert credential_in_file(spanning, deadline=later) == "a private key"


def test_a_non_pdf_is_declined_from_its_signature(tmp_path, monkeypatch):
    """The PDF handler reads five bytes before it reads a file.

    Every top-level file reaches this handler after the other probes decline, so an unconditional
    `read_bytes` allocated a second whole copy of every ordinary model shard in the package.
    """
    from flash import env_secrets

    shard = tmp_path / "model.bin"
    shard.write_bytes(random.Random(9).randbytes(4 << 20))

    read_whole = []
    original = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self: (read_whole.append(self), original(self))[1],
    )
    assert env_secrets._credential_in_pdf(shard, deadline=time.monotonic() + 120, depth=0) is None
    assert read_whole == [], f"read the whole non-PDF: {read_whole}"

    # a real PDF still reaches the stream walk
    key = f"fslo_{_FAKE_KEY_BODY}".encode()
    import zlib

    payload = zlib.compress(b"export FSLO_API_KEY=" + key + b"\n")
    document = tmp_path / "doc.pdf"
    document.write_bytes(
        b"%PDF-1.4\n1 0 obj\n<< /Filter /FlateDecode /Length "
        + str(len(payload)).encode()
        + b" >>\nstream\n"
        + payload
        + b"\nendstream\nendobj\n"
    )
    found = env_secrets._credential_in_pdf(document, deadline=time.monotonic() + 120, depth=0)
    assert found == "a Freesolo API key"


def test_a_chance_container_in_prose_does_not_refuse(tmp_path):
    """A base64-shaped run inside prose stays speculative, so its refusal is swallowed.

    Propagating a refusal from any exactly-decoding run made two real hub datasets unpublishable: a
    9.7 MB JSONL of issue text holds base64-shaped words by chance, and one decoded to bytes
    beginning `x\\x9c` with the FDICT bit set, so the dictionary refusal fired on bytes nobody
    encoded. Bounding characters cannot separate the two -- prose is full of them, and a real
    `zip -P` archive is 142 bytes, SMALLER than the accidental runs -- so the test is whether the
    run was ASSIGNED.
    """
    import zlib

    from flash.env_secrets import _Unscannable, credential_in_file

    # a REAL preset-dictionary stream. Setting the FDICT bit by hand does not work: the two header
    # bytes must stay a multiple of 31, so flipping one breaks the check that identifies zlib at
    # all and the refusal is never reached. zlib writes both the flag and a valid checksum here.
    compressor = zlib.compressobj(
        6,
        zlib.DEFLATED,
        zlib.MAX_WBITS,
        zlib.DEF_MEM_LEVEL,
        zlib.Z_DEFAULT_STRATEGY,
        b"the quick brown fox",
    )
    stream = compressor.compress(b"the quick brown fox jumps over the lazy dog" * 4)
    stream += compressor.flush()
    assert stream[1] & 0x20, "fixture must carry the FDICT flag"
    run = base64.b64encode(stream).decode().rstrip("=")

    later = time.monotonic() + 120
    prose = tmp_path / "issues.jsonl"
    prose.write_text(
        '{"title":"crash on startup","body":"the traceback mentions ' + run + ' in the log"}\n'
    )
    assert credential_in_file(prose, deadline=later) is None

    # the same bytes ASSIGNED are a value someone encoded, and stay refused
    assigned = tmp_path / "config.env"
    assigned.write_text("PAYLOAD=" + run + "\n")
    with pytest.raises(_Unscannable):
        credential_in_file(assigned, deadline=later)


def test_raw_deflate_stops_when_its_output_budget_reaches_zero():
    """A block boundary landing exactly on the budget must not inflate the next block whole.

    `max_length=0` means UNLIMITED to `zlib.decompressobj().decompress`, not "no output", so a
    budget computed as `max(0, budget - produced)` handed the decompressor a blank cheque exactly
    when it was out of room. Measured 11,024 bytes returned under a 1,024-byte budget, which is the
    buffer cap the bound exists to enforce.
    """
    import zlib

    from flash.env_deflate import _raw_deflate_from

    budget = 1024
    compressor = zlib.compressobj(6, zlib.DEFLATED, -zlib.MAX_WBITS)
    # the first block ends, flushed, at exactly the budget; the second is the over-budget payload
    first = compressor.compress(b"A" * budget) + compressor.flush(zlib.Z_SYNC_FLUSH)
    second = compressor.compress(b"B" * 10000) + compressor.flush()

    assert _raw_deflate_from(iter((first, second)), budget) is None

    # the bound still lets a stream that FITS through, so the fix is not "refuse everything"
    small = zlib.compressobj(6, zlib.DEFLATED, -zlib.MAX_WBITS)
    fits = small.compress(b"hello world" * 10) + small.flush()
    assert _raw_deflate_from(iter((fits,)), budget) == b"hello world" * 10


def test_a_filter_applied_after_flatedecode_is_undone(tmp_path):
    """`/Filter [/FlateDecode /ASCII85Decode]` inflates TO ASCII85 text, not to the key.

    Filters after the flate stage were left alone on the reasoning that the inflated bytes are
    scanned anyway. That holds for a layer this scans through; ASCII85 re-encodes the key's own
    bytes, so the scan read printable noise and the credential published.
    """
    import zlib

    from flash.env_secrets import credential_in_file

    key = "fslo_" + _FAKE_KEY_BODY
    encoded = base64.a85encode(key.encode(), adobe=False) + b"~>"
    stream = zlib.compress(encoded)
    document = (
        b"%PDF-1.4\n1 0 obj\n<< /Length "
        + str(len(stream)).encode()
        + b" /Filter [/FlateDecode /ASCII85Decode] >>\nstream\n"
        + stream
        + b"\nendstream\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n"
    )
    chained = tmp_path / "chained.pdf"
    chained.write_bytes(document)
    assert credential_in_file(chained, deadline=time.monotonic() + 120) == "a Freesolo API key"

    # an ordinary flate-only document is still clean, so the walk did not become indiscriminate
    plain = zlib.compress(b"just the ordinary text of a document")
    innocent = tmp_path / "innocent.pdf"
    innocent.write_bytes(
        b"%PDF-1.4\n1 0 obj\n<< /Length "
        + str(len(plain)).encode()
        + b" /Filter /FlateDecode >>\nstream\n"
        + plain
        + b"\nendstream\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n"
    )
    assert credential_in_file(innocent, deadline=time.monotonic() + 120) is None


def test_a_chance_zlib_header_is_not_read_whole(tmp_path, monkeypatch):
    """Bytes that merely satisfy the zlib header rule must not be copied entire to disprove it.

    The rule is about eleven bits, so roughly one file in 2,000 trips it by chance, and a member may
    be 256 MiB with the request body and staged file already live. A bounded prefix decides it.
    """
    import zlib

    from flash.env_secrets import _Unscannable, credential_in_file

    later = time.monotonic() + 120
    accidental = tmp_path / "shard.bin"
    accidental.write_bytes(b"\x78\x9c" + b"ordinary binary content, not deflate at all " * 500)

    reads: list[str] = []
    original = Path.read_bytes
    monkeypatch.setattr(
        Path, "read_bytes", lambda self: (reads.append(self.name), original(self))[1]
    )
    assert credential_in_file(accidental, deadline=later) is None
    assert "shard.bin" not in reads, "the whole member was read to disprove a chance header"

    # a REAL stream is still expanded and its credential found
    real = tmp_path / "real.z"
    real.write_bytes(zlib.compress(("fslo_" + _FAKE_KEY_BODY).encode()))
    assert credential_in_file(real, deadline=later) == "a Freesolo API key"

    # and the dictionary refusal, which cannot inflate at all, still fires
    compressor = zlib.compressobj(
        6, zlib.DEFLATED, zlib.MAX_WBITS, zlib.DEF_MEM_LEVEL, zlib.Z_DEFAULT_STRATEGY, b"fox"
    )
    stream = compressor.compress(b"the quick brown fox jumps over the lazy dog" * 4)
    stream += compressor.flush()
    assert stream[1] & 0x20, "fixture must carry the FDICT flag"
    dictionary = tmp_path / "dict.z"
    dictionary.write_bytes(stream)
    with pytest.raises(_Unscannable):
        credential_in_file(dictionary, deadline=later)


def test_a_secret_key_packet_past_the_first_chunk_is_not_reported_clean(tmp_path):
    """A keyring whose secret packet sits beyond the scan chunk must not publish.

    The sequence walk runs on the FIRST chunk alone, and it stopped as soon as a packet declared
    more body than was in hand -- which is exactly what a well-formed keyring looks like when a
    large early packet crosses the boundary. Measured with a real GnuPG export: `gpg --import`
    installed the secret key from bytes this reported clean.
    """
    from flash.env_buffers import _SCAN_CHUNK_BYTES
    from flash.env_secrets import _Unscannable, credential_in_file

    # a public key packet, then filler, then the secret key: the order `gpg --export` followed by
    # `gpg --export-secret-keys` writes, which is what a "back up my keys" one-liner produces.
    public = b"\x99\x00\x10" + b"\x04" + b"\x6a\x7e\x7e\x1e" + b"\x01" + b"\x00" * 10
    secret = b"\x95\x03\x98\x04" + b"\x6a\x7e\x7e\x1e" + b"\x01" + b"\x00" * 16
    # old-format tag 12, four-byte length: legal, parsed, and long enough to span the chunk.
    filler = lambda size: b"\xb2" + size.to_bytes(4, "big") + b"\x00" * size  # noqa: E731

    compact = tmp_path / "compact.pgp"
    compact.write_bytes(public + filler(16) + secret)
    assert credential_in_file(compact) == "a private key"

    spanning = tmp_path / "spanning.pgp"
    spanning.write_bytes(public + filler(_SCAN_CHUNK_BYTES + 4096) + secret)
    with pytest.raises(_Unscannable, match="cannot walk to the end"):
        credential_in_file(spanning)

    # a keyring cut short at TRUE end of file has nothing unread behind it, so refusing it would
    # fail a publish over bytes that demonstrably hold no key.
    truncated = tmp_path / "truncated.pgp"
    truncated.write_bytes(public + b"\xb2" + (4096).to_bytes(4, "big") + b"\x00" * 32)
    assert credential_in_file(truncated) is None


def test_ordinary_binary_is_not_refused_as_an_unfinishable_openpgp_sequence():
    """Random bytes must not be walked as a packet sequence and then refused for outrunning it.

    The walk entered on bit 7 of the first byte alone -- set on HALF of all random bytes -- and the
    next four bytes were then read as a body length, averaging two gigabytes. That declared body
    ran past the buffer, which the caller reports as "cannot walk to the end" and refuses. Measured
    at 18.8% of random 4 KiB blocks: roughly one upload in five blocked over nothing, and the shape
    is ordinary (a model shard, a random-padded archive member), not crafted.

    Two things were missing and BOTH are needed. The tag must name a packet type the format
    defines, which removes the new-format half. That is not sufficient on its own: an old-format
    tag is four bits, so a random byte names a real type most of the time -- the payload that
    actually failed CI was `0x92`, a legal one-pass signature header -- so a body larger than the
    scan could ever hold is also required to end the walk rather than report it undecided.

    Asserted at the chunk size the scan really reads. A 4 KiB probe still shows a residual from the
    two-byte length form, which can only claim 64 KB and so cannot outrun a megabyte chunk.

    The bodies are drawn from a SEEDED generator, and the rate is asserted with a margin rather than
    at zero. A megabyte of random bytes lands on a walkable header about twice in a thousand -- a
    real residual, measured at 0.20% before any of this and 0.23% after -- so 64 unseeded draws fail
    roughly one run in seven, on a property that has not regressed. Seeding makes the verdict a
    function of the code alone; the margin is what the assertion is actually about, since the point
    is that ordinary binary is not refused WHOLESALE, as it was at 18.8%.
    """
    import random

    from flash.env_buffers import _SCAN_CHUNK_BYTES
    from flash.env_openpgp import _openpgp_secret_key_in_sequence

    bodies = random.Random(20260815)
    draws = 256
    refused = sum(
        1
        for _ in range(draws)
        if _openpgp_secret_key_in_sequence(bodies.randbytes(_SCAN_CHUNK_BYTES), truncated=True)
        is None
    )
    assert refused <= draws // 64, f"{refused}/{draws} random chunks refused as unfinishable"

    # the shape that failed CI, pinned exactly: an old-format one-pass signature tag whose 4-byte
    # length declares a body no scan holds. Undecided here is what refused the publish.
    implausible = b"\x92" + (1 << 31).to_bytes(4, "big") + bodies.randbytes(4096)
    assert _openpgp_secret_key_in_sequence(implausible, truncated=True) is False

    # and a body that IS plausible still reports undecided, which is the property being preserved:
    # the fix must not turn "a real packet I could not finish reading" into a confident clean.
    spanning = b"\x92" + (1 << 20).to_bytes(4, "big") + bodies.randbytes(4096)
    assert _openpgp_secret_key_in_sequence(spanning, truncated=True) is None


def test_an_openssl_envelope_written_in_base64_is_refused(tmp_path):
    """`openssl enc -a` is the same ciphertext as `openssl enc`, and just as unreadable.

    The binary form was refused at the anchored format check, but the base64 form reached the
    decoded-bytes hook, which asked only whether the decode looked like a CONTAINER. An encrypted
    envelope is not a container, so it returned None and a whole Freesolo key published.
    """
    from flash.env_secrets import _Unscannable, credential_in_file

    # `Salted__`, an 8-byte salt, then ciphertext: the layout `openssl enc -salt` writes.
    envelope = b"Salted__" + bytes(range(8)) + bytes(range(256)) * 2

    binary = tmp_path / "enc.bin"
    binary.write_bytes(envelope)
    with pytest.raises(_Unscannable, match="OpenSSL-encrypted"):
        credential_in_file(binary)

    # the sidecar form: one encoded blob and the trailing newline every text file carries
    sidecar = tmp_path / "enc.b64"
    sidecar.write_bytes(base64.b64encode(envelope) + b"\n")
    with pytest.raises(_Unscannable, match="OpenSSL-encrypted"):
        credential_in_file(sidecar)

    # and as an assigned value, which is how a Kubernetes Secret carries one
    assigned = tmp_path / "secret.yaml"
    assigned.write_bytes(b"data:\n  creds: " + base64.b64encode(envelope) + b"\n")
    with pytest.raises(_Unscannable, match="OpenSSL-encrypted"):
        credential_in_file(assigned)

    # prose ABOUT the format is not an envelope, and an ordinary encoded file is still publishable
    prose = tmp_path / "notes.md"
    prose.write_text("the `Salted__` header precedes the salt.\n" * 20)
    assert credential_in_file(prose) is None
    harmless = tmp_path / "config.b64"
    harmless.write_bytes(base64.b64encode(b"ordinary configuration text\n" * 8) + b"\n")
    assert credential_in_file(harmless) is None


def test_a_pdf_stream_beyond_the_dictionary_gap_is_refused(tmp_path):
    """A PDF object may carry any amount of metadata before its `stream` keyword.

    The filter name and the keyword were paired within a fixed gap, on the convention that a stream
    dictionary is short. Padding one past that bound hid the stream entirely, and the credential in
    it published while the compact form of the same document was caught.
    """
    import zlib

    from flash.env_secrets import _Unscannable, credential_in_file

    body = zlib.compress(f"FREESOLO_API_KEY=fslo_{_FAKE_KEY_BODY}\n".encode())

    def document(dictionary: bytes) -> bytes:
        return (
            b"%PDF-1.4\n1 0 obj\n<< /Filter /FlateDecode "
            + dictionary
            + b" /Length "
            + str(len(body)).encode()
            + b" >>\nstream\n"
            + body
            + b"\nendstream\nendobj\n%%EOF\n"
        )

    compact = tmp_path / "compact.pdf"
    compact.write_bytes(document(b""))
    assert credential_in_file(compact) == "a Freesolo API key"

    padded = tmp_path / "padded.pdf"
    padded.write_bytes(document(b"/Meta (" + b"z" * 600 + b")"))
    with pytest.raises(_Unscannable, match="could not locate"):
        credential_in_file(padded)

    # a document that merely MENTIONS the filter -- and happens to contain the word `stream` later
    # -- has no dictionary to close, so it must stay publishable at any distance.
    prose = tmp_path / "prose.pdf"
    prose.write_bytes(
        b"%PDF-1.4\n% the /FlateDecode filter is the usual one.\n"
        + b"x" * 4000
        + b"\nstream\nnot an object\n%%EOF\n"
    )
    assert credential_in_file(prose) is None

    # an ordinary PDF whose streams all pair normally is not refused over the surplus test
    harmless = zlib.compress(b"BT /F1 12 Tf (hello) Tj ET\n" * 40)
    ordinary = tmp_path / "ordinary.pdf"
    ordinary.write_bytes(
        b"%PDF-1.4\n1 0 obj\n<< /Filter /FlateDecode /Length "
        + str(len(harmless)).encode()
        + b" >>\nstream\n"
        + harmless
        + b"\nendstream\nendobj\n%%EOF\n"
    )
    assert credential_in_file(ordinary) is None


def test_a_private_key_armor_with_extension_headers_is_found(tmp_path):
    """RFC 4880 permits ANY header name in an armored block, so a name list cannot gate the body.

    The armor pattern reached the base64 body directly, which a real `gpg --export-secret-keys
    --armor` block does not: it writes headers, and the blank line after them is what separates the
    two. Requiring a known header NAME instead was the same mistake one layer in -- the standard
    puts no limit on which names appear, so an unrecognised one hid the key.
    """
    from flash.env_secrets import credential_in_file

    body = "\n".join([_FAKE_KEY_BODY] * 6)

    def armored(headers: str) -> bytes:
        return f"-----BEGIN PGP PRIVATE KEY BLOCK-----\n{headers}\n{body}\n".encode()

    # the shape GnuPG writes: a recognised header, then a blank line, then the body
    known = tmp_path / "known.asc"
    known.write_bytes(armored("Version: GnuPG v2\n"))
    assert credential_in_file(known) == "a private key block"

    # an extension header nobody has an allowlist entry for is still an armored private key
    extended = tmp_path / "extended.asc"
    extended.write_bytes(armored("X-Custom-Exporter: acme-backup/3\nComment: nightly\n"))
    assert credential_in_file(extended) == "a private key block"

    # a headerless block puts the body on the line straight after BEGIN, with no blank line at all
    headerless = tmp_path / "bare.asc"
    headerless.write_bytes(f"-----BEGIN PRIVATE KEY-----\n{body}\n".encode())
    assert credential_in_file(headerless) == "a private key block"

    # prose naming the marker, with no body behind it, is not a key
    mention = tmp_path / "notes.md"
    mention.write_text("paste the -----BEGIN PGP PRIVATE KEY BLOCK----- line here.\n")
    assert credential_in_file(mention) is None


def test_a_certificate_only_pkcs12_is_not_reported_as_a_private_key(tmp_path):
    """A `.p12` holding only certificates carries no key, and refusing it fails a real publish.

    Both shapes are PBES2-encrypted, so the encryption OID cannot tell them apart. The key bag OID
    can: `pkcs8ShroudedKeyBag` is present exactly when a key is, and the certificate bag OID sits
    INSIDE the encrypted SafeContents where nothing can see it.
    """
    from flash.env_secrets import credential_in_file

    # The DER body both files share: a PKCS#8 RSA PrivateKeyInfo, which is what the private-key
    # pattern matches on. Built rather than asserted about, so the test exercises the PFX wrapper
    # around a body already known to match rather than a hand-guessed byte string.
    body = b"\x02\x01\x00\x30\x0d\x06\x09\x2a\x86\x48\x86\xf7\x0d\x01\x01\x01" + bytes(range(200))
    # the PFX preamble openssl writes: SEQUENCE, a two-byte long-form length, then the version 3
    # INTEGER PKCS#12 requires
    preamble = b"\x30\x82\x09\xdf" + b"\x02\x01\x03"
    shrouded_key_bag = b"\x2a\x86\x48\x86\xf7\x0d\x01\x0c\x0a\x01\x02"
    pbes2 = b"\x2a\x86\x48\x86\xf7\x0d\x01\x05\x0d"

    # the body alone, with no PFX wrapper, is a private key and stays one
    plain_der = tmp_path / "key.der"
    plain_der.write_bytes(body)
    assert credential_in_file(plain_der) == "a private key"

    with_key = tmp_path / "bundle.p12"
    with_key.write_bytes(preamble + pbes2 + shrouded_key_bag + body)
    assert credential_in_file(with_key) == "a private key"

    # the same envelope with no key bag: a certificate chain, which is public material. Both files
    # carry PBES2, so the encryption OID cannot be what decides.
    certificates_only = tmp_path / "chain.p12"
    certificates_only.write_bytes(preamble + pbes2 + body)
    assert credential_in_file(certificates_only) is None


def test_a_new_format_packet_length_is_decoded_for_every_tag(tmp_path):
    """Which length ENCODING applies is decided by bit 6 of the tag byte, not by the tag.

    Naming only the two new-format secret-key tags meant every other new-format packet fell to the
    old-format branch, which reads the length as one big-endian integer. That coincidentally agreed
    for the one-byte form and was wrong for the other two: a five-byte length returned a nonsense
    trillion-byte body assembled from four bytes of the packet's own payload, so the walk stopped
    and a secret key behind a modern public-key or literal packet was reported clean.
    """
    from flash.env_openpgp import _openpgp_body_length
    from flash.env_secrets import credential_in_file

    # a new-format PUBLIC key packet (tag 6), which is what Sequoia, RNP and
    # `--use-new-packet-format` write, across all three length encodings RFC 4880 defines
    for length_bytes, declared in (
        (bytes([30]), 30),
        (bytes([193, 52]), 500),
        (b"\xff" + (70000).to_bytes(4, "big"), 70000),
    ):
        head = bytes([0xC6]) + length_bytes + b"\x00" * 16
        first = head[1]
        offset = 2 if first < 192 else (3 if first < 224 else (6 if first == 0xFF else 0))
        assert _openpgp_body_length(head, offset) == declared, length_bytes

    secret = b"\x95\x03\x98\x04" + b"\x6a\x7e\x7e\x1e" + b"\x01" + b"\x00" * 16

    # a keyring whose leading packet uses the new format: the secret key behind it is still found.
    # The declared length must equal the bytes actually carried, or the walk lands mid-packet.
    body = b"\x04" + b"\x6a\x7e\x7e\x1e" + b"\x01" + b"\x00" * 25
    modern = tmp_path / "modern.gpg"
    modern.write_bytes(bytes([0xC6, len(body)]) + body + secret)
    assert credential_in_file(modern) == "a private key"

    # the same through a new-format LITERAL packet using the five-byte length form, which is the
    # encoding the old branch got most wrong
    filler = b"\x00" * 16
    literal = tmp_path / "literal.gpg"
    literal.write_bytes(bytes([0xCB, 0xFF]) + len(filler).to_bytes(4, "big") + filler + secret)
    assert credential_in_file(literal) == "a private key"


def test_jwk_markers_pair_within_one_record_not_across_a_dataset(tmp_path):
    """Two harmless JSONL rows are not one private key.

    The halves were paired across the whole stream, so a dataset holding a PUBLIC JWK in one row
    and an ordinary high-entropy string under a private member name in another -- a build id, a
    timestamped artifact name -- had both markers present and neither row held a key. The publish
    was refused over a file with no credential in it, and the entropy test cannot separate the two
    because a build id scores exactly as random as a key body does.
    """
    from flash.env_secrets import credential_in_file

    unrelated = tmp_path / "shard.jsonl"
    unrelated.write_bytes(
        b'{"kty":"RSA","n":"public-only","e":"AQAB"}\n{"d":"2026-08-14T12-34-56Z-build-123456"}\n'
    )
    assert credential_in_file(unrelated) is None

    # the same two markers in ONE object is a real key and still refuses
    together = tmp_path / "key.jwk"
    together.write_bytes(b'{"kty":"RSA","n":"public-only","d":"%s"}' % _FAKE_KEY_BODY.encode())
    assert credential_in_file(together) == "a private key"


def test_a_jwk_pairs_across_any_distance_inside_one_object(tmp_path):
    """Record scoping must not reintroduce the window it replaced.

    JWK members may sit any distance apart with arbitrary extension members between them, so a real
    key with megabytes of metadata between `kty` and `d` -- either way round, and spanning several
    read chunks -- has to pair. A boundary that split on distance rather than on the enclosing
    object would publish exactly that key.
    """
    from flash.env_secrets import credential_in_file

    padding = b'"pad":"' + b"m" * (3 << 20) + b'"'
    for name, body in (
        ("forward.jwk", b'{"kty":"RSA",' + padding + b',"d":"%s"}' % _FAKE_KEY_BODY.encode()),
        ("reverse.jwk", b'{"d":"%s",' % _FAKE_KEY_BODY.encode() + padding + b',"kty":"RSA"}'),
        # nested members must not close the record: only a brace back to depth zero ends it
        ("nested.jwk", b'{"kty":"RSA","x":{"y":{"z":1}},"d":"%s"}' % _FAKE_KEY_BODY.encode()),
        # a brace inside a STRING is not a boundary, or a crafted value splits a real key
        ("quoted.jwk", b'{"kty":"RSA","c":"} {","d":"%s"}' % _FAKE_KEY_BODY.encode()),
        # a format with no braces at all is one record, which is what a netrc relies on
        ("bare.jwk", b'"kty":"RSA"\n"d":"%s"\n' % _FAKE_KEY_BODY.encode()),
    ):
        published = tmp_path / name
        published.write_bytes(body)
        assert credential_in_file(published) == "a private key", name


def test_record_boundaries_survive_a_value_split_by_a_read_boundary(tmp_path):
    """Rows must not merge because a quoted value straddles the chunk the scan reads in.

    The record splitter carries its state between windows, and only the depth crossed. A string
    spanning the cut left the next window resuming half a string out: it read that value's CLOSING
    quote as an opening one, mispaired every quote after it, and matched `"}\\n{"` as a single
    string token -- so the brace between two JSONL rows vanished and they merged into one record.
    A public JWK in one row then paired with an ordinary high-entropy `d` in the other as a private
    key that neither row held, on a file that scanned clean in a single buffer.

    Swept over where the padding row ENDS rather than tested at one offset: what matters is which
    byte of the value the cut lands on, and the desync only reaches the rows behind it when the
    quote that closes the padding sits past that point. Escaped padding is swept too, in whole
    pairs -- a value truncated mid-pair ends on a lone backslash that escapes its own closing
    quote, and the string is then genuinely unterminated in any buffering, so refusing it would be
    correct JSON reading rather than the defect under test.
    """
    from flash.env_secrets import _SCAN_CHUNK_BYTES, _SCAN_OVERLAP_BYTES, credential_in_file

    public = b'{"kty":"EC","crv":"P-256","x":"%s","y":"%s"}' % (
        _FAKE_KEY_BODY.encode(),
        _FAKE_KEY_BODY.encode(),
    )
    # an ordinary high-entropy value under a private member name, in a row of its own: a build id
    # scores exactly as random as a key body, so only the record boundary separates them
    unrelated = b'{"note":"nightly","d":"%s"}' % _FAKE_KEY_BODY.encode()
    opener = b'{"pad":"'
    fillers = (b"A", b'\\"', b"\\\\")
    offsets = (-600, -8, 0, 8, 600, 900, 1200)
    for filler in fillers:
        for offset in offsets:
            width = _SCAN_CHUNK_BYTES - _SCAN_OVERLAP_BYTES + offset - len(opener)
            value = (filler * width)[: width // len(filler) * len(filler)]
            published = tmp_path / "shard.jsonl"
            published.write_bytes(opener + value + b'"}\n' + public + b"\n" + unrelated + b"\n")
            assert credential_in_file(published) is None, (filler, offset)

    # the same two markers inside ONE row is a real key and must still refuse, at every offset
    for filler in fillers:
        for offset in offsets:
            width = _SCAN_CHUNK_BYTES - _SCAN_OVERLAP_BYTES + offset - len(opener)
            value = (filler * width)[: width // len(filler) * len(filler)]
            together = tmp_path / "key.jsonl"
            together.write_bytes(
                opener
                + value
                + b'"}\n'
                + b'{"kty":"EC","crv":"P-256","d":"%s"}\n' % _FAKE_KEY_BODY.encode()
            )
            assert credential_in_file(together) == "a private key", (filler, offset)


def test_a_netrc_entry_spanning_lines_is_still_one_record(tmp_path):
    """A `.netrc` writes one entry over several lines, and both halves are in it.

    Splitting records at the newline read like the separator of a line-delimited file, but a netrc
    entry is `machine`, `login` and `password` on three separate lines -- so it put the two halves
    in different records and published the key.
    """
    from flash.env_secrets import credential_in_file

    entry = b"machine api.wandb.ai\n  login user\n  password %s\n" % _FAKE_KEY_BODY.encode()
    for name, body in (
        (".netrc", entry),
        # a brace ANYWHERE in the file is what makes the boundary scan run at all, so the
        # line-spanning entry has to survive it too -- a comment naming `${HOME}` is enough, and
        # so is an unrelated JSON line in the same file.
        ("commented.netrc", b"# see ${HOME}/.netrc\n" + entry),
        ("mixed.netrc", b'{"note":"creds below"}\n' + entry),
    ):
        published = tmp_path / name
        published.write_bytes(body)
        assert credential_in_file(published) == "a machine password in a netrc file", name


def test_the_wrapped_block_guard_is_linear_in_the_run_it_rejects(tmp_path):
    """The cheap guard in front of the joining pattern must not be quadratic.

    Widening it to accept narrower columns as an open-ended run made it take the longest match at
    every start position and backtrack it away one character at a time when no break followed:
    53 seconds on 100 KB of one long non-matching run, and hours on the megabyte chunks the scan
    actually reads. A single unbroken run of base64 characters is an ordinary thing for a file to
    contain, so this is reached without anyone crafting it.
    """
    import time

    from flash.env_base64 import _WRAPPED_HINT

    # one long run with no break after it: the shape that has no match to find
    haystack = b'{"pad":"' + b"z" * (1 << 20) + b'"}\n'
    started = time.monotonic()
    assert _WRAPPED_HINT.search(haystack) is None
    assert time.monotonic() - started < 5

    # and it still recognises the blocks it was widened for, at every column in the range
    for width in (32, 64, 76, 128):
        wrapped = (b"a" * width + b"\n") * 2
        assert _WRAPPED_HINT.search(wrapped), width
    assert _WRAPPED_HINT.search(b"a" * 31 + b"\n") is None


def test_an_overlay_search_does_not_reprobe_identical_candidates(tmp_path):
    """Overlapping magics are the same probe repeated, not a search.

    A file of adjacent gzip magics presents one candidate every three bytes, and each used to reopen
    the file, re-read 64 KiB and run a decompressor over it: 349,522 probes and 21 GB of reads for
    one megabyte of upload, which one authenticated publish could spend a worker's time on. Those
    candidates are 22 DISTINCT probes.

    Deduplicating cannot hide a payload -- two candidates with identical leading bytes decode
    identically -- which is why the cap itself stays where it was, applied per window rather than
    inside one.
    """
    import time

    from flash.env_formats import OVERLAY_UNPROBED, _overlay_offset

    spam = tmp_path / "spam.run"
    spam.write_bytes(b"#!/bin/sh\n" + b"\x1f\x8b\x08" * 350_000)
    started = time.monotonic()
    assert _overlay_offset(spam) == OVERLAY_UNPROBED
    assert time.monotonic() - started < 10


def _flate_pdf(dictionary: bytes, body: bytes, eol: bytes = b"\r\n") -> bytes:
    """A one-object PDF whose stream carries `body`, for the filter-chain tests below."""
    return (
        b"%PDF-1.7\n1 0 obj\n<< "
        + dictionary
        + b" /Length "
        + str(len(body)).encode()
        + b" >>\nstream"
        + eol
        + body
        + b"\nendstream\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n"
    )


def test_a_pdf_stream_is_found_after_every_legal_line_ending(tmp_path):
    """PDF allows CRLF, LF or a bare CR after the `stream` keyword.

    Matching only the first two left a CR-wrapped document's stream unrecognised, so its deflated
    bytes were scanned as opaque content and the credential inside published. All three are legal
    and generators emit all three.
    """
    import zlib

    from flash.env_secrets import credential_in_file

    body = zlib.compress(b'export FREESOLO_API_KEY="fslo_%s"\n' % _FAKE_KEY_BODY.encode())
    for name, eol in (("cr.pdf", b"\r"), ("lf.pdf", b"\n"), ("crlf.pdf", b"\r\n")):
        published = tmp_path / name
        published.write_bytes(_flate_pdf(b"/Filter /FlateDecode", body, eol=eol))
        assert credential_in_file(published) == "a Freesolo API key", name


def test_a_pdf_filter_name_written_with_hex_escapes_is_still_flate(tmp_path):
    """`/Flate#44ecode` IS `/FlateDecode`: `#44` is the PDF escape for `D` in a name.

    The rule applies to every character of every name, so one filter has 4096 legal spellings and a
    reader resolves all of them identically. Matching the literal characters recognised exactly one,
    which left the stream unrecognised, scanned as opaque bytes, and published with its credential
    inside -- and a document is trivially rewritten into a spelling nobody matches.
    """
    import zlib

    from flash.env_secrets import _Unscannable, credential_in_file

    body = zlib.compress(b'export FREESOLO_API_KEY="fslo_%s"\n' % _FAKE_KEY_BODY.encode())
    for name, spelling in (
        ("named.pdf", b"/Flate#44ecode"),
        ("first.pdf", b"/#46lateDecode"),
        ("last.pdf", b"/FlateDecod#65"),
        ("upper.pdf", b"/F#6CateDecode"),
        # hex digits are case-insensitive in a name escape, so the lowercase form is the same name
        ("lower.pdf", b"/F#6cateDecode"),
        ("every.pdf", b"/" + b"".join(b"#%02X" % c for c in b"FlateDecode")),
        ("chain.pdf", b"[/ASCII85Decode /Flate#44ecode]"),
    ):
        published = tmp_path / name
        stream = base64.a85encode(body) if spelling.startswith(b"[") else body
        published.write_bytes(_flate_pdf(b"/Filter " + spelling, stream))
        assert credential_in_file(published) == "a Freesolo API key", name

    # resolving escapes must not turn a DIFFERENT filter into Flate, and a name that merely starts
    # with the same characters is still a different name. None of these is read as flate, so each
    # is a chain this cannot undo -- refused rather than reported clean, which is the same answer
    # every other uninspectable stream gets.
    for name, spelling in (
        ("dct.pdf", b"/DCTDecode"),
        ("lzw.pdf", b"/LZWDecode"),
        ("longer.pdf", b"/FlateDecodeX"),
        ("shorter.pdf", b"/Flate"),
        ("case.pdf", b"/flatedecode"),
    ):
        other = tmp_path / name
        other.write_bytes(_flate_pdf(b"/Filter " + spelling, body))
        with pytest.raises(_Unscannable, match="cannot undo"):
            credential_in_file(other)


def test_a_pdf_predictor_refuses_rather_than_reading_differences(tmp_path):
    """A predictor is applied to the INFLATED bytes, so zlib's output is not the content.

    The stream inflates successfully and holds none of its own literal bytes -- they are horizontal
    or PNG differences -- so scanning what came out of zlib found nothing and the key published.
    Undoing it needs the colour and column parameters, so the honest answer is undecided.
    """
    import zlib

    from flash.env_secrets import _Unscannable, credential_in_file

    body = zlib.compress(b'export FREESOLO_API_KEY="fslo_%s"\n' % _FAKE_KEY_BODY.encode())
    predicted = tmp_path / "predicted.pdf"
    predicted.write_bytes(
        _flate_pdf(b"/Filter /FlateDecode /DecodeParms << /Predictor 12 /Columns 4 >>", body)
    )
    with pytest.raises(_Unscannable):
        credential_in_file(predicted)

    # `/Predictor 1` is the identity, so it must NOT refuse -- a document that names it is
    # ordinary, and refusing on the parameter's presence would fail a legitimate publish.
    identity = tmp_path / "identity.pdf"
    identity.write_bytes(_flate_pdf(b"/Filter /FlateDecode /DecodeParms << /Predictor 1 >>", body))
    assert credential_in_file(identity) == "a Freesolo API key"


def test_an_indirect_pdf_filter_reference_refuses(tmp_path):
    """`/Filter 2 0 R` names its filter through another object.

    A pattern matching the filter NAME directly never associated the stream with flate, so its
    deflated bytes were scanned as opaque content and the credential published. Resolving the
    reference means parsing the xref table; refusing is the bounded answer.
    """
    import zlib

    from flash.env_secrets import _Unscannable, credential_in_file

    body = zlib.compress(b'export FREESOLO_API_KEY="fslo_%s"\n' % _FAKE_KEY_BODY.encode())
    indirect = tmp_path / "indirect.pdf"
    indirect.write_bytes(_flate_pdf(b"/Filter 2 0 R", body))
    with pytest.raises(_Unscannable):
        credential_in_file(indirect)


def test_an_encrypted_pdf_refuses_rather_than_skipping_its_streams(tmp_path):
    """An encrypted PDF reverses stream encryption BEFORE the declared filters.

    What follows `stream` is ciphertext, so zlib rejects it -- which the skip treated as "not
    really a stream" and the document passed as clean. That made an encrypted PDF the one container
    shape this let through, while encrypted zip, OpenSSL and OpenPGP payloads are all refused.
    """
    import zlib

    from flash.env_secrets import _Unscannable, credential_in_file

    body = zlib.compress(b'export FREESOLO_API_KEY="fslo_%s"\n' % _FAKE_KEY_BODY.encode())
    encrypted = tmp_path / "encrypted.pdf"
    encrypted.write_bytes(
        b"%PDF-1.7\ntrailer\n<< /Encrypt 5 0 R /Root 1 0 R >>\n"
        + _flate_pdf(b"/Filter /FlateDecode", body)
    )
    with pytest.raises(_Unscannable):
        credential_in_file(encrypted)


def test_an_encrypt_key_ends_at_any_legal_separator(tmp_path):
    """A PDF name ends at whitespace, at a delimiter, or at the `%` that opens a comment.

    Requiring whitespace meant `/Encrypt%c\\n2 0 R` named no encryption dictionary, so the document
    was treated as unencrypted: its ciphertext streams went to the declared filters, failed to
    inflate, and were skipped as "not really a stream" -- publishing clean while being, by
    construction, unreadable to this check. Every character may also be written `#XX`, so the name
    has to be recognised by what it MEANS rather than by one spelling.
    """
    import os
    import zlib

    from flash.env_secrets import _Unscannable, credential_in_file

    # really ciphertext: if the key were findable in these bytes, a refusal would prove nothing
    cipher = bytes(
        stream ^ pad
        for stream, pad in zip(
            zlib.compress(b'export FREESOLO_API_KEY="fslo_%s"\n' % _FAKE_KEY_BODY.encode()),
            os.urandom(4096) * 8,
            strict=False,
        )
    )
    assert b"fslo_" not in cipher

    for name, entry in (
        ("comment.pdf", b"/Encrypt%c\n2 0 R"),
        ("escaped.pdf", b"/Encryp#74 2 0 R"),
        ("both.pdf", b"/#45ncrypt%x\n2 0 R"),
        ("dict.pdf", b"/Encrypt<</O 1>>"),
        ("array.pdf", b"/Encrypt[1 0 R]"),
    ):
        published = tmp_path / name
        published.write_bytes(
            b"%PDF-1.7\ntrailer\n<< "
            + entry
            + b" /Root 1 0 R >>\n"
            + _flate_pdf(b"/Filter /FlateDecode", cipher)
        )
        with pytest.raises(_Unscannable, match="encrypted document"):
            credential_in_file(published)

    # a LONGER name merely starting with those characters is a different name, and its document is
    # not encrypted -- refusing there would refuse ordinary documents on a substring
    plain = zlib.compress(b'export FREESOLO_API_KEY="fslo_%s"\n' % _FAKE_KEY_BODY.encode())
    other = tmp_path / "other.pdf"
    other.write_bytes(
        b"%PDF-1.7\ntrailer\n<< /Encryptionless 1 /Root 1 0 R >>\n"
        + _flate_pdf(b"/Filter /FlateDecode", plain)
    )
    assert credential_in_file(other) == "a Freesolo API key"


def test_narrowed_wide_text_keeps_the_truncation_state_of_its_chunk(tmp_path):
    """A UTF-16 file must reach the same verdict as the same bytes written narrow.

    Dropping `truncated` on the way into the narrowed run told the base64 path that every run
    ended where the file did, so an encoded container crossing a chunk boundary had its first
    fragment treated as a complete value while later fragments began mid-stream and could be
    expanded from neither side -- the wide form returned clean where the narrow form refused.
    """
    import base64
    import gzip
    import os

    from flash.env_secrets import _Unscannable, credential_in_file

    # incompressible padding, so the ENCODED form really does exceed one read chunk
    blob = base64.b64encode(
        gzip.compress(os.urandom(900_000) + b'export KEY="fslo_%s"\n' % _FAKE_KEY_BODY.encode())
    )
    assert len(blob) > (1 << 20)

    verdicts = []
    for name, data in (("narrow.txt", blob), ("wide.txt", blob.decode().encode("utf-16-le"))):
        published = tmp_path / name
        published.write_bytes(data)
        try:
            verdicts.append(credential_in_file(published))
        except _Unscannable as refusal:
            verdicts.append(str(refusal))
    assert verdicts[0] == verdicts[1], verdicts


def test_a_credential_split_by_any_supported_escape_is_rejoined(tmp_path):
    """`\\U000000XX` and `\\xXX` name the same character `\\u00XX` does.

    Only the JSON spelling was decoded, so the same key written for TOML or Python published:
    `tomllib.loads('key="fslo_AbCd\\\\U00000045..."')` and a Python sidecar using `\\x45` both hand
    the complete credential to whatever reads them, while the raw bytes a pattern sees are split by
    the escape and match nothing.
    """
    from flash.env_secrets import credential_in_file

    body = _FAKE_KEY_BODY.encode()
    for name, spelling in (
        ("json.json", b'{"key":"fslo_\\u0041%s"}' % body),
        ("toml.toml", b'key="fslo_\\U00000041%s"' % body),
        ("python.py", b'KEY = "fslo_\\x41%s"' % body),
    ):
        published = tmp_path / name
        published.write_bytes(spelling)
        assert credential_in_file(published) == "a Freesolo API key", name


def test_adjacent_literals_join_across_different_quote_styles(tmp_path):
    """Python concatenates `'a'"b"` exactly as it concatenates `'a''b'`.

    The seam pattern required the two quotes to MATCH, so a key split across literals written with
    different delimiters stayed split while the matched spelling was refused. What separates a
    concatenation from two list elements is the whitespace-only separator, not which quote each
    side uses.
    """
    from flash.env_secrets import credential_in_file

    body = _FAKE_KEY_BODY.encode()
    for name, source in (
        ("matched.py", b"KEY = 'fslo_Ab''%s'" % body),
        ("mixed.py", b"KEY = 'fslo_Ab'\"%s\"" % body),
        ("reversed.py", b"KEY = \"fslo_Ab\"'%s'" % body),
    ):
        published = tmp_path / name
        published.write_bytes(source)
        assert credential_in_file(published) == "a Freesolo API key", name

    # a separator that is not whitespace is two values, not one literal, and must not be welded
    for name, source in (
        ("list.py", b'V = ["fslo_Ab", "%s"]' % body),
        ("dict.py", b'D = {"fslo_Ab": "%s"}' % body),
        ("csv.csv", b'"fslo_Ab","%s"\n' % body),
    ):
        published = tmp_path / name
        published.write_bytes(source)
        assert credential_in_file(published) is None, name


def test_a_partial_length_packet_does_not_end_the_openpgp_walk(tmp_path):
    """A partial body length is walkable, and stopping at one hid the packets behind it.

    RFC 4880 4.2.2.4 lets a data packet arrive in chunks, each declaring its own length until a
    definite one ends it. There is no single stated length to read, so the walk treated it as an
    unreadable header and returned NORMALLY -- a confident "no key here" about a remainder it never
    examined. A legal partial-length literal packet between a public packet and a secret-key packet
    therefore published the secret key.
    """
    import os

    from flash.env_secrets import credential_in_file

    def packet(tag: int, body: bytes) -> bytes:
        return bytes([0xC0 | tag, len(body)]) + body

    material = bytes([4]) + b"\x67\x00\x00\x00" + bytes([1]) + os.urandom(70)
    public, secret = packet(6, material), packet(5, material)
    # tag 11 (literal data) may carry a partial length: 0xE2 declares a 4-byte chunk, then a
    # definite chunk ends the packet
    partial = bytes([0xC0 | 11, 0xE2]) + b"ABCD" + bytes([3]) + b"xyz"
    for name, data in (
        ("plain.pgp", public + secret),
        ("definite.pgp", public + packet(11, b"b\x00\x00\x00\x00\x00hello") + secret),
        ("partial.pgp", public + partial + secret),
    ):
        published = tmp_path / name
        published.write_bytes(data)
        assert credential_in_file(published) == "a private key", name

    # ordinary binary must not be refused because a random byte lands in the partial range: the
    # length is only read that way on the data packets the format allows it on.
    #
    # Seeded, and the gate is a RATE rather than zero. A chunk of random bytes refuses at some small
    # rate whatever this does -- the walk has other ways to run out of sequence -- so demanding zero
    # made the test fail on its own noise at 64 draws. Measured over 1024 draws: 2-3 refusals with
    # the tag gate against 10-17 with every tag allowed a partial length, which is the regression
    # this exists to catch, so the ceiling sits between the two.
    import random

    from flash.env_buffers import _SCAN_CHUNK_BYTES
    from flash.env_openpgp import _openpgp_secret_key_in_sequence

    draws, ceiling = 1024, 6
    rng = random.Random(20260815)
    refused = sum(
        1
        for _ in range(draws)
        if _openpgp_secret_key_in_sequence(rng.randbytes(_SCAN_CHUNK_BYTES), truncated=True) is None
    )
    assert refused <= ceiling, f"{refused}/{draws} random chunks refused over a partial length"


def test_a_version_5_session_packet_is_read_with_its_own_layout(tmp_path):
    """A v5 SKESK carries an AEAD algorithm byte the v4 layout does not have.

    Reading every version with the v4 layout tested that AEAD byte as though it were the S2K type.
    `gpg --force-aead --aead-algo OCB` writes AEAD 2, which is not a defined S2K type, so the
    packet failed validation, the message was reported "not encrypted", and ciphertext `gpg
    --decrypt` reads back as a key published clean.
    """
    import os

    from flash.env_secrets import _Unscannable, credential_in_file

    def packet(tag: int, body: bytes) -> bytes:
        return bytes([0xC0 | tag, len(body)]) + body

    data = packet(18, b"\x01" + os.urandom(64))
    salt = os.urandom(8)
    for name, skesk in (
        ("v4.gpg", packet(3, bytes([4, 9, 3, 8]) + salt + bytes([96]))),
        ("v5.gpg", packet(3, bytes([5, 9, 2, 3, 8]) + salt + bytes([96]))),
        ("v6.gpg", packet(3, bytes([6, 9, 2, 0, 3, 8]) + salt + bytes([96]))),
    ):
        published = tmp_path / name
        published.write_bytes(skesk + data)
        with pytest.raises(_Unscannable, match="encrypted OpenPGP message"):
            credential_in_file(published)


def test_a_one_asymmetric_key_declaring_version_1_is_recognised(tmp_path):
    """RFC 5958 declares `v2(1)` when the optional public-key field is present.

    The version INTEGER was pinned to 0, so an Ed25519 key carrying its public half -- what a
    conversion from OpenSSH produces -- matched nothing. Unlike RSA its private scalar holds no
    nested DER for another branch to catch, so the key published intact.
    """
    import os

    from flash.env_secrets import credential_in_file

    algorithm = b"\x30\x05\x06\x03\x2b\x65\x70"
    private = b"\x04\x22\x04\x20" + os.urandom(32)
    public = b"\x81\x21\x00" + os.urandom(32)
    for name, body in (
        ("v0.der", b"\x02\x01\x00" + algorithm + private),
        ("v1.der", b"\x02\x01\x01" + algorithm + private + public),
    ):
        published = tmp_path / name
        published.write_bytes(b"\x30" + bytes([len(body)]) + body)
        assert credential_in_file(published) == "a private key", name


def test_a_stream_behind_an_unreversible_filter_is_refused(tmp_path):
    """A PDF stream this cannot decode is undecided, not clean.

    The walk enumerated only streams naming `/FlateDecode`, so a stream declaring any other filter
    was never looked at: a conforming `/ASCIIHexDecode` payload holding the hex spelling of a key
    returned clean, while the same key behind flate was caught. Decoding every filter PDF defines
    is a document parser's job; refusing what cannot be reversed is the bounded answer.
    """
    import zlib

    from flash.env_secrets import _Unscannable, credential_in_file

    key = b"fslo_%s" % _FAKE_KEY_BODY.encode()
    for name, dictionary, body in (
        ("hex.pdf", b"/Filter /ASCIIHexDecode", key.hex().encode() + b">"),
        ("lzw.pdf", b"/Filter /LZWDecode", zlib.compress(key)),
        ("escaped.pdf", b"/#46ilter /ASCIIHexDecode", key.hex().encode() + b">"),
    ):
        published = tmp_path / name
        published.write_bytes(_flate_pdf(dictionary, body))
        with pytest.raises(_Unscannable, match="cannot undo"):
            credential_in_file(published)

    # a stream with NO filter is uncompressed, so the ordinary literal pass over the document
    # already covers it and it must not be refused
    plain = tmp_path / "plain.pdf"
    plain.write_bytes(_flate_pdf(b"/Type /XObject", b"harmless text"))
    assert credential_in_file(plain) is None


def test_an_escaped_predictor_key_is_still_a_predictor(tmp_path):
    """`/#50redictor` names `Predictor` to every reader.

    The key was matched literally, so a predictor written with an escape named nothing here: the
    stream inflated to horizontal differences and was scanned as though those were content, and a
    key a conforming decode reconstructs published intact.
    """
    import zlib

    from flash.env_secrets import _Unscannable, credential_in_file

    key = b"fslo_%s" % _FAKE_KEY_BODY.encode()
    differences = bytearray()
    previous = 0
    for byte in key:
        differences.append((byte - previous) & 0xFF)
        previous = byte
    body = zlib.compress(bytes(differences))
    for name, spelling in (("literal.pdf", b"/Predictor"), ("escaped.pdf", b"/#50redictor")):
        published = tmp_path / name
        published.write_bytes(
            _flate_pdf(
                b"/Filter /FlateDecode /DecodeParms << "
                + spelling
                + b" 2 /Colors 1 /BitsPerComponent 8 /Columns "
                + str(len(key)).encode()
                + b" >>",
                body,
            )
        )
        with pytest.raises(_Unscannable, match="cannot undo"):
            credential_in_file(published)


def test_a_pdf_larger_than_the_scan_buffer_is_refused(tmp_path):
    """A real PDF is held whole, so its size has to be bounded like every other container.

    The signature check in front of it stopped an ordinary shard being materialized, but a file
    that really begins `%PDF-` still reached an unconditional read: a package may carry a 256 MiB
    document, and holding one is a second complete copy while the decoded request and the staged
    file are both still live. Measured 224 MiB of RSS for a 200 MiB PDF against 26 MiB for the same
    bytes with a different first line.

    The grammar is not streamable from here -- the trailer names the encryption dictionary and an
    object's filters may sit at any offset -- so the bound refuses rather than parsing incrementally,
    which is the answer every other oversized container already gets.
    """
    from flash.env_secrets import _MAX_NESTED_BUFFER_BYTES, _Unscannable, credential_in_file

    def document(payload: bytes) -> bytes:
        return (
            b"%PDF-1.7\n1 0 obj\n<< /Length "
            + str(len(payload)).encode()
            + b" >>\nstream\n"
            + payload
            + b"\nendstream\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n"
        )

    oversized = tmp_path / "big.pdf"
    oversized.write_bytes(document(b"A" * (_MAX_NESTED_BUFFER_BYTES + 1)))
    with pytest.raises(_Unscannable, match="too large to inspect"):
        credential_in_file(oversized)

    # a document inside the bound is still scanned rather than refused over its size
    ordinary = tmp_path / "small.pdf"
    ordinary.write_bytes(document(b"harmless text"))
    assert credential_in_file(ordinary) is None


def test_a_pdf_filter_declared_across_a_comment_is_still_decoded(tmp_path):
    """A `%` comment is a token separator, so it may sit between `/Filter` and its value.

    Accepting whitespace alone left `/Filter%c\\n[/ASCII85Decode /FlateDecode]` unrecovered: the
    chain went unrecognised, the ASCII85 body was handed straight to zlib, the resulting error was
    read as "not really a stream", and the key inside published. A reader resolves the commented
    spelling exactly as it resolves the spaced one.
    """
    import base64
    import zlib

    from flash.env_secrets import credential_in_file

    key = b'FREESOLO_API_KEY="fslo_%s"\n' % _FAKE_KEY_BODY.encode()
    stream = base64.a85encode(zlib.compress(key)) + b"~>"
    for name, separator in (
        ("spaced.pdf", b" "),
        ("commented.pdf", b"%a comment before the value\n"),
    ):
        published = tmp_path / name
        published.write_bytes(
            _flate_pdf(b"/Filter" + separator + b"[/ASCII85Decode /FlateDecode]", stream)
        )
        assert credential_in_file(published) == "a Freesolo API key", name


def test_a_zip_member_name_is_checked_like_a_tar_member_name(tmp_path):
    """A zip entry's NAME publishes whatever it spells, exactly as a tar member's does.

    The tar walk checked names and the zip walk did not, so `fslo_<key>.json` inside a zip was read
    only for its (empty) contents and published with the key in the archive's listing. A name that
    is itself an encoded container is refused rather than decoded speculatively -- and the raw scan
    over the archive's own bytes swallowed that refusal, so a member named with base64 of an
    OpenSSL-encrypted file returned clean while the same string passed to the name scanner refused.
    """
    import base64
    import zipfile

    from flash.env_secrets import _Unscannable, credential_in_file, credential_in_name

    keyed = tmp_path / "keyed.zip"
    with zipfile.ZipFile(keyed, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"fslo_{_FAKE_KEY_BODY}.json", "{}")
    assert credential_in_file(keyed) == "a Freesolo API key"

    # a name the scan cannot finish reading is a refusal here too, the same answer the name
    # scanner already gives for that string on its own
    unscannable = base64.b64encode(b"Salted__12345678ciphertext").decode()
    with pytest.raises(_Unscannable, match="OpenSSL-encrypted"):
        credential_in_name(unscannable)
    hidden = tmp_path / "hidden.zip"
    with zipfile.ZipFile(hidden, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(unscannable, "{}")
    with pytest.raises(_Unscannable, match="OpenSSL-encrypted"):
        credential_in_file(hidden)

    # ordinary member names must not become refusals or false matches
    plain = tmp_path / "plain.zip"
    with zipfile.ZipFile(plain, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("model-00001-of-00002.safetensors", "weights")
        archive.writestr("configs/train.toml", "[train]\nsteps = 10\n")
    assert credential_in_file(plain) is None


def test_an_openssh_key_with_its_armor_stripped_is_still_a_private_key(tmp_path):
    """`-----BEGIN OPENSSH PRIVATE KEY-----` wraps base64 of a blob that IS the key.

    Every pattern matched the armor, so the same key with its header removed -- which is what
    `base64 -d` on the body produces -- matched nothing and published intact. Re-adding the header
    reconstructs a usable key, so the decoded blob is the whole secret.
    """
    import base64

    from flash.env_secrets import credential_in_file

    blob = b"openssh-key-v1\x00" + b"\x00\x00\x00\x04none" * 2 + b"\x00\x00\x00\x00" + b"\x11" * 96

    armored = tmp_path / "id_ed25519"
    armored.write_bytes(
        b"-----BEGIN OPENSSH PRIVATE KEY-----\n"
        + base64.encodebytes(blob)
        + b"-----END OPENSSH PRIVATE KEY-----\n"
    )
    assert credential_in_file(armored) == "a private key block"

    stripped = tmp_path / "id_ed25519.raw"
    stripped.write_bytes(blob)
    assert credential_in_file(stripped) == "a private key"

    # the magic alone is not enough: the ciphername length that follows bounds it to a real header,
    # so prose and binary that happen to carry the string are not refused
    for ordinary in (b"see openssh-key-v1 format docs\n", b"openssh-key-v1\x00\x00\x01\x00\x00"):
        prose = tmp_path / f"doc{len(ordinary)}.txt"
        prose.write_bytes(ordinary)
        assert credential_in_file(prose) is None, ordinary


def test_scanning_a_name_is_charged_to_the_packages_budget(tmp_path):
    """A name's expansion is bounded by the package's budget, not a fresh one per name.

    A name is a few hundred bytes, which reads like nothing to multiply -- but what it ENCODES is
    not, and giving each one its own 60-second budget made the package-wide bound advisory: a
    thousand members each got a full budget, so the limit the package scan enforces was never
    reached no matter how many expensive names were present.
    """
    import time

    from flash.env_secrets import credential_in_name

    # a deadline already in the past leaves no budget, so an expansion cannot be started under it
    assert credential_in_name(f"fslo_{_FAKE_KEY_BODY}", deadline=time.monotonic() - 1.0) == (
        "a Freesolo API key"
    ), "a literal match needs no expansion and must survive an exhausted budget"

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "ordinary.txt").write_bytes(b"nothing here\n")
    spent: list[float | None] = []
    import flash.env_secrets as secrets

    original = secrets.credential_in_name

    def record(name, *, deadline=None):
        spent.append(deadline)
        return original(name, deadline=deadline)

    secrets.credential_in_name = record
    try:
        secrets.reject_credential_bearing_package(package, display={})
    finally:
        secrets.credential_in_name = original

    assert spent, "the package walk must scan its member names"
    assert all(deadline is not None for deadline in spent), "each name must carry a budget"
    assert len(set(spent)) == 1, "every name in one package shares ONE budget"


def test_adjacent_literals_join_across_a_literal_prefix(tmp_path):
    """`b'fslo_AbCd' b'Ef...'` is one string to Python, exactly as the bare pair is.

    The seam pattern could not cross the `b`/`r`/`f`/`u` prefix on the second literal, so a key
    written that way published while the unprefixed spelling of the same value was refused -- and
    an executable sidecar is exactly where a `b''` pair is ordinary.
    """
    from flash.env_secrets import credential_in_file

    half, rest = f"fslo_{_FAKE_KEY_BODY[:12]}", _FAKE_KEY_BODY[12:]
    for name, prefix in (
        ("bare.py", ""),
        ("bytes.py", "b"),
        ("raw.py", "r"),
        ("rawbytes.py", "rb"),
    ):
        published = tmp_path / name
        published.write_bytes(f"KEY = {prefix}'{half}' {prefix}'{rest}'\n".encode())
        assert credential_in_file(published) == "a Freesolo API key", name

    # two list elements are still two values: the separator carries a comma, so nothing is joined
    listed = tmp_path / "list.py"
    listed.write_bytes(f"KEYS = [b'{half}', b'{rest}']\n".encode())
    assert credential_in_file(listed) is None


def test_a_tar_entry_that_is_not_a_file_still_has_its_name_checked(tmp_path):
    """A directory's NAME is in the archive listing exactly as a file's is.

    The entry-type filter ran first, so a directory, symlink or device node never reached the name
    scanner: a directory named with base64 of an OpenSSL-encrypted file returned clean here while
    the same string handed to `credential_in_name` was refused.
    """
    import base64
    import tarfile

    from flash.env_secrets import _Unscannable, credential_in_file

    def archive(path, name, kind):
        with tarfile.open(path, "w") as tar:
            entry = tarfile.TarInfo(name)
            entry.type, entry.mode = kind, 0o755
            tar.addfile(entry)

    keyed = tmp_path / "dir.tar"
    archive(keyed, f"fslo_{_FAKE_KEY_BODY}", tarfile.DIRTYPE)
    assert credential_in_file(keyed) == "a Freesolo API key"

    unscannable = tmp_path / "unscannable.tar"
    archive(unscannable, base64.b64encode(b"Salted__12345678ciphertext").decode(), tarfile.DIRTYPE)
    with pytest.raises(_Unscannable, match="OpenSSL-encrypted"):
        credential_in_file(unscannable)

    # an ordinary directory entry is still not a refusal
    plain = tmp_path / "plain.tar"
    archive(plain, "checkpoints", tarfile.DIRTYPE)
    assert credential_in_file(plain) is None


def test_a_zip_directory_entry_still_has_its_name_checked(tmp_path):
    """The same hole in the other format, which the trailing slash hides twice over.

    `ZipInfo.is_dir` IS the trailing `/`, so a directory entry skipped the name scanner -- and the
    slash also lands inside the base64 run, which stops the raw pass over the archive's own bytes
    from recovering the encoded stream. A directory named with base64 of a gzipped key returned
    clean while the name scanner refused that same string on its own.
    """
    import base64
    import gzip
    import io
    import zipfile

    from flash.env_secrets import credential_in_file, credential_in_name

    def archive(path, name):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zipped:
            zipped.writestr(name + "/", b"")
            zipped.writestr("readme.txt", b"hello\n")
        path.write_bytes(buffer.getvalue())

    encoded = (
        base64.b64encode(gzip.compress(f"fslo_{_FAKE_KEY_BODY}".encode())).decode().rstrip("=")
    )
    assert credential_in_name(encoded) == "a Freesolo API key"  # the control the zip has to match

    keyed = tmp_path / "dir.zip"
    archive(keyed, encoded)
    assert credential_in_file(keyed) == "a Freesolo API key"

    # the plain spelling too, which needs no decoding to be visible in the listing
    named = tmp_path / "named.zip"
    archive(named, f"fslo_{_FAKE_KEY_BODY}")
    assert credential_in_file(named) == "a Freesolo API key"

    # an ordinary directory entry is still not a refusal
    plain = tmp_path / "plain.zip"
    archive(plain, "checkpoints")
    assert credential_in_file(plain) is None


def test_a_pdf_dictionary_is_read_to_its_own_opening_bracket(tmp_path):
    """The object's `<<` bounds its dictionary, not a fixed 512 bytes.

    A dictionary may legally carry any amount of metadata. With 600 bytes of it, `/DecodeParms <<
    /Predictor 2 ... >>` fell outside the window and named nothing, so the stream was inflated and
    horizontal differences were scanned as though they were content; and a filter array split the
    same way reported no pre-filter, so zlib was handed ASCII85 text and the failure read as "not
    really a stream". Both spellings decode to the key for any conforming reader.
    """
    import base64
    import zlib

    from flash.env_secrets import _Unscannable, credential_in_file

    key = b"fslo_%s" % _FAKE_KEY_BODY.encode()
    gap = b" " * 600

    # a filter ARRAY whose two names sit either side of the gap
    stream = base64.a85encode(zlib.compress(key)) + b"~>"
    for name, declaration in (
        ("compact.pdf", b"/Filter [/ASCII85Decode /FlateDecode]"),
        ("spread.pdf", b"/Filter [/ASCII85Decode" + gap + b"/FlateDecode]"),
    ):
        published = tmp_path / name
        published.write_bytes(_flate_pdf(declaration, stream))
        assert credential_in_file(published) == "a Freesolo API key", name

    # a PREDICTOR written far ahead of the filter it applies to
    differences = bytearray()
    previous = 0
    for byte in key:
        differences.append((byte - previous) & 0xFF)
        previous = byte
    parms = (
        b"/DecodeParms << /Predictor 2 /Colors 1 /BitsPerComponent 8 /Columns "
        + str(len(key)).encode()
        + b" >>"
    )
    for name, filler in (("near.pdf", b""), ("far.pdf", b"/Meta (" + b"x" * 600 + b")")):
        published = tmp_path / name
        published.write_bytes(
            _flate_pdf(parms + filler + b" /Filter /FlateDecode", zlib.compress(bytes(differences)))
        )
        with pytest.raises(_Unscannable, match="cannot undo"):
            credential_in_file(published)


def test_the_all_stream_pass_refuses_rather_than_stopping_at_its_cap(tmp_path):
    """Stopping at the bound called every stream past it readable.

    The flate walk already refuses a surplus; this pass returned instead, so 4,096 unfiltered
    streams followed by an `/ASCIIHexDecode` stream holding a hex-spelled key never reached the
    filtered one -- and the flate walk counts only flate streams, so it did not reach it either.
    """
    from flash.env_deflate import _MAX_PDF_STREAMS
    from flash.env_secrets import _Unscannable, credential_in_file

    def document(objects):
        body = b"%PDF-1.7\n"
        for index, (dictionary, payload) in enumerate(objects, 1):
            body += b"%d 0 obj\n<< %s /Length %d >>\nstream\n%s\nendstream\nendobj\n" % (
                index,
                dictionary,
                len(payload),
                payload,
            )
        return body + b"trailer\n<< /Root 1 0 R >>\n%%EOF\n"

    hexed = (b"/Filter /ASCIIHexDecode", (b"fslo_%s" % _FAKE_KEY_BODY.encode()).hex().encode())
    over = tmp_path / "over.pdf"
    over.write_bytes(
        document([(b"", b"plain%d" % i) for i in range(_MAX_PDF_STREAMS + 8)] + [hexed])
    )
    with pytest.raises(_Unscannable, match="more compressed streams"):
        credential_in_file(over)

    # inside the bound the same document is decided rather than refused over its stream count
    under = tmp_path / "under.pdf"
    under.write_bytes(document([(b"", b"plain"), hexed]))
    with pytest.raises(_Unscannable, match="cannot undo"):
        credential_in_file(under)


def test_a_partial_packet_whose_final_chunk_outruns_the_buffer_is_undecided(tmp_path):
    """A declared length reaching past the bytes in hand is unread, not empty.

    The offset was returned unchecked, so slicing at it produced an EMPTY remainder -- and an empty
    remainder walks to the end finding nothing. A secret-key packet behind such a chunk was
    reported clean while `gpg --import` installed the key from those same bytes.
    """
    import os
    import struct

    from flash.env_buffers import _SCAN_CHUNK_BYTES
    from flash.env_secrets import _Unscannable, credential_in_file

    material = bytes([4]) + b"\x67\x00\x00\x00" + bytes([1]) + os.urandom(70)

    def packet(tag, body):
        return bytes([0xC0 | tag, len(body)]) + body

    public, secret = packet(6, material), packet(5, material)

    def sequence(body):
        # a partial chunk, then a five-octet definite length naming `body`
        return (
            public
            + bytes([0xC0 | 11, 0xE2])
            + b"ABCD"
            + bytes([0xFF])
            + struct.pack(">I", len(body))
            + body
            + secret
        )

    outruns = tmp_path / "outruns.pgp"
    outruns.write_bytes(sequence(b"Z" * (_SCAN_CHUNK_BYTES * 2)))
    with pytest.raises(_Unscannable, match="cannot walk to the end"):
        credential_in_file(outruns)

    # the same shape whose final chunk FITS is still walked to the secret key behind it
    fits = tmp_path / "fits.pgp"
    fits.write_bytes(sequence(b"Z" * 100) + b"\x00" * (_SCAN_CHUNK_BYTES + 16))
    assert credential_in_file(fits) == "a private key"


def test_a_version_3_openpgp_secret_key_is_recognised(tmp_path):
    """A v3 secret key is obsolete, not unreadable.

    GnuPG still names a structurally complete tag-5 v3 packet an (obsolete) secret key, and its raw
    MPI material matches no textual or DER detector -- so rejecting the version published the key.
    The v3 layout puts a two-byte validity period after the timestamp, so the algorithm sits two
    bytes further on than v4's and reading it at the v4 offset lands inside that field.
    """
    import os

    from flash.env_secrets import credential_in_file

    body = bytes([3]) + b"\x67\x00\x00\x00" + b"\x00\x1e" + bytes([1]) + os.urandom(70)
    published = tmp_path / "v3.pgp"
    published.write_bytes(bytes([0x80 | (5 << 2)]) + bytes([len(body)]) + body)
    assert credential_in_file(published) == "a private key"

    # the version is still a filter: a byte outside the three real ones is not a key packet
    for version in (0, 2, 5, 7):
        ordinary = tmp_path / f"v{version}.bin"
        ordinary.write_bytes(
            bytes([0x80 | (5 << 2)]) + bytes([len(body)]) + bytes([version]) + body[1:]
        )
        assert credential_in_file(ordinary) is None, version


def test_a_name_detected_only_after_rejoining_is_withheld_not_printed(tmp_path):
    """Masking cannot redact a key whose body is split by the seam that hid it.

    The substitutions run over the RAW name, so a key spelled `fslo_AbCd\\x45f...` or across two
    adjacent literals matched none of them and the refusal printed the complete reversible spelling
    into the terminal and any collected logs -- the one thing this module is careful never to do.
    """
    from flash.env_secrets import _redacted, credential_in_name

    half, rest = f"fslo_{_FAKE_KEY_BODY[:12]}", _FAKE_KEY_BODY[12:]
    for name in (f"{half}\\x{ord(rest[0]):02x}{rest[1:]}", f"'{half}' '{rest}'"):
        assert credential_in_name(name) == "a Freesolo API key", name
        shown = _redacted(name)
        assert _FAKE_KEY_BODY[12:] not in shown, f"the refusal echoed the key: {shown}"
        assert "encodes a credential" in shown

    # an ordinary name is still shown verbatim, which is what makes the refusal actionable
    assert _redacted("configs/train.toml") == "configs/train.toml"


def test_an_ar_archive_is_walked_member_by_member(tmp_path):
    """A `.deb` is an ar, and none of the other handlers enumerates one.

    Its members' magic sits at their own offsets rather than at byte zero, so the anchored format
    check could not see it and the raw pass read compressed member bytes as opaque content: a
    package whose `data.tar.zst` this cannot expand published clean, while the same zstd payload
    standing alone was refused.
    """
    import os

    from flash.env_secrets import _Unscannable, credential_in_file

    def ar(members):
        out = b"!<arch>\n"
        for name, payload in members:
            out += (
                f"{name + '/':<16}{0:<12}{0:<6}{0:<6}{0o100644:<8o}{len(payload):<10}`\n"
            ).encode()
            out += payload + (b"\n" if len(payload) % 2 else b"")
        return out

    opaque = b"\x28\xb5\x2f\xfd" + os.urandom(300)
    package = tmp_path / "pkg.deb"
    package.write_bytes(ar([("debian-binary", b"2.0\n"), ("data.tar.zst", opaque)]))
    with pytest.raises(_Unscannable, match="zstd"):
        credential_in_file(package)

    # a member's CONTENTS are scanned like any other file's
    keyed = tmp_path / "keyed.deb"
    keyed.write_bytes(
        ar([("debian-binary", b"2.0\n"), ("control", b"fslo_%s\n" % _FAKE_KEY_BODY.encode())])
    )
    assert credential_in_file(keyed) == "a Freesolo API key"

    # and an ordinary package is not refused merely for being one
    plain = tmp_path / "plain.deb"
    plain.write_bytes(ar([("debian-binary", b"2.0\n"), ("readme", b"nothing to see\n")]))
    assert credential_in_file(plain) is None


def test_an_age_encrypted_file_is_refused_like_every_other_ciphertext(tmp_path):
    """age was the one encryption this let through.

    Its body is ciphertext, so no pattern and no base64 decode can see the key inside, and it
    scanned as ordinary clean content while OpenSSL, OpenPGP, PDF and ZIP encryption were all
    refused. `secrets.age` beside an environment is ordinary -- several secret managers write it.
    """
    import base64
    import os

    from flash.env_secrets import _Unscannable, credential_in_file

    native = tmp_path / "secrets.age"
    native.write_bytes(b"age-encryption.org/v1\n-> X25519 abcd\n" + os.urandom(200))
    with pytest.raises(_Unscannable, match="age-encrypted"):
        credential_in_file(native)

    armored = tmp_path / "armored.age"
    armored.write_bytes(
        b"-----BEGIN AGE ENCRYPTED FILE-----\n"
        + base64.encodebytes(os.urandom(200))
        + b"-----END AGE ENCRYPTED FILE-----\n"
    )
    with pytest.raises(_Unscannable, match="age-encrypted"):
        credential_in_file(armored)

    # prose that merely mentions age is not ciphertext: the header is anchored at byte zero
    prose = tmp_path / "notes.md"
    prose.write_bytes(b"We encrypt these with age-encryption.org/v1 before sharing.\n")
    assert credential_in_file(prose) is None


def test_an_ar_trailer_too_short_for_a_header_is_refused(tmp_path):
    """The ar walk's end condition could not tell "finished" from "gave up".

    Fewer bytes left than a 60-byte header ended the walk as if the archive were complete, so a
    remainder that is not a member -- a bare compressed record appended after one -- was neither
    parsed nor reported. A 29-byte zlib record holding a key published clean while the same record
    standing alone was refused.
    """
    import zlib

    from flash.env_secrets import _Unscannable, credential_in_file

    record = zlib.compress(f"fslo_{_FAKE_KEY_BODY}".encode())
    assert len(record) < 60  # the fixture only works if it fits inside a header's width

    control = tmp_path / "record.zlib"
    control.write_bytes(record)
    assert credential_in_file(control) == "a Freesolo API key"  # the control the ar must not lose

    header = b"empty.txt".ljust(16) + b"0".ljust(12) + b"0".ljust(6) + b"0".ljust(6)
    header += b"100644".ljust(8) + b"0".ljust(10) + b"`\n"
    short = tmp_path / "short.a"
    short.write_bytes(b"!<arch>\n" + header + record)
    with pytest.raises(_Unscannable, match="cannot read"):
        credential_in_file(short)

    # an archive ending exactly on a member boundary is still clean, so this did not refuse every ar
    plain = tmp_path / "plain.a"
    plain.write_bytes(b"!<arch>\n" + header)
    assert credential_in_file(plain) is None


def test_a_codec_error_in_one_member_does_not_abandon_the_archive(tmp_path):
    """`zlib.error` and `LZMAError` inherit straight from `Exception`, so nothing caught them.

    An uncaught codec error escaped the member loop entirely and the dispatcher read it as "not a
    zip after all", moving on to the next handler. A corrupt FIRST member therefore hid a perfectly
    readable second member holding the key.
    """
    import struct
    import zipfile
    import zlib

    from flash.env_secrets import credential_in_file

    buffered = io.BytesIO()
    with zipfile.ZipFile(buffered, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("a_corrupt.bin", os.urandom(4000))
        archive.writestr("b_key.txt", f"fslo_{_FAKE_KEY_BODY}".encode())
    raw = bytearray(buffered.getvalue())

    # damage the FIRST member's deflate stream deep enough that it is rejected rather than truncated
    signature = raw.find(b"PK\x03\x04")
    names = struct.unpack_from("<H", raw, signature + 26)[0]
    extra = struct.unpack_from("<H", raw, signature + 28)[0]
    payload = signature + 30 + names + extra
    for index in range(payload + 3, payload + 60):
        raw[index] ^= 0xFF

    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(bytes(raw))
    # the fixture really does raise the error that used to escape, rather than a caught one
    with zipfile.ZipFile(io.BytesIO(bytes(raw))) as archive, pytest.raises(zlib.error):
        archive.read("a_corrupt.bin")
    assert credential_in_file(corrupt) == "a Freesolo API key"


def test_a_zip_appended_to_another_zip_is_still_walked(tmp_path):
    """`zipfile` reads the LAST end-of-central-directory record, so the first archive vanishes.

    `cat a.zip b.zip` leaves the whole of `a.zip` as a prefix `namelist()` never reports, and a key
    DEFLATED inside it is not in the concatenation's raw bytes either -- so both passes returned
    clean on a file every unzip recovers the key from.
    """
    import zipfile

    from flash.env_secrets import credential_in_file

    first = io.BytesIO()
    with zipfile.ZipFile(first, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("key.txt", f"fslo_{_FAKE_KEY_BODY}\n" * 40)
    second = io.BytesIO()
    with zipfile.ZipFile(second, "w") as archive:
        archive.writestr("readme.txt", b"hello\n")

    hidden = first.getvalue()
    assert f"fslo_{_FAKE_KEY_BODY}".encode() not in hidden  # compression hides it from the raw pass

    control = tmp_path / "first.zip"
    control.write_bytes(hidden)
    assert credential_in_file(control) == "a Freesolo API key"  # the control the pair must match

    joined = tmp_path / "joined.zip"
    joined.write_bytes(hidden + second.getvalue())
    assert credential_in_file(joined) == "a Freesolo API key"

    # an ordinary single zip has no prefix at all and is unaffected
    plain = tmp_path / "plain.zip"
    plain.write_bytes(second.getvalue())
    assert credential_in_file(plain) is None


def test_an_octal_escape_is_resolved_like_the_hex_ones(tmp_path):
    """Python resolves `\\105` to `E` exactly as it resolves `\\x45`.

    Only the hex and unicode spellings were rejoined, so a key body carrying one octal escape
    evaluated to the complete credential at runtime while the raw bytes a pattern reads were split
    by the escape and matched nothing.
    """
    from flash.env_secrets import credential_in_file

    body = _FAKE_KEY_BODY[:2] + "\\105" + _FAKE_KEY_BODY[3:]
    escaped = tmp_path / "sidecar.py"
    escaped.write_text(f'KEY = "fslo_{body}"\n')
    assert eval(f'"fslo_{body}"')[7] == "E"  # the fixture must really decode to the key
    assert credential_in_file(escaped) == "a Freesolo API key"

    # a short octal escape is left alone, so an ordinary regex string is not rewritten
    ordinary = tmp_path / "pattern.py"
    ordinary.write_text('SPLIT = "\\1 and \\2"\n')
    assert credential_in_file(ordinary) is None


def test_a_comment_between_adjacent_literals_is_still_a_seam(tmp_path):
    """Inside parentheses the tokenizer discards a trailing comment exactly as it does whitespace.

    `('fslo_AbCd'  # prefix\\n 'Ef...')` is one string at runtime, so the seam had to close for the
    same reason the plain adjacent pair does -- and the comment text between the quotes kept it
    open.
    """
    from flash.env_secrets import credential_in_file

    split = tmp_path / "helper.py"
    split.write_text(
        f"KEY = ('fslo_{_FAKE_KEY_BODY[:8]}'  # the issuer prefix\n       '{_FAKE_KEY_BODY[8:]}')\n"
    )
    assert credential_in_file(split) == "a Freesolo API key"

    # two list elements separated by a comment are still two values, not one
    listed = tmp_path / "names.py"
    listed.write_text('NAMES = ["alpha",  # first\n         "beta"]\n')
    assert credential_in_file(listed) is None


def test_jwk_halves_in_sibling_objects_are_not_one_key(tmp_path):
    """Records split at depth ZERO, so two sibling objects shared one record.

    `{"public":{"kty":"RSA"},"artifact":{"d":"..."}}` holds a public key and an unrelated
    high-entropy value under a private member name, in different objects -- and pairing them
    refused a legitimate publish. The entropy test cannot separate the two, because a build id
    scores exactly as random as a key body.
    """
    from flash.env_secrets import credential_in_file

    payload = "abcdefghijklmnopqrstuvwxyz012345"

    siblings = tmp_path / "manifest.json"
    siblings.write_text(f'{{"public":{{"kty":"RSA"}},"artifact":{{"d":"{payload}"}}}}\n')
    assert credential_in_file(siblings) is None

    reversed_order = tmp_path / "reversed.json"
    reversed_order.write_text(f'{{"artifact":{{"d":"{payload}"}},"public":{{"kty":"RSA"}}}}\n')
    assert credential_in_file(reversed_order) is None

    # every shape that IS one key still pairs, including across nested metadata and one level down
    for name, text in {
        "flat.json": f'{{"kty":"RSA","d":"{payload}"}}',
        "swapped.json": f'{{"d":"{payload}","kty":"RSA"}}',
        "nested.json": f'{{"kty":"RSA","meta":{{"x":1}},"d":"{payload}"}}',
        "wrapped.json": f'{{"keys":[{{"kty":"RSA","d":"{payload}"}}]}}',
        "descended.json": f'{{"kty":"RSA","sub":{{"d":"{payload}"}}}}',
    }.items():
        real = tmp_path / name
        real.write_text(text + "\n")
        assert credential_in_file(real) == "a private key", name


def test_prose_naming_the_pgp_armor_is_not_ciphertext(tmp_path):
    """The armor LINE alone refused the document most likely to quote it.

    A README saying "look for a block starting with -----BEGIN PGP MESSAGE-----" carries no
    ciphertext, and refusing it named a remedy -- decrypt and remove the message -- for a file that
    has none.
    """
    from flash.env_secrets import _Unscannable, credential_in_file

    prose = tmp_path / "README.md"
    prose.write_text("Look for a block starting with -----BEGIN PGP MESSAGE----- in the archive.\n")
    assert credential_in_file(prose) is None

    # a header with no body at all is not a message either
    empty = tmp_path / "empty.asc"
    empty.write_text("-----BEGIN PGP MESSAGE-----\n-----END PGP MESSAGE-----\n")
    assert credential_in_file(empty) is None

    # a real armored message is still refused, so the narrowing kept the true positive
    real = tmp_path / "secret.asc"
    real.write_text(
        "-----BEGIN PGP MESSAGE-----\n\n"
        "hQEMAwm7ZLQ8Kd1kAQf/abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGH\n"
        "=abcd\n-----END PGP MESSAGE-----\n"
    )
    with pytest.raises(_Unscannable, match="encrypted OpenPGP"):
        credential_in_file(real)


def _ar_archive(members: list[tuple[str, bytes]]) -> bytes:
    """An ar archive of `members`, each header 60 bytes and each body padded to an even offset."""
    out = bytearray(b"!<arch>\n")
    for name, body in members:
        out += name.ljust(16).encode()[:16]
        out += b"0".ljust(12) + b"0".ljust(6) + b"0".ljust(6) + b"100644".ljust(8)
        out += str(len(body)).ljust(10).encode()[:10] + b"`\n"
        out += body
        if len(body) % 2:
            out += b"\n"
    return bytes(out)


def test_an_ar_long_name_is_resolved_before_it_is_scanned(tmp_path):
    """GNU ar stores a long member name in the `//` table and writes `/<offset>` in the header.

    Only the placeholder was scanned, so the name that actually leaks through the archive's listing
    was never read: a member whose resolved name is base64 of an OpenSSL-encrypted file returned
    clean here while that same string handed to the name scanner refused it.
    """
    import base64

    from flash.env_secrets import _Unscannable, credential_in_file, credential_in_name

    encrypted = b"Salted__12345678" + b"ciphertextciphertextciphertext"
    long_name = base64.b64encode(encrypted).decode()

    # the control: this name on its own is refused, so the archive carrying it must be too
    with pytest.raises(_Unscannable, match="OpenSSL-encrypted"):
        credential_in_name(long_name)

    archive = tmp_path / "long.a"
    archive.write_bytes(
        _ar_archive([("//", (long_name + "/\n").encode()), ("/0", b"harmless contents here")])
    )
    with pytest.raises(_Unscannable, match="OpenSSL-encrypted"):
        credential_in_file(archive)


def test_a_clean_ar_walk_settles_a_deferred_refusal(tmp_path):
    """A completed ar walk has enumerated the whole file exactly as a zip or tar walk has.

    Only those two counted as settling the question, so an ordinary two-member ar -- a Debian
    package is exactly this shape -- kept a deferred refusal from a stream handler that had read
    one member and could say nothing about the rest, and a harmless archive failed to publish.
    """
    import gzip

    from flash.env_secrets import credential_in_file

    archive = tmp_path / "clean.a"
    archive.write_bytes(
        _ar_archive(
            [
                ("debian-binary", b"2.0\n"),
                ("blob.gz", gzip.compress(b"nothing sensitive in here at all, just prose")),
            ]
        )
    )
    assert credential_in_file(archive) is None


def test_an_ar_inside_a_zip_is_still_expanded(tmp_path):
    """The container predicate never learned ar magic, so a nested one was final content.

    A top-level ar holding a zlib-compressed key is reported, but the same archive stored as a zip
    member returned clean -- compression keeps the key out of the raw byte pass, and nothing
    reopened the member to walk it.
    """
    import zipfile
    import zlib

    from flash.env_secrets import credential_in_file

    inner = _ar_archive([("payload.z", zlib.compress(f"fslo_{_FAKE_KEY_BODY}".encode()))])

    control = tmp_path / "top.a"
    control.write_bytes(inner)
    assert credential_in_file(control) == "a Freesolo API key"  # the control the pair must match

    nested = tmp_path / "outer.zip"
    with zipfile.ZipFile(nested, "w") as archive:
        archive.writestr("nested.a", inner)
    assert credential_in_file(nested) == "a Freesolo API key"


def test_exactly_the_member_limit_is_not_too_many_members(tmp_path):
    """Exhausting the member range entered the `else` branch unconditionally.

    An archive holding exactly the configured count was reported as having too many members, while
    a zip or tar of the same size is allowed -- a false refusal that blocks an honest publish.
    """
    from flash.env_archive import credential_in_ar
    from flash.env_secrets import _Unscannable

    def _named(_name: str) -> str | None:
        return None

    def _scan(_handle, _deadline, _depth) -> str | None:
        return None

    at_limit = _ar_archive([("only.txt", b"nothing to see here at all")])
    assert (
        credential_in_ar(
            at_limit,
            deadline=1e18,
            depth=0,
            scan=_scan,
            refusal=_Unscannable,
            named=_named,
            member_limit=1,
        )
        is None
    )

    # one MORE than the limit is still refused, so the bound itself still holds
    over_limit = _ar_archive([("one.txt", b"aa"), ("two.txt", b"bb")])
    with pytest.raises(_Unscannable, match="too many members"):
        credential_in_ar(
            over_limit,
            deadline=1e18,
            depth=0,
            scan=_scan,
            refusal=_Unscannable,
            named=_named,
            member_limit=1,
        )


def test_bytes_after_the_zip_end_record_are_still_scanned(tmp_path):
    """`is_zipfile` stays true with arbitrary bytes appended after the end-of-central-directory.

    The prefix before the first member is walked, but the suffix after the end record was not: a
    zlib record carrying a key, appended to an ordinary zip, published intact while that same
    record standing alone was reported.
    """
    import zipfile
    import zlib

    from flash.env_secrets import credential_in_file

    trailing = zlib.compress(f"fslo_{_FAKE_KEY_BODY}".encode())

    control = tmp_path / "alone.z"
    control.write_bytes(trailing)
    assert credential_in_file(control) == "a Freesolo API key"  # the control the pair must match

    plain = io.BytesIO()
    with zipfile.ZipFile(plain, "w") as archive:
        archive.writestr("readme.txt", b"nothing here\n")
    ordinary = plain.getvalue()

    clean = tmp_path / "plain.zip"
    clean.write_bytes(ordinary)
    assert credential_in_file(clean) is None  # an ordinary zip has no suffix and is unaffected

    appended = tmp_path / "appended.zip"
    appended.write_bytes(ordinary + trailing)
    assert zipfile.is_zipfile(appended)  # still a valid zip to every reader
    assert credential_in_file(appended) == "a Freesolo API key"


def test_a_tar_link_target_is_scanned_like_a_member_name(tmp_path):
    """A symlink's TARGET is in the archive's listing exactly as its own name is.

    Only `info.name` was read, so a link whose target is base64 of an OpenSSL-encrypted file was
    passed over -- the entry is not a file, so the type filter skipped it before anything looked at
    where it pointed, while the same string handed to the name scanner refused it.
    """
    import base64
    import tarfile

    from flash.env_secrets import _Unscannable, credential_in_file, credential_in_name

    encrypted = b"Salted__12345678" + b"ciphertextciphertextciphertext"
    target = base64.b64encode(encrypted).decode()

    # the control: this target on its own is refused, so the tar carrying it must be too
    with pytest.raises(_Unscannable, match="OpenSSL-encrypted"):
        credential_in_name(target)

    archive = tmp_path / "linked.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("safe-link")
        info.type = tarfile.SYMTYPE
        info.linkname = target
        tar.addfile(info)
    with pytest.raises(_Unscannable, match="OpenSSL-encrypted"):
        credential_in_file(archive)


def _pdf_with_stream(stream: bytes, entries: bytes, extra: bytes = b"") -> bytes:
    """A one-object PDF whose stream declares `entries` in its dictionary."""
    body = b"%PDF-1.4\n1 0 obj\n<< /Length " + str(len(stream)).encode() + entries
    body += b" >>\nstream\n" + stream + b"\nendstream\nendobj\n" + extra
    return body + b"trailer\n<< /Root 1 0 R >>\n%%EOF\n"


def test_a_comment_before_an_indirect_pdf_filter_is_still_a_reference(tmp_path):
    """PDF counts a `%` comment as whitespace, so it may sit between a key and its value.

    The indirect-filter check accepted only whitespace before the reference, so `/Filter%c\\n2 0 R`
    named nothing: the stream was inflated and its raw bytes scanned while a conforming reader
    resolves object 2 to a filter this cannot undo. The spaced spelling was refused correctly.
    """
    import zlib

    from flash.env_secrets import _Unscannable, credential_in_file

    stream = zlib.compress(f"fslo_{_FAKE_KEY_BODY}".encode())
    resolved = b"2 0 obj\n/FlateDecode\nendobj\n"

    # the control: the same reference written with a space is refused
    spaced = tmp_path / "spaced.pdf"
    spaced.write_bytes(_pdf_with_stream(stream, b" /Filter 2 0 R", resolved))
    with pytest.raises(_Unscannable, match="filter this cannot undo"):
        credential_in_file(spaced)

    commented = tmp_path / "commented.pdf"
    commented.write_bytes(_pdf_with_stream(stream, b" /Filter%comment\n2 0 R", resolved))
    with pytest.raises(_Unscannable, match="filter this cannot undo"):
        credential_in_file(commented)


def test_an_indirect_pdf_decode_parameter_is_refused(tmp_path):
    """`/DecodeParms 2 0 R` puts the predictor in another object, out of the dictionary slice.

    The predictor check reads the local dictionary, so an indirect reference declared nothing and
    the inflated bytes were scanned as content while they were still PNG row differences -- a
    conforming decode reconstructs the key, which is why the direct spelling is refused.
    """
    import zlib

    from flash.env_secrets import _Unscannable, credential_in_file

    columns = 8
    raw = f"fslo_{_FAKE_KEY_BODY}".encode()
    rows = [raw[at : at + columns].ljust(columns, b"\0") for at in range(0, len(raw), columns)]
    encoded = bytearray()
    previous = bytes(columns)
    for row in rows:
        encoded += b"\x02"  # the PNG `Up` predictor tag
        encoded += bytes((row[at] - previous[at]) % 256 for at in range(columns))
        previous = row
    stream = zlib.compress(bytes(encoded))
    assert raw not in zlib.decompress(stream)  # inflation alone does not reveal it

    # the control: the same parameters written inline are refused
    direct = tmp_path / "direct.pdf"
    direct.write_bytes(
        _pdf_with_stream(
            stream, b" /Filter /FlateDecode /DecodeParms << /Predictor 12 /Columns 8 >>"
        )
    )
    with pytest.raises(_Unscannable, match="filter this cannot undo"):
        credential_in_file(direct)

    # the referenced object sits beyond the slice the predictor check reads
    filler = b"%" + b"x" * 900 + b"\n"
    indirect = tmp_path / "indirect.pdf"
    indirect.write_bytes(
        _pdf_with_stream(
            stream,
            b" /Filter /FlateDecode /DecodeParms 2 0 R",
            filler + b"2 0 obj\n<< /Predictor 12 /Columns 8 >>\nendobj\n",
        )
    )
    with pytest.raises(_Unscannable, match="filter this cannot undo"):
        credential_in_file(indirect)
