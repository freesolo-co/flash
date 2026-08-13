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

    from flash.env_secrets import credential_in_file

    plain = tmp_path / "plain.zip"
    with zipfile.ZipFile(plain, "w") as archive:
        archive.writestr("member.txt", "ordinary content " * 20)
    source = plain.read_bytes()

    # bit 0 of the general-purpose flag marks a member encrypted -> RuntimeError
    encrypted = tmp_path / "encrypted.zip"
    encrypted.write_bytes(_flip_zip_flags(source, offset_local=6, offset_central=8, value=b"\x01"))
    assert credential_in_file(encrypted) is None

    # method 99 is AES, which zipfile does not implement -> NotImplementedError
    unsupported = tmp_path / "unsupported.zip"
    unsupported.write_bytes(
        _flip_zip_flags(source, offset_local=8, offset_central=10, value=struct.pack("<H", 99))
    )
    assert credential_in_file(unsupported) is None

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
    import zipfile

    from flash.env_secrets import credential_in_file

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

    assert credential_in_file(path) is None


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

    from flash.env_secrets import _BASE64_WINDOW, _credential_kind, credential_in_file

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
    """
    import io
    import tarfile

    from flash.env_secrets import credential_in_file

    whole = io.BytesIO()
    with tarfile.open(fileobj=whole, mode="w") as archive:
        body = b"x" * 4096
        info = tarfile.TarInfo("a.txt")
        info.size = len(body)
        archive.addfile(info, io.BytesIO(body))

    truncated = tmp_path / "half.tar"
    truncated.write_bytes(whole.getvalue()[:900])
    assert credential_in_file(truncated) is None


def test_an_oversized_tar_refuses_rather_than_passing(tmp_path, monkeypatch):
    """The buffer-cap escape recognised only compressed magic, so a big tar took the pass branch.

    A tar past the cap is exactly as unverifiable as a gzip past it -- its members can be
    compressed, and those hold the credential nowhere the literal scan can see.
    """
    import gzip
    import io
    import os
    import tarfile

    from flash import env_secrets as secrets
    from flash.env_secrets import _Unscannable, credential_in_file

    monkeypatch.setattr(secrets, "_MAX_NESTED_BUFFER_BYTES", 1 << 20)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as archive:
        for name, body in (
            ("pad.bin", os.urandom(4 << 20)),
            ("shard.gz", gzip.compress(f'export KEY="fslo_{_FAKE_KEY_BODY}"\n'.encode())),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))

    oversized = tmp_path / "big.tar.gz"
    oversized.write_bytes(gzip.compress(buf.getvalue()))
    with pytest.raises(_Unscannable, match="too large to inspect"):
        credential_in_file(oversized)


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
