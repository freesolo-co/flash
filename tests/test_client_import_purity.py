"""`pip install freesolo-flash` must stay pure standard library.

[project] dependencies is empty: every runtime dep lives in an extra (`gpu` for the worker,
`server` for the control plane). So a module-level `import httpx` on a client-reachable path
is not a slow import, it is a crash for anyone who installed the base package -- and CI cannot
see it, because CI syncs `--extra server --dev` and therefore has every extra installed.

The check runs in a subprocess whose sys.path carries only the repo and the standard library.
Running it in-process would import against the fully-provisioned test venv and pass no matter
what the code does.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import sysconfig
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# imported only by the control plane (`server` extra) or by the gpu worker (`gpu` extra), never
# by a client command. deployment/deploy.py and preflight.py are reached exclusively through
# function-local imports in flash/runner/ and flash/server/, which keeps httpx and jsonschema off
# the client path. use exact relocated module prefixes: `flash.serve.deploy` would also match every
# client-reachable module under `flash.serve.deployment`.
SERVER_ONLY_PREFIXES = (
    "flash.server",
    "flash.engine.worker",
    "flash.serve.app",
    "flash.serve.deployment.deploy",
    "flash.serve.deployment.export",
    "flash.serve.deployment.preflight",
    # The serving app runs only inside the GPU container built from Dockerfile.serve, which
    # installs the `serving` extra (modal, orjson, pydantic-settings) plus vLLM. Nothing a client
    # imports reaches it, so its fastapi/pydantic/PIL imports stay at module scope.
    "flash.serving",
)

_PROBE = r"""
import importlib
import json
import os
import sys

import flash

SERVER_ONLY_PREFIXES = tuple(json.loads(os.environ["FLASH_SERVER_ONLY"]))

package_dir = flash.__path__[0]
parent = os.path.dirname(package_dir)

modules = set()
for root, dirs, files in os.walk(package_dir):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for filename in files:
        if not filename.endswith(".py"):
            continue
        rel = os.path.relpath(os.path.join(root, filename[:-3]), parent)
        name = rel.replace(os.sep, ".")
        if name.endswith(".__init__"):
            name = name[: -len(".__init__")]
        modules.add(name)

failures = []
for name in sorted(modules):
    if name.startswith(SERVER_ONLY_PREFIXES):
        continue
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as exc:
        # Only a missing third-party distribution is a purity violation; a missing flash.* module
        # would be an unrelated packaging bug and should not be reported as one.
        if not (exc.name or "").startswith("flash"):
            failures.append({"module": name, "missing": exc.name})
    except Exception:
        # Import-time side effects (needs config, touches the network) are out of scope here.
        # Only the dependency question matters, and a non-ModuleNotFoundError answers it "fine".
        pass

print(json.dumps(failures))
"""


def _stdlib_only_path() -> list[str]:
    """Repo + stdlib, with site-packages removed, so no extra is importable."""
    stdlib = {sysconfig.get_path("stdlib"), sysconfig.get_path("platstdlib")}
    keep = [str(REPO_ROOT)]
    for entry in sys.path:
        if not entry or "site-packages" in entry or "dist-packages" in entry:
            continue
        resolved = str(Path(entry).resolve())
        if resolved in {str(Path(p).resolve()) for p in stdlib if p} or entry.endswith(".zip"):
            keep.append(entry)
    return keep


def _import_failures(extra_env: dict[str, str] | None = None) -> list[dict[str, str]]:
    env = {
        "PYTHONPATH": ":".join(_stdlib_only_path()),
        "PYTHONNOUSERSITE": "1",
        "PATH": "/usr/bin:/bin",
        "FLASH_SERVER_ONLY": json.dumps(list(SERVER_ONLY_PREFIXES)),
        **(extra_env or {}),
    }
    proc = subprocess.run(
        [sys.executable, "-S", "-c", _PROBE],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )
    assert proc.returncode == 0, f"probe crashed:\n{proc.stderr}"
    return json.loads(proc.stdout)


def test_client_modules_import_without_any_third_party_package():
    failures = _import_failures()
    assert failures == [], (
        "these client-reachable modules import a package that `pip install freesolo-flash` does "
        "not install, so the base install crashes on them: "
        + ", ".join(f"{f['module']} needs {f['missing']}" for f in failures)
        + ". Move the import inside the function that uses it, or list the module in "
        "SERVER_ONLY_PREFIXES if it is genuinely server-only or worker-only."
    )


def test_serving_contract_imports_only_shared_identity_modules():
    contract = REPO_ROOT / "flash" / "serve" / "contract"
    allowed_flash_prefixes = ("flash.schema", "flash.serve.contract")
    violations: list[str] = []

    for path in sorted(contract.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                allowed_flash = any(
                    module == prefix or module.startswith(prefix + ".")
                    for prefix in allowed_flash_prefixes
                )
                root = module.split(".", 1)[0]
                if (
                    module != "__future__"
                    and root not in sys.stdlib_module_names
                    and not allowed_flash
                ):
                    violations.append(f"{path.name}: {module}")

    assert violations == [], (
        "flash.serve.contract is shared identity and HTTP protocol; move module-level deployment, "
        "control, provisioning, app, runtime, or provider-aware imports to their owning package: "
        + ", ".join(violations)
    )


def test_chat_transport_boundaries_import_without_httpx():
    env = {
        "PYTHONPATH": ":".join(_stdlib_only_path()),
        "PYTHONNOUSERSITE": "1",
        "PATH": "/usr/bin:/bin",
    }
    proc = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import flash.serve.request.transport; import flash.serve.request.streaming",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )

    assert proc.returncode == 0, proc.stderr


def test_probe_detects_a_planted_violation(tmp_path):
    """The guard above is worthless if it cannot fail, so prove it fails on a real violation.

    Copies the tree, plants a module-level third-party import on a client path, and requires the
    probe to report it. A green run of the test above only means something if this one is red
    whenever the invariant is actually broken.
    """
    import shutil

    tree = tmp_path / "repo"
    shutil.copytree(
        REPO_ROOT / "flash",
        tree / "flash",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    target = tree / "flash" / "cost" / "__init__.py"
    source = target.read_text().split("\n")
    for index, line in enumerate(source):
        if line.startswith("from __future__"):
            # Must land after `from __future__`, which is required to be the first statement;
            # planting above it raises SyntaxError and tests the wrong failure.
            source.insert(index + 1, "import httpx  # planted by the test")
            break
    else:  # pragma: no cover - flash/cost/__init__.py has carried this import for its whole life
        raise AssertionError("expected a `from __future__` line to plant after")
    target.write_text("\n".join(source))

    env = {
        "PYTHONPATH": ":".join([str(tree), *_stdlib_only_path()[1:]]),
        "PYTHONNOUSERSITE": "1",
        "PATH": "/usr/bin:/bin",
        "FLASH_SERVER_ONLY": json.dumps(list(SERVER_ONLY_PREFIXES)),
    }
    proc = subprocess.run(
        [sys.executable, "-S", "-c", _PROBE],
        capture_output=True,
        text=True,
        cwd=str(tree),
        env=env,
    )
    assert proc.returncode == 0, f"probe crashed:\n{proc.stderr}"
    failures = json.loads(proc.stdout)

    assert any(f["module"] == "flash.cost" and f["missing"] == "httpx" for f in failures), (
        f"the probe did not notice a module-level `import httpx` in flash/cost/__init__.py, so "
        f"it cannot protect the base install. reported: {failures}"
    )


def test_implementation_package_markers_import_no_children():
    markers = (
        "flash.cli",
        "flash.cli.commands",
        "flash.runner",
        "flash.engine.worker",
    )
    code = r"""
import importlib
import json
import sys

markers = tuple(json.loads(sys.argv[1]))
loaded = {}
for marker in markers:
    before = set(sys.modules)
    importlib.import_module(marker)
    loaded[marker] = sorted(
        name for name in set(sys.modules) - before if name.startswith(marker + ".")
    )
print(json.dumps(loaded))
"""
    proc = subprocess.run(
        [sys.executable, "-S", "-c", code, json.dumps(markers)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={
            "PYTHONPATH": ":".join(_stdlib_only_path()),
            "PYTHONNOUSERSITE": "1",
            "PATH": "/usr/bin:/bin",
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {marker: [] for marker in markers}
