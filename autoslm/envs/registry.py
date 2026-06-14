"""Environment registry used by specs, worker, CLI, and server."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from .base import Environment, load_environment_from_path
from .freesolo import load_environment as load_freesolo
from .tests_pass import load_environment as load_tests_pass

# Example environments live next to the package (examples/<name>/), outside the
# importable package tree so the core package carries no task-specific code. The
# wheel force-includes examples/ (see pyproject.toml), so they are present in wheel
# installs too. They are loaded by path and *lazily* — importing the registry must
# not import the examples, which would cycle back through
# envs.base -> envs/__init__ -> registry.
_EXAMPLES_ROOT = Path(__file__).resolve().parents[2] / "examples"
_EXAMPLE_ENVS = ("gsm8k", "math")


def _example_loader(name: str):
    """Return a loader that path-imports examples/<name>/ as a package on demand."""

    def _load(**params) -> Environment:
        mod_name = f"autoslm_example_{name}"
        mod = sys.modules.get(mod_name)
        if mod is None:
            pkg_dir = _EXAMPLES_ROOT / name
            init = pkg_dir / "__init__.py"
            if not init.exists():
                raise FileNotFoundError(
                    f"example environment {name!r} not found at {pkg_dir}; the examples/ "
                    "tree should be present in source checkouts and wheel installs alike "
                    "(force-included) — this likely means an incomplete install/build"
                )
            spec = importlib.util.spec_from_file_location(
                mod_name, init, submodule_search_locations=[str(pkg_dir)]
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
        return mod.load_environment(**params)

    return _load


_BUILTINS = {
    "freesolo": load_freesolo,
    "tests_pass": load_tests_pass,
    **{name: _example_loader(name) for name in _EXAMPLE_ENVS},
}


def _freesolo_worker_pip(params: dict) -> list[str]:
    """The freesolo bridge needs the freesolo package on the worker only when it
    reconstructs an environment for scoring (GRPO); SFT jobs are self-contained."""
    return [] if (params or {}).get("mode") == "sft" else ["freesolo[full]"]


# Built-ins whose worker-side execution needs extra pip packages.
_BUILTIN_WORKER_PIP = {
    "freesolo": _freesolo_worker_pip,
}

# Manifest of installed verifiers / Prime Hub environments (written by `slm env install`).
INSTALLED_MANIFEST = Path(
    os.environ.get("AUTOSLM_ENVS_MANIFEST", str(Path.home() / ".autoslm" / "envs.json"))
)


def list_environments() -> list[str]:
    return sorted(_BUILTINS)


def load_installed_manifest() -> dict:
    try:
        return json.loads(INSTALLED_MANIFEST.read_text())
    except (OSError, ValueError):
        return {}


def list_installed_verifiers_envs() -> list[str]:
    """Names of verifiers/Hub environments installed via `slm env install`."""
    return sorted(load_installed_manifest())


def record_installed_env(env_id: str, package: str, extras: dict | None = None) -> None:
    manifest = load_installed_manifest()
    manifest[env_id] = {"package": package, **(extras or {})}
    INSTALLED_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    # The manifest can hold a credentialed --extra-index-url. Create/truncate with 0600
    # from the start (not write_text + chmod, which leaves it umask-readable in between);
    # O_NOFOLLOW refuses a symlink planted at the path. chmod after covers a pre-existing
    # file created before this code path.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(INSTALLED_MANIFEST, flags, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    with contextlib.suppress(OSError):
        os.chmod(INSTALLED_MANIFEST, 0o600)


def _bare_wheel_name(env_ref: str) -> str:
    """``owner/name`` Hub slug -> the bare pip wheel name (``name``)."""
    return env_ref.split("/", 1)[1] if "/" in env_ref else env_ref


def worker_pip_for_env(env_id: str, params: dict | None = None) -> list[str]:
    """Pip requirements the GPU worker needs to run ``env_id`` (verifiers Hub envs).

    Installs ``verifiers`` + the recorded env wheel, and carries any ``extra_index_url``
    recorded at ``slm env install`` time (e.g. the Prime Intellect Hub index) through to
    the worker's ``pip install``.

    A non-built-in env MUST be recorded (``slm env install <env>``) so the wheel name and
    index are known. We deliberately do NOT guess a wheel name from an ``owner/name`` slug
    — that could install an unrelated PyPI package on a name collision and hides the real
    requirement; callers that want to bypass the manifest set ``[environment] pip`` instead
    (the caller prefers ``spec.environment.pip`` over this function).
    """
    params = params or {}
    if env_id in _BUILTINS:
        extra = _BUILTIN_WORKER_PIP.get(env_id)
        return list(extra(params)) if callable(extra) else list(extra or [])
    manifest = load_installed_manifest()
    entry = manifest.get(env_id)
    if entry is None:
        raise ValueError(
            f"environment {env_id!r} is not a built-in and is not recorded as installed. "
            f"Run `slm env install {env_id}` first, or set [environment] pip = [...] in "
            "the config to declare the worker requirements explicitly."
        )
    deps = ["verifiers", entry.get("package") or _bare_wheel_name(env_id)]
    out = list(dict.fromkeys(deps))  # dedupe, preserve order
    idx = entry.get("extra_index_url")
    if idx:
        out += ["--extra-index-url", idx]
    return out


def load_environment(
    env_id: str, params: dict | None = None, path: str | None = None
) -> Environment:
    params = params or {}
    if path:
        return load_environment_from_path(path, **params)
    if env_id in _BUILTINS:
        return _BUILTINS[env_id](**params)
    # Fall through to a verifiers / Prime Hub environment (installed via `slm env install`
    # or resolvable by verifiers). This is the verifiers/Hub interop path.
    from .verifiers_adapter import load_verifiers_environment

    try:
        return load_verifiers_environment(env_id, **params)
    except ImportError:
        raise
    except Exception as exc:
        allowed = ", ".join(list_environments())
        raise ValueError(
            f"could not load environment {env_id!r} as a built-in ({allowed}), a verifiers/Hub "
            f"env, or a path: {exc}"
        ) from exc
