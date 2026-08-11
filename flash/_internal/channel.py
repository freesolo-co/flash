"""Single switch between prod and dev channels; ``scripts/build_dev_dist.py`` rewrites CHANNEL to "dev"."""

import os
import sys

CHANNEL = "prod"

# Must stay in lockstep with [project.scripts] in pyproject.toml.
_DEFAULT_CLI_NAME = "flash-dev" if CHANNEL == "dev" else "flash"

# Every console script in [project.scripts] that enters flash.cli:main, per channel. `flash` is
# only correct to PRINT when the operator actually reached us through it: the `server` and `dev`
# extras install runpod-flash, which claims the same script name, so on such a host `flash` may be
# RunPod's CLI. Printing `flash runs cancel <id>` there tells the operator to run a command that
# exits 0 without cancelling, leaving a billed run alive.
_CLI_SCRIPT_NAMES = (
    frozenset({"flash-dev", "flash-dev-cli"})
    if CHANNEL == "dev"
    else frozenset({"flash", "flash-cli"})
)

# The `-m` form of the same entry point, which README/CONTRIBUTING/SELF_HOSTING all name as the
# way around the shadowing. Not channel-dependent: the dev build renames the console scripts but
# ships the identical `flash` package, so this module path is its escape hatch too.
_CLI_MODULE = "flash.cli"


def _entered_via_dash_m() -> bool:
    """Whether this process was started as `python -m flash.cli`.

    Two signals, because neither alone is sound at the moment this runs. Under ``-m``, runpy
    imports the target's parent package -- which is what imports this module -- BEFORE it rewrites
    ``sys.argv``, so ``sys.argv[0]`` is still the bare string ``"-m"``. That proves a module launch
    is underway but not WHICH module, since ``python -m pytest`` looks identical. The module name
    lives in ``sys.orig_argv``, the interpreter's untouched command line (3.10+).

    Its position is derived rather than searched: ``sys.argv[1:]`` is exactly the trailing script
    arguments, so the module sits one slot before them, at ``orig_argv[-len(sys.argv)]``. Scanning
    for ``"-m"`` instead would mistake a script's own ``-m`` argument for the interpreter flag, and
    would miss the real one behind interpreter flags that take values (``python -W ignore -m ...``).
    """
    if not sys.argv or sys.argv[0] != "-m":
        return False
    orig = getattr(sys, "orig_argv", None) or []  # absent when embedded, or before 3.10
    if len(orig) < len(sys.argv) + 1:
        return False
    return orig[-len(sys.argv)] == _CLI_MODULE


def _invoked_cli_name() -> str:
    """The name to print commands under: the console script typed, when it is one of ours.

    `python -m flash.cli` is the documented way OUT of the shadowing, so it is exactly the case
    that must not fall back to `flash`: an operator who reached us that way is on a host where
    `flash` may be RunPod's CLI, and echoing `flash` back sends them to a command that exits 0
    without doing anything. Name the same escape hatch they used instead.

    Anything else -- a test runner, a renamed copy -- keeps the channel default, which is correct
    on a client-only install and stays copy-pasteable rather than naming `__main__.py`.
    """
    if _entered_via_dash_m():
        return f"python -m {_CLI_MODULE}"
    argv0 = os.path.basename((sys.argv[0] or "").strip()) if sys.argv else ""
    if argv0.endswith(".exe"):  # windows console scripts
        argv0 = argv0[: -len(".exe")]
    return argv0 if argv0 in _CLI_SCRIPT_NAMES else _DEFAULT_CLI_NAME


CLI_NAME = _invoked_cli_name()

# Must stay in lockstep with [project].name in pyproject.toml.
DIST_NAME = "freesolo-flash-dev" if CHANNEL == "dev" else "freesolo-flash"
