"""Single switch between prod and dev channels; ``scripts/build_dev_dist.py`` rewrites CHANNEL to "dev"."""

import os
import shlex
import sys

CHANNEL = "prod"

# Must stay in lockstep with [project.scripts] in pyproject.toml.
_DEFAULT_CLI_NAME = "flash-dev" if CHANNEL == "dev" else "flash"

# The product's name, for places that IDENTIFY the tool rather than tell the operator what
# to type: the styled wordmark, `flash version`, `--version`. Fixed per channel, never per
# invocation -- reaching the same tool through a different entry point does not rename it, and
# `python -m flash.cli 1.1.43` reads as a command someone forgot to finish rather than a version
# banner.
BRAND_NAME = _DEFAULT_CLI_NAME

# every console script in [project.scripts] that enters flash.cli:main, per channel. preserve the
# exact name the operator used when rendering follow-up commands.
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

    That slot holds either ``flash.cli`` or ``-mflash.cli``: a short option may be written with its
    value attached, and ``python -mflash.cli`` is an ordinary invocation, not a curiosity. Reading
    only the separated form makes the attached one fall back to the default script name.
    """
    if not sys.argv or sys.argv[0] != "-m":
        return False
    orig = getattr(sys, "orig_argv", None) or []  # absent when embedded, or before 3.10
    if len(orig) < len(sys.argv) + 1:
        return False
    slot = orig[-len(sys.argv)]
    return slot == _CLI_MODULE or slot == f"-m{_CLI_MODULE}"


def _on_windows() -> bool:
    """Whether printed commands need Windows quoting.

    A seam, because tests cannot mutate process-wide ``os.name``: doing so breaks ``pathlib.Path``
    on 3.11. ``flash.cli.commands.env.push`` carries its own copy for the same reason; that one
    may return "" to refuse an unsafe destination, which this caller cannot do, so they stay
    separate rather than share a helper this module would have to import back from its own importer.
    """
    return os.name == "nt"


# cmd.exe's own command separators, all of which are legal in a Windows path and all of which
# double quotes render inert. `%` and `!` are deliberately absent: they are variable expansion, not
# splitting, and quoting does not stop either -- but neither breaks the command into two, which is
# the failure this guards. `"` cannot appear in a Windows path at all.
_CMD_METACHARACTERS = frozenset("&|<>^()")


def _quote_interpreter(path: str) -> str:
    """Quote an interpreter path as one token for the shell it will be pasted into.

    `shlex.quote` is POSIX-only, and not merely suboptimal on Windows: a backslash is outside its
    safe set, so EVERY Windows path -- `C:\\Python311\\python.exe`, which needed no quoting at all
    -- comes back wrapped in single quotes. cmd.exe does not strip those, so it looks for a program
    whose name literally starts with a quote and fails. That would break the ordinary Windows
    install to fix the spaced one.

    `list2cmdline` is the Windows counterpart: it leaves that path bare and double-quotes only when
    a space or tab makes it necessary, which both cmd.exe and PowerShell honour.

    Whitespace is not the only thing that needs the quotes, though. `list2cmdline` quotes for the C
    runtime's argv parser, which runs AFTER cmd.exe has already split on its own metacharacters, so
    `C:\\Tools&SDK\\python.exe` comes back bare and the `&` ends the command -- running `SDK\\...`
    as a second one. Force the quotes when any of those are present.
    """
    if _on_windows():
        # deferred rather than top-level: unused off Windows, and this module is imported early
        import subprocess

        quoted = subprocess.list2cmdline([path])
        if quoted == path and _CMD_METACHARACTERS & set(path):
            return f'"{path}"'
        return quoted
    return shlex.quote(path)


def _module_launch_interpreter() -> str:
    """The interpreter to print `-m` commands under: the one this process is actually running.

    Hardcoding `python` reintroduces this module's own bug in a second form. `python` is absent on
    a host that ships only `python3`, and inside a virtualenv invoked by absolute path it may
    resolve to a system interpreter with no flash installed -- so the printed cancel command fails
    while the invocation that printed it worked, which is exactly the "exits without cancelling"
    hazard the console-script fix exists to close.

    Prefer the typed word from `sys.orig_argv[0]` when it is a bare name: it resolved for the
    operator moments ago, and it keeps the hint short and copy-pasteable. A path (`./venv/bin/python`,
    an absolute interpreter) is echoed as `sys.executable` instead, because a relative one is only
    valid from the directory they happened to be in.

    The result is quoted for the local shell, because it is pasted into one. `C:\\Program
    Files\\...` and a virtualenv under a spaced directory otherwise split into several argv words,
    so the printed cancel command reaches a path that does not exist -- the same silent non-cancel
    this whole module exists to prevent. Quoting is a no-op for the ordinary unspaced path.
    """
    orig = getattr(sys, "orig_argv", None) or []
    typed = orig[0] if orig else ""
    # a bare word is a PATH lookup that just succeeded; anything with a separator is location-bound
    if typed and os.sep not in typed and (os.altsep is None or os.altsep not in typed):
        return _quote_interpreter(typed)
    return _quote_interpreter(sys.executable) if sys.executable else "python3"


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
        return f"{_module_launch_interpreter()} -m {_CLI_MODULE}"
    argv0 = os.path.basename((sys.argv[0] or "").strip()) if sys.argv else ""
    # Windows filenames are case-insensitive, so `FLASH-CLI.EXE` is the same file our installer
    # wrote and must be recognised as ours; the suffix and the name are folded together, since
    # `.EXE` survives a lowercase-only strip and then fails the match for a second reason. POSIX
    # keeps both exact: there `FLASH-CLI` is a DIFFERENT file, and naming it back would print a
    # command the operator does not have.
    if _on_windows():
        argv0 = argv0.lower()
    if argv0.endswith(".exe"):  # windows console scripts
        argv0 = argv0[: -len(".exe")]
    return argv0 if argv0 in _CLI_SCRIPT_NAMES else _DEFAULT_CLI_NAME


CLI_NAME = _invoked_cli_name()

# Must stay in lockstep with [project].name in pyproject.toml.
DIST_NAME = "freesolo-flash-dev" if CHANNEL == "dev" else "freesolo-flash"
