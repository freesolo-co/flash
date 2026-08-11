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


def _invoked_cli_name() -> str:
    """The console script the user actually typed, when it is one of ours.

    Anything else -- `python -m flash.cli`, a test runner, a renamed copy -- falls back to the
    channel default, so generated commands stay copy-pasteable rather than naming `__main__.py`.
    """
    argv0 = os.path.basename((sys.argv[0] or "").strip()) if sys.argv else ""
    if argv0.endswith(".exe"):  # windows console scripts
        argv0 = argv0[: -len(".exe")]
    return argv0 if argv0 in _CLI_SCRIPT_NAMES else _DEFAULT_CLI_NAME


CLI_NAME = _invoked_cli_name()

# Must stay in lockstep with [project].name in pyproject.toml.
DIST_NAME = "freesolo-flash-dev" if CHANNEL == "dev" else "freesolo-flash"
