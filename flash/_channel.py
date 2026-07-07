"""Single switch between prod and dev channels; ``scripts/build_dev_dist.py`` rewrites CHANNEL to "dev"."""

from __future__ import annotations

CHANNEL = "prod"

# Must stay in lockstep with [project.scripts] in pyproject.toml.
CLI_NAME = "flash-dev" if CHANNEL == "dev" else "flash"

# Must stay in lockstep with [project].name in pyproject.toml.
DIST_NAME = "freesolo-flash-dev" if CHANNEL == "dev" else "freesolo-flash"

# The chalk kernel package flash installs on workers (providers/_worker.py): the dev channel
# installs the dev-channel chalk build (freesolo-chalk-dev), prod installs stable freesolo-chalk.
CHALK_DIST = "freesolo-chalk-dev" if CHANNEL == "dev" else "freesolo-chalk"
