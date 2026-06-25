"""Release channel of the installed distribution — the single switch between prod and dev.

The production package ``freesolo-flash`` ships ``CHANNEL = "prod"`` (the checked-in default
below): it installs a ``flash`` CLI that talks to the production control plane. The dev-channel
package ``freesolo-flash-dev`` is built from this *same source* with only this one line rewritten
to ``CHANNEL = "dev"`` (see ``scripts/build_dev_dist.py``); everything that differs between the two
channels — the CLI name, the PyPI distribution name, the default control-plane URL — derives from
it below, so there is exactly one thing to flip. An explicit ``FLASH_API_URL`` /
``flash login --api-url`` always wins; the channel only picks the *default* plane.
"""

from __future__ import annotations

# The one line scripts/build_dev_dist.py rewrites to "dev" for the dev-channel build.
CHANNEL = "prod"

# Console-script + argparse program name. Kept in lockstep with [project.scripts] in
# pyproject.toml (which the build script also rewrites: flash -> flash-dev).
CLI_NAME = "flash-dev" if CHANNEL == "dev" else "flash"

# PyPI distribution name. Used to read back __version__ from installed metadata and to point the
# update check at the right project (kept in lockstep with [project].name in pyproject.toml).
DIST_NAME = "freesolo-flash-dev" if CHANNEL == "dev" else "freesolo-flash"
