"""The one place Flash decides where its local state lives.

Everything Flash persists on the machine it runs on -- the CLI's saved login, the control
plane's SQLite database, run records and results, and the RunPod SDK's resource registry --
hangs off a single root. ``FLASH_DATA_DIR`` moves that root; unset, it is ``~/.flash``, which
is where every one of those things already lived.

Why this module exists at all: four call sites used to spell ``~/.flash`` out for themselves.
Nothing kept them in agreement, so a deployment could only relocate state by editing Python in
four places and state could silently split if it edited three. Resolution is centralized here
so there is exactly one answer to "where does Flash keep its files".

The root is resolved per call rather than captured at import, because a process that sets the
variable after importing flash (tests, embedders) would otherwise get the old answer.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV = "FLASH_DATA_DIR"
DEFAULT_DATA_DIR_NAME = ".flash"


def data_dir() -> Path:
    """The root under which Flash keeps its local state.

    A relative ``FLASH_DATA_DIR`` is resolved against the current working directory, which makes
    the path a process-lifetime-stable answer only if the process does not chdir. Absolute paths
    are what a deployment should set; ``expanduser`` keeps ``~/...`` working for a human typing
    it into a shell.
    """
    configured = os.environ.get(DATA_DIR_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / DEFAULT_DATA_DIR_NAME
