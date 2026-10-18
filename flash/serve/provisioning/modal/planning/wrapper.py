"""fixed modal web-server wrapper for the packaged root launcher."""

from __future__ import annotations

from os import getpid, kill
from signal import SIGTERM
from threading import Thread


def launch_modal_server() -> None:
    """start serving and return so Modal can probe the port and set up its proxy.

    a background thread watches the launcher. when it exits, the thread signals the parent
    so Modal can shut down and replace the container without waiting for a failed request.
    """

    from flash.serve.app.launch import (
        LauncherShutdownUnconfirmed,
        start_launcher_process,
        stop_launcher_process,
    )

    process = start_launcher_process()

    def stop_parent_when_child_exits() -> None:
        # a healthy server can run indefinitely; this daemon thread does not block startup or exit.
        process.wait()
        kill(getpid(), SIGTERM)

    try:
        watcher = Thread(target=stop_parent_when_child_exits, daemon=True)
        watcher.start()
    except BaseException as startup_error:
        try:
            stop_launcher_process(process)
        except LauncherShutdownUnconfirmed as cleanup_error:
            raise cleanup_error from startup_error
        raise
