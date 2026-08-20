"""fixed modal web-server wrapper for the packaged root launcher."""

from __future__ import annotations


def launch_modal_server() -> None:
    """start the packaged child boundary after it scrubs runtime secret inputs."""

    from flash.serve.app.launch import start_launcher_process

    start_launcher_process()
