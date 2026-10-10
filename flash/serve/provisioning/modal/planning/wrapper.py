"""fixed modal web-server wrapper for the packaged root launcher."""

from __future__ import annotations

from os import getpid, kill
from signal import SIGTERM
from threading import Thread


def launch_modal_server() -> None:
    """spawn the packaged child boundary and return immediately.

    Returning while the child still runs is required, not an oversight. Modal's runtime calls this
    function and only *afterwards* probes the port and builds the proxy that serves the endpoint
    (`modal/_runtime/user_code_imports.py`, `WEBHOOK_TYPE_WEB_SERVER`)::

        user_defined_callable()
        host = asgi.get_ip_address(b"eth0")
        asgi.wait_for_web_server(host, port, timeout=startup_timeout)
        return asgi.asgi_app_wrapper(asgi.web_server_proxy(host, port), container_io_manager)

    Waiting on the child here never returns, so `wait_for_web_server` and the proxy are never
    reached and the endpoint serves nothing at all. `startup_timeout` is the deadline for the port
    to become reachable, not for this call to finish, and modal's own documented example is the
    same spawn-and-return shape (`subprocess.Popen(...)` with no wait).

    Modal notices a post-startup dead port when a request fails and then gracefully stops the
    container, but an idle child exit otherwise leaves a dead-port window and usually sacrifices
    that first request. Do not add a blocking `process.wait()` here; it trades that recovery gap for
    a total outage. A daemon watcher waits off the caller thread and sends SIGTERM to this supervised
    parent on any child exit, including uvicorn's clean engine-death exit, so Modal's installed
    handler performs its normal container cleanup and replacement.
    """

    from flash.serve.app.launch import start_launcher_process

    process = start_launcher_process()

    def stop_parent_when_child_exits() -> None:
        process.wait()
        kill(getpid(), SIGTERM)

    Thread(target=stop_parent_when_child_exits, daemon=True).start()
