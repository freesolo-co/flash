"""fixed modal web-server wrapper for the packaged root launcher."""

from __future__ import annotations


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

    The consequence is that a dead engine is not recovered by this process exiting -- see
    `_exit_on_engine_death` in `app/__main__.py`, which asks uvicorn to exit for exactly that
    purpose. That signal reaches modal through the child's own exit and the port going unreachable,
    not through this wrapper. Do not add a `process.wait()` here to close that gap; it trades a
    recovery-path weakness for a total outage.
    """

    from flash.serve.app.launch import start_launcher_process

    start_launcher_process()
