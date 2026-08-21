"""fixed modal web-server wrapper for the packaged root launcher."""

from __future__ import annotations


def launch_modal_server() -> None:
    """start the packaged child boundary and stay alive for exactly as long as it does.

    `@modal.web_server` calls this once and treats it as the container's lifetime, but
    `start_launcher_process` spawns the child and returns a live `Popen` immediately. Returning
    that handle to modal instead of waiting on it made the wrapper finish within milliseconds
    while the server it started kept running behind it.

    The damage is not the orphan itself -- modal keeps the container up for the web server -- but
    that it severs the one mechanism the app has for recovering from a dead engine.
    `_exit_on_engine_death` in `app/__main__.py` asks uvicorn to exit precisely so the process
    ends and "both providers treat [it] as a container to restart". With the handle discarded
    nothing observes that exit, so the child dies, the container stays up bound to a port nothing
    is listening on, and every later request fails. The topology makes it terminal rather than
    merely degraded: `_modal_plan.py` pins `max_containers=1` and `min_containers=0`, so there is
    no second container to answer and no replacement is ever triggered.

    Propagating the child's exit code keeps that signal intact for modal, and reusing
    `terminate_and_reap` means a wrapper interrupted mid-wait tears the child down on the way out
    rather than leaving a gpu process running on a container modal believes is finished.
    """

    from flash.serve.app.launch import start_launcher_process, terminate_and_reap

    process = start_launcher_process()
    try:
        exit_code = process.wait()
    except BaseException:
        terminate_and_reap(process)
        raise
    if exit_code != 0:
        raise SystemExit(exit_code)
