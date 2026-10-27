"""Immutable, independently addressable one-container GPU deployments."""

import modal

from flash.serving.deepinfer.contract.config import (
    GPU,
    PORT,
    RUNTIME_PYTHON,
    STARTUP_SECONDS,
)


def replica_app(name, image, version):
    app = modal.App(name, image=image)

    @app.server(
        name="Engine",
        serialized=True,
        # The image already contains this release's source. Auto-mounted source
        # from a newer controller must not overwrite it during later scale-out.
        include_source=False,
        gpu=GPU,
        cpu=32,
        memory=524288,
        compute_region="us",
        routing_region="us-east",
        volumes={"/weights": modal.Volume.from_name("freesolo-deepseek-1017-weights")},
        min_containers=1,
        max_containers=1,
        buffer_containers=0,
        target_concurrency=None,
        # HTTP admission enforces MAX_INPUTS and can prove a rejection happened
        # before native work. A proxy-generated rejection cannot attest that.
        max_concurrency=0,
        startup_timeout=STARTUP_SECONDS,
        port=PORT,
        unauthenticated=False,
        exit_grace_period=3600,
        env={"DEEPINFER_ROLLOUT_ID": version, "DEEPINFER_REPLICA_NAME": name},
    )
    class Engine:
        @modal.enter()
        def start(self):
            import os
            import subprocess
            import threading

            process = subprocess.Popen([RUNTIME_PYTHON, "-m", "flash.serving.deepinfer.app.http"])

            def watch():
                process.wait()
                os._exit(1)

            threading.Thread(target=watch, daemon=True).start()

    return app
