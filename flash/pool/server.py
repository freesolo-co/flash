"""Run the rollout-pool router (operator-side, like ``flash-server``).

``flash-pool serve`` starts the FastAPI router on a CPU box; GPU workers and trainers then talk to
it over HTTP. Requires the ``server`` extra (``pip install 'flash[server]'``).
"""

from __future__ import annotations

import os

from flash.pool.config import RouterConfig


def build_app(config: RouterConfig | None = None):
    """Construct the router app (importable by uvicorn / ASGI servers)."""
    from flash.pool.router import create_pool_app

    return create_pool_app(config=config or RouterConfig.from_env())


def _maybe_seed_backends(app) -> None:
    """Optionally pre-register backends listed in ``FLASH_POOL_BACKENDS`` (a comma-separated list of
    ``id=url@base_model`` triples) so a static fleet comes up wired without a separate call."""
    raw = os.environ.get("FLASH_POOL_BACKENDS", "").strip()
    if not raw:
        return
    from flash.pool.state import Backend

    router = app.state.router
    for spec in raw.split(","):
        spec = spec.strip()
        if not spec or "=" not in spec or "@" not in spec:
            continue
        bid, rest = spec.split("=", 1)
        url, base = rest.rsplit("@", 1)
        router.state.add_backend(Backend(id=bid.strip(), url=url.strip(), base_model=base.strip()))


def serve(host: str = "0.0.0.0", port: int = 8077, config: RouterConfig | None = None) -> None:
    import uvicorn

    app = build_app(config)
    _maybe_seed_backends(app)
    uvicorn.run(app, host=host, port=port)
