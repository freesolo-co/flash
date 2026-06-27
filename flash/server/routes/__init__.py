"""Control-plane API routers, grouped by domain.

Each module exposes an ``APIRouter`` named ``router`` that ``create_app`` mounts. These modules
import fastapi (and ``flash.server._deps``), so they must only be imported from ``create_app()``
— never at ``flash.server.app`` import time, which keeps the optional-server-extras guard intact.
"""
