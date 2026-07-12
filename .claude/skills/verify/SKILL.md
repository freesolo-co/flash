# flash verification

Use isolated loopback services to exercise the public FastAPI and CLI surfaces.

1. set `HOME` to a temporary directory so run JSON and `server.db` stay isolated.
2. launch a synthetic serving backend on an unused localhost port that implements `/adapters` and `/v1/chat/completions` and records request paths.
3. launch `flash.server.app.create_app()` with uvicorn through a temporary launcher. patch startup credential and external-provider preflights only, and set `FREESOLO_INTERNAL_KEY`, `FLASH_DEPLOY_SYNC=1`, and `FREESOLO_SERVING_URL` to the synthetic backend.
4. drive the real HTTP routes with curl using the internal bearer key. create dry-run records through `POST /v1/runs`; when a completed artifact is needed, edit only the isolated temporary run JSON before driving deploy, chat, list, cancel, and undeploy routes.
5. capture API response bodies and the synthetic backend request log. stop both loopback processes after verification.

Use `uv run --project <repo> python ...` for all Python launchers. Never use production credentials or network services.
