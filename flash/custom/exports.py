"""One temporary ZIP export per session, separate from its bounded checkpoint cache."""

import shutil
import tempfile
import time
from pathlib import Path


def checkpoint_response(session, name):
    from fastapi import HTTPException
    from starlette.concurrency import run_in_threadpool
    from starlette.responses import StreamingResponse

    with session.lock:
        if session.active_exports:
            raise HTTPException(409, "another checkpoint export is in progress")
        workspace = tempfile.TemporaryDirectory(prefix="flash-custom-export-")
        try:
            backend = session.backend
            store = getattr(backend, "checkpoints", None)
            source = (
                store.export_source(name, Path(workspace.name))
                if store is not None
                else backend.checkpoint_dir / name
            )
            if not (source / "state.pt").is_file():
                raise FileNotFoundError("unknown checkpoint")
            path = Path(shutil.make_archive(str(Path(workspace.name) / name), "zip", source))
            size = path.stat().st_size
            handle = path.open("rb")
        except BaseException as error:
            workspace.cleanup()
            if isinstance(error, FileNotFoundError):
                raise HTTPException(404, "unknown checkpoint") from error
            raise
        session.active_exports += 1
        session.last_activity = time.monotonic()

    cleaned = False

    def cleanup():
        nonlocal cleaned
        with session.lock:
            if cleaned:
                return
            cleaned = True
            try:
                handle.close()
                workspace.cleanup()
            finally:
                session.active_exports -= 1
                session.last_activity = time.monotonic()

    async def content():
        try:
            while chunk := await run_in_threadpool(handle.read, 1024 * 1024):
                yield chunk
        finally:
            cleanup()

    class CheckpointResponse(StreamingResponse):
        async def __call__(self, scope, receive, send):
            try:
                await super().__call__(scope, receive, send)
            finally:
                # also release resources if sending headers fails before the iterator starts.
                cleanup()

    return CheckpointResponse(
        content(),
        media_type="application/zip",
        headers={
            "Content-Length": str(size),
            "Content-Disposition": f'attachment; filename="{name}.zip"',
        },
    )
