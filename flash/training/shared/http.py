"""Small authenticated API for an operator-owned shared training pool.

Authentication resolves tenant and owner from server configuration. Request bodies
never choose tenant identity, model code, reward code, credentials, or local paths.
"""

import hashlib
import hmac
import json
from contextlib import asynccontextmanager

from flash.training.shared.contracts import (
    AdmissionError,
    ConflictError,
    Principal,
    UnavailableError,
    fields,
)


def token_authentication(credentials):
    if not credentials:
        raise ValueError("shared pool requires authenticated credentials")
    identities = []
    for token, identity in credentials.items():
        if not isinstance(token, str) or len(token) < 32:
            raise ValueError("pool credentials must contain at least 32 characters")
        identities.append((hashlib.sha256(token.encode()).digest(), Principal(**identity)))

    def authenticate(header):
        if not isinstance(header, str) or not header.startswith("Bearer "):
            return None
        digest = hashlib.sha256(header[7:].encode()).digest()
        for expected, principal in identities:
            if hmac.compare_digest(digest, expected):
                return principal
        return None

    return authenticate


def create_app(runtime, authenticate):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    @asynccontextmanager
    async def lifespan(app):
        runtime.start()
        try:
            yield
        finally:
            runtime.stop()

    app = FastAPI(lifespan=lifespan)

    async def payload(request):
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 2 * 1024 * 1024:
                raise AdmissionError("request body exceeds 2 MiB")
        try:
            return json.loads(raw)
        except (ValueError, UnicodeError) as error:
            raise ValueError("invalid JSON body") from error

    @app.middleware("http")
    async def authorize(request, call_next):
        principal = authenticate(request.headers.get("authorization"))
        if principal is None:
            return JSONResponse({"error": "invalid credentials"}, status_code=401)
        request.state.principal = principal
        if request.method != "GET" and (runtime.failed or runtime.stopping.is_set()):
            return JSONResponse({"error": "shared runtime unavailable"}, status_code=503)
        return await call_next(request)

    async def rejected(request, error):
        code = (
            503
            if isinstance(error, UnavailableError)
            else 429
            if isinstance(error, AdmissionError)
            else 409
            if isinstance(error, ConflictError)
            else 404
            if isinstance(error, KeyError)
            else 400
        )
        return JSONResponse(
            {"error": "unknown shared resource" if code == 404 else str(error)}, status_code=code
        )

    for error_type in (AdmissionError, ConflictError, UnavailableError, KeyError, ValueError):
        app.add_exception_handler(error_type, rejected)

    @app.post("/sessions")
    async def create(request: Request):
        body = await payload(request)
        fields(body, ("key", "config"))
        session = runtime.accept("create", request.state.principal, body["key"], body["config"])
        return {"session": session}

    @app.get("/sessions/{session}")
    def status(session: str, request: Request):
        return runtime.journal.status(request.state.principal, session)

    @app.post("/sessions/{session}/commands")
    async def submit(session: str, request: Request):
        body = await payload(request)
        fields(body, ("key", "operation", "payload"))
        command = runtime.accept(
            "submit",
            request.state.principal,
            session,
            body["key"],
            body["operation"],
            body["payload"],
        )
        return {"command": command}

    @app.delete("/sessions/{session}/commands/{command}")
    def cancel(session: str, command: str, request: Request):
        runtime.accept("cancel", request.state.principal, session, command)
        return {"cancelled": True}

    @app.get("/usage")
    def usage(request: Request):
        return runtime.journal.usage(request.state.principal)

    return app
