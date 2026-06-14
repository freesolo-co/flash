"""Local OpenAI-compatible HTTP shim for a deployed adapter.

Exposes ``/v1/chat/completions`` and ``/v1/models`` on localhost and forwards to the
AutoSLM control plane's chat endpoint, so ANY OpenAI SDK client works against an
AutoSLM deployment:

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
    client.chat.completions.create(model="autoslm-<run_id>", messages=[...])

Uses only the standard library (http.server) — no fastapi/uvicorn requirement — since
it forwards one request at a time to a single-worker deployment anyway. Streaming is
not supported (responses return whole); cold starts in dev mode mean the FIRST request
after idle can take minutes — that's the explicit dev-mode trade-off.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from autoslm.client import ApiClient


def run_proxy(
    client: ApiClient,
    run_id: str,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    served_model = f"autoslm-{run_id}"
    lock = threading.Lock()  # single-worker deployment -> serialize upstream calls

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._send(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": served_model, "object": "model", "owned_by": "autoslm"}],
                    },
                )
            else:
                self._send(404, {"error": {"message": f"unknown path {self.path}"}})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send(404, {"error": {"message": f"unknown path {self.path}"}})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(length) or b"{}")
                if req.get("stream"):
                    self._send(
                        400,
                        {"error": {"message": "streaming is not supported by the AutoSLM proxy"}},
                    )
                    return
                with lock:
                    resp = client.chat(
                        run_id,
                        messages=req.get("messages") or [],
                        temperature=float(req.get("temperature") or 0.0),
                        max_tokens=int(req.get("max_tokens") or 512),
                    )
                self._send(200, resp)
            except Exception as exc:
                self._send(502, {"error": {"message": f"upstream failure: {exc}"}})

        def log_message(self, *_args):  # quiet
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        # Ctrl-C is the normal way to stop the local proxy; shut down cleanly via finally.
        pass
    finally:
        server.server_close()
