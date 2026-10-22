"""Local HTTP transport for the authenticated OPD teacher bridge."""

import http.client
import json
from http.server import BaseHTTPRequestHandler


def make_teacher_bridge_handler(bridge):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def _send_json(self, status: int, payload: dict) -> None:
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self):
            recovered_failures: list[tuple[str, str]] = []
            request_succeeded = False
            try:
                if self.headers.get("Authorization") != f"Bearer {bridge.token}":
                    raise PermissionError("flash OPD bridge authorization failed")
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                route = bridge._routes(recovered_failures).get(self.path)
                if route is None:
                    raise ValueError("flash OPD bridge path is unknown")
                result = route(payload)
                request_succeeded = True
                self._send_json(200, result)
            except Exception as error:
                # only a route that already answered can be failing on DELIVERY; before that, an
                # OSError is the teacher call itself.
                delivery = (
                    request_succeeded
                    and self.path in {"/score", "/multiturn/score"}
                    and isinstance(error, (OSError, http.client.HTTPException))
                )
                classification = bridge._classify_failure(error, delivery=delivery)
                bridge._record_route_failure(
                    self.path,
                    error,
                    classification,
                    delivery=delivery,
                    recovered=recovered_failures[0] if recovered_failures else None,
                )
                self._send_json(
                    503 if classification == "transient" else 422,
                    {
                        "error": {
                            "classification": classification,
                            "message": str(error),
                        }
                    },
                )

    return Handler
