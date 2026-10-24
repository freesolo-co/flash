import hashlib
import hmac
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

objects = {}
lock = threading.Lock()
limit = 1024**3


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def authenticated(self):
        return hmac.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + os.environ["KV_RELAY_TOKEN"]
        )

    def do_PUT(self):
        if not self.authenticated():
            self.send_error(403)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= 128 * 1024**2:
            self.send_error(413)
            return
        data = self.rfile.read(length)
        if len(data) != length:
            self.send_error(400)
            return
        with lock:
            if sum(map(len, objects.values())) - len(objects.get(self.path, b"")) + length > limit:
                self.send_error(507)
                return
            objects[self.path] = data
        self.send_response(200)
        self.end_headers()
        self.wfile.write(hashlib.sha256(data).hexdigest().encode())

    def do_GET(self):
        if not self.authenticated():
            self.send_error(403)
            return
        with lock:
            data = objects.get(self.path)
        if data is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-SHA256", hashlib.sha256(data).hexdigest())
        self.end_headers()
        self.wfile.write(data)

    def do_DELETE(self):
        if not self.authenticated():
            self.send_error(403)
            return
        with lock:
            objects.clear()
        self.send_response(204)
        self.end_headers()


ThreadingHTTPServer(("0.0.0.0", 8443), Handler).serve_forever()
