"""Standard-library client for explicit shared-training sessions."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Client:
    def __init__(self, url, token, *, timeout=30):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" and not (
            parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1")
        ):
            raise ValueError("shared endpoint requires HTTPS except on loopback")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("invalid shared endpoint")
        self.url, self._token, self.timeout = url.rstrip("/"), token, timeout
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, method, path, payload=None):
        body = json.dumps(payload, allow_nan=False).encode() if payload is not None else None
        request = urllib.request.Request(
            self.url + path,
            data=body,
            method=method,
            headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"shared training HTTP {error.code}: {error.read(4096).decode(errors='replace')}"
            ) from None

    def create(self, algorithm, *, seed=0, learning_rate=1e-4, key=None):
        result = self.request(
            "POST",
            "/sessions",
            {
                "key": key or uuid.uuid4().hex,
                "config": {"algorithm": algorithm, "seed": seed, "learning_rate": learning_rate},
            },
        )
        return Session(self, result["session"])


class Session:
    def __init__(self, client, session_id):
        self.client, self.id = client, session_id
        self.path = "/sessions/" + urllib.parse.quote(session_id, safe="")
        self.version = 0

    def status(self):
        return self.client.request("GET", self.path)

    def submit(self, operation, payload, *, key=None):
        return self.client.request(
            "POST",
            self.path + "/commands",
            {"key": key or uuid.uuid4().hex, "operation": operation, "payload": payload},
        )["command"]

    def wait(self, command, *, timeout=600):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.status()
            for row in status["commands"]:
                if row["id"] != command:
                    continue
                if row["state"] == "completed":
                    result = row["result"]
                    if "version" in result:
                        self.version = result["version"]
                    return result
                if row["state"] in ("failed", "cancelled", "unknown"):
                    raise RuntimeError(f"shared operation {row['state']}: {row['result']}")
            if status["state"] == "lost":
                raise RuntimeError("shared session lost; restore a completed checkpoint")
            time.sleep(0.1)
        raise TimeoutError("shared operation still pending; retain its id before retrying")

    def call(self, operation, payload, **kwargs):
        return self.wait(self.submit(operation, payload, **kwargs))

    def sample(self, prompt, *, max_tokens, seed):
        return self.call(
            "sample",
            {"version": self.version, "prompt": prompt, "max_tokens": max_tokens, "seed": seed},
        )

    def checkpoint(self):
        return self.call("checkpoint", {})

    def close(self):
        return self.call("close", {})

    def restore(self, source_session, version):
        return self.call("restore", {"source_session": source_session, "version": version})
