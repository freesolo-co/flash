"""End-to-end managed CLI flow against a stub control plane (subprocess, CPU-only)."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def stub_server():
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/v1/runs":
                if self.headers.get("Authorization") != "Bearer fs-stub-key":
                    self._send(401, {"detail": "invalid or missing API key"})
                    return
                self._send(
                    200,
                    {"run_id": "flash-1-stub", "state": "queued", "spec": body.get("spec")},
                )
            else:
                self._send(404, {"detail": f"unknown path {self.path}"})

        def do_GET(self):
            if self.path == "/api/auth/verify":
                # Login is handled by the freesolo backend: it verifies the bearer key.
                ok = self.headers.get("Authorization") == "Bearer fs-stub-key"
                self._send(200 if ok else 401, {"ok": ok})
                return
            if self.path == "/v1/me":
                # After storing the key, login fetches the identity card from the control plane.
                self._send(
                    200,
                    {
                        "kind": "freesolo_api_key",
                        "key_prefix": "fs-stub",
                        "email": "stub@example.com",
                    },
                )
                return
            if self.path == "/v1/runs":
                self._send(
                    200,
                    {
                        "runs": [
                            {
                                "run_id": "flash-1-stub",
                                "state": "done",
                                "cost_usd": 0.25,
                                "updated_at": 1.0,
                                "spec": {"model": "Qwen/Qwen3.5-4B", "algorithm": "grpo"},
                            }
                        ]
                    },
                )
            else:
                self._send(404, {"detail": f"unknown path {self.path}"})

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _run(args, home: str, api_url: str):
    env = os.environ.copy()
    env.pop("FREESOLO_API_KEY", None)
    env.pop("FLASH_API_KEY", None)
    # The stub serves both the freesolo verify endpoint and the flash control plane.
    env.update({"HOME": home, "FLASH_API_URL": api_url, "FREESOLO_BASE_URL": api_url})
    return subprocess.run(
        [sys.executable, "-m", "flash.cli.main", *args],
        cwd=ROOT,
        text=True,
        env=env,
        capture_output=True,
        timeout=30,
    )


def test_login_verifies_freesolo_key_and_train_submits(stub_server, tmp_path):
    home = str(tmp_path)

    # `flash login` verifies the freesolo key against the freesolo backend, then stores it.
    proc = _run(["login", "--api-key", "fs-stub-key"], home=home, api_url=stub_server)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # The key and local config path must never be echoed.
    assert "fs-stub-key" not in proc.stdout
    assert "saved to" not in proc.stdout
    assert ".flash" not in proc.stdout
    assert "you're ready to train" not in proc.stdout
    # Login shows who you are straight away, so a separate `whoami` isn't needed.
    assert "logged in to flash" in proc.stdout
    assert "stub@example.com" in proc.stdout

    cfg = os.path.join(home, ".flash", "config.json")
    with open(cfg) as f:
        cfg_data = json.load(f)
    assert cfg_data["api_key"] == "fs-stub-key"
    assert stat.S_IMODE(os.stat(cfg).st_mode) == 0o600

    # `flash train --background` posts the locally-validated spec and prints the handle.
    toml = tmp_path / "run.toml"
    toml.write_text(
        'model = "Qwen/Qwen3.5-4B"\nalgorithm = "grpo"\n'
        '[environment]\nid = "freesolo-co/gsm8k"\n[train]\nsteps = 1\nseeds = [0]\nhf_repo = "owner/runs"\n'
    )
    proc = _run(["train", str(toml), "--background"], home=home, api_url=stub_server)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = json.loads(proc.stdout)
    assert out["run_id"] == "flash-1-stub"
    assert out["spec"]["model"] == "Qwen/Qwen3.5-4B"

    # `flash runs` renders the server's run list.
    proc = _run(["runs"], home=home, api_url=stub_server)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "flash-1-stub" in proc.stdout
    assert "GRPO" in proc.stdout
    assert "0.2500" in proc.stdout
