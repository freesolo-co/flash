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
sys.path.insert(0, ROOT)


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
            if self.path == "/v1/keys":
                self._send(
                    200,
                    {
                        "api_key": "sk-autoslm-stub0000",
                        "key_prefix": "sk-autoslm-stub",
                        "email": body.get("email"),
                        "created_at": 0,
                    },
                )
            elif self.path == "/v1/runs":
                if self.headers.get("Authorization") != "Bearer sk-autoslm-stub0000":
                    self._send(401, {"detail": "invalid or missing API key"})
                    return
                self._send(
                    200,
                    {"run_id": "autoslm-1-stub", "state": "queued", "spec": body.get("spec")},
                )
            else:
                self._send(404, {"detail": f"unknown path {self.path}"})

        def do_GET(self):
            if self.path == "/v1/runs":
                self._send(
                    200,
                    {
                        "runs": [
                            {
                                "run_id": "autoslm-1-stub",
                                "state": "done",
                                "cost_usd": 0.25,
                                "updated_at": 1.0,
                                "spec": {"model": "Qwen/Qwen3-4B-Instruct-2507"},
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
    env.pop("AUTOSLM_API_KEY", None)
    env.update({"HOME": home, "AUTOSLM_API_URL": api_url})
    return subprocess.run(
        [sys.executable, "-m", "autoslm.cli.main", *args],
        cwd=ROOT,
        text=True,
        env=env,
        capture_output=True,
        timeout=30,
    )


def test_login_claims_key_and_train_submits(stub_server, tmp_path):
    home = str(tmp_path)

    # `slm login` (no args) claims a key from the control plane.
    proc = _run(["login"], home=home, api_url=stub_server)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # The key must never be echoed; only the storage location is printed.
    assert "sk-autoslm-stub" not in proc.stdout
    assert "key saved to" in proc.stdout

    cfg = os.path.join(home, ".autoslm", "config.json")
    with open(cfg) as f:
        cfg_data = json.load(f)
    assert cfg_data["api_key"] == "sk-autoslm-stub0000"
    assert stat.S_IMODE(os.stat(cfg).st_mode) == 0o600

    # `slm train --background` posts the locally-validated spec and prints the handle.
    toml = tmp_path / "run.toml"
    toml.write_text(
        'model = "Qwen/Qwen3-4B-Instruct-2507"\nalgorithm = "grpo"\n'
        '[environment]\nid = "gsm8k"\n[train]\nsteps = 1\nseeds = [0]\n'
    )
    proc = _run(["train", str(toml), "--background"], home=home, api_url=stub_server)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = json.loads(proc.stdout)
    assert out["run_id"] == "autoslm-1-stub"
    assert out["spec"]["model"] == "Qwen/Qwen3-4B-Instruct-2507"

    # `slm ps` renders the server's run list.
    proc = _run(["ps"], home=home, api_url=stub_server)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "autoslm-1-stub" in proc.stdout
    assert "0.2500" in proc.stdout
