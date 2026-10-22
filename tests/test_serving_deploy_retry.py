"""A retry must not mistake a failed release's healthy router for promotion proof."""

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/deploy-modal.yml"


def _gate():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    return next(step for step in workflow["jobs"]["deploy"]["steps"] if step.get("id") == "gate")


@pytest.mark.parametrize(
    ("event", "attempt", "changed_file", "expected"),
    [
        ("push", "1", None, False),
        ("push", "2", None, True),
        ("push", "3", "README.md", True),
        ("push", "1", "README.md", False),
        ("push", "1", "flash/example.py", True),
        ("workflow_dispatch", "1", None, True),
    ],
)
def test_retry_requalifies_even_when_failed_release_already_reports_target_sha(
    tmp_path, event, attempt, changed_file, expected
):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Workflow test")
    git("config", "user.email", "workflow@example.invalid")
    git("-c", "commit.gpgsign=false", "commit", "--allow-empty", "-qm", "baseline")
    deployed = git("rev-parse", "HEAD")
    if changed_file:
        path = tmp_path / changed_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("changed\n")
        git("add", changed_file)
        git("-c", "commit.gpgsign=false", "commit", "-qm", "change")
    current = git("rev-parse", "HEAD")

    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"ok": True, "deployment_sha": deployed}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Health)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "outputs"
    try:
        result = subprocess.run(
            ["bash", "-e", "-c", _gate()["run"]],
            cwd=tmp_path,
            env={
                **os.environ,
                "EVENT": event,
                "RUN_ATTEMPT": attempt,
                "CURRENT_SHA": current,
                "DEPLOYED_HEALTH_URL": f"http://127.0.0.1:{server.server_port}/healthz",
                "GITHUB_OUTPUT": str(output),
            },
            capture_output=True,
            text=True,
            timeout=15,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert result.returncode == 0, result.stdout + result.stderr
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert outputs["previous_sha"] == deployed
    assert outputs["deploy"] == str(expected).lower()


def test_retry_number_is_wired_from_github_run_attempt():
    assert _gate()["env"]["RUN_ATTEMPT"] == "${{ github.run_attempt }}"
