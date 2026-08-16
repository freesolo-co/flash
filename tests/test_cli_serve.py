"""`flash serve` -- generating and deploying a self-hosted serving backend.
The setup path spends money: it starts a GPU container on the user's Modal account. So the tests
that matter most here are the ones asserting it does NOT do that without consent, and that every
failure is caught before anything is written or deployed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
import urllib.error
from pathlib import Path

import pytest

from flash.cli.commands import serve as serve_cmd
from flash.serve.probe import ProbeResult


def _probe_result(payload=None, status_code=200, error=""):
    return ProbeResult(status_code=status_code, payload=payload, error=error)


def _patch_status_probes(monkeypatch, request):
    def call(url, headers, path):
        try:
            payload = request(url, headers, path)
        except urllib.error.HTTPError as exc:
            return _probe_result(status_code=exc.code, error=str(exc))
        return _probe_result(payload)

    monkeypatch.setattr(
        serve_cmd, "request_json", lambda url, headers: call(url, headers, "/healthz")
    )
    monkeypatch.setattr(
        serve_cmd, "probe_serving_key", lambda url, headers: call(url, headers, "/adapters/probe")
    )


def _args(**kwargs):
    return types.SimpleNamespace(**kwargs)


def _setup_args(tmp_path, **overrides):
    base = {
        "model": "Qwen/Qwen3.5-4B",
        "output": str(tmp_path / "flash_serving_app.py"),
        "scaledown_window": None,
        "dry_run": True,
        "force": False,
        "yes": False,
    }
    base.update(overrides)
    return _args(**base)


def test_unknown_model_is_rejected_with_the_supported_list(capsys):
    from flash.cli import main

    assert main(["serve", "setup", "--model", "Llama/Nope", "--dry-run"]) == 1
    assert "Qwen/Qwen3.5-4B" in capsys.readouterr().err


def test_setup_dry_run_writes_the_app_without_deploying(tmp_path, capsys, monkeypatch):
    def _never(*args, **kwargs):
        raise AssertionError("dry run must not shell out")

    monkeypatch.setattr(subprocess, "run", _never)
    assert serve_cmd.cmd_serve_setup(_setup_args(tmp_path)) == 0
    written = tmp_path / "flash_serving_app.py"
    assert written.exists()
    assert "BASE_MODEL = 'Qwen/Qwen3.5-4B'" in written.read_text()


def test_setup_does_not_deploy_without_consent(tmp_path, monkeypatch, capsys):
    deployed = []
    monkeypatch.setattr(serve_cmd, "_modal_cli", lambda: "/usr/bin/modal")
    monkeypatch.setattr(serve_cmd, "_modal_is_authenticated", lambda: True)
    monkeypatch.setattr(serve_cmd, "_deploy", lambda path: deployed.append(path) or 0)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    assert serve_cmd.cmd_serve_setup(_setup_args(tmp_path, dry_run=False)) == 1
    assert deployed == []
    assert (tmp_path / "flash_serving_app.py").exists()
    assert "modal deploy" in capsys.readouterr().err


def test_a_closed_stdin_is_treated_as_a_refusal(tmp_path, monkeypatch):
    def _eof(prompt=""):
        raise EOFError

    deployed = []
    monkeypatch.setattr(serve_cmd, "_modal_cli", lambda: "/usr/bin/modal")
    monkeypatch.setattr(serve_cmd, "_modal_is_authenticated", lambda: True)
    monkeypatch.setattr(serve_cmd, "_deploy", lambda path: deployed.append(path) or 0)
    monkeypatch.setattr("builtins.input", _eof)
    assert serve_cmd.cmd_serve_setup(_setup_args(tmp_path, dry_run=False)) == 1
    assert deployed == []


def test_yes_deploys_without_prompting(tmp_path, monkeypatch):
    def _never(prompt=""):
        raise AssertionError("--yes must not prompt")

    deployed = []
    monkeypatch.setattr(serve_cmd, "_modal_cli", lambda: "/usr/bin/modal")
    monkeypatch.setattr(serve_cmd, "_modal_is_authenticated", lambda: True)
    monkeypatch.setattr(serve_cmd, "_deploy", lambda path: deployed.append(path) or 0)
    monkeypatch.setattr("builtins.input", _never)
    assert serve_cmd.cmd_serve_setup(_setup_args(tmp_path, dry_run=False, yes=True)) == 0
    assert len(deployed) == 1


def test_missing_modal_setup_is_reported_before_deploying(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(serve_cmd, "_modal_cli", lambda: None)
    assert serve_cmd.cmd_serve_setup(_setup_args(tmp_path, dry_run=False, yes=True)) == 1
    err = capsys.readouterr().err
    assert "modal setup" in err
    assert "modal secret create" in err


def test_the_rerun_those_instructions_ask_for_actually_works(tmp_path, monkeypatch, capsys):
    args = _setup_args(tmp_path, dry_run=False, yes=True)
    monkeypatch.setattr(serve_cmd, "_modal_cli", lambda: None)
    assert serve_cmd.cmd_serve_setup(args) == 1
    assert not Path(args.output).exists(), "the app was written before modal was even checked"
    monkeypatch.setattr(serve_cmd, "_modal_cli", lambda: "/usr/bin/modal")
    monkeypatch.setattr(serve_cmd, "_modal_is_authenticated", lambda: True)
    monkeypatch.setattr(serve_cmd, "_deploy", lambda app_file: 0)
    assert serve_cmd.cmd_serve_setup(args) == 0
    assert Path(args.output).exists()


def test_a_dry_run_does_not_require_modal_to_be_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(serve_cmd, "_modal_cli", lambda: None)
    args = _setup_args(tmp_path, dry_run=True)
    assert serve_cmd.cmd_serve_setup(args) == 0
    assert Path(args.output).exists()


def test_setup_refuses_to_overwrite_an_existing_app(tmp_path, capsys):
    destination = tmp_path / "flash_serving_app.py"
    destination.write_text("# my edits\n")
    assert serve_cmd.cmd_serve_setup(_setup_args(tmp_path)) == 1
    assert destination.read_text() == "# my edits\n"
    assert "--force" in capsys.readouterr().err


def test_setup_creates_the_parent_directory_of_a_custom_output(tmp_path):
    destination = tmp_path / "deploy" / "generated" / "serving.py"
    assert serve_cmd.cmd_serve_setup(_setup_args(tmp_path, output=str(destination))) == 0
    assert destination.exists()
    assert "BASE_MODEL = 'Qwen/Qwen3.5-4B'" in destination.read_text()


def test_force_overwrites(tmp_path):
    destination = tmp_path / "flash_serving_app.py"
    destination.write_text("# my edits\n")
    assert serve_cmd.cmd_serve_setup(_setup_args(tmp_path, force=True)) == 0
    assert "BASE_MODEL" in destination.read_text()


def test_deployed_url_picks_the_web_endpoint_not_the_dashboard():
    output = (
        "Created objects.\n"
        "View Deployment: https://modal.com/apps/acme/main/deployed/flash-serve-qwen3-5-4b\n"
        "https://acme--flash-serve-qwen3-5-4b-api.modal.run\n"
    )
    assert serve_cmd._deployed_url(output) == "https://acme--flash-serve-qwen3-5-4b-api.modal.run"


def test_deployed_url_strips_trailing_punctuation():
    output = "endpoint at https://acme--app-api.modal.run.\n"
    assert serve_cmd._deployed_url(output) == "https://acme--app-api.modal.run"


def test_deployed_url_is_empty_when_absent():
    assert serve_cmd._deployed_url("nothing here") == ""


def test_deploy_reports_the_export_line_users_need(tmp_path, monkeypatch, capsys):
    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="https://acme--flash-serve-api.modal.run\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert serve_cmd._deploy(tmp_path / "app.py") == 0
    out = capsys.readouterr().out
    assert "export FREESOLO_SERVING_URL=https://acme--flash-serve-api.modal.run" in out


def test_an_explicit_scaledown_window_is_not_collapsed_by_the_default(tmp_path, capsys):
    args = _setup_args(tmp_path, scaledown_window=0)
    assert serve_cmd.cmd_serve_setup(args) == 1
    assert "outside Modal's supported range" in capsys.readouterr().err


def test_a_valid_non_default_scaledown_window_reaches_the_generated_app(tmp_path):
    args = _setup_args(tmp_path, scaledown_window=60)
    assert serve_cmd.cmd_serve_setup(args) == 0
    assert "SCALEDOWN_WINDOW_SECONDS = 60" in Path(args.output).read_text()


def test_status_sends_the_internal_key_the_way_deploy_does(monkeypatch, capsys):
    from flash.serve import urls as urls_mod

    seen: list[tuple[str, dict]] = []

    def _fake_request(url, headers, path="/healthz"):
        seen.append((url, headers))
        if path != "/healthz":
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)
        return {
            "ok": True,
            "base_models": ["Qwen/Qwen3.5-4B"],
            "capabilities": ["immutable_adapter_revisions", "alias_compare_and_swap"],
        }

    monkeypatch.setattr(urls_mod, "serving_base_url", lambda: "https://acme.modal.run")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "sekrit")
    _patch_status_probes(monkeypatch, _fake_request)
    assert serve_cmd.cmd_serve_status(_args()) == 0
    assert seen == [("https://acme.modal.run", {"X-Freesolo-Internal-Key": "sekrit"})] * 2, (
        "status did not send the internal key, so an authenticated backend reads as unreachable"
    )


def test_status_runs_without_the_optional_http_dependency(monkeypatch):
    blocker = (
        "import sys\n"
        "class _NoHttpx:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name == 'httpx' or name.startswith('httpx.'):\n"
        "            raise ImportError('httpx is not installed in a base install')\n"
        "        return None\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        return self.find_module(name, path)\n"
        "sys.meta_path.insert(0, _NoHttpx())\n"
        "from flash.cli.commands import serve as s\n"
        "import types\n"
        "code = s.cmd_serve_status(types.SimpleNamespace())\n"
        "print('EXIT', code)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", blocker],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=Path(serve_cmd.__file__).resolve().parents[3],
        env={
            **os.environ,
            "FREESOLO_SERVING_URL": "https://status-probe.invalid",
            "FREESOLO_INTERNAL_KEY": "x",
        },
        check=False,
    )
    missing_dependency = (
        f"serve status could not run without httpx, which a base `pip install freesolo-flash` "
        f"does not provide:\n{done.stderr[-2000:]}"
    )
    assert "ModuleNotFoundError" not in done.stderr, missing_dependency
    assert "ImportError" not in done.stderr, missing_dependency
    assert "EXIT 1" in done.stdout, (
        f"expected the command to reach its own error path; got {done.stdout!r} "
        f"{done.stderr[-500:]!r}"
    )


@pytest.mark.parametrize(
    "payload",
    ["a string", ["a", "list"], 42, {"capabilities": "not-a-list"}, {"capabilities": [1, 2]}],
    ids=["string", "list", "number", "capabilities-not-list", "capabilities-not-strings"],
)
def test_a_malformed_health_payload_is_diagnosed_not_a_traceback(monkeypatch, capsys, payload):
    from flash.serve import urls as urls_mod

    monkeypatch.setattr(urls_mod, "serving_base_url", lambda: "https://acme.modal.run")
    monkeypatch.setattr(serve_cmd, "request_json", lambda url, headers: _probe_result(payload))
    assert serve_cmd.cmd_serve_status(_args()) == 1
    assert "serving backend at" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("url", "secret"),
    [
        ("https://user:sup3rsecret@acme.modal.run", "sup3rsecret"),
        ("https://t0ken@acme.modal.run", "t0ken"),
        ("https://user:p@ss:word@acme.modal.run", "p@ss:word"),
    ],
    ids=["user-and-password", "token-only", "password-with-separators"],
)
def test_a_credential_in_the_serving_url_is_never_printed(monkeypatch, capsys, url, secret):
    from flash.serve import urls as urls_mod

    def _boom(request_url, headers):
        raise ValueError(f"nonnumeric port: {request_url.split('://', 1)[-1]!r}")

    monkeypatch.setattr(urls_mod, "serving_base_url", lambda: url)
    monkeypatch.setattr(serve_cmd, "request_json", _boom)
    assert serve_cmd.cmd_serve_status(_args()) == 1
    err = capsys.readouterr().err
    assert secret not in err, f"the serving url's credential was printed: {err!r}"
    assert "acme.modal.run" in err, "the redaction dropped the host, so the error names nothing"


@pytest.mark.parametrize(
    "same_origin",
    [True, False],
    ids=["same-origin-redirect-is-followed", "cross-origin-redirect-drops-the-key"],
)
def test_status_never_sends_the_serving_key_to_another_origin(same_origin):
    import http.server
    import threading

    seen: list[dict] = []

    class _Target(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # matches BaseHTTPRequestHandler's own naming
            seen.append({k.lower(): v for k, v in self.headers.items()})
            body = b'{"ok": true, "capabilities": []}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    class _Redirector(_Target):
        def do_GET(self):
            if self.path == "/healthz":
                self.send_response(302)
                self.send_header("Location", target_url + "/elsewhere")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            super().do_GET()

    target = http.server.HTTPServer(("127.0.0.1", 0), _Target)
    handler = _Redirector if not same_origin else _Target
    front = http.server.HTTPServer(("127.0.0.1", 0), handler)
    target_url = f"http://127.0.0.1:{target.server_port}"
    front_url = f"http://127.0.0.1:{front.server_port}"
    if same_origin:

        class _SelfRedirector(_Target):
            def do_GET(self):
                if self.path == "/healthz":
                    self.send_response(302)
                    self.send_header("Location", front_url + "/poll")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                super().do_GET()

        front.RequestHandlerClass = _SelfRedirector
    for server in (target, front):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        headers = {"X-Freesolo-Internal-Key": "sekrit"}
        from flash.serve.probe import request_json

        result = request_json(front_url, headers)
        if same_origin:
            assert result.status_code == 200
            assert result.payload["ok"] is True, "a same-origin redirect must still be followed"
            assert seen, "the redirect target was never reached"
            assert seen[-1].get("x-freesolo-internal-key") == "sekrit", (
                "the key was dropped on a same-origin redirect, so modal's async-result poll "
                "would arrive unauthenticated"
            )
        else:
            assert result.status_code == 302
            assert "another origin" in result.error
            assert not any(h.get("x-freesolo-internal-key") for h in seen), (
                "the serving key was sent to the redirect target, disclosing the standalone "
                "plane's root credential to whoever controls it"
            )
    finally:
        for server in (target, front):
            server.shutdown()
            server.server_close()


def test_the_authenticated_probe_waits_as_long_as_the_control_plane_does(monkeypatch):
    import inspect
    import re

    import flash.serve.deploy as deploy_mod
    import flash.serve.probe as probe_mod

    source = inspect.getsource(deploy_mod._serving_request)
    ceiling = float(re.search(r"timeout = ([\d.]+) if timeout_s is None", source).group(1))
    assert ceiling == probe_mod.PROBE_TIMEOUT_SECONDS
    seen: dict[str, object] = {}

    class _Opener:
        def open(self, request, timeout=None):
            seen["timeout"] = timeout
            raise urllib.error.HTTPError(request.full_url, 404, "nope", {}, None)

    monkeypatch.setattr(probe_mod.urllib.request, "build_opener", lambda *a: _Opener())
    probe_mod.request_json("https://acme.modal.run", {})
    assert seen["timeout"] == ceiling, (
        f"the probe used {seen['timeout']}s against a {ceiling}s control-plane ceiling; a cold "
        "backend the control plane reaches would be reported unreachable"
    )


def test_the_printed_key_is_kept_where_the_user_can_send_it_back():
    text = serve_cmd._setup_instructions()
    assert "export FREESOLO_INTERNAL_KEY=$(" in text
    assert 'FLASH_SERVING_KEY="$FREESOLO_INTERNAL_KEY"' in text
    assert "FLASH_SERVING_KEY=$(python" not in text, "the generated key is discarded inline"


def test_the_deploy_output_points_at_the_control_plane_process(tmp_path, monkeypatch, capsys):
    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="https://acme--flash-serve-api.modal.run\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert serve_cmd._deploy(tmp_path / "app.py") == 0
    out = capsys.readouterr().out
    assert serve_cmd.SERVER_NAME in out, "the output never names the control-plane process"
    assert "FREESOLO_SERVING_URL=https://acme--flash-serve-api.modal.run" in out
    assert "FREESOLO_INTERNAL_KEY" in out, "the matching key was not mentioned"
    assert "restart" in out.lower(), "an already-running server needs a restart to see them"


def test_a_failed_deploy_is_reported_as_a_failure(tmp_path, monkeypatch, capsys):
    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert serve_cmd._deploy(tmp_path / "app.py") == 1
    assert "modal deploy failed" in capsys.readouterr().err


def test_teardown_confirms_before_stopping(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(serve_cmd, "_modal_cli", lambda: "/usr/bin/modal")
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    assert serve_cmd.cmd_serve_teardown(_args(model="Qwen/Qwen3.5-4B", yes=False)) == 1
    assert calls == []


def test_teardown_stops_the_apps_own_name(monkeypatch, capsys):
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(serve_cmd, "_modal_cli", lambda: "/usr/bin/modal")
    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert serve_cmd.cmd_serve_teardown(_args(model="Qwen/Qwen3.5-4B", yes=True)) == 0
    assert calls == [["modal", "app", "stop", "flash-serve-qwen3-5-4b"]]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["serve", "setup", "--model", "Qwen/Qwen3.5-4B"], "cmd_serve_setup"),
        (["serve", "status"], "cmd_serve_status"),
        (["serve", "teardown", "--model", "Qwen/Qwen3.5-4B"], "cmd_serve_teardown"),
    ],
)
def test_subcommands_are_registered(argv, expected):
    from flash.cli import _build_parser

    args = _build_parser().parse_args(argv)
    assert args.func.__name__ == expected


def test_setup_defaults_are_safe():
    from flash.cli import _build_parser

    args = _build_parser().parse_args(["serve", "setup", "--model", "Qwen/Qwen3.5-4B"])
    assert args.yes is False
    assert args.force is False
    assert args.dry_run is False


_MISSING = object()
_HEALTH = {
    "ok": True,
    "base_models": ["Qwen/Qwen3.5-4B"],
    "capabilities": ["immutable_adapter_revisions", "alias_compare_and_swap"],
}


@pytest.mark.parametrize(
    ("requires_key", "probe_status", "expected_code", "expected_text", "probe_expected"),
    [
        (True, 401, 1, "FREESOLO_INTERNAL_KEY", True),
        (True, 403, 1, "FREESOLO_INTERNAL_KEY", True),
        (True, 500, 1, "500", True),
        (True, 400, 1, "400", True),
        (True, 405, 1, "405", True),
        (True, 422, 1, "422", True),
        (True, 404, 0, "ready. deploy a run", True),
        (False, None, 0, "ready. deploy a run", False),
        (_MISSING, 401, 1, "FREESOLO_INTERNAL_KEY", True),
        (True, None, 1, "404", True),
    ],
)
def test_status_probe_matrix(
    monkeypatch, capsys, requires_key, probe_status, expected_code, expected_text, probe_expected
):
    from flash.serve import urls as urls_mod

    health = dict(_HEALTH)
    if requires_key is not _MISSING:
        health["requires_key"] = requires_key
    asked = []

    def request(url, headers, path="/healthz"):
        asked.append(path)
        if path == "/healthz":
            return health
        if probe_status is not None:
            raise urllib.error.HTTPError(url, probe_status, "probe result", {}, None)
        return {
            "adapter": {
                "adapter_id": path.rsplit("/", 1)[-1],
                "status": "ready",
                "base_model": "Qwen/Qwen3.5-4B",
                "metadata": {"lifecycle_state": "ready"},
            }
        }

    monkeypatch.setattr(urls_mod, "serving_base_url", lambda: "https://acme.modal.run")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "test-key")
    _patch_status_probes(monkeypatch, request)
    assert serve_cmd.cmd_serve_status(_args()) == expected_code
    output = capsys.readouterr()
    combined = output.out + output.err
    assert expected_text in combined
    assert any(path != "/healthz" for path in asked) is probe_expected
    assert ("ready. deploy a run" in output.out) is (expected_code == 0)
