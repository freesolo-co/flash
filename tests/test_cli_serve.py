"""`flash serve` -- generating and deploying a self-hosted serving backend.

The setup path spends money: it starts a GPU container on the user's Modal account. So the tests
that matter most here are the ones asserting it does NOT do that without consent, and that every
failure is caught before anything is written or deployed.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import types
import urllib.error
from pathlib import Path

import pytest

from flash.cli.commands import serve as serve_cmd
from flash.core.catalog import MODELS
from flash.serve.probe import ProbeResult

_REQUIRED_CAPABILITIES = ["immutable_adapter_revisions", "alias_compare_and_swap"]


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


def _health_payload(**overrides):
    payload = {
        "ok": True,
        "base_models": ["Qwen/Qwen3.5-4B"],
        "capabilities": list(_REQUIRED_CAPABILITIES),
    }
    payload.update(overrides)
    return payload


def _args(**kwargs):
    return types.SimpleNamespace(**kwargs)


def _setup_args(tmp_path, **overrides):
    base = {
        "model": "Qwen/Qwen3.5-4B",
        "gpu": None,
        "output": str(tmp_path / "flash_serving_app.py"),
        "scaledown_window": None,
        "dry_run": True,
        "force": False,
        "yes": False,
    }
    base.update(overrides)
    return _args(**base)


# --- serve gpus -------------------------------------------------------------------------------


def test_gpus_lists_cards_and_marks_the_validated_one(capsys):
    assert serve_cmd.cmd_serve_gpus(_args(model="Qwen/Qwen3.5-4B", context_len=0)) == 0
    out = capsys.readouterr().out
    assert "L4" in out
    assert "H200" in out
    assert "validated on real hardware" in out


def test_gpus_labels_the_numbers_as_estimates(capsys):
    """These are computed, not measured, and a reader deciding what to rent must be told.

    flash has no serving throughput dataset, so an unqualified speed or fit column would present
    arithmetic as though it were a benchmark.
    """
    serve_cmd.cmd_serve_gpus(_args(model="Qwen/Qwen3.5-4B", context_len=0))
    out = capsys.readouterr().out
    assert "ESTIMATES" in out
    assert "not a measured tokens/sec" in out


def test_gpus_shows_the_largest_model_fitting_only_the_largest_cards(capsys):
    from flash.serve.backend.gpus import MODAL_GPUS_BY_NAME

    serve_cmd.cmd_serve_gpus(_args(model="Qwen/Qwen3.6-35B-A3B", context_len=0))
    # read only the table rows: the tip below the table is prose and would otherwise be parsed
    # as data. a row starts with a known card name.
    fits = set()
    for line in capsys.readouterr().out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] in MODAL_GPUS_BY_NAME and parts[2] != "no":
            fits.add(parts[0])
    assert fits == {"H200", "B200"}


def test_unknown_model_is_rejected_with_the_supported_list(capsys):
    from flash.cli import main

    assert main(["serve", "gpus", "--model", "Llama/Nope"]) == 1
    assert "Qwen/Qwen3.5-4B" in capsys.readouterr().err


# --- serve setup ------------------------------------------------------------------------------


def test_setup_dry_run_writes_the_app_without_deploying(tmp_path, capsys, monkeypatch):
    def _never(*args, **kwargs):
        raise AssertionError("dry run must not shell out")

    monkeypatch.setattr(subprocess, "run", _never)
    assert serve_cmd.cmd_serve_setup(_setup_args(tmp_path)) == 0
    written = tmp_path / "flash_serving_app.py"
    assert written.exists()
    assert "BASE_MODEL = 'Qwen/Qwen3.5-4B'" in written.read_text()


def test_setup_does_not_deploy_without_consent(tmp_path, monkeypatch, capsys):
    """The confirmation is the gate on spending money, so a refusal must stop the deploy.

    Declining still leaves the generated app on disk and prints the command to deploy it later --
    the user rejected the automatic deploy, not the file.
    """
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
    """Non-interactive use must not deploy by default.

    A CI job or a piped invocation gets EOF from input(); reading that as consent would deploy a
    GPU container nobody asked for. --yes is the explicit opt in.
    """

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
    """Nothing may be written before the Modal check that ends with "re-run this command".

    A file written on the way out is precisely what makes the re-run die with FileExistsError, so
    the user follows correct instructions and gets an error telling them to pass --force.
    """
    args = _setup_args(tmp_path, dry_run=False, yes=True)
    monkeypatch.setattr(serve_cmd, "_modal_cli", lambda: None)
    assert serve_cmd.cmd_serve_setup(args) == 1
    assert not Path(args.output).exists(), "the app was written before modal was even checked"

    # Now satisfy what the instructions asked for and re-run exactly as told, no --force.
    monkeypatch.setattr(serve_cmd, "_modal_cli", lambda: "/usr/bin/modal")
    monkeypatch.setattr(serve_cmd, "_modal_is_authenticated", lambda: True)
    monkeypatch.setattr(serve_cmd, "_deploy", lambda app_file: 0)
    assert serve_cmd.cmd_serve_setup(args) == 0
    assert Path(args.output).exists()


def test_a_dry_run_does_not_require_modal_to_be_installed(tmp_path, monkeypatch):
    """`--dry-run` deploys nothing, so it must still work with no modal CLI and no account."""
    monkeypatch.setattr(serve_cmd, "_modal_cli", lambda: None)
    args = _setup_args(tmp_path, dry_run=True)
    assert serve_cmd.cmd_serve_setup(args) == 0
    assert Path(args.output).exists()


def test_setup_refuses_to_overwrite_an_existing_app(tmp_path, capsys):
    """The generated app is meant to be edited; silently regenerating over it destroys that work."""
    destination = tmp_path / "flash_serving_app.py"
    destination.write_text("# my edits\n")
    assert serve_cmd.cmd_serve_setup(_setup_args(tmp_path)) == 1
    assert destination.read_text() == "# my edits\n"
    assert "--force" in capsys.readouterr().err


def test_setup_creates_the_parent_directory_of_a_custom_output(tmp_path):
    """`--output deploy/generated/app.py` names a destination, not a path that must already exist.

    Without this the write fails on a bare errno 2 that names the FILE, so the user is told the
    app does not exist when the real problem is the directory above it. Creating the parent is
    what the flag asks for; the overwrite guard still runs first, so a refused run leaves no
    directories behind.
    """
    destination = tmp_path / "deploy" / "generated" / "serving.py"
    assert serve_cmd.cmd_serve_setup(_setup_args(tmp_path, output=str(destination))) == 0
    assert destination.exists()
    assert "BASE_MODEL = 'Qwen/Qwen3.5-4B'" in destination.read_text()


def test_force_overwrites(tmp_path):
    destination = tmp_path / "flash_serving_app.py"
    destination.write_text("# my edits\n")
    assert serve_cmd.cmd_serve_setup(_setup_args(tmp_path, force=True)) == 0
    assert "BASE_MODEL" in destination.read_text()


def test_an_unknown_gpu_is_rejected_before_anything_is_written(tmp_path, capsys):
    """T4 is the case that matters: excluded because vLLM silently downgrades bf16 on pre-Ampere
    silicon, so it would serve at degraded quality with no error to see."""
    assert serve_cmd.cmd_serve_setup(_setup_args(tmp_path, gpu="T4")) == 1
    assert not (tmp_path / "flash_serving_app.py").exists()
    assert "unknown Modal GPU" in capsys.readouterr().err


def test_a_chosen_gpu_is_honored(tmp_path):
    assert serve_cmd.cmd_serve_setup(_setup_args(tmp_path, gpu="H100")) == 0
    assert "GPU = 'H100'" in (tmp_path / "flash_serving_app.py").read_text()


# --- deploy output parsing --------------------------------------------------------------------


def test_deployed_url_picks_the_web_endpoint_not_the_dashboard():
    """modal prints a dashboard link too, and pointing flash at that would 404 every request."""
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
    """The URL is the whole point of the command: it is what connects flash to the new backend."""

    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="https://acme--flash-serve-api.modal.run\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert serve_cmd._deploy(tmp_path / "app.py") == 0
    out = capsys.readouterr().out
    assert "export FREESOLO_SERVING_URL=https://acme--flash-serve-api.modal.run" in out


def test_an_explicit_scaledown_window_is_not_collapsed_by_the_default(tmp_path, capsys):
    """`or DEFAULT` rewrites any falsy value, so an explicit 0 silently became 300.

    0 is outside Modal's supported range, so the right answer is a clear error before anything is
    written -- not a file that quietly disagrees with what was asked for.
    """
    args = _setup_args(tmp_path, scaledown_window=0)
    assert serve_cmd.cmd_serve_setup(args) == 1
    assert "outside Modal's supported range" in capsys.readouterr().err


def test_a_gpu_that_cannot_hold_the_model_is_refused(tmp_path, capsys):
    """An explicit `--gpu` must face the same fit check `serve gpus` shows.

    Skipped on the one path that spends money: setup writes the catalog's fixed 32K config,
    deploys, pulls the weights, and the engine OOMs on a cold start the user paid for. The refusal
    carries the numbers so the choice is actionable rather than a bare verdict.
    """
    args = _setup_args(tmp_path, model="Qwen/Qwen3.6-35B-A3B", gpu="L4")
    assert serve_cmd.cmd_serve_setup(args) == 1
    err = capsys.readouterr().err
    assert "cannot serve" in err
    assert "L4" in err
    assert not Path(args.output).exists(), "an app was written for a card that cannot run it"


def test_the_validated_gpu_for_every_model_still_passes_the_fit_check(tmp_path):
    """The guard must not reject the catalog's own production-validated card.

    A fit estimate stricter than reality would make `serve setup` refuse exactly the configuration
    Freesolo runs in production, which is worse than the bug being fixed.
    """
    for model_id in MODELS:
        args = _setup_args(tmp_path, model=model_id, output=str(tmp_path / f"{hash(model_id)}.py"))
        assert serve_cmd.cmd_serve_setup(args) == 0, (
            f"the validated card was refused for {model_id}"
        )


def test_a_valid_non_default_scaledown_window_reaches_the_generated_app(tmp_path):
    args = _setup_args(tmp_path, scaledown_window=60)
    assert serve_cmd.cmd_serve_setup(args) == 0
    assert "SCALEDOWN_WINDOW_SECONDS = 60" in Path(args.output).read_text()


@pytest.mark.parametrize(
    ("healthz", "warns"),
    [
        (_health_payload(requires_key=False), True),
        (_health_payload(requires_key=True), False),
        # An older app predating the field, and an unreachable one. Neither is evidence the app is
        # unauthenticated, so neither may cry wolf.
        ({"ok": True}, False),
        (None, False),
    ],
    ids=["no-key", "keyed", "field-absent", "unreachable"],
)
def test_a_keyless_deploy_warns_that_anyone_can_spend_the_gpu_budget(
    tmp_path, monkeypatch, capsys, healthz, warns
):
    """A Modal URL is public. Deploying without a key is a real exposure, so say so out loud."""

    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="https://acme--flash-serve-api.modal.run\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(serve_cmd, "healthz_with_retry", lambda url: healthz)
    assert serve_cmd._deploy(tmp_path / "app.py") == 0
    err = capsys.readouterr().err
    assert ("no FLASH_SERVING_KEY" in err) is warns


def test_the_keyless_warning_redeploys_the_file_that_was_actually_deployed(
    tmp_path, monkeypatch, capsys
):
    """The remediation must name the deployed file, not the default one.

    Under `--output` a hardcoded `flash_serving_app.py` either fails or deploys an unrelated app,
    so a user who follows the warning exactly is left with the same public keyless endpoint they
    were warned about -- the one instruction that has to work is the one closing the exposure.
    """

    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="https://acme--flash-serve-api.modal.run\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(serve_cmd, "healthz_with_retry", lambda url: _health_payload(requires_key=False))
    app_file = tmp_path / "serving" / "my_app.py"
    assert serve_cmd._deploy(app_file) == 0
    err = capsys.readouterr().err
    assert f"modal deploy {app_file}" in err
    assert serve_cmd.DEFAULT_APP_FILE not in err


def test_status_sends_the_internal_key_the_way_deploy_does(monkeypatch, capsys):
    """`serve status` must carry the internal key.

    The contract permits a backend to authenticate /healthz. A bare unauthenticated GET then 401s
    and status reports the backend as unreachable while every deploy against it works fine.
    """
    from flash.serve import urls as urls_mod

    seen: list[tuple[str, dict]] = []

    def _fake_request(url, headers, path="/healthz"):
        seen.append((url, headers))
        if path != "/healthz":
            # The key probe. This health payload omits `requires_key`, which is the backend making
            # no claim -- so status probes rather than assuming open, and 404 is the pass answer.
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
    """`serve status` must work on a bare `pip install freesolo-flash`.

    `[project].dependencies` is empty by design and the client CLI is pure standard library, but
    `flash.serve.deploy` imports httpx at module scope. Importing it from this command made status
    die with `ModuleNotFoundError: httpx` BEFORE any of its error handling ran -- and this is the
    command a user reaches for when their backend is not answering, so it is the worst one to make
    conditional on an extra.

    Run in a SUBPROCESS with httpx made unimportable, rather than by asserting on the import lines
    in this file. Import reachability is transitive -- a helper three modules down that pulls httpx
    would break the base install just as thoroughly -- and only actually running the command with
    httpx absent covers that. The command is pointed at an unroutable host so it fails on the
    request, which is the success condition here: reaching its own error path means every import
    it needed resolved.
    """
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
    """Diagnosing a broken backend is what `serve status` is FOR.

    Valid JSON that is not the expected shape decodes fine and escapes the decode guard, so an
    unchecked `.get` reaches the user as an AttributeError traceback -- the command failing at
    exactly the moment its job starts. Every shape below has to come back as a stated error.
    """
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
    """A base URL is user-supplied and may carry credentials in its authority.

    Printed verbatim, they land in the terminal and in whatever captures its output. Redacting the
    URL we print is necessary but not sufficient: urllib quotes the URL it was handed back into its
    OWN errors, and it reports FRAGMENTS -- parsing `https://user:pw@host` for a port raises
    "nonnumeric port: 'pw@host'", a string containing neither the full base nor the full userinfo.
    So the exception text has to be redacted too, token by token.
    """
    from flash.serve import urls as urls_mod

    def _boom(request_url, headers):
        # Exactly what urllib does: the value it was given, quoted back at the caller.
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
    """A `/healthz` redirect must not carry the plane's root credential off-origin.

    urllib re-sends custom headers to a redirect target -- it strips only the ones IT set -- so a
    backend that 302s elsewhere collects `X-Freesolo-Internal-Key`, which on a standalone plane is
    the credential that controls the plane. `flash.serve.deploy` installs an httpx hook for exactly
    this; the dependency-free status path needs its own.

    Driven against two real loopback servers rather than a mocked opener, because the leak IS
    urllib's redirect behavior: a stub would only test the stub. The same-origin case is not
    decoration -- Modal 303s slow requests to a same-origin poll url, so a handler that refused
    every redirect would break the ordinary path while passing the security half.
    """
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
        # redirect to a different PATH on the same origin, which is modal's async-result poll.
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


def test_the_printed_key_is_kept_where_the_user_can_send_it_back(monkeypatch, capsys):
    """Both key-setup paths must generate into a variable, not inline into `modal secret create`.

    The key is symmetric: flash sends the same value back on every request. Generated inline, the
    only copy goes into Modal and is unrecoverable, so the app is authenticated against a secret
    nobody holds and every deploy 401s -- with a setup transcript that looked like it worked.
    """
    monkeypatch.setattr(serve_cmd, "healthz_with_retry", lambda url: _health_payload(requires_key=False))
    serve_cmd._warn_if_unauthenticated("https://acme--flash-serve-api.modal.run")
    for text in (serve_cmd._setup_instructions(), capsys.readouterr().err):
        assert "export FREESOLO_INTERNAL_KEY=$(" in text
        assert 'FLASH_SERVING_KEY="$FREESOLO_INTERNAL_KEY"' in text
        assert "FLASH_SERVING_KEY=$(python" not in text, "the generated key is discarded inline"


def test_the_key_warning_survives_a_cold_start_on_the_first_probe(monkeypatch):
    import flash.serve.probe as probe_mod

    attempts = {"n": 0}

    class _Payload:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _urlopen(target, timeout=None):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TimeoutError("cold start")
        return _Payload()

    monkeypatch.setattr(probe_mod.urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(probe_mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        probe_mod.json,
        "load",
        lambda _: _health_payload(requires_key=False),
    )

    assert probe_mod.healthz_with_retry(
        "https://acme--flash-serve-api.modal.run",
        retry_delay_s=0.0,
    ) == _health_payload(requires_key=False)
    assert attempts["n"] == 3, f"the probe did not retry: {attempts['n']} attempt(s)"


def test_the_key_probe_gives_up_within_its_budget(monkeypatch):
    import flash.serve.probe as probe_mod

    attempts = {"n": 0}

    def _always_fails(target, timeout=None):
        attempts["n"] += 1
        raise TimeoutError("never comes up")

    monkeypatch.setattr(probe_mod.urllib.request, "urlopen", _always_fails)
    monkeypatch.setattr(probe_mod.time, "sleep", lambda _: None)

    assert probe_mod.healthz_with_retry("https://acme.modal.run", budget_s=0.0) is None
    assert attempts["n"] == 1, (
        f"a zero budget still made {attempts['n']} attempts; the probe is not bounded by it"
    )


def test_deploy_instructions_quote_a_path_with_spaces(tmp_path, monkeypatch, capsys):
    """These lines are meant to be COPIED into a shell, so an unquoted path is a broken command.

    `--output` accepts any writable path. One containing a space splits into two arguments when
    pasted, so the instruction for deploying the file that was just written fails to deploy it --
    and the keyless-warning variant leaves a public endpoint up after the user did exactly what
    they were told.
    """
    spaced = tmp_path / "my apps" / "flash_serving_app.py"
    spaced.parent.mkdir()
    spaced.write_text("# generated\n")

    monkeypatch.setattr(serve_cmd, "healthz_with_retry", lambda url: _health_payload(requires_key=False))
    serve_cmd._warn_if_unauthenticated("https://acme.modal.run", spaced)
    warning = capsys.readouterr().err
    assert f"modal deploy {shlex.quote(str(spaced))}" in warning, (
        f"the redeploy instruction did not quote a path containing a space: {warning}"
    )
    assert f"modal deploy {spaced}\n" not in warning, "the unquoted form is still being printed"


def test_the_deploy_output_points_at_the_control_plane_process(tmp_path, monkeypatch, capsys):
    """The serving variables belong to `flash-server`, not to the shell that ran `serve setup`.

    `models deploy`/`chat`/`undeploy` are control-plane routes: the SERVER reads
    FREESOLO_SERVING_URL and contacts the backend. An instruction to export it here reaches the
    CLI and not the server, so an operator who follows it exactly still has every deploy fail on
    an unset serving URL.
    """

    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="https://acme--flash-serve-api.modal.run\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(serve_cmd, "healthz_with_retry", lambda url: _health_payload(requires_key=True))
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


# --- serve teardown ---------------------------------------------------------------------------


def test_teardown_confirms_before_stopping(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(serve_cmd, "_modal_cli", lambda: "/usr/bin/modal")
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    assert serve_cmd.cmd_serve_teardown(_args(model="Qwen/Qwen3.5-4B", yes=False)) == 1
    assert calls == []


def test_teardown_stops_the_apps_own_name(monkeypatch, capsys):
    """Stopping the wrong app name would silently leave the real one running and billing."""
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(serve_cmd, "_modal_cli", lambda: "/usr/bin/modal")
    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert serve_cmd.cmd_serve_teardown(_args(model="Qwen/Qwen3.5-4B", yes=True)) == 0
    assert calls == [["modal", "app", "stop", "flash-serve-qwen3-5-4b"]]


# --- parser wiring ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["serve", "gpus", "--model", "Qwen/Qwen3.5-4B"], "cmd_serve_gpus"),
        (["serve", "setup", "--model", "Qwen/Qwen3.5-4B"], "cmd_serve_setup"),
        (["serve", "status"], "cmd_serve_status"),
        (["serve", "teardown", "--model", "Qwen/Qwen3.5-4B"], "cmd_serve_teardown"),
    ],
)
def test_subcommands_are_registered(argv, expected):
    from flash.cli import _build_parser

    args = _build_parser().parse_args(argv)
    assert args.func.__name__ == expected


def test_a_negative_context_length_is_rejected_at_parse_time():
    """A negative `--context-len` inverts the sizing math instead of failing.

    The estimator multiplies KV bytes per token by this value, so a negative SUBTRACTS memory:
    `--context-len -900000` reports a 24 GB A10 as fitting `ample` with 124 GB spare. The table
    looks entirely ordinary, which is what makes it worth rejecting at the boundary -- the command
    exists to answer "will this fit", and this is the one input that makes it answer backwards.

    Zero stays valid: it is the documented sentinel for the model's own serving context.
    """
    from flash.cli import _build_parser

    parser = _build_parser()
    assert parser.parse_args(["serve", "gpus", "--model", "Qwen/Qwen3.5-4B"]).context_len == 0
    ok = parser.parse_args(["serve", "gpus", "--model", "Qwen/Qwen3.5-4B", "--context-len", "0"])
    assert ok.context_len == 0
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["serve", "gpus", "--model", "Qwen/Qwen3.5-4B", "--context-len", "-900000"]
        )


def test_setup_defaults_are_safe():
    """Nothing that spends money may be on by default."""
    from flash.cli import _build_parser

    args = _build_parser().parse_args(["serve", "setup", "--model", "Qwen/Qwen3.5-4B"])
    assert args.yes is False
    assert args.force is False
    assert args.dry_run is False


@pytest.mark.parametrize("status_code", [401, 403])
def test_status_verifies_the_serving_key_against_an_authenticated_route(
    monkeypatch, capsys, status_code
):
    """A wrong key must not read as `ready`, whichever rejection code the backend uses.

    /healthz is deliberately unauthenticated, so a missing or mismatched key sails through every
    check the command makes and it prints `ready` -- then the very next `models deploy` 401s on
    /adapters. That is the exact misconfiguration an operator runs `serve status` to diagnose, so
    the key has to be exercised against a route that actually checks it.

    BOTH codes, because the contract is written for any backend, not just the generated app. The
    conformance suite accepts 401 or 403 as a valid rejection, so recognizing only 401 here would
    report `ready` against a backend that suite would certify and every deploy would then fail on.
    """
    import urllib.error

    from flash.serve import urls as urls_mod

    asked: list[str] = []

    def _fake_request(url, headers, path="/healthz"):
        asked.append(path)
        if path == "/healthz":
            return {
                "ok": True,
                "requires_key": True,
                "base_models": ["Qwen/Qwen3.5-4B"],
                "capabilities": ["immutable_adapter_revisions", "alias_compare_and_swap"],
            }
        raise urllib.error.HTTPError(url, status_code, "invalid serving key", {}, None)

    monkeypatch.setattr(urls_mod, "serving_base_url", lambda: "https://acme.modal.run")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "wrong")
    _patch_status_probes(monkeypatch, _fake_request)

    code = serve_cmd.cmd_serve_status(_args())
    out = capsys.readouterr()
    assert code == 1, (
        f"status reported success against a backend that rejects the key with {status_code}, so "
        f"the next deploy fails on a backend this command just called ready"
    )
    assert "ready. deploy a run" not in out.out
    assert str(status_code) in out.err, f"the failure did not name the status it got: {out.err!r}"
    assert "FREESOLO_INTERNAL_KEY" in out.err, (
        f"the failure did not name the variable to fix: {out.err!r}"
    )
    assert any(p != "/healthz" for p in asked), (
        "only /healthz was probed, which is unauthenticated -- nothing exercised the key"
    )


def test_status_does_not_report_ready_when_the_key_probe_errors(monkeypatch, capsys):
    """A 5xx on the probe means the key was never verified, which is not `ready`.

    The backend answered /healthz a moment ago and then failed the authenticated call, so the
    operator cannot deploy either way -- and treating "not a 401" as "the key was accepted" turns
    a broken backend into a green status.
    """
    import urllib.error

    from flash.serve import urls as urls_mod

    def _fake_request(url, headers, path="/healthz"):
        if path == "/healthz":
            return {
                "ok": True,
                "requires_key": True,
                "base_models": ["Qwen/Qwen3.5-4B"],
                "capabilities": ["immutable_adapter_revisions", "alias_compare_and_swap"],
            }
        raise urllib.error.HTTPError(url, 500, "internal server error", {}, None)

    monkeypatch.setattr(urls_mod, "serving_base_url", lambda: "https://acme.modal.run")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "some-key")
    _patch_status_probes(monkeypatch, _fake_request)

    code = serve_cmd.cmd_serve_status(_args())
    out = capsys.readouterr()
    assert code == 1, "a 500 on the key probe was reported as ready"
    assert "ready. deploy a run" not in out.out
    assert "500" in out.err, f"the failure did not name the status it got: {out.err!r}"


@pytest.mark.parametrize("status_code", [400, 405, 422])
def test_status_does_not_report_ready_on_a_non_404_read_back(monkeypatch, capsys, status_code):
    """Only 404 proves the read-back route works. Any other 4xx is not a pass.

    404 for an id that cannot exist means two things at once: the key got past authentication, and
    the route resolved the id to "no such record". A 400/405/422 proves neither -- it is the backend
    saying it does not answer this route the way the contract requires (missing route, rejected path
    shape, a handler wanting query parameters).

    Which matters because `models deploy` polls this exact route: `_registered_adapter_response`
    passes `ok_statuses=(404,)` and `_wait_revision_ready` re-raises any status below 500, so a
    non-404 4xx is fatal there. Falling through to `ready` tells the operator the backend is good
    and then loses the deploy to the same status, after registration has already started.
    """
    import urllib.error

    from flash.serve import urls as urls_mod

    def _fake_request(url, headers, path="/healthz"):
        if path == "/healthz":
            return {
                "ok": True,
                "requires_key": True,
                "base_models": ["Qwen/Qwen3.5-4B"],
                "capabilities": ["immutable_adapter_revisions", "alias_compare_and_swap"],
            }
        raise urllib.error.HTTPError(url, status_code, "unexpected", {}, None)

    monkeypatch.setattr(urls_mod, "serving_base_url", lambda: "https://acme.modal.run")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "right")
    _patch_status_probes(monkeypatch, _fake_request)

    code = serve_cmd.cmd_serve_status(_args())
    out = capsys.readouterr()
    assert code == 1, (
        f"a {status_code} read-back was reported as ready, but deploy polls this same route and "
        f"treats any non-404 4xx as fatal -- so the operator is told to deploy against a backend "
        f"that cannot complete one"
    )
    assert "ready. deploy a run" not in out.out
    assert str(status_code) in out.err, f"the failure did not name the status it got: {out.err!r}"


def test_status_reports_ready_when_the_serving_key_is_accepted(monkeypatch, capsys):
    """The probe must pass a CORRECT key through.

    Without this, rejecting every authenticated backend would satisfy the test above while making
    `serve status` useless on a properly configured deployment.
    """
    import urllib.error

    from flash.serve import urls as urls_mod

    def _fake_request(url, headers, path="/healthz"):
        if path == "/healthz":
            return {
                "ok": True,
                "requires_key": True,
                "base_models": ["Qwen/Qwen3.5-4B"],
                "capabilities": ["immutable_adapter_revisions", "alias_compare_and_swap"],
            }
        # What the generated app answers a good key asking for an id that does not exist: the
        # request got PAST authentication, which is the whole question.
        raise urllib.error.HTTPError(url, 404, "unknown adapter id", {}, None)

    monkeypatch.setattr(urls_mod, "serving_base_url", lambda: "https://acme.modal.run")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "right")
    _patch_status_probes(monkeypatch, _fake_request)

    assert serve_cmd.cmd_serve_status(_args()) == 0
    assert "ready. deploy a run" in capsys.readouterr().out


def test_status_skips_the_key_probe_on_a_backend_without_one(monkeypatch, capsys):
    """A backend that authenticates nothing has no key to verify.

    Probing anyway would send a request no answer could inform, and a backend that returns
    something other than 404 for an unknown id would then read as broken.
    """
    from flash.serve import urls as urls_mod

    asked: list[str] = []

    def _fake_request(url, headers, path="/healthz"):
        asked.append(path)
        return {
            "ok": True,
            "requires_key": False,
            "base_models": ["Qwen/Qwen3.5-4B"],
            "capabilities": ["immutable_adapter_revisions", "alias_compare_and_swap"],
        }

    monkeypatch.setattr(urls_mod, "serving_base_url", lambda: "https://acme.modal.run")
    _patch_status_probes(monkeypatch, _fake_request)
    assert serve_cmd.cmd_serve_status(_args()) == 0
    assert asked == ["/healthz"]
    assert "ready. deploy a run" in capsys.readouterr().out


def test_status_probes_the_key_when_the_backend_makes_no_claim(monkeypatch, capsys):
    """An OMITTED `requires_key` must still be probed, because the field is optional.

    The contract explicitly makes it optional, so absence is a backend declining to say -- not a
    backend declaring itself open. Treated as open, a protected custom backend that omits the field
    skipped the probe entirely, printed `ready` with a missing or wrong key, and 401'd on the very
    next deploy: exactly the misconfiguration this command exists to diagnose.
    """
    from flash.serve import urls as urls_mod

    asked: list[str] = []

    def _fake_request(url, headers, path="/healthz"):
        asked.append(path)
        if path == "/healthz":
            # No `requires_key` at all. Everything else is a conforming health payload.
            return {
                "ok": True,
                "base_models": ["Qwen/Qwen3.5-4B"],
                "capabilities": ["immutable_adapter_revisions", "alias_compare_and_swap"],
            }
        raise urllib.error.HTTPError(url, 401, "invalid serving key", {}, None)

    monkeypatch.setattr(urls_mod, "serving_base_url", lambda: "https://acme.modal.run")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "wrong")
    _patch_status_probes(monkeypatch, _fake_request)

    code = serve_cmd.cmd_serve_status(_args())
    out = capsys.readouterr()
    assert any(p != "/healthz" for p in asked), (
        "a backend that omits `requires_key` was never probed, so a wrong key reads as ready"
    )
    assert code == 1, (
        "status reported ready against a protected backend that omits `requires_key`, so the "
        "next deploy 401s on a backend this command just called ready"
    )
    assert "ready. deploy a run" not in out.out
    assert "FREESOLO_INTERNAL_KEY" in out.err


def test_status_does_not_report_ready_when_the_read_back_fabricates_a_record(monkeypatch, capsys):
    """A 200 for an id that was never registered is a contract violation, not a passing probe.

    Every other branch of the key probe handles a RAISING response, so a backend that answers the
    made-up id with a fabricated record returned normally and fell through to `ready` -- the one
    outcome proving the backend does not implement unknown-record semantics was reported as
    success. `models deploy` polls this exact route and cross-checks the record against the
    identity it registered, so a fabricating backend answers that poll with a mismatch the client
    reads as an immutability violation and refuses.
    """
    from flash.serve import urls as urls_mod

    def _fake_request(url, headers, path="/healthz"):
        if path == "/healthz":
            return {
                "ok": True,
                "requires_key": True,
                "base_models": ["Qwen/Qwen3.5-4B"],
                "capabilities": ["immutable_adapter_revisions", "alias_compare_and_swap"],
            }
        # The fabricating backend: a plausible-looking record for an id it has never seen.
        return {
            "adapter": {
                "adapter_id": path.rsplit("/", 1)[-1],
                "status": "ready",
                "base_model": "Qwen/Qwen3.5-4B",
                "metadata": {"lifecycle_state": "ready"},
            }
        }

    monkeypatch.setattr(urls_mod, "serving_base_url", lambda: "https://acme.modal.run")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "some-key")
    _patch_status_probes(monkeypatch, _fake_request)

    code = serve_cmd.cmd_serve_status(_args())
    out = capsys.readouterr()
    assert code == 1, (
        "status called a record-fabricating backend ready; every deploy against it then fails its "
        "read-back identity check, which is exactly what this command exists to catch first"
    )
    assert "ready. deploy a run" not in out.out
    assert "404" in out.err, (
        f"the failure did not name the status the contract requires: {out.err!r}"
    )
