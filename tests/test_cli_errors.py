"""The CLI reports ordinary user errors as a clean one-line message, not a traceback."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# `_run` launches the child with sys.executable, an absolute path, so the CLI names that
# interpreter back rather than a bare `python` -- which is absent entirely on a python3-only host.
# Derived from the same value the subprocess uses, so the two cannot drift apart.
INVOKED_CLI_NAME = f"{sys.executable} -m flash.cli"


def _run(args, env=None):
    full_env = os.environ.copy()
    full_env.pop("FREESOLO_API_KEY", None)  # never let the host's login leak into tests
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "flash.cli", *args],
        cwd=ROOT,
        text=True,
        env=full_env,
        capture_output=True,
        timeout=30,
    )


def _logged_out_env(tmp):
    home = os.path.join(tmp, "home")
    os.makedirs(home, exist_ok=True)
    return {"HOME": home}  # no ~/.flash/config.json -> no Flash key


def test_logged_out_status_is_friendly():
    with tempfile.TemporaryDirectory() as tmp:
        proc = _run(["runs", "status", "does-not-exist"], env=_logged_out_env(tmp))
    assert proc.returncode == 1
    assert proc.stderr.startswith("error:")
    assert f"{INVOKED_CLI_NAME} login" in proc.stderr
    # No raw traceback on stderr.
    assert "Traceback (most recent call last)" not in proc.stderr


def test_bad_model_is_friendly():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, "run.toml")
        with open(cfg, "w") as f:
            f.write(
                'model = "Not/AReal-Model"\nproject = "11111111-1111-4111-8111-111111111111"\nalgorithm = "grpo"\n'
                '[environment]\nid = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
            )
        proc = _run(["train", cfg, "--dry-run"], env=_logged_out_env(tmp))
    assert proc.returncode == 1
    assert proc.stderr.startswith("error:")
    assert "unsupported model" in proc.stderr
    assert "Traceback (most recent call last)" not in proc.stderr


def test_missing_config_is_friendly():
    with tempfile.TemporaryDirectory() as tmp:
        proc = _run(
            ["train", os.path.join(tmp, "nope.toml"), "--dry-run"], env=_logged_out_env(tmp)
        )
    assert proc.returncode == 1
    assert proc.stderr.startswith("error:")
    assert "config file not found" in proc.stderr
    assert f"{INVOKED_CLI_NAME} env setup" in proc.stderr
    # a bare [Errno 2] string and a traceback are both the wrong UX for a mistyped path.
    assert "Errno" not in proc.stderr
    assert "Traceback (most recent call last)" not in proc.stderr


def test_config_pointed_at_a_directory_is_friendly():
    with tempfile.TemporaryDirectory() as tmp:
        # `flash train configs/` or `flash train .` — a directory, not a .toml file.
        proc = _run(["train", tmp, "--dry-run"], env=_logged_out_env(tmp))
    assert proc.returncode == 1
    assert proc.stderr.startswith("error:")
    assert "is a directory" in proc.stderr
    assert "Traceback (most recent call last)" not in proc.stderr


def test_debug_flag_shows_traceback():
    with tempfile.TemporaryDirectory() as tmp:
        proc = _run(["--debug", "runs", "status", "does-not-exist"], env=_logged_out_env(tmp))
    assert proc.returncode != 0
    assert "Traceback (most recent call last)" in proc.stderr


def test_unexpected_error_suggestion_does_not_replay_a_credential():
    """The `run ... --debug` suggestion echoes argv, so it must not echo the key inside it.

    Both argv spellings matter: argparse accepts `--api-key secret` as two entries and
    `--api-key=secret` as one, and only the first is caught by looking at the following element.
    """
    from flash.cli.parsing.main import _redacted_args

    separate = _redacted_args(
        ["models", "export", "--repository", "alice/model", "--api-key", "hf_SUPERSECRET"]
    )
    assert "hf_SUPERSECRET" not in separate
    assert separate == [
        "models",
        "export",
        "--repository",
        "alice/model",
        "--api-key",
        "<redacted>",
    ]

    joined = _redacted_args(["login", "--api-key=fs_SUPERSECRET"])
    assert "fs_SUPERSECRET" not in joined
    assert joined == ["login", "--api-key=<redacted>"]

    # a non-credential flag keeps its value: the suggestion is only useful if it stays runnable.
    assert _redacted_args(["runs", "status", "--json"]) == ["runs", "status", "--json"]
    assert _redacted_args(["login", "--api-url", "https://x"]) == [
        "login",
        "--api-url",
        "https://x",
    ]


def test_unexpected_error_suggestion_redacts_abbreviated_credential_flags():
    """argparse accepts any unambiguous prefix of a long option, and those bind the value too.

    `flash login --api-k SECRET` and `--api-ke=SECRET` both populate `api_key`, so matching only the
    full `--api-key` spelling leaves every abbreviated invocation printing the credential. The
    parser is asserted here alongside the redaction: the point is that these spellings are real,
    not hypothetical.
    """
    from flash.cli.parsing.main import _build_parser, _redacted_args

    parser = _build_parser()
    for argv in (["login", "--api-k", "fs_SUPERSECRET"], ["login", "--api-ke=fs_SUPERSECRET"]):
        assert parser.parse_args(argv).api_key == "fs_SUPERSECRET", "abbreviation must be real"
        assert not any("fs_SUPERSECRET" in part for part in _redacted_args(argv))

    # which prefixes are ambiguous depends on the SUBPARSER. `flash login` also defines --api-url,
    # so `--api` is refused there; `flash models export` does not, so `--ap`, `--api` and `--api-`
    # are all accepted and bind the HuggingFace token. the required args are supplied because an
    # incomplete command exits on those first, which hides whether the abbreviation was accepted.
    export = ["models", "export", "--adapter-id", "a1", "--repository", "alice/m"]
    for flag in ("--ap", "--api", "--api-", "--api-k", "--api-key"):
        for argv in ([*export, flag, "hf_SUPERSECRET"], [*export, f"{flag}=hf_SUPERSECRET"]):
            assert parser.parse_args(argv).api_key == "hf_SUPERSECRET", f"{flag} must bind"
            assert not any("hf_SUPERSECRET" in part for part in _redacted_args(argv))


def test_train_without_login_fails_fast():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, "run.toml")
        with open(cfg, "w") as f:
            f.write(
                'model = "Qwen/Qwen3.5-9B"\nproject = "11111111-1111-4111-8111-111111111111"\nalgorithm = "grpo"\n'
                '[environment]\nid = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
                "[train]\nepochs = 1\nmax_examples = 1\n"
            )
        proc = _run(["train", cfg], env=_logged_out_env(tmp))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    # It must fail *before* contacting anything, with the fix spelled out.
    assert "not logged in" in proc.stderr
    assert f"{INVOKED_CLI_NAME} login" in proc.stderr


def test_missing_env_id_rejected_client_side():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, "run.toml")
        with open(cfg, "w") as f:
            f.write(
                'model = "Qwen/Qwen3.5-9B"\nproject = "11111111-1111-4111-8111-111111111111"\n'
                'algorithm = "grpo"\n[environment]\n[train]\n'
            )
        # A config without [environment] id is rejected before any network call.
        submit = _run(["train", cfg], env=_logged_out_env(tmp))
        assert submit.returncode == 1
        assert "[environment] id" in submit.stderr
        assert f"{INVOKED_CLI_NAME} env push" in submit.stderr


def test_dry_run_without_login_fails_fast():
    # dry-run uses server submit-time preflights, so it requires login. client validation still runs
    # first; see test_bad_model_is_friendly.
    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, "run.toml")
        with open(cfg, "w") as f:
            f.write(
                'model = "Qwen/Qwen3.5-9B"\nproject = "11111111-1111-4111-8111-111111111111"\nalgorithm = "grpo"\n'
                '[environment]\nid = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
                "[train]\nepochs = 1\nmax_examples = 1\n"
            )
        proc = _run(["train", cfg, "--dry-run"], env=_logged_out_env(tmp))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "not logged in" in proc.stderr
    assert f"{INVOKED_CLI_NAME} login" in proc.stderr
    assert "Traceback (most recent call last)" not in proc.stderr


def test_cost_requires_login_for_server_dry_run_preparation():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, "run.toml")
        with open(cfg, "w") as f:
            f.write(
                'model = "Qwen/Qwen3.5-9B"\nproject = "11111111-1111-4111-8111-111111111111"\nalgorithm = "grpo"\n'
                '[environment]\nid = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
                "[train]\nepochs = 1\nmax_examples = 1\n"
            )
        proc = _run(["train", cfg, "--cost"], env=_logged_out_env(tmp))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "not logged in" in proc.stderr
    assert f"{INVOKED_CLI_NAME} login" in proc.stderr
    assert "Traceback (most recent call last)" not in proc.stderr


@contextlib.contextmanager
def _stub_plane(content_type: str, body: bytes):
    """Stand in for the control plane, answering every GET 200 with one fixed body.

    `--api-url` is a supported path, so "a reverse proxy or an older plane answered" is a real
    user state. Neither body below is something the CLI can use.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_wrong_shape_plane_response_is_friendly():
    with (
        _stub_plane("application/json", b'{"hello": "world"}') as url,
        tempfile.TemporaryDirectory() as tmp,
    ):
        proc = _run(
            ["runs", "list"],
            env={
                **_logged_out_env(tmp),
                "FLASH_API_URL": url,
                "FREESOLO_API_KEY": "fslo-user-test",
            },
        )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert proc.stderr.startswith("error:")
    assert f"{url}/v1/runs returned an unexpected response shape" in proc.stderr
    assert "'runs'" in proc.stderr
    assert "rather than at a proxy or another service" in proc.stderr
    assert "Traceback (most recent call last)" not in proc.stderr


def test_non_json_login_response_is_friendly():
    with (
        _stub_plane("text/html", b"<html>hi</html>") as url,
        tempfile.TemporaryDirectory() as tmp,
    ):
        env = _logged_out_env(tmp)
        proc = _run(["login", "--api-key", "fslo-user-bogus", "--api-url", url], env=env)
        saved = os.path.join(env["HOME"], ".flash", "config.json")
        assert not os.path.exists(saved), "an unverified key must not be saved"
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "login failed" in proc.stderr
    assert f"{url}/v1/me did not return JSON (Content-Type: text/html)" in proc.stderr
    assert "rather than at a proxy or another service" in proc.stderr
    assert "Expecting value" not in proc.stderr
    assert "Traceback (most recent call last)" not in proc.stderr
