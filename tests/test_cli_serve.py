"""`flash serve` -- generating and deploying a self-hosted serving backend.

The setup path spends money: it starts a GPU container on the user's Modal account. So the tests
that matter most here are the ones asserting it does NOT do that without consent, and that every
failure is caught before anything is written or deployed.
"""

from __future__ import annotations

import subprocess
import types
from pathlib import Path

import pytest

from flash.cli.commands import serve as serve_cmd


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
    assert serve_cmd.cmd_serve_gpus(_args(model="Llama/Nope", context_len=0)) == 1
    assert "Qwen/Qwen3.5-4B" in capsys.readouterr().err


# --- serve setup ------------------------------------------------------------------------------


def test_setup_dry_run_writes_the_app_without_deploying(tmp_path, capsys, monkeypatch):
    def _never(*args, **kwargs):
        raise AssertionError("dry run must not shell out")

    monkeypatch.setattr(subprocess, "run", _never)
    assert serve_cmd.cmd_serve_setup(_setup_args(tmp_path)) == 0
    written = tmp_path / "flash_serving_app.py"
    assert written.exists()
    assert 'BASE_MODEL = "Qwen/Qwen3.5-4B"' in written.read_text()


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
    assert 'GPU = "H100"' in (tmp_path / "flash_serving_app.py").read_text()


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


def test_a_valid_non_default_scaledown_window_reaches_the_generated_app(tmp_path):
    args = _setup_args(tmp_path, scaledown_window=60)
    assert serve_cmd.cmd_serve_setup(args) == 0
    assert "SCALEDOWN_WINDOW_SECONDS = 60" in Path(args.output).read_text()


@pytest.mark.parametrize(
    ("healthz", "warns"),
    [
        ({"ok": True, "requires_key": False}, True),
        ({"ok": True, "requires_key": True}, False),
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
    monkeypatch.setattr(serve_cmd, "_healthz", lambda url: healthz)
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
    monkeypatch.setattr(serve_cmd, "_healthz", lambda url: {"ok": True, "requires_key": False})
    app_file = tmp_path / "serving" / "my_app.py"
    assert serve_cmd._deploy(app_file) == 0
    err = capsys.readouterr().err
    assert f"modal deploy {app_file}" in err
    assert serve_cmd.DEFAULT_APP_FILE not in err


def test_status_asks_the_backend_the_way_deploy_does(monkeypatch, capsys):
    """`serve status` must use the client's authenticated request path.

    The contract permits a backend to authenticate /healthz. A bare unauthenticated GET then 401s
    and status reports the backend as unreachable while every deploy against it works fine.
    """
    from flash.serve import deploy as deploy_mod

    seen: list[tuple[str, str]] = []

    class _Response:
        @staticmethod
        def json():
            return {
                "ok": True,
                "base_models": ["Qwen/Qwen3.5-4B"],
                "capabilities": ["immutable_adapter_revisions", "alias_compare_and_swap"],
            }

    def _fake_request(method, url, **kwargs):
        seen.append((method, url))
        return _Response()

    monkeypatch.setattr(deploy_mod, "serving_base_url", lambda: "https://acme.modal.run")
    monkeypatch.setattr(deploy_mod, "_serving_request", _fake_request)
    assert serve_cmd.cmd_serve_status(_args()) == 0
    assert seen == [("GET", "https://acme.modal.run/healthz")], (
        "status did not go through the authenticated serving request path"
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
    from flash.serve import deploy as deploy_mod

    monkeypatch.setattr(deploy_mod, "serving_base_url", lambda: "https://acme.modal.run")
    monkeypatch.setattr(
        deploy_mod,
        "_serving_request",
        lambda method, url, **kwargs: types.SimpleNamespace(json=lambda: payload),
    )
    assert serve_cmd.cmd_serve_status(_args()) == 1
    assert "serving backend at" in capsys.readouterr().err


def test_the_printed_key_is_kept_where_the_user_can_send_it_back(monkeypatch, capsys):
    """Both key-setup paths must generate into a variable, not inline into `modal secret create`.

    The key is symmetric: flash sends the same value back on every request. Generated inline, the
    only copy goes into Modal and is unrecoverable, so the app is authenticated against a secret
    nobody holds and every deploy 401s -- with a setup transcript that looked like it worked.
    """
    monkeypatch.setattr(serve_cmd, "_healthz", lambda url: {"ok": True, "requires_key": False})
    serve_cmd._warn_if_unauthenticated("https://acme--flash-serve-api.modal.run")
    for text in (serve_cmd._setup_instructions(), capsys.readouterr().err):
        assert "export FREESOLO_INTERNAL_KEY=$(" in text
        assert 'FLASH_SERVING_KEY="$FREESOLO_INTERNAL_KEY"' in text
        assert "FLASH_SERVING_KEY=$(python" not in text, "the generated key is discarded inline"


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
    monkeypatch.setattr(serve_cmd, "_healthz", lambda url: {"ok": True, "requires_key": True})
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


def test_setup_defaults_are_safe():
    """Nothing that spends money may be on by default."""
    from flash.cli import _build_parser

    args = _build_parser().parse_args(["serve", "setup", "--model", "Qwen/Qwen3.5-4B"])
    assert args.yes is False
    assert args.force is False
    assert args.dry_run is False
