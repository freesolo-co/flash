"""credential isolation for managed environment git operations."""

from __future__ import annotations

import stat
import subprocess
import tempfile
import urllib.parse
from pathlib import Path

import pytest

from flash.server.domain.registry import envs

_TOKEN = "synthetic/token:secret"
_ENCODED_TOKEN = urllib.parse.quote(_TOKEN, safe="")
_MIXED_ENCODED_TOKEN = _ENCODED_TOKEN.replace("%2F", "%2f").replace("%3A", "%3a")
_OTHER_MIXED_ENCODED_TOKEN = _ENCODED_TOKEN.replace("%3A", "%3a")
_PUBLISH_ROOT = "ns/project/env"


def _invoke_clone_flow(flow: str, tmp_path: Path) -> object:
    if flow == "publish":
        package = tmp_path / "package"
        package.mkdir()
        (package / "environment.py").write_text("def load_environment(**k): pass\n")
        return envs._github_publish_once(
            dest=package,
            repo="owner/repo",
            token=_TOKEN,
            publish_root=_PUBLISH_ROOT,
            message="upload test environment",
        )
    if flow == "download":
        return envs._github_download_once(
            repo="owner/repo",
            token=_TOKEN,
            publish_root=_PUBLISH_ROOT,
        )
    if flow == "delete":
        return envs._github_delete_once(
            repo="owner/repo",
            token=_TOKEN,
            publish_root=_PUBLISH_ROOT,
            message="delete test environment",
        )
    raise AssertionError(f"unknown flow: {flow}")


@pytest.mark.parametrize("flow", ["publish", "download", "delete"])
def test_clone_flows_keep_token_out_of_process_and_repository_state(flow, tmp_path, monkeypatch):
    calls: list[tuple[list[str], dict[str, str] | None]] = []
    auth_paths: list[tuple[Path, Path]] = []
    repository_configs: list[str] = []
    encoded = _ENCODED_TOKEN

    def fake_run(args, *, cwd, env=None, **_kwargs):
        command = [str(arg) for arg in args]
        assert env is not None
        copied_env = dict(env)
        calls.append((command, copied_env))

        assert _TOKEN not in "\0".join(command)
        assert encoded not in "\0".join(command)
        serialized_env = "\0".join(f"{key}={value}" for key, value in copied_env.items())
        assert _TOKEN not in serialized_env
        assert encoded not in serialized_env
        assert _MIXED_ENCODED_TOKEN not in serialized_env
        assert _OTHER_MIXED_ENCODED_TOKEN not in serialized_env
        assert copied_env["GIT_TERMINAL_PROMPT"] == "0"
        assert copied_env["GIT_CONFIG_KEY_0"] == "credential.helper"
        assert copied_env["GIT_CONFIG_VALUE_0"] == ""
        assert copied_env["SAFE_INHERITED_VALUE"] == "preserved"
        assert "GITHUB_TOKEN" not in copied_env
        assert secret_name not in copied_env
        assert encoded_secret_name not in copied_env
        assert "INHERITED_RAW_VALUE" not in copied_env
        assert "INHERITED_ENCODED_VALUE" not in copied_env

        askpass = Path(copied_env["GIT_ASKPASS"])
        token_file = Path(copied_env["FLASH_GIT_TOKEN_FILE"])
        auth_paths.append((askpass, token_file))
        assert stat.S_IMODE(askpass.stat().st_mode) == 0o700
        assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
        assert _TOKEN not in askpass.read_text()
        assert token_file.read_text() == _TOKEN

        if len(command) > 1 and command[1] == "clone":
            assert "https://github.com/owner/repo.git" in command
            checkout = Path(command[-1])
            checkout.mkdir(parents=True)
            config = (
                '[remote "origin"]\n'
                f"\turl = {command[-2]}\n"
                "\tfetch = +refs/heads/*:refs/remotes/origin/*\n"
            )
            git_dir = checkout / ".git"
            git_dir.mkdir()
            (git_dir / "config").write_text(config)
            repository_configs.append(config)
            target = checkout / _PUBLISH_ROOT
            target.mkdir(parents=True)
            (target / "environment.py").write_text("def load_environment(**k): pass\n")

        return subprocess.CompletedProcess(command, 0, "", "")

    secret_name = f"INHERITED_{_TOKEN}_NAME"
    encoded_secret_name = f"INHERITED_{_MIXED_ENCODED_TOKEN}_NAME"
    monkeypatch.setenv("GITHUB_TOKEN", _TOKEN)
    monkeypatch.setenv("GH_TOKEN", encoded)
    monkeypatch.setenv(secret_name, "ordinary-value")
    monkeypatch.setenv(encoded_secret_name, "ordinary-value")
    monkeypatch.setenv("INHERITED_RAW_VALUE", f"prefix-{_TOKEN}-suffix")
    monkeypatch.setenv("INHERITED_ENCODED_VALUE", f"prefix-{_OTHER_MIXED_ENCODED_TOKEN}-suffix")
    monkeypatch.setenv("SAFE_INHERITED_VALUE", "preserved")
    monkeypatch.setattr(envs.subprocess, "run", fake_run)

    result = _invoke_clone_flow(flow, tmp_path)

    if flow == "download":
        assert isinstance(result, bytes)
    elif flow == "delete":
        assert result is False
    else:
        assert result is None

    clone_calls = [call for call in calls if len(call[0]) > 1 and call[0][1] == "clone"]
    assert len(clone_calls) == 1
    assert repository_configs
    assert all(_TOKEN not in config and encoded not in config for config in repository_configs)
    assert auth_paths
    assert all(
        not askpass.exists() and not token_file.exists() for askpass, token_file in auth_paths
    )


def test_askpass_returns_credentials_without_embedding_them_in_process_state():
    auth_paths: tuple[Path, Path] | None = None

    with envs._git_credential_env(_TOKEN) as git_env:
        askpass = Path(git_env["GIT_ASKPASS"])
        token_file = Path(git_env["FLASH_GIT_TOKEN_FILE"])
        auth_paths = (askpass, token_file)
        username = subprocess.run(
            [str(askpass), "Username for https://github.com"],
            env=git_env,
            check=True,
            capture_output=True,
            text=True,
        )
        password = subprocess.run(
            [str(askpass), "Password for https://github.com"],
            env=git_env,
            check=True,
            capture_output=True,
            text=True,
        )
        serialized_env = "\0".join(f"{key}={value}" for key, value in git_env.items())
        assert _TOKEN not in serialized_env
        assert username.stdout == "x-access-token\n"
        assert password.stdout == _TOKEN

    assert auth_paths is not None
    assert all(not path.exists() for path in auth_paths)


def test_inherited_command_scope_git_helpers_cannot_read_the_token(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="flash-malicious-helper-", dir="/dev/shm") as tmp:
        attack_dir = Path(tmp)
        helper = attack_dir / "credential-helper.sh"
        invoked = attack_dir / "invoked"
        exfiltrated = attack_dir / "exfiltrated-token"
        helper.write_text(
            "#!/bin/sh\n"
            'printf invoked > "$MALICIOUS_HELPER_MARKER"\n'
            'cat "$FLASH_GIT_TOKEN_FILE" > "$MALICIOUS_HELPER_EXFIL"\n'
        )
        helper.chmod(0o700)
        helper_command = f"!/bin/sh {helper}"

        monkeypatch.setenv(
            "GIT_CONFIG_PARAMETERS",
            f"'credential.helper'='{helper_command}'",
        )
        monkeypatch.setenv("GIT_CONFIG_COUNT", "2")
        monkeypatch.setenv("GIT_CONFIG_KEY_0", "credential.helper")
        monkeypatch.setenv("GIT_CONFIG_VALUE_0", helper_command)
        monkeypatch.setenv("GIT_CONFIG_KEY_1", "credential.helper")
        monkeypatch.setenv("GIT_CONFIG_VALUE_1", helper_command)
        monkeypatch.setenv("GIT_CONFIG_KEY_27", "credential.helper")
        monkeypatch.setenv("GIT_CONFIG_VALUE_27", helper_command)
        monkeypatch.setenv("MALICIOUS_HELPER_MARKER", str(invoked))
        monkeypatch.setenv("MALICIOUS_HELPER_EXFIL", str(exfiltrated))
        monkeypatch.setattr(tempfile, "tempdir", "/dev/shm")

        with envs._git_credential_env(_TOKEN) as git_env:
            command_scope_keys = {
                key
                for key in git_env
                if key == "GIT_CONFIG_COUNT"
                or key == "GIT_CONFIG_PARAMETERS"
                or key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))
            }
            assert command_scope_keys == {
                "GIT_CONFIG_COUNT",
                "GIT_CONFIG_KEY_0",
                "GIT_CONFIG_VALUE_0",
            }
            assert git_env["GIT_CONFIG_COUNT"] == "1"
            assert git_env["GIT_CONFIG_KEY_0"] == "credential.helper"
            assert git_env["GIT_CONFIG_VALUE_0"] == ""
            subprocess.run(
                ["git", "credential", "fill"],
                env=git_env,
                input="protocol=https\nhost=example.invalid\n\n",
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )

        assert not invoked.exists()
        assert not exfiltrated.exists()


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (RuntimeError("git transport failed"), RuntimeError),
        (subprocess.TimeoutExpired(cmd=["git", "clone"], timeout=1), envs.EnvPublishError),
        (KeyboardInterrupt(), KeyboardInterrupt),
    ],
    ids=["failure", "timeout", "signal_like"],
)
def test_git_credential_material_is_cleaned_after_exception(
    raised, expected, tmp_path, monkeypatch, caplog
):
    auth_paths: list[tuple[Path, Path]] = []

    def fail_run(_args, *, env, **_kwargs):
        askpass = Path(env["GIT_ASKPASS"])
        token_file = Path(env["FLASH_GIT_TOKEN_FILE"])
        auth_paths.append((askpass, token_file))
        assert stat.S_IMODE(askpass.stat().st_mode) == 0o700
        assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
        raise raised

    monkeypatch.setattr(envs.subprocess, "run", fail_run)

    with pytest.raises(expected):
        envs._run_git(
            tmp_path,
            ["clone", "https://github.com/owner/repo.git", "checkout"],
            token=_TOKEN,
            operation="download",
        )

    assert auth_paths
    assert all(
        not askpass.exists() and not token_file.exists() for askpass, token_file in auth_paths
    )
    assert _TOKEN not in caplog.text
    assert urllib.parse.quote(_TOKEN, safe="") not in caplog.text


def test_redact_matches_only_exact_raw_or_hex_case_equivalent_encoded_tokens():
    safe_case_neighbor = _MIXED_ENCODED_TOKEN.replace("synthetic", "Synthetic")
    safe_octet_neighbor = _MIXED_ENCODED_TOKEN.replace("%2f", "%3f")
    payload = (
        f"raw={_TOKEN}|canonical={_ENCODED_TOKEN}|"
        f"before={_MIXED_ENCODED_TOKEN}|"
        f"crossing={'x' * 960}{_OTHER_MIXED_ENCODED_TOKEN}|"
        f"beyond={'y' * 1100}{_MIXED_ENCODED_TOKEN}|"
        f"safe_case={safe_case_neighbor}|safe_octet={safe_octet_neighbor}"
    )

    redacted = envs._redact(payload, _TOKEN)

    assert _TOKEN not in redacted
    assert _ENCODED_TOKEN not in redacted
    assert _MIXED_ENCODED_TOKEN not in redacted
    assert _OTHER_MIXED_ENCODED_TOKEN not in redacted
    assert redacted.count("<redacted>") == 5
    assert redacted.endswith(f"safe_case={safe_case_neighbor}|safe_octet={safe_octet_neighbor}")


@pytest.mark.parametrize(
    ("flow", "operation"),
    [("publish", "upload"), ("download", "download"), ("delete", "delete")],
)
def test_git_failures_redact_mixed_percent_encoding_around_truncation_boundary(
    flow, operation, tmp_path, monkeypatch, caplog
):
    def fail_run(args, **_kwargs):
        return subprocess.CompletedProcess(
            args,
            128,
            f"before: {_MIXED_ENCODED_TOKEN}",
            f"{'x' * 910} crossing: {_OTHER_MIXED_ENCODED_TOKEN} beyond: "
            f"{'y' * 1100}{_MIXED_ENCODED_TOKEN}",
        )

    monkeypatch.setattr(envs.subprocess, "run", fail_run)

    with pytest.raises(envs.EnvPublishError) as excinfo:
        _invoke_clone_flow(flow, tmp_path)

    message = str(excinfo.value)
    assert f"environment {operation} failed" in message
    assert _TOKEN not in message
    assert _ENCODED_TOKEN not in message
    assert _MIXED_ENCODED_TOKEN not in message
    assert _OTHER_MIXED_ENCODED_TOKEN not in message
    assert message.count("<redacted>") == 2
    assert len(message) < 1200
    assert _TOKEN not in caplog.text
    assert _ENCODED_TOKEN not in caplog.text
    assert _MIXED_ENCODED_TOKEN not in caplog.text
    assert _OTHER_MIXED_ENCODED_TOKEN not in caplog.text
