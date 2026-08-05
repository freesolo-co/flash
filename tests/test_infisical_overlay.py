"""The opt-in Infisical overlay under deploy/infisical/.

Flash reads secrets from its process environment, so the published image needs no secret-manager
tooling. The overlay exists for deployments that pull secrets from Infisical at container start.
These tests pin the two contracts that are easy to break by reading the files and impossible to
notice until a deployment fails to start.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
OVERLAY_DIR = REPO_ROOT / "deploy" / "infisical"
ENTRYPOINT = OVERLAY_DIR / "entrypoint.sh"


def _cmd_line(dockerfile: Path) -> str:
    lines = [ln for ln in dockerfile.read_text().splitlines() if ln.startswith("CMD ")]
    assert len(lines) == 1, f"{dockerfile} should declare exactly one CMD, found {len(lines)}"
    return lines[0]


def test_base_image_bakes_in_no_secret_manager():
    """The published image stays vendor-free: injection is the overlay's job, not the base's."""
    base = (REPO_ROOT / "Dockerfile").read_text()
    assert "infisical" not in base.lower().replace("deploy/infisical", "")
    assert "ENTRYPOINT" not in base, "the base image must leave ENTRYPOINT free for an overlay"


def test_overlay_restates_the_base_cmd_verbatim():
    """Setting ENTRYPOINT in a DERIVED image resets the inherited CMD to null.

    Docker does not carry a base image's CMD into a child that declares its own ENTRYPOINT. An
    overlay that relies on inheriting it hands the wrapper no command, and the container exits
    without ever starting the server. The two CMD lines must therefore stay byte-identical.
    """
    assert _cmd_line(OVERLAY_DIR / "Dockerfile") == _cmd_line(REPO_ROOT / "Dockerfile")


def test_overlay_builds_on_top_of_the_published_image():
    """The overlay must not rebuild flash: it layers onto an existing image via FLASH_IMAGE."""
    text = (OVERLAY_DIR / "Dockerfile").read_text()
    assert re.search(r"^ARG FLASH_IMAGE=", text, re.MULTILINE)
    assert re.search(r"^FROM \$\{FLASH_IMAGE\}", text, re.MULTILINE)
    assert "pip install" not in text


@pytest.mark.skipif(shutil.which("sh") is None, reason="POSIX sh required")
class TestEntrypointBehaviour:
    """The wrapper's control flow, exercised with a stub `infisical` on PATH.

    Nothing here contacts a real tenant: the stub records what it was asked to do and execs the
    command it was handed, which is exactly the part that has to keep working.
    """

    @staticmethod
    def _stub_dir(tmp_path: Path) -> Path:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "infisical"
        stub.write_text(
            "#!/bin/sh\n"
            'case "$1" in\n'
            "  login) echo stub-token ;;\n"
            '  run) shift; while [ "$1" != "--" ]; do shift; done; shift;\n'
            '       exec env FROM_VAULT=vault_value "$@" ;;\n'
            "esac\n"
        )
        stub.chmod(0o755)
        return bin_dir

    def _run(self, tmp_path, env, args):
        return subprocess.run(
            ["sh", str(ENTRYPOINT), *args],
            capture_output=True,
            text=True,
            env={"PATH": f"{self._stub_dir(tmp_path)}:/usr/bin:/bin", **env},
        )

    def test_no_client_id_is_a_transparent_passthrough(self, tmp_path):
        """Without the switch set the wrapper must be invisible: same env, same command."""
        result = self._run(
            tmp_path,
            {"HF_TOKEN": "from_container"},
            ["sh", "-c", 'echo "$HF_TOKEN ${FROM_VAULT:-no_vault}"'],
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "from_container no_vault"

    def test_client_id_injects_before_running_the_command(self, tmp_path):
        result = self._run(
            tmp_path,
            {
                "INFISICAL_CLIENT_ID": "cid",
                "INFISICAL_CLIENT_SECRET": "csec",
                "INFISICAL_PROJECT_ID": "proj",
                "INFISICAL_PATH": "/flash",
            },
            ["sh", "-c", 'echo "${FROM_VAULT:-no_vault}"'],
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "vault_value"

    def test_keep_reapplies_container_values_after_injection(self, tmp_path):
        """`infisical run` overrides existing env, so KEEP names must win over the vault.

        This is what lets a compose file point FREESOLO_BASE_URL at a service on its own Docker
        network while every other secret still comes from the vault.
        """
        result = self._run(
            tmp_path,
            {
                "INFISICAL_CLIENT_ID": "cid",
                "INFISICAL_CLIENT_SECRET": "csec",
                "INFISICAL_PROJECT_ID": "proj",
                "INFISICAL_PATH": "/flash",
                "INFISICAL_KEEP": "FROM_VAULT",
                "FROM_VAULT": "container_wins",
            },
            ["sh", "-c", 'echo "$FROM_VAULT"'],
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "container_wins"

    def test_keep_preserves_values_containing_spaces(self, tmp_path):
        """KEEP values are re-applied as quoted `K=V` args, so metacharacters survive intact.

        Word-splitting an unquoted string here would silently corrupt any value with a space in
        it -- and a corrupted URL or token fails much later, far from this line.
        """
        tricky = 'a b  c $HOME `id` "q"'
        result = self._run(
            tmp_path,
            {
                "INFISICAL_CLIENT_ID": "cid",
                "INFISICAL_CLIENT_SECRET": "csec",
                "INFISICAL_PROJECT_ID": "proj",
                "INFISICAL_PATH": "/flash",
                "INFISICAL_KEEP": "TRICKY",
                "TRICKY": tricky,
            },
            ["sh", "-c", 'printf "%s" "$TRICKY"'],
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == tricky

    def test_empty_cmd_reports_the_missing_command_not_a_credential(self, tmp_path):
        """`exec "$@"` with no arguments is a no-op that falls THROUGH to the injection path.

        An overlay that forgets to restate CMD lands here, and `set -u` would then blame
        INFISICAL_CLIENT_ID for what is really an empty command -- sending the operator to debug
        their vault credentials over a Dockerfile mistake.
        """
        result = self._run(tmp_path, {}, [])
        assert result.returncode == 2
        assert "no command given" in result.stderr
        assert "INFISICAL_CLIENT_ID" not in result.stderr

    def test_missing_required_setting_fails_instead_of_booting_uncredentialed(self, tmp_path):
        """With the switch on and configuration incomplete, refuse to start.

        Booting anyway would hand Flash's preflight a half-configured plane and report the error
        one layer away from its cause.
        """
        result = self._run(
            tmp_path,
            {"INFISICAL_CLIENT_ID": "cid", "INFISICAL_CLIENT_SECRET": "csec"},
            ["sh", "-c", "echo should_not_run"],
        )
        assert result.returncode != 0
        assert "should_not_run" not in result.stdout
