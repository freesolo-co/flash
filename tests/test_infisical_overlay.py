"""Infisical secret injection: the entrypoint wrapper and how the CLI reaches the image.

Flash reads its secrets from the process environment. This module pins the contract that lets
ONE image serve both ways of producing that environment -- `--env-file` and an Infisical vault --
selected at runtime by INFISICAL_CLIENT_ID rather than by which image was pulled.

These are the parts that are easy to break by editing a Dockerfile and impossible to notice
until a deployment fails to start.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_DOCKERFILE = REPO_ROOT / "Dockerfile"
OVERLAY_DIR = REPO_ROOT / "deploy" / "infisical"
OVERLAY_DOCKERFILE = OVERLAY_DIR / "Dockerfile"
ENTRYPOINT = OVERLAY_DIR / "entrypoint.sh"

# Every file that may install software from the network. A new one must be added here
# deliberately, which is the point: the checks below then apply to it.
INSTALLER_FILES = (
    BASE_DOCKERFILE,
    OVERLAY_DOCKERFILE,
    REPO_ROOT / "Dockerfile.worker",
    *sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")),
)

# `curl … | bash`, `wget … | sh`, and friends. Matches a fetch whose output is piped into a
# shell, with or without sudo, on one line.
PIPE_TO_SHELL = re.compile(r"(?:curl|wget)[^|\n]*\|\s*(?:sudo\s+)?(?:ba|z|k)?sh\b")


def _cmd_line(dockerfile: Path) -> str:
    lines = [ln for ln in dockerfile.read_text().splitlines() if ln.startswith("CMD ")]
    assert len(lines) == 1, f"{dockerfile} should declare exactly one CMD, found {len(lines)}"
    return lines[0]


@pytest.mark.parametrize("path", INSTALLER_FILES, ids=lambda p: p.name)
def test_nothing_pipes_a_downloaded_script_into_a_shell(path: Path):
    """`curl … | bash` executes whatever the vendor serves at request time.

    It cannot be reviewed, cannot be reproduced, and pins nothing -- and in a public repository
    it teaches the pattern to everyone who reads the file. Fetch to disk, verify a digest, then
    execute. This is a repo-wide rule, not an Infisical one; the parametrization is what keeps
    it that way when the next installer is added.
    """
    offenders = [
        ln.strip()
        for ln in path.read_text().splitlines()
        # Comments are stripped first: prose ABOUT the pattern (including the comments explaining
        # why these files avoid it) is not an instance of it. Both Dockerfiles and YAML use `#`.
        if PIPE_TO_SHELL.search(ln.split("#", 1)[0])
    ]
    assert not offenders, f"{path.name} pipes a downloaded script into a shell: {offenders}"


@pytest.mark.parametrize(
    "path", [BASE_DOCKERFILE, OVERLAY_DOCKERFILE], ids=lambda p: str(p.parent.name)
)
def test_the_infisical_cli_is_pinned_and_checksum_verified(path: Path):
    """A version pin alone still trusts whatever that URL serves later.

    The digest is what makes the install reproducible, so both must be present, and the
    verification has to happen where a mismatch aborts the build.
    """
    text = path.read_text()
    assert re.search(r"^ARG INFISICAL_VERSION=\d+\.\d+\.\d+$", text, re.MULTILINE)
    for arch in ("AMD64", "ARM64"):
        assert re.search(rf"^ARG INFISICAL_SHA256_{arch}=[0-9a-f]{{64}}$", text, re.MULTILINE), (
            f"{path} must pin a full sha256 for {arch}"
        )
    assert "sha256sum -c -" in text, f"{path} must verify the download before installing it"


def test_both_dockerfiles_pin_the_same_infisical_build():
    """The overlay re-runs the base's install step; a drifted pin would ship a different CLI."""
    pins = re.compile(r"^ARG (INFISICAL_(?:VERSION|SHA256_\w+))=(\S+)$", re.MULTILINE)
    base = dict(pins.findall(BASE_DOCKERFILE.read_text()))
    overlay = dict(pins.findall(OVERLAY_DOCKERFILE.read_text()))
    # Non-empty first: two files that both stopped pinning would compare equal and pass.
    assert base, "the base Dockerfile pins no infisical version or digest"
    assert base == overlay


def test_the_base_image_does_not_install_the_cli_by_default():
    """An open-source consumer's `docker build .` must produce a vendor-free image.

    The CLI is opt-in, so the build argument has to DEFAULT to false and the install has to be
    guarded by it. A default of true would ship vendor tooling to everyone who builds from the
    checkout, which is the thing this design is arranged to avoid.
    """
    text = BASE_DOCKERFILE.read_text()
    assert re.search(r"^ARG INSTALL_INFISICAL=false$", text, re.MULTILINE)
    assert re.search(r'if \[ "\$\{INSTALL_INFISICAL\}" = "true" \]', text)


def test_published_images_are_built_with_the_cli():
    """The deployments select Infisical by env var, which requires the binary to be present.

    Without this build argument the published image has no CLI, and setting INFISICAL_CLIENT_ID
    on a deployment would fail at start instead of switching it to vault-backed secrets.
    """
    workflow = (REPO_ROOT / ".github" / "workflows" / "publish-image.yml").read_text()
    assert "INSTALL_INFISICAL=true" in workflow


def test_base_image_declares_the_entrypoint():
    """One image serves both paths, so the wrapper ships in the base and always runs."""
    text = BASE_DOCKERFILE.read_text()
    assert 'ENTRYPOINT ["/usr/local/bin/flash-infisical-entrypoint"]' in text
    assert "COPY deploy/infisical/entrypoint.sh" in text


def test_publish_rebuilds_on_every_file_the_image_copies_in():
    """A file baked into the image must also trigger the build that publishes it.

    The path filter and the Dockerfile's COPY list are two statements of the same fact, kept in
    step by hand. When they drift, a change to the copied file merges without rebuilding, and the
    published tags keep serving the old copy until some unrelated watched file happens to change
    -- so a security fix to the secrets wrapper would look shipped while every deployment still
    ran the previous script. Deriving the requirement from the COPY lines keeps the next file
    added to the image from inheriting the same gap silently.
    """
    copied = re.findall(r"^COPY\s+(?!--)(\S+)", BASE_DOCKERFILE.read_text(), re.MULTILINE)
    # `COPY . .` already covers the whole tree; only specific paths need their own filter entry.
    specific = [src for src in copied if src not in {".", "./"}]
    assert specific, "expected the base image to COPY at least one specific path"

    workflow = (REPO_ROOT / ".github" / "workflows" / "publish-image.yml").read_text()
    paths_block = workflow.split("paths:", 1)[1].split("workflow_dispatch", 1)[0]
    for src in specific:
        assert f'"{src}"' in paths_block, (
            f"Dockerfile copies {src} into the image, but publish-image.yml would not rebuild "
            f"when it changes -- add it to on.push.paths"
        )


def test_overlay_restates_the_base_cmd_verbatim():
    """Setting ENTRYPOINT in a DERIVED image resets the inherited CMD to null.

    Docker does not carry a base image's CMD into a child that declares its own ENTRYPOINT. An
    overlay that relies on inheriting it hands the wrapper no command, and the container exits
    without ever starting the server. The two CMD lines must therefore stay byte-identical.
    """
    assert _cmd_line(OVERLAY_DOCKERFILE) == _cmd_line(BASE_DOCKERFILE)


def test_overlay_builds_on_top_of_the_published_image():
    """The overlay must not rebuild flash: it layers onto an existing image via FLASH_IMAGE."""
    text = OVERLAY_DOCKERFILE.read_text()
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

    def _run(self, tmp_path, env, args, *, with_cli=True):
        path = f"{self._stub_dir(tmp_path)}:/usr/bin:/bin" if with_cli else "/usr/bin:/bin"
        return subprocess.run(
            ["sh", str(ENTRYPOINT), *args],
            capture_output=True,
            text=True,
            env={"PATH": path, **env},
        )

    def test_env_file_deployment_works_without_the_cli_installed(self, tmp_path):
        """The vendor-free image must still run: no CLI on PATH, no INFISICAL_* set, no problem.

        This is the default `docker build .` artifact and the `--env-file` deployment. If the
        wrapper ever required the binary unconditionally, this is the test that fails.
        """
        result = self._run(
            tmp_path,
            {"HF_TOKEN": "from_container"},
            ["sh", "-c", 'echo "$HF_TOKEN"'],
            with_cli=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "from_container"

    def test_switch_on_without_the_cli_names_the_build_argument(self, tmp_path):
        """Asking for injection from an image built without the CLI must fail loudly, and say why.

        Left to `set -e` this is `infisical: not found` -- a message that names a missing command
        but not the build argument that provides it. Booting anyway would be worse: the server
        would start with the vault's secrets absent and fail far from the cause.
        """
        result = self._run(
            tmp_path,
            {
                "INFISICAL_CLIENT_ID": "cid",
                "INFISICAL_CLIENT_SECRET": "csec",
                "INFISICAL_PROJECT_ID": "proj",
                "INFISICAL_PATH": "/flash",
            },
            ["sh", "-c", "echo should_not_run"],
            with_cli=False,
        )
        assert result.returncode == 2
        assert "should_not_run" not in result.stdout
        assert "INSTALL_INFISICAL=true" in result.stderr

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

    def test_keep_naming_an_unset_variable_leaves_the_vault_value_alone(self, tmp_path):
        """A KEEP entry the container never set must NOT wipe the injected secret.

        The loop re-applies each KEEP name as a `K=V` argument to `env`. An unset name expands
        to nothing, so a bare `K=` would overwrite the vault's value with an empty string --
        turning a typo in INFISICAL_KEEP into a silently missing credential, which surfaces as
        an authentication failure far from its cause.
        """
        result = self._run(
            tmp_path,
            {
                "INFISICAL_CLIENT_ID": "cid",
                "INFISICAL_CLIENT_SECRET": "csec",
                "INFISICAL_PROJECT_ID": "proj",
                "INFISICAL_PATH": "/flash",
                "INFISICAL_KEEP": "FROM_VAULT",
            },
            ["sh", "-c", 'printf "%s" "${FROM_VAULT:-EMPTY}"'],
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == "vault_value"

    def test_keep_honours_an_explicitly_empty_container_value(self, tmp_path):
        """Set-but-empty is a deliberate choice and still beats the vault.

        This is the flip side of the test above: the fix must distinguish "unset" from "set to
        an empty string" rather than treating every falsy value as absent.
        """
        result = self._run(
            tmp_path,
            {
                "INFISICAL_CLIENT_ID": "cid",
                "INFISICAL_CLIENT_SECRET": "csec",
                "INFISICAL_PROJECT_ID": "proj",
                "INFISICAL_PATH": "/flash",
                "INFISICAL_KEEP": "FROM_VAULT",
                "FROM_VAULT": "",
            },
            ["sh", "-c", 'printf "%s" "${FROM_VAULT?unset}"'],
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == ""

    def test_keep_rejects_a_name_that_is_not_an_identifier(self, tmp_path):
        """KEEP names reach `eval`, so a non-identifier is refused rather than executed."""
        marker = tmp_path / "executed"
        result = self._run(
            tmp_path,
            {
                "INFISICAL_CLIENT_ID": "cid",
                "INFISICAL_CLIENT_SECRET": "csec",
                "INFISICAL_PROJECT_ID": "proj",
                "INFISICAL_PATH": "/flash",
                "INFISICAL_KEEP": f"$(touch {marker})",
            },
            ["sh", "-c", "echo reached"],
        )
        assert result.returncode == 2
        assert not marker.exists(), "a KEEP entry was executed as a command"
        assert "not a variable name" in result.stderr
        assert "reached" not in result.stdout

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
