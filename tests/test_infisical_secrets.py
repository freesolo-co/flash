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
PUBLISH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-image.yml"
ENTRYPOINT = REPO_ROOT / "deploy" / "infisical" / "entrypoint.sh"


def test_the_infisical_cli_is_pinned_and_checksum_verified():
    """A version pin alone still trusts whatever that URL serves later.

    The digest is what makes the install reproducible, so both must be present, and the
    verification has to happen where a mismatch aborts the build.
    """
    text = BASE_DOCKERFILE.read_text()
    assert re.search(r"^ARG INFISICAL_VERSION=\d+\.\d+\.\d+$", text, re.MULTILINE)
    for arch in ("AMD64", "ARM64"):
        assert re.search(rf"^ARG INFISICAL_SHA256_{arch}=[0-9a-f]{{64}}$", text, re.MULTILINE), (
            f"the Dockerfile must pin a full sha256 for {arch}"
        )
    # Substring presence cannot see ORDER, so it passed just as happily with the check moved
    # below the install -- where dpkg has already run the package's maintainer scripts by the
    # time the mismatch is noticed. Assert the actual sequence, and that the file being checked
    # is the file being installed.
    verify = text.index("sha256sum -c -")
    install = text.index('apt-get install -y --no-install-recommends "${deb}"')
    assert verify < install, "the checksum must be verified BEFORE the package is installed"
    checked = re.search(r'echo "\$\{sha\}  (\S+)" \| sha256sum -c -', text)
    assert checked, "the checksum line must name the file it verifies"
    assert checked.group(1) == "${deb}", (
        f"the verified file must be the downloaded package, not {checked.group(1)}"
    )


def test_the_base_image_does_not_install_the_cli_by_default():
    """An open-source consumer's `docker build .` must produce a vendor-free image.

    The CLI is opt-in, so the build argument has to DEFAULT to false and the install has to be
    guarded by it. A default of true would ship vendor tooling to everyone who builds from the
    checkout, which is the thing this design is arranged to avoid.
    """
    text = BASE_DOCKERFILE.read_text()
    assert re.search(r"^ARG INSTALL_INFISICAL=false$", text, re.MULTILINE)
    assert re.search(r'if \[ "\$\{INSTALL_INFISICAL\}" = "true" \]', text)


def _yaml_block(text: str, key: str) -> list[str]:
    """Return the entries under the first `key:` block, by indentation.

    Done by hand rather than with a yaml parser: pyyaml reaches this environment only as a
    transitive dependency of datasets/transformers, so importing it here would make a workflow
    test quietly contingent on the ML stack staying installed.

    Indentation is what makes this a structural read rather than a substring search. A comment
    explaining a build argument contains the same text as the argument itself, so a whole-file
    check for that text passes just as well when the real entry is gone. Entries are collected
    only while they stay indented past the key, which a comment above the step never is.
    """
    entries: list[str] = []
    depth: int | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if depth is None:
            if stripped == f"{key}:" or stripped == f"{key}: |":
                depth = indent
            continue
        if indent <= depth:
            break
        entries.append(stripped.removeprefix("- ").strip().strip("\"'"))
    return entries


def test_published_images_are_built_with_the_cli():
    """The deployments select Infisical by env var, which requires the binary to be present.

    Without this build argument the published image has no CLI, and setting INFISICAL_CLIENT_ID
    on a deployment would fail at start instead of switching it to vault-backed secrets.

    Read out of the `build-args:` block rather than the whole file: the step is introduced by a
    comment that names the same argument, so a substring check over the file stays green after
    the real entry is deleted and the published images silently lose the CLI.
    """
    assert "INSTALL_INFISICAL=true" in _yaml_block(PUBLISH_WORKFLOW.read_text(), "build-args")


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

    watched = _yaml_block(PUBLISH_WORKFLOW.read_text(), "paths")
    assert watched, "could not read on.push.paths out of publish-image.yml"

    for src in specific:
        assert src in watched, (
            f"Dockerfile copies {src} into the image, but publish-image.yml would not rebuild "
            f"when it changes -- add it to on.push.paths"
        )


def test_the_image_declares_exactly_one_cmd():
    """The entrypoint execs its arguments, so an absent or ambiguous CMD leaves it nothing to run.

    A derived image that declares its own ENTRYPOINT also resets this to null and must restate
    it; the entrypoint's empty-argv branch is what turns that mistake into a legible error.
    """
    cmds = [ln for ln in BASE_DOCKERFILE.read_text().splitlines() if ln.startswith("CMD ")]
    assert cmds == ['CMD ["python", "-m", "flash.server", "--host", "0.0.0.0", "--port", "8080"]']


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

    def test_an_empty_client_id_refuses_rather_than_passing_through(self, tmp_path):
        """Set-but-empty is an accident, not a way to turn the switch off.

        A compose `${VAR}` with nothing behind it, or a secretKeyRef to an absent key, produces
        exactly this. Reading it as "unset" boots the control plane on whatever ambient
        environment happened to be present -- the silent-credentials failure the switch exists to
        prevent -- so it must fail, and must not be confused with the deliberate passthrough.
        """
        result = self._run(
            tmp_path,
            {"INFISICAL_CLIENT_ID": "", "HF_TOKEN": "ambient"},
            ["sh", "-c", "echo should_not_run"],
        )
        assert result.returncode == 2
        assert "should_not_run" not in result.stdout
        assert "set but empty" in result.stderr

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

    def test_the_wrapper_does_not_delete_variables_it_was_never_asked_about(self, tmp_path):
        """The wrapper must not consume a name out of the child's environment as scratch space.

        The KEEP loop needs somewhere to hold each name's value while it builds the argument
        list. Holding it in a shell variable makes that variable's NAME part of the contract:
        whatever the container set under it is clobbered, and the cleanup `unset` afterwards
        deletes it from the child entirely -- a variable the caller never mentioned in
        INFISICAL_KEEP and had no reason to expect this script to touch.

        Worse, the `unset` sat outside the loop, so it fired even when INFISICAL_KEEP was unset
        and the loop body never ran: a deployment using no KEEP list at all still lost these two
        names. `keep_value` and `keep_is_set` are ordinary enough that a container could plausibly
        define them, and the failure is invisible until something downstream reads one.
        """
        result = self._run(
            tmp_path,
            {
                "INFISICAL_CLIENT_ID": "cid",
                "INFISICAL_CLIENT_SECRET": "csec",
                "INFISICAL_PROJECT_ID": "proj",
                "INFISICAL_PATH": "/flash",
                # deliberately NO INFISICAL_KEEP: the loop never runs, and these must survive
                "keep_value": "container_value",
                "keep_is_set": "container_flag",
            },
            ["sh", "-c", 'echo "${keep_value-<GONE>}/${keep_is_set-<GONE>}"'],
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "container_value/container_flag"

    def test_the_loop_iterator_does_not_leak_into_the_child(self, tmp_path):
        """The `for` variable is scratch space too, and it clobbers its own name.

        Removing the scratch variable from the loop BODY left the iterator itself. POSIX `for`
        assigns to the name in place, so a container that exported a variable of that name hands
        the loop an already-exported name and the child receives the list's last entry instead.
        Same defect as the body's scratch name, one line further up, and the same invisibility --
        nothing fails until something downstream reads it.

        The name here must track the iterator in entrypoint.sh. An earlier revision of this test
        pinned the old name `k`; renaming the iterator silently made it vacuous, since a leak
        through `_infisical_keep_name` cannot show up in a variable called `k`. A test naming the
        wrong variable passes for the wrong reason, so it is asserted against the real one.

        Distinct from the sibling test above: that one covers KEEP being *unset*, so the loop
        never runs. This one needs the loop to actually iterate, which is when the iterator is
        assigned.
        """
        result = self._run(
            tmp_path,
            {
                "INFISICAL_CLIENT_ID": "cid",
                "INFISICAL_CLIENT_SECRET": "csec",
                "INFISICAL_PROJECT_ID": "proj",
                "INFISICAL_PATH": "/flash",
                "INFISICAL_KEEP": "FROM_VAULT",
                "_infisical_keep_name": "ORIGINAL",
            },
            ["sh", "-c", 'echo "${_infisical_keep_name-<GONE>}"'],
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ORIGINAL"

    def test_keeping_the_iterators_own_name_does_not_invent_a_value(self, tmp_path):
        """A KEEP entry naming the iterator must not materialise out of the loop's own state.

        With `INFISICAL_KEEP=_infisical_keep_name` and NO container variable of that name, the
        loop's `${$name}` indirection resolves through the iterator: while processing that entry
        the iterator's value *is* its own name, so the buggy version emitted
        `_infisical_keep_name=_infisical_keep_name` and the child received a variable nobody set,
        holding a value that is just its own name.

        Asserting `<GONE>` rather than a value is deliberate. Every other KEEP test supplies a
        container value, so all of them stay green while this hole is open -- the wrapper is
        only correct here if it emits nothing at all.
        """
        result = self._run(
            tmp_path,
            {
                "INFISICAL_CLIENT_ID": "cid",
                "INFISICAL_CLIENT_SECRET": "csec",
                "INFISICAL_PROJECT_ID": "proj",
                "INFISICAL_PATH": "/flash",
                "INFISICAL_KEEP": "_infisical_keep_name",
                # deliberately NO container variable of that name: nothing to re-apply
            },
            ["sh", "-c", 'echo "${_infisical_keep_name-<GONE>}"'],
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "<GONE>"

    def test_keeping_the_iterators_own_name_returns_the_container_value(self, tmp_path):
        """Both iterator failures at once: KEEP names it AND the container exported it.

        The two tests above isolate one failure each -- clobbering a bystander, and inventing a
        value from nothing. This is the case where they overlap, and it is the one a "skip that
        name" fix gets wrong: skipping is right when there is nothing to re-apply, but here the
        container really did set the variable and really did ask to keep it, so the honest answer
        is its own value. Reading it inside the loop cannot produce that -- by then the iterator
        has overwritten it -- so the value has to be captured before the loop runs.
        """
        result = self._run(
            tmp_path,
            {
                "INFISICAL_CLIENT_ID": "cid",
                "INFISICAL_CLIENT_SECRET": "csec",
                "INFISICAL_PROJECT_ID": "proj",
                "INFISICAL_PATH": "/flash",
                "INFISICAL_KEEP": "_infisical_keep_name",
                "_infisical_keep_name": "ORIGINAL",
            },
            ["sh", "-c", 'echo "${_infisical_keep_name-<GONE>}"'],
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ORIGINAL"

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

    def test_the_vault_still_wins_for_the_iterator_name_nobody_kept(self, tmp_path):
        """Fixing the iterator leak must not promote that name to a permanent KEEP entry.

        The previous fix restored the container's value by handing it to `env` as an operand.
        Operands are applied AFTER injection, which is exactly how a KEEP entry beats the vault
        -- so the iterator's name won over the vault unconditionally, including when the keep
        list was empty and there was nothing to protect it from. Only names listed in
        INFISICAL_KEEP are allowed to do that.

        The default stub cannot see this: it injects `FROM_VAULT` alone, so the vault never
        defines the iterator's name and container-wins is indistinguishable from vault-wins.
        This one injects that name, which is what makes the two outcomes differ.
        """
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "infisical"
        stub.write_text(
            "#!/bin/sh\n"
            'case "$1" in\n'
            "  login) echo stub-token ;;\n"
            '  run) shift; while [ "$1" != "--" ]; do shift; done; shift;\n'
            '       exec env _infisical_keep_name=vault_value "$@" ;;\n'
            "esac\n"
        )
        stub.chmod(0o755)

        def run(keep: str | None) -> str:
            env = {
                "PATH": f"{bin_dir}:/usr/bin:/bin",
                "INFISICAL_CLIENT_ID": "cid",
                "INFISICAL_CLIENT_SECRET": "csec",
                "INFISICAL_PROJECT_ID": "proj",
                "INFISICAL_PATH": "/flash",
                "_infisical_keep_name": "CONTAINER_VAL",
            }
            if keep is not None:
                env["INFISICAL_KEEP"] = keep
            result = subprocess.run(
                ["sh", str(ENTRYPOINT), "sh", "-c", 'echo "${_infisical_keep_name-<GONE>}"'],
                capture_output=True,
                text=True,
                env=env,
            )
            assert result.returncode == 0, result.stderr
            return result.stdout.strip()

        # Not kept -- by the list being absent, or by naming something else entirely. Both are
        # the ordinary case for a container that happens to export this name for its own reasons.
        assert run(None) == "vault_value"
        assert run("SOMETHING_ELSE") == "vault_value"
        # Kept explicitly: now, and only now, the container's value is the right answer.
        assert run("_infisical_keep_name") == "CONTAINER_VAL"

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

    def test_keep_preserves_a_multiline_value_byte_for_byte(self, tmp_path):
        """A kept PEM key must arrive exactly as the container set it, trailing newline included.

        Command substitution strips ALL trailing newlines, so re-applying the value through
        `$(...)` shortened precisely the values most likely to break something downstream: a
        private key, a certificate chain, anything a parser reads line-wise. It failed silently
        and only for multiline values, so the single-line tests above stayed green.
        """
        pem = "-----BEGIN KEY-----\nabc\ndef\n-----END KEY-----\n"
        result = self._run(
            tmp_path,
            {
                "INFISICAL_CLIENT_ID": "cid",
                "INFISICAL_CLIENT_SECRET": "csec",
                "INFISICAL_PROJECT_ID": "proj",
                "INFISICAL_PATH": "/flash",
                "INFISICAL_KEEP": "PEM",
                "PEM": pem,
            },
            ["sh", "-c", 'printf "%s" "$PEM"'],
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == pem, "the kept value lost bytes on its way through the wrapper"

    def test_empty_cmd_reports_the_missing_command_not_a_credential(self, tmp_path):
        """`exec "$@"` with no arguments is a no-op that falls THROUGH to the injection path.

        An image built FROM ours that forgets to restate CMD lands here, and `set -u` would then
        blame INFISICAL_CLIENT_ID for what is really an empty command -- sending the operator to
        debug their vault credentials over a Dockerfile mistake.
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
