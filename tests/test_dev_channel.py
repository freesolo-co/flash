"""The dev-channel (`freesolo-flash-dev` / `flash-dev`) derivation and build transforms.

The dev channel is the same source as `freesolo-flash`, built with a single flipped line in
flash/_internal/channel.py (CHANNEL "prod" -> "dev"). These tests pin both ends of that contract: the
checked-in source is the prod channel, and the build-time rewrites produce a coherent dev package.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import flash.serve.contract.urls as serving_urls

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_build_module():
    path = REPO_ROOT / "scripts" / "build_dev_dist.py"
    spec = importlib.util.spec_from_file_location("build_dev_dist", path)
    assert spec is not None, f"could not load a spec for {path}"
    assert spec.loader is not None, f"spec for {path} has no loader"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ui_leaf_modules_import_cold() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import flash.cli.ui.env_panels; import flash.cli.ui.tables",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert result.returncode == 0, result.stderr


def test_heartbeat_interpretation_does_not_import_rendering() -> None:
    script = (
        "import sys; from flash.cli.ui import heartbeat; "
        "heartbeat._heartbeat_pairs({'state': 'running', "
        "'last_heartbeat': {'stage': 'sft_initializing', 'ts': 1}}); "
        "assert 'flash.cli.ui.render' not in sys.modules"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert result.returncode == 0, result.stderr


def test_checked_in_source_is_prod_channel():
    from flash._internal import channel as _channel
    from flash.client import config

    assert _channel.CHANNEL == "prod"
    assert _channel.CLI_NAME == "flash"
    assert _channel.DIST_NAME == "freesolo-flash"
    # The functional defaults everything reads from.
    assert config.DEFAULT_API_URL == "https://flash.freesolo.co"

    assert serving_urls.default_serving_url() == "https://serve.freesolo.co"


def test_default_api_url_follows_channel():
    from flash.client.config import DEV_API_URL, PROD_API_URL, default_api_url

    assert default_api_url("prod") == PROD_API_URL == "https://flash.freesolo.co"
    assert default_api_url("dev") == DEV_API_URL == "https://flash-dev.freesolo.co"


def test_default_serving_url_follows_channel():
    """The serving plane is per-channel, exactly like the control plane above.

    Each channel's serving app is backed by its own Supabase project, so an org
    row exists in one of them and not the other. A dev-channel client defaulting to prod serving
    posts a dev `org_id` into the prod database and takes a 23503 foreign-key violation -- which
    reads as a serving outage but is a routing defect, and which no retry can clear.
    """
    from flash.serve.contract.urls import (
        DEV_FREESOLO_SERVING_URL,
        PROD_FREESOLO_SERVING_URL,
        default_serving_url,
    )

    assert default_serving_url("prod") == PROD_FREESOLO_SERVING_URL == "https://serve.freesolo.co"
    assert default_serving_url("dev") == DEV_FREESOLO_SERVING_URL == "https://serve-dev.freesolo.co"
    # The two planes must not collapse onto one host: that is the whole defect being fixed, and an
    # equality here would make every assertion above pass while routing dev traffic to prod.
    assert PROD_FREESOLO_SERVING_URL != DEV_FREESOLO_SERVING_URL


def test_every_hosted_default_flips_with_the_channel():
    """Build the dev channel for real and assert NO hosted default is left pointing at prod.

    This is the regression that was missing. `serve.freesolo.co` sat hardcoded in
    `flash/serve/deployment/deploy.py` while `client/config.py` derived its URL from CHANNEL, so the dev
    package shipped a prod serving endpoint and every dev deploy failed the prod org_id FK.
    Each half was internally consistent, so nothing was red -- the same shape as the earlier
    catalog drift between the flash and serving rank limits.

    Asserting per-constant would just re-encode the mistake, so this scans the built dev tree for
    any surviving prod host. It fails on the NEXT hosted endpoint added without a channel split,
    not only on the one already fixed.
    """
    import re

    build = _load_build_module()
    dev_channel_src = build.rewrite_channel(
        (REPO_ROOT / "flash" / "_internal" / "channel.py").read_text()
    )
    namespace: dict = {}
    exec(compile(dev_channel_src, "flash/_internal/channel.py[dev]", "exec"), namespace)
    dev_channel = namespace["CHANNEL"]
    assert dev_channel == "dev"

    from flash.client.config import default_api_url
    from flash.serve.contract.urls import default_serving_url

    # Every channel-derived default, evaluated as the built dev package would.
    dev_defaults = {
        "control plane (flash.client.config.default_api_url)": default_api_url(dev_channel),
        "serving plane (flash.serve.deployment.serving_urls.default_serving_url)": default_serving_url(
            dev_channel
        ),
    }
    assert dev_defaults, "no hosted defaults collected -- the scan would pass vacuously"

    # A prod host is any freesolo.co host without a dev marker in its leftmost label.
    prod_host = re.compile(r"^https://(?!.*-dev\.)(?!.*dev\.)[a-z0-9-]+\.freesolo\.co$")
    leaked = {name: url for name, url in dev_defaults.items() if prod_host.match(url)}
    assert not leaked, (
        "dev-channel default(s) still address production: "
        + "; ".join(f"{name} -> {url}" for name, url in sorted(leaked.items()))
        + ". Each hosted plane is backed by its own Supabase project, so a dev client hitting a "
        "prod endpoint fails the org_id foreign key. Give the constant a prod/dev pair that "
        "derives from CHANNEL, as flash.client.config and flash.serve.deployment.deploy both do."
    )


def test_dev_channel_marker_derives_dev_names():
    # Run the real flash/_internal/channel.py through the build rewrite and confirm everything that differs
    # between channels (CLI name, dist name, and thus the default URL) flips together.
    build = _load_build_module()
    channel_src = (REPO_ROOT / "flash" / "_internal" / "channel.py").read_text()
    dev_src = build.rewrite_channel(channel_src)

    namespace: dict = {}
    exec(compile(dev_src, "flash/_internal/channel.py[dev]", "exec"), namespace)
    assert namespace["CHANNEL"] == "dev"
    assert namespace["CLI_NAME"] == "flash-dev"
    assert namespace["DIST_NAME"] == "freesolo-flash-dev"


def test_rewrite_channel_requires_exactly_one_prod_marker():
    import pytest

    build = _load_build_module()
    with pytest.raises(SystemExit):
        build.rewrite_channel('CHANNEL = "dev"\n')  # already flipped -> zero matches


def _real_console_scripts() -> dict[str, str]:
    """The checked-in [project.scripts] table, not a synthetic stand-in."""
    import tomllib

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    return dict(data["project"]["scripts"])


def test_flash_cli_alias_reaches_the_same_entry_point():
    """`flash-cli` must exist and be the same entry point as `flash`.

    The `server` and `dev` extras install runpod-flash, which declares its own `flash` console
    script. Whichever distribution is installed last wins, so on a control-plane host `flash` can
    silently belong to runpod-flash -- it exits 0 without doing anything. `flash-cli` is the name
    nothing else claims, so it is what SELF_HOSTING.md points operators at.
    """
    scripts = _real_console_scripts()
    assert scripts.get("flash") == "flash.cli.parsing.main:main"
    assert scripts.get("flash-cli") == scripts["flash"], (
        "flash-cli must stay an alias of flash. SELF_HOSTING.md tells self-hosters to use it "
        "when runpod-flash's console script shadows `flash`."
    )


def test_printed_commands_name_the_executable_the_operator_invoked():
    """Generated commands must name the script actually run, not the channel default.

    On the documented `[server]` install, `flash` can be runpod-flash's console script. Printing
    `flash runs cancel <id>` there hands the operator a command that exits 0 without cancelling,
    leaving a billed run alive -- so an operator who reached us via `flash-cli` must be told
    `flash-cli`. Anything that is not one of our scripts (a test runner, a renamed copy) falls
    back to the channel default so the hint stays copy-pasteable.
    """
    import importlib
    import sys
    from unittest import mock

    from flash._internal import channel

    for argv0, expected in (
        ("/usr/local/bin/flash-cli", "flash-cli"),
        ("/usr/local/bin/flash", "flash"),
        ("flash-cli.exe", "flash-cli"),
        ("/tmp/x/__main__.py", "flash"),
        ("", "flash"),
    ):
        with mock.patch.object(sys, "argv", [argv0]):
            resolved = importlib.reload(channel).CLI_NAME
        assert resolved == expected, argv0

    importlib.reload(channel)  # restore the pytest-invoked value for later tests


def test_operator_hints_follow_flash_cli_invocation():
    """Operator hints rendered after a flash-cli entry must keep naming flash-cli."""
    import importlib
    import re
    import sys
    from unittest import mock

    from flash._internal import channel
    from flash.cli.ui import env_panels, render
    from flash.cli.ui import env_panels as env_panels_module
    from flash.cli.ui import render as render_module
    from flash.cli.ui import tables as tables_module

    original_argv = list(sys.argv)
    outputs: dict[str, str] = {}
    try:
        with mock.patch.object(sys, "argv", ["/usr/local/bin/flash-cli"]):
            # reload each canonical owner after channel so its imported CLI_NAME follows argv.
            importlib.reload(channel)
            importlib.reload(tables_module)
            importlib.reload(env_panels_module)
            render = importlib.reload(render_module)
            outputs = {
                "flash/cli/ui/render.py:login_ok": render.login_ok(None),
                "flash/cli/ui/render.py:login_failed": render.login_failed("bad key"),
                # _QUIET_HEARTBEAT_HINT is deliberately absent: it now points at the age on the
                # panel rather than at `flash runs log`, so it names no command to rename. Add it
                # back here only if it starts spelling one again.
                "flash/cli/ui/render.py:env_setup": render.env_setup(
                    ["environment.py"], "11111111-1111-4111-8111-111111111111"
                ),
                "flash/cli/ui/render.py:env_list(local)": env_panels.env_list(["."]),
                "flash/cli/ui/render.py:env_list(empty)": env_panels.env_list([]),
                "flash/cli/ui/tables.py:models_table": tables_module.models_table(
                    [{"id": "acme/model"}]
                ),
                "flash/cli/ui/tables.py:projects_table": tables_module.projects_table([]),
                "flash/cli/ui/tables.py:checkpoints_table": tables_module.checkpoints_table(
                    "run-1", [{"step": 1}]
                ),
            }
    finally:
        with mock.patch.object(sys, "argv", original_argv):
            importlib.reload(channel)
            importlib.reload(tables_module)
            importlib.reload(env_panels_module)
            importlib.reload(render_module)

    bare_command = re.compile(
        r"(?<![\w-])flash (?=(?:env|login|models|projects|runs|traces|train|whoami)\b)"
    )
    for site, output in outputs.items():
        assert "flash-cli " in output, f"{site} did not render a flash-cli command: {output!r}"
        match = bare_command.search(output)
        assert match is None, f"{site} rendered a bare flash command: {output!r}"


def test_scaffolded_files_follow_flash_cli_invocation(tmp_path, monkeypatch):
    """The GENERATED environment.py / evaluations.py must name the invoked CLI too.

    The terminal output above already did, but the scaffold guidance was a literal `flash env push`.
    Those files outlive the terminal session -- an operator on a host where `flash` is RunPod's CLI
    reads the docstring later, runs the shadowed binary, and it exits 0 having published nothing.
    That is the failure mode the alias exists to prevent, so the two renderings have to agree.
    """
    import importlib
    import sys
    from argparse import Namespace
    from unittest import mock

    from flash._internal import channel
    from flash.cli.commands.env.ops import retained as retained_module
    from flash.cli.commands.env.ops import setup as setup_module

    project = "11111111-1111-4111-8111-111111111111"
    original_argv = list(sys.argv)
    try:
        with mock.patch.object(sys, "argv", ["/usr/local/bin/flash-cli"]):
            importlib.reload(channel)
            importlib.reload(retained_module)
            env_setup = importlib.reload(setup_module)
            monkeypatch.setattr(
                env_setup,
                "_require_setup_project",
                lambda _args, **_kwargs: {"id": project, "name": "Test"},
            )
            monkeypatch.setattr(
                "flash.client.config.load_credentials",
                lambda: ("https://flash.freesolo.co", "key"),
            )
            monkeypatch.chdir(tmp_path)
            rc = env_setup.cmd_env_setup(
                Namespace(
                    project=project,
                    yes=True,
                    turn_mode=None,
                    reasoning=None,
                    from_traces=None,
                    trace=None,
                    force=False,
                )
            )
            assert rc == 0
    finally:
        with mock.patch.object(sys, "argv", original_argv):
            importlib.reload(channel)
            importlib.reload(retained_module)
            importlib.reload(setup_module)

    written = {p.name: p.read_text(encoding="utf-8") for p in tmp_path.rglob("*") if p.is_file()}
    for name in ("environment.py", "evaluations.py"):
        assert "flash-cli env push" in written[name], f"{name}: {written[name]!r}"
        assert "`flash env push" not in written[name], f"{name} names the shadowed binary"
    assert "flash-cli env eval" in written["evaluations.py"]


def test_module_invocation_is_named_back_as_itself_not_as_flash():
    """`python -m flash.cli` must print itself, because it is the escape hatch FROM `flash`.

    README, CONTRIBUTING and SELF_HOSTING all send an operator whose `flash` is shadowed to the
    `-m` form. Falling back to the channel default there answers that operator with the exact name
    they were told to stop using.

    Resolution reads `sys.orig_argv`, NOT `sys.argv`: runpy imports the target's parent package --
    which is what imports channel -- before rewriting `sys.argv`, so `sys.argv[0]` is the bare
    string `-m` at that moment, for `python -m pytest` just as much as for us. Hence the pytest row
    below: it is what distinguishes reading the real command line from matching on `-m` alone.

    `sys.argv` is passed per row rather than held fixed because the module's position in
    `orig_argv` is derived from the length of the trailing script arguments.
    """
    import importlib
    import sys
    from unittest import mock

    from flash._internal import channel

    py = "/usr/bin/python3"
    # an absolute interpreter is echoed as sys.executable -- the running one, which is the only
    # path guaranteed to still have flash installed. Bare names are covered by the test below.
    exe = sys.executable
    for orig_argv, argv, expected in (
        ([py, "-m", "flash.cli", "runs", "list"], ["-m", "runs", "list"], f"{exe} -m flash.cli"),
        ([py, "-m", "pytest", "tests/"], ["-m", "tests/"], "flash"),
        ([py, "-m", "flash.server"], ["-m"], "flash"),
        # interpreter flags shift the module's index, including one that takes its own value
        ([py, "-W", "ignore", "-m", "flash.cli", "-v"], ["-m", "-v"], f"{exe} -m flash.cli"),
        # a console script names no module: `-m` in ITS arguments must not read as the flag
        (
            [py, "/usr/local/bin/flash-cli", "-m", "flash.cli"],
            ["/usr/local/bin/flash-cli", "-m", "flash.cli"],
            "flash-cli",
        ),
        ([py, "-m"], ["-m"], "flash"),  # truncated command line must not raise
        ([], ["-m"], "flash"),  # orig_argv absent (embedded interpreter) must not raise
    ):
        with (
            mock.patch.object(sys, "orig_argv", orig_argv),
            mock.patch.object(sys, "argv", argv),
        ):
            resolved = importlib.reload(channel).CLI_NAME
        assert resolved == expected, orig_argv

    importlib.reload(channel)  # restore the pytest-invoked value for later tests


def test_module_invocation_names_a_runnable_interpreter_not_bare_python():
    """Printing `python` reintroduces this module's own bug in a second form.

    A host that ships only `python3` has no `python` on PATH at all, so the printed cancel command
    fails with "command not found" while the invocation that printed it worked -- the same
    "operator believes the run is cancelled while it keeps billing" hazard the console-script fix
    exists to close. Inside a virtualenv it is worse than absent: bare `python` may RESOLVE, to a
    system interpreter running a different flash entirely.

    So the interpreter is preserved, not assumed. A bare word typed by the operator is echoed back
    (it resolved for them a moment ago, and stays short); anything carrying a path separator is
    location-bound, so `sys.executable` is printed instead -- the interpreter actually running,
    which is the one that definitely has this flash installed.
    """
    import importlib
    import sys
    from unittest import mock

    from flash._internal import channel

    exe = sys.executable
    for orig_argv, expected in (
        # bare names are echoed as typed: they are a PATH lookup that just succeeded
        (["python3", "-m", "flash.cli"], "python3 -m flash.cli"),
        (["python3.12", "-m", "flash.cli"], "python3.12 -m flash.cli"),
        (["python", "-m", "flash.cli"], "python -m flash.cli"),
        # paths are location-bound; echo the running interpreter instead
        (["/usr/bin/python3", "-m", "flash.cli"], f"{exe} -m flash.cli"),
        ([".venv/bin/python", "-m", "flash.cli"], f"{exe} -m flash.cli"),
        (["./python3", "-m", "flash.cli"], f"{exe} -m flash.cli"),
    ):
        with (
            mock.patch.object(sys, "orig_argv", orig_argv),
            mock.patch.object(sys, "argv", ["-m"]),
        ):
            resolved = importlib.reload(channel).CLI_NAME
        assert resolved == expected, orig_argv
        # whatever we print, it must never be a bare `python` the operator did not type
        assert not resolved.startswith("python -m") or orig_argv[0] == "python", orig_argv

    importlib.reload(channel)  # restore the pytest-invoked value for later tests


def test_module_launch_interpreter_survives_a_path_with_spaces():
    """An interpreter path is pasted into a shell, so it has to be one shell token.

    `C:\\Program Files\\Python\\python.exe` and a virtualenv under a spaced directory otherwise
    split into several argv words: the printed `runs cancel` resolves `/opt/My` and dies with "No
    such file or directory" while the run keeps billing -- the same silent non-cancel the invoked-
    name fix exists to close, reintroduced one layer down.

    Quoting must stay conditional. An ordinary unspaced path gains no quotes, because a hint the
    operator reads is worth keeping clean when there is nothing to escape.
    """
    import importlib
    import shlex
    import sys
    from unittest import mock

    from flash._internal import channel

    spaced = "/opt/My Python/bin/python"
    with (
        mock.patch.object(sys, "executable", spaced),
        mock.patch.object(sys, "orig_argv", [spaced, "-m", "flash.cli"]),
        mock.patch.object(sys, "argv", ["-m"]),
    ):
        resolved = importlib.reload(channel).CLI_NAME

    interpreter = resolved[: -len(" -m flash.cli")]
    # the whole point: split the printed command the way a shell would, and the path must survive
    assert shlex.split(resolved) == [spaced, "-m", "flash.cli"], resolved
    assert interpreter != spaced, "an unquoted spaced path splits into three shell words"

    # ...while an ordinary path is left alone rather than gratuitously quoted
    plain = "/usr/bin/python3"
    with (
        mock.patch.object(sys, "executable", plain),
        mock.patch.object(sys, "orig_argv", [plain, "-m", "flash.cli"]),
        mock.patch.object(sys, "argv", ["-m"]),
    ):
        unquoted = importlib.reload(channel).CLI_NAME
    assert unquoted == f"{plain} -m flash.cli"

    importlib.reload(channel)  # restore the pytest-invoked value for later tests


def test_windows_interpreter_paths_are_quoted_for_cmd_not_for_sh():
    """Windows quoting is not POSIX quoting, and using the POSIX form breaks the COMMON case.

    A backslash is outside `shlex.quote`'s safe set, so `C:\\Python311\\python.exe` -- a path with
    nothing in it that needs escaping -- comes back `'C:\\Python311\\python.exe'`. cmd.exe does not
    strip single quotes, so it searches for a program whose name begins with one and fails. That is
    the ordinary Windows install, regressed by the quoting added for the spaced one.

    `list2cmdline` is the platform's own answer: bare when it can be, double-quoted when a space
    forces it, which is the form both cmd.exe and PowerShell accept.
    """
    import sys
    from unittest import mock

    from flash._internal import channel

    # called directly rather than through `importlib.reload`: a reload re-executes the module and
    # would restore the real `_on_windows`, discarding the patch that makes this test Windows.
    for exe, expected in (
        # the common install: no space, so no quoting at all is the correct output
        (r"C:\Python311\python.exe", r"C:\Python311\python.exe"),
        (r"C:\Users\dev\.venv\Scripts\python.exe", r"C:\Users\dev\.venv\Scripts\python.exe"),
        # ...and the spaced ones the quoting exists for get cmd.exe's double quotes
        (r"C:\Program Files\Python\python.exe", '"C:\\Program Files\\Python\\python.exe"'),
        # parentheses are inert inside double quotes, so the most common 32-bit path still works
        (
            r"C:\Program Files (x86)\Python\python.exe",
            '"C:\\Program Files (x86)\\Python\\python.exe"',
        ),
    ):
        with (
            mock.patch.object(channel, "_on_windows", return_value=True),
            mock.patch.object(sys, "executable", exe),
            mock.patch.object(sys, "orig_argv", [exe, "-m", "flash.cli"]),
            mock.patch.object(sys, "argv", ["-m"]),
        ):
            resolved = channel._invoked_cli_name()
        assert resolved == f"{expected} -m flash.cli", exe
        assert "'" not in resolved, f"single quotes are not quoting to cmd.exe: {resolved}"


def test_cmd_metacharacters_in_an_interpreter_path_are_quoted_not_just_spaces():
    """`list2cmdline` quotes for the C runtime, which is a later stage than cmd.exe's own splitting.

    So `C:\\Tools&SDK\\python.exe` comes back bare -- correct argv, useless command line: cmd.exe
    has already treated `&` as a statement separator, ending the command at `C:\\Tools` and running
    `SDK\\python.exe -m flash.cli ...` as a second one. The operator sees an error from a command
    they did not type while the run they meant to cancel keeps billing.

    Double quotes make every one of these inert, so force them when whitespace alone would not.
    """
    import sys
    from unittest import mock

    from flash._internal import channel

    for exe in (
        r"C:\Tools&SDK\python.exe",  # `&` runs a second command
        r"C:\a|b\python.exe",  # `|` pipes into one
        r"C:\a^b\python.exe",  # `^` escapes the next character
        r"C:\Py(3.12)\python.exe",  # unspaced parens, which list2cmdline also leaves bare
    ):
        with (
            mock.patch.object(channel, "_on_windows", return_value=True),
            mock.patch.object(sys, "executable", exe),
            mock.patch.object(sys, "orig_argv", [exe, "-m", "flash.cli"]),
            mock.patch.object(sys, "argv", ["-m"]),
        ):
            resolved = channel._invoked_cli_name()
        assert resolved == f'"{exe}" -m flash.cli', exe


def test_the_attached_dash_m_form_is_recognised_as_a_module_launch():
    """`python -mflash.cli` is the same launch as `python -m flash.cli`, and must read the same.

    A short option may carry its value attached, and CPython accepts it; the only difference is
    that `sys.orig_argv` holds one token, `-mflash.cli`, instead of two. Matching the separated
    spelling alone sends this form to the `flash` fallback -- on the host where `flash` may be
    RunPod's console script, so the printed cancel exits 0 without cancelling. That is the exact
    failure this module exists to prevent, reached through a spelling rather than a bug.
    """
    import sys
    from unittest import mock

    from flash._internal import channel

    exe = sys.executable
    for orig_argv, argv in (
        (["python3", "-mflash.cli"], ["-m"]),
        (["python3", "-mflash.cli", "runs", "cancel", "abc"], ["-m", "runs", "cancel", "abc"]),
        (["python3", "-W", "ignore", "-mflash.cli"], ["-m"]),
    ):
        with (
            mock.patch.object(sys, "orig_argv", orig_argv),
            mock.patch.object(sys, "argv", argv),
        ):
            assert channel._invoked_cli_name() == "python3 -m flash.cli", orig_argv

    # ...and the attached form of a DIFFERENT module is still not us
    for orig_argv in (["python3", "-mpytest"], ["python3", "-mflash.cliX"]):
        with (
            mock.patch.object(sys, "orig_argv", orig_argv),
            mock.patch.object(sys, "argv", ["-m"]),
        ):
            assert channel._invoked_cli_name() == "flash", orig_argv

    assert exe  # the real executable is untouched by the patches above


def test_windows_console_script_names_match_case_insensitively():
    """Windows filenames are case-insensitive, so `FLASH-CLI.EXE` IS the script we installed.

    Rejecting it falls back to `flash` -- which on the `[server]` install may be RunPod's, so the
    operator is pointed at the shadowed name the alias exists to escape. `.EXE` also survives a
    lowercase-only suffix strip, so the name then fails the membership test for a second reason.

    POSIX must NOT fold: there `FLASH-CLI` is a different file, and echoing it back names a command
    the operator does not have.
    """
    import sys
    from unittest import mock

    from flash._internal import channel

    # bare filenames, not `C:\...` paths: `os.path.basename` is posixpath on the test host and does
    # not split on backslashes, so a full Windows path would test that accident instead of the
    # case-folding under test. Windows supplies the basename through `ntpath` at run time.
    for argv0, on_windows, expected in (
        ("FLASH-CLI.EXE", True, "flash-cli"),
        ("Flash-Cli.exe", True, "flash-cli"),
        ("FLASH.EXE", True, "flash"),
        ("flash-cli.exe", True, "flash-cli"),
        # on posix, case is identity: a differently-cased name is not our script
        ("/usr/local/bin/FLASH-CLI", False, "flash"),
        ("/usr/local/bin/flash-cli", False, "flash-cli"),
    ):
        with (
            mock.patch.object(channel, "_on_windows", return_value=on_windows),
            mock.patch.object(sys, "argv", [argv0]),
        ):
            assert channel._invoked_cli_name() == expected, argv0


def test_the_wordmark_does_not_follow_the_invoked_entry_point():
    """BRAND_NAME identifies the product; CLI_NAME says what to type. Only the latter varies.

    `flash version` prints `flash 1.2.3` -- a name and a version, not a command. Deriving it from
    the invoked entry point yields `python -m flash.cli 1.2.3`, which reads as a half-typed command
    and breaks the `flash <version>` contract tests/test_version.py pins. Reaching the same tool
    through a different script does not rename the tool.
    """
    import importlib
    import sys
    from unittest import mock

    from flash._internal import channel

    for orig_argv, argv in (
        ([sys.executable, "-m", "flash.cli", "version"], ["-m", "version"]),
        ([sys.executable, "/usr/local/bin/flash-cli", "version"], ["/usr/local/bin/flash-cli"]),
    ):
        with (
            mock.patch.object(sys, "orig_argv", orig_argv),
            mock.patch.object(sys, "argv", argv),
        ):
            reloaded = importlib.reload(channel)
            assert reloaded.BRAND_NAME == "flash", orig_argv

    importlib.reload(channel)  # restore the pytest-invoked value for later tests


def test_every_console_script_has_a_dev_rename():
    """A script key missing from SCRIPT_RENAMES ships unrenamed in the dev distribution.

    rewrite_pyproject only renames keys present in SCRIPT_RENAMES and passes everything else
    through, so a new console script added to pyproject without a matching entry would collide
    with the prod package on a side-by-side install.
    """
    build = _load_build_module()
    missing = sorted(set(_real_console_scripts()) - set(build.SCRIPT_RENAMES))
    assert not missing, (
        "console script(s) with no SCRIPT_RENAMES entry: "
        + ", ".join(missing)
        + ". scripts/build_dev_dist.py passes unknown keys through unrenamed, so the dev "
        "distribution would install a script that collides with freesolo-flash."
    )


def test_rewrite_pyproject_retargets_only_the_project_table():
    build = _load_build_module()
    src = (
        "[project]\n"
        'name = "freesolo-flash"\n'
        'version = "0.2.25"\n'
        "dependencies = []\n"
        "\n"
        "[project.scripts]\n"
        'flash = "flash.cli.main:main"\n'
        "# Operator-only console script.\n"
        'flash-cli = "flash.cli.main:main"\n'
        'flash-server = "flash.server.asgi.cli:main"\n'
        "\n"
        "[tool.flash-dev]\n"
        'version = "9.9.9"\n'
    )
    out = build.rewrite_pyproject(src, "9.9.9")

    # Package + scripts renamed for a side-by-side dev install.
    assert 'name = "freesolo-flash-dev"' in out
    assert 'flash-dev = "flash.cli.main:main"' in out
    assert 'flash-dev-cli = "flash.cli.main:main"' in out
    assert 'flash-dev-server = "flash.server.asgi.cli:main"' in out
    # No un-renamed script key survives: each would collide with the prod package.
    assert "\nflash = " not in out
    assert "\nflash-cli = " not in out
    # The console-script *values* (import targets) are untouched.
    assert '"flash.cli.main:main"' in out
    # The prod version is replaced by the dev version, and the identically-valued
    # [tool.flash-dev].version line is left alone (so both now read 9.9.9 -> two occurrences).
    assert '"0.2.25"' not in out
    assert out.count('version = "9.9.9"') == 2
    # The dist-name retarget did not bleed into the [tool.flash-dev] table header.
    assert "[tool.flash-dev]" in out


def test_read_dev_version():
    build = _load_build_module()
    src = '[project]\nname = "freesolo-flash"\nversion = "1.0.0"\n[tool.flash-dev]\nversion = "2.3.4"\n'
    assert build.read_dev_version(src) == "2.3.4"


def test_no_build_flips_channel_and_pyproject_together(tmp_path):
    """--no-build (publish-image.yml, before the `:dev` Docker build) must apply the FULL
    transform, not just the channel flip. flash/__init__.py resolves __version__ via
    importlib.metadata.version(DIST_NAME), and DIST_NAME derives from CHANNEL -- so a
    channel-only flip would point that lookup at a distribution name nothing installed,
    silently downgrading __version__ to "0+unknown" in the built image."""
    build = _load_build_module()
    (tmp_path / "flash" / "_internal").mkdir(parents=True)
    channel_path = tmp_path / "flash" / "_internal" / "channel.py"
    channel_path.write_text('CHANNEL = "prod"\n')
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        '[project]\nname = "freesolo-flash"\nversion = "1.0.0"\n'
        '[tool.flash-dev]\nversion = "1.0.0"\n'
    )

    rc = build.main(["--root", str(tmp_path), "--no-build"])

    assert rc == 0
    assert channel_path.read_text() == 'CHANNEL = "dev"\n'
    # The installed package name must move in lockstep with CHANNEL/DIST_NAME.
    assert 'name = "freesolo-flash-dev"' in pyproject_path.read_text()


def test_channels_are_at_the_same_version():
    # The prod (freesolo-flash) and dev (freesolo-flash-dev) channels must ship the same version;
    # .github/workflows/version-parity.yml enforces this in CI too.
    import tomllib

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert data["project"]["version"] == data["tool"]["flash-dev"]["version"]
