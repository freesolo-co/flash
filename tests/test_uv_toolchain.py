"""single-source contracts for the uv toolchain used by automation."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
TOOLCHAIN_PATH = ROOT / "uv-toolchain.json"
SETUP_ACTION_DIR = ROOT / ".github" / "actions" / "setup-uv"
SETUP_ACTION_PATH = SETUP_ACTION_DIR / "action.yml"
INSTALL_SCRIPT_PATH = SETUP_ACTION_DIR / "install.sh"
LOCAL_SETUP_UV = "./.github/actions/setup-uv"
EXPECTED_CONSUMERS = {
    ("bake-kernel-cache.yml", "bake"),
    ("ci.yml", "test"),
    ("ci.yml", "typecheck"),
    ("deploy-modal-dev.yml", "deploy"),
    ("publish-dev.yml", "publish-pypi-dev"),
    ("publish.yml", "publish-pypi"),
}
TOOLCHAIN_KEYS = {
    "version",
    "github_release_linux_x86_64_sha256",
}
SHELL_CONTROL_WORDS = {
    "!",
    "coproc",
    "do",
    "elif",
    "else",
    "if",
    "then",
    "until",
    "while",
}
SHELL_WRAPPERS = {"command", "env", "exec", "nice", "sudo", "time"}
SHELL_INTERPRETERS = {"bash", "dash", "ksh", "sh", "zsh"}
REDIRECTION_OPERATORS = {"<", ">", ">>", "<<<", "<>", ">|", ">&", "<&"}
DOWNLOAD_OPTIONS_WITH_VALUES = {
    "--output",
    "--output-document",
    "--output-file",
    "-O",
    "-o",
}
WRAPPER_OPTIONS_WITH_VALUES = {
    "env": {"--argv0", "--chdir", "--split-string", "--unset", "-C", "-S", "-u"},
    "exec": {"--argv0", "-a"},
    "nice": {"--adjustment", "-n"},
    "sudo": {
        "--chdir",
        "--close-from",
        "--group",
        "--host",
        "--other-user",
        "--prompt",
        "--role",
        "--type",
        "--user",
        "-C",
        "-D",
        "-g",
        "-h",
        "-p",
        "-r",
        "-t",
        "-u",
    },
    "time": {"--format", "--output", "-f", "-o"},
}
PYTHON_OPTIONS_WITH_VALUES = {"--check-hash-based-pycs", "-W", "-X"}
PIP_GLOBAL_OPTIONS_WITH_VALUES = {
    "--cache-dir",
    "--cert",
    "--client-cert",
    "--exists-action",
    "--log",
    "--proxy",
    # without this, `pip --python /usr/bin/python3 install uv` reads the interpreter path as the
    # subcommand and the install is never inspected.
    "--python",
    "--retries",
    "--timeout",
    "--trusted-host",
}
PIP_INSTALL_OPTIONS_WITH_VALUES = PIP_GLOBAL_OPTIONS_WITH_VALUES | {
    "--abi",
    "--config-settings",
    "--constraint",
    "--extra-index-url",
    "--find-links",
    "--global-option",
    "--group",
    "--implementation",
    "--index-url",
    "--install-option",
    "--platform",
    "--prefix",
    "--progress-bar",
    "--python-version",
    "--report",
    "--requirement",
    "--root",
    "--root-user-action",
    "--src",
    "--target",
    "--upgrade-strategy",
    "-C",
    "-c",
    "-f",
    "-i",
    "-r",
    "-t",
}
PIP_SHORT_OPTIONS_WITH_ATTACHED_VALUES = {"-C", "-c", "-f", "-i", "-r", "-t"}
# pipx takes its own install options. borrowing pip's table read `--suffix uv` as a package named
# uv, so `pipx install --suffix uv black` was rejected as a uv install.
PIPX_INSTALL_OPTIONS_WITH_VALUES = PIP_INSTALL_OPTIONS_WITH_VALUES | {
    "--fetch-missing-python",
    "--preinstall",
    "--pypackages",
    "--python",
    "--suffix",
    "--system-site-packages",
}
UV_PACKAGE = re.compile(
    # PEP 508 also allows a direct reference (`uv @ https://.../uv.whl`), which installs uv just as
    # surely as a version specifier does.
    r"uv(?:\[[^\]\s]+\])?(?:\s*@\s*\S+|(?:==|>=|<=|~=|!=|>|<).+)?",
    re.IGNORECASE,
)
PYTHON_EXECUTABLE = re.compile(r"python(?:3(?:\.\d+)?)?$")
PIP_EXECUTABLE = re.compile(r"pip(?:3(?:\.\d+)?)?$")
ANSI_C_ESCAPES = {
    "a": "\a",
    "b": "\b",
    "e": "\x1b",
    "E": "\x1b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
    "\\": "\\",
    "'": "'",
    '"': '"',
}


def _read_ansi_c_escape(source: str, index: int) -> tuple[str, int]:
    character = source[index]
    if character in ANSI_C_ESCAPES:
        return ANSI_C_ESCAPES[character], index + 1
    if character in "01234567":
        end = index + 1
        while end < min(index + 3, len(source)) and source[end] in "01234567":
            end += 1
        return chr(int(source[index:end], 8)), end
    widths = {"x": 2, "u": 4, "U": 8}
    if character in widths:
        end = min(index + 1 + widths[character], len(source))
        digits = source[index + 1 : end]
        if digits and all(digit in "0123456789abcdefABCDEF" for digit in digits):
            return chr(int(digits, 16)), end
    return character, index + 1


def _read_shell_word(source: str, index: int) -> tuple[str, int]:
    word: list[str] = []
    while index < len(source) and source[index] not in " \t\r\n;&|(){}<>":
        if source.startswith("$'", index):
            index += 2
            while index < len(source) and source[index] != "'":
                if source[index] == "\\" and index + 1 < len(source):
                    escaped, index = _read_ansi_c_escape(source, index + 1)
                    word.append(escaped)
                    continue
                word.append(source[index])
                index += 1
            index += int(index < len(source))
            continue
        character = source[index]
        if character in "'\"":
            quote = character
            index += 1
            while index < len(source) and source[index] != quote:
                if source[index] == "\\" and quote == '"' and index + 1 < len(source):
                    index += 1
                word.append(source[index])
                index += 1
            index += int(index < len(source))
            continue
        if character == "\\" and index + 1 < len(source):
            index += 1
            if source[index] == "\n":
                index += 1
                continue
        word.append(source[index])
        index += 1
    return "".join(word), index


def _command_substitutions(script: str) -> list[str]:
    """inner scripts of `$(...)` and backtick command substitutions.

    these execute as commands in their own right, so a consumer hidden inside one is a real
    consumer. nested `$(...)` is tracked by depth; backticks do not nest.
    """

    inner: list[str] = []
    index = 0
    while index < len(script):
        if script.startswith("$(", index):
            depth = 1
            cursor = index + 2
            while cursor < len(script) and depth:
                if script.startswith("$(", cursor):
                    depth += 1
                    cursor += 2
                    continue
                if script[cursor] == ")":
                    depth -= 1
                    if not depth:
                        break
                cursor += 1
            if not depth:
                inner.append(script[index + 2 : cursor])
                index = cursor + 1
                continue
        if script[index] == "`":
            end = script.find("`", index + 1)
            if end != -1:
                inner.append(script[index + 1 : end])
                index = end + 1
                continue
        index += 1
    return inner


def _shell_tokens(script: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    while index < len(script):
        character = script[index]
        if character == "\n":
            tokens.append("\n")
            index += 1
            continue
        if character in " \t\r":
            index += 1
            continue
        if character == "#":
            newline = script.find("\n", index)
            index = len(script) if newline == -1 else newline
            continue
        if character in ";&|(){}<>":
            end = index + 1
            while end < len(script) and script[end] == character:
                end += 1
            tokens.append(script[index:end])
            index = end
            continue
        word, index = _read_shell_word(script, index)
        if word:
            tokens.append(word)
    return tokens


def _heredoc_declarations(line: str) -> list[tuple[str, bool]]:
    declarations: list[tuple[str, bool]] = []
    index = 0
    quote: str | None = None
    while index < len(line):
        character = line[index]
        if quote is not None:
            if character == quote:
                quote = None
            elif character == "\\" and quote == '"':
                index += 1
            index += 1
            continue
        if character in "'\"":
            quote = character
            index += 1
            continue
        if character == "\\":
            index += 2
            continue
        if (
            not line.startswith("<<", index)
            or line.startswith("<<<", index)
            or (index > 0 and line[index - 1] == "<")
        ):
            index += 1
            continue

        index += 2
        strip_tabs = index < len(line) and line[index] == "-"
        index += int(strip_tabs)
        while index < len(line) and line[index] in " \t":
            index += 1
        delimiter, index = _read_shell_word(line, index)
        if delimiter:
            declarations.append((delimiter, strip_tabs))
    return declarations


def _basename(word: str) -> str:
    return word.rsplit("/", 1)[-1]


def _is_assignment(word: str) -> bool:
    return re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", word) is not None


def _option_consumes_next(
    option: str,
    options_with_values: set[str],
    short_options_with_attached_values: set[str] | None = None,
) -> bool:
    if "=" in option:
        return False
    if option in options_with_values:
        return True
    return not any(
        option.startswith(short) and option != short
        for short in short_options_with_attached_values or set()
    ) and option in (short_options_with_attached_values or set())


def _expand_env_split_strings(words: list[str]) -> list[str]:
    expanded = list(words)
    index = 0
    while index < len(expanded):
        if _basename(expanded[index]) != "env":
            index += 1
            continue
        option_index = index + 1
        while option_index < len(expanded):
            option = expanded[option_index]
            if option in {"-S", "--split-string"} and option_index + 1 < len(expanded):
                replacement = [
                    token for token in _shell_tokens(expanded[option_index + 1]) if token != "\n"
                ]
                expanded[option_index : option_index + 2] = replacement
                continue
            for prefix in ("--split-string=",):
                if option.startswith(prefix):
                    replacement = [
                        token
                        for token in _shell_tokens(option.removeprefix(prefix))
                        if token != "\n"
                    ]
                    expanded[option_index : option_index + 1] = replacement
                    break
            else:
                if option.startswith("-") or _is_assignment(option):
                    option_index += 1
                    if _option_consumes_next(option, WRAPPER_OPTIONS_WITH_VALUES["env"]):
                        option_index += 1
                    continue
            break
        index = max(index + 1, option_index)
    return expanded


def _unwrap_command(words: list[str]) -> int:
    index = 0
    while index < len(words) and words[index] in SHELL_CONTROL_WORDS:
        index += 1
    while index < len(words) and _is_assignment(words[index]):
        index += 1

    while index < len(words) and _basename(words[index]) in SHELL_WRAPPERS:
        wrapper = _basename(words[index])
        index += 1
        while index < len(words):
            word = words[index]
            if _is_assignment(word):
                index += 1
                continue
            if word == "--":
                index += 1
                break
            if not word.startswith("-") or word == "-":
                break
            consumes_next = _option_consumes_next(
                word,
                WRAPPER_OPTIONS_WITH_VALUES.get(wrapper, set()),
            )
            index += 1
            if consumes_next and index < len(words):
                index += 1
        while index < len(words) and _is_assignment(words[index]):
            index += 1
    return index


def _python_module_command(words: list[str], index: int) -> tuple[str | None, int]:
    index += 1
    while index < len(words):
        word = words[index]
        if word == "-m" and index + 1 < len(words):
            return words[index + 1], index + 2
        if word == "--" or not word.startswith("-"):
            return None, index
        consumes_next = _option_consumes_next(word, PYTHON_OPTIONS_WITH_VALUES)
        index += 1
        if consumes_next and index < len(words):
            index += 1
    return None, index


def _find_pip_subcommand(words: list[str], index: int) -> tuple[str | None, int]:
    while index < len(words):
        word = words[index]
        if word in {"install", "run"}:
            return word, index + 1
        if word == "--":
            return None, index
        if word.startswith("-"):
            consumes_next = _option_consumes_next(word, PIP_GLOBAL_OPTIONS_WITH_VALUES)
            index += 1
            if consumes_next and index < len(words):
                index += 1
            continue
        return None, index
    return None, index


def _pip_installs_uv(
    words: list[str],
    index: int,
    options_with_values: set[str] | None = None,
) -> bool:
    while index < len(words):
        word = words[index]
        if word == "--":
            return any(UV_PACKAGE.fullmatch(package) for package in words[index + 1 :])
        if word.startswith("-"):
            consumes_next = _option_consumes_next(
                word,
                options_with_values or PIP_INSTALL_OPTIONS_WITH_VALUES,
                PIP_SHORT_OPTIONS_WITH_ATTACHED_VALUES,
            )
            index += 1
            if consumes_next and index < len(words):
                index += 1
            continue
        if UV_PACKAGE.fullmatch(word):
            return True
        index += 1
    return False


def _download_urls(words: list[str], index: int) -> list[str]:
    """the fetch targets of a curl or wget command, excluding local output paths."""

    urls: list[str] = []
    cursor = index + 1
    while cursor < len(words):
        word = words[cursor]
        if word in DOWNLOAD_OPTIONS_WITH_VALUES:
            cursor += 2
            continue
        if word.startswith("-") and word != "-":
            # attached forms (`-o/tmp/x`, `--output=/tmp/x`) carry their value in the same word.
            cursor += 1
            continue
        urls.append(word)
        cursor += 1
    return urls


def _shell_interpreter_scripts(words: list[str], index: int) -> list[str]:
    """the script arguments a `sh -c`-style invocation will execute."""

    scripts: list[str] = []
    cursor = index + 1
    while cursor < len(words):
        word = words[cursor]
        if word.startswith("-c") and word != "-c":
            scripts.append(word[2:])
            cursor += 1
            continue
        if word == "-c" and cursor + 1 < len(words):
            scripts.append(words[cursor + 1])
            cursor += 2
            continue
        cursor += 1
    return scripts


def _command_uses_or_installs_uv(words: list[str]) -> bool:
    words = _expand_env_split_strings(words)
    index = _unwrap_command(words)
    if index >= len(words):
        return False

    executable = _basename(words[index])
    # `uvx` is uv's own tool runner, so it is the uv toolchain by another name.
    if executable in {"uv", "uvx"}:
        return True

    # `sh -c "..."` carries a whole script in one argument. treat that argument as a script rather
    # than an opaque word, or every consumer inside it is invisible to this contract.
    if executable in SHELL_INTERPRETERS:
        return any(
            _command_uses_or_installs_uv(command)
            for argument in _shell_interpreter_scripts(words, index)
            for command in _shell_commands(argument)
        )

    module: str | None = None
    arguments = index + 1
    if PYTHON_EXECUTABLE.fullmatch(executable):
        module, arguments = _python_module_command(words, index)
        if module not in {"pip", "pipx"}:
            return False
    elif PIP_EXECUTABLE.fullmatch(executable):
        module = "pip"
    elif executable == "pipx":
        module = "pipx"
    elif executable in {"curl", "wget"}:
        # only the URL being fetched decides this. matching the markers against every argument
        # rejected `curl --output /tmp/uv-installer ...`, where the uv-looking text is the local
        # destination and nothing uv is downloaded at all.
        return any(
            marker in argument
            for argument in _download_urls(words, index)
            for marker in (
                "astral.sh/uv",
                "astral-sh/uv",
                "/uv/releases/",
                "uv-installer",
                "uv-x86_64",
            )
        )
    else:
        return False

    subcommand, arguments = _find_pip_subcommand(words, arguments)
    if module == "pip":
        return subcommand == "install" and _pip_installs_uv(words, arguments)
    return subcommand in {"install", "run"} and _pip_installs_uv(
        words, arguments, PIPX_INSTALL_OPTIONS_WITH_VALUES
    )


def _strip_leading_redirections(words: list[str]) -> list[str]:
    """drop redirections written before the executable, e.g. `> log uv pip install x`.

    the tokenizer emits every redirection operator as its own token and splits an fd prefix off
    ahead of it, so `2>/tmp/err uv sync` arrives as ['2', '>', '/tmp/err', 'uv', 'sync']. without
    this the command's first word is the fd or the redirect target and the real executable is
    never examined.
    """

    index = 0
    while index < len(words):
        word = words[index]
        # an fd prefix only counts as one when a redirection operator follows it.
        if word.isdigit() and index + 1 < len(words) and words[index + 1] in REDIRECTION_OPERATORS:
            index += 1
            continue
        if word in REDIRECTION_OPERATORS:
            index += 2
            continue
        break
    return words[index:]


def _shell_commands(script: str) -> list[list[str]]:
    commands: list[list[str]] = []
    current: list[str] = []
    separators = {"\n", ";", ";;", "&", "&&", "|", "||", "(", ")", "{", "}"}
    for token in _shell_tokens(script):
        if token in separators:
            if current:
                commands.append(_strip_leading_redirections(current))
                current = []
            continue
        current.append(token)
    if current:
        commands.append(_strip_leading_redirections(current))
    for substitution in _command_substitutions(script):
        commands.extend(_shell_commands(substitution))
    return [command for command in commands if command]


def _workflow_documents() -> dict[str, dict[str, Any]]:
    paths = sorted(
        path for extension in ("*.yml", "*.yaml") for path in WORKFLOW_DIR.glob(extension)
    )
    assert paths
    return {path.name: yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths}


def _load_toolchain(source: str | None = None) -> dict[str, str]:
    document = json.loads(source if source is not None else TOOLCHAIN_PATH.read_text())
    assert set(document) == TOOLCHAIN_KEYS
    assert re.fullmatch(r"[0-9]+(?:\.[0-9]+){2}", document["version"])
    checksum = document["github_release_linux_x86_64_sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", checksum)
    assert checksum != "0" * 64
    return document


def _assert_setup_action_contract(source: str) -> None:
    action = yaml.safe_load(source)
    steps = action["runs"]["steps"]
    python_step = next(step for step in steps if step.get("name") == "Set up Python")
    installer = next(step for step in steps if step.get("id") == "toolchain")
    cache = next(step for step in steps if str(step.get("uses", "")).startswith("actions/cache@"))

    assert python_step["if"] == "inputs.python-version != ''"
    assert python_step["uses"] == ("actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97")
    assert python_step["with"]["python-version"] == "${{ inputs.python-version }}"

    assert installer["shell"] == "bash --noprofile --norc -euo pipefail {0}"
    assert installer["env"] == {
        "RUNNER_OS": "${{ runner.os }}",
        "RUNNER_ARCH": "${{ runner.arch }}",
        "TOOLCHAIN_PATH": "${{ github.action_path }}/../../../uv-toolchain.json",
    }
    assert installer["run"] == "${{ github.action_path }}/install.sh"

    assert cache["if"] == "inputs.enable-cache != 'false'"
    assert cache["uses"] == "actions/cache@0057852bfaa89a56745cba8c7296529d2fc39830"
    assert cache["with"]["path"] == "/dev/shm/flash-uv-cache"
    assert "steps.toolchain.outputs.version" in cache["with"]["key"]
    assert "hashFiles('**/uv.lock')" in cache["with"]["key"]

    assert "astral-sh/setup-uv@" not in source
    assert "download-from-astral-mirror" not in source
    for value in _load_toolchain().values():
        assert value not in source


def _assert_install_script_contract(source: str) -> None:
    assert "mktemp -d /dev/shm/" in source
    assert 'install_root=$(mktemp -d "/dev/shm/' in source
    assert "UV_CACHE_DIR=/dev/shm/flash-uv-cache" in source
    assert "https://github.com/astral-sh/uv/releases/download/" in source
    assert "curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location" in source
    assert "sha256sum --check --strict" in source
    assert source.index("sha256sum --check --strict") < source.index("tar --extract")
    assert source.index("tar --extract") < source.index('>> "$GITHUB_PATH"')
    assert 'checksum == "0" * 64' in source
    assert "/opt/hostedtoolcache" not in source
    assert "download-from-astral-mirror" not in source


def _without_heredoc_payloads(script: str) -> str:
    kept: list[str] = []
    pending: list[tuple[str, bool]] = []
    for line in script.splitlines():
        if pending:
            delimiter, strip_tabs = pending[0]
            candidate = line.lstrip("\t") if strip_tabs else line
            if candidate == delimiter:
                pending.pop(0)
            continue

        kept.append(line)
        pending.extend(_heredoc_declarations(line))

    assert not pending, f"unterminated heredoc delimiter {pending[0][0]!r}"
    return "\n".join(kept)


def _run_lines(step: dict[str, Any]) -> list[str]:
    script = _without_heredoc_payloads(str(step.get("run", "")))
    return [
        line.strip()
        for line in script.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _step_uses_or_installs_uv(step: dict[str, Any]) -> bool:
    script = "\n".join(_run_lines(step))
    return any(_command_uses_or_installs_uv(command) for command in _shell_commands(script))


def _assert_workflow_contracts(documents: dict[str, dict[str, Any]]) -> None:
    consumers: set[tuple[str, str]] = set()
    for workflow_name, document in documents.items():
        for job_name, job in document.get("jobs", {}).items():
            steps = job.get("steps", [])
            direct = [
                step
                for step in steps
                if str(step.get("uses", "")).startswith("astral-sh/setup-uv@")
            ]
            assert not direct, f"{workflow_name}:{job_name} bypasses the verified local action"

            installers = [
                (index, step)
                for index, step in enumerate(steps)
                if step.get("uses") == LOCAL_SETUP_UV
            ]
            uv_steps = [
                index for index, step in enumerate(steps) if _step_uses_or_installs_uv(step)
            ]
            if installers:
                consumers.add((workflow_name, job_name))
                assert str(job.get("runs-on", "")).startswith("ubuntu")
                assert len(installers) == 1
                assert any(
                    str(step.get("uses", "")).startswith("actions/checkout@")
                    for step in steps[: installers[0][0]]
                )
                if (workflow_name, job_name) == ("bake-kernel-cache.yml", "bake"):
                    assert any(
                        step.get("with", {}).get("ref")
                        == "${{ steps.base.outputs.revision || github.sha }}"
                        for step in steps[installers[0][0] + 1 :]
                    )
            if uv_steps:
                assert len(installers) == 1, (
                    f"{workflow_name}:{job_name} uses or installs uv without the verified local action"
                )
                assert installers[0][0] < min(uv_steps), (
                    f"{workflow_name}:{job_name} uses or installs uv before the verified local action"
                )

    assert consumers == EXPECTED_CONSUMERS


def _assert_docker_hash_enforcement(source: str) -> None:
    requirement = re.search(
        r"'uv==[0-9]+(?:\.[0-9]+){2} --hash=sha256:[0-9a-f]{64}'",
        source,
    )
    assert requirement is not None
    assert source.count("uv==") == 1
    assert "--require-hashes -r /tmp/uv-requirements.txt" in source
    assert source.index(requirement.group()) < source.index("--require-hashes")


def _make_uv_archive(root: Path, version: str) -> Path:
    payload = root / "payload" / "uv-x86_64-unknown-linux-gnu"
    payload.mkdir(parents=True)
    uv = payload / "uv"
    uv.write_text(f"#!/bin/sh\nprintf '%s\\n' 'uv {version}'\n", encoding="utf-8")
    uv.chmod(0o755)
    uvx = payload / "uvx"
    uvx.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    uvx.chmod(0o755)
    archive = root / "uv.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(payload, arcname=payload.name)
    return archive


def _run_installer(
    *,
    checksum: str | None,
    poison_path: bool,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    toolchain = _load_toolchain()
    state = Path(tempfile.mkdtemp(prefix="flash-uv-test-", dir="/dev/shm"))
    archive = _make_uv_archive(state, toolchain["version"])
    expected_checksum = checksum or hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = state / "uv-toolchain.json"
    manifest.write_text(
        json.dumps(
            {
                "version": toolchain["version"],
                "github_release_linux_x86_64_sha256": expected_checksum,
            }
        ),
        encoding="utf-8",
    )
    bin_dir = state / "bin"
    bin_dir.mkdir()
    curl_marker = state / "curl-called"
    fake_curl = bin_dir / "curl"
    fake_curl.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        'printf called > "$UV_TEST_CURL_MARKER"\n'
        "out=\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  case "$1" in\n'
        "    --output) out=$2; shift 2 ;;\n"
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
        'cp "$UV_TEST_ARCHIVE" "$out"\n',
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)
    poison_marker = state / "poison-called"
    tool_cache = state / "tool-cache"
    if poison_path:
        poison_script = (
            f"#!/bin/sh\nprintf called > {poison_marker}\n"
            f"printf '%s\\n' 'uv {toolchain['version']}'\n"
        )
        poison_uv = bin_dir / "uv"
        poison_uv.write_text(poison_script, encoding="utf-8")
        poison_uv.chmod(0o755)
        cached_uv = tool_cache / "uv" / toolchain["version"] / "x64" / "uv"
        cached_uv.parent.mkdir(parents=True)
        cached_uv.write_text(poison_script, encoding="utf-8")
        cached_uv.chmod(0o755)
        (cached_uv.parent.parent / "x64.complete").touch()

    github_path = state / "github-path"
    github_env = state / "github-env"
    github_output = state / "github-output"
    for path in (github_path, github_env, github_output):
        path.touch()
    completed = subprocess.run(
        ["bash", "--noprofile", "--norc", str(INSTALL_SCRIPT_PATH)],
        cwd=ROOT,
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "RUNNER_OS": "Linux",
            "RUNNER_ARCH": "X64",
            "TOOLCHAIN_PATH": str(manifest),
            "GITHUB_PATH": str(github_path),
            "GITHUB_ENV": str(github_env),
            "GITHUB_OUTPUT": str(github_output),
            "GITHUB_RUN_ID": "pytest",
            "GITHUB_RUN_ATTEMPT": "1",
            "RUNNER_TOOL_CACHE": str(tool_cache),
            "UV_TEST_ARCHIVE": str(archive),
            "UV_TEST_CURL_MARKER": str(curl_marker),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    return completed, state


def _cleanup_installer_state(state: Path) -> None:
    github_path = state / "github-path"
    if github_path.exists() and (install_root := github_path.read_text().strip()):
        shutil.rmtree(install_root, ignore_errors=True)
    shutil.rmtree(state, ignore_errors=True)


def test_uv_toolchain_declares_one_exact_version_and_release_hash() -> None:
    _load_toolchain()


def test_uv_toolchain_rejects_all_zero_checksum() -> None:
    document = _load_toolchain()
    document["github_release_linux_x86_64_sha256"] = "0" * 64
    with pytest.raises(AssertionError):
        _load_toolchain(json.dumps(document))


def test_setup_action_installs_from_the_verified_archive_and_caches_only_dependencies() -> None:
    _assert_setup_action_contract(SETUP_ACTION_PATH.read_text(encoding="utf-8"))
    _assert_install_script_contract(INSTALL_SCRIPT_PATH.read_text(encoding="utf-8"))


def test_installer_ignores_a_poisoned_preexisting_uv_and_downloads_every_run() -> None:
    toolchain = _load_toolchain()
    completed, state = _run_installer(checksum=None, poison_path=True)
    try:
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert (state / "curl-called").read_text() == "called"
        assert not (state / "poison-called").exists()
        install_root = (state / "github-path").read_text().strip()
        assert install_root.startswith("/dev/shm/flash-uv-")
        installed = subprocess.run(
            [str(Path(install_root) / "uv"), "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert installed.returncode == 0
        assert installed.stdout.strip() == f"uv {toolchain['version']}"
    finally:
        _cleanup_installer_state(state)


def test_installer_does_not_expose_uv_when_checksum_verification_fails() -> None:
    completed, state = _run_installer(checksum="1" * 64, poison_path=False)
    try:
        assert completed.returncode != 0
        assert (state / "curl-called").read_text() == "called"
        assert (state / "github-path").read_text() == ""
    finally:
        _cleanup_installer_state(state)


def test_installer_rejects_all_zero_checksum_before_download() -> None:
    completed, state = _run_installer(checksum="0" * 64, poison_path=False)
    try:
        assert completed.returncode != 0
        assert "must not be all zeroes" in completed.stderr
        assert not (state / "curl-called").exists()
        assert (state / "github-path").read_text() == ""
    finally:
        _cleanup_installer_state(state)


def test_every_current_ubuntu_uv_consumer_uses_the_verified_action() -> None:
    _assert_workflow_contracts(_workflow_documents())


def test_each_workflow_consumer_rejects_direct_setup_uv_sabotage() -> None:
    documents = _workflow_documents()
    for workflow_name, job_name in EXPECTED_CONSUMERS:
        sabotaged = copy.deepcopy(documents)
        step = next(
            step
            for step in sabotaged[workflow_name]["jobs"][job_name]["steps"]
            if step.get("uses") == LOCAL_SETUP_UV
        )
        step["uses"] = "astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d"
        step["with"] = {"version": "9.9.9"}

        with pytest.raises(AssertionError):
            _assert_workflow_contracts(sabotaged)


@pytest.mark.parametrize(
    "sabotage",
    [
        "uv sync --frozen",
        "FOO=bar uv run pytest",
        "env FOO=bar command uv build",
        "env --unset FOO command -- nice --adjustment 5 sudo -E uv build",
        "env -S 'sudo -u root command uv build'",
        "env --split-string='nice -n 5 uv build'",
        "env -u FOO /usr/bin/env --chdir /tmp command uv build",
        "/usr/bin/env FOO=bar uv build",
        "/usr/bin/command uv build",
        "/usr/bin/env FOO=bar /usr/bin/command uv build",
        'sudo -E env UV_CACHE_DIR="/dev/shm/cache dir" /usr/local/bin/uv --version',
        "exec uv sync",
        "nice -n 5 exec command uv sync",
        "/usr/local/bin/uv publish",
        '"uv" sync',
        '"/opt/tools/uv" sync',
        "if uv --version; then printf ready; fi",
        "if true; then uv --version; fi",
        "if false; then :; elif uv --version; then :; fi",
        "if false; then :; else uv --version; fi",
        "while uv --version; do break; done",
        "until uv --version; do break; done",
        "while false; do uv sync; done",
        "! uv --version",
        "{ uv --version; }",
        "coproc uv --version",
        "python -m pip install uv",
        '"/usr/bin/python3" -I -W ignore -X dev -m pip install uv',
        'python3 -m pip --disable-pip-version-check install "uv==0.12.5"',
        "python3 -m pip --proxy https://proxy.invalid install uv",
        "python3 -m pip install --index-url https://index.invalid/simple uv",
        "python3 -m pip install --no-cache-dir \\\n  uv==0.12.5",
        "/usr/bin/python3 -m pip install --no-cache-dir uv==0.12.5",
        "pip install uv>=0.12",
        "pip install uv\\[aws\\]==0.12.5",
        "pip install 'uv[aws]==0.12.5'",
        "pip install --target build 'uv[aws]'",
        "pip --proxy 'https://proxy.invalid/a;b|c&d' install uv",
        "pip install --index-url 'https://index.invalid/a;b|c&d' uv",
        "/usr/local/bin/pip3 install --upgrade uv",
        "pipx install uv",
        "python -m pipx install uv",
        "pipx run uv --version",
        "/usr/bin/python3 -m pipx run uv sync",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "env --unset HTTP_PROXY /usr/bin/curl -LsSf https://astral.sh/uv/install.sh | sh",
        "/usr/bin/curl -LsSf https://astral.sh/uv/install.sh | sh",
        "wget -qO- https://github.com/astral-sh/uv/releases/download/0.12.5/uv-installer.sh | bash",
        "/usr/bin/wget -qO- https://github.com/astral-sh/uv/releases/download/0.12.5/uv-installer.sh | bash",
    ],
)
def test_workflow_contract_rejects_uv_use_or_install_before_local_action(sabotage: str) -> None:
    documents = _workflow_documents()
    workflow_name, job_name = "ci.yml", "test"
    steps = documents[workflow_name]["jobs"][job_name]["steps"]
    installer_index = next(
        index for index, step in enumerate(steps) if step.get("uses") == LOCAL_SETUP_UV
    )
    steps.insert(installer_index, {"name": "sabotage", "run": sabotage})
    with pytest.raises(AssertionError):
        _assert_workflow_contracts(documents)


@pytest.mark.parametrize(
    "valid",
    [
        "pip install --target uv requests",
        "pip install -tuv requests",
        "pip install --config-settings 'package=uv;mode=build|safe' requests",
        "env --chdir uv printf '%s\\n' requests",
        "python -I script.py -m pip install uv",
        "printf '%s\\n' uv",
    ],
)
def test_workflow_contract_does_not_confuse_option_values_or_arguments_with_uv(
    valid: str,
) -> None:
    assert not _step_uses_or_installs_uv({"run": valid})


@pytest.mark.parametrize(
    "heredoc",
    [
        "cat <<'EOF'\nuv sync\n/usr/bin/curl https://astral.sh/uv/install.sh\nEOF",
        "cat <<E'OF'\npip install uv\nEOF",
        "cat <<$'EOF'\nexec uv sync\nEOF",
        "cat <<$'E\\x4fF'\nexec uv sync\nEOF",
        "cat <<-EOF\n\tpip install 'uv[aws]'\n\tpython -m pipx run uv\n\tEOF",
    ],
)
def test_workflow_contract_ignores_uv_text_in_heredoc_payloads(heredoc: str) -> None:
    documents = _workflow_documents()
    workflow_name, job_name = "ci.yml", "test"
    steps = documents[workflow_name]["jobs"][job_name]["steps"]
    installer_index = next(
        index for index, step in enumerate(steps) if step.get("uses") == LOCAL_SETUP_UV
    )
    steps.insert(installer_index, {"name": "heredoc", "run": heredoc})
    _assert_workflow_contracts(documents)


def test_workflows_do_not_duplicate_the_central_version_or_hash() -> None:
    sources = [
        path.read_text(encoding="utf-8")
        for extension in ("*.yml", "*.yaml")
        for path in WORKFLOW_DIR.glob(extension)
    ]
    action_sources = [
        SETUP_ACTION_PATH.read_text(encoding="utf-8"),
        INSTALL_SCRIPT_PATH.read_text(encoding="utf-8"),
    ]
    for value in _load_toolchain().values():
        assert all(value not in source for source in sources)
        assert all(value not in source for source in action_sources)


def test_serving_dockerfile_preserves_its_existing_hash_enforcement() -> None:
    _assert_docker_hash_enforcement((ROOT / "Dockerfile.serve").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "sabotage",
    [
        lambda source: source.replace("--require-hashes", "--no-deps"),
        lambda source: source.replace("--hash=sha256:", "--trusted-host="),
    ],
)
def test_serving_docker_contract_rejects_hash_bypass_sabotage(sabotage: Any) -> None:
    source = (ROOT / "Dockerfile.serve").read_text(encoding="utf-8")
    with pytest.raises(AssertionError):
        _assert_docker_hash_enforcement(sabotage(source))


@pytest.mark.parametrize(
    "bypass",
    [
        # uvx IS the uv toolchain, under uv's tool-runner name.
        "uvx ruff check .",
        # a script carried in one -c argument is still a script.
        'bash -c "uv pip install foo"',
        'sh -c"uv sync"',
        'zsh -c "uvx ruff"',
        'env FOO=1 bash -c "uv sync"',
        # command substitutions execute as commands in their own right.
        "echo `uv --version`",
        "echo $(echo $(uv --version))",
        # a redirection written before the executable hides it from a naive first-word check.
        "> /tmp/log uv pip install foo",
        "2>/tmp/err uv sync",
        # pip's --python takes a value; without that, its path was read as the subcommand.
        "python -m pip --python /usr/bin/python3 install uv",
        # PEP 508 direct references install uv just as surely as a version specifier does.
        "python -m pip install 'uv @ https://example.invalid/uv.whl'",
    ],
)
def test_workflow_contract_detects_uv_bypass_forms(bypass: str) -> None:
    """each of these installs or runs uv while evading a first-word executable check."""

    assert _step_uses_or_installs_uv({"run": bypass}), (
        f"{bypass!r} reached uv without tripping the single-source contract"
    )


@pytest.mark.parametrize(
    "valid",
    [
        # uv-looking text in a LOCAL destination path is not a uv download.
        "curl --output /tmp/uv-installer https://example.invalid/thing.sh",
        "wget -O /tmp/uv-x86_64 https://example.invalid/thing",
        # pipx --suffix takes a value; borrowing pip's table read it as a package named uv.
        "pipx install --suffix uv black",
        "pipx install --suffix=uv black",
        # near-misses that must stay quiet.
        "uvicorn app:main",
        "mkdir -p /opt/uv",
    ],
)
def test_workflow_contract_does_not_reject_uv_shaped_text_that_never_installs_uv(
    valid: str,
) -> None:
    assert not _step_uses_or_installs_uv({"run": valid}), (
        f"{valid!r} was rejected even though it neither runs nor installs uv"
    )


@pytest.mark.parametrize(
    "download",
    [
        "curl -o /tmp/x https://astral.sh/uv/install.sh",
        "wget -O /tmp/x https://astral.sh/uv/install.sh",
    ],
)
def test_workflow_contract_still_detects_real_uv_downloads_with_output_paths(
    download: str,
) -> None:
    """narrowing curl/wget to the fetched URL must not blind it to a real uv download."""

    assert _step_uses_or_installs_uv({"run": download}), (
        f"{download!r} downloads uv but was not detected"
    )
