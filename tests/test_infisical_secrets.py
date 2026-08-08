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


def _installer_files() -> tuple[Path, ...]:
    """Every file that may install software from the network.

    DISCOVERED, not enumerated: a hand-kept list silently stops covering the next Dockerfile
    someone adds, and a guard that quietly narrows its own scope reports the same green as one
    that found nothing. `docker/Dockerfile.kernelcache*` were both missed by an enumerated list
    that claimed to be repo-wide.

    A function rather than a module constant so the coverage tests can re-run the real discovery
    against a planted file. Asserting instead on a copy of these patterns would test the copy.
    """
    workflows = REPO_ROOT / ".github" / "workflows"
    return (
        *sorted(REPO_ROOT.glob("Dockerfile*")),
        *sorted(REPO_ROOT.glob("docker/Dockerfile*")),
        # Both extensions: GitHub Actions accepts `.yaml` too, and a workflow renamed to it would
        # drop out of a `*.yml`-only scan silently.
        *sorted(p for ext in ("*.yml", "*.yaml") for p in workflows.glob(ext)),
    )


# Resolved once for parametrization; the tests below call `_installer_files()` directly when they
# need discovery re-run.
INSTALLER_FILES = _installer_files()

# `curl … | bash`, `wget … | sh`, and friends: a fetch whose output is piped into a shell.
#
# This was a single regex through eleven rounds of review, and each fix created the next hole:
# widening the gap before the pipe made `||` fallbacks match, and adding flag operands made
# `env -u bash cat` match at the OPERAND. That is not bad luck. Deciding which token is the
# command requires knowing how many operands each flag consumes, and a regex cannot carry that
# state -- so its optional groups backtrack and re-read an operand as the command. Walking the
# tokens models the actual grammar and makes both failure directions expressible.
_SHELL_NAME = re.compile(r"(?:[\w./-]*/)?(?:ba|z|k|da)?sh$")

# Wrappers that EXEC what follows them, so the real command is further right. Open-ended by
# nature; this is the part of the guard most likely to need extending, and the only part where
# a miss is a silent under-match rather than a parse error.
_EXEC_WRAPPERS = {"sudo", "env", "xargs", "nohup", "exec", "command", "timeout", "stdbuf"}

# Flags that take a SEPARATE operand, per wrapper. Keyed by wrapper because the same spelling
# means different things: `env -u NAME` unsets a variable, `sudo -u USER` picks a user. Knowing
# the arity is what stops the operand from being mistaken for the command -- `env -u bash cat`
# runs `cat`, not `bash`, and flagging it would fail CI on a legitimate pipeline.
_FLAGS_WITH_OPERAND = {
    "sudo": {"-u", "-g", "-h", "-p", "-C", "-U", "-r", "-t", "--user", "--group", "--prompt"},
    "env": {"-u", "-C", "-S", "--unset", "--chdir", "--split-string"},
    "timeout": {"-k", "-s", "--kill-after", "--signal"},
    "xargs": {"-a", "-E", "-I", "-L", "-n", "-P", "-s", "-d", "--delimiter", "--max-args"},
    "stdbuf": {"-i", "-o", "-e", "--input", "--output", "--error"},
}

# A bare numeric operand sitting between a wrapper and its command: `timeout 30 sh` requires one,
# and an unrecognized flag may introduce one (`nohup -n 5 bash`). Skipped after ANY wrapper rather
# than special-cased to `timeout`, because nothing is ever *named* `30` -- so treating a bare
# numeral as an operand cannot swallow a real command, while guessing its arity wrong the other
# way silently drops the guard.
_NUMERIC_OPERAND = re.compile(r"\d+(?:\.\d+)?[smhd]?")


def _piped_into_a_shell(line: str) -> bool:
    """True when a `curl`/`wget` fetch has its output piped into a shell.

    The rule is "do not execute a download", not "do not use a pipe": `curl … | jq`,
    `curl … | tar -xz`, and `curl … | grep bash` are all fine, and `curl … || bash recover.sh`
    is a fallback on FAILURE with nothing piped into it.
    """
    return any(_stage_runs_a_shell(s) for s in _pipeline_stages_after_a_fetch(line))


# Words and shell operators. Quotes are STRIPPED rather than honoured, so `sh -c "curl … | bash"`
# is still seen as a pipe into a shell: the quoting is the outer command's, and the inner text is
# a real pipeline that really runs. A `>&`-style redirection is matched first so its `&` is not
# taken for a separator.
_TOKEN = re.compile(r"\d*[<>]{1,2}&\d*|\|\||&&|\||;|&|[^\s;|&]+")
_FETCH = re.compile(r"(?:[\w./-]*/)?(?:curl|wget)")


def _tokenize(line: str) -> list[str]:
    return _TOKEN.findall(line.replace('"', " ").replace("'", " "))


def _pipeline_stages_after_a_fetch(line: str) -> list[list[str]]:
    """Every stage that a curl/wget's output flows into, as token lists.

    All later stages are returned, not just the next one: a pipeline may pass through
    intermediate stages first (`curl … | tee /tmp/i.sh | bash`), and looking only at the stage
    directly after the fetch was blind to every such spelling.

    `||` is NOT a stage separator, and neither are `&&`, `;`, or `&`. They end the pipeline: what
    follows is a separate command with nothing piped into it. Reading the second bar of a `||` as
    a pipe made legitimate `curl … || bash recover.sh` fallbacks match.
    """
    pipelines: list[list[list[str]]] = [[[]]]
    for tok in _tokenize(line):
        if tok in ("||", "&&", ";", "&"):
            pipelines.append([[]])
        elif tok == "|":
            pipelines[-1].append([])
        else:
            pipelines[-1][-1].append(tok)

    downstream: list[list[str]] = []
    for stages in pipelines:
        fetched = next((i for i, s in enumerate(stages) if any(_FETCH.fullmatch(t) for t in s)), -1)
        if fetched >= 0:
            downstream.extend(stages[fetched + 1 :])
    return downstream


def _stage_runs_a_shell(tokens: list[str]) -> bool:
    """Walk one pipeline stage's tokens to the command word, and say whether it is a shell.

    Assignments and redirections may PRECEDE the command (`MODE=install bash`, `2>/dev/null sh`)
    -- both spellings run the downloaded stream, and the second needs no wrapper at all. Exec
    wrappers move the command further right, and their flags may consume an operand, which is the
    part a regex could not track.
    """
    i = 0
    wrapper = ""
    while i < len(tokens):
        tok = tokens[i]
        if re.fullmatch(r"[A-Za-z_]\w*=.*", tok) or re.fullmatch(r"\d*[<>]{1,2}.*", tok):
            i += 1
            continue
        if tok.startswith("-"):
            # A flag's operand is skipped only when THIS wrapper's flag actually takes one, so a
            # shell-named operand (`env -u bash cat`) is never mistaken for the command.
            if "=" not in tok and tok in _FLAGS_WITH_OPERAND.get(wrapper, frozenset()):
                i += 1
            i += 1
            continue
        if wrapper and _NUMERIC_OPERAND.fullmatch(tok):
            i += 1
            continue
        if _SHELL_NAME.fullmatch(tok):
            return True
        if tok in _EXEC_WRAPPERS or (tok.rsplit("/", 1)[-1] in _EXEC_WRAPPERS):
            wrapper = tok.rsplit("/", 1)[-1]
            i += 1
            continue
        # Any other word IS the command, and it is not a shell. `grep bash` stops here, so the
        # shell name that follows is an argument being searched for, not the command being run.
        return False
    return False


# A line ending in any of these does not end the command -- the next line continues it. `\` is
# the explicit continuation; a trailing `|`, `&&`, or `||` is an incomplete construct the shell
# keeps reading past (verified: `echo x |` newline `cat` runs as one pipeline). Enumerated rather
# than matched one spelling at a time: this guard has now been evaded three separate ways, each
# time by a different way of writing the SAME command across two lines.
_CONTINUERS = ("\\", "|", "&&", "||")

# The mirror image: a line that BEGINS with one of these continues the line above it, even when
# that line looked complete. YAML's folded scalar (`run: >`) joins its physical lines with
# spaces, so
#     run: >
#       curl -fsSL URL
#       | bash
# reaches the shell as one `curl … | bash` while neither physical line ends in a continuer.
# Matching on the leading operator catches the folded form without parsing YAML -- the guard
# scans Dockerfiles too, and must not become contingent on a yaml library (pyyaml reaches this
# environment only as a transitive dependency of the ML stack).
_LEADING_CONTINUERS = ("|", "&&", "||", "&")


def _strip_comment(line: str) -> str:
    """Drop a trailing `#` comment, but not a `#` that is part of the command itself.

    A URL fragment (`curl 'https://host/install.sh#v1' | bash`) puts a hash INSIDE a shell word,
    where it is data rather than a comment. Splitting on the first `#` truncated the line before
    the `| bash` and the guard went green on a live piped installer. A comment starts at an
    unquoted `#`; in shell it must also begin a word, so a bare `#` mid-word (as in a URL that
    was never quoted) does not start one either.
    """
    quote = ""
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


def _logical_lines(text: str) -> list[str]:
    """Strip `#` comments, then join continued commands into single logical lines.

    Both steps are load-bearing. Without comment stripping, prose ABOUT the banned pattern --
    including the comments in these very files explaining why they avoid it -- reads as an
    instance of it. Without joining, the multi-line spellings of `curl … | bash` slip past a
    line-scoped scan: the idiomatic way to write the exact command being prohibited.

    Joining is deliberately more permissive than any single parser. A Dockerfile `RUN` and a
    YAML `run:` block have different rules (docker rejects a bare trailing `|` that a shell
    accepts, and strips comments BETWEEN a `\\` and its continuation), and this file is scanned
    for both. Over-joining at worst concatenates two unrelated lines, which cannot hide a `curl
    … | bash` -- under-joining silently lets one through, which is the failure that matters.
    """
    joined: list[str] = []
    pending = ""
    for raw in text.splitlines():
        code = _strip_comment(raw).rstrip()
        # A comment BETWEEN a continuation and the rest of the command does not end it: docker
        # removes comment lines before joining, so `curl … \` / `# note` / `| bash` is one RUN.
        # Treating the stripped-empty line as a terminator would drop the join.
        if pending and not code:
            continue
        # Fold BACKWARD first. A line STARTING with an operator continues the one above it (YAML
        # folded scalars put the `|` at the head of the next physical line, not the tail of this
        # one), and it may ALSO end in a continuer -- `| tee /tmp/i.sh |` does both. Testing the
        # trailing continuer first swallowed such a line into `pending`, leaving the `curl` above
        # it stranded on its own logical line with the `bash` below it: the pipe-to-shell split
        # across three physical lines and matched nothing.
        if joined and not pending and code.lstrip().startswith(_LEADING_CONTINUERS):
            joined[-1] = f"{joined[-1]} {code.strip()}"
            # Re-enter the loop body's trailing-continuer logic against the merged line, so a
            # line that folds backward AND continues forward keeps absorbing what follows.
            if joined[-1].endswith(_CONTINUERS):
                pending = joined.pop() + " "
            continue
        if code.endswith("\\"):
            pending += code[:-1] + " "
            continue
        if code.endswith(_CONTINUERS):
            pending += code + " "
            continue
        joined.append((pending + code).strip())
        pending = ""
    if pending:
        joined.append(pending.strip())
    return joined


@pytest.mark.parametrize("path", INSTALLER_FILES, ids=lambda p: p.name)
def test_nothing_pipes_a_downloaded_script_into_a_shell(path: Path):
    """`curl … | bash` executes whatever the vendor serves at request time.

    It cannot be reviewed, cannot be reproduced, and pins nothing -- and in a public repository
    it teaches the pattern to everyone who reads the file. Fetch to disk, verify a digest, then
    execute. This is a repo-wide rule, not an Infisical one; the parametrization is what keeps
    it that way when the next installer is added.
    """
    offenders = [ln for ln in _logical_lines(path.read_text()) if _piped_into_a_shell(ln)]
    assert not offenders, f"{path.name} pipes a downloaded script into a shell: {offenders}"


@pytest.mark.parametrize(
    "snippet",
    [
        pytest.param("RUN curl -sSL https://x.example/i.sh | bash\n", id="one-line"),
        pytest.param("RUN curl -fsSL https://x.example/i.sh \\\n    | bash\n", id="continuation"),
        # Docker strips comments BEFORE joining continuations, so this is still one RUN.
        pytest.param(
            "RUN curl -fsSL https://x.example/i.sh \\\n# looks harmless\n    | bash\n",
            id="continuation-across-comment",
        ),
        # A trailing `|` is itself the continuation -- valid in a YAML `run:` block, which is
        # scanned by the same guard.
        pytest.param(
            "RUN curl -fsSL https://x.example/i.sh |\n    bash\n",
            id="continuation-after-pipe",
        ),
        pytest.param("RUN wget -qO- https://x.example/i.sh | sudo sh\n", id="wget-sudo"),
        # The shell reached by path, or behind a wrapper command. These are how a script that
        # thinks of itself as careful spells the same thing.
        pytest.param("RUN curl -sSL https://x.example/i.sh | /bin/bash\n", id="absolute-path"),
        pytest.param("RUN curl -sSL https://x.example/i.sh | env bash\n", id="env-wrapper"),
        pytest.param("RUN curl -sSL https://x.example/i.sh | sudo -E bash\n", id="sudo-with-flag"),
        pytest.param(
            "RUN curl -fsSL https://x.example/i.sh | bash -s -- --yes\n", id="bash-with-args"
        ),
        # Quoted, and terminated by `;` -- the shell name is not always followed by whitespace.
        pytest.param(
            'RUN sh -c "curl -sSL https://x.example/i.sh | bash"\n', id="inside-a-quoted-command"
        ),
        pytest.param("RUN curl -sSL https://x.example/i.sh | bash;\n", id="semicolon-terminated"),
        # The shell need not be the FIRST stage after the fetch.
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | tee /tmp/i.sh | bash\n", id="multi-stage-pipe"
        ),
        # A `#` inside the URL is a fragment, not a comment -- splitting on it truncated the
        # line before the `| bash`.
        pytest.param(
            "RUN curl 'https://x.example/install.sh#v1' | bash\n", id="hash-in-a-quoted-url"
        ),
        # Assignments and redirections may precede the command word. Both spellings run the
        # downloaded stream; the second needs no wrapper command at all.
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | env MODE=install bash\n", id="env-assignment"
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | MODE=install sh\n", id="bare-assignment"
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | 2>/dev/null bash\n", id="redirection-first"
        ),
        # A flag with an operand: consuming `-u` but not `root` left an unmatched word.
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | sudo -u root bash\n", id="flag-with-operand"
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | env -C /tmp sh\n", id="env-flag-with-operand"
        ),
        # `timeout` takes a bare duration before the command, so recognizing the wrapper name
        # alone is not enough -- the duration read as the command and the guard went green.
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | timeout 30 sh\n", id="timeout-wrapper"
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | timeout -k 5 30 bash\n", id="timeout-with-flag"
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | nohup -n 5 bash\n", id="numeric-flag-operand"
        ),
        # Arity is PER-WRAPPER, not global. `-s` takes an operand for `timeout` (a signal) but
        # not for `sudo`, so a wrapper-blind flag list skips the word after it and walks straight
        # past the shell -- a silent under-match. This pairing is what distinguishes the two
        # designs; the `env -u bash` cases below cannot, since `-u` takes an operand either way.
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | sudo -s bash\n", id="flag-arity-is-per-wrapper"
        ),
    ],
)
def test_the_pipe_to_shell_guard_catches_what_it_claims_to(snippet: str, tmp_path: Path):
    """The guard above is only worth having if it fails on the thing it prohibits.

    A scanner that silently matches nothing reports the same green as a clean repository. Every
    spelling here evaded an earlier version of this guard, so they are pinned rather than
    assumed: each one is the same prohibited command, written a different way.
    """
    planted = tmp_path / "Dockerfile"
    planted.write_text("FROM scratch\n" + snippet)

    with pytest.raises(AssertionError, match="pipes a downloaded script into a shell"):
        test_nothing_pipes_a_downloaded_script_into_a_shell(planted)


@pytest.mark.parametrize(
    "block",
    [
        # `>` folds its physical lines with spaces, so the shell receives one `curl … | bash`
        # even though neither line ends in a continuer.
        pytest.param(">", id="folded-scalar"),
        pytest.param("|", id="literal-scalar"),
    ],
)
def test_the_guard_catches_a_piped_installer_written_as_a_yaml_block(block: str, tmp_path: Path):
    """Workflows are scanned too, and YAML has its own way of splitting one command over lines.

    A `run:` block scalar puts the pipe at the HEAD of the next physical line rather than the
    tail of the previous one, which is invisible to a scan that only looks for trailing
    continuers. Verified against pyyaml that `>` really does fold these into a single command.
    """
    planted = tmp_path / "wf.yml"
    planted.write_text(
        "jobs:\n  j:\n    steps:\n      - run: "
        + block
        + "\n          curl -fsSL https://x.example/i.sh\n          | bash\n"
    )

    with pytest.raises(AssertionError, match="pipes a downloaded script into a shell"):
        test_nothing_pipes_a_downloaded_script_into_a_shell(planted)


def test_the_guard_catches_a_yaml_line_that_both_folds_back_and_continues_on(tmp_path: Path):
    """`| tee /tmp/i.sh |` starts with an operator AND ends with one.

    It has to fold backward onto the `curl` above it and keep absorbing the `bash` below it. An
    earlier ordering tested the trailing continuer first, so the line was swallowed forward and
    the fetch was left stranded on its own logical line: the pipe-to-shell split across three
    physical lines and matched nothing.
    """
    planted = tmp_path / "wf.yml"
    planted.write_text(
        "jobs:\n  j:\n    steps:\n      - run: >\n"
        "          curl -fsSL https://x.example/i.sh\n"
        "          | tee /tmp/i.sh |\n"
        "          bash\n"
    )

    with pytest.raises(AssertionError, match="pipes a downloaded script into a shell"):
        test_nothing_pipes_a_downloaded_script_into_a_shell(planted)


@pytest.mark.parametrize(
    "line",
    [
        pytest.param('echo "${sha}  ${deb}" | sha256sum -c -', id="verify-a-local-file"),
        pytest.param("curl -sSL https://x.example/c.tgz | sudo tar -xz -C /usr/bin", id="into-tar"),
        pytest.param("curl -s https://x.example/api | jq .version", id="into-jq"),
        pytest.param("curl -s https://x.example/list | grep bash", id="shell-name-as-argument"),
        pytest.param("curl -s https://x.example/l | grep bash-completion", id="name-is-a-prefix"),
        # The assignment rule must not turn an assignment-shaped ARGUMENT into a match: here the
        # shell name is still just text being searched for, not the command being run.
        pytest.param("curl -s https://x.example/l | grep mode=install bash", id="assignment-arg"),
        # `||` is a fallback, not a pipe: the shell runs on FAILURE of the fetch and nothing is
        # piped into it. Reading the second bar of a `||` as the pipe made this legitimate
        # recovery branch match -- an over-match introduced by the multi-stage-pipe fix.
        pytest.param(
            "curl -fsSL https://x.example/f || bash recover.sh", id="or-fallback-to-a-shell"
        ),
        pytest.param("wget -q https://x.example/f || sh -c 'exit 1'", id="or-fallback-to-sh"),
        # A flag's OPERAND that happens to be named like a shell. `env -u bash cat` unsets the
        # variable `bash` and runs `cat`; flagging it would fail CI on a legitimate pipeline.
        # Knowing each flag's arity is what tells the operand apart from the command.
        pytest.param("curl -s https://x.example/f | env -u bash cat", id="shell-named-operand"),
        pytest.param("curl -s https://x.example/f | sudo -u bash whoami", id="shell-named-user"),
        # `&&` and `;` end the pipeline: the shell that follows gets nothing piped into it.
        pytest.param("curl -s https://x.example/f > /tmp/f && bash /tmp/o.sh", id="and-then-shell"),
        pytest.param("curl -s https://x.example/f ; bash unrelated.sh", id="semicolon-then-shell"),
        # A wrapper in front of something that is not a shell is still not a shell.
        pytest.param("curl -s https://x.example/f | timeout 30 jq .", id="timeout-into-jq"),
        # The command name merely STARTS with the shell name.
        pytest.param("curl -s https://x.example/f | sudo -u shane cat", id="operand-shell-prefix"),
    ],
)
def test_the_pipe_to_shell_guard_does_not_flag_ordinary_pipelines(line: str, tmp_path: Path):
    """The rule is "do not execute a download", not "do not use a pipe".

    A guard that fires on every pipeline gets suppressed rather than fixed, so the non-shell
    destinations are pinned too. `grep bash` is the sharp one: the shell name appears as an
    ARGUMENT being searched for, not as the command being run, and an earlier draft of the
    wrapper handling matched it.
    """
    planted = tmp_path / "Dockerfile"
    planted.write_text(f"FROM scratch\nRUN {line}\n")

    test_nothing_pipes_a_downloaded_script_into_a_shell(planted)


def test_the_installer_scan_covers_every_dockerfile_in_the_repo():
    """The scan discovers its own inputs, so a new Dockerfile cannot silently escape it.

    An enumerated list looks identical to a complete one right up until someone adds a file:
    `docker/Dockerfile.kernelcache` and its relayer were both outside a list whose own docstring
    called the rule repo-wide. Deriving the set from the tree is what makes the claim true.
    """
    on_disk = {p.resolve() for p in REPO_ROOT.rglob("Dockerfile*") if ".git" not in p.parts}
    scanned = {p.resolve() for p in INSTALLER_FILES}
    missed = on_disk - scanned
    assert not missed, f"Dockerfiles not covered by the installer scan: {sorted(missed)}"


def test_the_installer_scan_covers_workflows_under_both_yaml_extensions():
    """GitHub Actions runs `.yaml` as readily as `.yml`, so the scan must accept both.

    There is no `.yaml` workflow today, which is exactly why this is worth pinning: a scan keyed
    to one extension goes green forever, and the day a workflow is added or renamed under the
    other one it silently leaves coverage. So plant one and re-run the REAL discovery. An earlier
    version of this test rebuilt the glob patterns inline and asserted against that copy, which
    passed unchanged when the production glob was narrowed back to `*.yml` -- it was testing its
    own literal, not the scan.
    """
    workflows = REPO_ROOT / ".github" / "workflows"
    on_disk = {p.resolve() for p in workflows.iterdir() if p.suffix in (".yml", ".yaml")}
    assert not on_disk - {p.resolve() for p in INSTALLER_FILES}, "a workflow is outside the scan"

    planted = workflows / "zz-extension-probe.yaml"
    planted.write_text("name: probe\non: workflow_dispatch\njobs: {}\n")
    try:
        assert planted.resolve() in {p.resolve() for p in _installer_files()}, (
            "a .yaml workflow would not be discovered"
        )
    finally:
        planted.unlink()


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
    assert "sha256sum -c -" in text, "the download must be verified before it is installed"


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
    assert "INSTALL_INFISICAL=true" in PUBLISH_WORKFLOW.read_text()


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

    # Collect the `- "..."` entries under `paths:` up to the next key at the same indent. Done by
    # hand rather than with a yaml parser: pyyaml reaches this environment only as a transitive
    # dependency of datasets/transformers, so importing it here would make a workflow test quietly
    # contingent on the ML stack staying installed.
    watched: list[str] = []
    in_paths = False
    for line in PUBLISH_WORKFLOW.read_text().splitlines():
        stripped = line.strip()
        if stripped == "paths:":
            in_paths = True
            continue
        if in_paths:
            if stripped.startswith("- "):
                watched.append(stripped[2:].strip().strip("\"'"))
            elif stripped and not stripped.startswith("#"):
                break
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
