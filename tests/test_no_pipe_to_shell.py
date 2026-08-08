"""No file in this repo may pipe a downloaded script into a shell.

`curl … | bash` executes whatever the vendor serves at request time: it cannot be reviewed,
cannot be reproduced, and pins nothing. This module scans every file that may install software
from the network and models enough shell grammar to tell a real pipe-to-shell from an ordinary
pipeline, so the rule can be enforced without failing CI on `curl … | jq`.

The scan discovers its own inputs. An enumerated list looks identical to a complete one right up
until someone adds a file.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


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
# `ash` is BusyBox's shell and therefore `/bin/sh` on every Alpine image, which is the base a
# vendor install script is most likely to run under.
_SHELL_NAME = re.compile(r"(?:[\w./-]*/)?(?:ba|a|z|k|da)?sh$")

# Wrappers that EXEC what follows them, so the real command is further right. Open-ended by
# nature; this is the part of the guard most likely to need extending, and the only part where
# a miss is a silent under-match rather than a parse error.
_EXEC_WRAPPERS = {
    "sudo",
    "env",
    "xargs",
    "nohup",
    "exec",
    "command",
    "timeout",
    "stdbuf",
    # `busybox ash` runs ash through BusyBox's applet dispatch. Unlike the others this one also
    # dispatches non-shells (`busybox wget`), so it only matters that the walk keeps GOING -- the
    # applet after it is judged on its own name, exactly as any other command word would be.
    "busybox",
}

# Flags that take a SEPARATE operand, per wrapper. Keyed by wrapper because the same spelling
# means different things: `env -u NAME` unsets a variable, `sudo -u USER` picks a user. Knowing
# the arity is what stops the operand from being mistaken for the command -- `env -u bash cat`
# runs `cat`, not `bash`, and flagging it would fail CI on a legitimate pipeline.
_FLAGS_WITH_OPERAND = {
    "sudo": {"-u", "-g", "-h", "-p", "-C", "-U", "-r", "-t", "--user", "--group", "--prompt"},
    "env": {"-u", "-C", "--unset", "--chdir"},
    "timeout": {"-k", "-s", "--kill-after", "--signal"},
    "xargs": {"-a", "-E", "-I", "-L", "-n", "-P", "-s", "-d", "--delimiter", "--max-args"},
    "stdbuf": {"-i", "-o", "-e", "--input", "--output", "--error"},
}

# `env -S 'bash -s'` splits its operand into words and runs them, so the operand is not a value
# to skip -- it is the command. Kept apart from _FLAGS_WITH_OPERAND because those two treatments
# are opposites: one steps over the operand, this one steps INTO it.
_SPLITS_ITS_OPERAND = {"-S", "--split-string"}

# The same flag with its operand fused into the token: `-S'bash -s'`, `--split-string='bash -s'`.
_FUSED_SPLIT_STRING = re.compile(r"(?:-S|--split-string=)(?P<command>.+)")

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

    The file's own keyword is dropped first. `RUN`/`run:` introduces a shell command but is not
    part of one, and a walk that reads it as the command word stops before reaching the shell.
    """
    return any(_stage_runs_a_shell(s) for s in _pipeline_stages_after_a_fetch(_shell_part(line)))


# What the file writes BEFORE the shell command: a Dockerfile `RUN`, a workflow `run:` (with the
# block scalar its value may open on). Stripped once, at the boundary between file syntax and
# shell syntax, so nothing downstream has to know which kind of file the line came from.
_FORMAT_KEYWORD = re.compile(r"^\s*(?:RUN|ENTRYPOINT|CMD|-?\s*(?:name:.*)?run:)\s*[|>]?[+-]?\d*\s*")


def _shell_part(line: str) -> str:
    return _FORMAT_KEYWORD.sub("", line, count=1)


# Words and shell operators. A quoted span is ONE token, so shell metacharacters inside it stay
# data: a query string (`curl 'https://host/i.sh?a=1&b=2' | bash`) must not have its `&` read as
# a command separator, which would file the fetch and the shell into different pipelines and let
# a live installer through. A `>&`-style redirection is matched before the bare `&` so its `&` is
# not taken for a separator either.
_TOKEN = re.compile(
    r"""'[^']*'|"(?:[^"\\]|\\.)*"|\d*[<>]{1,2}&\d*|\|\||&&|\||;|&"""
    r"""|(?:'[^']*'|"(?:[^"\\]|\\.)*"|\\.|[^\s;|&'"])+"""
)
_FETCH = re.compile(r"(?:[\w./-]*/)?(?:curl|wget)")

# A quoted span is a COMMAND when it is the argument to a shell's `-c`, and a VALUE everywhere
# else. That is a POSITIONAL test, and it replaces three attempts at a content-based one.
#
# The content heuristics all failed in both directions, because a URL may legally contain any
# operator a command can. Requiring whitespace around the operator missed the tight
# `sh -c "curl …|bash"`; counting a bare `|` anywhere then broke `curl 'http://host/a||b' | bash`
# (the `||` ended the pipeline early) AND invented a phantom stage from `curl 'https://host/f|bash'
# | jq .`, flagging a pipeline that feeds `jq`. Each fix traded one failure direction for the
# other, which is the signature of asking the wrong question: no amount of looking at the
# characters distinguishes a command from a URL that resembles one.
#
# Where the span SITS does distinguish them. `sh -c "…"` is the only construct here that takes a
# command as a string, so that is the only place the quotes get opened.
#
# Short options fuse: `sh -ec`, `bash -lc`, `sh -euc` all end in `c` and all take the next word
# as the command. Matching the flag exactly as `-c` missed every fused spelling -- including
# `bash -lc '…'`, which this repo already writes in docker/bake_kernel_cache.py. A long `--command`
# counts too; `--config` and other `c`-words must not, hence the exact long-form match.
_TAKES_A_COMMAND_STRING = re.compile(r"-[A-Za-z]*c|--command")


def _tokenize(line: str) -> list[str]:
    """Split a logical line into words and operators, opening `sh -c "…"` command strings.

    Two requirements pull against each other. Quoting must not HIDE a real pipeline from the
    scan -- `sh -c "curl … | bash"` runs exactly the command this guard exists to reject -- but
    it must not EXPOSE a quoted value's own characters as operators either, or a URL containing
    `&`, `;`, or `|` gets split into phantom commands.

    Both are satisfied by opening a quoted span only in the one position where the shell itself
    treats it as code: immediately after a shell name's `-c`.
    """
    tokens: list[str] = []
    raw = _TOKEN.findall(line)
    for index, tok in enumerate(raw):
        quoted = len(tok) > 1 and tok[0] == tok[-1] and tok[0] in "\"'"
        inner = tok[1:-1] if quoted else ""
        if inner and _is_a_command_string(raw, index):
            tokens.extend(_tokenize(inner))
        else:
            tokens.append(tok[1:-1] if quoted else tok)
    return tokens


def _is_a_command_string(tokens: list[str], index: int) -> bool:
    """Is `tokens[index]` the command string of a `sh -c` (or `bash -lc`, `zsh -euc`, …)?

    Walks the stage forward to its command word rather than scanning left from the `-c`, so a
    shell option that takes its own operand cannot hide the flag behind it: a leftward scan over
    `bash -o pipefail -c "…"` stops at `pipefail` and concludes the command is not a shell. Only
    a shell counts -- `jq -c "…"` passes `-c` too, and its argument is a filter, not a command.
    """
    if index == 0 or not _TAKES_A_COMMAND_STRING.fullmatch(tokens[index - 1]):
        return False
    word, at = _command_word(tokens[: index - 1])
    return at is not None and bool(_SHELL_NAME.fullmatch(word))


def _command_word(tokens: list[str]) -> tuple[str, int | None]:
    """The word a stage actually runs, and its index -- or `("", None)` if there is none.

    Assignments and redirections may PRECEDE the command (`MODE=install bash`, `2>/dev/null sh`)
    -- both run the downloaded stream, and the second needs no wrapper at all. Exec wrappers move
    the command further right, and their flags may consume an operand, which is the part a regex
    could not track.

    One walk serves both callers. Splitting it in two is what let `bash -o pipefail -c` through:
    each copy has to know the same flag arities, and only one of them was taught this one.
    """
    i = 0
    wrapper = ""
    while i < len(tokens):
        tok = tokens[i]
        if re.fullmatch(r"[A-Za-z_]\w*=.*", tok):
            i += 1
            continue
        # A redirection may be written apart from its target (`2> /dev/null bash`), in which case
        # the target is a separate token that is not the command.
        if re.fullmatch(r"\d*[<>]{1,2}.*", tok):
            i += 2 if tok.rstrip().endswith((">", "<")) else 1
            continue
        if tok.startswith(("-", "+")):
            if "=" not in tok and tok in _FLAGS_WITH_OPERAND.get(wrapper, frozenset()):
                i += 1
            i += 1
            continue
        if wrapper and _NUMERIC_OPERAND.fullmatch(tok):
            i += 1
            continue
        if tok in _EXEC_WRAPPERS or (tok.rsplit("/", 1)[-1] in _EXEC_WRAPPERS):
            wrapper = tok.rsplit("/", 1)[-1]
            i += 1
            continue
        return tok, i
    return "", None


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
    """Does one pipeline stage run a shell?

    A stage may be a subshell (`curl … | ( bash )`), which runs its contents in a shell of its
    own -- so the grouping tokens are stepped over rather than read as the command. Any other
    word IS the command, and is not a shell: `grep bash` stops there, so the shell name after it
    is an argument being searched for, not the command being run.
    """
    inner = [t for t in tokens if t not in ("(", ")", "{", "}")]
    word, at = _command_word(inner)
    if at is not None and _SHELL_NAME.fullmatch(word):
        # `sh -c '…'` takes its program from the operand and leaves the pipe on stdin for
        # whatever that program is. `curl checksums.txt | sh -c 'sha256sum -c -'` therefore
        # VERIFIES the download rather than running it -- the stream is data, and flagging it
        # would fail CI on the exact pattern this guard is meant to encourage.
        #
        # So the operand replaces the question, it does not answer it: ask whether THAT command
        # is a shell. `sh -c 'bash'` still matches, because the thing reading stdin is a shell.
        program = _dash_c_operand(inner[at:])
        return _stage_runs_a_shell(_tokenize(program)) if program is not None else True
    # Checked even when the walk found no command word: `env -S'bash -s'` fuses the operand into
    # the flag, so every token is a flag and the walk runs off the end with the shell still in it.
    return any(_stage_runs_a_shell(_tokenize(cmd)) for cmd in _split_string_operands(inner))


def _dash_c_operand(tokens: list[str]) -> str | None:
    """The program string a shell's `-c` supplies, or None when the shell has no `-c`.

    Returns the operand even when it is empty, so `sh -c ''` is distinguishable from a shell
    invoked with no `-c` at all -- the first runs nothing, the second runs the piped stream.

    `-s` wins over `-c` when both appear: it makes the shell read its program from stdin, which
    is the piped download, so the operand is no longer the whole story.
    """
    for index, tok in enumerate(tokens):
        if tok.startswith("-") and "s" in tok[1:] and not tok.startswith("--"):
            return None
        if _TAKES_A_COMMAND_STRING.fullmatch(tok):
            return tokens[index + 1] if index + 1 < len(tokens) else ""
    return None


def _split_string_operands(tokens: list[str]) -> list[str]:
    """The command strings introduced by `env -S`, in every spelling it accepts.

    The operand may be a separate token (`-S 'bash -s'`) or fused into the flag itself
    (`-S'bash -s'`, `--split-string='bash -s'`). Normalized here rather than handled at two call
    sites: the separate and fused forms are the same instruction, and splitting them across two
    mechanisms is how the fused one went missing in the first place.
    """
    operands: list[str] = []
    for flag, following in itertools.zip_longest(tokens, tokens[1:], fillvalue=""):
        fused = _FUSED_SPLIT_STRING.fullmatch(flag)
        if fused:
            operands.append(fused["command"].strip("\"'"))
        elif flag in _SPLITS_ITS_OPERAND and following:
            operands.append(following)
    return operands


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
        # A query string's `&` and `;` are DATA. Reading them as command separators filed the
        # fetch and the shell into different pipelines, so a live installer scanned clean.
        pytest.param(
            "RUN curl -fsSL 'https://x.example/i.sh?a=1&b=2' | bash\n", id="quoted-url-ampersand"
        ),
        pytest.param(
            'RUN curl -fsSL "https://x.example/i.sh;v=1" | sh\n', id="quoted-url-semicolon"
        ),
        # A quoted pipe with NO spaces around it. `&` and `;` need whitespace to count (a query
        # string has neither), but `|` is rare in a URL and almost always a real pipe, so it
        # counts anywhere -- otherwise this whole command stays one opaque word.
        pytest.param(
            'RUN sh -c "curl -fsSL https://x.example/i.sh|bash"\n', id="tight-pipe-inside-quotes"
        ),
        pytest.param(
            "RUN sh -c 'curl -fsSL https://x.example/i.sh|sh'\n", id="tight-pipe-single-quoted"
        ),
        # A `||` inside the URL must stay data. Reading it as an operator ended the pipeline
        # early, so the real outer `| bash` landed in a pipeline with no fetch in it.
        pytest.param("RUN curl 'http://x.example/a||b' | bash\n", id="double-pipe-in-a-url"),
        # The `-c` string may sit behind the shell's own flags, or behind another wrapper.
        pytest.param(
            'RUN bash -eu -c "curl -sSL https://x.example/i.sh | sh"\n', id="dash-c-behind-flags"
        ),
        pytest.param(
            'RUN sudo sh -c "curl -sSL https://x.example/i.sh|bash"\n', id="dash-c-behind-sudo"
        ),
        # Short options FUSE. `-ec`, `-lc`, `-euc` all end in `c` and all take the next word as
        # the command, so matching the flag exactly as `-c` missed every fused spelling --
        # including `bash -lc`, which this repo already writes in docker/bake_kernel_cache.py.
        pytest.param('RUN sh -ec "curl -sSL https://x.example/i.sh|bash"\n', id="fused-flags-ec"),
        pytest.param("RUN bash -lc 'curl -sSL https://x.example/i.sh | sh'\n", id="fused-flags-lc"),
        pytest.param(
            'RUN sh -euc "curl -sSL https://x.example/i.sh | bash"\n', id="fused-flags-euc"
        ),
        # BusyBox's shell, and therefore `/bin/sh` on every Alpine image -- the base a vendor
        # install script is most likely to run under.
        pytest.param("RUN curl -sSL https://x.example/i.sh | ash\n", id="busybox-ash"),
        # The same shell reached through BusyBox's applet dispatch. `busybox` is a wrapper the
        # walk must step over, not the command word.
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | busybox ash\n", id="busybox-applet-dispatch"
        ),
        pytest.param("RUN curl -sSL https://x.example/i.sh | busybox sh\n", id="busybox-applet-sh"),
        # A shell whose `-c` operand is ITSELF a shell: the thing reading stdin is still a shell,
        # so exempting every `-c` would let this through.
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | sh -c 'bash'\n", id="dash-c-runs-a-shell"
        ),
        # `-s` makes the shell read its program from stdin -- the piped download -- so it wins
        # over the `-c` operand rather than deferring to it.
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | bash -s -c 'jq .'\n", id="dash-s-beats-dash-c"
        ),
        # A shell option that takes its own operand. Scanning LEFT from the `-c` stopped at
        # `pipefail` and concluded the command was not a shell, so the quoted pipeline stayed
        # closed -- and `-o pipefail` is exactly how a careful CI script opens.
        pytest.param(
            'RUN bash -o pipefail -c "curl -sSL https://x.example/i.sh | sh"\n',
            id="shell-option-with-operand",
        ),
        # Nested quotes reach the inner shell unescaped. Left escaped, the inner scan read
        # `\"https://…\"` as one bare word, and the pipe after it never separated the fetch
        # from the shell it feeds.
        pytest.param(
            'RUN bash -c "curl -sSL \\"https://x.example/i.sh\\" | bash"\n',
            id="escaped-quotes-inside-a-command-string",
        ),
        # `env -S` SPLITS its operand into words and runs them, the opposite of the flags above:
        # the operand is not a value to step over, it is the command.
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | env -S 'bash -s'\n", id="env-split-string"
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | env --split-string 'sh -e'\n",
            id="env-split-string-long",
        ),
        # The same flag with its operand FUSED into the token. Every token is then a flag, so the
        # walk runs off the end of the stage with the shell still inside one of them.
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | env -S'bash -s'\n", id="env-split-string-fused"
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | env --split-string='bash -s'\n",
            id="env-split-string-fused-long",
        ),
        # A subshell runs its contents in a shell of its own.
        pytest.param("RUN curl -sSL https://x.example/i.sh | ( bash )\n", id="subshell-stage"),
        # A redirection may be written apart from its target, which is not the command word.
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | 2> /dev/null bash\n", id="spaced-redirection"
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
        # The other side of the quoted-URL fix: a query string must not make an ordinary
        # pipeline match either, so the fix cannot be "treat quoted spans as opaque".
        pytest.param("curl -s 'https://x.example/f?a=1&b=2' | jq .", id="quoted-url-into-jq"),
        # A `|` inside a URL value: counting `|` anywhere must not make the surrounding pipeline
        # match when the command it feeds is not a shell.
        pytest.param("curl -s 'https://x.example/f?p=a|b' | jq .", id="pipe-inside-a-url-value"),
        # A URL segment that IS a shell name. Opening the quotes here invented a pipeline stage
        # that does not exist -- curl feeds `jq`, and the `bash` is part of the path.
        pytest.param("curl -s 'https://x.example/f|bash' | jq .", id="shell-named-url-segment"),
        # `-c` is not exclusive to shells: jq takes one too, and its argument is a filter.
        pytest.param(
            'jq -c "curl https://x.example/i.sh | bash" /tmp/f', id="dash-c-on-a-non-shell"
        ),
        # Accepting fused short options must not turn every long flag ending in a c-word into a
        # command opener: `--config` is not `--command`.
        pytest.param(
            'sh --config "curl https://x.example/i.sh | bash"', id="long-flag-is-not-command"
        ),
        # Accepting `ash` must not turn every word ending in those letters into a shell. `cash`
        # and `stash` end in `ash`; the name has to match WHOLE, not as a suffix.
        pytest.param("curl -s https://x.example/f | cash --report", id="ash-is-not-a-suffix"),
        pytest.param("curl -s https://x.example/f | git stash", id="stash-is-not-a-shell"),
        # `-S` splits its operand and runs it -- but only when the operand really is a shell.
        pytest.param("curl -s https://x.example/f | env -S 'jq .version'", id="split-into-jq"),
        pytest.param("curl -s https://x.example/f | env -S'jq .version'", id="split-fused-into-jq"),
        # `-S` on something that is not `env` is an ordinary flag: jq's is --sort-keys.
        pytest.param("curl -s https://x.example/f | jq -S .", id="dash-s-on-a-non-env"),
        # `sh -c '…'` takes its program from the operand and leaves the pipe on stdin for that
        # program, so the download is DATA. Verifying a checksum this way is the pattern this
        # guard exists to encourage, and flagging it would fail CI on the recommended spelling.
        pytest.param(
            "curl -s https://x.example/checksums.txt | sh -c 'sha256sum -c -'",
            id="dash-c-verifies-the-download",
        ),
        pytest.param("curl -s https://x.example/f | bash -c 'jq .'", id="dash-c-runs-a-non-shell"),
        # BusyBox dispatches non-shell applets too; stepping over the wrapper must not make the
        # applet after it match.
        pytest.param("curl -s https://x.example/f | busybox cat", id="busybox-non-shell-applet"),
        # A shell OPTION's operand is skipped, so a shell-named one must not read as a command.
        # `bash -o bash script.sh` sets an (invalid) option and runs a script, not a nested shell.
        pytest.param("curl -s https://x.example/f | jq -o bash .", id="option-operand-not-command"),
        # The format keyword is stripped, which must not strip a real command that starts with
        # the same letters. `runner` is not `run:`.
        pytest.param(
            "curl -s https://x.example/f | runner --exec bash", id="keyword-is-not-a-prefix"
        ),
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
