"""Enough shell grammar to tell a real pipe-to-shell from an ordinary pipeline.

`curl … | bash` executes whatever the vendor serves at request time: it cannot be reviewed,
cannot be reproduced, and pins nothing. Rejecting it needs more than a substring search --
`curl … | jq`, `grep bash`, and `sh -c 'sha256sum -c -'` are all legitimate and must stay
clean, while `xargs -a f bash`, `env -S'bash -s'`, and `nice sh` are all the real thing.

So this models the grammar directly: tokenize, split into pipelines and stages, walk each stage
past its wrappers and flags to the command WORD, and ask whether that word is a shell and
whether the stream actually reaches it. Every claim about a wrapper's behaviour here was
confirmed by running it, not read off a man page -- the comments record which.

The rule that uses this lives in `test_no_pipe_to_shell.py`. Kept separate because the grammar
is the part with the subtle cases, and it is easier to reason about without 500 lines of
parametrized fixtures underneath it.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path

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
    # Scheduling and privilege wrappers. Each adjusts one attribute of the process and then
    # execs its COMMAND with stdin untouched, so the piped stream reaches the shell exactly as
    # if the wrapper were not there. Confirmed by running each one:
    # `printf 'echo X\n' | nice sh` prints X, and likewise for the rest.
    "nice",
    "ionice",
    "setsid",
    "setpriv",
    "chrt",
    "taskset",
    "runuser",
}

# Flags that take a SEPARATE operand, per wrapper. Keyed by wrapper because the same spelling
# means different things: `env -u NAME` unsets a variable, `sudo -u USER` picks a user. Knowing
# the arity is what stops the operand from being mistaken for the command -- `env -u bash cat`
# runs `cat`, not `bash`, and flagging it would fail CI on a legitimate pipeline.
_FLAGS_WITH_OPERAND = {
    "sudo": {"-u", "-g", "-h", "-p", "-C", "-U", "-r", "-t", "--user", "--group", "--prompt"},
    "env": {"-u", "-C", "--unset", "--chdir"},
    "timeout": {"-k", "-s", "--kill-after", "--signal"},
    # Confirmed by running each one bare: `xargs --arg-file` reports "requires an argument",
    # while `--replace` and `--max-lines` do not (their operands are optional).
    "xargs": {
        "-a",
        "-E",
        "-I",
        "-L",
        "-n",
        "-P",
        "-s",
        "-d",
        "--arg-file",
        "--delimiter",
        "--max-args",
        "--max-procs",
        "--max-chars",
    },
    "stdbuf": {"-i", "-o", "-e", "--input", "--output", "--error"},
    "nice": {"-n", "--adjustment"},
    "ionice": {"-c", "-n", "-p", "-P", "--class", "--classdata", "--pid"},
    "chrt": {"-p", "--pid"},
    "taskset": {"-c", "-p", "--cpu-list", "--pid"},
    "setpriv": {
        "--reuid",
        "--regid",
        "--groups",
        "--inh-caps",
        "--bounding-set",
        "--selinux-label",
    },
    "runuser": {"-u", "-g", "-G", "--user", "--group", "--supp-group"},
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
    r"""'[^']*'|"(?:[^"\\]|\\.)*"|\d*[<>]{1,2}&\d*|\|\||&&|\||;|&|\(|\)"""
    r"""|(?:'[^']*'|"(?:[^"\\]|\\.)*"|\\.|[^\s;|&()'"])+"""
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
            if "=" not in tok and _takes_a_separate_operand(wrapper, tok):
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


def _pipelines_in(text: str) -> list[list[list[str]]]:
    """Split a command list into pipelines, and each pipeline into its stages.

    `||`, `&&`, `;`, and `&` end a pipeline: what follows is a separate command with nothing
    piped into it. Reading the second bar of a `||` as a pipe made legitimate
    `curl … || bash recover.sh` fallbacks match.

    Shared by the top-level line and by a shell's `-c` program, because they are the same
    grammar -- `sh -c 'true; bash'` is a command list exactly as a `RUN` line is. Splitting the
    two apart is what left the `-c` program resolving only its first command.
    """
    pipelines: list[list[list[str]]] = [[[]]]
    for tok in _tokenize(text):
        if tok in ("||", "&&", ";", "&"):
            pipelines.append([[]])
        elif tok == "|":
            pipelines[-1].append([])
        else:
            pipelines[-1][-1].append(tok)
    return pipelines


def _commands_in(text: str) -> list[list[str]]:
    """Every command in a command list, flattened across pipelines and pipe stages."""
    return [stage for pipeline in _pipelines_in(text) for stage in pipeline if stage]


def _pipeline_stages_after_a_fetch(line: str) -> list[list[str]]:
    """Every stage that a curl/wget's output flows into, as token lists.

    All later stages are returned, not just the next one: a pipeline may pass through
    intermediate stages first (`curl … | tee /tmp/i.sh | bash`), and looking only at the stage
    directly after the fetch was blind to every such spelling.

    `||` is NOT a stage separator, and neither are `&&`, `;`, or `&`. They end the pipeline: what
    follows is a separate command with nothing piped into it. Reading the second bar of a `||` as
    a pipe made legitimate `curl … || bash recover.sh` fallbacks match.
    """
    downstream: list[list[str]] = []
    for stages in _pipelines_in(line):
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
        # `xargs` reads the pipe ITSELF to build an argument list, so the shell it runs gets
        # /dev/null on stdin and the downloaded bytes as ARGV. That is a different question,
        # asked below -- not a quieter version of this one.
        if _stream_becomes_arguments(inner[:at]):
            return _arguments_reach_a_command_slot(inner[:at], inner[at:])
        # `sh -c '…'` takes its program from the operand and leaves the pipe on stdin for
        # whatever that program is. `curl checksums.txt | sh -c 'sha256sum -c -'` therefore
        # VERIFIES the download rather than running it -- the stream is data, and flagging it
        # would fail CI on the exact pattern this guard is meant to encourage.
        #
        # So the operand replaces the question, it does not answer it: ask whether THAT command
        # is a shell. `sh -c 'bash'` still matches, because the thing reading stdin is a shell.
        #
        # The operand is a command LIST, not one command: `sh -c 'true; bash'` runs two, and the
        # stream is still there for the second. So every command in it is asked, not just the
        # first -- which is also why this cannot be "does a shell name appear in the program":
        # `sh -c 'true; jq .'` names no shell that runs, and must stay clean.
        program = _dash_c_operand(inner[at:])
        if program is None:
            return True
        return any(_stage_runs_a_shell(cmd) for cmd in _commands_in(program))
    # Checked even when the walk found no command word: `env -S'bash -s'` fuses the operand into
    # the flag, so every token is a flag and the walk runs off the end with the shell still in it.
    return any(_stage_runs_a_shell(_tokenize(cmd)) for cmd in _split_string_operands(inner))


def _takes_a_separate_operand(wrapper: str, flag: str) -> bool:
    """Is the token AFTER this flag its operand rather than the command?

    Short flags cluster, and only the LAST letter in a cluster can take one: `xargs -0a f sh`
    is `-0` plus `-a f`, so the file is the operand and `sh` is still the command. Reading the
    cluster as a whole token misses that, and the walk then stops on the operand.
    """
    known = _FLAGS_WITH_OPERAND.get(wrapper, frozenset())
    if flag in known:
        return True
    return not flag.startswith("--") and len(flag) > 2 and f"-{flag[-1]}" in known


def _stream_becomes_arguments(wrappers: list[str]) -> bool:
    """Does an `xargs` in the wrapper prefix redirect the pipe from stdin into argv?

    `xargs` reads the stream itself to build an ARGUMENT list and runs its command with stdin on
    /dev/null. So `curl … | xargs bash` hands the shell the download as a FILENAME to open, and
    the bytes are never executed.

    `-a`/`--arg-file` takes the argument list from a file instead and leaves the pipe connected,
    which puts the stream back on the child's stdin -- the ordinary route, not this one.

    Confirmed rather than assumed, against GNU findutils:
        printf 'echo PWNED\\n' | xargs -0 sh      -> sh: cannot open 'echo PWNED'  (argv)
        printf 'echo PWNED\\n' | xargs -a f sh    -> PWNED                         (stdin)
    """
    if not any(tok.rsplit("/", 1)[-1] == "xargs" for tok in wrappers):
        return False
    # Every spelling of the operand: `-a f`, `-af`, `-0a f`, `--arg-file f`, `--arg-file=f`.
    return not any(
        tok in ("-a", "--arg-file")
        or tok.startswith("--arg-file=")
        or (tok.startswith("-") and not tok.startswith("--") and "a" in tok[1:])
        for tok in wrappers
    )


def _arguments_reach_a_command_slot(wrappers: list[str], command: list[str]) -> bool:
    """Once the stream is argv, does it still land somewhere the shell EXECUTES?

    Being argv is normally harmless -- a shell given a filename tries to open it. But two
    spellings feed argv straight back into the shell's command string:

      `xargs sh -c`     the `-c` operand is MISSING, so the first argument xargs appends becomes
                        the program text. The download is executed verbatim.
      `xargs -I{} sh -c '… {} …'`   the placeholder is substituted INTO the operand, so the
                        download is executed as part of it.

    Confirmed rather than assumed, against GNU findutils:
        printf 'echo PWNED\\n' | xargs -0 sh -c              -> PWNED
        printf 'echo PWNED\\n' | xargs -I{} sh -c '{}'       -> PWNED
        printf 'echo PWNED\\n' | xargs -0 sh -c 'echo SAFE'  -> SAFE   (operand present)
    """
    program = _dash_c_operand(command)
    if program is None:
        # No `-c` at all: the shell opens argv as a file. `-s` also lands here, but xargs gave
        # the shell /dev/null for stdin, so there is nothing to read.
        return False
    if program == "":
        return True
    return any(holder in program for holder in _replacement_placeholders(wrappers))


def _replacement_placeholders(wrappers: list[str]) -> list[str]:
    """The strings xargs will substitute the stream into, or NOTHING if it substitutes at all.

    An empty result is the common case and it matters: with no replace flag, `{}` in the command
    is ordinary text. `curl … | xargs -0 sh -c 'jq {}'` runs a jq filter, and treating `{}` as a
    placeholder there fails CI on a legitimate pipeline. So this returns [] rather than defaulting
    to `{}` -- the default belongs to the FLAG, not to the absence of one.

    The two flags differ in arity, which is the whole subtlety:

      `-I` REQUIRES an operand      -- `-I{}` fused, or `-I {}` separate.
      `-i` / `--replace` OPTIONAL   -- `-i` alone means `{}`; `-iQQ` / `--replace=QQ` fuse a custom
                                       one. A SEPARATE word is never its operand, so `xargs -i -0`
                                       is `-i` (meaning `{}`) plus `-0`, not the placeholder `-0`.

    Confirmed rather than assumed, against GNU findutils:
        printf 'echo PWNED\\n' | xargs -i -0 sh -c '{}'        -> PWNED   (`-0` was NOT the operand)
        printf 'echo PWNED\\n' | xargs -iQQ sh -c 'echo got QQ' -> got echo PWNED
        printf 'X\\n'          | xargs -0 sh -c 'echo {}'       -> {}      (literal, no flag)
    """
    placeholders: list[str] = []
    for flag, following in itertools.zip_longest(wrappers, wrappers[1:], fillvalue=""):
        if flag == "-I":
            # Required operand, so the next word IS it even when it looks like an option.
            placeholders.append(following)
        elif flag == "--replace":
            placeholders.append("{}")
        elif flag.startswith("--replace="):
            placeholders.append(flag.split("=", 1)[1])
        elif not flag.startswith("--"):
            placeholders.append(_fused_replacement(flag))
    return [p for p in placeholders if p]


def _fused_replacement(flag: str) -> str:
    """The placeholder inside a short-flag cluster, or "" when the cluster has no `-i`/`-I`.

    Short flags cluster, and `-i`/`-I` may sit at the end of one: `xargs -0i` is `-0` plus `-i`.
    Everything after that letter is the placeholder, not more flags -- `-in1` means the
    placeholder is literally `n1`, not `-i -n 1`. So the search is for the LAST `i`/`I`, and only
    when nothing before it could have consumed the rest.

    Confirmed rather than assumed, against GNU findutils:
        printf 'echo PWNED\\n' | xargs -0i sh -c '{}'        -> PWNED    (clustered, default `{}`)
        printf 'echo PWNED\\n' | xargs -in1 sh -c '{}'       -> {}: not found  (`n1` is the name)
    """
    for position, letter in enumerate(flag):
        if letter in ("i", "I") and position > 0:
            return flag[position + 1 :] or "{}"
    return ""


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
