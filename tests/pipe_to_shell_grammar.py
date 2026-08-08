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

WHICH text gets scanned -- the files discovered, and how continued lines are joined back into one
command -- is `pipe_to_shell_scan`. That half needs no notion of a shell, and this half needs no
notion of a file; keeping them apart is what stops either from being debugged through the other.

What the stream BECOMES once a wrapper has had its say -- argv rather than stdin, an `sh -c`
program, an `env -S` operand -- is `pipe_to_shell_argv`.
"""

from __future__ import annotations

import itertools
import json
import re
import shlex

from tests.pipe_to_shell_argv import (
    _TAKES_A_COMMAND_STRING,
    _arguments_reach_a_command_slot,
    _dash_c_operand,
    _split_string_operands,
    _stream_becomes_arguments,
)

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
#
#
# `su` is deliberately NOT here. It starts a shell by default, but `-s`/`--shell` can point it at
# any program, so matching it as a shell NAME flagged safe pipelines too. It is modelled as a
# privilege tool below, where the program it actually starts is the thing being judged.
_SHELL_NAME = re.compile(r"(?:[\w./-]*/)?(?:(?:ba|a|z|k|da)?sh)$")

# Privilege tools that START A PROGRAM as another user, defaulting to that user's SHELL. They are
# not shells themselves: `-s`/`--shell` chooses WHICH program runs, so the same tool may end up
# running a shell or not. Matching them as shell NAMES made every spelling dangerous, including
# the safe ones -- `su -s /usr/bin/sha256sum nobody` hashes the stream and executes nothing:
#   printf 'hello\n' | sudo -n su -s /usr/bin/sha256sum nobody  ->  5891b5b5…  (hashed, not run)
#   printf 'echo PWNED\n' | sudo -n su -s /bin/sh               ->  PWNED
_PRIVILEGE_TOOLS = {"su", "runuser", "sudo"}

# How each names the program it starts. `su`/`runuser` accept a bare positional USER, which is
# not the program -- their program comes from `-s`/`--shell` or the user's login shell.
_SHELL_SELECTING_FLAGS = ("-s", "--shell")

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
    # `unshare` runs its program with namespaces unshared, stdin untouched. Confirmed:
    #   sudo -n unshare bash < payload        ->  ran
    #   sudo -n unshare --fork sh < payload   ->  ran
    "unshare",
}

# Flags that take a SEPARATE operand, per wrapper. Keyed by wrapper because the same spelling
# means different things: `env -u NAME` unsets a variable, `sudo -u USER` picks a user. Knowing
# the arity is what stops the operand from being mistaken for the command -- `env -u bash cat`
# runs `cat`, not `bash`, and flagging it would fail CI on a legitimate pipeline.
_FLAGS_WITH_OPERAND = {
    "sudo": {"-u", "-g", "-h", "-p", "-C", "-U", "-r", "-t", "--user", "--group", "--prompt"},
    # `su -s /bin/sh` picks the shell to start; the operand is not the command being run.
    "su": {"-s", "-g", "-G", "--shell", "--group", "--supp-group"},
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
    # `unshare`'s namespace flags take an OPTIONAL fused operand (`--mount=<file>`), so they
    # consume nothing separate. Only these six take a separate one, per `unshare --help`.
    "unshare": {"-S", "-G", "--setuid", "--setgid", "--propagation", "--setgroups"},
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

    The file's own keyword is dropped first. `RUN`/`run:` introduces a shell command but is not
    part of one, and a walk that reads it as the command word stops before reaching the shell.
    """
    command = _shell_part(line)
    if any(_stage_runs_a_shell(s) for s in _pipeline_stages_after_a_fetch(command)):
        return True
    # A whole pipeline may sit INSIDE a group or clause, where the enclosing construct is a single
    # stage and the fetch never appears at this level at all. `_pipeline_stages_after_a_fetch`
    # therefore finds nothing, while the shell inside runs the download exactly as it would at the
    # top level. Confirmed to execute:
    #   if true; then cat p.sh | bash; fi   ->  ran
    #   ( cat p.sh | bash )                 ->  ran
    return any(_piped_into_a_shell(inner) for inner in _nested_command_lists(command))


# What the file writes BEFORE the shell command: a Dockerfile `RUN`, a workflow `run:` (with the
# block scalar its value may open on). Stripped once, at the boundary between file syntax and
# shell syntax, so nothing downstream has to know which kind of file the line came from.
_FORMAT_KEYWORD = re.compile(r"^\s*(?:RUN|ENTRYPOINT|CMD|-?\s*(?:name:.*)?run:)\s*[|>]?[+-]?\d*\s*")


# Dockerfile EXEC form: `RUN ["bash", "-c", "curl … | bash"]`. The shell command is a JSON string
# INSIDE a JSON array, so after the keyword is stripped the line still starts with `[`. Docker
# runs the array's members as argv with no shell of its own -- but here argv[0] IS a shell, and
# argv[2] is the command it runs, so the download is executed exactly as in the shell form.
_JSON_EXEC_FORM = re.compile(r"^\[\s*(?:\"(?:[^\"\\]|\\.)*\"\s*,?\s*)+\]\s*$")

# A workflow `run:` value may be a QUOTED scalar rather than a bare or block one. The quotes are
# YAML syntax, not shell syntax, so they have to come off at this boundary -- otherwise the whole
# command arrives as one quoted token and no pipeline is ever seen inside it.
_QUOTED_SCALAR = re.compile(r"^(?P<q>[\"'])(?P<body>.*)(?P=q)\s*$")


def _shell_part(line: str) -> str:
    """The shell command a line runs, with the FILE's own syntax taken off.

    Three spellings reach the same shell command, and the difference between them is the
    surrounding file's grammar, not the shell's:

        RUN curl … | bash                        <- shell form, the keyword is all there is
        RUN ["bash", "-c", "curl … | bash"]      <- docker exec form, a JSON array
        run: "curl … | bash"                     <- a quoted YAML scalar

    Stripping all of it here keeps the knowledge of which file a line came from at this one
    boundary, so the grammar below only ever sees shell syntax.
    """
    stripped = _FORMAT_KEYWORD.sub("", line, count=1).strip()
    if _JSON_EXEC_FORM.fullmatch(stripped):
        return _exec_form_command(stripped)
    scalar = _QUOTED_SCALAR.fullmatch(stripped)
    # Only unwrap when the quotes enclose the WHOLE value. `curl 'https://h/a|b' | bash` also
    # starts and ends with a quote, and stripping those would splice a URL onto a shell name.
    # An EMPTY scalar (`run: ""`) tokenizes to nothing, so indexing the first token raised
    # IndexError and aborted the entire scan -- a crash where the answer is simply "clean".
    body_tokens = _TOKEN.findall(scalar.group("body")) if scalar else []
    if scalar and body_tokens and not body_tokens[0].startswith(("'", '"')):
        return scalar.group("body")
    return stripped


def _exec_form_command(text: str) -> str:
    """The shell command inside a Dockerfile exec-form array, or the argv joined back up.

    `["bash", "-c", "curl … | bash"]` is a shell running a command string, so the array is
    rejoined into exactly the shell form the grammar below already understands. An array that
    runs no shell (`["python", "-m", "pip", "install", "x"]`) rejoins into a stage whose command
    word is not a shell, so it stays clean for the same reason its shell-form spelling would.
    """
    try:
        argv = json.loads(text)
    except ValueError:
        return text
    return " ".join(shlex.quote(str(arg)) for arg in argv)


# Words and shell operators. A quoted span is ONE token, so shell metacharacters inside it stay
# data: a query string (`curl 'https://host/i.sh?a=1&b=2' | bash`) must not have its `&` read as
# a command separator, which would file the fetch and the shell into different pipelines and let
# a live installer through. A `>&`-style redirection is matched before the bare `&` so its `&` is
# not taken for a separator either.
_TOKEN = re.compile(
    # `|&` is ONE operator (bash: pipe stdout AND stderr), so it is matched before `||` and the
    # bare `&` -- otherwise it splits into `|` then `&`, and the `&` ends the pipeline before the
    # shell, which read as clean:  cat payload |& bash  ->  ran
    r"""'[^']*'|"(?:[^"\\]|\\.)*"|\d*[<>]{1,2}&\d*|\|&|\|\||&&|\||;|&|\(|\)"""
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


# A compound command delimited by KEYWORDS rather than punctuation. `curl … | if true; then
# bash; fi` is one pipeline stage, and the download is on its stdin for every command inside it --
# exactly like the `{ …; }` group below, but `if`/`fi` are words, so the depth counter never saw
# them and the clause's own `;` split the pipeline, detaching `bash`. Confirmed to execute:
#   printf 'echo PWNED\n' | if true; then bash; fi      ->  PWNED
_COMPOUND_OPENERS = {"if", "while", "until", "for", "select", "case"}
_COMPOUND_CLOSERS = {"fi", "done", "esac"}
# Words that are shell syntax inside a compound, not commands in its body.
_COMPOUND_INTERNALS = {"then", "else", "elif", "do", "in"}
# A keyword is only a keyword in COMMAND position. `echo done` is an argument -- reading it as a
# closer would unbalance the counter and split the very clause it is meant to hold together:
#   if true; then echo done; fi   ->  `done` here is a word being printed, not the closer
_KEYWORD_FOLLOWS = {";", "&", "|", "|&", "||", "&&", "(", "{"} | _COMPOUND_INTERNALS


def _in_command_position(stage: list[str]) -> bool:
    """Would the next token start a command, rather than continue one as an argument?"""
    return not stage or stage[-1] in _KEYWORD_FOLLOWS


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
    depth = 0
    opened: list[str] = []
    for tok in _tokenize(text):
        if tok in ("(", "{"):
            depth += 1
            opened.append(tok)
            pipelines[-1][-1].append(tok)
        # A `)` that closes nothing is a `case` PATTERN terminator, not a group closer:
        # `case x in x) bash;; esac` has one per branch and never an opening `(`. Decrementing
        # there unbalanced the counter, so the clause's `;;` split the pipeline and detached the
        # shell from the fetch. Confirmed to execute:
        #   printf 'echo PWNED\n' | case x in x) bash;; esac   ->  PWNED
        elif tok in (")", "}") and opened:
            depth = max(depth - 1, 0)
            opened.pop()
            pipelines[-1][-1].append(tok)
        elif tok in (")", "}"):
            pipelines[-1][-1].append(tok)
        elif tok in _COMPOUND_OPENERS and _in_command_position(pipelines[-1][-1]):
            depth += 1
            pipelines[-1][-1].append(tok)
        elif tok in _COMPOUND_CLOSERS and _in_command_position(pipelines[-1][-1]):
            depth = max(depth - 1, 0)
            pipelines[-1][-1].append(tok)
        # A separator INSIDE a group belongs to the group's own command list, not to the
        # enclosing pipeline. `curl … | { true; bash; }` is one stage whose group runs both
        # commands with the download still on stdin, so the group must stay attached to the
        # fetch. Splitting on that `;` detached `bash` and the whole line read as clean:
        #   printf 'echo PWNED\n' | { true; bash; }   ->  PWNED
        #   printf 'echo PWNED\n' | (true; bash)      ->  PWNED
        elif tok in ("||", "&&", ";", "&") and depth == 0:
            pipelines.append([[]])
        elif tok in ("|", "|&") and depth == 0:
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
        fetched = next((i for i, s in enumerate(stages) if _stage_fetches(s)), -1)
        if fetched >= 0:
            downstream.extend(stages[fetched + 1 :])
    return downstream


# A fetch captured by COMMAND SUBSTITUTION before being piped. The download still reaches the
# shell, but it arrives inside a token rather than as one: `printf '%s' "$(curl URL)" | bash`
# tokenizes to one quoted argument, so a whole-token match for `curl` never fires and the line
# read as clean. Confirmed to execute in each spelling:
#   printf '%s' "$(cat p.sh)" | bash   ->  ran
#   echo "$(cat p.sh)" | bash          ->  ran
#   printf '%s' "`cat p.sh`" | bash    ->  ran
_SUBSTITUTED_FETCH = re.compile(r"(?:\$\(|`)[^)`]*\b(?:curl|wget)\b")

# The same fetch UNQUOTED. `echo $(curl URL) | bash` is not one token: the tokenizer's fallback
# alternation excludes `(` and `)`, so it splits to `echo`, `$`, `(`, `curl`, `)`, and no single
# token carries both `$(` and the fetch name -- the pattern above never fires. Joining the stage
# back together and matching across the whole string catches both spellings with one rule.
# Confirmed to execute:  echo $(cat p.sh) | bash  ->  ran
_SUBSTITUTION_OPENS = ("$(", "`")


def _stage_substitutes_a_fetch(stage: list[str]) -> bool:
    """Does a command substitution anywhere in this stage run a fetch?

    Matched against the joined stage rather than token by token, because the tokenizer breaks an
    unquoted `$(curl …)` apart while keeping a quoted `"$(curl …)"` whole. Both reach the shell
    downstream, so both must match.
    """
    # Joined WITHOUT a separator: the tokenizer split `$(` into `$` and `(`, and a space between
    # them would stop `\$\(` matching the very spelling this exists to catch. Testing the tokens
    # individually fails for the same reason -- no single one carries both halves. The pattern's
    # own `\b` still keeps `curl` a whole word, so `$(curlie …)` does not match.
    return bool(_SUBSTITUTED_FETCH.search("".join(stage)))


def _stage_fetches(stage: list[str]) -> bool:
    """Does this stage download something, as its own command or inside a substitution?

    The bare name must be the stage's COMMAND. Matching it anywhere made `grep curl commands.txt`
    a download -- there the word is a search PATTERN, and the shell downstream receives locally
    selected text:  grep echo commands.txt | bash  ->  runs the local file's line, fetches nothing.

    A substitution is different: `"$(curl URL)"` is not the command word and never will be, so it
    is matched wherever it appears.
    """
    word, at = _command_word(stage)
    if at is None:
        return _stage_substitutes_a_fetch(stage)
    if _FETCH.fullmatch(word):
        return True
    # A `sh -c "curl … | bash"` program is spliced INLINE into its stage, so the stage's own
    # command word is `sh` and the fetch sits further right. Only a shell's `-c` makes what
    # follows a command: after `grep`, the word is an ARGUMENT (a search pattern), which is why
    # this recurses past `-c` rather than re-walking every remaining token.
    #   grep curl commands.txt | bash   ->  runs locally selected text, fetches nothing
    if _SHELL_NAME.fullmatch(word):
        for index, tok in enumerate(stage[at + 1 :], start=at + 1):
            if _TAKES_A_COMMAND_STRING.fullmatch(tok):
                return _stage_fetches(stage[index + 1 :])
    return _stage_substitutes_a_fetch(stage)


def _stage_runs_a_shell(tokens: list[str]) -> bool:
    """Does one pipeline stage run a shell?

    A stage may be a subshell (`curl … | ( bash )`), which runs its contents in a shell of its
    own -- so the grouping tokens are stepped over rather than read as the command. Any other
    word IS the command, and is not a shell: `grep bash` stops there, so the shell name after it
    is an argument being searched for, not the command being run.
    """
    # A GROUP is a command list of its own, and the stream reaches every command in it. Asking
    # about the flattened token soup instead would read `{ true; bash; }` as the single command
    # `true bash`, whose command word is `true` -- clean, while it executes the download.
    if _is_a_group(tokens):
        return any(_stage_runs_a_shell(cmd) for cmd in _commands_in(" ".join(tokens[1:-1])))
    # A keyword compound is a command list too, for the same reason a group is: the download stays
    # on stdin for every command inside the clause. Its command word is `if`, so without this the
    # stage reads clean while `then bash` executes the stream.
    if _is_a_compound(tokens):
        return any(_stage_runs_a_shell(cmd) for cmd in _compound_body_commands(tokens))
    inner = [t for t in tokens if t not in ("(", ")", "{", "}")]
    if _requests_a_shell_by_flag(inner) or _sources_stdin(inner) or _evals_stdin(inner):
        return True
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


# The paths that name the stage's OWN standard input. Sourcing any of these reads the piped
# download and runs it in the current shell, so the download is executed with no shell name
# anywhere in the stage. Verified for each, rather than assumed from /dev/stdin alone:
#   printf 'echo PWNED\n' | bash -c 'source /dev/fd/0'        -> PWNED
#   printf 'echo PWNED\n' | bash -c 'source /proc/self/fd/0'  -> PWNED
_STDIN_PATHS = ("/dev/stdin", "/dev/fd/0", "/proc/self/fd/0")


def _sources_stdin(tokens: list[str]) -> bool:
    """Does this stage `source` its standard input?

    `source` and `.` are shell BUILTINS, not commands, so the recursion above -- which asks
    whether a stage's command word is a shell -- never sees them. `sh -c 'source /dev/stdin'`
    therefore read as clean while executing the download outright:

        printf 'echo PWNED\\n' | bash -c 'source /dev/stdin'  ->  PWNED
        printf 'echo PWNED\\n' | bash -c '. /dev/stdin'       ->  PWNED

    Only the paths that name stdin count. Sourcing an ordinary file runs THAT file and leaves the
    pipe unread (`source ./setup.sh` prints the file's own output), and a command merely reading
    stdin as data is not executing it (`cat /dev/stdin`), so neither matches.
    """
    word, at = _command_word(tokens)
    if at is None or word not in ("source", "."):
        return False
    return any(operand in _STDIN_PATHS for operand in tokens[at + 1 :])


# Ways a command string SLURPS its own stdin: a bare `cat`, or a redirect-from-stdin
# substitution. Whatever is captured this way is the piped download.
_READS_STDIN = re.compile(r"(?:\$\(|`)\s*(?:cat\s*|<\s*/dev/stdin\s*)(?:\)|`)")


def _evals_stdin(tokens: list[str]) -> bool:
    """Does this stage `eval` something it read from stdin?

    `eval` is a builtin, so the `-c` recursion -- which asks whether a command is a shell --
    stops at it. But `eval "$(cat)"` captures the pipe and executes it:

        printf 'echo PWNED\\n' | bash -c 'eval "$(cat)"'         ->  PWNED
        printf 'echo PWNED\\n' | bash -c 'eval "$(</dev/stdin)"' ->  PWNED
        printf 'echo PWNED\\n' | bash -c 'eval "`cat`"'          ->  PWNED

    The capture must actually read stdin. `eval "echo SAFE"` runs a literal and `eval "$FOO"`
    runs a variable, neither of which touches the pipe, so both stay clean.
    """
    word, at = _command_word(tokens)
    if at is None or word != "eval":
        return False
    return any(_READS_STDIN.search(tok) for tok in tokens[at + 1 :])


def _requests_a_shell_by_flag(tokens: list[str]) -> bool:
    """Does a privilege tool start a shell that then reads the pipe?

    `sudo -s`, `sudo -i`, and a bare `su` all spawn the target user's shell, which reads the
    download -- with no shell NAME anywhere in the stage, so the walk for a command word runs
    off the end and reports clean. Confirmed rather than assumed:

        printf 'echo PWNED\\n' | sudo -n -s   -> PWNED
        printf 'echo PWNED\\n' | sudo -n -i   -> PWNED
        printf 'echo PWNED\\n' | sudo -n -ns  -> PWNED   (clustered)
        printf 'echo PWNED\\n' | sudo -n su   -> PWNED   (su's default IS a shell)

    Three things stop it from firing:

    A command AFTER the tool is what actually runs, so the stream is untouched:
        printf 'echo X\\n' | sudo -n -s echo hi  ->  hi

    `-s`/`--shell` on `su`/`runuser` CHOOSES the program, and it need not be a shell:
        printf 'hello\\n' | sudo -n su -s /usr/bin/sha256sum nobody  ->  5891b5b5…  (hashed)

    And `-c` hands that chosen shell a program, exactly as `sh -c` does, which leaves the stream
    as data for it -- the verified-download pattern this guard exists to encourage:
        printf 'echo PWNED\\n' | sudo -n su -s /bin/sh -c 'echo RAN_C'  ->  RAN_C only

    The stage is reached through `_command_word`, so wrappers in front are stepped over:
    `env sudo -s` and `nice sudo -s` both execute the download and both match.
    """
    at = _privilege_tool_at(tokens)
    if at is None:
        return False
    tool = tokens[at]
    flags = tokens[at + 1 :]
    chosen = _selected_program(tool.rsplit("/", 1)[-1], flags)
    if chosen is not None:
        # An explicitly chosen program replaces the question: ask about THAT stage instead, so a
        # chosen shell still defers to its own `-c` and a chosen non-shell stays clean.
        return _stage_runs_a_shell([chosen, *_after_the_selection(flags)])
    # A command AFTER the tool is what runs, so the default shell never starts and the stream
    # stays untouched: `sudo -s echo hi` prints hi, and `runuser -u root sha256sum` hashes.
    # The positional USER that `su`/`runuser` accept is not such a command, so it is stepped over.
    if _command_after_the_tool(tool.rsplit("/", 1)[-1], flags):
        return False
    # `-c` hands the default shell a program, exactly as `sh -c` does, so the stream becomes data
    # for it -- unless that program is itself a shell (`su -c 'sh'`).
    program = _dash_c_operand([tool, *flags])
    if program is not None:
        return any(_stage_runs_a_shell(cmd) for cmd in _commands_in(program))
    # Nothing else runs, so the tool starts its default. `su`/`runuser` default to the user's
    # login SHELL; `sudo` needs a flag to ask for one, since a bare `sudo` is a usage error.
    if tool.rsplit("/", 1)[-1] in ("su", "runuser"):
        return True
    return any(
        flag in ("--shell", "--login")
        or (flag.startswith("-") and not flag.startswith("--") and {"s", "i"} & set(flag[1:]))
        for flag in flags
    )


def _command_after_the_tool(tool: str, flags: list[str]) -> bool:
    """Does a real command follow the privilege tool, replacing its default shell?

    `su`/`runuser` take a positional USER first, which is not a command -- `su root` still starts
    root's shell on the pipe. So for those the FIRST bare word is the user and only a SECOND one
    is the command; for `sudo` every bare word is already the command.
    """
    _, at = _command_word(flags)
    if at is None:
        return False
    if tool not in ("su", "runuser"):
        return True
    return _command_word(flags[at + 1 :])[1] is not None


def _is_a_group(tokens: list[str]) -> bool:
    """Is this whole stage a `( … )` or `{ …; }` group, rather than a single command?

    Only when the grouping tokens ENCLOSE the stage. A `)` in the middle belongs to something
    else, and treating the stage as a group would re-split it wrongly.
    """
    return len(tokens) > 2 and (tokens[0], tokens[-1]) in (("(", ")"), ("{", "}"))


def _is_a_compound(tokens: list[str]) -> bool:
    """Does this stage START with a compound keyword (`if`, `while`, `for`, `case`, …)?

    Keyed on the opener alone, not on a matching closer: a `RUN` line continued across several
    physical lines can put `fi` out of reach, and a clause whose body runs a shell executes the
    download whether or not this scanner can see the end of it.
    """
    return bool(tokens) and tokens[0] in _COMPOUND_OPENERS


def _compound_body_commands(tokens: list[str]) -> list[list[str]]:
    """The commands inside a keyword compound, with the syntax words removed.

    A `case` BRANCH is `pattern)` followed by its commands, and the pattern is matched text rather
    than anything that runs. Dropping only the keywords left `case x in x) bash` reading as the
    command `x`, so the body's real command was never reached and the clause scanned clean while
    executing the download. Everything up to and including each `)` is therefore discarded.
    """
    return _commands_in(_compound_body_text(tokens))


def _nested_command_lists(command: str) -> list[str]:
    """The interior of every group and keyword clause in this command, as shell text.

    Returned as text rather than token lists so the caller re-enters the whole pipeline analysis
    on it: what is inside a clause is an ordinary command list, and may contain its own fetch,
    its own pipe, and its own nested clause. Recursion terminates because each step strips at
    least the enclosing construct's own tokens.
    """
    inner: list[str] = []
    for stage in _commands_in(command):
        if _is_a_group(stage):
            inner.append(" ".join(stage[1:-1]))
        elif _is_a_compound(stage):
            # The body as ONE text, not its commands joined back together: `_compound_body_commands`
            # has already split across the `|`, so re-joining those pieces loses the pipe and the
            # fetch inside the clause stops flowing into the shell inside the clause.
            inner.append(_compound_body_text(stage))
    return inner


def _compound_body_text(tokens: list[str]) -> str:
    """A keyword compound's interior, with the syntax words and `case` labels removed."""
    body = [
        tok
        for tok in tokens
        if tok not in _COMPOUND_OPENERS | _COMPOUND_CLOSERS | _COMPOUND_INTERNALS
    ]
    if tokens and tokens[0] == "case":
        body = _drop_case_patterns(body)
    return " ".join(body)


def _drop_case_patterns(body: list[str]) -> list[str]:
    """Remove each `pattern)` label from a `case` body, keeping the commands it guards.

    The subject word (`case SUBJECT in …`) is dropped by the same rule: it sits before the first
    `)` and is a value being matched, never a command.
    """
    kept: list[str] = []
    pending: list[str] = []
    for tok in body:
        if tok == ")":
            pending = []
            continue
        if tok in (";", "&&", "||", "|", "&"):
            kept.extend(pending)
            kept.append(tok)
            pending = []
            continue
        pending.append(tok)
    kept.extend(pending)
    return kept


def _privilege_tool_at(tokens: list[str], wrapper: str = "") -> int | None:
    """Where the privilege tool sits, stepping over any wrappers in front of it.

    `_command_word` cannot answer this. `sudo` is itself an exec wrapper, so that walk steps
    straight over it and returns whatever it runs -- `("", None)` for a bare `sudo -s`, which is
    exactly the case that executes the download. And for `runuser root` it returns the positional
    USER, which is not a program at all.

    Walking to the tool rather than testing `tokens[0]` is what lets a wrapper sit in front:
    `env sudo -s` and `nice sudo -s` both execute the pipe.
    """
    skip_next = False
    for index, tok in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        name = tok.rsplit("/", 1)[-1]
        if name in _PRIVILEGE_TOOLS:
            # These CHAIN, and the LAST one decides. `sudo su` is the ordinary spelling, and
            # stopping at `sudo` reads `su` as an ordinary command that replaces the shell --
            # while it starts one:  printf 'echo PWNED\n' | sudo -n su  ->  PWNED
            # The tool becomes the wrapper for what follows, so its OWN flag arities are known:
            # `sudo -u root su` must read `root` as -u's operand, not as a command.
            later = _privilege_tool_at(tokens[index + 1 :], name)
            return index + 1 + later if later is not None else index
        if tok.startswith(("-", "+")):
            # A flag may take a SEPARATE operand, and that operand is not a command. Without
            # this, `env -u LANG sudo -s` stopped at `LANG` and never reached sudo -- while it
            # executes the download:  printf 'echo PWNED\n' | env -u LANG sudo -n -s  ->  PWNED
            skip_next = "=" not in tok and _takes_a_separate_operand(wrapper, tok)
            continue
        # Assignments and redirections may PRECEDE the tool, exactly as they may precede any
        # command: `MODE=x su` and `2>/dev/null su` both start a shell on the pipe.
        if re.fullmatch(r"[A-Za-z_]\w*=.*", tok):
            continue
        if re.fullmatch(r"\d*[<>]{1,2}.*", tok):
            skip_next = tok.rstrip().endswith((">", "<"))
            continue
        # A wrapper's own numeric operand is not a command either: `timeout 30 sudo -s` and
        # `nice -n 5 sudo -s` both reach sudo. Same rule the command-word walk applies.
        if wrapper and _NUMERIC_OPERAND.fullmatch(tok):
            continue
        if name not in _EXEC_WRAPPERS:
            return None
        wrapper = name
    return None


def _selected_program(tool: str, flags: list[str]) -> str | None:
    """The program `su`/`runuser` was told to start via `-s`/`--shell`, if any.

    `sudo -s` takes NO operand -- it means "a shell", not "this shell" -- so it never selects.
    """
    if tool not in ("su", "runuser"):
        return None
    for flag, following in itertools.zip_longest(flags, flags[1:], fillvalue=""):
        if flag in _SHELL_SELECTING_FLAGS:
            return following or None
        if flag.startswith("--shell="):
            return flag.split("=", 1)[1] or None
    return None


def _after_the_selection(flags: list[str]) -> list[str]:
    """The flags that still apply to the selected program -- `-c` and its operand.

    Everything else (`--login`, a positional user) belongs to the privilege tool, not to the
    program it starts, and must not be handed on as if the shell had received it.
    """
    for index, flag in enumerate(flags):
        if _TAKES_A_COMMAND_STRING.fullmatch(flag):
            return flags[index:]
    return []


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
