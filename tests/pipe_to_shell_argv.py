"""How a wrapper's flags decide whether the piped stream reaches a shell as INPUT or as ARGV.

Split out of `pipe_to_shell_grammar` because it answers a different question. That module decides
which token is the command; this one decides what the stream BECOMES once a wrapper has had its
say -- `xargs` reading the pipe itself to build an argument list, `sh -c` taking its program from
an operand, `env -S` splitting one. The distinction is the whole game for `xargs`: the same
`curl … | xargs sh` is harmless (the shell gets a filename) while `xargs sh -c` executes the
download, and only the flags tell the two apart.

Every claim here was confirmed by running the command against GNU findutils, not read off a man
page; the docstrings record the transcripts.
"""

from __future__ import annotations

import itertools
import re

# a flag whose operand is the shell's program: `sh -c '…'`, `bash -cxe '…'`,
# `su --command '…'`. `c` may sit anywhere in a short-option cluster; the operand parser below
# skips values taken by other options before locating the command string.
_TAKES_A_COMMAND_STRING = re.compile(r"-(?=[A-Za-z]*c)[A-Za-z]+|--command")

# shell invocation options that consume the next argv word before a script or `-c` program. keeping
# these here prevents an operand such as the `c` in `bash -o c verify.sh` from becoming an option.
_SHELL_OPTIONS_WITH_OPERAND = {"o", "O"}
_LONG_SHELL_OPTIONS_WITH_OPERAND = {"--init-file", "--rcfile"}

# `env -S 'bash -s'` splits its operand into words and runs them, so the operand is not a value to
# skip -- it is the command. The fused spelling puts it in the same token: `-S'bash -s'`.
_SPLITS_ITS_OPERAND = {"-S", "--split-string"}
_FUSED_SPLIT_STRING = re.compile(r"(?:-S|--split-string=)(?P<command>.+)")


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


def _shell_option_state(tokens: list[str]) -> tuple[int, bool, bool]:
    """Return the first operand index, whether `-c` appeared, and whether `-s` appeared.

    Bash parses a whole short-option cluster before taking operands. Confirmed:
        bash -cxe 'echo X'                  -> X
        bash -co pipefail 'echo X'          -> X
        bash -o c verify.sh                 -> `c` is the option name, not `-c`

    The `-o`/`-O` values therefore move the eventual `-c` program to the right, while a `c` in
    one of those separate values is data. Parsing the option prefix once keeps both questions in
    agreement: which string `-c` runs, and whether a script-file operand replaces stdin.
    """
    index = 1
    takes_command = False
    reads_stdin = False
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            index += 1
            break
        if token == "-" or not token.startswith(("-", "+")):
            break
        if token in _LONG_SHELL_OPTIONS_WITH_OPERAND:
            index += 2
            continue
        if token.startswith("--"):
            takes_command |= token == "--command"
            index += 1
            continue
        options = token[1:]
        takes_command |= "c" in options
        reads_stdin |= "s" in options
        index += 1 + sum(option in _SHELL_OPTIONS_WITH_OPERAND for option in options)
    return index, takes_command, reads_stdin


def _command_string_operand_index(tokens: list[str]) -> int | None:
    """The argv index of a shell's `-c` program, including clustered option operands."""
    index, takes_command, _ = _shell_option_state(tokens)
    return index if takes_command else None


def _dash_c_operand(tokens: list[str]) -> str | None:
    """The program string a shell's `-c` supplies, or None when the shell has no `-c`.

    Returns the operand even when it is empty, so `sh -c ''` is distinguishable from a shell
    invoked with no `-c` at all -- the first runs nothing, the second runs the piped stream.

    When BOTH `-s` and `-c` appear the answer is SHELL-DEPENDENT, so it cannot be read off the
    flags alone. Confirmed by running each:

        printf 'echo FROM_STDIN\\n' | bash       -s -c 'echo FROM_C'  -> FROM_C
        printf 'echo FROM_STDIN\\n' | busybox ash -s -c 'echo FROM_C' -> FROM_C
        printf 'echo FROM_STDIN\\n' | dash       -s -c 'echo FROM_C'  -> FROM_C
                                                                         FROM_STDIN  <-- executed

    bash and busybox ash run the `-c` program and leave the pipe as data. dash runs the program
    AND THEN reads stdin, so the download is executed. `/bin/sh` IS dash on Debian and Ubuntu --
    where a Dockerfile `RUN` lands -- so `sh -s -c '…'` is a real evasion, while `bash -s -c '…'`
    is a legitimate verified-download spelling that must stay clean.
    """
    if _reads_stdin_despite_dash_c(tokens):
        return None
    index = _command_string_operand_index(tokens)
    if index is None:
        return None
    return tokens[index] if index < len(tokens) else ""


def _shell_reads_stdin(tokens: list[str]) -> bool:
    """Does a shell invocation take its program from stdin rather than a script operand?

    A non-option operand is a SCRIPT FILE, so `bash verify.sh` runs that file and leaves the pipe as
    data. A bare shell, `-s`, and the special `-` operand all still read commands from stdin.
    """
    if _reads_stdin_despite_dash_c(tokens):
        return True
    index, takes_command, reads_stdin = _shell_option_state(tokens)
    if takes_command:
        return False
    if reads_stdin or index >= len(tokens):
        return True
    return tokens[index] == "-"


# Shells confirmed to IGNORE stdin once `-c` supplies the program, even with `-s` also given.
# dash is deliberately absent: it runs the `-c` program and THEN executes stdin, and it is what
# `/bin/sh` points at on Debian and Ubuntu.
_IGNORES_STDIN_GIVEN_DASH_C = ("bash", "ash", "busybox")


def _reads_stdin_despite_dash_c(tokens: list[str]) -> bool:
    """Would this shell still execute the pipe even though `-c` gave it a program?

    Only when `-s` is present AND the shell is not one of the few confirmed to ignore stdin.
    Unknown shell names are treated as reading stdin, so a miss is a false POSITIVE (a CI failure
    someone investigates) rather than a silent hole.
    """
    _, takes_command, reads_stdin = _shell_option_state(tokens)
    if not (takes_command and reads_stdin):
        return False
    name = tokens[0].rsplit("/", 1)[-1] if tokens else ""
    return name not in _IGNORES_STDIN_GIVEN_DASH_C


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
