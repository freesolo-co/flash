"""No file in this repo may pipe a downloaded script into a shell.

`curl … | bash` executes whatever the vendor serves at request time: it cannot be reviewed,
cannot be reproduced, and pins nothing. This module enforces that rule across every file that
may install software from the network.

The scan discovers its own inputs. An enumerated list looks identical to a complete one right up
until someone adds a file.

The shell grammar this relies on -- what counts as a pipe into a shell, and what is an ordinary
pipeline that must stay clean -- lives in `pipe_to_shell_grammar`, and the scan surface it runs
over -- which files, and how continued lines are joined -- lives in `pipe_to_shell_scan`. The
tests below are the specification for both: each case records a spelling confirmed by running it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.pipe_to_shell_grammar import _piped_into_a_shell
from tests.pipe_to_shell_scan import (
    INSTALLER_FILES,
    _logical_lines,
)


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
        # Scheduling and privilege wrappers exec their operand like any other. Confirmed by
        # running each one rather than inferred from the man page:
        #   printf 'echo X\n' | nice sh   ->   X
        pytest.param("RUN curl -sSL https://x.example/i.sh | nice bash\n", id="nice-wrapper"),
        # `su` starts the target user's DEFAULT shell, with stdin intact, when nothing else is
        # named. Its positional operand is a USER, not a command, so `su root` still runs a shell.
        # Confirmed as root so no password prompt intervenes:
        #   printf 'echo PWNED\n' | sudo -n su       ->   PWNED
        #   printf 'echo PWNED\n' | sudo -n su root  ->   PWNED
        pytest.param("RUN curl -sSL https://x.example/i.sh | su\n", id="su-is-a-shell"),
        pytest.param("RUN curl -sSL https://x.example/i.sh | su -\n", id="su-login-shell"),
        pytest.param("RUN curl -sSL https://x.example/i.sh | su root\n", id="su-named-user"),
        # `-s` picks WHICH program to start. Here it names a shell, so the download is executed;
        # the counter-cases below pin the spellings where it names something else.
        pytest.param("RUN curl -sSL https://x.example/i.sh | su -s /bin/sh\n", id="su-shell-flag"),
        # A FLAG can start a shell with no shell name present at all. The walk steps over `sudo`
        # and the flag, finds no command, and without this would read as clean:
        #   printf 'echo PWNED\n' | sudo -n -s   ->   PWNED
        pytest.param("RUN curl -sSL https://x.example/i.sh | sudo -s\n", id="sudo-dash-s"),
        pytest.param("RUN curl -sSL https://x.example/i.sh | sudo -i\n", id="sudo-dash-i"),
        pytest.param("RUN curl -sSL https://x.example/i.sh | sudo -ns\n", id="sudo-clustered-s"),
        # The long spellings of the same two flags, confirmed to execute the pipe as well:
        #   printf 'echo PWNED\n' | sudo -n --login   ->   PWNED
        pytest.param("RUN curl -sSL https://x.example/i.sh | sudo --login\n", id="sudo-long-login"),
        pytest.param("RUN curl -sSL https://x.example/i.sh | sudo --shell\n", id="sudo-long-shell"),
        # `su -c` behaves exactly like `sh -c`: a shell operand still reads the stream.
        pytest.param("RUN curl -sSL https://x.example/i.sh | su -c 'sh'\n", id="su-dash-c-shell"),
        pytest.param("RUN curl -sSL https://x.example/i.sh | setsid sh\n", id="setsid-wrapper"),
        pytest.param("RUN curl -sSL https://x.example/i.sh | ionice -c2 sh\n", id="ionice-wrapper"),
        # These two REQUIRE an operand, so a bare `chrt sh` is a usage error rather than proof
        # they cannot reach a shell -- the working spelling does.
        pytest.param("RUN curl -sSL https://x.example/i.sh | chrt -o 0 sh\n", id="chrt-wrapper"),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | taskset -c 0 bash\n", id="taskset-wrapper"
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | setpriv --reuid 0 sh\n", id="setpriv-wrapper"
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | runuser -u root sh\n", id="runuser-wrapper"
        ),
        # `xargs` usually turns the stream into ARGV, which is harmless -- but three spellings
        # feed it back somewhere the shell executes. Each confirmed by running it:
        #   printf 'echo PWNED\n' | xargs -0 sh -c          ->  PWNED   (empty -c slot)
        #   printf 'echo PWNED\n' | xargs -I{} sh -c '{}'   ->  PWNED   (interpolated)
        #   printf 'echo PWNED\n' | xargs -a f sh           ->  PWNED   (pipe left on stdin)
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | xargs -0 sh -c\n", id="xargs-fills-the-c-slot"
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | xargs -I{} sh -c '{}'\n",
            id="xargs-interpolates-into-c",
        ),
        # `-i` takes an OPTIONAL operand, so a following OPTION is not it. Reading `-0` as the
        # placeholder left nothing matching `{}` in the program, and the match was lost:
        #   printf 'echo PWNED\n' | xargs -i -0 sh -c '{}'   ->   PWNED
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | xargs -i -0 sh -c '{}'\n",
            id="xargs-i-does-not-eat-the-next-option",
        ),
        # A custom placeholder, fused. `-I` was handled and `-i` was not, so this spelling fell
        # through to a default that no longer exists.
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | xargs -iQQ sh -c 'echo got QQ'\n",
            id="xargs-fused-i-custom-placeholder",
        ),
        # `-i` at the end of a short-flag CLUSTER. `-0i` is `-0` plus `-i`, and everything after
        # the letter is the placeholder -- which is why `-in1` means the name `n1`, not `-n 1`.
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | xargs -0i sh -c '{}'\n",
            id="xargs-clustered-i",
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | xargs --replace sh -c '{}'\n",
            id="xargs-long-replace-defaults",
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | xargs -a args.txt bash\n",
            id="xargs-arg-file-leaves-stdin",
        ),
        # The same instruction in its other spellings: a clustered short flag, and the long form
        # with a separate operand. Both were missed while `-a f` matched.
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | xargs -0a /dev/null sh\n",
            id="xargs-clustered-arg-file",
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | xargs --arg-file args.txt bash\n",
            id="xargs-long-arg-file",
        ),
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
        # A wrapper in FRONT of a flag-spawned shell. The check used to test `tokens[0]`, so any
        # wrapper hid it -- and these all execute the download:
        #   printf 'echo PWNED\n' | env sudo -n -s   ->  PWNED
        #   printf 'echo PWNED\n' | nice sudo -n -s  ->  PWNED
        pytest.param("RUN curl -sSL https://x.example/i.sh | env sudo -s\n", id="wrapped-sudo-s"),
        pytest.param("RUN curl -sSL https://x.example/i.sh | nice sudo -s\n", id="nice-sudo-s"),
        # The wrapper's own NUMERIC operand is not a command, so the walk must step over it.
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | timeout 30 sudo --login\n",
            id="timeout-operand-then-sudo",
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | nice -n 5 sudo -s\n",
            id="nice-operand-then-sudo",
        ),
        # `unshare` runs its program with namespaces unshared and stdin untouched. Confirmed:
        #   sudo -n unshare bash < payload       ->  ran
        #   sudo -n unshare --fork sh < payload  ->  ran
        pytest.param("RUN curl -sSL https://x.example/i.sh | unshare bash\n", id="unshare-wrapper"),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | unshare --fork sh\n", id="unshare-fork"
        ),
        # Only a few unshare flags take a SEPARATE operand; the namespace ones fuse (`--mount=f`).
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | unshare -S 0 bash\n", id="unshare-setuid"
        ),
        # `|&` is ONE operator (pipe stdout AND stderr). Tokenized as `|` then `&`, the `&` ended
        # the pipeline before the shell and the line read as clean:
        #   cat payload |& bash  ->  ran
        pytest.param("RUN curl -sSL https://x.example/i.sh |& bash\n", id="pipe-both-streams"),
        # `eval` is a builtin, so the `-c` recursion stops at it -- but a capture of stdin is the
        # download, and eval executes it. Each spelling confirmed:
        #   printf 'echo PWNED\n' | bash -c 'eval "$(cat)"'          ->  PWNED
        #   printf 'echo PWNED\n' | bash -c 'eval "$(</dev/stdin)"'  ->  PWNED
        #   printf 'echo PWNED\n' | bash -c 'eval "`cat`"'           ->  PWNED
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | bash -c 'eval \"$(cat)\"'\n", id="eval-cat"
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | bash -c 'eval \"$(</dev/stdin)\"'\n",
            id="eval-redirect-stdin",
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | bash -c 'eval \"`cat`\"'\n",
            id="eval-backtick-cat",
        ),
        # Privilege tools CHAIN, and the last one decides. `sudo su` is the ordinary spelling;
        # stopping at `sudo` reads `su` as a command that replaces the shell, while it starts one:
        #   printf 'echo PWNED\n' | sudo -n su           ->  PWNED
        #   printf 'echo PWNED\n' | sudo -n env MODE=x su ->  PWNED
        pytest.param("RUN curl -sSL https://x.example/i.sh | sudo su\n", id="sudo-then-su"),
        pytest.param("RUN curl -sSL https://x.example/i.sh | sudo su -\n", id="sudo-then-login-su"),
        # The first tool's own flag operand must not be read as that command.
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | sudo -u root su\n", id="sudo-user-then-su"
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | sudo runuser -u root sh\n",
            id="sudo-then-runuser",
        ),
        # Assignments and redirections may precede the tool, as they may precede any command.
        pytest.param("RUN curl -sSL https://x.example/i.sh | MODE=x su\n", id="assignment-then-su"),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | 2>/dev/null su\n", id="redirection-then-su"
        ),
        # A wrapper FLAG's own operand is not a command either. `LANG` is not numeric, so the
        # walk stopped on it and never reached sudo -- while the line executes the download:
        #   printf 'echo PWNED\n' | env -u LANG sudo -n -s     ->  PWNED
        #   printf 'echo PWNED\n' | timeout -k 5 30 sudo -n -s ->  PWNED
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | env -u LANG sudo -s\n", id="env-unset-then-sudo"
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | timeout -k 5 30 sudo -s\n",
            id="timeout-kill-after-then-sudo",
        ),
        # `--shell=` FUSED. Exact-matching `--shell` missed it, and the selected program is a
        # shell:  printf 'echo PWNED\n' | sudo -n runuser --shell=/bin/sh root  ->  PWNED
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | runuser --shell=/bin/sh root\n",
            id="runuser-fused-shell",
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | su -s /bin/bash root\n", id="su-selects-a-shell"
        ),
        # A group is a command LIST, and the stream reaches every command in it. The `;` inside
        # used to end the whole pipeline, detaching the shell from the fetch:
        #   printf 'echo PWNED\n' | { true; bash; }  ->  PWNED
        #   printf 'echo PWNED\n' | (true; bash)     ->  PWNED
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | { true; bash; }\n", id="brace-group-list"
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | (true; bash)\n", id="subshell-group-list"
        ),
        # The fetch may be captured by COMMAND SUBSTITUTION before being piped. It arrives inside
        # a token rather than as one, so a whole-token match never fired. Each spelling runs:
        #   printf '%s' "$(cat p.sh)" | bash   ->  ran
        #   echo "$(cat p.sh)" | bash          ->  ran
        #   printf '%s' "`cat p.sh`" | bash    ->  ran
        pytest.param(
            'RUN printf "%s" "$(curl -fsSL https://x.example/i.sh)" | bash\n',
            id="substituted-fetch",
        ),
        pytest.param(
            'RUN echo "$(curl -fsSL https://x.example/i.sh)" | bash\n', id="substituted-fetch-echo"
        ),
        pytest.param(
            'RUN printf "%s" "`curl -fsSL https://x.example/i.sh`" | sh\n',
            id="substituted-fetch-backticks",
        ),
        pytest.param(
            'RUN printf "%s" "$(wget -qO- https://x.example/i.sh)" | bash\n',
            id="substituted-fetch-wget",
        ),
        # The same substitution UNQUOTED. Nothing keeps it in one token here: the tokenizer's
        # fallback alternation excludes `(` and `)`, so this arrives as `echo`, `$`, `(`, `curl`,
        # `)` and no single token carries both `$(` and the fetch name. Requiring the fetch to be
        # the stage's command word (which cleared `grep curl … | bash`) left only the token-wise
        # substitution match, which never fired for this spelling. Confirmed to execute:
        #   echo $(cat p.sh) | bash   ->  ran
        pytest.param(
            "RUN echo $(curl -fsSL https://x.example/i.sh) | bash\n",
            id="unquoted-substituted-fetch",
        ),
        pytest.param(
            "RUN echo $(wget -qO- https://x.example/i.sh) | sh\n",
            id="unquoted-substituted-fetch-wget",
        ),
        pytest.param(
            "RUN echo $( curl -fsSL https://x.example/i.sh ) | bash\n",
            id="unquoted-substituted-fetch-spaced",
        ),
        pytest.param(
            "RUN echo `curl -fsSL https://x.example/i.sh` | bash\n",
            id="unquoted-substituted-fetch-backticks",
        ),
        # A compound command delimited by KEYWORDS is a command list too, and the download is on
        # its stdin for every command inside it -- exactly like the `{ …; }` group above. `if` and
        # `fi` are words, so the depth counter never saw them and the clause's own `;` split the
        # pipeline, detaching the shell from the fetch. Confirmed to execute:
        #   printf 'echo PWNED\n' | if true; then bash; fi   ->  PWNED
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | if true; then bash; fi\n",
            id="if-clause-stage",
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | while read -r l; do bash; done\n",
            id="while-clause-stage",
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | for i in 1; do bash; done\n",
            id="for-clause-stage",
        ),
        # `case` needs one thing more than the other compounds. Its branch labels end in `)` with
        # no opening `(`, so counting that as a group closer unbalanced the depth and the `;;`
        # split the pipeline anyway; and once the depth was fixed, the branch PATTERN was still
        # read as the body's command. Confirmed to execute:
        #   printf 'echo PWNED\n' | case x in x) bash;; esac   ->  PWNED
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | case x in x) bash;; esac\n",
            id="case-clause-stage",
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | case $x in a) jq .;; *) bash;; esac\n",
            id="case-later-branch",
        ),
        # The WHOLE pipeline may sit inside the group or clause, rather than being piped into it.
        # Then the fetch never appears at the top level at all -- the enclosing construct is one
        # stage, and a scan that only walks top-level stages sees a single command with no pipe in
        # it. Every spelling runs the download:
        #   ( cat p.sh | bash )                  ->  ran
        #   if true; then cat p.sh | bash; fi    ->  ran
        pytest.param(
            "RUN ( curl -sSL https://x.example/i.sh | bash )\n", id="pipeline-inside-a-subshell"
        ),
        pytest.param(
            "RUN { curl -sSL https://x.example/i.sh | bash; }\n", id="pipeline-inside-a-brace-group"
        ),
        pytest.param(
            "RUN if true; then curl -sSL https://x.example/i.sh | bash; fi\n",
            id="pipeline-inside-an-if",
        ),
        pytest.param(
            "RUN while true; do curl -sSL https://x.example/i.sh | bash; done\n",
            id="pipeline-inside-a-while",
        ),
        pytest.param(
            "RUN case x in x) curl -sSL https://x.example/i.sh | bash;; esac\n",
            id="pipeline-inside-a-case",
        ),
        pytest.param(
            "RUN if true; then if true; then curl -sSL https://x.example/i.sh | bash; fi; fi\n",
            id="pipeline-inside-nested-clauses",
        ),
        pytest.param(
            "RUN if true; then ( curl -sSL https://x.example/i.sh | bash ); fi\n",
            id="pipeline-inside-a-group-inside-a-clause",
        ),
        # The shapes a real vendor install actually takes. A conditional guard around the fetch is
        # the ordinary way to write "install it if it is not already here", so these are the
        # spellings most likely to appear in a Dockerfile -- and each was invisible while a clause
        # counted as one opaque stage.
        pytest.param(
            'RUN set -eux; if [ "$INSTALL" = "true" ]; then'
            " curl -fsSL https://x.example/i.sh | bash; fi\n",
            id="guarded-vendor-install",
        ),
        pytest.param(
            "RUN if [ ! -f /usr/bin/tool ]; then curl -fsSL https://x.example/i.sh | sh -; fi\n",
            id="install-when-absent",
        ),
        pytest.param(
            'RUN for v in 1 2; do curl -fsSL "https://x.example/$v.sh" | bash; done\n',
            id="install-in-a-loop",
        ),
        # Dockerfile EXEC form. Docker runs the array as argv with no shell of its own, but here
        # argv[0] IS a shell and argv[2] is the command it runs, so the download is executed just
        # as in the shell form. The JSON quoting is FILE syntax; leaving it on made the whole
        # command arrive as one token with no pipeline visible inside it.
        pytest.param(
            'RUN ["bash", "-c", "curl -sSL https://x.example/i.sh | bash"]\n',
            id="exec-form-run",
        ),
        pytest.param(
            'RUN ["/bin/sh", "-c", "curl -sSL https://x.example/i.sh | sh"]\n',
            id="exec-form-absolute-path",
        ),
        pytest.param(
            'ENTRYPOINT ["bash", "-c", "curl -sSL https://x.example/i.sh | bash"]\n',
            id="exec-form-entrypoint",
        ),
        pytest.param(
            'CMD ["sh", "-c", "curl -sSL https://x.example/i.sh | sh"]\n', id="exec-form-cmd"
        ),
        # A workflow `run:` value may be a QUOTED scalar rather than a bare or block one. The
        # quotes are YAML syntax, so they come off at the same boundary as the keyword.
        pytest.param('run: "curl -sSL https://x.example/i.sh | bash"\n', id="quoted-scalar-double"),
        pytest.param("run: 'curl -sSL https://x.example/i.sh | bash'\n", id="quoted-scalar-single"),
        # `source` and `.` are BUILTINS, so no shell name appears as a command word anywhere in
        # the stage -- yet sourcing stdin reads the piped download and runs it. Each spelling was
        # confirmed by running it, not inferred from /dev/stdin:
        #   printf 'echo PWNED\n' | bash -c 'source /dev/stdin'       -> PWNED
        #   printf 'echo PWNED\n' | bash -c '. /dev/stdin'            -> PWNED
        #   printf 'echo PWNED\n' | bash -c 'source /dev/fd/0'        -> PWNED
        #   printf 'echo PWNED\n' | bash -c 'source /proc/self/fd/0'  -> PWNED
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | bash -c 'source /dev/stdin'\n",
            id="source-stdin",
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | bash -c '. /dev/stdin'\n", id="dot-stdin"
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | sh -c '. /dev/stdin'\n", id="sh-dot-stdin"
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | bash -c 'source /dev/fd/0'\n",
            id="source-dev-fd-zero",
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | bash -c 'source /proc/self/fd/0'\n",
            id="source-proc-self-fd-zero",
        ),
        # The operand is a command LIST, so the source need not be the first command in it.
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | bash -c 'true; source /dev/stdin'\n",
            id="source-stdin-second-command",
        ),
        # `-s` alongside `-c` is SHELL-DEPENDENT, and this case pins the shell where it executes.
        # Confirmed by running both, rather than reasoning from the flags:
        #   printf 'echo PWNED\n' | sh   -s -c 'echo RAN_C'  ->  RAN_C then PWNED  <-- executed
        #   printf 'echo PWNED\n' | bash -s -c 'echo RAN_C'  ->  RAN_C only
        # `/bin/sh` is dash on Debian and Ubuntu, which is where a Dockerfile RUN lands, so this
        # spelling really does execute the download. The bash spelling is a legitimate one and is
        # pinned as a counter-case below; an earlier version of this test asserted the bash form
        # was dangerous, which is the opposite of what bash does.
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | sh -s -c 'jq .'\n", id="dash-s-beats-dash-c"
        ),
        # A MULTI-WORD `-c` body. The operand is re-tokenized and walked like any other stage, so
        # the shell is found behind whatever precedes it -- judging the body by its first word
        # would miss every one of these.
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | sh -c 'exec bash'\n", id="dash-c-body-exec"
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | sh -c 'env bash'\n", id="dash-c-body-wrapper"
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | sh -c 'busybox ash'\n", id="dash-c-body-busybox"
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | sh -c 'timeout 30 sh'\n",
            id="dash-c-body-numeric-operand",
        ),
        # A `-c` program is a command LIST, not one command. Resolving only its first command
        # stopped at `true` and let the shell behind the separator through.
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | sh -c 'true; bash'\n", id="dash-c-body-list"
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | sh -c 'true && bash'\n",
            id="dash-c-body-list-and",
        ),
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | sh -c 'echo x | bash'\n",
            id="dash-c-body-inner-pipe",
        ),
        # A subshell written TIGHT. `( bash )` was already pinned, but the parentheses were not
        # tokenizer operators, so `(bash)` stayed one word and the grouping-strip never saw it.
        pytest.param("RUN curl -sSL https://x.example/i.sh | (bash)\n", id="tight-subshell"),
        pytest.param("RUN curl -sSL https://x.example/i.sh | (sh)\n", id="tight-subshell-sh"),
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


@pytest.mark.parametrize(
    "step",
    [
        pytest.param(
            "      - run: if true; then curl -fsSL https://x.example/i.sh | bash; fi\n",
            id="workflow-bare-scalar",
        ),
        pytest.param(
            '      - run: "if true; then curl -fsSL https://x.example/i.sh | bash; fi"\n',
            id="workflow-quoted-scalar",
        ),
        pytest.param(
            "      - name: install\n"
            "        run: if true; then curl -fsSL https://x.example/i.sh | bash; fi\n",
            id="workflow-named-step",
        ),
    ],
)
def test_the_guard_catches_a_conditional_installer_in_a_workflow(step: str, tmp_path: Path):
    """A clause reaches the guard through the YAML path as readily as the Dockerfile one.

    Worth pinning separately: the `run:` value arrives after keyword stripping and, in the quoted
    form, after unwrapping a scalar -- so a compound that is handled in a `RUN` line is not
    thereby proven handled here. This is also the shape a workflow actually uses to install a
    tool only when it is missing.
    """
    planted = tmp_path / "wf.yml"
    planted.write_text("jobs:\n  j:\n    steps:\n" + step)

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
