"""No file in this repo may pipe a downloaded script into a shell.

`curl … | bash` executes whatever the vendor serves at request time: it cannot be reviewed,
cannot be reproduced, and pins nothing. This module enforces that rule across every file that
may install software from the network.

The scan discovers its own inputs. An enumerated list looks identical to a complete one right up
until someone adds a file.

The shell grammar this relies on -- what counts as a pipe into a shell, and what is an ordinary
pipeline that must stay clean -- lives in `pipe_to_shell_grammar`. The tests below are its
specification: each case records a spelling that was confirmed by running it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.pipe_to_shell_grammar import (
    INSTALLER_FILES,
    REPO_ROOT,
    _installer_files,
    _logical_lines,
    _piped_into_a_shell,
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
        # `-s` makes the shell read its program from stdin -- the piped download -- so it wins
        # over the `-c` operand rather than deferring to it.
        pytest.param(
            "RUN curl -sSL https://x.example/i.sh | bash -s -c 'jq .'\n", id="dash-s-beats-dash-c"
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
        # `xargs` reads the pipe ITSELF to build an argument list and gives its child /dev/null
        # on stdin, so the download arrives as a FILENAME the shell tries to open, not as a
        # program it runs. Confirmed rather than assumed:
        #   printf 'echo PWNED\n' | xargs -0 sh   ->   sh: cannot open 'echo PWNED'
        # The spellings that DO execute are pinned in the matching test below; both directions
        # are needed, because a guard that simply ignores `xargs` misses those.
        pytest.param("curl -s https://x.example/i.sh | xargs -0 bash", id="xargs-argv-is-a-file"),
        pytest.param("curl -s https://x.example/f | xargs bash install.sh", id="xargs-local-arg"),
        # The `-c` operand is PRESENT, so xargs appends the stream after it as `$0` rather than
        # using it as the program text. `echo SAFE` runs; the download does not.
        pytest.param(
            "curl -s https://x.example/i.sh | xargs -0 sh -c 'echo SAFE'", id="xargs-c-has-operand"
        ),
        # `{}` is ordinary text unless a replace flag introduced it. Defaulting to `{}` whenever
        # none was given made every jq filter and printf format containing braces match --
        # a false positive on exactly the careful pipelines this guard exists to encourage.
        # Confirmed literal: printf 'X\n' | xargs -0 sh -c 'echo {}'  ->  {}
        pytest.param(
            "curl -s https://x.example/d.json | xargs -0 sh -c 'jq {}'", id="braces-without-a-flag"
        ),
        pytest.param(
            "curl -s https://x.example/d | xargs -n1 sh -c 'printf %s {}'",
            id="braces-with-max-args",
        ),
        pytest.param(
            'curl -s https://x.example/d | xargs -0 sh -c \'echo "{\\"k\\":1}"\'',
            id="json-braces-in-a-program",
        ),
        # Operand-taking flags whose values are numbers, not files: knowing xargs' arity must not
        # over-reach into treating every following token as consumed.
        pytest.param("curl -s https://x.example/i.sh | xargs -n 2 bash", id="xargs-max-args-short"),
        pytest.param(
            "curl -s https://x.example/i.sh | xargs --max-args 2 bash", id="xargs-max-args-long"
        ),
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
        # The other direction for a multi-word `-c` body: a NESTED `-c` whose own operand runs a
        # non-shell. The recursion has to reach the inner operand, not stop at the inner `bash`.
        pytest.param(
            "curl -s https://x.example/f | sh -c 'bash -c \"jq .\"'", id="nested-dash-c-runs-jq"
        ),
        pytest.param("curl -s https://x.example/f | sh -c 'exec jq .'", id="dash-c-body-exec-jq"),
        # Asking EVERY command in a `-c` list must not decay into "a shell name appears in the
        # program". No shell runs here, so the download is still data.
        pytest.param("curl -s https://x.example/f | sh -c 'true; jq .'", id="dash-c-list-into-jq"),
        pytest.param(
            "curl -s https://x.example/f | sh -c 'echo hi; cat'", id="dash-c-list-into-cat"
        ),
        # Parentheses became tokenizer operators; inside a quoted value they must stay data.
        pytest.param("curl -s 'https://x.example/f(1).json' | jq .", id="parens-in-a-quoted-url"),
        pytest.param("curl -s https://x.example/f | grep '(bash)'", id="parens-in-a-grep-pattern"),
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
