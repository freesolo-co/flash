"""Pipelines that must stay CLEAN: the counter-cases for the pipe-to-shell guard.

A guard that flags `curl … | jq .` is worse than useless -- it fails CI on correct code and
teaches people to disable it. Every case here is an ordinary pipeline, a fetch that lands on
disk, or a shell name appearing somewhere it is data rather than a command, and each one was
verified by running it. They carry exactly as much weight as the positive cases in
`test_no_pipe_to_shell`: most of the defects found in this guard were false POSITIVES introduced
while closing a real hole.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests import test_no_pipe_to_shell
from tests.pipe_to_shell_grammar import _piped_into_a_shell


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
        # the semicolon ends the pipeline, so the later group is a separate top-level command and
        # receives the parent shell's stdin rather than the fetch. confirmed with outer stdin closed:
        #   cat payload.sh | echo a; ( bash )  -> a, no pwned
        pytest.param(
            "curl -s https://x.example/f | echo a; ( bash )",
            id="semicolon-after-pipeline-before-a-group",
        ),
        pytest.param(
            "curl -s https://x.example/f | case x in x) echo a;; esac; echo case; ( bash )",
            id="closed-case-then-top-level-group",
        ),
        # A wrapper in front of something that is not a shell is still not a shell.
        pytest.param("curl -s https://x.example/f | timeout 30 jq .", id="timeout-into-jq"),
        # `-s`/`-i` start a shell only when NOTHING follows them. With a command present, sudo
        # runs THAT and the stream is never executed -- so the flag alone cannot be the signal.
        # Confirmed: printf 'echo X\n' | sudo -n -s echo hi   ->   hi
        # The fetch name must be the stage's COMMAND. Matching it in any token made a search
        # PATTERN a download, and the shell downstream receives locally selected text:
        #   grep echo commands.txt | bash  ->  runs the local file's line, fetches nothing
        pytest.param("grep curl commands.txt | bash", id="fetch-name-is-a-pattern"),
        pytest.param("grep -r wget src | bash", id="fetch-name-is-a-pattern-with-flags"),
        # `eval` only matters when the capture reads STDIN. A literal or a variable does not.
        pytest.param("curl -s https://x.example/f | bash -c 'eval \"$FOO\"'", id="eval-a-variable"),
        pytest.param(
            "curl -s https://x.example/f | bash -c 'eval \"echo SAFE\"'", id="eval-a-literal"
        ),
        # The new wrapper and operator must not make ordinary destinations match.
        pytest.param("curl -s https://x.example/f | unshare jq .", id="unshare-into-jq"),
        pytest.param("curl -s https://x.example/f |& jq .", id="pipe-both-streams-into-jq"),
        # A chained tool still honours the LAST tool's own selection and `-c`, so the safe
        # spellings stay safe through the chain.
        pytest.param(
            "curl -s https://x.example/f | sudo su -s /usr/bin/sha256sum nobody",
            id="sudo-then-su-selects-a-non-shell",
        ),
        pytest.param(
            "curl -s https://x.example/f | sudo su -s /bin/sh -c 'jq .'",
            id="sudo-then-su-defers-to-c",
        ),
        pytest.param("curl -s https://x.example/f | sudo -u root jq .", id="sudo-user-then-jq"),
        pytest.param("curl -s https://x.example/f | 2>/dev/null jq .", id="redirection-then-jq"),
        # `-s`/`--shell` CHOOSES which program `su`/`runuser` starts, and it need not be a shell.
        # Matching `su` as a shell NAME made every spelling dangerous, including safe ones:
        #   printf 'hello\n' | sudo -n su -s /usr/bin/sha256sum nobody  ->  5891b5b5…  (hashed)
        #   printf 'hello\n' | sudo -n runuser -u root sha256sum        ->  5891b5b5…  (hashed)
        pytest.param(
            "curl -s https://x.example/f | su -s /usr/bin/sha256sum nobody",
            id="su-selects-a-non-shell",
        ),
        pytest.param(
            "curl -s https://x.example/f | runuser -u root sha256sum", id="runuser-runs-a-non-shell"
        ),
        # A chosen shell still defers to its own `-c`, exactly as `sh -c` does -- this is the
        # verified-download pattern the guard exists to encourage:
        #   printf 'echo PWNED\n' | sudo -n su -s /bin/sh -c 'echo RAN_C'  ->  RAN_C only
        pytest.param(
            "curl -s https://x.example/f | su -s /bin/sh -c 'sha256sum -c -'",
            id="su-selected-shell-defers-to-c",
        ),
        pytest.param("curl -s https://x.example/f | su -c 'jq .'", id="su-c-non-shell"),
        # A group whose commands are all ordinary is still ordinary; the depth tracking must not
        # make every group suspicious.
        pytest.param("curl -s https://x.example/f | { true; jq .; }", id="brace-group-into-jq"),
        pytest.param("curl -s https://x.example/f | (true; jq .)", id="subshell-group-into-jq"),
        # A separator at depth 0 still ENDS the pipeline: what follows has nothing piped into it.
        pytest.param(
            "curl -s https://x.example/f | jq . ; bash recover.sh", id="semicolon-ends-it"
        ),
        # A substitution that feeds a non-shell, and a shell fed by a substitution with no fetch
        # in it. Neither downloads-then-executes.
        pytest.param(
            'printf "%s" "$(curl -fsSL https://x.example/d.json)" | jq .',
            id="substituted-fetch-into-jq",
        ),
        pytest.param('echo "$(date)" | bash', id="substitution-without-a-fetch"),
        # The unquoted spelling must stay just as discriminating: joining the stage back together
        # to see `$(` across two tokens must not turn a NON-fetch substitution, or a command whose
        # name merely starts with one, into a download.
        pytest.param("echo $(cat local.sh) | bash", id="unquoted-substitution-without-a-fetch"),
        pytest.param("echo $(curlie https://x.example/f) | bash", id="unquoted-substituted-curlie"),
        pytest.param("echo $(mycurl https://x.example/f) | bash", id="unquoted-substituted-mycurl"),
        # A keyword compound whose body runs no shell is clean, exactly as the brace group is. The
        # `sh -c "cat"` spelling PRINTS the stream rather than executing it -- confirmed:
        #   printf 'echo PWNED\n' | if true; then sh -c "cat"; fi   ->  echo PWNED  (not run)
        pytest.param(
            "curl -sSL https://x.example/f | if true; then jq .; fi", id="if-clause-into-jq"
        ),
        pytest.param(
            'curl -sSL https://x.example/f | if true; then sh -c "cat"; fi',
            id="if-clause-prints-the-stream",
        ),
        # `grep curl` upstream is still a search PATTERN, not a fetch, whatever the stage
        # downstream happens to be.
        pytest.param("grep curl commands.txt | if true; then bash; fi", id="grep-then-if-clause"),
        # A compound keyword is only a keyword in COMMAND position. `echo done` is a word being
        # printed; reading it as the closer would unbalance the depth counter and split the very
        # clause the counter exists to hold together.
        pytest.param("if true; then echo done; fi", id="closer-word-as-an-argument"),
        # In a `case`, both the SUBJECT and the branch PATTERNS are matched text, never commands.
        # A shell name in either position runs nothing, so dropping them is what keeps the clause
        # from reading as a pipe-to-shell on the strength of a word being compared.
        pytest.param(
            "curl -sSL https://x.example/f | case x in x) jq .;; esac", id="case-clause-into-jq"
        ),
        pytest.param(
            "curl -sSL https://x.example/f | case bash in x) jq .;; esac",
            id="shell-name-is-the-case-subject",
        ),
        pytest.param(
            "curl -sSL https://x.example/f | case x in bash) jq .;; esac",
            id="shell-name-is-a-case-pattern",
        ),
        # The same two positions, one `case` deeper. Widening the label strip to reach a NESTED
        # `case` must not start reading its subject or pattern as a command either.
        pytest.param(
            "curl -sSL https://x.example/f | case x in x) case y in bash) jq .;; esac;; esac",
            id="shell-name-is-a-nested-case-pattern",
        ),
        pytest.param(
            "curl -sSL https://x.example/f | if true; then case x in x) jq .;; esac; fi",
            id="nested-case-into-jq",
        ),
        # Keeping a group's `)` inside a branch must not also keep the group's CONTENTS suspicious:
        # a group is only a finding when what it runs is a shell.
        pytest.param(
            "curl -sSL https://x.example/f | case x in x) ( jq . );; esac",
            id="group-in-a-case-branch-into-jq",
        ),
        # `case` as an ordinary word, where no label logic should engage at all.
        pytest.param("apt-get install -y case && echo ok", id="case-as-a-package-name"),
        # Recursing into a group or clause must not make its CONTENTS suspicious by themselves.
        # A fetch that lands on disk, a shell running a local script, and a pipeline into a
        # non-shell are all as clean inside a clause as they are outside one.
        pytest.param(
            "if true; then curl -sSL https://x.example/f | jq .; fi", id="pipeline-in-an-if-into-jq"
        ),
        pytest.param(
            "if true; then curl -sSL https://x.example/f -o f.txt; fi",
            id="fetch-to-disk-inside-a-clause",
        ),
        pytest.param("if true; then bash setup.sh; fi", id="local-script-inside-a-clause"),
        pytest.param("( curl -sSL https://x.example/f | jq . )", id="pipeline-in-a-group-into-jq"),
        pytest.param('while read -r l; do echo "$l"; done', id="clause-with-no-fetch"),
        # The pattern this guard exists to ENCOURAGE, wrapped in the same conditional: fetch to
        # disk, then verify a digest. Flagging it would fail CI on the correct way to install.
        pytest.param(
            "set -eux; curl -fsSL https://x.example/t.tgz -o /tmp/t.tgz"
            ' && echo "abc  /tmp/t.tgz" | sha256sum -c -',
            id="pinned-and-verified-install",
        ),
        pytest.param(
            "if command -v apt-get; then apt-get install -y curl; fi",
            id="clause-that-installs-curl-itself",
        ),
        # An exec-form array that runs no shell rejoins into a stage whose command word is not a
        # shell, so it stays clean for the same reason its shell-form spelling would. Unwrapping
        # the JSON must not by itself make an array suspicious.
        pytest.param('RUN ["python", "-m", "pip", "install", "x"]', id="exec-form-no-shell"),
        pytest.param('CMD ["uvicorn", "app:main", "--host", "0.0.0.0"]', id="exec-form-server"),
        pytest.param(
            'RUN ["bash", "-c", "curl -sSL https://x.example/f | jq ."]', id="exec-form-into-jq"
        ),
        pytest.param('run: "curl -sSL https://x.example/f | jq ."', id="quoted-scalar-into-jq"),
        pytest.param('run: "make test"', id="quoted-scalar-no-fetch"),
        # A command may itself START and END with a quote without being a quoted scalar: the
        # quotes here belong to the URL argument, and stripping them would splice the URL onto
        # the shell name. Only a quote enclosing the WHOLE value is file syntax.
        pytest.param("curl -s 'https://x.example/a|b' | jq .", id="quoted-arg-is-not-a-scalar"),
        # The other half of the shell-dependent `-s -c` pair pinned in the matching test above.
        # bash and busybox ash run the `-c` program and leave the pipe as unread data; only dash
        # goes on to execute it. Verified: printf 'echo PWNED\n' | bash -s -c 'echo RAN_C'
        # prints RAN_C and no PWNED.
        pytest.param(
            "curl -sSL https://x.example/i.sh | bash -s -c 'jq .'", id="bash-s-c-is-clean"
        ),
        # Sourcing an ordinary FILE runs that file and leaves the pipe unread, and a command that
        # merely reads stdin as data is not executing it. Confirmed:
        #   printf 'echo PWNED\n' | bash -c 'source ./sf.sh'  ->  FROM_FILE, no PWNED
        # Without the stdin-path check, matching bare `source` would flag all of these.
        pytest.param(
            "curl -s https://x.example/i.sh | bash -c 'source ./setup.sh'", id="source-a-real-file"
        ),
        pytest.param("curl -s https://x.example/i.sh | bash -c '. ./env.sh'", id="dot-a-real-file"),
        pytest.param(
            "curl -s https://x.example/i.sh | bash -c 'cat /dev/stdin'", id="cat-stdin-is-data"
        ),
        # `source` as a search term or a JSON field is not a command being run.
        pytest.param("curl -s https://x.example/f | grep source", id="grep-for-source"),
        pytest.param("curl -s https://x.example/f | jq -r .source", id="jq-source-field"),
        pytest.param("curl -s https://x.example/f | sudo -s echo hi", id="sudo-s-with-a-command"),
        pytest.param("curl -s https://x.example/f | sudo -s jq .", id="sudo-s-into-jq"),
        # `-s` on a wrapper that is not a privilege tool means something else entirely (here, the
        # signal to send). Scoping the flag set to sudo/su/runuser is what keeps this clean.
        pytest.param(
            "curl -s https://x.example/f | timeout -s TERM 30 jq .", id="timeout-signal-s"
        ),
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
        # A command word after `-s` takes over, so the flag no longer starts a shell:
        #   printf 'X\n' | sudo -n -s jq .   ->   jq parses the stream, nothing is executed
        pytest.param("curl -s https://x.example/f | sudo -s jq .", id="sudo-dash-s-with-command"),
        # `-u shane` and `-p prompt` contain the letters the flag scan looks for, in an OPERAND.
        # Reading them as `-s`/`-i` would flag ordinary pipelines.
        pytest.param("curl -s https://x.example/f | sudo -u sysadmin cat", id="operand-has-s"),
        pytest.param(
            "curl -s https://x.example/f | sudo -p 'pass:' cat", id="prompt-operand-has-s"
        ),
        # `su -c 'echo hi'` runs the operand and the stream is data, exactly like `sh -c`.
        # Confirmed: printf 'X\n' | sudo -n su -c 'echo Y'  ->  Y
        pytest.param("curl -s https://x.example/f | su -c 'echo hi'", id="su-dash-c-non-shell"),
        # The command name merely CONTAINS `su`.
        pytest.param("curl -s https://x.example/f | sudo -u root sudoedit", id="name-contains-su"),
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
        # inspecting an upstream split-string command must still require curl or wget as its command.
        # `env -S 'grep curl commands.txt'` searches local data; the word `curl` is only a pattern.
        pytest.param(
            "env -S 'grep curl commands.txt' | bash", id="split-upstream-fetch-name-is-an-argument"
        ),
        pytest.param(
            "env --split-string='curlie https://x.example/f' | bash",
            id="split-upstream-command-is-only-a-prefix",
        ),
        # `-S` on something that is not `env` is an ordinary flag: jq's is --sort-keys.
        pytest.param("curl -s https://x.example/f | jq -S .", id="dash-s-on-a-non-env"),
        # a non-option shell operand is a script file, so the shell runs that file and leaves the
        # pipe as data. confirmed:
        #   printf 'echo payload_ran\n' | bash verify.sh  -> verifier_ran, no payload_ran
        pytest.param(
            "curl -s https://x.example/checksums.txt | bash verify.sh",
            id="bash-runs-a-script-file",
        ),
        pytest.param(
            "curl -s https://x.example/checksums.txt | sh ./verify.sh",
            id="sh-runs-a-script-file",
        ),
        # option operands precede the script file and are not `-c`. confirmed:
        #   bash -o pipefail verify.sh  -> verifier_ran
        #   bash -o c verify.sh         -> `c`: invalid option name, no payload execution
        pytest.param(
            "curl -s https://x.example/checksums.txt | bash -o pipefail verify.sh",
            id="shell-option-before-a-script-file",
        ),
        pytest.param(
            "curl -s https://x.example/checksums.txt | bash -o c verify.sh",
            id="c-is-a-shell-option-operand",
        ),
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

    # Through the REAL guard rather than `_piped_into_a_shell` directly, so each counter-case also
    # covers the file-scanning path that runs in CI. Reached via the module rather than imported
    # by name: importing the function would re-collect its own parametrized repo scan here, adding
    # a duplicate run of every Dockerfile and workflow case to this file.
    test_no_pipe_to_shell.test_nothing_pipes_a_downloaded_script_into_a_shell(planted)


@pytest.mark.parametrize(
    "line",
    [
        pytest.param('run: ""', id="empty-double-quoted-scalar"),
        pytest.param("run: ''", id="empty-single-quoted-scalar"),
        pytest.param('run: "   "', id="whitespace-only-scalar"),
    ],
)
def test_an_empty_quoted_scalar_is_clean_rather_than_a_crash(line: str):
    """An empty `run:` value must answer "clean", not abort the scan.

    Unwrapping a quoted scalar reads its first token to tell file syntax from a command whose
    own ARGUMENT is quoted. An empty body has no first token, so indexing it raised IndexError
    -- and an exception here fails the whole repo-wide guard rather than one line, turning a
    trivially clean input into a broken build.
    """
    assert not _piped_into_a_shell(line)
