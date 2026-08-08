"""Which text the pipe-to-shell guard reads: the files it scans, and where commands begin and end.

This is the half of the guard that answers "what do we look at", kept apart from
`pipe_to_shell_grammar`, which answers "what does this shell text mean". They fail in different
ways and are debugged with different questions -- a file this module misses is invisible to the
grammar no matter how good the grammar is, and every evasion fixed here was a way of writing a
command across lines rather than a way of spelling the command itself.

Nothing here knows what a shell is.
"""

from __future__ import annotations

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


# Resolved once for parametrization; the tests call `_installer_files()` directly when they need
# discovery re-run.
INSTALLER_FILES = _installer_files()


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
