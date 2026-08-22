"""Turning argparse's usage errors into the themed CLI's short suggestions.

argparse reports a mistyped flag by printing the whole usage block, which buries the one token that
was actually wrong. These helpers reduce that to a single line naming the token and, where one
exists, the closest real alternative.

The hard part is making the suggestion one the user can actually follow, and both difficulties come
from where argparse raises. A subcommand's unknown tokens are handed back to the ROOT parser to
report, so the error has to know about flags it does not own -- but drawing candidates from the
whole tree offers flags belonging to unrelated commands, which fail exactly like the original typo.
The pool is therefore the SELECTED command's options plus the root's, resolved from the invocation.

Root flags are also positional here (`flash --debug login`, never `flash login --debug`), so a
suggestion can name a real flag that still fails where the user typed it. Those carry their
position, and a correctly spelled one is repositioned rather than respelled.
"""

from __future__ import annotations

import argparse
import difflib
import re
from collections.abc import Iterable

from flash._internal.channel import CLI_NAME


def friendly_error(
    message: str, parser: argparse.ArgumentParser, argv: Iterable[str] | None
) -> str:
    """Rewrite one argparse usage error for the themed CLI.

    The whole suggestion policy lives behind this call: which parser's flags are candidates, which
    of them need repositioning, and how the message reads. The caller is an ``error()`` override on
    every parser in the tree and should stay a two-liner -- deciding a candidate pool inline there
    would put policy in a hook that exists to print.
    """
    if argv is None:
        # a subparser raising its own error: it IS the selected parser, and it holds no argv stash.
        selected, root_only = parser, frozenset()
    else:
        selected, root_only = _selected_parser(parser, argv), _root_only_options(parser)
    candidates = sorted({*_parser_options(selected), *root_only})
    return _friendly_message(message, candidates, root_only)


def _friendly_message(
    message: str, options: Iterable[str] = (), root_only: frozenset[str] = frozenset()
) -> str:
    """Shorten argparse's verbose errors into concise suggestions on the styled path.

    An ``invalid choice: 'x' (choose from a, b, c, ...)`` becomes ``unknown command 'x' (did you
    mean 'y'?)`` instead of dumping the whole list. An ``unrecognized arguments: --flag value``
    suggests the closest option registered on the parser that rejected it. Other messages pass
    through untouched. The machine path keeps argparse's exact text for scripts and error tests.
    """
    m = re.search(r"invalid choice: '([^']*)'(?: \(choose from (.*)\))?", message)
    if m:
        bad, raw_choices = m.group(1), m.group(2) or ""
        choices = [c.strip().strip("'\"") for c in raw_choices.split(",") if c.strip()]
        near = difflib.get_close_matches(bad, choices, n=1)
        return f"unknown command '{bad}'" + (f" (did you mean '{near[0]}'?)" if near else "")

    prefix = "unrecognized arguments: "
    if not message.startswith(prefix):
        return message
    bad = next(
        (
            token.split("=", 1)[0]
            for token in message[len(prefix) :].split()
            if token.startswith("-")
        ),
        None,
    )
    if bad is None:
        return message
    # a correctly spelled ROOT flag in the wrong position is not a typo, so there is nothing to
    # correct it to: the fix is to move it. this has to be answered before the exact-token
    # exclusion below, which would otherwise drop `--verbose` from the pool and let the nearest
    # sibling stand in -- `flash train x.toml --verbose` answered "did you mean '--version'?",
    # pointing at a different flag with a different meaning.
    if bad in root_only:
        return _reposition_message(bad)
    # the candidate pool spans the selected command's flags plus the root's, so a flag that is real
    # SOMEWHERE else can still reach this suggestion: `flash login --repository X` would otherwise
    # answer "did you mean '--repository'?" -- echoing the token the user just typed as its own
    # correction. dropping the exact token keeps the suggestion to flags that differ from it.
    candidates = [option for option in options if option != bad]
    near = difflib.get_close_matches(bad, candidates, n=1)
    if not near:
        return message
    suggestion = near[0]
    if suggestion in root_only:
        # a root flag only parses before the subcommand, so naming it alone hands back a correction
        # that fails the same way. say where it goes, or the second attempt is the first error again.
        return (
            f"unrecognized argument '{bad}' (did you mean '{suggestion}'? "
            f"it goes before the command: `{CLI_NAME} {suggestion} <command>`)"
        )
    return f"unrecognized argument '{bad}' (did you mean '{suggestion}'?)"


def _reposition_message(flag: str) -> str:
    return (
        f"unrecognized argument '{flag}' (it goes before the command: "
        f"`{CLI_NAME} {flag} <command>`)"
    )


def _parser_options(parser: argparse.ArgumentParser) -> Iterable[str]:
    """Yield option strings registered on a parser and its subparsers.

    argparse returns a subparser's unknown tokens to the root, which raises the final
    ``unrecognized arguments`` error. Walking descendants keeps that root error aware of the
    selected command's real flags without maintaining a second option registry.
    """
    for action in parser._actions:
        yield from action.option_strings
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                yield from _parser_options(subparser)


def _selected_parser(
    parser: argparse.ArgumentParser, argv: Iterable[str]
) -> argparse.ArgumentParser:
    """Walk the subcommand path in ``argv`` to the parser that owns the invocation.

    The whole-tree pool is what makes a suggestion reachable at all -- argparse reports a
    subcommand's unknown tokens from the ROOT parser -- but drawing candidates from the whole tree
    means an unrelated command's flag can be offered: `flash login --folow` answered '--follow',
    and `flash runs log ID --api-ke` answered '--api-key'. Both corrections are rejected again,
    because those flags belong to other commands. Narrowing to the parser actually selected keeps
    the candidates to flags that would work where the user typed them.
    """
    current = parser
    for token in argv:
        if token.startswith("-"):
            continue
        subparsers = next(
            (a for a in current._actions if isinstance(a, argparse._SubParsersAction)), None
        )
        if subparsers is None:
            break
        nxt = subparsers.choices.get(token)
        if nxt is None:
            # not a command name, so it is a positional argument (a config path, a run id) and the
            # command path has ended. anything after it cannot name a deeper subparser.
            break
        current = nxt
    return current


def _root_only_options(parser: argparse.ArgumentParser) -> frozenset[str]:
    """Option strings accepted only BEFORE the subcommand.

    This CLI's root flags are positional: `flash --debug login` parses, `flash login --debug` does
    not. A suggestion drawn from the whole option pool can therefore name a real flag that still
    fails in the position the user typed it, so the caller needs to know which ones to reposition.
    """
    root: set[str] = set()
    sub: set[str] = set()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                sub.update(_parser_options(subparser))
        else:
            root.update(action.option_strings)
    return frozenset(root - sub)
