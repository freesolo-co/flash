"""Turning argparse's usage errors into the themed CLI's short suggestions.

argparse reports a mistyped flag by printing the whole usage block, which buries the one token that
was actually wrong. These helpers reduce that to a single line naming the token and, where one
exists, the closest real alternative.

Two details make the suggestion trustworthy rather than merely present. The candidate pool spans
every subcommand, because argparse hands a subcommand's unknown tokens back to the ROOT parser to
report -- so the root error has to know about flags it does not own. And root flags are positional
in this CLI (`flash --debug login`, never `flash login --debug`), so a suggestion drawn from that
pool can name a real flag that still fails where the user typed it; those carry their position.
"""

from __future__ import annotations

import argparse
import difflib
import re
from collections.abc import Iterable

from flash._internal.channel import CLI_NAME


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
    # the candidate pool spans every subcommand's flags, so a flag that is real SOMEWHERE else
    # reaches this suggestion: `flash login --repository X` would otherwise answer "did you mean
    # '--repository'?" -- echoing the token the user just typed as its own correction. dropping the
    # exact token keeps the suggestion to flags that differ from what was rejected.
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
