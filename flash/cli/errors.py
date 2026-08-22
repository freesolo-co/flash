"""Turning argparse's usage errors into the themed CLI's short suggestions.

argparse reports a mistyped flag by printing the whole usage block, which buries the one token that
was actually wrong. These helpers reduce that to a single line naming the token and, where one
exists, the closest real alternative.

The hard part is making the suggestion one the user can actually follow. A subcommand's unknown
tokens are handed back to the root parser to report, so candidate options must be recovered from the
parser in effect at the rejected token's argv position. Looking only at the final selected command
crosses nested command boundaries, while looking through the whole tree offers unrelated flags.

Root flags are also positional here (`flash --debug login`, never `flash login --debug`), and `--`
stops option parsing. The suggestion context therefore carries both boundaries, plus the original
spelling of a repeated short root flag that must be moved without changing its count.
"""

from __future__ import annotations

import argparse
import difflib
import re
from collections.abc import Iterable
from dataclasses import dataclass

from flash._internal.channel import CLI_NAME


@dataclass(frozen=True)
class _SuggestionContext:
    options: tuple[str, ...]
    root_only: frozenset[str]
    reposition: str | None = None
    after_terminator: bool = False


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
        # a subparser raising its own error is already the parser in effect at the rejected token.
        context = _SuggestionContext(tuple(sorted(_direct_options(parser))), frozenset())
    else:
        context = _suggestion_context(message, parser, tuple(argv))
    return _friendly_message(message, context)


def _friendly_message(message: str, context: _SuggestionContext) -> str:
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
    if context.after_terminator:
        return message
    # a root flag in the wrong position is a placement error, not a typo. repeatable short flags
    # retain the complete spelling so moving `-vv` never downgrades it to one `-v`.
    if context.reposition is not None:
        return _reposition_message(context.reposition)
    if bad in context.root_only:
        return _reposition_message(bad)
    # the position-scoped pool can still include an exact spelling shared with the root. dropping it
    # keeps a rejected token from being echoed back as its own correction.
    candidates = [option for option in context.options if option != bad]
    near = difflib.get_close_matches(bad, candidates, n=1)
    if not near:
        return message
    suggestion = near[0]
    if suggestion in context.root_only:
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


def _suggestion_context(
    message: str, root: argparse.ArgumentParser, argv: tuple[str, ...]
) -> _SuggestionContext:
    root_only = _root_only_options(root)
    bad = _unrecognized_option(message)
    if bad is None:
        return _SuggestionContext((), root_only)
    location = _option_location(root, argv, bad)
    if location is None:
        return _SuggestionContext((), root_only)
    current, after_terminator = location
    reposition = bad if _is_repeated_root_short_option(root, bad, root_only) else None
    options = tuple(sorted({*_direct_options(current), *root_only}))
    return _SuggestionContext(options, root_only, reposition, after_terminator)


def _unrecognized_option(message: str) -> str | None:
    prefix = "unrecognized arguments: "
    if not message.startswith(prefix):
        return None
    return next(
        (
            token.split("=", 1)[0]
            for token in message[len(prefix) :].split()
            if token.startswith("-")
        ),
        None,
    )


def _option_location(
    root: argparse.ArgumentParser, argv: tuple[str, ...], bad: str
) -> tuple[argparse.ArgumentParser, bool] | None:
    """Return the parser and terminator state at the rejected token's argv position."""
    current = root
    path_open = True
    after_terminator = False
    fallback: tuple[argparse.ArgumentParser, bool] | None = None

    for token in argv:
        if token == "--":
            after_terminator = True
            path_open = False
            continue
        if token.split("=", 1)[0] == bad:
            location = (current, after_terminator)
            fallback = location
            # repeated spellings can include one accepted root occurrence and one rejected
            # subcommand occurrence. choose the occurrence argparse could not consume here.
            if after_terminator or len(current._get_option_tuples(token)) != 1:
                return location
        if after_terminator or token.startswith("-") or not path_open:
            continue
        subparsers = next(
            (a for a in current._actions if isinstance(a, argparse._SubParsersAction)), None
        )
        if subparsers is None:
            path_open = False
            continue
        nxt = subparsers.choices.get(token)
        if nxt is None:
            path_open = False
            continue
        current = nxt

    return fallback


def _is_repeated_root_short_option(
    root: argparse.ArgumentParser, token: str, root_only: frozenset[str]
) -> bool:
    """Whether token is a compact repeat of a count-style root option, such as ``-vv``."""
    if not token.startswith("-") or token.startswith("--"):
        return False
    parsed = root._parse_optional(token)
    if parsed is None:
        return False
    action, option, separator, explicit = parsed
    return (
        isinstance(action, argparse._CountAction)
        and option in root_only
        and separator == ""
        and explicit is not None
        and token == option + option[-1] * len(explicit)
    )


def _direct_options(parser: argparse.ArgumentParser) -> Iterable[str]:
    for action in parser._actions:
        yield from action.option_strings


def _parser_options(parser: argparse.ArgumentParser) -> Iterable[str]:
    """Yield option strings registered on a parser and its subparsers."""
    yield from _direct_options(parser)
    for action in parser._actions:
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
