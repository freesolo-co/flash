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
    # a bare `-` starts with a dash but names no option, so it is an ordinary positional argparse
    # could not place. requiring a character after the dash keeps it out of the typo machinery,
    # which would otherwise answer it with the nearest short flag.
    bad = next(
        (
            token.split("=", 1)[0]
            for token in message[len(prefix) :].split()
            if token.startswith("-") and len(token.split("=", 1)[0]) > 1
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
        return f"unrecognized argument '{bad}' (did you mean '{suggestion}'? {_placement_hint(suggestion)})"
    return f"unrecognized argument '{bad}' (did you mean '{suggestion}'?)"


def _placement_hint(flag: str) -> str:
    """Say where a root flag belongs, and claim nothing about the rest of the line.

    This hint knows one thing: the flag parses only ahead of the subcommand. It does not know what
    removing the flag from where it sat exposes. `--wait` takes an optional value, so a flag lifted
    from between `--wait` and a run id leaves the run id to be swallowed as the timeout -- which
    `_wait_seconds` then catches and explains precisely. Stating the destination and stopping there
    keeps this hint true for every line; the one case it cannot see already answers itself.

    Both the repositioning message and the respelling-plus-repositioning message end in this same
    advice, so they share it rather than each spelling it out and drifting apart.
    """
    return f"it goes before the command, as `{CLI_NAME} {flag} ...`"


def _reposition_message(flag: str) -> str:
    """Answer a correctly spelled root flag that landed after the subcommand."""
    return f"unrecognized argument '{flag}' ({_placement_hint(flag)})"


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
            if after_terminator or len(_option_tuples(current, token)) != 1:
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


def _option_tuples(parser: argparse.ArgumentParser, token: str) -> list:
    """The prefix matches for a token, for tokens argparse's own helper cannot be asked about.

    `_get_option_tuples` indexes the token's second character, so a bare ``-`` raises IndexError
    there. A lone dash is an ordinary unrecognized positional and must reach argparse's own message
    rather than crash the code that was trying to improve it.
    """
    if len(token) < 2:
        return []
    return parser._get_option_tuples(token)


def _parse_optional_fields(
    parser: argparse.ArgumentParser, token: str
) -> tuple[object, str | None, str | None] | None:
    """The action, option spelling and explicit remainder argparse reads out of a token.

    `_parse_optional` has three shapes across the interpreters `requires-python` admits: a 3-tuple
    before 3.11.9/3.12.3, a 4-tuple once a separator was inserted, and from 3.12.11 a LIST of those
    tuples, one per prefix match. Normalizing here keeps that churn in one place, and keeps the
    separator out of the caller: the exact-spelling reconstruction the caller does already settles
    what the separator would have told it. An unfamiliar shape returns None rather than raising,
    because this runs inside the handler whose whole job is to render someone else's error.
    """
    parsed = parser._parse_optional(token)
    if isinstance(parsed, list):
        # several prefix matches means the spelling is ambiguous, so it is not one repeated flag.
        parsed = parsed[0] if len(parsed) == 1 else None
    if not isinstance(parsed, tuple) or len(parsed) not in (3, 4):
        return None
    return parsed[0], parsed[1], parsed[-1]


def _is_repeated_root_short_option(
    root: argparse.ArgumentParser, token: str, root_only: frozenset[str]
) -> bool:
    """Whether token is a compact repeat of a count-style root option, such as ``-vv``."""
    if not token.startswith("-") or token.startswith("--"):
        return False
    fields = _parse_optional_fields(root, token)
    if fields is None:
        return False
    action, option, explicit = fields
    return (
        isinstance(action, argparse._CountAction)
        and option in root_only
        and isinstance(explicit, str)
        and bool(explicit)
        # the whole token must be that one short flag repeated, so `-vx` and `-v=2` are not repeats.
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
