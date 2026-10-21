"""Parsing for ``flash env test --param``, which mirrors ``[environment.params]``.

Split from ``test.py`` as a cohesive group: this is config-value parsing, independent of driving
episodes or judging what a run measured, and nothing here touches an environment.
"""

from __future__ import annotations

import json
import tomllib

# characters that give a TOML value structure. text containing none of them is a bare string, so a
# parse failure means the user typed unquoted words; text containing any of them was reaching for
# TOML syntax and a parse failure means they got it wrong.
_TOML_STRUCTURAL_CHARS = frozenset("\"'[]{}=,\n")
# the characters that make a TOML KEY mean something other than its literal spelling: `.` nests,
# quotes make a bare key hold characters it otherwise could not. deliberately narrower than
# _TOML_STRUCTURAL_CHARS, which describes values -- a key is checked before the `=` split, so the
# structural characters of a value are not applicable to it.
_TOML_KEY_STRUCTURAL_CHARS = frozenset(".\"'")
# non-word TOML scalars start with a digit or sign; a parse failure is malformed syntax, not prose.
# include `.` so unsigned `.5` is rejected consistently with signed `+.5`.
_TOML_SCALAR_LEADING_CHARS = frozenset("0123456789+-.")
# the TOML booleans, which are written as bare words rather than starting with a digit or sign and
# are therefore the blind spot of _TOML_SCALAR_LEADING_CHARS. TOML spells them in lowercase only, so
# a case variant is a malformed literal rather than prose and must not forward as a string.
_TOML_BOOLEAN_WORDS = frozenset({"true", "false"})
# reject case variants of TOML's lowercase non-finite floats rather than forwarding them as strings.
# include `infinity`: float coercion turns it back into the unsupported infinite value.
_TOML_NON_FINITE_WORDS = frozenset({"inf", "infinity", "nan"})
# TOML has no null. these are the spellings people reach for anyway, borrowed from json, python, and
# yaml -- all bare words, so they land in the same blind spot: the parse fails, the value carries no
# structural character, and it forwards as its own literal STRING. an env testing `if value is None`
# or `if not value` then reads a truthy string, and no [environment.params] assignment could have
# produced it, since the config has no way to spell an absent value either. omitting the parameter
# is what expresses that, so say so rather than forwarding text nothing asked for.
_TOML_NULL_WORDS = frozenset({"null", "none", "nil"})


def _reject_unsubmittable_param(key: str, value: object) -> None:
    """Reject TOML values that ``[environment.params]`` could not submit.

    JSON excludes TOML date/time objects and non-finite floats, so use ``allow_nan=False``.
    ``ensure_ascii=False`` exposes lone surrogates; encoding the JSON result catches them even when
    nested, because no UTF-8 config can carry that value.
    """
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"--param {key} is not JSON-serializable and could not be submitted "
            f"in [environment.params]: {exc}"
        ) from exc
    if not _is_expressible_in_toml(encoded):
        raise ValueError(
            f"--param {key} is not valid UTF-8, so no config file could carry it and the run "
            f"would never receive this value"
        )


def _parse_param_value(key: str, raw: str) -> object:
    """Parse one ``--param`` value the way ``[environment.params]`` would parse it.

    Bare unquoted text is not valid TOML but is what users type most often, so it falls back to a
    string. Everything else must parse, because silently keeping a malformed structured value as a
    literal string would validate the environment against parameters the equivalent config entry
    could never load -- the offline gate would pass on input training rejects.
    """
    value = raw.strip()
    try:
        document = tomllib.loads(f"v = {value}")
    except tomllib.TOMLDecodeError as exc:
        # the fallback is an allowlist, not a blocklist of opening delimiters: `filters=]` opens
        # nothing yet is still malformed TOML, and blocklisting only the openers let it through as
        # the literal string "]". a bare string is text with no TOML structural character in it.
        #
        # ...and no delimiter is needed to be reaching for TOML syntax. `cutoff=2026-13-01` holds
        # none of those characters, so it forwarded as the string "2026-13-01" while the equivalent
        # `[environment.params]` entry fails to load -- the gate passing on a config that cannot be
        # written. same for `1e`, `0x`, `007`, `1_`, `12:99:00`. a leading digit or sign is the
        # tell: every TOML scalar except the bare-word `true`/`false`/`inf`/`nan` spellings starts
        # with one, so such a token is a malformed number or date, not prose.
        if value and not (set(value) & _TOML_STRUCTURAL_CHARS):
            # the booleans are the family of TOML scalars that does NOT start with a digit or sign,
            # so the leading-character test below cannot see them. TOML spells them lowercase only,
            # which makes a python-style `strict=False` parse-fail and fall through here as the
            # STRING "False" -- and a non-empty string is truthy, so an env branching on `if strict`
            # reads it as enabled while the config spelling `false` disables it. the offline gate
            # would pass on the opposite of what the run trains with.
            if value.lower() in _TOML_BOOLEAN_WORDS:
                raise ValueError(
                    f"--param {key} is not a valid TOML value: {exc}. TOML spells "
                    f"{value.lower()} in lowercase; write --param {key}={value.lower()} for the "
                    f"boolean, or --param '{key}=\"{value}\"' to pass it as text"
                ) from exc
            # the non-finite floats are the same blind spot, minus their optional sign -- `-Inf`
            # does start with one of the leading characters, but the message that test raises talks
            # about numbers and dates and would not name what is actually wrong.
            if value.lstrip("+-").lower() in _TOML_NON_FINITE_WORDS:
                raise ValueError(
                    f"--param {key} is not a valid TOML value: {exc}. TOML spells the non-finite "
                    f"floats as lowercase inf and nan, and [environment.params] could not submit "
                    f"one anyway since it is not JSON; pass a finite number, or "
                    f"--param '{key}=\"{value}\"' to pass it as text"
                ) from exc
            # the null spellings, the last bare-word family. unlike the others there is no lowercase
            # form to point at, because TOML cannot express an absent value at all.
            if value.lower() in _TOML_NULL_WORDS:
                raise ValueError(
                    f"--param {key} is not a valid TOML value: {exc}. TOML has no null, so "
                    f"[environment.params] could not carry this either; omit --param {key} to "
                    f"leave it unset, or --param '{key}=\"{value}\"' to pass it as text"
                ) from exc
            if value[0] not in _TOML_SCALAR_LEADING_CHARS:
                # the bare-string path returns before the parsed-value checks below, so the one
                # that applies to text is asked here too. this is the route a surrogate actually
                # takes: it holds no structural character, so it is read as prose.
                _reject_unsubmittable_param(key, value)
                return value
            # quoting is the escape hatch, and it is the same spelling the config needs -- a
            # genuinely textual "3px" has to be written `"3px"` in `[environment.params]` too, so
            # pointing at it keeps the flag and the config in step rather than adding a second
            # spelling that only the flag accepts.
            raise ValueError(
                f"--param {key} is not a valid TOML value: {exc}. it starts like a number or "
                f"date, so [environment.params] would reject it too; quote it "
                f"(--param '{key}=\"{value}\"') to pass it as text"
            ) from exc
        raise ValueError(f"--param {key} is not a valid TOML value: {exc}") from exc
    # a value containing a newline makes `v = <value>` a multi-line document, so tomllib accepts
    # `max_rows=5\nstrict=true` as two assignments and taking only "v" drops the second silently.
    if set(document) != {"v"}:
        extra = ", ".join(sorted(set(document) - {"v"}))
        raise ValueError(
            f"--param {key} contains more than one assignment ({extra}); "
            "pass one KEY=VALUE per --param"
        )
    parsed = document["v"]
    _reject_unsubmittable_param(key, parsed)
    return parsed


def _is_expressible_in_toml(text: str) -> bool:
    """Report whether ``[environment.params]`` can carry ``text`` unchanged.

    TOML quoted keys and strings cover normal text; only values that cannot encode as UTF-8, such as
    lone surrogates, cannot appear in the config. Apply this to both assignment sides.
    """
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _literal_param_key(key: str) -> str:
    """Resolve a ``--param`` TOML key to its literal parameter name.

    Parse dotted and quoted keys with tomllib so ``difficulty.level`` nests while
    ``"release.channel"`` stays flat. Reject genuine nesting because params are splatted as kwargs
    in ``flash/envs/loading/base.py``; pass the containing inline table instead.
    """
    name = key
    if set(key) & _TOML_KEY_STRUCTURAL_CHARS:
        try:
            document = tomllib.loads(f"{key} = 0")
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(
                f"--param {key} is not a valid TOML key: {exc}. [environment.params] would reject "
                f"this spelling too; quote the name (--param '\"{key}\"=...') to pass it literally"
            ) from exc
        # one assignment yields one top-level entry, whatever the spelling. it is nested exactly
        # when the dots were read as structure, which is the case this flag cannot forward.
        ((name, resolved),) = document.items()
        if isinstance(resolved, dict):
            raise ValueError(
                f"--param {key} uses TOML key syntax that denotes structure, which this flag "
                f"cannot forward faithfully. pass the containing table as one value instead, for "
                f"example --param {name}='{{ level = 3 }}', or quote the name "
                f"(--param '\"{key}\"=...') if the dot is part of it"
            )
    # whether the config can hold the name at all. almost always yes: a QUOTED key carries `bad
    # key`, `a/b`, `café` and the rest, and the schema loader takes it, so those are configs a run
    # really can receive. an earlier guard here rejected anything outside the BARE-key grammar,
    # which blocked validating a working config while claiming the config could not hold the name.
    # what is left is the names a UTF-8 config file cannot physically contain.
    if not _is_expressible_in_toml(name):
        raise ValueError(
            f"--param {key!r} is not valid UTF-8, so no config file could carry it and the run "
            f"would never receive this parameter"
        )
    return name


def _quoted_key_end(item: str, start: int) -> int | None:
    """Index of the quote closing the one at ``start``, or None when it is never closed."""
    quote = item[start]
    index = start + 1
    while index < len(item):
        # only a basic string takes escapes. in a literal string a backslash is just a character,
        # so consuming the next one there would step over the closing quote.
        if item[index] == "\\" and quote == '"':
            index += 2
            continue
        if item[index] == quote:
            return index
        index += 1
    return None


def _split_param_assignment(item: str) -> tuple[str, str, str]:
    """Split ``--param`` at the assignment ``=`` outside quoted key text.

    TOML permits flat keys such as ``"a=b"``. Unterminated quotes fall back to ``partition`` so the
    key validator reports the malformed spelling.
    """
    index = 0
    while index < len(item):
        char = item[index]
        if char in "\"'":
            closing = _quoted_key_end(item, index)
            if closing is None:
                break
            index = closing + 1
            continue
        if char == "=":
            return item[:index], "=", item[index + 1 :]
        index += 1
    return item.partition("=")


def _env_params(args) -> dict:
    """Build the ``load_environment()`` kwargs from ``--split`` / ``--param KEY=VALUE``.

    Mirrors ``[environment.params]`` so the local gate can validate the split a run actually
    trains on. Without this the gate always loaded ``dataset/train.jsonl`` and could pass while
    the configured split was never exercised.
    """
    params: dict = {}
    for item in getattr(args, "param", None) or []:
        key, sep, raw = _split_param_assignment(str(item))
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"--param must be KEY=VALUE (got {item!r})")
        params[_literal_param_key(key)] = _parse_param_value(key, raw)
    split = getattr(args, "split", None)
    if split is not None:
        # distinguish "not passed" from "passed empty". `--split "$SPLIT"` with an unset variable
        # is an explicit request for a split, and treating it as absent leaves a `--param
        # split=...` in effect -- so the gate silently validates a different split than the one the
        # command asked for, which is the failure this flag exists to prevent.
        split = str(split).strip()
        if not split:
            raise ValueError("--split requires a non-empty split name")
        params["split"] = split
    return params
