"""Which recorded field NAMES denote a credential.

Split out of `trace_redaction` so the naming rules -- the exact keys, the conventional suffixes and
qualifiers, and the regex forms a `patternProperties` key can take -- sit together and can be read
without the traversal around them.

The rules are deliberately conservative in both directions. A name that denotes a credential must
be recognized however it is spelled, because a third-party credential is never in `context.secrets`
and only the name can protect it. A name that merely CONTAINS a sensitive word must be left alone,
because redacting `token_count` or `password_policy_url` destroys legitimate recorded content.
"""

from __future__ import annotations

from typing import Any

# `auth` is exact-matched, never a suffix: `author` and `oauth` end in it but carry no credential.
_SECRET_KEY_EXACT = frozenset({"authorization", "proxyauthorization", "auth", "xauth"})
_SECRET_KEY_SUFFIXES = (
    "apikey",
    # conventional cloud credential fields end in these normalized forms. bare `key` is deliberately
    # excluded because JSON schemas and tool arguments use it pervasively for harmless data.
    "accesskeyid",
    "secretkey",
    "accesskey",
    # api-management gateways spell their credential `subscription_key` / `Ocp-Apim-Subscription-Key`.
    # it ends in `key` but not in any suffix above, so it was persisted verbatim. named as the whole
    # compound word rather than by relaxing to bare `key`, which `primary_key` and `sort_key` share
    # while carrying no credential.
    "subscriptionkey",
    "secret",
    "token",
    "password",
    "passwd",
    # a passphrase protects a private key or an encrypted archive. it is the same kind of secret as
    # a password and is spelled this way by ssh, gpg and pkcs tooling, but it ends in neither
    # `password` nor `passwd`, so `key_passphrase` was persisted unchanged.
    "passphrase",
    "credential",
    "credentials",
    "privatekey",
)
# words that conventionally trail a credential field without changing what it names:
# `password_confirmation` is still a password, `client_secret_value` still a secret. kept to a
# closed list of meaningless qualifiers so that nouns which DO change the meaning -- `token_count`
# (a length), `password_policy_url` (a link) -- keep their content.
_SECRET_KEY_QUALIFIERS = (
    "confirmation",
    "confirm",
    "value",
    "plaintext",
    "plain",
    "raw",
    "new",
    "old",
    "current",
)


# separators that join the WORDS of a field name without changing what it names. a JSON key uses
# `_` or `-`, but a human-readable label written into a request, an assistant reply or a tool result
# spells the same field `"API Key"` -- and leaving the space in place meant no `apikey` suffix
# matched, so that credential was persisted unchanged.
_SECRET_KEY_SEPARATORS = ("_", "-", " ", "\t", "\n", ".", "/")


def _normalize_secret_key(key: Any) -> str:
    """A field name reduced to its letters, so every spelling of one name compares equal."""
    normalized = str(key).casefold()
    for separator in _SECRET_KEY_SEPARATORS:
        normalized = normalized.replace(separator, "")
    return normalized


def _is_secret_key(key: Any, *, allow_token: bool = False) -> bool:
    normalized = _normalize_secret_key(key)
    if normalized in _SECRET_KEY_EXACT:
        return True
    for end in _secret_key_candidate_ends(normalized):
        # compared against the bound rather than a sliced copy: the slice is what made a long
        # stacked-qualifier name cost quadratic memory here.
        if allow_token and end == len("token") and normalized.startswith("token"):
            continue
        if any(normalized.endswith(suffix, 0, end) for suffix in _SECRET_KEY_SUFFIXES):
            return True
    return False


def _secret_key_candidate_ends(normalized: str) -> list[int]:
    """Where the key ends, plus where each form left by peeling a trailing qualifier ends.

    The suffix rule needs the sensitive term to END the key, so `password_confirmation` and
    `client_secret_value` fell through and their credentials were persisted unchanged. Only the
    qualifiers above are stripped, and only as a whole trailing word: they carry no meaning of
    their own, so what precedes them is still the field being named.

    Qualifiers STACK in practice -- `new_password_confirmation_value`, `client_secret_plaintext_value`
    -- and peeling only the last one left a form that still did not end in a secret suffix, so the
    credential survived. Peeling continues until no qualifier remains.

    Free-form containment would be wrong here -- `token_count` is a length and `password_policy_url`
    is a link, and redacting either would eat legitimate recorded content.

    An END OFFSET rather than the peeled string: slicing copied the shrinking key once per
    qualifier and kept every copy, which is quadratic in the key's length. A name built from
    stacked qualifiers -- well inside the ingress limit -- peaked at gigabytes and stalled the
    worker persisting the trace. `str.endswith` accepts the bound directly, so the offset carries
    everything the caller needs.
    """
    ends = [len(normalized)]
    end = len(normalized)
    while True:
        for qualifier in _SECRET_KEY_QUALIFIERS:
            if end > len(qualifier) and normalized.endswith(qualifier, 0, end):
                end -= len(qualifier)
                ends.append(end)
                break
        else:
            return ends


def _strip_required_anchors(pattern: str) -> str | None:
    """The body of a fully anchored pattern, or None when it does not constrain the whole match.

    A `patternProperties` key is applied as a search, not a full match, so only `^...$` names one
    field. `^(password)$` is accepted -- the anchors sit outside the group and still bound both
    ends -- but a bare `password`, a one-sided `^password`, and `(password)` are all substring
    expressions that also match longer ordinary names.
    """
    if not (pattern.startswith("^") and pattern.endswith("$")):
        return None
    body = pattern[1:-1]
    # `password\$` ends in an ESCAPED dollar: a literal character, not an anchor. an odd run of
    # trailing backslashes means the final `$` was escaped, so the match is unbounded at the end.
    if (len(body) - len(body.rstrip("\\"))) % 2:
        return None
    return body


def _normalize_exact_quantifiers(pattern: str) -> str:
    """Remove `{1}` repetitions, which name exactly what the element they follow already named.

    An anchored `^passw{1}ord$` matches only `password`, but the braces read as metacharacters, so
    the pattern was refused and the credential-bearing `default`, `const` and `enum` values under
    that perfectly valid schema stayed visible in raw exports.

    Only the count `1` is removed. `{2}` repeats the element and so spells a different name, and a
    RANGE (`{1,2}`, `{1,}`) matches several names at once; both are left in place, where the brace
    fails the name test and nothing is judged secret -- the direction that preserves an ordinary
    schema's annotations. A `{1}` with nothing before it, or one inside a character class, is not a
    quantifier at all and is likewise left alone.
    """
    result: list[str] = []
    index = 0
    quantifiable = False
    in_class = False
    closing_brace_index: int | None = None
    while index < len(pattern):
        character = pattern[index]
        if character == "\\" and index + 1 < len(pattern):
            result.append(pattern[index : index + 2])
            index += 2
            quantifiable = not in_class
            continue
        if in_class:
            result.append(character)
            in_class = character != "]"
            index += 1
            continue
        if character == "{" and quantifiable:
            # the position of the next `}` is remembered rather than searched afresh per brace,
            # which re-read the whole tail each time: a pattern of stacked `{` -- caller-controlled,
            # and rejected moments later anyway -- turned post-call sanitization quadratic.
            # `-1` records that the tail holds NO `}`, which stays true as the scan advances, so it
            # is sticky; re-searching on it would restore the quadratic scan it exists to prevent.
            if closing_brace_index is None or (
                closing_brace_index != -1 and closing_brace_index <= index
            ):
                closing_brace_index = pattern.find("}", index + 1)
            end = closing_brace_index
            if end != -1 and pattern[index + 1 : end] == "1":
                index = end + 1
                quantifiable = False
                continue
        result.append(character)
        in_class = character == "["
        # `(` and `|` open an element rather than completing one, so a brace directly after either
        # repeats nothing and stays a metacharacter.
        quantifiable = character not in "(|"
        index += 1
    return "".join(result)


def _is_secret_property_pattern(pattern: Any) -> bool:
    """Whether a `patternProperties` key matches property names this module treats as secret.

    The key is a REGEX, not a name, so the raw spelling is the wrong thing to test: `^password$`
    names exactly the field `password`, but it ends in `$` and failed the suffix rule, so the
    pattern's schema was never marked secret and its literals stayed in the raw export.

    Anchors and redundant grouping are stripped -- both constrain WHERE or HOW the expression
    matches, not what it names, so `^(password)$` names the same field as `password`. Anything
    else (character classes, quantifiers) is left in place, so a genuinely non-secret pattern is
    still judged on its full spelling rather than guessed at.

    An alternation names several fields at once. It is secret only when EVERY branch is, since one
    non-secret branch means the pattern also matches an ordinary property whose literals must
    survive: `^(password|api_key)$` is a credential either way, while `^(password|city)$` is not.

    BOTH anchors are required before any of that runs. `patternProperties` applies its key as a
    SUBSTRING match, so the unanchored `password` also matches `password_policy_url` -- an ordinary
    field `_is_secret_key` deliberately does not treat as a credential. Reading it as the single
    name `password` rewrote that field's annotations to "[redacted]" and corrupted the stored raw
    schema, so a pattern that does not constrain the whole match names nothing here.
    """
    if not isinstance(pattern, str):
        return False
    # `^password$|^api_key$` anchors each branch separately rather than the alternation as a whole,
    # so the split happens before the anchor test and every branch must carry its own pair.
    top_level = _alternation_branches(pattern)
    if top_level is not None:
        return all(_is_secret_property_pattern(branch) for branch in top_level)
    anchored = _strip_required_anchors(pattern)
    if anchored is None:
        return False
    unwrapped = _unwrap_pattern_groups(_normalize_exact_quantifiers(anchored))
    branches = _alternation_branches(_peel_enclosing_group(unwrapped))
    if branches is not None:
        return all(_is_secret_branch(_unwrap_pattern_groups(branch)) for branch in branches)
    names = _character_class_expansions(unwrapped)
    if names is None:
        return _is_secret_literal(unwrapped)
    return all(_is_secret_literal(name) for name in names)


# characters that make a pattern match more than the literal name it appears to spell. `.` is the
# one that mattered: the name test strips it as a word separator (`api.key` -> `apikey`), so
# `^api.key$` was read as a credential -- but as a regex it also matches `apiXkey`, an ordinary
# field whose schema annotations were then rewritten to "[redacted]", corrupting the stored schema.
# the rest are refused for the same reason, and because a pattern carrying them is not a name.
_PATTERN_METACHARACTERS = frozenset(".+*?()[]{}|^$")

# what a backslash may legally escape and still denote one literal character. the metacharacters,
# plus the separators tooling escapes out of caution: `\-` is a hyphen everywhere, and refusing it
# left `^api\-key$` -- a perfectly ordinary spelling of a credential name -- unrecognized.
_ESCAPABLE_LITERALS = _PATTERN_METACHARACTERS | frozenset("-/\\ ")

# escapes that spell one fixed character by NUMBER rather than by itself, as `\x77`, `\167` and
# `w` all spell `w`. each is a literal, so `^pass\x77ord$` matches exactly `password` -- and
# reading the escape as an opaque construct left that credential's schema unrecognized.
_HEX_ESCAPE_WIDTHS: dict[str, int] = {"x": 2, "u": 4, "U": 8}
_OCTAL_ESCAPE_DIGITS = 3
_MAX_OCTAL_ESCAPE = 0o377
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_OCTAL_DIGITS = frozenset("01234567")


def _decode_numeric_escape(pattern: str, index: int) -> tuple[str, int] | None:
    """The character a fixed numeric escape at `index` denotes, and the index just past it.

    `index` addresses the character AFTER the backslash. None means this is not a fixed numeric
    escape, which leaves the caller to refuse the pattern rather than guess at what it matches.
    """
    marker = pattern[index]
    width = _HEX_ESCAPE_WIDTHS.get(marker)
    if width is not None:
        digits = pattern[index + 1 : index + 1 + width]
        if len(digits) != width or not all(digit in _HEX_DIGITS for digit in digits):
            return None
        return _character_for(int(digits, 16), index + 1 + width)
    if marker not in _OCTAL_DIGITS:
        return None
    digits = ""
    while (
        len(digits) < _OCTAL_ESCAPE_DIGITS
        and index + len(digits) < len(pattern)
        and pattern[index + len(digits)] in _OCTAL_DIGITS
    ):
        digits += pattern[index + len(digits)]
    # `\0` opens an octal escape, and so does any run of exactly three octal digits: `\167` is `w`.
    # a SHORTER run that does not start at zero is a group backreference, which matches whatever
    # that group captured rather than one fixed character.
    if marker != "0" and len(digits) != _OCTAL_ESCAPE_DIGITS:
        return None
    code = int(digits, 8)
    # past `\377` the escape is not a valid pattern at all, so it names nothing.
    if code > _MAX_OCTAL_ESCAPE:
        return None
    return _character_for(code, index + len(digits))


def _character_for(code: int, end: int) -> tuple[str, int] | None:
    """The character a decoded code point denotes. None past the Unicode range, which no name
    contains, so refusing it judges nothing secret -- the safe direction."""
    return (chr(code), end) if code <= 0x10FFFF else None


def _literal_name(pattern: str) -> str | None:
    """The single field name a pattern matches, or None if it matches more than that one name.

    An ESCAPED metacharacter is the literal character, not the construct: `^api\\.key$` matches only
    `api.key`, which is a credential by this module's rules. Refusing every backslash left that
    valid `patternProperties` entry unrecognized, so the schema beneath it kept its credential
    literals in the raw export -- the leak this test exists to close, arrived at from the other
    side. Escapes are therefore decoded to the character they denote.

    A FIXED numeric escape is a literal for the same reason: `\\x77`, `\\167` and `\\u0077` all spell
    `w`, so `^pass\\x77ord$` matches exactly `password`.

    What is not decoded is what does not denote one character. `\\d`, `\\w` and `\\s` are classes
    matching many names; `\\1` is a backreference to whatever a group captured; an unterminated
    trailing backslash is not a name at all. Each returns None, which judges nothing secret and is
    the safe direction.
    """
    decoded: list[str] = []
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "\\":
            if index + 1 >= len(pattern):
                return None
            following = pattern[index + 1]
            if following in _ESCAPABLE_LITERALS:
                decoded.append(following)
                index += 2
                continue
            numeric = _decode_numeric_escape(pattern, index + 1)
            if numeric is None:
                return None
            literal, index = numeric
            decoded.append(literal)
            continue
        if character in _PATTERN_METACHARACTERS:
            return None
        decoded.append(character)
        index += 1
    return "".join(decoded)


def _is_secret_literal(pattern: str) -> bool:
    """Whether a pattern spells exactly one field name, and that name is a credential."""
    name = _literal_name(pattern)
    return name is not None and _is_secret_key(name)


def _is_secret_branch(branch: str) -> bool:
    """Whether one branch of an alternation names a credential, by literal or by simple class.

    A branch is judged exactly as a whole pattern is, because it IS one: `^(api.key|password)$`
    otherwise passed its wildcard branch straight to the name test, which is the same corruption
    the whole-pattern case had.
    """
    names = _character_class_expansions(branch)
    if names is None:
        return _is_secret_literal(branch)
    return all(_is_secret_literal(name) for name in names)


# a class is only enumerated when the set stays small enough that expanding it is cheaper than the
# leak it prevents. names are compared case-folded, so the fully spelled `[Pp][Aa]...` form of one
# word collapses to a single name and stays well inside the cap; a genuinely wide class does not.
_MAX_CLASS_EXPANSIONS = 64


def _character_class_expansions(pattern: str) -> tuple[str, ...] | None:
    """Every name a pattern of literals and simple classes matches, or None if it is not one.

    Tooling writes a case-insensitive property as `^[Pp]assword$`, and both names it matches are
    credentials by this module's own rules -- but the pattern is neither a literal nor an
    alternation, so the name test saw a regex and every schema beneath it kept its literals.

    Only CLOSED classes of plain characters are expanded. A range (`[a-z]`), a negation (`[^x]`) or
    any class carrying an escape is left unexpanded, because `pass[a-z]ord` also matches `passzord`
    and treating it as a credential name would blank an ordinary property's schema. Returning None
    falls back to judging the pattern whole, which is the safe direction.
    """
    if not any(character in pattern for character in "[]"):
        return None
    names: set[str] = {""}
    # literal runs are accumulated and appended once per CLASS rather than once per character.
    # rebuilding every name for each literal made classification quadratic in a caller-controlled
    # pattern -- `^[Pp]` with an 80,000-character suffix took 0.69s, on the persistence path that
    # runs after the paid upstream call -- while the expanded set is unchanged.
    literal: list[str] = []
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "\\" or character == "]":
            return None
        if character != "[":
            literal.append(character)
            index += 1
            continue
        end = pattern.find("]", index + 1)
        if end == -1:
            return None
        members = pattern[index + 1 : end]
        if not members or "-" in members or members.startswith("^") or "\\" in members:
            return None
        prefix = "".join(literal)
        literal.clear()
        # names are deduplicated case-insensitively as they grow, because the comparison that
        # judges them is case-folded too. spelling one word as `[Pp][Aa][Ss][Ss]...` would
        # otherwise multiply out to hundreds of variants of a single name and blow the cap.
        names = {_dedupe_name(name + prefix + member) for name in names for member in members}
        if len(names) > _MAX_CLASS_EXPANSIONS:
            return None
        index = end + 1
    if literal:
        suffix = "".join(literal)
        names = {name + suffix for name in names}
    return tuple(names)


def _dedupe_name(name: str) -> str:
    """One spelling of a name, so case variants of the same word count once."""
    return name.casefold()


def _peel_enclosing_group(pattern: str) -> str:
    """Remove one group that wraps the WHOLE expression, keeping any alternation inside it.

    `_unwrap_pattern_groups` deliberately refuses to unwrap a group containing `|`, because
    `(password|city)` is not the single name `password`. The all-secret test needs to see those
    branches, so this peels the wrapper for that inspection only -- what it returns is never used
    as a field name.
    """
    if not (pattern.startswith("(") and pattern.endswith(")")):
        return pattern
    inner = pattern[1:-1].removeprefix("?:")
    depth = 0
    for character in inner:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                return pattern
    return inner if depth == 0 else pattern


def _alternation_branches(pattern: str) -> tuple[str, ...] | None:
    """Split a top-level alternation into its branches, or None if there is no top-level `|`.

    Only alternation bars OUTSIDE any group or character class separate branches: in `(a|b)c|d` the
    first bar belongs to the group and splitting on it would invent branches the regex never has.
    An escaped bar is a literal character, not a separator.
    """
    branches: list[str] = []
    current: list[str] = []
    depth = 0
    in_class = False
    escaped = False
    for character in pattern:
        if escaped:
            current.append(character)
            escaped = False
            continue
        if character == "\\":
            current.append(character)
            escaped = True
            continue
        if in_class:
            current.append(character)
            if character == "]":
                in_class = False
            continue
        if character == "[":
            in_class = True
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "|" and depth == 0:
            branches.append("".join(current))
            current = []
            continue
        current.append(character)
    if not branches or depth != 0 or in_class or escaped:
        # an unbalanced or truncated expression is untrustworthy recorded input; judge it whole
        # rather than acting on a split that may not reflect what it matches.
        return None
    branches.append("".join(current))
    return tuple(branches)


def _unwrap_pattern_groups(pattern: str) -> str:
    """Strip anchors and redundant enclosing groups from a regex, leaving the name it matches.

    `^(password)$` and `^password$` name the same property, but removing only the anchors left
    `(password)`, which the name test did not recognize -- so the pattern's schema was not marked
    secret and its credential literals stayed in the raw export.

    A group is only unwrapped when it encloses the WHOLE remaining expression; `(a)x(b)` keeps its
    parentheses because they no longer name a single field. Alternation inside a group is left
    alone for the same reason: `(password|city)` is not simply `password`.
    """
    # every parenthesis is matched, and every `|` counted, in ONE pass up front. asking per layer --
    # slicing the remainder and rescanning it for balance and bars -- re-read the same characters
    # once per level of nesting, so a pattern of redundant nesting, caller-controlled and inside the
    # ingress limit, cost quadratic time: 16,000 groups took 7.8s during post-call trace
    # persistence, which runs after the paid upstream call and occupies a worker. With the two
    # lookups precomputed, each layer costs a constant amount and the loop becomes linear overall.
    closes = _group_close_positions(pattern)
    preceding_bars = _preceding_bar_counts(pattern)
    start, end = 0, len(pattern)
    while True:
        # one anchor at each end per pass, repeated: a doubled `^^name$$` peels down to the name it
        # spells, exactly as the repeated `removeprefix`/`removesuffix` did.
        stripped = False
        if start < end and pattern[start] == "^":
            start += 1
            stripped = True
        if end > start and pattern[end - 1] == "$":
            end -= 1
            stripped = True
        peeled = False
        # the group opening the span must close exactly at its end, or it wraps only part of the
        # expression: `(a)x(b)` keeps its parentheses because they no longer name a single field.
        if end - start >= 2 and pattern[start] == "(" and closes[start] == end - 1:
            inner_start = start + 1
            if pattern.startswith("?:", inner_start, end - 1):
                inner_start += 2
            # a bar anywhere inside makes the group a branch set rather than a name:
            # `(password|city)` is not simply `password`.
            if preceding_bars[end - 1] == preceding_bars[inner_start]:
                start, end = inner_start, end - 1
                peeled = True
        if not (stripped or peeled):
            break
    return _flatten_inner_groups(pattern[start:end])


def _group_close_positions(pattern: str) -> list[int]:
    """For each `(`, the index of the `)` that closes it -- or -1 when it is never closed.

    Escapes are deliberately NOT honoured: the balance test this replaces read the group's raw
    text, where `\\(` counts as an opening parenthesis like any other. Honouring them here would
    unwrap a group the previous form refused, changing which patterns name a credential.
    """
    closes = [-1] * len(pattern)
    open_indices: list[int] = []
    for index, character in enumerate(pattern):
        if character == "(":
            open_indices.append(index)
        elif character == ")" and open_indices:
            closes[open_indices.pop()] = index
    return closes


def _preceding_bar_counts(pattern: str) -> list[int]:
    """How many `|` appear before each index, so any span's count is the difference of two reads.

    Escapes are not honoured here either, for the same reason: the containment test this replaces
    asked `"|" in inner` of the raw text, where an escaped bar counts.
    """
    counts = [0] * (len(pattern) + 1)
    for index, character in enumerate(pattern):
        counts[index + 1] = counts[index] + (character == "|")
    return counts


_QUANTIFIERS = frozenset({"?", "*", "+", "{"})


def _flatten_inner_groups(pattern: str) -> str:
    """Splice out groups that merely bracket part of a name, leaving the name itself.

    Unwrapping only whole-expression groups left `pass(?:word)` intact, so the name test saw a
    regex rather than `password` and the schema beneath the pattern kept its credential literals.
    Tooling brackets name fragments routinely, and the grouping does not change what is matched.

    Only groups that plainly consume their contents are spliced. A quantified group (`(?:word)?`)
    matches with OR without the fragment, so it names two different fields and is left alone rather
    than guessed at; a lookaround consumes nothing; alternation is a branch set handled separately.
    Anything left unspliced simply fails the name test, which is the safe direction.

    A level is spliced only when EVERY group at that level can be -- one that cannot leaves the
    expression as the previous level left it -- so the shallowest unspliceable group decides how
    far the splicing reaches. Finding that depth takes one pass over the extents, where repeating a
    whole pass per level rescanned and rebuilt the expression once per level of nesting: `pass`
    followed by 8,000 nested `(?:` cost 9.6s during post-call trace persistence, which runs after
    the paid upstream call and occupies a worker.
    """
    extents = _pattern_group_extents(pattern)
    if extents is None:
        # a `(` that is never closed leaves the extents unknowable, so the expression is judged
        # whole rather than on a splice that may not reflect what it matches.
        return pattern
    # the shallowest group that cannot be spliced stops the splicing one level above itself. with
    # none, every level splices -- the ordinary case, and the one a `default=0` would silently turn
    # into "splice nothing".
    deepest = max((depth for _, _, depth, _ in extents), default=0)
    limit = min((depth for _, _, depth, splices in extents if not splices), default=deepest + 1) - 1
    if limit < 1:
        return pattern
    dropped = bytearray(len(pattern))
    for start, end, depth, _ in extents:
        if depth > limit:
            continue
        dropped[start] = dropped[end] = 1
        if pattern.startswith("?:", start + 1, end):
            dropped[start + 1] = dropped[start + 2] = 1
    return "".join(character for index, character in enumerate(pattern) if not dropped[index])


def _pattern_group_extents(pattern: str) -> list[tuple[int, int, int, bool]] | None:
    """Each group's start, end, nesting depth, and whether it splices plainly -- in one pass.

    None when some `(` is never closed. Every ancestor of an unclosed group is unclosed too, so the
    outermost scan reached one and judged the expression whole; reporting it here says the same.
    """
    open_groups: list[tuple[int, int]] = []
    extents: list[tuple[int, int, int, bool]] = []
    matched_closers = bytearray(len(pattern))
    bars = 0
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "\\":
            # an ESCAPED bar counts: the containment test this replaces read the group's raw text,
            # where `\|` holds a `|` like any other. stepping over the escape without counting it
            # would splice a group the previous form refused.
            bars += index + 1 < len(pattern) and pattern[index + 1] == "|"
            index += 2
            continue
        if character == "|":
            bars += 1
        elif character == "(":
            open_groups.append((index, bars))
        elif character == ")" and open_groups:
            start, bars_before = open_groups.pop()
            extents.append((start, index, len(open_groups) + 1, bars > bars_before))
            matched_closers[index] = 1
        index += 1
    if open_groups:
        return None
    return _with_splice_verdicts(pattern, extents, matched_closers)


def _with_splice_verdicts(
    pattern: str, extents: list[tuple[int, int, int, bool]], matched_closers: bytearray
) -> list[tuple[int, int, int, bool]]:
    """Replace each extent's `|` flag with whether that group splices plainly.

    A group is left alone when it does not plainly consume its contents: `?` opens a lookaround or
    a flag group, a quantifier makes the contents optional or repeated, and `|` makes it a branch
    set. The quantifier sits after the group's `)` in the string AS IT WILL BE when this group is
    the outermost one, which is the original with its ancestors' closers gone -- so the closers are
    stepped over, and only those of groups that actually matched, since a stray `)` is a literal
    that stays put. That lookup is precomputed once; walking the run per group would restore the
    quadratic cost on the deeply nested patterns this exists to bound.
    """
    following_index = list(range(len(pattern) + 1))
    for index in range(len(pattern) - 1, -1, -1):
        if matched_closers[index]:
            following_index[index] = following_index[index + 1]
    verdicts = []
    for start, end, depth, contains_bar in extents:
        lookaround = pattern.startswith("?", start + 1, end) and not pattern.startswith(
            "?:", start + 1, end
        )
        following = following_index[end + 1]
        quantified = following < len(pattern) and pattern[following] in _QUANTIFIERS
        verdicts.append((start, end, depth, not (lookaround or quantified or contains_bar)))
    return verdicts
