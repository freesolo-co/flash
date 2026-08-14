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
    "secret",
    "token",
    "password",
    "passwd",
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
    for candidate in _secret_key_candidates(normalized):
        if not (allow_token and candidate == "token") and any(
            candidate.endswith(suffix) for suffix in _SECRET_KEY_SUFFIXES
        ):
            return True
    return False


def _secret_key_candidates(normalized: str) -> tuple[str, ...]:
    """The key itself, plus each form left as conventional trailing qualifiers are peeled off.

    The suffix rule needs the sensitive term to END the key, so `password_confirmation` and
    `client_secret_value` fell through and their credentials were persisted unchanged. Only the
    qualifiers above are stripped, and only as a whole trailing word: they carry no meaning of
    their own, so what precedes them is still the field being named.

    Qualifiers STACK in practice -- `new_password_confirmation_value`, `client_secret_plaintext_value`
    -- and peeling only the last one left a form that still did not end in a secret suffix, so the
    credential survived. Peeling continues until no qualifier remains.

    Free-form containment would be wrong here -- `token_count` is a length and `password_policy_url`
    is a link, and redacting either would eat legitimate recorded content.
    """
    candidates = [normalized]
    current = normalized
    while True:
        for qualifier in _SECRET_KEY_QUALIFIERS:
            if current.endswith(qualifier) and len(current) > len(qualifier):
                current = current[: -len(qualifier)]
                candidates.append(current)
                break
        else:
            return tuple(candidates)


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
    """
    if not isinstance(pattern, str):
        return False
    unwrapped = _unwrap_pattern_groups(pattern)
    branches = _alternation_branches(_peel_enclosing_group(unwrapped))
    if branches is not None:
        return all(_is_secret_key(_unwrap_pattern_groups(branch)) for branch in branches)
    return _is_secret_key(unwrapped)


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
    previous = None
    current = pattern
    while current != previous:
        previous = current
        current = current.removeprefix("^").removesuffix("$")
        if not (current.startswith("(") and current.endswith(")")):
            continue
        inner = current[1:-1].removeprefix("?:")
        depth = 0
        for character in inner:
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth < 0:
                    break
        if depth == 0 and "|" not in inner:
            current = inner
    return _flatten_inner_groups(current)


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
    """
    result: list[str] = []
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "\\" and index + 1 < len(pattern):
            result.append(pattern[index : index + 2])
            index += 2
            continue
        if character != "(":
            result.append(character)
            index += 1
            continue
        end = _group_end(pattern, index)
        if end is None:
            return pattern
        inner = pattern[index + 1 : end]
        quantified = end + 1 < len(pattern) and pattern[end + 1] in _QUANTIFIERS
        if inner.startswith("?") and not inner.startswith("?:"):
            return pattern
        if quantified or "|" in inner:
            return pattern
        result.append(inner.removeprefix("?:"))
        index = end + 1
    return "".join(result)


def _group_end(pattern: str, start: int) -> int | None:
    """Index of the `)` closing the group opened at `start`, or None if it is never closed."""
    depth = 0
    index = start
    while index < len(pattern):
        character = pattern[index]
        if character == "\\":
            index += 2
            continue
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None
