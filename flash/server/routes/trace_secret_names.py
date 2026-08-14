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


def _is_secret_key(key: Any, *, allow_token: bool = False) -> bool:
    normalized = str(key).casefold().replace("_", "").replace("-", "")
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
    else (alternation, character classes, quantifiers) is left in place, so a genuinely non-secret
    pattern is still judged on its full spelling rather than guessed at.
    """
    if not isinstance(pattern, str):
        return False
    return _is_secret_key(_unwrap_pattern_groups(pattern))


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
    return current
