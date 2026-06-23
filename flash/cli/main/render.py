"""Human-facing rendering for `flash login` / `flash whoami` (stdlib only).

The identity behind a stored key is shown as a small aligned card instead of raw
``json.dumps`` output. ANSI styling and unicode glyphs are applied only when stdout is
an interactive terminal that can encode them; piped, captured, or ASCII-locale output
stays plain so it can be grepped or parsed and never raises ``UnicodeEncodeError``.
"""

from __future__ import annotations

import os
import sys

# Identity fields the control plane may return (flash/server/app.py `/v1/me`), in the
# order we show them. ``key_prefix``/``kind`` render separately on the key line.
_ROWS = (
    ("email", "account"),
    ("org_id", "org"),
    ("user_id", "user"),
    ("project_id", "project"),
    ("training_agent_job_id", "job"),
)

_KIND_LABEL = {
    "internal": "internal key",
    "freesolo_api_key": "freesolo key",
}


def _color() -> bool:
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"


def _style(code: str, text: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if _color() else text


def _glyph(unicode_glyph: str, ascii_fallback: str) -> str:
    """Use the unicode glyph only if stdout can encode it (else an ASCII stand-in)."""
    enc = sys.stdout.encoding or "ascii"
    try:
        unicode_glyph.encode(enc)
    except (UnicodeError, LookupError):
        return ascii_fallback
    return unicode_glyph


# Common unicode punctuation we'd rather downgrade to a readable ASCII stand-in than to an
# escape sequence when stdout can't encode it (e.g. an ASCII/non-UTF-8 locale). Written as
# escapes so the source stays free of confusable characters.
_ASCII_PUNCT = {
    0x2014: "-",  # em dash
    0x2013: "-",  # en dash
    0x2026: "...",  # ellipsis
    0x2019: "'",  # right single quote
    0x201C: '"',  # left double quote
    0x201D: '"',  # right double quote
}


def _safe(text: str) -> str:
    """Guarantee ``text`` can be printed under the current stdout encoding.

    Identity values from ``/v1/me`` (e.g. an internationalized email) or punctuation in our
    own copy must never make ``print()`` raise ``UnicodeEncodeError`` after a login has
    already succeeded. On a normal UTF-8 terminal the text is returned unchanged; only when
    the active encoding can't represent it do we downgrade punctuation and escape the rest.
    """
    enc = sys.stdout.encoding or "ascii"
    try:
        text.encode(enc)
    except UnicodeEncodeError:
        return text.translate(_ASCII_PUNCT).encode(enc, "backslashreplace").decode(enc)
    return text


def _bold(s: str) -> str:
    return _style("1", s)


def _dim(s: str) -> str:
    return _style("2", s)


def format_identity(me: dict) -> str:
    """Render the stored key's identity as an aligned card (not raw JSON)."""
    rows = [(label, str(me[field])) for field, label in _ROWS if me.get(field)]
    prefix = me.get("key_prefix")
    kind = _KIND_LABEL.get(me.get("kind", ""), me.get("kind") or "api key")
    rows.append(("key", f"{prefix}{_glyph('…', '...')} {_dim(f'({kind})')}" if prefix else kind))
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"  {_dim(label.ljust(width))}  {value}" for label, value in rows)


def login_ok(me: dict | None) -> str:
    head = f"{_style('32', _glyph('✓', 'ok:'))} {_bold('logged in to flash')}"
    if not me:
        # The key is verified and stored; we just couldn't fetch the account card right now.
        return _safe(
            f"{head}\n{_dim('  account details unavailable right now — run `flash whoami` later')}"
        )
    return _safe(f"{head}\n\n{format_identity(me)}")


def whoami(me: dict) -> str:
    return _safe(f"{_bold('logged in to flash')}\n\n{format_identity(me)}")


def login_failed(reason: str) -> str:
    return _safe(
        f"{_style('31', _glyph('✗', 'x:'))} {_bold('login failed')}\n"
        f"  {reason}\n"
        f"  {_dim('then run `flash login --api-key <key>` to try again')}\n"
        f"  {_dim('if it keeps failing, email founders@freesolo.co')}"
    )


def run_banner(run_id: str, *, algorithm: str, model: str, gpu: str, seeds, resume_hint: str) -> str:
    """A boxed header for `flash train` so the run id and key facts stand out above the log stream.

    Box-drawing degrades to ASCII when stdout can't encode it. The allocated GPU price isn't known
    until the server allocates, so it surfaces in the streamed `allocated ...` line, not here.
    """
    tl, tr = _glyph("┌", "+"), _glyph("┐", "+")
    bl, br = _glyph("└", "+"), _glyph("┘", "+")
    hbar, vbar = _glyph("─", "-"), _glyph("│", "|")
    # plain text drives width/padding; styled text is what we print, so ANSI codes in the styled
    # run id never skew the box alignment (escape sequences are zero-width on screen).
    rows = [
        (run_id, _bold(run_id)),
        (f"{algorithm}  -  {model}",) * 2,
        (f"{gpu}  -  seeds {list(seeds)}",) * 2,
    ]
    title = f"{hbar} flash run "
    width = max(max(len(plain) for plain, _ in rows), len(title))
    span = width + 2  # a space of padding on each side of the content
    top = tl + title + hbar * (span - len(title)) + tr
    body = "\n".join(f"{vbar} {styled}{' ' * (width - len(plain))} {vbar}" for plain, styled in rows)
    bottom = bl + hbar * span + br
    return _safe(f"{top}\n{body}\n{bottom}\n{_dim(resume_hint)}")
