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
        return (
            f"{head}\n{_dim('  account details unavailable right now — run `flash whoami` later')}"
        )
    return f"{head}\n\n{format_identity(me)}"


def whoami(me: dict) -> str:
    return f"{_bold('logged in to flash')}\n\n{format_identity(me)}"


def login_failed(reason: str) -> str:
    return (
        f"{_style('31', _glyph('✗', 'x:'))} {_bold('login failed')}\n"
        f"  {reason}\n"
        f"  {_dim('then run `flash login --api-key <key>` to try again')}\n"
        f"  {_dim('if it keeps failing, email founders@freesolo.co')}"
    )
