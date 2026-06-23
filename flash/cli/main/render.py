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


def _flag(name: str) -> bool | None:
    """Tri-state read of an env flag: True/False when set to a recognizable value, else None."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    val = raw.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"", "0", "false", "no", "off"}:
        return False
    # Unrecognized value (e.g. a typo): stay tri-state None so styled() falls back to isatty
    # rather than silently forcing the theme on.
    return None


def styled() -> bool:
    """Whether to render the themed human layout (tables, badges, panels) instead of the plain
    machine form. True on an interactive stdout; ``FLASH_STYLE=1``/``0`` forces it on or off
    (used to render the docs preview, and to keep output deterministic in scripts)."""
    forced = _flag("FLASH_STYLE")
    if forced is not None:
        return forced
    return sys.stdout.isatty()


def _color() -> bool:
    """Whether ANSI color may be emitted: the themed layout, minus a NO_COLOR / dumb-terminal opt-out."""
    return styled() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"


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
    head = f"{_paint(_glyph('✓', 'ok:'), _GREEN, '1')} {_bold('logged in to flash')}"
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
        f"{_paint(_glyph('✗', 'x:'), _RED, '1')} {_bold('login failed')}\n"
        f"  {reason}\n"
        f"  {_dim('then run `flash login --api-key <key>` to try again')}\n"
        f"  {_dim('if it keeps failing, email founders@freesolo.co')}"
    )


# ---------------------------------------------------------------------------
# Theme — one visual language shared by every styled command.
#
# These renderers are only ever called when `styled()` is true (an interactive stdout, or
# FLASH_STYLE=1). Each command keeps its exact plain/JSON output on the machine path, so
# `jq`, scripts, and the agent contract are untouched; this is purely the human view.
# 256-color SGR keeps the palette readable on essentially every modern terminal.
# ---------------------------------------------------------------------------

# The Freesolo brand palette, pulled from the website (frontend/app/globals.css): navy
# `--special` #1b1b4b, periwinkle `--periwinkle` #5f72ff (the accent/ring), green `--green`
# #57ff8f, and the deep teal `--green-deep` #00695c. Each entry is (hex, xterm-256 fallback):
# truecolor terminals (and the docs gallery) get the exact brand hex, everything else the
# nearest 256-color approximation. Periwinkle is the accent; green is success; a brightened
# teal carries amounts; a soft indigo (between navy and periwinkle) marks JSON literals.
_ACCENT = ("5f72ff", 63)  # periwinkle — brand accent
_ACCENT2 = ("97a3ff", 111)  # light periwinkle — keys, ids, links, snippets
_GREEN = ("57ff8f", 84)  # brand green — success
_TEAL = ("24c2a8", 43)  # brightened green-deep — amounts / numbers
_RED = ("ff6b6b", 203)  # destructive
_VIOLET = ("9a8cff", 105)  # navy→periwinkle indigo — JSON literals
_GRAY = ("8a93a8", 245)  # cool neutral
_FAINT = ("4d5470", 240)  # muted indigo-grey — rules, punctuation


def _truecolor() -> bool:
    """Whether the terminal advertises 24-bit color (so we can emit exact brand hex)."""
    return os.environ.get("COLORTERM", "").lower() in {"truecolor", "24bit"}


def _sgr(part: str | tuple[str, int]) -> str:
    """Resolve one style token to an SGR parameter: a raw code (``"1"``) passes through; a
    ``(hex, fallback)`` color becomes a truecolor or 256-color foreground."""
    if isinstance(part, tuple):
        hex6, fallback = part
        if _truecolor():
            return f"38;2;{int(hex6[0:2], 16)};{int(hex6[2:4], 16)};{int(hex6[4:6], 16)}"
        return f"38;5;{fallback}"
    return part


def _paint(text: str, *codes: str | tuple[str, int]) -> str:
    """Apply one or more style tokens (raw SGR codes and/or brand colors), honoring the gate."""
    if not codes or not _color():
        return text
    return f"\x1b[{';'.join(_sgr(c) for c in codes)}m{text}\x1b[0m"


# state -> (color, unicode dot, ascii dot) for status badges
_STATE_STYLE: dict[str, tuple[str, str, str]] = {
    "queued": (_GRAY, "○", "o"),
    "provisioning": (_TEAL, "◐", "o"),
    "running": (_ACCENT, "●", "*"),
    "done": (_GREEN, "●", "*"),
    "deployed": (_VIOLET, "●", "*"),
    "ready": (_GREEN, "●", "*"),
    "failed": (_RED, "●", "*"),
    "error": (_RED, "●", "*"),
    "cancelled": (_GRAY, "●", "*"),
    "canceled": (_GRAY, "●", "*"),
    "dry_run": (_ACCENT2, "○", "o"),
}


def _term_width(cap: int = 80) -> int:
    import shutil

    cols = shutil.get_terminal_size((80, 24)).columns
    return max(24, min(cols, cap))


def _rule(width: int | None = None) -> str:
    return _paint(_glyph("─", "-") * (width or _term_width()), _FAINT)


def header(cmd: str, desc: str | None = None) -> str:
    """Brand header line + a faint rule: the wordmark, the command, and an optional descriptor."""
    mark = _paint("flash", _ACCENT, "1")
    sep = _paint(_glyph("›", ">"), _FAINT)  # noqa: RUF001 (the glyph is the point)
    line = f"{mark} {sep} {_bold(cmd)}"
    if desc:
        line += "  " + _paint(desc, _GRAY)
    return _safe(f"{line}\n{_rule()}")


def badge(state: str) -> str:
    """A colored status dot + label, e.g. a green ``● done``."""
    color, uni, ascii_dot = _STATE_STYLE.get((state or "").lower(), (_GRAY, "•", "-"))
    return _paint(f"{_glyph(uni, ascii_dot)} {state}", color)


def ok(msg: str) -> str:
    return _safe(f"{_paint(_glyph('✓', 'ok:'), _GREEN, '1')} {msg}")


def arrow(msg: str) -> str:
    """A de-emphasized pointer / next-step line."""
    return _safe(f"{_paint(_glyph('→', '->'), _ACCENT2)} {_dim(msg)}")


def money(value: float, decimals: int = 4) -> str:
    return _paint(f"${value:.{decimals}f}", _TEAL)


def _kv(pairs: list[tuple[str, str | None]], indent: int = 2) -> str:
    """Aligned ``key · value`` panel; keys dimmed and padded to a common width."""
    rows = [(k, v) for k, v in pairs if v is not None]
    if not rows:
        return ""
    keyw = max(len(k) for k, _ in rows)
    pad = " " * indent
    sep = _paint(_glyph("·", "-"), _FAINT)
    return "\n".join(f"{pad}{_paint(k.ljust(keyw), _GRAY)} {sep} {v}" for k, v in rows)


def _table(headers: list[str], rows: list[list], aligns: list[str] | None = None) -> str:
    """Aligned table with a dim header and faint underline. Each cell is a plain string or a
    ``(text, *sgr_codes)`` tuple; widths come from the plain text so color never skews them."""
    aligns = aligns or ["l"] * len(headers)
    cols = len(headers)

    def plain(cell) -> str:
        return cell[0] if isinstance(cell, tuple) else str(cell)

    widths = [len(h) for h in headers]
    for row in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(plain(row[i])))

    def fmt(cell, w: int, align: str) -> str:
        text = plain(cell)
        padded = f"{text:>{w}}" if align == "r" else f"{text:<{w}}"
        if isinstance(cell, tuple) and len(cell) > 1:
            return _paint(padded, *cell[1:])
        return padded

    head = "  ".join(
        _paint(f"{h:>{widths[i]}}" if aligns[i] == "r" else f"{h:<{widths[i]}}", _GRAY, "1")
        for i, h in enumerate(headers)
    )
    underline = _paint("  ".join(_glyph("─", "-") * w for w in widths), _FAINT)
    # rstrip each row so an uncolored trailing column leaves no dangling whitespace.
    body = [
        "  ".join(fmt(row[i], widths[i], aligns[i]) for i in range(cols)).rstrip() for row in rows
    ]
    return _safe("\n".join([head.rstrip(), underline, *body]))


def _json(obj) -> str:
    """Pretty JSON: plain ``json.dumps(indent=2)`` when color is off (byte-identical to the
    legacy output), syntax-highlighted when it's on. No data is ever dropped."""
    import json

    if not _color():
        return json.dumps(obj, indent=2)
    return _safe(_color_json(obj, 0))


def _color_json(obj, depth: int) -> str:
    import json

    pad_in = "  " * (depth + 1)
    pad_out = "  " * depth
    colon = _paint(":", _FAINT)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [
            f"{pad_in}{_paint(json.dumps(k), _ACCENT2)}{colon} {_color_json(v, depth + 1)}"
            for k, v in obj.items()
        ]
        return "{\n" + ",\n".join(items) + f"\n{pad_out}}}"
    if isinstance(obj, list):
        if not obj:
            return "[]"
        items = [f"{pad_in}{_color_json(v, depth + 1)}" for v in obj]
        return "[\n" + ",\n".join(items) + f"\n{pad_out}]"
    if obj is None:
        return _paint("null", _VIOLET)
    if isinstance(obj, bool):
        return _paint("true" if obj else "false", _VIOLET)
    if isinstance(obj, (int, float)):
        return _paint(json.dumps(obj), _TEAL)
    return _paint(json.dumps(obj), _GREEN)


# ---------------------------------------------------------------------------
# Per-command renderers (themed view only; the command supplies the data).
# ---------------------------------------------------------------------------


def version(value: str) -> str:
    """The wordmark + version + tagline."""
    mark = _paint("flash", _ACCENT, "1")
    return _safe(f"{mark} {_dim('v' + value)}\n{_dim('managed lora post-training')}")


def models_table(rows: list[dict]) -> str:
    """Supported base models — a clean themed list of ids (the CLI lists ids only)."""
    ids = "\n".join(f"  {_paint('•', _FAINT)} {_paint(r['id'], _ACCENT2)}" for r in rows)
    foot = arrow("train one with: flash train configs/rl.toml")
    return f"{header('models', 'supported base models')}\n{ids}\n\n{foot}"


def gpus_table(rows: list[tuple[str, int, float | None]], tip: str) -> str:
    """GPU classes: (name, vram_gb, $/hr or None)."""
    body = []
    for name, vram, rate in rows:
        rate_cell = (f"${rate:.2f}", _TEAL) if rate else ("-", _FAINT)
        body.append([(name, _ACCENT2), (f"{vram} GB", _GRAY), rate_cell])
    table = _table(["GPU", "VRAM", "$/HR"], body, aligns=["l", "r", "r"])
    return f"{header('gpus', 'managed GPU classes')}\n{table}\n\n{_dim(tip)}"


def runs_table(runs: list[dict]) -> str:
    """Runs list: state badges + cost, newest first."""
    body = []
    for r in sorted(runs, key=lambda r: r.get("updated_at", 0), reverse=True):
        spec = r.get("spec") or {}
        model = spec.get("model", "")
        algorithm = str(spec.get("algorithm") or "-").upper()
        remote = r.get("remote") or {}
        provider = remote.get("provider") or (
            "runpod" if remote else (spec.get("gpu") or {}).get("provider", "")
        )
        gpu = remote.get("gpu") or (spec.get("gpu") or {}).get("type", "")
        where = f"{gpu}@{provider}" if provider else gpu
        color, uni, ascii_dot = _STATE_STYLE.get(str(r.get("state", "")).lower(), (_GRAY, "•", "-"))
        body.append(
            [
                (r["run_id"], _ACCENT2),
                (f"{_glyph(uni, ascii_dot)} {r.get('state', '')}", color),
                (algorithm, _GRAY),
                (f"${r.get('cost_usd', 0.0):.4f}", _TEAL),
                (where, _GRAY),
                model,
            ]
        )
    table = _table(
        ["RUN ID", "STATE", "ALGO", "COST", "GPU", "MODEL"],
        body,
        aligns=["l", "l", "l", "r", "l", "l"],
    )
    return f"{header('runs', f'{len(runs)} run(s)')}\n{table}"


def deployments_table(rows: list[dict]) -> str:
    body = []
    for r in rows:
        d = r.get("deployment") or {}
        body.append(
            [
                (r["run_id"], _ACCENT2),
                (d.get("gpu", "?"), _GRAY),
                (d.get("endpoint_name", ""), _GREEN),
            ]
        )
    table = _table(["RUN ID", "GPU", "ENDPOINT"], body)
    return f"{header('deployments', f'{len(rows)} active')}\n{table}"


def empty(cmd: str, desc: str, message: str) -> str:
    """A styled empty state (e.g. no runs yet)."""
    return f"{header(cmd, desc)}\n{_dim('  ' + message)}"


def _humanize_ts(value) -> str | None:
    """Format an epoch seconds value as a compact UTC timestamp, leaving non-numbers alone."""
    if not isinstance(value, (int, float)) or value <= 0:
        return None
    import datetime

    return datetime.datetime.fromtimestamp(value, datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")


def run_status(obj: dict) -> str:
    """A curated status panel for `flash status`, with the full JSON below for completeness."""
    spec = obj.get("spec") or {}
    remote = obj.get("remote") or {}
    gpu = remote.get("gpu") or (spec.get("gpu") or {}).get("type")
    provider = remote.get("provider")
    where = f"{gpu} @ {provider}" if gpu and provider else gpu
    pairs = [
        ("run", _paint(obj.get("run_id", ""), _ACCENT2)),
        ("model", spec.get("model")),
        ("algorithm", (spec.get("algorithm") or "").upper() or None),
        ("gpu", where),
        ("cost", money(obj.get("cost_usd", 0.0))),
    ]
    realized = obj.get("realized_cost_usd")
    if realized is not None:
        pairs.append(("realized", money(realized)))
    pairs += [
        ("created", _humanize_ts(obj.get("created_at"))),
        ("updated", _humanize_ts(obj.get("updated_at"))),
    ]
    if obj.get("artifacts_dir"):
        pairs.append(("artifacts", obj["artifacts_dir"]))
    if obj.get("error"):
        pairs.append(("error", _paint(str(obj["error"]), _RED)))
    head = f"{header('status')}\n  {badge(obj.get('state', 'unknown'))}\n\n{_kv(pairs)}"
    raw = f"{_dim('details')}\n{_json(obj)}"
    return f"{head}\n\n{raw}"


def object_panel(cmd: str, obj: dict, desc: str | None = None) -> str:
    """Header (+ state badge when present) over syntax-highlighted JSON. Lossless."""
    parts = [header(cmd, desc)]
    if isinstance(obj, dict) and obj.get("state"):
        rid = obj.get("run_id")
        line = "  " + badge(obj["state"])
        if rid:
            line += "   " + _paint(rid, _ACCENT2)
        parts.append(line + "\n")
    parts.append(_json(obj))
    return "\n".join(parts)


def cost_panel(est) -> str:
    """Pre-flight cost estimate (the themed twin of CostEstimate.breakdown())."""
    setup_extra = " + vLLM init" if est.method == "grpo" else ""
    pairs = [
        (
            "run",
            f"{_paint(est.model_id, _ACCENT2)}  {_dim(f'[{est.method.upper()}, {est.steps} steps]')}",
        ),
        (
            "gpu",
            f"{est.gpu} on {est.provider}  "
            f"{_dim(f'({est.gpu_vram_gb} GB; needs >= {est.required_vram_gb} GB)')}  "
            f"@ {money(est.gpu_hourly_usd, 2)}/hr",
        ),
        (
            "setup",
            f"{est.setup_seconds / 60:.1f} min  {_dim(f'(cold start: boot + deps + model load{setup_extra})')}",
        ),
        ("per step", f"{est.seconds_per_step:.2f} s"),
        (
            "train",
            f"{est.train_seconds / 60:.1f} min"
            + (_paint("  [capped at wall-clock limit]", _RED) if est.wall_capped else ""),
        ),
        ("wall clock", f"{est.wall_clock_hours:.2f} h"),
    ]
    panel = _kv(pairs)
    total = f"  {_paint('TOTAL'.ljust(10), _GRAY, '1')} {_paint(_glyph('·', '-'), _FAINT)} {_paint(f'${est.total_usd:.2f}', _TEAL, '1')}"
    out = f"{header('train', 'pre-flight cost estimate')}\n{panel}\n{_rule()}\n{total}"
    if est.notes:
        notes = "\n".join(f"  {_paint(_glyph('·', '-'), _FAINT)} {_dim(n)}" for n in est.notes)
        out += f"\n\n{_dim('notes')}\n{notes}"
    return out


def env_setup(paths: list[str]) -> str:
    """Confirmation + file tree for `flash env setup`."""
    labels = {
        "environment.py": "env entrypoint — edit the reward + prompt",
        "datasets/train.jsonl": "starter training rows",
        "configs/rl.toml": "GRPO run config",
        "configs/sft.toml": "SFT run config",
    }
    keyw = max(len(p) for p in paths)
    tree = "\n".join(
        f"  {_paint(p.ljust(keyw), _ACCENT2)}  {_dim(labels.get(p, ''))}" for p in paths
    )
    head = f"{header('env setup', 'starter Freesolo environment')}\n{ok('scaffold ready')}\n"
    nxt = arrow("publish it: flash env push --name my-env .")
    return f"{head}\n{tree}\n\n{nxt}"


def env_list(installed: list[str], local: list[str]) -> str:
    parts = [header("env list", "installed + local environments")]
    if installed:
        parts.append(_paint("installed", _GRAY, "1"))
        parts.extend(
            f"  {_paint(_glyph('·', '-'), _FAINT)} {_paint(e, _ACCENT2)}" for e in installed
        )
    if local:
        if installed:
            parts.append("")
        parts.append(
            _paint("local sources", _GRAY, "1")
            + _dim("  (publish with flash env push --name <name> <path>)")
        )
        parts.extend(f"  {_paint(_glyph('·', '-'), _FAINT)} {_paint(p, _ACCENT2)}" for p in local)
    if not installed and not local:
        parts.append(_dim("  no environments yet — scaffold one with `flash env setup`"))
    return "\n".join(parts)


def env_installed(env_id: str, manifest: str) -> str:
    snippet = f'[environment]\nid = "{env_id}"'
    body = "\n".join(f"  {_paint(line, _ACCENT2)}" for line in snippet.splitlines())
    return (
        f"{ok(f'recorded {_bold(env_id)}')}\n"
        f"{_dim(f'  manifest: {manifest}')}\n\n"
        f"{_dim('use it in your config:')}\n{body}"
    )


def chat_label() -> str:
    """A faint speaker label printed above a styled chat reply (the reply itself stays plain
    text so a piped transcript is unchanged)."""
    return _paint("assistant", _ACCENT2, "1")


def env_published(slug: str) -> str:
    snippet = f'[environment]\nid = "{slug}"'
    body = "\n".join(f"  {_paint(line, _ACCENT2)}" for line in snippet.splitlines())
    return f"{ok(f'published {_bold(slug)}')}\n\n{_dim('reference it in your config:')}\n{body}"
