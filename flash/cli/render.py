"""Human-facing rendering for `flash login` / `flash whoami` (stdlib only).

The identity behind a stored key is shown as a small aligned card instead of raw
``json.dumps`` output. ANSI styling and unicode glyphs are applied only when stdout is
an interactive terminal that can encode them; piped, captured, or ASCII-locale output
stays plain so it can be grepped or parsed and never raises ``UnicodeEncodeError``.
"""

from __future__ import annotations

import os
import sys

from flash._channel import CLI_NAME

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


def error(msg: str) -> str:
    """Themed twin of the plain ``error: {msg}`` line — the red ✗ idiom of ``login_failed``, used
    by main()'s catch-all so every command's failure is styled on a TTY, not just ``flash login``.
    The machine path keeps the plain ``error: {msg}`` prefix that scripts and tests match on."""
    mark = _paint(_glyph("✗", "x:"), _RED, "1")
    return _safe(f"{mark} {_paint('error:', _RED, '1')} {msg}")


def warn(msg: str) -> str:
    """A themed warning (amber ⚠) — distinct from the red error idiom: the command still
    proceeds or succeeded. Machine path keeps the plain ``warning: {msg}`` text."""
    mark = _paint(_glyph("⚠", "!"), _AMBER, "1")
    return _safe(f"{mark} {_paint('warning:', _AMBER, '1')} {msg}")


def note(msg: str) -> str:
    """A quiet, dimmed line for transient progress / info (e.g. ``exporting ...``). The
    machine path prints the same text undimmed."""
    return _safe(_dim(msg))


# These renderers are only ever called when `styled()` is true (an interactive stdout, or
# FLASH_STYLE=1). Each command keeps its exact plain/JSON output on the machine path, so
# `jq`, scripts, and the agent contract are untouched; this is purely the human view.
# 256-color SGR keeps the palette readable on essentially every modern terminal.

# The Freesolo brand palette, pulled from the website (frontend/app/globals.css): navy
# `--special` #1b1b4b, periwinkle `--periwinkle` #5f72ff (the accent/ring), green `--green`
# #57ff8f, and the deep teal `--green-deep` #00695c. The site ships a light and a dark theme,
# so the CLI mirrors that: each semantic color carries a (hex, xterm-256 fallback) for each
# mode. On a dark terminal periwinkle accents and the bright green read well; on a light
# terminal those wash out, so — exactly as the website does — green falls back to the deep
# teal and the accents deepen. Truecolor terminals (and the docs gallery) get the exact hex;
# everything else the nearest 256-color approximation.
_PALETTE: dict[str, dict[str, tuple[str, int]]] = {
    # role            dark (on navy)        light (on white)
    "accent": {"dark": ("5f72ff", 63), "light": ("5f72ff", 63)},  # periwinkle — brand accent
    "accent2": {"dark": ("97a3ff", 111), "light": ("3a46c8", 62)},  # keys, ids, links, snippets
    "green": {"dark": ("57ff8f", 84), "light": ("00695c", 23)},  # success (deep teal on light)
    "teal": {"dark": ("24c2a8", 43), "light": ("0e7490", 30)},  # amounts / numbers
    "red": {"dark": ("ff6b6b", 203), "light": ("cc3b3b", 160)},  # destructive
    "violet": {"dark": ("9a8cff", 105), "light": ("6d28d9", 92)},  # JSON literals
    "gray": {"dark": ("8a93a8", 245), "light": ("5b6472", 242)},  # neutral
    "faint": {"dark": ("4d5470", 240), "light": ("9aa1b5", 248)},  # rules, punctuation
    "amber": {"dark": ("ffb454", 215), "light": ("b45309", 130)},  # warnings (non-fatal)
}
# semantic handles used throughout the renderers (resolved per mode at paint time)
_ACCENT, _ACCENT2, _GREEN, _TEAL, _RED, _VIOLET, _GRAY, _FAINT, _AMBER = (
    "accent",
    "accent2",
    "green",
    "teal",
    "red",
    "violet",
    "gray",
    "faint",
    "amber",
)


def _truecolor() -> bool:
    """Whether the terminal advertises 24-bit color (so we can emit exact brand hex)."""
    return os.environ.get("COLORTERM", "").lower() in {"truecolor", "24bit"}


def _theme() -> str:
    """Active theme: ``light`` or ``dark``. ``FLASH_THEME`` wins; otherwise a light terminal
    background (via the de-facto ``COLORFGBG`` env) selects light. Defaults to ``dark`` (the
    safe default for the colors, and unchanged behavior when nothing is set)."""
    forced = os.environ.get("FLASH_THEME", "").strip().lower()
    if forced in {"light", "dark"}:
        return forced
    fgbg = os.environ.get("COLORFGBG", "")
    if fgbg:
        try:
            bg = int(fgbg.split(";")[-1])
        except ValueError:
            return "dark"
        # ANSI bg 7 (light grey) and 11-15 (bright/white) mean a light terminal background
        return "light" if bg == 7 or bg >= 11 else "dark"
    return "dark"


def _sgr(part: str) -> str:
    """Resolve one style token to an SGR parameter: a brand color name resolves to a truecolor
    or 256-color foreground for the active theme; any other code (e.g. ``"1"``) passes through."""
    color = _PALETTE.get(part)
    if color is None:
        return part
    hex6, fallback = color[_theme()]
    if _truecolor():
        return f"38;2;{int(hex6[0:2], 16)};{int(hex6[2:4], 16)};{int(hex6[4:6], 16)}"
    return f"38;5;{fallback}"


def _paint(text: str, *codes: str) -> str:
    """Apply one or more style tokens (raw SGR codes and/or brand color names), honoring the gate."""
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
    "cancelling": (_GRAY, "◐", "o"),
    "deploying": (_TEAL, "◐", "o"),
    "torn_down": (_GRAY, "○", "o"),
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
    mark = _paint(CLI_NAME, _ACCENT, "1")
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


# Per-command renderers (themed view only; the command supplies the data).


def version(value: str) -> str:
    """The wordmark + version."""
    mark = _paint(CLI_NAME, _ACCENT, "1")
    return _safe(f"{mark} {_dim('v' + value)}")


def submitted(run_id: str) -> str:
    """The `flash train` hand-off note (printed to stderr before logs start streaming)."""
    head = ok(f"run {_paint(run_id, _ACCENT2)} submitted")
    hint = _dim(f"following logs — Ctrl-C detaches; resume with `flash status {run_id} --follow`")
    return _safe(f"{head}\n{hint}")


def models_table(rows: list[dict]) -> str:
    """Supported base models — a clean themed list of ids (the CLI lists ids only)."""
    dot = _glyph("•", "-")
    ids = "\n".join(f"  {_paint(dot, _FAINT)} {_paint(r['id'], _ACCENT2)}" for r in rows)
    foot = arrow("train one with: flash train configs/rl.toml")
    return _safe(f"{header('models', 'supported base models')}\n{ids}\n\n{foot}")


def gpus_table(rows: list[tuple[str, int, float | None]], tip: str) -> str:
    """GPU classes: (name, vram_gb, $/hr or None)."""
    body = []
    for name, vram, rate in rows:
        rate_cell = (f"${rate:.2f}", _TEAL) if rate else ("-", _FAINT)
        body.append([(name, _ACCENT2), (f"{vram} GB", _GRAY), rate_cell])
    table = _table(["GPU", "VRAM", "$/HR"], body, aligns=["l", "r", "r"])
    return _safe(f"{header('gpus', 'managed GPU classes')}\n{table}\n\n{_dim(tip)}")


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
    return _safe(f"{header('runs', f'{len(runs)} run(s)')}\n{table}")


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
    return _safe(f"{header('deployments', f'{len(rows)} active')}\n{table}")


def checkpoints_table(run_id: str, rows: list[dict]) -> str:
    """Deployable per-step RL checkpoints: the step number + the stable repo path it lives at."""
    body = [
        [
            (str(c.get("step", "")), _TEAL),
            (f"{c.get('repo_id', '')}:{c.get('subfolder', '')}", _ACCENT2),
        ]
        for c in sorted(rows, key=lambda c: c.get("step", 0))
    ]
    table = _table(["STEP", "CHECKPOINT"], body, aligns=["r", "l"])
    foot = arrow(f"deploy one with: flash deploy {run_id} --step <STEP>")
    return _safe(f"{header('checkpoints', f'{len(rows)} deployable')}\n{table}\n\n{foot}")


def empty(cmd: str, desc: str, message: str) -> str:
    """A styled empty state (e.g. no runs yet)."""
    return _safe(f"{header(cmd, desc)}\n{_dim('  ' + message)}")


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
        ("run id", _paint(obj.get("run_id", ""), _ACCENT2)),
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
    return _safe(f"{head}\n\n{raw}")


def object_panel(cmd: str, obj: dict, desc: str | None = None) -> str:
    """Header (+ state badge when present) over syntax-highlighted JSON. Lossless.

    Used for `flash train --dry-run` / `--background`, where the full validated spec is the
    point. The run-lifecycle mutations (cancel/deploy/undeploy/export) instead use the curated
    confirmation cards below — their machine path still emits the full JSON for scripts."""
    parts = [header(cmd, desc)]
    if isinstance(obj, dict) and obj.get("state"):
        rid = obj.get("run_id")
        line = "  " + badge(obj["state"])
        if rid:
            line += "   " + _paint(rid, _ACCENT2)
        parts.append(line + "\n")
    parts.append(_json(obj))
    return _safe("\n".join(parts))


# Run-lifecycle mutation confirmations — the curated `✓ ... + detail card` idiom (same as
# login_ok / env_published), not a raw JSON dump. The machine path keeps json.dumps untouched.


def cancelled(payload: dict) -> str:
    """`flash cancel`: a green confirmation only when the run actually flips to `cancelled`. A
    no-op cancel of an already-terminal run (done/failed/dry_run) shows a neutral "already ..."
    line instead, so a finished or failed run is never dressed up as a fresh cancellation."""
    state = payload.get("state", "cancelled")
    rid = _paint(payload.get("run_id", ""), _ACCENT2)
    if state == "cancelled":
        head = ok(f"cancel requested for {rid}")
    else:
        head = note(f"{payload.get('run_id', '')} already {state} — nothing to cancel")
    return _safe(f"{head}\n  {badge(state)}")


def deployed(dep: dict) -> str:
    """`flash deploy`: the endpoint, gpu, and serving url as an aligned card (not a JSON dump)."""
    pairs = [
        ("run", _paint(dep.get("run_id", ""), _ACCENT2)),
        ("endpoint", _paint(dep["endpoint_name"], _GREEN) if dep.get("endpoint_name") else None),
        ("gpu", dep.get("gpu")),
        ("url", _paint(dep["url"], _ACCENT2) if dep.get("url") else None),
    ]
    state = dep.get("state", "deployed")
    if state == "dry_run":
        # a dry run validates and shapes the deployment without creating one, so don't dress it
        # up as a successful deploy — a neutral validation line instead of the green ✓.
        head = note(
            f"validated {_paint(dep.get('run_id', ''), _ACCENT2)} (dry run — nothing deployed)"
        )
    elif state == "deployed":
        head = ok("deployed")
    else:
        head = f"{ok('deploy')}  {badge(state)}"
    return _safe(f"{head}\n{_kv(pairs)}")


def undeployed(result: dict) -> str:
    """`flash undeploy`: confirm the run's serving deployment was torn down. The server clears the
    deployment record idempotently (an already-absent serving adapter that 404s still counts as a
    teardown), and the response can't distinguish that from a true no-op, so we always confirm;
    when the serving backend actually deregistered endpoints we name them."""
    rid = _paint(result.get("run_id", ""), _ACCENT2)
    deleted = result.get("deleted_endpoints") or []
    line = ok(f"torn down {rid}")
    if deleted:
        line += "\n" + _dim(f"  deregistered {', '.join(deleted)}")
    return _safe(line)


def exported(result: dict) -> str:
    """`flash export`: where the adapter landed on HuggingFace, as an aligned card."""
    pairs = [
        ("adapter", _paint(result.get("adapter_id", ""), _ACCENT2)),
        ("repo", _paint(result.get("repository", ""), _ACCENT2)),
        ("url", _paint(result["url"], _ACCENT2) if result.get("url") else None),
        ("visibility", "private" if result.get("private") else "public"),
    ]
    return _safe(f"{ok('exported to HuggingFace')}\n{_kv(pairs)}")


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
    return _safe(out)


def env_setup(paths: list[str]) -> str:
    """Confirmation + file tree for `flash env setup`."""
    labels = {
        "environment.py": "env entrypoint — edit the reward + prompt",
        "datasets/train.jsonl": "starter training rows",
        "configs/rl.toml": "GRPO run config",
        "configs/sft.toml": "SFT run config",
        "TRAINING.md": "how to train well — read this first",
    }
    keyw = max(len(p) for p in paths)
    tree = "\n".join(
        f"  {_paint(p.ljust(keyw), _ACCENT2)}  {_dim(labels.get(p, ''))}" for p in paths
    )
    head = f"{header('env setup', 'starter Freesolo environment')}\n{ok('scaffold ready')}\n"
    nxt = arrow("publish it: flash env push --name my-env .")
    return _safe(f"{head}\n{tree}\n\n{nxt}")


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
    return _safe("\n".join(parts))


def env_installed(env_id: str, manifest: str) -> str:
    snippet = f'[environment]\nid = "{env_id}"'
    body = "\n".join(f"  {_paint(line, _ACCENT2)}" for line in snippet.splitlines())
    return _safe(
        f"{ok(f'recorded {_bold(env_id)}')}\n"
        f"{_dim(f'  manifest: {manifest}')}\n\n"
        f"{_dim('use it in your config:')}\n{body}"
    )


def chat_label() -> str:
    """A faint speaker label printed above a styled chat reply (the reply itself stays plain
    text so a piped transcript is unchanged)."""
    return _paint("assistant", _ACCENT2, "1")


def log_section(name: str) -> str:
    """A themed divider above a passthrough worker-log section in `flash status --logs` — the same
    idiom as chat_label sitting above a raw chat reply (the log body stays raw). The machine path
    keeps the plain ``----- name -----`` divider that scripts and tests match on."""
    rule = _paint(_glyph("─", "-") * 3, _FAINT)
    return _safe(f"{rule} {_paint(name, _ACCENT2, '1')} {rule}")


def env_published(slug: str) -> str:
    snippet = f'[environment]\nid = "{slug}"'
    body = "\n".join(f"  {_paint(line, _ACCENT2)}" for line in snippet.splitlines())
    return _safe(
        f"{ok(f'published {_bold(slug)}')}\n\n{_dim('reference it in your config:')}\n{body}"
    )


def help_page(
    tagline: str,
    usage: str,
    groups: list[tuple[str, list[tuple[str, str]]]],
    options: list[tuple[str, str]],
    footers: list[str],
) -> str:
    """The themed ``flash --help`` page — the styled twin of argparse's flat default.

    Mirrors the rest of the CLI: brand banner + faint rule (like ``header``), dim-bold
    section titles (like ``env_list``), accent command names with dimmed summaries (like
    ``env_setup``), and ``arrow`` next-step hints. Commands arrive pre-grouped as
    ``(title, [(command, summary)])`` so the workflow ordering lives with the parser, not
    here. Only ever called on the styled path; piped/scripted ``--help`` keeps argparse's
    plain text (see ``flash.cli._FlashParser``), so existing greps stay byte-for-byte.
    """
    mark = _paint(CLI_NAME, _ACCENT, "1")
    banner = f"{mark}  {_paint(tagline, _GRAY)}"
    usage_line = f"{_dim('usage:')} {_paint(usage, _GRAY)}"
    # one name-column width across every group AND the options block, so every summary lines up
    # down the whole page (same single-shared-width discipline as _table).
    names = [name for _, rows in groups for name, _ in rows] + [flag for flag, _ in options]
    width = max((len(n) for n in names), default=0)

    def section(title: str, rows: list[tuple[str, str]]) -> str:
        head = _paint(title, _GRAY, "1")
        body = "\n".join(
            f"  {_paint(name.ljust(width), _ACCENT2)}  {_dim(summary)}" for name, summary in rows
        )
        return f"{head}\n{body}"

    blocks = [section(title, rows) for title, rows in groups]
    blocks.append(section("options", options))

    body = "\n\n".join(blocks)
    foot = "\n".join(arrow(line) for line in footers)
    # trailing newline so the styled page matches argparse's newline-terminated help (argparse
    # writes format_help() verbatim via print_help, with no print() to add one).
    return _safe(f"{banner}\n{_rule()}\n{usage_line}\n\n{body}\n\n{foot}\n")
