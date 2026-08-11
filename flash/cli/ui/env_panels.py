"""Styled panels for the `flash env` subcommands.

The primitives these build on (`_paint`, `header`, `ok`, ...) stay in `flash.cli.ui.render`; this
module holds the per-command env layouts. Split out to keep `render.py` under the file-size limit,
mirroring `flash.cli.ui.tables`.

Imported back into `flash.cli.ui.render` so `render.env_list(...)` keeps resolving, which is how
every call site and `monkeypatch.setattr(commands.render, ...)` reach these.
"""

from __future__ import annotations

from flash.cli.ui.render import (
    _ACCENT2,
    _FAINT,
    _GRAY,
    _bold,
    _dim,
    _glyph,
    _paint,
    _safe,
    arrow,
    header,
    ok,
)


def env_setup(paths: list[str], project_id: str) -> str:
    """Confirmation + file tree for `flash env setup`."""
    labels = {
        "environment.py": "env entrypoint — edit the reward + prompt",
        "dataset/train.jsonl": "starter training rows",
        "configs/sft.toml": "SFT run config",
        "configs/rl.toml": "GRPO run config",
        "configs/opd.toml": "OPD (distillation) run config",
        "TRAINING.md": "how to train well — read this first",
    }
    keyw = max(len(p) for p in paths)
    tree = "\n".join(
        f"  {_paint(p.ljust(keyw), _ACCENT2)}  {_dim(labels.get(p, ''))}" for p in paths
    )
    head = f"{header('env setup', 'starter Freesolo environment')}\n{ok('scaffold ready')}\n"
    next_step = arrow(f"publish it: flash env push --project {project_id} --name my-env .")
    return _safe(f"{head}\n{tree}\n\n{next_step}")


def env_list(
    local: list[str], *, published: list[str] | None = None, unavailable: str | None = None
) -> str:
    published = published or []
    parts = [header("env list", "published and local environments")]
    if published:
        parts.append(
            _paint("published", _GRAY, "1")
            + _dim('  (reference one with [environment] id = "<id>")')
        )
        parts.extend(
            f"  {_paint(_glyph('·', '-'), _FAINT)} {_paint(env_id, _ACCENT2)}"
            for env_id in published
        )
    elif unavailable:
        # never fold this into the "no environments yet" line: an unchecked hub must not read as an
        # empty one, or a publish that worked looks like it silently did nothing.
        parts.append(_dim(f"  published environments unavailable: {unavailable}"))
    if local:
        if published or unavailable:
            parts.append("")
        parts.append(
            _paint("local sources", _GRAY, "1")
            + _dim("  (publish with flash env push --project <project-uuid> --name <name> <path>)")
        )
        parts.extend(f"  {_paint(_glyph('·', '-'), _FAINT)} {_paint(p, _ACCENT2)}" for p in local)
    if not local and not published:
        parts.append(_dim("  no environments yet — scaffold one with `flash env setup`"))
    return _safe("\n".join(parts))


def env_published(slug: str) -> str:
    snippet = f'[environment]\nid = "{slug}"'
    body = "\n".join(f"  {_paint(line, _ACCENT2)}" for line in snippet.splitlines())
    return _safe(
        f"{ok(f'published {_bold(slug)}')}\n\n{_dim('reference it in your config:')}\n{body}"
    )


def env_pulled(dest: str, detail: str = "") -> str:
    line = ok(f"pulled {_bold(dest)}")
    if detail:
        line += f"\n{_dim(f'  {detail}')}"
    return _safe(line)
