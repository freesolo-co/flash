"""Styled output for ``flash env list``."""

from __future__ import annotations

from flash._internal.channel import CLI_NAME
from flash.cli.ui.render import _ACCENT2, _FAINT, _GRAY, _dim, _glyph, _paint, _safe, header


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
        parts.append(_dim(f"  published environments unavailable: {unavailable}"))
    if local:
        if published or unavailable:
            parts.append("")
        parts.append(
            _paint("local sources", _GRAY, "1")
            + _dim(
                f"  (publish with {CLI_NAME} env push --project <project-uuid> --name <name> <path>)"
            )
        )
        parts.extend(
            f"  {_paint(_glyph('·', '-'), _FAINT)} {_paint(path, _ACCENT2)}" for path in local
        )
    if not local and not published:
        parts.append(_dim(f"  no environments yet — scaffold one with `{CLI_NAME} env setup`"))
    return _safe("\n".join(parts))
