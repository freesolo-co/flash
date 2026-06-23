# Flash CLI output style

A standardized, production-grade output theme for every `flash` command.

- **Theme:** one visual language across the CLI: a brand header (`flash › <cmd>`), colored
  status badges, aligned tables, key/value panels, and syntax-highlighted JSON. Built on a
  256-color palette (electric-blue accent, cyan structure, green/amber/red/violet semantics).
- **TTY-gated, machine-safe:** the theme renders only on an interactive terminal. Piped,
  redirected, captured, or agent-driven output stays byte-for-byte plain/JSON, so `jq`,
  scripts, and the trainer-agent contract are untouched. Force it with `FLASH_STYLE=1`;
  disable the themed layout with `FLASH_STYLE=0`. `NO_COLOR` keeps the layout but drops ANSI color.
- **No new dependencies:** pure standard library, like the rest of the client CLI.

The rendering lives in `flash/cli/main/render.py`; the command wiring is in
`flash/cli/main/commands.py` and `envpush.py`.

## Preview

`index.html` is a self-contained before/after gallery of all 18 commands (open it directly,
or view `preview.png`). Regenerate after changing the theme:

```bash
uv run python docs/cli-style/generate.py        # writes index.html
```

The page drives the real command handlers against a fake control-plane client and captures
both the plain (`FLASH_STYLE=0`) and themed (`FLASH_STYLE=1`) output, converting ANSI to
inline-styled HTML.
