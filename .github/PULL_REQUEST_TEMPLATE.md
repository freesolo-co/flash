<!--
PRs go into `dev`, not `main`. PRs into `main` must come from `dev`
(.github/workflows/main-source-guard.yml).
-->

## What this changes

<!-- One or two sentences. What behavior is different after this PR? -->

## Why

<!-- The problem this solves. Link the issue if there is one: Fixes #123 -->

## How it was verified

<!-- Commands you ran and what you observed. If it needed a GPU or a live provider,
     say so explicitly and say what was NOT verified locally. -->

- [ ] `uv run pytest`
- [ ] `uv run ruff check .`

## Checklist

- [ ] Targets the `dev` branch
- [ ] Tests cover the changed behavior
- [ ] Version bumped in `pyproject.toml` (both `[project].version` and
      `[tool.flash-dev].version`) if this touches `flash/`, `pyproject.toml`, or `uv.lock`
- [ ] No credentials, tokens, or internal hostnames in the diff
- [ ] No AI-tool trailers in the commit messages
