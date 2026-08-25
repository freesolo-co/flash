# Contributing to Flash

Thanks for taking the time to contribute.

Before you start, read [README.md](README.md) — in particular the section on what this
repository does and does not include. Flash is the client and control plane for
Freesolo's hosted post-training service; some components it talks to are not part of this
repository. That shapes what can be built and tested locally.

## What you can run locally

The test suite is CPU-only, offline by default, and needs no GPU, no network, and no
credentials:

```bash
uv sync --extra server --dev
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

CI (`.github/workflows/ci.yml`) additionally enforces two size gates and runs the
timing-sensitive tests in their own serial pass:

```bash
uv run python scripts/check_file_size.py      # no module in flash/ over 1000 lines
uv run python scripts/check_function_size.py  # no function in flash/ over 150 lines
uv run pytest -q -m wallclock                 # asserts on real elapsed time; runs alone
```

The size gates fail on the _whole_ file or function, so the fix is always to extract a
cohesive piece into a sibling, never to reflow lines.

If you change the verl pin (`Dockerfile.worker` or
`flash/engine/worker/backend_common.py`), also run the one check that needs the network:

```bash
uv run python scripts/check_verl_pin_exists.py   # the pinned commit is reachable on the fork
```

The offline tests only compare the pin against a hardcoded copy of itself, so a
mis-transcribed sha passes every one of them while naming a commit that was never pushed —
which then fails every worker install at provisioning time.

A green local run usually means a green CI run. CI runs them on both supported interpreters (3.11 and 3.12);
locally, whichever one `uv` picks is normally enough.

CI also runs `uv run mypy flash`, but that job is advisory: it reports the existing type
errors without failing the build, so you do not need a clean mypy run to contribute.

To run the CLI from a dev checkout, use `uv run python -m flash.cli --help`. The
`flash-cli` script remains a stable alias for operator automation.

Formatting is enforced: CI runs `ruff format --check .` and fails on any drift, so run
`uv run ruff format .` before you push. The whole tree is already formatted, so this
touches only the files you changed.

Running the control plane (`flash-server`) additionally requires operator credentials for
GPU providers and the Freesolo backend — see `.env.example` and the self-hosting notes in
the README. You do not need any of that to change code and run the tests.

## Branching and pull requests

The repository uses two long-lived branches:

- `dev` — integration branch. **Open your pull request against `dev`.**
- `main` — release branch. PRs into `main` must come from `dev`
  (enforced by `.github/workflows/main-source-guard.yml`), and merging a version bump
  into `main` publishes to PyPI.

So the flow is: your branch → `dev` → `main`.

Before opening a pull request:

1. Rebase onto the current `dev`.
2. Make sure the checks above pass — `ruff check`, `ruff format --check`, `pytest`, and both
   size gates.
3. Add tests for the behavior you changed. The suite is large and treated as the
   specification for existing behavior; a change in behavior should show up as a change in
   tests.
4. Keep the change scoped to one thing. Unrelated cleanup in the same PR makes review
   harder and is usually asked to be split out.

### Version bumps

If your change touches `flash/`, `pyproject.toml`, or `uv.lock`, the release workflow
requires the version to increase before it can land on `main`. Bump **both**
`[project].version` and `[tool.flash-dev].version` in `pyproject.toml` — CI enforces that
they match (`.github/workflows/version-parity.yml`). Documentation-only changes do not
need a bump.

## Commit messages

Use conventional-commit prefixes, lowercase:

```
feat: add multi-turn rollout support
fix(server): reject unknown project ids before allocation
docs: document the self-hosting requirements
```

Prefixes in use: `feat`, `fix`, `docs`, `refactor`, `chore`, `test`, `style`, `perf`,
`build`, `ci`, `revert`.

Write a body explaining _why_ the change is needed, not just what it does. Most commits in
this repository have one, and it is the main reason the history is readable.

Do not add tool-generated trailers (co-author lines for AI assistants, session links, or
"generated with" footers) to commits.

That rule governs new commits only. Commits already in this history carry such trailers,
plus a build box that committed under an internal hostname. `scripts/scrub_history.sh`
rewrites both out of every ref and refuses to print publication instructions unless the
residual counts reach zero. It has to run on a fresh mirror clone **before** the
repository is made public: rewriting afterwards accomplishes nothing, because the old
shas stay reachable through forks, caches, and the GitHub API.

## Code style

- Ruff is the formatter and linter; its configuration lives in `pyproject.toml`. Run
  `uv run ruff format <the files you changed>` rather than hand-formatting.
- Line length is 100.
- Comments are lowercase.
- Match the conventions of the file you are editing.

## Reporting bugs and requesting features

Use the issue templates. For a bug, the single most useful thing you can include is the
training TOML (with credentials removed) and the exact command you ran. For anything
security-related, follow [SECURITY.md](SECURITY.md) instead of opening a public issue.

## Licensing

Flash is licensed under the Apache License 2.0. By contributing, you agree that your
contributions are licensed under the same terms (Apache-2.0 §5).
