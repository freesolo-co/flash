# Contributing to Flash

Thanks for hacking on Flash. This is the short version of how to get set up,
run the checks, and land a change.

## Setup

Flash uses [`uv`](https://docs.astral.sh/uv/). From the repo root:

```bash
make setup        # uv sync --extra server --dev
```

That installs the control-plane (`server`) extra plus the dev tools (ruff,
mypy, pytest). The heavy GPU/worker deps (`torch`, `vllm`, …) live behind the
`gpu` extra and are installed on the GPU worker, not for local dev.

Run `make` (or `make help`) to see every task.

## Running the checks

```bash
make check        # ruff check + offline test suite — this is what CI runs
```

Individually:

```bash
make lint         # ruff check
make fmt          # ruff format  (use fmt-check to verify without writing)
make test         # pytest, offline
make typecheck    # mypy (advisory; not gated in CI)
```

The test suite runs **offline by default** (`FLASH_SKIP_NET=1`) — no GPU and no
provider/network calls, so it's fast and safe to run anywhere. Tests that hit
real provider APIs are marked `live` and opt-in:

```bash
make test-live    # needs provider creds; sets FLASH_LIVE=1
```

Run `make check` before pushing — it mirrors CI (`.github/workflows/ci.yml`), so
a green local run means a green CI run.

## Branches & pull requests

- **Work off `dev` and open your PR into `dev`.** That's the integration branch.
- **`main` is release-only.** It's protected, and the *Main source guard*
  workflow enforces that PRs into `main` may only come from `dev` — promote with
  a `dev -> main` PR, never a feature branch straight into `main`.
- Keep PRs focused; CI must be green before merge.

## Layout

See the **Layout** section of [`README.md`](README.md) for the package map
(`flash/catalog.py`, `flash/providers/`, `flash/engine/`, `flash/server/`, …).
