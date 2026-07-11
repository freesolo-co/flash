# local Flash development harness

`flashdev` runs a control plane and CLI directly from a selected Flash checkout. It uses `uv run` and never invokes the ambiguous `flash` console script.

## setup

The only required host tool is `uv`. Every selected checkout must have its server and development dependencies installed before it is launched:

```bash
cd /absolute/path/to/flash-checkout
uv sync --extra server --dev
```

Create a private operator environment file in the checkout containing the launcher:

```bash
install -m 600 dev/local.env.example dev/local.env
```

The checked-in example contains intentionally invalid local-only placeholders. They satisfy startup preflight, including two distinct RunPod entries, but must not be used for provider operations. Put real operator credentials only in an ignored `dev/local.env` when provider integration is intentional. `flashdev` rejects non-example environment files with group or world permission bits when the host provides the standard macOS or Linux `stat` interface.

## usage

Run against the checkout containing the launcher:

```bash
./dev/flashdev serve
./dev/flashdev status
./dev/flashdev cli whoami
./dev/flashdev cli models
./dev/flashdev verify
./dev/flashdev logs
./dev/flashdev stop
```

Select another checkout explicitly when comparing branches. No explicit environment path is needed:

```bash
./dev/flashdev --checkout /absolute/path/to/flash-checkout serve
./dev/flashdev --checkout /absolute/path/to/flash-checkout cli whoami
```

The default environment resolution order is:

1. the selected checkout's ignored `dev/local.env`
2. the launcher checkout's ignored `dev/local.env`
3. the launcher checkout's checked-in `dev/local.env.example`

This allows the canonical launcher to target an older checkout that does not contain the `dev` harness files. The selected checkout must still contain the server-side `FLASH_LOCAL_CONTROL_PLANE=1` safety change in `flash/server/app.py`. `serve` fails closed before launch when that support is absent. Use a branch containing the change or cherry-pick it into older checkouts; the launcher never falls back to unsafe startup. `--env-file` overrides environment resolution. Global options must appear before the command. `FLASHDEV_CHECKOUT`, `FLASHDEV_PORT`, and `FLASHDEV_ENV_FILE` provide equivalent defaults. The port defaults to `8080`.

Startup fails immediately if the selected port already accepts connections. Stop the occupying service or select another port with `--port`.

## isolation and source selection

Each selected checkout stores runtime files under its ignored `.flashdev/` directory:

- `.flashdev/home/.flash/server.db`
- `.flashdev/home/.flash/runs/`
- `.flashdev/home/.flash/results/`
- `.flashdev/home/.flash/config.json`
- `.flashdev/server.pid`
- `.flashdev/server.log`
- `.flashdev/server.port`
- `.flashdev/lifecycle.lock/owner.pid` while `serve` or `stop` is mutating state

`serve` and `stop` use an atomic checkout-local directory lock around PID and port checks, metadata writes, launch, failed-start cleanup, and shutdown. A lock is reclaimed only when its owner file contains a numeric PID that is no longer running. Missing, invalid, or live owners fail clearly, and a launcher removes only the lock it owns. If the launcher receives HUP, INT, or TERM during startup or shutdown, it attempts to stop its tracked server child before releasing the lock.

`server.port` contains only a validated numeric port. The selected checkout remains authoritative, and every other state path plus the API URL is derived again from that checkout and port. Process ownership requires the recorded command to contain `python -m flash.server`, the checkout-derived `HOME`, and the exact `--port` value. `status`, `verify`, and `stop` reject an explicit port that conflicts with persisted metadata instead of associating the PID with another service.

The launcher starts `uv` from the selected project with the real user home, then applies the isolated `HOME` to each Python server or CLI child. This keeps uv caches and project discovery normal while preventing either child from reading or writing the shared `~/.flash/config.json`. CLI calls also receive explicit `FLASH_API_URL`, `FREESOLO_API_KEY`, and `FLASH_NO_UPDATE_CHECK=1` values. `login` is rejected even when it follows supported root flags such as `--debug`, `--verbose`, `-v`, `-vv`, or `--`.

Both `flash.server` and `flash.cli` are imported from the selected checkout. Remote worker code snapshots also come from that imported checkout-local package.

## local control-plane safety mode

The launcher forces `FLASH_LOCAL_CONTROL_PLANE=1` for the server. Startup preflight still validates the operator environment because it is network-free. The server does not automatically recover runs, retry charges, reconcile endpoint slots or costs, reap idle RunPod endpoints, or sweep Lambda orphans while this mode is active. Explicit HTTP routes and explicitly submitted runs remain available.

This suppression is a safety boundary for local startup, not a provider sandbox. An explicit real training submission can still allocate provider resources. If the local plane restarts or crashes, it will not recover the run or reap its provider resources. Monitor explicit real runs and clean up provider resources manually.

## verification boundary

`verify` is deliberately local, sub-minute, and no-spend. It checks `/v1/health`, then runs the real checkout-local `python -m flash.cli whoami` with the local bearer. It does not submit training or call provider APIs.
