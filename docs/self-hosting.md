# Self-hosting the AutoSLM control plane (operator guide)

End users only ever run `slm login` and `slm train` against a control plane. This guide
is for the **operator** who hosts that control plane: it holds the provider credentials,
mints user keys, supervises GPU runs, and proxies inference.

```
user ──slm login / slm train──> control plane (this guide) ──> RunPod Flash GPUs
        sk-autoslm-... key        RUNPOD_API_KEY + HF creds        + HF artifact repo
```

## Requirements

- Python 3.11/3.12 and an install of this package from the monorepo source —
  `pip install './autoslm[server]'` (adds FastAPI + uvicorn) — or the Docker image
  below. It is internal and not published to PyPI.
- A [RunPod](https://runpod.io) account API key (this account pays for all GPU time).
- A Hugging Face token with write access to a **dataset** repo for run artifacts.

## Configuration (environment variables)

| variable | required | meaning |
|---|---|---|
| `RUNPOD_API_KEY` | yes | RunPod key used to provision Flash GPUs |
| `HUGGINGFACE_TOKEN` | yes | HF token with write access to `HF_REPO` |
| `HF_REPO` | yes | HF *dataset* repo for adapters/checkpoints/heartbeats, e.g. `your-org/autoslm-runs` |
| `AUTOSLM_VRAM_HEADROOM` | no | open-model VRAM sizing headroom for smart allocation (default `1.15`) |

The server fails fast at startup if a required variable is missing. Provider credentials
are **env-only by design** — they are never read from `~/.autoslm/config.json` (that
file holds the client-side AutoSLM key) and never sent to clients.

## Run it

```bash
pip install './autoslm[server]'   # internal package — install from the monorepo source, not an index
export RUNPOD_API_KEY=... HUGGINGFACE_TOKEN=hf_... HF_REPO=your-org/autoslm-runs
slm server --host 0.0.0.0 --port 8080        # or: python -m autoslm.server
```

Docker:

```bash
docker build -t autoslm-server .
docker run -p 8080:8080 \
  -e RUNPOD_API_KEY=... -e HUGGINGFACE_TOKEN=... -e HF_REPO=your-org/autoslm-runs \
  -v autoslm-state:/root/.autoslm autoslm-server
```

Point clients at it:

```bash
AUTOSLM_API_URL=https://your-host:8080 slm login
# or persist it: slm login --api-url https://your-host:8080
```

## Operational model (read this before going public)

- **One instance only.** All state lives under the fixed `~/.autoslm/` directory: SQLite
  keys/ownership (`server.db`), run status/logs (`runs/`), and metrics (`results/`). Mount
  one volume at `~/.autoslm` (`/root/.autoslm` in the container) to persist it; back it up.
  Run exactly one uvicorn worker per state volume; do not scale horizontally or `--reload`.
  *Upgrading from an older build?* On first start the server best-effort migrates state from
  the pre-consolidation locations (cwd-relative `.autoslm/runs` + `results/`, or the old
  `/state` Docker mount) into `~/.autoslm`. Docker users who mounted the volume at `/state`
  should just remount that same volume at `/root/.autoslm` (its contents land in place).
- **Run behind a reverse proxy.** The per-IP `POST /v1/keys` claim throttle keys on the
  last `X-Forwarded-For` hop (the control plane assumes a trusted proxy / load balancer
  that strips client-supplied XFF). Don't expose it directly — a direct deployment would
  let a client spoof `X-Forwarded-For` to bypass the throttle.
- **Open key claiming = open GPU spend.** There is no billing yet: anyone who can reach
  `POST /v1/keys` can claim a key and start runs **on your RunPod account**. Guards in
  place: a per-IP claim throttle (5/hour) and the per-run `gpu.max_wall_seconds` execution
  cap. For anything beyond a trusted group, put the server behind a
  network boundary (VPN/allow-list) or a reverse proxy with its own auth.
- **TLS** is out of scope for the server itself — terminate it at a reverse proxy
  (Caddy/nginx/Traefik).
- **Restart recovery.** On startup the server re-attaches to every non-terminal training
  run via its persisted RunPod job handle; GPU runs are unaffected by control-plane
  restarts.
- **Tenant isolation.** Keys are stored hashed (sha256); every run is owned by the key
  that created it, and other keys get 404s. Note that all tenants' artifacts live in the
  operator's single `HF_REPO`, namespaced by run id — per-tenant repos are on the
  roadmap, so don't make that repo public.

## API surface (for building your own clients)

All endpoints are under `/v1`; authenticated ones take `Authorization: Bearer
sk-autoslm-...`.

| method & path | auth | purpose |
|---|---|---|
| `GET /v1/health` | — | liveness + version |
| `POST /v1/keys` | — (IP-throttled) | claim a key (`{"email": "..."}` optional) |
| `GET /v1/me` | key | identity behind the key |
| `GET /v1/models` | key | curated model catalog |
| `POST /v1/runs` | key | `{"spec": {...}, "dry_run": false}` → run status |
| `GET /v1/runs` | key | your runs |
| `GET /v1/runs/{id}` | owner | status (+ metrics when done) |
| `GET /v1/runs/{id}/logs?offset=N` | owner | incremental logs `{logs, offset, state}` |
| `POST /v1/runs/{id}/cancel` | owner | best-effort cancel |
| `POST /v1/runs/{id}/deploy` | owner | `{"mode": "dev"\|"always-on", "idle_timeout_s": 300}` |
| `DELETE /v1/runs/{id}/deploy` | owner | tear down serving |
| `GET /v1/deployments` | key | your active deployments |
| `POST /v1/runs/{id}/chat` | owner | OpenAI-shaped chat completion (proxied inference) |
