# Flash

Managed LoRA post-training service: SFT and GRPO on managed GPUs across multiple
providers — RunPod Flash (serverless queue; RTX 4090/5090 classes) and Vast.ai
(rented verified-datacenter instances; L40S / RTX Pro 4000 / A100 classes). The
allocator picks the cheapest GPU class that fits the run across both providers.

## Scope

- `flash train <cfg.toml>` / control-plane `POST /runs` — submit a training job;
  one dedicated GPU per run, supervised server-side (stall watchdog, bounded
  auto-retry resuming from the last streamed checkpoint, endpoint GC).
- `flash deploy` (scale-to-zero or always-on), `flash chat` —
  serving for trained adapters.
- **Freesolo SDK environments.** Every run names a Freesolo environment id.
  Scaffold a local env, upload it with `flash env push --name <name>`, then reference the
  returned id. The worker loads it through `freesolo.environments`. There are no
  built-in task environments. Single-turn and bounded multi-turn environments are
  supported.

## Layout

- `flash/catalog.py` — curated model catalog (Qwen3 dense supported tier;
  Qwen3.5/3.6 experimental tier) + `model_policy = "allow"` VRAM-fit check + each
  model's `thinking` capability (opt-in reasoning mode `thinking = true`)
- `flash/schema.py`, `flash/spec.py` — TOML → `JobSpec`
- `flash/runner.py` — server-side run supervisor (durable job handle,
  retries, cost guard, endpoint GC)
- `flash/providers/` — RunPod Flash + Vast.ai provider subtrees (pricing,
  gpus, durable submit/poll, preflight) behind one `base.Provider` protocol,
  with a cross-provider `allocator.py` that picks the cheapest fitting class
- `flash/engine/` — the on-GPU worker (TRL + colocated vLLM rollouts) and the
  shared recipe; SFT targets and RL rewards route through the active environment
  (task-specific grading lives with its example, not in the engine)
- `flash/envs/` — environment machinery: registry and the adapter that loads
  Freesolo SDK environments onto the worker's interface
- `flash env setup` / `flash env init` — scaffold a starter local Freesolo env and a
  ready-to-run config to start from
- `flash/serve/`, `flash/server/` — adapter serving and the FastAPI control
  plane (run operator-side via the separate `flash-server` command)
- `flash/mcp/` — stdio MCP bridge for coding agents
- `Dockerfile` — the control-plane image (used by the repo docker-compose)
- `tests/` — pytest suite (CPU-only; offline-by-default, no GPU/network)

## Local commands

```bash
cd flash
uv sync --extra server
uv run pytest                           # CPU tests (offline-by-default, no GPU/network)
uv run ruff check . && uv run ruff format .
uv run flash --help
uv run flash-server                      # control plane (operator-side, run once)
```

The control plane owns provider credentials: `RUNPOD_API_KEY` is always required
(RunPod is the default substrate), `VAST_API_KEY` is opt-in (only checked when set),
plus the shared `HF_TOKEN`.
The artifact repo is per-run (the run TOML's `[train] hf_repo`), not an
operator-wide env var. Clients authenticate with their freesolo API key (`flash login`).
