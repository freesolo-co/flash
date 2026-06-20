# Flash

Managed LoRA post-training service: SFT and GRPO on managed GPUs across multiple
providers — RunPod Flash (serverless queue; RTX 4090/5090 classes) and Vast.ai
(rented verified-datacenter instances; L40S / RTX Pro 4000 / A100 classes). The
allocator picks the cheapest GPU class that fits the run across both providers.

## Scope

- `slm train <cfg.toml>` / control-plane `POST /runs` — submit a training job;
  one dedicated GPU per run, supervised server-side (stall watchdog, bounded
  auto-retry resuming from the last streamed checkpoint, endpoint GC).
- `slm deploy` (scale-to-zero or always-on), `slm chat` —
  serving for trained adapters.
- **Verifiers-only environments.** Every run names a Prime Intellect `verifiers`
  environment by its published Hub slug (`[environment] id = "owner/name"`).
  Scaffold a local env, publish it with `slm env push`, then reference it by id.
  The worker wraps it via `flash/envs/adapter.py`. There are no
  built-in task environments and no freesolo bridge. Single-turn environments
  are fully supported (SFT/GRPO/eval).

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
- `flash/envs/` — environment machinery: registry and the
  `adapter` that wraps Prime Intellect / Hub `verifiers`
  environments onto the worker's interface
- `slm lab setup` / `slm env init` — scaffold a starter local verifiers env and a
  ready-to-run config to start from
- `flash/serve/`, `flash/server/` — adapter serving and the FastAPI control
  plane (run operator-side via the separate `flash-server` command)
- `flash/mcp/` — stdio MCP bridge for coding agents
- `Dockerfile` — the control-plane image (used by the repo docker-compose)
- `tests/` — pytest suite (CPU-only with `FLASH_SKIP_NET=1`)

## Local commands

Everything runs from the repo root and needs [`uv`](https://docs.astral.sh/uv/).
A `Makefile` wraps the common tasks — run `make` to list them:

```bash
make setup        # uv sync --extra server --dev
make test         # CPU tests, offline (FLASH_SKIP_NET=1, no GPU/network)
make lint         # ruff check
make fmt          # ruff format
make check        # lint + tests (what CI runs)

uv run slm --help        # the client CLI
uv run flash-server      # control plane (operator-side, run once)
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the dev → main branch flow.

The control plane owns provider credentials: `RUNPOD_API_KEY` is always required
(RunPod is the default substrate), `VAST_API_KEY` is opt-in (only checked when set),
plus the shared `HF_TOKEN`.
The artifact repo is per-run (the run TOML's `[train] hf_repo`), not an
operator-wide env var. Clients authenticate with their freesolo API key (`slm login`).
