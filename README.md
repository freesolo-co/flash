# AutoSLM

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
  The worker wraps it via `autoslm/envs/adapter.py`. There are no
  built-in task environments and no freesolo bridge. Single-turn environments
  are fully supported (SFT/GRPO/eval).

## Layout

- `autoslm/catalog.py` — curated model catalog (Qwen3 dense supported tier;
  Qwen3.5/3.6 experimental tier) + `model_policy = "allow"` VRAM-fit check + each
  model's `thinking` capability (opt-in reasoning mode `thinking = true`; see
  [docs/config-reference.md](docs/config-reference.md#thinking-mode-thinking--true))
- `autoslm/schema.py`, `autoslm/spec.py` — TOML → `JobSpec`
- `autoslm/runner.py` — server-side run supervisor (durable job handle,
  retries, cost guard, endpoint GC)
- `autoslm/providers/` — RunPod Flash + Vast.ai provider subtrees (pricing,
  gpus, durable submit/poll, preflight) behind one `base.Provider` protocol,
  with a cross-provider `allocator.py` that picks the cheapest fitting class
- `autoslm/engine/` — the on-GPU worker (TRL + colocated vLLM rollouts) and the
  shared recipe; SFT targets and RL rewards route through the active environment
  (task-specific grading lives with its example, not in the engine)
- `autoslm/envs/` — environment machinery: registry and the
  `adapter` that wraps Prime Intellect / Hub `verifiers`
  environments onto the worker's interface
- `slm lab setup` / `slm env init` — scaffold a starter local verifiers env and a
  ready-to-run config to start from
- `autoslm/serve/`, `autoslm/server/` — adapter serving and the FastAPI control
  plane (run operator-side via the separate `autoslm-server` command)
- `autoslm/mcp/` — stdio MCP bridge for coding agents
- `Dockerfile` — the control-plane image (used by the repo docker-compose)
- `tests/` — pytest suite (CPU-only with `AUTOSLM_SKIP_NET=1`)

## Local commands

```bash
cd autoslm
uv sync --extra server
AUTOSLM_SKIP_NET=1 uv run pytest          # CPU tests (no GPU/network)
uv run ruff check . && uv run ruff format .
uv run slm --help
uv run autoslm-server                      # control plane (operator-side, run once)
```

The control plane owns provider credentials (`RUNPOD_API_KEY` and/or
`VAST_API_KEY`, plus `HUGGINGFACE_TOKEN`); the artifact repo is per-run (the run TOML's
`[train] hf_repo`), not an operator-wide env var. Clients authenticate with their
freesolo API key (`slm login`). See `docs/config-reference.md` for the run TOML schema,
`docs/algorithms.md` for choosing and tuning SFT/GRPO, and
`docs/environments.md` for authoring a task.
