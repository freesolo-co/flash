# AutoSLM

Managed LoRA post-training service: SFT and GRPO on dedicated consumer GPUs
(RunPod Flash, RTX 4090/5090 classes). The Freesolo SDK submits training jobs here
instead of running Tinker loops client-side.

## Scope

- `slm train <cfg.toml>` / control-plane `POST /runs` — submit a training job;
  one dedicated GPU per run, supervised server-side (stall watchdog, bounded
  auto-retry resuming from the last streamed checkpoint, endpoint GC).
- `slm deploy` (scale-to-zero or always-on), `slm chat`, `slm serve-proxy` —
  serving for trained adapters.
- **Verifiers-only environments.** Every run names a Prime Intellect `verifiers`
  environment — a published Hub slug (`[environment] id = "owner/name"`) or, for
  local runs, a local verifiers env module (`[environment] path`). The worker
  wraps it via `autoslm/envs/verifiers_adapter.py`. There are no built-in task
  environments and no freesolo bridge. Single-turn environments are fully
  supported (SFT/GRPO/eval).

## Layout

- `autoslm/catalog.py` — curated model catalog (Qwen3 dense supported tier;
  Qwen3.5/3.6 experimental tier) + `model_policy = "allow"` VRAM-fit check + each
  model's `thinking` capability (opt-in reasoning mode `thinking = true`; see
  [docs/config-reference.md](docs/config-reference.md#thinking-mode-thinking--true))
- `autoslm/config_schema.py`, `autoslm/worker_spec.py` — TOML → `JobSpec`
- `autoslm/orchestrator.py` — server-side run supervisor (durable job handle,
  retries, cost guard, endpoint GC)
- `autoslm/flash/` — RunPod Flash provisioning, durable submit/poll, pricing
- `autoslm/engine/` — the on-GPU worker (TRL + colocated vLLM rollouts) and the
  shared recipe; SFT targets and RL rewards route through the active environment
  (task-specific grading lives with its example, not in the engine)
- `autoslm/envs/` — environment machinery: registry, local-path loader
  (`base.py`), and the `verifiers_adapter` that wraps Prime Intellect / Hub
  `verifiers` environments onto the worker's interface
- `slm lab setup` / `slm env init` — scaffold a starter local verifiers env and a
  ready-to-run config to start from
- `autoslm/serve/`, `autoslm/server/` — adapter serving and the FastAPI control
  plane (`slm server`, see `docs/self-hosting.md`)
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
uv run slm server                          # control plane (operator-side)
```

The control plane owns provider credentials (`RUNPOD_API_KEY`,
`HUGGINGFACE_TOKEN`, `HF_REPO`); clients authenticate with a claimed AutoSLM
key (`slm login`). See `docs/config-reference.md` for the run TOML schema,
`docs/algorithms.md` for choosing and tuning SFT/GRPO,
`docs/environments.md` for authoring a task, and `docs/self-hosting.md` for
operating the service.
