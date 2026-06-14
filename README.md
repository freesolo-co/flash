# AutoSLM

Managed LoRA post-training service: SFT, GRPO, on-policy distillation, and DPO
on dedicated consumer GPUs (RunPod Flash, RTX 4090/5090 classes). The Freesolo
SDK submits training jobs here instead of running Tinker loops client-side.

## Scope

- `slm train <cfg.toml>` / control-plane `POST /runs` — submit a training job;
  one dedicated GPU per run, supervised server-side (stall watchdog, bounded
  auto-retry resuming from the last streamed checkpoint, endpoint GC).
- `slm eval`, `slm deploy` (scale-to-zero or always-on), `slm chat`,
  `slm serve-proxy` — eval and serving for trained adapters.
- The `freesolo` built-in environment (`autoslm/envs/freesolo.py`) bridges
  freesolo SDK contracts/datasets onto the worker: the SDK passes contract text
  and dataset records as environment params; the worker pip-installs
  `freesolo[full]` and reconstructs the environment for prompting and scoring.
  Single-turn environments only.

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
  shared recipe; SFT targets, RL rewards, and evals all route through the active
  environment so every arm scores identically (task-specific grading lives with
  its example, not in the engine)
- `autoslm/envs/` — environment machinery: registry, base loader, `freesolo`
  bridge, `tests_pass` built-in + verifiers/Prime Hub interop
- `examples/` — repo-root example task environments, each a self-contained folder
  (`examples/gsm8k/`, `examples/math/`: env + grader + data + ready-to-run TOMLs).
  The registry path-loads them as the `gsm8k`/`math` built-ins; they ship with the
  source checkout rather than the installed package.
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
key (`slm login`). See `docs/config-reference.md` for the run TOML schema and
`docs/self-hosting.md` for operating the service.
