# Flash

Managed LoRA post-training service: SFT and GRPO on managed RunPod Flash GPUs.
The allocator picks the cheapest validated RunPod GPU class that fits the run.

## Scope

- `flash train <cfg.toml>` / control-plane `POST /runs` — submit a training job;
  one dedicated GPU per run, supervised server-side (stall watchdog, bounded
  auto-retry resuming from the last streamed checkpoint, endpoint GC).
- `flash deploy`, `flash chat` — serving for trained adapters.
- **Freesolo SDK environments.** Every run names a Freesolo environment id.
  Scaffold `environment.py` plus `datasets/train.jsonl`, upload `.` or another
  folder with `flash env push --name <name> <folder>`, then reference the
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
- `flash/providers/` — RunPod Flash provider code (pricing, gpus, durable
  submit/poll, preflight) behind the `base.Provider` protocol, with an
  `allocator.py` that picks the cheapest fitting class
- `flash/engine/` — the on-GPU worker (TRL + colocated vLLM rollouts) and the
  shared recipe; SFT targets and RL rewards route through the active environment
  (task-specific grading lives with its example, not in the engine)
- `flash/envs/` — environment machinery: registry and the adapter that loads
  Freesolo SDK environments onto the worker's interface
- `flash env setup` — scaffold a starter local Freesolo env, `datasets/train.jsonl`,
  and ready-to-run configs to start from
- `flash/serve/`, `flash/server/` — adapter serving and the FastAPI control
  plane (run operator-side via the separate `flash-server` command)
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

The control plane owns provider credentials: `RUNPOD_API_KEY` is always required,
plus the shared `HF_TOKEN`.
The artifact repo is platform-managed and per-run (each run gets its own
`Freesolo-Co/flashrun-<run_id>`, written by the operator `HF_TOKEN`); it is not a user
knob and not an operator-wide env var. Clients authenticate with their freesolo API key
(`flash login`).

## Serving From an API

`flash chat` is a CLI wrapper around the Flash control-plane chat endpoint. To call a
deployed adapter from your own app, deploy the finished run once and then POST chat
requests with your freesolo API key:

```bash
export FLASH_API_URL=https://flash.freesolo.co
export FREESOLO_API_KEY=fslo_...
export RUN_ID=flash-1782194170-ce1cfcff

curl -X POST "$FLASH_API_URL/v1/runs/$RUN_ID/deploy" \
  -H "Authorization: Bearer $FREESOLO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"dry_run": false}'

curl -X POST "$FLASH_API_URL/v1/runs/$RUN_ID/chat" \
  -H "Authorization: Bearer $FREESOLO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Write a two-sentence summary of the run."}
    ],
    "temperature": 0.0,
    "max_tokens": 256
  }'
```

The response uses the OpenAI chat-completions shape:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "..."
      }
    }
  ]
}
```

Use `choices[0].message.content` for the generated text. The run id is the adapter id
for serving. If the run is not deployed yet, `/v1/runs/<run_id>/chat` returns `409`
with a hint to deploy first.

Operators can also call the Modal serving app directly after the adapter is registered.
The default serving app is `https://clado-ai--freesolo-lora-serving.modal.run`, and
operators can point Flash at another serving app by setting `FREESOLO_SERVING_URL`.
Use that same base URL when calling the app directly; pass the run id as `model`:

```bash
export FREESOLO_SERVING_URL=https://clado-ai--freesolo-lora-serving.modal.run

curl -X POST "$FREESOLO_SERVING_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "flash-1782194170-ce1cfcff",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.0,
    "max_tokens": 256
  }'
```

Prefer the Flash control-plane endpoint for user apps because it enforces run ownership
and forwards per-run serving options such as thinking-mode parity.
