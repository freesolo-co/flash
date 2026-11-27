# Reserved H100 serving

> **Status (2026-11-27): 6 x `Qwen/Qwen3.5-9B-Base`, being acquired.** The earlier 5 x 27B + 3 x 0.8B Base fleet is retired and its secret deleted. `reserved_h100_acquire` launches and qualifies the six 9B Base H100s on Lambda. Reserved routing switches on only when an operator creates the workers secret ("Switching on" below), so 9B Base keeps the on-demand pool until then.

## Fleet

| Slots | Model | Engine |
|---|---|---|
| 1–6 | `Qwen/Qwen3.5-9B-Base` | BF16, one H100 80 GB each |

- Every other hosted model, including `Qwen/Qwen3.8-27B` and `Qwen/Qwen3.5-0.8B-Base`, runs on the on-demand Lambda pool with Modal fallback.
- **Reserved serving is switched on by the workers secret.** `reserved.active(env)` is the single switch. It is on only in main, and only when the Secret `flash-reserved-h100-workers-20261127` exists there.
  - Each `modal deploy` looks the secret up once and logs `reserved H100 serving ON|OFF in main …`.
  - The decision is baked into the serving, Lambda-controller and fleet-control images as `FLASH_RESERVED_H100_ACTIVE`, so remote containers follow it.
  - The lookup is rerun only by the next deploy, so creating or deleting the secret has no effect until you redeploy both the control app and serving.
- **ON:** 9B Base is served only by its slots, with no lease-pool workers and no Modal fallback.
- **OFF** (secret not created yet, or dev): 9B Base uses the ordinary on-demand Lambda pool and Modal fallback; no reserved proxy or watchdog is registered and the deploy references no reserved secret.
- **Source of truth:** `flash/serving/lambda_cloud/reserved.py` (`SLOT_MODELS`), `reserved_acquire.py` and `reserved_watchdog.py`.

| Kind | Name | Holds |
|---|---|---|
| Modal Secret (main) | `flash-reserved-h100-workers-20261127` | `FLASH_RESERVED_H100_WORKERS` (the worker list) and `LAMBDA_API_KEY` (reboots only) |
| Modal Dict (main) | `flash-reserved-h100-acquire-20261127` | acquisition records per slot (`slot-<n>`), `workers` (qualified, secret-shaped entries), `target_slots` |
| Modal Dict (main) | `flash-reserved-h100-health-20261127` | per-slot watchdog state and `configured_slots`, written by the watchdog |
| Lambda instances | `flash-h100-reservation-20261127-<n>` | the six workers; the training provider never sweeps this name prefix |

**Workers secret entry**, one per slot (exactly what the acquire Dict publishes under `workers`):

```json
{"slot": 1, "model": "Qwen/Qwen3.5-9B-Base", "instance_id": "…", "ip": "…", "revision": "<40-hex release>", "tls": {"cert": "…", "token": "…"}}
```

Every entry's `model` must be its slot's model. Otherwise the proxies and the watchdog reject the whole list, and the deploy serves nothing for 9B Base.

## Acquisition (`reserved_h100_acquire`)
Scheduled every 120 s on the main serving app, whenever the deploy has `LAMBDA_API_KEY`. For each target slot (default all six; set `target_slots` to `[]` in the acquire Dict to stop) that the workers secret does not hold:
- **Launch:** at most one per cycle, for the lowest empty slot, only on a single-H100 US Lambda shape with capacity. With no capacity it retries next cycle, indefinitely. Workers boot with the standard cloud-init from `bootstrap.user_data` at the deploy's `FREESOLO_DEPLOYMENT_SHA`, mounting the region's hosting-cache filesystem. The record (with the TLS pin and token, never the private key) is written before the launch request.
- **Reconcile:** records are matched to instances by name. An ambiguous launch is not resubmitted for 30 minutes. An instance with no record is never adopted. A record whose instance disappeared is dropped and the slot reacquired.
- **Qualify and publish:** an active worker must report `ok`, 9B Base, the record's revision and `configured_gpu == "H100"`, then answer a one-token canary. Its entry is then published under `workers`.
- Publishing does not route anything. Nothing here terminates an instance or touches slots the secret holds.

## Switching on (operator)
1. Wait until the deploy logs show `reserved H100 acquire: {... 'published': [1, 2, 3, 4, 5, 6]}`, or read `workers` from the acquire Dict and check it has six entries.
2. Create the Secret `flash-reserved-h100-workers-20261127` in main with `FLASH_RESERVED_H100_WORKERS` = that `workers` list as JSON, and `LAMBDA_API_KEY`.
3. Re-run the "Deploy serving Modal app" workflow on main (control app, then serving).
   - Check the deploy log for `reserved H100 serving ON in main`.
   - Check the logs for `reserved H100 slot <n>: ok` for all six.
   - Smoke-test a 9B Base chat and a 9B Base LoRA chat.
4. The on-demand 9B Base lease workers drain and are reaped by the fleet controller.

## Health and recovery
The watchdog, `reserved_h100_watchdog`, is scheduled every 120 s on the serving app while reserved serving is on.
- **Probe:** each worker gets `/health`, which must report its slot model, plus a one-token canary admitted through the worker's dedicated canary lane. A saturated but healthy worker still answers the canary.
- **Quarantine and reboot:** two failed probes quarantine a worker. After 10 minutes unhealthy it is rebooted through the Lambda API, at most one per cycle, only if another worker passed, and with a 30 minute cooldown.
- **Routing:** requests round-robin over the healthy, configured slots, and fail open if all are quarantined.

## Ownership (ongoing)
- **Billing:** the instances are plain on-demand Lambda VMs. They run, and bill, until an operator terminates them, so treat them as a fixed monthly cost (6 x H100).
- **Lost or replaced instance:** remove the slot's entry from the workers secret and redeploy (9B Base keeps serving on the other slots). Acquisition then reacquires the slot. Copy its newly published entry back into the secret and redeploy again.
- **New release:** a proxy refuses a worker whose reported `deployment_sha` differs from its entry's `revision`. Before deploying a release that changes the worker protocol, re-bootstrap each worker: edit `/opt/flash/worker.env` (`FREESOLO_DEPLOYMENT_SHA`) and `/opt/flash/setup.sh` (revision and `RUNTIME_IMAGE`), then `sudo docker rm -f flash-worker && sudo rm -rf /opt/flash-source/* && sudo bash /opt/flash/setup.sh`. Then update each entry's `revision` in the secret. SSH as `ubuntu@<ip>` with key `freesolo-serving`.

## Retiring the fleet
1. Set `target_slots` to `[]` in the acquire Dict, so nothing is reacquired.
2. Delete the Secret `flash-reserved-h100-workers-20261127` in main and redeploy (control app and serving). The deploy logs `reserved H100 serving OFF`, and 9B Base returns to the on-demand pool and fallback with no code change.
3. Wait out the 2 h router drain.
4. Terminate the instances: for each, `POST https://cloud.lambda.ai/api/v1/instance-operations/terminate` with `{"instance_ids": ["<id>"]}`, then poll `GET /api/v1/instances/<id>` until it returns 404 or `terminated`. Do not use Flash's `terminate_confirmed`: it refuses serving VMs by design.
5. Delete the acquire and health Dicts.
